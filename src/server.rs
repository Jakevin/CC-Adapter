use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum::http::StatusCode;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::config::{Config, ServeArgs};
use crate::convert::request::convert_request;
use crate::convert::response::{convert_response, response_to_sse_events};
use crate::error::AppError;
use crate::providers::openai::OpenAIProvider;
use crate::providers::chatgpt::ChatGPTProvider;
use crate::providers::anthropic::AnthropicCompatibleProvider;
use crate::types::anthropic::MessagesRequest;

/// 供應商列舉：支援 OpenAI 相容（含 Grok）、ChatGPT OAuth、Anthropic 相容
/// Provider enum: supports OpenAI-compatible (including Grok), ChatGPT OAuth, and Anthropic-compatible
pub enum ProviderKind {
    OpenAI(OpenAIProvider),
    ChatGPT(ChatGPTProvider),
    Anthropic(AnthropicCompatibleProvider),
}

/// 應用程式共享狀態（支援熱重載）
/// Application shared state (supports hot-reload)
pub struct AppState {
    pub config: RwLock<Config>,
    pub providers: RwLock<HashMap<String, ProviderKind>>,
    pub serve_args: ServeArgs,
}

/// 根據配置建構所有供應商實例
/// Build all provider instances from configuration
async fn build_providers(config: &Config) -> anyhow::Result<HashMap<String, ProviderKind>> {
    let mut providers = HashMap::new();

    for (name, provider_config) in &config.providers {
        let kind = match provider_config.provider_type.as_str() {
            "chatgpt" => {
                let chatgpt = ChatGPTProvider::new(name).await?;
                ProviderKind::ChatGPT(chatgpt)
            }
            "anthropic-compatible" => {
                let anth = AnthropicCompatibleProvider::new(
                    provider_config.api_key.clone(),
                    provider_config.base_url.clone(),
                );
                ProviderKind::Anthropic(anth)
            }
            _ => {
                let openai = OpenAIProvider::new(
                    provider_config.api_key.clone(),
                    provider_config.base_url.clone(),
                );
                ProviderKind::OpenAI(openai)
            }
        };
        providers.insert(name.clone(), kind);
    }

    Ok(providers)
}

/// 根據配置建構 AppState
/// Build AppState based on configuration
pub async fn build_app_state(config: Config, serve_args: ServeArgs) -> anyhow::Result<Arc<AppState>> {
    let providers = build_providers(&config).await?;

    Ok(Arc::new(AppState {
        config: RwLock::new(config),
        providers: RwLock::new(providers),
        serve_args,
    }))
}

/// 重新載入配置並重建供應商（熱重載核心）
/// Reload config and rebuild provider (hot-reload core)
pub async fn reload_config(state: &Arc<AppState>) -> anyhow::Result<()> {
    let new_config = Config::load(&state.serve_args)?;
    let new_providers = build_providers(&new_config).await?;

    let default_provider = new_config.models.default_provider.clone();
    let default_model = new_config.models.default_model.clone();
    let provider_names: Vec<String> = new_config.providers.keys().cloned().collect();

    *state.providers.write().await = new_providers;
    *state.config.write().await = new_config;

    eprintln!("\n  ⟳ config.toml 已更新，配置已熱重載");
    eprintln!("  ⟳ config.toml updated, configuration hot-reloaded");
    eprintln!("    供應商 / Providers: {}", provider_names.join(", "));
    eprintln!("    預設供應商 / Default provider: {}", default_provider);
    eprintln!("    預設模型 / Default model: {}\n", default_model);

    info!(
        providers = ?provider_names,
        default_provider = %default_provider,
        default_model = %default_model,
        "配置已熱重載 / Configuration hot-reloaded"
    );

    Ok(())
}

/// 處理 POST /v1/messages 請求：接收 Anthropic 格式、轉換、轉發、回傳
/// 根據模型路由表選擇對應的供應商。
///
/// Handle POST /v1/messages: receive Anthropic format, convert, forward, and return.
/// Selects provider based on model routing table.
pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<MessagesRequest>,
) -> Result<Response, AppError> {
    let anthropic_version = headers
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");
    let wants_stream = request.stream.unwrap_or(false);
    info!(
        model = %request.model,
        anthropic_version = %anthropic_version,
        max_tokens = request.max_tokens,
        message_count = request.messages.len(),
        has_tools = request.tools.is_some(),
        stream = wants_stream,
        "收到 Anthropic Messages API 請求 / Received Anthropic Messages API request"
    );

    if let Ok(json) = serde_json::to_string_pretty(&request) {
        debug!("Anthropic 請求內容 / Anthropic request body:\n{}", json);
    }

    let original_model = request.model.clone();

    let config = state.config.read().await;
    let providers = state.providers.read().await;

    // 根據模型路由表解析目標供應商與模型
    // Resolve target provider and model from the routing table
    let (provider_name, target_model) = config.models.resolve(&request.model);

    info!(
        provider = %provider_name,
        target_model = %target_model,
        "路由至供應商 / Routing to provider"
    );

    let provider = providers.get(&provider_name).ok_or_else(|| {
        error!(
            provider = %provider_name,
            "路由表指定的供應商不存在 / Provider specified in routing table not found"
        );
        AppError::internal(format!(
            "Provider '{}' not found. Available providers: {}",
            provider_name,
            providers.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    })?;

    let anthropic_response = match provider {
        ProviderKind::OpenAI(p) => {
            handle_openai_request(p, request, &target_model, &original_model).await?
        }
        ProviderKind::ChatGPT(p) => {
            handle_chatgpt_request(p, request, &target_model, &original_model).await?
        }
        ProviderKind::Anthropic(p) => {
            // 從 provider config 讀取 supports_streaming，決定是否對後端開啟 stream
            // Read supports_streaming from provider config to decide whether to enable stream for backend
            let stream = config
                .providers
                .get(&provider_name)
                .map(|c| c.supports_streaming)
                .unwrap_or(false);

            handle_anthropic_request(p, request, &target_model, anthropic_version, stream).await?
        }
    };

    drop(config);
    drop(providers);

    info!(
        stop_reason = ?anthropic_response.stop_reason,
        content_blocks = anthropic_response.content.len(),
        input_tokens = anthropic_response.usage.input_tokens,
        output_tokens = anthropic_response.usage.output_tokens,
        response_mode = if wants_stream { "SSE 串流 / SSE stream" } else { "JSON" },
        "回傳 Anthropic 回應 / Returning Anthropic response"
    );

    if let Ok(json) = serde_json::to_string_pretty(&anthropic_response) {
        debug!("Anthropic 回應內容 / Anthropic response body:\n{}", json);
    }

    if wants_stream {
        let events = response_to_sse_events(&anthropic_response).map_err(|e| {
            error!(error = %e, "SSE 事件轉換失敗 / Failed to convert to SSE events");
            AppError::internal(format!("SSE conversion failed: {}", e))
        })?;

        debug!(
            event_count = events.len(),
            "模擬 SSE 串流回傳 / Simulating SSE stream response"
        );

        let stream = tokio_stream::iter(events.into_iter().map(Ok::<_, Infallible>));
        Ok(Sse::new(stream).into_response())
    } else {
        Ok(Json(anthropic_response).into_response())
    }
}

/// 處理 OpenAI / Grok 供應商的請求轉換與轉發
/// Handle request conversion and forwarding for OpenAI / Grok provider
async fn handle_openai_request(
    provider: &OpenAIProvider,
    request: MessagesRequest,
    target_model: &str,
    original_model: &str,
) -> Result<crate::types::anthropic::MessagesResponse, AppError> {
    let openai_request = convert_request(request, target_model)
        .map_err(|e| {
            error!(error = %e, "請求轉換失敗 / Failed to convert request");
            AppError::bad_request(format!("Request conversion failed: {}", e))
        })?;

    debug!(
        openai_model = %openai_request.model,
        openai_messages = openai_request.messages.len(),
        "已轉換為 OpenAI 格式 / Converted to OpenAI format"
    );

    if let Ok(json) = serde_json::to_string_pretty(&openai_request) {
        debug!("OpenAI 請求內容 / OpenAI request body:\n{}", json);
    }

    let openai_response = provider
        .chat_completion(openai_request)
        .await
        .map_err(|e| {
            error!(error = %e, "供應商請求失敗 / Provider request failed");
            AppError::internal(format!("Provider error: {}", e))
        })?;

    convert_response(openai_response, original_model).map_err(|e| {
        error!(error = %e, "回應轉換失敗 / Failed to convert response");
        AppError::internal(format!("Response conversion failed: {}", e))
    })
}

/// 處理 ChatGPT 供應商的請求轉換與轉發
/// Handle request conversion and forwarding for ChatGPT provider
async fn handle_chatgpt_request(
    provider: &ChatGPTProvider,
    request: MessagesRequest,
    target_model: &str,
    original_model: &str,
) -> Result<crate::types::anthropic::MessagesResponse, AppError> {
    use crate::convert::request_responses::convert_request_to_responses;
    use crate::convert::response_responses::convert_responses_to_anthropic;

    let responses_request = convert_request_to_responses(request, target_model)
        .map_err(|e| {
            error!(error = %e, "Responses API 請求轉換失敗 / Responses API request conversion failed");
            AppError::bad_request(format!("Responses API conversion failed: {}", e))
        })?;

    debug!(
        model = %responses_request.model,
        input_count = responses_request.input.len(),
        "已轉換為 Responses API 格式 / Converted to Responses API format"
    );

    if let Ok(json) = serde_json::to_string_pretty(&responses_request) {
        debug!("Responses API 請求內容 / Responses API request body:\n{}", json);
    }

    let sse_text = provider
        .send_request(&responses_request)
        .await
        .map_err(|e| {
            error!(error = %e, "ChatGPT 供應商請求失敗 / ChatGPT provider request failed");
            AppError::internal(format!("ChatGPT provider error: {}", e))
        })?;

    convert_responses_to_anthropic(&sse_text, original_model).map_err(|e| {
        error!(error = %e, "Responses API 回應轉換失敗 / Responses API response conversion failed");
        AppError::internal(format!("Responses API response conversion failed: {}", e))
    })
}

/// 處理 Anthropic 相容供應商的請求轉發（不需格式轉換，只會覆寫目標模型）
/// Handle request forwarding for Anthropic-compatible provider (no format conversion, only model override)
async fn handle_anthropic_request(
    provider: &AnthropicCompatibleProvider,
    mut request: MessagesRequest,
    target_model: &str,
    anthropic_version: &str,
    stream: bool,
) -> Result<crate::types::anthropic::MessagesResponse, AppError> {
    // 覆寫模型為路由後的目標模型
    // Override model with routed target model
    request.model = target_model.to_string();

    // 由 Provider 設定檔中的 supports_streaming 控制是否對後端開啟 stream。
    // For Anthropic-compatible providers, the supports_streaming config flag controls whether
    // we ask the backend for a streaming response or a single JSON response.
    //
    // 若後端在 stream=true 時回傳 Anthropic Messages SSE，provider 會聚合為單一 JSON 回應。
    // When stream=true, if the backend returns Anthropic Messages SSE, the provider aggregates it into one JSON response.
    request.stream = Some(stream);

    provider
        .messages(&request, anthropic_version)
        .await
        .map_err(|e| {
            error!(
                error = %e,
                "Anthropic 相容供應商請求失敗 / Anthropic-compatible provider request failed"
            );
            AppError::internal(format!("Anthropic-compatible provider error: {}", e))
        })
}

/// 健康檢查端點
/// Health check endpoint
pub async fn handle_health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}
