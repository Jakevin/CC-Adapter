use anyhow::{Context, Result};
use reqwest::Client;
use tracing::{debug, info};

use crate::types::openai::{ChatCompletionRequest, ChatCompletionResponse};

/// OpenAI / Grok 供應商的 HTTP 客戶端
/// HTTP client for OpenAI / Grok provider
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    /// 將請求轉發至 OpenAI 相容的 Chat Completions 端點
    /// Forward the request to an OpenAI-compatible Chat Completions endpoint
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        debug!(model = %request.model, url = %url, "轉發請求至供應商 / Forwarding request to provider");

        // debug 等級時印出即將送出的請求 JSON
        // Print the outgoing request JSON at debug level
        if let Ok(json) = serde_json::to_string_pretty(&request) {
            debug!("送出請求內容 / Outgoing request body:\n{}", json);
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("無法傳送請求至供應商 / Failed to send request to provider")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "無法讀取錯誤回應 / Failed to read error body".to_string());
            // 在錯誤訊息中包含 API URL，方便偵錯 404 / Include API URL in error message for easier 404 debugging
            anyhow::bail!(
                "供應商回傳 HTTP {} (URL: {}) / Provider returned HTTP {} at {}: {}",
                status.as_u16(),
                url,
                status.as_u16(),
                url,
                body
            );
        }

        let body = resp
            .text()
            .await
            .context("無法讀取回應內容 / Failed to read response body")?;

        info!(
            status = %status,
            body_len = body.len(),
            "收到供應商回應 / Received response from provider"
        );
        debug!(body = %body, "供應商回應內容 / Provider response body");

        let response: ChatCompletionResponse =
            serde_json::from_str(&body).context("無法解析供應商回應 / Failed to parse provider response")?;

        Ok(response)
    }
}
