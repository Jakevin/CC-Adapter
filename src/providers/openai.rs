use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
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
            .map_err(|err| {
                anyhow::anyhow!(
                    "{} request failed (model: {}, URL: {}): {}",
                    classify_reqwest_error(&err),
                    request.model,
                    url,
                    err
                )
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "無法讀取錯誤回應 / Failed to read error body".to_string());
            // 在錯誤訊息中包含 API URL，方便偵錯 404 / Include API URL in error message for easier 404 debugging
            anyhow::bail!(
                "{}: provider returned HTTP {} (model: {}, URL: {}): {}",
                classify_http_status(status),
                status.as_u16(),
                request.model,
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

        let response: ChatCompletionResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "json_parse_failed: failed to parse provider response (model: {}, URL: {}, HTTP {}, body preview: {})",
                request.model,
                url,
                status.as_u16(),
                preview_body(&body, 1000)
            )
        })?;

        Ok(response)
    }
}

fn classify_reqwest_error(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else {
        "request_failed"
    }
}

fn classify_http_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "bad_request",
        422 => "unprocessable_entity",
        500..=599 => "provider_5xx",
        _ => "provider_http_error",
    }
}

fn preview_body(body: &str, max_chars: usize) -> String {
    body.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_status_classification() {
        assert_eq!(classify_http_status(StatusCode::BAD_REQUEST), "bad_request");
        assert_eq!(
            classify_http_status(StatusCode::UNPROCESSABLE_ENTITY),
            "unprocessable_entity"
        );
        assert_eq!(
            classify_http_status(StatusCode::INTERNAL_SERVER_ERROR),
            "provider_5xx"
        );
    }
}
