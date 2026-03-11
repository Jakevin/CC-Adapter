use anyhow::{Context, Result};
use reqwest::Client;
use tracing::{debug, info};

use crate::types::anthropic::{MessagesRequest, MessagesResponse};

/// Anthropic 相容供應商的 HTTP 客戶端（API 介面與 Anthropic 相同，只是 base_url 不同）
/// HTTP client for Anthropic-compatible provider (same API as Anthropic, different base_url)
pub struct AnthropicCompatibleProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicCompatibleProvider {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    /// 將 Anthropic Messages 請求直接轉發到相容後端
    /// Forward Anthropic Messages request directly to a compatible backend
    pub async fn messages(
        &self,
        request: &MessagesRequest,
        anthropic_version: &str,
    ) -> Result<MessagesResponse> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        debug!(
            model = %request.model,
            url = %url,
            "轉發請求至 Anthropic 相容供應商 / Forwarding request to Anthropic-compatible provider"
        );

        // debug 等級時印出即將送出的請求 JSON
        // Print the outgoing request JSON at debug level
        if let Ok(json) = serde_json::to_string_pretty(request) {
            debug!("送出請求內容 / Outgoing request body:\n{}", json);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", anthropic_version)
            .json(request)
            .send()
            .await
            .context("無法傳送請求至 Anthropic 相容供應商 / Failed to send request to Anthropic-compatible provider")?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .context(
                "無法讀取 Anthropic 相容供應商回應 / Failed to read Anthropic-compatible provider response",
            )?;

        info!(
            status = %status,
            body_len = body.len(),
            "收到 Anthropic 相容供應商回應 / Received response from Anthropic-compatible provider"
        );
        debug!(
            body = %body,
            "Anthropic 相容供應商回應內容 / Anthropic-compatible provider response body"
        );

        if !status.is_success() {
            // 在錯誤訊息中包含 API URL，方便偵錯 404 / Include API URL in error message for easier 404 debugging
            anyhow::bail!(
                "Anthropic 相容供應商回傳 HTTP {} (URL: {}) / Anthropic-compatible provider returned HTTP {} at {}: {}",
                status.as_u16(),
                url,
                status.as_u16(),
                url,
                body
            );
        }

        // 解析失敗時，把原始 body 一併寫進錯誤，方便比對實際回傳格式
        // When parsing fails, include raw body in the error to inspect actual response shape
        let response: MessagesResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!(
                "無法解析 Anthropic 相容供應商回應: {}，原始回應內容 / Failed to parse Anthropic-compatible provider response: {}. Raw body: {}",
                e,
                e,
                body
            )
        })?;

        Ok(response)
    }
}

