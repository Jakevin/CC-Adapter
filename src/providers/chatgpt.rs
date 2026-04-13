use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::auth::oauth;
use crate::auth::token_store::{self, TokenData};
use crate::types::responses::ResponsesRequest;

const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";

/// ChatGPT OAuth 供應商：透過 ChatGPT Plus/Pro 訂閱認證存取 Codex API
/// ChatGPT OAuth provider: access Codex API via ChatGPT Plus/Pro subscription authentication
pub struct ChatGPTProvider {
    client: Client,
    token: RwLock<TokenData>,
    account_id: RwLock<String>,
    token_name: String,
}

impl ChatGPTProvider {
    /// 建構 ChatGPTProvider，從 token store 載入認證資訊
    /// Construct ChatGPTProvider, loading auth info from token store
    pub async fn new(name: &str) -> Result<Self> {
        let token_data = token_store::load_named(name)?
            .context(
                "未找到 OAuth token，請先執行 `claude-adapter login` / \
                 No OAuth token found, please run `claude-adapter login` first",
            )?;

        let account_id = oauth::extract_account_id(&token_data.access_token)
            .unwrap_or_default();

        if account_id.is_empty() {
            warn!(
                "無法從 token 中提取 account ID，部分功能可能受影響 / \
                 Could not extract account ID from token, some features may be affected"
            );
        }

        info!(
            account_id = %if account_id.is_empty() { "<unknown>" } else { &account_id },
            "ChatGPT Provider 已初始化 / ChatGPT Provider initialized"
        );

        Ok(Self {
            client: Client::new(),
            token: RwLock::new(token_data),
            account_id: RwLock::new(account_id),
            token_name: name.to_string(),
        })
    }

    /// 確保 token 未過期，若已過期則自動刷新
    /// Ensure the token is not expired; automatically refresh if it is
    async fn ensure_valid_token(&self) -> Result<TokenData> {
        {
            let token = self.token.read().await;
            if !token.is_expired() {
                return Ok(token.clone());
            }
        }

        info!("Token 已過期，正在刷新... / Token expired, refreshing...");

        let current = self.token.read().await.clone();
        let new_token = oauth::refresh_token(&current.refresh_token).await?;

        token_store::save_named(&self.token_name, &new_token)?;

        if let Ok(new_account_id) = oauth::extract_account_id(&new_token.access_token) {
            *self.account_id.write().await = new_account_id;
        }

        *self.token.write().await = new_token.clone();

        info!("Token 刷新成功 / Token refreshed successfully");
        Ok(new_token)
    }

    /// OAuth 設定檔名稱（與 `claude-adapter login` 相同），供 server 在需 `Send` 的串流任務中重建實例
    /// OAuth profile name (same as `claude-adapter login`), for server to rebuild in Send streaming tasks
    pub fn token_name(&self) -> &str {
        &self.token_name
    }

    /// 傳送請求至 Codex API 並回傳 SSE 串流的原始文字
    /// Send a request to the Codex API and return the raw SSE stream text
    pub async fn send_request(&self, request: &ResponsesRequest) -> Result<String> {
        let token = self.ensure_valid_token().await?;
        let account_id = self.account_id.read().await.clone();

        let url = format!("{}/codex/responses", CODEX_BASE_URL);

        debug!(model = %request.model, url = %url, "轉發請求至 ChatGPT Codex / Forwarding request to ChatGPT Codex");

        // debug 等級時印出即將送出的請求 JSON
        // Print the outgoing request JSON at debug level
        if let Ok(json) = serde_json::to_string_pretty(request) {
            debug!("送出請求內容 / Outgoing request body:\n{}", json);
        }

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "codex_cli_rs");

        if !account_id.is_empty() {
            req_builder = req_builder.header("chatgpt-account-id", &account_id);
        }

        let resp = req_builder
            .json(request)
            .send()
            .await
            .context("無法傳送請求至 ChatGPT Codex / Failed to send request to ChatGPT Codex")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "無法讀取錯誤回應 / Failed to read error body".to_string());
            // 在錯誤訊息中包含 API URL，方便偵錯 404 / Include API URL in error message for easier 404 debugging
            anyhow::bail!(
                "ChatGPT Codex 回傳 HTTP {} (URL: {}) / ChatGPT Codex returned HTTP {} at {}: {}",
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
            .context("無法讀取 ChatGPT Codex 回應 / Failed to read ChatGPT Codex response")?;

        log_received_response(status, body.len(), &request.model);
        debug!(body = %body, "ChatGPT Codex 回應內容 / ChatGPT Codex response body");

        // 印出 SSE 原始前 1500 字元，協助偵測事件格式（INFO，可部署後移除）
        // Log first 1500 chars of raw SSE body to detect event format (INFO, remove after fix)
        {
            let preview: String = body.chars().take(1500).collect();
            info!("SSE 原始前 1500 字元 / Raw SSE body (first 1500 chars):\n{}", preview);
        }

        Ok(body)
    }
}

fn log_received_response(status: StatusCode, body_len: usize, model: &str) {
    info!(
        status = %status,
        body_len,
        model = %model,
        "收到 ChatGPT Codex 回應 / Received response from ChatGPT Codex"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing::subscriber::with_default;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct TestWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl<'a> MakeWriter<'a> for TestWriter {
        type Writer = TestWriterGuard;

        fn make_writer(&'a self) -> Self::Writer {
            TestWriterGuard {
                buffer: self.buffer.clone(),
            }
        }
    }

    struct TestWriterGuard {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl io::Write for TestWriterGuard {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_received_response_log_includes_model() {
        let writer = TestWriter::default();
        let buffer = writer.buffer.clone();

        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .without_time()
            .with_ansi(false)
            .finish();

        with_default(subscriber, || {
            log_received_response(StatusCode::OK, 123, "gpt-5.3-codex");
        });

        let output = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(output.contains("Received response from ChatGPT Codex"));
        assert!(output.contains("model=gpt-5.3-codex"));
    }
}
