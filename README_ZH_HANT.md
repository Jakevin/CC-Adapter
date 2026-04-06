# Claude API 轉接器

[English](README.md) | [简体中文](README_ZH_HANS.md)

以 Rust 開發的 API 轉接器，透過在 Anthropic Messages API 與供應商特定 API 格式之間進行轉換，讓 **Claude Code** 能夠使用其他 LLM 供應商（OpenAI、Grok/xAI、ChatGPT Plus/Pro），並支援同時配置多個 Provider、依模型名稱路由到不同後端。

## 運作原理

```
                               ┌──[OpenAI Chat API]──▶ OpenAI / Grok
Claude Code ──[Anthropic API]──▶ Adapter (localhost) ─┤
                               └──[Responses API + OAuth]──▶ ChatGPT Codex
```

轉接器會啟動一個本地 HTTP 伺服器：
1. 接收 Anthropic Messages API 格式的請求（`POST /v1/messages`）
2. 轉換為目標供應商的格式（Chat Completions 或 Responses API）
3. 轉發至設定的供應商
4. 將回應轉換回 Anthropic 格式
5. 回傳結果給 Claude Code

**支援的供應商：**
- **OpenAI** — 透過 API 金鑰 + Chat Completions API
- **Grok (xAI)** — 透過 API 金鑰 + Chat Completions API
- **ChatGPT Plus/Pro** — 透過 OAuth + Responses API（Codex 後端）
- **任何 OpenAI 相容 API** — 透過 API 金鑰
- **Anthropic 相容 API** — 與 Anthropic Messages API 相同，只是 `base_url` 不同

**支援功能：**
- 文字訊息與多輪對話
- 工具呼叫（完整的來回轉換）
- 系統提示
- 圖片輸入（base64）
- 可配置的模型映射
- SSE 串流模擬（用於不支援串流的供應商）
- 可選：透過 `[server] claude_stream_idle_timeout_ms` 寫入 Claude Code 的 `CLAUDE_STREAM_IDLE_TIMEOUT_MS`（結束時自備份還原）
- 設定檔熱重載（檔案監聽）
- 優雅關閉時還原 `~/.claude/settings.json`（SIGINT / SIGTERM / SIGHUP）
- ChatGPT OAuth 認證（PKCE 流程）

## AI Agent 安裝指引

將以下提示貼到你的 LLM Agent（Claude Code、Cursor 等）中：

```
Install and configure CC-Adapter by following the instructions here:
https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

或讓 Agent 直接取得安裝指引：

```bash
curl -s https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

## 快速開始

### 1. 安裝

**方式 A：下載預編譯檔（不需安裝 Rust）**

從 [GitHub Releases](https://github.com/Jakevin/CC-Adapter/releases) 下載最新版本，解壓即可使用：

```bash
tar xzf claude-adapter-<platform>.tar.gz
cd claude-adapter
```

壓縮檔內含執行檔與 `config-example.toml` 範本。

Windows 請下載 `claude-adapter-windows-amd64.zip` 並解壓（內含 `claude-adapter.exe` 與 `config-example.toml`）。

**方式 B：從原始碼編譯**

```bash
cargo build --release
```

編譯產物位於 `target/release/claude-adapter`。

### 2. 配置

`config.toml` 支援同時定義多個 Provider，並用 `models.routing` 將不同的 Claude 模型名稱路由到對應的供應商與模型。

#### 多供應商配置（建議，v0.3.0+）

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "info"
log_file = "adapter.log"
# log_file_enabled = true   # 設為 false 可關閉寫入日誌檔案（預設 true）
# claude_stream_idle_timeout_ms = 300000   # 選填：毫秒，寫入 ~/.claude/settings.json；結束時還原。預設 300000（5 分鐘）。設為 0 則不注入、不備份還原。

[providers.chatgpt]
type = "chatgpt"
# ChatGPT 使用 OAuth，不需要 api_key / base_url

[providers.openai-compatible]
type = "openai"
# API 金鑰（也可透過環境變數 ADAPTER_API_KEY 設定）
api_key = "sk-your-openai-or-grok-key"
# OpenAI / Grok / 其他相容 API 的 Base URL
base_url = "https://api.openai.com/v1"
# 後端是否回傳串流 SSE（通常建議關閉，由 Adapter 統一模擬 SSE）
supports_streaming = false

[providers.opencode-go-anthropic]
type = "anthropic-compatible"
api_key = "sk-your-key"
# Anthropic 相容 Messages API 的 Base URL
base_url = "https://opencode.ai/zen/go"
# false = 請求後端回傳單一 JSON（後端支援時建議）。
# true = 請求串流；轉接器會將 Anthropic Messages 的 SSE 聚合成單一回應。
# 若後端仍回傳 SSE（例如忽略 stream=false），轉接器會在可行時同樣聚合。
supports_streaming = false

[models]
# 找不到路由時的預設供應商與模型
default_provider = "chatgpt"
default_model = "gpt-5.4"

# 模型路由表：Anthropic 模型名稱 → 供應商 + 模型
# 支援「最長前綴比對」，適合帶有日期後綴的模型名稱。
[models.routing]
"claude-sonnet-4-6" = { provider = "openai-compatible",        model = "gpt-4.1" }
"claude-opus-4-6"   = { provider = "chatgpt",                  model = "gpt-5.4" }
"claude-haiku-4-5"  = { provider = "opencode-go-anthropic",    model = "MiniMax-M2.5" }
```

`models.routing` 解析規則：

1. 先嘗試完整模型名稱精確比對（例如 `"claude-opus-4-6"`）。
2. 若無精確比對，則尋找「最長前綴 key」，滿足 `incoming_model.starts_with(key)`。  
   例如 key 為 `"claude-haiku-4-5"`，可匹配 `"claude-haiku-4-5-20251001"`。
3. 若仍無對應，則回退到 `default_provider` + `default_model`。

#### 舊版單一 Provider 配置（仍然支援）

若場景簡單，也可以沿用舊版的單一 `[provider]` + `models.mapping` 格式：

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"        # 或 "grok" / "chatgpt"
api_key = "sk-your-api-key-here"
base_url = "https://api.openai.com/v1"

[models]
default = "gpt-5.4"

[models.mapping]
"claude-sonnet-4-6" = "gpt-5.4"
"claude-opus-4-6"   = "gpt-5.4"
```

> **註：** 內部實作上，Adapter 會將新舊兩種格式統一轉換為同一種「多 Provider 結構」，所以可以逐步遷移。

### 3. 登入（僅限 ChatGPT 訂閱使用者）

若使用 ChatGPT 訂閱方案，請先執行 OAuth 登入流程：

```bash
./target/release/claude-adapter login
```

這會：
1. 開啟瀏覽器前往 OpenAI 登入頁面
2. 登入後自動接收 OAuth token
3. 將 token 儲存至 `~/.claude-adapter/tokens-chatgpt.json`（仍向後相容舊的 `tokens.json`）

Token 過期時會自動刷新。

#### 多個 ChatGPT 帳號

你可以把不同的 ChatGPT 帳號綁定到不同的 Provider 名稱（對應 `config.toml` 的 `[providers.<name>]`）：

```bash
# 預設帳號 -> [providers.chatgpt]
./target/release/claude-adapter login

# 第二個帳號 -> [providers.chatgpt2]
./target/release/claude-adapter login --name chatgpt2
```

每個帳號的 token 會分別儲存在 `~/.claude-adapter/tokens-<name>.json`。

### 4. 執行

```bash
# 使用配置檔（預設命令）
./target/release/claude-adapter

# 明確指定 serve 子命令
./target/release/claude-adapter serve --config config.toml

# 使用 CLI 參數
./target/release/claude-adapter serve --api-key sk-xxx --model gpt-5.4

# 透過環境變數設定 API 金鑰
ADAPTER_API_KEY=sk-xxx ./target/release/claude-adapter
```

### 5. 搭配 Claude Code 使用

啟動時會更新 `~/.claude/settings.json`：至少寫入指向轉接器的 `ANTHROPIC_BASE_URL`；除非 `[server] claude_stream_idle_timeout_ms = 0`，還會寫入較長串流閒置逾時的 `CLAUDE_STREAM_IDLE_TIMEOUT_MS`（預設 300000 毫秒）。無須手動設環境變數或 shell hook。優雅結束時會依備份還原上述鍵的先前值。

然後開啟新終端執行：

```bash
claude
```

## 供應商範例

### OpenAI

```bash
./target/release/claude-adapter serve \
  --api-key sk-your-openai-key \
  --base-url https://api.openai.com/v1 \
  --model gpt-5.4
```

### Grok (xAI)

```bash
./target/release/claude-adapter serve \
  --api-key xai-your-grok-key \
  --base-url https://api.x.ai/v1 \
  --model grok-3
```

### ChatGPT Plus/Pro (OAuth)

```bash
# 第一次使用：登入
./target/release/claude-adapter login

# 之後直接啟動（config.toml 中 type = "chatgpt"）
./target/release/claude-adapter
```

### 任何 OpenAI 相容 API

```bash
./target/release/claude-adapter serve \
  --api-key your-key \
  --base-url https://your-provider.com/v1 \
  --model your-model-name
```

## Docker

### 建置

```bash
docker build -t claude-adapter .
```

### 執行

```bash
# OpenAI / Grok — 透過環境變數傳入 API 金鑰
docker run -d -p 8080:8080 \
  -e ADAPTER_API_KEY=sk-your-key \
  claude-adapter

# 掛載自訂 config.toml
docker run -d -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter

# ChatGPT OAuth — 掛載 token 目錄
# （請先在主機上執行 `claude-adapter login`）
docker run -d -p 8080:8080 \
  -v ~/.claude-adapter:/root/.claude-adapter \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter
```

容器預設監聽 `0.0.0.0:8080`。將 Claude Code 指向 adapter：

```bash
export ANTHROPIC_BASE_URL=http://<docker-host>:8080
claude
```

## CLI 參考

```
Usage: claude-adapter [OPTIONS] [COMMAND]

Commands:
  serve    啟動 Adapter 代理伺服器（預設）
  login    執行 ChatGPT OAuth 登入流程
  logout   清除已儲存的 OAuth token
  help     顯示說明

Serve 選項:
  -c, --config <CONFIG>      配置檔路徑 [預設: config.toml]
      --host <HOST>          覆寫監聽主機
  -p, --port <PORT>          覆寫監聽埠號
      --api-key <API_KEY>    覆寫 API 金鑰
      --base-url <BASE_URL>  覆寫 Base URL
      --model <MODEL>        覆寫預設模型

全域選項:
      --log-level <LEVEL>    日誌等級 [預設: info]
  -h, --help                 顯示說明
```

**API 金鑰優先順序：** CLI `--api-key` > 環境變數 `ADAPTER_API_KEY` > `config.toml`

## API 轉換細節

### OpenAI/Grok：請求映射 (Anthropic → Chat Completions)

| Anthropic | OpenAI |
|-----------|--------|
| `system`（頂層欄位） | `{role: "system"}` message |
| `max_tokens` | `max_completion_tokens` |
| `stop_sequences` | `stop` |
| `tool_choice: {type: "auto"}` | `tool_choice: "auto"` |
| `tool_choice: {type: "any"}` | `tool_choice: "required"` |
| `tool_choice: {type: "tool", name}` | `tool_choice: {type: "function", function: {name}}` |
| `tools[].input_schema` | `tools[].function.parameters` |
| 內容區塊 `tool_use` | `tool_calls[]` |
| 內容區塊 `tool_result` | `{role: "tool"}` message |

### ChatGPT：請求映射 (Anthropic → Responses API)

| Anthropic | Responses API |
|-----------|---------------|
| `system` | `instructions` |
| `messages[role=user]` | `input[type=message, role=user]` |
| `messages[role=assistant]` | `input[type=message, role=assistant]` |
| 內容區塊 `tool_use` | `input[type=function_call]` |
| 內容區塊 `tool_result` | `input[type=function_call_output]` |
| `tools` | `tools` (function type) |

### 回應映射 (Provider → Anthropic)

| OpenAI / Responses API | Anthropic |
|------------------------|-----------|
| `finish_reason: "stop"` / `status: "completed"` | `stop_reason: "end_turn"` |
| `finish_reason: "tool_calls"` / has function_call output | `stop_reason: "tool_use"` |
| `finish_reason: "length"` / `status: "incomplete"` | `stop_reason: "max_tokens"` |
| `usage.prompt_tokens` / `usage.input_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` / `usage.output_tokens` | `usage.output_tokens` |

## 健康檢查

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

## 目前限制

- Anthropic 相容後端的**串流**回應：轉接器會將 **文字** 的 SSE delta 聚合成單一回覆；工具呼叫的串流尚未完整還原。
- 來自第三方 Anthropic 相容 API 的 thinking 區塊會作為正式的 `thinking` 內容區塊透過 SSE 轉發，  
  依 UI 實作不同，可能會選擇隱藏或以特定方式顯示，不再當作一般文字插入回應。
- ChatGPT OAuth 使用與官方 Codex CLI 相同的流程，僅限個人使用。

## 專案結構

```
src/
├── main.rs                       # 入口、CLI 子命令、伺服器啟動
├── config.rs                     # TOML 配置 + clap CLI 解析，多供應商與模型路由
├── server.rs                     # Axum 路由處理、多供應商分派與熱重載
├── error.rs                      # 統一錯誤型別（Anthropic 格式）
├── auth/
│   ├── oauth.rs                  # PKCE OAuth 流程（ChatGPT 登入）
│   ├── callback_server.rs        # 本地 OAuth 回調伺服器
│   └── token_store.rs            # Token 持久化與過期檢查
├── types/
│   ├── anthropic.rs              # Anthropic API serde 型別（請求 + 回應，含 thinking/tool_use/text）
│   ├── openai.rs                 # OpenAI Chat Completions API serde 型別
│   └── responses.rs              # OpenAI Responses API serde 型別
├── convert/
│   ├── anthropic_sse.rs          # 將 Anthropic Messages SSE 聚合成單一 MessagesResponse
│   ├── request.rs                # Anthropic → Chat Completions 請求轉換
│   ├── response.rs               # Chat Completions → Anthropic 回應轉換
│   ├── request_responses.rs      # Anthropic → Responses API 請求轉換
│   └── response_responses.rs     # Responses API → Anthropic 回應轉換
└── providers/
    ├── openai.rs                 # OpenAI/Grok/OpenAI 相容 HTTP 客戶端（Chat Completions）
    ├── chatgpt.rs                # ChatGPT Codex HTTP 客戶端（Responses API + OAuth）
    └── anthropic.rs              # Anthropic 相容 HTTP 客戶端（Messages API 直通轉發）
```

## 合規聲明

ChatGPT OAuth 流程使用 OpenAI 官方的 OAuth 認證方式（與 Codex CLI 相同）。僅供個人開發使用，需搭配使用者自己的 ChatGPT Plus/Pro 訂閱。使用者需自行確保遵守 OpenAI 的使用條款。

## 授權

MIT
