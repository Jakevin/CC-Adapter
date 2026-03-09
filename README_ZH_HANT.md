# Claude API 轉接器

[English](README.md) | [简体中文](README_ZH_HANS.md)

以 Rust 開發的 API 轉接器，透過在 Anthropic Messages API 與供應商特定 API 格式之間進行轉換，讓 **Claude Code** 能夠使用其他 LLM 供應商（OpenAI、Grok/xAI、ChatGPT Plus/Pro）。

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

**支援功能：**
- 文字訊息與多輪對話
- 工具呼叫（完整的來回轉換）
- 系統提示
- 圖片輸入（base64）
- 可配置的模型映射
- SSE 串流模擬（用於不支援串流的供應商）
- ChatGPT OAuth 認證（PKCE 流程）

## 快速開始

### 1. 安裝

**方式 A：下載預編譯檔（不需安裝 Rust）**

從 [GitHub Releases](https://github.com/Jakevin/CC-Adapter/releases) 下載最新版本，解壓即可使用：

```bash
tar xzf claude-adapter-<platform>.tar.gz
cd claude-adapter
```

壓縮檔內含執行檔與 `config-example.toml` 範本。

**方式 B：從原始碼編譯**

```bash
cargo build --release
```

編譯產物位於 `target/release/claude-adapter`。

### 2. 配置

編輯 `config.toml` 或使用 CLI 參數 / 環境變數：

#### 用於 OpenAI / Grok（API 金鑰）

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"
api_key = "sk-your-api-key-here"
base_url = "https://api.openai.com/v1"

[models]
default = "gpt-5.4"

[models.mapping]
"claude-sonnet-4-6" = "gpt-5.4"
"claude-opus-4-6" = "gpt-5.4"
```

#### 用於 ChatGPT Plus/Pro（OAuth）

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "chatgpt"

[models]
default = "gpt-5.4"
```

### 3. 登入（僅限 ChatGPT 訂閱使用者）

若使用 ChatGPT 訂閱方案，請先執行 OAuth 登入流程：

```bash
./target/release/claude-adapter login
```

這會：
1. 開啟瀏覽器前往 OpenAI 登入頁面
2. 登入後自動接收 OAuth token
3. 將 token 儲存至 `~/.claude-adapter/tokens.json`

Token 過期時會自動刷新。

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

Adapter 啟動時會自動設定 `~/.claude/settings.json`——只需開啟新終端執行：

```bash
claude
```

不需要設定環境變數或 shell hook。Adapter 停止時會自動還原設定。

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

- 延伸思考區塊會轉換為帶 `<thinking>` 標籤的文字。
- ChatGPT OAuth 使用與官方 Codex CLI 相同的流程，僅限個人使用。

## 專案結構

```
src/
├── main.rs                       # 入口、CLI 子命令、伺服器啟動
├── config.rs                     # TOML 配置 + clap CLI 解析
├── server.rs                     # Axum 路由處理、多供應商分派
├── error.rs                      # 統一錯誤型別（Anthropic 格式）
├── auth/
│   ├── oauth.rs                  # PKCE OAuth 流程（ChatGPT 登入）
│   ├── callback_server.rs        # 本地 OAuth 回調伺服器
│   └── token_store.rs            # Token 持久化與過期檢查
├── types/
│   ├── anthropic.rs              # Anthropic API serde 型別
│   ├── openai.rs                 # OpenAI Chat Completions API serde 型別
│   └── responses.rs              # OpenAI Responses API serde 型別
├── convert/
│   ├── request.rs                # Anthropic → Chat Completions 請求轉換
│   ├── response.rs               # Chat Completions → Anthropic 回應轉換
│   ├── request_responses.rs      # Anthropic → Responses API 請求轉換
│   └── response_responses.rs     # Responses API → Anthropic 回應轉換
└── providers/
    ├── openai.rs                 # OpenAI/Grok HTTP 客戶端（API 金鑰）
    └── chatgpt.rs                # ChatGPT Codex HTTP 客戶端（OAuth）
```

## 合規聲明

ChatGPT OAuth 流程使用 OpenAI 官方的 OAuth 認證方式（與 Codex CLI 相同）。僅供個人開發使用，需搭配使用者自己的 ChatGPT Plus/Pro 訂閱。使用者需自行確保遵守 OpenAI 的使用條款。

## 授權

MIT
