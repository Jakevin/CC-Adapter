# Claude API Adapter

[繁中版](README_ZH_HANT.md) | [簡中版](README_ZH_HANS.md) | [日本語](README_JA.md)

A Rust-based API adapter that lets **Claude Code** use other LLM providers (OpenAI, Grok/xAI, ChatGPT Plus/Pro) by translating between Anthropic's Messages API and provider-specific API formats.

## How It Works

```
                               ┌──[OpenAI Chat API]──▶ OpenAI / Grok
Claude Code ──[Anthropic API]──▶ Adapter (localhost) ─┤
                               └──[Responses API + OAuth]──▶ ChatGPT Codex
```

The adapter runs a local HTTP server that:
1. Accepts requests in Anthropic Messages API format (`POST /v1/messages`)
2. Converts to the target provider's format (Chat Completions or Responses API)
3. Forwards to the configured provider
4. Converts the response back to Anthropic format
5. Returns the result to Claude Code

**Supported providers:**
- **OpenAI** — via API key + Chat Completions API
- **Grok (xAI)** — via API key + Chat Completions API
- **ChatGPT Plus/Pro** — via OAuth + Responses API (Codex backend)
- **Any OpenAI-compatible API** — via API key
- **Anthropic-compatible APIs** — same Messages API as Anthropic, different `base_url`

**Supported features:**
- Text messages and multi-turn conversations
- Tool Use / Function Calling (full round-trip conversion)
- System prompts
- Image inputs (base64)
- Configurable model mapping
- SSE streaming simulation (for non-streaming providers)
- Real-time config hot reload with filesystem watcher
- Graceful shutdown with settings restoration on SIGINT / SIGTERM
- OAuth authentication for ChatGPT (PKCE flow)

## Installation for AI Agents

Copy and paste this prompt to your LLM agent (Claude Code, Cursor, etc.):

```
Install and configure CC-Adapter by following the instructions here:
https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

Or let the agent fetch it directly:

```bash
curl -s https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

## Quick Start

### 1. Install

**Option A: Download pre-built binary (no Rust required)**

Download the latest release from [GitHub Releases](https://github.com/Jakevin/CC-Adapter/releases), extract and you're ready to go:

```bash
tar xzf claude-adapter-<platform>.tar.gz
cd claude-adapter
```

The archive includes the binary and a `config-example.toml` template.

On Windows, download `claude-adapter-windows-amd64.zip` and extract it (it contains `claude-adapter.exe` and `config-example.toml`).

**Option B: Build from source**

```bash
cargo build --release
```

The binary will be at `target/release/claude-adapter`.

### 2. Configure

You can configure **multiple providers at the same time** in `config.toml`, then route each Claude model name to a specific provider/model pair.

#### Multi-provider config (recommended, v0.3.0+)

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "info"
log_file = "adapter.log"
# log_file_enabled = true   # set to false to disable writing logs to file (default: true)

[providers.chatgpt]
type = "chatgpt"
# ChatGPT uses OAuth, no api_key/base_url needed

[providers.openai-compatible]
type = "openai"
# API key (can also be set via ADAPTER_API_KEY)
api_key = "sk-your-openai-or-grok-key"
# Base URL for OpenAI-compatible API (OpenAI / Grok / others)
base_url = "https://api.openai.com/v1"
# Whether the backend returns streaming SSE (usually keep false and let Adapter simulate SSE)
supports_streaming = false

[providers.opencode-go-anthropic]
type = "anthropic-compatible"
api_key = "sk-your-key"
# Base URL for Anthropic-compatible Messages API
base_url = "https://opencode.ai/zen/go"
# Most Anthropic-compatible backends support non-streaming JSON responses; keep false unless you know it's SSE-only
supports_streaming = false

[models]
# Default provider/model when no routing match is found
default_provider = "chatgpt"
default_model = "gpt-5.4"

# Routing table: Anthropic model name → provider + model
# The adapter supports **longest-prefix matching** for model names that include a changing date suffix.
# Example: key "claude-haiku-4-5" matches "claude-haiku-4-5-20251001"
[models.routing]
"claude-sonnet-4-6" = { provider = "openai-compatible", model = "gpt-4.1" }
"claude-opus-4-6"   = { provider = "chatgpt",           model = "gpt-5.4" }
"claude-haiku-4-5"  = { provider = "opencode-go-anthropic", model = "MiniMax-M2.5" }
```

Resolution rules for `models.routing`:

1. Exact match on the full Anthropic model name (e.g. `"claude-opus-4-6"`).
2. If no exact match, use the **longest prefix** key where `incoming_model.starts_with(key)`.  
   This is ideal for models whose name includes a date suffix (e.g. `claude-haiku-4-5-20251001`).
3. If still no match, fall back to `default_provider` + `default_model`.

#### Legacy single-provider config (still supported)

For simple setups you can still use the original single-`[provider]` + `models.mapping` format:

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"        # or "grok" / "chatgpt"
api_key = "sk-your-api-key-here"
base_url = "https://api.openai.com/v1"

[models]
default = "gpt-5.4"

[models.mapping]
"claude-sonnet-4-6" = "gpt-5.4"
"claude-opus-4-6"   = "gpt-5.4"
```

> **Note:** Internally, the adapter normalizes both formats into the same multi-provider structure, so you can safely migrate at your own pace.

### 3. Login (ChatGPT subscribers only)

If using a ChatGPT subscription, run the OAuth login flow first:

```bash
./target/release/claude-adapter login
```

This will:
1. Open your browser to the OpenAI login page
2. After login, automatically receive the OAuth token
3. Save the token to `~/.claude-adapter/tokens-chatgpt.json` (legacy `tokens.json` is still supported)

The token will be automatically refreshed when expired.

#### Multiple ChatGPT accounts

You can bind multiple ChatGPT accounts to different provider names:

```bash
# Default account -> [providers.chatgpt]
./target/release/claude-adapter login

# Second account -> [providers.chatgpt2]
./target/release/claude-adapter login --name chatgpt2
```

Tokens are stored separately as `~/.claude-adapter/tokens-<name>.json`.

### 4. Run

```bash
# Using config file (default command)
./target/release/claude-adapter

# Explicitly use serve subcommand
./target/release/claude-adapter serve --config config.toml

# Using CLI arguments
./target/release/claude-adapter serve --api-key sk-xxx --model gpt-5.4

# Using environment variable for API key
ADAPTER_API_KEY=sk-xxx ./target/release/claude-adapter
```

### 5. Use with Claude Code

The adapter automatically configures `~/.claude/settings.json` on startup — just open a new terminal and run:

```bash
claude
```

No environment variables or shell hooks needed. When the adapter stops, the settings are automatically restored.

## Provider Examples

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
# First time: login
./target/release/claude-adapter login

# Then start directly (type = "chatgpt" in config.toml)
./target/release/claude-adapter
```

### Any OpenAI-compatible API

```bash
./target/release/claude-adapter serve \
  --api-key your-key \
  --base-url https://your-provider.com/v1 \
  --model your-model-name
```

## Docker

### Build

```bash
docker build -t claude-adapter .
```

### Run

```bash
# OpenAI / Grok — pass API key via environment variable
docker run -d -p 8080:8080 \
  -e ADAPTER_API_KEY=sk-your-key \
  claude-adapter

# Mount a custom config.toml
docker run -d -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter

# ChatGPT OAuth — mount token directory
# (run `claude-adapter login` on the host first)
docker run -d -p 8080:8080 \
  -v ~/.claude-adapter:/root/.claude-adapter \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter
```

The container listens on `0.0.0.0:8080` by default. Point Claude Code at the adapter by setting:

```bash
export ANTHROPIC_BASE_URL=http://<docker-host>:8080
claude
```

## CLI Reference

```
Usage: claude-adapter [OPTIONS] [COMMAND]

Commands:
  serve    Start the Adapter proxy server (default)
  login    Run the ChatGPT OAuth login flow
  logout   Clear saved OAuth tokens
  help     Print help

Serve Options:
  -c, --config <CONFIG>      Path to config file [default: config.toml]
      --host <HOST>          Override listen host
  -p, --port <PORT>          Override listen port
      --api-key <API_KEY>    Override provider API key
      --base-url <BASE_URL>  Override provider base URL
      --model <MODEL>        Override default model

Global Options:
      --log-level <LEVEL>    Log level [default: info]
  -h, --help                 Print help
```

**API key priority:** CLI `--api-key` > env `ADAPTER_API_KEY` > `config.toml`

## API Conversion Details

### OpenAI/Grok: Request Mapping (Anthropic → Chat Completions)

| Anthropic | OpenAI |
|-----------|--------|
| `system` (top-level) | `{role: "system"}` message |
| `max_tokens` | `max_completion_tokens` |
| `stop_sequences` | `stop` |
| `tool_choice: {type: "auto"}` | `tool_choice: "auto"` |
| `tool_choice: {type: "any"}` | `tool_choice: "required"` |
| `tool_choice: {type: "tool", name}` | `tool_choice: {type: "function", function: {name}}` |
| `tools[].input_schema` | `tools[].function.parameters` |
| Content block `tool_use` | `tool_calls[]` |
| Content block `tool_result` | `{role: "tool"}` message |

### ChatGPT: Request Mapping (Anthropic → Responses API)

| Anthropic | Responses API |
|-----------|---------------|
| `system` | `instructions` |
| `messages[role=user]` | `input[type=message, role=user]` |
| `messages[role=assistant]` | `input[type=message, role=assistant]` |
| Content block `tool_use` | `input[type=function_call]` |
| Content block `tool_result` | `input[type=function_call_output]` |
| `tools` | `tools` (function type) |

### Response Mapping (Provider → Anthropic)

| OpenAI / Responses API | Anthropic |
|------------------------|-----------|
| `finish_reason: "stop"` / `status: "completed"` | `stop_reason: "end_turn"` |
| `finish_reason: "tool_calls"` / has function_call output | `stop_reason: "tool_use"` |
| `finish_reason: "length"` / `status: "incomplete"` | `stop_reason: "max_tokens"` |
| `usage.prompt_tokens` / `usage.input_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` / `usage.output_tokens` | `usage.output_tokens` |

## Health Check

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

## Current Limitations

- Thinking blocks from third-party Anthropic-compatible APIs are forwarded as proper `thinking` content blocks in SSE.
  - They are not shown as normal text, but some UIs or tools may choose to hide or ignore them.
- ChatGPT OAuth uses the same flow as the official Codex CLI, for personal use only.

## Project Structure

```
src/
├── main.rs                       # Entry point, CLI subcommands, server startup
├── config.rs                     # TOML config + clap CLI parsing, multi-provider & model routing
├── server.rs                     # Axum route handlers, multi-provider dispatch & hot-reload
├── error.rs                      # Unified error types (Anthropic format)
├── auth/
│   ├── oauth.rs                  # PKCE OAuth flow (ChatGPT login)
│   ├── callback_server.rs        # Local OAuth callback server
│   └── token_store.rs            # Token persistence and expiry check
├── types/
│   ├── anthropic.rs              # Anthropic API serde types (requests + responses, thinking/tool_use/text)
│   ├── openai.rs                 # OpenAI Chat Completions API serde types
│   └── responses.rs              # OpenAI Responses API serde types
├── convert/
│   ├── request.rs                # Anthropic → Chat Completions request conversion
│   ├── response.rs               # Chat Completions → Anthropic response conversion
│   ├── request_responses.rs      # Anthropic → Responses API request conversion
│   └── response_responses.rs     # Responses API → Anthropic response conversion
└── providers/
    ├── openai.rs                 # OpenAI/Grok/OpenAI-compatible HTTP client (Chat Completions)
    ├── chatgpt.rs                # ChatGPT Codex HTTP client (Responses API + OAuth)
    └── anthropic.rs              # Anthropic-compatible HTTP client (Messages API passthrough)
```

## Compliance Notice

The ChatGPT OAuth flow uses OpenAI's official OAuth authentication method (the same as the Codex CLI). It is intended for personal development use with your own ChatGPT Plus/Pro subscription. Users are responsible for ensuring their usage complies with OpenAI's Terms of Service.

## License

MIT
