# Claude API 转接器

[English](README.md) | [繁體中文](README_ZH_HANT.md)

基于 Rust 开发的 API 转接器，通过在 Anthropic Messages API 与供应商特定 API 格式之间进行转换，让 **Claude Code** 能够使用其他 LLM 供应商（OpenAI、Grok/xAI、ChatGPT Plus/Pro），并支持同时配置多个 Provider，按模型名称路由到不同后端。

## 工作原理

```
                               ┌──[OpenAI Chat API]──▶ OpenAI / Grok
Claude Code ──[Anthropic API]──▶ Adapter (localhost) ─┤
                               └──[Responses API + OAuth]──▶ ChatGPT Codex
```

转接器会启动一个本地 HTTP 服务器：
1. 接收 Anthropic Messages API 格式的请求（`POST /v1/messages`）
2. 转换为目标供应商的格式（Chat Completions 或 Responses API）
3. 转发至配置的供应商
4. 将响应转换回 Anthropic 格式
5. 返回结果给 Claude Code

**支持的供应商：**
- **OpenAI** — 通过 API 密钥 + Chat Completions API
- **Grok (xAI)** — 通过 API 密钥 + Chat Completions API
- **ChatGPT Plus/Pro** — 通过 OAuth + Responses API（Codex 后端）
- **任何 OpenAI 兼容 API** — 通过 API 密钥
- **Anthropic 兼容 API** — 与 Anthropic Messages API 相同，只是 `base_url` 不同

**支持功能：**
- 文本消息与多轮对话
- 工具调用（完整的往返转换）
- 系统提示
- 图片输入（base64）
- 可配置的模型映射
- SSE 流式模拟（用于不支持流式的供应商）
- ChatGPT OAuth 认证（PKCE 流程）

## AI Agent 安装指引

将以下提示粘贴到你的 LLM Agent（Claude Code、Cursor 等）中：

```
Install and configure CC-Adapter by following the instructions here:
https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

或让 Agent 直接获取安装指引：

```bash
curl -s https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

## 快速开始

### 1. 安装

**方式 A：下载预编译文件（无需安装 Rust）**

从 [GitHub Releases](https://github.com/Jakevin/CC-Adapter/releases) 下载最新版本，解压即可使用：

```bash
tar xzf claude-adapter-<platform>.tar.gz
cd claude-adapter
```

压缩包内含可执行文件与 `config-example.toml` 模板。

Windows 请下载 `claude-adapter-windows-amd64.zip` 并解压（包含 `claude-adapter.exe` 与 `config-example.toml`）。

**方式 B：从源码编译**

```bash
cargo build --release
```

编译产物位于 `target/release/claude-adapter`。

### 2. 配置

`config.toml` 支持同时定义多个 Provider，并用 `models.routing` 将不同 Claude 模型名称路由到对应的供应商与模型。

#### 多 Provider 配置（推荐，v0.3.0+）

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "info"
log_file = "adapter.log"
# log_file_enabled = true   # 设为 false 可关闭写入日志文件（默认 true）

[providers.chatgpt]
type = "chatgpt"
# ChatGPT 使用 OAuth，不需要 api_key / base_url

[providers.openai-compatible]
type = "openai"
# API 密钥（也可通过环境变量 ADAPTER_API_KEY 设置）
api_key = "sk-your-openai-or-grok-key"
# OpenAI / Grok / 其他兼容 API 的 Base URL
base_url = "https://api.openai.com/v1"
# 后端是否返回流式 SSE（通常建议关闭，由 Adapter 统一模拟 SSE）
supports_streaming = false

[providers.opencode-go-anthropic]
type = "anthropic-compatible"
api_key = "sk-your-key"
# Anthropic 兼容 Messages API 的 Base URL
base_url = "https://opencode.ai/zen/go"
# 大多数 Anthropic 兼容后端支持非流式 JSON 响应，建议保持 false
supports_streaming = false

[models]
# 找不到路由时使用的默认供应商和默认模型
default_provider = "chatgpt"
default_model = "gpt-5.4"

# 模型路由表：Anthropic 模型名称 → 供应商 + 模型
# 支持「最长前缀匹配」，适合名称中带日期后缀的模型。
[models.routing]
"claude-sonnet-4-6" = { provider = "openai-compatible",        model = "gpt-4.1" }
"claude-opus-4-6"   = { provider = "chatgpt",                  model = "gpt-5.4" }
"claude-haiku-4-5"  = { provider = "opencode-go-anthropic",    model = "MiniMax-M2.5" }
```

`models.routing` 解析规则：

1. 先尝试完整模型名称精确匹配（例如 `"claude-opus-4-6"`）。
2. 若没有精确匹配，则寻找「最长前缀 key」，满足 `incoming_model.starts_with(key)`。  
   例如 key 为 `"claude-haiku-4-5"`，可匹配 `"claude-haiku-4-5-20251001"`。
3. 若仍无匹配，则回退到 `default_provider` + `default_model`。

#### 旧版单 Provider 配置（仍然支持）

如果场景比较简单，也可以继续使用旧版的单一 `[provider]` + `models.mapping` 格式：

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

> **注意：** 在内部实现上，Adapter 会把新旧两种格式统一转换为同一种「多 Provider 结构」，可以按需逐步迁移。

### 3. 登录（仅限 ChatGPT 订阅用户）

若使用 ChatGPT 订阅方案，请先执行 OAuth 登录流程：

```bash
./target/release/claude-adapter login
```

这会：
1. 打开浏览器前往 OpenAI 登录页面
2. 登录后自动接收 OAuth token
3. 将 token 保存至 `~/.claude-adapter/tokens-chatgpt.json`（仍向后兼容旧的 `tokens.json`）

Token 过期时会自动刷新。

#### 多个 ChatGPT 账号

你可以把不同的 ChatGPT 账号绑定到不同的 Provider 名称（对应 `config.toml` 的 `[providers.<name>]`）：

```bash
# 默认账号 -> [providers.chatgpt]
./target/release/claude-adapter login

# 第二个账号 -> [providers.chatgpt2]
./target/release/claude-adapter login --name chatgpt2
```

每个账号的 token 会分别保存在 `~/.claude-adapter/tokens-<name>.json`。

### 4. 运行

```bash
# 使用配置文件（默认命令）
./target/release/claude-adapter

# 明确指定 serve 子命令
./target/release/claude-adapter serve --config config.toml

# 使用 CLI 参数
./target/release/claude-adapter serve --api-key sk-xxx --model gpt-5.4

# 通过环境变量设置 API 密钥
ADAPTER_API_KEY=sk-xxx ./target/release/claude-adapter
```

### 5. 搭配 Claude Code 使用

Adapter 启动时会自动配置 `~/.claude/settings.json`——只需打开新终端运行：

```bash
claude
```

不需要设置环境变量或 shell hook。Adapter 停止时会自动还原设置。

## 供应商示例

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
# 首次使用：登录
./target/release/claude-adapter login

# 之后直接启动（config.toml 中 type = "chatgpt"）
./target/release/claude-adapter
```

### 任何 OpenAI 兼容 API

```bash
./target/release/claude-adapter serve \
  --api-key your-key \
  --base-url https://your-provider.com/v1 \
  --model your-model-name
```

## Docker

### 构建

```bash
docker build -t claude-adapter .
```

### 运行

```bash
# OpenAI / Grok — 通过环境变量传入 API 密钥
docker run -d -p 8080:8080 \
  -e ADAPTER_API_KEY=sk-your-key \
  claude-adapter

# 挂载自定义 config.toml
docker run -d -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter

# ChatGPT OAuth — 挂载 token 目录
# （请先在主机上运行 `claude-adapter login`）
docker run -d -p 8080:8080 \
  -v ~/.claude-adapter:/root/.claude-adapter \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter
```

容器默认监听 `0.0.0.0:8080`。将 Claude Code 指向 adapter：

```bash
export ANTHROPIC_BASE_URL=http://<docker-host>:8080
claude
```

## CLI 参考

```
Usage: claude-adapter [OPTIONS] [COMMAND]

Commands:
  serve    启动 Adapter 代理服务器（默认）
  login    执行 ChatGPT OAuth 登录流程
  logout   清除已保存的 OAuth token
  help     显示说明

Serve 选项:
  -c, --config <CONFIG>      配置文件路径 [默认: config.toml]
      --host <HOST>          覆盖监听主机
  -p, --port <PORT>          覆盖监听端口
      --api-key <API_KEY>    覆盖 API 密钥
      --base-url <BASE_URL>  覆盖 Base URL
      --model <MODEL>        覆盖默认模型

全局选项:
      --log-level <LEVEL>    日志级别 [默认: info]
  -h, --help                 显示说明
```

**API 密钥优先顺序：** CLI `--api-key` > 环境变量 `ADAPTER_API_KEY` > `config.toml`

## API 转换细节

### OpenAI/Grok：请求映射 (Anthropic → Chat Completions)

| Anthropic | OpenAI |
|-----------|--------|
| `system`（顶层字段） | `{role: "system"}` message |
| `max_tokens` | `max_completion_tokens` |
| `stop_sequences` | `stop` |
| `tool_choice: {type: "auto"}` | `tool_choice: "auto"` |
| `tool_choice: {type: "any"}` | `tool_choice: "required"` |
| `tool_choice: {type: "tool", name}` | `tool_choice: {type: "function", function: {name}}` |
| `tools[].input_schema` | `tools[].function.parameters` |
| 内容块 `tool_use` | `tool_calls[]` |
| 内容块 `tool_result` | `{role: "tool"}` message |

### ChatGPT：请求映射 (Anthropic → Responses API)

| Anthropic | Responses API |
|-----------|---------------|
| `system` | `instructions` |
| `messages[role=user]` | `input[type=message, role=user]` |
| `messages[role=assistant]` | `input[type=message, role=assistant]` |
| 内容块 `tool_use` | `input[type=function_call]` |
| 内容块 `tool_result` | `input[type=function_call_output]` |
| `tools` | `tools` (function type) |

### 响应映射 (Provider → Anthropic)

| OpenAI / Responses API | Anthropic |
|------------------------|-----------|
| `finish_reason: "stop"` / `status: "completed"` | `stop_reason: "end_turn"` |
| `finish_reason: "tool_calls"` / has function_call output | `stop_reason: "tool_use"` |
| `finish_reason: "length"` / `status: "incomplete"` | `stop_reason: "max_tokens"` |
| `usage.prompt_tokens` / `usage.input_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` / `usage.output_tokens` | `usage.output_tokens` |

## 健康检查

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

## 当前限制

- 来自第三方 Anthropic 兼容 API 的 thinking 块会作为正式的 `thinking` 内容块通过 SSE 转发，  
  具体是否显示由上游 UI 或工具决定，不再当作普通文本插入到最终回答中。
- ChatGPT OAuth 使用与官方 Codex CLI 相同的流程，仅限个人使用。

## 项目结构

```
src/
├── main.rs                       # 入口、CLI 子命令、服务器启动
├── config.rs                     # TOML 配置 + clap CLI 解析，多 Provider 与模型路由
├── server.rs                     # Axum 路由处理、多供应商分派与热重载
├── error.rs                      # 统一错误类型（Anthropic 格式）
├── auth/
│   ├── oauth.rs                  # PKCE OAuth 流程（ChatGPT 登录）
│   ├── callback_server.rs        # 本地 OAuth 回调服务器
│   └── token_store.rs            # Token 持久化与过期检查
├── types/
│   ├── anthropic.rs              # Anthropic API serde 类型（请求 + 响应，含 thinking/tool_use/text）
│   ├── openai.rs                 # OpenAI Chat Completions API serde 类型
│   └── responses.rs              # OpenAI Responses API serde 类型
├── convert/
│   ├── request.rs                # Anthropic → Chat Completions 请求转换
│   ├── response.rs               # Chat Completions → Anthropic 响应转换
│   ├── request_responses.rs      # Anthropic → Responses API 请求转换
│   └── response_responses.rs     # Responses API → Anthropic 响应转换
└── providers/
    ├── openai.rs                 # OpenAI/Grok/OpenAI 兼容 HTTP 客户端（Chat Completions）
    ├── chatgpt.rs                # ChatGPT Codex HTTP 客户端（Responses API + OAuth）
    └── anthropic.rs              # Anthropic 兼容 HTTP 客户端（Messages API 直通转发）
```

## 合规声明

ChatGPT OAuth 流程使用 OpenAI 官方的 OAuth 认证方式（与 Codex CLI 相同）。仅供个人开发使用，需搭配用户自己的 ChatGPT Plus/Pro 订阅。用户需自行确保遵守 OpenAI 的使用条款。

## 许可证

MIT
