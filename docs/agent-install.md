# CC-Adapter Installation Guide (for AI Agents)

You are installing **CC-Adapter** (Claude API Adapter) — a local proxy that translates Anthropic Messages API into OpenAI/Grok/ChatGPT format, so Claude Code can use other LLM providers.

Repository: https://github.com/Jakevin/CC-Adapter

## Prerequisites

- macOS (arm64 / amd64), Linux (arm64 / amd64), or Windows (amd64)
- An API key from OpenAI, Grok (xAI), or any OpenAI-compatible provider
  - OR a ChatGPT Plus/Pro subscription (uses OAuth, no API key needed)

## Step 1: Download and Extract

Detect the platform and download the correct binary.

### macOS / Linux

```bash
# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')     # darwin or linux
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
esac

# Download latest release
curl -sL "https://github.com/Jakevin/CC-Adapter/releases/latest/download/claude-adapter-${OS}-${ARCH}.tar.gz" \
  -o /tmp/claude-adapter.tar.gz

# Extract to ~/.local/bin (or any directory in PATH)
mkdir -p ~/.local/bin
tar xzf /tmp/claude-adapter.tar.gz -C /tmp
cp /tmp/claude-adapter/claude-adapter ~/.local/bin/
cp /tmp/claude-adapter/config-example.toml ~/.local/bin/config-example.toml
rm -rf /tmp/claude-adapter /tmp/claude-adapter.tar.gz

# Verify
claude-adapter --help
```

If `~/.local/bin` is not in PATH, add it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Windows (PowerShell)

Download the latest Windows build and extract it to a folder in PATH (example uses `C:\Tools\claude-adapter`):

```powershell
$ErrorActionPreference = "Stop"

$dest = "C:\Tools\claude-adapter"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$zip = Join-Path $env:TEMP "claude-adapter.zip"
Invoke-WebRequest `
  -Uri "https://github.com/Jakevin/CC-Adapter/releases/latest/download/claude-adapter-windows-amd64.zip" `
  -OutFile $zip

Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip -Force

# Verify
& (Join-Path $dest "claude-adapter.exe") --help
```

## Step 2: Create Config

```bash
mkdir -p ~/.config/claude-adapter
cp ~/.local/bin/config-example.toml ~/.config/claude-adapter/config.toml
```

On Windows, you can put `config.toml` anywhere and pass it via `--config` in Step 3.

Edit `~/.config/claude-adapter/config.toml`. The user MUST provide their own values for the following fields. Ask the user which provider they want to use:

### Option A: OpenAI

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"
api_key = "<ASK_USER>"
base_url = "https://api.openai.com/v1"

[models]
default = "gpt-5.4"
```

### Option B: Grok (xAI)

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"
api_key = "<ASK_USER>"
base_url = "https://api.x.ai/v1"

[models]
default = "grok-3"
```

### Option C: ChatGPT Plus/Pro (OAuth, no API key)

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "chatgpt"

[models]
default = "gpt-5.4"
```

After saving config, if using ChatGPT, run the OAuth login:

```bash
claude-adapter login
```

If the user wants multiple ChatGPT accounts, bind them by name:

```bash
# Default account -> [providers.chatgpt]
claude-adapter login

# Second account -> [providers.chatgpt2]
claude-adapter login --name chatgpt2
```

### Option D: Any OpenAI-compatible API

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"
api_key = "<ASK_USER>"
base_url = "<ASK_USER>"

[models]
default = "<ASK_USER>"
```

### Option E (advanced): Multiple providers + routing

For advanced setups, the agent can help the user configure **multiple providers at once** and route each Claude model name to a specific provider/model pair:

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "info"
log_file = "adapter.log"

[providers.chatgpt]
type = "chatgpt"
# ChatGPT uses OAuth, no api_key/base_url needed

[providers.openai-compatible]
type = "openai"
api_key = "<ASK_USER>"                     # e.g. OpenAI / Grok key
base_url = "https://api.openai.com/v1"    # or https://api.x.ai/v1, etc.
supports_streaming = false                # let the adapter simulate SSE

[providers.anthropic-compatible]
type = "anthropic-compatible"
api_key = "<ASK_USER>"
base_url = "<ASK_USER_ANTHROPIC_BASE_URL>"  # e.g. https://your-host/v1
supports_streaming = false

[models]
default_provider = "chatgpt"
default_model = "gpt-5.4"

# Routing table: Anthropic model name → provider + model
# Longest-prefix matching is supported, which is useful for dated model names like
# "claude-haiku-4-5-20251001" (using key "claude-haiku-4-5").
[models.routing]
"claude-opus-4-6"   = { provider = "chatgpt",             model = "gpt-5.4" }
"claude-sonnet-4-6" = { provider = "openai-compatible",   model = "gpt-4.1" }
"claude-haiku-4-5"  = { provider = "anthropic-compatible", model = "<ASK_USER_MODEL>" }
```

The adapter resolves `models.routing` as follows:

1. Exact match on the full model name.
2. If no exact match, use the **longest prefix** key where `incoming_model.starts_with(key)`.
3. If still no match, fall back to `default_provider` + `default_model`.

## Step 3: Start the Adapter

```bash
claude-adapter serve --config ~/.config/claude-adapter/config.toml
```

Or pass the API key via environment variable instead of config:

```bash
ADAPTER_API_KEY=sk-xxx claude-adapter serve --config ~/.config/claude-adapter/config.toml
```

The adapter will:
1. Start listening on `http://127.0.0.1:8080`
2. Automatically configure `~/.claude/settings.json` with `ANTHROPIC_BASE_URL`
3. Hot-reload `config.toml` changes automatically while running
4. Restore the original settings when stopped (Ctrl+C / SIGTERM)

## Step 4: Verify

```bash
curl http://127.0.0.1:8080/health
# Expected: {"status":"ok"}
```

## Step 5: Use with Claude Code

Open a **new terminal** and run:

```bash
claude
```

No extra environment variables needed. The adapter auto-configured everything in Step 3.

## Troubleshooting

- **"connection refused"**: Adapter is not running. Start it first (Step 3).
- **Config changes not taking effect immediately**: Most environments hot-reload `config.toml` automatically. If filesystem watching is unavailable, the adapter will fall back to polling.
- **API key errors**: Check that `api_key` in config.toml is correct, or set `ADAPTER_API_KEY` env var.
- **ChatGPT token expired**: Run `claude-adapter login` again.
- **Port conflict**: Change `port` in config.toml or use `--port <PORT>` flag.

## Docker Alternative

```bash
docker run -d -p 8080:8080 \
  -e ADAPTER_API_KEY=sk-your-key \
  ghcr.io/jakevin/cc-adapter:latest
# Or build from source:
# git clone https://github.com/Jakevin/CC-Adapter.git && cd CC-Adapter
# docker build -t claude-adapter . && docker run -d -p 8080:8080 -e ADAPTER_API_KEY=sk-xxx claude-adapter
```
