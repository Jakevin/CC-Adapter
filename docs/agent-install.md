# CC-Adapter Installation Guide (for AI Agents)

You are installing **CC-Adapter** (Claude API Adapter) — a local proxy that translates Anthropic Messages API into OpenAI/Grok/ChatGPT format, so Claude Code can use other LLM providers.

Repository: https://github.com/Jakevin/CC-Adapter

## Prerequisites

- macOS (arm64 / amd64) or Linux (arm64 / amd64)
- An API key from OpenAI, Grok (xAI), or any OpenAI-compatible provider
  - OR a ChatGPT Plus/Pro subscription (uses OAuth, no API key needed)

## Step 1: Download and Extract

Detect the platform and download the correct binary:

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

## Step 2: Create Config

```bash
mkdir -p ~/.config/claude-adapter
cp ~/.local/bin/config-example.toml ~/.config/claude-adapter/config.toml
```

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
3. Restore the original settings when stopped (Ctrl+C)

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
