# Claude API Adapter

[English](README.md) | [繁中版](README_ZH_HANT.md) | [簡中版](README_ZH_HANS.md)

Rust 製の API アダプターです。**Claude Code** が Anthropic の Messages API と各プロバイダー固有の API 形式の間を変換することで、他の LLM プロバイダー（OpenAI、Grok/xAI、ChatGPT Plus/Pro）を利用できるようにします。

## 動作の仕組み

```
                               ┌──[OpenAI Chat API]──▶ OpenAI / Grok
Claude Code ──[Anthropic API]──▶ Adapter (localhost) ─┤
                               └──[Responses API + OAuth]──▶ ChatGPT Codex
```

アダプターはローカル HTTP サーバーとして動作し、次の処理を行います：

1. Anthropic Messages API 形式のリクエストを受け付ける（`POST /v1/messages`）
2. 対象プロバイダーの形式（Chat Completions または Responses API）に変換する
3. 設定されたプロバイダーへ転送する
4. レスポンスを Anthropic 形式に戻す
5. 結果を Claude Code に返す

**対応プロバイダー：**
- **OpenAI** — API キー + Chat Completions API
- **Grok (xAI)** — API キー + Chat Completions API
- **ChatGPT Plus/Pro** — OAuth + Responses API（Codex バックエンド）
- **OpenAI 互換 API** — API キー
- **Anthropic 互換 API** — Anthropic と同じ Messages API、異なる `base_url`

**対応機能：**
- テキストメッセージとマルチターン会話
- Tool Use / Function Calling（往復変換対応）
- システムプロンプト
- 画像入力（base64）
- モデルマッピングの設定
- SSE ストリーミングのシミュレーション（非ストリーミングプロバイダー向け）
- 任意：`[server] claude_stream_idle_timeout_ms` により Claude Code 設定へ `CLAUDE_STREAM_IDLE_TIMEOUT_MS` を注入（終了時にバックアップから復元）
- ファイルシステムウォッチャーによる設定のホットリロード
- SIGINT / SIGTERM / SIGHUP での Graceful シャットダウンと設定の復元
- ChatGPT 用 OAuth 認証（PKCE フロー）

## AI エージェント向けインストール

LLM エージェント（Claude Code、Cursor など）に次のプロンプトをコピー＆ペーストしてください：

```
Install and configure CC-Adapter by following the instructions here:
https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

エージェントに直接取得させる場合：

```bash
curl -s https://raw.githubusercontent.com/Jakevin/CC-Adapter/master/docs/agent-install.md
```

## クイックスタート

### 1. インストール

**方法 A: ビルド済みバイナリをダウンロード（Rust 不要）**

[GitHub Releases](https://github.com/Jakevin/CC-Adapter/releases) から最新リリースをダウンロードし、解凍するだけです：

```bash
tar xzf claude-adapter-<platform>.tar.gz
cd claude-adapter
```

アーカイブにはバイナリと `config-example.toml` のテンプレートが含まれています。

Windows の場合は `claude-adapter-windows-amd64.zip` をダウンロードして展開してください（`claude-adapter.exe` と `config-example.toml` が含まれます）。

**方法 B: ソースからビルド**

```bash
cargo build --release
```

バイナリは `target/release/claude-adapter` に出力されます。

### 2. 設定

`config.toml` で**複数のプロバイダーを同時に**設定し、各 Claude モデル名を特定のプロバイダー／モデルの組み合わせにルーティングできます。

#### マルチプロバイダー設定（推奨、v0.3.0+）

```toml
[server]
host = "127.0.0.1"
port = 8080
log_level = "info"
log_file = "adapter.log"
# log_file_enabled = true   # false にするとログをファイルに書き出さない（既定: true）
# claude_stream_idle_timeout_ms = 300000   # 任意: ms を ~/.claude/settings.json の env に書き込み、終了時に復元。既定 300000（5 分）。0 で注入・バックアップをスキップ

[providers.chatgpt]
type = "chatgpt"
# ChatGPT は OAuth のため api_key/base_url 不要

[providers.openai-compatible]
type = "openai"
# API キー（ADAPTER_API_KEY 環境変数でも指定可能）
api_key = "sk-your-openai-or-grok-key"
# OpenAI 互換 API の Base URL（OpenAI / Grok / その他）
base_url = "https://api.openai.com/v1"
# バックエンドがストリーミング SSE を返すか（通常は false のまま Adapter が SSE をシミュレート）
supports_streaming = false

[providers.opencode-go-anthropic]
type = "anthropic-compatible"
api_key = "sk-your-key"
# Anthropic 互換 Messages API の Base URL
base_url = "https://opencode.ai/zen/go"
# false = バックエンドに単一 JSON を要求（対応している場合は推奨）。
# true = ストリーミングを要求；アダプターは Anthropic Messages の SSE を 1 つのレスポンスに集約する。
# バックエンドが stream=false を無視して SSE を返す場合も、可能な限り同様に集約する。
supports_streaming = false

[models]
# ルーティングに一致しない場合のデフォルトプロバイダー／モデル
default_provider = "chatgpt"
default_model = "gpt-5.4"

# ルーティング表: Anthropic モデル名 → プロバイダー + モデル
# 日付サフィックス付きのモデル名には **最長一致** をサポート
# 例: キー "claude-haiku-4-5" は "claude-haiku-4-5-20251001" に一致
[models.routing]
"claude-sonnet-4-6" = { provider = "openai-compatible", model = "gpt-4.1" }
"claude-opus-4-6"   = { provider = "chatgpt",           model = "gpt-5.4" }
"claude-haiku-4-5"  = { provider = "opencode-go-anthropic", model = "MiniMax-M2.5" }
```

`models.routing` の解決ルール：

1. Anthropic モデル名の完全一致（例: `"claude-opus-4-6"`）
2. 完全一致がなければ、`incoming_model.starts_with(key)` を満たす**最長のプレフィックス**キーを使用  
   日付サフィックス付きモデル（例: `claude-haiku-4-5-20251001`）に最適です
3. それでも一致しなければ `default_provider` + `default_model` にフォールバック

#### 従来の単一プロバイダー設定（引き続き対応）

シンプルな構成では、従来の単一 `[provider]` + `models.mapping` 形式も利用できます：

```toml
[server]
host = "127.0.0.1"
port = 8080

[provider]
type = "openai"        # または "grok" / "chatgpt"
api_key = "sk-your-api-key-here"
base_url = "https://api.openai.com/v1"

[models]
default = "gpt-5.4"

[models.mapping]
"claude-sonnet-4-6" = "gpt-5.4"
"claude-opus-4-6"   = "gpt-5.4"
```

> **注:** 内部では両形式とも同じマルチプロバイダー構造に正規化されるため、移行は自分のペースで問題ありません。

### 3. ログイン（ChatGPT 契約者のみ）

ChatGPT 契約を利用する場合は、まず OAuth ログインフローを実行してください：

```bash
./target/release/claude-adapter login
```

これにより：

1. ブラウザが OpenAI のログインページを開く
2. ログイン後、OAuth トークンを自動で受け取る
3. トークンを `~/.claude-adapter/tokens-chatgpt.json` に保存（従来の `tokens.json` も対応）

トークンは期限切れ時に自動で更新されます。

#### 複数の ChatGPT アカウント

複数の ChatGPT アカウントを異なるプロバイダー名に紐づけられます：

```bash
# デフォルトアカウント → [providers.chatgpt]
./target/release/claude-adapter login

# 2 つ目のアカウント → [providers.chatgpt2]
./target/release/claude-adapter login --name chatgpt2
```

トークンは `~/.claude-adapter/tokens-<name>.json` に個別に保存されます。

### 4. 起動

```bash
# 設定ファイルを使用（デフォルトコマンド）
./target/release/claude-adapter

# serve サブコマンドを明示
./target/release/claude-adapter serve --config config.toml

# CLI 引数で指定
./target/release/claude-adapter serve --api-key sk-xxx --model gpt-5.4

# 環境変数で API キーを指定
ADAPTER_API_KEY=sk-xxx ./target/release/claude-adapter
```

### 5. Claude Code で利用する

起動時に `~/.claude/settings.json` を更新します。少なくともアダプター向けの `ANTHROPIC_BASE_URL` を設定し、`[server] claude_stream_idle_timeout_ms = 0` でない限り、長めのストリームアイドルタイムアウト用に `CLAUDE_STREAM_IDLE_TIMEOUT_MS`（既定 300000 ms）も書き込みます。手動の環境変数やシェルフックは不要です。Graceful 終了時にバックアップからこれらのキーを復元します。

その後、新しいターミナルを開いて次を実行します：

```bash
claude
```

## プロバイダー別の例

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

### ChatGPT Plus/Pro（OAuth）

```bash
# 初回: ログイン
./target/release/claude-adapter login

# その後は直接起動（config.toml で type = "chatgpt"）
./target/release/claude-adapter
```

### OpenAI 互換 API（任意）

```bash
./target/release/claude-adapter serve \
  --api-key your-key \
  --base-url https://your-provider.com/v1 \
  --model your-model-name
```

## Docker

### ビルド

```bash
docker build -t claude-adapter .
```

### 実行

```bash
# OpenAI / Grok — 環境変数で API キーを渡す
docker run -d -p 8080:8080 \
  -e ADAPTER_API_KEY=sk-your-key \
  claude-adapter

# カスタム config.toml をマウント
docker run -d -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter

# ChatGPT OAuth — トークンディレクトリをマウント
# （ホストで先に `claude-adapter login` を実行）
docker run -d -p 8080:8080 \
  -v ~/.claude-adapter:/root/.claude-adapter \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  claude-adapter
```

コンテナはデフォルトで `0.0.0.0:8080` で待ち受けます。Claude Code をアダプターに向けるには：

```bash
export ANTHROPIC_BASE_URL=http://<docker-host>:8080
claude
```

## CLI リファレンス

```
Usage: claude-adapter [OPTIONS] [COMMAND]

Commands:
  serve    Adapter プロキシサーバーを起動（デフォルト）
  login    ChatGPT OAuth ログインフローを実行
  logout   保存済み OAuth トークンを削除
  help     ヘルプを表示

Serve Options:
  -c, --config <CONFIG>      設定ファイルのパス [default: config.toml]
      --host <HOST>          リスンホストを上書き
  -p, --port <PORT>          リスンポートを上書き
      --api-key <API_KEY>    プロバイダー API キーを上書き
      --base-url <BASE_URL>  プロバイダー Base URL を上書き
      --model <MODEL>        デフォルトモデルを上書き

Global Options:
      --log-level <LEVEL>    ログレベル [default: info]
  -h, --help                 ヘルプを表示
```

**API キーの優先順位:** CLI `--api-key` > 環境変数 `ADAPTER_API_KEY` > `config.toml`

## API 変換の詳細

### OpenAI/Grok: リクエストマッピング（Anthropic → Chat Completions）

| Anthropic | OpenAI |
|-----------|--------|
| `system`（トップレベル） | `{role: "system"}` メッセージ |
| `max_tokens` | `max_completion_tokens` |
| `stop_sequences` | `stop` |
| `tool_choice: {type: "auto"}` | `tool_choice: "auto"` |
| `tool_choice: {type: "any"}` | `tool_choice: "required"` |
| `tool_choice: {type: "tool", name}` | `tool_choice: {type: "function", function: {name}}` |
| `tools[].input_schema` | `tools[].function.parameters` |
| コンテンツブロック `tool_use` | `tool_calls[]` |
| コンテンツブロック `tool_result` | `{role: "tool"}` メッセージ |

### ChatGPT: リクエストマッピング（Anthropic → Responses API）

| Anthropic | Responses API |
|-----------|---------------|
| `system` | `instructions` |
| `messages[role=user]` | `input[type=message, role=user]` |
| `messages[role=assistant]` | `input[type=message, role=assistant]` |
| コンテンツブロック `tool_use` | `input[type=function_call]` |
| コンテンツブロック `tool_result` | `input[type=function_call_output]` |
| `tools` | `tools`（function 型） |

### レスポンスマッピング（プロバイダー → Anthropic）

| OpenAI / Responses API | Anthropic |
|------------------------|-----------|
| `finish_reason: "stop"` / `status: "completed"` | `stop_reason: "end_turn"` |
| `finish_reason: "tool_calls"` / function_call 出力あり | `stop_reason: "tool_use"` |
| `finish_reason: "length"` / `status: "incomplete"` | `stop_reason: "max_tokens"` |
| `usage.prompt_tokens` / `usage.input_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` / `usage.output_tokens` | `usage.output_tokens` |

## ヘルスチェック

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

## 現在の制限事項

- Anthropic 互換バックエンドの**ストリーミング**レスポンス：アダプターは **テキスト** の SSE delta を 1 回の応答に集約します。ツール呼び出しのストリーミングはまだ完全には再構成していません。
- サードパーティの Anthropic 互換 API からの thinking ブロックは、SSE 内で適切な `thinking` コンテンツブロックとして転送されます。
  - 通常のテキストとしては表示されませんが、一部の UI やツールでは非表示または無視する場合があります。
- ChatGPT OAuth は公式 Codex CLI と同じフローを使用しており、個人利用を想定しています。

## プロジェクト構成

```
src/
├── main.rs                       # エントリポイント、CLI サブコマンド、サーバー起動
├── config.rs                     # TOML 設定 + clap CLI 解析、マルチプロバイダー＆モデルルーティング
├── server.rs                     # Axum ルートハンドラ、マルチプロバイダー振り分け＆ホットリロード
├── error.rs                      # 統一エラー型（Anthropic 形式）
├── auth/
│   ├── oauth.rs                  # PKCE OAuth フロー（ChatGPT ログイン）
│   ├── callback_server.rs        # ローカル OAuth コールバックサーバー
│   └── token_store.rs           # トークン永続化と有効期限チェック
├── types/
│   ├── anthropic.rs             # Anthropic API serde 型（リクエスト・レスポンス、thinking/tool_use/text）
│   ├── openai.rs                # OpenAI Chat Completions API serde 型
│   └── responses.rs             # OpenAI Responses API serde 型
├── convert/
│   ├── anthropic_sse.rs          # Anthropic Messages SSE を 1 つの MessagesResponse に集約
│   ├── request.rs                # Anthropic → Chat Completions リクエスト変換
│   ├── response.rs               # Chat Completions → Anthropic レスポンス変換
│   ├── request_responses.rs      # Anthropic → Responses API リクエスト変換
│   └── response_responses.rs     # Responses API → Anthropic レスポンス変換
└── providers/
    ├── openai.rs                 # OpenAI/Grok/OpenAI 互換 HTTP クライアント（Chat Completions）
    ├── chatgpt.rs                # ChatGPT Codex HTTP クライアント（Responses API + OAuth）
    └── anthropic.rs             # Anthropic 互換 HTTP クライアント（Messages API パススルー）
```

## コンプライアンスに関する注意

ChatGPT OAuth フローは OpenAI 公式の OAuth 認証（Codex CLI と同じ）を使用しています。ご自身の ChatGPT Plus/Pro 契約による個人開発利用を想定しています。利用は OpenAI の利用規約に従う責任はユーザーにあります。

## ライセンス

MIT
