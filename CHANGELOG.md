## v0.5.2

### 變更紀錄（繁體中文）

- **Anthropic-compatible base_url 更直覺**：`anthropic-compatible` 供應商的 `base_url` 現在可直接填到 API 前綴（例如 `.../v1`），程式會以 `.../messages` 作為端點組合（避免重複 `/v1`）。
- **更新範例設定**：`config-example.toml` 的 `opencode-go-anthropic.base_url` 更新為包含 `/v1` 的範例。

---

### Change Log (English)

- **More intuitive anthropic-compatible base_url**: For `anthropic-compatible` providers, `base_url` can now include the API prefix (e.g. `.../v1`), and the adapter will POST to `.../messages` (avoids duplicating `/v1`).
- **Updated example config**: `config-example.toml` now uses a `/v1` base URL for `opencode-go-anthropic`.

## v0.5.1

### 變更紀錄（繁體中文）

- **日誌檔案寫入開關**：`config.toml` 的 `[server]` 區段新增 `log_file_enabled`（預設 `true`）。設為 `false` 時，即使有設定 `log_file` 也不會寫入檔案，僅輸出到 console。

---

### Change Log (English)

- **Log file write toggle**: Added `log_file_enabled` (default `true`) under `[server]` in `config.toml`. When set to `false`, logs are not written to file even if `log_file` is set; output goes to console only.

## v0.5.0

### 變更紀錄（繁體中文）

- **即時 config 熱重載**：改用 `notify` 監控 `config.toml` 的檔案系統事件，比單純輪詢修改時間更即時。
- **自動退回 polling**：若檔案監控器在目前環境無法啟動或無法監控目錄，會自動退回原本的 polling 模式。
- **更完整的 graceful shutdown**：伺服器現在同時處理 `SIGINT`（Ctrl+C）與 `SIGTERM`，並在關閉時通知背景 watcher 停止，讓 `~/.claude/settings.json` 的還原更可靠。
- **標記舊 token store API 為 deprecated**：舊的 `save/load/delete` 介面保留向後相容，但現在會明確提示改用 `save_named/load_named/delete_named`。
- **ChatGPT Responses 轉換更貼近 Anthropic 語義**：`thinking` 區塊不再當作普通文字轉送，並補上 `incomplete_details.reason` 與 cache token 使用量映射，讓 `stop_reason` 與 usage 統計更準確。

---

### Change Log (English)

- **Real-time config hot reload**: `config.toml` changes are now monitored with `notify` filesystem events instead of relying only on mtime polling.
- **Automatic polling fallback**: if the filesystem watcher cannot start or cannot watch the target directory in the current environment, the adapter automatically falls back to the original polling mode.
- **More complete graceful shutdown**: the server now handles both `SIGINT` (Ctrl+C) and `SIGTERM`, and notifies the background watcher to stop during shutdown so restoring `~/.claude/settings.json` is more reliable.
- **Legacy token store APIs are now deprecated**: the old `save/load/delete` helpers remain for backward compatibility, but now explicitly point users to `save_named/load_named/delete_named`.
- **ChatGPT Responses conversion now matches Anthropic semantics more closely**: `thinking` blocks are no longer forwarded as plain text, and `incomplete_details.reason` plus cache token usage are now mapped back so `stop_reason` and usage statistics are more accurate.

## v0.4.0

### 變更紀錄（繁體中文）

- **ChatGPT 多帳號支援**：`login/logout` 新增 `--name`（預設 `chatgpt`），可用 `claude-adapter login --name chatgpt2` 新增第二個帳號，並在 `config.toml` 中以 `[providers.chatgpt2]` 使用。
- **Token 依 Provider 名稱分檔保存**：OAuth token 會保存到 `~/.claude-adapter/tokens-<name>.json`（例如 `tokens-chatgpt2.json`），`chatgpt` 仍向後相容讀取舊的 `tokens.json`。

---

### Change Log (English)

- **Multiple ChatGPT accounts**: `login/logout` now support `--name` (default `chatgpt`). Use `claude-adapter login --name chatgpt2` to add a second account and reference it via `[providers.chatgpt2]` in `config.toml`.
- **Per-provider token files**: OAuth tokens are stored as `~/.claude-adapter/tokens-<name>.json` (e.g. `tokens-chatgpt2.json`). The default `chatgpt` provider remains backward compatible with the legacy `tokens.json`.

## v0.3.0

### 變更紀錄（繁體中文）

#### 多 Provider 與路由

- **支援多個 Provider 同時配置**：透過 `config.toml` 的 `[providers.*]` 定義多個供應商（`chatgpt` / `openai` / `anthropic-compatible` 等）。
- **新增 `anthropic-compatible` Provider 類型**：直接透過 Anthropic Messages API 格式轉發到相容後端，只需設定 `base_url` + `api_key`。
- **模型路由表改為結構化**：使用 `[models]` + `[models.routing]`，每個 key 映射到 `{ provider, model }`。
- **支援「最長前綴匹配」模型路由**：`models.routing` 的 key 可以只寫「型號前綴」（如 `claude-haiku-4-5`），自動匹配 `claude-haiku-4-5-20251001` 等帶日期後綴的變體。
- **保留舊版單一 Provider 格式**：仍支援原本的 `[provider]` + `models.mapping`，內部會自動轉成多 Provider 結構。

#### 串流與錯誤處理

- **`supports_streaming` 設定啟用**：在每個 Provider 的 config 裡用 `supports_streaming` 控制是否對後端開啟 stream；預設建議 `false`，由 Adapter 統一模擬 SSE。
- **請求失敗時在錯誤訊息中加入完整 API URL**（OpenAI / ChatGPT / Anthropic-compatible），方便排查 404 / 路徑錯誤。
- **針對 Anthropic-compatible 後端回傳 SSE 的情況，強制 `stream=false` 時期望 JSON 回應；若仍為 SSE，會在錯誤訊息中附上 Raw body 以方便 debug。**

#### thinking 區塊支援

- **回應型別新增 `thinking` 變體**：`MessagesResponse.content` 現在支援 `type: "thinking"`，包含 `thinking` 與可選的 `signature` 欄位。
- **Anthropic-compatible 回應可正確解析含 `thinking` 的 content 陣列**，不再因未知 variant 而失敗。
- **SSE 串流輸出時，`thinking` 會用 Anthropic 官方 SSE 事件格式**（`content_block_start` + `thinking_delta` + `content_block_stop`），而不是當作一般文字輸出；UI 可以選擇隱藏或特殊處理 thinking，而不會顯示 `<thinking>...</thinking>` 純文字。

#### 文件與樣板更新

- **更新 `config-example.toml`**：展示多 Provider 配置（含 `supports_streaming` 與前綴路由註解）。
- **更新 README（英文 / 繁中 / 簡中）**：說明多 Provider 結構、`models.routing` 前綴匹配規則、Anthropic-compatible Provider、thinking 區塊行為。

---

### Change Log (English)

#### Multi-provider & routing

- **Support configuring multiple providers at once** via `[providers.*]` in `config.toml` (e.g. `chatgpt`, `openai`, `anthropic-compatible`).
- **New `anthropic-compatible` provider type**: forwards Anthropic Messages API requests directly to compatible backends by configuring `base_url` + `api_key`.
- **Structured model routing**: use `[models]` + `[models.routing]`, each key maps to `{ provider, model }`.
- **Longest-prefix model routing**: `models.routing` keys can be “family prefixes” (e.g. `claude-haiku-4-5`), automatically matching dated variants like `claude-haiku-4-5-20251001`.
- **Legacy single-provider format still supported**: original `[provider]` + `models.mapping` is normalized internally into the same multi-provider structure.

#### Streaming & error handling

- **`supports_streaming` flag activated per provider**: controls whether the adapter asks the backend for streaming responses; recommended `false` so the adapter simulates SSE uniformly.
- **Include full API URL in provider error messages** (OpenAI / ChatGPT / Anthropic-compatible) to quickly diagnose 404 and path issues.
- **For Anthropic-compatible backends that return SSE while `stream=false`**, parse failures now include the full raw body to aid debugging.

#### Thinking block support

- **Response type extended with `thinking` variant**: `MessagesResponse.content` now supports `type: "thinking"` with `thinking` text and optional `signature`.
- **Anthropic-compatible JSON responses containing `thinking` blocks can be deserialized correctly**, instead of failing on unknown variants.
- **When emitting SSE, thinking blocks are forwarded using Anthropic’s official SSE schema** (`content_block_start` + `thinking_delta` + `content_block_stop`), not as plain text; UIs can hide or specially render thinking instead of showing it inline as `<thinking>...</thinking>`.

#### Docs & templates

- **Updated `config-example.toml`** to showcase multi-provider configuration, `supports_streaming`, and prefix-based routing with detailed comments.
- **Updated README (EN / ZH-HANT / ZH-HANS)** to document multi-provider architecture, `models.routing` resolution rules, Anthropic-compatible providers, and thinking block behavior.

