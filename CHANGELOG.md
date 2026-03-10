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

