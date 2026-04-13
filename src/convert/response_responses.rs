use anyhow::{Context, Result};
use serde_json::Value;
use tracing::warn;
use uuid::Uuid;

use crate::types::anthropic::{MessagesResponse, ResponseContentBlock, Usage};
use crate::types::responses::{OutputContent, OutputItem, ResponsesResponse};

/// 從 Codex 後端 SSE 串流文字中解析完整回應，轉換為 Anthropic 格式
/// Parse a complete response from Codex backend SSE stream text, convert to Anthropic format
pub fn convert_responses_to_anthropic(
    sse_text: &str,
    original_model: &str,
) -> Result<MessagesResponse> {
    let mut response = parse_sse_to_response(sse_text)?;

    // Codex 有時 response.completed 的 output:[] 是空的，實際 output item 只在 SSE 串流事件裡
    // Codex sometimes returns output:[] in response.completed; real items are only in SSE events
    if response.output.is_empty() {
        let event_items = collect_output_items_from_events(sse_text);
        if !event_items.is_empty() {
            response.output = event_items;
        }
    }

    let mut msg = convert_parsed_response(response, original_model)?;
    let completed_raw = last_completed_response_json(sse_text);

    // completed 裡的 message.content 有時不含可解析的 output_text（或僅在串流事件帶全文）
    // Some backends omit parseable output_text in completed payload; full text may only exist in stream events
    if assistant_visible_text_is_empty(&msg.content) {
        let recovered = completed_raw
            .as_ref()
            .map(|v| collect_visible_text_from_response_value(v))
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                completed_raw
                    .as_ref()
                    .map(collect_visible_text_from_response_root_fallback)
                    .filter(|s| !s.trim().is_empty())
            })
            .or_else(|| {
                completed_raw.as_ref().and_then(|v| {
                    longest_assistive_text_anywhere_in_response(v, msg.usage.output_tokens)
                })
            })
            .or_else(|| scrape_max_visible_text_from_sse_json_blobs(sse_text))
            .or_else(|| extract_visible_text_from_sse_events(sse_text));
        if let Some(text) = recovered {
            msg.content = vec![ResponseContentBlock::Text { text }];
        }
    }

    // 僅 tool 呼叫時 struct 可能變成空 message；從原始 output 再掃 function_call / tool_call
    // Tool-only turns may deserialize as an empty message; re-scan raw output for function_call / tool_call
    if assistant_visible_text_is_empty(&msg.content) {
        if let Some(ref v) = completed_raw {
            let tools = extract_tool_use_blocks_from_response_value(v);
            if !tools.is_empty() {
                msg.content = tools;
                msg.stop_reason = Some("tool_use".to_string());
            }
        }
    }

    // Codex 的 function call 有時只在 SSE output_item.done 裡，completed.output 是空的；再掃一次 SSE 事件
    // Codex may only put function_call in SSE output_item.done events; re-scan events as last resort
    if assistant_visible_text_is_empty(&msg.content) {
        let tools = collect_tool_use_blocks_from_events(sse_text);
        if !tools.is_empty() {
            msg.content = tools;
            msg.stop_reason = Some("tool_use".to_string());
        }
    }

    // 無論是否已有文字，都補入 SSE 的工具呼叫（模型同時輸出 thinking text + function call 時）
    // Always merge SSE tool calls regardless of existing text (model may output thinking text + tool call)
    {
        let sse_tools = collect_tool_use_blocks_from_events(sse_text);
        if !sse_tools.is_empty() {
            // 移除初始化時插入的空文字佔位符（保留真實文字）
            // Remove empty-text placeholder blocks (keep real text)
            msg.content.retain(|b| !matches!(b, ResponseContentBlock::Text { text } if text.trim().is_empty()));
            // 將 SSE 工具加入（若已存在相同 id 的 tool_use 則不重複）
            // Add SSE tools, avoiding duplicates by id
            let existing_ids: std::collections::HashSet<String> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ResponseContentBlock::ToolUse { id, .. } => Some(id.clone()),
                    _ => None,
                })
                .collect();
            for tool in sse_tools {
                if let ResponseContentBlock::ToolUse { ref id, .. } = tool {
                    if !existing_ids.contains(id) {
                        msg.content.push(tool);
                    }
                }
            }
            // 若有工具呼叫，stop_reason 一律設為 tool_use
            let has_tool = msg
                .content
                .iter()
                .any(|b| matches!(b, ResponseContentBlock::ToolUse { .. }));
            if has_tool {
                msg.stop_reason = Some("tool_use".to_string());
            }
        }
    }

    if assistant_visible_text_is_empty(&msg.content) {
        if let Some(ref v) = completed_raw {
            let raw = serde_json::to_string(v).unwrap_or_default();
            let top_keys: Vec<String> = v
                .as_object()
                .map(|m| {
                    m.keys()
                        .filter(|k| k.as_str() != "instructions")
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            let output_preview = v
                .get("output")
                .map(|o| {
                    let s = serde_json::to_string_pretty(o).unwrap_or_default();
                    s.chars().take(2000).collect::<String>()
                })
                .unwrap_or_else(|| "<欄位不存在 / field absent>".to_string());
            // top-level `text` 與 `reasoning` 欄位（前 500 字元）
            let text_field = v
                .get("text")
                .map(|x| serde_json::to_string(x).unwrap_or_default())
                .map(|s| s.chars().take(500).collect::<String>())
                .unwrap_or_else(|| "<absent>".to_string());
            let reasoning_field = v
                .get("reasoning")
                .map(|x| serde_json::to_string(x).unwrap_or_default())
                .map(|s| s.chars().take(500).collect::<String>())
                .unwrap_or_else(|| "<absent>".to_string());
            // SSE 串流中出現的事件類型統計
            let mut sse_event_counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for_each_sse_block(sse_text, |et, data| {
                let jtype = serde_json::from_str::<serde_json::Value>(data)
                    .ok()
                    .and_then(|v| v.get("type").and_then(|x| x.as_str()).map(str::to_string));
                let key = if et.is_empty() {
                    jtype.unwrap_or_else(|| "<no_type>".to_string())
                } else {
                    et.to_string()
                };
                *sse_event_counts.entry(key).or_insert(0) += 1;
            });
            warn!(
                response_chars = raw.len(),
                "Codex 仍無可見文字 / Still no visible text\n\
                 頂層 keys (excl. instructions): {:?}\n\
                 output 欄位 / output field:\n{}\n\
                 text 欄位 / text field: {}\n\
                 reasoning 欄位 / reasoning field: {}\n\
                 SSE 事件統計 / SSE event counts: {:?}",
                top_keys,
                output_preview,
                text_field,
                reasoning_field,
                sse_event_counts
            );
        }
    }

    Ok(msg)
}

/// 在整份 `response` JSON 內遞迴找最長敘述字串；並依 `output_tokens` 要求最小長度，避免把 `verbosity: "detailed"` 等 metadata 當成正文
/// Longest narrative string in `response` JSON; enforces min length vs output_tokens to avoid metadata false positives
fn longest_assistive_text_anywhere_in_response(resp: &Value, output_tokens: u32) -> Option<String> {
    let mut best: Option<String> = None;
    visit_longest_assistive_string(resp, &mut best, 0, 20, false);
    let s = best.filter(|s| !s.trim().is_empty())?;
    let t = s.trim();
    if is_assistive_noise_token(t) {
        return None;
    }
    let min_chars = plausible_min_chars_for_longest_fallback(output_tokens);
    if t.len() < min_chars {
        return None;
    }
    Some(s)
}

/// API 設定裡常見的短字串，易被遞迴掃到誤當成 assistant 正文
/// Short API config strings often mistaken for assistant text when scraping JSON
fn is_assistive_noise_token(t: &str) -> bool {
    matches!(
        t.to_ascii_lowercase().as_str(),
        "detailed"
            | "medium"
            | "low"
            | "high"
            | "xhigh"
            | "auto"
            | "minimal"
            | "none"
            | "default"
            | "verbose"
            | "concise"
            | "text"
            | "json"
            | "markdown"
    )
}

/// 有 N 個 output token 時，合理正文至少應有的字元數下限（粗略；避免 metadata 單字）
/// Lower bound on chars for N output tokens (rough; filters single-token metadata)
fn plausible_min_chars_for_longest_fallback(output_tokens: u32) -> usize {
    if output_tokens <= 12 {
        return 2;
    }
    let est = (output_tokens as usize).saturating_mul(11) / 20;
    est.clamp(18, 200)
}

fn visit_longest_assistive_string(
    v: &Value,
    best: &mut Option<String>,
    depth: usize,
    max_depth: usize,
    inside_arguments: bool,
) {
    if depth > max_depth || inside_arguments {
        return;
    }

    match v {
        Value::Object(m) => {
            for (k, child) in m {
                let k = k.as_str();
                if matches!(
                    k,
                    "encrypted_content"
                        | "ciphertext"
                        | "encryption_key"
                        | "encryption_nonce"
                ) {
                    continue;
                }
                // 回應內嵌的請求 instructions（可達十萬字以上），絕非助理正文
                // Echoed request `instructions` in response body — not assistant-visible text
                if matches!(k, "instructions" | "input") {
                    continue;
                }
                if matches!(
                    k,
                    "input_schema" | "schema" | "definitions" | "properties" | "usage"
                ) {
                    continue;
                }

                let in_args = k == "arguments";

                // 略過 text/verbosity 等設定區塊底下的字串（例如 verbosity: detailed）
                // Skip strings under config-like keys (e.g. verbosity: detailed)
                if matches!(k, "verbosity" | "effort" | "format") {
                    continue;
                }

                if matches!(k, "text" | "summary" | "refusal" | "thinking") {
                    if let Some(s) = child.as_str() {
                        consider_longer_string(best, s);
                    }
                }
                if matches!(k, "content" | "value" | "body" | "markdown" | "delta" | "output_text")
                {
                    if let Some(s) = child.as_str() {
                        consider_longer_string(best, s);
                    }
                }

                visit_longest_assistive_string(
                    child,
                    best,
                    depth + 1,
                    max_depth,
                    inside_arguments || in_args,
                );
            }
        }
        Value::Array(a) => {
            for x in a {
                visit_longest_assistive_string(x, best, depth + 1, max_depth, inside_arguments);
            }
        }
        _ => {}
    }
}

fn consider_longer_string(best: &mut Option<String>, s: &str) {
    let t = s.trim();
    if t.len() < 2 {
        return;
    }
    if is_assistive_noise_token(t) {
        return;
    }
    let replace = match best {
        None => true,
        Some(b) => t.len() > b.trim().len(),
    };
    if replace {
        *best = Some(s.to_string());
    }
}

/// 從原始 `response` JSON 的 `output` 擷取 Anthropic `tool_use` 區塊（略過 serde Unknown）
/// Extract Anthropic tool_use blocks from raw `response` output (skips serde Unknown losses)
fn extract_tool_use_blocks_from_response_value(resp: &Value) -> Vec<ResponseContentBlock> {
    let mut out = Vec::new();
    let items: Vec<&Value> = match resp.get("output") {
        Some(Value::Array(a)) => a.iter().collect(),
        Some(v) if v.as_object().is_some() => vec![v],
        _ => return out,
    };
    for item in items {
        let t = item.get("type").and_then(|x| x.as_str());
        if !matches!(
            t,
            Some("function_call") | Some("tool_call") | Some("custom_tool_call")
        ) {
            continue;
        }
        let id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .or_else(|| item.get("tool_call_id"))
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("call_unknown")
            .to_string();
        let name = item
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let args_val = item.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));
        let arguments = match args_val {
            Value::String(s) => s,
            _ => serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string()),
        };
        let input: Value =
            serde_json::from_str(&arguments).unwrap_or_else(|_| Value::Object(Default::default()));
        out.push(ResponseContentBlock::ToolUse { id, name, input });
    }
    out
}

/// 合併單一 SSE 區塊內多行 `data:`（RFC 8895），避免 JSON 被截斷後無法解析
/// Merge multiple `data:` lines in one SSE block (RFC 8895) so split JSON still parses
fn merge_sse_event_block(block: &str) -> (String, String) {
    let mut event_type = String::new();
    let mut data_parts: Vec<String> = Vec::new();
    for line in block.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event_type = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            let body = rest.trim_start_matches(' ');
            data_parts.push(body.to_string());
        }
    }
    (event_type, data_parts.concat())
}

fn for_each_sse_block(sse_text: &str, mut f: impl FnMut(&str, &str)) {
    let normalized = sse_text.replace("\r\n", "\n");
    for block in normalized.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let (event_type, data) = merge_sse_event_block(block);
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        f(&event_type, &data);
    }
}

/// 從 `response` 物件的 `output` 擷取可顯示文字（略過嚴格 struct 反序列化差異）
/// Extract visible text from `output` on a `response` object (tolerates Codex JSON quirks)
fn collect_visible_text_from_response_value(resp: &Value) -> String {
    let mut out = Vec::new();
    let items: Vec<&Value> = match resp.get("output") {
        Some(Value::Array(a)) => a.iter().collect(),
        Some(v) if v.as_object().is_some() => vec![v],
        _ => return String::new(),
    };
    for item in items {
        let item_type = item.get("type").and_then(|t| t.as_str());
        if item_type == Some("reasoning") {
            match item.get("summary") {
                Some(Value::String(s)) if !s.trim().is_empty() => {
                    out.push(s.clone());
                }
                Some(Value::Array(parts)) => {
                    // Codex 有時 summary 是 [{type:"summary_text",text:"..."},...] 陣列
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
                            if !t.trim().is_empty() && !is_assistive_noise_token(t.trim()) {
                                out.push(t.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        if matches!(
            item_type,
            Some("function_call" | "tool_call" | "custom_tool_call")
        ) {
            continue;
        }
        let len_before = out.len();
        append_texts_from_message_content(item.get("content"), &mut out);
        if let Some(s) = item.get("text").and_then(|x| x.as_str()) {
            if !s.trim().is_empty() {
                out.push(s.to_string());
            }
        }
        // Codex 有時把正文拆成多段短字串，單段不滿 longest 後備門檻；改為在整個 output item 內依序拼接
        // Codex may shard visible text; single fragments fail `longest_assistive` min-length — concat shards in-order
        if out.len() == len_before {
            collect_text_shards_from_output_item(item, &mut out, 0, 24);
        }
    }
    out.concat()
}

/// 不依賴 `output` 形狀的最後保底：在整份 response 內收集可見文字片段（排除 request echo、schema、tool args）
/// Last-resort extraction when `output` shape drifts: collect visible text shards from full response
fn collect_visible_text_from_response_root_fallback(resp: &Value) -> String {
    let mut out = Vec::new();
    collect_visible_text_shards_from_response_root(resp, &mut out, 0, 28, false);
    out.concat()
}

fn collect_visible_text_shards_from_response_root(
    v: &Value,
    out: &mut Vec<String>,
    depth: usize,
    max_depth: usize,
    inside_arguments: bool,
) {
    if depth > max_depth || inside_arguments {
        return;
    }

    match v {
        Value::Object(m) => {
            for (k, child) in m {
                let k = k.as_str();
                if matches!(
                    k,
                    "instructions"
                        | "input"
                        | "tools"
                        | "messages"
                        | "system"
                        | "input_schema"
                        | "schema"
                        | "definitions"
                        | "properties"
                        | "usage"
                        | "encrypted_content"
                        | "ciphertext"
                        | "encryption_key"
                        | "encryption_nonce"
                ) {
                    continue;
                }
                if matches!(k, "verbosity" | "effort" | "format") {
                    continue;
                }
                if matches!(k, "id" | "call_id" | "tool_call_id" | "name" | "title" | "type") {
                    continue;
                }
                let in_args = k == "arguments";
                if matches!(k, "text" | "output_text" | "summary" | "delta" | "content" | "value") {
                    match child {
                        Value::String(s) => {
                            let t = s.trim();
                            if !t.is_empty() && !is_assistive_noise_token(t) {
                                out.push(s.clone());
                            }
                        }
                        Value::Object(o) => {
                            if let Some(s) = o.get("value").and_then(|x| x.as_str()) {
                                let t = s.trim();
                                if !t.is_empty() && !is_assistive_noise_token(t) {
                                    out.push(s.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                collect_visible_text_shards_from_response_root(
                    child,
                    out,
                    depth + 1,
                    max_depth,
                    in_args,
                );
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_visible_text_shards_from_response_root(
                    x,
                    out,
                    depth + 1,
                    max_depth,
                    inside_arguments,
                );
            }
        }
        _ => {}
    }
}

/// 在單一 `output` 項目中遞迴收集短字串片段（略過工具／加密／請求回顯）
/// Recursively collect short text shards inside one `output` item (skips tools, crypto, request echo)
fn collect_text_shards_from_output_item(
    v: &Value,
    out: &mut Vec<String>,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        return;
    }
    match v {
        Value::Object(m) => {
            for (k, child) in m {
                let k = k.as_str();
                if matches!(
                    k,
                    "arguments"
                        | "instructions"
                        | "input"
                        | "encrypted_content"
                        | "ciphertext"
                        | "encryption_key"
                        | "encryption_nonce"
                        | "input_schema"
                ) {
                    continue;
                }
                if matches!(k, "verbosity" | "effort" | "format") {
                    continue;
                }
                if matches!(k, "id" | "call_id" | "tool_call_id" | "name" | "title" | "type") {
                    continue;
                }
                if matches!(k, "text" | "output_text" | "summary" | "delta") {
                    match child {
                        Value::String(s) => {
                            let t = s.trim();
                            if !t.is_empty() && !is_assistive_noise_token(t) {
                                out.push(s.clone());
                            }
                        }
                        Value::Object(o) => {
                            if let Some(s) = o.get("value").and_then(|x| x.as_str()) {
                                let t = s.trim();
                                if !t.is_empty() && !is_assistive_noise_token(t) {
                                    out.push(s.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                collect_text_shards_from_output_item(child, out, depth + 1, max_depth);
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_text_shards_from_output_item(x, out, depth + 1, max_depth);
            }
        }
        _ => {}
    }
}

/// 掃描整段 SSE 內每個可解析的 JSON，對含 `output` 的物件取可見文字，回傳最長的一筆（Codex 有時事件名或包裝與預期不同）
/// Scan every JSON blob in SSE; collect visible text from objects with `output`; return longest match
fn scrape_max_visible_text_from_sse_json_blobs(sse_text: &str) -> Option<String> {
    let mut best: Option<String> = None;
    for_each_sse_block(sse_text, |_et, data| {
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let mut candidates: Vec<String> = Vec::new();
        if v.get("output").is_some() {
            candidates.push(collect_visible_text_from_response_value(&v));
        }
        if let Some(r) = v.get("response") {
            candidates.push(collect_visible_text_from_response_value(r));
        }
        for s in candidates {
            let t = s.trim();
            if t.is_empty() {
                continue;
            }
            let replace = match &best {
                None => true,
                Some(prev) => t.len() > prev.len(),
            };
            if replace {
                best = Some(t.to_string());
            }
        }
    });
    best
}

fn append_texts_from_message_content(content: Option<&Value>, out: &mut Vec<String>) {
    let Some(c) = content else {
        return;
    };
    if let Some(s) = c.as_str() {
        let t = s.trim();
        if !t.is_empty() && !is_assistive_noise_token(t) {
            out.push(s.to_string());
        }
        return;
    }
    if let Some(arr) = c.as_array() {
        for part in arr {
            extract_text_from_content_part(part, out);
        }
        return;
    }
    // Codex 有時 `content` 是單一物件而非陣列
    // Codex sometimes sends `content` as a single object instead of an array
    if c.is_object() {
        extract_text_from_content_part(c, out);
    }
}

fn extract_text_from_content_part(part: &Value, out: &mut Vec<String>) {
    if let Some(s) = part.get("refusal").and_then(|x| x.as_str()) {
        if !s.trim().is_empty() {
            out.push(s.to_string());
        }
    }

    match part.get("text") {
        Some(Value::String(s)) if !s.trim().is_empty() => {
            let t = s.trim();
            if !is_assistive_noise_token(t) {
                out.push(s.clone());
            }
            return;
        }
        Some(Value::Object(o)) => {
            if let Some(s) = o.get("value").and_then(|x| x.as_str()) {
                let t = s.trim();
                if !t.is_empty() && !is_assistive_noise_token(t) {
                    out.push(s.to_string());
                    return;
                }
            }
        }
        _ => {}
    }

    if let Some(s) = part.get("output_text").and_then(|x| x.as_str()) {
        let t = s.trim();
        if !t.is_empty() && !is_assistive_noise_token(t) {
            out.push(s.to_string());
        }
    }

    let ptype = part.get("type").and_then(|x| x.as_str());
    if matches!(ptype, Some("output_text") | Some("text")) {
        if let Value::Object(o) = part {
            if let Some(Value::String(s)) = o.get("content") {
                let t = s.trim();
                if !t.is_empty() && !is_assistive_noise_token(t) {
                    out.push(s.clone());
                }
            }
        }
    }
}

/// 取得最後一則 `response.completed` / `response.done` 的原始 `response` JSON
fn last_completed_response_json(sse_text: &str) -> Option<Value> {
    let mut last: Option<Value> = None;
    for_each_sse_block(sse_text, |event_type, data| {
        let Ok(json) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let jtype = json.get("type").and_then(|x| x.as_str());
        let is_completed = event_type == "response.completed"
            || event_type == "response.done"
            || jtype == Some("response.completed")
            || jtype == Some("response.done");
        if !is_completed {
            return;
        }
        let response_json = if json.get("response").is_some() {
            json["response"].clone()
        } else {
            json
        };
        last = Some(response_json);
    });
    last
}

/// 是否沒有可顯示的 assistant 文字（且無 tool_use，避免覆蓋工具回合）
/// Whether there is no visible assistant text (and no tool_use, to avoid breaking tool turns)
fn assistant_visible_text_is_empty(content: &[ResponseContentBlock]) -> bool {
    let has_tool = content
        .iter()
        .any(|b| matches!(b, ResponseContentBlock::ToolUse { .. }));
    if has_tool {
        return false;
    }
    let joined: String = content
        .iter()
        .filter_map(|b| match b {
            ResponseContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .concat();
    joined.trim().is_empty()
}

/// 從 SSE 串流事件擷取模型文字（completed 內容為空時的後備）
/// Extract model-visible text from SSE events (fallback when completed body has no text)
fn extract_visible_text_from_sse_events(sse_text: &str) -> Option<String> {
    let mut done_texts: Vec<String> = Vec::new();
    let mut delta_acc = String::new();
    let mut content_part_texts: Vec<String> = Vec::new();

    for_each_sse_block(sse_text, |event_type, data_line| {
        let Ok(v) = serde_json::from_str::<Value>(data_line) else {
            return;
        };

        let et = if event_type.is_empty() {
            v.get("type").and_then(|x| x.as_str()).unwrap_or("")
        } else {
            event_type
        };

        match et {
            "response.output_text.done" => {
                if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                    done_texts.push(t.to_string());
                }
            }
            "response.output_text.delta" => {
                match v.get("delta") {
                    Some(Value::String(s)) => delta_acc.push_str(s),
                    Some(Value::Object(o)) => {
                        if let Some(s) = o.get("text").and_then(|x| x.as_str()) {
                            delta_acc.push_str(s);
                        }
                        if let Some(s) = o.get("output_text").and_then(|x| x.as_str()) {
                            delta_acc.push_str(s);
                        }
                    }
                    _ => {}
                }
            }
            "response.content_part.done" => {
                if let Some(part) = v.get("part") {
                    let ptype = part.get("type").and_then(|x| x.as_str());
                    if matches!(ptype, Some("text") | Some("output_text")) {
                        if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
                            content_part_texts.push(t.to_string());
                        }
                    }
                }
            }
            "response.output_item.done" => {
                if let Some(item) = v.get("item") {
                    let wrapped = serde_json::json!({ "output": [item] });
                    let s = collect_visible_text_from_response_value(&wrapped);
                    if !s.trim().is_empty() {
                        done_texts.push(s);
                    }
                }
            }
            _ => {}
        }
    });

    if !done_texts.is_empty() {
        return Some(done_texts.concat());
    }
    if !content_part_texts.is_empty() {
        return Some(content_part_texts.concat());
    }
    if !delta_acc.is_empty() {
        return Some(delta_acc);
    }

    let greedy = greedy_collect_stream_text_events(sse_text);
    if !greedy.trim().is_empty() {
        return Some(greedy);
    }
    None
}

/// 依事件／JSON `type` 字串猜測與輸出文字相關的事件，從常見欄位撈字串（最後手段，避免 Codex 事件命名與官方文件不一致）
/// Heuristic scrape for stream events that look text-related when Codex event names diverge from docs
fn greedy_collect_stream_text_events(sse_text: &str) -> String {
    let mut acc = String::new();
    for_each_sse_block(sse_text, |event_type, data| {
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let jt = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        let label = format!("{event_type} {jt}");
        if !label.contains("output_text")
            && !label.contains("text_delta")
            && !label.contains("content_part")
            && !label.contains("output_item")
        {
            return;
        }
        if let Some(s) = v.get("text").and_then(|x| x.as_str()) {
            acc.push_str(s);
        }
        if let Some(d) = v.get("delta") {
            match d {
                Value::String(s) => acc.push_str(s),
                Value::Object(o) => {
                    if let Some(s) = o.get("text").and_then(|x| x.as_str()) {
                        acc.push_str(s);
                    }
                }
                _ => {}
            }
        }
        if let Some(p) = v.get("part").and_then(|x| x.as_object()) {
            if let Some(s) = p.get("text").and_then(|x| x.as_str()) {
                acc.push_str(s);
            }
        }
        if let Some(item) = v.get("item") {
            let wrapped = serde_json::json!({ "output": [item.clone()] });
            acc.push_str(&collect_visible_text_from_response_value(&wrapped));
        }
    });
    acc
}

/// 從 SSE 事件流中找到 response.completed 事件並解析完整回應
/// Find the response.completed event in the SSE stream and parse the complete response
fn parse_sse_to_response(sse_text: &str) -> Result<ResponsesResponse> {
    let mut last_response: Option<ResponsesResponse> = None;
    let normalized = sse_text.replace("\r\n", "\n");

    for block in normalized.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let (event_type, data_line) = merge_sse_event_block(block);
        if data_line.is_empty() || data_line == "[DONE]" {
            continue;
        }

        let json: Value = serde_json::from_str(&data_line).with_context(|| {
            format!(
                "無法解析 SSE 事件資料 / Failed to parse SSE event data: {}",
                event_type
            )
        })?;

        let jtype = json.get("type").and_then(|x| x.as_str());
        let is_completed = event_type == "response.completed"
            || event_type == "response.done"
            || jtype == Some("response.completed")
            || jtype == Some("response.done");
        if !is_completed {
            continue;
        }

        let response_json = if json.get("response").is_some() {
            json["response"].clone()
        } else {
            json
        };

        let response: ResponsesResponse = serde_json::from_value(response_json.clone())
            .with_context(|| {
                let preview: String = response_json.to_string().chars().take(500).collect();
                format!(
                    "無法解析 Responses 回應 / Failed to parse Responses response (preview: {})",
                    preview
                )
            })?;

        last_response = Some(response);
    }

    if last_response.is_none() {
        last_response = try_reconstruct_from_events(sse_text);
    }

    last_response.context(
        "SSE 串流中未找到完整回應 / No complete response found in SSE stream",
    )
}

/// 從 SSE `response.output_item.done` 事件直接建立 Anthropic tool_use 區塊（完整 fallback）
/// Build Anthropic tool_use blocks directly from SSE `response.output_item.done` events (full fallback)
fn collect_tool_use_blocks_from_events(sse_text: &str) -> Vec<ResponseContentBlock> {
    let mut out = Vec::new();
    for_each_sse_block(sse_text, |event_type, data_line| {
        if event_type != "response.output_item.done" {
            return;
        }
        let Ok(json) = serde_json::from_str::<Value>(data_line) else {
            return;
        };
        let item = json.get("item").cloned().unwrap_or(json.clone());
        let t = item.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if !matches!(t, "function_call" | "tool_call" | "custom_tool_call") {
            return;
        }
        let id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .or_else(|| item.get("tool_call_id"))
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("call_unknown")
            .to_string();
        let name = item
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let args_val = item
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let arguments = match args_val {
            Value::String(s) => s,
            _ => serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string()),
        };
        let input: Value =
            serde_json::from_str(&arguments).unwrap_or(Value::Object(Default::default()));
        out.push(ResponseContentBlock::ToolUse { id, name, input });
    });
    out
}

/// 從 SSE `response.output_item.done` 事件收集 OutputItem（用於補齊 completed 裡的空 output）
/// Collect OutputItems from SSE `response.output_item.done` events (patches empty output in completed)
fn collect_output_items_from_events(sse_text: &str) -> Vec<OutputItem> {
    let mut items: Vec<OutputItem> = Vec::new();
    for_each_sse_block(sse_text, |event_type, data_line| {
        if event_type != "response.output_item.done" {
            return;
        }
        let Ok(json) = serde_json::from_str::<Value>(data_line) else {
            return;
        };
        let item_val = json.get("item").cloned().unwrap_or(json.clone());
        if let Ok(item) = serde_json::from_value::<OutputItem>(item_val) {
            items.push(item);
        }
    });
    items
}

/// 嘗試從各個 SSE 事件中重建回應（當缺少 completed 事件時的後備方案）
/// Try to reconstruct a response from individual SSE events (fallback when completed is missing)
fn try_reconstruct_from_events(sse_text: &str) -> Option<ResponsesResponse> {
    let mut response_id = String::new();
    let mut model = String::new();
    let mut output_items: Vec<OutputItem> = Vec::new();
    let mut usage = None;
    let mut found_created = false;

    for_each_sse_block(sse_text, |event_type, data_line| {
        let Ok(json) = serde_json::from_str::<Value>(data_line) else {
            return;
        };

        match event_type {
            "response.created" => {
                found_created = true;
                if let Some(resp) = json.get("response") {
                    response_id = resp["id"].as_str().unwrap_or("").to_string();
                    model = resp["model"].as_str().unwrap_or("").to_string();
                }
            }
            "response.output_item.done" => {
                if let Ok(item) = serde_json::from_value::<OutputItem>(json["item"].clone()) {
                    output_items.push(item);
                }
            }
            "response.usage" | "response.completed" => {
                if let Some(u) = json.get("usage") {
                    usage = serde_json::from_value(u.clone()).ok();
                }
            }
            _ => {}
        }
    });

    if !found_created {
        return None;
    }

    Some(ResponsesResponse {
        id: if response_id.is_empty() {
            format!("resp_{}", Uuid::new_v4().simple())
        } else {
            response_id
        },
        status: "completed".to_string(),
        incomplete_details: None,
        model,
        output: output_items,
        usage,
    })
}

/// 將已解析的 Responses API 回應轉換為 Anthropic Messages 回應
/// Convert a parsed Responses API response into an Anthropic Messages response
fn convert_parsed_response(
    resp: ResponsesResponse,
    original_model: &str,
) -> Result<MessagesResponse> {
    let mut content: Vec<ResponseContentBlock> = Vec::new();

    for item in &resp.output {
        match item {
            OutputItem::Message {
                content: parts,
                ..
            } => {
                for part in parts {
                    match part {
                        OutputContent::Text { text } => {
                            content.push(ResponseContentBlock::Text { text: text.clone() });
                        }
                        OutputContent::Unknown => {}
                    }
                }
            }
            OutputItem::FunctionCall {
                name,
                arguments,
                call_id,
            } => {
                let input: serde_json::Value =
                    serde_json::from_str(arguments).unwrap_or(serde_json::Value::Object(
                        serde_json::Map::new(),
                    ));
                content.push(ResponseContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                });
            }
            OutputItem::Unknown => {}
        }
    }

    // 若內容為空，插入空文字區塊
    // If content is empty, push an empty text block
    if content.is_empty() {
        content.push(ResponseContentBlock::Text {
            text: String::new(),
        });
    }

    let stop_reason = convert_status_to_stop_reason(
        &resp.status,
        resp.incomplete_details.as_ref().and_then(|d| d.reason.as_deref()),
        &content,
    );

    let usage = match &resp.usage {
        Some(u) => Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_read_input_tokens: u
                .cache_read_input_tokens
                .or_else(|| u.input_tokens_details.as_ref().map(|d| d.cached_tokens))
                .or_else(|| u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens)),
        },
        None => Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    };

    let id = format!("msg_{}", Uuid::new_v4().simple());

    Ok(MessagesResponse {
        id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: original_model.to_string(),
        content,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage,
    })
}

/// 將 Responses API status 映射為 Anthropic stop_reason
/// Map Responses API status to Anthropic stop_reason
fn convert_status_to_stop_reason(
    status: &str,
    incomplete_reason: Option<&str>,
    content: &[ResponseContentBlock],
) -> String {
    let has_tool_use = content.iter().any(|b| matches!(b, ResponseContentBlock::ToolUse { .. }));

    if has_tool_use {
        return "tool_use".to_string();
    }

    match status {
        "completed" => "end_turn".to_string(),
        "incomplete" | "truncated" => match incomplete_reason {
            Some("max_output_tokens") | Some("max_tokens") | None => "max_tokens".to_string(),
            _ => "end_turn".to_string(),
        },
        "cancelled" => "end_turn".to_string(),
        _ => "end_turn".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_completed_event() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_abc","status":"completed","model":"gpt-5-codex","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();

        assert_eq!(result.response_type, "message");
        assert_eq!(result.role, "assistant");
        assert_eq!(result.model, "claude-sonnet-4-6");
        assert_eq!(result.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_incomplete_response_maps_cache_tokens_and_stop_reason() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_incomplete","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-5-codex","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Partial answer"}]}],"usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150,"input_tokens_details":{"cached_tokens":25},"cache_creation_input_tokens":10}}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();

        assert_eq!(result.stop_reason.as_deref(), Some("max_tokens"));
        assert_eq!(result.usage.input_tokens, 100);
        assert_eq!(result.usage.output_tokens, 50);
        assert_eq!(result.usage.cache_read_input_tokens, Some(25));
        assert_eq!(result.usage.cache_creation_input_tokens, Some(10));
    }

    #[test]
    fn test_message_content_uses_type_text_alias() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_x","status":"completed","model":"gpt-5","output":[{"type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}]}]}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-haiku").unwrap();
        match &result.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hi"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_fallback_output_text_done_when_completed_message_has_no_text() {
        let sse = r#"event: response.output_text.done
data: {"type":"response.output_text.done","text":"From stream"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_x","status":"completed","model":"gpt-5","output":[{"type":"message","role":"assistant","content":[]}]}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-haiku").unwrap();
        match &result.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "From stream"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_collect_text_from_reasoning_summary_when_message_empty() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"r","status":"completed","model":"m","output":[{"type":"reasoning","summary":"Summary line"},{"type":"message","role":"assistant","content":[]}]}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();
        match &result.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Summary line"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_content_part_text_as_nested_value_object() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"r","status":"completed","model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":{"value":"From nested value"}}]}]}}

"#;

        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();
        match &result.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "From nested value"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_collect_visible_when_content_is_single_object() {
        let v: Value = serde_json::from_str(
            r#"{"output":[{"type":"message","content":{"type":"output_text","text":"One block"}}]}"#,
        )
        .unwrap();
        assert_eq!(
            collect_visible_text_from_response_value(&v),
            "One block"
        );
    }

    #[test]
    fn test_collect_visible_concatenates_nested_shards_when_content_empty() {
        let v: Value = serde_json::from_str(
            r#"{"output":[{"type":"message","role":"assistant","content":[],"parts":[{"nested":{"text":"One "}},{"nested":{"text":"two "}},{"nested":{"text":"three."}}]}]}"#,
        )
        .unwrap();
        assert_eq!(
            collect_visible_text_from_response_value(&v),
            "One two three."
        );
    }

    #[test]
    fn test_root_fallback_collects_when_output_missing() {
        let v: Value = serde_json::from_str(
            r#"{"instructions":"very long system prompt","result":{"message":{"content":[{"type":"output_text","text":"Recovered from root fallback."}]}}}"#,
        )
        .unwrap();
        assert_eq!(
            collect_visible_text_from_response_root_fallback(&v),
            "Recovered from root fallback."
        );
    }

    #[test]
    fn test_greedy_stream_collects_output_text_delta_object() {
        let sse = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":{"text":"Hello"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"r","status":"completed","model":"m","output":[{"type":"message","content":[]}]}}

"#;
        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();
        match &result.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_call_recovered_from_raw_output_when_message_empty() {
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"r","status":"completed","model":"m","output":[{"type":"message","role":"assistant","content":[]},{"type":"tool_call","call_id":"c1","name":"Bash","arguments":{"cmd":"ls"}}]}}

"#;
        let result = convert_responses_to_anthropic(sse, "claude-sonnet-4-6").unwrap();
        assert_eq!(result.stop_reason.as_deref(), Some("tool_use"));
        match &result.content[0] {
            ResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "Bash");
                assert_eq!(input.get("cmd").and_then(|x| x.as_str()), Some("ls"));
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn test_longest_assistive_finds_deeply_nested_text() {
        let v: Value = serde_json::from_str(
            r#"{"output":[{"type":"custom","payload":{"text":"Deep assistant reply that is long enough for the token threshold check here."}}]}"#,
        )
        .unwrap();
        assert_eq!(
            longest_assistive_text_anywhere_in_response(&v, 60).as_deref(),
            Some("Deep assistant reply that is long enough for the token threshold check here.")
        );
    }

    #[test]
    fn test_longest_assistive_rejects_detailed_noise_even_if_long_tokens() {
        let v: Value = serde_json::from_str(
            r#"{"output":[{"type":"message","content":{"verbosity":"detailed","text":"detailed"}}]}"#,
        )
        .unwrap();
        assert!(longest_assistive_text_anywhere_in_response(&v, 200).is_none());
    }
}
