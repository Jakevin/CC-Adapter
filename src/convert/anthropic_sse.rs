//! 將 Anthropic Messages API 的 SSE 串流文字聚合為單一 `MessagesResponse`
//! Aggregate Anthropic Messages API SSE stream text into one `MessagesResponse`

use anyhow::{Context, Result};
use serde_json::Value;
use uuid::Uuid;

use crate::types::anthropic::{MessagesResponse, ResponseContentBlock, Usage};

/// 從相容後端回傳的 Anthropic Messages SSE 串流組出完整回應
/// Build a complete response from an Anthropic Messages SSE stream returned by a compatible backend
pub fn parse_anthropic_messages_sse(sse_text: &str) -> Result<MessagesResponse> {
    let normalized = sse_text.replace("\r\n", "\n");

    let mut msg_id = String::new();
    let mut response_type = "message".to_string();
    let mut role = "assistant".to_string();
    let mut model = String::new();
    let mut input_tokens = 0u32;
    let mut output_tokens = 0u32;
    let mut stop_reason: Option<String> = None;
    let mut stop_sequence: Option<String> = None;
    let mut text_by_block: Vec<String> = Vec::new();

    for block in normalized.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let mut event_type = "";
        let mut data_line = "";

        for line in block.lines() {
            if line.starts_with(':') {
                continue;
            }
            if let Some(et) = line.strip_prefix("event: ") {
                event_type = et.trim();
            } else if let Some(et) = line.strip_prefix("event:") {
                event_type = et.trim();
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_line = d.trim();
            } else if let Some(d) = line.strip_prefix("data:") {
                data_line = d.trim();
            }
        }

        if data_line.is_empty() {
            continue;
        }

        let v: Value = serde_json::from_str(data_line).with_context(|| {
            format!(
                "無法解析 Anthropic SSE 事件 JSON / Failed to parse Anthropic SSE event JSON: {}",
                data_line.chars().take(200).collect::<String>()
            )
        })?;

        match event_type {
            "message_start" => {
                if let Some(m) = v.get("message") {
                    msg_id = json_str(m, "id").unwrap_or_default().to_string();
                    response_type = json_str(m, "type")
                        .unwrap_or("message")
                        .to_string();
                    role = json_str(m, "role").unwrap_or("assistant").to_string();
                    model = json_str(m, "model").unwrap_or_default().to_string();
                    if let Some(u) = m.get("usage") {
                        input_tokens = json_u32(u, "input_tokens");
                        output_tokens = json_u32(u, "output_tokens");
                    }
                }
            }
            "content_block_delta" => {
                let index = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                if let Some(delta) = v.get("delta") {
                    let delta_type = delta.get("type").and_then(|x| x.as_str());
                    if delta_type == Some("text_delta") {
                        let piece = delta
                            .get("text")
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        ensure_vec_len(&mut text_by_block, index);
                        text_by_block[index].push_str(piece);
                    }
                }
            }
            "message_delta" => {
                if let Some(u) = v.get("usage") {
                    if u.get("input_tokens").is_some() {
                        input_tokens = json_u32(u, "input_tokens");
                    }
                    if u.get("output_tokens").is_some() {
                        output_tokens = json_u32(u, "output_tokens");
                    }
                }
                if let Some(d) = v.get("delta") {
                    if let Some(sr) = d.get("stop_reason").and_then(|x| x.as_str()) {
                        stop_reason = Some(sr.to_string());
                    }
                    if let Some(ss) = d.get("stop_sequence").and_then(|x| x.as_str()) {
                        stop_sequence = Some(ss.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let mut content: Vec<ResponseContentBlock> = Vec::new();
    for t in text_by_block {
        if !t.is_empty() {
            content.push(ResponseContentBlock::Text { text: t });
        }
    }
    if content.is_empty() {
        content.push(ResponseContentBlock::Text {
            text: String::new(),
        });
    }

    if msg_id.is_empty() {
        msg_id = format!("msg_{}", Uuid::new_v4().simple());
    }

    Ok(MessagesResponse {
        id: msg_id,
        response_type,
        role,
        model,
        content,
        stop_reason,
        stop_sequence,
        usage: Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    })
}

fn json_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

fn json_u32(v: &Value, key: &str) -> u32 {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|n| n as u32)
        .unwrap_or(0)
}

fn ensure_vec_len(v: &mut Vec<String>, i: usize) {
    while v.len() <= i {
        v.push(String::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_text_deltas_and_usage() {
        let sse = r#": keep-alive

event: message_start
data: {"type": "message_start", "message": {"id": "msg_test", "type": "message", "role": "assistant", "model": "gemma-test", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 204, "output_tokens": 0}}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 2, "input_tokens": 204}}

event: message_stop
data: {"type": "message_stop"}

"#;

        let r = parse_anthropic_messages_sse(sse).unwrap();
        assert_eq!(r.id, "msg_test");
        assert_eq!(r.model, "gemma-test");
        assert_eq!(r.usage.input_tokens, 204);
        assert_eq!(r.usage.output_tokens, 2);
        assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(r.content.len(), 1);
        match &r.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hi"),
            _ => panic!("expected text block"),
        }
    }
}
