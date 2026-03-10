use anyhow::Result;
use axum::response::sse::Event;
use serde_json::Value;
use uuid::Uuid;

use crate::types::anthropic::{MessagesResponse, ResponseContentBlock, Usage};
use crate::types::openai::ChatCompletionResponse;

/// 將 OpenAI Chat Completions 回應轉換為 Anthropic Messages 回應格式
/// Convert an OpenAI Chat Completions response into Anthropic Messages response format
pub fn convert_response(
    resp: ChatCompletionResponse,
    original_model: &str,
) -> Result<MessagesResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("OpenAI 回應中沒有 choices / No choices in OpenAI response"))?;

    let mut content: Vec<ResponseContentBlock> = Vec::new();

    // 轉換文字內容
    // Convert text content
    if let Some(text) = &choice.message.content
        && !text.is_empty()
    {
        content.push(ResponseContentBlock::Text { text: text.clone() });
    }

    // 轉換工具呼叫：OpenAI tool_calls → Anthropic tool_use 內容區塊
    // Convert tool calls: OpenAI tool_calls → Anthropic tool_use content blocks
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            // 將 JSON 字串解析為物件，解析失敗則使用空物件
            // Parse JSON string into an object; fall back to empty object on failure
            let input: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::Object(serde_json::Map::new()));

            content.push(ResponseContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    // 若內容仍為空，插入空文字區塊以滿足 Anthropic API 要求
    // If content is still empty, push an empty text block to satisfy Anthropic API requirements
    if content.is_empty() {
        content.push(ResponseContentBlock::Text {
            text: String::new(),
        });
    }

    let stop_reason = convert_finish_reason(choice.finish_reason.as_deref());

    let usage = match &resp.usage {
        Some(u) => Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
        None => Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    };

    // 產生 Anthropic 格式的訊息 ID（msg_ 前綴 + UUID）
    // Generate an Anthropic-style message ID (msg_ prefix + UUID)
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

/// 將 OpenAI 的 finish_reason 映射為 Anthropic 的 stop_reason
/// Map OpenAI's finish_reason to Anthropic's stop_reason
fn convert_finish_reason(reason: Option<&str>) -> String {
    match reason {
        Some("stop") => "end_turn".to_string(),
        Some("tool_calls") => "tool_use".to_string(),
        Some("length") => "max_tokens".to_string(),
        Some("content_filter") => "end_turn".to_string(),
        Some(other) => other.to_string(),
        None => "end_turn".to_string(),
    }
}

/// 將完整的 Anthropic 回應轉換為 SSE 事件序列，模擬串流回應。
/// 用於供應商不支援串流但 Claude Code 請求串流的情況。
///
/// Convert a complete Anthropic response into a sequence of SSE events, simulating streaming.
/// Used when the provider doesn't support streaming but Claude Code requests it.
pub fn response_to_sse_events(resp: &MessagesResponse) -> Result<Vec<Event>> {
    let mut events = Vec::new();

    // 1. message_start：包含初始訊息物件（content 為空）
    // 1. message_start: contains initial message object (content is empty)
    let message_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": resp.id,
            "type": "message",
            "role": resp.role,
            "content": [],
            "model": resp.model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": 0
            }
        }
    });
    events.push(Event::default().event("message_start").data(message_start.to_string()));

    // 2. 依序發送每個內容區塊的 start / delta / stop 事件
    // 2. Emit start / delta / stop events for each content block in order
    for (index, block) in resp.content.iter().enumerate() {
        match block {
            ResponseContentBlock::Text { text } => {
                // content_block_start
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": { "type": "text", "text": "" }
                });
                events.push(Event::default().event("content_block_start").data(block_start.to_string()));

                // content_block_delta（將完整文字一次送出）
                // content_block_delta (send complete text in one chunk)
                if !text.is_empty() {
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "text_delta", "text": text }
                    });
                    events.push(Event::default().event("content_block_delta").data(delta.to_string()));
                }

                // content_block_stop
                let block_stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": index
                });
                events.push(Event::default().event("content_block_stop").data(block_stop.to_string()));
            }
            ResponseContentBlock::ToolUse { id, name, input } => {
                // content_block_start（tool_use 區塊，input 為空物件）
                // content_block_start (tool_use block, input is empty object)
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {}
                    }
                });
                events.push(Event::default().event("content_block_start").data(block_start.to_string()));

                // content_block_delta（以 input_json_delta 一次送出完整 JSON）
                // content_block_delta (send complete JSON via input_json_delta in one chunk)
                let input_json = serde_json::to_string(input)?;
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "input_json_delta", "partial_json": input_json }
                });
                events.push(Event::default().event("content_block_delta").data(delta.to_string()));

                // content_block_stop
                let block_stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": index
                });
                events.push(Event::default().event("content_block_stop").data(block_stop.to_string()));
            }
            ResponseContentBlock::Thinking { thinking, .. } => {
                // 將 thinking 區塊以 Anthropic 官方 SSE 格式轉發：
                // thinking content block（對 UI 隱藏，不再作為一般文字顯示）。
                // Forward thinking blocks using Anthropic SSE format so UIs can hide them.
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": { "type": "thinking", "thinking": "" }
                });
                events.push(Event::default().event("content_block_start").data(block_start.to_string()));

                if !thinking.is_empty() {
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "thinking_delta", "thinking": thinking }
                    });
                    events.push(Event::default().event("content_block_delta").data(delta.to_string()));
                }

                let block_stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": index
                });
                events.push(Event::default().event("content_block_stop").data(block_stop.to_string()));
            }
        }
    }

    // 3. message_delta：包含 stop_reason 和最終 usage
    // 3. message_delta: contains stop_reason and final usage
    let message_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": resp.stop_reason,
            "stop_sequence": resp.stop_sequence
        },
        "usage": {
            "output_tokens": resp.usage.output_tokens
        }
    });
    events.push(Event::default().event("message_delta").data(message_delta.to_string()));

    // 4. message_stop
    let message_stop = serde_json::json!({ "type": "message_stop" });
    events.push(Event::default().event("message_stop").data(message_stop.to_string()));

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{
        Choice, ChoiceMessage, FunctionCall, ResponseUsage, ToolCall,
    };

    #[test]
    fn test_basic_text_response() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(ResponseUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            system_fingerprint: None,
        };

        let result = convert_response(resp, "claude-sonnet-4-6").unwrap();

        assert_eq!(result.response_type, "message");
        assert_eq!(result.role, "assistant");
        assert_eq!(result.model, "claude-sonnet-4-6");
        assert_eq!(result.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn test_tool_call_response() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-xyz".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("Let me check.".to_string()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_001".to_string(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"SF"}"#.to_string(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(ResponseUsage {
                prompt_tokens: 50,
                completion_tokens: 20,
                total_tokens: 70,
            }),
            system_fingerprint: None,
        };

        let result = convert_response(resp, "claude-sonnet-4-6").unwrap();

        assert_eq!(result.stop_reason.as_deref(), Some("tool_use"));
        // 應包含文字 + tool_use 兩個內容區塊
        // Should contain text + tool_use — 2 content blocks
        assert_eq!(result.content.len(), 2);
    }
}
