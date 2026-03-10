use anyhow::Result;

use crate::types::anthropic::{
    ContentBlock, Message, MessageContent, MessagesRequest, SystemPrompt, ToolChoice as AnthropicToolChoice,
    ToolDefinition, ToolResultContent,
};
use crate::types::openai::{
    ChatCompletionRequest, ChatMessage, ChatMessageContent, ContentPart, FunctionCall, FunctionDef,
    ImageUrl, Tool, ToolCall, ToolChoice as OpenAIToolChoice, ToolChoiceFunction, ToolChoiceObject,
};

/// 將 Anthropic Messages API 請求轉換為 OpenAI Chat Completions API 請求
/// Convert an Anthropic Messages API request into an OpenAI Chat Completions API request
pub fn convert_request(
    req: MessagesRequest,
    resolved_model: &str,
) -> Result<ChatCompletionRequest> {
    let model = resolved_model.to_string();

    let mut messages = Vec::new();

    // 將 Anthropic 頂層 system 欄位轉換為 OpenAI 的 system 角色訊息
    // Convert Anthropic's top-level system field into an OpenAI system role message
    if let Some(system) = &req.system {
        let content = match system {
            SystemPrompt::Text(text) => text.clone(),
            SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(ChatMessageContent::Text(content)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // 逐一轉換 Anthropic 訊息
    // Convert each Anthropic message one by one
    for msg in &req.messages {
        convert_message(msg, &mut messages)?;
    }

    let tools = req.tools.as_ref().map(|tools| convert_tools(tools));
    let tool_choice = req.tool_choice.as_ref().map(convert_tool_choice);

    Ok(ChatCompletionRequest {
        model,
        messages,
        tools,
        tool_choice,
        max_completion_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences.clone(),
        stream: Some(false),
        parallel_tool_calls: None,
    })
}

/// 轉換單一 Anthropic 訊息
/// Convert a single Anthropic message
fn convert_message(msg: &Message, out: &mut Vec<ChatMessage>) -> Result<()> {
    match &msg.content {
        MessageContent::Text(text) => {
            out.push(ChatMessage {
                role: msg.role.clone(),
                content: Some(ChatMessageContent::Text(text.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        MessageContent::Blocks(blocks) => {
            convert_content_blocks(&msg.role, blocks, out)?;
        }
    }
    Ok(())
}

/// 轉換 Anthropic 的內容區塊陣列。
/// 一條 Anthropic 訊息可能會展開為多條 OpenAI 訊息
/// （例如 tool_result 區塊會變成獨立的 role:"tool" 訊息）。
///
/// Convert Anthropic content blocks within a single message.
/// One Anthropic message may expand into multiple OpenAI messages
/// (e.g. tool_result blocks become separate `role: "tool"` messages).
fn convert_content_blocks(
    role: &str,
    blocks: &[ContentBlock],
    out: &mut Vec<ChatMessage>,
) -> Result<()> {
    match role {
        "assistant" => convert_assistant_blocks(blocks, out),
        "user" => convert_user_blocks(blocks, out),
        _ => {
            let text = extract_text_from_blocks(blocks);
            out.push(ChatMessage {
                role: role.to_string(),
                content: Some(ChatMessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
            });
            Ok(())
        }
    }
}

/// 轉換 assistant 角色的內容區塊：
/// 文字區塊合併為 content，tool_use 區塊轉為 tool_calls 陣列
///
/// Convert assistant role content blocks:
/// text blocks are merged into content, tool_use blocks become tool_calls array
fn convert_assistant_blocks(blocks: &[ContentBlock], out: &mut Vec<ChatMessage>) -> Result<()> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input)?,
                    },
                });
            }
            ContentBlock::Thinking { thinking, .. } => {
                // 將思考區塊以 XML 標籤包裹後嵌入文字內容
                // Wrap thinking block in XML tags and embed in text content
                text_parts.push(format!("<thinking>{}</thinking>", thinking));
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(ChatMessageContent::Text(text_parts.join("")))
    };

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    out.push(ChatMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tool_calls_opt,
        tool_call_id: None,
    });

    Ok(())
}

/// 轉換 user 角色的內容區塊：
/// 文字與圖片區塊組成使用者訊息，tool_result 區塊各自獨立成 role:"tool" 訊息
///
/// Convert user role content blocks:
/// text/image blocks form the user message, tool_result blocks become separate role:"tool" messages
fn convert_user_blocks(blocks: &[ContentBlock], out: &mut Vec<ChatMessage>) -> Result<()> {
    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut tool_results: Vec<(String, Option<String>, Option<bool>)> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                content_parts.push(ContentPart::Text { text: text.clone() });
            }
            ContentBlock::Image { source } => {
                // 將 base64 圖片轉為 data URI 格式
                // Convert base64 image into data URI format
                let data_url = format!(
                    "data:{};base64,{}",
                    source.media_type, source.data
                );
                content_parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: data_url,
                        detail: Some("auto".to_string()),
                    },
                });
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let text = match content {
                    Some(ToolResultContent::Text(t)) => Some(t.clone()),
                    Some(ToolResultContent::Blocks(blocks)) => {
                        Some(extract_text_from_blocks(blocks))
                    }
                    None => None,
                };
                tool_results.push((tool_use_id.clone(), text, *is_error));
            }
            _ => {}
        }
    }

    // 如果有文字或圖片內容，產生使用者訊息
    // If there are text/image parts, emit a user message
    if !content_parts.is_empty() {
        let content = if content_parts.len() == 1 {
            if let ContentPart::Text { text } = &content_parts[0] {
                ChatMessageContent::Text(text.clone())
            } else {
                ChatMessageContent::Parts(content_parts)
            }
        } else {
            ChatMessageContent::Parts(content_parts)
        };

        out.push(ChatMessage {
            role: "user".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // 每個 tool_result 區塊轉為獨立的 role:"tool" 訊息
    // Each tool_result block becomes a separate role:"tool" message
    for (tool_call_id, text, is_error) in tool_results {
        let content_str = if is_error.unwrap_or(false) {
            format!("Error: {}", text.unwrap_or_default())
        } else {
            text.unwrap_or_default()
        };

        out.push(ChatMessage {
            role: "tool".to_string(),
            content: Some(ChatMessageContent::Text(content_str)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        });
    }

    Ok(())
}

/// 從內容區塊陣列中擷取所有文字並串接
/// Extract all text from content blocks and concatenate
fn extract_text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// 將 Anthropic 工具定義轉換為 OpenAI 函式工具格式
/// Convert Anthropic tool definitions to OpenAI function tool format
fn convert_tools(tools: &[ToolDefinition]) -> Vec<Tool> {
    tools
        .iter()
        .map(|t| Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
                strict: None,
            },
        })
        .collect()
}

/// 將 Anthropic tool_choice 轉換為 OpenAI tool_choice 格式
/// Convert Anthropic tool_choice to OpenAI tool_choice format
fn convert_tool_choice(tc: &AnthropicToolChoice) -> OpenAIToolChoice {
    match tc {
        AnthropicToolChoice::Auto { .. } => OpenAIToolChoice::String("auto".to_string()),
        AnthropicToolChoice::Any { .. } => OpenAIToolChoice::String("required".to_string()),
        AnthropicToolChoice::None {} => OpenAIToolChoice::String("none".to_string()),
        AnthropicToolChoice::Tool { name, .. } => {
            OpenAIToolChoice::Object(ToolChoiceObject {
                choice_type: "function".to_string(),
                function: ToolChoiceFunction {
                    name: name.clone(),
                },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_text_conversion() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            }],
            system: Some(SystemPrompt::Text("You are helpful.".to_string())),
            tools: None,
            tool_choice: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
        };

        let result = convert_request(req, "gpt-4o").unwrap();

        assert_eq!(result.model, "gpt-4o");
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[1].role, "user");
    }

    #[test]
    fn test_tool_use_conversion() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 1024,
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("What's the weather?".to_string()),
                },
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Text {
                            text: "Let me check.".to_string(),
                        },
                        ContentBlock::ToolUse {
                            id: "toolu_01".to_string(),
                            name: "get_weather".to_string(),
                            input: json!({"location": "SF"}),
                        },
                    ]),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: "toolu_01".to_string(),
                        content: Some(ToolResultContent::Text("72F sunny".to_string())),
                        is_error: None,
                    }]),
                },
            ],
            system: None,
            tools: Some(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
                cache_control: None,
            }]),
            tool_choice: Some(AnthropicToolChoice::Auto {
                disable_parallel_tool_use: None,
            }),
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
        };

        let result = convert_request(req, "gpt-4o").unwrap();

        // 應產生：user、assistant（含 tool_calls）、tool 三條訊息
        // Should produce: user, assistant (with tool_calls), tool — 3 messages
        assert_eq!(result.messages.len(), 3);
        assert_eq!(result.messages[1].role, "assistant");
        assert!(result.messages[1].tool_calls.is_some());
        assert_eq!(result.messages[2].role, "tool");
        assert_eq!(
            result.messages[2].tool_call_id.as_deref(),
            Some("toolu_01")
        );

        // 驗證工具定義已正確轉換
        // Verify tool definitions are correctly converted
        assert!(result.tools.is_some());
        let tools = result.tools.unwrap();
        assert_eq!(tools[0].function.name, "get_weather");
    }
}
