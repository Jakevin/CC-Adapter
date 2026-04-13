use anyhow::Result;

use crate::types::anthropic::{
    ContentBlock, Message, MessageContent, MessagesRequest, SystemPrompt,
    ToolDefinition, ToolResultContent,
};
use crate::types::responses::{
    InputContent, InputContentPart, InputItem, ReasoningConfig, ResponsesRequest,
    ResponsesTool, TextConfig,
};

/// ChatGPT Codex `codex/responses` 要求請求必須帶非空 `instructions`（對應 system）
/// ChatGPT Codex requires a non-empty `instructions` field (maps from system prompt)
const DEFAULT_CODEX_INSTRUCTIONS: &str = "You are a helpful assistant.";

/// 將 Anthropic Messages API 請求轉換為 OpenAI Responses API 請求
/// Convert an Anthropic Messages API request into an OpenAI Responses API request
pub fn convert_request_to_responses(
    req: MessagesRequest,
    resolved_model: &str,
) -> Result<ResponsesRequest> {
    let model = resolved_model.to_string();

    // 提取系統提示作為 instructions（缺省或全空白時填預設，否則 Codex 回 HTTP 400）
    // Extract system prompt as instructions (default if missing/blank, else Codex returns HTTP 400)
    let from_system = req.system.as_ref().map(|s| match s {
        SystemPrompt::Text(text) => text.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    });
    let instructions = match from_system {
        Some(ref s) if !s.trim().is_empty() => s.clone(),
        _ => DEFAULT_CODEX_INSTRUCTIONS.to_string(),
    };

    let mut input: Vec<InputItem> = Vec::new();

    // 轉換每條 Anthropic 訊息為 Responses API 的 input items
    // Convert each Anthropic message into Responses API input items
    for msg in &req.messages {
        convert_message_to_input(msg, &mut input)?;
    }

    let tools = req.tools.as_ref().map(|tools| convert_tools(tools));

    Ok(ResponsesRequest {
        model,
        input,
        store: false,
        stream: true,
        instructions,
        tools,
        reasoning: Some(ReasoningConfig {
            effort: Some("medium".to_string()),
            summary: Some("auto".to_string()),
        }),
        text: Some(TextConfig {
            verbosity: Some("medium".to_string()),
        }),
        // 勿要求 reasoning.encrypted_content：Codex 常把可讀回覆壓進加密區塊，導致 message 正文為空、Claude Code 無內容
        // Avoid requesting reasoning.encrypted_content: Codex often hides visible text there, leaving message body empty
        include: None,
    })
}

/// 將單一 Anthropic 訊息轉換為 Responses API input items
/// Convert a single Anthropic message into Responses API input items
fn convert_message_to_input(msg: &Message, out: &mut Vec<InputItem>) -> Result<()> {
    match &msg.content {
        MessageContent::Text(text) => {
            out.push(InputItem::Message {
                role: msg.role.clone(),
                content: InputContent::Text(text.clone()),
            });
        }
        MessageContent::Blocks(blocks) => {
            convert_blocks_to_input(&msg.role, blocks, out)?;
        }
    }
    Ok(())
}

/// 轉換 Anthropic 內容區塊為 Responses API input items
/// Convert Anthropic content blocks into Responses API input items
fn convert_blocks_to_input(
    role: &str,
    blocks: &[ContentBlock],
    out: &mut Vec<InputItem>,
) -> Result<()> {
    match role {
        "assistant" => convert_assistant_blocks(blocks, out),
        "user" => convert_user_blocks(blocks, out),
        _ => {
            let text = extract_text_from_blocks(blocks);
            out.push(InputItem::Message {
                role: role.to_string(),
                content: InputContent::Text(text),
            });
            Ok(())
        }
    }
}

/// 轉換 assistant 區塊：文字 → message，tool_use → function_call
/// Convert assistant blocks: text → message, tool_use → function_call
fn convert_assistant_blocks(blocks: &[ContentBlock], out: &mut Vec<InputItem>) -> Result<()> {
    let mut text_parts: Vec<String> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::ToolUse { id, name, input } => {
                // 先輸出累積的文字
                // Flush accumulated text first
                if !text_parts.is_empty() {
                    out.push(InputItem::Message {
                        role: "assistant".to_string(),
                        content: InputContent::Text(text_parts.join("")),
                    });
                    text_parts.clear();
                }
                out.push(InputItem::FunctionCall {
                    name: name.clone(),
                    arguments: serde_json::to_string(input)?,
                    call_id: id.clone(),
                });
            }
            ContentBlock::Thinking { .. } => {}
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        out.push(InputItem::Message {
            role: "assistant".to_string(),
            content: InputContent::Text(text_parts.join("")),
        });
    }

    Ok(())
}

/// 轉換 user 區塊：文字/圖片 → message，tool_result → function_call_output
/// Convert user blocks: text/image → message, tool_result → function_call_output
fn convert_user_blocks(blocks: &[ContentBlock], out: &mut Vec<InputItem>) -> Result<()> {
    let mut content_parts: Vec<InputContentPart> = Vec::new();
    let mut tool_results: Vec<(String, String, bool)> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                content_parts.push(InputContentPart::Text { text: text.clone() });
            }
            ContentBlock::Image { source } => {
                let data_url = format!(
                    "data:{};base64,{}",
                    source.media_type, source.data
                );
                content_parts.push(InputContentPart::Image {
                    image_url: data_url,
                    detail: Some("auto".to_string()),
                });
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let text = match content {
                    Some(ToolResultContent::Text(t)) => t.clone(),
                    Some(ToolResultContent::Blocks(inner_blocks)) => {
                        extract_text_from_blocks(inner_blocks)
                    }
                    None => String::new(),
                };
                let is_err = is_error.unwrap_or(false);
                let output = if is_err {
                    format!("Error: {}", text)
                } else {
                    text
                };
                tool_results.push((tool_use_id.clone(), output, is_err));
            }
            _ => {}
        }
    }

    // 輸出使用者文字/圖片訊息
    // Emit user text/image message
    if !content_parts.is_empty() {
        let content = if content_parts.len() == 1 {
            if let InputContentPart::Text { text } = &content_parts[0] {
                InputContent::Text(text.clone())
            } else {
                InputContent::Parts(content_parts)
            }
        } else {
            InputContent::Parts(content_parts)
        };

        out.push(InputItem::Message {
            role: "user".to_string(),
            content,
        });
    }

    // 每個 tool_result → function_call_output
    for (call_id, output, _) in tool_results {
        out.push(InputItem::FunctionCallOutput {
            call_id,
            output,
        });
    }

    Ok(())
}

/// 從區塊中提取所有文字
/// Extract all text from blocks
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

/// 轉換 Anthropic 工具定義為 Responses API 工具格式
/// Convert Anthropic tool definitions to Responses API tool format
fn convert_tools(tools: &[ToolDefinition]) -> Vec<ResponsesTool> {
    tools
        .iter()
        .map(|t| ResponsesTool {
            tool_type: "function".to_string(),
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: Some(t.input_schema.clone()),
        })
        .collect()
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

        let result = convert_request_to_responses(req, "gpt-5-codex").unwrap();

        assert_eq!(result.model, "gpt-5-codex");
        assert_eq!(result.instructions, "You are helpful.");
        assert!(!result.store);
        assert!(result.stream);
        assert_eq!(result.input.len(), 1);
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
            tool_choice: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
        };

        let result = convert_request_to_responses(req, "gpt-5-codex").unwrap();

        // 應產生：user message、assistant message、function_call、function_call_output
        // Should produce: user message, assistant message, function_call, function_call_output
        assert_eq!(result.input.len(), 4);
        assert!(result.tools.is_some());
        assert_eq!(result.instructions, DEFAULT_CODEX_INSTRUCTIONS);
    }

    #[test]
    fn test_thinking_blocks_are_dropped() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "Visible text".to_string(),
                    },
                    ContentBlock::Thinking {
                        thinking: "internal reasoning".to_string(),
                        signature: None,
                    },
                ]),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
        };

        let result = convert_request_to_responses(req, "gpt-5-codex").unwrap();

        assert_eq!(result.input.len(), 1);
        match &result.input[0] {
            InputItem::Message { role, content } => {
                assert_eq!(role, "assistant");
                match content {
                    InputContent::Text(text) => assert_eq!(text, "Visible text"),
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected message input item"),
        }
    }
}
