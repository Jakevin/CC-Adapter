use std::collections::BTreeMap;

use anyhow::{Context, Result};
use axum::response::sse::Event;
use serde_json::Value;
use uuid::Uuid;

use crate::convert::tool_sanitizer::sanitize_tool_input;
use crate::types::openai::ChatCompletionStreamChunk;

#[derive(Debug, Clone, Default)]
struct ToolAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Clone)]
pub struct OpenAIStreamAccumulator {
    message_id: String,
    original_model: String,
    text_started: bool,
    text_index: Option<usize>,
    next_index: usize,
    tool_calls: BTreeMap<u32, ToolAccumulator>,
    finish_reason: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
}

impl OpenAIStreamAccumulator {
    pub fn new(original_model: &str) -> Self {
        Self {
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            original_model: original_model.to_string(),
            text_started: false,
            text_index: None,
            next_index: 0,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    pub fn start_event(&self) -> Event {
        let message_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.original_model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 0
                }
            }
        });
        Event::default()
            .event("message_start")
            .data(message_start.to_string())
    }

    pub fn absorb_chunk(&mut self, chunk: ChatCompletionStreamChunk) -> Result<Vec<Event>> {
        if let Some(usage) = chunk.usage {
            self.input_tokens = usage.prompt_tokens;
            self.output_tokens = usage.completion_tokens;
        }

        let mut events = Vec::new();
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            if let Some(text) = choice.delta.content
                && !text.is_empty()
            {
                events.extend(self.text_delta_events(&text));
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for call in tool_calls {
                    let acc = self.tool_calls.entry(call.index).or_default();
                    if let Some(id) = call.id
                        && !id.is_empty()
                    {
                        acc.id = Some(id);
                    }
                    if let Some(function) = call.function {
                        if let Some(name) = function.name
                            && !name.is_empty()
                        {
                            acc.name = Some(name);
                        }
                        if let Some(arguments) = function.arguments {
                            acc.arguments.push_str(&arguments);
                        }
                    }
                }
            }
        }

        Ok(events)
    }

    pub fn finish_events(&mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();

        if self.text_started {
            let index = self.text_index.expect("text index must exist");
            events.push(content_block_stop_event(index));
            self.text_started = false;
        }

        for tool in self.tool_calls.values() {
            let block_index = self.next_index;
            self.next_index += 1;
            events.push(tool_start_event(
                block_index,
                tool.id.clone(),
                tool.name.clone(),
            ));

            let name = tool.name.clone().unwrap_or_default();
            let input: Value = serde_json::from_str(&tool.arguments)
                .unwrap_or(Value::Object(serde_json::Map::new()));
            let input = sanitize_tool_input(&name, input);
            let input_json = serde_json::to_string(&input)?;
            let delta = serde_json::json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": { "type": "input_json_delta", "partial_json": input_json }
            });
            events.push(
                Event::default()
                    .event("content_block_delta")
                    .data(delta.to_string()),
            );
            events.push(content_block_stop_event(block_index));
        }

        let stop_reason = self.anthropic_stop_reason();
        let message_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": {
                "output_tokens": self.output_tokens
            }
        });
        events.push(
            Event::default()
                .event("message_delta")
                .data(message_delta.to_string()),
        );
        events.push(
            Event::default()
                .event("message_stop")
                .data(serde_json::json!({ "type": "message_stop" }).to_string()),
        );

        Ok(events)
    }

    fn text_delta_events(&mut self, text: &str) -> Vec<Event> {
        let mut events = Vec::new();
        let index = match self.text_index {
            Some(index) => index,
            None => {
                let index = self.next_index;
                self.next_index += 1;
                self.text_index = Some(index);
                self.text_started = true;
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": { "type": "text", "text": "" }
                });
                events.push(
                    Event::default()
                        .event("content_block_start")
                        .data(block_start.to_string()),
                );
                index
            }
        };

        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "text_delta", "text": text }
        });
        events.push(
            Event::default()
                .event("content_block_delta")
                .data(delta.to_string()),
        );
        events
    }

    fn anthropic_stop_reason(&self) -> &'static str {
        if !self.tool_calls.is_empty() {
            return "tool_use";
        }

        match self.finish_reason.as_deref() {
            Some("length") => "max_tokens",
            _ => "end_turn",
        }
    }
}

pub fn parse_openai_stream_data_line(data: &str) -> Result<Option<ChatCompletionStreamChunk>> {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(data)
        .map(Some)
        .with_context(|| format!("failed to parse OpenAI stream chunk: {}", data))
}

fn content_block_stop_event(index: usize) -> Event {
    let block_stop = serde_json::json!({
        "type": "content_block_stop",
        "index": index
    });
    Event::default()
        .event("content_block_stop")
        .data(block_stop.to_string())
}

fn tool_start_event(index: usize, id: Option<String>, name: Option<String>) -> Event {
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": id.unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple())),
            "name": name.unwrap_or_default(),
            "input": {}
        }
    });
    Event::default()
        .event("content_block_start")
        .data(block_start.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_done_as_none() {
        assert!(parse_openai_stream_data_line("[DONE]").unwrap().is_none());
    }

    #[test]
    fn streams_text_deltas_and_end_turn() {
        let mut acc = OpenAIStreamAccumulator::new("claude-sonnet-4-6");
        let chunk = parse_openai_stream_data_line(
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
        )
        .unwrap()
        .unwrap();

        let events = acc.absorb_chunk(chunk).unwrap();
        assert_eq!(events.len(), 2);
        let preview = format!("{:?}{:?}", events[0], events[1]);
        assert!(preview.contains("content_block_start"));
        assert!(preview.contains("hello"));

        let done = parse_openai_stream_data_line(
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
        )
        .unwrap()
        .unwrap();
        assert!(acc.absorb_chunk(done).unwrap().is_empty());
        let finish = acc.finish_events().unwrap();
        let joined = format!("{:?}", finish);
        assert!(joined.contains("content_block_stop"));
        assert!(joined.contains("end_turn"));
        assert!(joined.contains("output_tokens"));
    }

    #[test]
    fn accumulates_tool_call_and_sanitizes_read_pages() {
        let mut acc = OpenAIStreamAccumulator::new("claude-sonnet-4-6");
        let first = parse_openai_stream_data_line(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Read","arguments":"{\"file_path\":\"/tmp/a.md\","}}]},"finish_reason":null}]}"#,
        )
        .unwrap()
        .unwrap();
        let second = parse_openai_stream_data_line(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"pages\":\"1\",\"limit\":50}"}}]},"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap()
        .unwrap();

        assert!(acc.absorb_chunk(first).unwrap().is_empty());
        assert!(acc.absorb_chunk(second).unwrap().is_empty());
        let events = acc.finish_events().unwrap();
        let joined = format!("{:?}", events);

        assert!(joined.contains("tool_use"));
        assert!(joined.contains("Read"));
        assert!(joined.contains("limit"));
        assert!(!joined.contains("pages"));
        assert!(joined.contains("tool_use"));
    }
}
