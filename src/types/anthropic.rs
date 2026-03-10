use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── 請求型別 ───
// ─── Request Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// 系統提示：可以是純文字字串，也可以是含快取控制的內容區塊陣列
/// System prompt: either a plain text string, or an array of content blocks with cache control
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// 訊息內容：可以是純文字字串，也可以是結構化的內容區塊陣列
/// Message content: either a plain text string, or an array of structured content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// 內容區塊：依 type 欄位區分為文字、工具呼叫、工具結果、圖片、思考等類型
/// Content block: discriminated by `type` field into text, tool_use, tool_result, image, thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// 工具結果內容：可以是純文字，也可以是包含文字和圖片的區塊陣列
/// Tool result content: either plain text, or an array of blocks containing text and images
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

/// 工具選擇策略：auto（自動）、any（必須呼叫）、tool（指定工具）、none（禁止）
/// Tool choice strategy: auto, any (must call), tool (specific tool), none (disabled)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "any")]
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "tool")]
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "none")]
    None {},
}

// ─── 回應型別 ───
// ─── Response Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// 回應內容區塊：文字、工具呼叫或思考區塊
/// Response content block: text, tool_use, or thinking block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// 思考區塊（部分第三方 Anthropic 相容 API 會回傳）
    /// Thinking block (some Anthropic-compatible APIs may return this)
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// ─── 錯誤型別 ───
// ─── Error Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}
