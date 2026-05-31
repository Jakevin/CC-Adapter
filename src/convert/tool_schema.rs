use serde_json::{Value, json};

pub fn normalize_tool_input_schema(tool_name: &str, input_schema: Option<&Value>) -> Value {
    if let Some(schema) = input_schema.filter(|schema| !schema.is_null()) {
        return schema.clone();
    }

    match tool_name {
        "WebSearch" => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "allowed_domains": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "blocked_domains": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["query"],
            "additionalProperties": true
        }),
        "WebFetch" => json!({
            "type": "object",
            "properties": {
                "url": { "type": "string" },
                "prompt": { "type": "string" }
            },
            "required": ["url"],
            "additionalProperties": true
        }),
        _ => generic_object_schema(),
    }
}

fn generic_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_preserves_existing_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string" }
            }
        });

        assert_eq!(normalize_tool_input_schema("Read", Some(&schema)), schema);
    }

    #[test]
    fn test_web_search_default_schema() {
        let schema = normalize_tool_input_schema("WebSearch", None);
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        assert!(schema["properties"].get("query").is_some());
        assert!(schema["properties"].get("allowed_domains").is_some());
        assert!(schema["properties"].get("blocked_domains").is_some());
    }

    #[test]
    fn test_web_fetch_default_schema() {
        let schema = normalize_tool_input_schema("WebFetch", None);
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        assert!(schema["properties"].get("url").is_some());
        assert!(schema["properties"].get("prompt").is_some());
    }

    #[test]
    fn test_unknown_tool_default_schema() {
        let schema = normalize_tool_input_schema("UnknownTool", None);
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": true
            })
        );
    }
}
