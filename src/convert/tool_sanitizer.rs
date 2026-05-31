use serde_json::Value;

/// Clean model-produced tool inputs before returning them to Claude Code.
pub fn sanitize_tool_input(tool_name: &str, mut input: Value) -> Value {
    remove_top_level_nulls(&mut input);

    if tool_name == "Read" {
        sanitize_read_input(&mut input);
    }
    if tool_name == "TodoWrite" {
        sanitize_todo_write_input(&mut input);
    }

    input
}

fn remove_top_level_nulls(input: &mut Value) {
    if let Value::Object(map) = input {
        map.retain(|_, value| !value.is_null());
    }
}

fn sanitize_read_input(input: &mut Value) {
    let Value::Object(map) = input else {
        return;
    };

    let pages_is_empty = map
        .get("pages")
        .and_then(|value| value.as_str())
        .is_some_and(|pages| pages.is_empty());

    if pages_is_empty {
        map.remove("pages");
        return;
    }

    let is_pdf = map
        .get("file_path")
        .and_then(|value| value.as_str())
        .is_some_and(|file_path| file_path.to_ascii_lowercase().ends_with(".pdf"));

    if !is_pdf {
        map.remove("pages");
    }
}

fn sanitize_todo_write_input(input: &mut Value) {
    let Value::Object(map) = input else {
        return;
    };

    let Some(Value::Array(todos)) = map.get_mut("todos") else {
        return;
    };

    for todo in todos {
        let Value::Object(todo_map) = todo else {
            continue;
        };

        let content = todo_map
            .get("content")
            .and_then(|value| value.as_str())
            .filter(|content| !content.is_empty())
            .map(str::to_string);
        let active_form = todo_map
            .get("activeForm")
            .and_then(|value| value.as_str())
            .filter(|active_form| !active_form.is_empty())
            .map(str::to_string);

        if content.is_none()
            && let Some(active_form) = active_form.as_ref()
        {
            todo_map.insert("content".to_string(), Value::String(active_form.clone()));
        }

        if active_form.is_none()
            && let Some(content) = content.as_ref()
        {
            todo_map.insert("activeForm".to_string(), Value::String(content.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_md_removes_pages() {
        let input = json!({
            "file_path": "C:/workspace/PROBLEM_ANALYSIS.md",
            "limit": 2000,
            "offset": 0,
            "pages": "1"
        });

        let result = sanitize_tool_input("Read", input);

        assert!(result.get("pages").is_none());
        assert_eq!(result.get("limit").and_then(|v| v.as_i64()), Some(2000));
        assert_eq!(result.get("offset").and_then(|v| v.as_i64()), Some(0));
    }

    #[test]
    fn read_txt_removes_empty_pages() {
        let input = json!({
            "file_path": "C:/workspace/A.txt",
            "limit": 2000,
            "offset": 0,
            "pages": ""
        });

        let result = sanitize_tool_input("Read", input);

        assert!(result.get("pages").is_none());
        assert_eq!(
            result.get("file_path").and_then(|v| v.as_str()),
            Some("C:/workspace/A.txt")
        );
    }

    #[test]
    fn read_pdf_keeps_valid_pages() {
        let input = json!({
            "file_path": "C:/workspace/A.PDF",
            "pages": "1-5"
        });

        let result = sanitize_tool_input("Read", input);

        assert_eq!(result.get("pages").and_then(|v| v.as_str()), Some("1-5"));
    }

    #[test]
    fn read_pdf_removes_empty_pages() {
        let input = json!({
            "file_path": "C:/workspace/A.pdf",
            "pages": ""
        });

        let result = sanitize_tool_input("Read", input);

        assert!(result.get("pages").is_none());
    }

    #[test]
    fn non_read_keeps_empty_string_but_removes_null() {
        let input = json!({
            "command": "",
            "description": null
        });

        let result = sanitize_tool_input("Bash", input);

        assert_eq!(result.get("command").and_then(|v| v.as_str()), Some(""));
        assert!(result.get("description").is_none());
    }

    #[test]
    fn todo_write_fills_missing_content_from_active_form() {
        let input = json!({
            "todos": [
                {
                    "activeForm": "修复工具参数",
                    "status": "in_progress"
                }
            ]
        });

        let result = sanitize_tool_input("TodoWrite", input);

        assert_eq!(
            result
                .get("todos")
                .and_then(|value| value.as_array())
                .and_then(|todos| todos.first())
                .and_then(|todo| todo.get("content"))
                .and_then(|value| value.as_str()),
            Some("修复工具参数")
        );
    }

    #[test]
    fn todo_write_fills_missing_active_form_from_content() {
        let input = json!({
            "todos": [
                {
                    "content": "修复工具参数",
                    "status": "pending"
                }
            ]
        });

        let result = sanitize_tool_input("TodoWrite", input);

        assert_eq!(
            result
                .get("todos")
                .and_then(|value| value.as_array())
                .and_then(|todos| todos.first())
                .and_then(|todo| todo.get("activeForm"))
                .and_then(|value| value.as_str()),
            Some("修复工具参数")
        );
    }
}
