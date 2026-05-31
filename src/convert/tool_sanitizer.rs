use serde_json::Value;

/// Clean model-produced tool inputs before returning them to Claude Code.
pub fn sanitize_tool_input(tool_name: &str, mut input: Value) -> Value {
    remove_top_level_nulls(&mut input);

    if tool_name == "Read" {
        sanitize_read_input(&mut input);
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
}
