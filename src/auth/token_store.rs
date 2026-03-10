use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// OAuth token 資料結構
/// OAuth token data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub access_token: String,
    pub refresh_token: String,
    /// 過期時間（Unix 毫秒）
    /// Expiration time (Unix milliseconds)
    pub expires_at: u64,
}

impl TokenData {
    /// 檢查 token 是否已過期（提前 60 秒判定）
    /// Check whether the token has expired (60-second early buffer)
    pub fn is_expired(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now_ms + 60_000 >= self.expires_at
    }
}

/// 取得 token 儲存目錄：~/.claude-adapter/
/// Get the token storage directory: ~/.claude-adapter/
fn storage_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context(
        "無法取得使用者家目錄 / Failed to get home directory",
    )?;
    Ok(home.join(".claude-adapter"))
}

fn normalize_name(name: &str) -> String {
    // 僅允許簡單檔名字元，其他一律轉為 '_'，避免路徑注入或不合法檔名
    // Allow only safe filename characters; replace others with '_'
    let trimmed = name.trim();
    let mut out = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "chatgpt".to_string()
    } else {
        out
    }
}

/// 取得 token 檔案路徑：~/.claude-adapter/tokens-<name>.json
/// Get the token file path: ~/.claude-adapter/tokens-<name>.json
fn token_path_named(name: &str) -> Result<PathBuf> {
    let n = normalize_name(name);
    Ok(storage_dir()?.join(format!("tokens-{}.json", n)))
}

/// 舊版 token 檔案路徑：~/.claude-adapter/tokens.json（向後相容）
/// Legacy token file path: ~/.claude-adapter/tokens.json (backward compatibility)
fn legacy_token_path() -> Result<PathBuf> {
    Ok(storage_dir()?.join("tokens.json"))
}

/// 儲存 token 至磁碟
/// Save token to disk
pub fn save_named(name: &str, data: &TokenData) -> Result<()> {
    let dir = storage_dir()?;
    if !dir.exists() {
        std::fs::create_dir_all(&dir).with_context(|| {
            format!(
                "無法建立目錄 / Failed to create directory: {}",
                dir.display()
            )
        })?;
    }
    let path = token_path_named(name)?;
    let json = serde_json::to_string_pretty(data)
        .context("無法序列化 token / Failed to serialize token")?;
    std::fs::write(&path, json).with_context(|| {
        format!(
            "無法寫入 token 檔案 / Failed to write token file: {}",
            path.display()
        )
    })?;
    Ok(())
}

/// 舊 API（向後相容）：等同 save_named(\"chatgpt\", ...)
/// Legacy API (backward compatibility): same as save_named(\"chatgpt\", ...)
#[allow(dead_code)]
#[deprecated(note = "Use save_named instead")]
pub fn save(data: &TokenData) -> Result<()> {
    save_named("chatgpt", data)
}

/// 從磁碟載入 token，不存在則回傳 None
/// Load token from disk; returns None if the file does not exist
pub fn load_named(name: &str) -> Result<Option<TokenData>> {
    let named = token_path_named(name)?;
    let legacy = legacy_token_path()?;

    // 優先讀取 tokens-<name>.json；如果 name 為 chatgpt 且不存在，退回 tokens.json
    // Prefer tokens-<name>.json; if name is chatgpt and missing, fall back to tokens.json
    let path = if named.exists() {
        named
    } else if normalize_name(name) == "chatgpt" && legacy.exists() {
        legacy
    } else {
        return Ok(None);
    };

    let json = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "無法讀取 token 檔案 / Failed to read token file: {}",
            path.display()
        )
    })?;
    let data: TokenData = serde_json::from_str(&json)
        .context("無法解析 token 檔案 / Failed to parse token file")?;
    Ok(Some(data))
}

/// 舊 API（向後相容）：等同 load_named(\"chatgpt\")
/// Legacy API (backward compatibility): same as load_named(\"chatgpt\")
#[allow(dead_code)]
#[deprecated(note = "Use load_named instead")]
pub fn load() -> Result<Option<TokenData>> {
    load_named("chatgpt")
}

/// 刪除已儲存的 token 檔案
/// Delete the saved token file
pub fn delete_named(name: &str) -> Result<()> {
    let named = token_path_named(name)?;
    if named.exists() {
        std::fs::remove_file(&named).with_context(|| {
            format!(
                "無法刪除 token 檔案 / Failed to delete token file: {}",
                named.display()
            )
        })?;
    }

    // 若 name 是 chatgpt，一併清除舊檔 tokens.json（避免舊行為誤讀）
    // If name is chatgpt, also remove legacy tokens.json to avoid accidental fallback
    if normalize_name(name) == "chatgpt" {
        let legacy = legacy_token_path()?;
        if legacy.exists() {
            let _ = std::fs::remove_file(&legacy);
        }
    }
    Ok(())
}

/// 舊 API（向後相容）：等同 delete_named(\"chatgpt\")
/// Legacy API (backward compatibility): same as delete_named(\"chatgpt\")
#[allow(dead_code)]
#[deprecated(note = "Use delete_named instead")]
pub fn delete() -> Result<()> {
    delete_named("chatgpt")
}
