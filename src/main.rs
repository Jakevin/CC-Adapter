mod auth;
mod config;
mod convert;
mod error;
mod providers;
mod server;
mod types;

use std::path::PathBuf;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::prelude::*;

use config::{Cli, Commands, ServeArgs};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // 決定日誌等級，優先順序：CLI --log-level > 環境變數 RUST_LOG > config.toml > 預設 "info"
    // Determine log level priority: CLI --log-level > env RUST_LOG > config.toml > default "info"
    let log_level = resolve_log_level(&cli);
    let log_file = resolve_log_file(&cli);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&log_level));

    // 若有設定 log_file，日誌同時輸出到 console 和檔案
    // If log_file is set, output logs to both console and file
    if let Some(ref log_file_path) = log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file_path)
            .map_err(|e| anyhow::anyhow!(
                "無法開啟日誌檔案 / Failed to open log file '{}': {}",
                log_file_path, e
            ))?;

        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::sync::Mutex::new(file))
            .with_ansi(false);

        let console_layer = tracing_subscriber::fmt::layer();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(console_layer)
            .with(file_layer)
            .init();

        info!(
            path = %log_file_path,
            "日誌同時寫入檔案 / Logs are also written to file"
        );
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    match cli.command {
        Some(Commands::Login) => run_login().await,
        Some(Commands::Logout) => run_logout(),
        Some(Commands::Serve(args)) => run_serve(args).await,
        // 未指定子命令時預設啟動伺服器
        // Default to serve when no subcommand is given
        None => run_serve(ServeArgs::default()).await,
    }
}

/// 決定最終的日誌等級
/// Determine the final log level
fn resolve_log_level(cli: &Cli) -> String {
    // 1. CLI --log-level 最優先
    // 1. CLI --log-level takes highest priority
    if let Some(level) = &cli.log_level {
        return level.clone();
    }

    // 2. 環境變數 RUST_LOG 由 EnvFilter::try_from_default_env 處理，這裡不介入
    // 2. RUST_LOG env var is handled by EnvFilter::try_from_default_env, skip here

    // 3. 從 config.toml 中讀取（僅 serve 命令適用）
    // 3. Read from config.toml (applicable to serve command only)
    let config_path = match &cli.command {
        Some(Commands::Serve(args)) => args.config.clone(),
        None => std::path::PathBuf::from("config.toml"),
        _ => return "info".to_string(),
    };

    if let Some(level) = config::Config::peek_log_level(&config_path) {
        return level;
    }

    // 4. 預設值
    // 4. Default
    "info".to_string()
}

/// 決定日誌檔案路徑
/// Determine the log file path
fn resolve_log_file(cli: &Cli) -> Option<String> {
    // 從 config.toml 中讀取（僅 serve 命令適用）
    // Read from config.toml (applicable to serve command only)
    let config_path = match &cli.command {
        Some(Commands::Serve(args)) => args.config.clone(),
        None => std::path::PathBuf::from("config.toml"),
        _ => return None,
    };

    config::Config::peek_log_file(&config_path)
}

/// 確保 Claude Code 的 onboarding 已略過（修改 ~/.claude.json）
/// Ensure Claude Code onboarding is skipped (modifies ~/.claude.json)
fn ensure_claude_onboarding() {
    let Some(claude_json) = dirs::home_dir().map(|h| h.join(".claude.json")) else {
        return;
    };

    // 讀取現有設定（若存在），避免覆寫使用者其他欄位
    // Read existing config (if present) to avoid overwriting other fields
    let mut config: serde_json::Value = if claude_json.exists() {
        std::fs::read_to_string(&claude_json)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let Some(obj) = config.as_object_mut() else { return };

    // 已經設定過就跳過
    // Skip if already configured
    if obj.get("hasCompletedOnboarding").and_then(|v| v.as_bool()) == Some(true) {
        return;
    }

    obj.insert("hasCompletedOnboarding".to_string(), serde_json::json!(true));
    obj.insert("hasTrustDialogAccepted".to_string(), serde_json::json!(true));

    if !obj.contains_key("customApiKeyResponses") {
        obj.insert("customApiKeyResponses".to_string(), serde_json::json!({
            "approved": []
        }));
    }

    // 原子寫入
    // Atomic write
    let tmp_path = claude_json.with_extension("tmp");
    if let Ok(content) = serde_json::to_string_pretty(&config)
        && std::fs::write(&tmp_path, &content).is_ok()
    {
        let _ = std::fs::rename(&tmp_path, &claude_json);
    }

    info!(
        path = %claude_json.display(),
        "已自動略過 Claude Code 首次登入設定 / Auto-skipped Claude Code onboarding"
    );
}

/// 執行 OAuth 登入流程
/// Run the OAuth login flow
async fn run_login() -> anyhow::Result<()> {
    println!("正在啟動 ChatGPT OAuth 登入流程...");
    println!("Starting ChatGPT OAuth login flow...\n");

    let pkce = auth::oauth::generate_pkce();
    let state = auth::oauth::generate_state();
    let url = auth::oauth::build_authorize_url(&pkce.challenge, &state);

    println!("正在開啟瀏覽器前往 OpenAI 登入頁面...");
    println!("Opening browser to OpenAI login page...\n");

    if let Err(e) = open::that(&url) {
        println!("無法自動開啟瀏覽器：{}", e);
        println!("Failed to open browser automatically: {}\n", e);
        println!("請手動開啟以下 URL / Please open this URL manually:\n{}\n", url);
    }

    println!("等待 OAuth 回調中（超時 120 秒）...");
    println!("Waiting for OAuth callback (120 second timeout)...\n");

    let auth_code = auth::callback_server::wait_for_callback(state).await?;

    println!("收到授權碼，正在交換 token...");
    println!("Received authorization code, exchanging for token...\n");

    let token_data = auth::oauth::exchange_code(&auth_code.code, &pkce.verifier).await?;
    auth::token_store::save(&token_data)?;

    match auth::oauth::extract_account_id(&token_data.access_token) {
        Ok(account_id) => {
            println!("登入成功！ChatGPT Account ID: {}", account_id);
            println!("Login successful! ChatGPT Account ID: {}\n", account_id);
        }
        Err(_) => {
            println!("登入成功！（無法提取 Account ID，可能需要重新登入）");
            println!("Login successful! (Could not extract Account ID, re-login may be needed)\n");
        }
    }

    println!("Token 已儲存至 ~/.claude-adapter/tokens.json");
    println!("Token saved to ~/.claude-adapter/tokens.json\n");
    println!("現在可以使用 `claude-adapter serve` 或 `claude-adapter` 啟動伺服器。");
    println!("You can now start the server with `claude-adapter serve` or `claude-adapter`.");

    Ok(())
}

/// 清除已儲存的 OAuth token
/// Clear saved OAuth tokens
fn run_logout() -> anyhow::Result<()> {
    auth::token_store::delete()?;
    println!("已清除 OAuth token。");
    println!("OAuth token cleared.");
    Ok(())
}


// ---------------------------------------------------------------------------
// Claude Code settings.json 管理 / Claude Code settings.json management
// ---------------------------------------------------------------------------

/// ~/.claude/settings.json 路徑
fn claude_settings_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".claude").join("settings.json"))
}

/// 備份檔路徑：~/.claude-adapter/base_url_backup.json
fn backup_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".claude-adapter").join("base_url_backup.json"))
}

/// 將 ANTHROPIC_BASE_URL 注入 ~/.claude/settings.json，讓 Claude Code 自動使用本地代理
fn inject_claude_settings(host: &str, port: u16) {
    let Some(path) = claude_settings_path() else { return };

    let mut settings: serde_json::Value = if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let connect_host = if host == "0.0.0.0" { "127.0.0.1" } else { host };
    let proxy_url = format!("http://{}:{}", connect_host, port);

    // 備份當前的 ANTHROPIC_BASE_URL（若有），供關閉時還原
    let old_value = settings
        .get("env")
        .and_then(|env| env.get("ANTHROPIC_BASE_URL"))
        .cloned();

    if let Some(bp) = backup_path() {
        if let Some(parent) = bp.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let backup = serde_json::json!({
            "had_value": old_value.is_some(),
            "old_value": old_value,
        });
        if let Ok(json) = serde_json::to_string_pretty(&backup) {
            let _ = std::fs::write(&bp, json);
        }
    }

    if !settings.get("env").is_some_and(|v| v.is_object()) {
        settings["env"] = serde_json::json!({});
    }
    settings["env"]["ANTHROPIC_BASE_URL"] = serde_json::Value::String(proxy_url.clone());

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // 原子寫入：先寫 temp 再 rename，防止損壞
    let tmp_path = path.with_extension("tmp");
    match serde_json::to_string_pretty(&settings) {
        Ok(content) => {
            if std::fs::write(&tmp_path, &content).is_ok()
                && std::fs::rename(&tmp_path, &path).is_err()
            {
                let _ = std::fs::write(&path, &content);
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "無法序列化 settings.json / Failed to serialize settings.json");
            return;
        }
    }

    info!(
        path = %path.display(),
        url = %proxy_url,
        "已注入 ANTHROPIC_BASE_URL 至 Claude settings.json / Injected ANTHROPIC_BASE_URL into Claude settings.json"
    );
}

/// 還原 ~/.claude/settings.json 中的 ANTHROPIC_BASE_URL
fn restore_claude_settings() {
    let Some(path) = claude_settings_path() else { return };
    let Some(bp) = backup_path() else { return };

    if !path.exists() { return; }

    let mut settings: serde_json::Value = match std::fs::read_to_string(&path) {
        Ok(c) => serde_json::from_str(&c).unwrap_or_default(),
        Err(_) => return,
    };

    let backup: Option<serde_json::Value> = std::fs::read_to_string(&bp)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    if let Some(env) = settings.get_mut("env").and_then(|v| v.as_object_mut()) {
        match &backup {
            Some(b) if b.get("had_value").and_then(|v| v.as_bool()) == Some(true) => {
                if let Some(old) = b.get("old_value") {
                    env.insert("ANTHROPIC_BASE_URL".to_string(), old.clone());
                }
            }
            _ => {
                env.remove("ANTHROPIC_BASE_URL");
            }
        }

        if env.is_empty()
            && let Some(obj) = settings.as_object_mut()
        {
            obj.remove("env");
        }
    }

    let tmp_path = path.with_extension("tmp");
    if let Ok(content) = serde_json::to_string_pretty(&settings) {
        let _ = std::fs::write(&tmp_path, &content)
            .and_then(|_| std::fs::rename(&tmp_path, &path));
    }

    let _ = std::fs::remove_file(&bp);

    eprintln!(
        "\n  已還原 ~/.claude/settings.json — ANTHROPIC_BASE_URL 已移除"
    );
    eprintln!(
        "  Restored ~/.claude/settings.json — ANTHROPIC_BASE_URL removed\n"
    );
}

/// 啟動 Adapter 代理伺服器
/// Start the Adapter proxy server
async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config = config::Config::load(&args)?;

    let provider_names: Vec<String> = config.providers.keys().cloned().collect();
    info!(
        host = %config.server.host,
        port = config.server.port,
        providers = ?provider_names,
        default_provider = %config.models.default_provider,
        default_model = %config.models.default_model,
        "啟動 Claude API Adapter / Starting Claude API Adapter"
    );

    let state = server::build_app_state(config.clone(), args).await?;

    // 設定路由：POST /v1/messages 為主要端點，GET /health 為健康檢查
    // Setup routes: POST /v1/messages as main endpoint, GET /health for health check
    let app = Router::new()
        .route("/v1/messages", post(server::handle_messages))
        .route("/health", get(server::handle_health))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // 確保 Claude Code onboarding 已略過 + 注入 ANTHROPIC_BASE_URL
    // Ensure Claude Code onboarding is skipped + inject ANTHROPIC_BASE_URL
    ensure_claude_onboarding();
    inject_claude_settings(&config.server.host, config.server.port);

    // 啟動 config.toml 檔案監控任務（背景輪詢 mtime）
    // Spawn config.toml file watcher task (background mtime polling)
    let config_path = state.serve_args.config.clone();
    let watcher_state = state.clone();
    tokio::spawn(async move {
        watch_config(watcher_state, config_path).await;
    });

    info!(addr = %addr, "Adapter 正在監聽 / Adapter is listening");
    println!("\n  Claude API Adapter 執行中 / running at http://{}", addr);
    println!();
    for (name, provider_cfg) in &config.providers {
        let type_info = &provider_cfg.provider_type;
        let url_info = if provider_cfg.base_url.is_empty() {
            "(OAuth)".to_string()
        } else {
            provider_cfg.base_url.clone()
        };
        println!("  供應商 / Provider: {} [{}] ({})", name, type_info, url_info);
    }
    println!("  預設供應商 / Default provider: {}", config.models.default_provider);
    println!("  預設模型 / Default model: {}", config.models.default_model);
    if !config.models.routing.is_empty() {
        println!();
        println!("  模型路由 / Model routing:");
        for (anthropic, route) in &config.models.routing {
            println!("    {} → {} ({})", anthropic, route.model, route.provider);
        }
    }
    println!("\n  已自動設定 ~/.claude/settings.json 中的 ANTHROPIC_BASE_URL");
    println!("  ANTHROPIC_BASE_URL auto-configured in ~/.claude/settings.json");
    println!("  直接開啟新終端執行 claude 即可使用，無需任何環境變數或 shell hook");
    println!("  Just open a new terminal and run `claude` — no env vars or shell hooks needed");
    println!("\n  ⟳ 支援熱重載：修改 config.toml 後自動生效，無需重啟");
    println!("  ⟳ Hot-reload enabled: changes to config.toml take effect automatically");
    println!();

    // 使用 graceful shutdown 確保退出時清理 env 檔案
    // Use graceful shutdown to ensure env file cleanup on exit
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // 伺服器關閉後還原 ~/.claude/settings.json
    // Restore ~/.claude/settings.json after server shutdown
    restore_claude_settings();

    Ok(())
}

// ---------------------------------------------------------------------------
// config.toml 檔案監控 / config.toml file watcher
// ---------------------------------------------------------------------------

/// 輪詢 config.toml 的修改時間，偵測到變更時觸發熱重載
/// Poll config.toml modification time and trigger hot-reload on change
async fn watch_config(state: std::sync::Arc<server::AppState>, config_path: PathBuf) {
    use std::time::Duration;

    let mut last_modified = std::fs::metadata(&config_path)
        .and_then(|m| m.modified())
        .ok();

    let mut interval = tokio::time::interval(Duration::from_secs(2));
    // 跳過第一次立即觸發
    // Skip the first immediate tick
    interval.tick().await;

    loop {
        interval.tick().await;

        let current_modified = std::fs::metadata(&config_path)
            .and_then(|m| m.modified())
            .ok();

        if current_modified != last_modified {
            last_modified = current_modified;

            info!("偵測到 config.toml 變更，正在重新載入... / config.toml changed, reloading...");

            match server::reload_config(&state).await {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "配置重載失敗，維持目前配置 / Config reload failed, keeping current config"
                    );
                    eprintln!("  ✗ 配置重載失敗：{}", e);
                    eprintln!("  ✗ Config reload failed: {}\n", e);
                }
            }
        }
    }
}

/// 等待關閉信號（Ctrl+C）
/// Wait for shutdown signal (Ctrl+C)
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("無法安裝 Ctrl+C 處理器 / Failed to install Ctrl+C handler");
    eprintln!("\n  收到關閉信號，正在停止伺服器...");
    eprintln!("  Received shutdown signal, stopping server...");
}
