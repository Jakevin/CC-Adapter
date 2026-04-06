use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "claude-adapter")]
#[command(about = "API 轉接器，讓 Claude Code 能使用 OpenAI/Grok/ChatGPT 供應商 / API adapter that lets Claude Code use OpenAI/Grok/ChatGPT providers")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// 覆寫日誌等級 (trace, debug, info, warn, error)
    /// Override log level (trace, debug, info, warn, error)
    #[arg(long, global = true)]
    pub log_level: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// 啟動 Adapter 代理伺服器（預設命令）
    /// Start the Adapter proxy server (default command)
    Serve(ServeArgs),

    /// 執行 ChatGPT OAuth 登入流程
    /// Run the ChatGPT OAuth login flow
    Login(LoginArgs),

    /// 清除已儲存的 OAuth token
    /// Clear saved OAuth tokens
    Logout(LogoutArgs),
}

#[derive(Parser, Debug, Clone)]
pub struct LoginArgs {
    /// 要綁定到哪個 Provider 名稱（對應 config.toml 的 [providers.<name>]）
    /// Provider name to bind this login to (matches [providers.<name>] in config.toml)
    #[arg(long, default_value = "chatgpt")]
    pub name: String,
}

#[derive(Parser, Debug, Clone)]
pub struct LogoutArgs {
    /// 要清除哪個 Provider 名稱的 token（對應 config.toml 的 [providers.<name>]）
    /// Provider name whose token should be cleared (matches [providers.<name>] in config.toml)
    #[arg(long, default_value = "chatgpt")]
    pub name: String,
}

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// 配置檔路徑
    /// Path to config file
    #[arg(short, long, default_value = "config.toml")]
    pub config: PathBuf,

    /// 覆寫監聽主機
    /// Override listen host
    #[arg(long)]
    pub host: Option<String>,

    /// 覆寫監聽埠號
    /// Override listen port
    #[arg(short, long)]
    pub port: Option<u16>,

    /// 覆寫供應商 API 金鑰（僅適用於舊版單一供應商配置）
    /// Override provider API key (only for legacy single-provider config)
    #[arg(long)]
    pub api_key: Option<String>,

    /// 覆寫供應商 Base URL（僅適用於舊版單一供應商配置）
    /// Override provider base URL (only for legacy single-provider config)
    #[arg(long)]
    pub base_url: Option<String>,

    /// 覆寫預設模型
    /// Override default model
    #[arg(long)]
    pub model: Option<String>,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            config: PathBuf::from("config.toml"),
            host: None,
            port: None,
            api_key: None,
            base_url: None,
            model: None,
        }
    }
}

// ---------------------------------------------------------------------------
// 最終解析後的配置結構 / Resolved configuration structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub providers: HashMap<String, ProviderConfig>,
    pub models: ModelsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_log_level")]
    #[allow(dead_code)]
    pub log_level: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub log_file: Option<String>,
    /// 是否啟用寫入日誌檔案（預設為啟用）
    /// Whether to enable writing logs to file (enabled by default)
    #[serde(default = "default_log_file_enabled")]
    #[allow(dead_code)]
    pub log_file_enabled: bool,
    /// 寫入 ~/.claude/settings.json 的 `CLAUDE_STREAM_IDLE_TIMEOUT_MS`（毫秒），停止 adapter 時還原。
    /// 預設 300000（5 分鐘）；設為 0 表示不注入、不備份還原此變數。
    /// Writes `CLAUDE_STREAM_IDLE_TIMEOUT_MS` to ~/.claude/settings.json env; restored on shutdown.
    /// Default 300000 (5 min); 0 = do not inject or backup/restore this variable.
    #[serde(default = "default_claude_stream_idle_timeout_ms")]
    pub claude_stream_idle_timeout_ms: u64,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_file_enabled() -> bool {
    true
}

fn default_claude_stream_idle_timeout_ms() -> u64 {
    300_000
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub supports_streaming: bool,
}

#[derive(Debug, Clone)]
pub struct ModelsConfig {
    pub default_provider: String,
    pub default_model: String,
    pub routing: HashMap<String, ModelRoute>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelRoute {
    pub provider: String,
    pub model: String,
}

impl ModelsConfig {
    /// 根據 Anthropic 模型名稱解析目標供應商與模型
    ///
    /// 解析規則：
    /// 1. 先嘗試精確比對 key == anthropic_model
    /// 2. 若無精確比對，則尋找「最長的前綴 key」，滿足 anthropic_model.starts_with(key)
    ///    例如 key = "claude-haiku-4-5"，可匹配 "claude-haiku-4-5-20251001"
    /// 3. 若仍無對應，回退到 default_provider + default_model
    ///
    /// Resolve target provider and model from an Anthropic model name.
    /// Resolution order:
    /// 1. Exact match on the full model name
    /// 2. Longest prefix match where anthropic_model.starts_with(key)
    ///    e.g. key "claude-haiku-4-5" matches "claude-haiku-4-5-20251001"
    /// 3. Fallback to default_provider + default_model
    pub fn resolve(&self, anthropic_model: &str) -> (String, String) {
        // 1. 精確比對 / Exact match
        if let Some(route) = self.routing.get(anthropic_model) {
            return (route.provider.clone(), route.model.clone());
        }

        // 2. 最長前綴比對 / Longest prefix match
        let mut best_route: Option<(&String, &ModelRoute)> = None;

        for (key, route) in &self.routing {
            if anthropic_model.starts_with(key.as_str()) {
                match best_route {
                    Some((best_key, _)) if best_key.len() >= key.len() => {
                        // 已有更長或同長度的前綴，略過
                    }
                    _ => {
                        best_route = Some((key, route));
                    }
                }
            }
        }

        if let Some((_, route)) = best_route {
            return (route.provider.clone(), route.model.clone());
        }

        // 3. 回退預設值 / Fallback to defaults
        (self.default_provider.clone(), self.default_model.clone())
    }
}

// ---------------------------------------------------------------------------
// 用於 TOML 反序列化的中間結構（支援新舊格式）
// Intermediate structs for TOML deserialization (supports both old and new format)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawConfig {
    server: ServerConfig,
    /// 舊格式：單一供應商 / Legacy: single provider
    provider: Option<RawProviderConfig>,
    /// 新格式：多供應商 / New: multiple providers
    providers: Option<HashMap<String, ProviderConfig>>,
    models: RawModelsConfig,
}

/// 舊格式的 ProviderConfig（api_key/base_url 為必填 String）
/// Legacy ProviderConfig (api_key/base_url are required Strings)
#[derive(Debug, Deserialize)]
struct RawProviderConfig {
    #[serde(rename = "type")]
    provider_type: String,
    api_key: String,
    base_url: String,
    #[serde(default)]
    #[allow(dead_code)]
    supports_streaming: bool,
}

#[derive(Debug, Deserialize)]
struct RawModelsConfig {
    /// 舊格式：預設模型 / Legacy: default model
    default: Option<String>,
    /// 舊格式：映射表 / Legacy: mapping table
    #[serde(default)]
    mapping: Option<HashMap<String, toml::Value>>,

    /// 新格式：預設供應商 / New: default provider
    default_provider: Option<String>,
    /// 新格式：預設模型 / New: default model
    default_model: Option<String>,
    /// 新格式：路由表 / New: routing table
    #[serde(default)]
    routing: Option<HashMap<String, ModelRoute>>,
}

impl Config {
    /// 從配置檔中快速讀取 log_level（不需要完整驗證 config）
    /// Quickly read log_level from config file (without full config validation)
    pub fn peek_log_level(config_path: &std::path::Path) -> Option<String> {
        let content = std::fs::read_to_string(config_path).ok()?;
        let table: toml::Table = toml::from_str(&content).ok()?;
        table
            .get("server")?
            .get("log_level")?
            .as_str()
            .map(|s| s.to_string())
    }

    /// 從配置檔中快速讀取 log_file（不需要完整驗證 config）
    /// Quickly read log_file from config file (without full config validation)
    pub fn peek_log_file(config_path: &std::path::Path) -> Option<String> {
        let content = std::fs::read_to_string(config_path).ok()?;
        let table: toml::Table = toml::from_str(&content).ok()?;
        table
            .get("server")?
            .get("log_file")?
            .as_str()
            .map(|s| s.to_string())
    }

    /// 從配置檔中快速讀取 log_file_enabled（不需要完整驗證 config）
    /// Quickly read log_file_enabled from config file (without full config validation)
    pub fn peek_log_file_enabled(config_path: &std::path::Path) -> Option<bool> {
        let content = std::fs::read_to_string(config_path).ok()?;
        let table: toml::Table = toml::from_str(&content).ok()?;
        table
            .get("server")?
            .get("log_file_enabled")?
            .as_bool()
    }

    /// 載入配置：依序從配置檔、環境變數、CLI 參數合併設定
    /// Load config: merge settings from config file, environment variables, and CLI arguments
    pub fn load(args: &ServeArgs) -> Result<Self> {
        let config_path = &args.config;

        let mut config = if config_path.exists() {
            let content = std::fs::read_to_string(config_path)
                .with_context(|| format!("無法讀取配置檔 / Failed to read config file: {}", config_path.display()))?;
            let raw: RawConfig = toml::from_str(&content)
                .with_context(|| format!("無法解析配置檔 / Failed to parse config file: {}", config_path.display()))?;
            Self::resolve_raw(raw)?
        } else {
            Config {
                server: ServerConfig {
                    host: "127.0.0.1".to_string(),
                    port: 8080,
                    log_level: "info".to_string(),
                    log_file: None,
                    log_file_enabled: true,
                    claude_stream_idle_timeout_ms: default_claude_stream_idle_timeout_ms(),
                },
                providers: HashMap::new(),
                models: ModelsConfig {
                    default_provider: "openai".to_string(),
                    default_model: "gpt-4o".to_string(),
                    routing: HashMap::new(),
                },
            }
        };

        // CLI 參數覆寫
        // CLI argument overrides
        if let Some(host) = &args.host {
            config.server.host = host.clone();
        }
        if let Some(port) = args.port {
            config.server.port = port;
        }
        if let Some(model) = &args.model {
            config.models.default_model = model.clone();
        }

        // API 金鑰覆寫（僅作用於預設供應商）
        // API key override (applies to default provider only)
        let api_key_override = args
            .api_key
            .clone()
            .or_else(|| std::env::var("ADAPTER_API_KEY").ok());

        if let Some(key) = api_key_override {
            if let Some(provider) = config.providers.get_mut(&config.models.default_provider) {
                provider.api_key = key;
            }
        }

        // Base URL 覆寫（僅作用於預設供應商）
        // Base URL override (applies to default provider only)
        if let Some(base_url) = &args.base_url {
            if let Some(provider) = config.providers.get_mut(&config.models.default_provider) {
                provider.base_url = base_url.clone();
            }
        }

        // 驗證所有非 chatgpt 供應商都有 API key（或嘗試自動偵測 OAuth）
        // Validate all non-chatgpt providers have an API key (or try OAuth auto-detection)
        Self::validate_providers(&mut config, config_path)?;

        Ok(config)
    }

    /// 將 RawConfig 解析為最終 Config（處理新舊格式）
    /// Resolve RawConfig into final Config (handles old and new format)
    fn resolve_raw(raw: RawConfig) -> Result<Config> {
        let providers = Self::resolve_providers(&raw)?;
        let models = Self::resolve_models(&raw.models, &providers)?;

        Ok(Config {
            server: raw.server,
            providers,
            models,
        })
    }

    /// 解析供應商配置
    /// Resolve provider configuration
    fn resolve_providers(raw: &RawConfig) -> Result<HashMap<String, ProviderConfig>> {
        if let Some(providers) = &raw.providers {
            // 新格式：直接使用 [providers.*]
            Ok(providers.clone())
        } else if let Some(legacy) = &raw.provider {
            // 舊格式：將 [provider] 轉為單一條目的 providers map
            // Legacy: convert [provider] into a single-entry providers map
            let name = legacy.provider_type.clone();
            let config = ProviderConfig {
                provider_type: legacy.provider_type.clone(),
                api_key: legacy.api_key.clone(),
                base_url: legacy.base_url.clone(),
                supports_streaming: legacy.supports_streaming,
            };
            let mut map = HashMap::new();
            map.insert(name, config);
            Ok(map)
        } else {
            anyhow::bail!(
                "配置檔中未找到 [providers] 或 [provider] 區段 / \
                 No [providers] or [provider] section found in config file"
            );
        }
    }

    /// 解析模型配置（routing 表 + 預設值）
    /// Resolve model configuration (routing table + defaults)
    fn resolve_models(
        raw: &RawModelsConfig,
        providers: &HashMap<String, ProviderConfig>,
    ) -> Result<ModelsConfig> {
        // 新格式優先
        // New format takes priority
        if let Some(routing) = &raw.routing {
            let default_provider = raw.default_provider.clone().unwrap_or_else(|| {
                providers.keys().next().cloned().unwrap_or_default()
            });
            let default_model = raw
                .default_model
                .clone()
                .or_else(|| raw.default.clone())
                .unwrap_or_else(|| "gpt-4o".to_string());
            return Ok(ModelsConfig {
                default_provider,
                default_model,
                routing: routing.clone(),
            });
        }

        // 舊格式：將 mapping (HashMap<String, String>) 轉為 routing
        // Legacy: convert mapping into routing using the single provider
        let provider_name = providers.keys().next().cloned().unwrap_or_default();
        let default_model = raw
            .default
            .clone()
            .or_else(|| raw.default_model.clone())
            .unwrap_or_else(|| "gpt-4o".to_string());

        let mut routing = HashMap::new();
        if let Some(mapping) = &raw.mapping {
            for (anthropic_name, value) in mapping {
                match value {
                    // 舊格式純字串 → 使用預設供應商
                    // Legacy plain string → use default provider
                    toml::Value::String(model_name) => {
                        routing.insert(
                            anthropic_name.clone(),
                            ModelRoute {
                                provider: provider_name.clone(),
                                model: model_name.clone(),
                            },
                        );
                    }
                    // 新格式 inline table { provider = "...", model = "..." }
                    toml::Value::Table(table) => {
                        let provider = table
                            .get("provider")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&provider_name)
                            .to_string();
                        let model = table
                            .get("model")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&default_model)
                            .to_string();
                        routing.insert(
                            anthropic_name.clone(),
                            ModelRoute { provider, model },
                        );
                    }
                    _ => {
                        anyhow::bail!(
                            "模型映射 '{}' 的值格式無效，應為字串或 {{ provider, model }} 表格 / \
                             Invalid value format for model mapping '{}', expected string or {{ provider, model }} table",
                            anthropic_name, anthropic_name
                        );
                    }
                }
            }
        }

        let default_provider = raw
            .default_provider
            .clone()
            .unwrap_or(provider_name);

        Ok(ModelsConfig {
            default_provider,
            default_model,
            routing,
        })
    }

    /// 驗證供應商配置，嘗試自動偵測 OAuth token
    /// Validate provider configs, attempt OAuth token auto-detection
    fn validate_providers(config: &mut Config, config_path: &std::path::Path) -> Result<()> {
        let mut missing_keys: Vec<String> = Vec::new();

        for (name, provider) in &config.providers {
            if provider.provider_type == "chatgpt" {
                continue;
            }
            if provider.api_key.is_empty() {
                missing_keys.push(name.clone());
            }
        }

        if missing_keys.is_empty() {
            return Ok(());
        }

        // 如果只有一個供應商且缺少 key，嘗試 OAuth 自動切換（保持舊版行為）
        // If there's exactly one provider missing a key, try OAuth auto-switch (preserves legacy behavior)
        if config.providers.len() == 1 && missing_keys.len() == 1 {
            let has_tokens = dirs::home_dir()
                .map(|h| {
                    let dir = h.join(".claude-adapter");
                    dir.join("tokens.json").exists() || dir.join("tokens-chatgpt.json").exists()
                })
                .unwrap_or(false);

            if has_tokens {
                let name = &missing_keys[0];
                eprintln!(
                    "\n  ⟳ 供應商 '{}' 的 API key 為空但偵測到 OAuth token，自動切換為 chatgpt",
                    name
                );
                eprintln!(
                    "  ⟳ Provider '{}' API key is empty but OAuth token found, auto-switching to chatgpt\n",
                    name
                );

                let old_name = name.clone();
                let mut provider = config.providers.remove(&old_name).unwrap();
                provider.provider_type = "chatgpt".to_string();
                provider.base_url = "https://chatgpt.com/backend-api".to_string();
                config.providers.insert("chatgpt".to_string(), provider);

                if config.models.default_provider == old_name {
                    config.models.default_provider = "chatgpt".to_string();
                }
                for route in config.models.routing.values_mut() {
                    if route.provider == old_name {
                        route.provider = "chatgpt".to_string();
                    }
                }
                return Ok(());
            }
        }

        anyhow::bail!(
            "以下供應商未設定 API 金鑰：{}\n  \
             The following providers have no API key: {}\n  \
             請在 {} 的 [providers.*] 區段設定 api_key，\
             或使用 `claude-adapter login` 登入 ChatGPT OAuth\n  \
             Set api_key in [providers.*] sections of {}, \
             or use `claude-adapter login` for ChatGPT OAuth",
            missing_keys.join(", "),
            missing_keys.join(", "),
            config_path.display(),
            config_path.display()
        );
    }
}
