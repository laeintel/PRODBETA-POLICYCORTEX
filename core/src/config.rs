use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub environment: String,
    pub service_version: String,
    pub allowed_origins: Vec<String>,
    pub require_strict_audience: bool,
}

impl AppConfig {
    pub fn load() -> Self {
        let environment = env::var("ENVIRONMENT").unwrap_or_else(|_| "dev".to_string());

        // Prefer explicit SERVICE_VERSION, fall back to Cargo package version
        let service_version = env::var("SERVICE_VERSION")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());

        // Comma-separated list of origins
        let allowed_origins = env::var("ALLOWED_ORIGINS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        // Default to strict audience in non-dev environments
        let require_strict_audience = match env::var("REQUIRE_STRICT_AUDIENCE") {
            Ok(val) => matches!(val.as_str(), "1" | "true" | "TRUE"),
            Err(_) => environment != "dev",
        };

        Self {
            environment,
            service_version,
            allowed_origins,
            require_strict_audience,
        }
    }
}
