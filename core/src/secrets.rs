use azure_identity::DefaultAzureCredential;
use azure_security_keyvault::SecretClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Comprehensive secrets management with Azure Key Vault integration
/// Provides centralized, secure storage and retrieval of all application secrets
#[derive(Clone)]
pub struct SecretsManager {
    client: Option<SecretClient>,
    cache: Arc<RwLock<HashMap<String, CachedSecret>>>,
    vault_url: String,
    cache_ttl: Duration,
    rotation_check_interval: Duration,
}

#[derive(Clone, Debug)]
struct CachedSecret {
    value: String,
    cached_at: Instant,
    version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub name: String,
    pub version: Option<String>,
    pub enabled: bool,
    pub expires: Option<String>,
    pub created: String,
    pub updated: String,
}

impl SecretsManager {
    /// Create a new SecretsManager with Azure Key Vault integration
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let vault_url = std::env::var("KEY_VAULT_URL")
            .unwrap_or_else(|_| "https://policycortex-kv.vault.azure.net/".to_string());

        // Try to create Key Vault client
        let client = match Self::create_keyvault_client(&vault_url).await {
            Ok(c) => {
                info!("✅ Connected to Azure Key Vault: {}", vault_url);
                Some(c)
            }
            Err(e) => {
                warn!(
                    "⚠️ Key Vault not available, using environment variables: {}",
                    e
                );
                None
            }
        };

        Ok(Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            vault_url,
            cache_ttl: Duration::from_secs(300), // 5 minutes cache
            rotation_check_interval: Duration::from_secs(3600), // 1 hour
        })
    }

    async fn create_keyvault_client(
        vault_url: &str,
    ) -> Result<SecretClient, Box<dyn std::error::Error>> {
        let credential = DefaultAzureCredential::default();
        Ok(SecretClient::new(vault_url, Arc::new(credential))?)
    }

    /// Get a secret value by name
    pub async fn get_secret(&self, name: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(name) {
                if cached.cached_at.elapsed() < self.cache_ttl {
                    return Ok(cached.value.clone());
                }
            }
        }

        // Try Key Vault
        if let Some(ref client) = self.client {
            match client.get(name).await {
                Ok(secret) => {
                    let value = secret.value().to_string();

                    // Update cache
                    let mut cache = self.cache.write().await;
                    cache.insert(
                        name.to_string(),
                        CachedSecret {
                            value: value.clone(),
                            cached_at: Instant::now(),
                            version: None,
                        },
                    );

                    info!("Retrieved secret '{}' from Key Vault", name);
                    return Ok(value);
                }
                Err(e) => {
                    warn!("Failed to get secret '{}' from Key Vault: {}", name, e);
                }
            }
        }

        // Fallback to environment variable
        std::env::var(name).map_err(|e| {
            format!(
                "Secret '{}' not found in Key Vault or environment: {}",
                name, e
            )
            .into()
        })
    }

    /// Set or update a secret
    pub async fn set_secret(
        &self,
        name: &str,
        value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref client) = self.client {
            client.set(name, value).await?;

            // Invalidate cache
            let mut cache = self.cache.write().await;
            cache.remove(name);

            info!("Updated secret '{}' in Key Vault", name);
            Ok(())
        } else {
            Err("Key Vault client not available".into())
        }
    }

    /// Delete a secret
    pub async fn delete_secret(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref client) = self.client {
            client.start_delete(name).await?;

            // Remove from cache
            let mut cache = self.cache.write().await;
            cache.remove(name);

            info!("Deleted secret '{}' from Key Vault", name);
            Ok(())
        } else {
            Err("Key Vault client not available".into())
        }
    }

    /// List all secrets (metadata only, not values)
    pub async fn list_secrets(&self) -> Result<Vec<SecretMetadata>, Box<dyn std::error::Error>> {
        if let Some(ref client) = self.client {
            let mut secrets = Vec::new();
            let mut pages = client.list_secrets();

            while let Some(page) = pages.next().await {
                for secret in page? {
                    secrets.push(SecretMetadata {
                        name: secret.id().name().to_string(),
                        version: secret.id().version().map(|v| v.to_string()),
                        enabled: secret.attributes().enabled().unwrap_or(true),
                        expires: secret.attributes().expires().map(|e| e.to_string()),
                        created: secret
                            .attributes()
                            .created()
                            .map(|c| c.to_string())
                            .unwrap_or_default(),
                        updated: secret
                            .attributes()
                            .updated()
                            .map(|u| u.to_string())
                            .unwrap_or_default(),
                    });
                }
            }

            Ok(secrets)
        } else {
            Err("Key Vault client not available".into())
        }
    }

    /// Rotate a secret with a new value
    pub async fn rotate_secret(
        &self,
        name: &str,
        new_value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Store the new secret version
        self.set_secret(name, new_value).await?;

        // Log rotation for audit purposes
        info!("🔄 Rotated secret '{}'", name);

        Ok(())
    }

    /// Get all application secrets with proper defaults
    pub async fn get_all_app_secrets(&self) -> HashMap<String, String> {
        let mut secrets = HashMap::new();

        // List of required secrets for the application
        let required_secrets = vec![
            (
                "AZURE_SUBSCRIPTION_ID",
                "205b477d-17e7-4b3b-92c1-32cf02626b78",
            ),
            ("AZURE_TENANT_ID", "9ef5b184-d371-462a-bc75-5024ce8baff7"),
            ("AZURE_CLIENT_ID", "1ecc95d1-e5bb-43e2-9324-30a17cb6b01c"),
            (
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/policycortex",
            ),
            ("REDIS_URL", "redis://localhost:6379"),
            ("JWT_SECRET", "your-256-bit-secret-key-for-jwt-signing"),
            ("ENCRYPTION_KEY", "your-256-bit-encryption-key"),
            ("API_KEY", "your-api-key"),
        ];

        for (name, default) in required_secrets {
            let value = match self.get_secret(name).await {
                Ok(v) => v,
                Err(_) => {
                    warn!("Using default value for secret '{}'", name);
                    default.to_string()
                }
            };
            secrets.insert(name.to_string(), value);
        }

        secrets
    }

    /// Validate that all required secrets are present
    pub async fn validate_secrets(&self) -> Result<(), Vec<String>> {
        let mut missing = Vec::new();

        let required = vec![
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID",
            "DATABASE_URL",
        ];

        for name in required {
            if self.get_secret(name).await.is_err() {
                missing.push(name.to_string());
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    /// Clear the secret cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Cleared secrets cache");
    }
}

/// Secret scanning patterns for detecting leaked secrets
pub struct SecretScanner {
    patterns: Vec<SecretPattern>,
}

#[derive(Clone)]
struct SecretPattern {
    name: String,
    regex: regex::Regex,
    entropy_threshold: Option<f64>,
}

impl SecretScanner {
    pub fn new() -> Self {
        let patterns = vec![
            SecretPattern {
                name: "Azure Client Secret".to_string(),
                regex: regex::Regex::new(r"[a-zA-Z0-9~_.-]{34}").unwrap(),
                entropy_threshold: Some(4.5),
            },
            SecretPattern {
                name: "Azure Storage Key".to_string(),
                regex: regex::Regex::new(r"[a-zA-Z0-9/+=]{86}==").unwrap(),
                entropy_threshold: None,
            },
            SecretPattern {
                name: "JWT Token".to_string(),
                regex: regex::Regex::new(r"ey[A-Za-z0-9-_]+\.ey[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+").unwrap(),
                entropy_threshold: None,
            },
            SecretPattern {
                name: "Private Key".to_string(),
                regex: regex::Regex::new(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                entropy_threshold: None,
            },
            SecretPattern {
                name: "API Key".to_string(),
                regex: regex::Regex::new(r#"[aA][pP][iI][-_]?[kK][eE][yY]\s*[:=]\s*['"]?[a-zA-Z0-9]{32,}['"]?"#).unwrap(),
                entropy_threshold: None,
            },
            SecretPattern {
                name: "Connection String".to_string(),
                regex: regex::Regex::new(r"(DefaultEndpointsProtocol|Data Source|Server|Initial Catalog|User ID|Password)=[^;]+").unwrap(),
                entropy_threshold: None,
            },
        ];

        Self { patterns }
    }

    /// Scan text for potential secrets
    pub fn scan(&self, text: &str) -> Vec<SecretDetection> {
        let mut detections = Vec::new();

        for pattern in &self.patterns {
            for mat in pattern.regex.find_iter(text) {
                let matched_text = mat.as_str();

                // Check entropy if threshold is set
                if let Some(threshold) = pattern.entropy_threshold {
                    if Self::calculate_entropy(matched_text) < threshold {
                        continue;
                    }
                }

                detections.push(SecretDetection {
                    pattern_name: pattern.name.clone(),
                    location: mat.start(),
                    length: mat.len(),
                    severity: SecretSeverity::High,
                });
            }
        }

        detections
    }

    fn calculate_entropy(s: &str) -> f64 {
        let mut char_counts = HashMap::new();
        for c in s.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }

        let len = s.len() as f64;
        let mut entropy = 0.0;

        for count in char_counts.values() {
            let p = *count as f64 / len;
            entropy -= p * p.log2();
        }

        entropy
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SecretDetection {
    pub pattern_name: String,
    pub location: usize,
    pub length: usize,
    pub severity: SecretSeverity,
}

#[derive(Debug, Clone, Serialize)]
pub enum SecretSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_secret_scanner() {
        let scanner = SecretScanner::new();

        let test_text = r#"
            api_key: "sk-1234567890abcdef1234567890abcdef"
            connection: "Server=localhost;Database=test;User ID=admin;Password=secret123"
        "#;

        let detections = scanner.scan(test_text);
        assert!(!detections.is_empty());
    }
}
