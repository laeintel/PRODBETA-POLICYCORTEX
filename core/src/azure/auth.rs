// Azure Authentication Module
// Handles Azure AD authentication and token management

use anyhow::{Result, Context};
use azure_core::auth::{AccessToken, TokenCredential};
use azure_identity::{
    DefaultAzureCredential, DefaultAzureCredentialBuilder,
    ClientSecretCredential, ManagedIdentityCredential,
};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Azure authentication provider
#[derive(Clone)]
pub struct AzureAuthProvider {
    credential: Arc<DefaultAzureCredential>,
    token_cache: Arc<RwLock<Option<CachedToken>>>,
    config: super::AzureConfig,
}

#[derive(Clone, Debug)]
struct CachedToken {
    token: String,
    expires_at: DateTime<Utc>,
}

impl AzureAuthProvider {
    /// Create new authentication provider
    pub async fn new(config: super::AzureConfig) -> Result<Self> {
        info!("Initializing Azure authentication provider");
        
        // Build credential chain
        let credential = DefaultAzureCredentialBuilder::new()
            .exclude_azure_cli_credential()  // Use managed identity or service principal
            .build()
            .context("Failed to build Azure credential")?;

        Ok(Self {
            credential: Arc::new(credential),
            token_cache: Arc::new(RwLock::new(None)),
            config,
        })
    }

    /// Get access token for Azure Management API
    pub async fn get_management_token(&self) -> Result<String> {
        self.get_token("https://management.azure.com/.default").await
    }

    /// Get access token for Microsoft Graph API
    pub async fn get_graph_token(&self) -> Result<String> {
        self.get_token("https://graph.microsoft.com/.default").await
    }

    /// Get access token with caching
    async fn get_token(&self, scope: &str) -> Result<String> {
        // Check cache first
        {
            let cache = self.token_cache.read().await;
            if let Some(cached) = cache.as_ref() {
                if cached.expires_at > Utc::now() + Duration::minutes(5) {
                    debug!("Using cached token for scope: {}", scope);
                    return Ok(cached.token.clone());
                }
            }
        }

        // Get new token
        info!("Acquiring new token for scope: {}", scope);
        let token = self.credential
            .get_token(&[scope])
            .await
            .context("Failed to get Azure access token")?;

        // Cache the token
        {
            let mut cache = self.token_cache.write().await;
            *cache = Some(CachedToken {
                token: token.token.secret().to_string(),
                expires_at: DateTime::from_timestamp(token.expires_on.unix_timestamp(), 0)
                    .unwrap_or_else(|| Utc::now() + Duration::hours(1)),
            });
        }

        Ok(token.token.secret().to_string())
    }

    /// Get HTTP client with authentication headers
    pub async fn get_authenticated_client(&self, token_scope: &str) -> Result<reqwest::Client> {
        let token = self.get_token(token_scope).await?;
        
        let client = reqwest::Client::builder()
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", token).parse()?,
                );
                headers.insert(
                    reqwest::header::CONTENT_TYPE,
                    "application/json".parse()?,
                );
                headers
            })
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(client)
    }

    /// Get management API client
    pub async fn get_management_client(&self) -> Result<reqwest::Client> {
        self.get_authenticated_client("https://management.azure.com/.default").await
    }

    /// Get Graph API client
    pub async fn get_graph_client(&self) -> Result<reqwest::Client> {
        self.get_authenticated_client("https://graph.microsoft.com/.default").await
    }
}

/// Token response from Azure AD
#[derive(Debug, Deserialize, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub scope: String,
}

/// Helper function to create auth provider from environment
pub async fn from_env() -> Result<AzureAuthProvider> {
    let config = super::AzureConfig::from_env()?;
    AzureAuthProvider::new(config).await
}