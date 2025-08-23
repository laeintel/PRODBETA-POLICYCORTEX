// Azure Client Module
// Provides unified client for all Azure service interactions

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error, debug};
use tokio::time::{sleep, Duration};

use super::{AzureConfig, AzureResponse, auth::AzureAuthProvider};

/// Unified Azure client for all service interactions
#[derive(Clone)]
pub struct AzureClient {
    pub auth: Arc<AzureAuthProvider>,
    pub config: AzureConfig,
    pub http_client: reqwest::Client,
}

impl AzureClient {
    /// Create new Azure client
    pub async fn new() -> Result<Self> {
        let config = AzureConfig::from_env()?;
        let auth = Arc::new(AzureAuthProvider::new(config.clone()).await?);
        
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()?;

        Ok(Self {
            auth,
            config,
            http_client,
        })
    }

    /// Execute GET request to Azure Management API with retry logic
    pub async fn get_management<T>(&self, path: &str, api_version: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = super::build_management_url(path, api_version);
        self.execute_with_retry(&url, "management").await
    }

    /// Execute GET request to Microsoft Graph API with retry logic
    pub async fn get_graph<T>(&self, path: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = super::build_graph_url(path);
        self.execute_with_retry(&url, "graph").await
    }

    /// Execute POST request to Azure Management API
    pub async fn post_management<T, B>(&self, path: &str, api_version: &str, body: &B) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
        B: Serialize,
    {
        let url = super::build_management_url(path, api_version);
        let token = self.auth.get_management_token().await?;
        
        let response = self.http_client
            .post(&url)
            .bearer_auth(token)
            .json(body)
            .send()
            .await
            .context("Failed to send POST request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("Azure API error: {} - {}", status, text);
            anyhow::bail!("Azure API returned error: {} - {}", status, text);
        }

        response.json::<T>().await.context("Failed to deserialize response")
    }

    /// Execute request with retry logic
    async fn execute_with_retry<T>(&self, url: &str, api_type: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let max_retries = 3;
        let mut retry_count = 0;

        loop {
            let token = match api_type {
                "management" => self.auth.get_management_token().await?,
                "graph" => self.auth.get_graph_token().await?,
                _ => self.auth.get_management_token().await?,
            };

            let response = self.http_client
                .get(url)
                .bearer_auth(&token)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        return resp.json::<T>().await.context("Failed to deserialize response");
                    } else if resp.status() == 429 {
                        // Rate limited, wait and retry
                        retry_count += 1;
                        if retry_count >= max_retries {
                            anyhow::bail!("Max retries exceeded due to rate limiting");
                        }
                        
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(2_u64.pow(retry_count));
                        
                        warn!("Rate limited, retrying after {} seconds", retry_after);
                        sleep(Duration::from_secs(retry_after)).await;
                    } else {
                        let status = resp.status();
                        let text = resp.text().await.unwrap_or_default();
                        error!("Azure API error: {} - {}", status, text);
                        anyhow::bail!("Azure API returned error: {} - {}", status, text);
                    }
                }
                Err(e) if retry_count < max_retries => {
                    retry_count += 1;
                    warn!("Request failed, retry {} of {}: {}", retry_count, max_retries, e);
                    sleep(Duration::from_secs(2_u64.pow(retry_count))).await;
                }
                Err(e) => {
                    error!("Request failed after {} retries: {}", max_retries, e);
                    return Err(e.into());
                }
            }
        }
    }

    /// Get all pages of a paginated response
    pub async fn get_all_pages<T>(&self, initial_path: &str, api_version: &str) -> Result<Vec<T>>
    where
        T: for<'de> Deserialize<'de> + Clone,
    {
        let mut all_items = Vec::new();
        let mut next_link = Some(super::build_management_url(initial_path, api_version));

        while let Some(url) = next_link {
            let response: AzureResponse<T> = self.execute_with_retry(&url, "management").await?;
            all_items.extend(response.value);
            next_link = response.next_link;
        }

        Ok(all_items)
    }

    /// Health check for Azure connectivity
    pub async fn health_check(&self) -> Result<HealthCheckResult> {
        let mut result = HealthCheckResult::default();

        // Check management API
        match self.get_management::<serde_json::Value>(
            &format!("/subscriptions/{}", self.config.subscription_id),
            "2022-12-01"
        ).await {
            Ok(_) => {
                result.management_api = true;
                info!("Management API health check passed");
            }
            Err(e) => {
                warn!("Management API health check failed: {}", e);
            }
        }

        // Check Graph API
        match self.get_graph::<serde_json::Value>("/me").await {
            Ok(_) => {
                result.graph_api = true;
                info!("Graph API health check passed");
            }
            Err(e) => {
                warn!("Graph API health check failed: {}", e);
            }
        }

        result.healthy = result.management_api && result.graph_api;
        Ok(result)
    }
}

#[derive(Debug, Default, Serialize)]
pub struct HealthCheckResult {
    pub healthy: bool,
    pub management_api: bool,
    pub graph_api: bool,
}

/// Create a shared Azure client instance
pub async fn create_shared_client() -> Result<Arc<AzureClient>> {
    Ok(Arc::new(AzureClient::new().await?))
}