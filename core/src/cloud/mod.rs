pub mod models;
pub mod providers;
pub mod traits;

pub use models::*;
pub use providers::CloudProviderFactory;
pub use traits::*;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Cloud provider abstraction errors
#[derive(Error, Debug)]
pub enum CloudError {
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    #[error("Policy application failed: {0}")]
    PolicyApplicationFailed(String),
    
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Provider not supported: {0}")]
    ProviderNotSupported(String),
}

/// Result type for cloud operations
pub type CloudResult<T> = Result<T, CloudError>;

/// Supported cloud providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudProvider {
    Azure,
    AWS,
    GCP,
    Alibaba,
    Oracle,
    IBM,
}

impl CloudProvider {
    pub fn from_str(s: &str) -> CloudResult<Self> {
        match s.to_lowercase().as_str() {
            "azure" | "microsoft" => Ok(CloudProvider::Azure),
            "aws" | "amazon" => Ok(CloudProvider::AWS),
            "gcp" | "google" => Ok(CloudProvider::GCP),
            "alibaba" | "alicloud" => Ok(CloudProvider::Alibaba),
            "oracle" | "oci" => Ok(CloudProvider::Oracle),
            "ibm" | "ibmcloud" => Ok(CloudProvider::IBM),
            _ => Err(CloudError::ProviderNotSupported(s.to_string())),
        }
    }
    
    pub fn name(&self) -> &str {
        match self {
            CloudProvider::Azure => "Azure",
            CloudProvider::AWS => "AWS",
            CloudProvider::GCP => "GCP",
            CloudProvider::Alibaba => "Alibaba Cloud",
            CloudProvider::Oracle => "Oracle Cloud",
            CloudProvider::IBM => "IBM Cloud",
        }
    }
}

/// Cloud provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub provider: CloudProvider,
    pub credentials: HashMap<String, String>,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub max_retries: Option<u32>,
    pub enable_caching: bool,
    pub cache_ttl_seconds: Option<u64>,
}

impl CloudConfig {
    pub fn new(provider: CloudProvider) -> Self {
        CloudConfig {
            provider,
            credentials: HashMap::new(),
            region: None,
            endpoint: None,
            timeout_seconds: Some(30),
            max_retries: Some(3),
            enable_caching: true,
            cache_ttl_seconds: Some(300),
        }
    }
    
    pub fn with_credentials(mut self, key: String, value: String) -> Self {
        self.credentials.insert(key, value);
        self
    }
    
    pub fn with_region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }
}

/// Cloud provider health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudHealth {
    pub provider: CloudProvider,
    pub healthy: bool,
    pub latency_ms: u64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
}

/// Multi-cloud manager for handling multiple providers
pub struct MultiCloudManager {
    providers: HashMap<CloudProvider, Arc<dyn CloudProviderTrait>>,
    primary_provider: CloudProvider,
}

impl MultiCloudManager {
    pub fn new(primary: CloudProvider) -> Self {
        MultiCloudManager {
            providers: HashMap::new(),
            primary_provider: primary,
        }
    }
    
    pub fn add_provider(&mut self, provider: Arc<dyn CloudProviderTrait>) {
        self.providers.insert(provider.provider_type(), provider);
    }
    
    pub fn get_provider(&self, provider: CloudProvider) -> CloudResult<Arc<dyn CloudProviderTrait>> {
        self.providers
            .get(&provider)
            .cloned()
            .ok_or_else(|| CloudError::ProviderNotSupported(provider.name().to_string()))
    }
    
    pub fn get_primary(&self) -> CloudResult<Arc<dyn CloudProviderTrait>> {
        self.get_provider(self.primary_provider)
    }
    
    pub async fn health_check_all(&self) -> Vec<CloudHealth> {
        let mut results = Vec::new();
        
        for (provider_type, provider) in &self.providers {
            let start = std::time::Instant::now();
            let health = match provider.health_check().await {
                Ok(_) => CloudHealth {
                    provider: *provider_type,
                    healthy: true,
                    latency_ms: start.elapsed().as_millis() as u64,
                    last_check: chrono::Utc::now(),
                    error: None,
                },
                Err(e) => CloudHealth {
                    provider: *provider_type,
                    healthy: false,
                    latency_ms: start.elapsed().as_millis() as u64,
                    last_check: chrono::Utc::now(),
                    error: Some(e.to_string()),
                },
            };
            results.push(health);
        }
        
        results
    }
}