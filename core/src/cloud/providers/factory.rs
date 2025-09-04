use crate::cloud::{CloudConfig, CloudError, CloudProvider, CloudProviderTrait, CloudResult};
use std::sync::Arc;

use super::{aws::AWSProvider, azure::AzureProvider, gcp::GCPProvider};

/// Factory for creating cloud provider instances
pub struct CloudProviderFactory;

impl CloudProviderFactory {
    /// Create a cloud provider instance from configuration
    pub async fn create(config: CloudConfig) -> CloudResult<Arc<dyn CloudProviderTrait>> {
        match config.provider {
            CloudProvider::Azure => {
                let provider = AzureProvider::new(config).await?;
                Ok(Arc::new(provider))
            }
            CloudProvider::AWS => {
                let provider = AWSProvider::new(config).await?;
                Ok(Arc::new(provider))
            }
            CloudProvider::GCP => {
                let provider = GCPProvider::new(config).await?;
                Ok(Arc::new(provider))
            }
            _ => Err(CloudError::ProviderNotSupported(
                config.provider.name().to_string(),
            )),
        }
    }
    
    /// Create multiple providers from configurations
    pub async fn create_multiple(
        configs: Vec<CloudConfig>,
    ) -> CloudResult<Vec<Arc<dyn CloudProviderTrait>>> {
        let mut providers = Vec::new();
        
        for config in configs {
            let provider = Self::create(config).await?;
            providers.push(provider);
        }
        
        Ok(providers)
    }
    
    /// Create a provider with default configuration for testing
    pub async fn create_default(provider: CloudProvider) -> CloudResult<Arc<dyn CloudProviderTrait>> {
        let config = match provider {
            CloudProvider::Azure => {
                let mut config = CloudConfig::new(CloudProvider::Azure);
                config.region = Some("eastus".to_string());
                config
            }
            CloudProvider::AWS => {
                let mut config = CloudConfig::new(CloudProvider::AWS);
                config.region = Some("us-east-1".to_string());
                config
            }
            CloudProvider::GCP => {
                let mut config = CloudConfig::new(CloudProvider::GCP);
                config.region = Some("us-central1".to_string());
                config.credentials.insert("project_id".to_string(), "default-project".to_string());
                config.credentials.insert("use_default_credentials".to_string(), "true".to_string());
                config
            }
            _ => return Err(CloudError::ProviderNotSupported(provider.name().to_string())),
        };
        
        Self::create(config).await
    }
}