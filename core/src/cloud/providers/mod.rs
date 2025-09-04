pub mod aws;
pub mod azure;
pub mod factory;
pub mod gcp;

pub use factory::CloudProviderFactory;

use crate::cloud::{CloudConfig, CloudProvider, CloudProviderTrait, CloudResult};
use std::sync::Arc;

/// Initialize a cloud provider from configuration
pub async fn initialize_provider(config: CloudConfig) -> CloudResult<Arc<dyn CloudProviderTrait>> {
    CloudProviderFactory::create(config).await
}