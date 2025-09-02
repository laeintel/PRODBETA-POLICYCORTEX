// Azure Integration Module - Live Data Implementation
// This module provides comprehensive Azure service integration for PolicyCortex

pub mod client;
pub mod auth;
pub mod monitor;
pub mod governance;
pub mod security;
pub mod operations;
pub mod devops;
pub mod cost;
pub mod activity;
pub mod resource_graph;

// Re-export service structures
pub use client::{AzureClient, create_shared_client};
pub use monitor::MonitorService;
pub use governance::GovernanceService;
pub use security::SecurityService;
pub use operations::OperationsService;
pub use devops::DevOpsService;
pub use cost::CostService;
pub use activity::ActivityService;
pub use resource_graph::ResourceGraphService;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Azure Configuration
#[derive(Debug, Clone)]
pub struct AzureConfig {
    pub subscription_id: String,
    pub tenant_id: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub resource_graph_endpoint: String,
    pub management_endpoint: String,
    pub graph_endpoint: String,
    pub keyvault_endpoint: Option<String>,
}

impl AzureConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            subscription_id: std::env::var("AZURE_SUBSCRIPTION_ID")
                .unwrap_or_else(|_| "205b477d-17e7-4b3b-92c1-32cf02626b78".to_string()),
            tenant_id: std::env::var("AZURE_TENANT_ID")
                .unwrap_or_else(|_| "9ef5b184-d371-462a-bc75-5024ce8baff7".to_string()),
            client_id: std::env::var("AZURE_CLIENT_ID").ok(),
            client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
            resource_graph_endpoint: "https://management.azure.com".to_string(),
            management_endpoint: "https://management.azure.com".to_string(),
            graph_endpoint: "https://graph.microsoft.com".to_string(),
            keyvault_endpoint: std::env::var("AZURE_KEYVAULT_ENDPOINT").ok(),
        })
    }
}

/// Common Azure API response wrapper
#[derive(Debug, Deserialize, Serialize)]
pub struct AzureResponse<T> {
    pub value: Vec<T>,
    #[serde(rename = "nextLink")]
    pub next_link: Option<String>,
    #[serde(rename = "@odata.context")]
    pub odata_context: Option<String>,
}

/// Azure error response
#[derive(Debug, Deserialize, Serialize)]
pub struct AzureError {
    pub error: AzureErrorDetails,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AzureErrorDetails {
    pub code: String,
    pub message: String,
    pub details: Option<Vec<AzureErrorDetail>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AzureErrorDetail {
    pub code: String,
    pub message: String,
}

/// Common Azure resource properties
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub location: Option<String>,
    pub tags: Option<std::collections::HashMap<String, String>>,
    pub properties: Option<serde_json::Value>,
}

/// Azure API version constants
pub mod api_versions {
    pub const RESOURCE_GRAPH: &str = "2021-03-01";
    pub const MONITOR: &str = "2023-03-01-preview";
    pub const POLICY: &str = "2021-06-01";
    pub const SECURITY_CENTER: &str = "2022-01-01";
    pub const COST_MANAGEMENT: &str = "2023-11-01";
    pub const ACTIVITY_LOG: &str = "2015-04-01";
    pub const AUTOMATION: &str = "2019-06-01";
    pub const KEYVAULT: &str = "2023-07-01";
    pub const GRAPH: &str = "v1.0";
    pub const RBAC: &str = "2022-04-01";
    pub const ADVISOR: &str = "2020-01-01";
}

/// Helper function to build Azure management API URL
pub fn build_management_url(path: &str, api_version: &str) -> String {
    format!("https://management.azure.com{}?api-version={}", path, api_version)
}

/// Helper function to build Azure Graph API URL
pub fn build_graph_url(path: &str) -> String {
    format!("https://graph.microsoft.com/v1.0{}", path)
}