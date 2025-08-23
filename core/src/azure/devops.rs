// Azure DevOps Integration
// Provides CI/CD pipeline and deployment data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

use super::client::AzureClient;
use super::api_versions;

/// Container Registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRegistry {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub registry_type: String,
    pub location: String,
    pub sku: ContainerRegistrySku,
    pub properties: ContainerRegistryProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRegistrySku {
    pub name: String,
    pub tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRegistryProperties {
    #[serde(rename = "loginServer")]
    pub login_server: String,
    #[serde(rename = "creationDate")]
    pub creation_date: Option<DateTime<Utc>>,
    #[serde(rename = "provisioningState")]
    pub provisioning_state: String,
    #[serde(rename = "adminUserEnabled")]
    pub admin_user_enabled: Option<bool>,
    #[serde(rename = "storageAccount")]
    pub storage_account: Option<StorageAccountInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAccountInfo {
    pub id: String,
}

/// Container Image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerImage {
    pub registry: String,
    pub image_name: String,
    pub tags: Vec<String>,
    pub manifest: Option<ImageManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageManifest {
    pub digest: String,
    #[serde(rename = "imageSize")]
    pub image_size: i64,
    #[serde(rename = "createdTime")]
    pub created_time: DateTime<Utc>,
    #[serde(rename = "lastUpdateTime")]
    pub last_update_time: DateTime<Utc>,
}

/// Deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deployment {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub deployment_type: String,
    pub properties: DeploymentProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentProperties {
    #[serde(rename = "provisioningState")]
    pub provisioning_state: String,
    pub timestamp: DateTime<Utc>,
    pub duration: Option<String>,
    #[serde(rename = "correlationId")]
    pub correlation_id: String,
    pub providers: Option<Vec<Provider>>,
    pub dependencies: Option<Vec<Dependency>>,
    #[serde(rename = "outputResources")]
    pub output_resources: Option<Vec<OutputResource>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub namespace: String,
    #[serde(rename = "resourceTypes")]
    pub resource_types: Vec<ResourceTypeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTypeInfo {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    pub locations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    #[serde(rename = "dependsOn")]
    pub depends_on: Vec<DependencyInfo>,
    pub id: String,
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(rename = "resourceName")]
    pub resource_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub id: String,
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(rename = "resourceName")]
    pub resource_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputResource {
    pub id: String,
}

/// Web App (for deployments)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebApp {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub app_type: String,
    pub kind: String,
    pub location: String,
    pub properties: WebAppProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAppProperties {
    pub state: String,
    #[serde(rename = "hostNames")]
    pub host_names: Vec<String>,
    #[serde(rename = "repositorySiteName")]
    pub repository_site_name: Option<String>,
    #[serde(rename = "usageState")]
    pub usage_state: String,
    pub enabled: bool,
    #[serde(rename = "enabledHostNames")]
    pub enabled_host_names: Vec<String>,
    #[serde(rename = "availabilityState")]
    pub availability_state: String,
    #[serde(rename = "hostNameSslStates")]
    pub host_name_ssl_states: Option<Vec<HostNameSslState>>,
    #[serde(rename = "serverFarmId")]
    pub server_farm_id: String,
    #[serde(rename = "lastModifiedTimeUtc")]
    pub last_modified_time_utc: Option<DateTime<Utc>>,
    #[serde(rename = "siteConfig")]
    pub site_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostNameSslState {
    pub name: String,
    #[serde(rename = "sslState")]
    pub ssl_state: String,
    #[serde(rename = "hostType")]
    pub host_type: String,
}

/// Azure DevOps service
pub struct DevOpsService {
    client: AzureClient,
}

impl DevOpsService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get all container registries
    pub async fn get_container_registries(&self) -> Result<Vec<ContainerRegistry>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.ContainerRegistry/registries",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, "2023-01-01-preview").await
    }

    /// Get recent deployments
    pub async fn get_deployments(&self) -> Result<Vec<Deployment>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Resources/deployments",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, "2021-04-01").await
    }

    /// Get deployment by resource group
    pub async fn get_deployments_by_resource_group(&self, resource_group: &str) -> Result<Vec<Deployment>> {
        let path = format!(
            "/subscriptions/{}/resourcegroups/{}/providers/Microsoft.Resources/deployments",
            self.client.config.subscription_id, resource_group
        );

        self.client.get_all_pages(&path, "2021-04-01").await
    }

    /// Get web apps
    pub async fn get_web_apps(&self) -> Result<Vec<WebApp>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Web/sites",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, "2022-03-01").await
    }

    /// Get DevOps summary
    pub async fn get_devops_summary(&self) -> Result<DevOpsSummary> {
        info!("Fetching DevOps summary from Azure");

        let registries = self.get_container_registries().await.unwrap_or_default();
        let deployments = self.get_deployments().await.unwrap_or_default();
        let web_apps = self.get_web_apps().await.unwrap_or_default();

        // Count deployment states
        let successful_deployments = deployments.iter()
            .filter(|d| d.properties.provisioning_state == "Succeeded")
            .count();
        let failed_deployments = deployments.iter()
            .filter(|d| d.properties.provisioning_state == "Failed")
            .count();
        let running_deployments = deployments.iter()
            .filter(|d| d.properties.provisioning_state == "Running")
            .count();

        // Count app states
        let running_apps = web_apps.iter()
            .filter(|a| a.properties.state == "Running")
            .count();
        let stopped_apps = web_apps.iter()
            .filter(|a| a.properties.state == "Stopped")
            .count();

        Ok(DevOpsSummary {
            total_container_registries: registries.len(),
            total_deployments: deployments.len(),
            successful_deployments,
            failed_deployments,
            running_deployments,
            total_web_apps: web_apps.len(),
            running_apps,
            stopped_apps,
        })
    }

    /// Get pipeline-like data (from deployments)
    pub async fn get_pipelines(&self) -> Result<Vec<Pipeline>> {
        let deployments = self.get_deployments().await?;

        let pipelines = deployments
            .into_iter()
            .take(20) // Limit to recent 20
            .map(|d| Pipeline {
                id: d.id.clone(),
                name: d.name.clone(),
                status: match d.properties.provisioning_state.as_str() {
                    "Succeeded" => "completed".to_string(),
                    "Failed" => "failed".to_string(),
                    "Running" => "running".to_string(),
                    _ => "pending".to_string(),
                },
                last_run: Some(d.properties.timestamp),
                source: "ARM Deployment".to_string(),
                duration: d.properties.duration,
            })
            .collect();

        Ok(pipelines)
    }

    /// Get releases (from deployments)
    pub async fn get_releases(&self) -> Result<Vec<Release>> {
        let deployments = self.get_deployments().await?;

        let releases = deployments
            .into_iter()
            .filter(|d| d.properties.provisioning_state == "Succeeded")
            .take(20)
            .map(|d| Release {
                id: d.id.clone(),
                name: d.name.clone(),
                version: format!("v{}", d.properties.timestamp.format("%Y%m%d.%H%M")),
                status: "deployed".to_string(),
                created_at: d.properties.timestamp,
                environment: "Production".to_string(), // Would need additional API calls
            })
            .collect();

        Ok(releases)
    }

    /// Get artifacts (from container registries)
    pub async fn get_artifacts(&self) -> Result<Vec<Artifact>> {
        let registries = self.get_container_registries().await?;

        let mut artifacts = Vec::new();
        for registry in registries {
            artifacts.push(Artifact {
                id: registry.id.clone(),
                name: registry.name.clone(),
                artifact_type: "Container Registry".to_string(),
                location: registry.location.clone(),
                created_at: registry.properties.creation_date,
                size: None,
            });
        }

        Ok(artifacts)
    }
}

#[derive(Debug, Serialize)]
pub struct DevOpsSummary {
    pub total_container_registries: usize,
    pub total_deployments: usize,
    pub successful_deployments: usize,
    pub failed_deployments: usize,
    pub running_deployments: usize,
    pub total_web_apps: usize,
    pub running_apps: usize,
    pub stopped_apps: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Pipeline {
    pub id: String,
    pub name: String,
    pub status: String,
    pub last_run: Option<DateTime<Utc>>,
    pub source: String,
    pub duration: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Release {
    pub id: String,
    pub name: String,
    pub version: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub environment: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    pub id: String,
    pub name: String,
    pub artifact_type: String,
    pub location: String,
    pub created_at: Option<DateTime<Utc>>,
    pub size: Option<i64>,
}