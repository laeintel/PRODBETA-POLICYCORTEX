// Azure Resource Graph Integration
// Provides resource inventory and queries

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tracing::info;

use super::client::AzureClient;
use super::api_versions;

/// Resource Graph query request
#[derive(Debug, Serialize)]
pub struct QueryRequest {
    pub subscriptions: Vec<String>,
    pub query: String,
    pub options: Option<QueryOptions>,
}

#[derive(Debug, Serialize)]
pub struct QueryOptions {
    #[serde(rename = "$skip")]
    pub skip: Option<i32>,
    #[serde(rename = "$top")]
    pub top: Option<i32>,
    #[serde(rename = "$skipToken")]
    pub skip_token: Option<String>,
}

/// Resource Graph query response
#[derive(Debug, Deserialize)]
pub struct QueryResponse {
    #[serde(rename = "totalRecords")]
    pub total_records: i64,
    pub count: i64,
    pub data: Vec<HashMap<String, Value>>,
    #[serde(rename = "$skipToken")]
    pub skip_token: Option<String>,
}

/// Azure Resource details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDetails {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub kind: Option<String>,
    pub location: String,
    #[serde(rename = "resourceGroup")]
    pub resource_group: String,
    #[serde(rename = "subscriptionId")]
    pub subscription_id: String,
    pub tags: Option<HashMap<String, String>>,
    pub sku: Option<ResourceSku>,
    pub properties: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSku {
    pub name: Option<String>,
    pub tier: Option<String>,
    pub size: Option<String>,
    pub family: Option<String>,
    pub capacity: Option<i32>,
}

/// Resource statistics
#[derive(Debug, Clone, Serialize)]
pub struct ResourceStatistics {
    pub total_resources: usize,
    pub resources_by_type: HashMap<String, usize>,
    pub resources_by_location: HashMap<String, usize>,
    pub resources_by_group: HashMap<String, usize>,
    pub resources_by_subscription: HashMap<String, usize>,
    pub tagged_resources: usize,
    pub untagged_resources: usize,
}

/// Azure Resource Graph service
pub struct ResourceGraphService {
    client: AzureClient,
}

impl ResourceGraphService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Execute a Resource Graph query
    pub async fn query(&self, query: &str, top: Option<i32>) -> Result<QueryResponse> {
        let request = QueryRequest {
            subscriptions: vec![self.client.config.subscription_id.clone()],
            query: query.to_string(),
            options: Some(QueryOptions {
                skip: None,
                top,
                skip_token: None,
            }),
        };

        let response: QueryResponse = self.client
            .post_management(
                "/providers/Microsoft.ResourceGraph/resources",
                api_versions::RESOURCE_GRAPH,
                &request
            )
            .await?;

        Ok(response)
    }

    /// Get all resources with pagination
    pub async fn get_all_resources(&self) -> Result<Vec<ResourceDetails>> {
        info!("Fetching all resources from Azure Resource Graph");

        let query = r#"
            Resources
            | project id, name, type, kind, location, resourceGroup, subscriptionId, tags, sku, properties
            | order by name asc
        "#;

        let mut all_resources = Vec::new();
        let mut skip_token: Option<String> = None;

        loop {
            let request = QueryRequest {
                subscriptions: vec![self.client.config.subscription_id.clone()],
                query: query.to_string(),
                options: Some(QueryOptions {
                    skip: None,
                    top: Some(1000),
                    skip_token: skip_token.clone(),
                }),
            };

            let response: QueryResponse = self.client
                .post_management(
                    "/providers/Microsoft.ResourceGraph/resources",
                    api_versions::RESOURCE_GRAPH,
                    &request
                )
                .await?;

            for data in response.data {
                let resource = ResourceDetails {
                    id: data.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    name: data.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    resource_type: data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    kind: data.get("kind").and_then(|v| v.as_str()).map(String::from),
                    location: data.get("location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    resource_group: data.get("resourceGroup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    subscription_id: data.get("subscriptionId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    tags: data.get("tags").and_then(|v| serde_json::from_value(v.clone()).ok()),
                    sku: data.get("sku").and_then(|v| serde_json::from_value(v.clone()).ok()),
                    properties: data.get("properties").cloned(),
                };
                all_resources.push(resource);
            }

            skip_token = response.skip_token;
            if skip_token.is_none() {
                break;
            }
        }

        Ok(all_resources)
    }

    /// Get resources by type
    pub async fn get_resources_by_type(&self, resource_type: &str) -> Result<Vec<ResourceDetails>> {
        let query = format!(
            r#"
            Resources
            | where type =~ '{}'
            | project id, name, type, kind, location, resourceGroup, subscriptionId, tags, sku, properties
            | order by name asc
            "#,
            resource_type
        );

        let response = self.query(&query, Some(1000)).await?;
        
        let resources = response.data.into_iter()
            .map(|data| ResourceDetails {
                id: data.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                name: data.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_type: data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                kind: data.get("kind").and_then(|v| v.as_str()).map(String::from),
                location: data.get("location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_group: data.get("resourceGroup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                subscription_id: data.get("subscriptionId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                tags: data.get("tags").and_then(|v| serde_json::from_value(v.clone()).ok()),
                sku: data.get("sku").and_then(|v| serde_json::from_value(v.clone()).ok()),
                properties: data.get("properties").cloned(),
            })
            .collect();

        Ok(resources)
    }

    /// Get resource statistics
    pub async fn get_resource_statistics(&self) -> Result<ResourceStatistics> {
        info!("Calculating resource statistics");

        let resources = self.get_all_resources().await?;

        let mut resources_by_type = HashMap::new();
        let mut resources_by_location = HashMap::new();
        let mut resources_by_group = HashMap::new();
        let mut resources_by_subscription = HashMap::new();
        let mut tagged_resources = 0;
        let mut untagged_resources = 0;

        for resource in &resources {
            // Count by type
            *resources_by_type.entry(resource.resource_type.clone()).or_insert(0) += 1;

            // Count by location
            *resources_by_location.entry(resource.location.clone()).or_insert(0) += 1;

            // Count by resource group
            *resources_by_group.entry(resource.resource_group.clone()).or_insert(0) += 1;

            // Count by subscription
            *resources_by_subscription.entry(resource.subscription_id.clone()).or_insert(0) += 1;

            // Count tagged vs untagged
            if resource.tags.as_ref().map_or(false, |t| !t.is_empty()) {
                tagged_resources += 1;
            } else {
                untagged_resources += 1;
            }
        }

        Ok(ResourceStatistics {
            total_resources: resources.len(),
            resources_by_type,
            resources_by_location,
            resources_by_group,
            resources_by_subscription,
            tagged_resources,
            untagged_resources,
        })
    }

    /// Get virtual machines
    pub async fn get_virtual_machines(&self) -> Result<Vec<ResourceDetails>> {
        self.get_resources_by_type("microsoft.compute/virtualmachines").await
    }

    /// Get storage accounts
    pub async fn get_storage_accounts(&self) -> Result<Vec<ResourceDetails>> {
        self.get_resources_by_type("microsoft.storage/storageaccounts").await
    }

    /// Get network resources
    pub async fn get_network_resources(&self) -> Result<Vec<ResourceDetails>> {
        let query = r#"
            Resources
            | where type startswith 'microsoft.network/'
            | project id, name, type, kind, location, resourceGroup, subscriptionId, tags, sku, properties
            | order by type asc, name asc
        "#;

        let response = self.query(query, Some(1000)).await?;
        
        let resources = response.data.into_iter()
            .map(|data| ResourceDetails {
                id: data.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                name: data.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_type: data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                kind: data.get("kind").and_then(|v| v.as_str()).map(String::from),
                location: data.get("location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_group: data.get("resourceGroup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                subscription_id: data.get("subscriptionId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                tags: data.get("tags").and_then(|v| serde_json::from_value(v.clone()).ok()),
                sku: data.get("sku").and_then(|v| serde_json::from_value(v.clone()).ok()),
                properties: data.get("properties").cloned(),
            })
            .collect();

        Ok(resources)
    }

    /// Search resources with a custom query
    pub async fn search_resources(&self, search_query: &str) -> Result<Vec<ResourceDetails>> {
        let query = format!(
            r#"
            Resources
            | where name contains '{}' or tags contains '{}'
            | project id, name, type, kind, location, resourceGroup, subscriptionId, tags, sku, properties
            | order by name asc
            | limit 100
            "#,
            search_query, search_query
        );

        let response = self.query(&query, Some(100)).await?;
        
        let resources = response.data.into_iter()
            .map(|data| ResourceDetails {
                id: data.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                name: data.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_type: data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                kind: data.get("kind").and_then(|v| v.as_str()).map(String::from),
                location: data.get("location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                resource_group: data.get("resourceGroup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                subscription_id: data.get("subscriptionId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                tags: data.get("tags").and_then(|v| serde_json::from_value(v.clone()).ok()),
                sku: data.get("sku").and_then(|v| serde_json::from_value(v.clone()).ok()),
                properties: data.get("properties").cloned(),
            })
            .collect();

        Ok(resources)
    }
}