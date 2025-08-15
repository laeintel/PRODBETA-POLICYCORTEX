// Azure Resource Graph Integration
// Provides unified resource discovery and querying capabilities with enhanced 4,000 req/min quota

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use azure_core::{Result as AzureResult, Error as AzureError};
use crate::azure_client::AzureClient;
use crate::governance::{GovernanceError, GovernanceResult, ComponentHealth, HealthStatus};

/// Azure Resource Graph client with intelligent caching and quota management
pub struct ResourceGraphClient {
    /// Azure client for API calls
    azure_client: Arc<AzureClient>,

    /// Intelligent cache for resource data with TTL
    cache: Arc<DashMap<String, CachedResourceData>>,

    /// Query statistics for optimization
    query_stats: Arc<RwLock<QueryStatistics>>,

    /// Configuration for the client
    config: ResourceGraphConfig,
}

/// Configuration for Resource Graph client
#[derive(Debug, Clone)]
pub struct ResourceGraphConfig {
    /// Cache TTL for resource data (default: 5 minutes)
    pub cache_ttl: Duration,

    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,

    /// Query timeout
    pub query_timeout: Duration,

    /// Enable enhanced quota usage (useResourceGraph=true)
    pub use_enhanced_quota: bool,
}

impl Default for ResourceGraphConfig {
    fn default() -> Self {
        Self {
            cache_ttl: Duration::minutes(5),
            max_concurrent_queries: 10,
            query_timeout: Duration::seconds(30),
            use_enhanced_quota: true,
        }
    }
}

/// Cached resource data with timestamp
#[derive(Debug, Clone)]
pub struct CachedResourceData {
    /// The cached resource data
    pub data: ResourceQueryResult,

    /// When the data was cached
    pub cached_at: DateTime<Utc>,

    /// TTL for this cache entry
    pub ttl: Duration,
}

impl CachedResourceData {
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.cached_at + self.ttl
    }
}

/// Query statistics for optimization and monitoring
#[derive(Debug, Default)]
pub struct QueryStatistics {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_query_time_ms: f64,
    pub quota_remaining: Option<u32>,
    pub quota_reset_time: Option<DateTime<Utc>>,
}

/// Resource query result from Azure Resource Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQueryResult {
    /// Total count of resources matching the query
    pub total_records: u64,

    /// The actual resource data
    pub data: Vec<AzureResource>,

    /// Facets for aggregated data
    pub facets: Option<Vec<QueryFacet>>,

    /// Skip token for pagination
    pub skip_token: Option<String>,

    /// Result metadata
    pub result_truncated: bool,
}

/// Azure resource representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureResource {
    /// Resource ID
    pub id: String,

    /// Resource name
    pub name: String,

    /// Resource type
    pub resource_type: String,

    /// Resource group
    pub resource_group: String,

    /// Subscription ID
    pub subscription_id: String,

    /// Location
    pub location: String,

    /// Tags
    pub tags: HashMap<String, String>,

    /// Resource properties (dynamic JSON)
    pub properties: serde_json::Value,

    /// Compliance state
    pub compliance_state: Option<ComplianceState>,

    /// Last modified timestamp
    pub last_modified: Option<DateTime<Utc>>,
}

/// Resource compliance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceState {
    /// Overall compliance status
    pub status: ComplianceStatus,

    /// Policy evaluations
    pub policy_evaluations: Vec<PolicyEvaluation>,

    /// Last evaluation timestamp
    pub last_evaluation: DateTime<Utc>,
}

/// Compliance status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    Unknown,
    NotApplicable,
    Exempt,
}

/// Policy evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    /// Policy assignment ID
    pub policy_assignment_id: String,

    /// Policy definition ID
    pub policy_definition_id: String,

    /// Evaluation result
    pub result: ComplianceStatus,

    /// Evaluation reason
    pub reason: String,

    /// Evaluation timestamp
    pub evaluated_at: DateTime<Utc>,
}

/// Query facet for aggregated results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFacet {
    /// Facet expression
    pub expression: String,

    /// Result type
    pub result_type: String,

    /// Facet results
    pub results: Vec<FacetResult>,
}

/// Individual facet result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetResult {
    /// Facet value
    pub value: serde_json::Value,

    /// Count for this facet value
    pub count: u64,
}

/// Resource filter for discovery operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFilter {
    /// Resource types to include
    pub resource_types: Option<Vec<String>>,

    /// Subscription IDs to filter by
    pub subscription_ids: Option<Vec<String>>,

    /// Resource groups to filter by
    pub resource_groups: Option<Vec<String>>,

    /// Locations to filter by
    pub locations: Option<Vec<String>>,

    /// Tag filters
    pub tags: Option<HashMap<String, String>>,

    /// Compliance state filter
    pub compliance_state: Option<ComplianceStatus>,

    /// Custom KQL where clause
    pub custom_filter: Option<String>,
}

impl ResourceGraphClient {
    /// Create a new Resource Graph client
    pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
        let config = ResourceGraphConfig::default();

        Ok(Self {
            azure_client,
            cache: Arc::new(DashMap::new()),
            query_stats: Arc::new(RwLock::new(QueryStatistics::default())),
            config,
        })
    }

    /// Create client with custom configuration
    pub async fn with_config(azure_client: Arc<AzureClient>, config: ResourceGraphConfig) -> GovernanceResult<Self> {
        Ok(Self {
            azure_client,
            cache: Arc::new(DashMap::new()),
            query_stats: Arc::new(RwLock::new(QueryStatistics::default())),
            config,
        })
    }

    /// Execute a KQL query against Azure Resource Graph
    pub async fn query_resources(&self, query: &str) -> GovernanceResult<ResourceQueryResult> {
        let start_time = std::time::Instant::now();

        // Check cache first
        let cache_key = format!("query:{}", query);
        if let Some(cached) = self.cache.get(&cache_key) {
            if !cached.is_expired() {
                self.update_stats(true, start_time.elapsed().as_millis() as f64).await;
                return Ok(cached.data.clone());
            }
        }

        // Execute query against Azure Resource Graph
        let result = self.execute_query(query).await?;

        // Cache the result
        self.cache.insert(cache_key, CachedResourceData {
            data: result.clone(),
            cached_at: Utc::now(),
            ttl: self.config.cache_ttl,
        });

        self.update_stats(false, start_time.elapsed().as_millis() as f64).await;
        Ok(result)
    }

    /// Get compliance state for a specific scope
    pub async fn get_compliance_state(&self, scope: &str) -> GovernanceResult<ComplianceState> {
        let query = format!(
            r#"
            PolicyResources
            | where type == "microsoft.authorization/policyassignments"
            | where properties.scope startswith "{}"
            | join kind=inner (
                PolicyResources
                | where type == "microsoft.authorization/policyevaluations"
            ) on $left.id == $right.properties.policyAssignmentId
            | project
                policyAssignmentId = id,
                policyDefinitionId = properties.policyDefinitionId,
                complianceState = properties.complianceState,
                evaluationResult = properties.evaluationResult,
                timestamp = properties.timestamp
            "#,
            scope
        );

        let result = self.query_resources(&query).await?;

        // Process policy evaluations from the result
        let mut policy_evaluations = Vec::new();
        for resource in result.data {
            if let Ok(evaluation) = serde_json::from_value::<PolicyEvaluation>(resource.properties) {
                policy_evaluations.push(evaluation);
            }
        }

        // Determine overall compliance status
        let status = if policy_evaluations.is_empty() {
            ComplianceStatus::Unknown
        } else if policy_evaluations.iter().all(|e| e.result == ComplianceStatus::Compliant) {
            ComplianceStatus::Compliant
        } else if policy_evaluations.iter().any(|e| e.result == ComplianceStatus::NonCompliant) {
            ComplianceStatus::NonCompliant
        } else {
            ComplianceStatus::Unknown
        };

        Ok(ComplianceState {
            status,
            policy_evaluations,
            last_evaluation: Utc::now(),
        })
    }

    /// Discover resources by type with filtering
    pub async fn discover_resources_by_type(&self, resource_type: &str) -> GovernanceResult<Vec<AzureResource>> {
        let query = format!(
            r#"
            Resources
            | where type == "{}"
            | project
                id,
                name,
                type,
                resourceGroup,
                subscriptionId,
                location,
                tags,
                properties,
                kind
            | order by name asc
            "#,
            resource_type
        );

        let result = self.query_resources(&query).await?;
        Ok(result.data)
    }

    /// Discover resources with advanced filtering
    pub async fn discover_resources(&self, filter: &ResourceFilter) -> GovernanceResult<Vec<AzureResource>> {
        let mut query_parts = vec!["Resources".to_string()];
        let mut where_conditions = Vec::new();

        // Build WHERE conditions based on filter
        if let Some(ref types) = filter.resource_types {
            let types_str = types.iter()
                .map(|t| format!("\"{}\"", t))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("type in ({})", types_str));
        }

        if let Some(ref subscriptions) = filter.subscription_ids {
            let subs_str = subscriptions.iter()
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("subscriptionId in ({})", subs_str));
        }

        if let Some(ref groups) = filter.resource_groups {
            let groups_str = groups.iter()
                .map(|g| format!("\"{}\"", g))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("resourceGroup in ({})", groups_str));
        }

        if let Some(ref locations) = filter.locations {
            let locations_str = locations.iter()
                .map(|l| format!("\"{}\"", l))
                .collect::<Vec<_>>()
                .join(", ");
            where_conditions.push(format!("location in ({})", locations_str));
        }

        if let Some(ref tags) = filter.tags {
            for (key, value) in tags {
                where_conditions.push(format!("tags[\"{}\"] == \"{}\"", key, value));
            }
        }

        if let Some(ref custom) = filter.custom_filter {
            where_conditions.push(custom.clone());
        }

        // Build the complete query
        if !where_conditions.is_empty() {
            query_parts.push(format!("| where {}", where_conditions.join(" and ")));
        }

        query_parts.push(r#"
            | project
                id,
                name,
                type,
                resourceGroup,
                subscriptionId,
                location,
                tags,
                properties,
                kind
            | order by name asc
        "#.to_string());

        let query = query_parts.join("\n");
        let result = self.query_resources(&query).await?;
        Ok(result.data)
    }

    /// Get resource inventory with compliance information
    pub async fn get_resource_inventory(&self) -> GovernanceResult<ResourceInventory> {
        let query = r#"
            Resources
            | summarize
                TotalResources = count(),
                ResourcesByType = count() by type,
                ResourcesByLocation = count() by location,
                ResourcesBySubscription = count() by subscriptionId,
                ResourcesByResourceGroup = count() by resourceGroup
            | order by TotalResources desc
        "#;

        let result = self.query_resources(query).await?;

        // Process the aggregated data into inventory format
        let mut inventory = ResourceInventory {
            total_resources: 0,
            resources_by_type: HashMap::new(),
            resources_by_location: HashMap::new(),
            resources_by_subscription: HashMap::new(),
            resources_by_resource_group: HashMap::new(),
            last_updated: Utc::now(),
        };

        // Extract aggregated counts from the result
        for resource in result.data {
            if let Some(total) = resource.properties.get("TotalResources") {
                if let Some(count) = total.as_u64() {
                    inventory.total_resources = count;
                }
            }

            // Process other aggregations...
            // (Implementation would parse the JSON results and populate the HashMap fields)
        }

        Ok(inventory)
    }

    /// Execute raw KQL query (private helper)
    async fn execute_query(&self, query: &str) -> GovernanceResult<ResourceQueryResult> {
        // Build the request URL with enhanced quota parameter
        let mut url = "https://management.azure.com/providers/Microsoft.ResourceGraph/resources".to_string();
        if self.config.use_enhanced_quota {
            url.push_str("?useResourceGraph=true");
        }

        // Build request body
        let request_body = serde_json::json!({
            "subscriptions": [], // Will be populated from Azure client context
            "query": query,
            "options": {
                "resultFormat": "objectArray",
                "resultTruncated": false
            }
        });

        // Execute the HTTP request
        // This is a simplified version - actual implementation would use azure_core HTTP client
        let response = self.azure_client
            .http_client()
            .post(url, Some(request_body.to_string()))
            .await
            .map_err(|e| GovernanceError::AzureApi(e))?;

        // Parse response
        let result: ResourceQueryResult = serde_json::from_str(&response.body)
            .map_err(GovernanceError::Serialization)?;

        // Update quota statistics from response headers
        if let Some(remaining) = response.headers.get("x-ms-user-quota-remaining") {
            if let Ok(quota) = remaining.parse::<u32>() {
                let mut stats = self.query_stats.write().await;
                stats.quota_remaining = Some(quota);
            }
        }

        Ok(result)
    }

    /// Update query statistics
    async fn update_stats(&self, cache_hit: bool, query_time_ms: f64) {
        let mut stats = self.query_stats.write().await;
        stats.total_queries += 1;

        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }

        // Update average query time
        let total_time = stats.average_query_time_ms * (stats.total_queries - 1) as f64;
        stats.average_query_time_ms = (total_time + query_time_ms) / stats.total_queries as f64;
    }

    /// Get query statistics
    pub async fn get_statistics(&self) -> QueryStatistics {
        self.query_stats.read().await.clone()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Health check for the Resource Graph client
    pub async fn health_check(&self) -> ComponentHealth {
        let start_time = std::time::Instant::now();

        // Simple query to test connectivity
        let test_query = "Resources | take 1";

        match self.execute_query(test_query).await {
            Ok(_) => {
                let query_time = start_time.elapsed().as_millis() as f64;
                let stats = self.get_statistics().await;

                let mut metrics = HashMap::new();
                metrics.insert("query_time_ms".to_string(), query_time);
                metrics.insert("cache_hit_ratio".to_string(),
                    if stats.total_queries > 0 {
                        stats.cache_hits as f64 / stats.total_queries as f64
                    } else { 0.0 });
                metrics.insert("total_queries".to_string(), stats.total_queries as f64);

                ComponentHealth {
                    component: "ResourceGraph".to_string(),
                    status: if query_time < 5000.0 { HealthStatus::Healthy } else { HealthStatus::Degraded },
                    message: format!("Query executed in {:.2}ms", query_time),
                    last_check: Utc::now(),
                    metrics,
                }
            }
            Err(e) => ComponentHealth {
                component: "ResourceGraph".to_string(),
                status: HealthStatus::Unhealthy,
                message: format!("Health check failed: {}", e),
                last_check: Utc::now(),
                metrics: HashMap::new(),
            }
        }
    }
}

/// Resource inventory summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInventory {
    pub total_resources: u64,
    pub resources_by_type: HashMap<String, u64>,
    pub resources_by_location: HashMap<String, u64>,
    pub resources_by_subscription: HashMap<String, u64>,
    pub resources_by_resource_group: HashMap<String, u64>,
    pub last_updated: DateTime<Utc>,
}

// Mock HTTP response for compilation
struct HttpResponse {
    body: String,
    headers: HashMap<String, String>,
}