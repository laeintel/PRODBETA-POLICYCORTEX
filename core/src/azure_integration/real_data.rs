// Real Azure Data Integration - NO MOCK DATA
// This module ensures all API calls return real Azure data

use serde::{Deserialize, Serialize};
use azure_core::auth::TokenCredential;
use azure_identity::DefaultAzureCredential;
use reqwest::header::{self, HeaderMap, HeaderValue};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

pub struct RealAzureDataClient {
    credential: Arc<DefaultAzureCredential>,
    http_client: reqwest::Client,
    subscription_id: String,
    cache: Arc<RwLock<DataCache>>,
}

impl RealAzureDataClient {
    pub async fn new() -> Result<Self, AzureError> {
        let credential = Arc::new(DefaultAzureCredential::default());
        
        // Get subscription ID from environment or Azure CLI
        let subscription_id = std::env::var("AZURE_SUBSCRIPTION_ID")
            .or_else(|_| Self::get_subscription_from_cli())
            .map_err(|e| AzureError::Configuration(format!("No subscription found: {}", e)))?;

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AzureError::HttpClient(e.to_string()))?;

        Ok(Self {
            credential,
            http_client,
            subscription_id,
            cache: Arc::new(RwLock::new(DataCache::new())),
        })
    }

    fn get_subscription_from_cli() -> Result<String, std::env::VarError> {
        // Execute az account show to get current subscription
        let output = std::process::Command::new("az")
            .args(&["account", "show", "--query", "id", "-o", "tsv"])
            .output()
            .map_err(|_| std::env::VarError::NotPresent)?;
        
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    async fn get_access_token(&self) -> Result<String, AzureError> {
        let token = self.credential
            .get_token(&["https://management.azure.com/.default"])
            .await
            .map_err(|e| AzureError::Authentication(e.to_string()))?;
        
        Ok(token.token.secret().to_string())
    }

    pub async fn make_azure_request<T>(&self, url: &str) -> Result<T, AzureError> 
    where
        T: for<'de> Deserialize<'de>,
    {
        let token = self.get_access_token().await?;
        
        let response = self.http_client
            .get(url)
            .header(header::AUTHORIZATION, format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| AzureError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AzureError::Api(format!("Azure API error: {}", error_text)));
        }

        response.json::<T>()
            .await
            .map_err(|e| AzureError::Deserialization(e.to_string()))
    }

    // Real resource collection from Azure
    pub async fn get_all_resources(&self) -> Result<Vec<AzureResource>, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resources?api-version=2021-04-01",
            self.subscription_id
        );

        #[derive(Deserialize)]
        struct ResourceList {
            value: Vec<AzureResource>,
        }

        let result = self.make_azure_request::<ResourceList>(&url).await?;
        Ok(result.value)
    }

    // Real VM metrics from Azure Monitor
    pub async fn get_vm_metrics(&self, vm_id: &str) -> Result<VmMetrics, AzureError> {
        let url = format!(
            "https://management.azure.com{}/providers/Microsoft.Insights/metrics?\
            api-version=2021-05-01&metricnames=Percentage CPU,Available Memory Bytes,\
            Network In Total,Network Out Total&timespan=P7D&interval=PT1H&aggregation=average",
            vm_id
        );

        let response = self.make_azure_request::<MetricsResponse>(&url).await?;
        
        // Process real metrics
        let mut metrics = VmMetrics::default();
        for metric in response.value {
            match metric.name.value.as_str() {
                "Percentage CPU" => {
                    metrics.cpu_avg = Self::calculate_average(&metric.timeseries);
                    metrics.cpu_p95 = Self::calculate_percentile(&metric.timeseries, 95);
                }
                "Available Memory Bytes" => {
                    let available = Self::calculate_average(&metric.timeseries);
                    metrics.memory_avg = 100.0 - (available / 1024.0 / 1024.0 / 1024.0); // Convert to GB and percentage
                }
                "Network In Total" | "Network Out Total" => {
                    metrics.network_io_avg += Self::calculate_average(&metric.timeseries);
                }
                _ => {}
            }
        }
        
        Ok(metrics)
    }

    // Real cost data from Azure Cost Management
    pub async fn get_cost_data(&self) -> Result<CostData, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.CostManagement/query?\
            api-version=2021-10-01",
            self.subscription_id
        );

        let body = serde_json::json!({
            "type": "Usage",
            "timeframe": "MonthToDate",
            "dataset": {
                "granularity": "Daily",
                "aggregation": {
                    "totalCost": {
                        "name": "Cost",
                        "function": "Sum"
                    }
                },
                "grouping": [
                    {
                        "type": "Dimension",
                        "name": "ServiceName"
                    },
                    {
                        "type": "Dimension",
                        "name": "ResourceId"
                    }
                ]
            }
        });

        let token = self.get_access_token().await?;
        let response = self.http_client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", token))
            .json(&body)
            .send()
            .await
            .map_err(|e| AzureError::Request(e.to_string()))?;

        let cost_response: CostQueryResponse = response.json().await
            .map_err(|e| AzureError::Deserialization(e.to_string()))?;

        // Process real cost data
        let mut total_cost = 0.0;
        let mut cost_by_service = std::collections::HashMap::new();
        let mut idle_resource_costs = Vec::new();

        for row in cost_response.properties.rows {
            let cost = row[0].as_f64().unwrap_or(0.0);
            let service = row[1].as_str().unwrap_or("Unknown");
            let resource_id = row[2].as_str().unwrap_or("");
            
            total_cost += cost;
            *cost_by_service.entry(service.to_string()).or_insert(0.0) += cost;
            
            // Check if resource is idle (would check metrics)
            if let Ok(metrics) = self.get_vm_metrics(resource_id).await {
                if metrics.cpu_avg < 5.0 && metrics.network_io_avg < 1.0 {
                    idle_resource_costs.push(IdleResourceCost {
                        resource_id: resource_id.to_string(),
                        monthly_cost: cost * 30.0, // Extrapolate to monthly
                        cpu_avg: metrics.cpu_avg,
                    });
                }
            }
        }

        Ok(CostData {
            current_spend: total_cost,
            predicted_spend: total_cost * 1.05, // Would use ML model
            savings_identified: idle_resource_costs.iter().map(|r| r.monthly_cost).sum(),
            idle_resources: idle_resource_costs,
            cost_by_service,
        })
    }

    // Real policy compliance data from Azure Policy
    pub async fn get_policy_compliance(&self) -> Result<PolicyCompliance, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.PolicyInsights/\
            policyStates/latest/summarize?api-version=2019-10-01",
            self.subscription_id
        );

        let token = self.get_access_token().await?;
        let response = self.http_client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| AzureError::Request(e.to_string()))?;

        let summary: PolicySummaryResponse = response.json().await
            .map_err(|e| AzureError::Deserialization(e.to_string()))?;

        // Calculate real compliance rate
        let total = summary.value[0].results.resource_count;
        let compliant = summary.value[0].results.compliant_resource_count;
        let non_compliant = summary.value[0].results.non_compliant_resource_count;
        
        Ok(PolicyCompliance {
            total_policies: summary.value[0].policy_assignments.count,
            compliant_resources: compliant,
            non_compliant_resources: non_compliant,
            compliance_rate: (compliant as f64 / total as f64) * 100.0,
            violations: self.get_policy_violations().await?,
        })
    }

    // Real policy violations
    async fn get_policy_violations(&self) -> Result<Vec<PolicyViolation>, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.PolicyInsights/\
            policyStates/latest/queryResults?api-version=2019-10-01&$filter=complianceState eq 'NonCompliant'",
            self.subscription_id
        );

        let response = self.make_azure_request::<PolicyStatesResponse>(&url).await?;
        
        Ok(response.value.into_iter().map(|state| PolicyViolation {
            policy_name: state.policy_definition_name,
            resource_id: state.resource_id,
            resource_type: state.resource_type,
            compliance_state: state.compliance_state,
            timestamp: state.timestamp,
        }).collect())
    }

    // Real security alerts from Azure Security Center
    pub async fn get_security_alerts(&self) -> Result<Vec<SecurityAlert>, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.Security/alerts?\
            api-version=2021-01-01",
            self.subscription_id
        );

        let response = self.make_azure_request::<AlertsResponse>(&url).await?;
        
        Ok(response.value.into_iter().map(|alert| SecurityAlert {
            id: alert.id,
            name: alert.name,
            severity: alert.properties.severity,
            status: alert.properties.status,
            description: alert.properties.description,
            remediation_steps: alert.properties.remediation_steps,
            affected_resources: alert.properties.affected_resources,
        }).collect())
    }

    // Real RBAC data from Azure AD
    pub async fn get_rbac_data(&self) -> Result<RbacData, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.Authorization/\
            roleAssignments?api-version=2020-04-01-preview",
            self.subscription_id
        );

        let response = self.make_azure_request::<RoleAssignmentsResponse>(&url).await?;
        
        // Analyze for overprivileged accounts
        let mut overprivileged_count = 0;
        let mut risk_score = 0.0;
        
        for assignment in &response.value {
            // Check if role is Owner or Contributor at subscription level
            if assignment.properties.scope == format!("/subscriptions/{}", self.subscription_id) {
                if assignment.properties.role_definition_id.contains("Owner") ||
                   assignment.properties.role_definition_id.contains("Contributor") {
                    overprivileged_count += 1;
                    risk_score += 10.0;
                }
            }
        }
        
        Ok(RbacData {
            total_assignments: response.value.len(),
            overprivileged_count,
            risk_score: risk_score.min(100.0),
            role_assignments: response.value,
        })
    }

    // Real network topology from Azure Network Watcher
    pub async fn get_network_topology(&self) -> Result<NetworkTopology, AzureError> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/providers/Microsoft.Network/\
            networkSecurityGroups?api-version=2021-05-01",
            self.subscription_id
        );

        let response = self.make_azure_request::<NsgResponse>(&url).await?;
        
        let mut public_endpoints = 0;
        let mut critical_exposures = Vec::new();
        
        for nsg in response.value {
            for rule in nsg.properties.security_rules {
                // Check for public exposure
                if rule.properties.source_address_prefix == "*" || 
                   rule.properties.source_address_prefix == "Internet" {
                    if rule.properties.destination_port_range == "3389" || // RDP
                       rule.properties.destination_port_range == "22" ||   // SSH
                       rule.properties.destination_port_range == "445" {   // SMB
                        critical_exposures.push(CriticalExposure {
                            resource: nsg.name.clone(),
                            port: rule.properties.destination_port_range.clone(),
                            protocol: rule.properties.protocol.clone(),
                        });
                        public_endpoints += 1;
                    }
                }
            }
        }
        
        Ok(NetworkTopology {
            total_endpoints: response.value.len(),
            public_endpoints,
            critical_exposures,
        })
    }

    fn calculate_average(timeseries: &[TimeSeries]) -> f64 {
        if timeseries.is_empty() {
            return 0.0;
        }
        
        let sum: f64 = timeseries.iter()
            .flat_map(|ts| &ts.data)
            .filter_map(|d| d.average)
            .sum();
        
        let count = timeseries.iter()
            .flat_map(|ts| &ts.data)
            .filter(|d| d.average.is_some())
            .count();
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    fn calculate_percentile(timeseries: &[TimeSeries], percentile: usize) -> f64 {
        let mut values: Vec<f64> = timeseries.iter()
            .flat_map(|ts| &ts.data)
            .filter_map(|d| d.average)
            .collect();
        
        if values.is_empty() {
            return 0.0;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile as f64 / 100.0 * values.len() as f64) as usize;
        values[index.min(values.len() - 1)]
    }
}

// Data structures for real Azure responses
#[derive(Debug, Deserialize, Serialize)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub location: String,
    pub tags: Option<std::collections::HashMap<String, String>>,
    pub properties: Option<serde_json::Value>,
}

#[derive(Debug, Default)]
pub struct VmMetrics {
    pub cpu_avg: f64,
    pub cpu_p95: f64,
    pub memory_avg: f64,
    pub memory_p95: f64,
    pub network_io_avg: f64,
    pub disk_io_avg: f64,
}

#[derive(Debug, Deserialize)]
struct MetricsResponse {
    value: Vec<Metric>,
}

#[derive(Debug, Deserialize)]
struct Metric {
    name: MetricName,
    timeseries: Vec<TimeSeries>,
}

#[derive(Debug, Deserialize)]
struct MetricName {
    value: String,
}

#[derive(Debug, Deserialize)]
struct TimeSeries {
    data: Vec<MetricData>,
}

#[derive(Debug, Deserialize)]
struct MetricData {
    #[serde(rename = "timeStamp")]
    timestamp: String,
    average: Option<f64>,
}

#[derive(Debug)]
pub struct CostData {
    pub current_spend: f64,
    pub predicted_spend: f64,
    pub savings_identified: f64,
    pub idle_resources: Vec<IdleResourceCost>,
    pub cost_by_service: std::collections::HashMap<String, f64>,
}

#[derive(Debug)]
pub struct IdleResourceCost {
    pub resource_id: String,
    pub monthly_cost: f64,
    pub cpu_avg: f64,
}

#[derive(Debug, Deserialize)]
struct CostQueryResponse {
    properties: CostQueryProperties,
}

#[derive(Debug, Deserialize)]
struct CostQueryProperties {
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug)]
pub struct PolicyCompliance {
    pub total_policies: usize,
    pub compliant_resources: usize,
    pub non_compliant_resources: usize,
    pub compliance_rate: f64,
    pub violations: Vec<PolicyViolation>,
}

#[derive(Debug)]
pub struct PolicyViolation {
    pub policy_name: String,
    pub resource_id: String,
    pub resource_type: String,
    pub compliance_state: String,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
struct PolicySummaryResponse {
    value: Vec<PolicySummary>,
}

#[derive(Debug, Deserialize)]
struct PolicySummary {
    results: PolicyResults,
    #[serde(rename = "policyAssignments")]
    policy_assignments: PolicyAssignments,
}

#[derive(Debug, Deserialize)]
struct PolicyResults {
    #[serde(rename = "resourceCount")]
    resource_count: usize,
    #[serde(rename = "nonCompliantResourceCount")]
    non_compliant_resource_count: usize,
    #[serde(rename = "compliantResourceCount")]
    compliant_resource_count: usize,
}

#[derive(Debug, Deserialize)]
struct PolicyAssignments {
    count: usize,
}

#[derive(Debug, Deserialize)]
struct PolicyStatesResponse {
    value: Vec<PolicyState>,
}

#[derive(Debug, Deserialize)]
struct PolicyState {
    #[serde(rename = "resourceId")]
    resource_id: String,
    #[serde(rename = "resourceType")]
    resource_type: String,
    #[serde(rename = "policyDefinitionName")]
    policy_definition_name: String,
    #[serde(rename = "complianceState")]
    compliance_state: String,
    timestamp: String,
}

#[derive(Debug)]
pub struct SecurityAlert {
    pub id: String,
    pub name: String,
    pub severity: String,
    pub status: String,
    pub description: String,
    pub remediation_steps: Vec<String>,
    pub affected_resources: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AlertsResponse {
    value: Vec<Alert>,
}

#[derive(Debug, Deserialize)]
struct Alert {
    id: String,
    name: String,
    properties: AlertProperties,
}

#[derive(Debug, Deserialize)]
struct AlertProperties {
    severity: String,
    status: String,
    description: String,
    #[serde(rename = "remediationSteps")]
    remediation_steps: Vec<String>,
    #[serde(rename = "affectedResources")]
    affected_resources: Vec<String>,
}

#[derive(Debug)]
pub struct RbacData {
    pub total_assignments: usize,
    pub overprivileged_count: usize,
    pub risk_score: f64,
    pub role_assignments: Vec<RoleAssignment>,
}

#[derive(Debug, Deserialize)]
struct RoleAssignmentsResponse {
    value: Vec<RoleAssignment>,
}

#[derive(Debug, Deserialize)]
pub struct RoleAssignment {
    id: String,
    name: String,
    properties: RoleAssignmentProperties,
}

#[derive(Debug, Deserialize)]
struct RoleAssignmentProperties {
    #[serde(rename = "roleDefinitionId")]
    role_definition_id: String,
    #[serde(rename = "principalId")]
    principal_id: String,
    scope: String,
}

#[derive(Debug)]
pub struct NetworkTopology {
    pub total_endpoints: usize,
    pub public_endpoints: usize,
    pub critical_exposures: Vec<CriticalExposure>,
}

#[derive(Debug)]
pub struct CriticalExposure {
    pub resource: String,
    pub port: String,
    pub protocol: String,
}

#[derive(Debug, Deserialize)]
struct NsgResponse {
    value: Vec<NetworkSecurityGroup>,
}

#[derive(Debug, Deserialize)]
struct NetworkSecurityGroup {
    name: String,
    properties: NsgProperties,
}

#[derive(Debug, Deserialize)]
struct NsgProperties {
    #[serde(rename = "securityRules")]
    security_rules: Vec<SecurityRule>,
}

#[derive(Debug, Deserialize)]
struct SecurityRule {
    properties: SecurityRuleProperties,
}

#[derive(Debug, Deserialize)]
struct SecurityRuleProperties {
    protocol: String,
    #[serde(rename = "sourceAddressPrefix")]
    source_address_prefix: String,
    #[serde(rename = "destinationPortRange")]
    destination_port_range: String,
}

struct DataCache {
    entries: std::collections::HashMap<String, CacheEntry>,
}

struct CacheEntry {
    data: Vec<u8>,
    expires_at: DateTime<Utc>,
}

impl DataCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AzureError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Authentication error: {0}")]
    Authentication(String),
    #[error("HTTP client error: {0}")]
    HttpClient(String),
    #[error("Request error: {0}")]
    Request(String),
    #[error("API error: {0}")]
    Api(String),
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}