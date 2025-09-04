use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource types supported across cloud providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    VirtualMachine,
    Container,
    Database,
    Storage,
    Network,
    LoadBalancer,
    SecurityGroup,
    KeyVault,
    Function,
    ApiGateway,
    CDN,
    DNS,
    Monitoring,
    Custom(String),
}

/// Generic cloud resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    pub name: String,
    pub resource_type: ResourceType,
    pub provider: String,
    pub region: String,
    pub status: ResourceStatus,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Resource status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceStatus {
    Running,
    Stopped,
    Creating,
    Updating,
    Deleting,
    Failed,
    Unknown,
}

/// Request to create a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResourceRequest {
    pub name: String,
    pub resource_type: ResourceType,
    pub region: String,
    pub tags: HashMap<String, String>,
    pub configuration: HashMap<String, serde_json::Value>,
}

/// Policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub policy_type: PolicyType,
    pub rules: Vec<PolicyRule>,
    pub enforcement: EnforcementMode,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Policy types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyType {
    Security,
    Compliance,
    Cost,
    Performance,
    Governance,
    Custom(String),
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub condition: String,
    pub action: PolicyAction,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Policy action
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    Audit,
    Remediate,
    Alert,
}

/// Enforcement mode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementMode {
    Enforced,
    Disabled,
    AuditOnly,
}

/// Policy result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy_id: String,
    pub success: bool,
    pub applied_to: Vec<String>,
    pub failures: Vec<PolicyFailure>,
    pub timestamp: DateTime<Utc>,
}

/// Policy failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyFailure {
    pub resource_id: String,
    pub reason: String,
    pub details: Option<String>,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub timestamp: DateTime<Utc>,
    pub overall_score: f64,
    pub compliant_resources: u32,
    pub non_compliant_resources: u32,
    pub total_resources: u32,
    pub by_policy: Vec<PolicyCompliance>,
    pub by_resource_type: HashMap<String, ComplianceStats>,
}

/// Policy compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCompliance {
    pub policy_id: String,
    pub policy_name: String,
    pub compliance_percentage: f64,
    pub violations: Vec<Violation>,
}

/// Compliance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStats {
    pub total: u32,
    pub compliant: u32,
    pub non_compliant: u32,
    pub percentage: f64,
}

/// Violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub resource_id: String,
    pub resource_name: String,
    pub violation_type: String,
    pub severity: Severity,
    pub details: String,
    pub detected_at: DateTime<Utc>,
}

/// Severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub total_cost: f64,
    pub currency: String,
    pub by_service: HashMap<String, f64>,
    pub by_region: HashMap<String, f64>,
    pub by_tag: HashMap<String, f64>,
    pub trends: Vec<CostTrend>,
    pub recommendations: Vec<CostRecommendation>,
}

/// Cost trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTrend {
    pub date: DateTime<Utc>,
    pub cost: f64,
    pub forecast: Option<f64>,
}

/// Cost recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    pub id: String,
    pub description: String,
    pub potential_savings: f64,
    pub effort: EffortLevel,
    pub resources_affected: Vec<String>,
}

/// Effort level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

/// Time range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: DateTime<Utc>,
    pub labels: HashMap<String, String>,
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub source: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Resource relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRelationships {
    pub resource_id: String,
    pub parents: Vec<ResourceReference>,
    pub children: Vec<ResourceReference>,
    pub dependencies: Vec<ResourceReference>,
}

/// Resource reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReference {
    pub id: String,
    pub name: String,
    pub resource_type: ResourceType,
    pub relationship_type: String,
}

/// Resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfiguration {
    pub resource_id: String,
    pub configuration: HashMap<String, serde_json::Value>,
    pub configuration_version: String,
    pub last_modified: DateTime<Utc>,
}

/// Create policy request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePolicyRequest {
    pub name: String,
    pub description: String,
    pub policy_type: PolicyType,
    pub rules: Vec<PolicyRule>,
    pub enforcement: EnforcementMode,
}

/// Policy update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyUpdate {
    pub name: Option<String>,
    pub description: Option<String>,
    pub rules: Option<Vec<PolicyRule>>,
    pub enforcement: Option<EnforcementMode>,
}

/// Policy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub policy_id: String,
    pub evaluated_resources: Vec<String>,
    pub compliant: Vec<String>,
    pub non_compliant: Vec<String>,
    pub evaluation_time: DateTime<Utc>,
}

/// Policy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    pub id: String,
    pub policy_id: String,
    pub resource_id: String,
    pub violation_details: String,
    pub severity: Severity,
    pub detected_at: DateTime<Utc>,
}

/// Remediation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemediationType {
    Automatic,
    Manual,
    Scheduled,
}

/// Remediation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationResult {
    pub successful: Vec<String>,
    pub failed: Vec<RemediationFailure>,
    pub timestamp: DateTime<Utc>,
}

/// Remediation failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationFailure {
    pub violation_id: String,
    pub reason: String,
}

// Identity and Access Management models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub email: String,
    pub roles: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: Option<String>,
    pub roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: Vec<Permission>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRoleRequest {
    pub name: String,
    pub description: String,
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub id: String,
    pub name: String,
    pub resource: String,
    pub actions: Vec<String>,
}

// Network models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub id: String,
    pub name: String,
    pub cidr: String,
    pub subnets: Vec<Subnet>,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subnet {
    pub id: String,
    pub name: String,
    pub cidr: String,
    pub availability_zone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNetworkRequest {
    pub name: String,
    pub cidr: String,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityGroup {
    pub id: String,
    pub name: String,
    pub description: String,
    pub rules: Vec<SecurityRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    pub id: String,
    pub direction: String,
    pub protocol: String,
    pub port_range: String,
    pub source: String,
    pub destination: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSecurityGroupRequest {
    pub name: String,
    pub description: String,
    pub rules: Vec<SecurityRule>,
}

// Storage models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageContainer {
    pub id: String,
    pub name: String,
    pub storage_type: String,
    pub region: String,
    pub size_bytes: u64,
    pub object_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateStorageRequest {
    pub name: String,
    pub storage_type: String,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    pub key: String,
    pub size_bytes: u64,
    pub content_type: String,
    pub last_modified: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

// Monitoring models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub severity: Severity,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAlertRequest {
    pub name: String,
    pub condition: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertUpdate {
    pub name: Option<String>,
    pub condition: Option<String>,
    pub severity: Option<Severity>,
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricQuery {
    pub metric_name: String,
    pub aggregation: String,
    pub time_range: TimeRange,
    pub filters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    pub metric_name: String,
    pub data_points: Vec<DataPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub id: String,
    pub name: String,
    pub widgets: Vec<Widget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub id: String,
    pub widget_type: String,
    pub title: String,
    pub query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDashboardRequest {
    pub name: String,
    pub widgets: Vec<Widget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogQuery {
    pub query: String,
    pub time_range: TimeRange,
    pub limit: Option<u32>,
}

// Cost management models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentCosts {
    pub total: f64,
    pub currency: String,
    pub period: String,
    pub by_service: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostForecast {
    pub forecasted_cost: f64,
    pub confidence_interval: (f64, f64),
    pub forecast_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostGroupBy {
    Service,
    Region,
    Tag(String),
    ResourceType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub group_by: CostGroupBy,
    pub breakdown: HashMap<String, f64>,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub name: String,
    pub amount: f64,
    pub threshold_percentage: f64,
    pub notification_emails: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub budget_name: String,
    pub budget_amount: f64,
    pub spent: f64,
    pub percentage_used: f64,
    pub forecast_overspend: Option<f64>,
}