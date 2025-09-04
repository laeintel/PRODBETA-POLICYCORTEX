use super::{CloudError, CloudProvider, CloudResult};
use crate::cloud::models::*;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

/// Main trait for cloud provider abstraction
#[async_trait]
pub trait CloudProviderTrait: Send + Sync {
    /// Get the provider type
    fn provider_type(&self) -> CloudProvider;
    
    /// Check provider health
    async fn health_check(&self) -> CloudResult<()>;
    
    /// List all resources
    async fn list_resources(
        &self,
        resource_type: Option<ResourceType>,
        filters: HashMap<String, String>,
    ) -> CloudResult<Vec<Resource>>;
    
    /// Get a specific resource by ID
    async fn get_resource(&self, resource_id: &str) -> CloudResult<Resource>;
    
    /// Create a new resource
    async fn create_resource(
        &self,
        resource: CreateResourceRequest,
    ) -> CloudResult<Resource>;
    
    /// Update an existing resource
    async fn update_resource(
        &self,
        resource_id: &str,
        updates: HashMap<String, Value>,
    ) -> CloudResult<Resource>;
    
    /// Delete a resource
    async fn delete_resource(&self, resource_id: &str) -> CloudResult<()>;
    
    /// Apply a policy
    async fn apply_policy(&self, policy: Policy) -> CloudResult<PolicyResult>;
    
    /// Get compliance status
    async fn get_compliance_status(&self) -> CloudResult<ComplianceReport>;
    
    /// Get cost analysis
    async fn get_cost_analysis(
        &self,
        start_date: chrono::DateTime<chrono::Utc>,
        end_date: chrono::DateTime<chrono::Utc>,
    ) -> CloudResult<CostAnalysis>;
    
    /// Execute a custom command (provider-specific)
    async fn execute_custom(
        &self,
        command: &str,
        parameters: HashMap<String, Value>,
    ) -> CloudResult<Value>;
}

/// Trait for resource management
#[async_trait]
pub trait ResourceManager: Send + Sync {
    /// Tag a resource
    async fn tag_resource(
        &self,
        resource_id: &str,
        tags: HashMap<String, String>,
    ) -> CloudResult<()>;
    
    /// Get resource metrics
    async fn get_resource_metrics(
        &self,
        resource_id: &str,
        metric_names: Vec<String>,
        time_range: TimeRange,
    ) -> CloudResult<Vec<Metric>>;
    
    /// Get resource logs
    async fn get_resource_logs(
        &self,
        resource_id: &str,
        time_range: TimeRange,
        filters: Option<HashMap<String, String>>,
    ) -> CloudResult<Vec<LogEntry>>;
    
    /// Get resource relationships
    async fn get_resource_relationships(
        &self,
        resource_id: &str,
    ) -> CloudResult<ResourceRelationships>;
    
    /// Get resource configuration
    async fn get_resource_config(
        &self,
        resource_id: &str,
    ) -> CloudResult<ResourceConfiguration>;
}

/// Trait for policy management
#[async_trait]
pub trait PolicyManager: Send + Sync {
    /// List all policies
    async fn list_policies(
        &self,
        policy_type: Option<PolicyType>,
    ) -> CloudResult<Vec<Policy>>;
    
    /// Get a specific policy
    async fn get_policy(&self, policy_id: &str) -> CloudResult<Policy>;
    
    /// Create a new policy
    async fn create_policy(&self, policy: CreatePolicyRequest) -> CloudResult<Policy>;
    
    /// Update a policy
    async fn update_policy(
        &self,
        policy_id: &str,
        updates: PolicyUpdate,
    ) -> CloudResult<Policy>;
    
    /// Delete a policy
    async fn delete_policy(&self, policy_id: &str) -> CloudResult<()>;
    
    /// Evaluate policy compliance
    async fn evaluate_policy(
        &self,
        policy_id: &str,
        resource_ids: Vec<String>,
    ) -> CloudResult<PolicyEvaluation>;
    
    /// Get policy violations
    async fn get_policy_violations(
        &self,
        policy_id: Option<String>,
        time_range: TimeRange,
    ) -> CloudResult<Vec<PolicyViolation>>;
    
    /// Remediate policy violations
    async fn remediate_violations(
        &self,
        violation_ids: Vec<String>,
        remediation_type: RemediationType,
    ) -> CloudResult<RemediationResult>;
}

/// Trait for identity and access management
#[async_trait]
pub trait IdentityManager: Send + Sync {
    /// List users
    async fn list_users(&self) -> CloudResult<Vec<User>>;
    
    /// Get user details
    async fn get_user(&self, user_id: &str) -> CloudResult<User>;
    
    /// Create a user
    async fn create_user(&self, user: CreateUserRequest) -> CloudResult<User>;
    
    /// Delete a user
    async fn delete_user(&self, user_id: &str) -> CloudResult<()>;
    
    /// List roles
    async fn list_roles(&self) -> CloudResult<Vec<Role>>;
    
    /// Get role details
    async fn get_role(&self, role_id: &str) -> CloudResult<Role>;
    
    /// Create a role
    async fn create_role(&self, role: CreateRoleRequest) -> CloudResult<Role>;
    
    /// Assign role to user
    async fn assign_role(
        &self,
        user_id: &str,
        role_id: &str,
    ) -> CloudResult<()>;
    
    /// Revoke role from user
    async fn revoke_role(
        &self,
        user_id: &str,
        role_id: &str,
    ) -> CloudResult<()>;
    
    /// Get user permissions
    async fn get_user_permissions(&self, user_id: &str) -> CloudResult<Vec<Permission>>;
}

/// Trait for network management
#[async_trait]
pub trait NetworkManager: Send + Sync {
    /// List virtual networks
    async fn list_networks(&self) -> CloudResult<Vec<Network>>;
    
    /// Get network details
    async fn get_network(&self, network_id: &str) -> CloudResult<Network>;
    
    /// Create a network
    async fn create_network(&self, network: CreateNetworkRequest) -> CloudResult<Network>;
    
    /// Delete a network
    async fn delete_network(&self, network_id: &str) -> CloudResult<()>;
    
    /// List security groups
    async fn list_security_groups(&self) -> CloudResult<Vec<SecurityGroup>>;
    
    /// Get security group details
    async fn get_security_group(&self, group_id: &str) -> CloudResult<SecurityGroup>;
    
    /// Create a security group
    async fn create_security_group(
        &self,
        group: CreateSecurityGroupRequest,
    ) -> CloudResult<SecurityGroup>;
    
    /// Update security group rules
    async fn update_security_rules(
        &self,
        group_id: &str,
        rules: Vec<SecurityRule>,
    ) -> CloudResult<()>;
}

/// Trait for storage management
#[async_trait]
pub trait StorageManager: Send + Sync {
    /// List storage accounts/buckets
    async fn list_storage(&self) -> CloudResult<Vec<StorageContainer>>;
    
    /// Get storage details
    async fn get_storage(&self, storage_id: &str) -> CloudResult<StorageContainer>;
    
    /// Create storage
    async fn create_storage(
        &self,
        storage: CreateStorageRequest,
    ) -> CloudResult<StorageContainer>;
    
    /// Delete storage
    async fn delete_storage(&self, storage_id: &str) -> CloudResult<()>;
    
    /// List objects in storage
    async fn list_objects(
        &self,
        storage_id: &str,
        prefix: Option<String>,
    ) -> CloudResult<Vec<StorageObject>>;
    
    /// Upload object
    async fn upload_object(
        &self,
        storage_id: &str,
        object_key: &str,
        data: Vec<u8>,
        metadata: Option<HashMap<String, String>>,
    ) -> CloudResult<StorageObject>;
    
    /// Download object
    async fn download_object(
        &self,
        storage_id: &str,
        object_key: &str,
    ) -> CloudResult<Vec<u8>>;
    
    /// Delete object
    async fn delete_object(
        &self,
        storage_id: &str,
        object_key: &str,
    ) -> CloudResult<()>;
}

/// Trait for monitoring and observability
#[async_trait]
pub trait MonitoringManager: Send + Sync {
    /// Create an alert
    async fn create_alert(&self, alert: CreateAlertRequest) -> CloudResult<Alert>;
    
    /// List alerts
    async fn list_alerts(
        &self,
        filters: Option<HashMap<String, String>>,
    ) -> CloudResult<Vec<Alert>>;
    
    /// Get alert details
    async fn get_alert(&self, alert_id: &str) -> CloudResult<Alert>;
    
    /// Update alert
    async fn update_alert(
        &self,
        alert_id: &str,
        updates: AlertUpdate,
    ) -> CloudResult<Alert>;
    
    /// Delete alert
    async fn delete_alert(&self, alert_id: &str) -> CloudResult<()>;
    
    /// Get metrics
    async fn get_metrics(
        &self,
        metric_query: MetricQuery,
    ) -> CloudResult<Vec<MetricData>>;
    
    /// Create dashboard
    async fn create_dashboard(
        &self,
        dashboard: CreateDashboardRequest,
    ) -> CloudResult<Dashboard>;
    
    /// Get logs
    async fn get_logs(
        &self,
        log_query: LogQuery,
    ) -> CloudResult<Vec<LogEntry>>;
}

/// Trait for cost management
#[async_trait]
pub trait CostManager: Send + Sync {
    /// Get current costs
    async fn get_current_costs(&self) -> CloudResult<CurrentCosts>;
    
    /// Get cost forecast
    async fn get_cost_forecast(
        &self,
        days_ahead: u32,
    ) -> CloudResult<CostForecast>;
    
    /// Get cost breakdown
    async fn get_cost_breakdown(
        &self,
        group_by: CostGroupBy,
        time_range: TimeRange,
    ) -> CloudResult<CostBreakdown>;
    
    /// Get cost recommendations
    async fn get_cost_recommendations(&self) -> CloudResult<Vec<CostRecommendation>>;
    
    /// Set budget alert
    async fn set_budget_alert(
        &self,
        budget: BudgetAlert,
    ) -> CloudResult<()>;
    
    /// Get budget status
    async fn get_budget_status(&self) -> CloudResult<Vec<BudgetStatus>>;
}