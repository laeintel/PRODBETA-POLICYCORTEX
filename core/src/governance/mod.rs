// PolicyCortex Governance Module
// Implements the 15 Azure governance tools integration for unified cloud governance

pub mod resource_graph;
pub mod policy_engine;
pub mod identity;
pub mod monitoring;
pub mod cost_management;
pub mod security_posture;
pub mod access_control;
pub mod network;
pub mod optimization;
pub mod blueprints;
pub mod unified_api;

// AI-powered governance features (Patents 1, 2, 4)
pub mod ai;

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::azure_client::AzureClient;

/// Central governance coordinator that orchestrates all governance tools
pub struct GovernanceCoordinator {
    /// Azure Resource Graph for resource discovery and querying
    pub resource_graph: Arc<resource_graph::ResourceGraphClient>,

    /// Azure Policy engine for policy management and compliance
    pub policy_engine: Arc<policy_engine::PolicyEngine>,

    /// Identity governance for access control and identity management
    pub identity: Arc<identity::IdentityGovernanceClient>,

    /// Azure Monitor integration for governance monitoring
    pub monitoring: Arc<monitoring::GovernanceMonitor>,

    /// Cost management and financial governance
    pub cost_management: Arc<cost_management::CostGovernanceEngine>,

    /// Security posture management via Defender for Cloud
    pub security_posture: Arc<security_posture::SecurityPostureEngine>,

    /// Access control and RBAC management
    pub access_control: Arc<access_control::AccessGovernanceEngine>,

    /// Network security governance
    pub network: Arc<network::NetworkGovernanceEngine>,

    /// Optimization recommendations via Azure Advisor
    pub optimization: Arc<optimization::OptimizationEngine>,

    /// Environment governance via Azure Blueprints
    pub blueprints: Arc<blueprints::GovernanceBlueprints>,

    /// AI-powered governance intelligence
    pub ai_engine: Arc<ai::AIGovernanceEngine>,

    /// Unified API layer for governance operations
    pub unified_api: Arc<unified_api::UnifiedGovernanceAPI>,
}

impl GovernanceCoordinator {
    /// Create a new governance coordinator with all Azure integrations
    pub async fn new(azure_client: Arc<AzureClient>) -> Result<Self, GovernanceError> {
        let resource_graph = Arc::new(resource_graph::ResourceGraphClient::new(azure_client.clone()).await?);
        let policy_engine = Arc::new(policy_engine::PolicyEngine::new(azure_client.clone()).await?);
        let identity = Arc::new(identity::IdentityGovernanceClient::new(azure_client.clone()).await?);
        let monitoring = Arc::new(monitoring::GovernanceMonitor::new(azure_client.clone()).await?);
        let cost_management = Arc::new(cost_management::CostGovernanceEngine::new(azure_client.clone()).await?);
        let security_posture = Arc::new(security_posture::SecurityPostureEngine::new(azure_client.clone()).await?);
        let access_control = Arc::new(access_control::AccessGovernanceEngine::new(azure_client.clone()).await?);
        let network = Arc::new(network::NetworkGovernanceEngine::new(azure_client.clone()).await?);
        let optimization = Arc::new(optimization::OptimizationEngine::new(azure_client.clone()).await?);
        let blueprints = Arc::new(blueprints::GovernanceBlueprints::new(azure_client.clone()).await?);

        // AI engines can reference individual components instead of full coordinator
        let ai_engine = Arc::new(ai::AIGovernanceEngine::new(
            resource_graph.clone(),
            policy_engine.clone(),
            identity.clone(),
            monitoring.clone(),
        ).await?);

        let unified_api = Arc::new(unified_api::UnifiedGovernanceAPI::new(
            resource_graph.clone(),
            policy_engine.clone(),
            identity.clone(),
            monitoring.clone(),
            ai_engine.clone(),
        ).await);

        Ok(Self {
            resource_graph,
            policy_engine,
            identity,
            monitoring,
            cost_management,
            security_posture,
            access_control,
            network,
            optimization,
            blueprints,
            ai_engine,
            unified_api,
        })
    }

    /// Get comprehensive governance health status across all domains
    pub async fn get_governance_health(&self) -> Result<GovernanceHealthReport, GovernanceError> {
        // Parallel health checks across all governance domains
        let (
            resource_health,
            policy_health,
            identity_health,
            monitoring_health,
            cost_health,
            security_health,
            access_health,
            network_health,
            optimization_health,
            blueprints_health,
        ) = tokio::join!(
            self.resource_graph.health_check(),
            self.policy_engine.health_check(),
            self.identity.health_check(),
            self.monitoring.health_check(),
            self.cost_management.health_check(),
            self.security_posture.health_check(),
            self.access_control.health_check(),
            self.network.health_check(),
            self.optimization.health_check(),
            self.blueprints.health_check(),
        );

        Ok(GovernanceHealthReport {
            overall_status: calculate_overall_health(&[
                resource_health.status,
                policy_health.status,
                identity_health.status,
                monitoring_health.status,
                cost_health.status,
                security_health.status,
                access_health.status,
                network_health.status,
                optimization_health.status,
                blueprints_health.status,
            ]),
            resource_graph: resource_health,
            policy_engine: policy_health,
            identity: identity_health,
            monitoring: monitoring_health,
            cost_management: cost_health,
            security_posture: security_health,
            access_control: access_health,
            network: network_health,
            optimization: optimization_health,
            blueprints: blueprints_health,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Common error types for governance operations
#[derive(Debug, thiserror::Error)]
pub enum GovernanceError {
    #[error("Azure API error: {0}")]
    AzureApi(#[from] azure_core::Error),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Policy error: {0}")]
    Policy(String),

    #[error("Compliance error: {0}")]
    Compliance(String),

    #[error("Resource discovery error: {0}")]
    ResourceDiscovery(String),

    #[error("Network governance error: {0}")]
    Network(String),

    #[error("Cost governance error: {0}")]
    Cost(String),

    #[error("Security governance error: {0}")]
    Security(String),

    #[error("AI governance error: {0}")]
    AI(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Health status for governance components
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "Healthy"),
            HealthStatus::Degraded => write!(f, "Degraded"),
            HealthStatus::Unhealthy => write!(f, "Unhealthy"),
        }
    }
}

/// Individual component health report
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub component: String,
    pub status: HealthStatus,
    pub message: String,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Comprehensive governance health report
#[derive(Debug, Clone)]
pub struct GovernanceHealthReport {
    pub overall_status: HealthStatus,
    pub resource_graph: ComponentHealth,
    pub policy_engine: ComponentHealth,
    pub identity: ComponentHealth,
    pub monitoring: ComponentHealth,
    pub cost_management: ComponentHealth,
    pub security_posture: ComponentHealth,
    pub access_control: ComponentHealth,
    pub network: ComponentHealth,
    pub optimization: ComponentHealth,
    pub blueprints: ComponentHealth,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Calculate overall health from component statuses
fn calculate_overall_health(statuses: &[HealthStatus]) -> HealthStatus {
    if statuses.iter().any(|s| *s == HealthStatus::Unhealthy) {
        HealthStatus::Unhealthy
    } else if statuses.iter().any(|s| *s == HealthStatus::Degraded) {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

/// Common result type for governance operations
pub type GovernanceResult<T> = Result<T, GovernanceError>;