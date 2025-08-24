// Azure Integration Service - Main integration point for all Azure services
// This module provides a unified interface for accessing Azure data

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

// For now, we'll use a simplified approach without importing all service types
// The services will be created within the implementation

/// Main Azure integration service that coordinates all Azure service interactions
#[derive(Clone)]
pub struct AzureIntegrationService {
    client: Arc<AzureClient>,
    monitor: Arc<MonitorService>,
    governance: Arc<GovernanceService>,
    security: Arc<SecurityService>,
    operations: Arc<OperationsService>,
    devops: Arc<DevOpsService>,
    cost: Arc<CostService>,
    activity: Arc<ActivityService>,
    resource_graph: Arc<ResourceGraphService>,
    initialized: Arc<RwLock<bool>>,
}

impl AzureIntegrationService {
    /// Create a new Azure integration service
    pub async fn new() -> Result<Self> {
        info!("Initializing Azure Integration Service");
        
        let client = create_shared_client().await?;
        
        let service = Self {
            monitor: Arc::new(MonitorService::new((*client).clone())),
            governance: Arc::new(GovernanceService::new((*client).clone())),
            security: Arc::new(SecurityService::new((*client).clone())),
            operations: Arc::new(OperationsService::new((*client).clone())),
            devops: Arc::new(DevOpsService::new((*client).clone())),
            cost: Arc::new(CostService::new((*client).clone())),
            activity: Arc::new(ActivityService::new((*client).clone())),
            resource_graph: Arc::new(ResourceGraphService::new((*client).clone())),
            client,
            initialized: Arc::new(RwLock::new(false)),
        };

        // Perform initial health check
        service.health_check().await?;
        
        *service.initialized.write().await = true;
        info!("Azure Integration Service initialized successfully");
        
        Ok(service)
    }

    /// Perform health check on Azure connectivity
    pub async fn health_check(&self) -> Result<HealthStatus> {
        info!("Performing Azure health check");
        
        let mut status = HealthStatus::default();
        
        // Check management API
        match self.client.health_check().await {
            Ok(result) => {
                status.management_api = result.management_api;
                status.graph_api = result.graph_api;
            }
            Err(e) => {
                warn!("Health check failed: {}", e);
            }
        }

        // Check Resource Graph
        match self.resource_graph.query("Resources | summarize count()", Some(1)).await {
            Ok(_) => {
                status.resource_graph = true;
            }
            Err(e) => {
                warn!("Resource Graph check failed: {}", e);
            }
        }

        status.overall = status.management_api && status.graph_api && status.resource_graph;
        
        Ok(status)
    }

    /// Get monitor service
    pub fn monitor(&self) -> &MonitorService {
        &self.monitor
    }

    /// Get governance service
    pub fn governance(&self) -> &GovernanceService {
        &self.governance
    }

    /// Get security service
    pub fn security(&self) -> &SecurityService {
        &self.security
    }

    /// Get operations service
    pub fn operations(&self) -> &OperationsService {
        &self.operations
    }

    /// Get DevOps service
    pub fn devops(&self) -> &DevOpsService {
        &self.devops
    }

    /// Get cost service
    pub fn cost(&self) -> &CostService {
        &self.cost
    }

    /// Get activity service
    pub fn activity(&self) -> &ActivityService {
        &self.activity
    }

    /// Get resource graph service
    pub fn resource_graph(&self) -> &ResourceGraphService {
        &self.resource_graph
    }

    /// Check if service is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Get subscription ID
    pub fn subscription_id(&self) -> &str {
        &self.client.config.subscription_id
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct HealthStatus {
    pub overall: bool,
    pub management_api: bool,
    pub graph_api: bool,
    pub resource_graph: bool,
}

/// Create a global Azure integration service instance
lazy_static::lazy_static! {
    static ref AZURE_SERVICE: tokio::sync::OnceCell<Arc<AzureIntegrationService>> = tokio::sync::OnceCell::new();
}

/// Get or create the global Azure integration service
pub async fn get_azure_service() -> Result<Arc<AzureIntegrationService>> {
    AZURE_SERVICE
        .get_or_init(|| async {
            match AzureIntegrationService::new().await {
                Ok(service) => Arc::new(service),
                Err(e) => {
                    error!("Failed to initialize Azure service: {}", e);
                    panic!("Azure service initialization failed: {}", e);
                }
            }
        })
        .await
        .clone()
        .pipe(Ok)
}

// Helper trait for pipe operations
trait Pipe {
    fn pipe<T, F>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T,
        Self: Sized,
    {
        f(self)
    }
}

impl<T> Pipe for T {}