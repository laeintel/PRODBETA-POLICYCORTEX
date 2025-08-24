// Health check API for Azure service connectivity
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};

use crate::api::AppState;
// use crate::azure_integration::get_azure_service; // Temporarily commented out

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub services: ServiceHealthStatus,
    pub azure_connectivity: AzureConnectivityStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceHealthStatus {
    pub api: bool,
    pub database: bool,
    pub cache: bool,
    pub azure_integration: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AzureConnectivityStatus {
    pub management_api: bool,
    pub graph_api: bool,
    pub resource_graph: bool,
    pub monitor: bool,
    pub governance: bool,
    pub cost_management: bool,
    pub details: Vec<ServiceDetail>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceDetail {
    pub service: String,
    pub status: String,
    pub message: Option<String>,
    pub latency_ms: Option<u64>,
}

// GET /api/v1/health
pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    info!("Performing health check");
    
    let mut azure_connectivity = AzureConnectivityStatus {
        management_api: false,
        graph_api: false,
        resource_graph: false,
        monitor: false,
        governance: false,
        cost_management: false,
        details: Vec::new(),
    };

    let mut azure_integration_healthy = false;

    // Check Azure integration (temporarily disabled)
    /*
    match get_azure_service().await {
        Ok(azure) => {
            // Check basic connectivity
            match azure.health_check().await {
                Ok(status) => {
                    azure_connectivity.management_api = status.management_api;
                    azure_connectivity.graph_api = status.graph_api;
                    azure_connectivity.resource_graph = status.resource_graph;
                    azure_integration_healthy = status.overall;
                    
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Management API".to_string(),
                        status: if status.management_api { "healthy" } else { "unhealthy" }.to_string(),
                        message: None,
                        latency_ms: None,
                    });
                    
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Graph API".to_string(),
                        status: if status.graph_api { "healthy" } else { "unhealthy" }.to_string(),
                        message: None,
                        latency_ms: None,
                    });
                    
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Resource Graph".to_string(),
                        status: if status.resource_graph { "healthy" } else { "unhealthy" }.to_string(),
                        message: None,
                        latency_ms: None,
                    });
                }
                Err(e) => {
                    error!("Azure health check failed: {}", e);
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Azure Integration".to_string(),
                        status: "error".to_string(),
                        message: Some(format!("Health check failed: {}", e)),
                        latency_ms: None,
                    });
                }
            }

            // Test specific services with timing sequentially
            // Test Monitor service
            let start = std::time::Instant::now();
            match test_monitor_service(&azure).await {
                Ok(()) => {
                    let latency = start.elapsed().as_millis() as u64;
                    azure_connectivity.monitor = true;
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Monitor".to_string(),
                        status: "healthy".to_string(),
                        message: None,
                        latency_ms: Some(latency),
                    });
                }
                Err(e) => {
                    warn!("Monitor service check failed: {}", e);
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Monitor".to_string(),
                        status: "unhealthy".to_string(),
                        message: Some(e.to_string()),
                        latency_ms: None,
                    });
                }
            }

            // Test Governance service
            let start = std::time::Instant::now();
            match test_governance_service(&azure).await {
                Ok(()) => {
                    let latency = start.elapsed().as_millis() as u64;
                    azure_connectivity.governance = true;
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Governance".to_string(),
                        status: "healthy".to_string(),
                        message: None,
                        latency_ms: Some(latency),
                    });
                }
                Err(e) => {
                    warn!("Governance service check failed: {}", e);
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Governance".to_string(),
                        status: "unhealthy".to_string(),
                        message: Some(e.to_string()),
                        latency_ms: None,
                    });
                }
            }

            // Test Cost Management service
            let start = std::time::Instant::now();
            match test_cost_service(&azure).await {
                Ok(()) => {
                    let latency = start.elapsed().as_millis() as u64;
                    azure_connectivity.cost_management = true;
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Cost Management".to_string(),
                        status: "healthy".to_string(),
                        message: None,
                        latency_ms: Some(latency),
                    });
                }
                Err(e) => {
                    warn!("Cost Management service check failed: {}", e);
                    azure_connectivity.details.push(ServiceDetail {
                        service: "Cost Management".to_string(),
                        status: "unhealthy".to_string(),
                        message: Some(e.to_string()),
                        latency_ms: None,
                    });
                }
            }
        }
        Err(e) => {
            error!("Failed to get Azure service: {}", e);
            azure_connectivity.details.push(ServiceDetail {
                service: "Azure Integration".to_string(),
                status: "unavailable".to_string(),
                message: Some(format!("Service initialization failed: {}", e)),
                latency_ms: None,
            });
        }
    }
    */
    
    // Add mock Azure connectivity details for now
    azure_connectivity.details.push(ServiceDetail {
        service: "Azure Integration".to_string(),
        status: "disabled".to_string(),
        message: Some("Azure integration temporarily disabled for compilation".to_string()),
        latency_ms: None,
    });

    // Check other services (simplified)
    let service_health = ServiceHealthStatus {
        api: true, // API is responding if we got here
        database: check_database_health(&state).await,
        cache: check_cache_health(&state).await,
        azure_integration: azure_integration_healthy,
    };

    let overall_status = if service_health.api && azure_integration_healthy {
        "healthy"
    } else if service_health.api {
        "degraded"
    } else {
        "unhealthy"
    };

    let response = HealthCheckResponse {
        status: overall_status.to_string(),
        timestamp: Utc::now(),
        services: service_health,
        azure_connectivity,
    };

    (StatusCode::OK, Json(response))
}

// GET /api/v1/health/azure
pub async fn azure_health_check(_state: State<Arc<AppState>>) -> impl IntoResponse {
    info!("Performing detailed Azure health check");
    
    // Azure integration temporarily disabled
    let response = serde_json::json!({
        "status": "disabled",
        "timestamp": Utc::now(),
        "services": {
            "management_api": false,
            "graph_api": false,
            "resource_graph": false,
        },
        "message": "Azure integration temporarily disabled for compilation"
    });
    (StatusCode::OK, Json(response))
}

/*
async fn test_monitor_service(azure: &crate::azure_integration::AzureIntegrationService) -> anyhow::Result<()> {
    // Simple test - try to get system health
    let _ = azure.monitor().get_system_health().await?;
    Ok(())
}

async fn test_governance_service(azure: &crate::azure_integration::AzureIntegrationService) -> anyhow::Result<()> {
    // Simple test - try to get compliance summary
    let _ = azure.governance().get_compliance_summary().await?;
    Ok(())
}

async fn test_cost_service(azure: &crate::azure_integration::AzureIntegrationService) -> anyhow::Result<()> {
    // Simple test - try to get current month costs
    let _ = azure.cost().get_current_month_costs().await?;
    Ok(())
}
*/

async fn check_database_health(_state: &AppState) -> bool {
    // Database health check - currently not implemented
    // The application uses in-memory storage for now
    true
}

async fn check_cache_health(_state: &AppState) -> bool {
    // Cache health check - currently not implemented
    // The application uses in-memory caching for now
    true
}