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
use crate::azure_integration::get_azure_service;

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

    // Check Azure integration
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

            // Test specific services with timing
            let services_to_test = vec![
                ("Monitor", test_monitor_service(&azure)),
                ("Governance", test_governance_service(&azure)),
                ("Cost Management", test_cost_service(&azure)),
            ];

            for (service_name, test_future) in services_to_test {
                let start = std::time::Instant::now();
                match test_future.await {
                    Ok(()) => {
                        let latency = start.elapsed().as_millis() as u64;
                        match service_name {
                            "Monitor" => azure_connectivity.monitor = true,
                            "Governance" => azure_connectivity.governance = true,
                            "Cost Management" => azure_connectivity.cost_management = true,
                            _ => {}
                        }
                        azure_connectivity.details.push(ServiceDetail {
                            service: service_name.to_string(),
                            status: "healthy".to_string(),
                            message: None,
                            latency_ms: Some(latency),
                        });
                    }
                    Err(e) => {
                        warn!("{} service check failed: {}", service_name, e);
                        azure_connectivity.details.push(ServiceDetail {
                            service: service_name.to_string(),
                            status: "unhealthy".to_string(),
                            message: Some(e.to_string()),
                            latency_ms: None,
                        });
                    }
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
    
    match get_azure_service().await {
        Ok(azure) => {
            match azure.health_check().await {
                Ok(status) => {
                    let response = serde_json::json!({
                        "status": if status.overall { "healthy" } else { "unhealthy" },
                        "timestamp": Utc::now(),
                        "services": {
                            "management_api": status.management_api,
                            "graph_api": status.graph_api,
                            "resource_graph": status.resource_graph,
                        },
                        "message": if status.overall {
                            "All Azure services are accessible"
                        } else {
                            "Some Azure services are not accessible"
                        }
                    });
                    (StatusCode::OK, Json(response))
                }
                Err(e) => {
                    let response = serde_json::json!({
                        "status": "error",
                        "timestamp": Utc::now(),
                        "error": e.to_string()
                    });
                    (StatusCode::SERVICE_UNAVAILABLE, Json(response))
                }
            }
        }
        Err(e) => {
            let response = serde_json::json!({
                "status": "unavailable",
                "timestamp": Utc::now(),
                "error": format!("Azure service not initialized: {}", e)
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(response))
        }
    }
}

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

async fn check_database_health(state: &AppState) -> bool {
    // Check if database connection pool is healthy
    if let Some(ref pool) = state.pool {
        match sqlx::query("SELECT 1").fetch_one(pool).await {
            Ok(_) => true,
            Err(e) => {
                warn!("Database health check failed: {}", e);
                false
            }
        }
    } else {
        false
    }
}

async fn check_cache_health(state: &AppState) -> bool {
    // Check if Redis cache is healthy
    if let Some(ref cache) = state.cache {
        match cache.ping().await {
            Ok(_) => true,
            Err(e) => {
                warn!("Cache health check failed: {}", e);
                false
            }
        }
    } else {
        false
    }
}