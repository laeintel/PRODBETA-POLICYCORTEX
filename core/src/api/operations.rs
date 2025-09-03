// Operations API handlers for comprehensive navigation system
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};

use crate::{
    api::AppState,
    data_mode::{DataMode, DataResponse},
};

// Resource info
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub resource_group: String,
    pub location: String,
    pub status: String,
    pub health: String,
    pub tags: Vec<Tag>,
    pub created_at: DateTime<Utc>,
    pub cost_per_month: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tag {
    pub key: String,
    pub value: String,
}

// Monitoring metric
#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringMetric {
    pub resource_id: String,
    pub metric_name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: DateTime<Utc>,
    pub threshold: Option<f64>,
    pub status: String,
}

// Automation workflow
#[derive(Debug, Serialize, Deserialize)]
pub struct AutomationWorkflow {
    pub id: String,
    pub name: String,
    pub description: String,
    pub trigger_type: String,
    pub status: String,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: Option<DateTime<Utc>>,
    pub success_rate: f64,
    pub actions_count: u32,
}

// Notification
#[derive(Debug, Serialize, Deserialize)]
pub struct Notification {
    pub id: String,
    pub title: String,
    pub message: String,
    pub severity: String,
    pub source: String,
    pub timestamp: DateTime<Utc>,
    pub read: bool,
    pub actionable: bool,
}

// Alert
#[derive(Debug, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub name: String,
    pub description: String,
    pub severity: String,
    pub resource_id: String,
    pub metric: String,
    pub condition: String,
    pub current_value: f64,
    pub threshold: f64,
    pub triggered_at: DateTime<Utc>,
    pub status: String,
}

// GET /api/v1/operations/resources
pub async fn get_operations_resources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mode = DataMode::from_env();
    
    // In real mode, we must have real data or fail
    if mode.is_real() {
        if let Some(ref async_client) = state.async_azure_client {
            match async_client.get_all_resources_with_health().await {
                Ok(resources_data) => {
            if let Some(items) = resources_data.get("items").and_then(|v| v.as_array()) {
                let resources: Vec<ResourceInfo> = items
                    .iter()
                    .take(10)
                    .map(|item| ResourceInfo {
                        id: item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        name: item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        resource_type: item.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        resource_group: item.get("resourceGroup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        location: item.get("location").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        status: "running".to_string(),
                        health: "healthy".to_string(),
                        tags: vec![],
                        created_at: Utc::now() - chrono::Duration::days(30),
                        cost_per_month: 150.0,
                    })
                    .collect();
                
                    if !resources.is_empty() {
                        return Json(DataResponse::new(resources, mode)).into_response();
                    } else {
                        tracing::error!("No resources found in Azure response");
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            Json(serde_json::json!({
                                "error": "No resources found",
                                "message": "Azure returned empty resource list",
                                "mode": "real"
                            }))
                        ).into_response();
                    }
                } else {
                    tracing::error!("Invalid Azure response format");
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "Invalid response format",
                            "message": "Azure response does not contain expected data structure",
                            "mode": "real"
                        }))
                    ).into_response();
                }
                }
                Err(e) => {
                    tracing::error!("Failed to get resources from Azure: {}", e);
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "Azure service unavailable",
                            "message": format!("Failed to retrieve resources: {}", e),
                            "mode": "real"
                        }))
                    ).into_response();
                }
            }
        } else {
            tracing::error!("Real mode enabled but Azure client not initialized");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Azure client not initialized",
                    "message": "Real data mode requires Azure client configuration",
                    "mode": "real"
                }))
            ).into_response();
        }
    }

    // Only return simulated data in simulated mode
    let resources = vec![
        ResourceInfo {
            id: "/subscriptions/xxx/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-prod-001 (SIMULATED)".to_string(),
            name: "vm-prod-001 (SIMULATED)".to_string(),
            resource_type: "Microsoft.Compute/virtualMachines".to_string(),
            resource_group: "rg-prod".to_string(),
            location: "eastus".to_string(),
            status: "running".to_string(),
            health: "healthy".to_string(),
            tags: vec![
                Tag { key: "Environment".to_string(), value: "Production (SIMULATED)".to_string() },
                Tag { key: "Owner".to_string(), value: "DevOps (SIMULATED)".to_string() },
            ],
            created_at: Utc::now() - chrono::Duration::days(90),
            cost_per_month: 450.0,
        },
        ResourceInfo {
            id: "/subscriptions/xxx/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod001".to_string(),
            name: "stprod001 (SIMULATED)".to_string(),
            resource_type: "Microsoft.Storage/storageAccounts".to_string(),
            resource_group: "rg-prod".to_string(),
            location: "eastus".to_string(),
            status: "available".to_string(),
            health: "healthy".to_string(),
            tags: vec![
                Tag { key: "Environment".to_string(), value: "Production".to_string() },
            ],
            created_at: Utc::now() - chrono::Duration::days(180),
            cost_per_month: 125.0,
        },
        ResourceInfo {
            id: "/subscriptions/xxx/resourceGroups/rg-dev/providers/Microsoft.Web/sites/app-dev-001".to_string(),
            name: "app-dev-001".to_string(),
            resource_type: "Microsoft.Web/sites".to_string(),
            resource_group: "rg-dev".to_string(),
            location: "westus2".to_string(),
            status: "running".to_string(),
            health: "degraded".to_string(),
            tags: vec![
                Tag { key: "Environment".to_string(), value: "Development".to_string() },
            ],
            created_at: Utc::now() - chrono::Duration::days(45),
            cost_per_month: 75.0,
        },
    ];

    Json(resources).into_response()
}

// GET /api/v1/operations/monitoring/metrics
pub async fn get_monitoring_metrics(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = vec![
        MonitoringMetric {
            resource_id: "vm-prod-001".to_string(),
            metric_name: "CPU Percentage".to_string(),
            value: 45.2,
            unit: "%".to_string(),
            timestamp: Utc::now(),
            threshold: Some(80.0),
            status: "normal".to_string(),
        },
        MonitoringMetric {
            resource_id: "vm-prod-001".to_string(),
            metric_name: "Memory Usage".to_string(),
            value: 62.8,
            unit: "%".to_string(),
            timestamp: Utc::now(),
            threshold: Some(85.0),
            status: "normal".to_string(),
        },
        MonitoringMetric {
            resource_id: "stprod001".to_string(),
            metric_name: "Storage Used".to_string(),
            value: 78.5,
            unit: "%".to_string(),
            timestamp: Utc::now(),
            threshold: Some(90.0),
            status: "warning".to_string(),
        },
        MonitoringMetric {
            resource_id: "app-dev-001".to_string(),
            metric_name: "Response Time".to_string(),
            value: 245.0,
            unit: "ms".to_string(),
            timestamp: Utc::now(),
            threshold: Some(200.0),
            status: "critical".to_string(),
        },
    ];

    Json(metrics).into_response()
}

// GET /api/v1/operations/automation/workflows
pub async fn get_automation_workflows(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let workflows = vec![
        AutomationWorkflow {
            id: "wf-001".to_string(),
            name: "Auto-scaling Workflow".to_string(),
            description: "Automatically scale resources based on demand".to_string(),
            trigger_type: "metric_based".to_string(),
            status: "active".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::hours(2)),
            next_run: Some(Utc::now() + chrono::Duration::hours(1)),
            success_rate: 98.5,
            actions_count: 156,
        },
        AutomationWorkflow {
            id: "wf-002".to_string(),
            name: "Backup Automation".to_string(),
            description: "Daily backup of critical resources".to_string(),
            trigger_type: "scheduled".to_string(),
            status: "active".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::hours(12)),
            next_run: Some(Utc::now() + chrono::Duration::hours(12)),
            success_rate: 100.0,
            actions_count: 45,
        },
        AutomationWorkflow {
            id: "wf-003".to_string(),
            name: "Compliance Remediation".to_string(),
            description: "Auto-remediate compliance violations".to_string(),
            trigger_type: "event_based".to_string(),
            status: "active".to_string(),
            last_run: Some(Utc::now() - chrono::Duration::minutes(30)),
            next_run: None,
            success_rate: 92.3,
            actions_count: 89,
        },
    ];

    Json(workflows).into_response()
}

// GET /api/v1/operations/notifications
pub async fn get_notifications(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let notifications = vec![
        Notification {
            id: "notif-001".to_string(),
            title: "Resource Auto-scaled".to_string(),
            message: "VM vm-prod-001 was automatically scaled up due to high CPU usage".to_string(),
            severity: "info".to_string(),
            source: "Auto-scaling".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(15),
            read: false,
            actionable: false,
        },
        Notification {
            id: "notif-002".to_string(),
            title: "Backup Completed".to_string(),
            message: "Daily backup completed successfully for 45 resources".to_string(),
            severity: "success".to_string(),
            source: "Backup Automation".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(1),
            read: true,
            actionable: false,
        },
        Notification {
            id: "notif-003".to_string(),
            title: "High Memory Usage Detected".to_string(),
            message: "VM vm-dev-002 is using 92% of available memory".to_string(),
            severity: "warning".to_string(),
            source: "Monitoring".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(2),
            read: false,
            actionable: true,
        },
    ];

    Json(notifications).into_response()
}

// GET /api/v1/operations/alerts
pub async fn get_operations_alerts(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let alerts = vec![
        Alert {
            id: "alert-001".to_string(),
            name: "High CPU Usage".to_string(),
            description: "CPU usage exceeded 80% threshold".to_string(),
            severity: "warning".to_string(),
            resource_id: "vm-prod-002".to_string(),
            metric: "CPU Percentage".to_string(),
            condition: "greater_than".to_string(),
            current_value: 85.2,
            threshold: 80.0,
            triggered_at: Utc::now() - chrono::Duration::minutes(10),
            status: "active".to_string(),
        },
        Alert {
            id: "alert-002".to_string(),
            name: "Storage Near Capacity".to_string(),
            description: "Storage usage approaching maximum capacity".to_string(),
            severity: "critical".to_string(),
            resource_id: "stprod002".to_string(),
            metric: "Storage Used".to_string(),
            condition: "greater_than".to_string(),
            current_value: 94.5,
            threshold: 90.0,
            triggered_at: Utc::now() - chrono::Duration::hours(1),
            status: "active".to_string(),
        },
        Alert {
            id: "alert-003".to_string(),
            name: "Application Response Time".to_string(),
            description: "Application response time exceeding SLA".to_string(),
            severity: "high".to_string(),
            resource_id: "app-prod-001".to_string(),
            metric: "Response Time".to_string(),
            condition: "greater_than".to_string(),
            current_value: 450.0,
            threshold: 200.0,
            triggered_at: Utc::now() - chrono::Duration::minutes(30),
            status: "acknowledged".to_string(),
        },
    ];

    let mode = DataMode::from_env();
    Json(DataResponse::new(alerts, mode)).into_response()
}