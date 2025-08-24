// Dashboard API handlers for comprehensive navigation system
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use tracing::{info, warn};
use anyhow::Result;

use crate::api::AppState;
// use crate::azure_integration::get_azure_service; // Temporarily commented out

// Mock function to replace azure service during compilation
async fn mock_azure_result<T>() -> Result<T> where T: Default {
    Err(anyhow::anyhow!("Azure service temporarily disabled"))
}

// Dashboard metrics response
#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub total_resources: u32,
    pub compliant_resources: u32,
    pub non_compliant_resources: u32,
    pub critical_alerts: u32,
    pub high_alerts: u32,
    pub medium_alerts: u32,
    pub low_alerts: u32,
    pub total_cost: f64,
    pub projected_cost: f64,
    pub cost_savings: f64,
    pub security_score: f64,
    pub compliance_rate: f64,
    pub ai_predictions_made: u64,
    pub automations_executed: u64,
    pub timestamp: DateTime<Utc>,
}

// Alert summary
#[derive(Debug, Serialize, Deserialize)]
pub struct AlertSummary {
    pub id: String,
    pub severity: String,
    pub category: String,
    pub title: String,
    pub description: String,
    pub resource_count: u32,
    pub created_at: DateTime<Utc>,
    pub status: String,
}

// Recent activity
#[derive(Debug, Serialize, Deserialize)]
pub struct Activity {
    pub id: String,
    pub activity_type: String,
    pub user: String,
    pub action: String,
    pub resource: String,
    pub result: String,
    pub timestamp: DateTime<Utc>,
}

// GET /api/v1/dashboard/metrics
pub async fn get_dashboard_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    info!("Fetching dashboard metrics from Azure");
    
    // Get Azure integration service (temporarily disabled)
    // // Azure service temporarily disabled - using mock data
    match mock_azure_result().await {
    /*
    // Azure service temporarily disabled - using mock data
    match mock_azure_result().await {
        Ok(azure) => {
            // Fetch real data from multiple Azure services in parallel
            let (monitor_health, compliance_summary, cost_summary, resource_stats) = tokio::join!(
                azure.monitor().get_system_health(),
                azure.governance().get_compliance_summary(),
                azure.cost().get_current_month_costs(),
                azure.resource_graph().get_resource_statistics()
            );

            // Build dashboard metrics from real Azure data
            let dashboard_metrics = DashboardMetrics {
                total_resources: resource_stats.as_ref().map_or(0, |s| s.total_resources as u32),
                compliant_resources: compliance_summary.as_ref().map_or(0, |c| c.compliant_resources as u32),
                non_compliant_resources: compliance_summary.as_ref().map_or(0, |c| c.non_compliant_resources as u32),
                critical_alerts: monitor_health.as_ref()
                    .ok()
                    .and_then(|h| h.alert_severity_distribution.get("0"))
                    .map(|&count| count as u32)
                    .unwrap_or(0),
                high_alerts: monitor_health.as_ref()
                    .ok()
                    .and_then(|h| h.alert_severity_distribution.get("1"))
                    .map(|&count| count as u32)
                    .unwrap_or(0),
                medium_alerts: monitor_health.as_ref()
                    .ok()
                    .and_then(|h| h.alert_severity_distribution.get("2"))
                    .map(|&count| count as u32)
                    .unwrap_or(0),
                low_alerts: monitor_health.as_ref()
                    .ok()
                    .and_then(|h| h.alert_severity_distribution.get("3"))
                    .map(|&count| count as u32)
                    .unwrap_or(0),
                total_cost: cost_summary.as_ref().map_or(0.0, |c| c.total_cost),
                projected_cost: cost_summary.as_ref().map_or(0.0, |c| c.total_cost * 1.1), // Simple projection
                cost_savings: cost_summary.as_ref().map_or(0.0, |c| c.total_cost * 0.05), // Estimated savings
                security_score: compliance_summary.as_ref().map_or(0.0, |c| c.security_score),
                compliance_rate: compliance_summary.as_ref().map_or(0.0, |c| c.compliance_percentage),
                ai_predictions_made: 15234, // Would need AI service integration
                automations_executed: 8921, // Would need automation service integration
                timestamp: Utc::now(),
            };
            
            Json(dashboard_metrics).into_response()
        }
    */
    
    // Use mock data (Azure integration temporarily disabled)
    warn!("Using mock dashboard data (Azure integration temporarily disabled)");
    Json(DashboardMetrics {
                total_resources: 2843,
                compliant_resources: 2456,
                non_compliant_resources: 387,
                critical_alerts: 3,
                high_alerts: 7,
                medium_alerts: 15,
                low_alerts: 42,
                total_cost: 145832.0,
                projected_cost: 138500.0,
                cost_savings: 7332.0,
                security_score: 87.5,
                compliance_rate: 86.4,
                ai_predictions_made: 15234,
                automations_executed: 8921,
                timestamp: Utc::now(),
            }).into_response()
}

// GET /api/v1/dashboard/alerts
pub async fn get_dashboard_alerts(_state: State<Arc<AppState>>) -> impl IntoResponse {
    info!("Fetching dashboard alerts from Azure");
    
    // Azure service temporarily disabled - returning mock alerts data
    let mock_alerts = vec![
        Ok(azure) => {
            // Get real alerts from Azure Monitor
            match azure.monitor().get_alert_incidents().await {
                Ok(incidents) => {
                    let alerts: Vec<AlertSummary> = incidents
                        .into_iter()
                        .take(20) // Limit to 20 most recent
                        .map(|incident| {
                            let severity = if incident.rule_name.contains("Critical") {
                                "critical"
                            } else if incident.rule_name.contains("High") {
                                "high"
                            } else if incident.rule_name.contains("Warning") {
                                "medium"
                            } else {
                                "low"
                            };
                            
                            AlertSummary {
                                id: incident.name,
                                severity: severity.to_string(),
                                category: "monitoring".to_string(),
                                title: incident.rule_name.clone(),
                                description: format!("Alert rule: {}", incident.rule_name),
                                resource_count: 1,
                                created_at: incident.activated_time,
                                status: if incident.is_active { "active" } else { "resolved" }.to_string(),
                            }
                        })
                        .collect();
                    
                    if !alerts.is_empty() {
                        return Json(alerts).into_response();
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch alerts from Azure: {}", e);
                }
            }
        }
        Err(e) => {
            warn!("Failed to get Azure service: {}", e);
        }
    }
    
    // Return mock data as fallback
    let alerts = vec![
        AlertSummary {
            id: "alert-001".to_string(),
            severity: "critical".to_string(),
            category: "security".to_string(),
            title: "Unencrypted Storage Accounts Detected".to_string(),
            description: "3 storage accounts found without encryption at rest enabled".to_string(),
            resource_count: 3,
            created_at: Utc::now() - chrono::Duration::hours(2),
            status: "active".to_string(),
        },
        AlertSummary {
            id: "alert-002".to_string(),
            severity: "high".to_string(),
            category: "compliance".to_string(),
            title: "Policy Violations Detected".to_string(),
            description: "12 resources violating required tagging policy".to_string(),
            resource_count: 12,
            created_at: Utc::now() - chrono::Duration::hours(5),
            status: "active".to_string(),
        },
        AlertSummary {
            id: "alert-003".to_string(),
            severity: "medium".to_string(),
            category: "cost".to_string(),
            title: "Idle Resources Detected".to_string(),
            description: "234 resources identified as idle for over 30 days".to_string(),
            resource_count: 234,
            created_at: Utc::now() - chrono::Duration::days(1),
            status: "acknowledged".to_string(),
        },
    ];

    Json(alerts).into_response()
}

// GET /api/v1/dashboard/activities
pub async fn get_dashboard_activities(_state: State<Arc<AppState>>) -> impl IntoResponse {
    info!("Fetching dashboard activities from Azure");
    
    // Azure service temporarily disabled - using mock data
    match mock_azure_result().await {
        Ok(azure) => {
            // Get real activities from Azure Activity Log
            match azure.activity().get_recent_activities().await {
                Ok(azure_activities) => {
                    let activities: Vec<Activity> = azure_activities
                        .into_iter()
                        .take(20) // Limit to 20 most recent
                        .map(|act| {
                            let activity_type = if act.operation.contains("write") {
                                "modification"
                            } else if act.operation.contains("delete") {
                                "deletion"
                            } else if act.operation.contains("read") {
                                "access"
                            } else {
                                "other"
                            };
                            
                            Activity {
                                id: act.id,
                                activity_type: activity_type.to_string(),
                                user: "Azure User".to_string(), // Would need additional API call for user details
                                action: act.operation,
                                resource: act.resource_id.unwrap_or_else(|| "N/A".to_string()),
                                result: act.status,
                                timestamp: act.timestamp,
                            }
                        })
                        .collect();
                    
                    if !activities.is_empty() {
                        return Json(activities).into_response();
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch activities from Azure: {}", e);
                }
            }
        }
        Err(e) => {
            warn!("Failed to get Azure service: {}", e);
        }
    }
    
    // Return mock data as fallback
    let activities = vec![
        Activity {
            id: "act-001".to_string(),
            activity_type: "remediation".to_string(),
            user: "admin@contoso.com".to_string(),
            action: "Applied missing tags".to_string(),
            resource: "vm-prod-001".to_string(),
            result: "success".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(15),
        },
        Activity {
            id: "act-002".to_string(),
            activity_type: "policy".to_string(),
            user: "policy-admin@contoso.com".to_string(),
            action: "Updated compliance policy".to_string(),
            resource: "require-encryption-policy".to_string(),
            result: "success".to_string(),
            timestamp: Utc::now() - chrono::Duration::minutes(30),
        },
        Activity {
            id: "act-003".to_string(),
            activity_type: "approval".to_string(),
            user: "manager@contoso.com".to_string(),
            action: "Approved exception request".to_string(),
            resource: "exception-req-042".to_string(),
            result: "approved".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(1),
        },
        Activity {
            id: "act-004".to_string(),
            activity_type: "ai".to_string(),
            user: "system".to_string(),
            action: "Auto-remediated compliance drift".to_string(),
            resource: "rg-dev-resources".to_string(),
            result: "success".to_string(),
            timestamp: Utc::now() - chrono::Duration::hours(2),
        },
    ];

    Json(activities).into_response()
}