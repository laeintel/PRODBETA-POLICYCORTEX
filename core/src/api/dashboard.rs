// Dashboard API handlers for comprehensive navigation system
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};

use crate::api::AppState;

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
    // Try to get real metrics from Azure if available
    if let Some(ref async_client) = state.async_azure_client {
        if let Ok(metrics) = async_client.get_governance_metrics().await {
            let dashboard_metrics = DashboardMetrics {
                total_resources: metrics.resources.total,
                compliant_resources: metrics.resources.optimized,
                non_compliant_resources: metrics.policies.violations,
                critical_alerts: 2,
                high_alerts: 5,
                medium_alerts: 12,
                low_alerts: 28,
                total_cost: metrics.costs.current_spend,
                projected_cost: metrics.costs.predicted_spend,
                cost_savings: metrics.costs.savings_identified,
                security_score: 100.0 - metrics.rbac.risk_score,
                compliance_rate: metrics.policies.compliance_rate,
                ai_predictions_made: metrics.ai.predictions_made,
                automations_executed: metrics.ai.automations_executed,
                timestamp: Utc::now(),
            };
            return Json(dashboard_metrics).into_response();
        }
    }

    // Return mock data for development
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