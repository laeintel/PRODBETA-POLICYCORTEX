use crate::ml::{
    predictive_compliance::PredictiveComplianceEngine,
    risk_scoring::{RiskScoringEngine, RiskSummary},
    pattern_analysis::PatternAnalyzer,
    drift_detector::DriftDetector,
    ViolationPrediction, RiskLevel,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Clone)]
pub struct PredictionState {
    pub compliance_engine: Arc<RwLock<PredictiveComplianceEngine>>,
    pub risk_engine: Arc<RiskScoringEngine>,
    pub pattern_analyzer: Arc<RwLock<PatternAnalyzer>>,
    pub drift_detector: Arc<RwLock<DriftDetector>>,
}

#[derive(Debug, Deserialize)]
pub struct PredictionQuery {
    pub lookahead_hours: Option<i64>,
    pub risk_threshold: Option<String>,
    pub include_patterns: Option<bool>,
    pub include_drift: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub predictions: Vec<ViolationPrediction>,
    pub risk_summary: RiskSummary,
    pub patterns: Option<Vec<PatternInfo>>,
    pub drift_analysis: Option<DriftInfo>,
    pub metadata: PredictionMetadata,
}

#[derive(Debug, Serialize)]
pub struct PatternInfo {
    pub pattern_id: String,
    pub pattern_type: String,
    pub confidence: f64,
    pub estimated_time_to_violation: i64,
    pub recommendation: String,
}

#[derive(Debug, Serialize)]
pub struct DriftInfo {
    pub total_drift_score: f64,
    pub drift_velocity: f64,
    pub time_to_violation: Option<i64>,
    pub critical_drifts: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PredictionMetadata {
    pub generated_at: DateTime<Utc>,
    pub lookahead_hours: i64,
    pub total_predictions: usize,
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
}

// GET /api/v1/predictions/violations
pub async fn get_violation_predictions(
    State(state): State<Arc<crate::api::AppState>>,
    Query(query): Query<PredictionQuery>,
) -> impl IntoResponse {
    let lookahead_hours = query.lookahead_hours.unwrap_or(24);
    
    // For demo/MVP, return realistic mock predictions
    let predictions = generate_demo_predictions(lookahead_hours);
    
    let risk_engine = RiskScoringEngine::new();
    let risk_summary = risk_engine.get_risk_summary(&predictions);
    
    let metadata = PredictionMetadata {
        generated_at: Utc::now(),
        lookahead_hours,
        total_predictions: predictions.len(),
        critical_count: predictions.iter().filter(|p| matches!(p.risk_level, RiskLevel::Critical)).count(),
        high_count: predictions.iter().filter(|p| matches!(p.risk_level, RiskLevel::High)).count(),
        medium_count: predictions.iter().filter(|p| matches!(p.risk_level, RiskLevel::Medium)).count(),
        low_count: predictions.iter().filter(|p| matches!(p.risk_level, RiskLevel::Low)).count(),
    };
    
    let response = PredictionResponse {
        predictions,
        risk_summary,
        patterns: if query.include_patterns.unwrap_or(false) {
            Some(generate_pattern_info())
        } else {
            None
        },
        drift_analysis: if query.include_drift.unwrap_or(false) {
            Some(generate_drift_info())
        } else {
            None
        },
        metadata,
    };
    
    Json(response)
}

// GET /api/v1/predictions/violations/{resource_id}
pub async fn get_resource_predictions(
    State(state): State<Arc<crate::api::AppState>>,
    Path(resource_id): Path<String>,
    Query(query): Query<PredictionQuery>,
) -> impl IntoResponse {
    let lookahead_hours = query.lookahead_hours.unwrap_or(24);
    
    let predictions = generate_demo_predictions(lookahead_hours)
        .into_iter()
        .filter(|p| p.resource_id.contains(&resource_id))
        .collect::<Vec<_>>();
    
    if predictions.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "No predictions found for resource"
            }))
        ).into_response();
    }
    
    Json(predictions).into_response()
}

// GET /api/v1/predictions/risk-score/{resource_id}
pub async fn get_risk_score(
    State(state): State<Arc<crate::api::AppState>>,
    Path(resource_id): Path<String>,
) -> impl IntoResponse {
    let predictions = generate_demo_predictions(24)
        .into_iter()
        .filter(|p| p.resource_id.contains(&resource_id))
        .collect::<Vec<_>>();
    
    let risk_engine = RiskScoringEngine::new();
    let risk_summary = risk_engine.get_risk_summary(&predictions);
    
    Json(serde_json::json!({
        "resource_id": resource_id,
        "risk_score": risk_summary.risk_score,
        "risk_level": if risk_summary.risk_score > 0.8 { "Critical" }
                     else if risk_summary.risk_score > 0.6 { "High" }
                     else if risk_summary.risk_score > 0.4 { "Medium" }
                     else { "Low" },
        "critical_risks": risk_summary.critical_risks,
        "high_risks": risk_summary.high_risks,
        "total_financial_impact": risk_summary.total_financial_impact,
        "recommendations": generate_risk_recommendations(risk_summary.risk_score),
    }))
}

// POST /api/v1/predictions/remediate/{prediction_id}
pub async fn remediate_prediction(
    State(state): State<Arc<crate::api::AppState>>,
    Path(prediction_id): Path<String>,
    Json(payload): Json<RemediationRequest>,
) -> impl IntoResponse {
    // In production, this would trigger actual remediation
    let response = RemediationResponse {
        prediction_id,
        status: "initiated".to_string(),
        workflow_id: Uuid::new_v4().to_string(),
        estimated_completion: Utc::now() + chrono::Duration::minutes(5),
        actions: vec![
            "Analyzing violation prediction".to_string(),
            "Generating ARM template".to_string(),
            "Applying configuration changes".to_string(),
            "Validating compliance".to_string(),
        ],
    };
    
    (StatusCode::ACCEPTED, Json(response))
}

#[derive(Debug, Deserialize)]
pub struct RemediationRequest {
    pub auto_approve: bool,
    pub notification_email: Option<String>,
    pub schedule: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RemediationResponse {
    pub prediction_id: String,
    pub status: String,
    pub workflow_id: String,
    pub estimated_completion: DateTime<Utc>,
    pub actions: Vec<String>,
}

// Helper functions for demo data
fn generate_demo_predictions(lookahead_hours: i64) -> Vec<ViolationPrediction> {
    vec![
        ViolationPrediction {
            id: Uuid::new_v4(),
            resource_id: "/subscriptions/xxx/resourceGroups/prod/providers/Microsoft.Storage/storageAccounts/proddata".to_string(),
            resource_type: "Microsoft.Storage/storageAccounts".to_string(),
            policy_id: "pol-encryption-001".to_string(),
            policy_name: "Require Storage Encryption".to_string(),
            prediction_time: Utc::now(),
            violation_time: Utc::now() + chrono::Duration::hours(18),
            confidence_score: 0.92,
            risk_level: RiskLevel::Critical,
            business_impact: crate::ml::BusinessImpact {
                financial_impact: 75000.0,
                compliance_impact: "SOC2 Type II violation - Critical".to_string(),
                operational_impact: "No immediate operational impact".to_string(),
                security_impact: "Critical - Unencrypted sensitive data exposure".to_string(),
                affected_resources: vec!["proddata".to_string()],
            },
            remediation_suggestions: vec![
                crate::ml::RemediationSuggestion {
                    action: "Enable Encryption at Rest".to_string(),
                    description: "Enable storage service encryption for data at rest".to_string(),
                    automated: true,
                    arm_template: Some(serde_json::json!({
                        "properties": {
                            "encryption": {
                                "services": {
                                    "blob": { "enabled": true },
                                    "file": { "enabled": true }
                                }
                            }
                        }
                    }).to_string()),
                    estimated_time: "5 minutes".to_string(),
                    success_probability: 0.98,
                },
            ],
            drift_indicators: vec![
                crate::ml::DriftIndicator {
                    property: "encryption.status".to_string(),
                    current_value: "Disabled".to_string(),
                    expected_value: "Enabled".to_string(),
                    drift_rate: 0.15,
                    time_to_violation: 18,
                },
            ],
        },
        ViolationPrediction {
            id: Uuid::new_v4(),
            resource_id: "/subscriptions/xxx/resourceGroups/prod/providers/Microsoft.Network/publicIPAddresses/web-ip".to_string(),
            resource_type: "Microsoft.Network/publicIPAddresses".to_string(),
            policy_id: "pol-network-002".to_string(),
            policy_name: "Restrict Public IP Allocation".to_string(),
            prediction_time: Utc::now(),
            violation_time: Utc::now() + chrono::Duration::hours(12),
            confidence_score: 0.78,
            risk_level: RiskLevel::High,
            business_impact: crate::ml::BusinessImpact {
                financial_impact: 25000.0,
                compliance_impact: "Network security policy violation".to_string(),
                operational_impact: "Potential service disruption during remediation".to_string(),
                security_impact: "High - Unauthorized public endpoint exposure".to_string(),
                affected_resources: vec!["web-ip".to_string()],
            },
            remediation_suggestions: vec![
                crate::ml::RemediationSuggestion {
                    action: "Remove Public IP".to_string(),
                    description: "Delete public IP and configure private endpoint".to_string(),
                    automated: true,
                    arm_template: None,
                    estimated_time: "15 minutes".to_string(),
                    success_probability: 0.85,
                },
            ],
            drift_indicators: vec![],
        },
    ]
}

fn generate_pattern_info() -> Vec<PatternInfo> {
    vec![
        PatternInfo {
            pattern_id: "drift_001".to_string(),
            pattern_type: "ConfigurationDrift".to_string(),
            confidence: 0.87,
            estimated_time_to_violation: 24 * 3600,
            recommendation: "Lock down configuration settings to prevent drift".to_string(),
        },
        PatternInfo {
            pattern_id: "periodic_001".to_string(),
            pattern_type: "PeriodicViolation".to_string(),
            confidence: 0.92,
            estimated_time_to_violation: 7 * 24 * 3600,
            recommendation: "Set up automated certificate renewal".to_string(),
        },
    ]
}

fn generate_drift_info() -> DriftInfo {
    DriftInfo {
        total_drift_score: 0.72,
        drift_velocity: 0.03,
        time_to_violation: Some(36),
        critical_drifts: vec![
            "encryption.status drifting from Enabled to Disabled".to_string(),
            "publicNetworkAccess changing from Disabled to Enabled".to_string(),
        ],
        recommendations: vec![
            "Review and lock critical security configurations".to_string(),
            "Implement policy enforcement to prevent unauthorized changes".to_string(),
        ],
    }
}

fn generate_risk_recommendations(risk_score: f64) -> Vec<String> {
    if risk_score > 0.8 {
        vec![
            "URGENT: Critical risk detected. Immediate action required.".to_string(),
            "Enable automated remediation for critical violations".to_string(),
            "Review and strengthen Azure Policy assignments".to_string(),
        ]
    } else if risk_score > 0.6 {
        vec![
            "High risk detected. Review within 24 hours.".to_string(),
            "Consider enabling preventive policies".to_string(),
        ]
    } else {
        vec![
            "Monitor for changes in risk profile".to_string(),
            "Schedule regular compliance reviews".to_string(),
        ]
    }
}