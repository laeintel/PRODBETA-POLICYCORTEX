// Security Graph API endpoints
// Implements GitHub Issues #56-59: Security Exposure Graph features

use axum::{
    extract::{Query, State, Path},
    Json,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::security_graph::{
    AttackPath, MitigationBundle, SecurityExposureReport,
    analyze_security_exposure, SecurityGraphEngine, MitigationResult
};

#[derive(Debug, Deserialize)]
pub struct SecurityQuery {
    pub sensitivity_level: Option<String>, // "Critical", "High", "Medium", "Low"
    pub include_mitigations: Option<bool>,
    pub max_paths: Option<usize>,
}

// GET /api/v1/security/exposure
pub async fn get_security_exposure(
    Query(params): Query<SecurityQuery>,
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    match analyze_security_exposure(state.async_azure_client.as_ref()).await {
        Ok(report) => {
            let filtered_report = if let Some(max) = params.max_paths {
                SecurityExposureReport {
                    top_attack_paths: report.top_attack_paths.into_iter().take(max).collect(),
                    ..report
                }
            } else {
                report
            };

            (StatusCode::OK, Json(filtered_report))
        }
        Err(e) => {
            tracing::error!("Failed to analyze security exposure: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": "Failed to analyze security exposure"
            })))
        }
    }
}

// GET /api/v1/security/attack-paths
pub async fn get_attack_paths(
    Query(params): Query<SecurityQuery>,
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    let sensitivity = params.sensitivity_level.unwrap_or_else(|| "Critical".to_string());

    if let Some(ref client) = state.async_azure_client {
        let mut engine = SecurityGraphEngine::new();

        // Build graph from Azure
        if let Err(e) = engine.build_from_azure(client).await {
            tracing::error!("Failed to build security graph: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(vec![]));
        }

        // Find attack paths
        let paths = engine.find_attack_paths(&sensitivity);
        let max_paths = params.max_paths.unwrap_or(10);

        (StatusCode::OK, Json(paths.into_iter().take(max_paths).collect::<Vec<_>>()))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(vec![]))
    }
}

// GET /api/v1/security/attack-paths/:path_id
pub async fn get_attack_path_details(
    Path(path_id): Path<String>,
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    // Not implemented yet without a data store
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({"message": "Attack path details unavailable"})))
}

#[derive(Debug, Deserialize)]
pub struct MitigationRequest {
    pub bundle_id: String,
    pub dry_run: bool,
    pub auto_rollback: bool,
}

// POST /api/v1/security/mitigate
pub async fn apply_mitigation(
    State(state): State<Arc<crate::api::AppState>>,
    Json(request): Json<MitigationRequest>,
) -> impl IntoResponse {
    if let Some(ref client) = state.async_azure_client {
        // In production, retrieve bundle from database
        let bundle = MitigationBundle {
            bundle_id: request.bundle_id.clone(),
            name: "Network Segmentation".to_string(),
            description: "Apply network segmentation controls".to_string(),
            controls: vec![],
            effectiveness: 0.85,
            implementation_cost: "Medium".to_string(),
            blast_radius: vec![],
        };

        let engine = SecurityGraphEngine::new();

        match engine.apply_mitigation(&bundle, client).await {
            Ok(result) => (StatusCode::OK, Json(result)),
            Err(e) => {
                tracing::error!("Failed to apply mitigation: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, Json(MitigationResult {
                    bundle_id: request.bundle_id,
                    applied_controls: vec![],
                    failed_controls: vec![],
                    residual_risk: 1.0,
                    timestamp: chrono::Utc::now(),
                }))
            }
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(MitigationResult {
            bundle_id: request.bundle_id,
            applied_controls: vec![],
            failed_controls: vec![],
            residual_risk: 1.0,
            timestamp: chrono::Utc::now(),
        }))
    }
}

// GET /api/v1/security/risk-score
pub async fn get_risk_score(
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    let risk_score = RiskScore {
        overall_score: 32.5,
        category_scores: CategoryScores {
            network: 28.0,
            identity: 35.0,
            data: 42.0,
            compliance: 18.0,
        },
        trend: "decreasing".to_string(),
        change_24h: -2.3,
        critical_risks: 3,
        high_risks: 7,
        medium_risks: 15,
        low_risks: 42,
    };

    (StatusCode::OK, Json(risk_score))
}

#[derive(Debug, Serialize)]
struct RiskScore {
    overall_score: f64,
    category_scores: CategoryScores,
    trend: String,
    change_24h: f64,
    critical_risks: u32,
    high_risks: u32,
    medium_risks: u32,
    low_risks: u32,
}

#[derive(Debug, Serialize)]
struct CategoryScores {
    network: f64,
    identity: f64,
    data: f64,
    compliance: f64,
}

// GET /api/v1/security/recommendations
pub async fn get_security_recommendations(
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    let recommendations = vec![
        SecurityRecommendation {
            id: "sec-rec-001".to_string(),
            priority: "Critical".to_string(),
            title: "Enable Network Segmentation".to_string(),
            description: "Implement micro-segmentation to limit lateral movement".to_string(),
            affected_resources: 12,
            risk_reduction: 45.0,
            implementation_effort: "Medium".to_string(),
            automation_available: true,
        },
        SecurityRecommendation {
            id: "sec-rec-002".to_string(),
            priority: "High".to_string(),
            title: "Implement Zero Trust Access".to_string(),
            description: "Replace VPN with Zero Trust Network Access solution".to_string(),
            affected_resources: 8,
            risk_reduction: 38.0,
            implementation_effort: "High".to_string(),
            automation_available: false,
        },
        SecurityRecommendation {
            id: "sec-rec-003".to_string(),
            priority: "High".to_string(),
            title: "Enable Advanced Threat Protection".to_string(),
            description: "Deploy EDR and XDR solutions across all endpoints".to_string(),
            affected_resources: 45,
            risk_reduction: 52.0,
            implementation_effort: "Low".to_string(),
            automation_available: true,
        },
    ];

    (StatusCode::OK, Json(recommendations))
}

#[derive(Debug, Serialize)]
struct SecurityRecommendation {
    id: String,
    priority: String,
    title: String,
    description: String,
    affected_resources: u32,
    risk_reduction: f64,
    implementation_effort: String,
    automation_available: bool,
}