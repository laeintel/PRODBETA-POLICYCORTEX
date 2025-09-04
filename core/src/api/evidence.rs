use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    Router,
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::evidence::{
    HashChain, Evidence as ChainEvidence, ComplianceStatus as ChainComplianceStatus,
    ChainVerificationResult, ChainStatus, MerkleProof,
};

/// Evidence collection request
#[derive(Debug, Deserialize, Serialize)]
pub struct CollectEvidenceRequest {
    pub event_type: String,
    pub resource_id: String,
    pub policy_id: String,
    pub policy_name: String,
    pub compliance_status: String,
    pub actor: String,
    pub subscription_id: String,
    pub resource_group: String,
    pub resource_type: String,
    pub details: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

/// Evidence collection response
#[derive(Debug, Serialize)]
pub struct CollectEvidenceResponse {
    pub success: bool,
    pub evidence_id: String,
    pub hash: String,
    pub timestamp: DateTime<Utc>,
}

/// Report generation request
#[derive(Debug, Deserialize)]
pub struct GenerateReportRequest {
    pub subscription_id: String,
    pub format: ReportFormat,
    pub include_qr_code: bool,
    pub digital_signature: bool,
    pub include_evidence: bool,
    pub evidence_limit: Option<usize>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ReportFormat {
    Pdf,
    Ssp,
    Poam,
}

/// Report generation response
#[derive(Debug, Serialize)]
pub struct GenerateReportResponse {
    pub success: bool,
    pub report_id: String,
    pub report_path: String,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Serialize)]
pub struct ReportMetadata {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub generated_by: String,
    pub report_type: ReportFormat,
    pub evidence_count: usize,
    pub chain_status: ChainVerificationResult,
    pub signature: Option<String>,
    pub public_key: Option<String>,
}

/// Query parameters for evidence retrieval
#[derive(Debug, Deserialize)]
pub struct EvidenceQuery {
    pub limit: Option<usize>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
}

/// Application state
pub struct AppState {
    pub hash_chain: Arc<HashChain>,
    pub evidence_store: Arc<RwLock<HashMap<String, ChainEvidence>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            hash_chain: Arc::new(HashChain::new()),
            evidence_store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Create evidence router
pub fn create_evidence_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/evidence/collect", post(collect_evidence))
        .route("/api/v1/evidence/verify/:hash", get(verify_evidence))
        .route("/api/v1/evidence/report", post(generate_report))
        .route("/api/v1/evidence/chain", get(get_chain_status))
        .route("/api/v1/evidence/:id", get(get_evidence_by_id))
        .route("/api/v1/evidence/block/:index", get(get_block))
        .route("/api/v1/evidence/proof/:hash", get(get_merkle_proof))
        .with_state(state)
}

/// Collect and store evidence
pub async fn collect_evidence(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CollectEvidenceRequest>,
) -> Result<Json<CollectEvidenceResponse>, StatusCode> {
    // Convert request to chain evidence
    let evidence = ChainEvidence {
        id: Uuid::new_v4().to_string(),
        timestamp: Utc::now(),
        event_type: request.event_type,
        resource_id: request.resource_id,
        policy_id: request.policy_id,
        compliance_status: match request.compliance_status.as_str() {
            "Compliant" => ChainComplianceStatus::Compliant,
            "NonCompliant" => ChainComplianceStatus::NonCompliant,
            "Warning" => ChainComplianceStatus::Warning,
            "Error" => ChainComplianceStatus::Error,
            _ => ChainComplianceStatus::Pending,
        },
        actor: request.actor,
        details: request.details,
        metadata: request.metadata,
        hash: String::new(),
    };

    // Add to hash chain
    let hash = state.hash_chain
        .add_evidence(evidence.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Store in evidence store
    let mut store = state.evidence_store.write().await;
    store.insert(evidence.id.clone(), evidence.clone());

    Ok(Json(CollectEvidenceResponse {
        success: true,
        evidence_id: evidence.id,
        hash,
        timestamp: evidence.timestamp,
    }))
}

/// Verify evidence chain
pub async fn verify_evidence(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Generate Merkle proof
    let proof = state.hash_chain
        .generate_merkle_proof(&hash)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;

    // Verify the proof
    let is_valid = HashChain::verify_merkle_proof(&proof);

    Ok(Json(serde_json::json!({
        "valid": is_valid,
        "hash": hash,
        "proof": proof,
        "timestamp": Utc::now()
    })))
}

/// Generate audit report
pub async fn generate_report(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateReportRequest>,
) -> Result<Json<GenerateReportResponse>, StatusCode> {
    // Get chain verification status
    let chain_status = state.hash_chain.verify_chain().await;

    // Get evidence for subscription
    let store = state.evidence_store.read().await;
    let evidence: Vec<_> = store
        .values()
        .filter(|e| {
            // Filter by subscription and date range
            if let Some(sub_id) = e.metadata.get("subscription_id") {
                if sub_id != &request.subscription_id {
                    return false;
                }
            }
            
            if let Some(start) = request.start_date {
                if e.timestamp < start {
                    return false;
                }
            }
            
            if let Some(end) = request.end_date {
                if e.timestamp > end {
                    return false;
                }
            }
            
            true
        })
        .take(request.evidence_limit.unwrap_or(1000))
        .cloned()
        .collect();

    let report_id = Uuid::new_v4().to_string();
    let report_path = format!("reports/{}_audit_report.{}", 
        report_id, 
        match request.format {
            ReportFormat::Pdf => "pdf",
            ReportFormat::Ssp => "json",
            ReportFormat::Poam => "json",
        }
    );

    let metadata = ReportMetadata {
        report_id: report_id.clone(),
        generated_at: Utc::now(),
        generated_by: "PolicyCortex PROVE System".to_string(),
        report_type: request.format,
        evidence_count: evidence.len(),
        chain_status,
        signature: if request.digital_signature {
            Some("digital_signature_placeholder".to_string())
        } else {
            None
        },
        public_key: if request.digital_signature {
            Some("public_key_placeholder".to_string())
        } else {
            None
        },
    };

    Ok(Json(GenerateReportResponse {
        success: true,
        report_id,
        report_path,
        metadata,
    }))
}

/// Get chain status
pub async fn get_chain_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ChainStatus>, StatusCode> {
    let status = state.hash_chain.get_status().await;
    Ok(Json(status))
}

/// Get evidence by ID
pub async fn get_evidence_by_id(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ChainEvidence>, StatusCode> {
    let store = state.evidence_store.read().await;
    
    store
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// Get block by index
pub async fn get_block(
    State(state): State<Arc<AppState>>,
    Path(index): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.hash_chain
        .get_block(index)
        .await
        .map(|block| Json(serde_json::to_value(block).unwrap()))
        .ok_or(StatusCode::NOT_FOUND)
}

/// Get Merkle proof for evidence
pub async fn get_merkle_proof(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> Result<Json<MerkleProof>, StatusCode> {
    state.hash_chain
        .generate_merkle_proof(&hash)
        .await
        .map(Json)
        .map_err(|_| StatusCode::NOT_FOUND)
}

/// Export chain to JSON
async fn export_chain(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let chain_json = state.hash_chain
        .export_chain()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let value: serde_json::Value = serde_json::from_str(&chain_json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(value))
}

/// Health check endpoint
async fn health() -> &'static str {
    "Evidence chain service is healthy"
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, Method};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_collect_evidence() {
        let state = Arc::new(AppState::new());
        let app = create_evidence_router(state);

        let request_body = serde_json::json!({
            "event_type": "PolicyCheck",
            "resource_id": "vm-001",
            "policy_id": "policy-001",
            "policy_name": "VM Compliance Policy",
            "compliance_status": "Compliant",
            "actor": "system",
            "subscription_id": "sub-001",
            "resource_group": "rg-001",
            "resource_type": "Microsoft.Compute/virtualMachines",
            "details": {},
            "metadata": {}
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/evidence/collect")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_chain_status() {
        let state = Arc::new(AppState::new());
        let app = create_evidence_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/v1/evidence/chain")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}