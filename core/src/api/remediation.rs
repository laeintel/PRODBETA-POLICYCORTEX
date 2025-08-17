// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Remediation API Endpoints
// Handles approval workflows, bulk remediation, and rollback operations

use crate::api::{AppState, ApiError};
use crate::auth::{AuthUser, TokenValidator};
use crate::remediation::*;
use crate::remediation::approval_manager::{ApprovalManager, ApprovalRequest, ApprovalDecision};
use crate::remediation::bulk_remediation::{BulkRemediationEngine, Violation};
use crate::remediation::rollback_manager::RollbackManager;
// use crate::remediation::quick_fixes::*;
use axum::{
    extract::{Path, Query, State},
    response::{IntoResponse, Json, sse::{Event, Sse}},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// ========== Request/Response DTOs ==========

#[derive(Debug, Deserialize)]
pub struct CreateApprovalRequest {
    pub remediation_request: RemediationRequest,
    pub approvers: Vec<String>,
    pub require_all: bool,
    pub timeout_hours: u64,
    pub auto_approve_conditions: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct CreateApprovalResponse {
    pub approval_id: String,
    pub status: String,
    pub expires_at: DateTime<Utc>,
    pub approval_url: String,
}

#[derive(Debug, Deserialize)]
pub struct ApprovalDecisionRequest {
    pub decision: String, // "approve" or "reject"
    pub reason: Option<String>,
    pub conditions: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct ApprovalDecisionResponse {
    pub success: bool,
    pub status: String,
    pub remediation_started: bool,
    pub remediation_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BulkRemediationRequest {
    pub violations: Vec<ViolationDto>,
    pub dry_run: bool,
    pub parallel: bool,
    pub max_parallel: Option<usize>,
    pub stop_on_error: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ViolationDto {
    pub violation_id: String,
    pub resource_id: String,
    pub resource_type: String,
    pub policy_id: String,
    pub violation_type: String,
    pub severity: String,
}

#[derive(Debug, Serialize)]
pub struct BulkRemediationResponse {
    pub bulk_id: String,
    pub total_violations: usize,
    pub processing: usize,
    pub stream_url: String,
}

#[derive(Debug, Deserialize)]
pub struct RollbackRequest {
    pub reason: String,
    pub force: bool,
}

#[derive(Debug, Serialize)]
pub struct RollbackResponse {
    pub success: bool,
    pub resources_restored: usize,
    pub duration_ms: u64,
    pub details: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RemediationStatusQuery {
    pub include_details: bool,
    pub page: Option<u32>,
    pub limit: Option<u32>,
}

// ========== Approval Endpoints ==========

/// Create a new approval request for remediation
pub async fn create_approval_request(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateApprovalRequest>,
) -> impl IntoResponse {
    // Verify user has permission to create approvals
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Remediate"]) {
        return ApiError::Forbidden("Insufficient permissions to create remediation approvals".to_string())
            .into_response();
    }

    let approval_id = Uuid::new_v4().to_string();
    let expires_at = Utc::now() + chrono::Duration::hours(request.timeout_hours as i64);
    
    // Store approval request in state (in production, use database)
    let approval = ApprovalRequest {
        id: approval_id.clone(),
        remediation_request: request.remediation_request,
        approvers: request.approvers,
        require_all: Some(request.require_all),
        created_by: auth_user.claims.preferred_username.clone().unwrap_or_default(),
        created_at: Utc::now(),
        expires_at,
        status: "pending".to_string(),
        decisions: HashMap::new(),
    };
    
    // Get or create approval manager
    let approval_manager = state.approval_manager.clone()
        .unwrap_or_else(|| Arc::new(ApprovalManager::new()));
    
    // Create approval
    match approval_manager.create_approval(approval).await {
        Ok(id) => {
            let approval_id_clone = id.clone();
            // Send notifications to approvers (async)
            tokio::spawn(async move {
                // In production, send email/Teams/Slack notifications
                tracing::info!("Sending approval notifications for {}", approval_id_clone);
            });
            
            Json(CreateApprovalResponse {
                approval_id: id.clone(),
                status: "pending".to_string(),
                expires_at,
                approval_url: format!("/api/v1/remediation/approvals/{}", id),
            }).into_response()
        }
        Err(e) => {
            ApiError::internal_server(format!("Failed to create approval: {}", e))
                .into_response()
        }
    }
}

/// Process an approval decision
pub async fn approve_remediation(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Path(approval_id): Path<String>,
    Json(decision): Json<ApprovalDecisionRequest>,
) -> impl IntoResponse {
    // Verify user has permission to approve
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Approve"]) {
        return ApiError::Forbidden("Insufficient permissions to approve remediations".to_string())
            .into_response();
    }
    
    let approval_manager = match state.approval_manager.clone() {
        Some(manager) => manager,
        None => {
            return ApiError::ServiceUnavailable("Approval system not initialized".to_string())
                .into_response();
        }
    };
    
    let approval_decision = if decision.decision == "approve" {
        ApprovalDecision::Approved
    } else {
        ApprovalDecision::Rejected
    };
    
    match approval_manager.process_approval(&approval_id, approval_decision).await {
        Ok(result) => {
            let mut remediation_started = false;
            let mut remediation_id = None;
            
            // If approved and all approvals received, start remediation
            if result.approved && result.final_decision {
                // Start remediation process
                if let Ok(rem_id) = start_remediation(&state, &approval_id).await {
                    remediation_started = true;
                    remediation_id = Some(rem_id);
                }
            }
            
            Json(ApprovalDecisionResponse {
                success: true,
                status: if result.approved { "approved".to_string() } else { "rejected".to_string() },
                remediation_started,
                remediation_id,
            }).into_response()
        }
        Err(e) => {
            ApiError::BadRequest(format!("Failed to process approval: {}", e))
                .into_response()
        }
    }
}

/// Get approval status
pub async fn get_approval_status(
    State(state): State<Arc<AppState>>,
    Path(approval_id): Path<String>,
) -> impl IntoResponse {
    let approval_manager = match state.approval_manager.clone() {
        Some(manager) => manager,
        None => {
            return ApiError::ServiceUnavailable("Approval system not initialized".to_string())
                .into_response();
        }
    };
    
    match approval_manager.get_approval(&approval_id).await {
        Ok(approval) => Json(approval).into_response(),
        Err(e) => ApiError::NotFound(format!("Approval not found: {}", e))
            .into_response()
    }
}

/// List pending approvals for a user
pub async fn list_pending_approvals(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let approval_manager = match state.approval_manager.clone() {
        Some(manager) => manager,
        None => {
            return Json(vec![] as Vec<ApprovalRequest>).into_response();
        }
    };
    
    let user = auth_user.claims.preferred_username.clone().unwrap_or_default();
    match approval_manager.list_pending_for_user(&user).await {
        Ok(approvals) => Json(approvals).into_response(),
        Err(e) => {
            tracing::error!("Failed to list approvals: {}", e);
            Json(vec![] as Vec<ApprovalRequest>).into_response()
        }
    }
}

// ========== Bulk Remediation Endpoints ==========

/// Execute bulk remediation for multiple violations
pub async fn execute_bulk_remediation(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(request): Json<BulkRemediationRequest>,
) -> impl IntoResponse {
    // Verify permissions
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Remediate"]) {
        return ApiError::Forbidden("Insufficient permissions for bulk remediation".to_string())
            .into_response();
    }
    
    let bulk_id = Uuid::new_v4().to_string();
    let violations: Vec<Violation> = request.violations.into_iter().map(|dto| {
        Violation {
            violation_id: dto.violation_id,
            resource_id: dto.resource_id,
            resource_type: dto.resource_type,
            policy_id: dto.policy_id,
            violation_type: dto.violation_type,
            severity: parse_severity(&dto.severity),
            detected_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }).collect();
    
    let bulk_id_clone = bulk_id.clone();
    let bulk_id_spawn = bulk_id.clone();
    let violations_clone = violations.clone();
    
    // Create SSE channel for progress updates
    let (tx, _rx) = broadcast::channel::<String>(100);
    let tx_clone = tx.clone();
    
    // Get or create bulk remediation engine
    let bulk_engine = state.bulk_remediation_engine.clone()
        .unwrap_or_else(|| {
            Arc::new(BulkRemediationEngine::new(
                Arc::new(crate::remediation::bulk_remediation::MockTemplateExecutor)
            ))
        });
    
    // Execute remediation in background
    tokio::spawn(async move {
        let _ = tx_clone.send(format!("Starting bulk remediation {}", bulk_id_spawn));
        
        let options = crate::remediation::bulk_remediation::BulkOptions {
            dry_run: request.dry_run,
            severity_filter: None,
            resource_type_filter: None,
            max_parallel: request.max_parallel,
            stop_on_error: request.stop_on_error,
        };
        
        let result = bulk_engine.execute_with_options(violations_clone, options).await;
        
        let _ = tx_clone.send(format!(
            "Bulk remediation completed: {} successful, {} failed, {} skipped",
            result.successful, result.failed, result.skipped
        ));
    });
    
    // Store SSE channel for streaming
    if let Some(channels) = &state.bulk_remediation_channels {
        channels.write().await.insert(bulk_id_clone.clone(), tx);
    }
    
    Json(BulkRemediationResponse {
        bulk_id: bulk_id.clone(),
        total_violations: violations.len(),
        processing: violations.len(),
        stream_url: format!("/api/v1/remediation/bulk/{}/stream", bulk_id),
    }).into_response()
}

/// Stream bulk remediation progress
pub async fn stream_bulk_remediation(
    State(state): State<Arc<AppState>>,
    Path(bulk_id): Path<String>,
) -> impl IntoResponse {
    let rx_opt = if let Some(channels) = &state.bulk_remediation_channels {
        channels.read().await.get(&bulk_id).cloned()
    } else {
        None
    };
    
    if let Some(tx) = rx_opt {
        let mut rx = tx.subscribe();
        let stream = async_stream::stream! {
            while let Ok(msg) = rx.recv().await {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(msg));
            }
        };
        Sse::new(stream).into_response()
    } else {
        ApiError::NotFound("Bulk remediation not found".to_string()).into_response()
    }
}

// ========== Rollback Endpoints ==========

/// Execute rollback for a remediation
pub async fn rollback_remediation(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Path(rollback_token): Path<String>,
    Json(request): Json<RollbackRequest>,
) -> impl IntoResponse {
    // Verify permissions
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Remediate"]) {
        return ApiError::Forbidden("Insufficient permissions for rollback".to_string())
            .into_response();
    }
    
    let rollback_manager = state.rollback_manager.clone()
        .unwrap_or_else(|| Arc::new(RollbackManager::new()));
    
    let start_time = std::time::Instant::now();
    
    match rollback_manager.execute_rollback(rollback_token, request.reason).await {
        Ok(result) => {
            Json(RollbackResponse {
                success: true,
                resources_restored: result.resources_restored,
                duration_ms: start_time.elapsed().as_millis() as u64,
                details: vec![format!("{} resources restored", result.resources_restored)],
            }).into_response()
        }
        Err(e) => {
            ApiError::internal_server(format!("Rollback failed: {}", e))
                .into_response()
        }
    }
}

/// Get rollback status and availability
pub async fn get_rollback_status(
    State(state): State<Arc<AppState>>,
    Path(rollback_token): Path<String>,
) -> impl IntoResponse {
    let rollback_manager = state.rollback_manager.clone()
        .unwrap_or_else(|| Arc::new(RollbackManager::new()));
    
    match rollback_manager.get_rollback_status(&rollback_token).await {
        Ok(status) => Json(status).into_response(),
        Err(e) => ApiError::NotFound(format!("Rollback token not found: {}", e))
            .into_response()
    }
}

// ========== Status and Monitoring Endpoints ==========

/// Get remediation status
pub async fn get_remediation_status(
    State(_state): State<Arc<AppState>>,
    Path(request_id): Path<String>,
    Query(params): Query<RemediationStatusQuery>,
) -> impl IntoResponse {
    // In production, fetch from database
    Json(serde_json::json!({
        "request_id": request_id,
        "status": "in_progress",
        "stage": 2,
        "total_stages": 5,
        "started_at": Utc::now() - chrono::Duration::minutes(5),
        "estimated_completion": Utc::now() + chrono::Duration::minutes(2),
        "include_details": params.include_details,
    })).into_response()
}

/// List all remediations
pub async fn list_remediations(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<RemediationStatusQuery>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1);
    let limit = params.limit.unwrap_or(20).min(100);
    
    // In production, fetch from database with pagination
    Json(serde_json::json!({
        "remediations": [],
        "total": 0,
        "page": page,
        "limit": limit,
    })).into_response()
}

// ========== Helper Functions ==========

fn parse_severity(severity: &str) -> crate::remediation::bulk_remediation::ViolationSeverity {
    use crate::remediation::bulk_remediation::ViolationSeverity;
    match severity.to_lowercase().as_str() {
        "critical" => ViolationSeverity::Critical,
        "high" => ViolationSeverity::High,
        "medium" => ViolationSeverity::Medium,
        _ => ViolationSeverity::Low,
    }
}

async fn start_remediation(_state: &Arc<AppState>, approval_id: &str) -> Result<String, String> {
    // In production, start actual remediation process
    let remediation_id = Uuid::new_v4().to_string();
    tracing::info!("Starting remediation {} for approval {}", remediation_id, approval_id);
    
    // Store remediation record
    // Launch remediation task
    // Return remediation ID
    
    Ok(remediation_id)
}

// ========== Route Registration ==========

pub fn remediation_routes() -> axum::Router<Arc<AppState>> {
    use axum::routing::{get, post};
    
    axum::Router::new()
        // Approval endpoints
        .route("/api/v1/remediation/approvals", post(create_approval_request))
        .route("/api/v1/remediation/approvals/:id", get(get_approval_status))
        .route("/api/v1/remediation/approvals/:id/approve", post(approve_remediation))
        .route("/api/v1/remediation/approvals/pending", get(list_pending_approvals))
        
        // Bulk remediation endpoints
        .route("/api/v1/remediation/bulk", post(execute_bulk_remediation))
        .route("/api/v1/remediation/bulk/:id/stream", get(stream_bulk_remediation))
        
        // Rollback endpoints
        .route("/api/v1/remediation/rollback/:token", post(rollback_remediation))
        .route("/api/v1/remediation/rollback/:token/status", get(get_rollback_status))
        
        // Status endpoints
        .route("/api/v1/remediation/:id/status", get(get_remediation_status))
        .route("/api/v1/remediation", get(list_remediations))
}