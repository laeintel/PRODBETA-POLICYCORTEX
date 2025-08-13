use crate::auth::{AuthUser, OptionalAuthUser, TenantContext, TokenValidator};
use crate::secrets::SecretsManager;
use crate::slo::{Aggregation, ErrorBudget, SLIType, SLOManager, SLOWindow, SLI, SLO};
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use chrono::{DateTime, Utc};
use flate2::{write::GzEncoder, Compression};
use metrics::counter;
use metrics_exporter_prometheus::PrometheusHandle;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;
use tar::Builder as TarBuilder;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

// Patent 1: Unified AI Platform - Multi-service data aggregation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GovernanceMetrics {
    pub policies: PolicyMetrics,
    pub rbac: RbacMetrics,
    pub costs: CostMetrics,
    pub network: NetworkMetrics,
    pub resources: ResourceMetrics,
    pub ai: AIMetrics,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyMetrics {
    pub total: u32,
    pub active: u32,
    pub violations: u32,
    pub automated: u32,
    pub compliance_rate: f64,
    pub prediction_accuracy: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RbacMetrics {
    pub users: u32,
    pub roles: u32,
    pub violations: u32,
    pub risk_score: f64,
    pub anomalies_detected: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CostMetrics {
    pub current_spend: f64,
    pub predicted_spend: f64,
    pub savings_identified: f64,
    pub optimization_rate: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkMetrics {
    pub endpoints: u32,
    pub active_threats: u32,
    pub blocked_attempts: u32,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResourceMetrics {
    pub total: u32,
    pub optimized: u32,
    pub idle: u32,
    pub overprovisioned: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AIMetrics {
    pub accuracy: f64,
    pub predictions_made: u64,
    pub automations_executed: u64,
    pub learning_progress: f64,
}

// Patent 2: Predictive Compliance Engine
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompliancePrediction {
    pub resource_id: String,
    pub policy_id: String,
    pub prediction_time: DateTime<Utc>,
    pub violation_probability: f64,
    pub confidence_interval: (f64, f64),
    pub drift_detected: bool,
    pub recommended_actions: Vec<RemediationAction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RemediationAction {
    pub action_type: String,
    pub description: String,
    pub impact_score: f64,
    pub estimated_time_minutes: u32,
    pub automation_available: bool,
}

// Patent 3: Conversational Intelligence
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationRequest {
    pub query: String,
    pub context: Option<ConversationContext>,
    pub session_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationContext {
    pub previous_intents: Vec<String>,
    pub entities: Vec<Entity>,
    pub turn_count: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Entity {
    pub entity_type: String,
    pub value: String,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationResponse {
    pub response: String,
    pub intent: String,
    pub confidence: f64,
    pub suggested_actions: Vec<String>,
    pub generated_policy: Option<String>,
}

// Patent 4: Cross-Domain Correlation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CrossDomainCorrelation {
    pub correlation_id: String,
    pub domains: Vec<String>,
    pub correlation_strength: f64,
    pub causal_relationship: Option<CausalRelationship>,
    pub impact_predictions: Vec<ImpactPrediction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CausalRelationship {
    pub source_domain: String,
    pub target_domain: String,
    pub lag_time_hours: f64,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImpactPrediction {
    pub domain: String,
    pub metric: String,
    pub predicted_change: f64,
    pub time_to_impact_hours: f64,
}

// Proactive Recommendations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProactiveRecommendation {
    pub id: String,
    pub recommendation_type: String,
    pub severity: String,
    pub title: String,
    pub description: String,
    pub potential_savings: Option<f64>,
    pub risk_reduction: Option<f64>,
    pub automation_available: bool,
    pub confidence: f64,
}

pub struct AppState {
    pub metrics: Arc<RwLock<GovernanceMetrics>>,
    pub predictions: Arc<RwLock<Vec<CompliancePrediction>>>,
    pub recommendations: Arc<RwLock<Vec<ProactiveRecommendation>>>,
    pub start_time: std::time::Instant,
    pub azure_client: Option<crate::azure_client::AzureClient>,
    pub async_azure_client: Option<crate::azure_client_async::AsyncAzureClient>,
    // Phase 2 action orchestrator (in-memory)
    pub actions: Arc<RwLock<std::collections::HashMap<String, ActionRecord>>>,
    pub action_events: Arc<RwLock<std::collections::HashMap<String, broadcast::Sender<String>>>>,
    pub config: crate::config::AppConfig,
    pub approvals: Arc<RwLock<std::collections::HashMap<String, ApprovalRequest>>>,
    pub slo_manager: SLOManager,
    pub secrets: Option<SecretsManager>,
    pub prometheus: Option<PrometheusHandle>,
    pub db_pool: Option<sqlx::Pool<sqlx::Postgres>>,
}

impl AppState {
    pub fn new() -> Self {
        let metrics = GovernanceMetrics {
            policies: PolicyMetrics {
                total: 347,
                active: 298,
                violations: 12,
                automated: 285,
                compliance_rate: 99.8,
                prediction_accuracy: 92.3,
            },
            rbac: RbacMetrics {
                users: 1843,
                roles: 67,
                violations: 3,
                risk_score: 18.5,
                anomalies_detected: 7,
            },
            costs: CostMetrics {
                current_spend: 145832.0,
                predicted_spend: 98190.0,
                savings_identified: 47642.0,
                optimization_rate: 89.0,
            },
            network: NetworkMetrics {
                endpoints: 487,
                active_threats: 2,
                blocked_attempts: 127,
                latency_ms: 12.3,
            },
            resources: ResourceMetrics {
                total: 2843,
                optimized: 2456,
                idle: 234,
                overprovisioned: 153,
            },
            ai: AIMetrics {
                accuracy: 96.8,
                predictions_made: 15234,
                automations_executed: 8921,
                learning_progress: 87.3,
            },
        };

        let recommendations = vec![
            ProactiveRecommendation {
                id: "rec-001".to_string(),
                recommendation_type: "cost_optimization".to_string(),
                severity: "high".to_string(),
                title: "VM Right-Sizing Opportunity".to_string(),
                description:
                    "AI detected $12,450/month savings by right-sizing 47 VMs in production"
                        .to_string(),
                potential_savings: Some(12450.0),
                risk_reduction: None,
                automation_available: true,
                confidence: 94.5,
            },
            ProactiveRecommendation {
                id: "rec-002".to_string(),
                recommendation_type: "security".to_string(),
                severity: "critical".to_string(),
                title: "Unencrypted Storage Detected".to_string(),
                description:
                    "Found 3 storage accounts without encryption in production environment"
                        .to_string(),
                potential_savings: None,
                risk_reduction: Some(85.0),
                automation_available: true,
                confidence: 99.9,
            },
            ProactiveRecommendation {
                id: "rec-003".to_string(),
                recommendation_type: "rbac".to_string(),
                severity: "medium".to_string(),
                title: "Excessive Permissions Review".to_string(),
                description: "23 users have unused admin privileges for over 90 days".to_string(),
                potential_savings: None,
                risk_reduction: Some(62.0),
                automation_available: false,
                confidence: 87.3,
            },
        ];

        AppState {
            metrics: Arc::new(RwLock::new(metrics)),
            predictions: Arc::new(RwLock::new(Vec::new())),
            recommendations: Arc::new(RwLock::new(recommendations)),
            start_time: std::time::Instant::now(),
            azure_client: None,       // Will be initialized in main
            async_azure_client: None, // Will be initialized in main
            actions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            action_events: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config: crate::config::AppConfig::load(),
            approvals: Arc::new(RwLock::new(std::collections::HashMap::new())),
            slo_manager: SLOManager::new(),
            secrets: None,
            prometheus: None,
            db_pool: None,
        }
    }
}

// API Handlers
pub async fn get_metrics(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "metrics");
    // Log authenticated request
    // If authenticated, log who; if not, allow simulated data in dev/local flows
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for metrics from user: {:?}",
            user.claims.preferred_username
        );

        // Get tenant context for multi-tenant data access
        let _tenant_context = match TenantContext::from_user(user).await {
            Ok(context) => {
                tracing::debug!(
                    "User has access to tenant: {} with {} subscriptions",
                    context.tenant_id,
                    context.subscription_ids.len()
                );
                context
            }
            Err(e) => {
                tracing::error!("Failed to get tenant context: {:?}", e);
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "error": "tenant_access_denied",
                        "message": "Unable to determine tenant access"
                    })),
                )
                    .into_response();
            }
        };
    } else {
        tracing::info!("Unauthenticated metrics request - returning simulated data (dev mode)");
        let simulated_metrics = GovernanceMetrics {
            policies: PolicyMetrics {
                total: 15,
                active: 12,
                violations: 3,
                automated: 10,
                compliance_rate: 85.5,
                prediction_accuracy: 92.3,
            },
            rbac: RbacMetrics {
                users: 150,
                roles: 25,
                violations: 2,
                risk_score: 3.2,
                anomalies_detected: 1,
            },
            costs: CostMetrics {
                current_spend: 125000.0,
                predicted_spend: 118000.0,
                savings_identified: 7000.0,
                optimization_rate: 88.5,
            },
            network: NetworkMetrics {
                endpoints: 450,
                active_threats: 0,
                blocked_attempts: 127,
                latency_ms: 15.2,
            },
            resources: ResourceMetrics {
                total: 2500,
                optimized: 2100,
                idle: 200,
                overprovisioned: 200,
            },
            ai: AIMetrics {
                accuracy: 94.5,
                predictions_made: 12000,
                automations_executed: 8500,
                learning_progress: 85.0,
            },
        };
        return Json(simulated_metrics).into_response();
    }

    // Always try to get real Azure data when Azure client is available
    // This works for local development with Azure CLI authentication

    // Try high-performance async client first
    if let Some(ref async_azure_client) = state.async_azure_client {
        match async_azure_client.get_governance_metrics().await {
            Ok(real_metrics) => {
                tracing::info!("✅ Real Azure metrics fetched with async client (cached)");
                return Json(real_metrics).into_response();
            }
            Err(e) => {
                tracing::warn!("Async Azure client failed: {}", e);
            }
        }
    }

    // Fallback to synchronous client
    if let Some(ref azure_client) = state.azure_client {
        match azure_client.get_governance_metrics().await {
            Ok(real_metrics) => {
                tracing::info!("✅ Real Azure metrics fetched with sync client");
                return Json(real_metrics).into_response();
            }
            Err(e) => {
                tracing::warn!("Failed to fetch real Azure metrics: {}", e);
                tracing::debug!("Error details: {:?}", e);
            }
        }
    }

    // No Azure connection available - return simulated data
    let simulated_metrics = GovernanceMetrics {
        policies: PolicyMetrics {
            total: 15,
            active: 12,
            violations: 3,
            automated: 10,
            compliance_rate: 85.5,
            prediction_accuracy: 92.3,
        },
        rbac: RbacMetrics {
            users: 150,
            roles: 25,
            violations: 2,
            risk_score: 3.2,
            anomalies_detected: 1,
        },
        costs: CostMetrics {
            current_spend: 125000.0,
            predicted_spend: 118000.0,
            savings_identified: 7000.0,
            optimization_rate: 88.5,
        },
        network: NetworkMetrics {
            endpoints: 450,
            active_threats: 0,
            blocked_attempts: 127,
            latency_ms: 15.2,
        },
        resources: ResourceMetrics {
            total: 2500,
            optimized: 2100,
            idle: 200,
            overprovisioned: 200,
        },
        ai: AIMetrics {
            accuracy: 94.5,
            predictions_made: 12000,
            automations_executed: 8500,
            learning_progress: 85.0,
        },
    };

    // Return simulated data directly (flat shape)
    Json(simulated_metrics).into_response()
}

pub async fn get_predictions(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Verify authentication and tenant access
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for predictions from user: {:?}",
            user.claims.preferred_username
        );
        // Optional: load tenant context without failing if it errors (dev)
        if let Err(e) = TenantContext::from_user(user).await {
            tracing::warn!("Tenant context unavailable: {:?}", e);
        }
    } else {
        tracing::info!("Unauthenticated predictions request (dev mode)");
    }

    let predictions = state.predictions.read().await;
    Json(predictions.clone()).into_response()
}

pub async fn get_recommendations(
    auth_user: OptionalAuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "recommendations");
    // Verify authentication
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for recommendations from user: {:?}",
            user.claims.preferred_username
        );
    } else {
        tracing::info!("Unauthenticated recommendations request (dev mode)");
    }
    // If we have an Azure client, fetch real recommendations based on actual Azure data
    if let Some(ref async_azure_client) = state.async_azure_client {
        match async_azure_client.get_governance_metrics().await {
            Ok(metrics) => {
                // Generate recommendations based on real metrics
                let mut recommendations = Vec::new();

                // Cost recommendations based on actual spend
                if metrics.costs.current_spend > 50.0 {
                    let savings_percentage =
                        (metrics.costs.savings_identified / metrics.costs.current_spend) * 100.0;
                    recommendations.push(ProactiveRecommendation {
                        id: "cost-001".to_string(),
                        recommendation_type: "cost_optimization".to_string(),
                        severity: if savings_percentage > 20.0 { "high".to_string() } else { "medium".to_string() },
                        title: format!("Cost Optimization: ${:.2} potential savings", metrics.costs.savings_identified),
                        description: format!(
                            "Analysis shows ${:.2}/month in potential savings ({:.1}% of current spend). {} idle resources detected.",
                            metrics.costs.savings_identified,
                            savings_percentage,
                            metrics.resources.idle
                        ),
                        potential_savings: Some(metrics.costs.savings_identified),
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 95.0,
                    });
                }

                // Resource optimization recommendations
                if metrics.resources.idle > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "resource-001".to_string(),
                        recommendation_type: "resource_optimization".to_string(),
                        severity: if metrics.resources.idle > 5 { "high".to_string() } else { "medium".to_string() },
                        title: format!("{} Idle Resources Detected", metrics.resources.idle),
                        description: format!(
                            "Found {} resources that have been idle. Consider deallocating or deleting to reduce costs.",
                            metrics.resources.idle
                        ),
                        potential_savings: Some((metrics.resources.idle as f64) * 15.0), // Estimate $15/resource
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 98.0,
                    });
                }

                // Over-provisioned resources
                if metrics.resources.overprovisioned > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "resource-002".to_string(),
                        recommendation_type: "rightsizing".to_string(),
                        severity: "medium".to_string(),
                        title: format!("{} Over-provisioned Resources", metrics.resources.overprovisioned),
                        description: format!(
                            "{} resources are using less than 20% of allocated capacity. Consider downsizing.",
                            metrics.resources.overprovisioned
                        ),
                        potential_savings: Some((metrics.resources.overprovisioned as f64) * 25.0),
                        risk_reduction: None,
                        automation_available: true,
                        confidence: 92.0,
                    });
                }

                // Policy compliance recommendations
                if metrics.policies.violations > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "policy-001".to_string(),
                        recommendation_type: "compliance".to_string(),
                        severity: "critical".to_string(),
                        title: format!("{} Policy Violations Detected", metrics.policies.violations),
                        description: format!(
                            "{} resources are non-compliant with active policies. Immediate remediation recommended.",
                            metrics.policies.violations
                        ),
                        potential_savings: None,
                        risk_reduction: Some(85.0),
                        automation_available: metrics.policies.violations <= 5,
                        confidence: 99.9,
                    });
                }

                // RBAC recommendations
                if metrics.rbac.risk_score > 20.0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "rbac-001".to_string(),
                        recommendation_type: "security".to_string(),
                        severity: if metrics.rbac.risk_score > 50.0 { "critical".to_string() } else { "high".to_string() },
                        title: "Elevated RBAC Risk Score".to_string(),
                        description: format!(
                            "RBAC risk score is {:.1}%. Review permissions for {} users with elevated access.",
                            metrics.rbac.risk_score,
                            (metrics.rbac.users as f64 * 0.1) as u32
                        ),
                        potential_savings: None,
                        risk_reduction: Some(metrics.rbac.risk_score),
                        automation_available: false,
                        confidence: 88.0,
                    });
                }

                // Network security recommendations
                if metrics.network.active_threats > 0 {
                    recommendations.push(ProactiveRecommendation {
                        id: "network-001".to_string(),
                        recommendation_type: "security".to_string(),
                        severity: "critical".to_string(),
                        title: format!("{} Active Network Threats", metrics.network.active_threats),
                        description: format!(
                            "Active threats detected on {} endpoints. {} attempts blocked in last 24 hours.",
                            metrics.network.active_threats,
                            metrics.network.blocked_attempts
                        ),
                        potential_savings: None,
                        risk_reduction: Some(95.0),
                        automation_available: true,
                        confidence: 99.5,
                    });
                }

                return Json(recommendations);
            }
            Err(e) => {
                tracing::warn!("Failed to generate real-time recommendations: {}", e);
            }
        }
    }

    // Fallback to cached recommendations
    let recommendations = state.recommendations.read().await;
    Json(recommendations.clone())
}

// ===================== Config & Secrets Status =====================

#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    environment: String,
    version: String,
    approvals_required: bool,
    strict_audience: bool,
    allowed_origins: usize,
}

pub async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = &state.config;
    Json(ConfigResponse {
        environment: cfg.environment.clone(),
        version: cfg.service_version.clone(),
        approvals_required: cfg.require_approvals,
        strict_audience: cfg.require_strict_audience,
        allowed_origins: cfg.allowed_origins.len(),
    })
}

#[derive(Debug, Serialize)]
pub struct SecretsStatus {
    ok: bool,
    missing: Vec<String>,
}

pub async fn get_secrets_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref sm) = state.secrets {
        match sm.validate_secrets().await {
            Ok(_) => {
                return (
                    StatusCode::OK,
                    Json(SecretsStatus {
                        ok: true,
                        missing: vec![],
                    }),
                )
                    .into_response()
            }
            Err(missing) => {
                return (StatusCode::OK, Json(SecretsStatus { ok: false, missing })).into_response()
            }
        }
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "secrets manager unavailable"})),
    )
        .into_response()
}

pub async fn export_prometheus(
    State(state): State<Arc<AppState>>,
    auth: OptionalAuthUser,
) -> impl IntoResponse {
    // In production, require auth for metrics to avoid exposing internals
    let is_prod = matches!(
        std::env::var("ENVIRONMENT").as_deref(),
        Ok("production") | Ok("prod")
    );
    if is_prod && auth.0.is_none() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error":"unauthorized"})),
        )
            .into_response();
    }

    if let Some(ref h) = state.prometheus {
        let body = h.render();
        return (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4",
            )],
            body,
        )
            .into_response();
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "prometheus not initialized"})),
    )
        .into_response()
}

// Secrets cache control
pub async fn reload_secrets(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref sm) = state.secrets {
        sm.clear_cache().await;
        return Json(serde_json::json!({"success": true})).into_response();
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "secrets manager unavailable"})),
    )
        .into_response()
}

// Evidence pack and policy export stubs
pub async fn get_evidence_pack() -> impl IntoResponse {
    // Build a small tar.gz evidence pack on the fly with sample artifacts
    let mut buf = Vec::new();
    let enc = GzEncoder::new(&mut buf, Compression::default());
    let mut tar = TarBuilder::new(enc);
    let now = chrono::Utc::now().to_rfc3339();

    // policy_snapshot.json
    let policy = serde_json::json!({"snapshot_at": now, "items": [{"id":"require-tags","status":"noncompliant","count":58}]});
    let policy_bytes = match serde_json::to_vec_pretty(&policy) {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed_to_generate_policy_snapshot"})),
            )
                .into_response()
        }
    };
    let mut header = tar::Header::new_gnu();
    header.set_size(policy_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(chrono::Utc::now().timestamp() as u64);
    header.set_cksum();
    if tar
        .append_data(&mut header, "policy_snapshot.json", &policy_bytes[..])
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":"failed_to_append_policy_snapshot"})),
        )
            .into_response();
    }

    // rbac_assignments.csv
    let rbac_csv = b"principal,role,scope\nuser@contoso.com,Owner,/subscriptions/xxx\n";
    let mut header2 = tar::Header::new_gnu();
    header2.set_size(rbac_csv.len() as u64);
    header2.set_mode(0o644);
    header2.set_mtime(chrono::Utc::now().timestamp() as u64);
    header2.set_cksum();
    if tar
        .append_data(&mut header2, "rbac_assignments.csv", &rbac_csv[..])
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":"failed_to_append_rbac"})),
        )
            .into_response();
    }

    // cost_anomalies.csv
    let cost_csv = b"service,anomaly_usd,date\nStorage,2450.13,2025-08-01\n";
    let mut header3 = tar::Header::new_gnu();
    header3.set_size(cost_csv.len() as u64);
    header3.set_mode(0o644);
    header3.set_mtime(chrono::Utc::now().timestamp() as u64);
    header3.set_cksum();
    if tar
        .append_data(&mut header3, "cost_anomalies.csv", &cost_csv[..])
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":"failed_to_append_costs"})),
        )
            .into_response();
    }

    if let Err(_) = tar.finish() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":"failed_to_finalize_tar"})),
        )
            .into_response();
    }
    let enc = match tar.into_inner() {
        Ok(enc) => enc,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed_to_get_encoder"})),
            )
                .into_response()
        }
    };
    let body = match enc.finish() {
        Ok(body) => body,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error":"failed_to_finish_encoding"})),
            )
                .into_response()
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/gzip")
        .header(
            axum::http::header::CONTENT_DISPOSITION,
            "attachment; filename=evidence.tar.gz",
        )
        .body(Body::from(body.clone()))
        .unwrap()
}

pub async fn export_policies(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    // For now, return the same simulated policies used in get_policies
    use crate::simulated_data::SimulatedDataProvider;
    let items = SimulatedDataProvider::get_policies();
    Json(serde_json::json!({"items": items, "count": items.len()}))
}

// ===================== Actions Preflight (UI support) =====================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreflightChange {
    pub resource: String,
    pub additions: u32,
    pub deletions: u32,
    pub diff: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreflightResponse {
    pub status: String,
    pub message: String,
    pub changes: Vec<PreflightChange>,
    pub validations: Vec<serde_json::Value>,
}

pub async fn get_action_preflight(Path(_id): Path<String>) -> impl IntoResponse {
    // Stub preflight with deterministic payload for UI
    let changes = vec![PreflightChange {
        resource: "Microsoft.Compute/virtualMachines/vm-prod-001".to_string(),
        additions: 1,
        deletions: 0,
        diff: vec![
            serde_json::json!({"type":"add","lineNumber":42,"content":"tags.owner = 'FinOps'"}),
        ],
    }];
    let validations = vec![
        serde_json::json!({"name":"Policy evaluation","passed":true}),
        serde_json::json!({"name":"RBAC permission check","passed":true}),
    ];
    Json(PreflightResponse {
        status: "passed".to_string(),
        message: "Preflight checks passed".to_string(),
        changes,
        validations,
    })
}

// ===================== Compliance Frameworks =====================

#[derive(Debug, Serialize)]
pub struct FrameworkInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub controls: usize,
}

pub async fn list_frameworks() -> impl IntoResponse {
    Json(vec![
        FrameworkInfo {
            id: "cis-azure".into(),
            name: "CIS Microsoft Azure Foundations Benchmark".into(),
            version: "1.4".into(),
            controls: 92,
        },
        FrameworkInfo {
            id: "nist-800-53".into(),
            name: "NIST SP 800-53".into(),
            version: "rev5".into(),
            controls: 110,
        },
    ])
}

// ===================== Policy Drift (GitOps scaffolding) =====================

#[derive(Debug, Serialize)]
pub struct DriftItem {
    pub id: String,
    pub type_name: String,
    pub expected: serde_json::Value,
    pub actual: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct DriftReport {
    pub drifted: usize,
    pub items: Vec<DriftItem>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

pub async fn get_policy_drift() -> impl IntoResponse {
    // Stub drift report for UI integration; later compare desired (Git) vs. current (Azure)
    let items = vec![DriftItem {
        id: "policy:require-tags".into(),
        type_name: "AzurePolicy".into(),
        expected: serde_json::json!({"effect":"deny","requiredTags":["Owner","CostCenter"]}),
        actual: serde_json::json!({"effect":"audit","requiredTags":["Owner"]}),
    }];
    Json(DriftReport {
        drifted: items.len(),
        items,
        generated_at: chrono::Utc::now(),
    })
}

// ===================== Roadmap Status =====================

#[derive(Debug, Serialize)]
pub struct RoadmapItem {
    pub id: String,
    pub name: String,
    pub progress: u8,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct RoadmapStatus {
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub items: Vec<RoadmapItem>,
}

pub async fn get_roadmap_status(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    // Estimate based on implemented features; keep aligned with backend capabilities
    let items = vec![
        RoadmapItem{ id: "kv_secrets".into(), name: "KV-backed secrets for DB/JWT".into(), progress: 25, description: "Secrets manager wired; DB pool via KV/env implemented; JWT key pending.".into() },
        RoadmapItem{ id: "metrics".into(), name: "Status/latency histograms + /metrics".into(), progress: 80, description: "Prometheus exporter, counters, latency histograms with status; more business metrics TBD.".into() },
        RoadmapItem{ id: "compliance_evidence".into(), name: "Compliance mapping & evidence packs".into(), progress: 40, description: "Frameworks endpoints and downloadable evidence tar.gz (sample). Real artifacts mapping pending.".into() },
        RoadmapItem{ id: "gitops".into(), name: "GitOps PR flows & drift/rollback".into(), progress: 20, description: "Drift and policy generator stubs; PR creation & rollback logic TBD.".into() },
        RoadmapItem{ id: "guardrails".into(), name: "Auto-remediation guardrails".into(), progress: 40, description: "Approvals required in non-dev; preflight diff endpoint; staged rollout/rollback pending.".into() },
        RoadmapItem{ id: "sso_multicloud".into(), name: "SSO/RBAC breadth & multi-cloud".into(), progress: 10, description: "Scaffolding only; providers/adapters TBD.".into() },
    ];
    Json(RoadmapStatus {
        last_updated: chrono::Utc::now(),
        items,
    })
}

pub async fn get_framework(Path(id): Path<String>) -> impl IntoResponse {
    // Minimal mapping sample
    let controls = vec![
        serde_json::json!({"id":"AZ-1","title":"Ensure Storage Accounts have secure transfer required","policyId":"/providers/Microsoft.Authorization/policyDefinitions/secureTransfer"}),
        serde_json::json!({"id":"AZ-2","title":"Ensure VMs use managed disks","policyId":"/providers/Microsoft.Authorization/policyDefinitions/managedDisks"}),
    ];
    Json(serde_json::json!({"id": id, "controls": controls}))
}

// ===================== Approvals (Phase 1) =====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub requested_by: Option<String>,
    pub resource_id: String,
    pub action: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateApprovalPayload {
    pub resource_id: String,
    pub action: String,
}

pub async fn create_approval(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateApprovalPayload>,
) -> impl IntoResponse {
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Approve"]) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"insufficient_scope"})),
        )
            .into_response();
    }
    use chrono::Utc;
    use uuid::Uuid;
    let id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let requested_by = auth_user
        .claims
        .preferred_username
        .clone()
        .or(Some("anonymous".to_string()));

    let req = ApprovalRequest {
        id: id.clone(),
        requested_by,
        resource_id: payload.resource_id,
        action: payload.action,
        status: "Pending".to_string(),
        created_at: now,
        updated_at: now,
    };
    {
        let mut approvals = state.approvals.write().await;
        approvals.insert(id.clone(), req.clone());
    }
    Json(serde_json::json!({"success": true, "approval": req})).into_response()
}

pub async fn list_approvals(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let approvals = state.approvals.read().await;
    let mut items: Vec<_> = approvals.values().cloned().collect();
    items.sort_by_key(|a| a.created_at);
    Json(items)
}

#[derive(Debug, Deserialize)]
pub struct ApprovePayload {
    pub approve: bool,
}

pub async fn approve_request(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(id): Path<String>,
    Json(payload): Json<ApprovePayload>,
) -> impl IntoResponse {
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Approve"]) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"insufficient_scope"})),
        )
            .into_response();
    }
    let mut approvals = state.approvals.write().await;
    if let Some(a) = approvals.get_mut(&id) {
        a.status = if payload.approve {
            "Approved"
        } else {
            "Rejected"
        }
        .to_string();
        a.updated_at = chrono::Utc::now();
        return (
            StatusCode::OK,
            Json(serde_json::json!({"success": true, "approval": a})),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "approval not found"})),
    )
        .into_response()
}

// ===================== Policy-as-code scaffolding =====================

#[derive(Debug, Deserialize)]
pub struct GeneratePolicyPayload {
    pub requirement: String,
    pub provider: Option<String>,
    pub framework: Option<String>,
}

pub async fn generate_policy(Json(payload): Json<GeneratePolicyPayload>) -> impl IntoResponse {
    // Provide a minimal Azure policy skeleton based on a requirement string (stub)
    let policy = serde_json::json!({
        "mode": "All",
        "parameters": {},
        "policyRule": {
            "if": {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachines"}
                ]
            },
            "then": {"effect": "audit"}
        },
        "metadata": {"generatedFrom": payload.requirement, "provider": payload.provider, "framework": payload.framework}
    });
    Json(serde_json::json!({ "policy": policy }))
}

pub async fn process_conversation(
    auth_user: AuthUser,
    State(_state): State<Arc<AppState>>,
    Json(request): Json<ConversationRequest>,
) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "conversation");
    // Verify authentication and get user context for personalized responses
    tracing::info!(
        "Authenticated conversation request from user: {:?}",
        auth_user.claims.preferred_username
    );

    let user_context = format!(
        " (authenticated as {})",
        auth_user
            .claims
            .preferred_username
            .as_deref()
            .unwrap_or("unknown user")
    );

    // Simulate NLP processing with user context
    let response = ConversationResponse {
        response: format!("I understand you're asking about: {}{}. Based on your environment analysis, I recommend reviewing the cost optimization opportunities that could save you $12,450/month.", request.query, user_context),
        intent: "cost_inquiry".to_string(),
        confidence: 92.5,
        suggested_actions: vec![
            "Review VM sizing recommendations".to_string(),
            "Enable auto-scaling for development resources".to_string(),
            "Implement resource tagging policy".to_string(),
        ],
        generated_policy: Some(r#"{
            "mode": "Indexed",
            "policyRule": {
                "if": {
                    "field": "type",
                    "equals": "Microsoft.Compute/virtualMachines"
                },
                "then": {
                    "effect": "audit"
                }
            }
        }"#.to_string()),
    };

    Json(response)
}

pub async fn get_correlations(auth_user: OptionalAuthUser) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "correlations");
    // Verify authentication
    if let Some(ref user) = auth_user.0 {
        tracing::info!(
            "Authenticated request for correlations from user: {:?}",
            user.claims.preferred_username
        );
    } else {
        tracing::info!("Unauthenticated correlations request (dev mode)");
    }
    let correlation = CrossDomainCorrelation {
        correlation_id: "corr-001".to_string(),
        domains: vec!["cost".to_string(), "resources".to_string()],
        correlation_strength: 0.87,
        causal_relationship: Some(CausalRelationship {
            source_domain: "resources".to_string(),
            target_domain: "cost".to_string(),
            lag_time_hours: 24.0,
            confidence: 89.5,
        }),
        impact_predictions: vec![ImpactPrediction {
            domain: "cost".to_string(),
            metric: "monthly_spend".to_string(),
            predicted_change: -8.5,
            time_to_impact_hours: 72.0,
        }],
    };

    Json(vec![correlation])
}

// ===================== Deep Insights & Actions (Phase 1) =====================

#[derive(Debug, Deserialize)]
pub struct RemediateRequest {
    pub resource_id: String,
    pub action: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateExceptionRequest {
    pub resource_id: String,
    pub policy_id: String,
    pub reason: String,
}

// Get policies with data mode support
pub async fn get_policies(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "policies");
    use crate::data_mode::{DataMode, DataResponse};
    use crate::simulated_data::SimulatedDataProvider;

    let mode = DataMode::from_env();

    // Try to get real data if available and mode is Real
    if mode.is_real() {
        if let Some(ref async_azure_client) = state.async_azure_client {
            match async_azure_client.get_policies().await {
                Ok(policies) => {
                    return Json(DataResponse::new(
                        serde_json::json!({
                            "policies": policies,
                            "total": policies.len(),
                        }),
                        DataMode::Real,
                    ));
                }
                Err(e) => {
                    tracing::warn!("Failed to get real policies: {}", e);
                }
            }
        }
    }

    // Use simulated data as fallback or when in simulated mode
    let simulated_policies = SimulatedDataProvider::get_policies();
    Json(DataResponse::new(
        serde_json::json!({
            "policies": simulated_policies,
            "total": simulated_policies.len(),
        }),
        DataMode::Simulated,
    ))
}

// Policies Deep Compliance
pub async fn get_policies_deep() -> impl IntoResponse {
    // Try proxy to Python deep service if configured
    if let Some(json) = proxy_deep_get("/api/v1/policies/deep").await {
        return Json(json);
    }

    Json(serde_json::json!({
        "success": true,
        "message": "Deep policy compliance (core)",
        "complianceResults": [
            {
                "assignment": {
                    "name": "require-tags",
                    "displayName": "Require Resource Tags",
                    "scope": "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78"
                },
                "summary": {
                    "totalResources": 147,
                    "compliantResources": 89,
                    "nonCompliantResources": 58,
                    "compliancePercentage": 60.5
                },
                "nonCompliantResources": [
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-prod-001",
                        "resourceName": "vm-prod-001",
                        "resourceType": "Microsoft.Compute/virtualMachines",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Missing required tags: Environment, Owner, CostCenter",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Automatically add missing tags"},
                            {"action": "create-exception", "description": "Create policy exception"},
                            {"action": "manual-fix", "description": "Manually add tags"}
                        ]
                    },
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod001",
                        "resourceName": "stprod001",
                        "resourceType": "Microsoft.Storage/storageAccounts",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Missing required tags: Environment, CostCenter",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Automatically add missing tags"},
                            {"action": "create-exception", "description": "Create policy exception"}
                        ]
                    }
                ]
            },
            {
                "assignment": {
                    "name": "require-encryption",
                    "displayName": "Require Encryption at Rest",
                    "scope": "/subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78"
                },
                "summary": {
                    "totalResources": 83,
                    "compliantResources": 71,
                    "nonCompliantResources": 12,
                    "compliancePercentage": 85.5
                },
                "nonCompliantResources": [
                    {
                        "resourceId": "/subscriptions/205b477d/resourceGroups/rg-dev/providers/Microsoft.Storage/storageAccounts/stdev002",
                        "resourceName": "stdev002",
                        "resourceType": "Microsoft.Storage/storageAccounts",
                        "complianceState": "NonCompliant",
                        "complianceReason": "Encryption at rest is not enabled",
                        "remediationOptions": [
                            {"action": "auto-remediate", "description": "Enable encryption automatically"},
                            {"action": "create-exception", "description": "Create policy exception"}
                        ]
                    }
                ]
            }
        ]
    }))
}

// Initiate remediation (stub – Phase 1). Later, orchestrate jobs and stream progress.
pub async fn remediate(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RemediateRequest>,
) -> impl IntoResponse {
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Write"]) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"insufficient_scope"})),
        )
            .into_response();
    }
    // Enforce write safety: block writes in simulated mode
    use crate::data_mode::{DataMode, DataModeGuard, DataResponse};
    let guard = DataModeGuard::new();
    if let Err(e) = guard.ensure_write_allowed() {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "success": false,
                "status": "ReadOnlyMode",
                "message": e.to_string()
            })),
        )
            .into_response();
    }

    // Enforce approvals in non-dev environments
    let require_approvals = crate::config::AppConfig::load().require_approvals;
    if require_approvals {
        let approvals = state.approvals.read().await;
        let approved = approvals.values().any(|a| {
            a.status == "Approved"
                && a.resource_id == payload.resource_id
                && a.action == payload.action
        });
        if !approved {
            return Json(serde_json::json!({
                "success": false,
                "status": "PendingApproval",
                "message": "Remediation requires approval before execution",
                "next": "Submit approval via /api/v1/approvals"
            }))
            .into_response();
        }
    }

    Json(serde_json::json!({
        "success": true,
        "resourceId": payload.resource_id,
        "action": payload.action,
        "status": "Initiated",
        "estimatedCompletion": "5 minutes",
        "message": format!("Remediation '{}' initiated for resource {}", payload.action, payload.resource_id)
    })).into_response()
}

// Create a policy exception (stub – Phase 1)
pub async fn create_exception(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateExceptionRequest>,
) -> impl IntoResponse {
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Write"]) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"insufficient_scope"})),
        )
            .into_response();
    }
    use chrono::Utc;
    let id = Uuid::new_v4();
    let expires_at = Utc::now() + chrono::Duration::days(30);
    if let Some(ref pool) = state.db_pool {
        let _ = sqlx::query(
            r#"INSERT INTO exceptions (
                id, tenant_id, resource_id, policy_id, reason, status, created_by, created_at, expires_at, recertify_at, evidence, metadata
            ) VALUES ($1,$2,$3,$4,$5,'Approved',$6,NOW(),$7,$8,$9,$10)"#
        )
        .bind(&id)
        .bind(auth_user.claims.tid.clone().unwrap_or_else(|| "default".to_string()))
        .bind(&payload.resource_id)
        .bind(&payload.policy_id)
        .bind(&payload.reason)
        .bind(auth_user.claims.preferred_username.clone().unwrap_or_default())
        .bind(&expires_at)
        .bind(&expires_at)
        .bind(serde_json::json!({"requested_by": auth_user.claims.preferred_username}))
        .bind(serde_json::json!({}))
        .execute(pool)
        .await;
    }
    Json(serde_json::json!({
        "success": true,
        "exceptionId": id,
        "resourceId": payload.resource_id,
        "policyId": payload.policy_id,
        "reason": payload.reason,
        "expiresAt": expires_at.to_rfc3339(),
        "status": "Approved",
        "recertifyAt": expires_at.to_rfc3339(),
        "evidenceRequired": true
    }))
    .into_response()
}

// Helper: Proxy deep GET to Python service (Phase 3). Require DEEP_API_BASE in prod; no localhost fallback
async fn proxy_deep_get(path: &str) -> Option<serde_json::Value> {
    let is_prod = matches!(
        std::env::var("ENVIRONMENT").as_deref(),
        Ok("production") | Ok("prod")
    );
    let base = match std::env::var("DEEP_API_BASE").or_else(|_| std::env::var("API_GATEWAY_URL")) {
        Ok(v) => v,
        Err(_) if is_prod => return None,
        Err(_) => "http://localhost:8090".to_string(),
    };
    let url = format!("{}{}", base, path);
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => resp.json::<serde_json::Value>().await.ok(),
        _ => None,
    }
}

// Additional deep endpoints with REAL Azure data
pub async fn get_rbac_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Try Python service first for AI-enhanced analysis
    if let Some(json) = proxy_deep_get("/api/v1/rbac/deep").await {
        return Json(json);
    }

    // Fallback to direct Azure API
    if let Some(ref client) = state.async_azure_client {
        match client.get_rbac_analysis().await {
            Ok(rbac_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "roleAssignments": rbac_data.get("assignments"),
                    "riskAnalysis": {
                        "privilegedAccounts": rbac_data.get("privileged_count"),
                        "highRiskAssignments": rbac_data.get("high_risk_count"),
                        "staleAssignments": rbac_data.get("stale_count"),
                        "overprivilegedIdentities": rbac_data.get("overprivileged_identities")
                    },
                    "recommendations": rbac_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get RBAC data from Azure: {}", e);
            }
        }
    }

    // No Azure connection available
    Json(serde_json::json!({
        "error": "Azure connection not available",
        "status": "unavailable"
    }))
}

pub async fn get_costs_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/costs/deep").await {
        return Json(json);
    }

    // Get REAL cost data from Azure Cost Management
    if let Some(ref client) = state.async_azure_client {
        match client.get_cost_analysis().await {
            Ok(cost_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "totalCost": cost_data.get("total_cost"),
                    "breakdown": cost_data.get("service_breakdown"),
                    "anomalies": cost_data.get("anomalies"),
                    "trends": cost_data.get("trends"),
                    "forecast": cost_data.get("forecast"),
                    "optimizationPotential": cost_data.get("optimization_potential"),
                    "recommendations": cost_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get cost data from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Cost Management API not available",
        "status": "unavailable"
    }))
}

pub async fn get_network_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/network/deep").await {
        return Json(json);
    }

    // Get REAL network topology from Azure
    if let Some(ref client) = state.async_azure_client {
        match client.get_network_topology().await {
            Ok(network_data) => {
                return Json(serde_json::json!({
                    "success": true,
                    "networkSecurityGroups": network_data.get("nsgs"),
                    "virtualNetworks": network_data.get("vnets"),
                    "publicEndpoints": network_data.get("public_endpoints"),
                    "privateEndpoints": network_data.get("private_endpoints"),
                    "securityRisks": network_data.get("security_risks"),
                    "recommendations": network_data.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get network data from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Network API not available",
        "status": "unavailable"
    }))
}

pub async fn get_resources_deep(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(json) = proxy_deep_get("/api/v1/resources/deep").await {
        return Json(json);
    }

    // Get REAL resources from Azure Resource Graph
    if let Some(ref client) = state.async_azure_client {
        match client.get_all_resources_with_health().await {
            Ok(resources) => {
                return Json(serde_json::json!({
                    "success": true,
                    "resources": resources.get("items"),
                    "totalCount": resources.get("total_count"),
                    "healthSummary": resources.get("health_summary"),
                    "complianceSummary": resources.get("compliance_summary"),
                    "tagAnalysis": resources.get("tag_analysis"),
                    "recommendations": resources.get("recommendations")
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get resources from Azure: {}", e);
            }
        }
    }

    Json(serde_json::json!({
        "error": "Azure Resource Graph not available",
        "status": "unavailable"
    }))
}

// ===================== Additional Endpoints =====================

pub async fn get_compliance(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "compliance");
    use crate::data_mode::{DataMode, DataResponse};
    let mode = DataMode::from_env();
    // Try to get real compliance data from Azure
    if let Some(ref async_client) = state.async_azure_client {
        match async_client.get_governance_metrics().await {
            Ok(metrics) => {
                return Json(DataResponse::new(
                    serde_json::json!({
                        "status": "compliant",
                        "policies": {
                            "total": metrics.policies.total,
                            "compliant": metrics.policies.active,
                            "non_compliant": metrics.policies.violations,
                            "compliance_rate": metrics.policies.compliance_rate
                        },
                        "rbac": {
                            "violations": metrics.rbac.violations,
                            "risk_score": metrics.rbac.risk_score
                        },
                        "timestamp": chrono::Utc::now()
                    }),
                    DataMode::Real,
                ));
            }
            Err(e) => {
                tracing::warn!("Failed to get compliance data: {}", e);
            }
        }
    }

    // Return default compliance data wrapped with source info
    Json(DataResponse::new(
        serde_json::json!({
            "status": "compliant",
            "policies": {
                "total": 15,
                "compliant": 12,
                "non_compliant": 3,
                "compliance_rate": 80.0
            },
            "rbac": {
                "violations": 2,
                "risk_score": 3.2
            },
            "timestamp": chrono::Utc::now()
        }),
        DataMode::Simulated,
    ))
}

pub async fn get_resources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    counter!("api_requests_total", 1, "endpoint" => "resources");
    use crate::data_mode::{DataMode, DataResponse};
    let mode = DataMode::from_env();
    // Try to get real resource data from Azure
    if let Some(ref async_client) = state.async_azure_client {
        match async_client.get_governance_metrics().await {
            Ok(metrics) => {
                return Json(DataResponse::new(
                    serde_json::json!({
                        "resources": {
                            "total": metrics.resources.total,
                            "optimized": metrics.resources.optimized,
                            "idle": metrics.resources.idle,
                            "overprovisioned": metrics.resources.overprovisioned
                        },
                        "costs": {
                            "current_spend": metrics.costs.current_spend,
                            "savings_identified": metrics.costs.savings_identified
                        },
                        "timestamp": chrono::Utc::now()
                    }),
                    DataMode::Real,
                ));
            }
            Err(e) => {
                tracing::warn!("Failed to get resource data: {}", e);
            }
        }
    }

    // Return default resource data wrapped
    Json(DataResponse::new(
        serde_json::json!({
            "resources": {
                "total": 2500,
                "optimized": 1875,
                "idle": 125,
                "overprovisioned": 50
            },
            "costs": {
                "current_spend": 125000.0,
                "savings_identified": 7000.0
            },
            "timestamp": chrono::Utc::now()
        }),
        DataMode::Simulated,
    ))
}

// ===================== Action Orchestrator (Phase 2) =====================

#[derive(Debug, Deserialize)]
pub struct CreateActionRequest {
    pub action_type: String,
    pub resource_id: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub id: String,
    pub action_type: String,
    pub resource_id: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub params: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub stage: u8,
    pub total_stages: u8,
    pub rollback_available: bool,
}

pub async fn create_action(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateActionRequest>,
) -> impl IntoResponse {
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Write"]) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"insufficient_scope"})),
        )
            .into_response();
    }
    use chrono::Utc;
    use uuid::Uuid;

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let record = ActionRecord {
        id: id.clone(),
        action_type: payload.action_type.clone(),
        resource_id: payload.resource_id.clone(),
        status: "queued".to_string(),
        created_at: now,
        updated_at: now,
        params: payload.params.clone(),
        result: None,
        stage: 0,
        total_stages: 3,
        rollback_available: false,
    };

    {
        let mut actions = state.actions.write().await;
        actions.insert(id.clone(), record);
    }

    let (tx, _rx) = broadcast::channel::<String>(100);
    {
        let mut events = state.action_events.write().await;
        events.insert(id.clone(), tx.clone());
    }

    let state_clone = state.clone();
    let id_clone = id.clone();
    tokio::spawn(async move {
        let send_step = |m: &str| {
            let _ = tx.send(m.to_string());
        };
        send_step("queued");
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.status = "in_progress".to_string();
                a.updated_at = chrono::Utc::now();
            }
        }
        send_step("in_progress: preflight");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.stage = 1;
                a.updated_at = chrono::Utc::now();
            }
        }
        send_step("in_progress: executing");
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.stage = 2;
                a.updated_at = chrono::Utc::now();
                a.rollback_available = true;
            }
        }
        send_step("in_progress: verifying");
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        {
            let mut actions = state_clone.actions.write().await;
            if let Some(a) = actions.get_mut(&id_clone) {
                a.status = "completed".to_string();
                a.updated_at = chrono::Utc::now();
                a.result = Some(
                    serde_json::json!({"message": "Action executed successfully", "changes": 1}),
                );
                a.stage = 3;
            }
        }
        send_step("completed");
    });

    Json(serde_json::json!({"action_id": id})).into_response()
}

pub async fn get_action(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let actions = state.actions.read().await;
    if let Some(a) = actions.get(&action_id) {
        return Json(serde_json::json!(a));
    }
    Json(serde_json::json!({"error": "action not found"}))
}

#[derive(Debug, Deserialize)]
pub struct StageAdvance {
    pub stage: Option<u8>,
}

pub async fn advance_stage(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
    Json(payload): Json<StageAdvance>,
) -> impl IntoResponse {
    let mut actions = state.actions.write().await;
    if let Some(a) = actions.get_mut(&action_id) {
        let new_stage = payload.stage.unwrap_or(a.stage.saturating_add(1));
        a.stage = new_stage.min(a.total_stages);
        a.updated_at = chrono::Utc::now();
        if a.stage >= a.total_stages {
            a.status = "completed".to_string();
        }
        return Json(serde_json::json!({"success": true, "action": a})).into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "action not found"})),
    )
        .into_response()
}

pub async fn rollback_action(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let mut actions = state.actions.write().await;
    if let Some(a) = actions.get_mut(&action_id) {
        if a.rollback_available {
            a.status = "rolled_back".to_string();
            a.updated_at = chrono::Utc::now();
            a.result = Some(serde_json::json!({"message":"Rollback executed","reverted": true}));
            return Json(serde_json::json!({"success": true, "action": a})).into_response();
        } else {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "rollback not available"})),
            )
                .into_response();
        }
    }
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "action not found"})),
    )
        .into_response()
}

pub async fn get_action_impact(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let actions = state.actions.read().await;
    if actions.get(&action_id).is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "action not found"})),
        )
            .into_response();
    }
    Json(serde_json::json!({
        "resourceChanges": [
            {"resource":"Microsoft.Compute/virtualMachines/vm-prod-001","changes":[{"op":"add","path":"/tags/Owner","value":"FinOps"}]} 
        ],
        "blastRadius": {"resources": 1, "dependencies": 0},
        "riskScore": 15
    })).into_response()
}

pub async fn stream_action_events(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> impl IntoResponse {
    let rx_opt = {
        let events = state.action_events.read().await;
        events.get(&action_id).cloned()
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
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "action not found"})),
        )
            .into_response()
    }
}

// Global SSE stream for lightweight real-time updates (heartbeat + metric snapshots)
pub async fn stream_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stream = async_stream::stream! {
        // Send initial hello
        let hello = serde_json::json!({
            "type": "connected",
            "timestamp": chrono::Utc::now(),
        });
        yield Ok::<Event, std::convert::Infallible>(Event::default().data(hello.to_string()));

        // Periodic updates
        loop {
            // Snapshot a small subset of metrics to keep payload small
            let metrics = state.metrics.read().await.clone();
            let snapshot = serde_json::json!({
                "type": "metric_update",
                "timestamp": chrono::Utc::now(),
                "data": {
                    "policies": { "total": metrics.policies.total, "violations": metrics.policies.violations, "compliance_rate": metrics.policies.compliance_rate },
                    "costs": { "current_spend": metrics.costs.current_spend, "savings_identified": metrics.costs.savings_identified },
                    "security": { "risk_score": metrics.rbac.risk_score, "anomalies_detected": metrics.rbac.anomalies_detected }
                }
            });
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(snapshot.to_string()));
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        }
    };
    Sse::new(stream).into_response()
}

// ===================== Exceptions Management =====================

#[derive(Debug, Serialize)]
pub struct ExceptionRecord {
    pub id: Uuid,
    pub resource_id: String,
    pub policy_id: String,
    pub reason: String,
    pub status: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub recertify_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub async fn list_exceptions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref pool) = state.db_pool {
        if let Ok(rows) = sqlx::query(
            r#"SELECT id, resource_id, policy_id, reason, status, expires_at, recertify_at, created_at FROM exceptions ORDER BY created_at DESC LIMIT 100"#
        )
        .fetch_all(pool)
        .await
        {
            let items: Vec<ExceptionRecord> = rows
                .into_iter()
                .map(|r| {
                    let id: uuid::Uuid = r.try_get("id").unwrap_or_else(|_| uuid::Uuid::new_v4());
                    let resource_id: String = r.try_get("resource_id").unwrap_or_default();
                    let policy_id: String = r.try_get("policy_id").unwrap_or_default();
                    let reason: String = r.try_get("reason").unwrap_or_default();
                    let status: String = r.try_get("status").unwrap_or_else(|_| "Approved".to_string());
                    let expires_at: chrono::DateTime<chrono::Utc> = r.try_get("expires_at").unwrap_or(chrono::Utc::now());
                    let recertify_at: Option<chrono::DateTime<chrono::Utc>> = r.try_get("recertify_at").ok();
                    let created_at: chrono::DateTime<chrono::Utc> = r.try_get("created_at").unwrap_or(chrono::Utc::now());
                    ExceptionRecord {
                        id,
                        resource_id,
                        policy_id,
                        reason,
                        status,
                        expires_at,
                        recertify_at,
                        created_at,
                    }
                })
                .collect();
            return Json(serde_json::json!({"items": items}));
        }
    }
    Json(serde_json::json!({"items": []}))
}

#[derive(Debug, Serialize)]
pub struct ExpireResult {
    pub expired: i64,
}

pub async fn expire_exceptions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref pool) = state.db_pool {
        match sqlx::query(
            r#"UPDATE exceptions SET status = 'Expired' WHERE expires_at < NOW() AND status <> 'Expired' RETURNING 1"#
        )
        .fetch_all(pool)
        .await
        {
            Ok(rows) => return Json(ExpireResult { expired: rows.len() as i64 }).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
        }
    }
    Json(ExpireResult { expired: 0 }).into_response()
}
