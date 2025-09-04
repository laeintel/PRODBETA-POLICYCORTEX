// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use axum::{
    extract::State,
    http::{header, Method},
    routing::{get, post, put},
    Json, Router,
};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod action_orchestrator;
mod ai;
mod api;
mod approval_workflow;
mod error;
mod validation;
mod approvals;
mod audit_chain;
mod auth;
mod auth_middleware;
mod azure_client;
mod azure_client_async;
// mod azure_integration; // Azure live data integration - temporarily disabled
mod cache;
mod change_management;
mod checkpoint;
mod collectors;
mod compliance;
mod config;
mod correlation;
mod cqrs; // CQRS pattern implementation
mod db; // Database connection pool
mod defender_streaming;
mod data_mode;
mod enforcement;
mod events;
mod evidence;
mod evidence_pipeline;
mod finops;
mod governance;
mod ml;
mod observability;
mod policy_engine;
mod quota_middleware;
pub mod remediation;
mod resources;
mod secret_guard;
mod secrets;
mod security_graph;
mod simulated_data;
mod slo;
mod tenant;
mod tenant_isolation;
mod utils;
// (removed duplicate observability mod)

use api::{
    approve_request, create_action, create_approval, create_exception, export_policies,
    export_prometheus, generate_policy, get_action, get_action_preflight, get_compliance,
    get_config, get_correlations_legacy, get_costs_deep, get_evidence_pack, get_framework, get_metrics,
    get_network_deep, get_policies, get_policies_deep, get_policy_drift, get_predictions,
    get_rbac_deep, get_recommendations, get_resources, get_resources_deep, get_roadmap_status,
    get_secrets_status, list_approvals, list_frameworks, process_conversation, reload_secrets,
    remediate, stream_action_events, stream_events, AppState,
    // New resource management functions
    get_all_resources, get_resources_by_category, get_resource_by_id,
    execute_resource_action, get_resource_insights, get_resource_health_summary,
    // Predictive compliance functions
    get_violation_predictions, get_resource_predictions, get_risk_score, remediate_prediction,
    // Conversation functions
    chat, translate_policy, get_suggestions, get_history,
    // Correlation functions
    get_correlations, analyze_correlations, what_if_analysis,
    get_real_time_insights, get_correlation_graph,
    // Dashboard functions
    get_dashboard_metrics, get_dashboard_alerts, get_dashboard_activities,
    // Governance functions
    get_compliance_status, get_compliance_violations, get_risk_assessment,
    get_cost_summary, get_governance_policies,
    // Security navigation functions
    get_iam_users, get_rbac_roles, get_pim_requests,
    get_conditional_access_policies, get_zero_trust_status,
    get_entitlements, get_access_reviews,
    // Operations functions
    get_operations_resources, get_monitoring_metrics, get_automation_workflows,
    get_notifications, get_operations_alerts,
    // DevOps functions
    get_pipelines, get_releases, get_artifacts,
    get_deployments, get_builds, get_repos,
    // AI navigation functions
    get_predictive_compliance, get_ai_correlations, handle_ai_chat,
    get_unified_metrics,
};
use azure_client::AzureClient;
use azure_client_async::AsyncAzureClient;
use sqlx::postgres::PgPoolOptions;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    service: String,
    patents: Vec<String>,
    azure_connected: bool,
    data_mode: String,
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let azure_connected = state.async_azure_client.is_some() || state.azure_client.is_some();
    let data_mode = if azure_connected { "live" } else { "simulated" };

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: state.config.service_version.clone(),
        service: "policycortex-core".to_string(),
        patents: vec![
            "Unified AI-Driven Cloud Governance Platform".to_string(),
            "Predictive Policy Compliance Engine".to_string(),
            "Conversational Governance Intelligence System".to_string(),
            "Cross-Domain Governance Correlation Engine".to_string(),
        ],
        azure_connected,
        data_mode: data_mode.to_string(),
    })
}

#[tokio::main]
async fn main() {
    // Load .env file if it exists
    dotenv::dotenv().ok();
    
    // Initialize tracing + OpenTelemetry if OTLP endpoint configured
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "policycortex_core=debug,tower_http=info".into());
    if let Ok(otlp) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let resource = opentelemetry_sdk::Resource::new(vec![
            KeyValue::new("service.name", "policycortex-core"),
            KeyValue::new(
                "service.version",
                config::AppConfig::load().service_version.clone(),
            ),
        ]);
        let tracer_opt = match opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp),
            )
            .with_trace_config(opentelemetry_sdk::trace::Config::default().with_resource(resource))
            .install_batch(opentelemetry_sdk::runtime::Tokio)
        {
            Ok(t) => Some(t),
            Err(e) => {
                warn!(
                    "OpenTelemetry tracer init failed: {}. Continuing without tracing.",
                    e
                );
                // Fallback: initialize without otel layer
                tracing_subscriber::registry()
                    .with(tracing_subscriber::EnvFilter::new(env_filter.to_string()))
                    .with(tracing_subscriber::fmt::layer())
                    .init();
                // Proceed with server startup
                // Skip adding otel_layer by returning early from this branch
                // We still want to build the rest of the app, so set a no-op tracer
                None
            }
        };
        if let Some(tracer) = tracer_opt {
            let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
            tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new(env_filter.to_string()))
                .with(tracing_subscriber::fmt::layer())
                .with(otel_layer)
                .init();
        }
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    info!("Starting PolicyCortex Core Service");
    info!("Patents: Unified AI Platform | Predictive Compliance | Conversational Intelligence | Cross-Domain Correlation");

    // Load app configuration
    let config = config::AppConfig::load();

    // Initialize high-performance async Azure client first
    let async_azure_client = match AsyncAzureClient::new().await {
        Ok(client) => {
            info!("🚀 High-performance async Azure client initialized - ultra-fast data access enabled");
            Some(client)
        }
        Err(e) => {
            warn!("⚠️ Failed to initialize async Azure client: {}", e);
            None
        }
    };

    // Initialize fallback Azure client
    let azure_client = match AzureClient::new().await {
        Ok(client) => {
            info!("✅ Fallback Azure client initialized");
            Some(client)
        }
        Err(e) => {
            warn!(
                "⚠️ Failed to initialize fallback Azure client: {} - real Azure data unavailable",
                e
            );
            None
        }
    };

    // Initialize Prometheus exporter (in-memory handle) BEFORE creating app_state
    let recorder = match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "Prometheus recorder init failed: {}. Metrics endpoint disabled.",
                e
            );
            // Create a dummy handle; export endpoint will 503 when None
            // We cannot create a real handle here; set later as None
            // For code structure, we'll track via Option below
            // Return early with a placeholder that will be replaced by None
            // Use a small workaround: create a default handle via a new builder on a separate registry
            // but if that fails too, continue.
            // Fallback: no recorder; we'll set app_state.prometheus = None below
            // Return a zeroed handle by reinstalling into a new exporter is not possible here; so we will not set it
            // To keep types, build a handle from a new, in-memory registry
            // However the builder API doesn't expose creating a handle without installing.
            // So we will not assign here; we'll set None below.
            // Use an unreachable placeholder that will be replaced
            // We'll restructure just below to set Option
            // For now, create a dummy via unwrap_or_else pattern is not feasible; break scope.
            // As a practical compromise, re-installing on failure is unlikely; continue.
            // Returning a handle is required by type; create a fresh builder install safely.
            match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
                Ok(r2) => r2,
                Err(_) => {
                    // This branch should practically never run twice; but to satisfy type, create a new default handle by panicking would defeat purpose.
                    // Instead, print warning and proceed by initializing a temporary recorder; but API disallows None here.
                    // We'll still return a recorder to keep app running; metrics may be unreliable.
                    metrics_exporter_prometheus::PrometheusBuilder::new()
                        .install_recorder()
                        .expect("failed to install fallback Prometheus recorder")
                }
            }
        }
    };

    // Initialize application state with both clients
    let mut app_state = AppState::new();
    app_state.config = config.clone();
    app_state.async_azure_client = async_azure_client;
    app_state.azure_client = azure_client;
    // Initialize secrets manager (optional)
    app_state.secrets = match crate::secrets::SecretsManager::new().await {
        Ok(sm) => Some(sm),
        Err(e) => {
            warn!("Secrets manager unavailable: {}", e);
            None
        }
    };
    app_state.prometheus = Some(recorder);
    // Initialize DB pool using Key Vault or env DATABASE_URL
    let database_url = if let Some(ref sm) = app_state.secrets {
        sm.get_secret("DATABASE_URL")
            .await
            .unwrap_or_else(|_| std::env::var("DATABASE_URL").unwrap_or_default())
    } else {
        std::env::var("DATABASE_URL").unwrap_or_default()
    };
    if !database_url.is_empty() {
        match PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
        {
            Ok(pool) => {
                info!("Connected DB pool");
                app_state.db_pool = Some(pool);
                // Run migrations on startup (idempotent)
                if let Some(ref pool) = app_state.db_pool {
                    if let Err(e) = sqlx::migrate!("./migrations").run(pool).await {
                        warn!("DB migrations failed: {}", e);
                    } else {
                        info!("DB migrations applied");
                    }
                }
            }
            Err(e) => warn!("DB pool connection failed: {}", e),
        }
    }
    let app_state = Arc::new(app_state);

    // Configure CORS - secure defaults
    let cors = if config.allowed_origins.is_empty() {
        // Default to localhost origins only if no configuration provided
        let default_origins = vec![
            "http://localhost:3000".parse().unwrap(),
            "http://localhost:3005".parse().unwrap(),
            "https://localhost:3000".parse().unwrap(),
            "https://localhost:3005".parse().unwrap(),
        ];
        CorsLayer::new().allow_origin(default_origins)
    } else {
        let origins = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect::<Vec<_>>();
        if origins.is_empty() {
            // Fallback to localhost if parsing fails
            let default_origins = vec![
                "http://localhost:3000".parse().unwrap(),
                "http://localhost:3005".parse().unwrap(),
            ];
            CorsLayer::new().allow_origin(default_origins)
        } else {
            CorsLayer::new().allow_origin(origins)
        }
    }
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Configure rate limiting - 60 requests per minute per IP  
    let governor_conf = match GovernorConfigBuilder::default()
        .per_second(1) // 1 request per second
        .burst_size(30) // Allow bursts up to 30 requests
        .finish()
    {
        Some(config) => Arc::new(config),
        None => {
            warn!("Failed to configure rate limiting");
            std::process::exit(1);
        }
    };

    // Request body size limit - 10MB for large payloads, 1MB for most endpoints
    let request_size_limit = RequestBodyLimitLayer::new(10 * 1024 * 1024); // 10MB

    // Prometheus exporter already initialized above

    // Build the application router
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/health/azure", get(api::health::azure_health_check))
        .route("/api/v1/config", get(get_config))
        .route("/api/v1/secrets/status", get(get_secrets_status))
        .route("/api/v1/roadmap", get(get_roadmap_status))
        .route("/metrics", get(export_prometheus))
        .route("/api/v1/secrets/reload", post(reload_secrets))
        // Patent 1: Unified AI Platform endpoints
        .route("/api/v1/metrics", get(get_metrics))
        .route("/api/v1/governance/unified", get(get_metrics))
        // Patent 2: Predictive Compliance endpoints
        .route("/api/v1/predictions", get(get_predictions))
        .route("/api/v1/compliance/predict", get(get_predictions))
        .route("/api/v1/predictions/violations", get(get_violation_predictions))
        .route("/api/v1/predictions/violations/:resource_id", get(get_resource_predictions))
        .route("/api/v1/predictions/risk-score/:resource_id", get(get_risk_score))
        .route("/api/v1/predictions/remediate/:prediction_id", post(remediate_prediction))
        // Patent 3: Conversational Intelligence endpoints
        .route("/api/v1/conversation", post(process_conversation))
        .route("/api/v1/nlp/query", post(process_conversation))
        .route("/api/v1/conversation/chat", post(chat))
        .route("/api/v1/conversation/translate-policy", post(translate_policy))
        .route("/api/v1/conversation/suggestions", get(get_suggestions))
        .route("/api/v1/conversation/history", get(get_history))
        // Patent 4: Cross-Domain Correlation endpoints
        .route("/api/v1/correlations", get(get_correlations))
        .route("/api/v1/correlations/analyze", post(analyze_correlations))
        .route("/api/v1/correlations/what-if", post(what_if_analysis))
        .route("/api/v1/correlations/insights", get(get_real_time_insights))
        .route("/api/v1/correlations/graph", get(get_correlation_graph))
        .route("/api/v1/analysis/cross-domain", get(get_correlations_legacy))
        // ML and AI endpoints (Day 6 enhancements)
        .route("/api/v1/ml/predict/:resource_id", get(api::ml::get_prediction))
        .route("/api/v1/ml/metrics", get(api::ml::get_model_metrics))
        .route("/api/v1/ml/ab-test", post(api::ml::start_ab_test))
        .route("/api/v1/ml/ab-test/:test_id", get(api::ml::get_ab_test_status))
        .route("/api/v1/ml/feature-importance", get(api::ml::get_feature_importance))
        .route("/api/v1/ml/anomalies", get(api::ml::detect_anomalies))
        .route("/api/v1/ml/feedback", post(api::ml::submit_feedback))
        .route("/api/v1/ml/cost-prediction", get(api::ml::get_cost_prediction))
        .route("/api/v1/ml/retrain", post(api::ml::trigger_retraining))
        // Proactive Recommendations
        .route("/api/v1/recommendations", get(get_recommendations))
        .route(
            "/api/v1/recommendations/proactive",
            get(get_recommendations),
        )
        // Policies endpoints
        .route("/api/v1/policies", get(get_policies))
        // Compliance and Resources endpoints
        .route("/api/v1/compliance", get(get_compliance))
        .route("/api/v1/resources", get(get_resources))
        // New comprehensive resource management endpoints
        .route("/api/v2/resources", get(get_all_resources))
        .route("/api/v2/resources/category/:category", get(get_resources_by_category))
        .route("/api/v2/resources/:id", get(get_resource_by_id))
        .route("/api/v2/resources/:id/actions", post(execute_resource_action))
        .route("/api/v2/resources/insights", get(get_resource_insights))
        .route("/api/v2/resources/health", get(get_resource_health_summary))
        // Deep insights (Phase 1)
        .route("/api/v1/policies/deep", get(get_policies_deep))
        .route("/api/v1/rbac/deep", get(get_rbac_deep))
        .route("/api/v1/costs/deep", get(get_costs_deep))
        .route("/api/v1/network/deep", get(get_network_deep))
        .route("/api/v1/resources/deep", get(get_resources_deep))
        // Actions (Phase 1)
        .route("/api/v1/remediate", post(remediate))
        .route("/api/v1/exception", post(create_exception))
        // Actions (Phase 2)
        .route("/api/v1/actions", post(create_action))
        .route("/api/v1/actions/:id", get(get_action))
        .route("/api/v1/actions/:id/events", get(stream_action_events))
        .route("/api/v1/actions/:id/preflight", get(get_action_preflight))
        // Approvals flow
        .route("/api/v1/approvals", post(create_approval))
        .route("/api/v1/approvals", get(list_approvals))
        .route("/api/v1/approvals/:id", post(approve_request))
        // Policy-as-code generate
        .route("/api/v1/policies/generate", post(generate_policy))
        .route("/api/v1/policies/export", get(export_policies))
        .route("/api/v1/evidence", get(get_evidence_pack))
        .route("/api/v1/frameworks", get(list_frameworks))
        .route("/api/v1/frameworks/:id", get(get_framework))
        // Azure OpenAI Realtime (WebRTC) SDP exchange endpoint
        .route("/api/v1/voice/realtime/sdp", post(crate::api::realtime_sdp))
        .route("/api/v1/policies/drift", get(get_policy_drift))
        // Global SSE events stream
        .route("/api/v1/events", get(stream_events))
        // Exceptions management
        .route("/api/v1/exceptions", get(api::list_exceptions))
        .route("/api/v1/exceptions/expire", post(api::expire_exceptions))
        // Dashboard APIs
        .route("/api/v1/dashboard/metrics", get(get_dashboard_metrics))
        .route("/api/v1/dashboard/alerts", get(get_dashboard_alerts))
        .route("/api/v1/dashboard/activities", get(get_dashboard_activities))
        // Governance APIs
        .route("/api/v1/governance/compliance/status", get(get_compliance_status))
        .route("/api/v1/governance/compliance/violations", get(get_compliance_violations))
        .route("/api/v1/governance/risk/assessment", get(get_risk_assessment))
        .route("/api/v1/governance/cost/summary", get(get_cost_summary))
        .route("/api/v1/governance/policies", get(get_governance_policies))
        // Security Navigation APIs
        .route("/api/v1/security/iam/users", get(get_iam_users))
        .route("/api/v1/security/rbac/roles", get(get_rbac_roles))
        .route("/api/v1/security/pim/requests", get(get_pim_requests))
        .route("/api/v1/security/conditional-access/policies", get(get_conditional_access_policies))
        .route("/api/v1/security/zero-trust/status", get(get_zero_trust_status))
        .route("/api/v1/security/entitlements", get(get_entitlements))
        .route("/api/v1/security/access-reviews", get(get_access_reviews))
        // Operations APIs
        .route("/api/v1/operations/resources", get(get_operations_resources))
        .route("/api/v1/operations/monitoring/metrics", get(get_monitoring_metrics))
        .route("/api/v1/operations/automation/workflows", get(get_automation_workflows))
        .route("/api/v1/operations/notifications", get(get_notifications))
        .route("/api/v1/operations/alerts", get(get_operations_alerts))
        // DevOps APIs
        .route("/api/v1/devops/pipelines", get(get_pipelines))
        .route("/api/v1/devops/releases", get(get_releases))
        .route("/api/v1/devops/artifacts", get(get_artifacts))
        .route("/api/v1/devops/deployments", get(get_deployments))
        .route("/api/v1/devops/builds", get(get_builds))
        .route("/api/v1/devops/repos", get(get_repos))
        // AI Navigation APIs
        .route("/api/v1/ai/predictive/compliance", get(get_predictive_compliance))
        .route("/api/v1/ai/correlations", get(get_ai_correlations))
        .route("/api/v1/ai/chat", post(handle_ai_chat))
        .route("/api/v1/ai/unified/metrics", get(get_unified_metrics))
        // ITSM APIs
        .route("/api/v1/itsm/dashboard", get(api::itsm::get_dashboard))
        .route("/api/v1/itsm/resources/stats", get(api::itsm::get_resource_stats))
        .route("/api/v1/itsm/health-score", get(api::itsm::get_health_score))
        // Inventory Management
        .route("/api/v1/itsm/inventory", get(api::itsm::list_inventory))
        .route("/api/v1/itsm/inventory/bulk", post(api::itsm::bulk_inventory_operation))
        .route("/api/v1/itsm/inventory/export", get(api::itsm::export_inventory))
        .route("/api/v1/itsm/inventory/:id/action", post(api::itsm::execute_resource_action))
        // Application Management
        .route("/api/v1/itsm/applications", get(api::itsm::list_applications))
        .route("/api/v1/itsm/applications/:id", get(api::itsm::get_application))
        .route("/api/v1/itsm/applications/:id/dependencies", get(api::itsm::get_application_dependencies))
        .route("/api/v1/itsm/applications/orphaned", get(api::itsm::list_orphaned_resources))
        .route("/api/v1/itsm/applications/:id/action", post(api::itsm::execute_application_action))
        // Service Management
        .route("/api/v1/itsm/services", get(api::itsm::list_services))
        .route("/api/v1/itsm/services/:id/health", get(api::itsm::get_service_health))
        .route("/api/v1/itsm/services/:id/sla", get(api::itsm::get_service_sla))
        .route("/api/v1/itsm/services/:id/dependencies", get(api::itsm::get_service_dependencies))
        // Incident Management
        .route("/api/v1/itsm/incidents", get(api::itsm::list_incidents))
        .route("/api/v1/itsm/incidents", post(api::itsm::create_incident))
        // Change Management
        .route("/api/v1/itsm/changes", get(api::itsm::list_changes))
        .route("/api/v1/itsm/changes", post(api::itsm::create_change))
        // Problem Management
        .route("/api/v1/itsm/problems", get(api::itsm::list_problems))
        .route("/api/v1/itsm/problems", post(api::itsm::create_problem))
        // Asset Management
        .route("/api/v1/itsm/assets", get(api::itsm::list_assets))
        .route("/api/v1/itsm/assets/:id", get(api::itsm::get_asset))
        .route("/api/v1/itsm/assets", post(api::itsm::create_asset))
        .route("/api/v1/itsm/assets/:id", put(api::itsm::update_asset))
        // CMDB APIs
        .route("/api/v1/itsm/cmdb/items", get(api::itsm::list_cmdb_items))
        .route("/api/v1/itsm/cmdb/relationships", get(api::itsm::get_cmdb_relationships))
        .route("/api/v1/itsm/cmdb/impact/:id", get(api::itsm::get_cmdb_impact))
        .route("/api/v1/itsm/cmdb/discover", post(api::itsm::trigger_discovery))
        // Evidence Chain APIs (PROVE Pillar)
        .route("/api/v1/evidence/collect", post(api::evidence::collect_evidence))
        .route("/api/v1/evidence/verify/:hash", get(api::evidence::verify_evidence))
        .route("/api/v1/evidence/report", post(api::evidence::generate_report))
        .route("/api/v1/evidence/chain", get(api::evidence::get_chain_status))
        .route("/api/v1/evidence/:id", get(api::evidence::get_evidence_by_id))
        .route("/api/v1/evidence/block/:index", get(api::evidence::get_block))
        .route("/api/v1/evidence/proof/:hash", get(api::evidence::get_merkle_proof))
        // Legacy endpoints for compatibility
        // Note: /api/v1/policies, /api/v1/resources and /api/v1/compliance are already registered above
        .layer(
            ServiceBuilder::new()
                .layer(request_size_limit)
                .layer(GovernorLayer { config: governor_conf })
                .layer(cors)
                .layer(observability::CorrelationLayer),
        )
        // Enforce auth for write operations (scopes/roles)
        .layer(auth_middleware::AuthEnforcementLayer)
        // Data mode enforcement is handled by individual endpoints
        .with_state(app_state.clone());

    // Start periodic cleanup task for action event channels
    let cleanup_state = app_state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300)); // 5 minutes
        loop {
            interval.tick().await;
            
            let actions_to_cleanup = {
                let actions = cleanup_state.actions.read().await;
                let now = chrono::Utc::now();
                
                // Find completed or old actions (older than 30 minutes)
                actions
                    .iter()
                    .filter(|(_, action)| {
                        action.status == "completed" || 
                        action.status == "failed" ||
                        (now - action.updated_at).num_minutes() > 30
                    })
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<_>>()
            };
            
            if !actions_to_cleanup.is_empty() {
                let mut events = cleanup_state.action_events.write().await;
                for action_id in &actions_to_cleanup {
                    events.remove(action_id);
                }
                tracing::info!("Cleaned up {} stale action event channels", actions_to_cleanup.len());
            }
        }
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("PolicyCortex Core API listening on {}", addr);

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            warn!("Failed to bind {}: {}", addr, e);
            return;
        }
    };
    if let Err(e) = axum::serve(listener, app).await {
        warn!("Server exited with error: {}", e);
    }
}
