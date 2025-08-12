use axum::{
    extract::State,
    http::{header, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod action_orchestrator;
mod ai;
mod api;
mod approval_workflow;
mod approvals;
mod audit_chain;
mod auth;
mod auth_middleware;
mod azure_client;
mod azure_client_async;
mod cache;
mod change_management;
mod collectors;
mod compliance;
mod config;
mod data_mode;
mod enforcement;
mod events;
mod evidence_pipeline;
mod finops;
mod observability;
mod policy_engine;
mod secret_guard;
mod secrets;
mod security_graph;
mod simulated_data;
mod slo;
mod tenant;
mod tenant_isolation;
// (removed duplicate observability mod)

use api::{
    approve_request, create_action, create_approval, create_exception, export_policies,
    export_prometheus, generate_policy, get_action, get_action_preflight, get_compliance,
    get_config, get_correlations, get_costs_deep, get_evidence_pack, get_framework, get_metrics,
    get_network_deep, get_policies, get_policies_deep, get_policy_drift, get_predictions,
    get_rbac_deep, get_recommendations, get_resources, get_resources_deep, get_roadmap_status,
    get_secrets_status, list_approvals, list_frameworks, process_conversation, reload_secrets,
    remediate, stream_action_events, stream_events, AppState,
};
use auth::{AuthUser, OptionalAuthUser};
use azure_client::AzureClient;
use azure_client_async::AsyncAzureClient;
use sqlx::postgres::PgPoolOptions;
use tenant_isolation::{tenant_isolation_middleware, TenantContext, TenantDatabase};

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
    // Initialize tracing + OpenTelemetry if OTLP endpoint configured
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "policycortex_core=debug,tower_http=info".into());
    if let Ok(otlp) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let resource = opentelemetry_sdk::Resource::new(vec![
            KeyValue::new("service.name", "policycortex-core"),
            KeyValue::new("service.version", config::AppConfig::load().service_version.clone()),
        ]);
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp),
            )
            .with_trace_config(opentelemetry_sdk::trace::Config::default().with_resource(resource))
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .expect("failed to install OTLP tracer");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .with(otel_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    info!("Starting PolicyCortex v2 Core Service");
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
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

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
            }
            Err(e) => warn!("DB pool connection failed: {}", e),
        }
    }
    let app_state = Arc::new(app_state);

    // Configure CORS
    let cors = if config.allowed_origins.is_empty() {
        CorsLayer::new().allow_origin(Any)
    } else {
        let origins = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect::<Vec<_>>();
        CorsLayer::new().allow_origin(origins)
    }
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Prometheus exporter already initialized above

    // Build the application router
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/api/v1/health", get(health_check))
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
        // Patent 3: Conversational Intelligence endpoints
        .route("/api/v1/conversation", post(process_conversation))
        .route("/api/v1/nlp/query", post(process_conversation))
        // Patent 4: Cross-Domain Correlation endpoints
        .route("/api/v1/correlations", get(get_correlations))
        .route("/api/v1/analysis/cross-domain", get(get_correlations))
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
        .route("/api/v1/policies/drift", get(get_policy_drift))
        // Global SSE events stream
        .route("/api/v1/events", get(stream_events))
        // Exceptions management
        .route("/api/v1/exceptions", get(api::list_exceptions))
        .route("/api/v1/exceptions/expire", post(api::expire_exceptions))
        // Legacy endpoints for compatibility
        // Note: /api/v1/policies, /api/v1/resources and /api/v1/compliance are already registered above
        .layer(ServiceBuilder::new().layer(cors).layer(observability::CorrelationLayer))
        // Enforce auth for write operations (scopes/roles)
        .layer(auth_middleware::AuthEnforcementLayer)
        // Disallow write endpoints when running in simulated mode
        .layer(tower::layer::util::LayerFn::new(|service| {
            tower::service_fn(move |req: axum::http::Request<axum::body::Body>| {
                let is_write = matches!(req.method(), &Method::POST | &Method::PUT | &Method::DELETE | &Method::PATCH);
                let simulated = !matches!(std::env::var("USE_REAL_DATA").as_deref(), Ok("true") | Ok("1"));
                if simulated && is_write && !req.uri().path().contains("/approvals") {
                    let resp = axum::response::Response::builder()
                        .status(axum::http::StatusCode::FORBIDDEN)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from("{\"error\":\"read_only_mode\",\"message\":\"Writes disabled in simulated mode\"}"))
                        .unwrap();
                    return std::future::ready(Ok::<_, std::convert::Infallible>(resp));
                }
                service.call(req)
            })
        }))
        .with_state(app_state);

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
