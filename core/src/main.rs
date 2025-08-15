use axum::{
    extract::State,
    http::{header, Method},
    routing::{get, post},
    Json, Router,
};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
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
mod governance;
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
        // Azure OpenAI Realtime (WebRTC) SDP exchange endpoint
        .route("/api/v1/voice/realtime/sdp", post(crate::api::realtime_sdp))
        .route("/api/v1/policies/drift", get(get_policy_drift))
        // Global SSE events stream
        .route("/api/v1/events", get(stream_events))
        // Exceptions management
        .route("/api/v1/exceptions", get(api::list_exceptions))
        .route("/api/v1/exceptions/expire", post(api::expire_exceptions))
        // Legacy endpoints for compatibility
        // Note: /api/v1/policies, /api/v1/resources and /api/v1/compliance are already registered above
        .layer(
            ServiceBuilder::new()
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
