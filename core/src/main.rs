// PCG Platform - Predictive Cloud Governance
// Focus on three pillars: PREVENT, PROVE, PAYBACK
// Patent #4: Predictive Policy Compliance Engine
// © 2024 PolicyCortex. All rights reserved.

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
use tower_http::limit::RequestBodyLimitLayer;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Core modules only - focused on PCG pillars
mod api;
mod auth;
mod auth_middleware;
mod azure_client_async;
mod cache;
mod config;
mod correlation;
mod cqrs;
mod data_mode;
mod db;
mod error;
mod evidence;
mod evidence_pipeline;
mod ml;
mod observability;
mod policy_engine;
mod remediation;
mod resources;
mod secrets;
mod simulated_data;
mod tenant;
mod tenant_isolation;
mod utils;
mod validation;

use api::{
    AppState,
    // Health & Config
    export_prometheus, get_config,
    // PREVENT Pillar - Predictive Compliance (Patent #4)
    get_predictions, get_violation_predictions, get_resource_predictions,
    get_risk_score, remediate_prediction,
    // PROVE Pillar - Evidence Chain
    get_evidence_pack,
    // PAYBACK Pillar - ROI & Cost Optimization
    get_correlations_legacy as get_roi_correlations,
    // ML endpoints for predictions
};
use azure_client_async::AsyncAzureClient;
use sqlx::postgres::PgPoolOptions;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    service: String,
    platform: String,
    pillars: Vec<String>,
    data_mode: String,
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let data_mode = if state.async_azure_client.is_some() { "live" } else { "simulated" };

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: state.config.service_version.clone(),
        service: "pcg-core".to_string(),
        platform: "Predictive Cloud Governance (PCG)".to_string(),
        pillars: vec![
            "PREVENT - Predictive Compliance".to_string(),
            "PROVE - Evidence Chain".to_string(),
            "PAYBACK - ROI Optimization".to_string(),
        ],
        data_mode: data_mode.to_string(),
    })
}

#[tokio::main]
async fn main() {
    // Load .env file if it exists
    dotenv::dotenv().ok();
    
    // Initialize tracing
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "pcg_core=debug,tower_http=info".into());
    
    if let Ok(otlp) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let resource = opentelemetry_sdk::Resource::new(vec![
            KeyValue::new("service.name", "pcg-core"),
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
                warn!("OpenTelemetry init failed: {}. Continuing without tracing.", e);
                None
            }
        };
        
        if let Some(tracer) = tracer_opt {
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
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    info!("Starting PCG Core Service - Predictive Cloud Governance Platform");
    info!("Three Pillars: PREVENT | PROVE | PAYBACK");

    // Load configuration
    let config = config::AppConfig::load();

    // Initialize Azure client for live data
    let async_azure_client = match AsyncAzureClient::new().await {
        Ok(client) => {
            info!("✅ Azure client initialized - live data enabled");
            Some(client)
        }
        Err(e) => {
            warn!("⚠️ Azure client unavailable: {} - using simulated data", e);
            None
        }
    };

    // Initialize Prometheus metrics
    let recorder = match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
        Ok(r) => Some(r),
        Err(e) => {
            warn!("Prometheus metrics unavailable: {}", e);
            None
        }
    };

    // Initialize application state
    let mut app_state = AppState::new();
    app_state.config = config.clone();
    app_state.async_azure_client = async_azure_client;
    app_state.prometheus = recorder;
    
    // Initialize secrets manager
    app_state.secrets = match crate::secrets::SecretsManager::new().await {
        Ok(sm) => Some(sm),
        Err(e) => {
            warn!("Secrets manager unavailable: {}", e);
            None
        }
    };
    
    // Initialize database pool
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
                info!("Database connected");
                app_state.db_pool = Some(pool);
                
                // Run migrations
                if let Some(ref pool) = app_state.db_pool {
                    if let Err(e) = sqlx::migrate!("./migrations").run(pool).await {
                        warn!("Database migrations failed: {}", e);
                    } else {
                        info!("Database migrations applied");
                    }
                }
            }
            Err(e) => warn!("Database connection failed: {}", e),
        }
    }
    
    let app_state = Arc::new(app_state);

    // Configure CORS
    let cors = if config.allowed_origins.is_empty() {
        let default_origins = vec![
            "http://localhost:3000".parse().unwrap(),
            "http://localhost:3005".parse().unwrap(),
        ];
        CorsLayer::new().allow_origin(default_origins)
    } else {
        let origins = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect::<Vec<_>>();
        if origins.is_empty() {
            let default_origins = vec!["http://localhost:3000".parse().unwrap()];
            CorsLayer::new().allow_origin(default_origins)
        } else {
            CorsLayer::new().allow_origin(origins)
        }
    }
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Configure rate limiting - 60 requests per minute
    let governor_conf = match GovernorConfigBuilder::default()
        .per_second(1)
        .burst_size(30)
        .finish()
    {
        Some(config) => Arc::new(config),
        None => {
            warn!("Failed to configure rate limiting");
            std::process::exit(1);
        }
    };

    // Request body size limit - 10MB
    let request_size_limit = RequestBodyLimitLayer::new(10 * 1024 * 1024);

    // Build the PCG application router - focused on three pillars
    let app = Router::new()
        // === Core Health & Metrics ===
        .route("/health", get(health_check))
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/config", get(get_config))
        .route("/metrics", get(export_prometheus))
        
        // === PREVENT PILLAR: Predictive Compliance (Patent #4) ===
        .route("/api/v1/predict/compliance", get(get_predictions))
        .route("/api/v1/predict/violations", get(get_violation_predictions))
        .route("/api/v1/predict/violations/:resource_id", get(get_resource_predictions))
        .route("/api/v1/predict/risk-score/:resource_id", get(get_risk_score))
        .route("/api/v1/predict/remediate/:prediction_id", post(remediate_prediction))
        
        // ML endpoints for predictive analytics
        .route("/api/v1/ml/predict/:resource_id", get(api::ml::get_prediction))
        .route("/api/v1/ml/metrics", get(api::ml::get_model_metrics))
        .route("/api/v1/ml/anomalies", get(api::ml::detect_anomalies))
        .route("/api/v1/ml/cost-prediction", get(api::ml::get_cost_prediction))
        .route("/api/v1/ml/feedback", post(api::ml::submit_feedback))
        
        // === PROVE PILLAR: Evidence Chain ===
        .route("/api/v1/evidence", get(get_evidence_pack))
        .route("/api/v1/evidence/collect", post(api::evidence::collect_evidence))
        .route("/api/v1/evidence/verify/:hash", get(api::evidence::verify_evidence))
        .route("/api/v1/evidence/report", post(api::evidence::generate_report))
        .route("/api/v1/evidence/chain", get(api::evidence::get_chain_status))
        .route("/api/v1/evidence/:id", get(api::evidence::get_evidence_by_id))
        
        // === PAYBACK PILLAR: ROI & Cost Optimization ===
        .route("/api/v1/roi/correlations", get(get_roi_correlations))
        .route("/api/v1/roi/cost-savings", get(api::ml::get_cost_prediction))
        .route("/api/v1/roi/optimization", post(api::correlations::analyze_correlations))
        .route("/api/v1/roi/insights", get(api::correlations::get_real_time_insights))
        
        // === Core Resource Management (supporting all pillars) ===
        .route("/api/v1/resources", get(api::get_resources))
        .route("/api/v1/resources/:id", get(api::resources::get_resource_by_id))
        .route("/api/v1/resources/:id/actions", post(api::resources::execute_resource_action))
        .route("/api/v1/resources/insights", get(api::resources::get_resource_insights))
        
        // === Core Compliance (required for evidence) ===
        .route("/api/v1/compliance", get(api::get_compliance))
        .route("/api/v1/policies", get(api::get_policies))
        
        .layer(
            ServiceBuilder::new()
                .layer(request_size_limit)
                .layer(GovernorLayer { config: governor_conf })
                .layer(cors)
                .layer(observability::CorrelationLayer),
        )
        .layer(auth_middleware::AuthEnforcementLayer)
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("PCG Core API listening on {}", addr);

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