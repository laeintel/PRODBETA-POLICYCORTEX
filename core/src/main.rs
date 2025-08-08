use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    // Add your application state here
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "policycortex_core=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting PolicyCortex v2 Core Service");

    // Initialize application state
    let state = AppState {};

    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/policies", get(get_policies))
        .route("/api/v1/policies", post(create_policy))
        .route("/api/v1/resources", get(get_resources))
        .route("/api/v1/compliance", get(get_compliance))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Run our application
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("PolicyCortex Core listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    service: String,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "2.0.0".to_string(),
        service: "policycortex-core".to_string(),
    })
}

#[derive(Serialize)]
struct Policy {
    id: String,
    name: String,
    category: String,
    severity: String,
    status: String,
}

async fn get_policies() -> Json<Vec<Policy>> {
    // Mock data for now
    let policies = vec![
        Policy {
            id: "pol-1".to_string(),
            name: "Require HTTPS for Storage".to_string(),
            category: "Security".to_string(),
            severity: "High".to_string(),
            status: "Active".to_string(),
        },
        Policy {
            id: "pol-2".to_string(),
            name: "Enforce Tagging Standards".to_string(),
            category: "Governance".to_string(),
            severity: "Medium".to_string(),
            status: "Active".to_string(),
        },
    ];
    Json(policies)
}

#[derive(Deserialize)]
struct CreatePolicyRequest {
    name: String,
    category: String,
    severity: String,
}

async fn create_policy(Json(payload): Json<CreatePolicyRequest>) -> StatusCode {
    info!("Creating policy: {}", payload.name);
    StatusCode::CREATED
}

#[derive(Serialize)]
struct Resource {
    id: String,
    name: String,
    resource_type: String,
    location: String,
}

async fn get_resources() -> Json<Vec<Resource>> {
    let resources = vec![
        Resource {
            id: "res-1".to_string(),
            name: "stcontoso01".to_string(),
            resource_type: "Storage Account".to_string(),
            location: "East US".to_string(),
        },
    ];
    Json(resources)
}

#[derive(Serialize)]
struct ComplianceResult {
    policy_id: String,
    resource_id: String,
    status: String,
    checked_at: String,
}

async fn get_compliance() -> Json<Vec<ComplianceResult>> {
    let results = vec![
        ComplianceResult {
            policy_id: "pol-1".to_string(),
            resource_id: "res-1".to_string(),
            status: "Compliant".to_string(),
            checked_at: "2024-01-15T10:00:00Z".to_string(),
        },
    ];
    Json(results)
}