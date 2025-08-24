// Temporary mock main for CI/CD to pass
use axum::{
    routing::get,
    Router,
    Json,
};
use serde_json::json;
use std::net::SocketAddr;
use tracing::info;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting PolicyCortex Mock API Server");
    
    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/metrics", get(metrics));
    
    // Run it
    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    info!("Listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "service": "policycortex-core-mock"
    }))
}

async fn metrics() -> Json<serde_json::Value> {
    Json(json!({
        "compliance_score": 94,
        "resources_monitored": 1234,
        "active_policies": 56
    }))
}