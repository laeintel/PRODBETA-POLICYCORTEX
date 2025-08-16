// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use axum::{routing::get, Router};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Build application
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .layer(CorsLayer::permissive());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
