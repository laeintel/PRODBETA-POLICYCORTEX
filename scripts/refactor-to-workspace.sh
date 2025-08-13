#!/bin/bash
set -e

echo "🔧 Refactoring Rust core to workspace structure..."

# Create workspace structure
mkdir -p crates/{api,auth,evidence,orchestration,shared,models}

# Create workspace Cargo.toml
cat > Cargo.toml << 'EOF'
[workspace]
resolver = "2"
members = [
    "crates/api",
    "crates/auth",
    "crates/evidence",
    "crates/orchestration",
    "crates/shared",
    "crates/models"
]

[workspace.dependencies]
# Async runtime
tokio = { version = "1.38", features = ["full"] }
axum = { version = "0.7", features = ["ws", "macros"] }
tower = { version = "0.4", features = ["full"] }
tower-http = { version = "0.5", features = ["fs", "cors", "trace"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres", "json", "uuid", "chrono"] }

# Azure
azure_identity = "0.20"
azure_mgmt_monitor = "0.20"
azure_mgmt_compute = "0.20"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Utils
uuid = { version = "1.8", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
dotenv = "0.15"

[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
EOF

# Create shared crate for common types
cat > crates/shared/Cargo.toml << 'EOF'
[package]
name = "policycortex-shared"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
thiserror = { workspace = true }
EOF

# Create models crate
cat > crates/models/Cargo.toml << 'EOF'
[package]
name = "policycortex-models"
version = "0.1.0"
edition = "2021"

[dependencies]
policycortex-shared = { path = "../shared" }
serde = { workspace = true }
serde_json = { workspace = true }
sqlx = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
EOF

# Create auth crate
cat > crates/auth/Cargo.toml << 'EOF'
[package]
name = "policycortex-auth"
version = "0.1.0"
edition = "2021"

[dependencies]
policycortex-shared = { path = "../shared" }
policycortex-models = { path = "../models" }
axum = { workspace = true }
tower = { workspace = true }
azure_identity = { workspace = true }
jsonwebtoken = "9.2"
EOF

# Create evidence crate
cat > crates/evidence/Cargo.toml << 'EOF'
[package]
name = "policycortex-evidence"
version = "0.1.0"
edition = "2021"

[dependencies]
policycortex-shared = { path = "../shared" }
policycortex-models = { path = "../models" }
sqlx = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
EOF

# Create orchestration crate
cat > crates/orchestration/Cargo.toml << 'EOF'
[package]
name = "policycortex-orchestration"
version = "0.1.0"
edition = "2021"

[dependencies]
policycortex-shared = { path = "../shared" }
policycortex-models = { path = "../models" }
policycortex-evidence = { path = "../evidence" }
tokio = { workspace = true }
anyhow = { workspace = true }
EOF

# Create API crate (main application)
cat > crates/api/Cargo.toml << 'EOF'
[package]
name = "policycortex-api"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "policycortex"
path = "src/main.rs"

[dependencies]
policycortex-shared = { path = "../shared" }
policycortex-models = { path = "../models" }
policycortex-auth = { path = "../auth" }
policycortex-evidence = { path = "../evidence" }
policycortex-orchestration = { path = "../orchestration" }

axum = { workspace = true }
tower = { workspace = true }
tower-http = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
dotenv = { workspace = true }
EOF

# Create lib.rs files for each crate
for crate in api auth evidence orchestration shared models; do
    mkdir -p crates/$crate/src
    echo "// $crate module" > crates/$crate/src/lib.rs
done

# Create main.rs for API crate
cat > crates/api/src/main.rs << 'EOF'
use axum::{Router, routing::get};
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
EOF

echo "✅ Workspace structure created!"
echo ""
echo "Next steps:"
echo "1. Move existing code from core/ to appropriate crates/"
echo "2. Update imports to use workspace crates"
echo "3. Run 'cargo build' to verify compilation"
echo "4. Update Dockerfile to build from crates/api"
echo ""
echo "Benefits:"
echo "- Parallel compilation of crates"
echo "- Better code organization"
echo "- Faster incremental builds"
echo "- Independent testing of components"