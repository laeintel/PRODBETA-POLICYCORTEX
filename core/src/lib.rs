// Library interface for PolicyCortex Core
// This enables integration tests to access internal modules

pub mod remediation;
pub mod api;
pub mod ml;
pub mod governance;
pub mod correlation;
pub mod auth;
pub mod error;
pub mod config;
pub mod azure;
pub mod azure_client;
pub mod azure_client_async;
pub mod azure_integration;
pub mod resources;
pub mod validation;
pub mod secrets;
pub mod slo;
pub mod simulated_data;
pub mod data_mode;
pub mod utils;
pub mod cache;
pub mod ai;

// Re-export the main app state
pub use api::AppState;