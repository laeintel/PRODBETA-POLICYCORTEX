// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

//! PolicyCortex Core Library
//! Production-grade backend services for AI-powered Azure governance

// Core modules
pub mod remediation;
pub mod api;
pub mod ml;
pub mod governance;
pub mod correlation;
pub mod auth;
pub mod error;
pub mod evidence;
pub mod config;
pub mod azure;
pub mod azure_client;
pub mod azure_client_async;
// pub mod azure_integration; // Temporarily commented out due to import issues
pub mod resources;
pub mod validation;
pub mod secrets;
pub mod slo;
pub mod simulated_data;
pub mod data_mode;
pub mod cache;
pub mod utils;
pub mod ai;
pub mod finops;

// New production-ready modules
pub mod cqrs; // CQRS pattern implementation
pub mod db;   // Database connection pool with optimized configuration

// Re-export the main app state and commonly used types
pub use api::AppState;
pub use config::AppConfig;
pub use db::{DbConfig, DbPool, SharedDbPool};
pub use error::ApiError;

// CQRS exports for external use
pub use cqrs::{
    CQRSSystem,
    Command, Query, Aggregate, DomainEvent,
    CommandHandler, QueryHandler,
    EventStore, ReadStore,
};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
