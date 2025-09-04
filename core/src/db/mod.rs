// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

//! Database connection pool and configuration module
//! Implements production-grade connection pooling with environment-specific settings

use anyhow::{Context, Result};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{ConnectOptions, PgPool};
use std::env;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Database configuration with environment-specific pool settings
#[derive(Debug, Clone)]
pub struct DbConfig {
    /// Database connection URL
    pub database_url: String,
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    /// Minimum number of idle connections to maintain
    pub min_idle: u32,
    /// Connection timeout in seconds
    pub connect_timeout: u64,
    /// Idle timeout in seconds before closing a connection
    pub idle_timeout: u64,
    /// Maximum lifetime of a connection in seconds
    pub max_lifetime: u64,
    /// Enable statement-level logging
    pub enable_logging: bool,
}

impl DbConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Result<Self> {
        let database_url = env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://postgres:postgres@localhost:5432/policycortex".to_string());
        
        let environment = env::var("ENVIRONMENT").unwrap_or_else(|_| "dev".to_string());
        
        // Environment-specific pool sizes
        let (max_connections, min_idle) = match environment.as_str() {
            "production" | "prod" => (30, 10),
            "staging" | "stage" => (20, 5),
            "development" | "dev" => (10, 2),
            "test" => (5, 1),
            _ => (25, 5), // Default safe values
        };
        
        // Override with explicit environment variables if provided
        let max_connections = env::var("DB_MAX_CONNECTIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(max_connections);
            
        let min_idle = env::var("DB_MIN_IDLE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(min_idle);
        
        let connect_timeout = env::var("DB_CONNECT_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
            
        let idle_timeout = env::var("DB_IDLE_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600); // 10 minutes
            
        let max_lifetime = env::var("DB_MAX_LIFETIME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800); // 30 minutes
        
        let enable_logging = env::var("DB_ENABLE_LOGGING")
            .ok()
            .map(|v| v == "true" || v == "1")
            .unwrap_or_else(|| environment == "dev");
        
        info!(
            "Database configuration loaded - Environment: {}, Max connections: {}, Min idle: {}",
            environment, max_connections, min_idle
        );
        
        Ok(Self {
            database_url,
            max_connections,
            min_idle,
            connect_timeout,
            idle_timeout,
            max_lifetime,
            enable_logging,
        })
    }
}

/// Database connection pool manager
pub struct DbPool {
    pool: PgPool,
    config: DbConfig,
}

impl DbPool {
    /// Create a new database pool with the given configuration
    pub async fn new(config: DbConfig) -> Result<Self> {
        let mut connect_options = PgConnectOptions::from_str(&config.database_url)
            .context("Invalid database URL")?;
        
        // Configure logging
        if config.enable_logging {
            connect_options = connect_options
                .log_statements(tracing::log::LevelFilter::Debug)
                .log_slow_statements(tracing::log::LevelFilter::Warn, Duration::from_secs(1));
        } else {
            connect_options = connect_options.disable_statement_logging();
        }
        
        // Build the connection pool with production settings
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_idle)
            .acquire_timeout(Duration::from_secs(config.connect_timeout))
            .idle_timeout(Duration::from_secs(config.idle_timeout))
            .max_lifetime(Duration::from_secs(config.max_lifetime))
            .test_before_acquire(true)
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    // Set connection parameters for optimal performance
                    sqlx::query("SET statement_timeout = '30s'")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET lock_timeout = '10s'")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET idle_in_transaction_session_timeout = '60s'")
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                })
            })
            .connect_with(connect_options)
            .await
            .context("Failed to create database pool")?;
        
        // Warm up the connection pool
        Self::warmup_pool(&pool, config.min_idle).await?;
        
        info!(
            "Database pool initialized with {} connections (min: {}, max: {})",
            pool.size(), config.min_idle, config.max_connections
        );
        
        Ok(Self { pool, config })
    }
    
    /// Warm up the connection pool by pre-establishing minimum connections
    async fn warmup_pool(pool: &PgPool, min_connections: u32) -> Result<()> {
        info!("Warming up connection pool with {} connections", min_connections);
        
        let mut handles = Vec::new();
        for i in 0..min_connections {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                match pool.acquire().await {
                    Ok(conn) => {
                        debug!("Warmed up connection {}", i);
                        drop(conn); // Return to pool
                        Ok(())
                    }
                    Err(e) => {
                        warn!("Failed to warm up connection {}: {}", i, e);
                        Err(e)
                    }
                }
            }));
        }
        
        // Wait for all warmup connections
        for handle in handles {
            handle.await??;
        }
        
        info!("Connection pool warmup completed");
        Ok(())
    }
    
    /// Get a reference to the underlying connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
    
    /// Get the current pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            size: self.pool.size(),
            idle: self.pool.num_idle() as u32,
            max_connections: self.config.max_connections,
            min_idle: self.config.min_idle,
        }
    }
    
    /// Perform a health check on the database
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let start = std::time::Instant::now();
        
        match sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&self.pool)
            .await
        {
            Ok(_) => {
                let latency = start.elapsed();
                Ok(HealthStatus {
                    healthy: true,
                    latency_ms: latency.as_millis() as u64,
                    pool_stats: self.stats(),
                    error: None,
                })
            }
            Err(e) => {
                let latency = start.elapsed();
                warn!("Database health check failed: {}", e);
                Ok(HealthStatus {
                    healthy: false,
                    latency_ms: latency.as_millis() as u64,
                    pool_stats: self.stats(),
                    error: Some(e.to_string()),
                })
            }
        }
    }
    
    /// Gracefully shutdown the connection pool
    pub async fn shutdown(&self) {
        info!("Shutting down database connection pool");
        self.pool.close().await;
        info!("Database connection pool closed");
    }
}

/// Pool statistics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolStats {
    /// Current number of connections in the pool
    pub size: u32,
    /// Number of idle connections
    pub idle: u32,
    /// Maximum connections allowed
    pub max_connections: u32,
    /// Minimum idle connections to maintain
    pub min_idle: u32,
}

/// Database health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthStatus {
    /// Whether the database is healthy
    pub healthy: bool,
    /// Query latency in milliseconds
    pub latency_ms: u64,
    /// Current pool statistics
    pub pool_stats: PoolStats,
    /// Error message if unhealthy
    pub error: Option<String>,
}

/// Shared database pool type for dependency injection
pub type SharedDbPool = Arc<DbPool>;

/// Create a shared database pool from environment configuration
pub async fn create_pool() -> Result<SharedDbPool> {
    let config = DbConfig::from_env()?;
    let pool = DbPool::new(config).await?;
    Ok(Arc::new(pool))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_from_env() {
        // Test development environment
        env::set_var("ENVIRONMENT", "dev");
        let config = DbConfig::from_env().unwrap();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_idle, 2);
        
        // Test production environment
        env::set_var("ENVIRONMENT", "production");
        let config = DbConfig::from_env().unwrap();
        assert_eq!(config.max_connections, 30);
        assert_eq!(config.min_idle, 10);
        
        // Test custom overrides
        env::set_var("DB_MAX_CONNECTIONS", "50");
        env::set_var("DB_MIN_IDLE", "15");
        let config = DbConfig::from_env().unwrap();
        assert_eq!(config.max_connections, 50);
        assert_eq!(config.min_idle, 15);
        
        // Clean up
        env::remove_var("ENVIRONMENT");
        env::remove_var("DB_MAX_CONNECTIONS");
        env::remove_var("DB_MIN_IDLE");
    }
}