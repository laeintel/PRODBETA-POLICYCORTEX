// Event projections for updating read models in PolicyCortex

use super::*;
use crate::cqrs::events::*;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{info, warn, error};

/// Projection trait for handling events and updating read models
#[async_trait]
pub trait Projection: Send + Sync {
    /// Handle an event and update the read model
    async fn handle(&self, event: Box<dyn DomainEvent>) -> Result<()>;
    
    /// Get the name of this projection
    fn name(&self) -> &'static str;
}

/// Projection manager that coordinates event processing
pub struct ProjectionManager {
    projections: Vec<Box<dyn Projection>>,
    event_receiver: tokio::sync::mpsc::UnboundedReceiver<Box<dyn DomainEvent>>,
    pool: crate::db::SharedDbPool,
}

impl ProjectionManager {
    /// Create a new projection manager
    pub fn new(
        event_receiver: tokio::sync::mpsc::UnboundedReceiver<Box<dyn DomainEvent>>,
        pool: crate::db::SharedDbPool,
    ) -> Self {
        Self {
            projections: Vec::new(),
            event_receiver,
            pool,
        }
    }
    
    /// Register a projection
    pub fn register(&mut self, projection: Box<dyn Projection>) {
        info!("Registered projection: {}", projection.name());
        self.projections.push(projection);
    }
    
    /// Start processing events
    pub async fn start(mut self) {
        info!("Starting projection manager with {} projections", self.projections.len());
        
        loop {
            match self.event_receiver.recv().await {
                Some(event) => {
                    for projection in &self.projections {
                        if let Err(e) = projection.handle(event.clone_box()).await {
                            error!(
                                "Projection {} failed to handle event: {}",
                                projection.name(),
                                e
                            );
                        }
                    }
                }
                None => {
                    info!("Event channel closed, stopping projection manager");
                    break;
                }
            }
        }
    }
}

/// Policy read model projection
pub struct PolicyProjection {
    pool: crate::db::SharedDbPool,
}

impl PolicyProjection {
    pub fn new(pool: crate::db::SharedDbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Projection for PolicyProjection {
    async fn handle(&self, event: Box<dyn DomainEvent>) -> Result<()> {
        let mut conn = self.pool.pool().acquire().await?;
        
        // Determine event type and handle accordingly
        let event_type = event.event_type();
        let event_data = event.as_json();
        
        match event_type {
            "PolicyCreated" => {
                if let Ok(e) = serde_json::from_value::<PolicyCreatedEvent>(event_data) {
                    sqlx::query(
                        r#"
                        INSERT INTO policies (
                            id, name, description, rules, resource_types,
                            enforcement_mode, created_by, created_at, updated_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                        ON CONFLICT (id) DO NOTHING
                        "#
                    )
                    .bind(e.policy_id)
                    .bind(e.name)
                    .bind(e.description)
                    .bind(&e.rules[..])
                    .bind(&e.resource_types[..])
                    .bind(e.enforcement_mode)
                    .bind(e.created_by)
                    .bind(e.occurred_at)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Created policy {} in read model", e.policy_id);
                }
            }
            "PolicyUpdated" => {
                // Handle policy update
                info!("Updated policy in read model");
            }
            "PolicyDeleted" => {
                if let Ok(e) = serde_json::from_value::<PolicyDeletedEvent>(event_data) {
                    sqlx::query(
                        r#"
                        UPDATE policies
                        SET deleted_at = $2, deleted_by = $3, deletion_reason = $4
                        WHERE id = $1
                        "#
                    )
                    .bind(e.policy_id)
                    .bind(e.occurred_at)
                    .bind(e.deleted_by)
                    .bind(e.reason)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Marked policy {} as deleted in read model", e.policy_id);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "PolicyProjection"
    }
}

/// Resource read model projection
pub struct ResourceProjection {
    pool: crate::db::SharedDbPool,
}

impl ResourceProjection {
    pub fn new(pool: crate::db::SharedDbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Projection for ResourceProjection {
    async fn handle(&self, event: Box<dyn DomainEvent>) -> Result<()> {
        let mut conn = self.pool.pool().acquire().await?;
        let event_type = event.event_type();
        let event_data = event.as_json();
        
        match event_type {
            "ResourceCreated" => {
                if let Ok(e) = serde_json::from_value::<ResourceCreatedEvent>(event_data) {
                    sqlx::query(
                        r#"
                        INSERT INTO resources (
                            id, resource_type, name, location, tags,
                            compliance_status, risk_score, created_by, created_at, updated_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                        ON CONFLICT (id) DO NOTHING
                        "#
                    )
                    .bind(e.resource_id)
                    .bind(e.resource_type)
                    .bind(e.name)
                    .bind(e.location)
                    .bind(serde_json::to_value(&e.tags)?)
                    .bind("unknown")
                    .bind(0.0)
                    .bind(e.created_by)
                    .bind(e.occurred_at)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Created resource {} in read model", e.resource_id);
                }
            }
            "ResourceComplianceChecked" => {
                if let Ok(e) = serde_json::from_value::<ResourceComplianceCheckedEvent>(event_data) {
                    // Update resource compliance status
                    let status = if e.compliant { "compliant" } else { "non_compliant" };
                    let risk_score = e.violations.len() as f64 * 10.0;
                    
                    sqlx::query(
                        r#"
                        UPDATE resources
                        SET compliance_status = $2, risk_score = $3, updated_at = $4
                        WHERE id = $1
                        "#
                    )
                    .bind(e.resource_id)
                    .bind(status)
                    .bind(risk_score)
                    .bind(e.occurred_at)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Updated compliance status for resource {}", e.resource_id);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "ResourceProjection"
    }
}

/// Compliance read model projection
pub struct ComplianceProjection {
    pool: crate::db::SharedDbPool,
}

impl ComplianceProjection {
    pub fn new(pool: crate::db::SharedDbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Projection for ComplianceProjection {
    async fn handle(&self, event: Box<dyn DomainEvent>) -> Result<()> {
        let mut conn = self.pool.pool().acquire().await?;
        let event_type = event.event_type();
        let event_data = event.as_json();
        
        match event_type {
            "ComplianceChecked" => {
                if let Ok(e) = serde_json::from_value::<ComplianceCheckedEvent>(event_data) {
                    sqlx::query(
                        r#"
                        INSERT INTO compliance_checks (
                            id, resource_id, policy_id, compliant, violations,
                            checked_by, checked_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        "#
                    )
                    .bind(e.check_id)
                    .bind(e.resource_id)
                    .bind(e.policy_id)
                    .bind(e.compliant)
                    .bind(&e.violations[..])
                    .bind(e.checked_by)
                    .bind(e.occurred_at)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Recorded compliance check {} in read model", e.check_id);
                }
            }
            "ComplianceViolationDetected" => {
                if let Ok(e) = serde_json::from_value::<ComplianceViolationDetectedEvent>(event_data) {
                    sqlx::query(
                        r#"
                        INSERT INTO compliance_violations (
                            id, resource_id, policy_id, violation_type, severity,
                            description, detected_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        "#
                    )
                    .bind(e.violation_id)
                    .bind(e.resource_id)
                    .bind(e.policy_id)
                    .bind(e.violation_type)
                    .bind(e.severity)
                    .bind(e.description)
                    .bind(e.occurred_at)
                    .execute(&mut *conn)
                    .await?;
                    
                    info!("Recorded compliance violation {} in read model", e.violation_id);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "ComplianceProjection"
    }
}

/// Analytics projection for aggregated metrics
pub struct AnalyticsProjection {
    pool: crate::db::SharedDbPool,
    cache: crate::cache::CacheManager,
}

impl AnalyticsProjection {
    pub fn new(pool: crate::db::SharedDbPool, cache: crate::cache::CacheManager) -> Self {
        Self { pool, cache }
    }
    
    async fn update_metrics(&self) -> Result<()> {
        let mut conn = self.pool.pool().acquire().await?;
        
        // Update policy compliance metrics
        let compliance_stats = sqlx::query_as::<_, (Option<i64>, Option<f32>, Option<i64>)>(
            r#"
            SELECT 
                COUNT(DISTINCT resource_id) as total_resources,
                SUM(CASE WHEN compliant THEN 1 ELSE 0 END)::float / COUNT(*)::float * 100 as compliance_rate,
                COUNT(DISTINCT policy_id) as policies_checked
            FROM compliance_checks
            WHERE checked_at >= NOW() - INTERVAL '24 hours'
            "#
        )
        .fetch_one(&mut *conn)
        .await?;
        
        // Cache the metrics
        let metrics = serde_json::json!({
            "total_resources": compliance_stats.0,
            "compliance_rate": compliance_stats.1,
            "policies_checked": compliance_stats.2,
            "updated_at": chrono::Utc::now()
        });
        
        self.cache.set("metrics:compliance", &metrics, 60).await?;
        
        info!("Updated analytics metrics in cache");
        Ok(())
    }
}

#[async_trait]
impl Projection for AnalyticsProjection {
    async fn handle(&self, event: Box<dyn DomainEvent>) -> Result<()> {
        // Update analytics metrics after certain events
        match event.event_type() {
            "ComplianceChecked" | "PolicyCreated" | "ResourceCreated" => {
                // Update metrics asynchronously
                let pool = self.pool.clone();
                let cache = self.cache.clone();
                
                tokio::spawn(async move {
                    let projection = AnalyticsProjection::new(pool, cache);
                    if let Err(e) = projection.update_metrics().await {
                        warn!("Failed to update analytics metrics: {}", e);
                    }
                });
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AnalyticsProjection"
    }
}

/// Create and register all projections
pub fn create_projections(
    pool: crate::db::SharedDbPool,
    cache: crate::cache::CacheManager,
) -> Vec<Box<dyn Projection>> {
    vec![
        Box::new(PolicyProjection::new(pool.clone())),
        Box::new(ResourceProjection::new(pool.clone())),
        Box::new(ComplianceProjection::new(pool.clone())),
        Box::new(AnalyticsProjection::new(pool, cache)),
    ]
}