// Event sourcing implementation for PolicyCortex

use super::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Policy domain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyEvent {
    Created(PolicyCreatedEvent),
    Updated(PolicyUpdatedEvent),
    Deleted(PolicyDeletedEvent),
    Activated(PolicyActivatedEvent),
    Deactivated(PolicyDeactivatedEvent),
}

impl DomainEvent for PolicyEvent {
    fn aggregate_id(&self) -> Uuid {
        match self {
            PolicyEvent::Created(e) => e.policy_id,
            PolicyEvent::Updated(e) => e.policy_id,
            PolicyEvent::Deleted(e) => e.policy_id,
            PolicyEvent::Activated(e) => e.policy_id,
            PolicyEvent::Deactivated(e) => e.policy_id,
        }
    }
    
    fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            PolicyEvent::Created(e) => e.occurred_at,
            PolicyEvent::Updated(e) => e.occurred_at,
            PolicyEvent::Deleted(e) => e.occurred_at,
            PolicyEvent::Activated(e) => e.occurred_at,
            PolicyEvent::Deactivated(e) => e.occurred_at,
        }
    }
    
    fn event_type(&self) -> &'static str {
        match self {
            PolicyEvent::Created(_) => "PolicyCreated",
            PolicyEvent::Updated(_) => "PolicyUpdated",
            PolicyEvent::Deleted(_) => "PolicyDeleted",
            PolicyEvent::Activated(_) => "PolicyActivated",
            PolicyEvent::Deactivated(_) => "PolicyDeactivated",
        }
    }
    
    fn version(&self) -> u64 {
        1 // Event schema version
    }
    
    fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    
    fn clone_box(&self) -> Box<dyn DomainEvent> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCreatedEvent {
    pub policy_id: Uuid,
    pub name: String,
    pub description: String,
    pub rules: Vec<String>,
    pub resource_types: Vec<String>,
    pub enforcement_mode: String,
    pub created_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyUpdatedEvent {
    pub policy_id: Uuid,
    pub updates: crate::api::PolicyUpdate,
    pub updated_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDeletedEvent {
    pub policy_id: Uuid,
    pub deleted_by: String,
    pub reason: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyActivatedEvent {
    pub policy_id: Uuid,
    pub activated_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDeactivatedEvent {
    pub policy_id: Uuid,
    pub deactivated_by: String,
    pub reason: Option<String>,
    pub occurred_at: DateTime<Utc>,
}

/// Resource domain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceEvent {
    Created(ResourceCreatedEvent),
    Updated(ResourceUpdatedEvent),
    Deleted(ResourceDeletedEvent),
    Tagged(ResourceTaggedEvent),
    ComplianceChecked(ResourceComplianceCheckedEvent),
}

impl DomainEvent for ResourceEvent {
    fn aggregate_id(&self) -> Uuid {
        match self {
            ResourceEvent::Created(e) => e.resource_id,
            ResourceEvent::Updated(e) => e.resource_id,
            ResourceEvent::Deleted(e) => e.resource_id,
            ResourceEvent::Tagged(e) => e.resource_id,
            ResourceEvent::ComplianceChecked(e) => e.resource_id,
        }
    }
    
    fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            ResourceEvent::Created(e) => e.occurred_at,
            ResourceEvent::Updated(e) => e.occurred_at,
            ResourceEvent::Deleted(e) => e.occurred_at,
            ResourceEvent::Tagged(e) => e.occurred_at,
            ResourceEvent::ComplianceChecked(e) => e.occurred_at,
        }
    }
    
    fn event_type(&self) -> &'static str {
        match self {
            ResourceEvent::Created(_) => "ResourceCreated",
            ResourceEvent::Updated(_) => "ResourceUpdated",
            ResourceEvent::Deleted(_) => "ResourceDeleted",
            ResourceEvent::Tagged(_) => "ResourceTagged",
            ResourceEvent::ComplianceChecked(_) => "ResourceComplianceChecked",
        }
    }
    
    fn version(&self) -> u64 {
        1
    }
    
    fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    
    fn clone_box(&self) -> Box<dyn DomainEvent> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCreatedEvent {
    pub resource_id: Uuid,
    pub resource_type: String,
    pub name: String,
    pub location: String,
    pub tags: std::collections::HashMap<String, String>,
    pub created_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdatedEvent {
    pub resource_id: Uuid,
    pub updates: crate::api::ResourceUpdate,
    pub updated_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDeletedEvent {
    pub resource_id: Uuid,
    pub deleted_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTaggedEvent {
    pub resource_id: Uuid,
    pub tags: std::collections::HashMap<String, String>,
    pub tagged_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceComplianceCheckedEvent {
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub compliant: bool,
    pub violations: Vec<String>,
    pub occurred_at: DateTime<Utc>,
}

/// Compliance domain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceEvent {
    Checked(ComplianceCheckedEvent),
    ViolationDetected(ComplianceViolationDetectedEvent),
    Remediated(ComplianceRemediatedEvent),
}

impl DomainEvent for ComplianceEvent {
    fn aggregate_id(&self) -> Uuid {
        match self {
            ComplianceEvent::Checked(e) => e.check_id,
            ComplianceEvent::ViolationDetected(e) => e.violation_id,
            ComplianceEvent::Remediated(e) => e.remediation_id,
        }
    }
    
    fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            ComplianceEvent::Checked(e) => e.occurred_at,
            ComplianceEvent::ViolationDetected(e) => e.occurred_at,
            ComplianceEvent::Remediated(e) => e.occurred_at,
        }
    }
    
    fn event_type(&self) -> &'static str {
        match self {
            ComplianceEvent::Checked(_) => "ComplianceChecked",
            ComplianceEvent::ViolationDetected(_) => "ComplianceViolationDetected",
            ComplianceEvent::Remediated(_) => "ComplianceRemediated",
        }
    }
    
    fn version(&self) -> u64 {
        1
    }
    
    fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    
    fn clone_box(&self) -> Box<dyn DomainEvent> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheckedEvent {
    pub check_id: Uuid,
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub compliant: bool,
    pub violations: Vec<String>,
    pub checked_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolationDetectedEvent {
    pub violation_id: Uuid,
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub violation_type: String,
    pub severity: String,
    pub description: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRemediatedEvent {
    pub remediation_id: Uuid,
    pub violation_id: Uuid,
    pub resource_id: Uuid,
    pub action_taken: String,
    pub remediated_by: String,
    pub occurred_at: DateTime<Utc>,
}

/// Remediation domain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationEvent {
    Triggered(RemediationTriggeredEvent),
    Started(RemediationStartedEvent),
    Completed(RemediationCompletedEvent),
    Failed(RemediationFailedEvent),
}

impl DomainEvent for RemediationEvent {
    fn aggregate_id(&self) -> Uuid {
        match self {
            RemediationEvent::Triggered(e) => e.remediation_id,
            RemediationEvent::Started(e) => e.remediation_id,
            RemediationEvent::Completed(e) => e.remediation_id,
            RemediationEvent::Failed(e) => e.remediation_id,
        }
    }
    
    fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            RemediationEvent::Triggered(e) => e.occurred_at,
            RemediationEvent::Started(e) => e.occurred_at,
            RemediationEvent::Completed(e) => e.occurred_at,
            RemediationEvent::Failed(e) => e.occurred_at,
        }
    }
    
    fn event_type(&self) -> &'static str {
        match self {
            RemediationEvent::Triggered(_) => "RemediationTriggered",
            RemediationEvent::Started(_) => "RemediationStarted",
            RemediationEvent::Completed(_) => "RemediationCompleted",
            RemediationEvent::Failed(_) => "RemediationFailed",
        }
    }
    
    fn version(&self) -> u64 {
        1
    }
    
    fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    
    fn clone_box(&self) -> Box<dyn DomainEvent> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationTriggeredEvent {
    pub remediation_id: Uuid,
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub action_type: String,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    pub triggered_by: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationStartedEvent {
    pub remediation_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationCompletedEvent {
    pub remediation_id: Uuid,
    pub completed_at: DateTime<Utc>,
    pub result: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationFailedEvent {
    pub remediation_id: Uuid,
    pub failed_at: DateTime<Utc>,
    pub error_message: String,
    pub occurred_at: DateTime<Utc>,
}

/// PostgreSQL-based event store implementation
pub struct PostgresEventStore {
    pool: crate::db::SharedDbPool,
}

impl PostgresEventStore {
    pub fn new(pool: crate::db::SharedDbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl EventStore for PostgresEventStore {
    async fn save_events(&self, events: Vec<Box<dyn DomainEvent>>) -> Result<()> {
        let pool = self.pool.pool();
        let mut tx = pool.begin().await?;
        
        for event in events {
            let event_data = event.as_json();
            
            sqlx::query(
                r#"
                INSERT INTO event_store (
                    aggregate_id, event_type, event_data, 
                    event_version, occurred_at
                )
                VALUES ($1, $2, $3, $4, $5)
                "#
            )
            .bind(event.aggregate_id())
            .bind(event.event_type())
            .bind(event_data)
            .bind(event.version() as i64)
            .bind(event.occurred_at())
            .execute(tx.as_mut())
            .await?;
        }
        
        tx.commit().await?;
        Ok(())
    }
    
    async fn load_events(&self, aggregate_id: Uuid, after_version: Option<u64>) -> Result<Vec<Box<dyn DomainEvent>>> {
        let mut conn = self.pool.pool().acquire().await?;
        
        let version = after_version.unwrap_or(0) as i64;
        
        let rows = sqlx::query_as::<_, (String, serde_json::Value, i64)>(
            r#"
            SELECT event_type, event_data, event_version
            FROM event_store
            WHERE aggregate_id = $1 AND event_version > $2
            ORDER BY event_version ASC
            "#
        )
        .bind(aggregate_id)
        .bind(version)
        .fetch_all(&mut *conn)
        .await?;
        
        let mut events = Vec::new();
        for row in rows {
            // Deserialize based on event type
            // This would need proper implementation based on event type
            // For now, returning empty vec as placeholder
        }
        
        Ok(events)
    }
    
    async fn get_version(&self, aggregate_id: Uuid) -> Result<u64> {
        let mut conn = self.pool.pool().acquire().await?;
        
        let result = sqlx::query_as::<_, (Option<i64>,)>(
            r#"
            SELECT MAX(event_version) as version
            FROM event_store
            WHERE aggregate_id = $1
            "#
        )
        .bind(aggregate_id)
        .fetch_one(&mut *conn)
        .await?;
        
        Ok(result.0.unwrap_or(0) as u64)
    }
}