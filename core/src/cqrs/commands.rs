// CQRS Command implementations for PolicyCortex

use super::*;
// Removed unused update types - will be replaced with PCG-focused types
// use crate::api::{PolicyUpdate, ResourceUpdate, ComplianceUpdate};
use crate::cqrs::aggregate::{PolicyAggregate, ResourceAggregate, ComplianceAggregate, RemediationAggregate};
use crate::cqrs::events::{
    PolicyEvent, PolicyCreatedEvent, PolicyUpdatedEvent, PolicyDeletedEvent,
    ResourceEvent, ResourceCreatedEvent, ResourceUpdatedEvent,
    ComplianceEvent, ComplianceCheckedEvent,
    RemediationEvent, RemediationTriggeredEvent
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Command to create a new policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePolicyCommand {
    pub policy_id: Uuid,
    pub name: String,
    pub description: String,
    pub rules: Vec<String>,
    pub resource_types: Vec<String>,
    pub enforcement_mode: String,
    pub created_by: String,
}

#[async_trait]
impl Command for CreatePolicyCommand {
    type Aggregate = PolicyAggregate;
    type Result = Uuid;
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        // Create policy created event
        let event = PolicyCreatedEvent {
            policy_id: self.policy_id,
            name: self.name.clone(),
            description: self.description.clone(),
            rules: self.rules.clone(),
            resource_types: self.resource_types.clone(),
            enforcement_mode: self.enforcement_mode.clone(),
            created_by: self.created_by.clone(),
            occurred_at: Utc::now(),
        };
        
        // Apply event to aggregate
        aggregate.apply(&PolicyEvent::Created(event));
        aggregate.increment_version();
        
        Ok(self.policy_id)
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.policy_id
    }
    
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow::anyhow!("Policy name cannot be empty"));
        }
        if self.rules.is_empty() {
            return Err(anyhow::anyhow!("Policy must have at least one rule"));
        }
        Ok(())
    }
}

/// Command to update a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePolicyCommand {
    pub policy_id: Uuid,
    pub updates: PolicyUpdate,
    pub updated_by: String,
}

#[async_trait]
impl Command for UpdatePolicyCommand {
    type Aggregate = PolicyAggregate;
    type Result = ();
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = PolicyUpdatedEvent {
            policy_id: self.policy_id,
            updates: self.updates.clone(),
            updated_by: self.updated_by.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&PolicyEvent::Updated(event));
        aggregate.increment_version();
        
        Ok(())
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.policy_id
    }
}

/// Command to delete a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletePolicyCommand {
    pub policy_id: Uuid,
    pub deleted_by: String,
    pub reason: String,
}

#[async_trait]
impl Command for DeletePolicyCommand {
    type Aggregate = PolicyAggregate;
    type Result = ();
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = PolicyDeletedEvent {
            policy_id: self.policy_id,
            deleted_by: self.deleted_by.clone(),
            reason: self.reason.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&PolicyEvent::Deleted(event));
        aggregate.increment_version();
        
        Ok(())
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.policy_id
    }
}

/// Command to create a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResourceCommand {
    pub resource_id: Uuid,
    pub resource_type: String,
    pub name: String,
    pub location: String,
    pub tags: std::collections::HashMap<String, String>,
    pub created_by: String,
}

#[async_trait]
impl Command for CreateResourceCommand {
    type Aggregate = ResourceAggregate;
    type Result = Uuid;
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = ResourceCreatedEvent {
            resource_id: self.resource_id,
            resource_type: self.resource_type.clone(),
            name: self.name.clone(),
            location: self.location.clone(),
            tags: self.tags.clone(),
            created_by: self.created_by.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&ResourceEvent::Created(event));
        aggregate.increment_version();
        
        Ok(self.resource_id)
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.resource_id
    }
}

/// Command to update a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResourceCommand {
    pub resource_id: Uuid,
    pub updates: ResourceUpdate,
    pub updated_by: String,
}

#[async_trait]
impl Command for UpdateResourceCommand {
    type Aggregate = ResourceAggregate;
    type Result = ();
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = ResourceUpdatedEvent {
            resource_id: self.resource_id,
            updates: self.updates.clone(),
            updated_by: self.updated_by.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&ResourceEvent::Updated(event));
        aggregate.increment_version();
        
        Ok(())
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.resource_id
    }
}

/// Command to record compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordComplianceCheckCommand {
    pub check_id: Uuid,
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub compliant: bool,
    pub violations: Vec<String>,
    pub checked_by: String,
}

#[async_trait]
impl Command for RecordComplianceCheckCommand {
    type Aggregate = ComplianceAggregate;
    type Result = Uuid;
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = ComplianceCheckedEvent {
            check_id: self.check_id,
            resource_id: self.resource_id,
            policy_id: self.policy_id,
            compliant: self.compliant,
            violations: self.violations.clone(),
            checked_by: self.checked_by.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&ComplianceEvent::Checked(event));
        aggregate.increment_version();
        
        Ok(self.check_id)
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.check_id
    }
}

/// Command to trigger remediation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerRemediationCommand {
    pub remediation_id: Uuid,
    pub resource_id: Uuid,
    pub policy_id: Uuid,
    pub action_type: String,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    pub triggered_by: String,
}

#[async_trait]
impl Command for TriggerRemediationCommand {
    type Aggregate = RemediationAggregate;
    type Result = Uuid;
    
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result> {
        let event = RemediationTriggeredEvent {
            remediation_id: self.remediation_id,
            resource_id: self.resource_id,
            policy_id: self.policy_id,
            action_type: self.action_type.clone(),
            parameters: self.parameters.clone(),
            triggered_by: self.triggered_by.clone(),
            occurred_at: Utc::now(),
        };
        
        aggregate.apply(&RemediationEvent::Triggered(event));
        aggregate.increment_version();
        
        Ok(self.remediation_id)
    }
    
    fn aggregate_id(&self) -> Uuid {
        self.remediation_id
    }
}

/// Command handlers implementation
pub struct PolicyCommandHandler {
    event_store: Box<dyn EventStore>,
}

impl PolicyCommandHandler {
    pub fn new(event_store: Box<dyn EventStore>) -> Self {
        Self { event_store }
    }
}

#[async_trait]
impl CommandHandler<CreatePolicyCommand> for PolicyCommandHandler {
    async fn handle(&self, command: CreatePolicyCommand) -> Result<Uuid> {
        let mut aggregate = PolicyAggregate::new(command.policy_id);
        let result = command.execute(&mut aggregate).await?;
        
        // Save events
        let events: Vec<Box<dyn DomainEvent>> = aggregate
            .pending_events()
            .into_iter()
            .map(|e| Box::new(e) as Box<dyn DomainEvent>)
            .collect();
        
        self.event_store.save_events(events).await?;
        
        Ok(result)
    }
}

/// Resource command handler
pub struct ResourceCommandHandler {
    event_store: Box<dyn EventStore>,
}

impl ResourceCommandHandler {
    pub fn new(event_store: Box<dyn EventStore>) -> Self {
        Self { event_store }
    }
}

#[async_trait]
impl CommandHandler<CreateResourceCommand> for ResourceCommandHandler {
    async fn handle(&self, command: CreateResourceCommand) -> Result<Uuid> {
        let mut aggregate = ResourceAggregate::new(command.resource_id);
        let result = command.execute(&mut aggregate).await?;
        
        let events: Vec<Box<dyn DomainEvent>> = aggregate
            .pending_events()
            .into_iter()
            .map(|e| Box::new(e) as Box<dyn DomainEvent>)
            .collect();
        
        self.event_store.save_events(events).await?;
        
        Ok(result)
    }
}