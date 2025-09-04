// Aggregate implementations for PolicyCortex domain entities

use super::*;
use crate::cqrs::events::*;
use uuid::Uuid;
use std::collections::HashMap;

/// Policy aggregate
pub struct PolicyAggregate {
    id: Uuid,
    version: u64,
    name: Option<String>,
    description: Option<String>,
    rules: Vec<String>,
    resource_types: Vec<String>,
    enforcement_mode: Option<String>,
    is_active: bool,
    is_deleted: bool,
    pending_events: Vec<PolicyEvent>,
}

impl PolicyAggregate {
    pub fn pending_events(&mut self) -> Vec<PolicyEvent> {
        std::mem::take(&mut self.pending_events)
    }
}

impl Aggregate for PolicyAggregate {
    type Event = PolicyEvent;
    
    fn new(id: Uuid) -> Self {
        Self {
            id,
            version: 0,
            name: None,
            description: None,
            rules: Vec::new(),
            resource_types: Vec::new(),
            enforcement_mode: None,
            is_active: false,
            is_deleted: false,
            pending_events: Vec::new(),
        }
    }
    
    fn apply(&mut self, event: &Self::Event) {
        match event {
            PolicyEvent::Created(e) => {
                self.name = Some(e.name.clone());
                self.description = Some(e.description.clone());
                self.rules = e.rules.clone();
                self.resource_types = e.resource_types.clone();
                self.enforcement_mode = Some(e.enforcement_mode.clone());
                self.is_active = true;
            }
            PolicyEvent::Updated(e) => {
                // Apply updates from PolicyUpdate struct
                // This would need proper implementation based on PolicyUpdate fields
            }
            PolicyEvent::Deleted(_) => {
                self.is_deleted = true;
                self.is_active = false;
            }
            PolicyEvent::Activated(_) => {
                self.is_active = true;
            }
            PolicyEvent::Deactivated(_) => {
                self.is_active = false;
            }
        }
        
        // Add to pending events for persistence
        self.pending_events.push(event.clone());
    }
    
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn version(&self) -> u64 {
        self.version
    }
    
    fn increment_version(&mut self) {
        self.version += 1;
    }
}

/// Resource aggregate
pub struct ResourceAggregate {
    id: Uuid,
    version: u64,
    resource_type: Option<String>,
    name: Option<String>,
    location: Option<String>,
    tags: HashMap<String, String>,
    compliance_status: ComplianceStatus,
    risk_score: f64,
    is_deleted: bool,
    pending_events: Vec<ResourceEvent>,
}

#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    Unknown,
    Compliant,
    NonCompliant,
    PartiallyCompliant,
}

impl ResourceAggregate {
    pub fn pending_events(&mut self) -> Vec<ResourceEvent> {
        std::mem::take(&mut self.pending_events)
    }
}

impl Aggregate for ResourceAggregate {
    type Event = ResourceEvent;
    
    fn new(id: Uuid) -> Self {
        Self {
            id,
            version: 0,
            resource_type: None,
            name: None,
            location: None,
            tags: HashMap::new(),
            compliance_status: ComplianceStatus::Unknown,
            risk_score: 0.0,
            is_deleted: false,
            pending_events: Vec::new(),
        }
    }
    
    fn apply(&mut self, event: &Self::Event) {
        match event {
            ResourceEvent::Created(e) => {
                self.resource_type = Some(e.resource_type.clone());
                self.name = Some(e.name.clone());
                self.location = Some(e.location.clone());
                self.tags = e.tags.clone();
            }
            ResourceEvent::Updated(e) => {
                // Apply updates from ResourceUpdate struct
            }
            ResourceEvent::Deleted(_) => {
                self.is_deleted = true;
            }
            ResourceEvent::Tagged(e) => {
                self.tags.extend(e.tags.clone());
            }
            ResourceEvent::ComplianceChecked(e) => {
                self.compliance_status = if e.compliant {
                    ComplianceStatus::Compliant
                } else if e.violations.is_empty() {
                    ComplianceStatus::PartiallyCompliant
                } else {
                    ComplianceStatus::NonCompliant
                };
                
                // Update risk score based on violations
                self.risk_score = e.violations.len() as f64 * 10.0;
            }
        }
        
        self.pending_events.push(event.clone());
    }
    
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn version(&self) -> u64 {
        self.version
    }
    
    fn increment_version(&mut self) {
        self.version += 1;
    }
}

/// Compliance aggregate
pub struct ComplianceAggregate {
    id: Uuid,
    version: u64,
    resource_id: Option<Uuid>,
    policy_id: Option<Uuid>,
    compliant: bool,
    violations: Vec<String>,
    pending_events: Vec<ComplianceEvent>,
}

impl ComplianceAggregate {
    pub fn pending_events(&mut self) -> Vec<ComplianceEvent> {
        std::mem::take(&mut self.pending_events)
    }
}

impl Aggregate for ComplianceAggregate {
    type Event = ComplianceEvent;
    
    fn new(id: Uuid) -> Self {
        Self {
            id,
            version: 0,
            resource_id: None,
            policy_id: None,
            compliant: true,
            violations: Vec::new(),
            pending_events: Vec::new(),
        }
    }
    
    fn apply(&mut self, event: &Self::Event) {
        match event {
            ComplianceEvent::Checked(e) => {
                self.resource_id = Some(e.resource_id);
                self.policy_id = Some(e.policy_id);
                self.compliant = e.compliant;
                self.violations = e.violations.clone();
            }
            ComplianceEvent::ViolationDetected(e) => {
                self.compliant = false;
                self.violations.push(e.description.clone());
            }
            ComplianceEvent::Remediated(_) => {
                self.compliant = true;
                self.violations.clear();
            }
        }
        
        self.pending_events.push(event.clone());
    }
    
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn version(&self) -> u64 {
        self.version
    }
    
    fn increment_version(&mut self) {
        self.version += 1;
    }
}

/// Remediation aggregate
pub struct RemediationAggregate {
    id: Uuid,
    version: u64,
    resource_id: Option<Uuid>,
    policy_id: Option<Uuid>,
    status: RemediationStatus,
    action_type: Option<String>,
    parameters: HashMap<String, serde_json::Value>,
    error_message: Option<String>,
    pending_events: Vec<RemediationEvent>,
}

#[derive(Debug, Clone)]
pub enum RemediationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl RemediationAggregate {
    pub fn pending_events(&mut self) -> Vec<RemediationEvent> {
        std::mem::take(&mut self.pending_events)
    }
}

impl Aggregate for RemediationAggregate {
    type Event = RemediationEvent;
    
    fn new(id: Uuid) -> Self {
        Self {
            id,
            version: 0,
            resource_id: None,
            policy_id: None,
            status: RemediationStatus::Pending,
            action_type: None,
            parameters: HashMap::new(),
            error_message: None,
            pending_events: Vec::new(),
        }
    }
    
    fn apply(&mut self, event: &Self::Event) {
        match event {
            RemediationEvent::Triggered(e) => {
                self.resource_id = Some(e.resource_id);
                self.policy_id = Some(e.policy_id);
                self.action_type = Some(e.action_type.clone());
                self.parameters = e.parameters.clone();
                self.status = RemediationStatus::Pending;
            }
            RemediationEvent::Started(_) => {
                self.status = RemediationStatus::InProgress;
            }
            RemediationEvent::Completed(_) => {
                self.status = RemediationStatus::Completed;
                self.error_message = None;
            }
            RemediationEvent::Failed(e) => {
                self.status = RemediationStatus::Failed;
                self.error_message = Some(e.error_message.clone());
            }
        }
        
        self.pending_events.push(event.clone());
    }
    
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn version(&self) -> u64 {
        self.version
    }
    
    fn increment_version(&mut self) {
        self.version += 1;
    }
}