// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Event-Driven Architecture Implementation
// Based on Roadmap_02_System_Architecture.md
// Implements GitHub Issue #50: Event-driven architecture with NATS/Kafka

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::broadcast;
use uuid::Uuid;

#[cfg(feature = "events")]
use async_nats;

// Core event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum GovernanceEvent {
    // Policy Events
    PolicyViolation {
        policy_id: String,
        resource_id: String,
        severity: String,
        details: HashMap<String, serde_json::Value>,
    },
    PolicyRemediated {
        policy_id: String,
        resource_id: String,
        action_taken: String,
        success: bool,
    },

    // Resource Events
    ResourceCreated {
        resource_id: String,
        resource_type: String,
        metadata: HashMap<String, String>,
    },
    ResourceModified {
        resource_id: String,
        changes: Vec<Change>,
    },
    ResourceDeleted {
        resource_id: String,
        reason: String,
    },

    // Cost Events
    CostAnomaly {
        service: String,
        deviation_percentage: f64,
        normal_cost: f64,
        actual_cost: f64,
    },
    CostOptimization {
        optimization_id: String,
        savings_achieved: f64,
        resources_affected: Vec<String>,
    },

    // Security Events
    SecurityThreat {
        threat_id: String,
        severity: String,
        attack_vector: String,
        affected_resources: Vec<String>,
    },
    SecurityMitigation {
        threat_id: String,
        mitigation_applied: String,
        success: bool,
    },

    // Compliance Events
    ComplianceCheck {
        framework: String,
        control_id: String,
        status: String,
        evidence_id: Option<String>,
    },
    ComplianceDrift {
        framework: String,
        current_score: f64,
        previous_score: f64,
        controls_affected: Vec<String>,
    },

    // Action Lifecycle Events
    ActionInitiated {
        action_id: String,
        action_type: String,
        dry_run: bool,
        initiated_by: String,
    },
    ActionProgress {
        action_id: String,
        progress_percentage: f64,
        current_step: String,
        message: String,
    },
    ActionCompleted {
        action_id: String,
        success: bool,
        execution_time_ms: u64,
        result: serde_json::Value,
    },
    ActionFailed {
        action_id: String,
        error: String,
        rollback_available: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub field: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: String,
    pub correlation_id: String,
    pub source: String,
    pub event: GovernanceEvent,
    pub metadata: HashMap<String, String>,
}

impl EventEnvelope {
    pub fn new(event: GovernanceEvent, tenant_id: String, source: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            tenant_id,
            correlation_id: Uuid::new_v4().to_string(),
            source,
            event,
            metadata: HashMap::new(),
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = correlation_id;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// Event Bus trait for different implementations
#[async_trait]
pub trait EventBus: Send + Sync {
    async fn publish(&self, event: EventEnvelope) -> Result<(), EventError>;
    async fn subscribe(&self, topics: Vec<String>) -> Result<EventSubscription, EventError>;
    async fn subscribe_pattern(&self, pattern: String) -> Result<EventSubscription, EventError>;
}

// NATS implementation
#[cfg(feature = "events")]
pub struct NatsEventBus {
    client: async_nats::Client,
    jetstream: async_nats::jetstream::Context,
}

#[cfg(feature = "events")]
impl NatsEventBus {
    pub async fn new(urls: Vec<String>) -> Result<Self, EventError> {
        let client = async_nats::connect(&urls.join(","))
            .await
            .map_err(|e| EventError::ConnectionError(e.to_string()))?;

        let jetstream = async_nats::jetstream::new(client.clone());

        // Create streams for different event categories
        Self::ensure_streams(&jetstream).await?;

        Ok(Self { client, jetstream })
    }

    async fn ensure_streams(js: &async_nats::jetstream::Context) -> Result<(), EventError> {
        use async_nats::jetstream::stream::Config;

        let streams = vec![
            ("GOVERNANCE", vec!["governance.>".to_string()]),
            ("COSTS", vec!["costs.>".to_string()]),
            ("SECURITY", vec!["security.>".to_string()]),
            ("COMPLIANCE", vec!["compliance.>".to_string()]),
            ("ACTIONS", vec!["actions.>".to_string()]),
        ];

        for (name, subjects) in streams {
            let config = Config {
                name: name.to_string(),
                subjects,
                retention: async_nats::jetstream::stream::RetentionPolicy::Limits,
                max_messages: 1_000_000,
                max_age: std::time::Duration::from_secs(7 * 24 * 60 * 60), // 7 days
                ..Default::default()
            };

            js.get_or_create_stream(config)
                .await
                .map_err(|e| EventError::StreamError(e.to_string()))?;
        }

        Ok(())
    }

    fn get_subject(event: &GovernanceEvent) -> String {
        match event {
            GovernanceEvent::PolicyViolation { .. } | GovernanceEvent::PolicyRemediated { .. } => {
                "governance.policy"
            }

            GovernanceEvent::ResourceCreated { .. }
            | GovernanceEvent::ResourceModified { .. }
            | GovernanceEvent::ResourceDeleted { .. } => "governance.resources",

            GovernanceEvent::CostAnomaly { .. } | GovernanceEvent::CostOptimization { .. } => {
                "costs.events"
            }

            GovernanceEvent::SecurityThreat { .. } | GovernanceEvent::SecurityMitigation { .. } => {
                "security.events"
            }

            GovernanceEvent::ComplianceCheck { .. } | GovernanceEvent::ComplianceDrift { .. } => {
                "compliance.events"
            }

            GovernanceEvent::ActionInitiated { .. }
            | GovernanceEvent::ActionProgress { .. }
            | GovernanceEvent::ActionCompleted { .. }
            | GovernanceEvent::ActionFailed { .. } => "actions.lifecycle",
        }
        .to_string()
    }
}

#[async_trait]
#[cfg(feature = "events")]
impl EventBus for NatsEventBus {
    async fn publish(&self, event: EventEnvelope) -> Result<(), EventError> {
        let subject = Self::get_subject(&event.event);
        let payload = serde_json::to_vec(&event)
            .map_err(|e| EventError::SerializationError(e.to_string()))?;

        self.client
            .publish(subject, payload.into())
            .await
            .map_err(|e| EventError::PublishError(e.to_string()))?;

        Ok(())
    }

    async fn subscribe(&self, topics: Vec<String>) -> Result<EventSubscription, EventError> {
        let mut subscription = EventSubscription::new();

        for topic in topics {
            let sub = self
                .client
                .subscribe(topic)
                .await
                .map_err(|e| EventError::SubscriptionError(e.to_string()))?;

            subscription.add_subscription(sub);
        }

        Ok(subscription)
    }

    async fn subscribe_pattern(&self, pattern: String) -> Result<EventSubscription, EventError> {
        let sub = self
            .client
            .subscribe(pattern)
            .await
            .map_err(|e| EventError::SubscriptionError(e.to_string()))?;

        let mut subscription = EventSubscription::new();
        subscription.add_subscription(sub);

        Ok(subscription)
    }
}

// In-memory implementation for testing
pub struct InMemoryEventBus {
    sender: broadcast::Sender<EventEnvelope>,
}

impl InMemoryEventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self { sender }
    }
}

#[async_trait]
impl EventBus for InMemoryEventBus {
    async fn publish(&self, event: EventEnvelope) -> Result<(), EventError> {
        self.sender
            .send(event)
            .map_err(|e| EventError::PublishError(e.to_string()))?;
        Ok(())
    }

    async fn subscribe(&self, _topics: Vec<String>) -> Result<EventSubscription, EventError> {
        let receiver = self.sender.subscribe();
        let mut subscription = EventSubscription::new();
        subscription.add_broadcast_receiver(receiver);
        Ok(subscription)
    }

    async fn subscribe_pattern(&self, _pattern: String) -> Result<EventSubscription, EventError> {
        self.subscribe(vec![]).await
    }
}

// Event subscription handler
pub struct EventSubscription {
    #[cfg(feature = "events")]
    nats_subs: Vec<async_nats::Subscriber>,
    broadcast_receivers: Vec<broadcast::Receiver<EventEnvelope>>,
}

impl EventSubscription {
    fn new() -> Self {
        Self {
            #[cfg(feature = "events")]
            nats_subs: Vec::new(),
            broadcast_receivers: Vec::new(),
        }
    }

    #[cfg(feature = "events")]
    fn add_subscription(&mut self, sub: async_nats::Subscriber) {
        self.nats_subs.push(sub);
    }

    fn add_broadcast_receiver(&mut self, receiver: broadcast::Receiver<EventEnvelope>) {
        self.broadcast_receivers.push(receiver);
    }

    pub async fn next(&mut self) -> Option<EventEnvelope> {
        // Try NATS subscriptions first
        #[cfg(feature = "events")]
        {
            for sub in &mut self.nats_subs {
                if let Some(msg) = sub.next().await {
                    if let Ok(event) = serde_json::from_slice::<EventEnvelope>(&msg.payload) {
                        return Some(event);
                    }
                }
            }
        }

        // Try in-memory receivers
        for receiver in &mut self.broadcast_receivers {
            if let Ok(event) = receiver.try_recv() {
                return Some(event);
            }
        }

        None
    }
}

// Event processor for handling events
pub struct EventProcessor {
    handlers: HashMap<String, Box<dyn EventHandler>>,
}

impl EventProcessor {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register_handler(&mut self, event_type: String, handler: Box<dyn EventHandler>) {
        self.handlers.insert(event_type, handler);
    }

    pub async fn process(&self, event: EventEnvelope) -> Result<(), EventError> {
        let event_type = match &event.event {
            GovernanceEvent::PolicyViolation { .. } => "PolicyViolation",
            GovernanceEvent::PolicyRemediated { .. } => "PolicyRemediated",
            GovernanceEvent::ResourceCreated { .. } => "ResourceCreated",
            GovernanceEvent::ResourceModified { .. } => "ResourceModified",
            GovernanceEvent::ResourceDeleted { .. } => "ResourceDeleted",
            GovernanceEvent::CostAnomaly { .. } => "CostAnomaly",
            GovernanceEvent::CostOptimization { .. } => "CostOptimization",
            GovernanceEvent::SecurityThreat { .. } => "SecurityThreat",
            GovernanceEvent::SecurityMitigation { .. } => "SecurityMitigation",
            GovernanceEvent::ComplianceCheck { .. } => "ComplianceCheck",
            GovernanceEvent::ComplianceDrift { .. } => "ComplianceDrift",
            GovernanceEvent::ActionInitiated { .. } => "ActionInitiated",
            GovernanceEvent::ActionProgress { .. } => "ActionProgress",
            GovernanceEvent::ActionCompleted { .. } => "ActionCompleted",
            GovernanceEvent::ActionFailed { .. } => "ActionFailed",
        };

        if let Some(handler) = self.handlers.get(event_type) {
            handler.handle(event).await?;
        }

        Ok(())
    }
}

#[async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: EventEnvelope) -> Result<(), EventError>;
}

// Error types
#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Publish error: {0}")]
    PublishError(String),
    #[error("Subscription error: {0}")]
    SubscriptionError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Handler error: {0}")]
    HandlerError(String),
}

// Event aggregator for analytics
pub struct EventAggregator {
    window_size: std::time::Duration,
    events: tokio::sync::RwLock<Vec<EventEnvelope>>,
}

impl EventAggregator {
    pub fn new(window_size: std::time::Duration) -> Self {
        Self {
            window_size,
            events: tokio::sync::RwLock::new(Vec::new()),
        }
    }

    pub async fn add_event(&self, event: EventEnvelope) {
        let mut events = self.events.write().await;
        events.push(event);

        // Clean old events
        let cutoff = Utc::now() - chrono::Duration::from_std(self.window_size).unwrap();
        events.retain(|e| e.timestamp > cutoff);
    }

    pub async fn get_statistics(&self) -> EventStatistics {
        let events = self.events.read().await;

        let mut by_type = HashMap::new();
        let mut by_tenant = HashMap::new();

        for event in events.iter() {
            *by_type.entry(format!("{:?}", event.event)).or_insert(0) += 1;
            *by_tenant.entry(event.tenant_id.clone()).or_insert(0) += 1;
        }

        EventStatistics {
            total_events: events.len(),
            events_by_type: by_type,
            events_by_tenant: by_tenant,
            window_size: self.window_size,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EventStatistics {
    pub total_events: usize,
    pub events_by_type: HashMap<String, usize>,
    pub events_by_tenant: HashMap<String, usize>,
    pub window_size: std::time::Duration,
}
