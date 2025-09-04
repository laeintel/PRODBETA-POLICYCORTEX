// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

//! CQRS (Command Query Responsibility Segregation) implementation
//! Separates read and write models for optimal performance and scalability

pub mod commands;
pub mod queries;
pub mod events;
pub mod projections;
pub mod aggregate;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Base trait for all commands in the system
#[async_trait]
pub trait Command: Send + Sync + Debug {
    /// The type of aggregate this command targets
    type Aggregate: Aggregate;
    
    /// The result type returned by executing this command
    type Result: Send + Sync;
    
    /// Execute the command and return the result
    async fn execute(&self, aggregate: &mut Self::Aggregate) -> Result<Self::Result>;
    
    /// Get the aggregate ID this command targets
    fn aggregate_id(&self) -> Uuid;
    
    /// Validate the command before execution
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Base trait for all queries in the system
#[async_trait]
pub trait Query: Send + Sync + Debug {
    /// The result type returned by executing this query
    type Result: Send + Sync + for<'de> Deserialize<'de> + Serialize;
    
    /// Execute the query against the read model
    async fn execute(&self, read_store: &ReadStore) -> Result<Self::Result>;
    
    /// Cache key for this query (if cacheable)
    fn cache_key(&self) -> Option<String> {
        None
    }
    
    /// Cache TTL in seconds (if cacheable)
    fn cache_ttl(&self) -> Option<u64> {
        None
    }
}

/// Base trait for aggregates (domain entities)
#[async_trait]
pub trait Aggregate: Send + Sync + Sized {
    /// The type of events this aggregate produces
    type Event: DomainEvent;
    
    /// Create a new aggregate instance
    fn new(id: Uuid) -> Self;
    
    /// Apply an event to update the aggregate state
    fn apply(&mut self, event: &Self::Event);
    
    /// Get the aggregate ID
    fn id(&self) -> Uuid;
    
    /// Get the current version of the aggregate
    fn version(&self) -> u64;
    
    /// Increment the version
    fn increment_version(&mut self);
}

/// Base trait for domain events
pub trait DomainEvent: Send + Sync + Debug {
    /// Get the aggregate ID this event belongs to
    fn aggregate_id(&self) -> Uuid;
    
    /// Get the event timestamp
    fn occurred_at(&self) -> DateTime<Utc>;
    
    /// Get the event type name
    fn event_type(&self) -> &'static str;
    
    /// Get the event version
    fn version(&self) -> u64;
    
    /// Serialize the event to JSON
    fn as_json(&self) -> serde_json::Value;
    
    /// Clone the event as a boxed trait object
    fn clone_box(&self) -> Box<dyn DomainEvent>;
}

/// Command handler that processes commands
#[async_trait]
pub trait CommandHandler<C: Command>: Send + Sync {
    /// Handle a command and return the result
    async fn handle(&self, command: C) -> Result<C::Result>;
}

/// Query handler that processes queries
#[async_trait]
pub trait QueryHandler<Q: Query>: Send + Sync {
    /// Handle a query and return the result
    async fn handle(&self, query: Q) -> Result<Q::Result>;
}

/// Event store for persisting domain events
#[async_trait]
pub trait EventStore: Send + Sync {
    /// Save events to the store
    async fn save_events(&self, events: Vec<Box<dyn DomainEvent>>) -> Result<()>;
    
    /// Load events for an aggregate
    async fn load_events(&self, aggregate_id: Uuid, after_version: Option<u64>) -> Result<Vec<Box<dyn DomainEvent>>>;
    
    /// Get the current version of an aggregate
    async fn get_version(&self, aggregate_id: Uuid) -> Result<u64>;
}

/// Read store for querying projected data
pub struct ReadStore {
    pool: crate::db::SharedDbPool,
    cache: crate::cache::CacheManager,
}

impl ReadStore {
    /// Create a new read store
    pub fn new(pool: crate::db::SharedDbPool, cache: crate::cache::CacheManager) -> Self {
        Self { pool, cache }
    }
    
    /// Get a database connection from the pool
    pub async fn conn(&self) -> Result<sqlx::pool::PoolConnection<sqlx::Postgres>> {
        Ok(self.pool.pool().acquire().await?)
    }
    
    /// Get the cache manager
    pub fn cache(&self) -> &crate::cache::CacheManager {
        &self.cache
    }
}

/// Command bus for dispatching commands to handlers
pub struct CommandBus {
    handlers: std::collections::HashMap<std::any::TypeId, Box<dyn std::any::Any + Send + Sync>>,
    event_store: Box<dyn EventStore>,
}

impl CommandBus {
    /// Create a new command bus
    pub fn new(event_store: Box<dyn EventStore>) -> Self {
        Self {
            handlers: std::collections::HashMap::new(),
            event_store,
        }
    }
    
    /// Register a command handler
    pub fn register_handler<C: Command + 'static>(&mut self, handler: Box<dyn CommandHandler<C>>) {
        let type_id = std::any::TypeId::of::<C>();
        self.handlers.insert(type_id, Box::new(handler) as Box<dyn std::any::Any + Send + Sync>);
    }
    
    /// Dispatch a command to its handler
    pub async fn dispatch<C: Command + 'static>(&self, command: C) -> Result<C::Result> {
        let type_id = std::any::TypeId::of::<C>();
        
        let handler = self.handlers
            .get(&type_id)
            .ok_or_else(|| anyhow::anyhow!("No handler registered for command"))?;
        
        let handler = handler
            .downcast_ref::<Box<dyn CommandHandler<C>>>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast handler"))?;
        
        // Validate command
        command.validate()?;
        
        // Execute command
        handler.handle(command).await
    }
}

/// Query bus for dispatching queries to handlers
pub struct QueryBus {
    handlers: std::collections::HashMap<std::any::TypeId, Box<dyn std::any::Any + Send + Sync>>,
    read_store: ReadStore,
}

impl QueryBus {
    /// Create a new query bus
    pub fn new(read_store: ReadStore) -> Self {
        Self {
            handlers: std::collections::HashMap::new(),
            read_store,
        }
    }
    
    /// Register a query handler
    pub fn register_handler<Q: Query + 'static>(&mut self, handler: Box<dyn QueryHandler<Q>>) {
        let type_id = std::any::TypeId::of::<Q>();
        self.handlers.insert(type_id, Box::new(handler) as Box<dyn std::any::Any + Send + Sync>);
    }
    
    /// Dispatch a query to its handler
    pub async fn dispatch<Q: Query + 'static>(&self, query: Q) -> Result<Q::Result> {
        // Check cache if query is cacheable
        let cache_key = query.cache_key();
        if let Some(ref key) = cache_key {
            if let Some(cached) = self.read_store.cache().get::<Q::Result>(key).await {
                return Ok(cached);
            }
        }
        
        let type_id = std::any::TypeId::of::<Q>();
        
        let handler = self.handlers
            .get(&type_id)
            .ok_or_else(|| anyhow::anyhow!("No handler registered for query"))?;
        
        let handler = handler
            .downcast_ref::<Box<dyn QueryHandler<Q>>>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast handler"))?;
        
        let cache_ttl = query.cache_ttl();
        let result = handler.handle(query).await?;
        
        // Cache result if query is cacheable
        if let Some(key) = cache_key {
            if let Some(ttl) = cache_ttl {
                let _ = self.read_store.cache().set(&key, &result, ttl).await;
            }
        }
        
        Ok(result)
    }
}

/// CQRS system that coordinates commands and queries
pub struct CQRSSystem {
    command_bus: CommandBus,
    query_bus: QueryBus,
}

impl CQRSSystem {
    /// Create a new CQRS system
    pub fn new(
        event_store: Box<dyn EventStore>,
        pool: crate::db::SharedDbPool,
        cache: crate::cache::CacheManager,
    ) -> Self {
        let read_store = ReadStore::new(pool, cache);
        
        Self {
            command_bus: CommandBus::new(event_store),
            query_bus: QueryBus::new(read_store),
        }
    }
    
    /// Get a mutable reference to the command bus for registration
    pub fn command_bus_mut(&mut self) -> &mut CommandBus {
        &mut self.command_bus
    }
    
    /// Get a mutable reference to the query bus for registration
    pub fn query_bus_mut(&mut self) -> &mut QueryBus {
        &mut self.query_bus
    }
    
    /// Execute a command
    pub async fn execute_command<C: Command + 'static>(&self, command: C) -> Result<C::Result> {
        self.command_bus.dispatch(command).await
    }
    
    /// Execute a query
    pub async fn execute_query<Q: Query + 'static>(&self, query: Q) -> Result<Q::Result> {
        self.query_bus.dispatch(query).await
    }
}