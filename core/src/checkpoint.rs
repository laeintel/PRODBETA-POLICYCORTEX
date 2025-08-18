use chrono::{DateTime, Utc};
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, RedisError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionCheckpoint {
    pub source: String,
    pub last_processed_timestamp: DateTime<Utc>,
    pub last_processed_id: Option<String>,
    pub last_etag: Option<String>,
    pub records_processed: u64,
    pub last_error: Option<String>,
    pub retry_count: u32,
}

impl IngestionCheckpoint {
    pub fn new(source: String) -> Self {
        IngestionCheckpoint {
            source,
            last_processed_timestamp: Utc::now(),
            last_processed_id: None,
            last_etag: None,
            records_processed: 0,
            last_error: None,
            retry_count: 0,
        }
    }

    pub fn update_progress(&mut self, record_id: Option<String>, etag: Option<String>) {
        self.last_processed_timestamp = Utc::now();
        if let Some(id) = record_id {
            self.last_processed_id = Some(id);
        }
        if let Some(tag) = etag {
            self.last_etag = Some(tag);
        }
        self.records_processed += 1;
        self.last_error = None;
        self.retry_count = 0;
    }

    pub fn record_error(&mut self, error: String) {
        self.last_error = Some(error);
        self.retry_count += 1;
    }
}

pub struct CheckpointManager {
    redis_conn: ConnectionManager,
    namespace: String,
}

impl CheckpointManager {
    pub async fn new(redis_url: &str, namespace: &str) -> Result<Self, RedisError> {
        let client = redis::Client::open(redis_url)?;
        let conn = ConnectionManager::new(client).await?;
        
        Ok(CheckpointManager {
            redis_conn: conn,
            namespace: namespace.to_string(),
        })
    }

    fn get_key(&self, source: &str) -> String {
        format!("checkpoint:{}:{}", self.namespace, source)
    }

    pub async fn save_checkpoint(&mut self, checkpoint: &IngestionCheckpoint) -> Result<(), RedisError> {
        let key = self.get_key(&checkpoint.source);
        let serialized = serde_json::to_string(checkpoint)
            .map_err(|e| RedisError::from((redis::ErrorKind::TypeError, "Serialization failed", e.to_string())))?;
        
        // Set with 7-day expiration (604800 seconds)
        self.redis_conn.set_ex::<_, _, ()>(&key, serialized, 604800).await?;
        
        debug!(
            "Saved checkpoint for source '{}': {} records processed, last_id: {:?}",
            checkpoint.source, checkpoint.records_processed, checkpoint.last_processed_id
        );
        
        Ok(())
    }

    pub async fn load_checkpoint(&mut self, source: &str) -> Result<Option<IngestionCheckpoint>, RedisError> {
        let key = self.get_key(source);
        let value: Option<String> = self.redis_conn.get(&key).await?;
        
        match value {
            Some(json_str) => {
                let checkpoint: IngestionCheckpoint = serde_json::from_str(&json_str)
                    .map_err(|e| RedisError::from((redis::ErrorKind::TypeError, "Deserialization failed", e.to_string())))?;
                
                info!(
                    "Loaded checkpoint for source '{}': {} records processed, last update: {}",
                    source, checkpoint.records_processed, checkpoint.last_processed_timestamp
                );
                
                Ok(Some(checkpoint))
            }
            None => {
                info!("No checkpoint found for source '{}', starting fresh", source);
                Ok(None)
            }
        }
    }

    pub async fn delete_checkpoint(&mut self, source: &str) -> Result<(), RedisError> {
        let key = self.get_key(source);
        self.redis_conn.del::<_, ()>(&key).await?;
        info!("Deleted checkpoint for source '{}'", source);
        Ok(())
    }

    pub async fn list_checkpoints(&mut self) -> Result<HashMap<String, IngestionCheckpoint>, RedisError> {
        let pattern = format!("checkpoint:{}:*", self.namespace);
        let keys: Vec<String> = self.redis_conn.keys(&pattern).await?;
        
        let mut checkpoints = HashMap::new();
        
        for key in keys {
            let value: Option<String> = self.redis_conn.get(&key).await?;
            if let Some(json_str) = value {
                if let Ok(checkpoint) = serde_json::from_str::<IngestionCheckpoint>(&json_str) {
                    checkpoints.insert(checkpoint.source.clone(), checkpoint);
                }
            }
        }
        
        Ok(checkpoints)
    }

    pub async fn get_stale_checkpoints(&mut self, stale_after_hours: i64) -> Result<Vec<String>, RedisError> {
        let checkpoints = self.list_checkpoints().await?;
        let cutoff_time = Utc::now() - chrono::Duration::hours(stale_after_hours);
        
        let stale_sources: Vec<String> = checkpoints
            .into_iter()
            .filter(|(_, checkpoint)| checkpoint.last_processed_timestamp < cutoff_time)
            .map(|(source, _)| source)
            .collect();
        
        if !stale_sources.is_empty() {
            warn!(
                "Found {} stale checkpoints (no updates in {} hours): {:?}",
                stale_sources.len(),
                stale_after_hours,
                stale_sources
            );
        }
        
        Ok(stale_sources)
    }
}

// Azure-specific checkpoint sources
pub mod azure_sources {
    pub const RESOURCE_GRAPH: &str = "azure_resource_graph";
    pub const POLICY_COMPLIANCE: &str = "azure_policy_compliance";
    pub const COST_MANAGEMENT: &str = "azure_cost_management";
    pub const ACTIVITY_LOG: &str = "azure_activity_log";
    pub const DEFENDER_ALERTS: &str = "azure_defender_alerts";
    pub const ADVISOR_RECOMMENDATIONS: &str = "azure_advisor";
    pub const MONITOR_METRICS: &str = "azure_monitor_metrics";
    pub const KEYVAULT_EVENTS: &str = "azure_keyvault_events";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = IngestionCheckpoint::new("test_source".to_string());
        assert_eq!(checkpoint.source, "test_source");
        assert_eq!(checkpoint.records_processed, 0);
        assert!(checkpoint.last_processed_id.is_none());
        assert!(checkpoint.last_error.is_none());
    }

    #[test]
    fn test_checkpoint_update() {
        let mut checkpoint = IngestionCheckpoint::new("test_source".to_string());
        checkpoint.update_progress(Some("record_123".to_string()), Some("etag_456".to_string()));
        
        assert_eq!(checkpoint.records_processed, 1);
        assert_eq!(checkpoint.last_processed_id, Some("record_123".to_string()));
        assert_eq!(checkpoint.last_etag, Some("etag_456".to_string()));
        assert!(checkpoint.last_error.is_none());
        assert_eq!(checkpoint.retry_count, 0);
    }

    #[test]
    fn test_checkpoint_error_handling() {
        let mut checkpoint = IngestionCheckpoint::new("test_source".to_string());
        checkpoint.record_error("Connection timeout".to_string());
        
        assert_eq!(checkpoint.last_error, Some("Connection timeout".to_string()));
        assert_eq!(checkpoint.retry_count, 1);
        
        checkpoint.record_error("Another error".to_string());
        assert_eq!(checkpoint.retry_count, 2);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = IngestionCheckpoint::new("test_source".to_string());
        let serialized = serde_json::to_string(&checkpoint).unwrap();
        let deserialized: IngestionCheckpoint = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(checkpoint.source, deserialized.source);
        assert_eq!(checkpoint.records_processed, deserialized.records_processed);
    }
}