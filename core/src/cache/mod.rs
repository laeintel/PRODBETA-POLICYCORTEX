// Cache management module for PolicyCortex

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::time::{Duration, Instant};
use anyhow::Result;

/// Cache access pattern for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAccessPattern {
    ReadHeavy,
    WriteHeavy,
    Balanced,
    RealTime,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size: usize,
    pub default_ttl: Duration,
    pub access_pattern: CacheAccessPattern,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            default_ttl: Duration::from_secs(300),
            access_pattern: CacheAccessPattern::Balanced,
        }
    }
}

/// Common cache keys
pub struct CacheKeys;

impl CacheKeys {
    pub const RESOURCES: &'static str = "resources:all";
    pub const POLICIES: &'static str = "policies:all";
    pub const COMPLIANCE: &'static str = "compliance:status";
    pub const METRICS: &'static str = "metrics:current";
    
    pub fn resource(id: &str) -> String {
        format!("resource:{}", id)
    }
    
    pub fn policy(id: &str) -> String {
        format!("policy:{}", id)
    }
    
    pub fn user_session(id: &str) -> String {
        format!("session:{}", id)
    }
    
    pub fn governance_metrics() -> String {
        "governance:metrics".to_string()
    }
}

/// Cache entry with TTL support
#[derive(Debug, Clone)]
struct CacheEntry {
    value: Value,
    expires_at: Instant,
}

/// Thread-safe cache manager
#[derive(Debug, Clone)]
pub struct CacheManager {
    store: Arc<RwLock<HashMap<String, CacheEntry>>>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set a value in the cache with TTL in seconds
    pub async fn set<T: Serialize>(&self, key: &str, value: &T, ttl_seconds: u64) -> Result<()> {
        let json_value = serde_json::to_value(value)?;
        let entry = CacheEntry {
            value: json_value,
            expires_at: Instant::now() + Duration::from_secs(ttl_seconds),
        };
        
        let mut store = self.store.write().await;
        store.insert(key.to_string(), entry);
        Ok(())
    }

    /// Get a value from the cache
    pub async fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        let store = self.store.read().await;
        
        if let Some(entry) = store.get(key) {
            // Check if entry has expired
            if entry.expires_at > Instant::now() {
                // Try to deserialize the value
                if let Ok(value) = serde_json::from_value(entry.value.clone()) {
                    return Some(value);
                }
            }
        }
        
        None
    }

    /// Get raw JSON value from cache
    pub async fn get_raw(&self, key: &str) -> Option<Value> {
        let store = self.store.read().await;
        
        if let Some(entry) = store.get(key) {
            if entry.expires_at > Instant::now() {
                return Some(entry.value.clone());
            }
        }
        
        None
    }

    /// Remove a value from the cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        let mut store = self.store.write().await;
        store.remove(key);
        Ok(())
    }

    /// Clear all expired entries
    pub async fn clear_expired(&self) -> Result<()> {
        let mut store = self.store.write().await;
        let now = Instant::now();
        store.retain(|_, entry| entry.expires_at > now);
        Ok(())
    }

    /// Clear all cache entries
    pub async fn clear_all(&self) -> Result<()> {
        let mut store = self.store.write().await;
        store.clear();
        Ok(())
    }

    /// Get the number of cached entries
    pub async fn size(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }
    
    /// Get from hot cache (short TTL)
    pub async fn get_hot<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.get(key).await
    }
    
    /// Set with smart caching based on access pattern
    pub async fn set_smart<T: Serialize>(&self, key: &str, value: &T, _pattern: CacheAccessPattern) -> Result<()> {
        // Use default TTL for now, could optimize based on pattern
        self.set(key, value, 300).await
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_set_get() {
        let cache = CacheManager::new();
        
        // Test setting and getting a string value
        cache.set("test_key", &"test_value".to_string(), 60).await.unwrap();
        let value: Option<String> = cache.get("test_key").await;
        assert_eq!(value, Some("test_value".to_string()));
    }

    #[tokio::test]
    async fn test_cache_expiry() {
        let cache = CacheManager::new();
        
        // Set with 0 second TTL (immediate expiry)
        cache.set("expired_key", &"value", 0).await.unwrap();
        
        // Wait a moment
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Should not be retrievable
        let value: Option<String> = cache.get("expired_key").await;
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_cache_delete() {
        let cache = CacheManager::new();
        
        cache.set("delete_key", &"value", 60).await.unwrap();
        cache.delete("delete_key").await.unwrap();
        
        let value: Option<String> = cache.get("delete_key").await;
        assert_eq!(value, None);
    }
}