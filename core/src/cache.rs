use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, warn};
use tokio::time::{sleep, timeout};

// High-performance caching layer for Azure data
#[derive(Clone)]
pub struct CacheManager {
    redis_url: String,
    default_ttl: Duration,
    connection_pool: ConnectionManager,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub data: T,
    pub timestamp: i64,
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub redis_url: String,
    pub default_ttl_seconds: u64,
    pub max_connections: u32,
    pub connection_timeout_ms: u64,
    pub retry_attempts: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            default_ttl_seconds: 300, // 5 minutes default
            max_connections: 20,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
        }
    }
}

impl CacheManager {
    pub async fn new(config: CacheConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = redis::Client::open(config.redis_url.clone())?;
        let connection_manager = ConnectionManager::new(client).await?;

        Ok(Self {
            redis_url: config.redis_url,
            default_ttl: Duration::from_secs(config.default_ttl_seconds),
            connection_pool: connection_manager,
        })
    }

    // Hot data cache - ultra-fast access for critical governance data
    pub async fn get_hot<T>(&mut self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let hot_key = format!("hot:{}", key);
        self.get_with_retry(&hot_key).await
    }

    pub async fn set_hot<T>(&mut self, key: &str, value: &T, ttl: Option<Duration>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        let hot_key = format!("hot:{}", key);
        let cache_ttl = ttl.unwrap_or_else(|| Duration::from_secs(30)); // Hot data: 30 seconds
        self.set_with_retry(&hot_key, value, cache_ttl).await
    }

    // Warm data cache - frequently accessed data
    pub async fn get_warm<T>(&mut self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let warm_key = format!("warm:{}", key);
        self.get_with_retry(&warm_key).await
    }

    pub async fn set_warm<T>(&mut self, key: &str, value: &T, ttl: Option<Duration>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        let warm_key = format!("warm:{}", key);
        let cache_ttl = ttl.unwrap_or_else(|| Duration::from_secs(300)); // Warm data: 5 minutes
        self.set_with_retry(&warm_key, value, cache_ttl).await
    }

    // Cold data cache - infrequently accessed historical data
    pub async fn get_cold<T>(&mut self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let cold_key = format!("cold:{}", key);
        self.get_with_retry(&cold_key).await
    }

    pub async fn set_cold<T>(&mut self, key: &str, value: &T, ttl: Option<Duration>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        let cold_key = format!("cold:{}", key);
        let cache_ttl = ttl.unwrap_or_else(|| Duration::from_secs(3600)); // Cold data: 1 hour
        self.set_with_retry(&cold_key, value, cache_ttl).await
    }

    // Smart cache with automatic tiering based on access patterns
    pub async fn get_smart<T>(&mut self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de> + Clone + Serialize,
    {
        // Try hot cache first
        if let Some(data) = self.get_hot::<T>(key).await? {
            debug!("Cache HIT (hot): {}", key);
            return Ok(Some(data));
        }

        // Try warm cache
        if let Some(data) = self.get_warm::<T>(key).await? {
            debug!("Cache HIT (warm): {}", key);
            
            // Promote to hot cache on access
            if let Err(e) = self.set_hot(key, &data, None).await {
                warn!("Failed to promote to hot cache: {}", e);
            }
            
            return Ok(Some(data));
        }

        // Try cold cache
        if let Some(data) = self.get_cold::<T>(key).await? {
            debug!("Cache HIT (cold): {}", key);
            
            // Promote to warm cache on access
            if let Err(e) = self.set_warm(key, &data, None).await {
                warn!("Failed to promote to warm cache: {}", e);
            }
            
            return Ok(Some(data));
        }

        debug!("Cache MISS: {}", key);
        Ok(None)
    }

    pub async fn set_smart<T>(&mut self, key: &str, value: &T, access_pattern: CacheAccessPattern) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        match access_pattern {
            CacheAccessPattern::RealTime => self.set_hot(key, value, Some(Duration::from_secs(10))).await,
            CacheAccessPattern::Frequent => self.set_warm(key, value, Some(Duration::from_secs(300))).await,
            CacheAccessPattern::Occasional => self.set_cold(key, value, Some(Duration::from_secs(1800))).await,
            CacheAccessPattern::Rare => self.set_cold(key, value, Some(Duration::from_secs(7200))).await,
        }
    }

    // Batch operations for high-performance bulk caching
    pub async fn get_batch<T>(&mut self, keys: &[String]) -> Result<Vec<Option<T>>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut results = Vec::with_capacity(keys.len());
        
        // Use Redis pipeline for batch operations
        let mut pipe = redis::pipe();
        for key in keys {
            pipe.get(key);
        }

        let batch_results: Vec<Option<String>> = pipe.query_async(&mut self.connection_pool.clone()).await?;
        
        for result in batch_results {
            match result {
                Some(json_str) => {
                    match serde_json::from_str::<CacheEntry<T>>(&json_str) {
                        Ok(entry) => results.push(Some(entry.data)),
                        Err(_) => results.push(None),
                    }
                }
                None => results.push(None),
            }
        }

        Ok(results)
    }

    pub async fn set_batch<T>(&mut self, entries: &[(String, T, Duration)]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        let mut pipe = redis::pipe();
        
        for (key, value, ttl) in entries {
            let entry = CacheEntry {
                data: value,
                timestamp: chrono::Utc::now().timestamp(),
                version: "1.0".to_string(),
            };
            
            let json_str = serde_json::to_string(&entry)?;
            pipe.set_ex(key, json_str, ttl.as_secs() as usize);
        }

        let _: () = pipe.query_async(&mut self.connection_pool.clone()).await?;
        Ok(())
    }

    // Intelligent cache invalidation
    pub async fn invalidate_pattern(&mut self, pattern: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.connection_pool.clone();
        let keys: Vec<String> = conn.keys(pattern).await?;
        if keys.is_empty() {
            return Ok(0);
        }

        let deleted: u64 = conn.del(&keys).await?;
        debug!("Invalidated {} cache entries matching pattern: {}", deleted, pattern);
        Ok(deleted)
    }

    pub async fn invalidate_by_tags(&mut self, tags: &[String]) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut total_deleted = 0u64;
        
        for tag in tags {
            let pattern = format!("*:{}:*", tag);
            total_deleted += self.invalidate_pattern(&pattern).await?;
        }

        Ok(total_deleted)
    }

    // Cache statistics for monitoring
    pub async fn get_stats(&mut self) -> Result<CacheStats, Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.connection_pool.clone();
        let info: String = redis::cmd("INFO")
            .arg("memory")
            .query_async(&mut conn)
            .await?;

        let keyspace: String = redis::cmd("INFO")
            .arg("keyspace")
            .query_async(&mut conn)
            .await?;

        Ok(CacheStats {
            memory_info: info,
            keyspace_info: keyspace,
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    // Private helper methods
    async fn get_with_retry<T>(&mut self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: for<'de> Deserialize<'de>,
    {
        for attempt in 0..3 {
            let mut conn = self.connection_pool.clone();
            match timeout(
                Duration::from_millis(5000),
                conn.get::<&str, Option<String>>(key)
            ).await {
                Ok(Ok(Some(json_str))) => {
                    match serde_json::from_str::<CacheEntry<T>>(&json_str) {
                        Ok(entry) => return Ok(Some(entry.data)),
                        Err(e) => {
                            error!("Failed to deserialize cache entry: {}", e);
                            return Ok(None);
                        }
                    }
                }
                Ok(Ok(None)) => return Ok(None),
                Ok(Err(e)) => {
                    error!("Redis error on attempt {}: {}", attempt + 1, e);
                    if attempt < 2 {
                        sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                    }
                }
                Err(_) => {
                    error!("Cache timeout on attempt {}", attempt + 1);
                    if attempt < 2 {
                        sleep(Duration::from_millis(200)).await;
                    }
                }
            }
        }
        
        Ok(None)
    }

    async fn set_with_retry<T>(&mut self, key: &str, value: &T, ttl: Duration) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        let entry = CacheEntry {
            data: value,
            timestamp: chrono::Utc::now().timestamp(),
            version: "1.0".to_string(),
        };
        
        let json_str = serde_json::to_string(&entry)?;

        for attempt in 0..3 {
            let mut conn = self.connection_pool.clone();
            match timeout(
                Duration::from_millis(5000),
                conn.set_ex::<&str, String, ()>(key, json_str.clone(), ttl.as_secs() as usize)
            ).await {
                Ok(Ok(_)) => return Ok(()),
                Ok(Err(e)) => {
                    error!("Redis set error on attempt {}: {}", attempt + 1, e);
                    if attempt < 2 {
                        sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                    }
                }
                Err(_) => {
                    error!("Cache set timeout on attempt {}", attempt + 1);
                    if attempt < 2 {
                        sleep(Duration::from_millis(200)).await;
                    }
                }
            }
        }

        Err("Failed to set cache after retries".into())
    }
}

#[derive(Debug, Clone)]
pub enum CacheAccessPattern {
    RealTime,   // < 30 seconds (compliance violations, alerts)
    Frequent,   // 5 minutes (policies, resources, costs)
    Occasional, // 30 minutes (reports, analytics)
    Rare,       // 2+ hours (historical data, configurations)
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub memory_info: String,
    pub keyspace_info: String,
    pub timestamp: i64,
}

// Cache key builders for consistent naming
pub struct CacheKeys;

impl CacheKeys {
    pub fn governance_metrics(tenant_id: &str) -> String {
        format!("governance:metrics:tenant:{}", tenant_id)
    }

    pub fn policy_compliance(policy_id: &str) -> String {
        format!("policy:compliance:{}", policy_id)
    }

    pub fn resource_data(resource_id: &str) -> String {
        format!("resource:data:{}", resource_id)
    }

    pub fn cost_analysis(subscription_id: &str, date: &str) -> String {
        format!("cost:analysis:sub:{}:date:{}", subscription_id, date)
    }

    pub fn rbac_assignments(tenant_id: &str) -> String {
        format!("rbac:assignments:tenant:{}", tenant_id)
    }

    pub fn security_alerts(tenant_id: &str) -> String {
        format!("security:alerts:tenant:{}", tenant_id)
    }

    pub fn compliance_score(tenant_id: &str) -> String {
        format!("compliance:score:tenant:{}", tenant_id)
    }
}