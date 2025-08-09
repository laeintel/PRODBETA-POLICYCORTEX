# Performance Optimization

## Table of Contents
1. [Performance Architecture](#performance-architecture)
2. [Database Optimization](#database-optimization)
3. [Caching Strategies](#caching-strategies)
4. [API Performance](#api-performance)
5. [Frontend Optimization](#frontend-optimization)
6. [AI/ML Performance](#aiml-performance)
7. [System Monitoring](#system-monitoring)
8. [Scaling Strategies](#scaling-strategies)
9. [Benchmarks and Metrics](#benchmarks-and-metrics)

## Performance Architecture

PolicyCortex is designed for sub-millisecond response times and massive scale through multi-layered performance optimizations:

```
┌─────────────────────────────────────────────────────────────┐
│                Performance Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Edge Layer    │  │  Application    │  │   Data      │  │
│  │                 │  │     Layer       │  │   Layer     │  │
│  │ • CDN           │  │ • Rust Core     │  │ • PostgreSQL│  │
│  │ • Edge Cache    │  │ • Next.js SSG   │  │ • DragonflyDB│ │
│  │ • WebAssembly   │  │ • GraphQL       │  │ • EventStore│  │
│  │ • Geographic    │  │ • Load Balancer │  │ • Partitions│  │
│  │   Distribution  │  │ • Circuit Breaker│ │ • Indexes   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Performance Monitoring                     │  │
│  │  • Real-time metrics collection                        │  │
│  │  • Distributed tracing                                 │  │
│  │  • Performance regression detection                    │  │
│  │  • Automated optimization triggers                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Performance Targets

- **API Response Time**: < 50ms (P95), < 10ms (P50)
- **Database Query Time**: < 5ms (P95), < 1ms (P50)
- **Cache Hit Ratio**: > 95% for hot data
- **Throughput**: > 10,000 requests/second per instance
- **UI Load Time**: < 2 seconds (First Contentful Paint)
- **Real-time Updates**: < 100ms latency

## Database Optimization

### Query Optimization

```rust
// core/src/database/query_optimizer.rs
use sqlx::{PgPool, QueryBuilder, Postgres, Row};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};

pub struct QueryOptimizer {
    connection_pool: PgPool,
    query_cache: HashMap<String, CachedQuery>,
    performance_stats: QueryPerformanceStats,
}

#[derive(Debug, Clone)]
pub struct CachedQuery {
    pub sql: String,
    pub execution_plan: Option<String>,
    pub avg_duration_ms: f64,
    pub hit_count: u64,
    pub last_used: Instant,
}

#[derive(Debug, Default)]
pub struct QueryPerformanceStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub slow_queries: u64,
    pub avg_response_time: f64,
}

impl QueryOptimizer {
    pub fn new(pool: PgPool) -> Self {
        Self {
            connection_pool: pool,
            query_cache: HashMap::new(),
            performance_stats: QueryPerformanceStats::default(),
        }
    }

    // Optimized resource queries with intelligent indexing
    pub async fn get_resources_optimized(
        &mut self,
        filters: &ResourceFilters,
        pagination: &PaginationParams,
    ) -> Result<Vec<Resource>, sqlx::Error> {
        let start_time = Instant::now();
        let query_key = self.generate_query_key("resources", filters);

        // Check query cache
        if let Some(cached) = self.query_cache.get(&query_key) {
            self.performance_stats.cache_hits += 1;
            cached.hit_count += 1;
        }

        let mut query = QueryBuilder::new(
            "SELECT r.id, r.name, r.type, r.subscription_id, r.location, 
             r.resource_group_name, r.properties, r.tags, r.created_at, r.updated_at"
        );

        query.push(" FROM resources r");
        
        // Add covering indexes hints
        if filters.subscription_id.is_some() {
            query.push(" USE INDEX (idx_resources_subscription_covering)");
        }

        query.push(" WHERE r.deleted_at IS NULL");

        // Optimized filter application
        self.apply_optimized_filters(&mut query, filters);

        // Intelligent ordering based on most common access patterns
        if filters.order_by.is_none() {
            query.push(" ORDER BY r.updated_at DESC, r.id");
        }

        // Efficient pagination with cursor-based approach for large datasets
        if pagination.use_cursor && pagination.after_cursor.is_some() {
            query.push(" AND (r.updated_at, r.id) < ");
            query.push_bind(pagination.after_cursor.as_ref().unwrap());
        } else {
            query.push(" OFFSET ");
            query.push_bind((pagination.page - 1) * pagination.limit);
        }

        query.push(" LIMIT ");
        query.push_bind(pagination.limit);

        let built_query = query.build_query_as::<Resource>();
        let result = built_query.fetch_all(&self.connection_pool).await?;

        // Record performance metrics
        let duration = start_time.elapsed();
        self.update_performance_stats(duration, result.len());

        // Cache hot queries
        if result.len() > 0 {
            self.cache_query(query_key, built_query.sql(), duration);
        }

        Ok(result)
    }

    // Batch operations for improved throughput
    pub async fn batch_insert_resources(
        &self,
        resources: Vec<CreateResourceRequest>,
    ) -> Result<Vec<uuid::Uuid>, sqlx::Error> {
        let batch_size = 1000; // Optimal batch size for PostgreSQL
        let mut inserted_ids = Vec::new();

        for chunk in resources.chunks(batch_size) {
            let mut query_builder = QueryBuilder::new(
                "INSERT INTO resources (id, tenant_id, azure_resource_id, subscription_id, 
                 resource_group_name, name, type, location, properties, tags, created_at, updated_at)"
            );

            query_builder.push_values(chunk, |mut b, resource| {
                let id = uuid::Uuid::new_v4();
                let now = chrono::Utc::now();
                
                b.push_bind(id)
                    .push_bind(resource.tenant_id)
                    .push_bind(&resource.azure_resource_id)
                    .push_bind(&resource.subscription_id)
                    .push_bind(&resource.resource_group_name)
                    .push_bind(&resource.name)
                    .push_bind(&resource.resource_type)
                    .push_bind(&resource.location)
                    .push_bind(&resource.properties)
                    .push_bind(&resource.tags)
                    .push_bind(now)
                    .push_bind(now);

                inserted_ids.push(id);
            });

            query_builder.push(" RETURNING id");
            query_builder.build().execute(&self.connection_pool).await?;
        }

        Ok(inserted_ids)
    }

    // Connection pool optimization
    pub async fn optimize_connection_pool(&self) -> Result<(), sqlx::Error> {
        // Warm up connections
        let warm_up_queries = vec![
            "SELECT 1",
            "SELECT COUNT(*) FROM resources WHERE deleted_at IS NULL",
            "SELECT COUNT(*) FROM policies WHERE enabled = true",
        ];

        for query in warm_up_queries {
            sqlx::query(query).execute(&self.connection_pool).await?;
        }

        // Analyze and update table statistics
        self.update_table_statistics().await?;

        Ok(())
    }

    async fn update_table_statistics(&self) -> Result<(), sqlx::Error> {
        let tables = vec!["resources", "policies", "policy_evaluations", "cost_data"];
        
        for table in tables {
            let analyze_query = format!("ANALYZE {}", table);
            sqlx::query(&analyze_query).execute(&self.connection_pool).await?;
        }

        Ok(())
    }

    fn apply_optimized_filters(&self, query: &mut QueryBuilder<Postgres>, filters: &ResourceFilters) {
        // Most selective filters first for optimal query planning
        if let Some(subscription_id) = &filters.subscription_id {
            query.push(" AND r.subscription_id = ");
            query.push_bind(subscription_id);
        }

        if let Some(resource_type) = &filters.resource_type {
            query.push(" AND r.type = ");
            query.push_bind(resource_type);
        }

        if let Some(location) = &filters.location {
            query.push(" AND r.location = ");
            query.push_bind(location);
        }

        if let Some(resource_group) = &filters.resource_group_name {
            query.push(" AND r.resource_group_name = ");
            query.push_bind(resource_group);
        }

        // Tag filtering with GIN index optimization
        if let Some(tags) = &filters.tags {
            for tag in tags {
                if let Some((key, value)) = tag.split_once(':') {
                    query.push(" AND r.tags @> ");
                    query.push_bind(serde_json::json!({key: value}));
                } else {
                    query.push(" AND r.tags ? ");
                    query.push_bind(tag);
                }
            }
        }

        // Compliance status with subquery optimization
        if let Some(compliance_status) = &filters.compliance_status {
            query.push(" AND EXISTS (
                SELECT 1 FROM policy_evaluations pe 
                WHERE pe.resource_id = r.id 
                AND pe.result = ");
            query.push_bind(compliance_status);
            query.push(" AND pe.evaluated_at = (
                SELECT MAX(evaluated_at) 
                FROM policy_evaluations 
                WHERE resource_id = r.id AND policy_id = pe.policy_id
            ))");
        }
    }
}

// Specialized indexes for performance
const PERFORMANCE_INDEXES: &[&str] = &[
    // Covering index for most common resource queries
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_resources_subscription_covering 
     ON resources (subscription_id, deleted_at, updated_at) 
     INCLUDE (id, name, type, location, resource_group_name)",
    
    // Composite index for complex filtering
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_resources_multi_filter 
     ON resources (subscription_id, type, location, deleted_at) 
     WHERE deleted_at IS NULL",
    
    // GIN index for tag searches
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_resources_tags_gin 
     ON resources USING GIN (tags)",
    
    // Partial index for active policies
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policies_active 
     ON policies (type, category, severity) 
     WHERE enabled = true AND deleted_at IS NULL",
    
    // Composite index for policy evaluations
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policy_evaluations_latest 
     ON policy_evaluations (resource_id, policy_id, evaluated_at DESC, result)",
    
    // Time-based partitioning index for cost data
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_data_time_partition 
     ON cost_data (date, resource_id, service_name) 
     WHERE date >= CURRENT_DATE - INTERVAL '1 year'",
];
```

### Advanced Indexing Strategy

```sql
-- scripts/performance/advanced_indexes.sql

-- Composite partial indexes for hot queries
CREATE INDEX CONCURRENTLY idx_resources_hot_queries 
ON resources (tenant_id, subscription_id, type) 
WHERE deleted_at IS NULL AND updated_at >= NOW() - INTERVAL '30 days';

-- Expression indexes for computed columns
CREATE INDEX CONCURRENTLY idx_resources_tag_count 
ON resources ((jsonb_array_length(jsonb_object_keys(tags)))) 
WHERE deleted_at IS NULL;

-- Specialized index for compliance queries
CREATE INDEX CONCURRENTLY idx_compliance_current_status 
ON policy_evaluations (resource_id, result, evaluated_at DESC) 
WHERE evaluated_at >= NOW() - INTERVAL '24 hours';

-- Time-series optimization for metrics
CREATE INDEX CONCURRENTLY idx_metrics_time_series 
ON cost_data (resource_id, date DESC, service_name) 
WHERE date >= CURRENT_DATE - INTERVAL '90 days';

-- Full-text search optimization
CREATE INDEX CONCURRENTLY idx_resources_fulltext 
ON resources USING GIN (
    to_tsvector('english', name || ' ' || COALESCE(properties->>'description', ''))
);

-- Materialized views for complex aggregations
CREATE MATERIALIZED VIEW mv_compliance_summary AS
SELECT 
    r.tenant_id,
    r.subscription_id,
    r.type as resource_type,
    COUNT(*) as total_resources,
    COUNT(CASE WHEN pe.result = 'compliant' THEN 1 END) as compliant_resources,
    COUNT(CASE WHEN pe.result = 'non_compliant' THEN 1 END) as non_compliant_resources,
    ROUND(AVG(CASE WHEN pe.result = 'compliant' THEN 100.0 ELSE 0.0 END), 2) as compliance_percentage
FROM resources r
LEFT JOIN LATERAL (
    SELECT result 
    FROM policy_evaluations pe 
    WHERE pe.resource_id = r.id 
    ORDER BY evaluated_at DESC 
    LIMIT 1
) pe ON true
WHERE r.deleted_at IS NULL
GROUP BY r.tenant_id, r.subscription_id, r.type;

-- Unique index on materialized view
CREATE UNIQUE INDEX idx_mv_compliance_summary_pk 
ON mv_compliance_summary (tenant_id, subscription_id, resource_type);

-- Refresh strategy for materialized views
CREATE OR REPLACE FUNCTION refresh_compliance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_compliance_summary;
    PERFORM pg_stat_reset_single_table_counters('mv_compliance_summary'::regclass);
END;
$$ LANGUAGE plpgsql;

-- Automated refresh using pg_cron (if available)
-- SELECT cron.schedule('refresh-compliance-summary', '*/5 * * * *', 'SELECT refresh_compliance_summary();');
```

## Caching Strategies

### Multi-Level Caching Architecture

```rust
// core/src/cache/multi_level.rs
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

pub struct MultiLevelCache {
    // L1: In-memory cache (fastest)
    l1_cache: Arc<RwLock<L1Cache>>,
    
    // L2: DragonflyDB (distributed)
    l2_cache: DragonflyClient,
    
    // L3: Read replicas (slowest but most comprehensive)
    l3_cache: ReadReplicaCache,
    
    // Cache policies
    policies: CachingPolicies,
    
    // Performance metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

#[derive(Debug, Clone)]
pub struct CachingPolicies {
    pub l1_max_size: usize,
    pub l1_ttl: Duration,
    pub l2_ttl: Duration,
    pub l3_ttl: Duration,
    pub prefetch_threshold: f64,
    pub compression_threshold: usize,
}

impl Default for CachingPolicies {
    fn default() -> Self {
        Self {
            l1_max_size: 10_000,
            l1_ttl: Duration::from_secs(60),      // 1 minute
            l2_ttl: Duration::from_secs(1800),    // 30 minutes
            l3_ttl: Duration::from_secs(3600),    // 1 hour
            prefetch_threshold: 0.8,               // 80% hit rate triggers prefetch
            compression_threshold: 1024,           // 1KB
        }
    }
}

#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub prefetch_count: u64,
    pub evictions: u64,
    pub compression_saves: u64,
}

impl MultiLevelCache {
    pub fn new(l2_client: DragonflyClient, l3_client: ReadReplicaCache) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(L1Cache::new())),
            l2_cache: l2_client,
            l3_cache: l3_client,
            policies: CachingPolicies::default(),
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    // Intelligent get with automatic promotion
    pub async fn get<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de> + Serialize + Clone + Send + 'static,
    {
        let start_time = Instant::now();

        // Try L1 cache first
        if let Some(value) = self.get_from_l1::<T>(key).await {
            self.record_hit("l1", start_time.elapsed()).await;
            return Some(value);
        }

        // Try L2 cache (DragonflyDB)
        if let Ok(Some(value)) = self.get_from_l2::<T>(key).await {
            self.record_hit("l2", start_time.elapsed()).await;
            
            // Promote to L1
            self.set_l1(key, &value, self.policies.l1_ttl).await;
            return Some(value);
        }

        // Try L3 cache (Read replica)
        if let Ok(Some(value)) = self.get_from_l3::<T>(key).await {
            self.record_hit("l3", start_time.elapsed()).await;
            
            // Promote to L2 and L1
            let _ = self.set_l2(key, &value, self.policies.l2_ttl).await;
            self.set_l1(key, &value, self.policies.l1_ttl).await;
            return Some(value);
        }

        // Cache miss
        self.record_miss(start_time.elapsed()).await;
        None
    }

    // Intelligent set with write-through strategy
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Duration) -> Result<(), CacheError>
    where
        T: Serialize + Clone + Send + 'static,
    {
        // Determine optimal caching strategy based on data size and access patterns
        let serialized_size = self.estimate_serialized_size(value);
        let access_pattern = self.analyze_access_pattern(key).await;

        // Always cache in L1 for hot data
        if access_pattern.is_hot() || serialized_size < self.policies.compression_threshold {
            self.set_l1(key, value, self.policies.l1_ttl).await;
        }

        // Cache in L2 based on access pattern and size
        if access_pattern.should_cache_l2() {
            let compressed = if serialized_size > self.policies.compression_threshold {
                self.compress_value(value)?
            } else {
                serde_json::to_vec(value)?
            };
            
            self.set_l2_raw(key, &compressed, ttl).await?;
        }

        // L3 for long-term storage of expensive computations
        if access_pattern.should_cache_l3() {
            self.set_l3(key, value, self.policies.l3_ttl).await?;
        }

        Ok(())
    }

    // Prefetching for predictive caching
    pub async fn prefetch_related(&self, key: &str) -> Result<(), CacheError> {
        let related_keys = self.get_related_keys(key).await;
        let prefetch_tasks = related_keys.into_iter().map(|related_key| {
            let cache = self.clone();
            tokio::spawn(async move {
                if let Some(value) = cache.l3_cache.get_raw(&related_key).await.ok().flatten() {
                    cache.set_l2_raw(&related_key, &value, cache.policies.l2_ttl).await.ok();
                }
            })
        });

        // Execute prefetch tasks concurrently
        futures::future::join_all(prefetch_tasks).await;
        
        let mut metrics = self.metrics.write().await;
        metrics.prefetch_count += 1;
        
        Ok(())
    }

    // Cache warming for application startup
    pub async fn warm_cache(&self, warming_plan: Vec<CacheWarmingEntry>) -> Result<(), CacheError> {
        const BATCH_SIZE: usize = 100;
        
        for batch in warming_plan.chunks(BATCH_SIZE) {
            let warming_tasks = batch.iter().map(|entry| {
                let cache = self.clone();
                let entry = entry.clone();
                tokio::spawn(async move {
                    match entry.source {
                        WarmingSource::Database => {
                            // Load from database and cache
                            if let Ok(value) = cache.load_from_database(&entry.key).await {
                                cache.set_l2_raw(&entry.key, &value, entry.ttl).await.ok();
                            }
                        }
                        WarmingSource::Computation => {
                            // Perform expensive computation and cache
                            if let Ok(value) = cache.compute_value(&entry.key).await {
                                cache.set_l2_raw(&entry.key, &value, entry.ttl).await.ok();
                            }
                        }
                    }
                })
            });

            futures::future::join_all(warming_tasks).await;
        }

        Ok(())
    }

    // Cache invalidation with pattern matching
    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<u64, CacheError> {
        let mut total_invalidated = 0;

        // Invalidate L1
        total_invalidated += self.invalidate_l1_pattern(pattern).await;

        // Invalidate L2 (DragonflyDB supports pattern matching)
        total_invalidated += self.l2_cache.delete_pattern(pattern).await?;

        // Invalidate L3
        total_invalidated += self.l3_cache.delete_pattern(pattern).await?;

        Ok(total_invalidated)
    }

    // Performance analysis and optimization
    pub async fn analyze_performance(&self) -> CachePerformanceReport {
        let metrics = self.metrics.read().await;
        
        let l1_hit_rate = metrics.l1_hits as f64 / (metrics.l1_hits + metrics.l1_misses) as f64;
        let l2_hit_rate = metrics.l2_hits as f64 / (metrics.l2_hits + metrics.l2_misses) as f64;
        let l3_hit_rate = metrics.l3_hits as f64 / (metrics.l3_hits + metrics.l3_misses) as f64;
        let overall_hit_rate = (metrics.l1_hits + metrics.l2_hits + metrics.l3_hits) as f64 /
                              (metrics.l1_hits + metrics.l1_misses + 
                               metrics.l2_hits + metrics.l2_misses + 
                               metrics.l3_hits + metrics.l3_misses) as f64;

        CachePerformanceReport {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
            overall_hit_rate,
            prefetch_effectiveness: self.calculate_prefetch_effectiveness().await,
            compression_ratio: self.calculate_compression_ratio().await,
            recommendations: self.generate_optimization_recommendations(
                l1_hit_rate, l2_hit_rate, l3_hit_rate
            ),
        }
    }

    // Adaptive cache sizing
    pub async fn optimize_cache_sizes(&mut self) {
        let performance = self.analyze_performance().await;
        
        // Adjust L1 size based on hit rate
        if performance.l1_hit_rate < 0.8 {
            self.policies.l1_max_size = (self.policies.l1_max_size as f64 * 1.2) as usize;
        } else if performance.l1_hit_rate > 0.95 {
            self.policies.l1_max_size = (self.policies.l1_max_size as f64 * 0.9) as usize;
        }

        // Adjust TTL values based on access patterns
        let access_patterns = self.analyze_global_access_patterns().await;
        if access_patterns.average_reaccess_time < self.policies.l1_ttl {
            self.policies.l1_ttl = access_patterns.average_reaccess_time;
        }

        // Apply new policies
        self.l1_cache.write().await.resize(self.policies.l1_max_size);
    }
}

// Supporting structures
#[derive(Debug, Clone)]
pub struct CacheWarmingEntry {
    pub key: String,
    pub source: WarmingSource,
    pub ttl: Duration,
    pub priority: u8,
}

#[derive(Debug, Clone)]
pub enum WarmingSource {
    Database,
    Computation,
}

#[derive(Debug)]
pub struct AccessPattern {
    pub frequency: f64,
    pub recency: Duration,
    pub size_estimate: usize,
    pub read_write_ratio: f64,
}

impl AccessPattern {
    pub fn is_hot(&self) -> bool {
        self.frequency > 10.0 && self.recency < Duration::from_secs(300)
    }

    pub fn should_cache_l2(&self) -> bool {
        self.frequency > 1.0 || self.read_write_ratio > 3.0
    }

    pub fn should_cache_l3(&self) -> bool {
        self.size_estimate > 1024 && self.read_write_ratio > 10.0
    }
}

#[derive(Debug)]
pub struct CachePerformanceReport {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub overall_hit_rate: f64,
    pub prefetch_effectiveness: f64,
    pub compression_ratio: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug)]
pub struct GlobalAccessPatterns {
    pub average_reaccess_time: Duration,
    pub peak_usage_hours: Vec<u8>,
    pub common_key_patterns: Vec<String>,
    pub size_distribution: HashMap<String, usize>,
}

// L1 Cache implementation with LRU eviction
struct L1Cache {
    data: HashMap<String, CacheEntry>,
    access_order: std::collections::VecDeque<String>,
    max_size: usize,
    current_size: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: Instant,
    ttl: Duration,
    access_count: u64,
    last_accessed: Instant,
}

impl L1Cache {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            access_order: std::collections::VecDeque::new(),
            max_size: 10_000,
            current_size: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.data.get_mut(key) {
            // Check TTL
            if entry.created_at.elapsed() > entry.ttl {
                self.remove(key);
                return None;
            }

            // Update access statistics
            entry.access_count += 1;
            entry.last_accessed = Instant::now();

            // Move to front of access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
                self.access_order.push_back(key.to_string());
            }

            Some(entry)
        } else {
            None
        }
    }

    fn set(&mut self, key: String, data: Vec<u8>, ttl: Duration) {
        // Evict if necessary
        while self.current_size >= self.max_size {
            self.evict_lru();
        }

        let entry = CacheEntry {
            data,
            created_at: Instant::now(),
            ttl,
            access_count: 1,
            last_accessed: Instant::now(),
        };

        self.data.insert(key.clone(), entry);
        self.access_order.push_back(key);
        self.current_size += 1;
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.pop_front() {
            self.data.remove(&key);
            self.current_size -= 1;
        }
    }

    fn resize(&mut self, new_size: usize) {
        self.max_size = new_size;
        while self.current_size > self.max_size {
            self.evict_lru();
        }
    }
}
```

## API Performance

### High-Performance API Implementation

```rust
// core/src/api/performance.rs
use axum::{
    extract::{Query, Path, State},
    http::{StatusCode, HeaderMap},
    Json,
    middleware::{self, Next},
    response::Response,
};
use tower::{ServiceBuilder, ServiceExt};
use tower_http::{
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use std::sync::Arc;

// High-performance API server setup
pub fn create_optimized_router() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/v1/resources", axum::routing::get(list_resources_optimized))
        .route("/api/v1/resources/:id", axum::routing::get(get_resource_optimized))
        .route("/api/v1/resources/batch", axum::routing::post(batch_create_resources))
        .layer(
            ServiceBuilder::new()
                // Request timeout
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                // Rate limiting
                .layer(middleware::from_fn(rate_limiting_middleware))
                // Request size limits
                .layer(RequestBodyLimitLayer::new(1024 * 1024)) // 1MB limit
                // Compression
                .layer(CompressionLayer::new())
                // Circuit breaker
                .layer(middleware::from_fn(circuit_breaker_middleware))
                // Performance monitoring
                .layer(middleware::from_fn(performance_middleware))
                // Distributed tracing
                .layer(TraceLayer::new_for_http())
        )
}

// Optimized resource listing with intelligent pagination
pub async fn list_resources_optimized(
    Query(params): Query<ResourceQuery>,
    State(state): State<AppState>,
) -> Result<Json<ResourceResponse>, (StatusCode, String)> {
    let start_time = Instant::now();

    // Input validation and sanitization
    let validated_params = validate_resource_query(params)?;
    
    // Generate cache key based on query parameters
    let cache_key = generate_cache_key("resources", &validated_params);
    
    // Try cache first
    if let Some(cached_response) = state.cache.get::<ResourceResponse>(&cache_key).await {
        return Ok(Json(cached_response));
    }

    // Database query with optimizations
    let mut query_builder = sqlx::QueryBuilder::new(
        "SELECT r.id, r.name, r.type, r.subscription_id, r.location, 
         r.resource_group_name, r.properties, r.tags, r.created_at, r.updated_at,
         COUNT(*) OVER() as total_count"
    );

    query_builder.push(" FROM resources r");

    // Use query hints for PostgreSQL optimization
    if validated_params.subscription_id.is_some() {
        query_builder.push(" /*+ IndexScan(r idx_resources_subscription_covering) */");
    }

    query_builder.push(" WHERE r.deleted_at IS NULL");

    // Apply filters in optimal order (most selective first)
    apply_filters_optimized(&mut query_builder, &validated_params);

    // Intelligent ordering
    if validated_params.use_cursor_pagination {
        query_builder.push(" ORDER BY r.updated_at DESC, r.id DESC");
        if let Some(cursor) = &validated_params.cursor {
            query_builder.push(" AND (r.updated_at, r.id) < ");
            query_builder.push_bind(cursor);
        }
    } else {
        query_builder.push(" ORDER BY r.updated_at DESC");
        let offset = (validated_params.page - 1) * validated_params.limit;
        query_builder.push(" OFFSET ").push_bind(offset);
    }

    query_builder.push(" LIMIT ").push_bind(validated_params.limit);

    // Execute query with performance monitoring
    let query_start = Instant::now();
    let rows = query_builder
        .build()
        .fetch_all(&state.database)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let query_duration = query_start.elapsed();

    // Process results
    let total_count = if rows.is_empty() { 0 } else { 
        rows[0].get::<i64, _>("total_count") as u64 
    };

    let resources: Vec<Resource> = rows
        .into_iter()
        .map(|row| Resource::from_row(&row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Build response with metadata
    let response = ResourceResponse {
        data: resources,
        pagination: PaginationInfo {
            page: validated_params.page,
            limit: validated_params.limit,
            total_items: total_count,
            total_pages: ((total_count as f64) / (validated_params.limit as f64)).ceil() as u32,
            has_next_page: total_count > (validated_params.page * validated_params.limit) as u64,
            has_previous_page: validated_params.page > 1,
        },
        performance: PerformanceMetadata {
            query_time_ms: query_duration.as_millis() as u64,
            total_time_ms: start_time.elapsed().as_millis() as u64,
            cache_hit: false,
        },
    };

    // Cache successful responses
    if !resources.is_empty() {
        let cache_ttl = determine_cache_ttl(&validated_params);
        state.cache.set(&cache_key, &response, cache_ttl).await.ok();
    }

    // Record performance metrics
    state.metrics.record_api_call("list_resources", start_time.elapsed(), resources.len()).await;

    Ok(Json(response))
}

// Batch operations for improved throughput
pub async fn batch_create_resources(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateResourcesRequest>,
) -> Result<Json<BatchCreateResourcesResponse>, (StatusCode, String)> {
    let start_time = Instant::now();

    // Validate batch size
    if request.resources.len() > 1000 {
        return Err((StatusCode::BAD_REQUEST, "Batch size too large (max 1000)".to_string()));
    }

    // Validate all resources before processing
    for (index, resource) in request.resources.iter().enumerate() {
        if let Err(e) = validate_create_resource_request(resource) {
            return Err((StatusCode::BAD_REQUEST, format!("Invalid resource at index {}: {}", index, e)));
        }
    }

    // Process in parallel batches
    const PARALLEL_BATCH_SIZE: usize = 100;
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for chunk in request.resources.chunks(PARALLEL_BATCH_SIZE) {
        let batch_tasks = chunk.iter().enumerate().map(|(index, resource)| {
            let state = state.clone();
            let resource = resource.clone();
            tokio::spawn(async move {
                match create_single_resource(&state, resource).await {
                    Ok(created_resource) => Ok((index, created_resource)),
                    Err(e) => Err((index, e)),
                }
            })
        });

        let batch_results = futures::future::join_all(batch_tasks).await;
        
        for task_result in batch_results {
            match task_result {
                Ok(Ok((index, resource))) => results.push((index, resource)),
                Ok(Err((index, error))) => errors.push(BatchError {
                    index,
                    error: error.to_string(),
                }),
                Err(e) => errors.push(BatchError {
                    index: 0,
                    error: format!("Task execution error: {}", e),
                }),
            }
        }
    }

    let response = BatchCreateResourcesResponse {
        created: results.into_iter().map(|(_, resource)| resource).collect(),
        errors,
        performance: PerformanceMetadata {
            query_time_ms: 0, // Not applicable for batch operations
            total_time_ms: start_time.elapsed().as_millis() as u64,
            cache_hit: false,
        },
    };

    Ok(Json(response))
}

// Performance monitoring middleware
pub async fn performance_middleware(
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start_time = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;

    let duration = start_time.elapsed();
    let status = response.status();

    // Log performance metrics
    tracing::info!(
        method = %method,
        uri = %uri,
        status = %status.as_u16(),
        duration_ms = duration.as_millis(),
        "API request completed"
    );

    // Record metrics for monitoring
    // This would integrate with your metrics collection system
    
    Ok(response)
}

// Rate limiting middleware
pub async fn rate_limiting_middleware(
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let client_id = extract_client_id(&headers);
    let rate_limiter = get_rate_limiter(&client_id).await;

    if !rate_limiter.check_rate_limit().await {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let mut response = next.run(request).await;
    
    // Add rate limit headers
    let remaining = rate_limiter.remaining_requests().await;
    let reset_time = rate_limiter.reset_time().await;
    
    response.headers_mut().insert("X-RateLimit-Remaining", remaining.into());
    response.headers_mut().insert("X-RateLimit-Reset", reset_time.into());

    Ok(response)
}

// Circuit breaker middleware
pub async fn circuit_breaker_middleware(
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let circuit_breaker = get_circuit_breaker("api").await;

    if circuit_breaker.is_open() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    match next.run(request).await {
        response if response.status().is_server_error() => {
            circuit_breaker.record_failure().await;
            Ok(response)
        }
        response => {
            circuit_breaker.record_success().await;
            Ok(response)
        }
    }
}

// Connection pooling optimization
pub async fn optimize_database_pool(pool: &sqlx::PgPool) -> Result<(), sqlx::Error> {
    // Warm up connection pool
    let warm_up_queries = vec![
        "SELECT 1",
        "SELECT pg_backend_pid()",
        "SET statement_timeout = '30s'",
        "SET lock_timeout = '10s'",
    ];

    let tasks = warm_up_queries.into_iter().map(|query| {
        let pool = pool.clone();
        tokio::spawn(async move {
            sqlx::query(query).execute(&pool).await
        })
    });

    futures::future::try_join_all(tasks).await?;

    // Configure optimal settings
    sqlx::query("SET shared_preload_libraries = 'pg_stat_statements'")
        .execute(pool)
        .await?;

    sqlx::query("SET max_connections = 200")
        .execute(pool)
        .await?;

    sqlx::query("SET effective_cache_size = '1GB'")
        .execute(pool)
        .await?;

    Ok(())
}

// Supporting structures
#[derive(Debug, serde::Serialize)]
pub struct PerformanceMetadata {
    pub query_time_ms: u64,
    pub total_time_ms: u64,
    pub cache_hit: bool,
}

#[derive(Debug, serde::Deserialize)]
pub struct ResourceQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub subscription_id: Option<String>,
    pub resource_type: Option<String>,
    pub location: Option<String>,
    pub resource_group_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub use_cursor_pagination: Option<bool>,
    pub cursor: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct ResourceResponse {
    pub data: Vec<Resource>,
    pub pagination: PaginationInfo,
    pub performance: PerformanceMetadata,
}

#[derive(Debug, serde::Deserialize)]
pub struct BatchCreateResourcesRequest {
    pub resources: Vec<CreateResourceRequest>,
}

#[derive(Debug, serde::Serialize)]
pub struct BatchCreateResourcesResponse {
    pub created: Vec<Resource>,
    pub errors: Vec<BatchError>,
    pub performance: PerformanceMetadata,
}

#[derive(Debug, serde::Serialize)]
pub struct BatchError {
    pub index: usize,
    pub error: String,
}
```

This comprehensive performance optimization documentation provides detailed strategies and implementations for maximizing PolicyCortex's performance across all layers of the system. The optimizations ensure sub-millisecond response times and massive scalability while maintaining reliability and consistency.