# Core Rust API Documentation

## Overview

The Core Rust API is the backbone of PolicyCortex, providing high-performance endpoints for all governance operations. Built with Axum 0.7 and Tokio, it delivers sub-millisecond response times for cached operations and implements all four patented technologies.

## Architecture

### Technology Stack

- **Framework**: Axum 0.7 (Tower-based web framework)
- **Runtime**: Tokio (async runtime)
- **Database**: PostgreSQL with SQLx
- **Cache**: Redis/DragonflyDB
- **Serialization**: Serde JSON
- **Authentication**: jsonwebtoken with Azure AD
- **HTTP Client**: Reqwest
- **Metrics**: Prometheus

### Project Structure

```
core/
├── Cargo.toml              # Dependencies and metadata
├── src/
│   ├── main.rs            # Application entry point
│   ├── api/
│   │   ├── mod.rs         # API handlers and routing
│   │   ├── compliance.rs  # Compliance endpoints
│   │   ├── finops.rs      # FinOps endpoints
│   │   └── security.rs    # Security endpoints
│   ├── auth.rs            # Authentication middleware
│   ├── cache.rs           # Caching layer
│   ├── azure_client.rs    # Sync Azure SDK client
│   ├── azure_client_async.rs # Async Azure SDK client
│   ├── collectors/
│   │   └── azure_resource_collector.rs
│   ├── policy/
│   │   └── evaluation_engine.rs
│   ├── compliance/
│   │   ├── mod.rs
│   │   └── engine.rs
│   ├── finops/
│   │   ├── mod.rs
│   │   └── analyzer.rs
│   ├── security_graph/
│   │   ├── mod.rs
│   │   └── graph.rs
│   └── events/
│       ├── mod.rs
│       └── store.rs
```

## Core Components

### 1. Main Application Entry (main.rs)

```rust
#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Create application state
    let state = Arc::new(AppState::new().await);
    
    // Build router with all routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/metrics", get(get_governance_metrics))
        .route("/api/v1/predictions", get(get_predictions))
        .route("/api/v1/conversation", post(conversation))
        .route("/api/v1/correlations", get(get_correlations))
        .nest("/api/v1", api_routes())
        .layer(cors_layer())
        .layer(trace_layer())
        .with_state(state);
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### 2. Application State (api/mod.rs)

```rust
pub struct AppState {
    pub db_pool: PgPool,
    pub redis_pool: RedisPool,
    pub azure_client: Option<AzureClient>,
    pub async_azure_client: Option<AsyncAzureClient>,
    pub metrics: Arc<RwLock<GovernanceMetrics>>,
    pub predictions: Arc<RwLock<Vec<Prediction>>>,
    pub cache: Arc<Cache>,
    pub action_store: Arc<RwLock<HashMap<String, Action>>>,
}
```

### 3. Authentication Middleware (auth.rs)

The authentication system validates Azure AD JWT tokens:

```rust
pub struct Claims {
    pub sub: String,
    pub tenant_id: String,
    pub roles: Vec<String>,
    pub exp: i64,
}

pub async fn validate_token(token: &str) -> Result<Claims, AuthError> {
    // Fetch JWKS from Azure AD
    let jwks = fetch_jwks(&tenant_id).await?;
    
    // Decode and validate JWT
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_jwk(&jwk)?,
        &Validation::new(Algorithm::RS256),
    )?;
    
    // Verify expiration
    if token_data.claims.exp < current_timestamp() {
        return Err(AuthError::TokenExpired);
    }
    
    Ok(token_data.claims)
}
```

### 4. Caching System (cache.rs)

Multi-tier caching with different TTLs:

```rust
pub struct Cache {
    redis: RedisPool,
    hot_ttl: Duration,    // 30 seconds
    warm_ttl: Duration,   // 5 minutes
    cold_ttl: Duration,   // 1 hour
}

impl Cache {
    pub async fn get_or_compute<T, F>(
        &self,
        key: &str,
        tier: CacheTier,
        compute: F,
    ) -> Result<T, CacheError>
    where
        T: Serialize + DeserializeOwned,
        F: Future<Output = Result<T, Box<dyn Error>>>,
    {
        // Try to get from cache
        if let Some(cached) = self.get(key).await? {
            return Ok(cached);
        }
        
        // Compute value
        let value = compute.await?;
        
        // Store in cache with appropriate TTL
        let ttl = match tier {
            CacheTier::Hot => self.hot_ttl,
            CacheTier::Warm => self.warm_ttl,
            CacheTier::Cold => self.cold_ttl,
        };
        
        self.set(key, &value, ttl).await?;
        Ok(value)
    }
}
```

## API Endpoints

### Patent 1: Unified Governance Metrics

**Endpoint**: `GET /api/v1/metrics`

```rust
pub async fn get_governance_metrics(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Try cache first
    if let Ok(cached) = state.cache.get("metrics").await {
        return Json(cached);
    }
    
    // Fetch from Azure if available
    if let Some(ref client) = state.async_azure_client {
        let metrics = client.get_unified_metrics().await?;
        
        // Cache for 30 seconds
        state.cache.set("metrics", &metrics, Duration::from_secs(30)).await?;
        
        return Json(metrics);
    }
    
    // Return service unavailable if no Azure connection
    StatusCode::SERVICE_UNAVAILABLE
}
```

**Response Schema**:
```json
{
  "compliance": {
    "score": 85.5,
    "violations": 23,
    "trend": "improving"
  },
  "security": {
    "score": 92.0,
    "alerts": 5,
    "criticalFindings": 1
  },
  "costs": {
    "current": 125000,
    "projected": 135000,
    "savings": 15000
  },
  "resources": {
    "total": 1250,
    "compliant": 1100,
    "nonCompliant": 150
  }
}
```

### Patent 2: Predictive Compliance

**Endpoint**: `GET /api/v1/predictions`

```rust
pub async fn get_predictions(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let predictions = calculate_predictions(&state).await?;
    
    Json(json!({
        "predictions": predictions,
        "accuracy": 0.94,
        "modelVersion": "2.0.0",
        "timestamp": Utc::now()
    }))
}
```

**Prediction Algorithm**:
```rust
async fn calculate_predictions(state: &AppState) -> Vec<Prediction> {
    let mut predictions = Vec::new();
    
    // Analyze historical compliance data
    let history = fetch_compliance_history(&state.db_pool).await?;
    
    // Apply ML model
    for resource in history.resources {
        let drift_probability = ml_model.predict(&resource.features);
        
        if drift_probability > 0.7 {
            predictions.push(Prediction {
                resource_id: resource.id,
                prediction_type: "compliance_drift",
                probability: drift_probability,
                recommended_action: suggest_remediation(&resource),
                timeframe: "7_days",
            });
        }
    }
    
    predictions
}
```

### Patent 3: Conversational Intelligence

**Endpoint**: `POST /api/v1/conversation`

```rust
pub async fn conversation(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConversationRequest>,
) -> impl IntoResponse {
    // Process natural language query
    let intent = parse_intent(&request.query).await?;
    
    // Execute appropriate action
    let response = match intent {
        Intent::QueryPolicy => query_policies(&state, &request).await?,
        Intent::CreatePolicy => create_policy_from_nl(&state, &request).await?,
        Intent::AnalyzeCompliance => analyze_compliance_nl(&state, &request).await?,
        Intent::OptimizeCosts => suggest_cost_optimizations(&state).await?,
    };
    
    Json(response)
}
```

### Patent 4: Cross-Domain Correlation

**Endpoint**: `GET /api/v1/correlations`

```rust
pub async fn get_correlations(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Collect data from all domains
    let compliance_data = fetch_compliance_data(&state).await?;
    let security_data = fetch_security_data(&state).await?;
    let cost_data = fetch_cost_data(&state).await?;
    
    // Run correlation engine
    let correlations = correlation_engine::analyze(
        compliance_data,
        security_data,
        cost_data,
    ).await?;
    
    Json(json!({
        "correlations": correlations,
        "insights": generate_insights(&correlations),
        "recommendations": generate_recommendations(&correlations),
    }))
}
```

## Azure Integration

### Async Azure Client (azure_client_async.rs)

High-performance async client for Azure APIs:

```rust
pub struct AsyncAzureClient {
    credential: Arc<DefaultAzureCredential>,
    subscription_id: String,
    resource_client: ResourceClient,
    policy_client: PolicyClient,
    cost_client: CostManagementClient,
    graph_client: GraphClient,
}

impl AsyncAzureClient {
    pub async fn get_all_resources(&self) -> Result<Vec<Resource>> {
        let mut resources = Vec::new();
        let mut continuation_token = None;
        
        loop {
            let response = self.resource_client
                .resources()
                .list()
                .continuation(continuation_token)
                .send()
                .await?;
            
            resources.extend(response.value);
            
            if response.next_link.is_none() {
                break;
            }
            
            continuation_token = response.continuation_token;
        }
        
        Ok(resources)
    }
    
    pub async fn get_unified_metrics(&self) -> Result<GovernanceMetrics> {
        // Parallel fetch from multiple Azure services
        let (resources, policies, costs, security) = tokio::join!(
            self.get_all_resources(),
            self.get_policy_compliance(),
            self.get_cost_analysis(),
            self.get_security_alerts()
        );
        
        // Aggregate metrics
        Ok(GovernanceMetrics {
            compliance: calculate_compliance_metrics(policies?),
            security: calculate_security_metrics(security?),
            costs: calculate_cost_metrics(costs?),
            resources: calculate_resource_metrics(resources?),
        })
    }
}
```

## Action Orchestration

### Action Management System

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    pub action_type: ActionType,
    pub resource_id: String,
    pub status: ActionStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub blast_radius: BlastRadius,
    pub approvals: Vec<Approval>,
    pub execution_log: Vec<LogEntry>,
}

pub async fn create_action(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateActionRequest>,
) -> impl IntoResponse {
    // Validate request
    validate_action_request(&request)?;
    
    // Calculate blast radius
    let blast_radius = calculate_blast_radius(&state, &request).await?;
    
    // Create action
    let action = Action {
        id: Uuid::new_v4().to_string(),
        action_type: request.action_type,
        resource_id: request.resource_id,
        status: ActionStatus::Pending,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        blast_radius,
        approvals: Vec::new(),
        execution_log: Vec::new(),
    };
    
    // Store action
    state.action_store.write().await.insert(action.id.clone(), action.clone());
    
    // Send to execution queue if auto-approved
    if action.blast_radius.risk_level == RiskLevel::Low {
        execute_action(&state, &action).await?;
    }
    
    Json(action)
}
```

### Server-Sent Events for Real-time Updates

```rust
pub async fn action_events(
    State(state): State<Arc<AppState>>,
    Path(action_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            
            // Get action status
            let action = state.action_store.read().await.get(&action_id).cloned();
            
            if let Some(action) = action {
                yield Ok(Event::default()
                    .event("action-update")
                    .json_data(action)
                    .unwrap());
                
                // Stop streaming if action is complete
                if matches!(action.status, ActionStatus::Completed | ActionStatus::Failed) {
                    break;
                }
            }
        }
    };
    
    Sse::new(stream)
}
```

## Performance Optimizations

### 1. Connection Pooling

```rust
// Database connection pool
let db_pool = PgPoolOptions::new()
    .max_connections(100)
    .min_connections(10)
    .connect_timeout(Duration::from_secs(5))
    .idle_timeout(Duration::from_secs(300))
    .connect(&database_url)
    .await?;

// Redis connection pool
let redis_pool = RedisPool::new(
    redis_url,
    100,  // max connections
    10,   // min connections
)?;
```

### 2. Request Batching

```rust
pub async fn batch_get_resources(
    State(state): State<Arc<AppState>>,
    Json(resource_ids): Json<Vec<String>>,
) -> impl IntoResponse {
    // Batch fetch from cache
    let mut cached_resources = Vec::new();
    let mut missing_ids = Vec::new();
    
    for id in resource_ids {
        if let Ok(Some(resource)) = state.cache.get(&format!("resource:{}", id)).await {
            cached_resources.push(resource);
        } else {
            missing_ids.push(id);
        }
    }
    
    // Batch fetch missing from Azure
    if !missing_ids.is_empty() {
        let azure_resources = state.async_azure_client
            .batch_get_resources(missing_ids)
            .await?;
        
        // Cache fetched resources
        for resource in &azure_resources {
            state.cache.set(
                &format!("resource:{}", resource.id),
                resource,
                Duration::from_secs(300),
            ).await?;
        }
        
        cached_resources.extend(azure_resources);
    }
    
    Json(cached_resources)
}
```

### 3. Streaming Responses

```rust
pub async fn stream_large_dataset(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let stream = async_stream::stream! {
        let mut offset = 0;
        let limit = 1000;
        
        loop {
            let batch = fetch_batch(&state.db_pool, offset, limit).await?;
            
            if batch.is_empty() {
                break;
            }
            
            for item in batch {
                yield Ok::<_, Error>(Bytes::from(
                    format!("{}\n", serde_json::to_string(&item)?)
                ));
            }
            
            offset += limit;
        }
    };
    
    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}
```

## Error Handling

### Custom Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Authentication failed: {0}")]
    Auth(#[from] AuthError),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Azure API error: {0}")]
    Azure(#[from] AzureError),
    
    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal server error")]
    Internal,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::Auth(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Database error".to_string()),
            ApiError::Azure(_) => (StatusCode::BAD_GATEWAY, "Azure service error".to_string()),
            ApiError::Cache(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Cache error".to_string()),
            ApiError::Validation(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string()),
        };
        
        let body = Json(json!({
            "error": error_message,
            "timestamp": Utc::now(),
        }));
        
        (status, body).into_response()
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_governance_metrics() {
        let state = create_test_state().await;
        let response = get_governance_metrics(State(state)).await;
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let metrics: GovernanceMetrics = response.json().await;
        assert!(metrics.compliance.score >= 0.0);
        assert!(metrics.compliance.score <= 100.0);
    }
    
    #[tokio::test]
    async fn test_cache_hit_rate() {
        let cache = Cache::new(redis_url).await;
        
        // Set value
        cache.set("test_key", &"test_value", Duration::from_secs(60)).await.unwrap();
        
        // Get value (should hit)
        let result: String = cache.get("test_key").await.unwrap().unwrap();
        assert_eq!(result, "test_value");
        
        // Check metrics
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 0);
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_flow() {
    let app = create_test_app().await;
    let client = TestClient::new(app);
    
    // Authenticate
    let token = get_test_token().await;
    
    // Create action
    let response = client
        .post("/api/v1/actions")
        .header("Authorization", format!("Bearer {}", token))
        .json(&json!({
            "action_type": "remediate",
            "resource_id": "test-resource-123",
        }))
        .send()
        .await;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let action: Action = response.json().await;
    assert_eq!(action.status, ActionStatus::Pending);
    
    // Monitor action progress
    let mut event_stream = client
        .get(&format!("/api/v1/actions/{}/events", action.id))
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
        .sse();
    
    while let Some(event) = event_stream.next().await {
        let update: Action = serde_json::from_str(&event.data).unwrap();
        if matches!(update.status, ActionStatus::Completed) {
            break;
        }
    }
}
```

## Deployment Configuration

### Docker Build

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/policycortex-core /usr/local/bin/
EXPOSE 8080
CMD ["policycortex-core"]
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/policycortex

# Redis
REDIS_URL=redis://localhost:6379

# Azure
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-secret
AZURE_SUBSCRIPTION_ID=your-subscription

# Server
PORT=8080
RUST_LOG=info
ENVIRONMENT=production

# Features
ENABLE_CACHE=true
CACHE_TTL_SECONDS=300
MAX_CONNECTIONS=100
REQUEST_TIMEOUT_MS=30000
```

## Monitoring & Metrics

### Prometheus Metrics

```rust
lazy_static! {
    static ref HTTP_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "http_requests_total",
        "Total number of HTTP requests",
        &["method", "endpoint", "status"]
    ).unwrap();
    
    static ref HTTP_REQUEST_DURATION: HistogramVec = register_histogram_vec!(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["method", "endpoint"]
    ).unwrap();
    
    static ref CACHE_HITS: IntCounter = register_int_counter!(
        "cache_hits_total",
        "Total number of cache hits"
    ).unwrap();
    
    static ref AZURE_API_CALLS: IntCounterVec = register_int_counter_vec!(
        "azure_api_calls_total",
        "Total Azure API calls",
        &["service", "operation", "status"]
    ).unwrap();
}
```

### Health Checks

```rust
pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut status = HealthStatus {
        status: "healthy",
        checks: HashMap::new(),
    };
    
    // Check database
    match sqlx::query("SELECT 1").fetch_one(&state.db_pool).await {
        Ok(_) => status.checks.insert("database", "healthy"),
        Err(_) => {
            status.status = "unhealthy";
            status.checks.insert("database", "unhealthy");
        }
    }
    
    // Check Redis
    match state.redis_pool.get().await {
        Ok(mut conn) => {
            if redis::cmd("PING").query::<String>(&mut conn).await.is_ok() {
                status.checks.insert("cache", "healthy");
            } else {
                status.status = "degraded";
                status.checks.insert("cache", "unhealthy");
            }
        }
        Err(_) => {
            status.status = "degraded";
            status.checks.insert("cache", "unhealthy");
        }
    }
    
    // Check Azure
    if let Some(ref client) = state.async_azure_client {
        match client.health_check().await {
            Ok(_) => status.checks.insert("azure", "healthy"),
            Err(_) => {
                status.status = "degraded";
                status.checks.insert("azure", "unhealthy");
            }
        }
    }
    
    Json(status)
}
```