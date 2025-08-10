use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, span, Level, Span};
use opentelemetry::{global, sdk::propagation::TraceContextPropagator, trace::{Tracer, TracerProvider, TraceError}};
use opentelemetry_otlp::WithExportConfig;
use prometheus::{Encoder, TextEncoder, Counter, Gauge, Histogram, HistogramOpts, Registry};

/// Comprehensive observability system with distributed tracing, metrics, and logging
pub struct ObservabilitySystem {
    tracer: Arc<dyn Tracer>,
    metrics_registry: Registry,
    spans: Arc<RwLock<HashMap<String, SpanContext>>>,
    correlation_ids: Arc<RwLock<HashMap<String, CorrelationContext>>>,
    
    // Metrics
    request_counter: Counter,
    error_counter: Counter,
    latency_histogram: Histogram,
    active_requests: Gauge,
    
    // SLO tracking
    slo_metrics: Arc<RwLock<SloMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub service_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub status: SpanStatus,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error(String),
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationContext {
    pub correlation_id: Uuid,
    pub causation_id: Option<Uuid>,
    pub tenant_id: String,
    pub user_id: String,
    pub session_id: Option<String>,
    pub request_path: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct SloMetrics {
    total_requests: u64,
    successful_requests: u64,
    error_budget_consumed: f64,
    current_availability: f64,
    last_updated: DateTime<Utc>,
}

impl ObservabilitySystem {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize OpenTelemetry
        global::set_text_map_propagator(TraceContextPropagator::new());
        
        let tracer = Self::init_tracer()?;
        
        // Initialize Prometheus metrics
        let registry = Registry::new();
        
        let request_counter = Counter::new("requests_total", "Total number of requests")?;
        let error_counter = Counter::new("errors_total", "Total number of errors")?;
        
        let latency_histogram = Histogram::with_opts(
            HistogramOpts::new("request_duration_seconds", "Request duration in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
        )?;
        
        let active_requests = Gauge::new("active_requests", "Number of active requests")?;
        
        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(error_counter.clone()))?;
        registry.register(Box::new(latency_histogram.clone()))?;
        registry.register(Box::new(active_requests.clone()))?;
        
        Ok(Self {
            tracer: Arc::new(tracer),
            metrics_registry: registry,
            spans: Arc::new(RwLock::new(HashMap::new())),
            correlation_ids: Arc::new(RwLock::new(HashMap::new())),
            request_counter,
            error_counter,
            latency_histogram,
            active_requests,
            slo_metrics: Arc::new(RwLock::new(SloMetrics {
                total_requests: 0,
                successful_requests: 0,
                error_budget_consumed: 0.0,
                current_availability: 100.0,
                last_updated: Utc::now(),
            })),
        })
    }

    fn init_tracer() -> Result<impl Tracer, TraceError> {
        let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:4317".to_string());
        
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint)
            )
            .with_trace_config(
                opentelemetry::sdk::trace::config()
                    .with_resource(opentelemetry::sdk::Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", "policycortex"),
                        opentelemetry::KeyValue::new("service.version", "2.0.0"),
                    ]))
            )
            .install_batch(opentelemetry::runtime::Tokio)
            .map(|provider| provider.tracer("policycortex"))
    }

    /// Start a new trace span
    pub async fn start_span(
        &self,
        operation_name: &str,
        correlation_id: Option<Uuid>,
    ) -> String {
        let trace_id = Uuid::new_v4().to_string();
        let span_id = Uuid::new_v4().to_string();
        
        let span_context = SpanContext {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: None,
            operation_name: operation_name.to_string(),
            service_name: "policycortex".to_string(),
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
            events: Vec::new(),
        };
        
        let mut spans = self.spans.write().await;
        spans.insert(span_id.clone(), span_context);
        
        // Track correlation context
        if let Some(corr_id) = correlation_id {
            let correlation_context = CorrelationContext {
                correlation_id: corr_id,
                causation_id: None,
                tenant_id: String::new(), // To be set by caller
                user_id: String::new(),   // To be set by caller
                session_id: None,
                request_path: operation_name.to_string(),
                created_at: Utc::now(),
            };
            
            let mut correlations = self.correlation_ids.write().await;
            correlations.insert(corr_id.to_string(), correlation_context);
        }
        
        // Increment active requests
        self.active_requests.inc();
        
        info!(
            trace_id = %trace_id,
            span_id = %span_id,
            operation = %operation_name,
            "Started span"
        );
        
        span_id
    }

    /// End a trace span
    pub async fn end_span(&self, span_id: &str, status: SpanStatus) {
        let mut spans = self.spans.write().await;
        
        if let Some(span) = spans.get_mut(span_id) {
            span.end_time = Some(Utc::now());
            span.duration_ms = Some(
                (span.end_time.unwrap() - span.start_time).num_milliseconds() as u64
            );
            span.status = status.clone();
            
            // Record metrics
            self.request_counter.inc();
            
            if let SpanStatus::Error(_) = status {
                self.error_counter.inc();
            }
            
            if let Some(duration_ms) = span.duration_ms {
                self.latency_histogram.observe(duration_ms as f64 / 1000.0);
            }
            
            // Decrement active requests
            self.active_requests.dec();
            
            // Update SLO metrics
            self.update_slo_metrics(matches!(status, SpanStatus::Ok)).await;
            
            info!(
                span_id = %span_id,
                duration_ms = span.duration_ms.unwrap_or(0),
                status = ?status,
                "Ended span"
            );
        }
    }

    /// Add an event to a span
    pub async fn add_span_event(&self, span_id: &str, event_name: &str, attributes: HashMap<String, String>) {
        let mut spans = self.spans.write().await;
        
        if let Some(span) = spans.get_mut(span_id) {
            span.events.push(SpanEvent {
                timestamp: Utc::now(),
                name: event_name.to_string(),
                attributes,
            });
        }
    }

    /// Set span attributes
    pub async fn set_span_attributes(&self, span_id: &str, attributes: HashMap<String, String>) {
        let mut spans = self.spans.write().await;
        
        if let Some(span) = spans.get_mut(span_id) {
            span.attributes.extend(attributes);
        }
    }

    /// Get correlation context for a correlation ID
    pub async fn get_correlation_context(&self, correlation_id: &Uuid) -> Option<CorrelationContext> {
        let correlations = self.correlation_ids.read().await;
        correlations.get(&correlation_id.to_string()).cloned()
    }

    /// Record a custom metric
    pub fn record_metric(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        info!(
            metric_name = %name,
            value = %value,
            labels = ?labels,
            "Recorded metric"
        );
        
        // TODO: Send to metrics backend
    }

    /// Export metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let encoder = TextEncoder::new();
        let metric_families = self.metrics_registry.gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        
        Ok(String::from_utf8(buffer)?)
    }

    /// Update SLO metrics
    async fn update_slo_metrics(&self, success: bool) {
        let mut slo = self.slo_metrics.write().await;
        
        slo.total_requests += 1;
        if success {
            slo.successful_requests += 1;
        }
        
        // Calculate current availability
        if slo.total_requests > 0 {
            slo.current_availability = (slo.successful_requests as f64 / slo.total_requests as f64) * 100.0;
        }
        
        // Calculate error budget consumed (assuming 99.9% SLO)
        let target_slo = 99.9;
        let allowed_failures = slo.total_requests as f64 * (100.0 - target_slo) / 100.0;
        let actual_failures = slo.total_requests - slo.successful_requests;
        
        if allowed_failures > 0.0 {
            slo.error_budget_consumed = (actual_failures as f64 / allowed_failures) * 100.0;
        }
        
        slo.last_updated = Utc::now();
    }

    /// Get current SLO status
    pub async fn get_slo_status(&self) -> SloStatus {
        let slo = self.slo_metrics.read().await;
        
        SloStatus {
            availability: slo.current_availability,
            error_budget_consumed: slo.error_budget_consumed,
            total_requests: slo.total_requests,
            successful_requests: slo.successful_requests,
            failed_requests: slo.total_requests - slo.successful_requests,
            last_updated: slo.last_updated,
        }
    }

    /// Create a distributed trace context for cross-service calls
    pub fn create_trace_context(&self, parent_span_id: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        
        // W3C Trace Context headers
        headers.insert("traceparent".to_string(), format!("00-{}-{}-01", 
            Uuid::new_v4().to_string().replace("-", ""),
            parent_span_id.replace("-", "")
        ));
        
        headers.insert("tracestate".to_string(), "policycortex=active".to_string());
        
        headers
    }

    /// Extract trace context from incoming request headers
    pub fn extract_trace_context(&self, headers: &HashMap<String, String>) -> Option<(String, String)> {
        if let Some(traceparent) = headers.get("traceparent") {
            let parts: Vec<&str> = traceparent.split('-').collect();
            if parts.len() >= 3 {
                return Some((parts[1].to_string(), parts[2].to_string()));
            }
        }
        None
    }

    /// Log structured event
    pub fn log_event(&self, level: LogLevel, message: &str, fields: HashMap<String, String>) {
        let log_entry = LogEntry {
            timestamp: Utc::now(),
            level,
            message: message.to_string(),
            fields,
            service: "policycortex".to_string(),
        };
        
        match level {
            LogLevel::Error => error!("{:?}", log_entry),
            LogLevel::Warn => warn!("{:?}", log_entry),
            LogLevel::Info => info!("{:?}", log_entry),
            LogLevel::Debug => tracing::debug!("{:?}", log_entry),
        }
    }

    /// Health check for observability systems
    pub async fn health_check(&self) -> ObservabilityHealth {
        ObservabilityHealth {
            tracing_healthy: true, // TODO: Check actual OTLP connection
            metrics_healthy: true, // TODO: Check metrics endpoint
            logging_healthy: true, // TODO: Check log aggregation
            last_check: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloStatus {
    pub availability: f64,
    pub error_budget_consumed: f64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogEntry {
    timestamp: DateTime<Utc>,
    level: LogLevel,
    message: String,
    fields: HashMap<String, String>,
    service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityHealth {
    pub tracing_healthy: bool,
    pub metrics_healthy: bool,
    pub logging_healthy: bool,
    pub last_check: DateTime<Utc>,
}

/// Macro for easy span creation with automatic closing
#[macro_export]
macro_rules! trace_span {
    ($obs:expr, $name:expr) => {
        $obs.start_span($name, None).await
    };
    ($obs:expr, $name:expr, $corr_id:expr) => {
        $obs.start_span($name, Some($corr_id)).await
    };
}

/// Macro for recording metrics
#[macro_export]
macro_rules! record_metric {
    ($obs:expr, $name:expr, $value:expr) => {
        $obs.record_metric($name, $value, HashMap::new())
    };
    ($obs:expr, $name:expr, $value:expr, $labels:expr) => {
        $obs.record_metric($name, $value, $labels)
    };
}