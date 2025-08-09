# 15. Monitoring & Observability

## Table of Contents
1. [Observability Strategy](#observability-strategy)
2. [Metrics Collection](#metrics-collection)
3. [Logging Framework](#logging-framework)
4. [Distributed Tracing](#distributed-tracing)
5. [Health Monitoring](#health-monitoring)
6. [Application Performance Monitoring (APM)](#application-performance-monitoring-apm)
7. [Infrastructure Monitoring](#infrastructure-monitoring)
8. [Database Monitoring](#database-monitoring)
9. [Security Monitoring](#security-monitoring)
10. [Alerting & Notification](#alerting--notification)
11. [Dashboards & Visualization](#dashboards--visualization)
12. [SLI/SLO/SLA Management](#slislosla-management)
13. [Incident Response](#incident-response)
14. [Capacity Planning](#capacity-planning)
15. [Cost Monitoring](#cost-monitoring)

## Observability Strategy

### Three Pillars of Observability
```yaml
# Observability pillars implementation
Metrics:
  - Business metrics (policy evaluations, compliance rates)
  - Technical metrics (response times, error rates, throughput)
  - Infrastructure metrics (CPU, memory, disk, network)
  - Custom metrics (AI model performance, correlation strength)

Logs:
  - Application logs (structured JSON)
  - Access logs (API requests, user actions)
  - Audit logs (policy changes, admin actions)
  - Error logs (exceptions, failures)

Traces:
  - Request flow across microservices
  - Database query performance
  - External API call timing
  - AI processing workflows
```

### Observability Stack
```yaml
# Technology stack
Collection:
  - OpenTelemetry (unified collection)
  - Prometheus (metrics)
  - Fluent Bit (log shipping)
  - Jaeger (distributed tracing)

Storage:
  - Prometheus (metrics storage)
  - Loki (log aggregation)
  - Jaeger (trace storage)
  - InfluxDB (time-series data)

Visualization:
  - Grafana (primary dashboards)
  - Kibana (log analysis)
  - Jaeger UI (trace visualization)
  - Custom dashboards (business metrics)

Alerting:
  - AlertManager (Prometheus alerts)
  - PagerDuty (incident management)
  - Slack (team notifications)
  - Email (escalation)
```

## Metrics Collection

### Rust Backend Metrics
```rust
// core/src/metrics/mod.rs
use prometheus::{
    Counter, Histogram, Gauge, IntGauge, IntCounter,
    register_counter, register_histogram, register_gauge,
    register_int_gauge, register_int_counter, Encoder, TextEncoder
};
use lazy_static::lazy_static;

lazy_static! {
    // Request metrics
    pub static ref HTTP_REQUESTS_TOTAL: Counter = register_counter!(
        "http_requests_total", 
        "Total number of HTTP requests"
    ).unwrap();

    pub static ref HTTP_REQUEST_DURATION: Histogram = register_histogram!(
        "http_request_duration_seconds",
        "Duration of HTTP requests in seconds",
        vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();

    pub static ref HTTP_REQUEST_ERRORS: IntCounter = register_int_counter!(
        "http_request_errors_total",
        "Total number of HTTP request errors"
    ).unwrap();

    // Business metrics
    pub static ref POLICY_EVALUATIONS_TOTAL: IntCounter = register_int_counter!(
        "policy_evaluations_total",
        "Total number of policy evaluations performed"
    ).unwrap();

    pub static ref POLICY_VIOLATIONS_TOTAL: IntCounter = register_int_counter!(
        "policy_violations_total",
        "Total number of policy violations detected"
    ).unwrap();

    pub static ref ACTIVE_POLICIES: IntGauge = register_int_gauge!(
        "active_policies",
        "Number of active policies"
    ).unwrap();

    pub static ref COMPLIANCE_RATE: Gauge = register_gauge!(
        "compliance_rate",
        "Overall compliance rate percentage"
    ).unwrap();

    // AI metrics
    pub static ref AI_INFERENCE_DURATION: Histogram = register_histogram!(
        "ai_inference_duration_seconds",
        "Duration of AI inference requests in seconds"
    ).unwrap();

    pub static ref AI_MODEL_ACCURACY: Gauge = register_gauge!(
        "ai_model_accuracy",
        "Current AI model accuracy score"
    ).unwrap();

    pub static ref CORRELATION_STRENGTH: Histogram = register_histogram!(
        "correlation_strength",
        "Distribution of correlation strength values"
    ).unwrap();

    // Database metrics
    pub static ref DB_CONNECTION_POOL_SIZE: IntGauge = register_int_gauge!(
        "db_connection_pool_size",
        "Current database connection pool size"
    ).unwrap();

    pub static ref DB_QUERY_DURATION: Histogram = register_histogram!(
        "db_query_duration_seconds",
        "Duration of database queries in seconds"
    ).unwrap();
}

// Metrics middleware
use axum::{extract::Request, response::Response, middleware::Next};
use std::time::Instant;

pub async fn metrics_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    HTTP_REQUESTS_TOTAL.inc();

    let response = next.run(request).await;
    
    let duration = start.elapsed();
    HTTP_REQUEST_DURATION.observe(duration.as_secs_f64());

    if response.status().is_server_error() || response.status().is_client_error() {
        HTTP_REQUEST_ERRORS.inc();
    }

    // Log slow requests
    if duration.as_millis() > 1000 {
        tracing::warn!(
            method = %method,
            path = %path,
            duration_ms = duration.as_millis(),
            status = response.status().as_u16(),
            "Slow request detected"
        );
    }

    response
}

// Business metrics tracking
pub fn track_policy_evaluation(policy_id: &str, result: &EvaluationResult) {
    POLICY_EVALUATIONS_TOTAL.inc();
    
    if !result.violations.is_empty() {
        POLICY_VIOLATIONS_TOTAL.inc_by(result.violations.len() as u64);
    }

    // Update compliance rate
    let total_evaluations = POLICY_EVALUATIONS_TOTAL.get() as f64;
    let total_violations = POLICY_VIOLATIONS_TOTAL.get() as f64;
    let compliance = ((total_evaluations - total_violations) / total_evaluations) * 100.0;
    COMPLIANCE_RATE.set(compliance);
}

pub fn track_ai_inference(duration: f64, accuracy: f64) {
    AI_INFERENCE_DURATION.observe(duration);
    AI_MODEL_ACCURACY.set(accuracy);
}

pub fn track_correlation(strength: f64) {
    CORRELATION_STRENGTH.observe(strength);
}

// Metrics endpoint
use axum::{http::StatusCode, response::IntoResponse};

pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    
    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => (StatusCode::OK, metrics),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, String::from("Failed to encode metrics")),
    }
}
```

### Frontend Metrics (Next.js)
```typescript
// frontend/lib/metrics/index.ts
interface MetricData {
  name: string;
  value: number;
  labels?: Record<string, string>;
  timestamp?: number;
}

class MetricsCollector {
  private metrics: MetricData[] = [];
  private flushInterval: number = 30000; // 30 seconds
  private endpoint: string = '/api/metrics';

  constructor() {
    if (typeof window !== 'undefined') {
      this.startAutoFlush();
      this.setupPerformanceObserver();
      this.setupErrorTracking();
    }
  }

  // Counter metrics
  increment(name: string, labels?: Record<string, string>): void {
    this.addMetric({
      name: `${name}_total`,
      value: 1,
      labels,
      timestamp: Date.now()
    });
  }

  // Gauge metrics
  gauge(name: string, value: number, labels?: Record<string, string>): void {
    this.addMetric({
      name,
      value,
      labels,
      timestamp: Date.now()
    });
  }

  // Histogram metrics (for timing)
  timing(name: string, value: number, labels?: Record<string, string>): void {
    this.addMetric({
      name: `${name}_duration_ms`,
      value,
      labels,
      timestamp: Date.now()
    });
  }

  // Track user actions
  trackUserAction(action: string, component?: string): void {
    this.increment('user_actions', {
      action,
      component: component || 'unknown',
      page: window.location.pathname
    });
  }

  // Track page views
  trackPageView(path: string): void {
    this.increment('page_views', { path });
  }

  // Track API calls
  trackApiCall(endpoint: string, method: string, status: number, duration: number): void {
    this.increment('api_calls', {
      endpoint: endpoint.replace(/\/\d+/g, '/:id'), // normalize IDs
      method,
      status: status.toString()
    });
    
    this.timing('api_call_duration', duration, {
      endpoint: endpoint.replace(/\/\d+/g, '/:id'),
      method
    });
  }

  // Track errors
  trackError(error: Error, context?: string): void {
    this.increment('client_errors', {
      error_type: error.name,
      context: context || 'unknown',
      page: window.location.pathname
    });
  }

  private addMetric(metric: MetricData): void {
    this.metrics.push(metric);
  }

  private startAutoFlush(): void {
    setInterval(() => {
      this.flush();
    }, this.flushInterval);

    // Flush on page unload
    window.addEventListener('beforeunload', () => {
      this.flush(true);
    });
  }

  private flush(sync: boolean = false): void {
    if (this.metrics.length === 0) return;

    const metricsToSend = [...this.metrics];
    this.metrics = [];

    const payload = JSON.stringify({ metrics: metricsToSend });

    if (sync) {
      // Use sendBeacon for synchronous sending on page unload
      navigator.sendBeacon(this.endpoint, payload);
    } else {
      fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload
      }).catch(error => {
        console.error('Failed to send metrics:', error);
        // Put metrics back for retry
        this.metrics.unshift(...metricsToSend);
      });
    }
  }

  private setupPerformanceObserver(): void {
    if ('PerformanceObserver' in window) {
      // Web Vitals
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.timing('page_load_time', navEntry.loadEventEnd - navEntry.fetchStart);
            this.timing('dom_content_loaded', navEntry.domContentLoadedEventEnd - navEntry.fetchStart);
            this.timing('first_paint', navEntry.responseEnd - navEntry.fetchStart);
          }

          if (entry.entryType === 'largest-contentful-paint') {
            this.timing('largest_contentful_paint', entry.startTime);
          }

          if (entry.entryType === 'first-input') {
            this.timing('first_input_delay', (entry as any).processingStart - entry.startTime);
          }
        }
      });

      observer.observe({ entryTypes: ['navigation', 'largest-contentful-paint', 'first-input'] });
    }
  }

  private setupErrorTracking(): void {
    window.addEventListener('error', (event) => {
      this.trackError(event.error, 'global_error_handler');
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.trackError(new Error(event.reason), 'unhandled_promise_rejection');
    });
  }
}

export const metrics = new MetricsCollector();

// React hook for component-level metrics
import { useEffect } from 'react';

export function useMetrics(componentName: string) {
  useEffect(() => {
    metrics.increment('component_renders', { component: componentName });
    
    return () => {
      metrics.timing('component_lifetime', Date.now(), { component: componentName });
    };
  }, [componentName]);

  return {
    trackAction: (action: string) => metrics.trackUserAction(action, componentName),
    trackError: (error: Error) => metrics.trackError(error, componentName),
    timing: (name: string, value: number) => metrics.timing(name, value, { component: componentName })
  };
}
```

### Custom Business Metrics
```typescript
// frontend/lib/metrics/business.ts
export class BusinessMetrics {
  // Policy management metrics
  static trackPolicyCreated(category: string): void {
    metrics.increment('policies_created', { category });
  }

  static trackPolicyEvaluation(policyId: string, resourceCount: number, violationCount: number): void {
    metrics.increment('policy_evaluations', { policy_id: policyId });
    metrics.gauge('resources_evaluated', resourceCount);
    metrics.gauge('violations_found', violationCount);
    
    const complianceRate = ((resourceCount - violationCount) / resourceCount) * 100;
    metrics.gauge('compliance_rate', complianceRate, { policy_id: policyId });
  }

  // AI conversation metrics
  static trackAiQuery(queryType: string, responseTime: number, satisfaction?: number): void {
    metrics.increment('ai_queries', { type: queryType });
    metrics.timing('ai_response_time', responseTime, { type: queryType });
    
    if (satisfaction !== undefined) {
      metrics.gauge('ai_satisfaction_score', satisfaction, { type: queryType });
    }
  }

  // User engagement metrics
  static trackFeatureUsage(feature: string, duration: number): void {
    metrics.increment('feature_usage', { feature });
    metrics.timing('feature_session_duration', duration, { feature });
  }

  // Cost optimization metrics
  static trackCostSaving(amount: number, category: string): void {
    metrics.gauge('cost_savings', amount, { category });
    metrics.increment('cost_optimization_actions', { category });
  }
}
```

## Logging Framework

### Structured Logging (Rust)
```rust
// core/src/logging/mod.rs
use tracing::{info, warn, error, debug, trace};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    Registry,
    EnvFilter
};
use tracing_appender::non_blocking::WorkerGuard;
use serde_json::json;

pub fn init_logging() -> WorkerGuard {
    let file_appender = tracing_appender::rolling::daily("logs", "policycortex.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    Registry::default()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true)
                .json()
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .json()
        )
        .init();

    guard
}

// Structured logging macros
#[macro_export]
macro_rules! log_request {
    ($method:expr, $path:expr, $status:expr, $duration:expr) => {
        info!(
            method = $method,
            path = $path,
            status = $status,
            duration_ms = $duration.as_millis(),
            "HTTP request completed"
        );
    };
}

#[macro_export]
macro_rules! log_policy_evaluation {
    ($policy_id:expr, $resource_count:expr, $violation_count:expr) => {
        info!(
            policy_id = $policy_id,
            resource_count = $resource_count,
            violation_count = $violation_count,
            compliance_rate = (($resource_count - $violation_count) as f64 / $resource_count as f64) * 100.0,
            "Policy evaluation completed"
        );
    };
}

#[macro_export]
macro_rules! log_correlation {
    ($event_type:expr, $correlation_strength:expr, $confidence:expr) => {
        info!(
            event_type = $event_type,
            correlation_strength = $correlation_strength,
            confidence = $confidence,
            "Cross-domain correlation detected"
        );
    };
}

// Audit logging
pub struct AuditLogger;

impl AuditLogger {
    pub fn log_user_action(user_id: &str, action: &str, resource: &str, details: Option<&str>) {
        info!(
            event_type = "audit",
            user_id = user_id,
            action = action,
            resource = resource,
            details = details.unwrap_or(""),
            timestamp = chrono::Utc::now().to_rfc3339(),
            "User action audit log"
        );
    }

    pub fn log_policy_change(user_id: &str, policy_id: &str, change_type: &str, old_value: Option<&str>, new_value: Option<&str>) {
        info!(
            event_type = "audit",
            category = "policy_change",
            user_id = user_id,
            policy_id = policy_id,
            change_type = change_type,
            old_value = old_value.unwrap_or(""),
            new_value = new_value.unwrap_or(""),
            timestamp = chrono::Utc::now().to_rfc3339(),
            "Policy change audit log"
        );
    }

    pub fn log_security_event(event_type: &str, severity: &str, details: serde_json::Value) {
        warn!(
            event_type = "security",
            security_event_type = event_type,
            severity = severity,
            details = ?details,
            timestamp = chrono::Utc::now().to_rfc3339(),
            "Security event detected"
        );
    }
}
```

### Frontend Logging
```typescript
// frontend/lib/logging/index.ts
enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

interface LogEntry {
  level: LogLevel;
  message: string;
  context?: Record<string, any>;
  timestamp: number;
  userId?: string;
  sessionId: string;
  userAgent: string;
  url: string;
}

class Logger {
  private logLevel: LogLevel = LogLevel.INFO;
  private sessionId: string = this.generateSessionId();
  private buffer: LogEntry[] = [];
  private flushInterval: number = 10000; // 10 seconds

  constructor() {
    if (typeof window !== 'undefined') {
      this.logLevel = process.env.NODE_ENV === 'development' ? LogLevel.DEBUG : LogLevel.INFO;
      this.startAutoFlush();
      this.setupUnhandledErrorLogging();
    }
  }

  debug(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.DEBUG, message, context);
  }

  info(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.INFO, message, context);
  }

  warn(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.WARN, message, context);
  }

  error(message: string, context?: Record<string, any>): void {
    this.log(LogLevel.ERROR, message, context);
  }

  // Specialized logging methods
  logUserAction(action: string, component?: string, details?: Record<string, any>): void {
    this.info(`User action: ${action}`, {
      category: 'user_action',
      component,
      ...details
    });
  }

  logApiCall(endpoint: string, method: string, status: number, duration: number, error?: Error): void {
    const level = status >= 400 ? LogLevel.ERROR : LogLevel.INFO;
    this.log(level, `API call: ${method} ${endpoint}`, {
      category: 'api_call',
      endpoint,
      method,
      status,
      duration,
      error: error?.message
    });
  }

  logPolicyEvaluation(policyId: string, resourceCount: number, violationCount: number): void {
    this.info('Policy evaluation completed', {
      category: 'policy_evaluation',
      policy_id: policyId,
      resource_count: resourceCount,
      violation_count: violationCount,
      compliance_rate: ((resourceCount - violationCount) / resourceCount) * 100
    });
  }

  logPerformance(metric: string, value: number, context?: Record<string, any>): void {
    this.info(`Performance metric: ${metric}`, {
      category: 'performance',
      metric,
      value,
      ...context
    });
  }

  private log(level: LogLevel, message: string, context?: Record<string, any>): void {
    if (level < this.logLevel) return;

    const logEntry: LogEntry = {
      level,
      message,
      context,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    // Console output for development
    if (process.env.NODE_ENV === 'development') {
      const consoleMethods = ['debug', 'log', 'warn', 'error'];
      console[consoleMethods[level] as keyof Console](
        `[${new Date().toISOString()}] ${message}`,
        context || ''
      );
    }

    this.buffer.push(logEntry);

    // Immediate flush for errors
    if (level === LogLevel.ERROR) {
      this.flush();
    }
  }

  private startAutoFlush(): void {
    setInterval(() => {
      this.flush();
    }, this.flushInterval);

    window.addEventListener('beforeunload', () => {
      this.flush(true);
    });
  }

  private flush(sync: boolean = false): void {
    if (this.buffer.length === 0) return;

    const logs = [...this.buffer];
    this.buffer = [];

    const payload = JSON.stringify({ logs });

    if (sync) {
      navigator.sendBeacon('/api/logs', payload);
    } else {
      fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload
      }).catch(error => {
        console.error('Failed to send logs:', error);
        // Put logs back for retry
        this.buffer.unshift(...logs);
      });
    }
  }

  private setupUnhandledErrorLogging(): void {
    window.addEventListener('error', (event) => {
      this.error('Unhandled error', {
        category: 'unhandled_error',
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.error('Unhandled promise rejection', {
        category: 'unhandled_rejection',
        reason: event.reason
      });
    });
  }

  private generateSessionId(): string {
    return Math.random().toString(36).substr(2, 9);
  }
}

export const logger = new Logger();

// React hook for component logging
export function useLogger(componentName: string) {
  useEffect(() => {
    logger.debug(`Component mounted: ${componentName}`);
    
    return () => {
      logger.debug(`Component unmounted: ${componentName}`);
    };
  }, [componentName]);

  return {
    debug: (message: string, context?: Record<string, any>) => 
      logger.debug(`[${componentName}] ${message}`, context),
    info: (message: string, context?: Record<string, any>) => 
      logger.info(`[${componentName}] ${message}`, context),
    warn: (message: string, context?: Record<string, any>) => 
      logger.warn(`[${componentName}] ${message}`, context),
    error: (message: string, context?: Record<string, any>) => 
      logger.error(`[${componentName}] ${message}`, context),
    logUserAction: (action: string, details?: Record<string, any>) => 
      logger.logUserAction(action, componentName, details)
  };
}
```

## Distributed Tracing

### OpenTelemetry Configuration
```rust
// core/src/tracing/mod.rs
use opentelemetry::{
    global, trace::{TraceError, Tracer},
    sdk::{
        trace::{self, Sampler},
        Resource
    },
    KeyValue
};
use opentelemetry_jaeger::JaegerTraceExporter;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_tracing() -> Result<(), TraceError> {
    // Configure Jaeger exporter
    let jaeger_exporter = JaegerTraceExporter::builder()
        .with_agent_endpoint("http://localhost:14268/api/traces")
        .with_service_name("policycortex-core")
        .build()?;

    // Create tracer
    let tracer = opentelemetry::sdk::trace::TracerProvider::builder()
        .with_batch_exporter(jaeger_exporter, opentelemetry::runtime::Tokio)
        .with_sampler(Sampler::TraceIdRatioBased(0.1)) // Sample 10% of traces
        .with_resource(Resource::new(vec![
            KeyValue::new("service.name", "policycortex-core"),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("deployment.environment", std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string())),
        ]))
        .build();

    global::set_tracer_provider(tracer);

    // Set up tracing subscriber with OpenTelemetry layer
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer())
        .with(OpenTelemetryLayer::new(global::tracer("policycortex-core")))
        .init();

    Ok(())
}

// Custom tracing middleware
use axum::{extract::Request, response::Response, middleware::Next};
use tracing::{Span, instrument};
use opentelemetry::trace::{SpanKind, Status};

pub async fn tracing_middleware(request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let uri = request.uri().to_string();
    let headers = request.headers().clone();

    let span = tracing::info_span!(
        "http_request",
        method = %method,
        uri = %uri,
        otel.kind = "server"
    );

    // Extract trace context from headers
    if let Some(trace_header) = headers.get("traceparent") {
        if let Ok(trace_str) = trace_header.to_str() {
            span.record("trace.parent", &trace_str);
        }
    }

    let _guard = span.enter();
    
    let start = std::time::Instant::now();
    let response = next.run(request).await;
    let duration = start.elapsed();

    // Record span attributes
    let current_span = Span::current();
    current_span.record("http.status_code", &response.status().as_u16());
    current_span.record("http.response_time_ms", &duration.as_millis());

    if response.status().is_server_error() {
        current_span.record("error", &true);
        opentelemetry::trace::get_active_span(|span| {
            span.set_status(Status::error("Server error"));
        });
    }

    response
}

// Policy evaluation tracing
#[instrument(
    skip(engine, policy, resources),
    fields(
        policy.id = %policy.id,
        policy.name = %policy.name,
        resource.count = resources.len()
    )
)]
pub async fn trace_policy_evaluation(
    engine: &PolicyEngine,
    policy: &Policy,
    resources: &[Resource]
) -> Result<EvaluationResult, PolicyError> {
    let span = Span::current();
    
    // Start database span
    let db_span = tracing::info_span!("database_query", query.type = "policy_lookup");
    let _db_guard = db_span.enter();
    
    let start = std::time::Instant::now();
    let result = engine.evaluate(policy, resources).await;
    let duration = start.elapsed();
    
    span.record("evaluation.duration_ms", &duration.as_millis());
    
    match &result {
        Ok(eval_result) => {
            span.record("evaluation.violations", &eval_result.violations.len());
            span.record("evaluation.compliance_rate", 
                &(((resources.len() - eval_result.violations.len()) as f64 / resources.len() as f64) * 100.0));
        },
        Err(error) => {
            span.record("error", &true);
            span.record("error.message", &error.to_string());
            opentelemetry::trace::get_active_span(|span| {
                span.set_status(Status::error(error.to_string()));
            });
        }
    }

    result
}

// AI correlation tracing
#[instrument(
    skip(correlation_engine),
    fields(
        event.type = event_type,
        correlation.algorithm = "cross_domain"
    )
)]
pub async fn trace_correlation_analysis(
    correlation_engine: &CorrelationEngine,
    event_type: &str,
    events: &[Event]
) -> Result<Vec<Correlation>, CorrelationError> {
    let span = Span::current();
    span.record("events.count", &events.len());

    let start = std::time::Instant::now();
    let correlations = correlation_engine.analyze(events).await;
    let duration = start.elapsed();

    span.record("analysis.duration_ms", &duration.as_millis());

    match &correlations {
        Ok(corrs) => {
            span.record("correlations.found", &corrs.len());
            if let Some(strongest) = corrs.iter().max_by_key(|c| (c.strength * 1000.0) as u32) {
                span.record("correlation.max_strength", &strongest.strength);
            }
        },
        Err(error) => {
            span.record("error", &true);
            opentelemetry::trace::get_active_span(|span| {
                span.set_status(Status::error(error.to_string()));
            });
        }
    }

    correlations
}
```

### Frontend Tracing
```typescript
// frontend/lib/tracing/index.ts
import { trace, context, SpanStatusCode, SpanKind } from '@opentelemetry/api';
import { getWebAutoInstrumentations } from '@opentelemetry/auto-instrumentations-web';
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';

// Initialize OpenTelemetry
export function initTracing(): void {
  const provider = new WebTracerProvider({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: 'policycortex-frontend',
      [SemanticResourceAttributes.SERVICE_VERSION]: process.env.NEXT_PUBLIC_VERSION || '1.0.0',
      [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV,
    }),
  });

  // Configure Jaeger exporter
  const jaegerExporter = new JaegerExporter({
    endpoint: process.env.NEXT_PUBLIC_JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
  });

  provider.addSpanProcessor(new BatchSpanProcessor(jaegerExporter));
  provider.register();

  // Auto-instrument web APIs
  getWebAutoInstrumentations({
    '@opentelemetry/instrumentation-fetch': {
      propagateTraceHeaderCorsUrls: [/.*/],
      clearTimingResources: true,
    },
    '@opentelemetry/instrumentation-xml-http-request': {
      propagateTraceHeaderCorsUrls: [/.*/],
    },
  });
}

const tracer = trace.getTracer('policycortex-frontend');

// React component tracing
export function traceComponent<P extends {}>(
  componentName: string,
  WrappedComponent: React.ComponentType<P>
): React.ComponentType<P> {
  return (props: P) => {
    useEffect(() => {
      const span = tracer.startSpan(`component.${componentName}`, {
        kind: SpanKind.INTERNAL,
        attributes: {
          'component.name': componentName,
          'react.lifecycle': 'mount',
        },
      });

      return () => {
        span.setAttributes({
          'react.lifecycle': 'unmount',
        });
        span.end();
      };
    }, []);

    return <WrappedComponent {...props} />;
  };
}

// API call tracing
export async function traceApiCall<T>(
  endpoint: string,
  method: string,
  body?: any
): Promise<T> {
  const span = tracer.startSpan(`api.${method.toLowerCase()}.${endpoint}`, {
    kind: SpanKind.CLIENT,
    attributes: {
      'http.method': method,
      'http.url': endpoint,
      'http.scheme': 'https',
    },
  });

  try {
    const start = performance.now();
    
    const response = await fetch(endpoint, {
      method,
      headers: {
        'Content-Type': 'application/json',
        // Inject trace context
        ...getTraceHeaders(),
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    const duration = performance.now() - start;

    span.setAttributes({
      'http.status_code': response.status,
      'http.response_time': duration,
      'http.response_size': response.headers.get('content-length') || 0,
    });

    if (response.ok) {
      span.setStatus({ code: SpanStatusCode.OK });
      const data = await response.json();
      return data;
    } else {
      span.setStatus({ 
        code: SpanStatusCode.ERROR, 
        message: `HTTP ${response.status} ${response.statusText}` 
      });
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    span.setStatus({ 
      code: SpanStatusCode.ERROR, 
      message: error instanceof Error ? error.message : 'Unknown error' 
    });
    span.recordException(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    span.end();
  }
}

// Policy evaluation tracing
export function tracePolicyEvaluation(policyId: string, resourceCount: number) {
  const span = tracer.startSpan('policy.evaluate', {
    kind: SpanKind.INTERNAL,
    attributes: {
      'policy.id': policyId,
      'resources.count': resourceCount,
    },
  });

  return {
    recordResult: (violationCount: number) => {
      span.setAttributes({
        'policy.violations': violationCount,
        'policy.compliance_rate': ((resourceCount - violationCount) / resourceCount) * 100,
      });
    },
    recordError: (error: Error) => {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.recordException(error);
    },
    end: () => span.end(),
  };
}

// User action tracing
export function traceUserAction(action: string, component?: string) {
  const span = tracer.startSpan(`user.${action}`, {
    kind: SpanKind.INTERNAL,
    attributes: {
      'user.action': action,
      'ui.component': component || 'unknown',
      'page.url': window.location.pathname,
    },
  });

  return {
    recordDetails: (details: Record<string, any>) => {
      Object.entries(details).forEach(([key, value]) => {
        span.setAttribute(`user.${key}`, String(value));
      });
    },
    end: () => span.end(),
  };
}

// Get trace headers for API calls
function getTraceHeaders(): Record<string, string> {
  const activeSpan = trace.getActiveSpan();
  if (!activeSpan) return {};

  const spanContext = activeSpan.spanContext();
  return {
    'traceparent': `00-${spanContext.traceId}-${spanContext.spanId}-01`,
  };
}

// React hook for tracing
export function useTracing(operationName: string) {
  const [span, setSpan] = useState<any>(null);

  useEffect(() => {
    const newSpan = tracer.startSpan(operationName, {
      kind: SpanKind.INTERNAL,
    });
    setSpan(newSpan);

    return () => {
      newSpan.end();
    };
  }, [operationName]);

  return {
    recordAttribute: (key: string, value: any) => {
      span?.setAttribute(key, String(value));
    },
    recordError: (error: Error) => {
      span?.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span?.recordException(error);
    },
  };
}
```

## Health Monitoring

### Health Check Implementation
```rust
// core/src/health/mod.rs
use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Serialize, Deserialize};
use sqlx::PgPool;
use redis::aio::Connection;
use std::time::SystemTime;

#[derive(Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: String,
    pub version: String,
    pub uptime: u64,
    pub services: ServiceHealth,
}

#[derive(Serialize, Deserialize)]
pub struct ServiceHealth {
    pub database: ComponentHealth,
    pub redis: ComponentHealth,
    pub eventstore: ComponentHealth,
    pub ai_engine: ComponentHealth,
    pub external_apis: ComponentHealth,
}

#[derive(Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub response_time_ms: Option<u64>,
    pub error: Option<String>,
    pub details: Option<serde_json::Value>,
}

pub struct HealthChecker {
    db_pool: PgPool,
    redis_client: redis::Client,
    start_time: SystemTime,
}

impl HealthChecker {
    pub fn new(db_pool: PgPool, redis_client: redis::Client) -> Self {
        Self {
            db_pool,
            redis_client,
            start_time: SystemTime::now(),
        }
    }

    pub async fn check_health(&self) -> HealthStatus {
        let uptime = self.start_time.elapsed().unwrap_or_default().as_secs();
        
        let (database, redis, eventstore, ai_engine, external_apis) = tokio::join!(
            self.check_database(),
            self.check_redis(),
            self.check_eventstore(),
            self.check_ai_engine(),
            self.check_external_apis()
        );

        let overall_status = if [&database, &redis, &eventstore, &ai_engine, &external_apis]
            .iter()
            .any(|health| health.status != "healthy") 
        {
            "unhealthy"
        } else {
            "healthy"
        };

        HealthStatus {
            status: overall_status.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime,
            services: ServiceHealth {
                database,
                redis,
                eventstore,
                ai_engine,
                external_apis,
            },
        }
    }

    async fn check_database(&self) -> ComponentHealth {
        let start = std::time::Instant::now();
        
        match sqlx::query("SELECT 1 as health_check")
            .fetch_one(&self.db_pool)
            .await
        {
            Ok(_) => ComponentHealth {
                status: "healthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: None,
                details: Some(serde_json::json!({
                    "connections": {
                        "active": self.db_pool.size(),
                        "idle": self.db_pool.num_idle(),
                    }
                })),
            },
            Err(e) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(e.to_string()),
                details: None,
            },
        }
    }

    async fn check_redis(&self) -> ComponentHealth {
        let start = std::time::Instant::now();
        
        match self.redis_client.get_async_connection().await {
            Ok(mut conn) => {
                match redis::cmd("PING").query_async::<_, String>(&mut conn).await {
                    Ok(_) => ComponentHealth {
                        status: "healthy".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: None,
                        details: None,
                    },
                    Err(e) => ComponentHealth {
                        status: "unhealthy".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: Some(e.to_string()),
                        details: None,
                    },
                }
            },
            Err(e) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(e.to_string()),
                details: None,
            },
        }
    }

    async fn check_eventstore(&self) -> ComponentHealth {
        let start = std::time::Instant::now();
        let client = reqwest::Client::new();
        
        match client.get("http://localhost:2113/info").send().await {
            Ok(response) => {
                if response.status().is_success() {
                    ComponentHealth {
                        status: "healthy".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: None,
                        details: None,
                    }
                } else {
                    ComponentHealth {
                        status: "unhealthy".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: Some(format!("HTTP {}", response.status())),
                        details: None,
                    }
                }
            },
            Err(e) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(e.to_string()),
                details: None,
            },
        }
    }

    async fn check_ai_engine(&self) -> ComponentHealth {
        let start = std::time::Instant::now();
        let client = reqwest::Client::new();
        
        match client.get("http://localhost:8001/health").send().await {
            Ok(response) => {
                if response.status().is_success() {
                    ComponentHealth {
                        status: "healthy".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: None,
                        details: None,
                    }
                } else {
                    ComponentHealth {
                        status: "degraded".to_string(),
                        response_time_ms: Some(start.elapsed().as_millis() as u64),
                        error: Some(format!("HTTP {}", response.status())),
                        details: None,
                    }
                }
            },
            Err(e) => ComponentHealth {
                status: "unhealthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(e.to_string()),
                details: None,
            },
        }
    }

    async fn check_external_apis(&self) -> ComponentHealth {
        let start = std::time::Instant::now();
        let client = reqwest::Client::new();
        
        // Check Azure Resource Manager API
        let azure_health = match client.get("https://management.azure.com/subscriptions?api-version=2020-01-01")
            .header("Authorization", format!("Bearer {}", "test-token"))
            .send()
            .await
        {
            Ok(response) => response.status() == 401, // 401 is expected without proper token
            Err(_) => false,
        };

        if azure_health {
            ComponentHealth {
                status: "healthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: None,
                details: Some(serde_json::json!({
                    "azure_api": "reachable"
                })),
            }
        } else {
            ComponentHealth {
                status: "degraded".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                error: Some("Azure API unreachable".to_string()),
                details: None,
            }
        }
    }
}

// Health endpoints
pub async fn health_handler(health_checker: axum::extract::State<HealthChecker>) -> impl IntoResponse {
    let health = health_checker.check_health().await;
    let status_code = if health.status == "healthy" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    
    (status_code, Json(health))
}

pub async fn readiness_handler(health_checker: axum::extract::State<HealthChecker>) -> impl IntoResponse {
    let health = health_checker.check_health().await;
    
    // Readiness requires all critical services to be healthy
    let ready = health.services.database.status == "healthy" 
        && health.services.redis.status == "healthy";
    
    if ready {
        (StatusCode::OK, Json(serde_json::json!({
            "status": "ready",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "not ready",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }
}

pub async fn liveness_handler() -> impl IntoResponse {
    // Simple liveness check - if this endpoint responds, the app is alive
    (StatusCode::OK, Json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}
```

This comprehensive monitoring and observability strategy provides complete visibility into the PolicyCortex platform, enabling proactive issue detection, performance optimization, and reliable operations across all environments.