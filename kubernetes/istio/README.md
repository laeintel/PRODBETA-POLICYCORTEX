# Istio Service Mesh Configuration

## Overview
This directory contains Istio service mesh configurations for PolicyCortex, providing advanced traffic management, security, and observability capabilities.

## Architecture

```
                    ┌─────────────────────┐
                    │   Istio Gateway     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Virtual Services   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼──────────┐      ┌──────────▼──────────┐
     │  Frontend Service   │      │   Core API Service  │
     │   (Envoy Sidecar)   │      │   (Envoy Sidecar)   │
     └─────────────────────┘      └─────────────────────┘
                                            │
                               ┌────────────┴────────────┐
                               │                         │
                    ┌──────────▼──────────┐   ┌─────────▼──────────┐
                    │  GraphQL Gateway    │   │   Python Services  │
                    │   (Envoy Sidecar)   │   │  (Envoy Sidecar)   │
                    └─────────────────────┘   └────────────────────┘
```

## Directory Structure

- **base/** - Core Istio configuration and namespace setup
- **traffic-management/** - Load balancing, canary deployments, circuit breakers
- **security/** - mTLS, authorization policies, JWT validation
- **observability/** - Distributed tracing, metrics, logging
- **policies/** - Rate limiting, retry policies, timeout configurations

## Key Features

### Traffic Management
- **Canary Deployments**: Gradual rollout with percentage-based traffic splitting
- **Blue-Green Deployments**: Zero-downtime deployments with instant rollback
- **Circuit Breakers**: Automatic failure detection and recovery
- **Retries**: Configurable retry logic with exponential backoff
- **Load Balancing**: Multiple algorithms (round-robin, random, least-request)

### Security
- **mTLS**: Automatic mutual TLS between all services
- **Authorization**: Fine-grained RBAC policies
- **JWT Validation**: Token validation at mesh level
- **Network Policies**: Zero-trust network segmentation

### Observability
- **Distributed Tracing**: End-to-end request tracing with Jaeger
- **Metrics**: RED metrics (Rate, Errors, Duration) with Prometheus
- **Service Graph**: Visual representation of service dependencies
- **Access Logs**: Structured logging with correlation IDs

## Deployment

### Prerequisites
```bash
# Install Istio
istioctl install --set values.pilot.env.PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true

# Enable namespace for injection
kubectl label namespace policycortex istio-injection=enabled
```

### Apply Configurations
```bash
# Apply base configuration
kubectl apply -f kubernetes/istio/base/

# Apply traffic management
kubectl apply -f kubernetes/istio/traffic-management/

# Apply security policies
kubectl apply -f kubernetes/istio/security/

# Apply observability
kubectl apply -f kubernetes/istio/observability/

# Apply policies
kubectl apply -f kubernetes/istio/policies/
```

## Monitoring

### Access Dashboards
```bash
# Kiali - Service mesh observability
istioctl dashboard kiali

# Grafana - Metrics visualization
istioctl dashboard grafana

# Jaeger - Distributed tracing
istioctl dashboard jaeger

# Prometheus - Metrics collection
istioctl dashboard prometheus
```

## Traffic Management Examples

### Canary Deployment (10% traffic to v2)
```yaml
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: core-api
        subset: v2
      weight: 100
  - route:
    - destination:
        host: core-api
        subset: v1
      weight: 90
    - destination:
        host: core-api
        subset: v2
      weight: 10
```

### Circuit Breaker
```yaml
spec:
  outlierDetection:
    consecutiveErrors: 5
    interval: 30s
    baseEjectionTime: 30s
    maxEjectionPercent: 50
```

## Security Configuration

### mTLS Mode
- **STRICT**: Only mTLS traffic allowed
- **PERMISSIVE**: Both mTLS and plain text (migration mode)
- **DISABLE**: No mTLS (not recommended)

### Authorization Policy Example
```yaml
spec:
  selector:
    matchLabels:
      app: core-api
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/policycortex/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/*"]
```

## Performance Tuning

### Envoy Proxy Configuration
- Connection pool sizing
- Request timeout settings
- Retry budget configuration
- Circuit breaker thresholds

### Resource Limits
```yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 2000m
    memory: 1024Mi
```

## Troubleshooting

### Common Issues
1. **503 errors**: Check circuit breaker status
2. **Connection refused**: Verify mTLS configuration
3. **High latency**: Review retry and timeout policies
4. **Traffic not routing**: Check virtual service configuration

### Debug Commands
```bash
# Check proxy configuration
istioctl proxy-config cluster <pod-name> -n policycortex

# View proxy logs
kubectl logs <pod-name> -c istio-proxy -n policycortex

# Analyze configuration
istioctl analyze -n policycortex

# Check mTLS status
istioctl authn tls-check <pod-name> -n policycortex
```