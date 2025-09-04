# PolicyCortex DevOps Operations Guide

## Overview

This guide provides comprehensive documentation for the PolicyCortex DevOps infrastructure, including monitoring, observability, DORA metrics, supply chain security, and SLO management.

## Quick Start

```powershell
# Start the complete DevOps stack
.\scripts\start-devops-stack.ps1 -All

# Or start individual components
.\scripts\start-devops-stack.ps1 -StartMonitoring
.\scripts\start-devops-stack.ps1 -StartApplication
.\scripts\start-devops-stack.ps1 -CheckSLOs
```

## Architecture Components

### 1. Monitoring Stack (Prometheus + Grafana)

- **Prometheus**: Time-series metrics database
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and management
- **Node Exporter**: Host metrics collection
- **cAdvisor**: Container metrics

**Access Points:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3030 (admin/admin)
- AlertManager: http://localhost:9093

### 2. Distributed Tracing (OpenTelemetry + Jaeger)

- **OpenTelemetry Collector**: Trace/metric/log collection
- **Jaeger**: Distributed trace storage and UI
- **Auto-instrumentation**: Automatic trace generation

**Access Points:**
- Jaeger UI: http://localhost:16686
- OTLP Receiver: localhost:4317 (gRPC), localhost:4318 (HTTP)

### 3. Log Aggregation (Loki + Promtail)

- **Loki**: Log aggregation system
- **Promtail**: Log shipping agent
- **Grafana Integration**: Unified log/metric view

**Access Points:**
- Loki API: http://localhost:3100
- Logs in Grafana: http://localhost:3030

### 4. DORA Metrics

Track the four key DevOps metrics:

```powershell
# Track a deployment
.\scripts\dora-metrics\track-deployment.ps1 -Environment production -Version v2.1.0

# Track an incident
.\scripts\dora-metrics\track-incident.ps1 -Action open -IncidentId INC-001
.\scripts\dora-metrics\track-incident.ps1 -Action resolved -IncidentId INC-001

# Calculate DORA metrics
.\scripts\dora-metrics\calculate-dora-metrics.ps1 -Days 30 -ExportJson
```

**Metrics Tracked:**
- **Deployment Frequency**: How often code is deployed
- **Lead Time**: Time from commit to production
- **Change Failure Rate**: Percentage of deployments causing failures
- **MTTR**: Mean time to recovery from incidents

### 5. Supply Chain Security

#### SBOM Generation

```powershell
# Generate comprehensive SBOM
.\scripts\supply-chain\generate-sbom.ps1 -Format json -SignSBOM

# Generate and upload to registry
.\scripts\supply-chain\generate-sbom.ps1 -UploadToRegistry
```

**Outputs:**
- CycloneDX format SBOMs
- SaaSBOM for cloud dependencies
- Vulnerability reports

#### Container Signing

```powershell
# Sign containers with provenance
.\scripts\supply-chain\sign-containers.ps1 -GenerateProvenance -GenerateSLSA

# Verify signatures
.\scripts\supply-chain\sign-containers.ps1 -VerifySignatures
```

**Features:**
- Keyless signing with Sigstore/Cosign
- In-toto provenance attestations
- SLSA Level 3 compliance
- Signature verification

### 6. SLO Management

#### Configuration

SLOs are defined in `scripts/slo-management/slo-config.yaml`:

```yaml
slos:
  - service: policycortex-core
    objectives:
      - name: availability
        target: 99.9  # Three 9s
        window: 30d
```

#### Error Budget Calculation

```powershell
# Calculate error budgets
.\scripts\slo-management\calculate-error-budget.ps1 -Window 30 -AlertOnBreach

# Export report
.\scripts\slo-management\calculate-error-budget.ps1 -ExportReport
```

**Error Budget Policies:**
- Feature freeze at <25% budget
- Emergency response at <10% budget
- Postmortem required for rapid burn

## Dashboards

### SLO & Golden Signals Dashboard

Monitors the four golden signals:
- **Latency**: Response time distributions
- **Traffic**: Request rates
- **Errors**: Error rates and types
- **Saturation**: Resource utilization

### DORA Metrics Dashboard

Tracks DevOps performance:
- Deployment frequency trends
- Lead time distributions
- Change failure rates
- MTTR analysis

## Alerting Rules

### SLO-Based Alerts

| Alert | Threshold | Action |
|-------|-----------|---------|
| Burn Rate 25% | 5min @ 0.25% error | Warning |
| Burn Rate 50% | 3min @ 0.5% error | Ticket |
| Burn Rate 100% | 1min @ 1% error | Page |

### Service Health Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| Service Down | up == 0 for 1m | Critical |
| High Latency | p99 > 1s for 5m | Warning |
| High Error Rate | >5% for 5m | Critical |

## Operational Runbooks

### Service Degradation

1. Check Grafana dashboard for affected services
2. Review error logs in Loki
3. Check distributed traces in Jaeger
4. Identify root cause from metrics
5. Apply remediation
6. Track incident resolution time

### Error Budget Exhaustion

1. Implement feature freeze
2. Focus on reliability improvements
3. Conduct root cause analysis
4. Update monitoring/alerting
5. Document lessons learned

### Failed Deployment

1. Check deployment metrics
2. Review container signatures
3. Validate SBOM for vulnerabilities
4. Rollback if necessary
5. Update change failure rate

## Best Practices

### 1. Monitoring

- Define SLIs before SLOs
- Use percentiles (p95, p99) not averages
- Alert on symptoms, not causes
- Keep cardinality under control
- Use recording rules for complex queries

### 2. Tracing

- Sample intelligently (errors, high latency)
- Add correlation IDs to all requests
- Include business context in spans
- Keep span names consistent
- Limit span attributes

### 3. Security

- Generate SBOMs for every release
- Sign all container images
- Verify signatures before deployment
- Scan for vulnerabilities regularly
- Maintain attestation chain

### 4. DORA Metrics

- Automate metric collection
- Track all deployments (success and failure)
- Measure from commit to production
- Include rollbacks in failure rate
- Focus on trends, not absolute values

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check targets
curl http://localhost:9090/api/v1/targets

# Check service discovery
curl http://localhost:9090/api/v1/service-discovery
```

### Grafana Dashboard Empty

1. Check datasource configuration
2. Verify Prometheus is running
3. Check query syntax
4. Verify metrics exist

### Jaeger No Traces

1. Check OTLP collector status
2. Verify instrumentation
3. Check sampling configuration
4. Review network connectivity

### High Cardinality Issues

1. Identify high cardinality metrics
2. Review label usage
3. Implement recording rules
4. Adjust retention policies

## Maintenance

### Daily Tasks

- Review error budgets
- Check alert fatigue
- Monitor DORA metrics
- Verify backups

### Weekly Tasks

- Review SLO compliance
- Analyze incident trends
- Update runbooks
- Security scanning

### Monthly Tasks

- SLO calibration
- Capacity planning
- Cost optimization
- Compliance audit

## Integration Points

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Generate SBOM
  run: |
    ./scripts/supply-chain/generate-sbom.ps1 -Format json
    
- name: Sign Container
  run: |
    ./scripts/supply-chain/sign-containers.ps1 -GenerateProvenance
    
- name: Track Deployment
  run: |
    ./scripts/dora-metrics/track-deployment.ps1 \
      -Environment ${{ github.event.inputs.environment }} \
      -Version ${{ github.ref }}
```

### Kubernetes Integration

```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: policycortex
spec:
  selector:
    matchLabels:
      app: policycortex
  endpoints:
  - port: metrics
    interval: 30s
```

## Security Considerations

1. **Metrics Exposure**: Use authentication for metrics endpoints
2. **Secret Management**: Store credentials in secure vaults
3. **Network Isolation**: Separate monitoring network
4. **Data Retention**: Follow compliance requirements
5. **Access Control**: Implement RBAC for dashboards

## Performance Optimization

1. **Metric Cardinality**: Keep below 10M series
2. **Query Optimization**: Use recording rules
3. **Storage Efficiency**: Adjust retention policies
4. **Sampling Strategy**: Balance visibility vs. cost
5. **Dashboard Efficiency**: Limit concurrent queries

## Disaster Recovery

### Backup Strategy

```powershell
# Backup Prometheus data
docker exec prometheus tar czf /tmp/prometheus-backup.tar.gz /prometheus

# Backup Grafana dashboards
docker exec grafana grafana-cli admin export-dashboard

# Backup Loki data
docker exec loki tar czf /tmp/loki-backup.tar.gz /loki
```

### Recovery Procedures

1. Restore from backups
2. Replay WAL if available
3. Rebuild dashboards from code
4. Verify metric continuity
5. Test alerting pipeline

## Cost Management

### Optimization Tips

1. **Retention Policies**: Balance history vs. cost
2. **Sampling Rates**: Adjust based on criticality
3. **Storage Tiers**: Use appropriate storage classes
4. **Query Caching**: Implement result caching
5. **Resource Limits**: Set appropriate limits

## Support and Resources

- **Documentation**: [Internal Wiki](https://wiki.policycortex.com/devops)
- **Runbooks**: [Runbook Repository](https://github.com/policycortex/runbooks)
- **Slack Channel**: #devops-support
- **On-Call Schedule**: PagerDuty rotation

## License

Copyright (c) 2024 PolicyCortex. All rights reserved.