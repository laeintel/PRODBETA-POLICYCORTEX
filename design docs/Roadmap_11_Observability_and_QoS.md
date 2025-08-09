# 11. Observability & QoS

## 11.1 Metrics
- API latency/throughput, error rates, queue depth, action durations, savings realized

## 11.2 Traces & Logs
- Correlation IDs across FE→API→workers→AI; structured logs with PII scrubbing

## 11.3 SLOs
- API p99 < 300ms; 99.9% uptime; actions streamed within 200ms; evidence generation < 2 min for 90th percentile

## 11.4 Quality Gates
- Canary deploys; chaos tests on queues; DR drills; perf tests on deep endpoints

## 11.5 Alerting
- Budget burn, compliance drop, exposure spikes, action failures
