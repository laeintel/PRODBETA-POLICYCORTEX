# 11. Observability & QoS

## 11.1 Metrics
- API: latency p50/p95/p99, RPS, error rate, saturation
- Orchestrator: queue depth, actions/min, step latency, failure rate, rollback rate
- Savings: realized/month, time‑to‑savings, plan acceptance rate
- Compliance/Security: control pass rate, violation MTTR, active threats blocked

## 11.2 Traces & Logs
- Correlation IDs end‑to‑end; structured logs (JSON) with PII scrubbing

## 11.3 SLOs
- Core API p99 < 300ms; 99.9% uptime
- SSE event delivery < 200ms median
- Evidence generation < 2 min P90

## 11.4 Alerts (Playbooks)
- API error rate > 2% 5m → page; check recent deploy; rollback if needed
- Queue depth > threshold 10m → scale workers; investigate stuck actions
- Budget burn > threshold → auto guardrails on; notify FinOps
- Compliance drop > 5% in 24h → run drift scan; generate action batch

## 11.5 Dashboards
- Exec: Savings, Risk, Compliance score, Actions
- Ops: API SLO, queue health, action outcomes, error budgets
