# 2. System Architecture

## 2.1 High-Level View
- Public API: Rust/Axum core (`/api/v1/*`) — authn/authz, safety, orchestration, SSE
- Internal AI services: Python (FastAPI) — deep analyses, generation, predictions
- Event bus: NATS/Kafka — change detection, action lifecycle, telemetry fan‑out
- Storage: Postgres (state/audit), Redis (cache/queues), ClickHouse/Timescale (metrics), S3/Blob (evidence)
- Frontend: Next.js — drill‑downs, action drawer, voice/chat assistant

## 2.2 Core Components
- Gateway/API: request auth, tenancy context, rate-limits, schema validation
- Action Orchestrator: dry‑run diffs, approvals, blast‑radius, execution, rollback, audit
- Deep Insight Proxy: fan‑out to Python deep routes with timeouts/circuit breakers
- Evidence Factory: capture inputs/outputs, artifacts, and lineage per action/scan
- Policy Studio: NL→PAC generation, validation, equivalence mapping (Azure/AWS/GCP)

## 2.3 Data Flow (Detect→Decide→Act→Evidence)
1) Collectors/SDK ingest cloud facts → Core caches/DB
2) AI services analyze violations/opportunities → recommendations
3) User/AI triggers actions via Core → dry‑run → approvals → execute
4) Progress via SSE; results committed; evidence attached; KPIs updated

## 2.4 Security & Tenancy
- AAD/OIDC JWT verification; per‑tenant scoping; least privilege on provider creds
- PIM/JIT for privileged actions; no broad long‑lived credentials
- All actions signed and auditable; optional append-only ledger

## 2.5 Availability & Scalability
- Core stateless; horizontal scaling behind LB; shared Redis/Postgres
- Workers scale independently; backpressure via queues; idempotency keys
- SLOs: p99 < 300ms for reads; actions streamed within 200ms per step
