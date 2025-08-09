# 5. Action Orchestrator & Safety

## 5.1 Lifecycle
queued → preflight → in_progress → verifying → completed|failed|rolled_back

## 5.2 Preflight & Dry‑Run
- Compute change set; diff against current; estimate blast‑radius (resource count, spend impact, risk)
- Policy guards (e.g., never delete prod without tag + approval)
- Output human‑readable plan + machine evidence

## 5.3 Approvals
- Thresholds by environment/impact; multi‑party; expirations; JIT elevation (PIM)

## 5.4 Execution & Rollback
- Idempotent operations; partial rollback plans; compensating actions; checkpoints
- Timeouts + retry budgets; exponential backoff; abort on policy violation

## 5.5 Observability & Audit
- SSE stream per action; events persisted; correlation IDs; link to evidence artifacts
- Signed audit records; optional append-only ledger

## 5.6 Security
- Scoped credentials per tenant; short‑lived tokens; no standing admin
- Execution sandbox and rate caps; circuit breakers

## 5.7 Interfaces
- POST `/actions` (idempotent key)
- GET `/actions/{id}`, SSE `/actions/{id}/events`

## 5.8 Failure Playbooks
- Safe abort; quarantine; fallback to manual with context
- Auto ticketing (ITSM) with full evidence
