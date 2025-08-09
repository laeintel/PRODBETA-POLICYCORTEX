# 5. Action Orchestrator & Safety

## 5.1 Lifecycle State Machine
States: queued → preflight → in_progress → verifying → completed | failed | rolled_back

Transitions:
- queued → preflight: system picks up action; fetches resource facts
- preflight → in_progress: approvals satisfied, dry‑run diff accepted
- in_progress → verifying: execution finished, running validation checks
- verifying → completed: validations passed; evidence attached
- any → failed: fatal error; attach diagnostics; propose rollback
- in_progress/verifying → rolled_back: compensating actions executed

## 5.2 Preflight & Dry‑Run
- Compute change set: current vs target (policy/resource/infra)
- Blast‑radius scoring: #resources touched, cost delta, risk class, environment
- Guard policies: deny/require approvals for high‑risk ops (e.g., delete in prod)
- Output
  - Human: YAML/Markdown summary with diffs, impacts, and guard decisions
  - Machine: JSON plan, resource list, dependency order, idempotency key

## 5.3 Approvals
- Threshold matrix by environment and blast‑radius tier
- Multi‑party/role approvals (operator+approver; break‑glass with reason)
- Expirations and audit trail of decision

## 5.4 Execution & Rollback
- Idempotent step execution; retries with budgets
- Checkpoints for partial rollback; compensating plans generated at preflight time
- Abort switches on risk spikes or policy violations

## 5.5 Observability & Audit
- SSE broadcast per action; persisted `action_events`
- Evidence pack: input plan, step logs, diffs, post‑validation, screenshots/exports
- Correlation IDs: FE → API → workers → AI

## 5.6 Security
- Per‑tenant scoped creds; short‑lived tokens; PIM/JIT elevation via approval
- Rate limits and concurrency caps; circuit breakers

## 5.7 API Surfaces
- POST `/actions` (idempotency key header optional)
- GET `/actions/{id}`
- GET SSE `/actions/{id}/events`

## 5.8 Failure Playbooks
- Safe abort and quarantine; auto ticket creation with evidence
- Manual handoff with proposed next steps
