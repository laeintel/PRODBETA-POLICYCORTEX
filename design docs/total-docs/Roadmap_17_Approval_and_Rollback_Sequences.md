# 17. Approval & Rollback Sequences

## 17.1 Approval Flow (Mermaid)
```mermaid
sequenceDiagram
  participant FE as Frontend
  participant Core as Core API
  participant PG as Postgres
  participant Approver as Approver

  FE->>Core: POST /actions {dry_run:true}
  Core->>PG: create action (queued)
  Core-->>FE: action_id

  FE->>Core: GET SSE /actions/{id}/events
  Core-->>FE: preflight, diff, blast-radius

  FE->>Core: POST /actions/{id}/approve (operator)
  Core->>PG: record approval request (await approver)
  Core-->>FE: pending approval

  Approver->>Core: POST /actions/{id}/approve (approver)
  Core->>PG: approvals satisfied → transition to in_progress
  Core-->>FE: events: in_progress, verifying, completed
```

## 17.2 Rollback Flow (Mermaid)
```mermaid
sequenceDiagram
  participant FE as Frontend
  participant Core as Core API
  participant Worker as Exec Worker
  participant PG as Postgres

  note over Core,Worker: Execution error triggers rollback policy
  Worker-->>Core: event: failed (checkpoint=X)
  Core->>PG: update action status failed; attach diagnostics
  FE-->>Core: POST /actions/{id}/rollback
  Core->>PG: create rollback action, reference original
  Core->>Worker: execute rollback plan
  Worker-->>Core: events: in_progress → completed
  Core->>PG: update rollback action completed; audit both actions
  Core-->>FE: SSE updates and final status
```

## 17.3 Edge Cases
- Partial rollback: only revert successful steps; mark irrecoverable items and propose manual steps
- Expired approvals: require re‑approval if plan changed or TTL exceeded
- Blast‑radius increase post‑preflight: enforce re‑preflight and approval
