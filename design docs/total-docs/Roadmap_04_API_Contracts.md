# 4. API Contracts (Public)

Base: `/api/v1`

## 4.1 Metrics & Recommendations
- GET `/metrics`
Response:
```json
{
  "policies": {"total": 347, "active": 298, "violations": 12, "automated": 285, "compliance_rate": 99.8, "prediction_accuracy": 92.3},
  "rbac": {"users": 1843, "roles": 67, "violations": 3, "risk_score": 18.5, "anomalies_detected": 7},
  "costs": {"current_spend": 145832, "predicted_spend": 98190, "savings_identified": 47642, "optimization_rate": 89},
  "network": {"endpoints": 487, "active_threats": 2, "blocked_attempts": 127, "latency_ms": 12.3},
  "resources": {"total": 2843, "optimized": 2456, "idle": 234, "overprovisioned": 153},
  "ai": {"accuracy": 96.8, "predictions_made": 15234, "automations_executed": 8921, "learning_progress": 87.3}
}
```

- GET `/recommendations`
Response:
```json
[
  {"id":"rec-001","recommendation_type":"cost_optimization","severity":"high","title":"VM Right-Sizing Opportunity","description":"...","potential_savings":12450,"automation_available":true,"confidence":94.5}
]
```

## 4.2 Deep Insights
- GET `/policies/deep`
Response: `{ "complianceResults": [ { "assignment": { ... }, "summary": { ... }, "nonCompliantResources": [ ... ] } ] }`

- GET `/rbac/deep`, `/costs/deep`, `/network/deep`, `/resources/deep` — similar envelopes

## 4.3 Entities
- GET `/resources` → `{ resources: [...], total: n }`
- GET `/resources/{id}` → a single resource
- GET `/policies` / `/policies/{id}`

## 4.4 Actions & Exceptions
- POST `/actions`
Request:
```json
{ "action_type": "enable_encryption", "resource_id": "/subscriptions/.../storageAccounts/st1", "params": {"mode":"platformManaged"} }
```
Response:
```json
{ "action_id": "e1a6b1cd-..." }
```

- GET `/actions/{id}` →
```json
{ "id":"...","action_type":"enable_encryption","resource_id":"...","status":"in_progress","params":{},"result":null,"created_at":"...","updated_at":"..." }
```

- GET (SSE) `/actions/{id}/events` → event stream lines (`data: queued`, `data: in_progress: executing`, ...)

- POST `/exception` → `{ "exceptionId": "exc-20250101120000", "status": "Approved" }`

## 4.5 Conversation
- POST `/conversation` → `{ response, intent, confidence, suggested_actions[], generated_policy }`

## 4.6 Errors
All errors follow:
```json
{ "error": "message", "code": "ERR_CODE", "correlation_id": "uuid" }
```

# 4. API Contracts (Internal – Python)

- GET `DEEP_API_BASE/api/v1/*/deep`
- POST `DEEP_API_BASE/api/v1/policies/generate`
- POST `DEEP_API_BASE/api/v1/analyze`

Timeouts: 2–5s; circuit breaker open → Core returns fallback with `success:false`.
