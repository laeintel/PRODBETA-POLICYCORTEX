# 4. API Contracts (Public)

Base: `/api/v1`

## 4.1 Metrics & Recommendations
- GET `/metrics` → GovernanceMetrics
- GET `/recommendations` → ProactiveRecommendation[]

## 4.2 Deep Insights
- GET `/policies/deep` → { complianceResults[] }
- GET `/rbac/deep` → RBAC deep analysis
- GET `/costs/deep` → cost breakdown/anomalies
- GET `/network/deep` → network exposures
- GET `/resources/deep` → health + compliance

## 4.3 Entities
- GET `/resources` | `/resources/{id}`
- GET `/policies` | `/policies/{id}`

## 4.4 Actions & Exceptions
- POST `/actions` { action_type, resource_id, params } → { action_id }
- GET `/actions/{id}` → ActionRecord
- GET (SSE) `/actions/{id}/events` → streamed steps
- POST `/exception` { resource_id, policy_id, reason } → { exceptionId }

## 4.5 Conversation
- POST `/conversation` { query, context, session_id } → ConversationResponse

Notes:
- All requests authenticated via OIDC/JWT; tenant derived from claims; rate‑limited.
- Idempotency keys on POST /actions.

# 4. API Contracts (Internal – Python)

- GET `DEEP_API_BASE/api/v1/*/deep` — deep analyses
- POST `DEEP_API_BASE/api/v1/policies/generate` — NL→policy
- POST `DEEP_API_BASE/api/v1/analyze` — environment analysis

Error model: `{ error, code, correlation_id }`
Timeouts: 2–5s per deep call; circuit‑breaker in Core.
