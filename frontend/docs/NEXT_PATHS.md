# PolicyCortex — Next.js App Router & API Map

This doc explains what each **Next.js path** does, which **backend/API** it uses, and its **current wiring status** so devs know exactly where to plug real data.

> Auth & security: all routes (except `/` and `/api/auth/*`) are protected by middleware and get nonce-based CSP + security headers. API handlers use `withAuth`, rate-limits, and audit logging.  
> Sources: `frontend/middleware.ts`, `frontend/app/api/v1/*`.  

---

## Core App Pages

### `/tactical`
- **Purpose:** entry hub for governance, security, ops, cost cards with deep links.
- **Notes:** modern card grid; navigation objects already in code.  
  Source: `frontend/app/tactical/page.tsx`.

---

### `/finops` (+ `/finops/anomalies`, `/finops/forecasting`, …)
- **Purpose:** FinOps command center (predictions, anomalies, optimization).
- **Wire to:**  
  - `/api/v1/predictions` (ML forecasts)  
  - Executive ROI endpoints (backend: `/api/v1/executive/roi`)  
- **Status:** page exists; subroutes referenced; connect charts/cards to live endpoints.  
  Sources: `frontend/app/finops/page.tsx` (routes list); API routes below.

---

### `/audit`
- **Purpose:** comprehensive **Audit Trail** (filters, stats, drill-downs).
- **Wire to:** `/api/v1/blockchain/audit` (list), `/api/v1/blockchain/verify{?hash}` (tamper check).
- **Status:** **mock data** right now; swap to API + show **Merkle proof / signature** badges.  
  Sources: `frontend/app/audit/page.tsx` (mock); backend blockchain endpoints.

---

### `/blockchain`
- **Purpose:** blockchain audit explorer & verification.
- **Wire to:** `/api/v1/blockchain/verify`, `/api/v1/blockchain/smart-contracts`, `/api/v1/blockchain/audit`.
- **Status:** verification is **simulated** in current UI; wire to real endpoints.  
  Sources: Validation report & backend modules.

---

### `/rbac`, `/resources` (and `/resources/[view]`)
- **Purpose:** access governance and asset inventory with URL-driven filters.
- **Wire to:** `/api/v1/resources` (server route in Next) or backend resource APIs via rewrites.
- **Status:** URL segment → filter is implemented; replace mocks with API.  
  Sources: dynamic routing & API handler.

---

### `/devsecops/pipelines`, `/devsecops/policy-code`
- **Purpose:** CI/CD gates, policy-as-code, "create fix PR" actions.
- **Wire to:** `/api/v1/devsecops/*` (backend), and PR action endpoints when ready.
- **Status:** UI present; connect actions to backend.

---

### `/ai`
- **Purpose:** umbrella for predictive, correlations, chat, unified dashboards.
- **Wire to:** `/api/v1/predictions` and correlation backends.
- **Status:** page exists; keep **predictive** as first-class; chat remains Labs for now.  
  Source map within nav doc.

---

## "Labs" (secondary) App Pages

> Shipped UI but currently **mocked** or semi-wired — keep discoverable but decentered in nav.

- `/copilot` — simulated responses today. Wire to `/api/v1/copilot/*`.  
- `/quantum` — hardcoded PQC/secret data; wire to `/api/v1/quantum/*`.  
- `/edge` — hardcoded nodes/policies; wire to `/api/v1/edge/*`.

---

## Next.js API Routes (in `frontend/app/api/v1/*`)

- `GET /api/v1/predictions` — **ML predictions** (auth + rate limit + audit). Wire UI top-right widget.  
- `GET /api/v1/metrics` — realtime metrics (auth + rate limit + audit). Dash cards.  
- `POST /api/auth/set-cookie` — set/clear `auth-token` cookie for middleware.  
- `GET/POST /api/v1/resources` — resource list/create (auth, rate-limit, CSRF, audit).  

> Additional **backend** APIs (Rust `core/src/api/*`) surfaced via rewrites:  
> **Executive:** `/api/v1/executive/*` • **Blockchain:** `/api/v1/blockchain/*` • **Quantum:** `/api/v1/quantum/*` • **Edge:** `/api/v1/edge/*` • **Copilot:** `/api/v1/copilot/*`. Validate in **VALIDATION_REPORT.md**.

---

## Middleware & Security

- **Route protection:** everything except `/` and `/api/auth/*` requires cookie auth; unauthenticated → login.  
- **CSP / headers:** nonce-based CSP + HSTS, XFO, X-CTO, etc. applied per response.  
- **Rewrites/proxy:** Next rewrites or middleware proxy events/GraphQL/health to backend.

---

## "Wire It" Checklist (replace mocks with live)

- **Audit Trail → Blockchain:** swap UI mock array to `GET /api/v1/blockchain/audit`, add **Verify** (Merkle/signature).  
- **FinOps → Predictions/ROI:** charts call `/api/v1/predictions` and backend `/api/v1/executive/roi`.  
- **Copilot/Quantum/Edge:** keep visible but flag **Labs** and connect later. (Wiring gaps documented.)