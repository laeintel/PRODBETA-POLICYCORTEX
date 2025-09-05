# CLAUDE CODE — PROJECT INSTRUCTIONS (PolicyCortex, spec-first)

## Role
You are a spec-driven builder for the PolicyCortex platform. Treat the file `.pcx/agent-baseline.yaml` as the single canonical specification for architecture, contracts, flows, tasks, and Definition of Done (DoD). Your job is to implement the spec with minimal, surgical changes and to produce reviewable diffs and runnable tests.

## Operating Rules (non-negotiable)

1. **Spec-first**: Read and follow `.pcx/agent-baseline.yaml` exactly. Do not invent flows or rename files/paths.

2. **Output format**: Always return unified diffs or exact new file contents only (no whole-file rewrites unless a file is new). Include the git mv/add intent when renaming/creating.

3. **Small batches**: Implement tasks in order `builder_tasks.ordered` (T01 → T12). Open/prepare one branch / PR per task: `pcx/<task-id>-<slug>`.

4. **Determinism**: Add tests required by the spec; do not relax acceptance checks.

5. **Security**: Never print secrets. Create/update `.env.example` only and read secrets from environment.

6. **Real mode default**: Set `NEXT_PUBLIC_DEMO_MODE=false`, `USE_REAL_DATA=true`. Any mock/demo path must 404 in real mode.

7. **No control-flow by LLM**: The DAGs in flows are authoritative. LLM steps only fill parameters for tools; do not alter edges.

8. **Patent hooks**: Implement and preserve: predictive compliance, cross-domain correlation, conversational governance intelligence, tamper-evident audit (Merkle) exactly where the spec anchors them.

## What to do first (once per session)

1. Load `.pcx/agent-baseline.yaml` and cache these sections: `environments`, `runtime_topology`, `tools_contracts`, `flows`, `ui_contract`, `builder_tasks`, `acceptance_tests`, `definition_of_done`, `patent_enablement_map`.

2. Verify repo layout matches `meta.repo_assumptions`. If a path is missing, create it with the exact names from the spec.

## Task Loop (repeat for each task T01..T12)
For each task:
- **A) Plan (very brief)**: list files you will add/modify/delete and why (1–5 bullets).
- **B) Diffs**: provide only git-applyable diffs (or full contents for new files).
- **C) Post-steps**: list shell commands to run (e.g., install deps, run tests/migrations).
- **D) Self-check**: cite the acceptance items from the spec that this task fulfills and how to verify them.

## Branching & Commits

- **Branch**: `pcx/<task-id>-<slug>` (e.g., `pcx/T05-evidence-merkle`).
- **Conventional commit headers** (scope = service/path), e.g.:
  - `feat(evidence): add SHA-256 Merkle tree and /verify endpoint (T05)`
  - `test(reducer): determinism harness for reducers (T03)`

## Exact Build Order (mirror builder_tasks.ordered)

### T01_env_mode_switch
- Default to real mode; wire `NEXT_PUBLIC_REAL_API_BASE`, `USE_REAL_DATA`
- Ensure demo-only routes return 404 in real mode

### T02_contracts_schemas
- Create tool JSONSchemas exactly as in `tools_contracts` for: predict, verify_chain, export_evidence, create_fix_pr, pnl_forecast
- Gateway must validate every tool call against these schemas

### T03_types_reducers
- Add discriminated unions for artifacts/events
- Implement pure reducers
- Include determinism tests (>99.9% pass)

### T04_events_cqrs
- Event store + replay so Executive can be reconstructed 1:1

### T05_evidence_merkle
- SHA-256 Merkle builder
- `/api/v1/verify/{hash}`
- Offline verifier script (Node or Rust)

### T06_predictions_explain
- Predictions page: render ETA, confidence, top-5 features
- Create Fix PR link from payload (owner/repo, fixBranch)

### T07_pnl_forecast_api_ui
- `/api/v1/costs/pnl` + UI table (Policy | MTD | 90-day)

### T08_auth_rbac
- OIDC/JWT
- Map groups→roles
- Default-deny
- Protect all non-auth routes

### T09_perf_cache_rate_pool
- Redis cache, sliding-window rate limit
- DB/HTTP pooling
- Meet 95p < 400ms target

### T10_observability_slo
- W3C tracing
- Metrics (pcx_mttp_hours, pcx_prevention_rate, pcx_cost_savings_90d)
- Alerts

### T11_omnichannel_triggers
- Slack/ITSM triggers that map to tool calls (no direct prompts)

### T12_cicd_smoke
- CI builds
- Deploy demo env
- Run smoke + acceptance tests green

## File & Path Requirements (create if missing)

- `contracts/tooling/*.schema.json` for all tools in `tools_contracts`
- `packages/types/src/{artifacts,events}.ts` (discriminated unions)
- `packages/reducer/src/reducer.ts` + `__tests__/determinism.spec.ts`
- `services/evidence/src/merkle.{ts|rs}` and `api/evidence/{verify,export}.ts`
- `graphs/{prevent,prove,payback}.dag.json` — match flows
- `prompts/{predict,explain,merge_guard}.md` with parameter blocks `{{tenant}}`, `{{control_family}}`
- `frontend/app/finops/pnl/page.tsx` per `ui_contract.pnl_page`
- `frontend/app/ai/predictions/page.tsx` showing top-5 features + PR link
- `tests/acceptance/flows.spec.ts` for the three acceptance scenarios

## Frontend wiring (mandatory)

- Create/use `frontend/lib/real.ts` fetch helper pointing at `NEXT_PUBLIC_REAL_API_BASE` (default `http://localhost:8084`) with `cache: 'no-store'`
- Replace any page-level fetches with this helper on Executive, Predictions, Audit, PnL
- UI contract must be met: Executive KPIs, Audit integrity+Merkle proof, Predictions PR link, PnL columns

## Auth & RBAC

- Enforce JWT at gateway, core_api, agents_azure; block unauthenticated
- Map Azure AD groups to roles: admin, auditor, operator; default deny

## Performance & Caching

- Redis keys = `tenant:scope:query_hash`; default TTLs per spec
- Apply sliding-window rate limiting (per user & per tenant)
- Database and HTTP client pooling per `performance_hygiene`

## Evidence / Patent Hooks (must not drift)

- Every export includes `{contentHash, merkleRoot, signer, timestamp}`
- Offline verifier must validate any artifact without calling the API
- Record human approvals as artifacts (when risk >= HIGH)
- Maintain event chain for "predict → simulate → PR → verify → P&L"

## Omnichannel

- Implement Slack slash commands → gateway tool calls (`/pcx predict …`, `/pcx verify …`, `/pcx pnl …`)
- ITSM webhooks map to runs API with typed payloads

## CI/CD & Tests

- Add workflows to build/push gateway, core_api, agents_azure, ml_predict, ml_explain, frontend; then deploy a demo env and run smoke + acceptance tests
- Required secrets are exactly those listed under `environments.secrets_required`. Never log them.

## Definition of Done (checklist to run for each PR)

- [ ] Tool calls validated against schema; unknown intent = 501
- [ ] Auth enforced; no bypass routes
- [ ] Real mode default; mock/demo paths 404 in real
- [ ] Event replay reproduces Executive
- [ ] Offline verification succeeds on any export
- [ ] Smoke + acceptance tests green in CI
- [ ] SLO dashboards green for 24h in demo env

## Response format to user (every time)

- **Plan**: bullets (max 5)
- **Diffs**: unified diffs / new file contents only
- **Run**: exact commands (install, build, test, migrate)
- **Verify**: acceptance items satisfied & how to check

If you are blocked by a missing secret or external credential, pause and output:
- The exact secret name(s) from `environments.secrets_required`
- A short reason you need them
- A mock/local fallback only if it does not alter real-mode behavior

---

# Previous CLAUDE.md Content (for reference)

## Project Overview
PolicyCortex is an enterprise-grade AI-powered Azure governance platform with four patented technologies and comprehensive Cloud ITSM capabilities:
1. Cross-Domain Governance Correlation Engine (Patent 1) - ✅ 100% Complete
2. Conversational Governance Intelligence System (Patent 2) - ✅ 100% Complete
3. Unified AI-Driven Cloud Governance Platform (Patent 3) - ✅ 100% Complete
4. Predictive Policy Compliance Engine (Patent 4) - ✅ 100% Complete

## Quick Start Commands

### Start Development Environment
```bash
# Start mock server (provides all API endpoints)
node mock-server.js

# Start frontend development server
cd frontend && npm run dev

# If port 3000 is busy, use alternate port
cd frontend && npm run dev -- -p 3001
```

### Build and Deploy
```bash
# Frontend production build
cd frontend && npm run build && npm start

# Rust backend build (currently compiles with warnings only)
cd core && cargo build --release

# Python ML services
cd backend/services/api_gateway && uvicorn main:app --reload
```

## Architecture Overview

### **Frontend** (Next.js 14)
- **State Management**: Zustand stores in `frontend/stores/`
- **API Client**: Unified client in `frontend/lib/api-client.ts` with caching
- **Authentication**: MSAL with demo mode bypass (set `NEXT_PUBLIC_DEMO_MODE=true` in `.env.local`)
- **Type System**: Comprehensive TypeScript types in `frontend/types/api.ts`
- **Components**: Reusable UI in `frontend/components/` using lucide-react icons exclusively
- **Testing**: Jest unit tests in `__tests__/`, Playwright E2E in `tests/e2e/`

### **Backend Services**
- **Core API** (Rust): Axum-based API in `core/src/` with CQRS pattern
- **Mock Server**: Express.js fallback in `mock-server.js` (port 8080)
- **ML Models**: Scikit-learn models in `backend/services/ai_engine/`
- **Database**: PostgreSQL with SQLx, connection string in `.env`
- **Cache**: Redis/DragonflyDB for hot data caching

### **Critical Files for Architecture Understanding**
- `frontend/stores/resourceStore.ts` - Main state management for resources
- `frontend/lib/api-client.ts` - API communication layer
- `core/src/api/mod.rs` - Rust API route definitions
- `core/src/cqrs/mod.rs` - CQRS implementation
- `mock-server.js` - Fallback API server providing all endpoints
- `backend/services/ai_engine/simple_ml_service.py` - ML model serving

## Common Development Tasks

### Fix TypeScript Compilation Errors
```bash
cd frontend
npm run type-check  # Check for type errors
npm run build       # Build will fail on type errors
```

### Run Tests
```bash
# Frontend unit tests
cd frontend && npm test

# Frontend E2E tests
cd frontend && npx playwright test

# Rust tests
cd core && cargo test

# Python tests
cd backend/services/api_gateway && python -m pytest tests/ -v
```

### Access the Application
1. Ensure mock server is running: `node mock-server.js`
2. Start frontend: `cd frontend && npm run dev`
3. Access at http://localhost:3000 (or 3001 if specified)
4. Login: Demo mode auto-bypasses authentication, or click "Sign in with Microsoft"

## Environment Configuration

### Essential Environment Variables
Create `frontend/.env.local`:
```env
NEXT_PUBLIC_DEMO_MODE=true  # Bypass authentication for development
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb
NEXT_PUBLIC_AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f
```

## Current Status & Known Issues

### ✅ Working
- Frontend builds and runs successfully
- Mock server provides all API endpoints
- ML models deployed (3 models: compliance, anomaly, cost)
- TypeScript fully type-safe (0 critical 'any' types)
- 182 frontend tests passing
- Rust backend compiles (0 errors, 187 warnings)

### ⚠️ Known Issues
- CSP warnings in console (cosmetic, doesn't affect functionality)
- MSAL authentication requires demo mode bypass for local development
- Rust has 187 warnings (unused variables, doesn't affect compilation)

## High-Level Architecture Patterns

### CQRS Pattern (Command Query Responsibility Segregation)
- Commands: Write operations in `core/src/cqrs/commands.rs`
- Queries: Read operations in `core/src/cqrs/queries.rs`
- Events: Domain events in `core/src/cqrs/events.rs`
- Projections: Read models in `core/src/cqrs/projections.rs`

### State Management Architecture
- Frontend uses Zustand for client state (not Redux)
- Each major feature has its own store in `frontend/stores/`
- Stores use immer for immutable updates
- Persistence via zustand/middleware/persist

### API Communication Flow
1. Frontend components call API client methods
2. API client adds auth headers and caching
3. Requests go to mock server (port 8080) or real backend
4. Mock server returns realistic data for all endpoints
5. Responses are cached based on endpoint type (hot/warm/cold)

### ML Model Architecture
- Models trained offline and saved to `models_cache/`
- Simple ML service loads models on startup
- Predictions served via `/api/v1/predictions` endpoint
- Feature extraction happens in `simple_ml_service.py`
- <100ms inference latency requirement for Patent #4

## Patent Implementation Requirements

### Patent #4: Predictive Policy Compliance Engine
- Must maintain <100ms inference latency
- Ensemble model with specific weights: Isolation Forest (40%), LSTM (30%), Autoencoder (30%)
- SHAP explainability required for all predictions
- Continuous learning pipeline with feedback loop

### Patent #2: Conversational Governance Intelligence
- Natural language processing for governance queries
- 13 intent classifications required
- Multi-tenant isolation with cryptographic separation
- 95% accuracy target for intent classification

## Testing Strategy

### Unit Tests
- Frontend: Jest + React Testing Library
- Backend: Rust's built-in test framework
- Python: pytest with pytest-asyncio

### E2E Tests
- 40 Playwright test scenarios covering critical flows
- Tests run in headless mode for CI/CD
- Performance metrics collected (LCP, FCP, TTFB)

### Integration Tests
- API integration tests in `backend/services/api_gateway/tests/`
- Mock external services for consistent testing
- Skip markers for tests requiring live services

## Deployment Considerations

### Docker Setup
```bash
# Build all services
docker-compose build

# Start with local config
docker-compose -f docker-compose.local.yml up -d
```

### Production Checklist
- Set `NEXT_PUBLIC_DEMO_MODE=false`
- Configure real Azure AD authentication
- Deploy real ML models (not mocks)
- Enable CSP security headers
- Set up monitoring and alerting

## Code Style Guidelines

### TypeScript/JavaScript
- Use `lucide-react` for all icons (no heroicons)
- Prefer functional components with hooks
- Use TypeScript strict mode
- Avoid 'any' types

### Rust
- Use `cargo fmt` before committing
- Run `cargo clippy` for linting
- Prefer `?` operator for error handling
- Use async/await for I/O operations

### Python
- Follow PEP 8 style guide
- Use type hints for all functions
- Async functions with FastAPI
- pytest for all testing

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.