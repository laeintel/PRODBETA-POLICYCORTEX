# Live-Data Audit & Remediation Plan

## Objective
Identify every remaining mock/simulated data path across **frontend** and **backend**, then prescribe exact fixes so PolicyCortex always serves *real* data when `USE_REAL_DATA=true` / `NEXT_PUBLIC_DEMO_MODE=false`.

---

## 1. Environment Flags & Modes (root cause)
* **Frontend** relies on `NEXT_PUBLIC_DEMO_MODE` (string `'true'|'false'`) & optional `NEXT_PUBLIC_USE_REAL_DATA`.
* **Backend (Rust core)** uses `USE_REAL_DATA` (`true|1`) read via `data_mode::DataMode::from_env()`.
* When these are unset/`false`, code silently falls back to **simulated data**, masking broken integrations.

👉 **Fix:**
1. Set the following in all *non-demo* envs:
   ```env
   NEXT_PUBLIC_DEMO_MODE=false
   USE_REAL_DATA=true            # for Rust
   NEXT_PUBLIC_USE_REAL_DATA=true # for Next.js helpers (optional)
   ```
2. Add CI check that fails build if `NEXT_PUBLIC_DEMO_MODE` ≠ `false` for `main`/`prod` deploys.

---

## 2. Frontend – Mock Data Hot-Spots
| File | Mock Trigger | Action |
|------|--------------|--------|
| `app/api/v1/metrics/route.ts` | returns hard-coded KPI JSON when demo mode | Proxy to `/api/v1/metrics` (Rust) **only**; remove fallback. |
| `app/api/v1/predictions/route.ts` | same pattern | Remove fallback; surface 5xx to caller so UI shows error state. |
| `app/api/v1/correlations/route.ts` | same | Same removal. |
| `app/dashboard/page.tsx` | local `kpiData` array as default | Fetch from the API route above; show skeleton loader on failure. |
| `app/ai/predictions/page.tsx` | silent mock list on catch | Replace with error boundary + retry; no mock. |
| `api/graphql/mock.ts` + `graphqlMockMiddleware` | Entire fake GraphQL gateway | Guard with `if DEMO_MODE !== 'true' throw 404`; ensure *never* bundled in prod build. |
| `DemoDataProvider.tsx`, `AuthContext.tsx` | auto-injects demo user & fake streams | Disable provider when `NEXT_PUBLIC_DEMO_MODE=false`. |
| `components/AppShell.tsx` | `isDemoMode` gating certain UI | Remove paths that depend on demo data or hide with `<LabsBadge>` |

**Global remediation:**
* Delete or tree-shake `DemoDataProvider` from prod bundle.
* Standardise a single `isDemo()` util in `lib/env.ts`; all components must respect it.
* Add Playwright smoke tests (already drafted in UI docs) asserting zero mock banners in real-data mode.

---

## 3. Backend – Simulated Data Paths
| Module | Mock Evidence | Fix |
|--------|---------------|-----|
| `core/src/api/dashboard.rs` | `mock_azure_result` helper + `warn!("Using mock dashboard data")` | Re-enable Azure integration block, drop simulated fallback when `DataMode::Real`. |
| `core/src/api/governance.rs` | Multiple comment-guarded Azure calls, returns hard-coded `ComplianceStatus`, `PolicyViolation`, `CostSummary` etc. | Uncomment / implement Azure SDK calls; if unavailable → return `StatusCode::SERVICE_UNAVAILABLE` not mock. |
| `core/src/api/itsm.rs` | `health_score` etc. mocked at lines 585–593 | Wire to real ITSM source or feature-flag entire endpoint. |
| `core/src/api/health.rs` | Injects *mock* Azure connectivity details line 196 | Remove placeholder; query real health source. |
| `core/src/api/mod.rs` many handlers | `simulated_data::SimulatedDataProvider` used whenever `data_mode.is_real()==false` | In `DataMode::Real`, early-return 503 if dependencies unavailable. |
| `simulated_data` module | Central repository of demo payloads | Keep only for `DataMode::Simulated`; document clearly; unit-test that no call path reaches it when real. |

Additional checks:
* Search for `mock_azure_result` / `simulated_data` to find stragglers – ensure all guarded by `if data_mode.is_simulated()`.
* Enable `cargo clippy` / `cargo test` with env `USE_REAL_DATA=true` in CI; any simulated path executed must fail the test.

---

## 4. Broken / Missing Integrations
1. **Azure Policy & Resource Graph** – commented-out in governance modules.
   * Implement async client wrapper in `services/azure.rs` using `azure_identity` & `azure_mgmt_policy` crates.
   * Update handlers to construct `ComplianceStatus` from live query.
2. **Cost Management ROI (`/api/v1/executive/roi`)** – placeholder values.
   * Call `billing/benefitUtilizationSummaries` API; aggregate savings.
3. **Threat Metrics** – `active_threats: 2` constant.
   * Integrate Defender for Cloud alerts endpoint.
4. **Predictions Engine** – returns empty list when ML service missing.
   * Deploy Python ML service (`ml/app.py`) → set `PREDICTIONS_URL`; update `predictions.rs` to error if unset in real mode.
5. **GraphQL Gateway** – not wired.
   * Either remove GraphQL layer or implement Apollo Gateway proxy to existing REST core.

---

## 5. Enforcement Policy
* **Fail-fast principle:** In `DataMode::Real` *never* serve mocked payloads. Prefer 5xx so UI surfaces errors.
* **CI tests:** run backend integration tests with `USE_REAL_DATA=true` & dummy creds; assert 503 not 200 for missing deps.
* **Feature flags:**
  * `USE_REAL_DATA` (backend)
  * `NEXT_PUBLIC_DEMO_MODE` (frontend)
  * `MOCKS_ENABLED` (dev-only)

---

## 6. Step-by-Step Remediation Checklist
1. **Env Flip** – ensure staging/prod sets real-data flags (see §1).
2. **Backend:**
   1. Re-enable Azure SDK blocks, implement missing clients.
   2. Guard every handler: if `DataMode::Real` & dependency fails → `SERVICE_UNAVAILABLE`.
   3. Unit-test each endpoint in real-data CI.
3. **Frontend:**
   1. Delete/disable `DemoDataProvider` & GraphQL mocks for prod.
   2. Convert pages/api routes to *proxy only*; remove inline mock fallbacks.
   3. Implement UI error boundaries & skeletons.
4. **QA:**
   * Playwright smoke tests (Executive landing, Audit integrity chip, Predictions list) run against real API.
   * Automated scan grepping `MOCK`/`simulated` returns **zero** lines in shipped build.
5. **Docs:**
   * Update `README` deployment section with required env vars.
   * Add this audit file to release notes.

---

## 7. References (code locations)
* `core/src/api/dashboard.rs` lines 118-178 – mock dashboard.
* `core/src/api/governance.rs` lines 134-172, 339-437 – compliance & cost mock.
* `frontend/app/api/v1/metrics/route.ts` lines 55-62 – KPI mock.
* `frontend/api/graphql/mock.ts` full file – GraphQL mock.
* `frontend/app/dashboard/page.tsx` lines 204-214 – `kpiData` array.

Use these as starting points when removing mocks.

---

### Outcome
Once all steps are complete, **PolicyCortex will refuse to serve simulated data in production**, surfacing integration gaps instantly and ensuring demos == reality.
