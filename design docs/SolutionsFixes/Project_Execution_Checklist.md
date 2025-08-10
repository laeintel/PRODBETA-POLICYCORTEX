# Project Execution Checklist

Status key: [ ] Todo · [~] In progress · [x] Done

This living checklist tracks closure of the initial 50 Roasts and V2 gaps. Update as work progresses.

## Phase 0 — Baseline & Gating (Roasts: 4, 5, V2‑2, 23)
- [~] 0.1 One‑command FE/BE start (FE 3000, BE 8080)
  - Links: `frontend/package.json`, `start-api.bat`, `start-local.bat`
- [x] 0.2 Health includes `azure_connected`
  - Links: `backend/services/api_gateway/main.py` (`/health`)
- [ ] 0.3 Windows preflight (env/ports/MSAL) script
  - Links: `scripts/preflight.ps1`
- [ ] 0.4 Build/type‑check baseline green (CI)
  - Links: `.github/workflows/application.yml`

## Phase 1 — AuthN/AuthZ + Tenant Isolation (Roasts: 6, 7, 45, V2‑4)
- [~] 1.1 Enforce JWT (issuer/audience/scope) on all API routes
  - Links: `backend/services/api_gateway/main.py`
- [ ] 1.2 Propagate tenant (tid) & enforce tenant filters in DB queries

## Phase 2 — Evidence, Audit, Secrets (Roasts: 11, 12, 13, 14, 15)
- [ ] 2.1 Evidence pipeline (signed artifacts; WORM option)
- [ ] 2.2 Tamper‑evident audit (hash chain + Merkle)
- [ ] 2.3 Key Vault mandatory; CI secret scanning gate

## Phase 3 — Durability & Migrations (Roasts: 8, 34, 40)
- [ ] 3.1 Move in‑memory flows to Postgres with migrations
- [ ] 3.2 Exceptions lifecycle (expiry/recert/evidence)

## Phase 4 — Observability & SLOs (Roasts: 21, 22)
- [ ] 4.1 OpenTelemetry tracing/metrics/log correlation FE→BE
- [ ] 4.2 SLOs + error‑budget CI gate

## Phase 5 — Honest UI & Resilience (Roasts: 3, 10, 18, 19, 20, V2‑3, 11)
- [x] 5.1 Global mock/demo badges + dry‑run labels
- [~] 5.2 Table virtualization/pagination rollout (Resources/Policies/RBAC)
  - Links: `frontend/components/VirtualizedTable.tsx`, `frontend/components/ServerPagination.tsx`
- [ ] 5.3 Saved filter presets; conflict UX
- [ ] 5.4 Offline queue used by at least one real flow; 409 conflict UX

## Phase 6 — Approvals & Guardrails (Roasts: 9, 11, 27)
- [ ] 6.1 Idempotency/retries/compensations
- [ ] 6.2 Multi‑stage approvals; preflight/blast‑radius required

## Phase 7 — Policy Depth & Enforcement (Roasts: 26, 27)
- [ ] 7.1 Assignment lineage/effect parity
- [ ] 7.2 Drift detection + safe remediation plan workflow

## Phase 8 — FinOps Real Ingestion (Roast: 29, V2‑14)
- [~] 8.1 FinOps endpoints scaffolding
- [ ] 8.2 CUR/FOCUS ingestion; unit economics; RI/SP planning

## Phase 9 — IAM Graph & Attack Paths (Roast: 28)
- [ ] 9.1 Identity/permission graph; toxic combos; least‑privilege recs

## Phase 10 — Privacy/Residency & Controls (Roasts: 24, 25, 43)
- [ ] 10.1 Data classification; residency toggles; TTL/legal hold
- [ ] 10.2 Control mapping to SOC2/ISO/NIST; auditor export pack

## Phase 11 — Multi‑cloud & Extensions (Roasts: 2, 36, 39)
- [ ] 11.1 AWS/GCP collectors; normalized schemas
- [ ] 11.2 Plugin/event contracts; example extension

## Phase 12 — Testing & CI/CD Hardening (Roasts: 30, 32, 42)
- [ ] 12.1 Unit/integration/e2e/load baseline; coverage goals
- [ ] 12.2 SBOM/CVE scan; deploy smoke tests + rollback gates

## Phase 13 — AI Truth & Provenance (Roasts: 1, 37, 49, 50; V2‑16)
- [ ] 13.1 Replace mocks with evaluated models; model cards/metrics
- [ ] 13.2 Provenance + confidence surfaced in UI; evaluations reproducible

## Cross‑cutting trackers
- [~] A11y conformance (axe CI, contrast, keyboard paths)
- [~] i18n/RTL dictionaries + locale QA
- [ ] Docs: runbooks, day‑2 ops, incident playbooks

---

Quick status
- FE on 3000; `/health` exposes `azure_connected`
- Virtualized tables and server pagination live on Resources/Policies/RBAC
- JWT enforcement wired; tenant filters pending
