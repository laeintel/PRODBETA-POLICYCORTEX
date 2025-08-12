# Comprehensive Roast and Fixes (v1)

Date: 2025-08-12

Scope: 40 concrete reasons this application will not be purchased or used by enterprises today, with directly actionable fixes. Use this as a remediation backlog. Items intentionally overlap with prior docs but add prescriptive implementation steps.

## 1) Vaporware AI claims
- Why it fails: Claims “GPT-grade domain expertise” without reproducible models, datasets, or benchmarks.
- Fix: Ship a verifiable model registry (`core/src/ai/model_registry.rs`), public model cards, eval datasets, CI eval runs, and acceptance thresholds that gate releases.

## 2) Multi-cloud in name only
- Why it fails: Azure-centric stubs; no AWS/GCP collectors or normalized schema.
- Fix: Implement AWS/GCP collectors in `core/src/collectors/*`, define a cross-cloud resource and policy schema, add contract tests for parity.

## 3) Mock data masquerading as real
- Why it fails: Simulated data appears real; trust collapses.
- Fix: Force “Simulated vs Real” banners in FE, block write actions in simulated mode, log the data source with every response.

## 4) Brittle local runtime (especially on Windows)
- Why it fails: Manual steps, port collisions, and toolchain pain.
- Fix: Devcontainer + Docker Compose as the default dev path, preflight checks, dynamic port discovery, single command bootstrap.

## 5) Fragile service routing
- Why it fails: Hardcoded ports and rewrites drift; easy to break.
- Fix: Environment-driven discovery for FE rewrites (`frontend/middleware.ts`, `next.config.*`), central config for base URLs, health checks.

## 6) Authentication mismatch
- Why it fails: FE uses MSAL; APIs inconsistently validate `aud`/`iss` and scopes.
- Fix: Standardize JWT validation (`core/src/auth.rs`), enforce scopes/roles on every route, add token rotation and clock-skew handling.

## 7) Tenant isolation incomplete
- Why it fails: Risk of cross-tenant access; a non-starter for enterprises.
- Fix: Enforce RLS in DB, propagate `tenant_id` from token to all queries, add tenancy middleware (`core/src/tenant.rs`).

## 8) No durable system of record
- Why it fails: Actions/events/audit don’t persist; nothing survives restarts.
- Fix: Persist actions, events, evidence into Postgres; add migrations and PITR backups; introduce retention policies.

## 9) Unsafe action orchestrator
- Why it fails: No idempotency, retries, compensation, or approvals; outages waiting to happen.
- Fix: Idempotency keys, retry policies, saga/compensation patterns, and approval gates before execution.

## 10) Misleading “Remediate” UX
- Why it fails: Buttons imply safety/rollback that don’t exist.
- Fix: Preflight impact diffs, blast-radius display, confirmations, and disabled destructive ops without approvals.

## 11) No approvals/Separation of Duties (SoD)
- Why it fails: Violates change management standards.
- Fix: Multi-stage approvals, SoD policy engine, break-glass with recording and time-bound tokens.

## 12) Evidence pipeline missing
- Why it fails: “Compliance” without signed/immutable evidence.
- Fix: Build an Evidence Factory (`core/src/compliance/`), generate signed artifacts, WORM storage, auditor exports.

## 13) Tamper-evident logs absent
- Why it fails: Forensics can’t trust records.
- Fix: Append-only, hash-chained audit logs (`core/src/audit_chain.rs`), verification on startup, rotation and retention.

## 14) Weak secrets management
- Why it fails: Env vars instead of vaults; high blast radius.
- Fix: Move to Key Vault/Secrets Manager, workload identity, scoped access, rotation policies; remove plaintext envs from code and CI.

## 15) Secret leakage risks
- Why it fails: Keys can end up in logs/bundles.
- Fix: Global redaction middleware, static scans in CI to block bundling, runtime guards against printing sensitive values.

## 16) Accessibility ignored
- Why it fails: Fails WCAG; excludes users and fails policy.
- Fix: Adopt `frontend/lib/accessibility.ts` helpers across components, add automated a11y tests, fix contrast/focus/ARIA.

## 17) No i18n/l10n
- Why it fails: English-only UI; global orgs can’t roll out.
- Fix: Wrap app in i18n provider, migrate strings to locales, add ICU formatting; tenant-level language selection.

## 18) UI does not scale
- Why it fails: Large tables and lists will lag/crash.
- Fix: Virtualized lists, server-side pagination, cursor-based pagination, query caching and backpressure.

## 19) Primitive search/filters
- Why it fails: Daily workflows are slow and clumsy.
- Fix: Faceted filters, saved views, advanced query builder with permissions-aware scopes.

## 20) No offline/conflict strategy
- Why it fails: Real networks fail; users lose work.
- Fix: PWA offline queues, optimistic concurrency with ETags, conflict resolution UI.

## 21) Observability lip service
- Why it fails: No end-to-end tracing or metrics; ops fly blind.
- Fix: OpenTelemetry tracing + metrics + logs, correlation IDs, dashboards for RED/USE + business KPIs.

## 22) No SLOs/error budgets
- Why it fails: Reliability undefined; unmanaged risk.
- Fix: Define SLIs/SLOs per critical endpoint, track error budgets, gate releases on burn rate.

## 23) DR strategy missing
- Why it fails: Unknown RPO/RTO; unacceptable risk.
- Fix: Backups, cross-region replicas, tested restore drills, documented RPO/RTO.

## 24) Data retention undefined
- Why it fails: Legal and compliance blockers.
- Fix: TTL policies, legal hold workflows, purge flows with audit.

## 25) Control frameworks unmapped
- Why it fails: “Compliance” cannot be audited.
- Fix: Map features to SOC2/ISO/NIST, live control status, evidence linkage, auditor export packs.

## 26) Shallow policy engine
- Why it fails: Reduced to JSON rendering; lacks parity.
- Fix: Assignment graph, inheritance, parameter resolution, effect semantics parity across clouds.

## 27) No enforcement path
- Why it fails: Posture without control; drift persists.
- Fix: Integrate with Azure Policy/Terraform/Bicep and AWS/GCP equivalents; drift detection + safe auto-remediation.

## 28) Missing IAM/Exposure graph
- Why it fails: Superficial security value.
- Fix: Build identity/permission graph, attack path detection, least-privilege recommendations, PIM integration.

## 29) FinOps without real data
- Why it fails: CFOs won’t trust claims.
- Fix: Ingest billing exports/CUR, align with FOCUS, show unit economics, RI/SP planning, verified savings.

## 30) No automated testing
- Why it fails: High regression risk; buyers fear instability.
- Fix: CI unit/e2e/load tests, synthetic probes, test data fixtures, API contract tests.

## 31) No threat model
- Why it fails: Unknown risks; security teams block.
- Fix: STRIDE/LINDDUN per service, track mitigations, gate releases on security reviews.

## 32) Supply chain security unknown
- Why it fails: Vendor reviews will fail.
- Fix: SBOM (Syft), CVE scanning (Grype), signed releases, provenance (SLSA), dependency update policy.

## 33) Operability/docs weak
- Why it fails: Hard to operate; high toil.
- Fix: Helm/Kustomize, runbooks, incident playbooks, golden dashboards, on-call rotations.

## 34) No migration/import story
- Why it fails: High switching costs for prospects.
- Fix: Importers for existing policies/evidence/exceptions, mapping tools, migration dry-runs.

## 35) Pricing/ROI unclear
- Why it fails: Economic buyer can’t justify spend.
- Fix: Pricing tiers, outcome-based options, ROI calculators with case studies and benchmarks.

## 36) Azure lock-in signals
- Why it fails: Multi-cloud buyers churn.
- Fix: Neutral domain model, AWS/GCP parity, cross-cloud demos and case studies.

## 37) Patents ≠ product
- Why it fails: IP doesn’t prove outcomes.
- Fix: Translate patents into shipped, measured features with public benchmarks and customer results.

## 38) UX coherence gaps
- Why it fails: Users get lost; completion time increases.
- Fix: IA review, consistent navigation/drill-down, breadcrumbs, task-oriented flows.

## 39) RBAC not enforced end-to-end
- Why it fails: Dangerous pathways; least privilege broken.
- Fix: Default-deny route guards, ABAC where needed, policy-as-code for roles, audits of permissions.

## 40) Exceptions lifecycle absent
- Why it fails: GRC won’t accept unmanaged exceptions.
- Fix: Time-boxed exceptions with recertification, evidence linkage, automated expiry and alerts.

---

### How to execute
- Convert each “Fix” into an issue with owner, milestone, and acceptance criteria.
- Tag by risk and buyer impact; drive weekly burn-down against this list.
- Keep this file living; close items with links to PRs and release notes.


