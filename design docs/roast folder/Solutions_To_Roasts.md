# Solutions to the 50 Roasts (Remediation Plan)

This is a prioritized, actionable plan to remediate each adoption blocker. Sequenced for fastest enterprise viability.

## Living Status Checklist

Legend: [Full] implemented end-to-end; [Partial] exists but incomplete; [Missing] not implemented.

1) Reproducible AI — [Partial]
- Pointers: `core/src/ai/model_registry.rs` (model cards, eval harness); `training/` scaffolding
- Next: add datasets/benchmarks + CI eval runs; publish model cards

2) Real Multi‑Cloud — [Partial]
- Pointers: `core/src/collectors/aws_collector.rs` (stub); Azure clients across core/backend
- Next: implement AWS/GCP collectors + normalized schema; policy translation

3) Truthful Data Modes — [Partial]
- Pointers: `core/src/data_mode.rs`; gateway forces real Azure; core falls back in `core/src/api/mod.rs`
- Next: surface source banner in FE; block writes when simulated
  - Update: Demo/Simulated data banner added in `frontend/components/DemoModeBanner.tsx` and mounted globally in `frontend/components/AppLayout.tsx`.

4) Harden Local Runtime — [Partial]
- Pointers: `docker-compose*.yml`, `bootstrap.*`
- Next: devcontainer, preflight checks; reduce manual port rewrites

5) Env‑Driven Routing — [Partial]
- Pointers: `frontend/middleware.ts`, `frontend/next.config.docker.js`; hardcoded rewrites in `frontend/next.config.js`
- Next: env-driven discovery for all routes; remove hardcodes

6) Unified AuthN/AuthZ — [Partial]
- Pointers: FE MSAL `frontend/lib/auth-config.ts`; BE JWT validation `core/src/auth.rs`
- Next: enforce scopes/roles on handlers; propagate tenant context

7) Tenant Isolation — [Partial]
- Pointers: `scripts/migrations/001_add_tenant_isolation.sql` (tenant_id + RLS), `core/src/tenant.rs` (middleware/DB context)
- Next: wire middleware in router; migrate all queries to tenant-aware access

8) Durable SOR — [Partial]
- Pointers: Python actions persisted (`backend/services/api_gateway/main.py`); rich DB schema `scripts/init.sql`
- Next: move core actions/events to DB; add migrations/backups

9) Production Orchestrator — [Partial]
- Pointers: in-memory lifecycle + SSE `core/src/api/mod.rs`, gateway
- Next: idempotency keys/retries/compensation/rollback workflows

10) Honest UX — [Partial]
- Pointers: Action drawer tabs (preflight/approvals/evidence) in `frontend/components/ActionDrawer`
- Next: confirmations, disable destructive ops without approvals in UI
  - Update: Focus management, scroll lock, and topmost z-index applied to Action Drawer to ensure visibility and usability.

11) Approvals & SoD — [Partial]
- Pointers: `core/src/approvals.rs` (models/policies), `Roadmap_17_Approval_and_Rollback_Sequences.md`
- Next: add approve/rollback endpoints and integrate with action lifecycle

12) Evidence Factory — [Partial]
- Pointers: `core/src/compliance/mod.rs`, API in `core/src/api/compliance.rs`
- Next: immutable storage (WORM), signatures, auditor exports

13) Tamper‑Evident Logs — [Partial]
- Pointers: `core/src/audit_chain.rs` (hash chaining, WORM insert)
- Next: emit audit entries from auth/actions/compliance; verify chain on startup

14) Secrets Lifecycle — [Partial]
- Pointers: Key Vault hydration in gateway; Terraform KV
- Next: remove plaintext env secrets; rotation + workload identity

15) Secret Boundary Checks — [Missing]
- Next: log redaction, CI checks preventing secret bundling

16) Accessibility — [Partial]
- Pointers: `frontend/lib/accessibility.ts` (focus trap, ARIA helpers)
- Next: adopt across components; add automated a11y tests

17) i18n — [Partial]
- Pointers: `frontend/lib/i18n.ts` (provider, translations)
- Next: wrap app provider; migrate strings; load locales

18) UI Scale — [Partial]
- Pointers: `frontend/lib/performance-api.ts` (caching/backpressure/streaming)
- Next: virtualization and server pagination in large lists

19) Search/Filters — [Partial]
- Pointers: `frontend/app/policies/page.tsx` simple filters
- Next: faceted filters, saved views, query builder

20) Offline/Conflicts — [Missing]
- Next: PWA offline queues, optimistic concurrency (ETags)

21) Observability — [Partial]
- Pointers: docs `advance docs/15-monitoring-observability.md`; tracing init in core
- Next: wire OpenTelemetry exporter, dashboards, SLIs

22) SLOs & Error Budgets — [Partial]
- Pointers: `core/src/slo.rs` (SLO manager, error budget)
- Next: wire SLIs to metrics/exporter; alerting and release gates

23) DR Strategy — [Partial]
- Pointers: Terraform backups/retention
- Next: restore drills, geo-redundancy, RPO/RTO docs

24) Data Retention — [Partial]
- Pointers: storage retention in Terraform
- Next: app-level TTLs, legal holds

25) Control Framework Mapping — [Partial]
- Pointers: models in `core/src/compliance/mod.rs`
- Next: SOC2/ISO/NIST mapping + evidence linking

26) Policy Engine Depth — [Partial]
- Pointers: `core/src/policy/evaluation_engine.rs`
- Next: assignment graph/inheritance/effect parity

27) Enforcement Path — [Partial]
- Pointers: `backend/services/azure_active_governance.py` (not wired)
- Next: connect orchestrator to Azure Policy/Terraform/Bicep

28) IAM Graph — [Partial]
- Pointers: `core/src/security_graph/mod.rs`
- Next: integrate with detections/approvals; mitigation apply paths

29) FinOps Foundations — [Partial]
- Pointers: `core/src/finops/mod.rs`
- Next: billing ingestion (CUR), FOCUS alignment, unit economics

30) Testing — [Partial]
- Pointers: few unit tests; strategy doc
- Next: CI unit/e2e/load suites; smoke tests

31) Threat Modeling — [Partial]
- Pointers: `advance docs/07-security-architecture.md`
- Next: track mitigations, gate releases on reviews

32) Supply Chain Security — [Partial]
- Pointers: `scripts/supply-chain-security.sh`, `.bat` (SBOM/CVE)
- Next: integrate in CI; add provenance/signing policy gates

33) Operability — [Partial]
- Pointers: compose + Terraform; docs
- Next: Helm/Kustomize, runbooks, playbooks

34) Migration/Import — [Partial]
- Pointers: `Roadmap_19_Postgres_Migrations_and_Seeds.md`, `scripts/init.sql`
- Next: wire migration tool (Flyway/SQLx), importers

35) Pricing & ROI — [Partial]
- Pointers: ROI-like savings in FinOps
- Next: pricing tiers, ROI calculators

36) True Multi‑Cloud Messaging — [Partial]
- Pointers: AWS stub collector; multi‑cloud KB
- Next: parity features + case studies

37) From IP to Product — [Partial]
- Pointers: many endpoints implemented
- Next: benchmarks and outcome KPIs

38) Documentation — [Full]
- Pointers: extensive `design docs/` and roadmaps
- Next: keep current with releases

39) Extension Model — [Partial]
- Pointers: event bus contracts `core/src/events/mod.rs`
- Next: plugin SDK, stable APIs, examples

40) Schema Versioning — [Partial]
- Pointers: roadmap docs
- Next: versioning policy + CI checks

41) Event‑Driven Architecture — [Partial]
- Pointers: NATS impl `core/src/events/mod.rs`
- Next: publish/subscribe from action/compliance/cost flows

42) Performance Engineering — [Partial]
- Pointers: perf client; basic scripts
- Next: load/soak/chaos benchmarks and budgets

43) Privacy/Residency — [Missing/Partial]
- Pointers: docs mention; no code controls
- Next: DPAs, data flows, residency enforcement

44) UX Coherence — [Partial]
- Pointers: `frontend/components/AppLayout.tsx`
- Next: IA review, consistent drill‑down patterns

45) RBAC Enforcement — [Partial]
- Pointers: `core/src/auth.rs`
- Next: apply route guards everywhere, ABAC policies

46) Exceptions Lifecycle — [Partial]
- Pointers: stubs in `core/src/api/mod.rs` and gateway
- Next: expiry/recertification workflows

47) Change Management — [Partial]
- Pointers: `core/src/change_management.rs` (ServiceNow/Jira integrations)
- Next: invoke during high‑risk actions; freeze window checks

48) References & Community — [Missing]
- Next: add design partners, references, community space

49) Positioning & Messaging — [Missing]
- Next: focused messaging with proof points

50) Culture of Evidence — [Partial]
- Pointers: evidence models, audit tables in `scripts/init.sql`
- Next: signed artifacts, end‑to‑end attestations

1) Reproducible AI
- Deliver versioned models with training data lineage, eval harnesses, and benchmarks. Publish model cards and outcomes.

2) Real Multi‑Cloud
- Implement AWS/GCP collectors, normalize schema, and policy translation. Start with read‑only, then enforcement parity.

3) Truthful Data Modes
- Add prominent “Simulated vs Real” indicators; feature flag forced‑real for production. Block risky flows on simulated mode.

4) Harden Local Runtime
- One‑click bootstrap scripts (Win/Mac/Linux), preflight checks, and containerized dev with devcontainer/compose.

5) Env‑Driven Routing
- Replace hardcoded rewrites with config/env discovery. Add service registry or gateway.

6) Unified AuthN/AuthZ
- Standardize JWT format, scopes, and claim handling. Enforce at every API boundary. Include token rotation.

7) Tenant Isolation
- Add tenant_id to all tables, queries, and caches. Enforce row‑level policies and context propagation.

8) Durable SOR
- Postgres for actions, exceptions, violations, and audit. Versioned schemas with migrations and backups.

9) Production Orchestrator
- Introduce idempotency keys, retries, backoff, saga/compensation patterns, and state machines.

10) Honest UX
- Show preflight, impact analysis, and reversibility. Require confirmations and approvals as needed.

11) Approvals & SoD
- Configurable multi‑stage approvals, SoD rules, protected envs, and emergency break‑glass with recording.

12) Evidence Factory
- Signed evidence capture, immutable storage (WORM), and auditor‑grade reports with control mapping.

13) Tamper‑Evident Logs
- Use append‑only, hashed, chained logs (e.g., immudb/OPA bundles) with signatures and attestations.

14) Secrets Lifecycle
- Move secrets to Key Vault (and AWS/GCP equivalents), enable rotation, and use workload identity.

15) Secret Boundary Checks
- Redact sensitive values in logs, add static checks to block bundling keys to clients.

16) Accessibility
- Implement WCAG 2.1 AA: focus traps, ARIA roles, contrast, keyboard nav; add automated a11y tests.

17) i18n
- Localize strings; use ICU message format; load language packs per tenant.

18) UI Scale
- Virtualized tables, server‑side paging, cursor‑based pagination, query caching.

19) Search/Filters
- Faceted search with saved views and scoped permissions; query builder with presets.

20) Offline/Conflicts
- Optimistic concurrency control (ETags), offline queues with reconciliation, conflict UI.

21) Observability
- OpenTelemetry tracing + metrics + logs across all services; dashboards for RED/USE and business KPIs.

22) SLOs & Error Budgets
- Define SLOs per endpoint; alert on SLI breaches; error budget policies for release gating.

23) DR Strategy
- Backups, PITR, cross‑region replicas; document RPO/RTO and run restore drills quarterly.

24) Data Retention
- Configurable retention/TTL, legal holds, and purge flows with audit.

25) Control Framework Mapping
- Map features to SOC2/ISO/NIST; export auditor packages with evidence links and control status.

26) Policy Engine Depth
- Build assignment graph, inheritance, parameter resolution, and effect semantics; parity with Azure/GCP/AWS.

27) Enforcement Path
- Integrate with Azure Policy/Bicep/Terraform and GCP/AWS equivalents; drift detection + auto‑remediation.

28) IAM Graph
- Build identity/privilege graph, detect attack paths, and recommend least‑privilege fixes; support PIM.

29) FinOps Foundations
- Ingest CURs/billing exports, model unit economics, align with FOCUS standard, show RI/SP planning.

30) Testing
- CI with unit/e2e/load tests; synthetic probes; test data fixtures; contract tests for APIs.

31) Threat Modeling
- STRIDE/LINDDUN per service; mitigations tracked; security reviews on each release.

32) Supply Chain Security
- SBOM generation (Syft), CVE scanning (Grype), provenance (SLSA), and dependency update policies.

33) Operability
- Helm charts, Kustomize overlays, runbooks, incident playbooks, and upgrade guides.

34) Migration/Import
- Importers for policies/evidence/exceptions; mapping tools; migration dry‑runs.

35) Pricing & ROI
- Publish pricing tiers, usage‑based options, ROI calculators based on savings and risk reduction.

36) True Multi‑Cloud Messaging
- Neutral domain model; parity features across clouds; case studies demonstrating cross‑cloud value.

37) From IP to Product
- Translate patents into shipped, measured features with benchmarks and case studies.

38) Documentation
- Developer/operator docs, architecture diagrams, SLAs, FAQs; in‑product help and tooltips.

39) Extension Model
- Plugin framework with events and stable APIs; marketplace for actions, collectors, and policies.

40) Schema Versioning
- Semantic versioning of APIs/schemas; migration guides; backward‑compat layers.

41) Event‑Driven Architecture
- Introduce event bus (Kafka/Service Bus), define contracts, publish domain events; build consumers.

42) Performance Engineering
- Load and soak testing; performance budgets; profiling; auto‑scaling policies.

43) Privacy/Residency
- Data flow maps, DPAs, field‑level controls, residency options per tenant/region.

44) UX Coherence
- Information architecture pass, task‑oriented navigation, consistent drill‑downs and breadcrumbs.

45) RBAC Enforcement
- Policy‑based access control with fine‑grained permissions; ABAC where needed; default‑deny.

46) Exceptions Lifecycle
- Time‑boxed exceptions with re‑certification, evidence linking, and automated expiry.

47) Change Management
- Integrate with CAB/RFC tools (ServiceNow/JSM); freeze windows; change calendars; approval gates.

48) References & Community
- Design partners, lighthouse customers, public references, and community Slack/forum.

49) Positioning & Messaging
- Narrow to high‑value use cases per buyer persona; measurable outcomes and proof points.

50) Culture of Evidence
- Replace claims with proof: metrics, tests, audits, and case studies. Make trust earned and verifiable.
