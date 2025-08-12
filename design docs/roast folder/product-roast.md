# PolicyCortex Product Roast (Brutally Honest)

Below is a candid critique intended to surface blockers that make enterprise buyers hesitate. Use this to prioritize fixes that unlock sales, reduce risk, and build trust.

## Why buyers will hesitate or walk away (30 items)

1. Unclear core value vs. Azure native: Overlaps Azure Policy/Defender/Purview; lacking an obvious 10x advantage with demos and RFP-ready outcomes.
2. Patent-heavy, proof-light: Four patents, but few third-party benchmarks, case studies, quantified results; buyers want validated outcomes, not IP.
3. Brittle authentication: MSAL/JWT “allow any audience” dev shortcuts undermine trust; enterprises require strict OIDC/AAD/SAML + aud/iss validation and auditability.
4. Inconsistent API contracts: Mixed response shapes (flat vs. `data` envelope) signal immaturity and raise integration risk and maintenance cost.
5. GraphQL CSRF issues: Apollo CSRF prevention failures indicate security design gaps; a red flag for security-conscious orgs.
6. Dev/Prod parity gaps: Tailwind purge, header preflight, and env rewrites fail in prod; fear that “it works only in dev.”
7. Configuration sprawl: Many env vars across services without a typed schema or single source of truth; onboarding is fragile.
8. IaC reliability: Terraform backend/auth required workarounds and fallbacks; perceived deployment instability for scaled rollouts.
9. Secrets management: Application secrets via raw envs instead of Key Vault with RBAC and rotation; fails enterprise expectations.
10. Observability gaps: No cohesive tracing/logging/metrics with correlation IDs and SLOs; incident resolution and ops readiness are unclear.
11. Multi-tenancy ambiguity: Tenant isolation, data residency, and per-tenant RBAC segregation are not formalized.
12. Security posture unproven: No published threat model, pen test reports, or compliance artifacts (SOC2/ISO); risk-averse buyers stall.
13. Auto-remediation safety: Limited approvals/guardrails/blast-radius analysis/rollback; fears of unplanned outages.
14. “AI domain expert” claims: Lacks reproducible accuracy metrics, drift monitoring, model governance, and explainability; compliance risk.
15. Azure-only scope: No AWS/GCP support; multi-cloud/hybrid buyers will deprioritize.
16. UI polish over usability: Visual depth improved, but task flows, accessibility, and operator UX need work.
17. RBAC granularity: Claims fine-grained RBAC but lacks policy-as-code for roles, JIT elevation, and PAM integration.
18. Policy-as-code missing: No first-class GitOps for policies, drift diffs, approvals, rollbacks; this is table stakes.
19. Data/graph depth limited: “Cross-domain correlation” lacks transparent resource graph, lineage, and historical snapshots.
20. Real-time pipeline thin: One global SSE; enterprises expect scalable streaming/eventing (Event Grid/Kafka) with authz and backpressure.
21. FinOps features shallow: No FOCUS adoption, precise anomaly metrics, or verifiable savings; weak for CFO stakeholders.
22. Reference architectures absent: No blueprints for LZ/hub-spoke/zero-trust or policy baselines; orgs want “known-good” recipes.
23. Marketplace and procurement: Not on Azure Marketplace; no transactable offers or standard legal terms; sales friction.
24. Pricing opacity: No transparent pricing, unit economics, or ROI case studies; pilots stall without predictable cost/benefit.
25. Change management: No migration guides from Azure Policy/Defender/Purview; lacks coexistence patterns; buyers fear overlap chaos.
26. Extensibility gaps: Limited webhooks/integrations; no public plugin/connector story; ops ecosystems (Jira/ServiceNow/Splunk) underserved.
27. Compliance frameworks thin: Missing control mapping, evidence collection workflows, auditor-ready artifacts.
28. Offline/sovereign limitations: No support for sovereign/air-gapped scenarios; public-sector blockers.
29. Performance claims unproven: “Sub-millisecond” Rust claims without published workloads/scale tests/concurrency guarantees.
30. Support/readiness: No SLAs, LTS, or break-glass support channels; enterprises won’t entrust Tier-1 workloads.

---

Use this list to drive a remediation roadmap (security/compliance first, then DX/operability, then differentiated outcomes).