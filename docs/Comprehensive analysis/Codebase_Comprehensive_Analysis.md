# Comprehensive Codebase Analysis

Date: 2025-08-13

## Executive Summary

This document captures the key technical shortcomings, bottlenecks, and architectural gaps discovered in the **PolicyCortex** repository as of the most recent commit on `main`.  The goal is to give engineering, product, and leadership a clear snapshot of what still "sucks" for an MVP-grade live demo and what must be addressed to reach production-grade quality.

---

## Primary Problems (High-Level)

1. **Shallow Domain Functionality** – Many features are stubbed or mocked (e.g., evidence generation, policy graph, remediation engine) which limits demo depth.
2. **DX & Build Instability** – Docker/Rust builds require work-arounds (SQLX_OFFLINE, Alpine OpenSSL fixes) and often break on new crates.
3. **Observability Gaps** – OTLP plumbing exists but metrics/traces/logs are not exported in dev or prod; no SLO dashboards.
4. **Thin Test Coverage** – Unit and integration tests cover < 10 % of code paths; security-critical flows (auth, tenant isolation) untested.
5. **Security Baseline Drift** – 5 CRITICAL CVEs tolerated; baseline file may rot without auto-ratcheting.
6. **Frontend Performance** – No code-splitting, bundle > 1 MB, no list virtualisation or SSR caching.
7. **Terraform Drift** – Container App revision FQDNs cause plan churn; remote-state locking not configured.
8. **Secret Hygiene** – Push protection flagged embedded secrets; large binaries (MSI) also present in history.
9. **Multi-Cloud Claims vs Reality** – Collectors & ARM-specific code show Azure bias; AWS/GCP unsupported.
10. **Product Differentiation** – Core value props (continuous evidence, auto-remediation) not fully implemented, leaving room for competitors.

---

## Bottlenecks & "Doing Too Much"

| Area | Observation | Impact |
|------|-------------|--------|
| **Rust Modular Workspace** | Now using workspace with `crates/{api,auth,evidence,orchestration,shared,models}`. | Better compile times and separation; add per-crate CI/cache next. |
| **CI Matrix Explosion** | Path-based filters and conditional builds now active; still relies on Docker on self-hosted runner. | Faster feedback when areas unchanged; ensure runner has Docker to avoid skips. |
| **Terraform Apply in CI** | Plan on PR; Apply on main push (dev). Azure infra workflow supports manual apply. | Lower cost/drift; add remote state locking and apply approvals. |
| **Demo Mode Toggle** | `DemoModeBanner`, `MockDataIndicator`, `useMockDataStatus` exist; not yet centralized provider. | Partially improved; create `DemoDataProvider` to eliminate scattering. |
| **SBOM+SLSA+Trivy** scans | Baseline gating in CI; application workflow runs parallel security jobs; supply-chain script remains sequential. | Duration reduced; consider full parallelization and caching across jobs. |

---

## Top 30 Recommendations & Fixes

1. **Modularise Rust** – Split `core` into crates: `api`, `auth`, `evidence`, `remediation`, `db`.  Use Cargo workspaces.
2. **Introduce Feature Flags** – Compile optional SQLx, OpenTelemetry, Remediation logic only when needed.
3. **Adopt Buildkit Caching** – Layer caching for cargo builds to cut container build time ~60 %.
4. **Ratchet CVE Baseline** – Weekly job lowers `critical_allowance` automatically when CVEs drop.
5. **Enable Dependabot-Security Updates** – Auto PRs for vulnerable crates & npm modules.
6. **E2E Cypress Suite** – Cover auth login, tenant isolation, policy assignment flows.
7. **Add Migrations** – Use `sqlx migrate` and a `migrations/` directory; run on boot.
8. **Provision Dev DB** – Spin up Postgres in Container App or Azure DB for demo realism.
9. **Implement OpenTelemetry Export** – Send traces to `otelcol` ➜ Azure Monitor; create Grafana dashboards.
10. **Server-Side Rendering Cache** – Use Next.js `getServerSideProps` + `swr` cache for policy lists.
11. **Code-Split Frontend** – Dynamic import heavy pages; enable React 18 streaming.
12. **List Virtualisation** – Use `react-virtualized` for evidence tables > 1k rows.
13. **Image Optimisation** – Enable `next/image` and AVIF in CI.
14. **Remove Dead Assets** – Delete MSI & other large binaries; use Git LFS if truly needed.
15. **Secret Scanning Gate** – Block PR merge when secrets detected; migrate secrets to Key Vault.
16. **Readonly Service Principals** – Least privilege for collectors; Terraform uses Federated OIDC.
17. **SBOM Publishing** – Upload CycloneDX JSON to GH Releases; sign with cosign.
18. **SLSA Level 2** – Enable provenance attestation on GH Actions with Sigstore.
19. **Automated Canary Deploy** – Blue/green revision before dev CA flip to reduce downtime.
20. **Async Evidence Pipeline** – Use Azure Service Bus + worker app instead of in-request processing.
21. **Policy Graph DB** – Model resources & policy edges in Neo4j or Cosmos Gremlin.
22. **Fine-Grained RBAC** – Use `opa` or `cedar` policy engine; centralise authz.
23. **Config-as-Code** – Store policies in Git, sync via Flux/Kustomize.
24. **Client-Side Analytics Opt-In** – Add telemetry banner and anonymised analytics toggle.
25. **Accessibility Pass** – Run `axe-core` audit; fix missing ARIA labels & contrast.
26. **i18n Setup** – `next-i18next` scaffolding for English + one extra locale; show roadmap.
27. **FinOps Dashboard** – Basic cost charts using Azure Cost Management API.
28. **Threat Model Markdown** – Create `docs/threat-model.md` with STRIDE table; track mitigations.
29. **Runbooks** – Add `runbooks/alerts.md` linked from Grafana.
30. **Investor Demo Script** – Single script `make demo` spins infra, seeds data, opens browser.

---

## Conclusion

The project is now demo-ready for a **thin-slice** of the envisioned feature set, but significant architectural, security, and performance gaps remain.  The 30 items above provide a concrete path to evolve from MVP to a defensible, production-grade platform.
