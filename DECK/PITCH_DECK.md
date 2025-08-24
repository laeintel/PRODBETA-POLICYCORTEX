## PolicyCortex Pitch Deck

### How to use this deck
- Copy sections into your slide tool (Google Slides, PowerPoint, Pitch) as-is.
- Keep the Mermaid diagram or replace with your own architecture image.
- Replace placeholders like X/Y/A/B with pilot numbers.

## Executive summary
- **Tagline**: AI-driven cloud governance for Azure that turns policy sprawl into real-time, cross-domain decisions.
- **Problem**: Cloud security, compliance, and cost signals live in silos; teams drown in alerts and manual work.
- **Solution**: PolicyCortex correlates identity, network, policy, runtime, and spend to drive automated, auditable actions.
- **Why now**: Azure adoption + AI-native ops; regulators and boards demand provable controls; teams must do more with less.
- **Business value**: Fewer incidents, faster audits, lower spend, higher engineering velocity.

## The problem
- **Fragmentation**: Security, FinOps, and platform teams use different tools that don’t agree.
- **Noise**: High alert-to-action ratio; false positives burn cycles.
- **Manual controls**: Evidence collection and policy remediation are brittle and slow.
- **Risk exposure**: Unenforced least-privilege, drift from IaC, and missing runtime visibility.
- **Audit fatigue**: Controls exist but aren’t testable or provable at audit time.

## The solution (what PolicyCortex does)
- **Cross-domain correlation (patent-backed)**: Links identity, configs, runtime, and spend to surface true risk and impact.
- **Actionable AI**: Summarizes risk, explains cause, and recommends next action; can open PRs or run Terraform changes.
- **Continuous compliance**: Maps evidence to frameworks (SOC2, HIPAA, ISO 27001, CIS, NIST) and tracks control health.
- **Guardrails over gates**: Safe, reversible changes via PRs and Terraform state; policy as code.
- **Enterprise-grade security**: Azure AD login, httpOnly sessions, zero-trust defaults.

## Product overview
- **User personas**
  - CISO / VP Security: risk posture, board reporting, audit readiness
  - Cloud Security Architect: policies, detections, auto-remediation
  - Platform / SRE: drift control, golden paths, change velocity
  - FinOps Lead: cost anomalies, unit economics, chargeback
- **Core capabilities**
  - Asset inventory (Azure resources, identities, policies)
  - Risk analytics (misconfigs, over-permissioned identities, exposed surfaces)
  - Cost governance (budgets, anomalies, waste)
  - Policy as Code & remediation (PRs, Terraform, AKS rollouts)
  - Audit trails & evidence packages
  - GraphQL API and UI for workflows

### High-level architecture
```mermaid
graph LR
  U["Users (AAD)"] --> FE["Next.js Frontend"]
  FE --> GQL["GraphQL Gateway"]
  GQL --> CORE["Rust Core Service"]
  GQL --> PY["Python Services"]
  CORE --> DB[(Postgres)]
  CORE --> REDIS[(Redis)]
  subgraph Azure
    AKS["AKS (Kubernetes)"]
    ACR["Azure Container Registry"]
    KV["Key Vault"]
    AI["App Insights/Logs"]
  end
  FE --> AKS
  CORE --> AKS
  PY --> AKS
  AKS --> AI
  CI["GitHub Actions (CI/CD)"] --> ACR
  ACR --> AKS
  TF["Terraform + OIDC"] --> Azure
```

## What’s already real (repo evidence)
- **Auth enforced before UI**: Route gating via middleware; only login and auth endpoints are public (`frontend/middleware.ts`).
- **Azure AD + MSAL**: Login flow and token acquisition with secure, httpOnly `auth-token` cookie (`frontend/contexts/AuthContext.tsx`, `frontend/app/api/auth/set-cookie/route.ts`).
- **CI/CD**: Security scans, supply chain checks, infrastructure (Terraform OIDC), builds, tests, AKS deploys (`.github/workflows/entry.yml`, `application.yml`, `deploy-aks.yml`, `azure-infra.yml`).
- **Azure-native**: ACR/AKS images, runtime manifests under `k8s/dev` and `k8s/prod`.

## Differentiation (why we win)
- **Cross-domain correlation**: Not another scanner—connects identity, config, runtime, and spend to calculate real risk and business impact.
- **Explainable AI**: Human-readable rationales and remediation steps; integrates with PR workflows for trust and auditability.
- **Azure-first depth**: Native MSAL, ACR/AKS, Terraform OIDC, Azure Policy mapping.
- **Shift-left + guardrails**: PR-based changes, infra state awareness, and rollbacks.
- **Compliance-as-evidence**: Exportable evidence packs mapped to controls, not just dashboards.
- **Performance & safety**: Rust core for critical logic; zero-trust session model.

## Market and ICP
- **ICP**: Mid-market to large Azure-centric orgs (regulated: healthcare, fintech, public sector; complex: multi-subscription/multi-tenant).
- **Use cases that land**
  - Over-permissioned service principals and dangling access
  - Policy drift and audit-readiness for SOC2/ISO/NIST
  - Cost anomalies (unused IPs, idle clusters, zombie storage)
  - AKS misconfigurations (network policies, secrets, RBAC)

## Value and ROI
- **Risk**: Reduce high-severity misconfigs and over-permission by X–Y% in N weeks.
- **Cost**: Cut cloud waste by A–B% via continuous anomaly detection and guardrails.
- **Velocity**: Faster change lead time by enabling safe, PR-based remediations.
- **Audit**: Weeks → days to assemble evidence per control set.

> Replace X/Y/A/B with your pilot results.

## Business model
- **Pricing options** (choose one to test):
  - Per-subscription + per-node (AKS) tiering
  - Per-asset (resource) with volume breaks
  - Platform seat + usage add-ons (AI analyses, evidence packs)
- **Editions**: Starter (read-only), Pro (remediation), Enterprise (SAML/SCIM, custom controls, private SaaS/self-hosted)

## Go-to-market
- **Land**: Risk posture + 3–5 high-value remediations in first month.
- **Expand**: Compliance automation + FinOps, then platform engineering.
- **Channels**: Azure Marketplace private offers; MSSP and GSI partnerships.
- **Proof**: 2–4 week pilot, deploy in customer Azure, measurable outcomes.

## Security, privacy, compliance
- **AuthN**: Azure AD + httpOnly sessions; route gating.
- **Secrets**: Key Vault and GitHub OIDC (no long-lived secrets).
- **Isolation**: Customer data stays in their Azure; images in ACR; workloads in AKS.
- **Compliance**: Map detections to SOC2/ISO 27001/CIS; export evidence packs.

## Roadmap (next 2–3 quarters)
- Deeper Azure Policy graph and drift prevention
- IAM least-privilege recommender (graph-based access reduction)
- FinOps unit economics & anomaly root-cause (service-to-service)
- Control libraries and audit pack generator
- Golden path wizards (RBAC, network, secrets, AKS baselines)
- Partner integrations (Defender for Cloud, Sentinel, Wiz/Prisma, GitHub Advanced Security)

## Competitive landscape
- Microsoft Defender for Cloud, Azure Policy, Wiz, Prisma Cloud, Lacework, Orca
- **Our edge**: Cross-domain correlation + explainable remediation + PR-first change model in an Azure-native stack

## Demo storyline (10 min)
1. Login with Microsoft (MSAL); show blocked UI without auth.
2. Posture dashboard: top risky identities, misconfigs, and costs.
3. Drill: one critical identity/path; explain blast radius and evidence.
4. One-click fix: open PR and Terraform plan; safe rollout to AKS.
5. Evidence pack: export control mapping; share a link.
6. FinOps anomaly: identify cluster waste and demonstrate auto-scheduling policy.

## Slide-by-slide outline
- Problem • Why now
- Solution • Product overview
- Architecture (diagram) • Security model
- Differentiators • Competitive positioning
- ICP & Use cases • Value/ROI
- GTM & Pricing
- Traction (or early pilots) & Case studies
- Roadmap & Vision
- Team & IP (patent coverage)
- Ask & Use of funds

## KPIs to track
- Mean-time-to-remediate policy gaps
- % of auto/remediated changes via PRs
- Alert-to-action ratio
- Compliance control pass rate
- Waste hours/$ saved
- Enterprise expansion (subs, workloads)

## Risks and mitigations
- Buyer fatigue from “another dashboard” → Anchor on PR-based outcomes
- Over-automation risk → Guardrails, simulations, approvals
- Azure tie-in → Expand to multi-cloud with adapters (later)
- Data sensitivity → Keep data in customer tenant, evidentiary minimalism

## Appendix A: Tech deep-dive (stack)
- Frontend: Next.js, MSAL, GraphQL client
- Gateway: Node/GraphQL
- Core: Rust (policy/graph logic)
- Services: Python (analytics, jobs)
- Data: Postgres, Redis
- Infra: AKS, ACR, Key Vault, App Insights, Terraform (OIDC)
- CI/CD: GitHub Actions with security, supply chain, deploy gates

## Appendix B: CI/CD evidence (what runs)
- Runner setup, Docker health checks
- Secret scanning (Gitleaks), license and dependency review
- Supply chain (Trivy), security summary
- Infra: Terraform plan/apply with OIDC
- Build and push images to ACR (core, frontend, graphql)
- Integration tests, then AKS deploy (dev/prod)
- Tag-based production release

## Appendix C: One-liners for sales
- **From chaos to control**: We correlate across identity, config, runtime, and spend so you can act.
- **Confidence to automate**: Explainable recommendations, PR-based changes, full audit trails.
- **Azure-native**: We meet you where you are: MSAL, AKS, ACR, Terraform OIDC.


