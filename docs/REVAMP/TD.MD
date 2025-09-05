1) Where your codebase stands (post‑rebuild readiness)

Run these quick, non‑destructive checks to grade the rebuild in under 20 minutes. They will surface real vs mock, UI density, routing, observability, policy wiring, and predictive engine readiness.

A. Real data (no silent mocks)

What to verify

Real‑mode flags are default true in production: NEXT_PUBLIC_DEMO_MODE=false, USE_REAL_DATA=true.

In “real”, API handlers do not fall back to canned JSON; they return 503 with a helpful configuration banner in the UI.

GraphQL mocks are off in real mode.

How to check (copy/paste)

# find any lingering demo/mocks
rg -n --hidden "(MOCK|SIMULAT|DEMO_MODE|USE_MOCK)" frontend/ core/

# run core in real mode and curl top endpoints (should be 200 or explicit 503)
export USE_REAL_DATA=true
curl -i http://localhost:8080/api/v1/executive/roi
curl -i http://localhost:8080/api/v1/predictions
curl -i http://localhost:8080/api/v1/blockchain/verify


Pass criteria

No unexpected MOCK_* in production bundles.

503s include clear “how to configure” text (Azure/DB/Predictions_URL).

B. UI density & responsiveness (“Splunk‑grade”)

What to verify

Every page uses a bounded container (max-w-screen-2xl).

No w-[…] / min-w-[…] on content cards; use w-full min-w-0.

Headings/KPI use fluid tokens (no billboard text).

Chip/toolbars wrap (flex-wrap), not whitespace-nowrap.

How to check

rg -n "w-\[|min-w-\[|text-\[" frontend/
rg -n "max-w-screen-2xl" frontend/


Pass criteria

≤ 5 hits of w-[…] (only for icons/badges), ≥ 5 pages wrapped in a container.

Playwright “no horizontal scroll” test passes at 1366×768.

C. Navigation & IA

What to verify

Left‑nav order: Executive, Policy, Audit Trail, Predict, FinOps & ROI, Access Governance, Resources, DevSecOps, Settings.

Labs items grouped, noindex, badge visible.

How to check

rg -n "Executive|Policy|Audit Trail|Predict|FinOps" frontend/components frontend/app


Pass criteria

Routes match the order above; Labs collapsed; public‑site footer removed inside app.

D. Policy enforcement + evidence

What to verify

Policy Hub shows: Coverage, Predicted Drift (7‑day), Prevented (30/90d), $ Impact.

CI/CD gates connected; “Create Fix PR” works.

Audit Trail uses /api/v1/blockchain/audit & /verify, row‑level integrity chips render.

How to check

curl -s http://localhost:8080/api/v1/blockchain/audit | jq '.[0]'
rg -n "/api/v1/blockchain/(audit|verify)" frontend/


Pass criteria

Audit list non‑empty; Verify chips have OK/FAIL states; Evidence export link works.

E. Predictions + ROI

What to verify

Predictions page calls /api/v1/predictions (real ML if PREDICTIONS_URL set; else explicit 503 in real mode).

Executive & FinOps show ROI with savings now + 90‑day forecast.

How to check

rg -n "/api/v1/predictions|/api/v1/executive/roi" frontend/
curl -i http://localhost:8080/api/v1/predictions
curl -i http://localhost:8080/api/v1/executive/roi


Pass criteria

Predictions render list (or clear “configure ML endpoint” banner).

ROI cards render numbers; empty states are explicit, not blank.

F. Observability & health

What to verify

/healthz confirms DB, Azure token, PolicyInsights, Cost API.

Logs are structured (JSON) with request_id, Azure API error codes.

A single error banner component standardizes user‑facing config hints.

How to check

curl -s http://localhost:8080/healthz | jq
rg -n "tracing|request_id|StatusCode" core/
rg -n "Needs configuration" frontend/


Pass criteria

healthz includes sub‑checks; frontend uses one shared ErrorBanner.

2) Exactly where to improve next (code‑level backlog)

Assuming the rebuilt app still resembles your earlier structure, here’s the surgical backlog (no rewrites, only reversible patches):

Fail‑fast in real mode
Replace any “nice mock” with 503+hint (compliance, resources, threats, predictions).
Outcome: fake data cannot leak to prod; gaps surface fast.

Executive & Policy as the front door
Default route → /executive. Policy Hub: Coverage, Predicted Drift, Prevented, $ Impact with CTAs (New Policy, Install Pack, Run Simulation, Export Evidence).

Audit Trail: live + verifiable
Row‑level IntegrityChip hits /blockchain/verify?hash=…; “Verify Chain” banner; Export Signed Evidence JSON (hash chain, Merkle root, signature, timestamp).

Predictions list + “Fix PR”
Show ETA, confidence, top causes; one‑click PR template for CI gate patches.

FinOps ROI
/executive/roi: savings to date, forecast 90d, anomalies. Wire to cost API; what‑if simulator flagged as Labs until ML is ready.

Splunk‑grade density
Add PageContainer, fluid type tokens; remove w-[…]/min-w-[…] on content; flex-wrap chips; tabular-nums on KPIs.

E2E smoke (blocks regressions)
Playwright: no horizontal scroll, Executive landing, Audit verify chip, Predictions render + Fix PR, ROI render + error banner on 503.

3) Competitor gaps you can exploit (do these better than anyone)

Below I cite vendor pages to keep this grounded. The pattern is consistent: CNAPPs excel at reactive risk + compliance; FinOps tools excel at spend; governance‑as‑code tools enforce rules. Very few unify predictive governance + ROI + tamper‑evident evidence.

Competitor	Strengths (short)	Gaps you can own (short)
Prisma Cloud (PANW)	Broad CNAPP: CSPM/IaC/guardrails; 75+ frameworks; one‑click audits. 
Palo Alto Networks
+1
	Predictive look‑ahead (time‑based), FinOps ROI wiring, immutable evidence (no public claims).
InsightCloudSec (Rapid7)	Real‑time risk, no‑code automation, CIEM, IaC; multi‑cloud. 
Rapid7
	CFO‑grade Governance P&L; blockchain evidence; deep cost+policy coupling.
Wiz	Agentless CNAPP + security graph; strong compliance catalogs. 
EM360Tech
	FinOps ROI; time‑ahead policy drift forecasts (no explicit public focus).
Orca Security	Agentless SideScanning; unified CNAPP with attack path analysis; CI/CD checks. 
Orca Security
+2
Orca Security
+2
	FinOps ROI; tamper‑evident audit; forecasted policy drift with confidence.
Stacklet (Custodian)	Governance‑as‑code + FinOps guardrails & auto remediation. 
Stacklet
+1
	ML prediction; CFO dashboard; auditor‑ready chain.
Kion	FinOps + Compliance + Identity in one multicloud governance plane (budget enforcements, savings, forecast). 
Kion
+1
	Predictive drift, cryptographic evidence, cross‑tool ROI correlation.
CloudZero / Finout	Deep cost intelligence; Megabill (cloud+SaaS). 
CloudZero
docs.finout.io
	Compliance/security coupling; predictive compliance→$ impact.

Why this matters:

Prisma/InsightCloudSec/Wiz/Orca pages highlight visibility, compliance, guardrails, attack paths — not time‑ahead forecasting or CFO‑visible ROI. 
Palo Alto Networks
+1
Rapid7
Orca Security

Stacklet/Kion do FinOps+governance, but their public materials emphasize rule enforcement & tagging over ML predictions and immutable evidence. 
Stacklet
+1
Kion

FinOps leaders (CloudZero/Finout) publish cost analytics, not compliance enforcement. 
CloudZero
docs.finout.io

Your 3 moats to lean into (and ship well)

Predictive Governance: per‑policy Forecast Card (what, why, when, confidence).

Governance P&L: every policy shows $ saved + risk avoided; board‑ready report.

Tamper‑evident Evidence: hash‑chained events + “Verify Chain” + auditor portal.

4) Concrete “build & prove” plan (90 days)

Weeks 0–2 — Make real mode undeniable

Guard every endpoint with the real‑mode fail‑fast pattern.

Replace any remaining GraphQL mock with 404 in real.

Health endpoint: DB + Azure PolicyInsights + Cost checks return green or actionable red.

Weeks 3–6 — Ship the moat

Forecast Card v1: drift predictions for top 10 controls; confidence bands.

Governance P&L v1: show $ impact by policy + 90‑day forecast on Executive & FinOps.

Audit Mode v1: per‑row verify chip; chain banner; JSON export.

Weeks 7–10 — CI/CD & ROI story

GitHub/Azure DevOps app: PR comment with “Apply policy fix” link.

Board Report: 1‑click PDF with savings, risks avoided, evidence hashes.

Integrate 1 CNAPP (Wiz/Prisma/Orca) as a signal source (reduce their alert volume by X%).

Weeks 11–13 — Design partners (prove)

Healthcare + FinServ + Public sector pilots.

Targets: MTTP < 24h, Prevention rate ≥ 35% auto‑fix, Savings 8–12% in 90 days.

5) Drop‑in artifacts your agent can apply now
A. E2E smoke (Playwright)
import { test, expect } from '@playwright/test'

test('Executive landing + no horizontal scroll', async ({ page }) => {
  await page.goto('http://localhost:3000/')
  await expect(page).toHaveURL(/\/executive/)
  await page.evaluate(() => window.scrollBy(10000,0))
  expect(await page.evaluate(() => window.scrollX)).toBe(0)
})

test('Audit verify visible', async ({ page }) => {
  await page.goto('/audit')
  await expect(page.getByText(/Chain integrity:/)).toBeVisible()
  await expect(page.getByText(/Integrity (OK|FAIL|Checking)/).first()).toBeVisible()
})

test('Predictions render + Fix PR', async ({ page }) => {
  await page.goto('/ai/predictions')
  await expect(page.getByRole('heading', { name: /Predictions/i })).toBeVisible()
  await expect(page.getByRole('link', { name: /Create Fix PR/i }).first()).toBeVisible()
})

test('ROI shows values or helpful error', async ({ page }) => {
  await page.goto('/finops')
  await expect(page.getByText(/Savings this Quarter|Needs configuration/)).toBeVisible()
})

B. Real‑mode guard (Rust pattern)
let real = std::env::var("USE_REAL_DATA").unwrap_or_default() == "true";
if real {
  match call_provider().await {
    Ok(v) => return Ok(Json(v)),
    Err(e) => { tracing::error!("provider err: {e}"); return Err(StatusCode::SERVICE_UNAVAILABLE); }
  }
}
return Ok(Json(simulated_payload()));

C. UI container & type tokens (Tailwind)
:root{
  --font-body: clamp(0.95rem,0.25vw+0.9rem,1rem);
  --font-title: clamp(1.25rem,1vw+1rem,1.75rem);
  --font-kpi: clamp(1.25rem,0.8vw+1.1rem,1.75rem);
  --font-kpi-label: clamp(0.75rem,0.2vw+0.7rem,0.875rem);
}

// components/PageContainer.tsx
export default ({children}:{children:React.ReactNode}) =>
  <div className="mx-auto max-w-screen-2xl px-4 sm:px-6 lg:px-8">{children}</div>

6) “What they don’t do well” → your crisp differentiators

Time‑ahead prevention vs. reactive dashboards
CNAPP pages emphasize discovering, prioritizing, and remediating current risks (visibility, guardrails, attack paths) — not forecasting policy drift with confidence windows. 
Palo Alto Networks
Rapid7
Orca Security

→ Own Forecast Cards (per‑policy), MTTP SLO, and “prevented” counters.

$‑tied governance vs. security‑only
Prisma/Orca/Wiz call out compliance reporting and risk prioritization; Kion/Stacklet cover FinOps+guardrails; but CFO‑grade Governance P&L (policy→$$) is not their headline. 
Palo Alto Networks
Orca Security
Stacklet
Kion

→ Own ROI wiring: every control shows savings and risk‑to‑money.

Cryptographically provable evidence
Competitor materials tout “audit‑ready reports,” but do not claim blockchain‑anchored evidence chains. 
Palo Alto Networks

→ Own immutable evidence with chain verify + auditor portal.

Consolidation without lock‑in
Position as the governance brain that sits above CNAPPs/FinOps tools (ingest, correlate, prevent, prove). Keep integrations first‑class (Wiz/Prisma/Orca signals; Finout/CloudZero costs) so you reduce tool fatigue rather than replace everything. 
Orca Security
CloudZero

7) Acceptance metrics (to know you’re winning)

MTTP (Mean Time To Prevention) < 24h (prediction → merged fix).

Prevention rate ≥ 35% auto‑fixed; ≥ 70% with guided PR in 7 days.

Governance‑linked savings 8–12% in 90 days.

Audit readiness: “Days to Audit‑Ready” trending down; Evidence coverage ≥ 95%.

Time‑to‑value: first Forecast Card + $ impact ≤ 15 minutes after connect.

Citations (what I used to benchmark competitors)

Prisma Cloud emphasizes multi‑cloud compliance guardrails, asset visibility, and IaC scanning (reactive enforcement, “one‑click” audit reports). 
Palo Alto Networks
+1

Rapid7 InsightCloudSec focuses on real‑time risk, CIEM, IaC security, and no‑code compliance automation (security posture first). 
Rapid7

Orca Security positions agentless CNAPP, SideScanning, and attack‑path analysis; strong risk discovery and CI/CD checks (no public predictive horizon). 
Orca Security
+2
Orca Security
+2

Stacklet (Custodian) markets FinOps governance guardrails and automated remediation (policy‑as‑code, cost focus). 
Stacklet
+1

Kion delivers FinOps budgets/enforcements and compliance automation in a single platform (governance plane), but not predictive ML/immutable evidence in public materials. 
Kion
+1

CloudZero and Finout emphasize cost intelligence and a unified bill (Megabill) — FinOps depth, minimal compliance enforcement. 
CloudZero
docs.finout.io



Where we are (today)

Audit Trail → live & provable. UI now calls /api/v1/blockchain/audit and /api/v1/blockchain/verify, adds the IntegrityChip, chain-verify banner, and Export Evidence action. 

Merge PR-03: Audit Trail blockc…

 

feat(ui): PR-03 - Audit Trail b…

Predictions page is in. /ai/predictions renders PredictionCard(s) with risk/confidence, ETA, and Create Fix PR action (opens fix URL or falls back to pipelines). 

feat(ui): PR-04 - Predictions p…

 

feat(ui): PR-04 - Predictions p…

Nav/UX refactor toward “Splunk-grade”. “Labs” collapsed, DevSecOps nav cleaned, Settings promoted; aligns with your IA plan. 

feat(ui): Implement enterprise-…

Backend “real data” posture improving. Handlers refactored to live Azure integrations and the fail-fast pattern (no silent mocks) is being applied. 

feat: Enhance API handlers with…

PCG MVP direction merged. PREVENT/PROVE/PAYBACK pillars and repo cleanup guidance are codified (remove legacy dirs, keep core prediction/evidence/ROI pieces). 

feat: Implement Predictive Clou…

 

feat: Implement Predictive Clou…

Core library hardened. Patent notices + new CQRS/DB modules and exports landed. 

feat: Complete PolicyCortex pla…

 

feat: Complete PolicyCortex pla…

Likely gaps to finish (from repo notes)

Kill remaining mocks & deep/GraphQL stubs when running “real”. The PR notes still call out Deep/GraphQL/Chat mocks and instruct wiring Audit/Predictions/ROI fully. 

feat: Enhance API handlers with…

 

feat: Enhance API handlers with…

Ensure mocks are off via a single switch (e.g., NEXT_PUBLIC_USE_MOCKS=false) and unify UX actions per page (“Verify chain”, “Export Evidence”, etc.). 

Merge PR-03: Audit Trail blockc…

Targeted corrections (diff-only) and how to update

One fetch helper (no-store + consistent errors)
Create frontend/lib/api.ts and use everywhere you fetch platform APIs.

// frontend/lib/api.ts (NEW)
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(path, { cache: 'no-store', ...init });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`API ${r.status}: ${text || r.statusText}`);
  }
  return r.json() as Promise<T>;
}


How to update: replace page-level fetch(...) calls with api('/api/v1/...') (e.g., Predictions, ROI, Audit list/verify). This lines up with the blueprint’s “single fetch helper” guidance. 

Merge PR-03: Audit Trail blockc…

 

policycortex_blueprint

GraphQL mocks hard-off in real mode
In your GraphQL route (or any mock route), make real-mode return 404, not fallback data:

- if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') return res.json(MOCK)
- return res.json(MOCK) // (leaky)
+ if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') return res.json(MOCK)
+ return res.status(404).json({ error: 'GraphQL disabled in real-data mode' })


How to update: apply to all GraphQL/mock endpoints so “real” never shows simulated data. Mirrors your blueprint’s directive. 

policycortex_blueprint

Health endpoint exposes DB/provider status
Add explicit DB/provider checks to /healthz so the StatusBar and ops can trust it:

// core/src/api/health.rs (PATCH)
#[derive(serde::Serialize)]
struct Health { ok: bool, db_ok: bool, provider_ok: bool, error: Option<String> }

pub async fn healthz(State(app): State<AppState>) -> (StatusCode, Json<Health>) {
  let mut error = None;
  let db_ok = app.db.ping().await.unwrap_or_else(|e| { error = Some(format!("db:{e}")); false });
  let provider_ok = app.azure.ping().await.unwrap_or_else(|e| { 
    error = Some(format!("{} provider:{e}", error.unwrap_or_default())); false 
  });
  let ok = db_ok && provider_ok;
  (if ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE },
   Json(Health{ ok, db_ok, provider_ok, error }))
}


How to update: expose this route and surface it in your StatusBar; aligns with the “fail-fast/real-mode” posture already appearing in recent commits. 

feat: Enhance API handlers with…

 

feat: Complete PolicyCortex pla…

Playwright smoke (no horizontal scroll + key surfaces)
Drop in a minimal e2e that matches the blueprint (Executive/Audit/Predictions):

// frontend/tests/smoke.spec.ts (NEW)
import { test, expect } from '@playwright/test'

test('Executive is front door + no horizontal scroll', async ({ page }) => {
  await page.goto('http://localhost:3000/')
  await expect(page).toHaveURL(/\/executive/)
  await page.evaluate(() => window.scrollBy(10000,0))
  expect(await page.evaluate(() => window.scrollX)).toBe(0)
})

test('Audit shows chain integrity UI', async ({ page }) => {
  await page.goto('/audit')
  await expect(page.getByText(/Chain integrity:/)).toBeVisible()
})

test('Predictions render + Fix PR visible', async ({ page }) => {
  await page.goto('/ai/predictions')
  await expect(page.getByRole('link', { name: /Create Fix PR/i })).toBeVisible()
})


How to update: pnpm playwright test as part of CI; this matches your existing PR notes and blueprint smoke. 

feat: Enhance API handlers with…

 

feat: Enhance API handlers with…

 

policycortex_blueprint

Quick verify (real mode)
# Backend (real)
export USE_REAL_DATA=true AZURE_TENANT_ID=... AZURE_SUBSCRIPTION_ID=... AZURE_CLIENT_ID=... AZURE_CLIENT_SECRET=... DATABASE_URL=...
cargo run -p core --release

# Frontend
pnpm i
export NEXT_PUBLIC_DEMO_MODE=false
pnpm dev


Then hit /healthz and the three pages (Executive, Audit, Predictions). 

policycortex_blueprint