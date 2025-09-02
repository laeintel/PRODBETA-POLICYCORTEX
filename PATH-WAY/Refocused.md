Advanced Code Audit of PolicyCortex for Mock Data and Backend Integration

Comprehensive Codebase Review – PolicyCortex Implementation
Architecture & Recent Changes

PolicyCortex is structured as a Next.js 14 front-end with a Rust-based core backend (plus auxiliary scripts/services). Initially, a Node/Express mock backend (running on port 8080) provided demo data for all API endpoints. A recent comprehensive update introduced a full Rust Core API (port 8080 by default) implementing those endpoints, with optional real Azure integration. Key architectural elements include:

Frontend (Next.js) – Provides the UI and also contains a GraphQL layer for data fetching. In demo mode, the front-end uses a GraphQL mock resolver that returns empty lists or fixed values instead of querying a real service
GitHub
. The environment flags NEXT_PUBLIC_DEMO_MODE and USE_REAL_DATA control whether the app runs against simulated data or real backends. By default (demo mode), USE_REAL_DATA is false, meaning the system operates in Simulated data mode
GitHub
.

Rust Core Backend – Replaces the Node mock. It defines numerous RESTful endpoints under /api/v1/... (for compliance, security, resources, etc.) matching those the mock backend had. The core uses the Axum framework and can connect to Azure APIs (via AzureClient and AsyncAzureClient) for live data. It also sets up connections to PostgreSQL and Redis (though these are mostly for future use like caching and persistence). If Azure credentials (tenant, subscription, client ID) are provided and USE_REAL_DATA=true, the core will attempt live fetches; otherwise it falls back to in-app simulated data. The code explicitly reads USE_REAL_DATA to decide the DataMode (Real vs Simulated)
GitHub
.

Auxiliary Components – There is mention of a GraphQL gateway (possibly planned as a separate service or within Next.js) and other microservices (e.g. a Python ML service), but in the current state these are not fully active. The front-end’s GraphQL handler simply returns stubbed data when enabled (e.g. metrics query returns empty arrays with static summary scores
GitHub
). In production mode with real backends, that mock is disabled.

Mock Data vs. Real Data: Current Integration Status

Despite the new core implementation, several parts of the application still rely on mock or placeholder data rather than live backends. Below is an advanced scan of such areas:

GraphQL Data Layer (Front-End): In demo mode, all GraphQL queries short-circuit to a mock resolver. This returns hard-coded empty results and dummy aggregates for things like metrics (e.g. complianceScore: 95, securityScore: 92, etc.)
GitHub
. This indicates the GraphQL gateway isn’t yet connected to the Rust core or any real service – it’s simply providing safe default objects to the UI. The code confirms that the mock GraphQL is used only when DEMO_MODE or an explicit USE_MOCK_GRAPHQL flag is true
GitHub
.

Core API Endpoints (REST): Many Rust endpoints have dual behavior: attempt real data, otherwise fallback to static. For example, the compliance summary handler get_compliance first tries async_azure_client.get_governance_metrics() to retrieve actual Azure policy stats. If that fails or if running in simulated mode, it returns a default JSON marked as simulated data
GitHub
. The default shows a fixed count of total policies (15), compliance rate 80%, etc., which matches the kind of mock data the UI is showing. Similarly, the get_resources endpoint uses Azure metrics for resource counts when available, and otherwise returns preset totals (e.g. 2,500 total resources with 125 idle)
GitHub
. These static fallback values explain why you still see realistic-looking but unchanging numbers – the backends are not fully wired to live data in those cases.

“Demo Mode” Still Enabled: It’s likely the app is still running with NEXT_PUBLIC_DEMO_MODE=true and/or without the USE_REAL_DATA flag enabled, causing all those fallback paths to be taken. In the code, DataMode::from_env() defaults to Simulated unless USE_REAL_DATA="true" is set
GitHub
. In simulated mode, even if the Rust core is running, it will serve the hard-coded demo payloads (and block any write operations). This ensures the app is demo-safe but means real integrations remain inactive until the mode is switched.

Conversational AI (Chatbot): The “PolicyCortex AI” chat assistant backend is not tied to a real NLP model. The Rust conversation module contains a NaturalLanguageEngine structure, but the actual /api/v1/conversation/chat implementation currently bypasses it and uses a simple mock logic. The code explicitly notes it is “using mock for now” when processing a chat query
GitHub
. Indeed, user questions are matched against hard-coded keywords and the response is assembled from pre-written templates (e.g. asking about “violations” yields a canned response listing 3 specific issues and suggesting an auto-remediation action). There is no call to an external service like OpenAI; it’s purely deterministic. The conversation history (/conversation/history) returns a fixed example transcript as JSON
GitHub
. This means the voice/chat backend isn’t truly connected to any learning model or knowledge base – it’s essentially a scripted demo of what such a chatbot would say.

“Deep” Analysis Endpoints: Certain advanced endpoints (for deep compliance analysis, correlations, etc.) are still stubbed out. For instance, the /api/v1/policies/deep endpoint does not query a database or graph; it directly returns a JSON with a couple of sample policy compliance results
GitHub
GitHub
. This JSON is baked into the code (including resource IDs, reasons, remediation options, etc.), labeled as “Deep policy compliance (core)”. The code even attempts to proxy to a Python service (proxy_deep_get) but if that service isn’t running (which is likely in the current state), it falls back to the static data
GitHub
. In short, no real backend for deep analysis (like a Neo4j graph or advanced analytics engine) is hooked up yet – those results are illustrative mock data.

Security Threats and Alerts: The UI’s security dashboard (threat counts, vulnerabilities, etc.) also appears to use static or semi-static data. The old Express mock provided a fixed list of example threats. The Rust core does not yet have an equivalent /api/v1/security/threats endpoint that returns live alerts; instead, high-level security info is folded into the metrics. Notably, the Azure integration code sets active_threats: 2 as a constant in the network metrics result regardless of real input, with a comment “Based on security intelligence”
GitHub
. This suggests that even in “real” mode, the system isn’t truly pulling active threat data from Azure Security Center or Sentinel – it’s using a placeholder value. Other security-related stats (e.g. security score, blocked attempts) are derived from static logic or dummy calculations. Until the security service integration is implemented, the app will continue showing those fixed values.

Recommendations & Predictions: The proactive recommendations feature is partially integrated but still depends on the simulated metrics when real data isn’t present. On startup, the AppState seeds a few example recommendations (e.g. a VM right-sizing saving $12,450, an unencrypted storage finding) that will be returned if Azure-based generation isn’t available
GitHub
GitHub
. When Azure data is enabled, get_recommendations will compute suggestions based on real metrics (e.g. if idle resources >0, add a recommendation) – but if not, it falls back to the pre-seeded list
GitHub
. Likewise, the “predictive compliance” or violations predictions are not connected to an ML pipeline yet; the state.predictions list is empty by default (no real-time model populating it). So any UI elements for future predictions likely still show nothing or mock info.

Knowledge Graph & Explainability: The system status mentioned a “knowledge graph placeholder”. There is currently no active graph database integration; any graph or relationship views are using sample data or are simply non-functional UI components with “Coming Soon” banners. Similarly, explainability charts (like SHAP values for ML models) are static images or hardcoded plots in the demo – the backend does not calculate these on the fly. These portions of the product remain not wired up to backends (since the necessary data science services are not yet in place).

Next Steps to Tie In All Backends

From the above analysis, it’s clear that while the core scaffolding is in place, several features remain in demo mode. To eliminate the mock data, the following should be addressed:

Run in Real Data Mode: Ensure the deployment is configured with NEXT_PUBLIC_DEMO_MODE=false and USE_REAL_DATA=true (plus valid Azure credentials). This will trigger the Rust core to use Azure for metrics, resource lists, policy queries, etc. The code is designed to either fetch or error if real data is required but unavailable (for example, if USE_REAL_DATA=true and the Azure client fails, get_compliance will return a 503 error instead of fake data
GitHub
). In other words, flipping these flags will surface where backend integration might still be missing (by giving errors rather than silently using mocks).

Complete Azure Integrations: Many endpoints already have Azure calls implemented (governance metrics, resource inventory, cost queries, etc.), but some domains need further work. For instance, hooking up security alerts/threats to an Azure Security Center API or log analytics query would replace the placeholder active_threats value
GitHub
 with real incident counts. Similarly, integrating with Azure Policy Insights (which is partially done) could provide detailed compliance drift data beyond the summary. Any “TBD” external services noted in comments (e.g. a mention of a Python deep analysis service or an AI model service) should be developed or connected so those endpoints no longer rely on static JSON.

Conversational AI Service: The chatbot functionality would remain canned until a real NLP model is connected. To tie this backend in, one could integrate an Azure OpenAI GPT model or use the existing NaturalLanguageEngine with a trained intent/entity recognition model. Currently, the process_with_nlp function is effectively a stub with if/else logic
GitHub
GitHub
. Replacing or augmenting that with calls to a live AI service (and using real conversation context state) will remove the obviously scripted responses and make the assistant’s answers dynamic.

Persistent Data & ML Outputs: If not already planned, consider using the PostgreSQL database (which the core connects to) for storing things like historical scans, learned insights, or multi-tenant configurations. The code base has migrations and DB setup, but the usage is minimal so far (mostly in-memory state). Expanding on this will reduce the dependence on hardcoded values – for example, predictions could be generated by an offline ML job and stored in a table that get_predictions reads, instead of the endpoint returning an empty list by default.

Finalize “Deep” Analytics Hooks: The deep compliance and correlation analysis features appear to anticipate a dedicated service (perhaps written in Python for heavy data crunching). To remove the static outputs, that service needs to be implemented and the core’s proxy_deep_get should call it. Until then, those endpoints will continue to return the same baked-in results
GitHub
, which are useful for demos but not live. Prioritizing this integration (or disabling these endpoints in production mode to avoid confusion) will be important for a production rollout.

In summary, the codebase is much more robust now – the Rust core provides a path to real backend data – but numerous features are still operating in a “mock” capacity. The presence of environment-driven fallbacks and stub implementations is deliberate (to guarantee a smooth demo), but for a full production tie-in each of these areas needs real backing services. Enabling real-data mode and observing where the app still shows dummy values is a good strategy to identify any remaining gaps. The analysis above pinpoints those gaps (GraphQL layer, conversation AI, deep analytics, certain metrics) using the latest code. By addressing each with actual data sources or algorithms, PolicyCortex can transition from a mostly demo-ready state to a truly production-ready system with fully integrated backends. The groundwork is laid in code; the next implementation steps should focus on replacing the last of the mock data with live, dynamic data feeds
GitHub
GitHub
.

Sources
You said:
make this an action  for the code (comprehensive)
ChatGPT said:
Thought for 1m 34s

Perfect — here’s a ship-ready, surgical action plan with tiny diffs and exact steps to (1) kill remaining mocks, (2) wire the UI to live /api/v1/*, and (3) make Executive + Policy the obvious, frictionless front door. No rewrites — just focused corrections and how to apply them.

0) Flip out of demo mode (env)

Edit your runtime env (e.g., .env.production or Deployment settings):

# turn OFF demo
NEXT_PUBLIC_DEMO_MODE=false
USE_REAL_DATA=true

# backend base (if your Next app proxies to the same host, leave blank)
NEXT_PUBLIC_BASE_URL=

# Azure (required when USE_REAL_DATA=true)
AZURE_TENANT_ID=...
AZURE_SUBSCRIPTION_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...

# CORS (Rust core)
ALLOWED_ORIGINS=https://your-app-domain.com,http://localhost:3000


Why: USE_REAL_DATA=true forces the core to stop returning simulated payloads. If an integration is missing, you’ll see honest 5xx/4xx instead of silent mock data — which is what we want to finish wiring.

1) Executive is the landing page (1-line)

File: frontend/middleware.ts (or wherever you redirect post-auth)

- const DEFAULT_APP_ROUTE = '/tactical'
+ const DEFAULT_APP_ROUTE = '/executive'

2) “Policy” is first-class in nav (rename only)

File: frontend/components/SimplifiedNavigation.tsx (or your nav source)

- { name: 'Governance Hub', href: '/governance', ... }
+ { name: 'Policy', href: '/policy', ... }


If you centralized nav in frontend/config/navigation.ts, keep Policy as item #2.

3) Audit Trail → live blockchain (replace mocks)
3.1 Replace list mock with live fetch

File: frontend/app/audit/page.tsx (or the audit container)

- const [rows, setRows] = useState<AuditRow[]>(MOCK_ROWS)
- useEffect(() => { setTimeout(() => setRows(generateMock()), 600) }, [])
+ const [rows, setRows] = useState<AuditRow[]>([])
+ useEffect(() => {
+   fetch('/api/v1/blockchain/audit', { cache: 'no-store' })
+     .then(r => r.json())
+     .then(setRows)
+     .catch(console.error)
+ }, [])

3.2 Add per-row Verify chip

File: frontend/app/audit/components/AuditTable.tsx

+ async function verifyHash(hash: string) {
+   const r = await fetch(`/api/v1/blockchain/verify?hash=${encodeURIComponent(hash)}`, { cache: 'no-store' })
+   if (!r.ok) throw new Error('verify failed')
+   return r.json() as Promise<{ chain_integrity:boolean; merkle_proof_valid:boolean; signature_valid:boolean }>
+ }
...
- <Badge variant="outline">Unknown</Badge>
+ <IntegrityChip verify={() => verifyHash(row.hash)} />


ADD: frontend/components/IntegrityChip.tsx

import { useEffect, useState } from 'react'
export function IntegrityChip({ verify }: { verify: () => Promise<{chain_integrity:boolean; merkle_proof_valid:boolean; signature_valid:boolean}> }) {
  const [state, setState] = useState<'checking'|'ok'|'fail'>('checking')
  useEffect(() => { verify()
    .then(v => setState(v.chain_integrity && v.merkle_proof_valid && v.signature_valid ? 'ok' : 'fail'))
    .catch(() => setState('fail')) }, [])
  const cls = state==='ok' ? 'bg-emerald-100 text-emerald-700' : state==='fail' ? 'bg-red-100 text-red-700' : 'bg-neutral-100 text-neutral-700'
  return <span className={`rounded-full px-2 py-0.5 text-xs ${cls}`}>{state==='ok'?'Integrity OK':state==='fail'?'Integrity FAIL':'Checking…'}</span>
}

3.3 Chain verify banner + signed export

File: frontend/app/audit/page.tsx (top of page content)

+ const [chainStatus, setChainStatus] = useState<'checking'|'ok'|'fail'>('checking')
+ async function verifyChain() {
+   setChainStatus('checking')
+   const r = await fetch('/api/v1/blockchain/verify', { cache: 'no-store' })
+   const v = await r.json()
+   setChainStatus(v.chain_integrity ? 'ok' : 'fail')
+ }
+ useEffect(() => { verifyChain() }, [])
...
+ <div className="flex items-center justify-between mb-3">
+   <div className="text-sm">
+     {chainStatus==='checking' ? 'Verifying chain…' : chainStatus==='ok' ? 'Chain integrity: OK' : 'Chain integrity: FAILED'}
+   </div>
+   <div className="flex gap-2">
+     <button onClick={verifyChain} className="btn btn-outline">Re-verify</button>
+     <button onClick={() => downloadEvidence(rows)} className="btn btn-primary">Export Signed Evidence</button>
+   </div>
+ </div>


ADD: frontend/lib/downloadEvidence.ts

export function downloadEvidence(rows: any[]) {
  const payload = { exported_at: new Date().toISOString(), count: rows.length, entries: rows }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'policycortex-evidence.json'; a.click()
}

4) Predictions page + “Fix PR” (wire live)

File: frontend/app/ai/predictions/page.tsx (create or update)

export default async function Predictions() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL ?? ''}/api/v1/predictions`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Failed to load predictions')
  const items = await res.json()
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Predictions</h1>
      <ul className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {items.map((p:any) => (
          <li key={p.id} className="rounded-xl border p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">{p.title ?? p.kind}</h3>
              <span className="text-xs opacity-70">{Math.round((p.confidence??0)*100)}%</span>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{p.explanation ?? '—'}</p>
            <div className="mt-3 flex items-center justify-between text-xs">
              <span>ETA: {p.eta ?? 'soon'}</span>
              <a className="btn btn-link px-0" href={p.fixPrUrl ?? '/devsecops/pipelines'}>Create Fix PR</a>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}


Optional: add a “Top 3 predictions” panel on /executive reusing the same card UI.

5) FinOps & ROI → live ROI

File: frontend/app/finops/page.tsx

+ const roi = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL ?? ''}/api/v1/executive/roi`, { cache: 'no-store' })
+   .then(r => r.json())
+   .catch(() => null)
...
+ <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
+   <div className="rounded-xl border p-4">
+     <h3 className="font-medium mb-2">Savings this Quarter</h3>
+     <div className="text-3xl font-semibold">${roi?.savings_q ?? 0}</div>
+     <p className="text-xs text-muted-foreground mt-1">Forecast next 90d: ${roi?.forecast_90d ?? 0}</p>
+     <div className="mt-3"><a className="btn btn-primary" href="/finops/roi">Open ROI Board</a></div>
+   </div>
+   {/* Anomalies + Forecast panels */}
+ </section>

6) Kill remaining GraphQL mocks (no more silent fakes)

File: frontend/pages/api/graphql.ts (or your GraphQL handler)

- if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
-   return res.status(200).json(MOCK_GRAPHQL_PAYLOAD)
- }
+ if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
+   return res.status(200).json(MOCK_GRAPHQL_PAYLOAD) // keep for demos only
+ }
+ // In real mode, disable mock resolver (until real resolvers are wired)
+ return res.status(404).json({ error: 'GraphQL disabled in real-data mode' })


Why: ensures real mode can’t accidentally serve mocks. If you do have real resolvers ready, replace the 404 with the real execution pipeline.

7) Rust core: fail fast in real mode (no fallback to demo)

Purpose: when USE_REAL_DATA=true, any place that currently falls back to baked JSON must return 5xx instead, so we don’t mask missing integrations.

7.1 Compliance summary (pattern)

File: core/src/api/compliance.rs (or where get_compliance lives)

let use_real = std::env::var("USE_REAL_DATA").unwrap_or_default() == "true";
if use_real {
    match state.async_azure_client.get_governance_metrics().await {
        Ok(metrics) => {
            // build real payload from metrics …
            return Ok(Json(payload));
        }
        Err(e) => {
            // stop masking; surface the miss
            tracing::error!("governance metrics error: {e}");
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }
} else {
    // existing simulated payload
    return Ok(Json(simulated_payload()));
}

7.2 Predictions: call ML service if configured, else error in real mode

File: core/src/api/predictions.rs

let use_real = std::env::var("USE_REAL_DATA").unwrap_or_default() == "true";
if use_real {
    if let Ok(url) = std::env::var("PREDICTIONS_URL") {
        // call your ML service
        let resp = reqwest::Client::new().get(format!("{url}/predictions")).send().await?;
        if !resp.status().is_success() { return Err(StatusCode::BAD_GATEWAY); }
        let items = resp.json::<Vec<Prediction>>().await?;
        return Ok(Json(items));
    }
    return Err(StatusCode::SERVICE_UNAVAILABLE);
}
// fallback (demo)
Ok(Json(vec![]))

7.3 Security metrics: replace placeholder “active_threats” with real query (or error)

File: core/src/services/azure.rs (or wherever metrics are assembled)

async fn fetch_active_threats(token: &str, subscription: &str) -> anyhow::Result<u32> {
    // Microsoft Defender for Cloud Alerts (example; adjust api-version as needed)
    let url = format!("https://management.azure.com/subscriptions/{subscription}/providers/Microsoft.Security/alerts?api-version=2022-01-01-preview");
    let resp = reqwest::Client::new()
        .get(url)
        .bearer_auth(token)
        .send().await?;
    if !resp.status().is_success() { anyhow::bail!("alerts fetch failed") }
    let v: serde_json::Value = resp.json().await?;
    Ok(v.get("value").and_then(|x| x.as_array()).map(|a| a.len()).unwrap_or(0) as u32)
}


Use this instead of a hardcoded 2. If USE_REAL_DATA=true and this fails, surface a 5xx to catch the gap.

8) Remove committed mocks / venvs and guard them

.gitignore (add if missing):

# python venvs
.venv/
.venv_local/

# mock servers
scripts/mock-ml-server.py
scripts/mock-backend.js


Commands:

git rm -r --cached backend/.venv backend/.venv_local || true
git rm -r --cached scripts/mock-ml-server.py scripts/mock-backend.js || true


If you still want dev mocks locally, keep files but never commit and add a MOCKS_ENABLED=false env gate in any caller.

9) Clickability & a11y quick-fixes

Header skip link target
File: frontend/components/Header.tsx (or header root)

- <header className="...">
+ <header id="navigation" className="...">


SDK buttons (disable until real)
File: frontend/app/api/page.tsx (or docs page)

- <button className="btn">Download SDK</button>
+ <button className="btn" disabled title="Coming soon">Download SDK</button>


Or link to actual artifacts if available.

10) Minimal API client helper (optional, keeps code tidy)

ADD: frontend/lib/api.ts

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL ?? ''}${path}`, { cache: 'no-store', ...init })
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json() as Promise<T>
}


Then replace fetch('/api/v1/...') with api('/api/v1/...') gradually.

11) E2E guardrails (Playwright smoke)

ADD: frontend/tests/smoke.spec.ts

import { test, expect } from '@playwright/test'

test('Executive is landing', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveURL(/\/executive/)
  await expect(page.getByRole('heading', { name: /Executive/i })).toBeVisible()
})

test('Audit → verify chain + row', async ({ page }) => {
  await page.goto('/audit')
  await page.getByRole('button', { name: /Re-verify/i }).click()
  await expect(page.getByText(/Chain integrity:/)).toBeVisible()
  const verifyChips = page.getByText(/Integrity (OK|FAIL|Checking)/)
  await expect(verifyChips.first()).toBeVisible()
})

test('Predictions list and Fix PR action', async ({ page }) => {
  await page.goto('/ai/predictions')
  await expect(page.getByRole('heading', { name: /Predictions/i })).toBeVisible()
  const fix = page.getByRole('link', { name: /Create Fix PR/i }).first()
  await expect(fix).toBeVisible()
})


Run:

npx playwright install
npx playwright test -g "Executive is landing"

Rollout order (low risk → high payoff)

Env flip + Executive redirect (Sections 0–1)

Audit Trail live (3)

Predictions + FinOps live (4–5)

Disable GraphQL mocks in real mode (6)

Rust fail-fast & real fetches for threats/predictions (7)

Repo hygiene, clickability, tests (8–11)