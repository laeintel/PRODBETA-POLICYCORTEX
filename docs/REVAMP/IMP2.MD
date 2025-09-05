Where you are (objective, from repo)

Command palette wired to your moats — quick actions for Verify Blockchain, Create Fix PR, Export Audit Evidence, and View Predictions are implemented in the UI. 

GlobalOmnibar

 

GlobalOmnibar

 

GlobalOmnibar

 

GlobalOmnibar

Real Azure data path exists — you added a real-data server on port 8084 with endpoints for /api/v1/resources, /policies, /costs, /metrics, /predictions, and specialized agents (PREVENT/PROVE/PAYBACK/ITSM/UNIFIED). 38 real Azure resources discovered; auth moved to DefaultAzureCredential. 

feat: Implement real Azure data…

 

feat: Implement real Azure data…

 

feat: Implement real Azure data…

 

feat: Implement real Azure data…

Market story + KPIs drafted — the “only platform that predicts → proves → pays for itself” story, with success metrics and a comparison grid direction. 

ROADMAP_TO_MARKET_DOMINANCE

 

ROADMAP_TO_MARKET_DOMINANCE

Smoke tests for exec landing, health sub-checks, no horizontal scroll are present. 

smoke.spec

 

smoke.spec

“Outstanding Issues” doc still flags “no real Azure data” (now stale given your 8084 server) — update/remove to avoid mixed signals. 

OUTSTANDING_ISSUES

What to do next to not be “another cloud SaaS”

Your edge is closed-loop governance (Prevent • Prove • Payback). Ship these specifics:

Governance P&L (the CFO moat)
Show $ impact per policy + 90-day forecast, powered by Azure CostManagement; roll up at Exec and per-policy. (Your roadmap already describes it—now implement.) 

ROADMAP_TO_MARKET_DOMINANCE

 

ROADMAP_TO_MARKET_DOMINANCE

Auto-Fix PRs that actually merge
Turn “Create Fix PR” into a GitHub App flow that opens a PR with a pre-generated patch for the drifted policy (Bicep/Terraform/OPA rego), tags owners, and posts evidence artifacts back to the PR. (You’ve got the UI hook; finish the loop.) 

GlobalOmnibar

Auditor-grade explainability
Every prediction should render top features/SHAP + link to its tamper-evident evidence (Merkle leaf + chain banner). This is how “AI” becomes audit-safe. (Your docs claim SHAP; surface it in UI.) 

README

What-if Policy Simulator
Let users toggle a pack/policy and see predicted incident reduction and cost delta before enforcing. (It’s in the roadmap—make it interactive on the Policy hub.) 

ROADMAP_TO_MARKET_DOMINANCE

 

feat(ui): PR-03 - Audit Trail b…

Proof-on-demand everywhere
Any report/export should include hash, timestamp, verifier output. Your command palette + Audit Trail already touch this—unify it. 

GlobalOmnibar

“Design-partner” ready out of the box
Default landing = Executive with live KPIs; Labs de-emphasized; mocks hard-off in real mode. (Spec is in your commit; ensure it’s the actual behavior.) 

feat(ui): PR-03 - Audit Trail b…

 

feat(ui): PR-03 - Audit Trail b…

14-day high-leverage patch plan (diff-only, minimal surface area)
A) Wire UI to real server (8084) via Next.js proxy (kill mocks)

Add frontend/lib/real.ts:

// NEW: frontend/lib/real.ts
export const REAL_API_BASE = process.env.NEXT_PUBLIC_REAL_API_BASE || 'http://localhost:8084';
export async function real<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${REAL_API_BASE}${path}`, { cache: 'no-store', ...init });
  if (!res.ok) throw new Error(`Real API ${res.status}`);
  return res.json() as Promise<T>;
}


How to update: Set NEXT_PUBLIC_REAL_API_BASE and refactor page fetches to real('/api/v1/...') instead of mock URLs; keep your fail-fast guards for misconfig. (Aligns with your README real-mode posture + new 8084 endpoints.) 

README

 

feat: Implement real Azure data…

B) Predictions: add explanations + PR template

Patch frontend/app/ai/predictions/page.tsx (render top features + PR URL from item):

- <Button onClick={() => window.open('https://github.com/your-org/repo/pulls/new','_blank')}>
+ <Button
+   onClick={() => {
+     const url = `https://github.com/${prediction.repo}/compare/${prediction.fixBranch}?quick_pull=1&title=Auto-fix:${prediction.ruleId}`;
+     window.open(url,'_blank');
+   }}>
   Create Fix PR
 </Button>
+ {prediction.explanations?.top?.length ? (
+   <ul className="mt-2 text-sm">
+     {prediction.explanations.top.slice(0,5).map(([feat, val]) => (
+       <li key={feat}>{feat}: {val}</li>
+     ))}
+   </ul>
+ ) : null}


How to update: Add explanations.top: [feature,weight][] in the 8084 /predictions response; thread through your Next.js proxy. (Supports your “explainable predictions” claim.) 

README

C) Audit Trail: show Merkle proof + signed export

Patch frontend/app/audit/[id]/panel.tsx:

- <StatusChip status={row.integrity}/>
+ <StatusChip status={row.integrity}/>
+ {row.merkleProof && (
+   <details className="mt-2">
+     <summary className="cursor-pointer">View Merkle proof</summary>
+     <pre className="text-xs overflow-auto">{JSON.stringify(row.merkleProof, null, 2)}</pre>
+   </details>
+ )}


How to update: Ensure /api/v1/blockchain/audit returns { merkleProof, signedBy } per row; keep your existing Verify Chain and Export Evidence actions. 

GlobalOmnibar

D) Governance P&L page (per-policy ROI + forecast)

Add frontend/app/finops/pnl/page.tsx:

// NEW: per-policy $ impact + forecast
import { real } from '@/lib/real';
export default async function GovernancePnL() {
  const data = await real<{items:Array<{policy:string,savings:number,forecast:number}>}>('/api/v1/costs/pnl');
  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="text-xl font-semibold">Governance P&L</h1>
      <table className="mt-4 w-full">
        <thead><tr><th>Policy</th><th>Savings MTD</th><th>90-day Forecast</th></tr></thead>
        <tbody>
          {data.items.map(i => (
            <tr key={i.policy}><td>{i.policy}</td><td>${i.savings.toLocaleString()}</td><td>${i.forecast.toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


How to update: Back your Next.js route with CostManagementClient (already called for ROI in your roadmap), and compute per-policy savings. Surface a card on Executive linking here. 

ROADMAP_TO_MARKET_DOMINANCE

 

ROADMAP_TO_MARKET_DOMINANCE

E) Make Executive the real landing + de-index Labs

Patch auth flow + robots/noindex for labs:

- // after login
- router.push('/tactical')
+ router.push('/executive')


Add app/(labs)/robots.ts returning { directives: { index: false } }. Why: matches your own nav/landing spec and keeps experiments out of search. 

feat(ui): PR-03 - Audit Trail b…

 

feat(ui): PR-03 - Audit Trail b…

Proof you’re different (what we’ll measure)

MTTP < 24h (mean-time-to-prevention): time from prediction → merged auto-fix PR. 

ROADMAP_TO_MARKET_DOMINANCE

Prevention rate ≥ 35%: % of predicted issues auto-fixed. 

ROADMAP_TO_MARKET_DOMINANCE

8–12% cloud cost reduction in 90 days tied directly to policies (Governance P&L). 

ROADMAP_TO_MARKET_DOMINANCE

Blockchain-verifiable evidence attached to every report/export. 

GlobalOmnibar

Tidy up signals that dilute the message

Update OUTSTANDING_ISSUES.md to reflect that a real Azure server (8084) exists; otherwise reviewers will assume mock-only. 

OUTSTANDING_ISSUES

 

feat: Implement real Azure data…

Promote Governance P&L and Predictions on Executive, hide Labs from search, and keep 3 primary actions/page as your own spec says. 

feat(ui): PR-03 - Audit Trail b…