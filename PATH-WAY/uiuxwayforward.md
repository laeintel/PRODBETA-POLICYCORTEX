What I’m delivering here

Nav & routing fixes so Executive is the front door and Policy is first-class.

Audit Trail wired to blockchain verify (per-row + full-chain + evidence export).

Predict page + home widget using your predictions endpoint and “Create Fix PR” action.

FinOps & ROI wired card + board.

Policy Hub + Packs + Composer + Enforcement + Exceptions + Evidence (scaffolds you can wire incrementally).

Labs route-group to hide experimental areas without deleting them.

Omnibar & Quick Actions so core flows are 1–2 clicks.

If your paths differ, keep the same structure and adjust folder names; the components are self-contained.

PR-01 — Executive-first + Policy as top-level
A) Make Executive the default landing after login

File: frontend/middleware.ts

- const DEFAULT_APP_ROUTE = '/tactical'
+ const DEFAULT_APP_ROUTE = '/executive'

B) Centralize nav (clean order + labels)

ADD: frontend/config/navigation.ts

export type NavItem = { label: string; href: string; badge?: 'Beta'|'Labs' }

export const CORE: NavItem[] = [
  { label: 'Executive', href: '/executive' },
  { label: 'Policy', href: '/policy' },
  { label: 'Audit Trail', href: '/audit' },
  { label: 'Predict', href: '/ai/predictions' },
  { label: 'FinOps & ROI', href: '/finops' },
  { label: 'Access Governance', href: '/rbac' },
  { label: 'Resources', href: '/resources' },
  { label: 'DevSecOps', href: '/devsecops/pipelines' },
  { label: 'Settings', href: '/settings' },
]

export const LABS: NavItem[] = [
  { label: 'Blockchain Explorer', href: '/blockchain', badge: 'Labs' },
  { label: 'Copilot', href: '/copilot', badge: 'Labs' },
  { label: 'Cloud ITSM', href: '/itsm', badge: 'Labs' },
  { label: 'Quantum-Safe', href: '/quantum', badge: 'Labs' },
  { label: 'Edge Governance', href: '/edge', badge: 'Labs' },
]

C) Render that nav

File: frontend/components/SimplifiedNavigation.tsx

- // Existing hard-coded items ...
+ import { CORE, LABS } from '@/config/navigation'
+ // Render CORE in order, then one “Labs” group that expands to LABS


(Keep your existing icon set; only swap the data source to CORE/LABS.)

PR-02 — Move experiments to Labs (noindex)

ADD: frontend/app/(labs)/layout.tsx

export const metadata = { robots: { index: false, follow: true } }
export default function LabsLayout({ children }: { children: React.ReactNode }) {
  return <div data-section="labs">{children}</div>
}


Move directories (no URL change): put /copilot, /quantum, /edge, /blockchain under app/(labs)/….
(Blockchain Explorer stays in Labs until Audit Trail has full verify & export—in PR-03 we add that.)

PR-03 — Audit Trail → live blockchain verify + evidence export
A) Replace mock list with real fetch

File: frontend/app/audit/page.tsx

- const [rows, setRows] = useState<AuditRow[]>(MOCK_ROWS)
- useEffect(() => { setTimeout(() => setRows(generateMock()), 600) }, [])
+ const [rows, setRows] = useState<AuditRow[]>([])
+ useEffect(() => {
+   fetch('/api/v1/blockchain/audit', { cache: 'no-store' })
+     .then(r => r.json())
+     .then(setRows)
+     .catch(console.error)
+ }, [])

B) Per-row Verify action + integrity chip

File: frontend/app/audit/components/AuditTable.tsx

+ async function verifyHash(hash: string) {
+   const r = await fetch(`/api/v1/blockchain/verify?hash=${encodeURIComponent(hash)}`, { cache: 'no-store' })
+   return r.json() as Promise<{ chain_integrity: boolean; merkle_proof_valid: boolean; signature_valid: boolean }>
+ }
...
- <Badge variant="outline">Unknown</Badge>
+ <IntegrityChip verify={() => verifyHash(row.hash)} />


ADD: frontend/components/IntegrityChip.tsx

export function IntegrityChip({ verify }: { verify: () => Promise<{chain_integrity:boolean; merkle_proof_valid:boolean; signature_valid:boolean}> }) {
  const [state, setState] = useState<'idle'|'ok'|'fail'|'checking'>('idle')
  useEffect(() => { setState('checking'); verify().then(v => setState(v.chain_integrity && v.merkle_proof_valid && v.signature_valid ? 'ok' : 'fail')).catch(() => setState('fail')) }, [])
  const cls = state==='ok' ? 'bg-emerald-100 text-emerald-700' : state==='fail' ? 'bg-red-100 text-red-700' : 'bg-neutral-100 text-neutral-700'
  const label = state==='ok' ? 'Integrity OK' : state==='fail' ? 'Integrity FAIL' : 'Checking…'
  return <span className={`rounded-full px-2 py-0.5 text-xs ${cls}`}>{label}</span>
}

C) Full-chain verify banner + export

File: frontend/app/audit/page.tsx

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

export function downloadEvidence(rows: {hash:string; merkle_root?:string; signature?:string; [k:string]:any}[]) {
  const payload = { exported_at: new Date().toISOString(), count: rows.length, entries: rows }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'policycortex-evidence.json'; a.click()
}

PR-04 — Predict page + home widget

ADD: frontend/app/ai/predictions/page.tsx

export default async function Predictions() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL ?? ''}/api/v1/predictions`, { cache: 'no-store' })
  const items = await res.json()
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Predictions</h1>
      <ul className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {items.map((p:any) => <PredictionCard key={p.id} p={p} />)}
      </ul>
    </div>
  )
}


ADD: frontend/components/PredictionCard.tsx

export function PredictionCard({ p }: { p: any }) {
  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{p.title ?? p.kind}</h3>
        <span className="text-xs opacity-70">{Math.round((p.confidence??0)*100)}%</span>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{p.explanation ?? 'Pending explanation'}</p>
      <div className="mt-3 flex items-center justify-between text-xs">
        <span>ETA: {p.eta ?? 'soon'}</span>
        <button className="btn btn-link px-0" onClick={() => window.open(p.fixPrUrl ?? '/devsecops/pipelines', '_self')}>Create Fix PR</button>
      </div>
    </div>
  )
}


Home/Tactical widget: add a small “Top 3 predictions” panel that links to /ai/predictions (same card component in compact mode).

PR-05 — FinOps & ROI (card + board)

File: frontend/app/finops/page.tsx

+ const roi = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL ?? ''}/api/v1/executive/roi`, { cache: 'no-store' }).then(r => r.json()).catch(() => null)
...
+ <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
+   <div className="rounded-xl border p-4">
+     <h3 className="font-medium mb-2">Savings this Quarter</h3>
+     <div className="text-3xl font-semibold">${roi?.savings_q ?? 0}</div>
+     <p className="text-xs text-muted-foreground mt-1">Forecast next 90d: ${roi?.forecast_90d ?? 0}</p>
+     <div className="mt-3"><a className="btn btn-primary" href="/finops/roi">Open ROI Board</a></div>
+   </div>
+   {/* add Anomalies + Forecast panels adjacent */}
+ </section>

PR-06 — Policy suite (Hub, Packs, Composer, Enforcement, Exceptions, Evidence)

ADD: frontend/app/(core)/policy/page.tsx

export default function PolicyHub() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Policy</h1>
        <p className="text-muted-foreground">Define, simulate, enforce, and evidence guardrails.</p>
      </header>
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {/* Coverage, Predicted Drift, Prevented, $ Impact cards (fetch + render) */}
      </div>
      <div className="flex gap-2">
        <a className="btn btn-primary" href="/policy/composer">New Policy</a>
        <a className="btn" href="/policy/packs">Install Policy Pack</a>
        <a className="btn btn-outline" href="/policy/enforcement">Enforcement</a>
      </div>
      {/* Active Policies table; Exceptions expiring; Recent prevention events */}
    </div>
  )
}


ADD empty pages (scaffolds you’ll wire over time):

frontend/app/(core)/policy/packs/page.tsx
frontend/app/(core)/policy/composer/page.tsx
frontend/app/(core)/policy/enforcement/page.tsx
frontend/app/(core)/policy/exceptions/page.tsx
frontend/app/(core)/policy/evidence/page.tsx


Each page: title + placeholder sections. Keep Composer steps: Baseline → Scope → Parameters → Enforcement → Simulate (prediction + $$ impact) → Create.

You keep DevSecOps as the place gates actually run; link both ways (Policy ▸ Enforcement lists bindings; Gates detail links “Managed by Policy”).

PR-07 — Quick Actions & Omnibar (fast paths)

File: frontend/components/QuickActionsBar.tsx

+ { id: 'board-report', label: 'Board Report', action: () => router.push('/executive/reports') },
+ { id: 'verify-chain', label: 'Verify Audit Chain', action: async () => { await fetch('/api/v1/blockchain/verify', { cache: 'no-store' }); router.push('/audit') } },
+ { id: 'new-policy', label: 'New Policy', action: () => router.push('/policy/composer') },


ADD (optional): frontend/components/OmniK.tsx (Ctrl/⌘+K) with jump targets to Executive, Policy, Audit, Predict, FinOps.

PR-08 — Consistent “Labs” badging inside pages

At top of each Labs page, add:

<div className="mb-4 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs">
  <span>Labs</span>
  <span className="opacity-70">Feature preview — functionality limited</span>
</div>

PR-09 — Theming & density nits

Use Comfortable (default) + Compact toggle on large tables (audit/resources).

Keep semantic chips only (Success/Warning/Danger/Info). Remove “NEW/AI/ENHANCED/CRITICAL” labels from the core nav; put Beta chips inside headers only if a page is limited.

Smoke-test checklist (10 minutes)

Login lands on /executive; left-nav shows Executive first, Policy second.

Audit Trail lists real entries; Integrity chip evaluates; Verify Chain banner works; Export Signed Evidence downloads JSON.

Predict shows cards, confidence, ETA; Create Fix PR opens your PR flow (can be a template link).

FinOps & ROI displays numbers; link to ROI board.

Policy Hub loads with scaffolded cards and actions; Composer/Packs/Enforcement pages routable.

Labs menu collapses non-core areas; pages show the Labs badge; search engines won’t index them.

How to apply this quickly

Create a branch (ui/enterprise-cut-2025-09-02), paste the diffs/adds above, and run:
pnpm i (or npm i) → pnpm dev → verify the smoke-test.

If your folder names differ (e.g., apps/web/), keep the same relative structure under that root.

If your API base isn’t /api/v1, set NEXT_PUBLIC_BASE_URL or swap the fetch URLs.

If you want me to tailor the diffs to your exact file layout, paste (or share) the nav component file and the audit page file contents and I’ll output just the precise corrections for those files.

Worked for 36 seconds

I took a look at the container you shared, and unfortunately it doesn’t include any of your actual PolicyCortex UI code (there’s just the slides template in answer.js). Because of that, I can’t run or modify your UI directly.

To move forward, I would suggest the following high‑impact refactors, based on the architecture we discussed earlier. These changes make the UI enterprise‑grade while keeping your core business flows intact:

Executive as the Landing Page. Redirect authenticated users to /executive instead of /tactical, so that C‑suite KPIs, ROI metrics and board reports are the first thing they see. This matches your existing Executive dashboard code.

Top‑level “Policy” Section. In your nav component, replace “Governance Hub” with “Policy” and make it the second item in the list. The policy hub should show coverage, predicted drift, prevented violations and $ impact, with quick actions to create a policy, install packs and view enforcement status.

Audit Trail Wiring. Replace mock audit data with a fetch to /api/v1/blockchain/audit. Add a Verify button per log entry calling /api/v1/blockchain/verify?hash=…, and a banner that verifies the whole chain. Include an “Export Signed Evidence” button to download Merkle roots and signatures as JSON.

Prediction UI. Create a /ai/predictions page that lists predictions returned by /api/v1/predictions (title, ETA, confidence, recommended fix), with a “Create Fix PR” action that opens your DevSecOps pipeline.

FinOps & ROI Dashboard. On /finops, call the backend ROI endpoint and render savings this quarter, forecasted savings and anomalies. Provide a “View ROI Board” button leading to a more detailed ROI analysis.

DevSecOps & Enforcement. Keep your CI/CD gates in a dedicated DevSecOps section, but link each gate back to the policy that governs it. Conversely, in your policy hub, show which gates each policy is attached to.

Labs Route Group. Move experimental pages like Blockchain Explorer, Copilot, Quantum‑Safe and Edge Governance into a (labs) route group. Label them clearly as “Labs” and set robots: { index: false } so they’re not indexed until you’re ready.

Navigation Refactor. Create a central navigation config (navigation.ts) with two arrays: CORE and LABS. Render CORE items as your main nav and LABS as a single collapsed “Labs” entry. Use descriptive labels like “FinOps & ROI”, “Access Governance” and “Resources” instead of vague terms.

Quick Actions & Omnibar. Add shortcuts such as “New Policy”, “Board Report”, “Verify Audit Chain” and “Create Fix PR” to your quick actions bar. Implement an omnibar (⌘/Ctrl‑K) to jump quickly to pages.