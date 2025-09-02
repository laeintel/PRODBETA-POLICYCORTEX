✅ Updated left‑nav (final)

Core (top‑level)

Home (Tactical)

Policy ← NEW: elevate to core

Audit Trail (tamper‑evident, blockchain verify inside)

Predict (predictions list + detail)

FinOps & ROI

Access Governance (RBAC)

Resources

DevSecOps (pipelines, gates)

Settings

Labs (one collapsed menu)

Copilot, Cloud ITSM, Quantum‑Safe, Edge Governance, Blockchain Explorer (only until Audit Trail verify is fully wired)

Why: Policy is the control plane; DevSecOps is just one enforcement surface. Keeping Policy as #2 makes the story crystal clear: Define policy → see/prevent drift → prove it (audit) → show ROI.

🧭 What goes under Policy (pages & behavior)

Goal: let a CISO/Platform Lead define & enforce guardrails in minutes, with predictive impact and ROI visible before enabling.

1) Policy Hub (/policy)

Hero cards:

Coverage (accounts in scope, % resources covered)

Predicted drift (next 7 days)

Violations prevented (last 30 days)

Estimated cost impact (savings window)

Primary actions: New Policy ▸ wizard, Install Policy Pack, Run Simulation

Lists: Active Policies, Exceptions expiring soon, Recent prevention events

2) Policy Packs (/policy/packs)

Curated baselines: NIST 800‑53 Moderate, CIS AWS/Azure/K8s, SOC2, HIPAA, FedRAMP

Each pack shows: controls, defaults, predicted drift reduction, estimated $$ impact.

3) Composer (/policy/composer)

Step 1: Baseline (start from pack or blank)

Step 2: Scope (accounts, tags, resource selectors)

Step 3: Parameters (e.g., regions allowed, encryption on, KMS key policy)

Step 4: Enforcement mode (Observe → PR‑Block → Auto‑fix)

Step 5: Preview & Simulate (predicted drift, risk avoided, $ impact), then Create Policy

Inline YAML/rego view with linting and tests; Explain tooltip shows features feeding the prediction.

4) Enforcement & Gates (/policy/enforcement)

Where each policy is bound to one or more gates: CI (IaC checks), CD (deploy checks), Runtime (preventive guardrails)

Status chips: PASS/FAIL, MTTP (Mean Time To Prevention), auto‑fix PR links.

5) Exceptions & Waivers (/policy/exceptions)

Create time‑boxed exceptions with approvers, scope, attestation text, and review cadence.

Badges in tables if a violation is covered by a valid exception.

6) Evidence & Mapping (/policy/evidence)

Each control maps to audit evidence with Merkle‑root + signature (pulled from Audit Trail), exportable as JSON/CSV/PDF.

🏁 What moves out of Policy (or links out)

CI/CD details remain in DevSecOps, but are linked from Policy ▸ Enforcement & Gates.

Blockchain Explorer stays in Labs; Audit Trail contains all verify actions users need.

✨ Home hero (polish to make Policy obvious)

Top of Home shows three cards (each with one action):

Policy posture: Coverage, predicted drift → Open Policy

Audit integrity: Verify status → Open Audit Trail

CFO ROI: Savings to‑date → Open FinOps

This makes Policy discoverable in the first 10 seconds.

🔧 Tiny, PR‑ready changes (no rewrites)

1) Centralize nav and add “Policy”

// frontend/config/navigation.ts
export type NavItem = { label: string; href: string; ready?: boolean; badge?: 'Beta'|'Labs' }

export const CORE: NavItem[] = [
  { label: 'Home', href: '/tactical', ready: true },
  { label: 'Policy', href: '/policy', ready: true },               // ← NEW
  { label: 'Audit Trail', href: '/audit', ready: true },
  { label: 'Predict', href: '/ai/predictions', ready: true },
  { label: 'FinOps & ROI', href: '/finops', ready: true },
  { label: 'Access Governance', href: '/rbac', ready: true },
  { label: 'Resources', href: '/resources', ready: true },
  { label: 'DevSecOps', href: '/devsecops/pipelines', ready: true },
  { label: 'Settings', href: '/settings', ready: true },
]

export const LABS: NavItem[] = [
  { label: 'Blockchain Explorer', href: '/blockchain', badge: 'Labs' },
  { label: 'Copilot', href: '/copilot', badge: 'Labs' },
  { label: 'Cloud ITSM', href: '/itsm', badge: 'Labs' },
  { label: 'Quantum‑Safe', href: '/quantum', badge: 'Labs' },
  { label: 'Edge Governance', href: '/edge', badge: 'Labs' },
]


2) Add Policy hub route (scaffold)

// app/(core)/policy/page.tsx
export default function PolicyHub() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Policy</h1>
        <p className="text-muted-foreground">Define, simulate, enforce, and evidence cloud guardrails.</p>
      </header>

      {/* Hero cards: Coverage / Predicted drift / Prevented / $ impact */}
      {/* Buttons: New Policy, Install Policy Pack, Run Simulation */}

      {/* Lists: Active Policies, Exceptions expiring soon, Recent prevention events */}
    </div>
  )
}


3) Add stubs (can be empty pages initially)

app/(core)/policy/packs/page.tsx
app/(core)/policy/composer/page.tsx
app/(core)/policy/enforcement/page.tsx
app/(core)/policy/exceptions/page.tsx
app/(core)/policy/evidence/page.tsx


4) Link DevSecOps gates back to Policy

On each gate detail, show “Managed by Policy › [policy name]” linking to /policy/enforcement?policy=<id>.

📈 Policy KPIs (show on Policy Hub and Home)

Coverage (% resources/accounts under policy)

Predicted drift (7‑day) by framework/area

Violations prevented (30/90‑day) & MTTP

Estimated $ impact (savings / avoidance) tied to each policy

Exceptions aging (count expiring within 7/30 days)

👩‍💻 Implementation order (fastest path)

Policy Hub page + nav, with live metrics if available (can start with placeholders and wire quickly).

Composer wizard (MVP): baseline → scope → parameters → enforcement → simulate → create.

Enforcement & Gates: list bindings; show last gate status; deep link to DevSecOps gate details.

Exceptions table + create flow (expiry, approver, scope).

Evidence: reuse Audit Trail verification data for each control; export JSON/CSV.

✂️ What to de‑center now (to keep Policy prominent)

Keep Copilot, Cloud ITSM, Quantum‑Safe, Edge in Labs.

Remove “NEW/AI/CRITICAL” badges from core nav; use a small Beta chip inside headers only when truly limited.

Merge “Blockchain Audit” into Audit Trail; verification is a feature, not a separate product.