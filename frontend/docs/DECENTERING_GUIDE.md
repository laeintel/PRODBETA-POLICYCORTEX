# De-Centering Guide (Prioritize Impact, Park Experiments)

**Goal:** Make the product's unique moats (tamper-evident audit + predictive + ROI) the star. Keep experimental/partially wired areas discoverable but clearly secondary.

## 1) Route Grouping (no URL change) and Robots

- Create two route groups under `app/`:
  - `app/(core)/…` → `/tactical`, `/audit`, `/finops`, `/rbac`, `/resources`, `/devsecops/*`
  - `app/(labs)/…` → `/copilot`, `/quantum`, `/edge`, `/blockchain` (until verify is fully wired)
- Add `app/(labs)/layout.tsx` with:

```ts
// app/(labs)/layout.tsx
export const metadata = {
  robots: { index: false, follow: true }, // keep Labs pages out of search
  other: { 'x-section': 'labs' }
}
export default function LabsLayout({ children }: { children: React.ReactNode }) {
  return <div data-section="labs">{children}</div>
}
```

Route groups do not change URLs, but they let us style/flag Labs consistently and apply robots:noindex.

## 2) Navigation: move Labs behind a single menu

Add a thin config so the UI only renders Core items in the sidebar/top-nav; Labs collapses into one entry.

```ts
// frontend/config/navigation.ts
export type NavItem = { label: string; href: string; ready?: boolean; badge?: 'Labs'|'Beta' }
export const CORE: NavItem[] = [
  { label: 'Tactical', href: '/tactical', ready: true },
  { label: 'Audit Trail', href: '/audit', ready: true },
  { label: 'FinOps', href: '/finops', ready: true },
  { label: 'RBAC', href: '/rbac', ready: true },
  { label: 'Resources', href: '/resources', ready: true },
  { label: 'DevSecOps', href: '/devsecops/pipelines', ready: true },
]
export const LABS: NavItem[] = [
  { label: 'Blockchain', href: '/blockchain', badge: 'Labs' },
  { label: 'Copilot', href: '/copilot', badge: 'Labs' },
  { label: 'Quantum', href: '/quantum', badge: 'Labs' },
  { label: 'Edge', href: '/edge', badge: 'Labs' },
]
```

Your test nav map already enumerates these paths; this file centralizes order/flags.

Render CORE normally; render one "Labs" item that opens a sub-menu with LABS.

## 3) Inline badging for Labs pages

At the top of each Labs page:

```tsx
{/* Labs badge */}
<div className="mb-4 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs">
  <span>Labs</span>
  <span className="opacity-70">Feature preview — functionality limited</span>
</div>
```

## 4) Feature flags (optional)

Expose a single env var to hide Labs in production:

```env
NEXT_PUBLIC_SHOW_LABS=false
```

And guard render:

```tsx
{process.env.NEXT_PUBLIC_SHOW_LABS === 'true' && <LabsMenu items={LABS} />}
```

## 5) "Wire-First" micro-tasks (what moves out of Labs fastest)

**Blockchain → Core** once Verify is truly live:
- UI: add Verify button on each audit entry (calls `/api/v1/blockchain/verify/{hash}`) and show Merkle proof / signature valid chips.
- Export: "Download signed evidence (JSON)" that includes entry hashes and Merkle root.

**FinOps:** connect top widgets to `/api/v1/predictions` and backend ROI then keep it Core.

**Copilot/Quantum/Edge** stay Labs until API calls replace mocks (as flagged in Validation Report).

## 6) What not to surface by default

Any page still driven by simulated data (listed in VALIDATION_REPORT.md) remains in Labs until endpoints are connected.

---

## Tiny code nudges (surgical, not rewrites)

### 1) **Audit Trail → live blockchain data (replace mock):**

In `frontend/app/audit/page.tsx`, replace the local generator with:

```ts
// fetch on mount
const res = await fetch('/api/v1/blockchain/audit', { cache: 'no-store' })
const entries = await res.json() // map to UI shape
// Add Verify action per entry:
await fetch(`/api/v1/blockchain/verify?hash=${encodeURIComponent(entry.hash)}`)
```

Then render "Integrity OK" / "Tampering detected" from verification payload fields (chain_integrity, signature_valid, merkle_proof_valid).

### 2) **FinOps cards → predictions & ROI:**

Call `GET /api/v1/predictions` for forecast tiles; call backend `/api/v1/executive/roi` to fill the CFO snapshot.

### 3) **Keep everything auth-safe:**

Leave `frontend/middleware.ts` as-is so every route is gated; your set-cookie route is already in place for creating the httpOnly session.

## Why this works (fastest impact)

- It centers your moats (tamper-evident audit + predictive + ROI) in the first 60 seconds of a demo and hides noise until it's real.
- It matches the real wiring status already documented (mocks vs. live), so the work is small, visible, and compounding.

If you want, I can follow up with exact PR-ready diffs for audit/page.tsx (swap to /blockchain/audit + Verify), and a minimal navigation.ts that your existing nav reads from.