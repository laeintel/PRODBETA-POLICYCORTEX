# 10. Frontend UX

## 10.1 Principles
- Clarity, explainability, and speed; minimal clicks to evidence or action
- Deep links for every entity; breadcrumbs and drill‑downs; shareable URLs

## 10.2 Dashboard
- KPI tiles; proactive actions; real‑time activity; AI learning status
- Tiles deep‑link to filtered lists (e.g., `/dashboard?module=policies` or `/policies`)

## 10.3 Modules
- Policies, RBAC, Costs, Network, Resources — list → detail → action drawer
- Detail pages (`/resources/[id]`, `/policies/[id]`) show metadata, violations, recommendations, evidence

## 10.4 Action Drawer
- Sections: Summary, Preflight Diff, Blast‑Radius, Approvals, Live Progress, Evidence
- Uses: POST `/actions`, GET status, SSE events, download evidence
- Failure path: show diagnostics + rollback CTA

## 10.5 Conversational Assistant
- Suggestions map to actions; “approve and run” flows; voice + chat with context chips

## 10.6 Accessibility & Performance
- Lazy streaming; skeletons; keyboard nav; color‑contrast; toast errors

## 10.7 Deep-link Patterns
- Resource: `/resources/[id]`
- Policy: `/policies/[id]`
- Action: `/actions/[id]` (read‑only view with SSE replay)
