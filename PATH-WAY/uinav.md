What stays vs. what moves (actionable triage)
Current label (screenshot)	Keep in Core?	New label / Placement	Why	Immediate action
Dashboard	Keep	Home (or Tactical)	First‑impression hub for your 3 moats: Predict → Prevent, Tamper‑Evident Audit, CFO ROI.	Make this the default route. Show 3 hero KPI cards + single primary actions.
Executive (NEW)	Conditional	If live numbers exist: Executive; else fold into FinOps & ROI	If you have a working ROI endpoint, keep it; otherwise it fragments value.	If not fully wired, replace with an “Executive” section on FinOps and a Home hero card.
FinOps (AI)	Keep	FinOps & ROI	One of your differentiators (spend + governance outcomes).	Wire to predictions & ROI endpoints; show savings, forecast, anomalies.
Governance	Move to Labs or merge	Merge core policy work into DevSecOps until packs are mature	Avoid a top‑level that’s thin or overlaps with DevSecOps/policy‑as‑code.	Keep a “Policies (Beta)” tab inside DevSecOps for now.
Security & Access	Keep	Access Governance (RBAC)	Identity/permissions are core for audits and drift prevention.	Keep table + right‑drawer pattern; add saved views.
Operations	Keep (rename)	Resources	“Operations” is vague; “Resources” matches inventory mental models.	Keep filters (cloud/account/service/tag), bulk actions.
DevOps (ENHANCED)	Keep (rename)	DevSecOps	Clearer value: CI/CD gates, policy‑as‑code, auto‑fix PRs.	Add “Gate results” list + “Create Fix PR” action.
Cloud ITSM (NEW)	Move to Labs	Labs → Cloud ITSM	Large scope; not core to predictive governance story.	Hide from main nav; keep discoverable in Labs.
AI Intelligence	Remove as top‑level	Replace with Predict page	“AI Intelligence” is marketing‑vague. “Predict” is your moat.	Create Predict (list + drawer) using /predictions; move “AI” copy there.
Governance Copilot (AI)	Move to Labs	Labs → Copilot	Chat UX is secondary unless it explains predictions & proposes PRs.	Keep link; badge “Labs”.
Blockchain Audit (NEW)	Merge	Merge into Audit Trail	Two audit entries confuse users. Blockchain = how you secure audit, not a separate product.	Fold verification into Audit Trail: per‑row Verify, chain Verify banner, Evidence export.
Quantum‑Safe Security (CRITICAL)	Move to Labs	Labs → Quantum‑Safe	Great R&D, not day‑1 value for most buyers.	Keep as technical brief + endpoints preview.
Edge Governance (NEW)	Move to Labs	Labs → Edge Governance	Early/adjacent; avoid crowding core.	Hide from main nav; show only in Labs.
Audit Trail (NEW)	Keep	Audit Trail (tamper‑evident)	Star of the show together with Predict + ROI.	Replace mocks with live blockchain audit + Verify + Export signed evidence.
Settings	Keep	Settings	Standard.	Nothing special; keep sensible grouping.

Badge policy: drop “NEW/AI/ENHANCED/CRITICAL” labels in the core nav (they create noise). Use a subtle “Beta” chip inside page headers only where functionality is genuinely limited. Reserve “Labs” for items hidden behind the Labs menu.

Final nav (clean, enterprise, 7–8 items)

Core (left rail):

Home (Tactical)

Audit Trail (with blockchain verification inside)

Predict (Predictions list + detail)

FinOps & ROI

Access Governance (RBAC)

Resources

DevSecOps

Settings

Labs (single collapsed entry):

Blockchain Explorer (until verify is fully wired end‑to‑end, then it disappears because Audit Trail covers it)

Copilot

Cloud ITSM

Quantum‑Safe

Edge Governance

Why this specific cut

Centers your moats: Predict → Prevent, tamper‑evident audit, and CFO‑visible ROI are the first three tiles a buyer sees.

Reduces friction: 15 ➜ 8 core items; fewer decisions = faster orientation and higher demo clarity.

Avoids duplication: no separate “Blockchain Audit” vs. “Audit Trail”; “AI Intelligence” becomes “Predict” (concrete outcome, not jargon).

Keeps ambition visible, not distracting: Labs holds powerful but not‑yet‑central capabilities.

Two tiny implementation steps (so your coder ships this in a day)

Centralize nav config
Create frontend/config/navigation.ts and list CORE vs LABS arrays; render Labs as a single menu item that expands inline.

Route‑group Labs
Use Next.js route groups app/(core)/… and app/(labs)/…, and in app/(labs)/layout.tsx set robots: { index: false } and show a small “Labs” badge near the page title.

TL;DR

Keep: Home, Audit Trail, Predict, FinOps & ROI, Access Governance, Resources, DevSecOps, Settings.

Merge: Blockchain Audit → Audit Trail.

Hide behind Labs: Governance (until robust), Copilot, Cloud ITSM, Quantum‑Safe, Edge Governance.

Rename to reduce friction: “Dashboard”→Home, “DevOps”→DevSecOps, “Operations”→Resources, “AI Intelligence”→Predict, “Security & Access”→Access Governance.