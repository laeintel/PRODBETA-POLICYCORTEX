# 15. Risks & Mitigations

## Technical
- Cross‑cloud policy equivalence gaps → document and provide fallbacks; staged rollouts
- Action safety regressions → preflight checks, approvals, kill switches, canaries
- Data volume (telemetry/cost) → tiered retention; ClickHouse/Timescale

## Security/Compliance
- Credential sprawl → JIT/PIM, vault integration, short‑lived tokens
- Evidence integrity → signed artifacts; optional append‑only ledger

## Delivery
- Scope creep → milestone gates; outcomes‑first KPIs
- Multi‑cloud sequencing → Azure first, then AWS, then GCP; keep interfaces stable

## Adoption
- Change management → guided onboarding, quick wins dashboard, ROI calculator
- Auditor acceptance → evidence packs aligned to frameworks; mapping transparency
