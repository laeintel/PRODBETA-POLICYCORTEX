# 6. AI & Knowledge Base

## 6.1 Components
- Domain Expert (Python) – deep rules/heuristics + pattern knowledge
- GPT‑5 Integration – NL analysis, generation, predictions (fallback to expert if unavailable)
- Multi‑Cloud Knowledge Base – service mappings, patterns, controls, equivalence

## 6.2 Training & Context
- Use existing training specs and data; add retrieval into tenants’ facts for grounded responses
- Context windows constructed from resources, policies, costs, violations, actions

## 6.3 Capabilities
- Policy generation (multi‑cloud equivalence)
- Compliance prediction (drift, conflicts, effective policies)
- FinOps optimization strategies (commitments, rightsizing)
- Security exposure analysis (attack paths)

## 6.4 Guardrails
- Deterministic rules backstop; never run actions from LLMs without orchestrator safeguards
- Hallucination defenses: evidence requirement; confidence thresholds; red teaming

## 6.5 APIs
- POST `DEEP_API_BASE/api/v1/policies/generate`
- POST `DEEP_API_BASE/api/v1/analyze` (prompt‑engineered with evidence requests)
