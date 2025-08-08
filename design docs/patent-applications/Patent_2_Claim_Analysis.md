Title: Claim Strategy and Coverage Map — Conversational Governance Intelligence System

Scope Overview
- Domain NLU (transformer), dialogue state graph, policy synthesis, saga orchestration, multimodal output, secure tokenization.

Independent Claims
- System (NLU + state graph + synthesis + saga + latency/caching profile), Method, CRM.

Dependent Claims
- Context window and summarization; framework-aware validation; exception handling and parameters; compensation steps and audit; voice interface; token scopes.

Design-Arounds
- Remove policy synthesis → independent method/system still cover orchestration of governance actions; dependent claims ensure synthesis embodiment.
- Stateless chat → dependent claims tie to graph-based state.

Enforcement Examples
- Assistant that generates cloud policies from NL with validation and executes via orchestrated steps likely reads on core claims.

Implementation Evidence
- `frontend/lib/api.ts` useConversation/context; MSAL token; `domain_expert.py` policy generation; voice UI component.


