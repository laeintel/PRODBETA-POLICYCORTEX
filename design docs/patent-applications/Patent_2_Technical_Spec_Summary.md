Title: Technical Specification Summary — Conversational Governance Intelligence System

Interfaces
- POST /api/v1/conversation { query, context, session_id } → { response, intent, confidence, suggested_actions, generated_policy }

Latency Targets
- Chat POST uncached; p95 < 2s under 100–200 VUs; metrics endpoints hot cached.

Security
- OAuth delegated scopes; per-intent permission checks; audit logs with redaction.

Policy Synthesis
- Provider-specific rules; parameters; remediation; exceptions; monitoring (see `domain_expert.py`).

Dialogue State
- Last N turns, entities, and intents; summarization for long contexts (see `useConversation()` packing).


