# 50 Comprehensive Roasts: Why This Application Wouldn’t Be Bought or Used

These critiques are intentionally direct to expose adoption blockers. Use them as a pre‑mortem checklist to harden product, platform, and trust.

1) Vaporware AI claims
- Promises GPT‑grade “domain intelligence” without reproducible models, datasets, or benchmarks.
- Reads as demo theater; enterprise buyers won’t trust unmeasured AI.

2) Multi‑cloud in name only
- Azure‑centric stubs; no AWS/GCP collectors, schema normalization, or policy translation.
- “Unified governance” is untrue today.

3) Mock data everywhere
- Critical views fall back to fabricated data with no warning.
- Trust collapses if users can’t distinguish real from simulated.

4) Brittle local runtime
- Windows build fails (MSVC), multi‑port rewrites, and manual steps.
- Onboarding friction kills team adoption.

5) Fragile service routing
- Hardcoded ports (8080/8090/4000) and ad‑hoc rewrites.
- No env‑driven discovery; easy to drift and break.

6) Authentication mismatch
- MSAL in frontend; APIs lack consistent token validation and scopes.
- Cross‑service authZ story is incoherent.

7) No tenant isolation
- Data model lacks tenant boundaries/filters.
- Risk of cross‑tenant leakage is a non‑starter.

8) No durable system of record
- Actions/exceptions/audit are in‑memory or discarded.
- Nothing survives restarts, nothing queryable.

9) Toy action orchestrator
- No idempotency, retries, compensation, or approvals.
- Not safe for production changes.

10) Misleading “Remediate” UX
- Buttons imply safety and reversibility that don’t exist.
- Encourages risky, irreversible operations.

11) No approvals/guardrails
- Destructive changes lack workflows or separation of duties.
- Violates change management policies.

12) No evidence pipeline
- “Compliance” without signed/immutable evidence artifacts.
- Auditors will reject it outright.

13) Untamperable logs missing
- No immutability, signatures, or chaining for audit logs.
- Forensics cannot rely on records.

14) Secrets management weak
- Env vars over proper vault + rotation + scoping.
- High blast radius if compromised.

15) Key leakage risks
- No clear boundaries to prevent secrets in logs/bundles.
- Default logging can exfiltrate tokens.

16) Accessibility ignored
- Lacks focus management, ARIA, contrast, keyboard nav.
- Excludes users; fails a11y policies.

17) No i18n/l10n
- English‑only UI.
- Global orgs can’t deploy broadly.

18) Doesn’t scale in UI
- No virtualization for large tables; no paging strategy.
- UI will lag/crash on real data volumes.

19) Primitive search/filters
- No faceted filters, saved searches, or advanced queries.
- Daily workflows feel clumsy and slow.

20) Offline/conflict strategy absent
- No optimistic concurrency or offline queues.
- Real‑world network issues break flows.

21) Observability lip service
- No end‑to‑end tracing; sparse metrics.
- Ops will fly blind during incidents.

22) No SLOs/error budgets
- Reliability undefined; no availability targets.
- Unacceptable for critical governance systems.

23) DR not designed
- No backups, replication, or restore drills.
- Unknown RPO/RTO.

24) Data retention undefined
- No TTL, retention, or legal hold policies.
- Compliance blockers for many buyers.

25) Control frameworks unmapped
- Features not mapped to SOC2/ISO/NIST controls.
- “Compliance” lacks auditability.

26) Policy engine surface deep but shallow
- No assignment graph/inheritance/effect parity.
- “Policy” reduced to JSON rendering.

27) No enforcement path
- No bridge to Azure Policy/Terraform for drift correction.
- Posture management without control.

28) IAM graph missing
- No identity/permission graph or attack paths.
- Security value is superficial.

29) FinOps without data
- No CUR ingestion, no unit economics, no FOCUS compliance.
- Cost claims lack legs.

30) No automated testing
- Unit/e2e/load tests missing.
- High regression risk.

31) No threat model
- STRIDE/LINDDUN absent; critical risks unaddressed.
- Security teams will block.

32) Supply chain unknown
- No SBOM, CVE scanning, or provenance (SLSA).
- Fails vendor security reviews.

33) Complex deploy, weak docs
- Multi‑lang stack with sparse runbooks.
- Hard to operate; high toil.

34) No migration/import story
- Can’t ingest existing policies/evidence/exceptions.
- Switching costs block adoption.

35) Pricing/ROI unclear
- No ROI calculators, outcomes, or packaging.
- Economic buyer can’t justify purchase.

36) Azure lock‑in signals
- Object model and flows are Azure‑biased.
- Multi‑cloud buyers will churn.

37) Patents ≠ product
- IP docs don’t prove capability.
- Buyers want working, measured features.

38) Thin documentation
- Missing operator/developer guides and SLAs.
- Scalability and supportability unclear.

39) No extension model
- No plugin/event contracts for partners.
- Ecosystem potential near zero.

40) Immature data versioning
- No schema versioning/migrations policy.
- Breaking changes likely.

41) Eventing aspiration only
- No bus, contracts, or consumers.
- “Real‑time” isn’t real.

42) Performance unknown
- No load/soak/chaos benchmarks.
- Scalability claims unsubstantiated.

43) Privacy/residency unaddressed
- No DPA templates; data flows unmapped.
- Legal will slow or stop procurement.

44) UX coherence gaps
- Navigation/drill‑downs inconsistent; weak IA.
- Users struggle to find/complete tasks.

45) Roles not enforced
- RBAC lacks end‑to‑end checks and least privilege.
- Dangerous pathways exposed.

46) Exceptions lifecycle absent
- No expiry/re‑certification/evidence.
- GRC teams won’t accept it.

47) Change tickets not integrated
- No CAB/RFC linkage or freeze windows.
- Violates enterprise change policies.

48) No community or references
- No design partners, references, or champions.
- Social proof missing.

49) Weak differentiation messaging
- Vision broad; feature reality narrow.
- Competitors with focus will outposition.

50) “Trust us” culture
- Trust demanded, not earned via controls, tests, and results.
- Enterprises will pass.
