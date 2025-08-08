Title: Claim Strategy and Coverage Map — Cross-Domain Governance Correlation Engine

Scope Overview
- Core: Heterogeneous hierarchical governance graph; attention-weighted message passing; ensemble correlation with significance; Monte Carlo impact modeling; tiered caching; real-time ingestion; APIs with RBAC.
- Implemented Supports: `core/src/azure_client_async.rs` (parallel fetch, semaphore), `core/src/cache.rs` (hot/warm/cold tiers), `frontend/lib/api.ts` (hot/warm usage, batching), `backend/services/ai_engine/ml_models/governance_models.py` (graph-style scoring), `scripts/performance-tests.js` (p99 goals), `graphql/gateway.js`.

Independent Claims (IC)
- IC1 System (graph + attention + ensemble + impact + real-time + API)
- IC2 Method (normalize → graph → attention → ensemble → significance → simulate)
- IC3 CRM (instructions implementing IC2)

Dependent Coverage
- Typed attention per edge/domain level; multi-method dependence (Pearson/Spearman/MI/Granger/TE) with calibration; permutation/bootstrap significance; hot/warm/cold cache with promotion; REST/GraphQL with RBAC; digital twin snapshots; concurrency control via semaphores; uncertainty reporting.

Claim Chart (select)
- Graph building, typed entities → Novel in governance cross-domain context; supports hierarchy and time-windowed snapshots.
- Attention MP + statistical ensemble → Non-obvious combination tailored to governance signals.
- Monte Carlo impact with dependency-aware pruners → Differentiates from static correlation.
- Tiered caching + async concurrency meeting 100k+/min → Performance-limited embodiments.

Design-Around Analysis
- Attempt to remove attention: still captured via ensemble + significance + typed graph? Dependent claims secure attention specifics; core IC1 covers general graph + ensemble.
- Replace MC with point estimates: dependent claims cover uncertainty; IC1 still covers impact estimation module.

Enforcement Examples
- Competing platform computes cross-domain correlations using typed graphs with temporal causality and exposes impact recommendations with uncertainty bands under similar throughput—likely reads on IC1/IC2.

Continuation Opportunities
- Specialized edge kernels (causal attention); streaming TE approximations; GPU acceleration kernels; privacy-preserving federated correlation.


