Title: Technical Specification Summary — Cross-Domain Governance Correlation Engine

APIs
- Correlations: GET /api/v1/correlations?domains=... returns correlation edges with strength, direction, lag, p-values, CIs.
- What-if: POST /api/v1/whatif { action, scope } returns Δmetrics with percentile bands.

Performance Targets
- Ingestion ≥100k events/min; p99 <1.5s (see `scripts/performance-tests.js`).
- Median correlation query ≤500ms under 200 VUs.

Security
- RBAC/Scopes enforced; audit logs include model version and snapshot hash.

Caching
- Hot: 10–30s; Warm: ~5m; Cold: 30–120m; promotion on access; batch ops (see `core/src/cache.rs`).

Data Model
- Nodes: {Resource, Policy, RoleAssignment, Identity, NetworkRule, CostBucket, Metric}.
- Edges: {enforces, assignedTo, connectedTo, contributesTo, dependsOn} with timestamps.

ML
- Typed-attention GNN; correlation ensemble (Pearson/Spearman/MI/Granger/TE) with calibration; MC impact simulation.


