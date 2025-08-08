Title: Technical Specification Summary — Predictive Policy Compliance Engine

Signals
- Config vectors; policy deltas; RBAC changes; NSG updates; cost/utilization trends.

Models
- VAE drift score; STL components; ensemble predictors (sequence + booster); Bayesian averaging; uncertainty intervals.

Risk/MCDA
- Inputs: likelihood, impact, detectability, business weight; outputs: priority score and rank.

APIs
- GET /api/v1/predictions; GET /api/v1/drift; POST /api/v1/recommendations

Performance
- 24–168h horizons; AUROC ≥0.90; p99 ≤1.5s; 100k+ events/min.


