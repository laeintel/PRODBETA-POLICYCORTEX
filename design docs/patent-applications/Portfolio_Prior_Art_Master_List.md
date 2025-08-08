Title: PolicyCortex Patent Portfolio — Consolidated Prior Art Master List (Technical and Industry)

Purpose
Ground each application’s differentiation with a curated set of representative references across policy engines, governance orchestration, graph analytics, causality/uncertainty, and optimization/orchestration.

I. Cloud Governance Platforms and Policy Engines (Industry)
- Azure Policy Insights, Azure Policy/Blueprints (Microsoft)
- AWS Config Rules, AWS Control Tower (Amazon)
- Google Cloud Security Command Center (Google)
- Open Policy Agent (OPA)
- Cloud Custodian (Capital One / Open source)
- HashiCorp Sentinel (Policy as Code)

Notes on Differentiation
- Siloed domain scope vs. unified cross-domain correlation/impact; lack of typed heterogeneous graph and attention; limited or no uncertainty quantification and Monte Carlo impact; limited multi-objective optimization or saga-based rollback orchestration.

II. Graph Learning and Causality (Academic/Methods)
- Graph Attention Networks (GAT); Heterogeneous Graph Attention Networks (HAN)
- Message passing neural networks; typed/relational GNNs
- Granger causality for time series
- Transfer Entropy (information-theoretic)

Notes on Differentiation
- General-purpose GNNs vs. governance-typed hierarchical graphs with domain-specific attention and significance-pruned correlation ensembles.

III. Uncertainty and Ensemble Methods
- Bayesian Model Averaging (BMA)
- Bootstrap/residual-based uncertainty; Monte Carlo dropout

Notes on Differentiation
- Application to governance impact simulation with dependency-aware graph propagation and confidence intervals per metric.

IV. Temporal Analysis and SPC
- STL decomposition (trend/seasonality/residual)
- Statistical Process Control (SPC), 3-sigma rules

Notes on Differentiation
- Joint use with VAE-based drift and calibrated violation forecasting specific to compliance signals.

V. Optimization and Orchestration
- NSGA-II (Pareto multi-objective optimization)
- Saga pattern for distributed transactions

Notes on Differentiation
- Governance-constrained optimization (compliance non-regression, SLO/budget caps, least-privilege) and twin-validated changes with rollback/audit lineage.

Cross-Application Mapping
- Patent 1 (Correlation): typed GNN + causality ensemble + MC impact vs. domain-siloed tools and generic GNN papers
- Patent 2 (Conversational): domain NLU + policy synthesis + saga vs. generic chatbots and script-based policy tools
- Patent 3 (Unified Platform): hierarchical ensemble + twin + Pareto + saga vs. point solutions and AIOps w/o uncertainty/twin
- Patent 4 (Predictive Compliance): VAE drift + STL/ensemble + fuzzy/MCDA + constrained recommendations vs. reactive scanning and anomaly-only detectors

Disclosure Practices
- Cite representative product docs and method papers in IDS; explain governance-specific adaptations and integrated pipeline producing practical technical improvements.


