Title: Claim Strategy and Coverage Map — Predictive Policy Compliance Engine

Scope Overview
- VAE drift detection + SPC; STL decomposition; motif/regime; ensemble forecasts with Bayesian averaging; fuzzy + MCDA risk; CBR + constraints recommendations.

Independent Claims
- System, Method, CRM.

Dependent Claims
- Thresholding via rolling quantiles; residual bootstrap calibration; maintenance window and dependency constraints; explanation artifacts.

Design-Arounds
- Swap VAE with autoencoder → dependent claims enumerate VAE; IC covers learned representation drift scoring more generally.

Evidence
- Models in `governance_models.py`; expert rules and remediation in `domain_expert.py`; ingestion/caching in core.


