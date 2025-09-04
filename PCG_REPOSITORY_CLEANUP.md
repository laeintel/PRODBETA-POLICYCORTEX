# PolicyCortex Repository Cleanup for PCG Pivot

## Executive Summary
This document outlines the repository cleanup strategy for pivoting PolicyCortex to the Predictive Cloud Governance (PCG) platform focused on the three pillars: PREVENT, PROVE, and PAYBACK.

## GitHub Issues Management Summary

### Issues Closed (37 total)
- Closed 33 legacy issues not aligned with PCG vision
- Closed generic governance features, UX enhancements, and old roadmap items
- Retained 28 issues that can be repurposed for PCG pillars

### New Issues Created (10 total)

#### Epic Issues (3)
1. **#161** - [PCG EPIC] PREVENT Pillar - 7-Day Predictive Engine
2. **#162** - [PCG EPIC] PROVE Pillar - Immutable Evidence Chain
3. **#163** - [PCG EPIC] PAYBACK Pillar - Governance P&L and ROI Engine

#### Feature Issues (7)
1. **#164** - 7-Day Prediction Engine
2. **#165** - Immutable Evidence Chain with Cryptographic Signing
3. **#166** - Governance P&L Dashboard with Real-time ROI
4. **#167** - CI/CD Gates Integration with Shift-Left Predictions
5. **#168** - Auto-Fix PR Generation for Predicted Issues
6. **#169** - What-If Simulator for Governance Changes
7. **#170** - CNAPP Integration Hub (Wiz/Prisma/Orca)

### Issues to Keep and Repurpose (28)
- **Evidence/Compliance**: #104, #31, #60, #61, #66, #21
- **Security/Predictions**: #107, #30, #56, #57, #58, #59, #20, #81, #84, #85
- **Cost/ROI**: #29, #41, #52, #53, #54, #55, #88
- **Platform Foundations**: #106, #108

## Repository File Cleanup Recommendations

### Directories to REMOVE

#### 1. Generic Feature Directories
```
/frontend/app/ai/          # Generic AI features not prediction-focused
/frontend/app/tactical/    # Tactical views not aligned with PCG
/frontend/app/rbac/        # Generic RBAC management
/graphql/                  # GraphQL federation (overly complex for MVP)
```

#### 2. Legacy Documentation
```
/design docs/total-docs/Roadmap_*.md    # Old roadmap files (except FinOps, Security, Compliance)
/training instructions/                  # Generic training docs
/design docs/advance docs/              # Over-engineered documentation
```

#### 3. Non-Essential Services
```
/backend/services/model_server/         # Generic model server
/backend/mlops/                         # Generic MLOps (rebuild for predictions)
/infrastructure/disaster-recovery/      # Premature optimization
/kubernetes/istio/                       # Service mesh (not needed for MVP)
```

### Directories to KEEP and ENHANCE

#### 1. Core Prediction Infrastructure
```
/core/                                   # Rust core (refactor for predictions)
/backend/services/ai_engine/            # ML models (focus on predictions)
/backend/services/api_gateway/          # API gateway (add prediction endpoints)
```

#### 2. Evidence Chain Components
```
/core/src/cqrs/                         # CQRS for event sourcing
/infrastructure/database/               # Database for evidence storage
/monitoring/                            # For evidence collection
```

#### 3. ROI/Financial Components
```
/scripts/dora-metrics/                  # DORA metrics for ROI
/monitoring/grafana/dashboards/        # Financial dashboards
```

#### 4. Patent Documentation
```
/NON-PROVISIONAL/Patent4_PredictivePolicyCompliance/  # Core PCG patent
/design docs/patent-applications/Patent_4_*          # Predictive engine specs
```

### Files to CREATE

#### 1. PCG-Specific Documentation
```
/docs/PCG_ARCHITECTURE.md              # New architecture focused on 3 pillars
/docs/PREVENT_PILLAR.md                # Prediction engine documentation
/docs/PROVE_PILLAR.md                  # Evidence chain documentation
/docs/PAYBACK_PILLAR.md                # ROI engine documentation
```

#### 2. Integration Guides
```
/integrations/cnapp/                   # CNAPP connector implementations
/integrations/cicd/                    # CI/CD gate integrations
/integrations/audit/                   # Audit tool integrations
```

#### 3. Prediction Models
```
/models/predictions/                   # 7-day prediction models
/models/roi/                          # ROI calculation models
/models/evidence/                     # Evidence scoring models
```

## Migration Strategy

### Phase 1: Cleanup (Week 1)
1. Archive non-PCG directories to `/archive/` folder
2. Remove generic features from frontend
3. Clean up unused dependencies

### Phase 2: Refactor (Week 2)
1. Refactor core API to focus on predictions
2. Update ML service for prediction engine
3. Modify frontend to show 3 pillars

### Phase 3: Build (Weeks 3-4)
1. Implement 7-day prediction engine
2. Build evidence chain with crypto signing
3. Create P&L dashboard

### Phase 4: Integrate (Week 5)
1. Add CI/CD gate integrations
2. Connect CNAPP platforms
3. Enable auto-fix PR generation

## Success Metrics

### Technical Metrics
- **Prediction Accuracy**: >95% for 7-day forecasts
- **Evidence Integrity**: 100% tamper-proof with crypto signatures
- **ROI Calculation**: Real-time with <5 second latency
- **Integration Coverage**: 3+ CNAPP platforms, 3+ CI/CD systems

### Business Metrics
- **Customer Acquisition**: 10x improvement with clear value prop
- **Time to Value**: <30 minutes to first prediction
- **ROI Demonstration**: 10x ROI within 90 days
- **Market Differentiation**: Only platform with true 7-day predictions

## Recommended Immediate Actions

1. **Archive Legacy Code**: Move non-PCG code to archive branch
2. **Update README**: New messaging focused on Prevent/Prove/Payback
3. **Create Landing Page**: New frontend showing 3 pillars prominently
4. **Build Demo**: 7-day prediction demo with sample data
5. **Document APIs**: New API docs for prediction/evidence/ROI endpoints

## Repository Structure After Cleanup

```
policycortex/
├── core/                      # Prediction engine (Rust)
├── frontend/                  # PCG dashboard (Next.js)
│   ├── app/
│   │   ├── prevent/          # Prediction UI
│   │   ├── prove/            # Evidence UI
│   │   └── payback/          # ROI UI
├── backend/
│   └── services/
│       ├── prediction/       # 7-day prediction service
│       ├── evidence/         # Evidence chain service
│       └── roi/              # ROI calculation service
├── integrations/             # External integrations
├── models/                   # ML models
├── docs/                     # PCG documentation
└── patents/                  # Patent documentation
```

## Conclusion

This cleanup will transform PolicyCortex from a broad governance platform into a focused Predictive Cloud Governance solution with clear differentiation and measurable value. The three pillars (Prevent/Prove/Payback) provide a compelling narrative that resonates with enterprise buyers looking for proactive, evidence-based governance with demonstrable ROI.

**Next Steps:**
1. Review and approve cleanup plan
2. Create archive branch for legacy code
3. Execute phased cleanup over 2 weeks
4. Begin building PCG-specific features
5. Update all marketing/documentation to reflect PCG positioning