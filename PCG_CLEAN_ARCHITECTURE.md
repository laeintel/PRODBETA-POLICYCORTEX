# PCG Platform - Clean Architecture

## Platform Focus
**Predictive Cloud Governance (PCG)** - A focused platform with three core pillars:
- **PREVENT**: 7-day predictive compliance engine
- **PROVE**: Immutable evidence chain with cryptographic verification
- **PAYBACK**: ROI engine demonstrating 8-15% cost savings

## Simplified Architecture

### Frontend Structure (Next.js)
```
frontend/
├── app/
│   ├── page.tsx           # Executive dashboard
│   ├── prevent/          # Predictions & compliance
│   ├── prove/            # Evidence & audit trail
│   └── payback/          # ROI & cost optimization
├── components/pcg/       # PCG-specific components
│   ├── ForecastCard.tsx
│   ├── EvidenceChain.tsx
│   ├── ROIWidget.tsx
│   └── QuickActions.tsx
├── lib/
│   └── api-client.ts    # Simplified API client (~150 lines)
└── stores/
    └── pcgStore.ts      # Focused state management
```

### Backend Structure (Rust + Python)
```
core/
├── src/
│   ├── main.rs          # Simplified routing (3 pillars only)
│   ├── api/
│   │   ├── predictions.rs
│   │   ├── evidence.rs
│   │   └── roi.rs
│   └── evidence/        # Hash chain implementation
├── prediction/          # PREVENT services
│   ├── ml-engine/       # 7-day predictions
│   ├── drift-detector/  # Config monitoring
│   └── auto-fixer/      # PR generation
├── roi/                 # PAYBACK services
│   ├── cost-calculator/
│   ├── what-if-engine/
│   └── pl-dashboard/
└── evidence/            # PROVE services
    ├── collector/
    ├── hash-chain/
    └── audit-reporter/
```

## API Endpoints (Focused)

### PREVENT Endpoints
- `GET /api/v1/predict/violations` - Get 7-day predictions
- `POST /api/v1/predict/fix` - Generate auto-fix PR
- `GET /api/v1/predict/mttp` - Mean Time To Prevention metrics

### PROVE Endpoints  
- `POST /api/v1/evidence/collect` - Store evidence
- `GET /api/v1/evidence/verify/{hash}` - Verify chain
- `POST /api/v1/evidence/report` - Generate audit report

### PAYBACK Endpoints
- `GET /api/v1/roi/dashboard` - P&L metrics
- `POST /api/v1/roi/simulate` - What-if scenarios
- `GET /api/v1/roi/savings` - Cost savings data

## Removed Components
- ❌ ITSM features
- ❌ DevOps pipelines
- ❌ Security navigation
- ❌ Operations monitoring
- ❌ Generic AI chat
- ❌ Edge/Quantum experiments
- ❌ GraphQL federation
- ❌ Istio service mesh
- ❌ Complex authentication flows
- ❌ Over-engineered monitoring stack

## Technology Stack (Simplified)
- **Frontend**: Next.js 14, React, TypeScript
- **Backend**: Rust (core API), Python (ML/ROI)
- **Database**: PostgreSQL (events), TimescaleDB (predictions)
- **Cache**: Redis (hot data)
- **ML**: Scikit-learn, ONNX Runtime

## Performance Targets
- Prediction latency: <500ms
- Evidence verification: <100ms
- ROI calculation: <1s
- Time-to-value: 15 minutes

## Deployment (Simple)
```bash
# Frontend
cd frontend && npm run build && npm start

# Backend services
python core/prediction/start-prevent-services.py
python core/roi/start-roi-services.py
cargo run --release

# Or use Docker
docker-compose up
```

## Success Metrics
- **MTTP**: < 24 hours
- **Prevention Rate**: 35% auto-fixed
- **ROI**: 250-350% Year 1
- **Cloud Savings**: 8-15%
- **Evidence Coverage**: 95%

The platform is now laser-focused on demonstrating value through prevention, evidence, and ROI - no unnecessary complexity.