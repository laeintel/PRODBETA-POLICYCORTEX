# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PolicyCortex is an AI-powered Azure governance platform with four patented technologies:
1. Cross-Domain Governance Correlation Engine (Patent 1)
2. Conversational Governance Intelligence System (Patent 2) 
3. Unified AI-Driven Cloud Governance Platform (Patent 3)
4. Predictive Policy Compliance Engine (Patent 4)

## Architecture
- **Backend**: Rust modular monolith (core/) using Axum framework with async/await
- **Frontend**: Next.js 14 (frontend/) with App Router, Server Components, Zustand state management
- **GraphQL**: Apollo Federation gateway (graphql/) for unified API
- **AI Services**: Python-based domain expert AI (backend/services/ai_engine/)
- **Edge Functions**: WebAssembly functions (edge/) for sub-millisecond inference
- **Databases**: PostgreSQL (main), EventStore (event sourcing), DragonflyDB (Redis-compatible cache)

## Essential Commands

### Complete Testing (Recommended)
```bash
# Test everything on Windows
.\scripts\testing\test-all-windows.bat

# Test everything on Linux/Mac  
./scripts/testing/test-all-linux.sh
```

### Development
```bash
# Start full stack (Windows)
.\scripts\runtime\start-dev.bat

# Start with Docker Compose (Windows)
.\scripts\runtime\start-local.bat

# Start with Docker Compose (Linux/Mac)
./scripts/runtime/start-local.sh

# Frontend only (runs on port 3000)
cd frontend && npm run dev

# Backend only (Rust)
cd core && cargo watch -x run

# GraphQL gateway
cd graphql && npm run dev

# API Gateway (Python)
cd backend/services/api_gateway && uvicorn main:app --reload
```

### Building & Testing

**Recommended: Use the comprehensive test scripts**
```bash
# Complete test suite (Windows)
.\scripts\testing\test-all-windows.bat

# Complete test suite (Linux/Mac)
./scripts/testing/test-all-linux.sh
```

**Manual component testing:**
```bash
# Frontend
cd frontend
npm run build
npm run lint
npm run type-check
npm test

# Rust backend
cd core
cargo build --release
cargo test
cargo clippy -- -D warnings
cargo fmt --all -- --check

# Run specific Rust test
cd core && cargo test test_name

# Format Rust code
cd core && cargo fmt --all

# Python services
cd backend/services/api_gateway
python -m pytest tests/ --verbose

# GraphQL gateway
cd graphql
npm test

# Edge functions
cd edge
npm run build
npm test
```

### Database Operations
```bash
# Load sample data
.\scripts\seed-data.bat  # Windows
./scripts/seed-data.sh    # Linux/Mac

# Access PostgreSQL
psql postgresql://postgres:postgres@localhost:5432/policycortex

# Access Redis/DragonflyDB
redis-cli -h localhost -p 6379
```

## Service Endpoints
- **Frontend**: http://localhost:3000 (dev mode) or http://localhost:3005 (docker)
- **Core API**: http://localhost:8080 (local) or http://localhost:8085 (docker)
- **GraphQL**: http://localhost:4000/graphql (local) or http://localhost:4001 (docker)
- **EventStore UI**: http://localhost:2113 (admin/changeit)
- **Adminer DB UI**: http://localhost:8081

## Key API Routes
- `/api/v1/metrics` - Unified governance metrics (Patent 1)
- `/api/v1/predictions` - Predictive compliance (Patent 4)
- `/api/v1/conversation` - Conversational AI (Patent 2)
- `/api/v1/correlations` - Cross-domain correlations (Patent 1)
- `/api/v1/recommendations` - AI-driven recommendations
- `/health` - Service health check

## Azure Integration
The platform requires Azure credentials configured via environment variables:
- `AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78`
- `AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7`
- `AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c`

Use managed identity authentication in production. The system includes both sync (azure_client.rs) and async (azure_client_async.rs) Azure clients for optimal performance.

## AI Architecture
The AI engine uses a domain expert architecture (NOT generic AI) with specialized models for:
- Cloud governance policy analysis
- Compliance prediction
- Resource optimization
- Security threat detection
- Cost optimization

Training configuration is in `training/` with Azure AI Foundry integration.

## State Management
- Frontend uses Zustand (not Redux) for state management
- React Query for server state and caching
- Real-time updates via GraphQL subscriptions

## Performance Considerations
- Rust backend provides sub-millisecond response times
- Edge functions use WebAssembly for distributed processing
- DragonflyDB provides 25x faster Redis-compatible caching
- Event sourcing enables complete audit trail without performance impact

## Security Features
- Post-quantum cryptography (Kyber1024, Dilithium5)
- Blockchain-based immutable audit trail
- Zero-trust architecture
- End-to-end encryption
- RBAC with fine-grained permissions

## Current Known Issues
- **Rust Compilation**: The core service has unresolved compilation errors related to unclosed delimiter in src/api/mod.rs:2328. A mock server is used in Docker builds as a temporary workaround.
- **SQLx Offline Mode**: When running locally, you may need to unset `SQLX_OFFLINE` environment variable.
- **MSAL Authentication**: SSR issues have been resolved by providing default AuthContext values during server-side rendering.

## CI/CD Pipeline
GitHub Actions workflow (`application.yml`) includes:
- Linux runners (ubuntu-latest) for better performance
- Azure Container Registry: `crpcxdev.azurecr.io` (dev), `crcortexprodvb9v2h.azurecr.io` (prod)
- Terraform 1.6.0 for infrastructure deployment
- Security scanning with Trivy (non-blocking if Code Scanning not enabled)

## Development Workflow
1. Check Azure authentication: `az account show`
2. Start services with appropriate script (scripts/runtime/start-dev.bat or scripts/runtime/start-local.bat)
3. Frontend hot-reloads automatically
4. Backend requires restart for Rust changes (use cargo watch for auto-reload)
5. Test patent features with `scripts/test-workflow.sh`

## Important Files
- `core/src/main.rs` - Rust API entry point
- `core/src/api/mod.rs` - API route handlers
- `frontend/app/layout.tsx` - Next.js root layout
- `frontend/components/AppLayout.tsx` - Main app layout with navigation
- `backend/services/ai_engine/domain_expert.py` - Core AI engine
- `graphql/gateway.js` - GraphQL federation gateway

## Project Tracking Requirements
**CRITICAL**: After completing each day's implementation work, ALWAYS update the `docs/PROJECT_TRACKING.MD` file with:
- Main heading for the day completed
- Bullet list of all features/components implemented
- Status updates showing progress and completion
- Technical details and system capabilities achieved
This ensures continuous documentation of implementation progress and maintains project visibility.

## Testing Patent Features
The system includes four patented technologies that can be tested via their respective APIs:
1. Cross-Domain Correlation - Test via `/api/v1/correlations` for pattern detection
2. Conversational Intelligence - Test via `/api/v1/conversation` with natural language queries
3. Unified Platform - Test via `/api/v1/metrics` for cross-domain metrics
4. Predictive Compliance - Test via `/api/v1/predictions` for drift predictions

## Docker Operations
```bash
# Build all services
docker-compose build

# Start services (development)
docker-compose -f docker-compose.local.yml up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Clean up everything
docker-compose down -v --remove-orphans
```
- # Patent #2 Quick Reference Guide for AI Coders

## Core Patent Requirements - Must Implement Exactly

### System Architecture (8 Required Subsystems)
1. **Frontend Conversation Interface** - React + Next.js 14 + WebSocket
2. **Natural Language Processing Engine** - Intent classification + Entity extraction
3. **Domain Expert AI** - 175B parameter model (98.7% Azure, 98.2% AWS, 97.5% GCP)
4. **Policy Translation Engine** - Natural language to cloud policy JSON
5. **RLHF System** - Reward models + Preference learning + PPO
6. **Safety Gate System** - Risk assessment + Approval workflows
7. **Multi-Tenant Isolation** - Cryptographic separation + Row-level security
8. **Audit and Evidence Generation** - Immutable audit trails

### Critical Performance Metrics (Must Achieve)
- **Azure Operations Accuracy**: 98.7%
- **AWS Operations Accuracy**: 98.2%
- **GCP Operations Accuracy**: 97.5%
- **Intent Classification**: 95% accuracy across 13 intents
- **Entity Extraction**: 90% precision/recall across 10 entity types

### Domain Expert Model Specifications (Exact Requirements)
- **Parameters**: 175 billion
- **Layers**: 96
- **Attention Heads**: 96
- **Embedding Dimensions**: 12,288
- **Max Sequence Length**: 4,096
- **Training Data**: 2.3 terabytes governance-specific

### Intent Classification (13 Required Intents)
1. **ExplainPolicy** - Policy interpretation requests
2. **CheckCompliance** - Compliance status inquiries
3. **GeneratePolicy** - Policy creation requests
4. **RemediateViolation** - Violation resolution
5. **AnalyzeCost** - Cost analysis queries
6. **ReviewPermissions** - Access control reviews
7. **InvestigateIncident** - Security incident analysis
8. **ConfigureMonitoring** - Monitoring setup
9. **RequestApproval** - Approval workflows
10. **GenerateReport** - Compliance reporting
11. **PredictRisk** - Risk assessment
12. **OptimizeResources** - Resource optimization
13. **ValidateConfiguration** - Configuration validation

### Entity Extraction (10 Required Entity Types)
1. **ResourceId** - Cloud resource identifiers
2. **PolicyName** - Policy naming conventions
3. **ComplianceFramework** - Regulatory framework names
4. **TimeRange** - Temporal expressions
5. **UserIdentity** - Email and username formats
6. **RoleName** - Cloud provider role definitions
7. **CostAmount** - Monetary expressions
8. **RiskLevel** - Severity classifications
9. **CloudProvider** - Platform identifiers
10. **ActionType** - Governance operation types

## Implementation Checklist

### Phase 1: Infrastructure ✓
- [ ] React + Next.js 14 frontend with WebSocket
- [ ] Rust API server with Actix-web
- [ ] PostgreSQL with row-level security
- [ ] Redis for caching and sessions
- [ ] Kubernetes with Istio service mesh

### Phase 2: NLP Core ✓
- [ ] Intent classifier (13 governance intents)
- [ ] Entity extractor (10 entity types)
- [ ] Conversation context manager
- [ ] Knowledge graph representation
- [ ] Multi-turn conversation support

### Phase 3: Domain Expert AI ✓
- [ ] 175B parameter transformer model
- [ ] Multi-cloud expertise (Azure/AWS/GCP)
- [ ] Compliance framework knowledge
- [ ] 2.3TB governance training data
- [ ] Model serving with GPU acceleration

### Phase 4: Policy Translation ✓
- [ ] Natural language to policy conversion
- [ ] Multi-cloud policy support (Azure/AWS/GCP)
- [ ] Policy validation and simulation
- [ ] Template library and pattern matching
- [ ] Syntax and semantic verification

### Phase 5: RLHF System ✓
- [ ] Reward model implementation
- [ ] Preference learning (Bradley-Terry)
- [ ] PPO optimization
- [ ] Feedback collection system
- [ ] Organizational preference adaptation

### Phase 6: Safety Gates ✓
- [ ] Risk assessment and blast radius
- [ ] Approval workflow engine
- [ ] Dry-run simulation
- [ ] Rollback planning
- [ ] Emergency override procedures

### Phase 7: Multi-Tenant Isolation ✓
- [ ] Tenant context isolation
- [ ] Cryptographic data protection
- [ ] Row-level security policies
- [ ] Encrypted conversation storage
- [ ] Audit trail segregation

### Phase 8: APIs ✓
- [ ] Conversation processing endpoints
- [ ] Policy translation APIs
- [ ] Approval workflow APIs
- [ ] GraphQL schema and subscriptions
- [ ] Real-time WebSocket updates

### Phase 9: Testing ✓
- [ ] Conversation accuracy validation
- [ ] Policy generation quality testing
- [ ] Security and isolation testing
- [ ] Performance and scalability testing
- [ ] Compliance verification

## Key API Endpoints (Must Implement)

### Conversation APIs
- `POST /api/v1/conversation` - Process messages with NLP
- `GET /api/v1/conversation/history` - Retrieve conversation history
- `POST /api/v1/conversation/feedback` - Submit human feedback
- `GET /api/v1/conversation/suggestions` - Get proactive recommendations

### Policy Translation APIs
- `POST /api/v1/policy/translate` - Natural language to policy
- `POST /api/v1/policy/validate` - Policy syntax/semantic validation
- `POST /api/v1/policy/simulate` - Dry-run impact assessment
- `GET /api/v1/policy/templates` - Governance policy templates

### Approval Workflow APIs
- `POST /api/v1/approval/request` - Create approval requests
- `PUT /api/v1/approval/{id}/decision` - Process approval decisions
- `GET /api/v1/approval/pending` - Retrieve pending approvals
- `GET /api/v1/approval/history` - Approval audit trails

## Critical Implementation Notes

### Domain Expert Architecture
```python
class DomainExpertAI:
    def __init__(self):
        # 175B parameter transformer
        # 96 layers, 96 heads, 12288 dimensions
        # Multi-cloud expertise modules
        # Compliance framework knowledge
```

### Intent Classification
```rust
pub enum GovernanceIntent {
    ExplainPolicy,
    CheckCompliance,
    GeneratePolicy,
    RemediateViolation,
    // ... 9 more intents
}
```

### Policy Translation
```python
class PolicyTranslator:
    def translate(self, natural_language: str) -> CloudPolicy:
        # Requirement parsing
        # Template matching
        # Parameter extraction
        # Policy synthesis with validation
```

### RLHF System
```python
class RLHFSystem:
    def __init__(self):
        # Reward model (neural network)
        # Bradley-Terry preference model
        # PPO trainer
        # Feedback collection buffer
```

## Training Data Composition (2.3TB Total)

- **Cloud Provider Documentation**: 450GB (Azure, AWS, GCP)
- **Compliance Frameworks**: 380GB (NIST, ISO, PCI-DSS, HIPAA, SOX, GDPR, FedRAMP)
- **Policy Templates**: 290GB (50,000+ validated policies)
- **Governance Best Practices**: 310GB (Whitepapers, guidelines)
- **Incident Reports**: 280GB (Sanitized breach analyses)
- **Audit Logs**: 350GB (Anonymized operation histories)
- **Expert Annotations**: 240GB (Hand-labeled conversations)

## Success Validation

### Accuracy Testing
- Achieve 98.7% Azure accuracy
- Achieve 98.2% AWS accuracy
- Achieve 97.5% GCP accuracy
- Validate 95% intent classification accuracy
- Validate 90% entity extraction precision/recall

### Conversation Quality Testing
- Multi-turn context preservation
- Governance terminology understanding
- Policy generation accuracy
- Compliance framework adherence
- Safety gate effectiveness

### Security Testing
- Tenant isolation verification
- Cryptographic protection validation
- Row-level security testing
- Audit trail integrity
- Access control enforcement

### Performance Testing
- Real-time conversation processing
- Policy translation latency
- Approval workflow performance
- System scalability validation
- Load testing with concurrent users

## Common Implementation Pitfalls

1. **Incorrect Model Size** - Must use exactly 175B parameters with specified architecture
2. **Missing Intent Classes** - Must implement all 13 governance-specific intents
3. **Incomplete Entity Types** - Must extract all 10 governance entity types
4. **Inadequate Training Data** - Must use 2.3TB governance-specific training data
5. **Missing Multi-Cloud Support** - Must support Azure, AWS, and GCP policies
6. **Insufficient Safety Gates** - Must implement comprehensive risk assessment
7. **Incomplete Tenant Isolation** - Must enforce cryptographic separation
8. **Missing RLHF Components** - Must implement reward models and PPO
9. **Inadequate Policy Validation** - Must validate syntax and semantics
10. **Missing Audit Trails** - Must generate immutable audit evidence

## Compliance Requirements

### Patent Claim Compliance
- All 25 patent claims must be implemented
- Specific technical parameters must be met
- Architectural requirements must be followed
- Performance metrics must be achieved

### Regulatory Compliance
- GDPR data protection requirements
- SOX audit trail requirements
- HIPAA privacy protections
- PCI-DSS security standards
- FedRAMP security controls

### Security Standards
- AES-256-GCM encryption for data at rest
- TLS 1.3 for data in transit
- Row-level security for multi-tenancy
- Cryptographic key management
- Audit logging with immutable timestamps

  
# Patent #4 Quick Reference Guide for AI Coders

## Core Patent Requirements - Must Implement Exactly

### System Architecture (8 Required Subsystems)
1. **Feature Engineering Subsystem** - Multi-modal feature extraction
2. **Ensemble ML Engine** - LSTM + Attention + Gradient Boosting + Prophet
3. **Drift Detection Subsystem** - VAE + Reconstruction Error + SPC
4. **SHAP Explainability Engine** - Feature importance + Decision attribution
5. **Continuous Learning Pipeline** - Human feedback + Auto retraining
6. **Confidence Scoring Module** - Uncertainty quantification + Risk scoring
7. **Tenant-Isolated Infrastructure** - Secure multi-tenancy
8. **Real-time Prediction Serving** - Sub-100ms latency

### Critical Performance Metrics (Must Achieve)
- **Prediction Accuracy**: 99.2%
- **False Positive Rate**: <2%
- **Inference Latency**: <100ms
- **Training Throughput**: 10,000 samples/second

### LSTM Network Specifications (Exact Requirements)
- **Hidden Dimensions**: 512
- **Layers**: 3
- **Dropout Rate**: 0.2
- **Attention Heads**: 8
- **Sequence Length**: 100

### Ensemble Model Weights (Specified in Patent)
- **Isolation Forest**: 40%
- **LSTM Detector**: 30%
- **Autoencoder**: 30%

### VAE Configuration (Required Parameters)
- **Latent Space Dimensionality**: 128
- **Reconstruction Error Thresholds**: Dynamic (based on historical variance)
- **SPC Control Limits**: 3-sigma rules

## Implementation Checklist

### Phase 1: Infrastructure ✓
- [ ] Development environment with PyTorch 1.12+, TensorFlow 2.8+
- [ ] Database schema (configurations, policies, violations, predictions, models, feedback)
- [ ] API framework with REST + GraphQL + WebSocket
- [ ] Kubernetes deployment with GPU support

### Phase 2: ML Core ✓
- [ ] PolicyCompliancePredictor (LSTM + Attention)
- [ ] AnomalyDetectionPipeline (Ensemble)
- [ ] VAEDriftDetector (Drift detection)
- [ ] Feature engineering pipeline (multi-modal)
- [ ] Statistical Process Control

### Phase 3: Explainability ✓
- [ ] SHAP explainer (local + global)
- [ ] Attention visualization
- [ ] Decision tree extraction
- [ ] Interactive explanations

### Phase 4: Continuous Learning ✓
- [ ] Feedback collection system
- [ ] Online learning algorithms
- [ ] Automated retraining pipeline
- [ ] A/B testing framework

### Phase 5: Performance ✓
- [ ] TensorRT optimization
- [ ] Distributed training
- [ ] Model serving infrastructure
- [ ] Hyperparameter optimization

### Phase 6: Security ✓
- [ ] Tenant isolation
- [ ] Differential privacy
- [ ] Encrypted model parameters
- [ ] Secure aggregation

### Phase 7: APIs ✓
- [ ] Prediction endpoints
- [ ] Model management APIs
- [ ] Explainability APIs
- [ ] GraphQL subscriptions

### Phase 8: Testing ✓
- [ ] Unit tests (99.2% accuracy validation)
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Security validation

## Key API Endpoints (Must Implement)

### Prediction APIs
- `GET /api/v1/predictions` - All predictions
- `GET /api/v1/predictions/violations` - Violation forecasts
- `GET /api/v1/predictions/risk-score/{resource_id}` - Risk assessment
- `POST /api/v1/predictions/remediate/{prediction_id}` - Trigger remediation

### Model Management APIs
- `GET /api/v1/ml/feature-importance` - SHAP analysis
- `POST /api/v1/ml/retrain` - Manual retraining
- `GET /api/v1/ml/metrics` - Performance monitoring
- `POST /api/v1/ml/feedback` - Submit feedback

## Critical Implementation Notes

### LSTM Architecture
```python
class PolicyCompliancePredictor(nn.Module):
    def __init__(self):
        # Feature extractor: 256→512→1024→512
        # Multi-head attention: 8 heads
        # Prediction layers: 512→256→128→2
        # Confidence scorer: 512→1
```

### Ensemble Pipeline
```python
class AnomalyDetectionPipeline:
    def __init__(self):
        # Isolation Forest: 40% weight
        # LSTM Detector: 30% weight  
        # Autoencoder: 30% weight
```

### Drift Detection
```python
class VAEDriftDetector:
    def __init__(self):
        # Latent dimensions: 128
        # Reconstruction error thresholds
        # Bayesian uncertainty quantification
```

## Success Validation

### Accuracy Testing
- Achieve 99.2% prediction accuracy
- Maintain <2% false positive rate
- Validate with time-series cross-validation

### Performance Testing
- Sub-100ms inference latency
- 10,000 samples/second training throughput
- Horizontal scaling validation

### Security Testing
- Tenant isolation verification
- Differential privacy validation
- Encryption and secure aggregation testing

### Explainability Testing
- SHAP value accuracy
- Attention mechanism visualization
- Regulatory compliance reporting

## Common Implementation Pitfalls

1. **Incorrect LSTM Parameters** - Must use exact specifications (512 hidden, 3 layers, 0.2 dropout)
2. **Wrong Ensemble Weights** - Must use 40/30/30 split for IF/LSTM/AE
3. **Missing Tenant Isolation** - Critical for multi-tenant security
4. **Inadequate Explainability** - SHAP + Attention required for compliance
5. **Performance Shortcuts** - Must achieve specified latency and accuracy
6. **Incomplete API Coverage** - All specified endpoints must be implemented
7. **Missing Continuous Learning** - Feedback loop is patent requirement
8. **Insufficient Testing** - Must validate all performance metrics