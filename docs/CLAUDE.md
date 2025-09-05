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
- `/api/v1/metrics` - Unified governance metrics (Patent 3)
- `/api/v1/predictions` - Predictive compliance (Patent 4)
- `/api/v1/conversation` - Conversational AI (Patent 2)
- `/api/v1/correlations` - Cross-domain correlations (Patent 1)
- `/api/v1/recommendations` - AI-driven recommendations
- `/health` - Service health check

## Azure Integration
The platform requires Azure credentials configured via environment variables:
- `AZURE_SUBSCRIPTION_ID=6dc7cfa2-0332-4740-98b6-bac9f1a23de9`
- `AZURE_TENANT_ID=e1f3e196-aa55-4709-9c55-0e334c0b444f`
- `AZURE_CLIENT_ID=232c44f7-d0cf-4825-a9b5-beba9f587ffb`

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

## Patent Implementation Details

### Patent #2: Conversational Governance Intelligence System

**Core Requirements:**
- 175B parameter domain expert AI model
- 13 governance-specific intent classifications
- 10 entity extraction types
- Natural language to cloud policy translation
- RLHF system with PPO optimization
- Multi-tenant isolation with cryptographic separation

**Performance Targets:**
- Azure Operations: 98.7% accuracy
- AWS Operations: 98.2% accuracy
- GCP Operations: 97.5% accuracy
- Intent Classification: 95% accuracy
- Entity Extraction: 90% precision/recall

**Key APIs:**
- `POST /api/v1/conversation` - Process messages with NLP
- `POST /api/v1/policy/translate` - Natural language to policy
- `POST /api/v1/approval/request` - Create approval requests

### Patent #4: Predictive Policy Compliance Engine

**Core Requirements:**
- Ensemble ML with LSTM + Attention + Gradient Boosting
- VAE drift detection with 128-dimensional latent space
- SHAP explainability engine
- Continuous learning pipeline with human feedback
- Real-time prediction serving (<100ms latency)

**Performance Targets:**
- Prediction Accuracy: 99.2%
- False Positive Rate: <2%
- Inference Latency: <100ms
- Training Throughput: 10,000 samples/second

**Model Specifications:**
- LSTM: 512 hidden dims, 3 layers, 0.2 dropout, 8 attention heads
- Ensemble Weights: Isolation Forest (40%), LSTM (30%), Autoencoder (30%)

**Key APIs:**
- `GET /api/v1/predictions` - All predictions
- `GET /api/v1/predictions/risk-score/{resource_id}` - Risk assessment
- `GET /api/v1/ml/feature-importance` - SHAP analysis
- `POST /api/v1/ml/feedback` - Submit feedback

## Detailed Patent Implementation Guides

For comprehensive implementation details including checklists, code examples, and validation criteria:
- Patent #2 Implementation: See inline documentation in `backend/services/ai_engine/`
- Patent #4 Implementation: See inline documentation in `core/src/ml/`

These patents require exact implementation of specified architectures, performance metrics, and API endpoints to maintain compliance with patent claims.