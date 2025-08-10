# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PolicyCortex v2 is an AI-powered Azure governance platform with four patented technologies:
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

### Development
```bash
# Start full stack (Windows)
.\start-dev.bat

# Start with Docker Compose (Windows)
.\start-local.bat

# Start with Docker Compose (Linux/Mac)
./start-local.sh

# Restart all services (Windows)
.\restart-services.bat

# Frontend only (runs on port 3000 in dev mode, port 3005 with custom dev)
cd frontend && npm run dev

# Backend only (Rust) with auto-reload
cd core && cargo watch -x run

# Backend only (Rust) without auto-reload
cd core && cargo run

# GraphQL gateway
cd graphql && npm run dev

# API Gateway (Python)
cd backend/services/api_gateway && uvicorn main:app --reload --port 8000

# API Gateway without Docker (Windows)
.\start-api-only.bat

# AI Engine (Python)
cd backend/services/ai_engine && python app.py
```

### Building & Testing
```bash
# Frontend
cd frontend
npm run build
npm run lint
npm run type-check

# Rust backend  
cd core
cargo build --release
cargo test
cargo clippy

# Run full test suite
./scripts/test-workflow.sh  # Linux/Mac

# Run single Rust test
cd core && cargo test test_name

# Run Python tests
cd backend && pytest
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
- **Frontend**: http://localhost:3000 (docker/production) or http://localhost:3005 (dev mode)
- **Core API**: http://localhost:8080
- **GraphQL**: http://localhost:4000/graphql
- **API Gateway**: http://localhost:8000 (Python FastAPI)
- **AI Engine**: http://localhost:8001
- **EventStore UI**: http://localhost:2113 (admin/changeit)

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

Training configuration is in `training/` with Azure AI Foundry integration. Key AI components:
- `backend/services/ai_engine/domain_expert.py` - Core domain-specific AI engine
- `backend/services/ai_engine/continuous_learning.py` - Real-time learning system
- `backend/services/ai_engine/advanced_learning_integration.py` - Advanced learning features
- `backend/services/ai_engine/meta_learning_system.py` - Meta-learning capabilities

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

## Development Workflow
1. Check Azure authentication: `az account show`
2. Start services with appropriate script (start-dev.bat, start-local.bat, or restart-services.bat)
3. Frontend hot-reloads automatically
4. Backend requires restart for Rust changes (use `cargo watch` for auto-reload)
5. Python services auto-reload with uvicorn --reload flag
6. Test patent features with `scripts/test-workflow.sh`

## Important Files
- `core/src/main.rs` - Rust API entry point
- `core/src/api/mod.rs` - API route handlers
- `core/src/services/` - Core business logic services
- `frontend/app/layout.tsx` - Next.js root layout
- `frontend/components/AppLayout.tsx` - Main app layout with navigation
- `frontend/components/Dashboard/` - Dashboard components
- `backend/services/ai_engine/domain_expert.py` - Core AI engine
- `backend/services/api_gateway/main.py` - Python API gateway
- `graphql/gateway.js` - GraphQL federation gateway
- `docker-compose.local.yml` - Local Docker configuration
- `docker-compose.dev.yml` - Development Docker configuration

## Testing Patent Features
The system includes four patented technologies that can be tested via their respective APIs:
1. Unified Platform - Test via `/api/v1/metrics` for cross-domain metrics
2. Predictive Compliance - Test via `/api/v1/predictions` for drift predictions
3. Conversational Intelligence - Test via `/api/v1/conversation` with natural language queries
4. Cross-Domain Correlation - Test via `/api/v1/correlations` for pattern detection

## Frontend Package Scripts
- `npm run dev` - Start development server on port 3000
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run type-check` - TypeScript type checking

## Rust Cargo Commands
- `cargo build` - Build debug version
- `cargo build --release` - Build optimized release version
- `cargo test` - Run all tests
- `cargo clippy` - Run linter
- `cargo watch -x run` - Auto-reload on file changes

## Docker Operations
- `docker-compose -f docker-compose.local.yml up -d` - Start all services
- `docker-compose -f docker-compose.local.yml down` - Stop all services
- `docker-compose -f docker-compose.local.yml logs -f [service]` - View logs
- `docker ps` - List running containers
- `docker exec -it [container] /bin/sh` - Shell into container

## Troubleshooting
- If frontend port 3000 is in use, it runs on 3005 in dev mode
- Clear Redis cache: `redis-cli FLUSHALL`
- Reset PostgreSQL: Drop and recreate the database
- Check service health: `curl http://localhost:8080/health`
- View Rust logs: Set `RUST_LOG=debug` environment variable
- Python debugging: Check uvicorn output in API gateway window