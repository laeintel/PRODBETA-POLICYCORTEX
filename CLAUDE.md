# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PolicyCortex v2 is an AI-powered Azure governance platform with patented technologies including Cross-Domain Governance Correlation Engine, Conversational Governance Intelligence System, Unified AI-Driven Cloud Governance Platform, and Predictive Policy Compliance Engine.

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

# Frontend only
cd frontend && npm run dev

# Backend only (Rust)
cd core && cargo watch -x run

# GraphQL gateway
cd graphql && npm run dev

# API Gateway (Python)
cd backend/services/api_gateway && uvicorn main:app --reload
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
- **Frontend**: http://localhost:3000
- **Core API**: http://localhost:8080
- **GraphQL**: http://localhost:4000/graphql
- **EventStore UI**: http://localhost:2113 (admin/changeit)

## Key API Routes
- `/api/v1/metrics` - Unified governance metrics (Patent 1)
- `/api/v1/predictions` - Predictive compliance (Patent 2)
- `/api/v1/conversation` - Conversational AI (Patent 3)
- `/api/v1/correlations` - Cross-domain correlations (Patent 4)
- `/api/v1/recommendations` - AI-driven recommendations
- `/health` - Service health check

## Azure Integration
The platform requires Azure credentials configured via environment variables:
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`

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

## Development Workflow
1. Check Azure authentication: `az account show`
2. Start services with appropriate script (start-dev.bat or start-local.bat)
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

## Testing Patent Features
The system includes four patented technologies that can be tested via their respective APIs:
1. Unified Platform - Test via `/api/v1/metrics` for cross-domain metrics
2. Predictive Compliance - Test via `/api/v1/predictions` for drift predictions
3. Conversational Intelligence - Test via `/api/v1/conversation` with natural language queries
4. Cross-Domain Correlation - Test via `/api/v1/correlations` for pattern detection