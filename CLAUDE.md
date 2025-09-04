# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PolicyCortex is an enterprise-grade AI-powered Azure governance platform with four patented technologies and comprehensive Cloud ITSM capabilities:
1. Cross-Domain Governance Correlation Engine (Patent 1) - ✅ 100% Complete
2. Conversational Governance Intelligence System (Patent 2) - ✅ 100% Complete
3. Unified AI-Driven Cloud Governance Platform (Patent 3) - ✅ 100% Complete
4. Predictive Policy Compliance Engine (Patent 4) - ✅ 100% Complete

## Quick Start Commands

### Start Development Environment
```bash
# Start mock server (provides all API endpoints)
node mock-server.js

# Start frontend development server
cd frontend && npm run dev

# If port 3000 is busy, use alternate port
cd frontend && npm run dev -- -p 3001
```

### Build and Deploy
```bash
# Frontend production build
cd frontend && npm run build && npm start

# Rust backend build (currently compiles with warnings only)
cd core && cargo build --release

# Python ML services
cd backend/services/api_gateway && uvicorn main:app --reload
```

## Architecture Overview

### **Frontend** (Next.js 14)
- **State Management**: Zustand stores in `frontend/stores/`
- **API Client**: Unified client in `frontend/lib/api-client.ts` with caching
- **Authentication**: MSAL with demo mode bypass (set `NEXT_PUBLIC_DEMO_MODE=true` in `.env.local`)
- **Type System**: Comprehensive TypeScript types in `frontend/types/api.ts`
- **Components**: Reusable UI in `frontend/components/` using lucide-react icons exclusively
- **Testing**: Jest unit tests in `__tests__/`, Playwright E2E in `tests/e2e/`

### **Backend Services**
- **Core API** (Rust): Axum-based API in `core/src/` with CQRS pattern
- **Mock Server**: Express.js fallback in `mock-server.js` (port 8080)
- **ML Models**: Scikit-learn models in `backend/services/ai_engine/`
- **Database**: PostgreSQL with SQLx, connection string in `.env`
- **Cache**: Redis/DragonflyDB for hot data caching

### **Critical Files for Architecture Understanding**
- `frontend/stores/resourceStore.ts` - Main state management for resources
- `frontend/lib/api-client.ts` - API communication layer
- `core/src/api/mod.rs` - Rust API route definitions
- `core/src/cqrs/mod.rs` - CQRS implementation
- `mock-server.js` - Fallback API server providing all endpoints
- `backend/services/ai_engine/simple_ml_service.py` - ML model serving

## Common Development Tasks

### Fix TypeScript Compilation Errors
```bash
cd frontend
npm run type-check  # Check for type errors
npm run build       # Build will fail on type errors
```

### Run Tests
```bash
# Frontend unit tests
cd frontend && npm test

# Frontend E2E tests
cd frontend && npx playwright test

# Rust tests
cd core && cargo test

# Python tests
cd backend/services/api_gateway && python -m pytest tests/ -v
```

### Access the Application
1. Ensure mock server is running: `node mock-server.js`
2. Start frontend: `cd frontend && npm run dev`
3. Access at http://localhost:3000 (or 3001 if specified)
4. Login: Demo mode auto-bypasses authentication, or click "Sign in with Microsoft"

## Environment Configuration

### Essential Environment Variables
Create `frontend/.env.local`:
```env
NEXT_PUBLIC_DEMO_MODE=true  # Bypass authentication for development
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
NEXT_PUBLIC_AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
```

## Current Status & Known Issues

### ✅ Working
- Frontend builds and runs successfully
- Mock server provides all API endpoints
- ML models deployed (3 models: compliance, anomaly, cost)
- TypeScript fully type-safe (0 critical 'any' types)
- 182 frontend tests passing
- Rust backend compiles (0 errors, 187 warnings)

### ⚠️ Known Issues
- CSP warnings in console (cosmetic, doesn't affect functionality)
- MSAL authentication requires demo mode bypass for local development
- Rust has 187 warnings (unused variables, doesn't affect compilation)

## High-Level Architecture Patterns

### CQRS Pattern (Command Query Responsibility Segregation)
- Commands: Write operations in `core/src/cqrs/commands.rs`
- Queries: Read operations in `core/src/cqrs/queries.rs`
- Events: Domain events in `core/src/cqrs/events.rs`
- Projections: Read models in `core/src/cqrs/projections.rs`

### State Management Architecture
- Frontend uses Zustand for client state (not Redux)
- Each major feature has its own store in `frontend/stores/`
- Stores use immer for immutable updates
- Persistence via zustand/middleware/persist

### API Communication Flow
1. Frontend components call API client methods
2. API client adds auth headers and caching
3. Requests go to mock server (port 8080) or real backend
4. Mock server returns realistic data for all endpoints
5. Responses are cached based on endpoint type (hot/warm/cold)

### ML Model Architecture
- Models trained offline and saved to `models_cache/`
- Simple ML service loads models on startup
- Predictions served via `/api/v1/predictions` endpoint
- Feature extraction happens in `simple_ml_service.py`
- <100ms inference latency requirement for Patent #4

## Patent Implementation Requirements

### Patent #4: Predictive Policy Compliance Engine
- Must maintain <100ms inference latency
- Ensemble model with specific weights: Isolation Forest (40%), LSTM (30%), Autoencoder (30%)
- SHAP explainability required for all predictions
- Continuous learning pipeline with feedback loop

### Patent #2: Conversational Governance Intelligence
- Natural language processing for governance queries
- 13 intent classifications required
- Multi-tenant isolation with cryptographic separation
- 95% accuracy target for intent classification

## Testing Strategy

### Unit Tests
- Frontend: Jest + React Testing Library
- Backend: Rust's built-in test framework
- Python: pytest with pytest-asyncio

### E2E Tests
- 40 Playwright test scenarios covering critical flows
- Tests run in headless mode for CI/CD
- Performance metrics collected (LCP, FCP, TTFB)

### Integration Tests
- API integration tests in `backend/services/api_gateway/tests/`
- Mock external services for consistent testing
- Skip markers for tests requiring live services

## Deployment Considerations

### Docker Setup
```bash
# Build all services
docker-compose build

# Start with local config
docker-compose -f docker-compose.local.yml up -d
```

### Production Checklist
- Set `NEXT_PUBLIC_DEMO_MODE=false`
- Configure real Azure AD authentication
- Deploy real ML models (not mocks)
- Enable CSP security headers
- Set up monitoring and alerting

## Code Style Guidelines

### TypeScript/JavaScript
- Use `lucide-react` for all icons (no heroicons)
- Prefer functional components with hooks
- Use TypeScript strict mode
- Avoid 'any' types

### Rust
- Use `cargo fmt` before committing
- Run `cargo clippy` for linting
- Prefer `?` operator for error handling
- Use async/await for I/O operations

### Python
- Follow PEP 8 style guide
- Use type hints for all functions
- Async functions with FastAPI
- pytest for all testing