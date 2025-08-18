# PolicyCortex Development Guide

## Quick Start

### Prerequisites
- Rust 1.70+
- Node.js 18+
- PostgreSQL 14+
- Docker & Docker Compose
- Azure CLI

### Local Development Setup

```bash
# Start full development environment
.\scripts\runtime\start-dev.bat        # Windows
./start-dev.sh         # Linux/Mac

# Start with Docker Compose
.\scripts\runtime\start-local.bat      # Windows  
./scripts/runtime/start-local.sh       # Linux/Mac
```

### Individual Services

```bash
# Frontend only (port 3000)
cd frontend && npm run dev

# Backend only (Rust - port 8080)
cd core && cargo watch -x run

# GraphQL gateway (port 4000)
cd graphql && npm run dev

# Python AI engine
cd backend/services/api_gateway && uvicorn main:app --reload
```

## Testing

### Comprehensive Testing (Recommended)
```bash
.\scripts\testing\test-all-windows.bat    # Windows
./scripts/testing/test-all-linux.sh       # Linux/Mac
```

### Component Testing
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
cargo clippy -- -D warnings
cargo fmt --all -- --check
```

## Database Operations

```bash
# Load sample data
.\scripts\seed-data.bat   # Windows
./scripts/seed-data.sh    # Linux/Mac

# Access databases
psql postgresql://postgres:postgres@localhost:5432/policycortex
redis-cli -h localhost -p 6379
```

## Service Endpoints

- **Frontend**: http://localhost:3000 (dev) / http://localhost:3005 (docker)
- **Core API**: http://localhost:8080 (local) / http://localhost:8085 (docker)  
- **GraphQL**: http://localhost:4000/graphql (local) / http://localhost:4001 (docker)
- **EventStore UI**: http://localhost:2113 (admin/changeit)
- **Adminer DB**: http://localhost:8081

## Key API Routes

- `/api/v1/metrics` - Unified governance metrics (Patent 1)
- `/api/v1/predictions` - Predictive compliance (Patent 4)
- `/api/v1/conversation` - Conversational AI (Patent 2)
- `/api/v1/correlations` - Cross-domain correlations (Patent 1)
- `/api/v1/remediation` - Automated remediation workflows
- `/health` - Service health check

## Azure Integration

Required environment variables:
```bash
AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7  
AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
```

Authentication in production uses managed identity. The system includes both sync (`azure_client.rs`) and async (`azure_client_async.rs`) Azure clients.

## State Management & Architecture

- **Frontend**: Zustand for state management (not Redux)
- **React Query**: Server state and caching
- **Real-time**: GraphQL subscriptions
- **Backend**: Rust async/await with Axum framework
- **AI**: Domain expert architecture (specialized models)

## Performance Features

- Sub-millisecond response times (Rust backend)
- WebAssembly edge functions for distributed processing
- DragonflyDB provides 25x faster Redis-compatible caching
- Event sourcing for complete audit trail without performance impact

## Security

- Post-quantum cryptography (Kyber1024, Dilithium5)
- Blockchain-based immutable audit trail
- Zero-trust architecture with end-to-end encryption
- RBAC with fine-grained permissions

## Known Issues

- **Rust Compilation**: Core service uses mock server in Docker builds (temporary workaround)
- **SQLx Offline**: May need to unset `SQLX_OFFLINE` environment variable locally
- **MSAL Authentication**: SSR issues resolved with default AuthContext values

## Development Workflow

1. Check Azure auth: `az account show`
2. Start services with appropriate script
3. Frontend hot-reloads automatically  
4. Backend requires restart for Rust changes (use `cargo watch`)
5. Test patent features with `scripts/test-workflow.sh`

## Important Files

- `core/src/main.rs` - Rust API entry point
- `core/src/api/mod.rs` - API route handlers  
- `frontend/app/layout.tsx` - Next.js root layout
- `frontend/components/AppLayout.tsx` - Main app layout
- `backend/services/ai_engine/domain_expert.py` - Core AI engine
- `graphql/gateway.js` - GraphQL federation gateway

## CI/CD Pipeline

GitHub Actions workflow includes:
- Linux runners (ubuntu-latest)
- Azure Container Registry integration
- Terraform 1.6.0 for infrastructure
- Security scanning with Trivy

**Registries:**
- Dev: `crpcxdev.azurecr.io`  
- Prod: `crcortexprodvb9v2h.azurecr.io`

## Patent Technologies Testing

Test the four patented systems:

1. **Unified Platform** - `GET /api/v1/metrics` for cross-domain metrics
2. **Predictive Compliance** - `GET /api/v1/predictions` for drift predictions  
3. **Conversational Intelligence** - `POST /api/v1/conversation` with natural language
4. **Cross-Domain Correlation** - `GET /api/v1/correlations` for pattern detection