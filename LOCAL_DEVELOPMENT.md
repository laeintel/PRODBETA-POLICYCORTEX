# PolicyCortex v2 - Local Development Guide

## Prerequisites

1. **Docker Desktop** (Windows/Mac/Linux)
   - Download: https://www.docker.com/products/docker-desktop
   - Ensure WSL2 backend is enabled (Windows)
   - Allocate at least 8GB RAM in Docker settings

2. **Development Tools**
   - Git
   - Visual Studio Code (recommended)
   - Rust (optional for local development without Docker)
   - Node.js 20+ (optional for local development without Docker)

## Quick Start

### Windows
```powershell
# Clone the repository
git clone <repository-url>
cd policycortex\policycortex-v2

# Start all services
.\start-local.bat
```

### Linux/Mac
```bash
# Clone the repository
git clone <repository-url>
cd policycortex/policycortex-v2

# Make script executable
chmod +x start-local.sh

# Start all services
./start-local.sh
```

## Service Endpoints

Once started, the following services will be available:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Next.js 14 application with Module Federation |
| **GraphQL Gateway** | http://localhost:4000/graphql | Apollo Federation gateway |
| **Core API** | http://localhost:8080 | Rust modular monolith |
| **Edge Simulator** | http://localhost:8787 | WASM edge functions |
| **EventStore UI** | http://localhost:2113 | Event sourcing database UI |
| **PostgreSQL** | localhost:5432 | Main database (user: postgres, pass: postgres) |
| **DragonflyDB** | localhost:6379 | Redis-compatible cache |

## Development Workflow

### 1. Frontend Development

The frontend uses Next.js 14 with hot module reloading:

```bash
cd frontend
npm install
npm run dev
```

Access at http://localhost:3000

**Key Features:**
- Server Components
- Module Federation
- PWA capabilities
- Real-time GraphQL subscriptions
- Framer Motion animations

### 2. Backend Development

The Rust core service auto-reloads on file changes:

```bash
cd core
cargo watch -x run
```

**Key Modules:**
- `/api` - REST endpoints
- `/graphql` - GraphQL resolvers
- `/domain` - Business logic
- `/blockchain` - Audit trail
- `/quantum` - Quantum-ready cryptography
- `/ml` - Machine learning pipelines

### 3. GraphQL Development

The GraphQL gateway supports hot reloading:

```bash
cd graphql
npm install
npm run dev
```

Access GraphQL Playground at http://localhost:4000/graphql

### 4. Edge Functions

Develop WASM functions locally:

```bash
cd edge
npm install
npm run dev
```

Test edge functions at http://localhost:8787

## Testing

### Run All Tests
```bash
# In root directory
npm run test:all
```

### Frontend Tests
```bash
cd frontend
npm test                # Unit tests
npm run test:e2e       # E2E tests with Playwright
npm run test:coverage  # Coverage report
```

### Backend Tests
```bash
cd core
cargo test            # Unit tests
cargo test --all     # All tests including integration
cargo bench          # Performance benchmarks
```

### GraphQL Tests
```bash
cd graphql
npm test             # Schema validation and resolver tests
```

## Database Management

### PostgreSQL
- Connection: `postgresql://postgres:postgres@localhost:5432/policycortex`
- GUI: Use pgAdmin or DBeaver
- Migrations: `cd core && cargo run --bin migrate`

### EventStore
- UI: http://localhost:2113
- Username: admin
- Password: changeit
- View event streams and projections

### DragonflyDB (Redis)
- Connection: `redis://localhost:6379`
- CLI: `docker exec -it policycortex-v2-dragonfly-1 redis-cli`

## Seed Data

Load sample data for development:

```bash
# Load sample policies and resources
cd scripts
./seed-data.sh

# Or on Windows
.\seed-data.bat
```

## Environment Variables

Create `.env.local` files for custom configuration:

### Frontend (.env.local)
```env
NEXT_PUBLIC_GRAPHQL_ENDPOINT=http://localhost:4000/graphql
NEXT_PUBLIC_WEBSOCKET_ENDPOINT=ws://localhost:4000/subscriptions
NEXT_PUBLIC_AUTH_DOMAIN=your-auth0-domain
NEXT_PUBLIC_AUTH_CLIENT_ID=your-client-id
```

### Core Service (.env)
```env
RUST_LOG=debug
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
REDIS_URL=redis://localhost:6379
JWT_SECRET=development-secret-key
```

## Debugging

### View Logs
```bash
# All services
docker-compose -f docker-compose.local.yml logs -f

# Specific service
docker-compose -f docker-compose.local.yml logs -f core
docker-compose -f docker-compose.local.yml logs -f frontend
```

### Access Container Shell
```bash
# Core service
docker exec -it policycortex-v2-core-1 /bin/bash

# Frontend
docker exec -it policycortex-v2-frontend-1 /bin/sh
```

### Debug Rust Service
1. Install VS Code extension: rust-analyzer
2. Set breakpoints in code
3. Use launch configuration:

```json
{
  "type": "lldb",
  "request": "launch",
  "name": "Debug Core",
  "cargo": {
    "args": ["run", "--bin", "policycortex-core"],
    "env": {
      "RUST_LOG": "debug"
    }
  }
}
```

## Common Issues

### Port Already in Use
```bash
# Find process using port (Windows)
netstat -ano | findstr :3000

# Kill process (Windows)
taskkill /PID <process-id> /F

# Find process using port (Linux/Mac)
lsof -i :3000

# Kill process (Linux/Mac)
kill -9 <process-id>
```

### Docker Build Failures
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose -f docker-compose.local.yml build --no-cache
```

### Database Connection Issues
```bash
# Reset database
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d postgres
```

## Performance Monitoring

### Metrics Dashboard
Access Grafana at http://localhost:3001 (when enabled)

### Tracing
View distributed traces at http://localhost:16686 (Jaeger UI)

## Hot Module Reloading

All services support hot reloading:
- **Frontend**: Next.js Fast Refresh
- **Core**: cargo-watch for Rust
- **GraphQL**: nodemon for Node.js
- **Edge**: Wrangler dev server

## VS Code Extensions

Recommended extensions for development:
- Rust Analyzer
- ESLint
- Prettier
- GraphQL
- Docker
- GitLens
- Thunder Client (API testing)

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Next.js 14     │────▶│  GraphQL        │────▶│  Rust Core      │
│  Frontend       │     │  Gateway        │     │  Service        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Edge Functions │     │  EventStore     │     │  PostgreSQL     │
│  (WASM)         │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Additional Resources

- [Architecture Documentation](./docs/ARCHITECTURE.md)
- [API Documentation](http://localhost:8080/swagger)
- [GraphQL Schema](http://localhost:4000/graphql)
- [Component Storybook](http://localhost:6006) (when enabled)