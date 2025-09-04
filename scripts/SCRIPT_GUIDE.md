# PolicyCortex Script Usage Guide

## Quick Start - Which Script to Use?

### For Complete Local Testing (RECOMMENDED)
```bash
.\scripts\runtime\start-complete.bat
```
**Use this when:** You want to start the entire PolicyCortex stack for comprehensive testing
- ✅ Starts all infrastructure (PostgreSQL, Redis, EventStore)
- ✅ Attempts Docker Compose first, falls back to local if needed
- ✅ Verifies all services are running
- ✅ Best for full integration testing

### For Docker-Based Development
```bash
.\scripts\runtime\start-local.bat
```
**Use this when:** You want everything running in Docker containers
- ✅ Uses docker-compose.local.yml
- ✅ All services in containers
- ✅ Good for consistent environment
- ⚠️ May have issues with Core service compilation

### For Local Development with Hot Reload
```bash
.\scripts\runtime\start-dev.bat
```
**Use this when:** You're actively developing and need hot reload
- ✅ Runs databases in Docker
- ✅ Runs applications locally (Core, GraphQL, Frontend)
- ✅ Best for active development
- ✅ Supports hot reload for all services

### For Running Complete Test Suite
```bash
.\scripts\testing\test-all-windows.bat
```
**Use this when:** You want to run all tests before committing
- ✅ Tests Frontend (TypeScript, Lint, Build)
- ✅ Tests Core (Rust tests, Clippy, Format)
- ✅ Tests GraphQL
- ✅ Tests Python backend
- ✅ Verifies Docker builds

## Service Architecture

### Infrastructure Services (Always in Docker)
- **PostgreSQL**: Main database (port 5432)
- **Redis/DragonflyDB**: Caching layer (port 6379)
- **EventStore**: Event sourcing (port 2113)

### Application Services
- **Frontend**: Next.js application (port 3000)
- **Core API**: Rust backend (port 8080)
- **GraphQL Gateway**: Apollo Federation (port 4000)

## Common Commands

### Start Everything for Testing
```bash
# Recommended - tries Docker first, falls back to local
.\scripts\runtime\start-complete.bat
```

### Start for Development
```bash
# Hot reload enabled
.\scripts\runtime\start-dev.bat
```

### Run All Tests
```bash
.\scripts\testing\test-all-windows.bat
```

### Stop Everything
```bash
# If using Docker
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.dev.yml down

# If running locally
# Close all command windows
# Then clean up Docker containers
docker-compose -f docker-compose.dev.yml down
```

### View Logs
```bash
# Docker logs
docker-compose -f docker-compose.local.yml logs -f [service-name]

# Core logs
docker logs pcx_core -f

# All infrastructure logs
docker-compose -f docker-compose.dev.yml logs -f
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using the port (example for 3000)
netstat -ano | findstr :3000

# Kill the process
taskkill /PID [process-id] /F
```

### Docker Issues
```bash
# Clean everything
docker-compose down -v --remove-orphans
docker system prune -a

# Restart Docker Desktop
```

### Core Service Won't Compile in Docker
This is a known issue. Use one of these workarounds:
1. Run `start-complete.bat` (auto-fallback to local)
2. Run `start-dev.bat` (runs Core locally)
3. Use the mock server (automatically enabled in Docker)

### Database Connection Issues
```bash
# Reset databases
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up -d postgres redis
```

## Environment Variables

Create a `.env.development` file with:
```env
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-secret

# Optional
OPENAI_API_KEY=your-openai-key
ENABLE_REAL_AZURE_DATA=true
ENABLE_CACHE=true
```

## Quick Status Check

After starting services, verify they're running:

| Service | Health Check URL | Expected Response |
|---------|-----------------|-------------------|
| Frontend | http://localhost:3000 | Next.js app |
| Core API | http://localhost:8080/health | `{"status":"healthy"}` |
| GraphQL | http://localhost:4000/graphql | GraphQL Playground |
| PostgreSQL | `docker exec pcx_postgres psql -U postgres -c "SELECT 1"` | 1 |
| Redis | `docker exec pcx_redis redis-cli ping` | PONG |

## Recommended Development Workflow

1. **Initial Setup**
   ```bash
   # Clone repo
   git clone [repo-url]
   cd policycortex
   
   # Create environment file
   copy .env.example .env.development
   # Edit .env.development with your Azure credentials
   ```

2. **Start Development**
   ```bash
   # For first time or full testing
   .\scripts\runtime\start-complete.bat
   
   # For daily development
   .\scripts\runtime\start-dev.bat
   ```

3. **Before Committing**
   ```bash
   # Run full test suite
   .\scripts\testing\test-all-windows.bat
   ```

4. **Clean Shutdown**
   ```bash
   # Stop all services
   docker-compose -f docker-compose.local.yml down
   docker-compose -f docker-compose.dev.yml down
   ```