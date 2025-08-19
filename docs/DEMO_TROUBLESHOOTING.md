# PolicyCortex Demo Troubleshooting Guide

## Common Issues and Solutions

### 1. Cold Start / Service Startup Issues

#### Problem: Services take too long to start
**Symptoms:**
- Frontend shows "Loading..." indefinitely
- API calls timeout
- GraphQL gateway not responding

**Solutions:**
```bash
# Pre-warm containers before demo
docker-compose -f docker-compose.local.yml up -d postgres redis core graphql frontend
# Wait 30-60 seconds for all services to initialize

# Check service health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Restart specific service if needed
docker-compose -f docker-compose.local.yml restart [service-name]
```

#### Problem: Rust compilation timeout
**Symptoms:**
- Core API fails to start
- cargo build hangs

**Solutions:**
```bash
# Build in release mode beforehand
cd core
cargo build --release

# Use pre-built binary
./target/release/policycortex-core

# Or use Docker image
docker run -p 8080:8080 policycortex-core:latest
```

### 2. Database Connection Issues

#### Problem: PostgreSQL not accessible
**Symptoms:**
- "Connection refused" errors
- Empty data in UI

**Solutions:**
```bash
# Check PostgreSQL status
docker exec policycortex-postgres pg_isready -U postgres

# Restart PostgreSQL
docker-compose -f docker-compose.local.yml restart postgres

# Re-run seed data
.\scripts\seed-data.bat  # Windows
./scripts/seed-data.sh    # Linux/Mac

# Verify data exists
docker exec -it policycortex-postgres psql -U postgres -d policycortex -c "SELECT COUNT(*) FROM organizations;"
```

### 3. Frontend Issues

#### Problem: Next.js dev server not starting
**Symptoms:**
- Port 3000 already in use
- Module not found errors

**Solutions:**
```bash
# Kill existing process on port 3000
# Windows
netstat -ano | findstr :3000
taskkill /F /PID [process-id]

# Linux/Mac
lsof -i :3000
kill -9 [process-id]

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

#### Problem: Blank page or components not rendering
**Symptoms:**
- White screen
- Console errors about missing components

**Solutions:**
```bash
# Clear Next.js cache
cd frontend
rm -rf .next
npm run dev

# Check for TypeScript errors
npm run type-check

# Verify environment variables
echo %NODE_ENV%           # Windows
echo $NODE_ENV            # Linux/Mac
```

### 4. GraphQL Issues

#### Problem: GraphQL queries failing
**Symptoms:**
- Network errors in browser console
- "Cannot read property of undefined"

**Solutions:**
```javascript
// Frontend automatically falls back to mock resolver
// Verify mock resolver is active:
// Check browser console for "Using GraphQL mock resolver" message

// Manual override in frontend/.env.local
USE_MOCK_GRAPHQL=true
```

### 5. Authentication Issues

#### Problem: MSAL authentication errors in demo
**Symptoms:**
- Login redirects fail
- "Tenant not found" errors

**Solutions:**
```bash
# Disable auth for demo
# In frontend/.env.local
REQUIRE_AUTH=false

# Or use demo mode
NODE_ENV=demo
```

### 6. Performance Issues

#### Problem: Slow API responses
**Symptoms:**
- Requests take >5 seconds
- UI freezes

**Solutions:**
```bash
# Increase Docker resources
# Docker Desktop > Settings > Resources
# - CPUs: 4+
# - Memory: 8GB+

# Use production builds
cd frontend && npm run build && npm start
cd core && cargo build --release && ./target/release/policycortex-core

# Enable caching
docker exec policycortex-redis redis-cli FLUSHALL
```

### 7. Demo Mode Issues

#### Problem: Real data showing instead of demo data
**Symptoms:**
- Actual Azure resources visible
- Real costs displayed

**Solutions:**
```bash
# Force demo mode
set USE_REAL_DATA=false     # Windows
export USE_REAL_DATA=false  # Linux/Mac

# Verify demo mode in UI
# Look for "Demo Mode" indicators in:
# - Security panel
# - Cost optimization panel
# - Tenant header
```

### 8. Port Conflicts

Common ports used by PolicyCortex:
- **3000**: Frontend (Next.js)
- **8080**: Core API (Rust)
- **4000**: GraphQL Gateway
- **5432**: PostgreSQL
- **6379**: Redis/DragonflyDB
- **2113**: EventStore
- **8081**: Adminer (DB UI)

```bash
# Find what's using a port (Windows)
netstat -ano | findstr :[port]

# Find what's using a port (Linux/Mac)
lsof -i :[port]

# Change ports in docker-compose.local.yml if needed
```

### 9. Quick Recovery Script

If demo fails during presentation, run:

```bash
# Windows - Quick Recovery
@echo off
echo Recovering PolicyCortex Demo...
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up -d postgres redis
timeout /t 5
call scripts\seed-data.bat
cd frontend && start cmd /c "npm run dev"
echo Recovery complete! Access http://localhost:3000

# Linux/Mac - Quick Recovery
#!/bin/bash
echo "Recovering PolicyCortex Demo..."
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up -d postgres redis
sleep 5
./scripts/seed-data.sh
cd frontend && npm run dev &
echo "Recovery complete! Access http://localhost:3000"
```

### 10. Pre-Demo Checklist

Run this before any demo:

- [ ] Docker Desktop running with adequate resources
- [ ] All containers healthy: `docker ps`
- [ ] Database seeded: Check for 3 organizations
- [ ] Frontend accessible: http://localhost:3000
- [ ] Core API healthy: http://localhost:8080/health
- [ ] Demo mode active: Check for demo indicators
- [ ] Network stable: No VPN conflicts
- [ ] Browser cache cleared
- [ ] Backup plan ready (screenshots/video)

### Emergency Fallback

If all else fails:
1. Use the production deployment: https://policycortex.azurecontainerapps.io
2. Show recorded demo video (keep updated version ready)
3. Use static screenshots in `/docs/demo-screenshots/`

### Support

For urgent issues during demo:
- Check #demo-support Slack channel
- Review latest known issues in GitHub Issues
- Contact: demo-support@policycortex.com