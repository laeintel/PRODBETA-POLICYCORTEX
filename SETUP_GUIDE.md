# PolicyCortex Setup Guide

## 🚀 Quick Start

Run the complete setup with one command:

```bash
# Windows (Run as Administrator)
.\scripts\setup\setup-all.bat

# Linux/Mac
./scripts/setup/setup-all.sh
```

This will automatically:
- ✅ Check and install prerequisites
- ✅ Set up Docker services (databases, cache, monitoring)
- ✅ Configure Azure resources
- ✅ Set up GitHub secrets and variables
- ✅ Initialize databases with schema
- ✅ Configure authentication (MSAL)
- ✅ Deploy ML models
- ✅ Start all services
- ✅ Run smoke tests

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (latest version)
- **Node.js** (v18+ LTS)
- **Rust** (latest stable)
- **Python** (3.11+)
- **Azure CLI** (latest)
- **GitHub CLI** (latest)
- **Git** (latest)

### Azure Requirements
- Active Azure subscription
- Owner or Contributor role
- Resource provider registrations:
  - Microsoft.ContainerRegistry
  - Microsoft.ContainerService
  - Microsoft.CognitiveServices
  - Microsoft.KeyVault

### GitHub Requirements
- Repository admin access
- Personal access token (for GitHub CLI)

## 🔧 Manual Setup Steps

If you prefer manual setup or need to configure specific components:

### 1. Environment Configuration

```bash
# Copy environment templates
cp .env.development .env.local
cp .env.production .env.production.local

# Edit with your values
notepad .env.local  # Windows
nano .env.local     # Linux/Mac
```

### 2. Docker Services

```bash
# Start all infrastructure services
docker-compose -f scripts/setup/docker-services.yml up -d

# Verify services are running
docker ps

# Check service health
docker-compose -f scripts/setup/docker-services.yml ps
```

### 3. Database Setup

```bash
# Initialize database schema
docker exec -i policycortex-postgres psql -U postgres -d policycortex < scripts/setup/init-db.sql

# Verify database
docker exec -it policycortex-postgres psql -U postgres -d policycortex -c "\dt"
```

### 4. Azure Resources

```bash
# Login to Azure
az login

# Run Azure setup
.\scripts\setup\azure-setup.bat  # Windows
./scripts/setup/azure-setup.sh   # Linux/Mac
```

### 5. GitHub Configuration

```bash
# Login to GitHub
gh auth login

# Set up secrets and variables
.\scripts\setup\github-setup.bat  # Windows
./scripts/setup/github-setup.sh   # Linux/Mac
```

### 6. Application Build

```bash
# Build frontend
cd frontend
npm install
npm run build

# Build backend (Rust)
cd ../core
cargo build --release

# Build Python services
cd ../backend/services/api_gateway
pip install -r requirements.txt
```

### 7. Start Services

```bash
# Development mode
.\scripts\runtime\start-dev.bat  # Windows
./scripts/runtime/start-dev.sh   # Linux/Mac

# Production mode
.\scripts\runtime\start-production.bat  # Windows
./scripts/runtime/start-production.sh   # Linux/Mac
```

## 🌐 Service URLs

Once running, access services at:

### Application
- **Frontend**: http://localhost:3000 (dev) / http://localhost:3005 (docker)
- **Backend API**: http://localhost:8080 (dev) / http://localhost:8085 (docker)
- **GraphQL**: http://localhost:4000/graphql
- **API Gateway**: http://localhost:8000

### Infrastructure
- **PostgreSQL**: localhost:5432
- **DragonflyDB**: localhost:6379
- **EventStore**: http://localhost:2113
- **NATS**: localhost:4222

### Monitoring
- **Grafana**: http://localhost:3010 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Adminer**: http://localhost:8081

### ML/AI
- **MLflow**: http://localhost:5000
- **Training API**: http://localhost:8001

## 🔑 Authentication Setup

### Azure AD (MSAL)
1. Register application in Azure AD
2. Configure redirect URIs:
   - http://localhost:3000 (development)
   - https://your-domain.com (production)
3. Update `.env.local` with:
   ```
   NEXT_PUBLIC_AZURE_AD_CLIENT_ID=your-client-id
   NEXT_PUBLIC_AZURE_AD_TENANT_ID=your-tenant-id
   ```

### Service Principal
```bash
# Create service principal
az ad sp create-for-rbac --name "PolicyCortex-SP" --role Contributor

# Save the output to GitHub secrets
```

## 🧪 Testing

### Unit Tests
```bash
# Frontend
cd frontend && npm test

# Backend (Rust)
cd core && cargo test

# Python
cd backend/services/api_gateway && pytest
```

### E2E Tests
```bash
cd frontend
npm run test:e2e
```

### Integration Tests
```bash
.\scripts\testing\test-all-windows.bat  # Windows
./scripts/testing/test-all-linux.sh     # Linux/Mac
```

## 🚀 Deployment

### Development
```bash
# Deploy to dev environment
gh workflow run deploy-aks.yml -f environment=dev
```

### Production
```bash
# Deploy to production
gh workflow run deploy-aks.yml -f environment=prod
```

## 📊 Monitoring

### Business Metrics
Access Grafana dashboards:
1. Navigate to http://localhost:3010
2. Login with admin/admin
3. View dashboards:
   - Business KPIs
   - Compliance Metrics
   - Cost Optimization
   - Security Posture

### Technical Metrics
- Application performance
- Infrastructure health
- Error rates and logs
- Distributed tracing

## 🔧 Troubleshooting

### Docker Issues
```bash
# Reset Docker services
docker-compose -f scripts/setup/docker-services.yml down -v
docker-compose -f scripts/setup/docker-services.yml up -d

# Check logs
docker-compose -f scripts/setup/docker-services.yml logs -f [service-name]
```

### Database Connection
```bash
# Test connection
docker exec -it policycortex-postgres psql -U postgres -c "SELECT 1"

# Reset database
docker exec -i policycortex-postgres psql -U postgres -c "DROP DATABASE IF EXISTS policycortex"
docker exec -i policycortex-postgres psql -U postgres -c "CREATE DATABASE policycortex"
docker exec -i policycortex-postgres psql -U postgres -d policycortex < scripts/setup/init-db.sql
```

### Azure Authentication
```bash
# Clear Azure CLI cache
az account clear
az login

# Verify subscription
az account show
```

### Port Conflicts
```bash
# Find process using port (Windows)
netstat -ano | findstr :3000

# Kill process (Windows)
taskkill /PID [process-id] /F

# Find and kill process (Linux/Mac)
lsof -i :3000
kill -9 [process-id]
```

## 📚 Additional Resources

- [Architecture Documentation](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Patent Implementation Guide](docs/PATENTS.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/laeintel/policycortex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/laeintel/policycortex/discussions)
- **Email**: support@policycortex.com

## 📄 License

PolicyCortex is protected by patents and proprietary technology.
See [LICENSE](LICENSE) for details.