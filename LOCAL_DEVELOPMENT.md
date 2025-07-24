# PolicyCortex Local Development Guide

This guide provides comprehensive instructions for setting up and running PolicyCortex locally.

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ and npm
- Python 3.11+
- Azure CLI (optional, for Azure resource access)
- Git

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd policycortex
```

### 2. Create Environment Files

Create `.env` files for each service with the required environment variables:

#### Backend Services `.env` Template

Create these files:
- `backend/services/api_gateway/.env`
- `backend/services/azure_integration/.env`
- `backend/services/ai_engine/.env`
- `backend/services/data_processing/.env`
- `backend/services/conversation/.env`
- `backend/services/notification/.env`

```env
# Environment Configuration
ENVIRONMENT=development
SERVICE_NAME=api_gateway  # Change for each service
PORT=8000  # 8000-8005 for each service

# Security
JWT_SECRET_KEY=local-dev-secret-key-change-in-production

# Azure Configuration (use dummy values for local)
AZURE_CLIENT_ID=00000000-0000-0000-0000-000000000000
TENANT_ID=00000000-0000-0000-0000-000000000000
CLIENT_ID=00000000-0000-0000-0000-000000000000
CLIENT_SECRET=dummy-secret
SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000
RESOURCE_GROUP=local-dev
KEY_VAULT_NAME=local-dev
MANAGED_IDENTITY_CLIENT_ID=00000000-0000-0000-0000-000000000000

# Storage Configuration
STORAGE_ACCOUNT_NAME=localstoragedev

# Database Configuration (Cosmos DB Emulator)
COSMOS_CONNECTION_STRING=AccountEndpoint=https://localhost:8081/;AccountKey=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==
COSMOS_ENDPOINT=https://localhost:8081/
COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==

# SQL Database (optional)
SQL_SERVER=localhost
SQL_USERNAME=sa
SQL_PASSWORD=YourStrong@Passw0rd

# Redis Configuration
REDIS_CONNECTION_STRING=localhost:6379,password=,ssl=False,abortConnect=False

# Application Insights (dummy for local)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://eastus-1.in.applicationinsights.azure.com/

# Cognitive Services (dummy for local)
COGNITIVE_SERVICES_KEY=dummy-key
COGNITIVE_SERVICES_ENDPOINT=https://dummy.cognitiveservices.azure.com/

# Service Bus (dummy for local)
SERVICE_BUS_NAMESPACE=dummy-namespace

# Machine Learning (dummy for local)
ML_WORKSPACE_NAME=dummy-workspace

# Feature Flags
ENABLE_REDIS_CACHE=true
ENABLE_COSMOS_DB=true
ENABLE_SQL_DB=false
ENABLE_SERVICE_BUS=false
ENABLE_ML_FEATURES=false

# Logging
LOG_LEVEL=DEBUG

# Local Development
USE_LOCAL_STORAGE=true
USE_IN_MEMORY_CACHE=true
BYPASS_AUTH=true  # Set to true for easier local testing

# CORS (for API Gateway only)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

#### Frontend `.env` Template

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_AZURE_CLIENT_ID=00000000-0000-0000-0000-000000000000
VITE_AZURE_TENANT_ID=00000000-0000-0000-0000-000000000000
VITE_AZURE_REDIRECT_URI=http://localhost:5173
VITE_APP_VERSION=1.0.0-local
```

### 3. Run with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.local.yml up -d

# View logs
docker-compose -f docker-compose.local.yml logs -f

# Stop all services
docker-compose -f docker-compose.local.yml down
```

### 4. Run Services Individually (Alternative)

#### Start Infrastructure Services
```bash
# Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Cosmos DB Emulator (Windows/Linux only)
docker run -d -p 8081:8081 -p 10251-10254:10251-10254 --name cosmosdb mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest
```

#### Start Backend Services
```bash
# API Gateway
cd backend/services/api_gateway
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Repeat for other services on ports 8001-8005
```

#### Start Frontend
```bash
cd frontend
npm install
npm run dev
```

## Service URLs

- **Frontend**: http://localhost:5173
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Azure Integration**: http://localhost:8001
- **AI Engine**: http://localhost:8002
- **Data Processing**: http://localhost:8003
- **Conversation**: http://localhost:8004
- **Notification**: http://localhost:8005
- **Redis**: localhost:6379
- **Cosmos DB Emulator**: https://localhost:8081

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   netstat -ano | findstr :8000  # Windows
   lsof -i :8000  # Mac/Linux
   ```

2. **Cosmos DB Emulator Certificate Issues**
   - Download certificate from https://localhost:8081/_explorer/emulator.pem
   - Install it as trusted certificate

3. **Service Can't Connect to Redis/Cosmos**
   - Ensure infrastructure services are running
   - Check firewall settings
   - Verify connection strings in .env files

4. **Frontend Can't Connect to Backend**
   - Ensure API Gateway is running on port 8000
   - Check CORS configuration
   - Verify VITE_API_BASE_URL in frontend .env

### Debug Mode

To run services in debug mode:

```bash
# Backend service with debugging
cd backend/services/api_gateway
python -m debugpy --listen 5678 --wait-for-client -m uvicorn main:app --reload

# Frontend with debugging
cd frontend
npm run dev -- --debug
```

## Testing

### Run Unit Tests
```bash
# Backend tests
cd backend/services/api_gateway
pytest tests/

# Frontend tests
cd frontend
npm test
```

### Run Integration Tests
```bash
# Ensure all services are running
docker-compose -f docker-compose.local.yml up -d

# Run integration tests
cd tests/integration
pytest
```

## Development Workflow

1. **Make Code Changes**
2. **Test Locally** using the setup above
3. **Run Tests** to ensure nothing is broken
4. **Commit Changes** with descriptive messages
5. **Push to Branch** and create PR
6. **CI/CD Pipeline** will deploy to dev environment

## Environment-Specific Configuration

### Using Real Azure Services Locally

If you want to connect to real Azure services:

1. Login to Azure CLI:
   ```bash
   az login
   ```

2. Update .env files with real Azure resource values:
   - Get values from Key Vault
   - Update connection strings
   - Set BYPASS_AUTH=false

3. Ensure your IP is whitelisted in Azure services

### Switching Between Environments

Use different .env files:
- `.env.local` - Local development with emulators
- `.env.dev` - Connected to dev Azure resources
- `.env.staging` - Connected to staging resources

## Additional Tools

### Monitoring
- **Redis Commander**: `docker run -d -p 8081:8081 rediscommander/redis-commander`
- **Cosmos DB Explorer**: Access at https://localhost:8081/_explorer/index.html

### API Testing
- Use Postman or Thunder Client
- Import API collection from `docs/api/postman_collection.json`
- Set environment variables for local testing

## Tips

1. **Use Docker Compose** for consistent environment
2. **Enable hot reloading** for faster development
3. **Use debug logging** to troubleshoot issues
4. **Test with minimal services** first, then add complexity
5. **Document any local changes** needed for your setup 