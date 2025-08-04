# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolicyCortex is an AI-powered Azure governance platform that transforms complex cloud management into intelligent, conversational experiences. It provides predictive insights, automated optimization, and natural language interaction for Azure Policy, RBAC, Network Security, and Cost Management.

## Architecture

The system follows a microservices architecture with:

- **Frontend**: React + TypeScript SPA with Azure AD authentication
- **API Gateway**: FastAPI-based gateway with authentication, rate limiting, and service routing
- **Core Services**: 5 Python microservices (Azure Integration, AI Engine, Data Processing, Conversation, Notification)
- **Infrastructure**: Azure Container Apps, AKS, SQL Database, Cosmos DB, Redis, ML Workspace
- **AI/ML**: PyTorch models, Azure OpenAI, Azure ML for predictive analytics

## Common Development Commands

### Frontend (React + TypeScript)
```bash
cd frontend
npm install                  # Install dependencies
npm run dev                 # Start development server (port 5173)
npm run build               # Build for production
npm run build:no-tsc        # Build without TypeScript checking
npm run lint                # Run ESLint
npm run lint:fix           # Fix ESLint issues
npm run type-check         # TypeScript type checking
npm run test               # Run Vitest tests
npm run test:ui            # Run tests with UI
npm run test:coverage      # Run tests with coverage
npm run format             # Format code with Prettier
npm run format:check       # Check formatting without fixing
npm run preview            # Preview production build
```

### Backend Services
Each service (api_gateway, azure_integration, ai_engine, data_processing, conversation, notification) has:
```bash
cd backend/services/{service_name}
pip install -r requirements.txt    # Install dependencies
python main.py                     # Run service directly
python main_simple.py             # Run simplified dev version
uvicorn main:app --reload --port 8000    # Run with hot reload
pytest                            # Run tests
pytest --cov=.                    # Run tests with coverage
pytest -v --tb=short              # Verbose with short traceback
flake8 . --max-line-length=100    # Lint Python code
black . --line-length=100         # Format Python code
isort . --profile black           # Sort imports
```

### Docker Development
```bash
# Frontend development with Azure backend services
docker-compose -f docker-compose.dev.yml up

# Full local development stack with all services
docker-compose -f docker-compose.local.yml up

# Simple development setup for quick testing
docker-compose -f docker-compose.simple.yml up

# Hybrid development: frontend dev with backend services
docker-compose -f docker-compose.hybrid.yml up

# Test environment with realistic data simulation
docker-compose -f docker-compose.test.yml up
```

### Infrastructure
```bash
# Terraform deployment with environment-specific configurations
cd infrastructure/terraform
terraform init
terraform plan -var-file="environments/dev/terraform.tfvars"
terraform apply -var-file="environments/dev/terraform.tfvars"

# Bicep deployment with parameter files
cd infrastructure/bicep
az deployment group create --resource-group rg-policycortex-dev --template-file main.bicep --parameters environments/dev.bicepparam

# Kubernetes deployment
cd infrastructure/kubernetes
kubectl apply -f manifests/

# Infrastructure automation scripts
./scripts/setup-keyvault-secrets.sh          # Provision Key Vault secrets
./scripts/sync-keyvault-to-containerapp.sh   # Sync secrets to Container Apps
```

### Testing
```bash
# Run comprehensive test suite with coverage and performance metrics
powershell -File testing/scripts/run-complete-test-coverage.ps1

# Quick test run for rapid feedback
powershell -File testing/scripts/quick-test.ps1

# Individual service tests with isolated environments
powershell -File testing/scripts/test-{service-name}.ps1

# Multi-stage test orchestration with result aggregation
powershell -File testing/scripts/run-all-tests.ps1

# Run a single test file with verbose output
cd backend/services/{service_name}
pytest tests/test_specific.py -v --tb=short

# Run tests with specific pattern and coverage
pytest -k "test_pattern" -v --cov=.

# Frontend test with watch mode and coverage
cd frontend
npm run test -- --watch --coverage
```

## Key Architectural Patterns

### Microservices Communication
- API Gateway acts as single entry point routing requests to services
- Services communicate via HTTP REST APIs
- Authentication handled centrally by API Gateway
- Circuit breaker pattern implemented for resilience

### Configuration Management
- **Hierarchical Configuration System**: Layered configuration classes (`DatabaseConfig`, `AzureConfig`, `AIConfig`, `SecurityConfig`) with environment-specific composition
- **Feature Flag Integration**: Built-in feature flags for controlled rollouts (`enable_cost_optimization`, `enable_policy_automation`)
- **Configuration Validation**: Automatic URL generation and cross-field validation using Pydantic validators
- Shared configuration in `backend/shared/config.py`
- Azure Key Vault for secrets in production with managed identity authentication
- Environment variables for local development with `.env` file cascading

### Database Architecture
- **Dual Database Architecture**: Both sync and async SQLAlchemy engines for different use cases
- **Audit Trail System**: Built-in `AuditLog` model with automatic change tracking for governance compliance
- **Service Health Monitoring**: `ServiceHealth` model for distributed service monitoring with database-backed status tracking
- **Transaction Management**: `async_db_transaction()` context manager for consistent transaction handling
- Azure SQL Database for relational data
- Azure Cosmos DB for real-time NoSQL data
- Redis for caching, session management, and rate limiting with sliding window algorithms

### AI/ML Pipeline
- **Patent-Protected Implementations**: Unified AI Platform with hierarchical neural networks, Cross-Domain Correlation Engine using Graph Neural Networks, and Conversational Intelligence System
- **Multi-Objective Optimization**: NSGA-II genetic algorithm integration for governance optimization
- **Context-Aware Processing**: Conversation state management with policy synthesis and natural language query translation
- Models in `backend/services/ai_engine/ml_models/` including specialized compliance predictors and correlation engines
- Azure ML Workspace for training and deployment
- PyTorch for deep learning models with custom GNN architectures
- Azure OpenAI for conversational AI with domain-specific fine-tuning

### Authentication & Security
- **Azure AD Integration**: MSAL-based authentication with automatic token refresh handling
- **Multi-Layered Security**: JWT token management with configurable lifecycles, environment-specific CORS policies
- **Circuit Breaker Pattern**: Three-state circuit breaker (CLOSED/OPEN/HALF_OPEN) with configurable failure thresholds
- **Advanced Rate Limiting**: Redis-based sliding window algorithm with burst allowances and user/IP-based limits
- RBAC for fine-grained permissions with Azure AD group integration
- Input validation with Pydantic models and automatic sanitization

## Service Structure

Each backend service follows this structure:
```
services/{service_name}/
├── main.py              # FastAPI app entry point
├── main_simple.py       # Simplified development version
├── auth.py              # Authentication logic
├── models.py            # Pydantic models
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── pytest.ini         # Test configuration
├── services/           # Business logic modules
└── tests/             # Unit and integration tests
```

### API Gateway Service Routes
- `/api/v1/azure/{path}` → Azure Integration Service
- `/api/v1/ai/{path}` → AI Engine Service  
- `/api/v1/chat/{path}` → Conversation Service
- `/api/v1/data/{path}` → Data Processing Service
- `/api/v1/notifications/{path}` → Notification Service

## Development Workflows

### Adding New Features
1. Create feature branch from main
2. Implement changes in appropriate service(s)
3. Add/update tests for new functionality
4. Run linting and type checking
5. Test locally using docker-compose
6. Create pull request with Azure Pipeline validation

### Deployment Pipeline
The Azure Pipeline (`azure-pipelines.yml`) provides:
- Multi-stage builds for all services
- Unit and integration testing
- Security scanning (OWASP, Bandit)
- Performance testing with K6
- Blue-green deployment to production

### Environment Configuration
- **Development**: Local development with docker-compose
- **Staging**: Azure Container Apps staging environment
- **Production**: Azure Container Apps with dedicated workload profiles

## Infrastructure as Code

### Terraform Modules
- `modules/networking/` - VNet, subnets, NSGs
- `modules/data-services/` - SQL, Cosmos DB, Redis
- `modules/ai-services/` - ML Workspace, Cognitive Services
- `modules/monitoring/` - Log Analytics, Application Insights

### Key Azure Resources
- Container Apps Environment with workload profiles
- User-assigned managed identity for service authentication
- Key Vault for secrets management
- Container Registry for Docker images
- Virtual Network with service-specific subnets

## Monitoring & Observability

- **Logging**: Structured logging with structlog
- **Metrics**: Prometheus metrics exposed at `/metrics`
- **Tracing**: Application Insights integration
- **Health Checks**: `/health` and `/ready` endpoints
- **Alerting**: Azure Monitor alerts for critical issues

## Security Considerations

- All services use managed identity for Azure authentication
- Secrets stored in Azure Key Vault
- Network isolation with VNet integration
- HTTPS/TLS encryption for all communications
- Input validation with Pydantic models
- Rate limiting and authentication on all APIs

## Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop
- Azure CLI (`az login` for authentication)
- Visual Studio Code with recommended extensions

### Quick Start for Local Development
```bash
# 1. Clone and setup
git clone <repository-url>
cd policycortex

# 2. Backend setup
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 3. Frontend setup
cd ../frontend
npm install

# 4. Run services (choose one)
# Option A: Docker Compose (recommended)
docker-compose -f docker-compose.simple.yml up

# Option B: Manual start
# Terminal 1: API Gateway
cd backend/services/api_gateway
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Service Ports
- Frontend: http://localhost:5173
- API Gateway: http://localhost:8000
- Azure Integration: http://localhost:8001
- AI Engine: http://localhost:8002
- Data Processing: http://localhost:8003
- Conversation: http://localhost:8004
- Notification: http://localhost:8005

## Advanced Development Patterns

### Resilience and Circuit Breaking
- **Circuit Breaker Implementation**: `backend/services/api_gateway/circuit_breaker.py` with three-state management
- **Health Check Integration**: Automatic service discovery and health monitoring with circuit breaker failover
- **Retry Strategies**: Exponential backoff with jitter for external service calls

### Configuration Patterns
- **Environment Cascading**: `.env` files override environment variables which override defaults
- **Feature Flag Management**: Runtime feature toggling without deployment for A/B testing and gradual rollouts
- **Secret Management**: Azure Key Vault integration with automatic secret rotation and managed identity authentication

### Service Communication Patterns
- **Service Discovery**: Environment-based URL configuration with fallback mechanisms
- **Request Context Propagation**: Distributed tracing context across microservice boundaries
- **Error Boundary Patterns**: Centralized error handling with service-specific error translation

### Testing Orchestration
- **Multi-Stage Testing**: PowerShell orchestration scripts coordinate testing across all services with dependency management
- **Test Environment Management**: Isolated test environments with realistic data simulation and cleanup automation
- **Performance Benchmarking**: Integrated load testing with K6 and performance regression detection

### Infrastructure Automation
- **Resource Naming Conventions**: Environment-specific resource naming with automatic Azure resource discovery
- **Secret Provisioning**: Automated Key Vault secret setup with service principal authentication
- **Container Image Management**: Multi-stage Docker builds with base image optimization and security scanning

## Troubleshooting Common Issues

### Python/Backend Issues
- **Import errors**: Ensure you're in the virtual environment
- **Module not found**: Check if you're in the correct directory (backend/services/{service_name})
- **Port already in use**: Kill existing processes or use different ports
- **Azure SDK errors**: Run `az login` and ensure credentials are valid

### Frontend Issues
- **npm install fails**: Clear cache with `npm cache clean --force`
- **TypeScript errors**: Run `npm run type-check` to see all issues
- **Build fails**: Try `npm run build:no-tsc` to skip type checking

### Docker Issues
- **Container fails to start**: Check logs with `docker logs <container-name>`
- **Network issues**: Ensure Docker Desktop is running and networks are created
- **Volume mount issues on Windows**: Use WSL2 or adjust paths in docker-compose