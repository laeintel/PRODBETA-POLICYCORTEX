# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolicyCortex is an AI-powered Azure governance platform built as a microservices architecture with:
- **Backend**: Python 3.11+ microservices using FastAPI
- **Frontend**: React 18 + TypeScript (skeleton structure)
- **Infrastructure**: Azure-native services deployed via Terraform to AKS
- **AI/ML**: PyTorch, Transformers, and Azure AI services

## Development Commands

### Backend Development
```bash
# Setup virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run individual services
cd services/api_gateway
uvicorn main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Format code
black . --line-length 100
isort . --profile black

# Lint code
flake8 . --max-line-length 100
mypy . --ignore-missing-imports
```

### Infrastructure
```bash
# Terraform commands (from infrastructure/terraform/)
terraform init
terraform plan -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars

# Docker build
docker build -t policycortex-api-gateway:latest -f backend/services/api_gateway/Dockerfile backend/
```

## Architecture Overview

### Microservices Structure
The platform consists of 6 core services in `backend/services/`:
1. **api_gateway**: Central entry point, authentication, routing
2. **azure_integration**: Direct Azure API integration
3. **ai_engine**: ML model inference and NLP
4. **data_processing**: ETL pipelines and stream processing
5. **conversation**: Natural language interface
6. **notification**: Alert and notification management

### Key Architectural Patterns
- **Service Communication**: REST APIs with FastAPI, event-driven via Azure Service Bus
- **Data Layer**: Azure SQL (structured), Cosmos DB (real-time), Redis (caching)
- **Security**: JWT auth, Azure AD integration, Key Vault for secrets
- **AI/ML**: Model registry, A/B testing, continuous learning

### Configuration Management
- Environment configs in `backend/core/config.py` using Pydantic Settings
- Secrets via Azure Key Vault
- Feature flags for gradual rollouts
- Environment-specific configs: dev, staging, prod

### Testing Strategy
- Unit tests with pytest in each service's `tests/` directory
- Integration tests for API endpoints
- Test fixtures in `conftest.py` files
- Mock Azure services for testing

## Important Notes

1. **Azure Integration**: All Azure operations go through `azure_integration` service
2. **Authentication**: JWT tokens required for all API calls except health checks
3. **Async Operations**: Use async/await patterns throughout for performance
4. **Error Handling**: Consistent error responses using `backend/core/exceptions.py`
5. **Logging**: Structured logging with correlation IDs for tracing
6. **Database Migrations**: Use Alembic for schema changes
7. **API Versioning**: Version in URL path (e.g., `/api/v1/`)

## Common Development Tasks

### Adding a New API Endpoint
1. Define route in appropriate service's `routes/` directory
2. Add business logic in `services/` directory
3. Update OpenAPI schema
4. Add unit tests
5. Update API documentation

### Modifying Database Schema
1. Update SQLAlchemy models in `models/`
2. Generate migration: `alembic revision --autogenerate -m "description"`
3. Review and apply migration: `alembic upgrade head`

### Adding AI/ML Models
1. Place model artifacts in `backend/services/ai_engine/models/`
2. Create model wrapper in `ml_models/`
3. Update model registry configuration
4. Add inference endpoint in AI engine service

### Infrastructure Changes
1. Modify appropriate Terraform module in `infrastructure/terraform/modules/`
2. Update environment-specific variables
3. Run `terraform plan` to review changes
4. Apply changes with appropriate approval
