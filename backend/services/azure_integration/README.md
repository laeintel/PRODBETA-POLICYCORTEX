# Azure Integration Service

The Azure Integration Service is a core component of the PolicyCortex platform that provides comprehensive Azure SDK integrations for policy, RBAC, cost, network, and resource management.

## Features

### Policy Management
- List, create, update, and delete Azure policies
- Get policy compliance status and reports
- Policy assignment management
- Built-in and custom policy support

### RBAC Management
- Role definitions and assignments
- Permission analysis and validation
- Security recommendations
- Access control auditing

### Cost Management
- Cost usage analysis and reporting
- Budget management and alerts
- Cost forecasting and predictions
- Cost optimization recommendations

### Network Management
- Virtual network discovery and analysis
- Network security group management
- Security rule validation
- Network topology visualization

### Resource Management
- Resource discovery and inventory
- Resource group operations
- Tag management and governance
- Resource health monitoring

## Architecture

The service follows a modular architecture with:
- **FastAPI** for high-performance async API endpoints
- **Azure SDK** for native Azure service integration
- **Pydantic** for data validation and serialization
- **Structured logging** for comprehensive observability
- **Prometheus metrics** for monitoring and alerting
- **Circuit breaker pattern** for resilience

## API Endpoints

### Authentication
- `POST /auth/login` - Azure AD authentication
- `POST /auth/refresh` - Token refresh

### Policy Management
- `GET /api/v1/policies` - List policies
- `GET /api/v1/policies/{id}` - Get policy details
- `POST /api/v1/policies` - Create policy
- `PUT /api/v1/policies/{id}` - Update policy
- `DELETE /api/v1/policies/{id}` - Delete policy
- `GET /api/v1/policies/{id}/compliance` - Get compliance status

### RBAC Management
- `GET /api/v1/rbac/roles` - List roles
- `GET /api/v1/rbac/assignments` - List role assignments
- `POST /api/v1/rbac/assignments` - Create role assignment
- `DELETE /api/v1/rbac/assignments/{id}` - Delete role assignment

### Cost Management
- `GET /api/v1/costs/usage` - Get cost usage
- `GET /api/v1/costs/forecast` - Get cost forecast
- `GET /api/v1/costs/budgets` - List budgets
- `GET /api/v1/costs/recommendations` - Get cost recommendations

### Network Management
- `GET /api/v1/networks` - List virtual networks
- `GET /api/v1/networks/security-groups` - List NSGs
- `GET /api/v1/networks/security-analysis` - Analyze network security

### Resource Management
- `GET /api/v1/resources` - List resources
- `GET /api/v1/resources/{id}` - Get resource details
- `GET /api/v1/resources/groups` - List resource groups
- `POST /api/v1/resources/tags/{id}` - Update resource tags

## Installation

### Prerequisites
- Python 3.11+
- Azure subscription with appropriate permissions
- Redis for caching
- Azure AD application registration

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd backend/services/azure_integration
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure credentials and configuration
```

5. Run the service:
```bash
python main.py
```

### Docker Development

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the service:
   - API: http://localhost:8001
   - Health check: http://localhost:8001/health
   - API docs: http://localhost:8001/docs

## Configuration

### Azure Credentials
The service requires Azure service principal credentials:
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

### Required Azure Permissions
The service principal needs the following permissions:
- `Policy Contributor` for policy management
- `User Access Administrator` for RBAC management
- `Cost Management Contributor` for cost analysis
- `Network Contributor` for network management
- `Reader` for resource discovery

### Environment Variables
See `.env.example` for all available configuration options.

## Monitoring

### Health Checks
- `/health` - Basic health check
- `/ready` - Readiness check including Azure connectivity

### Metrics
- `/metrics` - Prometheus metrics endpoint

### Logging
Structured logging with correlation IDs for request tracing.

## Security

### Authentication
- JWT-based authentication
- Azure AD integration
- Role-based access control

### Authorization
- Request-level user context
- Subscription-level access control
- Resource-level permissions

### Rate Limiting
- Per-user rate limiting
- Azure API rate limit handling
- Circuit breaker for resilience

## Testing

Run tests with:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=. tests/
```

## Deployment

### Production Deployment
1. Build Docker image:
```bash
docker build -t azure-integration-service .
```

2. Deploy to container registry and orchestrator (AKS, Docker Swarm, etc.)

### Environment Configuration
- Set `ENVIRONMENT=production`
- Use Azure Key Vault for secrets
- Configure proper logging and monitoring
- Set up health checks and auto-scaling

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include proper logging and metrics
4. Write tests for new functionality
5. Update documentation

## License

This project is licensed under the MIT License.