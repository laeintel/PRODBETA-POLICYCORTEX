# PolicyCortex Comprehensive Testing Guide

## Overview

This guide provides detailed instructions for running comprehensive local tests on the PolicyCortex platform. The testing suite covers all microservices, frontend components, and integration scenarios.

## Prerequisites

Before running tests, ensure you have:

1. **Python 3.11+** installed
2. **Node.js 18+** and npm installed
3. **Docker Desktop** running
4. **PowerShell 7+** (on Windows)
5. **Git** for version control

## Quick Start

```powershell
# Navigate to testing directory
cd testing/scripts

# Setup test environment (first time only)
.\setup-test-env.ps1

# Run all tests
.\run-all-tests.ps1

# Run quick structural test
.\quick-test.ps1
```

## Test Structure

```
testing/
├── configs/          # Test configuration files
├── plans/            # Detailed test plans
├── reports/          # Generated test reports
├── results/          # Raw test results
└── scripts/          # Test execution scripts
```

## Individual Service Testing

### 1. API Gateway (Port 8000)
```powershell
.\test-api-gateway.ps1 -TestType all -Coverage
```
Tests:
- Authentication & JWT validation
- Service routing & load balancing
- Rate limiting & circuit breaker
- CORS configuration

### 2. Azure Integration (Port 8001)
```powershell
.\test-azure-integration.ps1 -TestType all
```
Tests:
- Azure API connections
- Resource management
- Cost management APIs
- Service Bus integration

### 3. AI Engine (Port 8002)
```powershell
.\test-ai-engine.ps1 -TestType all -Performance
```
Tests:
- Model inference
- Policy analysis
- Anomaly detection
- NLP processing

### 4. Data Processing (Port 8003)
```powershell
.\test-data-processing.ps1 -TestType all
```
Tests:
- ETL pipelines
- Data validation
- Stream processing
- Transformation logic

### 5. Conversation (Port 8004)
```powershell
.\test-conversation.ps1 -TestType all
```
Tests:
- Message handling
- Context management
- WebSocket connections
- Intent recognition

### 6. Notification (Port 8005)
```powershell
.\test-notification.ps1 -TestType all
```
Tests:
- Multi-channel delivery
- Template rendering
- Preference management
- Retry logic

### 7. Frontend (Port 3000)
```powershell
.\test-frontend.ps1 -TestType all
```
Tests:
- Build process
- TypeScript compilation
- Authentication flow
- Component rendering

## Test Types

### Unit Tests
- Test individual functions and components
- Mock external dependencies
- Fast execution
- Run with: `-TestType unit`

### Integration Tests
- Test service interactions
- Use test database
- Verify API contracts
- Run with: `-TestType integration`

### API Tests
- Test REST endpoints
- Verify request/response
- Check authentication
- Run with: `-TestType api`

### Performance Tests
- Measure response times
- Test under load
- Check resource usage
- Run with: `-TestType performance`

## Running Tests

### Sequential Execution
```powershell
.\run-all-tests.ps1 -Sequential
```

### Parallel Execution (Default)
```powershell
.\run-all-tests.ps1
```

### Quick Tests Only
```powershell
.\run-all-tests.ps1 -Quick
```

### With Coverage Reports
```powershell
.\run-all-tests.ps1 -Coverage
```

## Test Configuration

### Environment Variables
Edit `testing/configs/test.env`:
```env
API_GATEWAY_PORT=8000
AZURE_INTEGRATION_PORT=8001
# ... other configurations
```

### Mock Services
- Azure APIs use mock responses
- External services are stubbed
- Database uses test instance

## Interpreting Results

### Test Reports
Reports are generated in `testing/reports/`:
- `comprehensive_report.html` - Full HTML report
- `comprehensive_report.md` - Markdown summary
- Service-specific reports in subdirectories

### Success Criteria
- ✅ **PASSED**: All assertions successful
- ❌ **FAILED**: One or more assertions failed
- ⚠️ **SKIPPED**: Test skipped due to conditions

### Performance Benchmarks
| Service | Target Response Time | Acceptable Range |
|---------|---------------------|------------------|
| API Gateway | < 50ms | 10-200ms |
| Azure Integration | < 100ms | 50-500ms |
| AI Engine | < 200ms | 100-1000ms |
| Data Processing | < 150ms | 50-800ms |
| Conversation | < 100ms | 50-400ms |
| Notification | < 80ms | 20-300ms |

## Troubleshooting

### Service Won't Start
```powershell
# Check if port is already in use
netstat -ano | findstr :8000

# Kill process using port
taskkill /PID <process_id> /F
```

### Test Failures
1. Check service logs in `results/<service>/`
2. Verify environment configuration
3. Ensure dependencies are installed
4. Check Docker containers are running

### Missing Dependencies
```powershell
# Reinstall Python dependencies
cd backend/services/<service>
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Reinstall frontend dependencies
cd frontend
npm install
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    cd testing/scripts
    ./setup-test-env.ps1
    ./run-all-tests.ps1 -Coverage
```

### Azure DevOps
```yaml
- script: |
    cd testing/scripts
    pwsh setup-test-env.ps1
    pwsh run-all-tests.ps1
  displayName: 'Run Tests'
```

## Best Practices

1. **Run tests before commits**
   ```powershell
   .\run-all-tests.ps1 -Quick
   ```

2. **Full test suite before deployment**
   ```powershell
   .\run-all-tests.ps1 -Coverage
   ```

3. **Test specific changes**
   ```powershell
   .\test-api-gateway.ps1 -TestType api
   ```

4. **Clean test environment regularly**
   ```powershell
   docker stop policycortex-postgres-test policycortex-redis-test
   docker rm policycortex-postgres-test policycortex-redis-test
   ```

## Advanced Testing

### Custom Test Scenarios
Create custom tests in `testing/scripts/custom/`:
```powershell
# Example: End-to-end user journey
.\test-user-journey.ps1
```

### Load Testing
```powershell
# Run load tests (requires additional setup)
.\run-load-tests.ps1 -Users 100 -Duration 300
```

### Security Testing
```powershell
# Run security scans
.\run-security-tests.ps1
```

## Maintenance

### Update Test Dependencies
```powershell
# Update all service dependencies
.\update-test-deps.ps1
```

### Clean Test Data
```powershell
# Remove old test results
.\cleanup-test-data.ps1 -DaysOld 7
```

## Support

For issues or questions:
1. Check test logs in `results/`
2. Review service-specific documentation
3. Contact the development team

---

*Last Updated: $(Get-Date -Format "yyyy-MM-dd")*