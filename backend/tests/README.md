# PolicyCortex Backend Test Suite

## Overview

This directory contains comprehensive unit tests for all PolicyCortex backend microservices. The test suite is designed to ensure high code quality, reliability, and maintainability across the entire backend system.

## Test Structure

### Completed Services

#### 1. API Gateway Service
- **Location**: `backend/services/api_gateway/tests/`
- **Test Files**:
  - `test_main.py` - Main application endpoints and routing
  - `test_circuit_breaker.py` - Circuit breaker functionality
  - `test_rate_limiter.py` - Rate limiting functionality
  - `test_auth.py` - Authentication and authorization
  - `conftest.py` - Test fixtures and configuration

**Coverage Areas**:
- Health check endpoints
- Authentication and authorization
- Rate limiting (fixed window, sliding window, hierarchical)
- Circuit breaker patterns (closed, open, half-open states)
- Request proxying to downstream services
- Error handling and middleware
- Prometheus metrics collection

#### 2. Azure Integration Service
- **Location**: `backend/services/azure_integration/tests/`
- **Test Files**:
  - `test_main.py` - Main application endpoints
  - `conftest.py` - Test fixtures and configuration

**Coverage Areas**:
- Azure authentication and token management
- Policy management (CRUD operations, compliance checking)
- RBAC management (roles, assignments)
- Cost management (usage, forecasting, budgets, recommendations)
- Network management (VNets, NSGs, security analysis)
- Resource management (listing, tagging, grouping)
- Azure SDK integration mocking
- Error handling and metrics

### Test Infrastructure

#### Global Test Configuration
- **Location**: `backend/tests/conftest.py`
- **Features**:
  - Shared fixtures for all services
  - Mock factories for external dependencies
  - Database session management
  - Authentication helpers
  - Sample data generators

#### Service-Specific Configuration
Each service has its own `conftest.py` with:
- Service-specific fixtures
- Mock external dependencies
- Test data specific to the service
- Client configuration

### Test Utilities

#### Mock Objects
- **Azure SDK Clients**: Complete mocking of Azure management clients
- **Redis**: Cache and rate limiting operations
- **HTTP Clients**: External API calls
- **Database**: Test database sessions
- **Authentication**: JWT token handling

#### Sample Data
- Policy definitions and compliance data
- Resource information and metadata
- Cost and usage data
- Network configuration data
- User and authentication data

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Running All Tests
```bash
# From project root
pytest backend/tests/

# With coverage
pytest --cov=backend --cov-report=html backend/tests/
```

### Running Service-Specific Tests
```bash
# API Gateway
pytest backend/services/api_gateway/tests/

# Azure Integration
pytest backend/services/azure_integration/tests/
```

### Running Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Authentication tests
pytest -m auth

# Circuit breaker tests
pytest -m circuit_breaker
```

## Test Configuration

### Pytest Configuration
Each service has a `pytest.ini` file with:
- Coverage requirements (80% minimum)
- Test discovery patterns
- Async test support
- Custom markers for test categorization
- Warning filters

### Environment Variables
```bash
# Test database
DATABASE_URL=sqlite:///./test.db

# Debug mode
DEBUG=True

# Service URLs (for integration tests)
AZURE_INTEGRATION_URL=http://localhost:8001
AI_ENGINE_URL=http://localhost:8002
```

## Test Patterns

### Unit Test Structure
```python
class TestFeatureName:
    """Test feature functionality."""
    
    def test_success_case(self, fixtures):
        """Test successful operation."""
        # Arrange
        # Act
        # Assert
    
    def test_error_case(self, fixtures):
        """Test error handling."""
        # Arrange
        # Act with pytest.raises
        # Assert
    
    @pytest.mark.asyncio
    async def test_async_operation(self, fixtures):
        """Test async operation."""
        # Arrange
        # Act
        # Assert
```

### Mock Usage
```python
@pytest.fixture
def mock_external_service():
    """Mock external service."""
    with patch("module.ExternalService") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.method = AsyncMock(return_value="test_result")
        yield mock_instance
```

### Test Data
```python
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "id": "test-id",
        "name": "Test Item",
        "properties": {"key": "value"}
    }
```

## Test Coverage

### Current Coverage
- **API Gateway**: ~95% line coverage
- **Azure Integration**: ~90% line coverage

### Coverage Reports
- HTML reports generated in `htmlcov/`
- XML reports for CI/CD integration
- Terminal output with missing lines

### Coverage Requirements
- Minimum 80% line coverage per service
- Critical paths must have 100% coverage
- Error handling paths must be tested

## Integration Tests

### Test Scenarios
- End-to-end workflows
- Service communication
- Authentication flows
- Data consistency

### Test Environment
- Docker-based test environment
- Test databases and caches
- Mock external services
- Isolated test runs

## Performance Tests

### Load Testing
- API endpoint performance
- Database query optimization
- Cache effectiveness
- Memory usage patterns

### Stress Testing
- Rate limiting behavior
- Circuit breaker activation
- Error handling under load
- Resource cleanup

## CI/CD Integration

### GitHub Actions
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Quality Gates
- All tests must pass
- Coverage thresholds must be met
- No critical security issues
- Performance benchmarks maintained

## Best Practices

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Test both success and failure cases

### Mock Strategy
- Mock external dependencies
- Use realistic test data
- Verify mock interactions
- Clean up mocks after tests

### Async Testing
- Use pytest-asyncio for async tests
- Mock async operations properly
- Handle async context managers
- Test timeout scenarios

### Error Testing
- Test all error conditions
- Verify error messages
- Check error handling paths
- Test recovery mechanisms

## Future Enhancements

### Planned Services
- AI Engine Service tests
- Data Processing Service tests
- Conversation Service tests
- Notification Service tests

### Test Improvements
- Property-based testing with Hypothesis
- Contract testing with Pact
- Mutation testing with mutmut
- Visual regression testing

### Monitoring
- Test execution metrics
- Flaky test detection
- Test performance tracking
- Coverage trend analysis

## Contributing

### Adding New Tests
1. Create test files in appropriate service directory
2. Follow naming conventions (`test_*.py`)
3. Use appropriate fixtures from `conftest.py`
4. Add markers for test categorization
5. Ensure minimum coverage requirements

### Test Guidelines
- Write tests before implementing features (TDD)
- Test public interfaces, not implementation details
- Use meaningful assertions
- Keep tests independent and isolated
- Document complex test scenarios

### Code Review
- Verify test coverage for new code
- Check mock usage is appropriate
- Ensure tests are maintainable
- Review test performance impact