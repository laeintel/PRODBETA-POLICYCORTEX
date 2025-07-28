# PolicyCortex Comprehensive Testing Suite

This directory contains comprehensive local testing for all PolicyCortex microservices and components.

## Directory Structure

```
testing/
├── plans/           # Detailed test plans for each service
├── scripts/         # Executable test scripts
├── reports/         # Test execution reports
├── results/         # Raw test results and logs
├── configs/         # Test configuration files
└── README.md        # This file
```

## Quick Start

1. **Setup Test Environment:**
   ```bash
   cd testing/scripts
   ./setup-test-env.sh  # or setup-test-env.ps1 for Windows
   ```

2. **Run All Tests:**
   ```bash
   ./run-all-tests.sh
   ```

3. **Run Individual Service Tests:**
   ```bash
   ./test-api-gateway.sh
   ./test-azure-integration.sh
   ./test-ai-engine.sh
   # ... etc
   ```

4. **Generate Test Report:**
   ```bash
   ./generate-report.sh
   ```

## Test Coverage

### Microservices Testing
- **API Gateway**: Authentication, routing, rate limiting
- **Azure Integration**: Azure API interactions, service bus
- **AI Engine**: Model inference, NLP processing
- **Data Processing**: ETL pipelines, data validation
- **Conversation**: Natural language interface, context management
- **Notification**: Alert delivery, notification channels

### Frontend Testing
- Authentication flow (Azure AD)
- API integration
- Component rendering
- State management

### Integration Testing
- Inter-service communication
- End-to-end workflows
- Error handling and recovery
- Performance benchmarks

## Test Environments

- **Local**: Full microservices running on localhost
- **Docker**: Containerized testing environment
- **Mock**: Isolated unit tests with mocked dependencies

## Reporting

Test reports are generated in `reports/` directory with:
- Summary statistics
- Pass/fail rates
- Performance metrics
- Error logs
- Recommendations