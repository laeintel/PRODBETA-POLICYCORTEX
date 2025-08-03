# PolicyCortex API Integration Tests

This directory contains comprehensive Postman collections for testing the PolicyCortex API.

## 📁 Files

- `PolicyCortex-API-Tests.postman_collection.json` - Main test collection with all API endpoints
- `PolicyCortex-Local-Environment.postman_environment.json` - Environment variables for local testing
- `run-api-tests.js` - Newman test runner script for CI/CD integration

## 🚀 Quick Start

### Using Postman GUI

1. Open Postman
2. Import the collection: File → Import → Select `PolicyCortex-API-Tests.postman_collection.json`
3. Import the environment: File → Import → Select `PolicyCortex-Local-Environment.postman_environment.json`
4. Select "PolicyCortex Local Environment" from the environment dropdown
5. Run the collection: Collection → Run → Start Run

### Using Newman (Command Line)

1. Install Newman:
   ```bash
   npm install -g newman
   npm install -g newman-reporter-html
   ```

2. Run all tests:
   ```bash
   newman run PolicyCortex-API-Tests.postman_collection.json \
     -e PolicyCortex-Local-Environment.postman_environment.json \
     -r cli,html \
     --reporter-html-export test-results.html
   ```

3. Run specific folder:
   ```bash
   newman run PolicyCortex-API-Tests.postman_collection.json \
     -e PolicyCortex-Local-Environment.postman_environment.json \
     --folder "Authentication Flow"
   ```

### Using the Test Runner Script

1. Install dependencies:
   ```bash
   npm install newman newman-reporter-html
   ```

2. Run tests:
   ```bash
   node run-api-tests.js
   ```

## 📋 Test Coverage

### 1. Authentication Flow
- Health check endpoint
- User login with JWT tokens
- Get current user profile
- Token refresh
- Logout

### 2. Policy Management
- List all policies
- Get policy details with compliance data
- Search policies by query
- Filter policies by type/category
- Policy creation/update (admin only)

### 3. Compliance Monitoring
- Get overall compliance summary
- View compliance trends over time
- List non-compliant resources
- Get resource-specific compliance details
- Export compliance reports

### 4. Conversation & AI
- Start new conversation
- Send messages and receive AI responses
- Get conversation history
- Real-time WebSocket updates
- Delete conversations

### 5. Recommendations & Insights
- Get AI-powered recommendations
- Cost optimization insights
- Security recommendations
- Performance suggestions

### 6. Integration Tests
- Full workflow tests
- Data consistency validation
- Error handling scenarios
- Rate limiting tests

## 🔧 Environment Variables

Update these in the environment file before running:

- `base_url` - API base URL (default: http://localhost:8000)
- `test_username` - Test user email
- `test_password` - Test user password
- `azure_tenant_id` - Your Azure AD tenant ID
- `azure_client_id` - Your Azure AD app client ID

## ✅ Test Assertions

Each test includes assertions for:
- Response status codes
- Response time (<5s for AI endpoints, <1s for others)
- Required fields in response
- Data type validation
- Business logic validation

## 🔄 CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Run API Tests
  run: |
    npm install -g newman newman-reporter-html
    newman run postman/PolicyCortex-API-Tests.postman_collection.json \
      -e postman/PolicyCortex-Local-Environment.postman_environment.json \
      -r cli,junit \
      --reporter-junit-export results/api-tests.xml
```

## 🐛 Debugging Tips

1. **Check request console** in Postman for detailed request/response
2. **Use console.log()** in test scripts for debugging
3. **Enable request timeout** for slow endpoints
4. **Check collection variables** after failed tests
5. **Review newman HTML reports** for detailed failure analysis

## 📊 Performance Benchmarks

Expected response times:
- Authentication endpoints: <500ms
- Policy list/search: <1s
- AI chat responses: <5s
- Compliance calculations: <2s
- WebSocket connection: <100ms

## 🔒 Security Notes

- Never commit real credentials to the environment file
- Use Postman's environment variables for sensitive data
- Enable SSL certificate validation for production
- Rotate test user credentials regularly
- Use separate test tenant/subscription for testing