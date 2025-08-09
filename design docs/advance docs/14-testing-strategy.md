# 14. Testing Strategy

## Table of Contents
1. [Testing Philosophy](#testing-philosophy)
2. [Test Pyramid](#test-pyramid)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [End-to-End Testing](#end-to-end-testing)
6. [Performance Testing](#performance-testing)
7. [Security Testing](#security-testing)
8. [API Testing](#api-testing)
9. [Frontend Testing](#frontend-testing)
10. [AI Model Testing](#ai-model-testing)
11. [Database Testing](#database-testing)
12. [Load Testing](#load-testing)
13. [Chaos Engineering](#chaos-engineering)
14. [Test Data Management](#test-data-management)
15. [Continuous Testing](#continuous-testing)
16. [Test Reporting](#test-reporting)

## Testing Philosophy

### Core Principles
1. **Shift-Left Testing**: Catch issues early in the development cycle
2. **Test Automation**: Minimize manual testing through comprehensive automation
3. **Risk-Based Testing**: Focus testing efforts on high-risk, high-impact areas
4. **Continuous Feedback**: Provide immediate feedback to developers
5. **Test-Driven Development**: Write tests before implementation when appropriate

### Quality Gates
```yaml
# Quality gates for different stages
Development:
  - Unit test coverage: 85%+
  - Integration test coverage: 70%+
  - Zero critical security vulnerabilities
  - Code quality score: A+

Staging:
  - All unit and integration tests pass
  - E2E test coverage: 60%+
  - Performance benchmarks meet SLA
  - Security scan passes

Production:
  - All test suites pass
  - Load testing validates capacity
  - Chaos engineering tests pass
  - Monitoring alerts configured
```

## Test Pyramid

### Testing Levels Distribution
```
       /\
      /  \     E2E Tests (10%)
     /    \    - Happy path scenarios
    /      \   - Critical user journeys
   /________\  - Cross-browser testing
   
  /__________\  Integration Tests (20%)
 /            \ - API integration tests
/              \- Database integration
\              /- Service communication
 \____________/ - External service mocks

/________________\ Unit Tests (70%)
|                | - Business logic
|                | - Individual functions
|                | - Component behavior
|________________| - Error conditions
```

## Unit Testing

### Rust Backend Unit Tests
```rust
// core/src/api/policies/mod.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Policy;
    use crate::test_utils::{create_test_db, create_mock_policy};
    use axum_test::TestServer;
    use serde_json::json;

    #[tokio::test]
    async fn test_create_policy_success() {
        // Arrange
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();
        let policy_data = json!({
            "name": "Test Policy",
            "description": "Test policy description",
            "rules": [{
                "condition": "resource.type == 'Microsoft.Compute/virtualMachines'",
                "action": "deny"
            }],
            "category": "Security"
        });

        // Act
        let response = server
            .post("/api/v1/policies")
            .json(&policy_data)
            .await;

        // Assert
        assert_eq!(response.status_code(), 201);
        let policy: Policy = response.json();
        assert_eq!(policy.name, "Test Policy");
        assert_eq!(policy.rules.len(), 1);
    }

    #[tokio::test]
    async fn test_validate_policy_invalid_syntax() {
        // Arrange
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();
        let invalid_policy = json!({
            "name": "Invalid Policy",
            "rules": [{
                "condition": "invalid syntax here",
                "action": "deny"
            }]
        });

        // Act
        let response = server
            .post("/api/v1/policies")
            .json(&invalid_policy)
            .await;

        // Assert
        assert_eq!(response.status_code(), 400);
        let error: serde_json::Value = response.json();
        assert!(error["message"].as_str().unwrap().contains("syntax"));
    }

    #[test]
    fn test_policy_rule_evaluation() {
        // Arrange
        let rule = PolicyRule {
            condition: "resource.type == 'Microsoft.Compute/virtualMachines'".to_string(),
            action: PolicyAction::Deny,
        };
        let resource = json!({
            "type": "Microsoft.Compute/virtualMachines",
            "name": "vm-test-001"
        });

        // Act
        let result = rule.evaluate(&resource);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PolicyAction::Deny);
    }

    #[test]
    fn test_correlation_engine() {
        // Arrange
        let engine = CrossDomainCorrelationEngine::new();
        let security_event = SecurityEvent {
            event_type: "unauthorized_access".to_string(),
            resource_id: "vm-001".to_string(),
            severity: "high".to_string(),
            timestamp: Utc::now(),
        };
        let cost_anomaly = CostAnomaly {
            resource_id: "vm-001".to_string(),
            cost_increase: 150.0,
            threshold: 20.0,
            timestamp: Utc::now(),
        };

        // Act
        let correlation = engine.correlate(&security_event, &cost_anomaly);

        // Assert
        assert!(correlation.is_some());
        let corr = correlation.unwrap();
        assert_eq!(corr.strength, CorrelationStrength::High);
        assert!(corr.confidence > 0.8);
    }
}

// Test utilities
// core/src/test_utils.rs
use axum::Router;
use sqlx::PgPool;
use std::env;

pub async fn create_test_db() -> PgPool {
    let database_url = env::var("TEST_DATABASE_URL")
        .expect("TEST_DATABASE_URL must be set");
    
    let pool = PgPool::connect(&database_url).await.unwrap();
    
    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    
    pool
}

pub async fn create_test_app() -> Router {
    let pool = create_test_db().await;
    create_app(pool).await
}

pub fn create_mock_policy() -> Policy {
    Policy {
        id: uuid::Uuid::new_v4(),
        name: "Mock Policy".to_string(),
        description: Some("Mock policy for testing".to_string()),
        rules: vec![PolicyRule {
            condition: "true".to_string(),
            action: PolicyAction::Allow,
        }],
        category: PolicyCategory::Security,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}
```

### JavaScript/TypeScript Unit Tests
```typescript
// frontend/components/PolicyEditor/PolicyEditor.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PolicyEditor } from './PolicyEditor';
import { createMockPolicy } from '../../test-utils/mocks';
import * as policyApi from '../../lib/api/policies';

// Mock API calls
jest.mock('../../lib/api/policies');
const mockPolicyApi = policyApi as jest.Mocked<typeof policyApi>;

describe('PolicyEditor', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <PolicyEditor {...props} />
      </QueryClientProvider>
    );
  };

  it('should render policy editor form', () => {
    renderComponent();
    
    expect(screen.getByLabelText(/policy name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
    expect(screen.getByText(/add rule/i)).toBeInTheDocument();
  });

  it('should validate required fields', async () => {
    renderComponent();
    
    const submitButton = screen.getByText(/save policy/i);
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/policy name is required/i)).toBeInTheDocument();
    });
  });

  it('should create policy successfully', async () => {
    const mockPolicy = createMockPolicy();
    mockPolicyApi.createPolicy.mockResolvedValue(mockPolicy);

    renderComponent();
    
    fireEvent.change(screen.getByLabelText(/policy name/i), {
      target: { value: 'Test Policy' },
    });
    fireEvent.change(screen.getByLabelText(/description/i), {
      target: { value: 'Test description' },
    });

    const submitButton = screen.getByText(/save policy/i);
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockPolicyApi.createPolicy).toHaveBeenCalledWith({
        name: 'Test Policy',
        description: 'Test description',
        rules: [],
        category: 'Security',
      });
    });
  });

  it('should handle API errors gracefully', async () => {
    mockPolicyApi.createPolicy.mockRejectedValue(
      new Error('API Error: Invalid policy syntax')
    );

    renderComponent();
    
    fireEvent.change(screen.getByLabelText(/policy name/i), {
      target: { value: 'Test Policy' },
    });

    const submitButton = screen.getByText(/save policy/i);
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/invalid policy syntax/i)).toBeInTheDocument();
    });
  });
});

// frontend/lib/utils/policyValidation.test.ts
import { validatePolicyRule, PolicyValidationError } from './policyValidation';

describe('policyValidation', () => {
  describe('validatePolicyRule', () => {
    it('should validate correct policy syntax', () => {
      const rule = "resource.type == 'Microsoft.Compute/virtualMachines'";
      const result = validatePolicyRule(rule);
      
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect syntax errors', () => {
      const rule = "resource.type == 'unclosed string";
      const result = validatePolicyRule(rule);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Unterminated string literal');
    });

    it('should detect unsupported operators', () => {
      const rule = "resource.type ~= 'pattern'";
      const result = validatePolicyRule(rule);
      
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Unsupported operator: ~=');
    });
  });
});
```

## Integration Testing

### API Integration Tests
```rust
// core/tests/integration_tests.rs
use policycortex_core::create_app;
use axum_test::TestServer;
use serde_json::json;
use sqlx::PgPool;
use std::env;

#[tokio::test]
async fn test_policy_workflow_integration() {
    let pool = setup_test_db().await;
    let app = create_app(pool).await;
    let server = TestServer::new(app).unwrap();

    // 1. Create a policy
    let policy_data = json!({
        "name": "VM Security Policy",
        "description": "Ensure VMs have security configurations",
        "rules": [{
            "condition": "resource.type == 'Microsoft.Compute/virtualMachines' && !resource.properties.securityProfile",
            "action": "deny"
        }],
        "category": "Security"
    });

    let create_response = server
        .post("/api/v1/policies")
        .json(&policy_data)
        .await;

    assert_eq!(create_response.status_code(), 201);
    let policy: serde_json::Value = create_response.json();
    let policy_id = policy["id"].as_str().unwrap();

    // 2. Evaluate policy against resources
    let resource_data = json!({
        "resources": [{
            "type": "Microsoft.Compute/virtualMachines",
            "name": "vm-test-001",
            "properties": {
                "vmSize": "Standard_D2s_v3"
                // Missing securityProfile
            }
        }]
    });

    let eval_response = server
        .post(&format!("/api/v1/policies/{}/evaluate", policy_id))
        .json(&resource_data)
        .await;

    assert_eq!(eval_response.status_code(), 200);
    let eval_result: serde_json::Value = eval_response.json();
    assert_eq!(eval_result["violations"].as_array().unwrap().len(), 1);

    // 3. Get policy insights
    let insights_response = server
        .get(&format!("/api/v1/policies/{}/insights", policy_id))
        .await;

    assert_eq!(insights_response.status_code(), 200);
    let insights: serde_json::Value = insights_response.json();
    assert!(insights["compliance_rate"].as_f64().unwrap() < 100.0);

    // 4. Update policy
    let update_data = json!({
        "description": "Updated security policy with enhanced rules"
    });

    let update_response = server
        .patch(&format!("/api/v1/policies/{}", policy_id))
        .json(&update_data)
        .await;

    assert_eq!(update_response.status_code(), 200);

    // 5. Delete policy
    let delete_response = server
        .delete(&format!("/api/v1/policies/{}", policy_id))
        .await;

    assert_eq!(delete_response.status_code(), 204);
}

#[tokio::test]
async fn test_cross_domain_correlation() {
    let pool = setup_test_db().await;
    let app = create_app(pool).await;
    let server = TestServer::new(app).unwrap();

    // 1. Submit security event
    let security_event = json!({
        "event_type": "unauthorized_access",
        "resource_id": "vm-001",
        "severity": "high",
        "details": {
            "source_ip": "192.168.1.100",
            "user": "unknown"
        }
    });

    server
        .post("/api/v1/events/security")
        .json(&security_event)
        .await;

    // 2. Submit cost anomaly
    let cost_event = json!({
        "resource_id": "vm-001",
        "cost_increase": 200.0,
        "threshold": 20.0,
        "anomaly_type": "sudden_spike"
    });

    server
        .post("/api/v1/events/cost")
        .json(&cost_event)
        .await;

    // 3. Check correlations
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let correlations_response = server
        .get("/api/v1/correlations?resource_id=vm-001")
        .await;

    assert_eq!(correlations_response.status_code(), 200);
    let correlations: serde_json::Value = correlations_response.json();
    let items = correlations["items"].as_array().unwrap();
    
    assert!(items.len() > 0);
    let correlation = &items[0];
    assert_eq!(correlation["strength"], "High");
    assert!(correlation["confidence"].as_f64().unwrap() > 0.7);
}

async fn setup_test_db() -> PgPool {
    let database_url = env::var("TEST_DATABASE_URL")
        .expect("TEST_DATABASE_URL must be set for integration tests");
    
    let pool = PgPool::connect(&database_url).await.unwrap();
    
    // Clean database
    sqlx::query("TRUNCATE TABLE policies, policy_evaluations, events CASCADE")
        .execute(&pool)
        .await
        .unwrap();
    
    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    
    pool
}
```

### Database Integration Tests
```rust
// core/tests/database_tests.rs
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[sqlx::test]
async fn test_policy_crud_operations(pool: PgPool) {
    // Create
    let policy_id = Uuid::new_v4();
    let result = sqlx::query!(
        r#"
        INSERT INTO policies (id, name, description, rules, category)
        VALUES ($1, $2, $3, $4, $5)
        "#,
        policy_id,
        "Test Policy",
        Some("Test description"),
        serde_json::json!([{
            "condition": "resource.type == 'test'",
            "action": "allow"
        }]),
        "Security"
    )
    .execute(&pool)
    .await;

    assert!(result.is_ok());

    // Read
    let policy = sqlx::query!(
        "SELECT * FROM policies WHERE id = $1",
        policy_id
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(policy.name, "Test Policy");
    assert_eq!(policy.category, "Security");

    // Update
    let update_result = sqlx::query!(
        "UPDATE policies SET description = $1 WHERE id = $2",
        "Updated description",
        policy_id
    )
    .execute(&pool)
    .await;

    assert!(update_result.is_ok());
    assert_eq!(update_result.unwrap().rows_affected(), 1);

    // Delete
    let delete_result = sqlx::query!(
        "DELETE FROM policies WHERE id = $1",
        policy_id
    )
    .execute(&pool)
    .await;

    assert!(delete_result.is_ok());
    assert_eq!(delete_result.unwrap().rows_affected(), 1);
}

#[sqlx::test]
async fn test_policy_evaluation_history(pool: PgPool) {
    // Insert test policy
    let policy_id = Uuid::new_v4();
    sqlx::query!(
        r#"
        INSERT INTO policies (id, name, rules, category)
        VALUES ($1, $2, $3, $4)
        "#,
        policy_id,
        "History Test Policy",
        serde_json::json!([]),
        "Compliance"
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert evaluation records
    for i in 0..10 {
        sqlx::query!(
            r#"
            INSERT INTO policy_evaluations 
            (id, policy_id, resource_id, result, evaluation_time)
            VALUES ($1, $2, $3, $4, NOW() - INTERVAL '%s seconds')
            "#,
            Uuid::new_v4(),
            policy_id,
            format!("resource-{}", i),
            if i % 2 == 0 { "passed" } else { "failed" },
            i * 60
        )
        .execute(&pool)
        .await
        .unwrap();
    }

    // Query evaluation history
    let evaluations = sqlx::query!(
        r#"
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN result = 'passed' THEN 1 END) as passed,
               COUNT(CASE WHEN result = 'failed' THEN 1 END) as failed
        FROM policy_evaluations 
        WHERE policy_id = $1
        "#,
        policy_id
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(evaluations.total, Some(10));
    assert_eq!(evaluations.passed, Some(5));
    assert_eq!(evaluations.failed, Some(5));
}
```

## End-to-End Testing

### Playwright E2E Tests
```typescript
// e2e/tests/policy-management.spec.ts
import { test, expect, Page } from '@playwright/test';

test.describe('Policy Management', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    
    // Login
    await page.goto('http://localhost:3000/login');
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'password123');
    await page.click('[data-testid="login-button"]');
    
    // Wait for redirect to dashboard
    await page.waitForURL('**/dashboard');
  });

  test('should create new policy', async () => {
    // Navigate to policies page
    await page.click('[data-testid="nav-policies"]');
    await page.waitForURL('**/policies');

    // Click create policy button
    await page.click('[data-testid="create-policy-button"]');

    // Fill policy form
    await page.fill('[data-testid="policy-name"]', 'E2E Test Policy');
    await page.fill('[data-testid="policy-description"]', 'Created by E2E test');
    
    // Select category
    await page.selectOption('[data-testid="policy-category"]', 'Security');

    // Add rule
    await page.click('[data-testid="add-rule-button"]');
    await page.fill(
      '[data-testid="rule-condition"]',
      "resource.type == 'Microsoft.Compute/virtualMachines'"
    );
    await page.selectOption('[data-testid="rule-action"]', 'deny');

    // Save policy
    await page.click('[data-testid="save-policy-button"]');

    // Verify success message
    await expect(page.locator('[data-testid="success-message"]')).toContainText(
      'Policy created successfully'
    );

    // Verify policy appears in list
    await page.waitForURL('**/policies');
    await expect(page.locator('[data-testid="policy-list"]')).toContainText(
      'E2E Test Policy'
    );
  });

  test('should evaluate policy against resources', async () => {
    // Create test policy first
    await createTestPolicy(page);

    // Navigate to evaluation page
    await page.click('[data-testid="nav-evaluation"]');
    await page.waitForURL('**/evaluation');

    // Select policy
    await page.selectOption('[data-testid="policy-select"]', 'E2E Test Policy');

    // Upload resource file or enter manually
    await page.fill('[data-testid="resource-data"]', JSON.stringify({
      resources: [{
        type: 'Microsoft.Compute/virtualMachines',
        name: 'test-vm',
        properties: { vmSize: 'Standard_B1s' }
      }]
    }));

    // Run evaluation
    await page.click('[data-testid="evaluate-button"]');

    // Wait for results
    await page.waitForSelector('[data-testid="evaluation-results"]');

    // Verify results
    await expect(page.locator('[data-testid="violations-count"]')).toContainText('1');
    await expect(page.locator('[data-testid="violation-details"]')).toContainText(
      'test-vm'
    );
  });

  test('should show policy insights and analytics', async () => {
    await createTestPolicy(page);

    // Navigate to analytics
    await page.click('[data-testid="nav-analytics"]');
    await page.waitForURL('**/analytics');

    // Select policy for detailed view
    await page.click('[data-testid="policy-E2E Test Policy"]');

    // Verify charts are displayed
    await expect(page.locator('[data-testid="compliance-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="trend-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="resource-breakdown"]')).toBeVisible();

    // Verify metrics
    await expect(page.locator('[data-testid="total-evaluations"]')).toBeVisible();
    await expect(page.locator('[data-testid="compliance-rate"]')).toBeVisible();
    await expect(page.locator('[data-testid="avg-response-time"]')).toBeVisible();
  });

  test('should handle real-time updates via WebSocket', async () => {
    await page.goto('http://localhost:3000/dashboard');

    // Wait for WebSocket connection
    await page.waitForFunction(() => window.WebSocket !== undefined);

    // Listen for notifications
    const notificationPromise = page.waitForSelector('[data-testid="notification"]');

    // Trigger server-side event (simulate policy violation)
    await page.evaluate(() => {
      // This would normally be triggered by actual policy evaluation
      window.dispatchEvent(new CustomEvent('policy-violation', {
        detail: {
          policy: 'Security Policy',
          resource: 'vm-001',
          severity: 'high'
        }
      }));
    });

    // Verify notification appears
    const notification = await notificationPromise;
    await expect(notification).toContainText('Policy violation detected');
  });

  async function createTestPolicy(page: Page) {
    await page.goto('http://localhost:3000/policies/new');
    await page.fill('[data-testid="policy-name"]', 'E2E Test Policy');
    await page.fill('[data-testid="policy-description"]', 'Test policy for E2E');
    await page.selectOption('[data-testid="policy-category"]', 'Security');
    await page.click('[data-testid="add-rule-button"]');
    await page.fill(
      '[data-testid="rule-condition"]',
      "resource.type == 'Microsoft.Compute/virtualMachines'"
    );
    await page.selectOption('[data-testid="rule-action"]', 'deny');
    await page.click('[data-testid="save-policy-button"]');
    await page.waitForURL('**/policies');
  }
});
```

### Cypress E2E Tests
```typescript
// cypress/e2e/ai-conversation.cy.ts
describe('AI Conversation System', () => {
  beforeEach(() => {
    cy.login('admin@example.com', 'password');
    cy.visit('/chat');
  });

  it('should handle natural language policy queries', () => {
    // Type natural language query
    cy.get('[data-cy="chat-input"]')
      .type('Show me all VMs that are not compliant with security policies{enter}');

    // Wait for AI response
    cy.get('[data-cy="ai-response"]', { timeout: 10000 })
      .should('be.visible')
      .and('contain', 'I found');

    // Verify results display
    cy.get('[data-cy="query-results"]').should('be.visible');
    cy.get('[data-cy="resource-item"]').should('have.length.at.least', 1);

    // Verify export options
    cy.get('[data-cy="export-button"]').should('be.visible');
  });

  it('should provide policy recommendations', () => {
    cy.get('[data-cy="chat-input"]')
      .type('Recommend security policies for my Azure environment{enter}');

    cy.get('[data-cy="ai-response"]', { timeout: 15000 })
      .should('contain', 'recommendations');

    // Verify recommendation cards
    cy.get('[data-cy="recommendation-card"]').should('have.length.at.least', 3);
    
    // Click on first recommendation
    cy.get('[data-cy="recommendation-card"]').first().click();
    cy.get('[data-cy="recommendation-details"]').should('be.visible');

    // Apply recommendation
    cy.get('[data-cy="apply-recommendation"]').click();
    cy.get('[data-cy="success-notification"]')
      .should('contain', 'Policy created successfully');
  });

  it('should explain policy violations', () => {
    // Navigate to violations page
    cy.visit('/violations');
    
    // Click on a violation
    cy.get('[data-cy="violation-item"]').first().click();
    
    // Click explain button
    cy.get('[data-cy="explain-violation"]').click();
    
    // Verify AI explanation
    cy.get('[data-cy="violation-explanation"]', { timeout: 10000 })
      .should('be.visible')
      .and('contain', 'This violation occurred because');

    // Verify remediation suggestions
    cy.get('[data-cy="remediation-suggestions"]')
      .should('be.visible')
      .find('[data-cy="suggestion-item"]')
      .should('have.length.at.least', 1);
  });
});
```

## Performance Testing

### Load Testing with Artillery
```yaml
# performance/load-test.yml
config:
  target: 'http://localhost:8080'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      name: "Normal load"
    - duration: 60
      arrivalRate: 100
      name: "High load"
    - duration: 30
      arrivalRate: 200
      name: "Stress test"
  
  variables:
    auth_token: "{{ $env.TEST_AUTH_TOKEN }}"
  
  processor: "./load-test-functions.js"

scenarios:
  - name: "Policy CRUD Operations"
    weight: 40
    flow:
      - post:
          url: "/api/v1/policies"
          headers:
            Authorization: "Bearer {{ auth_token }}"
          json:
            name: "Load Test Policy {{ $randomString() }}"
            description: "Created during load test"
            rules:
              - condition: "resource.type == 'test'"
                action: "allow"
            category: "Testing"
          capture:
            - json: "$.id"
              as: "policy_id"
      
      - get:
          url: "/api/v1/policies/{{ policy_id }}"
          headers:
            Authorization: "Bearer {{ auth_token }}"
      
      - patch:
          url: "/api/v1/policies/{{ policy_id }}"
          headers:
            Authorization: "Bearer {{ auth_token }}"
          json:
            description: "Updated during load test"
      
      - delete:
          url: "/api/v1/policies/{{ policy_id }}"
          headers:
            Authorization: "Bearer {{ auth_token }}"

  - name: "Policy Evaluation"
    weight: 60
    flow:
      - post:
          url: "/api/v1/evaluate"
          headers:
            Authorization: "Bearer {{ auth_token }}"
          json:
            resources:
              - type: "Microsoft.Compute/virtualMachines"
                name: "vm-{{ $randomInt(1, 1000) }}"
                properties:
                  vmSize: "Standard_D2s_v3"
          expect:
            - statusCode: 200
            - hasProperty: "results"
```

### Performance Test Functions
```javascript
// performance/load-test-functions.js
module.exports = {
  generateTestResource: function(userContext, events, done) {
    const resourceTypes = [
      'Microsoft.Compute/virtualMachines',
      'Microsoft.Storage/storageAccounts',
      'Microsoft.Network/virtualNetworks',
      'Microsoft.KeyVault/vaults'
    ];
    
    userContext.vars.resourceType = resourceTypes[
      Math.floor(Math.random() * resourceTypes.length)
    ];
    userContext.vars.resourceName = `resource-${Math.random().toString(36).substr(2, 9)}`;
    
    return done();
  },

  validateResponse: function(requestParams, response, context, ee, next) {
    if (response.statusCode === 200) {
      const body = JSON.parse(response.body);
      if (body.results && Array.isArray(body.results)) {
        ee.emit('histogram', 'policy_evaluation.results_count', body.results.length);
      }
    }
    return next();
  }
};
```

### Rust Backend Benchmarks
```rust
// core/benches/policy_evaluation.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use policycortex_core::engine::{PolicyEngine, PolicyRule};
use serde_json::json;

fn benchmark_policy_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_evaluation");
    
    let engine = PolicyEngine::new();
    let rule = PolicyRule {
        condition: "resource.type == 'Microsoft.Compute/virtualMachines' && resource.properties.vmSize == 'Standard_D2s_v3'".to_string(),
        action: crate::models::PolicyAction::Deny,
    };

    // Single resource evaluation
    let resource = json!({
        "type": "Microsoft.Compute/virtualMachines",
        "name": "test-vm",
        "properties": {
            "vmSize": "Standard_D2s_v3",
            "location": "eastus"
        }
    });

    group.bench_function("single_resource", |b| {
        b.iter(|| {
            engine.evaluate_rule(black_box(&rule), black_box(&resource))
        })
    });

    // Multiple resources evaluation
    for resource_count in [10, 100, 1000] {
        let resources: Vec<_> = (0..resource_count)
            .map(|i| json!({
                "type": "Microsoft.Compute/virtualMachines",
                "name": format!("vm-{}", i),
                "properties": {
                    "vmSize": if i % 2 == 0 { "Standard_D2s_v3" } else { "Standard_B1s" },
                    "location": "eastus"
                }
            }))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("multiple_resources", resource_count),
            &resources,
            |b, resources| {
                b.iter(|| {
                    for resource in resources {
                        engine.evaluate_rule(black_box(&rule), black_box(resource));
                    }
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_policy_evaluation);
criterion_main!(benches);
```

## Security Testing

### OWASP ZAP Integration
```python
# security/zap_scan.py
import time
import requests
import json
from zapv2 import ZAPv2

class SecurityScanner:
    def __init__(self, proxy_host='localhost', proxy_port=8080):
        self.zap = ZAPv2(proxies={
            'http': f'http://{proxy_host}:{proxy_port}',
            'https': f'http://{proxy_host}:{proxy_port}'
        })
        self.target_url = 'http://localhost:3000'
        self.api_url = 'http://localhost:8080'

    def perform_scan(self):
        print("Starting security scan...")
        
        # Spider the application
        print("Spidering application...")
        scan_id = self.zap.spider.scan(self.target_url)
        
        while int(self.zap.spider.status(scan_id)) < 100:
            print(f"Spider progress: {self.zap.spider.status(scan_id)}%")
            time.sleep(2)

        # Active scan
        print("Starting active scan...")
        scan_id = self.zap.ascan.scan(self.target_url)
        
        while int(self.zap.ascan.status(scan_id)) < 100:
            print(f"Active scan progress: {self.zap.ascan.status(scan_id)}%")
            time.sleep(5)

        # Generate report
        print("Generating report...")
        alerts = self.zap.core.alerts(baseurl=self.target_url)
        
        high_risk = [alert for alert in alerts if alert['risk'] == 'High']
        medium_risk = [alert for alert in alerts if alert['risk'] == 'Medium']
        
        report = {
            'scan_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target': self.target_url,
            'total_alerts': len(alerts),
            'high_risk': len(high_risk),
            'medium_risk': len(medium_risk),
            'alerts': alerts
        }
        
        with open('security_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

    def test_api_security(self):
        """Test API-specific security vulnerabilities"""
        test_results = []
        
        # Test SQL injection
        sql_payloads = ["'; DROP TABLE policies; --", "1' OR '1'='1", "admin'/*"]
        for payload in sql_payloads:
            response = requests.get(f"{self.api_url}/api/v1/policies", 
                                  params={'name': payload})
            test_results.append({
                'test': 'SQL Injection',
                'payload': payload,
                'status_code': response.status_code,
                'vulnerable': 'error' not in response.text.lower()
            })
        
        # Test authentication bypass
        endpoints = ['/api/v1/policies', '/api/v1/users', '/api/v1/admin']
        for endpoint in endpoints:
            response = requests.get(f"{self.api_url}{endpoint}")
            test_results.append({
                'test': 'Authentication Bypass',
                'endpoint': endpoint,
                'status_code': response.status_code,
                'vulnerable': response.status_code == 200
            })
        
        return test_results

if __name__ == '__main__':
    scanner = SecurityScanner()
    report = scanner.perform_scan()
    api_results = scanner.test_api_security()
    
    print(f"Security scan completed. Found {report['high_risk']} high-risk vulnerabilities.")
```

### Penetration Testing Script
```bash
#!/bin/bash
# security/pentest.sh

set -e

TARGET_HOST="localhost:3000"
API_HOST="localhost:8080"
REPORT_DIR="security/reports/$(date +%Y%m%d-%H%M%S)"

mkdir -p "$REPORT_DIR"

echo "🔐 Starting penetration testing suite"
echo "Target: $TARGET_HOST"
echo "API: $API_HOST"
echo "Report directory: $REPORT_DIR"

# 1. Port scanning
echo "📡 Scanning ports..."
nmap -sS -O "$TARGET_HOST" > "$REPORT_DIR/nmap_scan.txt"

# 2. SSL/TLS testing
echo "🔒 Testing SSL/TLS configuration..."
testssl.sh --jsonfile "$REPORT_DIR/ssl_test.json" "https://$TARGET_HOST"

# 3. Directory brute force
echo "📁 Directory enumeration..."
gobuster dir -u "http://$TARGET_HOST" -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -o "$REPORT_DIR/directories.txt"

# 4. API testing with custom payloads
echo "🔍 Testing API endpoints..."
python3 security/api_security_test.py --host "$API_HOST" --output "$REPORT_DIR/api_test.json"

# 5. Cross-site scripting (XSS) testing
echo "⚡ XSS testing..."
python3 security/xss_test.py --host "$TARGET_HOST" --output "$REPORT_DIR/xss_test.json"

# 6. Generate consolidated report
echo "📊 Generating consolidated report..."
python3 security/generate_report.py --input-dir "$REPORT_DIR" --output "$REPORT_DIR/consolidated_report.html"

echo "✅ Penetration testing completed"
echo "📋 Report available at: $REPORT_DIR/consolidated_report.html"
```

## API Testing

### Postman/Newman API Tests
```json
{
  "info": {
    "name": "PolicyCortex API Tests",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "variable": [
    {
      "key": "baseUrl",
      "value": "http://localhost:8080",
      "type": "string"
    }
  ],
  "item": [
    {
      "name": "Authentication",
      "item": [
        {
          "name": "Login",
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test('Login successful', function () {",
                  "    pm.response.to.have.status(200);",
                  "    const response = pm.response.json();",
                  "    pm.expect(response.token).to.exist;",
                  "    pm.collectionVariables.set('authToken', response.token);",
                  "});"
                ]
              }
            }
          ],
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n    \"email\": \"admin@example.com\",\n    \"password\": \"password123\"\n}"
            },
            "url": {
              "raw": "{{baseUrl}}/api/v1/auth/login",
              "host": ["{{baseUrl}}"],
              "path": ["api", "v1", "auth", "login"]
            }
          }
        }
      ]
    },
    {
      "name": "Policies",
      "item": [
        {
          "name": "Create Policy",
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test('Policy created successfully', function () {",
                  "    pm.response.to.have.status(201);",
                  "    const response = pm.response.json();",
                  "    pm.expect(response.id).to.exist;",
                  "    pm.expect(response.name).to.equal('API Test Policy');",
                  "    pm.collectionVariables.set('policyId', response.id);",
                  "});",
                  "",
                  "pm.test('Response time is acceptable', function () {",
                  "    pm.expect(pm.response.responseTime).to.be.below(1000);",
                  "});"
                ]
              }
            }
          ],
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Authorization",
                "value": "Bearer {{authToken}}"
              },
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n    \"name\": \"API Test Policy\",\n    \"description\": \"Policy created during API testing\",\n    \"rules\": [\n        {\n            \"condition\": \"resource.type == 'Microsoft.Compute/virtualMachines'\",\n            \"action\": \"deny\"\n        }\n    ],\n    \"category\": \"Security\"\n}"
            },
            "url": {
              "raw": "{{baseUrl}}/api/v1/policies",
              "host": ["{{baseUrl}}"],
              "path": ["api", "v1", "policies"]
            }
          }
        },
        {
          "name": "Get Policy",
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test('Policy retrieved successfully', function () {",
                  "    pm.response.to.have.status(200);",
                  "    const response = pm.response.json();",
                  "    pm.expect(response.id).to.equal(pm.collectionVariables.get('policyId'));",
                  "    pm.expect(response.rules).to.be.an('array').that.is.not.empty;",
                  "});"
                ]
              }
            }
          ],
          "request": {
            "method": "GET",
            "header": [
              {
                "key": "Authorization",
                "value": "Bearer {{authToken}}"
              }
            ],
            "url": {
              "raw": "{{baseUrl}}/api/v1/policies/{{policyId}}",
              "host": ["{{baseUrl}}"],
              "path": ["api", "v1", "policies", "{{policyId}}"]
            }
          }
        },
        {
          "name": "Evaluate Policy",
          "event": [
            {
              "listen": "test",
              "script": {
                "exec": [
                  "pm.test('Policy evaluation successful', function () {",
                  "    pm.response.to.have.status(200);",
                  "    const response = pm.response.json();",
                  "    pm.expect(response.results).to.be.an('array');",
                  "    pm.expect(response.summary).to.exist;",
                  "});",
                  "",
                  "pm.test('Evaluation results contain violations', function () {",
                  "    const response = pm.response.json();",
                  "    pm.expect(response.results.length).to.be.greaterThan(0);",
                  "    pm.expect(response.results[0].result).to.equal('violation');",
                  "});"
                ]
              }
            }
          ],
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Authorization",
                "value": "Bearer {{authToken}}"
              },
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n    \"resources\": [\n        {\n            \"type\": \"Microsoft.Compute/virtualMachines\",\n            \"name\": \"test-vm-001\",\n            \"properties\": {\n                \"vmSize\": \"Standard_D2s_v3\",\n                \"location\": \"eastus\"\n            }\n        }\n    ]\n}"
            },
            "url": {
              "raw": "{{baseUrl}}/api/v1/policies/{{policyId}}/evaluate",
              "host": ["{{baseUrl}}"],
              "path": ["api", "v1", "policies", "{{policyId}}", "evaluate"]
            }
          }
        }
      ]
    }
  ]
}
```

### REST Assured API Tests (Java)
```java
// src/test/java/com/policycortex/api/PolicyApiTest.java
import io.restassured.RestAssured;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.*;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

public class PolicyApiTest {
    
    private static String authToken;
    private static String policyId;
    
    @BeforeAll
    public static void setup() {
        RestAssured.baseURI = "http://localhost";
        RestAssured.port = 8080;
        RestAssured.basePath = "/api/v1";
        
        // Authenticate and get token
        authToken = given()
            .contentType(ContentType.JSON)
            .body("{\n" +
                  "  \"email\": \"admin@example.com\",\n" +
                  "  \"password\": \"password123\"\n" +
                  "}")
            .when()
            .post("/auth/login")
            .then()
            .statusCode(200)
            .extract()
            .path("token");
    }
    
    @Test
    @Order(1)
    public void testCreatePolicy() {
        policyId = given()
            .header("Authorization", "Bearer " + authToken)
            .contentType(ContentType.JSON)
            .body("{\n" +
                  "  \"name\": \"Java Test Policy\",\n" +
                  "  \"description\": \"Created by Java REST Assured test\",\n" +
                  "  \"rules\": [\n" +
                  "    {\n" +
                  "      \"condition\": \"resource.type == 'Microsoft.Storage/storageAccounts'\",\n" +
                  "      \"action\": \"audit\"\n" +
                  "    }\n" +
                  "  ],\n" +
                  "  \"category\": \"Compliance\"\n" +
                  "}")
            .when()
            .post("/policies")
            .then()
            .statusCode(201)
            .body("name", equalTo("Java Test Policy"))
            .body("category", equalTo("Compliance"))
            .body("rules", hasSize(1))
            .time(lessThan(2000L))
            .extract()
            .path("id");
        
        Assertions.assertNotNull(policyId);
    }
    
    @Test
    @Order(2)
    public void testGetPolicy() {
        given()
            .header("Authorization", "Bearer " + authToken)
            .when()
            .get("/policies/" + policyId)
            .then()
            .statusCode(200)
            .body("id", equalTo(policyId))
            .body("name", equalTo("Java Test Policy"))
            .body("rules[0].condition", containsString("Microsoft.Storage/storageAccounts"));
    }
    
    @Test
    @Order(3)
    public void testUpdatePolicy() {
        given()
            .header("Authorization", "Bearer " + authToken)
            .contentType(ContentType.JSON)
            .body("{\n" +
                  "  \"description\": \"Updated by Java test\"\n" +
                  "}")
            .when()
            .patch("/policies/" + policyId)
            .then()
            .statusCode(200)
            .body("description", equalTo("Updated by Java test"));
    }
    
    @Test
    @Order(4)
    public void testEvaluatePolicy() {
        given()
            .header("Authorization", "Bearer " + authToken)
            .contentType(ContentType.JSON)
            .body("{\n" +
                  "  \"resources\": [\n" +
                  "    {\n" +
                  "      \"type\": \"Microsoft.Storage/storageAccounts\",\n" +
                  "      \"name\": \"teststorageaccount\",\n" +
                  "      \"properties\": {\n" +
                  "        \"accessTier\": \"Hot\",\n" +
                  "        \"encryption\": {\n" +
                  "          \"services\": {\n" +
                  "            \"blob\": {\n" +
                  "              \"enabled\": true\n" +
                  "            }\n" +
                  "          }\n" +
                  "        }\n" +
                  "      }\n" +
                  "    }\n" +
                  "  ]\n" +
                  "}")
            .when()
            .post("/policies/" + policyId + "/evaluate")
            .then()
            .statusCode(200)
            .body("results", hasSize(greaterThan(0)))
            .body("summary.total_resources", equalTo(1))
            .body("summary.evaluations", equalTo(1));
    }
    
    @Test
    @Order(5)
    public void testDeletePolicy() {
        given()
            .header("Authorization", "Bearer " + authToken)
            .when()
            .delete("/policies/" + policyId)
            .then()
            .statusCode(204);
        
        // Verify policy is deleted
        given()
            .header("Authorization", "Bearer " + authToken)
            .when()
            .get("/policies/" + policyId)
            .then()
            .statusCode(404);
    }
    
    @Test
    public void testUnauthorizedAccess() {
        given()
            .contentType(ContentType.JSON)
            .body("{\"name\": \"Unauthorized Policy\"}")
            .when()
            .post("/policies")
            .then()
            .statusCode(401);
    }
    
    @Test
    public void testInvalidPolicyData() {
        given()
            .header("Authorization", "Bearer " + authToken)
            .contentType(ContentType.JSON)
            .body("{\n" +
                  "  \"name\": \"\",\n" +
                  "  \"rules\": [\n" +
                  "    {\n" +
                  "      \"condition\": \"invalid syntax here\",\n" +
                  "      \"action\": \"invalid_action\"\n" +
                  "    }\n" +
                  "  ]\n" +
                  "}")
            .when()
            .post("/policies")
            .then()
            .statusCode(400)
            .body("errors", hasSize(greaterThan(0)));
    }
}
```

This comprehensive testing strategy covers all aspects of the PolicyCortex platform, from unit tests to end-to-end scenarios. The strategy emphasizes automation, early detection of issues, and continuous feedback to ensure high-quality deliveries.