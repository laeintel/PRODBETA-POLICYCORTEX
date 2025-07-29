// K6 Load Test Script for PolicyCortex
// Comprehensive performance testing for all services

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';
import { randomString, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const loginSuccess = new Rate('login_success');
const apiGatewayErrors = new Rate('api_gateway_errors');
const azureIntegrationErrors = new Rate('azure_integration_errors');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '3m', target: 200 },  // Ramp up to 200 users
    { duration: '5m', target: 200 },  // Stay at 200 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.05'],                  // Error rate under 5%
    errors: ['rate<0.05'],                           // Custom error rate under 5%
    login_success: ['rate>0.95'],                    // 95% login success rate
  },
};

// Base URLs
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const SERVICES = {
  apiGateway: BASE_URL,
  azureIntegration: 'http://localhost:8001',
  aiEngine: 'http://localhost:8002',
  dataProcessing: 'http://localhost:8003',
  conversation: 'http://localhost:8004',
  notification: 'http://localhost:8005',
};

// Test data
const testUsers = [
  { email: 'test1@example.com', password: 'TestPass123!' },
  { email: 'test2@example.com', password: 'TestPass123!' },
  { email: 'test3@example.com', password: 'TestPass123!' },
];

const testPolicies = [
  'All VMs must have backup enabled',
  'Storage accounts must use encryption',
  'Network security groups must restrict SSH',
];

// Helper functions
function authenticateUser() {
  const user = randomItem(testUsers);
  const loginRes = http.post(`${BASE_URL}/api/v1/auth/login`, JSON.stringify(user), {
    headers: { 'Content-Type': 'application/json' },
  });

  const loginSuccessful = check(loginRes, {
    'login successful': (r) => r.status === 200,
    'token received': (r) => r.json('access_token') !== undefined,
  });

  loginSuccess.add(loginSuccessful);

  if (loginSuccessful) {
    return loginRes.json('access_token');
  }
  return null;
}

// Test scenarios
export function setup() {
  // Setup code - create test data if needed
  console.log('Setting up load test...');
  
  // Check all services are healthy
  const services = Object.entries(SERVICES);
  for (const [name, url] of services) {
    const res = http.get(`${url}/health`);
    if (res.status !== 200) {
      throw new Error(`${name} service is not healthy`);
    }
  }
  
  return { startTime: new Date() };
}

export default function () {
  // Scenario 1: User Authentication Flow
  const token = authenticateUser();
  if (!token) {
    errorRate.add(1);
    return;
  }

  const authHeaders = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };

  // Scenario 2: Dashboard Data Loading
  const dashboardRes = http.get(`${BASE_URL}/api/v1/dashboard`, { headers: authHeaders });
  check(dashboardRes, {
    'dashboard loaded': (r) => r.status === 200,
    'dashboard data valid': (r) => r.json('data') !== undefined,
  });

  sleep(1);

  // Scenario 3: Policy Management
  const policiesRes = http.get(`${BASE_URL}/api/v1/policies`, { headers: authHeaders });
  const policiesSuccess = check(policiesRes, {
    'policies retrieved': (r) => r.status === 200,
    'policies array': (r) => Array.isArray(r.json('data')),
  });

  if (policiesSuccess) {
    // Create a new policy
    const newPolicy = {
      name: `Test Policy ${randomString(8)}`,
      description: randomItem(testPolicies),
      type: 'compliance',
      category: 'security',
    };

    const createPolicyRes = http.post(
      `${BASE_URL}/api/v1/policies`,
      JSON.stringify(newPolicy),
      { headers: authHeaders }
    );

    check(createPolicyRes, {
      'policy created': (r) => r.status === 201,
      'policy id returned': (r) => r.json('data.id') !== undefined,
    });
  }

  sleep(2);

  // Scenario 4: Resource Discovery
  const resourcesRes = http.get(`${BASE_URL}/api/v1/resources`, { headers: authHeaders });
  const resourcesSuccess = check(resourcesRes, {
    'resources retrieved': (r) => r.status === 200,
    'resources data valid': (r) => r.json('data') !== undefined,
  });

  if (!resourcesSuccess) {
    apiGatewayErrors.add(1);
  }

  sleep(1);

  // Scenario 5: AI Analysis
  const analysisPayload = {
    policy_text: randomItem(testPolicies),
    policy_type: 'compliance',
  };

  const analysisRes = http.post(
    `${BASE_URL}/api/v1/ai/analyze`,
    JSON.stringify(analysisPayload),
    { headers: authHeaders }
  );

  check(analysisRes, {
    'AI analysis completed': (r) => r.status === 200,
    'analysis results returned': (r) => r.json('data.analysis') !== undefined,
  });

  sleep(2);

  // Scenario 6: Conversation Flow
  const conversationPayload = {
    message: 'Show me all non-compliant resources',
    context: {},
  };

  const conversationRes = http.post(
    `${BASE_URL}/api/v1/conversation/message`,
    JSON.stringify(conversationPayload),
    { headers: authHeaders }
  );

  check(conversationRes, {
    'conversation response': (r) => r.status === 200,
    'response message': (r) => r.json('data.response') !== undefined,
  });

  sleep(1);

  // Scenario 7: Notification Check
  const notificationsRes = http.get(`${BASE_URL}/api/v1/notifications`, { headers: authHeaders });
  check(notificationsRes, {
    'notifications retrieved': (r) => r.status === 200,
    'notifications array': (r) => Array.isArray(r.json('data')),
  });

  // Random wait between iterations
  sleep(Math.random() * 3 + 1);
}

export function teardown(data) {
  // Cleanup code
  console.log(`Load test completed. Duration: ${new Date() - data.startTime}ms`);
}

// Custom summary
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'summary.json': JSON.stringify(data),
    'summary.html': htmlReport(data),
  };
}

function textSummary(data, options) {
  // Custom text summary
  let summary = '\n=== PolicyCortex Load Test Results ===\n\n';
  
  // Add key metrics
  summary += 'Key Metrics:\n';
  summary += `  Total Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += `  Failed Requests: ${data.metrics.http_req_failed.values.passes}\n`;
  summary += `  Avg Response Time: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms\n`;
  summary += `  P95 Response Time: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += `  P99 Response Time: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n`;
  
  return summary;
}

function htmlReport(data) {
  // Generate HTML report
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <title>PolicyCortex Load Test Report</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; padding: 10px; background: #f0f0f0; }
        .passed { color: green; }
        .failed { color: red; }
      </style>
    </head>
    <body>
      <h1>PolicyCortex Load Test Report</h1>
      <div class="metric">
        <h3>Test Summary</h3>
        <p>Total Requests: ${data.metrics.http_reqs.values.count}</p>
        <p>Duration: ${data.state.testRunDurationMs}ms</p>
      </div>
      <div class="metric">
        <h3>Response Times</h3>
        <p>Average: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms</p>
        <p>P95: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms</p>
        <p>P99: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms</p>
      </div>
    </body>
    </html>
  `;
}