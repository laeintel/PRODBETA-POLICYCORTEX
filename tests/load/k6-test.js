import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiResponseTime = new Trend('api_response_time');
const graphqlResponseTime = new Trend('graphql_response_time');

// Test configuration based on test type
const TEST_TYPE = __ENV.TEST_TYPE || 'standard';
const DURATION = __ENV.DURATION || '300';
const API_URL = __ENV.API_URL || 'http://localhost:8080';
const FRONTEND_URL = __ENV.FRONTEND_URL || 'http://localhost:3000';
const GRAPHQL_URL = __ENV.GRAPHQL_URL || 'http://localhost:4000';

// Load test scenarios
const scenarios = {
  standard: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 20 },  // Ramp up to 20 users
      { duration: '3m', target: 20 },  // Stay at 20 users
      { duration: '1m', target: 0 },   // Ramp down to 0 users
    ],
    gracefulRampDown: '30s',
  },
  stress: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '2m', target: 50 },   // Ramp up to 50 users
      { duration: '5m', target: 100 },  // Ramp up to 100 users
      { duration: '2m', target: 200 },  // Ramp up to 200 users
      { duration: '5m', target: 200 },  // Stay at 200 users
      { duration: '2m', target: 0 },    // Ramp down to 0 users
    ],
    gracefulRampDown: '1m',
  },
  spike: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '10s', target: 10 },   // Baseline load
      { duration: '5s', target: 200 },   // Spike to 200 users
      { duration: '3m', target: 200 },   // Stay at 200 users
      { duration: '5s', target: 10 },    // Drop back to baseline
      { duration: '30s', target: 10 },   // Continue at baseline
      { duration: '10s', target: 0 },    // Ramp down
    ],
    gracefulRampDown: '30s',
  },
  soak: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '5m', target: 50 },    // Ramp up to 50 users
      { duration: '24h', target: 50 },   // Stay at 50 users for 24 hours
      { duration: '5m', target: 0 },     // Ramp down to 0 users
    ],
    gracefulRampDown: '1m',
  },
};

// Export test options
export const options = {
  scenarios: {
    main_scenario: scenarios[TEST_TYPE] || scenarios.standard,
  },
  thresholds: {
    // Response time thresholds
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // 95% of requests under 500ms, 99% under 1s
    api_response_time: ['p(95)<300', 'p(99)<500'],
    graphql_response_time: ['p(95)<400', 'p(99)<700'],
    
    // Error rate thresholds
    errors: ['rate<0.01'], // Error rate less than 1%
    http_req_failed: ['rate<0.05'], // HTTP failure rate less than 5%
    
    // Throughput thresholds
    http_reqs: ['rate>100'], // At least 100 requests per second
  },
};

// API endpoints to test
const endpoints = {
  api: {
    health: `${API_URL}/health`,
    metrics: `${API_URL}/api/v1/metrics`,
    correlations: `${API_URL}/api/v1/correlations`,
    predictions: `${API_URL}/api/v1/predictions`,
    recommendations: `${API_URL}/api/v1/recommendations`,
  },
  graphql: {
    health: `${GRAPHQL_URL}/health`,
    query: `${GRAPHQL_URL}/graphql`,
  },
  frontend: {
    home: `${FRONTEND_URL}/`,
    dashboard: `${FRONTEND_URL}/dashboard`,
  },
};

// GraphQL queries
const graphqlQueries = {
  getMetrics: `
    query GetMetrics {
      metrics {
        id
        name
        value
        timestamp
      }
    }
  `,
  getResources: `
    query GetResources {
      resources {
        id
        name
        type
        status
      }
    }
  `,
  getPolicies: `
    query GetPolicies {
      policies {
        id
        name
        status
        complianceScore
      }
    }
  `,
};

// Main test function
export default function () {
  // Test API endpoints
  testAPIEndpoints();
  
  // Test GraphQL endpoints
  testGraphQLEndpoints();
  
  // Test Frontend (if not in soak test)
  if (TEST_TYPE !== 'soak') {
    testFrontendEndpoints();
  }
  
  // Random sleep between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

function testAPIEndpoints() {
  // Health check
  let healthRes = http.get(endpoints.api.health);
  check(healthRes, {
    'API health check status is 200': (r) => r.status === 200,
    'API health check response time < 100ms': (r) => r.timings.duration < 100,
  });
  errorRate.add(healthRes.status !== 200);
  apiResponseTime.add(healthRes.timings.duration);
  
  // Metrics endpoint
  let metricsRes = http.get(endpoints.api.metrics, {
    headers: { 'Content-Type': 'application/json' },
  });
  check(metricsRes, {
    'Metrics endpoint status is 200': (r) => r.status === 200,
    'Metrics response has data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body && body.data;
      } catch {
        return false;
      }
    },
  });
  errorRate.add(metricsRes.status !== 200);
  apiResponseTime.add(metricsRes.timings.duration);
  
  // Correlations endpoint (with simulated data)
  let correlationsRes = http.post(
    endpoints.api.correlations,
    JSON.stringify({
      resourceIds: ['res1', 'res2', 'res3'],
      timeRange: '24h',
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );
  check(correlationsRes, {
    'Correlations endpoint responds': (r) => r.status < 500,
  });
  errorRate.add(correlationsRes.status >= 500);
  apiResponseTime.add(correlationsRes.timings.duration);
  
  // Predictions endpoint
  let predictionsRes = http.get(endpoints.api.predictions);
  check(predictionsRes, {
    'Predictions endpoint responds': (r) => r.status < 500,
  });
  errorRate.add(predictionsRes.status >= 500);
  apiResponseTime.add(predictionsRes.timings.duration);
}

function testGraphQLEndpoints() {
  // GraphQL health check
  let healthRes = http.get(endpoints.graphql.health);
  check(healthRes, {
    'GraphQL health check status is 200': (r) => r.status === 200,
  });
  errorRate.add(healthRes.status !== 200);
  
  // GraphQL queries
  for (const [queryName, query] of Object.entries(graphqlQueries)) {
    let graphqlRes = http.post(
      endpoints.graphql.query,
      JSON.stringify({ query }),
      { headers: { 'Content-Type': 'application/json' } }
    );
    
    check(graphqlRes, {
      [`GraphQL ${queryName} status is 200`]: (r) => r.status === 200,
      [`GraphQL ${queryName} has no errors`]: (r) => {
        try {
          const body = JSON.parse(r.body);
          return !body.errors;
        } catch {
          return false;
        }
      },
    });
    
    errorRate.add(graphqlRes.status !== 200);
    graphqlResponseTime.add(graphqlRes.timings.duration);
  }
}

function testFrontendEndpoints() {
  // Frontend home page
  let homeRes = http.get(endpoints.frontend.home);
  check(homeRes, {
    'Frontend home page loads': (r) => r.status === 200,
    'Frontend home page has content': (r) => r.body && r.body.length > 1000,
  });
  errorRate.add(homeRes.status !== 200);
  
  // Dashboard page (might require auth in real scenario)
  let dashboardRes = http.get(endpoints.frontend.dashboard);
  check(dashboardRes, {
    'Frontend dashboard responds': (r) => r.status < 500,
  });
  errorRate.add(dashboardRes.status >= 500);
}

// Setup function (runs once before the test)
export function setup() {
  console.log(`Starting ${TEST_TYPE} load test`);
  console.log(`API URL: ${API_URL}`);
  console.log(`Frontend URL: ${FRONTEND_URL}`);
  console.log(`GraphQL URL: ${GRAPHQL_URL}`);
  
  // Verify endpoints are accessible
  let healthCheck = http.get(endpoints.api.health);
  if (healthCheck.status !== 200) {
    console.warn('API health check failed during setup');
  }
  
  return { startTime: Date.now() };
}

// Teardown function (runs once after the test)
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration} seconds`);
}