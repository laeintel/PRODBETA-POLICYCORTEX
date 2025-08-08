import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up to 100 users over 2 minutes
    { duration: '5m', target: 100 }, // Maintain 100 users for 5 minutes
    { duration: '2m', target: 200 }, // Ramp up to 200 users over 2 minutes
    { duration: '5m', target: 200 }, // Maintain 200 users for 5 minutes
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(99)<1500'], // 99% of requests must complete below 1.5s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate must be below 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  // Test 1: Homepage load
  let response = http.get(`${BASE_URL}/`);
  check(response, {
    'homepage status is 200': (r) => r.status === 200,
    'homepage loads in <2s': (r) => r.timings.duration < 2000,
  }) || errorRate.add(1);

  sleep(1);

  // Test 2: Dashboard access
  response = http.get(`${BASE_URL}/dashboard`);
  check(response, {
    'dashboard status is 200': (r) => r.status === 200,
    'dashboard loads in <3s': (r) => r.timings.duration < 3000,
  }) || errorRate.add(1);

  sleep(1);

  // Test 3: API health check
  response = http.get(`${BASE_URL}/api/v1/metrics`);
  check(response, {
    'metrics API status is 200': (r) => r.status === 200,
    'metrics API responds in <1s': (r) => r.timings.duration < 1000,
    'metrics API returns valid JSON': (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch {
        return false;
      }
    },
  }) || errorRate.add(1);

  sleep(1);

  // Test 4: Chat interface
  response = http.get(`${BASE_URL}/chat`);
  check(response, {
    'chat status is 200': (r) => r.status === 200,
    'chat loads in <2s': (r) => r.timings.duration < 2000,
  }) || errorRate.add(1);

  sleep(1);

  // Test 5: GraphQL endpoint
  const graphqlPayload = JSON.stringify({
    query: '{ __schema { types { name } } }'
  });

  response = http.post(`${BASE_URL}/graphql`, graphqlPayload, {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  check(response, {
    'GraphQL status is 200': (r) => r.status === 200,
    'GraphQL responds in <1s': (r) => r.timings.duration < 1000,
    'GraphQL returns valid response': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.data && data.data.__schema;
      } catch {
        return false;
      }
    },
  }) || errorRate.add(1);

  sleep(2);
}