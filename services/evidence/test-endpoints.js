/**
 * Test script for Evidence Server endpoints
 */

const http = require('http');

const BASE_URL = 'http://localhost:8081';

// Helper function to make HTTP requests
function makeRequest(method, path, data) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          resolve({
            status: res.statusCode,
            data: JSON.parse(body)
          });
        } catch {
          resolve({
            status: res.statusCode,
            data: body
          });
        }
      });
    });

    req.on('error', reject);

    if (data) {
      req.write(JSON.stringify(data));
    }
    req.end();
  });
}

async function runTests() {
  console.log('Testing Evidence Server Endpoints\n');
  console.log('==================================\n');

  try {
    // Test 1: Health check
    console.log('1. Testing /health endpoint...');
    const health = await makeRequest('GET', '/health');
    console.log('   Status:', health.status);
    console.log('   Response:', JSON.stringify(health.data, null, 2));
    console.log();

    // Test 2: Add event (T04)
    console.log('2. Testing POST /api/v1/events...');
    const eventData = {
      payload: {
        type: 'PredictionIssued',
        ruleId: 'AZ-NSG-OPEN-443',
        etaDays: 5,
        confidence: 0.91,
        repo: 'org/infra',
        fixBranch: 'pcx/autofix/AZ-NSG-OPEN-443',
        timestamp: new Date().toISOString()
      }
    };
    const newEvent = await makeRequest('POST', '/api/v1/events', eventData);
    console.log('   Status:', newEvent.status);
    console.log('   Response:', JSON.stringify(newEvent.data, null, 2));
    console.log();

    // Test 3: List events
    console.log('3. Testing GET /api/v1/events...');
    const events = await makeRequest('GET', '/api/v1/events');
    console.log('   Status:', events.status);
    console.log('   Event count:', events.data.length);
    console.log('   First event:', JSON.stringify(events.data[0], null, 2));
    console.log();

    // Test 4: Replay events
    console.log('4. Testing GET /api/v1/events/replay...');
    const replay = await makeRequest('GET', '/api/v1/events/replay');
    console.log('   Status:', replay.status);
    console.log('   Replay count:', replay.data.length);
    console.log();

    // Test 5: Verify hash (T05)
    console.log('5. Testing GET /api/v1/verify/:hash...');
    const testHash = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const verify = await makeRequest('GET', `/api/v1/verify/${testHash}`);
    console.log('   Status:', verify.status);
    console.log('   Response:', JSON.stringify(verify.data, null, 2));
    console.log();

    // Test 6: Export evidence with Merkle proofs
    console.log('6. Testing POST /api/v1/evidence/export...');
    const exportReq = {
      contentHashes: [
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
      ]
    };
    const exported = await makeRequest('POST', '/api/v1/evidence/export', exportReq);
    console.log('   Status:', exported.status);
    console.log('   Merkle Root:', exported.data.merkleRoot);
    console.log('   Export count:', exported.data.totalCount);
    
    if (exported.data.exports && exported.data.exports.length > 0) {
      console.log('   First export:', JSON.stringify(exported.data.exports[0], null, 2));
    }
    console.log();

    console.log('✅ All tests completed successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

// Run tests
runTests();