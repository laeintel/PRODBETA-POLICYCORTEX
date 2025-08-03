/**
 * PolicyCortex API Test Runner
 * 
 * This script runs the Postman collection tests using Newman
 * Install Newman: npm install -g newman
 * Run tests: node run-api-tests.js
 */

const newman = require('newman');
const path = require('path');

// Test configuration
const config = {
    collection: path.join(__dirname, 'PolicyCortex-API-Tests.postman_collection.json'),
    environment: path.join(__dirname, 'PolicyCortex-Local-Environment.postman_environment.json'),
    reporters: ['cli', 'html', 'json'],
    reporter: {
        html: {
            export: './test-results/api-test-report.html'
        },
        json: {
            export: './test-results/api-test-results.json'
        }
    },
    insecure: true, // For local development with self-signed certs
    timeout: 30000, // 30 second timeout
    timeoutRequest: 10000, // 10 second request timeout
    color: true,
    suppressExitCode: false
};

// Run the collection
console.log('🚀 Starting PolicyCortex API Integration Tests...\n');

newman.run(config, function (err, summary) {
    if (err) {
        console.error('❌ Collection run failed:', err);
        process.exit(1);
    }
    
    console.log('\n📊 Test Results Summary:');
    console.log(`Total Requests: ${summary.run.stats.tests.total}`);
    console.log(`✅ Passed: ${summary.run.stats.tests.pending}`);
    console.log(`❌ Failed: ${summary.run.stats.tests.failed}`);
    console.log(`⏱️  Duration: ${summary.run.timings.completed - summary.run.timings.started}ms`);
    
    if (summary.run.failures.length > 0) {
        console.log('\n❌ Failed Tests:');
        summary.run.failures.forEach((failure, index) => {
            console.log(`${index + 1}. ${failure.error.test}: ${failure.error.message}`);
        });
    }
    
    console.log('\n✨ Test run complete! Check test-results folder for detailed reports.');
});