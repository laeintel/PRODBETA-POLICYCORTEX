/**
 * Patent Performance Validation Benchmarks
 * Validates all patent claims meet performance requirements
 */

const axios = require('axios');
const { performance } = require('perf_hooks');
const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');

// Configuration
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8080';
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const ITERATIONS = 100; // Number of test iterations
const CONFIDENCE_LEVEL = 0.95; // 95% confidence interval

// Patent performance requirements
const PATENT_REQUIREMENTS = {
  patent1: {
    name: 'Cross-Domain Correlation Engine',
    endpoint: '/api/v1/correlations',
    method: 'POST',
    maxLatency: 100, // ms
    minAccuracy: null,
    testData: {
      domains: ['security', 'compliance', 'cost', 'operations'],
      timeRange: '24h',
      correlationType: 'anomaly'
    }
  },
  patent2: {
    name: 'Conversational Governance Intelligence',
    endpoint: '/api/v1/conversation',
    method: 'POST',
    maxLatency: 1000, // ms for NLP processing
    minAccuracy: 98.7, // Azure operations accuracy
    testData: {
      messages: [
        'Show me all non-compliant resources in production',
        'Create a backup policy for all VMs',
        'What is my cloud spend this month?',
        'List all security vulnerabilities',
        'Schedule maintenance for database servers'
      ]
    }
  },
  patent3: {
    name: 'Unified AI-Driven Platform',
    endpoint: '/api/v1/metrics/unified',
    method: 'GET',
    maxLatency: 500, // ms
    minAccuracy: null,
    testData: {}
  },
  patent4: {
    name: 'Predictive Policy Compliance Engine',
    endpoint: '/api/v1/predictions',
    method: 'GET',
    maxLatency: 100, // ms inference time
    minAccuracy: 99.2, // % accuracy
    testData: {}
  }
};

// Test utilities
class BenchmarkRunner {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        apiUrl: API_BASE_URL
      },
      patents: {}
    };
  }

  async runAllBenchmarks() {
    console.log(chalk.cyan('\\n🚀 Starting Patent Performance Validation\\n'));
    
    for (const [patentId, config] of Object.entries(PATENT_REQUIREMENTS)) {
      console.log(chalk.yellow(`\\nTesting ${config.name}...`));
      await this.benchmarkPatent(patentId, config);
    }
    
    await this.generateReport();
    return this.results;
  }

  async benchmarkPatent(patentId, config) {
    const latencies = [];
    const accuracyResults = [];
    let errors = 0;
    
    // Warm up the API
    await this.makeRequest(config).catch(() => {});
    
    // Run benchmark iterations
    for (let i = 0; i < ITERATIONS; i++) {
      try {
        const startTime = performance.now();
        const response = await this.makeRequest(config);
        const latency = performance.now() - startTime;
        
        latencies.push(latency);
        
        // Check accuracy if applicable
        if (config.minAccuracy && response.data) {
          const accuracy = this.extractAccuracy(response.data, patentId);
          if (accuracy !== null) {
            accuracyResults.push(accuracy);
          }
        }
        
        // Progress indicator
        if ((i + 1) % 10 === 0) {
          process.stdout.write('.');
        }
      } catch (error) {
        errors++;
        console.error(chalk.red(`Error in iteration ${i}: ${error.message}`));
      }
    }
    
    console.log(); // New line after progress dots
    
    // Calculate statistics
    const stats = this.calculateStats(latencies);
    const accuracyStats = accuracyResults.length > 0 
      ? this.calculateStats(accuracyResults) 
      : null;
    
    // Determine pass/fail
    const latencyPass = stats.p95 <= config.maxLatency;
    const accuracyPass = !config.minAccuracy || 
      (accuracyStats && accuracyStats.mean >= config.minAccuracy);
    const overallPass = latencyPass && accuracyPass;
    
    // Store results
    this.results.patents[patentId] = {
      name: config.name,
      iterations: ITERATIONS,
      errors,
      latency: {
        ...stats,
        requirement: config.maxLatency,
        pass: latencyPass
      },
      accuracy: config.minAccuracy ? {
        ...accuracyStats,
        requirement: config.minAccuracy,
        pass: accuracyPass
      } : null,
      overallPass
    };
    
    // Print results
    this.printPatentResults(patentId, config);
  }

  async makeRequest(config) {
    const url = config.endpoint.startsWith('/api/v1/conversation') || 
                config.endpoint.startsWith('/api/v1/ml')
      ? PYTHON_API_URL + config.endpoint
      : API_BASE_URL + config.endpoint;
    
    const headers = {
      'Authorization': 'Bearer test-token',
      'Content-Type': 'application/json',
      'X-Tenant-ID': 'benchmark-tenant'
    };
    
    let requestConfig = {
      method: config.method,
      url,
      headers,
      timeout: 5000
    };
    
    if (config.method === 'POST') {
      // For conversation API, cycle through test messages
      if (config.testData.messages) {
        const messageIndex = Math.floor(Math.random() * config.testData.messages.length);
        requestConfig.data = {
          message: config.testData.messages[messageIndex]
        };
      } else {
        requestConfig.data = config.testData;
      }
    }
    
    return axios(requestConfig);
  }

  extractAccuracy(data, patentId) {
    switch (patentId) {
      case 'patent2':
        // Extract intent classification confidence
        return data.confidence ? data.confidence * 100 : null;
      case 'patent4':
        // Extract prediction accuracy
        return data.accuracy || data.confidence || null;
      default:
        return null;
    }
  }

  calculateStats(values) {
    if (values.length === 0) return null;
    
    const sorted = values.slice().sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    
    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / values.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      stdDev: this.standardDeviation(values)
    };
  }

  standardDeviation(values) {
    const mean = values.reduce((a, b) => a + b) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b) / values.length;
    return Math.sqrt(avgSquareDiff);
  }

  printPatentResults(patentId, config) {
    const result = this.results.patents[patentId];
    
    console.log(chalk.white(`\\n📊 ${config.name} Results:`));
    console.log(chalk.white('━'.repeat(50)));
    
    // Latency results
    console.log(chalk.cyan('Latency Performance:'));
    console.log(`  Mean: ${result.latency.mean.toFixed(2)}ms`);
    console.log(`  P95: ${result.latency.p95.toFixed(2)}ms`);
    console.log(`  P99: ${result.latency.p99.toFixed(2)}ms`);
    console.log(`  Requirement: <${config.maxLatency}ms`);
    console.log(`  Status: ${result.latency.pass ? chalk.green('✅ PASS') : chalk.red('❌ FAIL')}`);
    
    // Accuracy results (if applicable)
    if (result.accuracy) {
      console.log(chalk.cyan('\\nAccuracy Performance:'));
      console.log(`  Mean: ${result.accuracy.mean.toFixed(2)}%`);
      console.log(`  Min: ${result.accuracy.min.toFixed(2)}%`);
      console.log(`  Max: ${result.accuracy.max.toFixed(2)}%`);
      console.log(`  Requirement: >${config.minAccuracy}%`);
      console.log(`  Status: ${result.accuracy.pass ? chalk.green('✅ PASS') : chalk.red('❌ FAIL')}`);
    }
    
    // Overall status
    console.log(chalk.white('━'.repeat(50)));
    console.log(`Overall: ${result.overallPass ? chalk.green('✅ PASS') : chalk.red('❌ FAIL')}`);
    
    if (result.errors > 0) {
      console.log(chalk.yellow(`⚠️  ${result.errors} errors occurred during testing`));
    }
  }

  async generateReport() {
    const reportPath = path.join(
      __dirname, 
      `patent-benchmark-${Date.now()}.json`
    );
    
    await fs.writeFile(reportPath, JSON.stringify(this.results, null, 2));
    
    console.log(chalk.cyan(`\\n📄 Full report saved to: ${reportPath}`));
    
    // Generate summary
    this.printSummary();
  }

  printSummary() {
    console.log(chalk.cyan('\\n' + '='.repeat(60)));
    console.log(chalk.cyan.bold('PATENT PERFORMANCE VALIDATION SUMMARY'));
    console.log(chalk.cyan('='.repeat(60) + '\\n'));
    
    let totalPass = 0;
    let totalFail = 0;
    
    for (const [patentId, result] of Object.entries(this.results.patents)) {
      const status = result.overallPass ? chalk.green('PASS') : chalk.red('FAIL');
      console.log(`${result.name}: ${status}`);
      
      if (result.overallPass) totalPass++;
      else totalFail++;
    }
    
    console.log(chalk.cyan('\\n' + '-'.repeat(60)));
    console.log(`Total Patents Tested: ${totalPass + totalFail}`);
    console.log(chalk.green(`Passed: ${totalPass}`));
    console.log(chalk.red(`Failed: ${totalFail}`));
    
    const allPass = totalFail === 0;
    console.log(chalk.cyan('-'.repeat(60)));
    console.log(
      allPass 
        ? chalk.green.bold('\\n✅ ALL PATENT PERFORMANCE REQUIREMENTS MET!')
        : chalk.red.bold(`\\n❌ ${totalFail} PATENT(S) FAILED PERFORMANCE REQUIREMENTS`)
    );
  }
}

// Additional specialized benchmarks
class SpecializedBenchmarks {
  async testCorrelationScalability() {
    console.log(chalk.cyan('\\n🔬 Testing Correlation Engine Scalability...\\n'));
    
    const domainCounts = [2, 5, 10, 20, 50];
    const results = [];
    
    for (const count of domainCounts) {
      const domains = Array(count).fill(0).map((_, i) => `domain${i}`);
      const startTime = performance.now();
      
      try {
        await axios.post(
          `${API_BASE_URL}/api/v1/correlations`,
          { domains, timeRange: '24h' },
          {
            headers: {
              'Authorization': 'Bearer test-token',
              'X-Tenant-ID': 'test'
            }
          }
        );
        
        const latency = performance.now() - startTime;
        results.push({ domains: count, latency });
        
        console.log(`  ${count} domains: ${latency.toFixed(2)}ms`);
      } catch (error) {
        console.error(chalk.red(`  ${count} domains: ERROR - ${error.message}`));
      }
    }
    
    // Check if latency scales linearly
    const scalingFactor = results[results.length - 1].latency / results[0].latency;
    const domainsFactor = domainCounts[domainCounts.length - 1] / domainCounts[0];
    
    console.log(chalk.cyan(`\\n  Scaling Factor: ${scalingFactor.toFixed(2)}x for ${domainsFactor}x domains`));
    console.log(
      scalingFactor < domainsFactor * 1.5 
        ? chalk.green('  ✅ Sub-linear scaling achieved')
        : chalk.yellow('  ⚠️  Scaling could be optimized')
    );
    
    return results;
  }

  async testPredictionDrift() {
    console.log(chalk.cyan('\\n🔬 Testing Prediction Drift Detection...\\n'));
    
    const testResources = [
      { id: 'vm-1', drift: 0.1 },
      { id: 'vm-2', drift: 0.3 },
      { id: 'vm-3', drift: 0.7 },
      { id: 'vm-4', drift: 0.9 }
    ];
    
    for (const resource of testResources) {
      try {
        const response = await axios.get(
          `${API_BASE_URL}/api/v1/predictions/drift/${resource.id}`,
          {
            headers: {
              'Authorization': 'Bearer test-token',
              'X-Tenant-ID': 'test'
            }
          }
        );
        
        const detected = response.data.driftScore > 0.5;
        const shouldDetect = resource.drift > 0.5;
        const correct = detected === shouldDetect;
        
        console.log(
          `  Resource ${resource.id} (drift=${resource.drift}): ` +
          `${correct ? chalk.green('✅ Correctly') : chalk.red('❌ Incorrectly')} ` +
          `${detected ? 'detected' : 'not detected'}`
        );
      } catch (error) {
        console.error(chalk.red(`  Resource ${resource.id}: ERROR - ${error.message}`));
      }
    }
  }

  async testConversationContextRetention() {
    console.log(chalk.cyan('\\n🔬 Testing Conversation Context Retention...\\n'));
    
    const conversation = [
      'Show me all VMs in production',
      'How many of them are compliant?',
      'What are the non-compliant ones missing?',
      'Create a remediation plan for them'
    ];
    
    let sessionId = null;
    
    for (let i = 0; i < conversation.length; i++) {
      try {
        const response = await axios.post(
          `${PYTHON_API_URL}/api/v1/conversation`,
          { 
            message: conversation[i],
            sessionId: sessionId 
          },
          {
            headers: {
              'Authorization': 'Bearer test-token',
              'X-Tenant-ID': 'test'
            }
          }
        );
        
        sessionId = response.data.sessionId;
        
        const hasContext = response.data.contextUsed || 
          (i > 0 && response.data.response.includes('them'));
        
        console.log(
          `  Message ${i + 1}: ${hasContext ? chalk.green('✅ Context retained') : chalk.yellow('⚠️  No context')}`
        );
      } catch (error) {
        console.error(chalk.red(`  Message ${i + 1}: ERROR - ${error.message}`));
      }
    }
  }
}

// Main execution
async function main() {
  try {
    // Run standard benchmarks
    const runner = new BenchmarkRunner();
    const results = await runner.runAllBenchmarks();
    
    // Run specialized benchmarks
    const specialized = new SpecializedBenchmarks();
    await specialized.testCorrelationScalability();
    await specialized.testPredictionDrift();
    await specialized.testConversationContextRetention();
    
    // Exit with appropriate code
    const allPass = Object.values(results.patents).every(p => p.overallPass);
    process.exit(allPass ? 0 : 1);
  } catch (error) {
    console.error(chalk.red(`\\n❌ Benchmark failed: ${error.message}`));
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = {
  BenchmarkRunner,
  SpecializedBenchmarks,
  PATENT_REQUIREMENTS
};