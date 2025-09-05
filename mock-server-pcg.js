// PCG-focused Mock Server - Provides API endpoints for PREVENT, PROVE, PAYBACK
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// Check if we should use fail-fast mode for real data
const USE_REAL_DATA = process.env.USE_REAL_DATA === 'true';
const FAIL_FAST = process.env.FAIL_FAST_MODE === 'true' || USE_REAL_DATA;

// Middleware to enforce fail-fast in real mode
const failFastMiddleware = (serviceName) => (req, res, next) => {
  if (FAIL_FAST && USE_REAL_DATA) {
    return res.status(503).json({
      error: 'service_unavailable',
      message: `Service '${serviceName}' unavailable in real mode`,
      hint: `Real mode requires actual Azure connections. Configure:
- AZURE_SUBSCRIPTION_ID
- AZURE_TENANT_ID  
- AZURE_CLIENT_ID
- AZURE_CLIENT_SECRET
- PREDICTIONS_URL (for ML service)
- DATABASE_URL (for persistence)

See docs/REVAMP/REAL_MODE_SETUP.md for details.`,
      timestamp: new Date().toISOString()
    });
  }
  next();
};

// === PREVENT PILLAR: Predictive Compliance ===
app.get('/api/v1/predictions', (req, res) => {
  res.json({
    predictions: [
      {
        id: 'pred-001',
        resource_id: '/subscriptions/demo/resourceGroups/prod-rg/providers/Microsoft.Storage/storageAccounts/prodstorage',
        resource_name: 'prodstorage',
        violation_type: 'data_encryption',
        prediction_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
        probability: 0.89,
        severity: 'HIGH',
        estimated_impact: '$45,000',
        causal_factors: ['No encryption at rest', 'Public access enabled', 'No access logs'],
        recommended_action: 'Enable encryption and restrict public access'
      },
      {
        id: 'pred-002',
        resource_id: '/subscriptions/demo/resourceGroups/dev-rg/providers/Microsoft.Compute/virtualMachines/devvm01',
        resource_name: 'devvm01',
        violation_type: 'unpatched_system',
        prediction_date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
        probability: 0.76,
        severity: 'MEDIUM',
        estimated_impact: '$12,000',
        causal_factors: ['Missing critical patches', 'Auto-update disabled'],
        recommended_action: 'Apply security patches immediately'
      },
      {
        id: 'pred-003',
        resource_id: '/subscriptions/demo/resourceGroups/prod-rg/providers/Microsoft.Network/networkSecurityGroups/prod-nsg',
        resource_name: 'prod-nsg',
        violation_type: 'excessive_permissions',
        prediction_date: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
        probability: 0.92,
        severity: 'CRITICAL',
        estimated_impact: '$125,000',
        causal_factors: ['Port 3389 open to internet', 'No IP restrictions'],
        recommended_action: 'Restrict RDP access to specific IPs'
      }
    ],
    summary: {
      total: 3,
      critical: 1,
      high: 1,
      medium: 1,
      low: 0,
      prevented_last_30_days: 12,
      mttp_hours: 18
    }
  });
});

app.get('/api/v1/predict/mttp', (req, res) => {
  res.json({
    current_mttp: 18,
    target_mttp: 24,
    trend: 'improving',
    last_7_days: [22, 20, 19, 18, 17, 18, 18],
    prevention_rate: 0.35
  });
});

// === PROVE PILLAR: Evidence Chain ===
app.get('/api/v1/evidence', (req, res) => {
  res.json({
    evidence_items: [
      {
        id: 'ev-001',
        timestamp: new Date().toISOString(),
        control: 'NIST-800-53-AC-2',
        status: 'compliant',
        hash: '0x3f4a8b9c2d1e5f6a7b8c9d0e1f2a3b4c5d6e7f8a',
        verified: true,
        details: 'User access controls verified and compliant'
      },
      {
        id: 'ev-002',
        timestamp: new Date(Date.now() - 60000).toISOString(),
        control: 'CIS-1.2.3',
        status: 'compliant',
        hash: '0x8f7e6d5c4b3a2918f6e5d4c3b2a1908f7e6d5c4b',
        verified: true,
        details: 'Encryption at rest enabled for all databases'
      },
      {
        id: 'ev-003',
        timestamp: new Date(Date.now() - 120000).toISOString(),
        control: 'SOC2-CC6.1',
        status: 'non_compliant',
        hash: '0x2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b',
        verified: true,
        details: 'Logical access controls need strengthening'
      }
    ],
    chain_status: {
      total_blocks: 1247,
      latest_block_hash: '0x9f8e7d6c5b4a3928271e6d5c4b3a2918f6e5d4c3',
      chain_verified: true,
      last_anchor: new Date(Date.now() - 3600000).toISOString(),
      integrity_score: 100
    }
  });
});

app.get('/api/v1/evidence/chain', (req, res) => {
  res.json({
    chain_id: 'pcg-chain-001',
    status: 'active',
    total_blocks: 1247,
    total_evidence: 3891,
    latest_block: {
      index: 1247,
      hash: '0x9f8e7d6c5b4a3928271e6d5c4b3a2918f6e5d4c3',
      previous_hash: '0x8e7d6c5b4a3927160d5c4b3a2917e5d4c3b2a190',
      timestamp: new Date().toISOString(),
      evidence_count: 3,
      merkle_root: '0x5d4c3b2a1907e6d5c4b3a2816f5e4d3c2b1a0908'
    },
    integrity: {
      verified: true,
      last_verification: new Date().toISOString(),
      verification_method: 'SHA3-256',
      signature_valid: true
    }
  });
});

// === PAYBACK PILLAR: ROI Metrics ===
app.get('/api/v1/roi/metrics', failFastMiddleware('ROI Calculator'), (req, res) => {
  res.json({
    summary: {
      total_savings: 485000,
      monthly_savings: 45000,
      roi_percentage: 350,
      payback_period_months: 3.5,
      cloud_cost_reduction: 0.12
    },
    breakdown: {
      prevented_incidents: {
        count: 12,
        value: 285000,
        details: 'Prevented 12 incidents that would have cost $285,000'
      },
      resource_optimization: {
        count: 45,
        value: 125000,
        details: 'Optimized 45 resources saving $125,000'
      },
      automated_remediation: {
        hours_saved: 280,
        value: 42000,
        details: 'Saved 280 hours of manual work worth $42,000'
      },
      compliance_penalties_avoided: {
        count: 2,
        value: 33000,
        details: 'Avoided 2 compliance penalties worth $33,000'
      }
    },
    projections: {
      next_30_days: 52000,
      next_60_days: 108000,
      next_90_days: 168000,
      annual_projection: 640000
    }
  });
});

app.post('/api/v1/roi/simulate', (req, res) => {
  const { prevention_rate = 0.35, automation_level = 0.6, incident_reduction = 0.5 } = req.body;
  
  const baselineIncidents = 20;
  const baselineCost = 500000;
  const baselineHours = 500;
  
  const projectedIncidents = Math.round(baselineIncidents * (1 - incident_reduction));
  const projectedCost = Math.round(baselineCost * (1 - prevention_rate));
  const projectedHours = Math.round(baselineHours * (1 - automation_level));
  
  res.json({
    simulation_id: 'sim-' + Date.now(),
    parameters: { prevention_rate, automation_level, incident_reduction },
    results: {
      incidents: {
        baseline: baselineIncidents,
        projected: projectedIncidents,
        reduction: baselineIncidents - projectedIncidents
      },
      costs: {
        baseline: baselineCost,
        projected: projectedCost,
        savings: baselineCost - projectedCost
      },
      hours: {
        baseline: baselineHours,
        projected: projectedHours,
        saved: baselineHours - projectedHours
      },
      roi: {
        percentage: Math.round(((baselineCost - projectedCost) / 150000) * 100),
        payback_months: (150000 / ((baselineCost - projectedCost) / 12)).toFixed(1)
      }
    },
    confidence: {
      level: 'HIGH',
      accuracy: 0.85,
      factors: ['Historical data available', 'Similar environment patterns', 'Validated models']
    }
  });
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'pcg-mock-server',
    pillars: ['PREVENT', 'PROVE', 'PAYBACK'],
    timestamp: new Date().toISOString()
  });
});

app.get('/api/v1/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'pcg-mock-server',
    version: '1.0.0',
    platform: 'Predictive Cloud Governance'
  });
});

// Default config
app.get('/api/v1/config', (req, res) => {
  res.json({
    pillars: {
      prevent: { enabled: true, features: ['7-day-prediction', 'auto-fix-pr', 'drift-detection'] },
      prove: { enabled: true, features: ['hash-chain', 'audit-reports', 'compliance-mapping'] },
      payback: { enabled: true, features: ['roi-tracking', 'what-if-simulator', 'cost-optimization'] }
    },
    demo_mode: true
  });
});

// Compliance endpoint (needed by some components)
app.get('/api/v1/compliance', (req, res) => {
  res.json({
    compliance_status: {
      overall: 0.92,
      frameworks: {
        'NIST-800-53': 0.94,
        'CIS': 0.91,
        'SOC2': 0.89,
        'ISO-27001': 0.93
      }
    },
    violations: [],
    last_scan: new Date().toISOString()
  });
});

// Resources endpoint (needed by some components)
app.get('/api/v1/resources', (req, res) => {
  res.json({
    resources: [
      {
        id: '/subscriptions/demo/resourceGroups/prod-rg/providers/Microsoft.Storage/storageAccounts/prodstorage',
        name: 'prodstorage',
        type: 'Microsoft.Storage/storageAccounts',
        status: 'at-risk',
        compliance_score: 0.65,
        cost: 1250,
        risk_level: 'HIGH'
      },
      {
        id: '/subscriptions/demo/resourceGroups/dev-rg/providers/Microsoft.Compute/virtualMachines/devvm01',
        name: 'devvm01',
        type: 'Microsoft.Compute/virtualMachines',
        status: 'compliant',
        compliance_score: 0.88,
        cost: 450,
        risk_level: 'LOW'
      }
    ],
    total: 2,
    compliant: 1,
    at_risk: 1
  });
});

// Policies endpoint
app.get('/api/v1/policies', (req, res) => {
  res.json({
    policies: [
      {
        id: 'pol-001',
        name: 'Require Encryption at Rest',
        category: 'Security',
        enforcement: 'deny',
        compliance_impact: 'HIGH',
        resources_affected: 45
      },
      {
        id: 'pol-002',
        name: 'Require Tags for Cost Tracking',
        category: 'FinOps',
        enforcement: 'audit',
        compliance_impact: 'MEDIUM',
        resources_affected: 123
      }
    ],
    total: 2
  });
});

const PORT = process.env.PORT || 8081;
app.listen(PORT, () => {
  console.log(`PCG Mock Server running on port ${PORT}`);
  console.log('Endpoints available:');
  console.log('  PREVENT: /api/v1/predictions, /api/v1/predict/mttp');
  console.log('  PROVE: /api/v1/evidence, /api/v1/evidence/chain');
  console.log('  PAYBACK: /api/v1/roi/metrics, /api/v1/roi/simulate');
});