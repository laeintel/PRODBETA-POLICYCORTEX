// Mock API Server for PolicyCortex
// Provides fallback endpoints when backend services are not running

const express = require('express');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 8080;

// Middleware
app.use(cors());
app.use(express.json());

// Mock data
const mockResources = [
  {
    id: "res-001",
    name: "PolicyCortex-VM-01",
    type: "Virtual Machine",
    location: "East US",
    compliance_status: "compliant",
    risk_score: 15,
    tags: { environment: "production", owner: "devops" },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: "res-002",
    name: "PolicyCortex-Storage",
    type: "Storage Account",
    location: "West US",
    compliance_status: "non-compliant",
    risk_score: 75,
    tags: { environment: "production", criticality: "high" },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
];

const mockPolicies = [
  {
    id: "pol-001",
    name: "Encryption at Rest",
    description: "Ensures all storage accounts have encryption enabled",
    enforcement_mode: "enforced",
    compliance_rate: 92.5,
    resources_affected: 15,
    created_at: new Date().toISOString()
  },
  {
    id: "pol-002",
    name: "Network Security",
    description: "Enforces network security group rules",
    enforcement_mode: "audit",
    compliance_rate: 87.3,
    resources_affected: 23,
    created_at: new Date().toISOString()
  }
];

// Health endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'mock-server',
    version: '2.26.0'
  });
});

// API v1 endpoints
app.get('/api/v1/health', (req, res) => {
  res.json({
    status: 'healthy',
    azure: { connected: false, reason: 'Using mock data' },
    database: { connected: true },
    cache: { connected: true },
    timestamp: new Date().toISOString()
  });
});

app.get('/api/v1/resources', (req, res) => {
  res.json({
    resources: mockResources,
    total: mockResources.length,
    summary: {
      total_resources: mockResources.length,
      compliant: 1,
      non_compliant: 1,
      at_risk: 0
    }
  });
});

app.get('/api/v1/policies', (req, res) => {
  res.json({
    policies: mockPolicies,
    total: mockPolicies.length,
    summary: {
      total_policies: mockPolicies.length,
      enforced: 1,
      audit: 1,
      disabled: 0
    }
  });
});

app.get('/api/v1/metrics', (req, res) => {
  res.json({
    metrics: {
      total_resources: 125,
      compliance_score: 89.5,
      policies_active: 42,
      cost_savings: 12500,
      risk_score: 28,
      incidents_resolved: 7
    },
    timestamp: new Date().toISOString()
  });
});

app.get('/api/v1/compliance', (req, res) => {
  res.json({
    overall_compliance: 89.5,
    by_category: {
      security: 92.0,
      cost: 87.5,
      performance: 88.3,
      availability: 90.1
    },
    recent_checks: [
      {
        resource_id: "res-001",
        policy_id: "pol-001",
        status: "passed",
        checked_at: new Date().toISOString()
      }
    ]
  });
});

app.get('/api/v1/correlations', (req, res) => {
  res.json([
    {
      id: "corr-001",
      source: "Resource Configuration Change",
      target: "Compliance Drift",
      strength: 0.85,
      confidence: 0.92
    },
    {
      id: "corr-002",
      source: "Cost Spike",
      target: "Untagged Resources",
      strength: 0.78,
      confidence: 0.88
    }
  ]);
});

app.get('/api/v1/predictions', (req, res) => {
  res.json({
    predictions: [
      {
        id: "pred-001",
        type: "compliance_drift",
        resource_id: "res-002",
        prediction: "Non-compliance likely in 7 days",
        confidence: 0.89,
        recommended_action: "Update security configuration"
      }
    ]
  });
});

app.post('/api/v1/conversation', (req, res) => {
  const { message } = req.body;
  res.json({
    response: `I understand you said: "${message}". This is a mock response from the development server.`,
    intent: "query",
    entities: [],
    confidence: 0.95
  });
});

// API v2 endpoints
app.get('/api/v2/resources', (req, res) => {
  res.json({
    data: mockResources,
    metadata: {
      total: mockResources.length,
      page: 1,
      per_page: 10
    }
  });
});

// Governance endpoints
app.get('/api/v1/governance/metrics', (req, res) => {
  res.json({
    governance_score: 87.5,
    policy_compliance: 89.2,
    resource_optimization: 85.3,
    security_posture: 90.1,
    cost_efficiency: 84.7
  });
});

// AI endpoints
app.get('/api/v1/ai/unified', (req, res) => {
  res.json({
    metrics: {
      total: 250,
      active: 187,
      alerts: 12,
      savings: 45000
    },
    insights: [
      "15 resources need attention",
      "Cost optimization opportunity: $5,000/month",
      "3 security recommendations pending"
    ]
  });
});

// ITSM endpoints
app.get('/api/v1/itsm/summary', (req, res) => {
  res.json({
    incidents: { open: 5, resolved: 42, in_progress: 3 },
    changes: { pending: 8, approved: 15, completed: 127 },
    problems: { identified: 3, investigating: 2, resolved: 18 },
    requests: { new: 12, assigned: 8, completed: 95 }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    path: req.path,
    timestamp: new Date().toISOString()
  });
});

// === PAYBACK PILLAR: ROI Metrics ===
app.get('/api/v1/roi/metrics', (req, res) => {
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
        verified: true
      },
      {
        id: 'ev-002',
        timestamp: new Date(Date.now() - 60000).toISOString(),
        control: 'CIS-1.2.3',
        status: 'compliant',
        hash: '0x8f7e6d5c4b3a2918f6e5d4c3b2a1908f7e6d5c4b',
        verified: true
      }
    ],
    chain_status: {
      total_blocks: 1247,
      latest_block_hash: '0x9f8e7d6c5b4a3928271e6d5c4b3a2918f6e5d4c3',
      chain_verified: true,
      integrity_score: 100
    }
  });
});

app.get('/api/v1/evidence/chain', (req, res) => {
  res.json({
    status: 'active',
    total_blocks: 1247,
    total_evidence: 3891,
    integrity: {
      verified: true,
      last_verification: new Date().toISOString()
    }
  });
});

app.get('/api/v1/predict/mttp', (req, res) => {
  res.json({
    current_mttp: 18,
    target_mttp: 24,
    trend: 'improving',
    prevention_rate: 0.35
  });
});

app.post('/api/v1/roi/simulate', (req, res) => {
  res.json({
    simulation_id: 'sim-' + Date.now(),
    results: {
      costs: {
        baseline: 500000,
        projected: 325000,
        savings: 175000
      },
      roi: {
        percentage: 350,
        payback_months: 3.5
      }
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
========================================
PolicyCortex Mock Server Started
========================================
Port: ${PORT}
Mode: Development
URL: http://localhost:${PORT}
Health: http://localhost:${PORT}/health
========================================
  `);
});