const express = require('express');
const cors = require('cors');
const app = express();

// Middleware
app.use(cors({
  origin: true,
  credentials: true
}));
app.use(express.json());

// Log requests for debugging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  if (req.headers.authorization) {
    console.log('  Authorization header present');
  }
  next();
});

// Health endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      core: 'operational',
      database: 'operational',
      cache: 'operational'
    }
  });
});

// API v1 endpoints
app.get('/api/v1/health', (req, res) => {
  res.json({ status: 'healthy', version: '1.0.0' });
});

app.get('/api/v1/compliance', (req, res) => {
  res.json({
    overallScore: 98.7,
    totalResources: 2847,
    compliant: 2810,
    nonCompliant: 37,
    policies: [
      { id: 'p1', name: 'Encryption at Rest', status: 'compliant', resources: 450 },
      { id: 'p2', name: 'Network Security', status: 'compliant', resources: 380 }
    ]
  });
});

app.get('/api/v1/security/threats', (req, res) => {
  res.json({
    threatLevel: 'MEDIUM',
    activeThreats: [
      { id: 't1', type: 'Suspicious Login', severity: 'medium', source: '192.168.1.100', target: 'webapp-01', detected: '2 min ago', status: 'active', description: 'Multiple failed login attempts detected' },
      { id: 't2', type: 'Port Scan', severity: 'low', source: '10.0.0.45', target: 'sql-server-01', detected: '15 min ago', status: 'mitigating', description: 'Automated port scanning activity' },
      { id: 't3', type: 'Brute Force Attempt', severity: 'high', source: '203.0.113.42', target: 'ssh-gateway', detected: '1 hour ago', status: 'active', description: 'Brute force attack on SSH service' }
    ],
    totalScans: 1247,
    blockedAttempts: 89,
    vulnerabilities: {
      critical: 0,
      high: 2,
      medium: 5,
      low: 12
    },
    securityScore: 87,
    lastScan: new Date().toISOString()
  });
});

app.get('/api/v1/resources', (req, res) => {
  res.json({
    resources: [
      { id: 'vm-01', name: 'VM-PROD-WEB-01', type: 'VirtualMachine', status: 'running' },
      { id: 'sql-01', name: 'SQL-PROD-01', type: 'Database', status: 'running' },
      { id: 'storage-01', name: 'STORAGE-PROD-01', type: 'StorageAccount', status: 'running' }
    ],
    total: 2847
  });
});

app.get('/api/v1/cost/analysis', (req, res) => {
  res.json({
    currentMonth: 127439,
    previousMonth: 138472,
    trend: -8,
    forecast: 145000,
    breakdown: {
      compute: 45678,
      storage: 23456,
      network: 12890,
      database: 34567,
      other: 10848
    }
  });
});

app.get('/api/v1/correlations', (req, res) => {
  res.json({
    correlations: [
      { id: 'c1', source: 'Security', target: 'Cost', strength: 87, type: 'positive' },
      { id: 'c2', source: 'Compliance', target: 'Performance', strength: 73, type: 'positive' }
    ],
    patterns: [
      { id: 'p1', name: 'Cost-Security Cascade', occurrences: 47, severity: 'high' }
    ],
    total: 892
  });
});

app.get('/api/v1/predictions', (req, res) => {
  res.json({
    predictions: [
      { id: 'pred1', type: 'cost_overrun', probability: 87, timeframe: '30d' },
      { id: 'pred2', type: 'security_incident', probability: 42, timeframe: '7d' }
    ],
    modelsActive: 14,
    accuracy: 94.7
  });
});

app.get('/api/v1/metrics', (req, res) => {
  res.json({
    resources: 2847,
    compliance: 98.7,
    threats: 3,
    cost: 127439,
    policies: 412,
    alerts: 7,
    correlations: 892,
    predictions: 47
  });
});

app.get('/api/v1/recommendations', (req, res) => {
  res.json({
    recommendations: [
      { id: 'r1', title: 'Enable SQL Auditing', priority: 'high', impact: 'security' },
      { id: 'r2', title: 'Optimize VM Sizes', priority: 'medium', impact: 'cost' }
    ]
  });
});

// Additional endpoints for tactical pages
app.post('/api/v1/compliance/scan', (req, res) => {
  res.json({ status: 'initiated', scanId: `scan-${Date.now()}` });
});

app.post('/api/v1/security/mitigate/:id', (req, res) => {
  res.json({ status: 'mitigating', threatId: req.params.id });
});

app.get('/api/v1/policies', (req, res) => {
  res.json({
    policies: [
      { id: 'pol1', name: 'Require Encryption', status: 'active', resources: 450 },
      { id: 'pol2', name: 'Network Isolation', status: 'active', resources: 380 }
    ]
  });
});

app.post('/api/v1/policies/enforce', (req, res) => {
  res.json({ status: 'enforcing', timestamp: new Date().toISOString() });
});

app.get('/api/v1/rbac/deep', (req, res) => {
  res.json({ roles: [], permissions: [], users: [] });
});

app.get('/api/v1/costs/deep', (req, res) => {
  res.json({ totalCost: 127439, breakdown: {} });
});

app.get('/api/v1/policies/deep', (req, res) => {
  res.json({ policies: [], compliance: 98.7 });
});

app.get('/api/v1/performance', (req, res) => {
  res.json({ score: 94.3, metrics: {} });
});

app.get('/api/v1/monitoring', (req, res) => {
  res.json({ status: 'active', alerts: 7 });
});

app.get('/api/v1/roadmap', (req, res) => {
  res.json({ items: [] });
});

app.get('/api/v1/exceptions', (req, res) => {
  res.json({ exceptions: [] });
});

app.post('/api/v1/conversation/chat', (req, res) => {
  res.json({ 
    response: 'I can help you manage your Azure resources. What would you like to know?',
    suggestions: ['Check compliance', 'View costs', 'Security status']
  });
});

// Action endpoints
app.post('/api/v1/actions', (req, res) => {
  const { action_type, resource_id, params } = req.body;
  res.json({
    action_id: `action-${Date.now()}`,
    action_type,
    resource_id,
    status: 'initiated',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/v1/actions/:id', (req, res) => {
  res.json({
    action_id: req.params.id,
    status: 'completed',
    result: 'success'
  });
});

// Server-sent events for action monitoring
app.get('/api/v1/actions/:id/events', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });
  
  let counter = 0;
  const interval = setInterval(() => {
    res.write(`data: Progress: ${counter * 20}%\n\n`);
    counter++;
    if (counter > 5) {
      res.write('data: Action completed successfully\n\n');
      clearInterval(interval);
      res.end();
    }
  }, 1000);
  
  req.on('close', () => {
    clearInterval(interval);
  });
});

// Start server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Mock backend server running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});