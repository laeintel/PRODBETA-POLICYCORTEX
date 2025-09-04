/**
 * Enhanced GraphQL Gateway with WebSocket Subscription Support
 * PATENT NOTICE: This code implements methods covered by PolicyCortex patents.
 * © 2024 PolicyCortex. All rights reserved.
 */

const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { ApolloServerPluginDrainHttpServer } = require('@apollo/server/plugin/drainHttpServer');
const { makeExecutableSchema } = require('@graphql-tools/schema');
const { WebSocketServer } = require('ws');
const { useServer } = require('graphql-ws/lib/use/ws');
const { PubSub } = require('graphql-subscriptions');
const express = require('express');
const http = require('http');
const cors = require('cors');
const bodyParser = require('body-parser');
const { gql } = require('graphql-tag');

// Create PubSub instance for subscriptions
const pubsub = new PubSub();

// Subscription event types
const EVENTS = {
  RESOURCE_CREATED: 'RESOURCE_CREATED',
  RESOURCE_UPDATED: 'RESOURCE_UPDATED',
  RESOURCE_DELETED: 'RESOURCE_DELETED',
  POLICY_CHANGED: 'POLICY_CHANGED',
  COMPLIANCE_ALERT: 'COMPLIANCE_ALERT',
  COST_ALERT: 'COST_ALERT',
  SECURITY_ALERT: 'SECURITY_ALERT',
  GOVERNANCE_UPDATE: 'GOVERNANCE_UPDATE',
};

// Enhanced GraphQL schema with subscriptions
const typeDefs = gql`
  type Query {
    policies(filter: PolicyFilter): [Policy!]!
    policy(id: ID!): Policy
    resources(filter: ResourceFilter): [Resource!]!
    resource(id: ID!): Resource
    compliance(filter: ComplianceFilter): [ComplianceResult!]!
    metrics: Metrics!
    alerts(severity: String): [Alert!]!
    costs(timeRange: TimeRange!): CostAnalysis!
  }

  type Mutation {
    createPolicy(input: CreatePolicyInput!): Policy!
    updatePolicy(id: ID!, input: UpdatePolicyInput!): Policy!
    deletePolicy(id: ID!): Boolean!
    createResource(input: CreateResourceInput!): Resource!
    updateResource(id: ID!, input: UpdateResourceInput!): Resource!
    deleteResource(id: ID!): Boolean!
    triggerCompliance(policyId: ID!, resourceIds: [ID!]!): ComplianceResult!
    acknowledgeAlert(alertId: ID!): Alert!
    updateCostBudget(budget: CostBudgetInput!): CostBudget!
  }

  type Subscription {
    resourceChanged(resourceType: String): ResourceEvent!
    policyChanged(category: String): PolicyEvent!
    complianceAlert(severity: String): ComplianceAlert!
    costAlert(threshold: Float): CostAlert!
    securityAlert(severity: String): SecurityAlert!
    governanceUpdate(domain: String): GovernanceUpdate!
    metricsUpdate(interval: Int): Metrics!
    systemStatus: SystemStatus!
  }

  type Policy {
    id: ID!
    name: String!
    description: String
    category: String!
    severity: String!
    status: String!
    rules: [PolicyRule!]!
    affectedResources: Int!
    complianceRate: Float!
    createdAt: String!
    updatedAt: String!
    tenant: Tenant!
  }

  type PolicyRule {
    id: ID!
    condition: String!
    action: String!
    parameters: String
  }

  type Resource {
    id: ID!
    name: String!
    type: String!
    provider: String!
    location: String!
    tags: [Tag!]!
    status: ResourceStatus!
    compliance: ComplianceStatus!
    cost: ResourceCost
    relationships: [ResourceRelationship!]!
    metrics: ResourceMetrics
    createdAt: String!
    updatedAt: String!
    tenant: Tenant!
  }

  type ResourceStatus {
    state: String!
    health: String!
    lastChecked: String!
  }

  type ComplianceStatus {
    compliant: Boolean!
    violations: [PolicyViolation!]!
    score: Float!
  }

  type PolicyViolation {
    policyId: String!
    policyName: String!
    severity: String!
    description: String!
  }

  type ResourceCost {
    daily: Float!
    monthly: Float!
    forecast: Float!
    currency: String!
  }

  type ResourceRelationship {
    type: String!
    targetId: String!
    targetName: String!
    targetType: String!
  }

  type ResourceMetrics {
    cpu: Float
    memory: Float
    disk: Float
    network: NetworkMetrics
  }

  type NetworkMetrics {
    inbound: Float!
    outbound: Float!
  }

  type Tag {
    key: String!
    value: String!
  }

  type ComplianceResult {
    id: ID!
    policyId: String!
    policy: Policy!
    resourceId: String!
    resource: Resource!
    status: String!
    details: String
    severity: String!
    remediationSteps: [String!]!
    autoRemediated: Boolean!
    checkedAt: String!
  }

  type Metrics {
    totalResources: Int!
    complianceScore: Float!
    costOptimizationScore: Float!
    securityScore: Float!
    operationalScore: Float!
    trends: MetricTrends!
    byProvider: [ProviderMetrics!]!
  }

  type MetricTrends {
    daily: [TrendPoint!]!
    weekly: [TrendPoint!]!
    monthly: [TrendPoint!]!
  }

  type TrendPoint {
    timestamp: String!
    value: Float!
  }

  type ProviderMetrics {
    provider: String!
    resources: Int!
    cost: Float!
    compliance: Float!
  }

  type Alert {
    id: ID!
    type: String!
    severity: String!
    title: String!
    description: String!
    resourceId: String
    resource: Resource
    acknowledged: Boolean!
    acknowledgedBy: String
    acknowledgedAt: String
    createdAt: String!
  }

  type CostAnalysis {
    total: Float!
    byService: [ServiceCost!]!
    byRegion: [RegionCost!]!
    forecast: CostForecast!
    recommendations: [CostRecommendation!]!
  }

  type ServiceCost {
    service: String!
    cost: Float!
    percentage: Float!
  }

  type RegionCost {
    region: String!
    cost: Float!
    percentage: Float!
  }

  type CostForecast {
    nextDay: Float!
    nextWeek: Float!
    nextMonth: Float!
  }

  type CostRecommendation {
    id: ID!
    description: String!
    potentialSavings: Float!
    effort: String!
    resources: [Resource!]!
  }

  type CostBudget {
    id: ID!
    amount: Float!
    period: String!
    alertThresholds: [Float!]!
    currentSpend: Float!
    projectedSpend: Float!
  }

  type Tenant {
    id: ID!
    name: String!
    tier: String!
  }

  # Subscription Event Types
  type ResourceEvent {
    type: String!
    action: String!
    resource: Resource!
    timestamp: String!
    triggeredBy: String!
  }

  type PolicyEvent {
    type: String!
    action: String!
    policy: Policy!
    timestamp: String!
    triggeredBy: String!
  }

  type ComplianceAlert {
    id: ID!
    severity: String!
    policyId: String!
    policy: Policy!
    resourceId: String!
    resource: Resource!
    violation: String!
    timestamp: String!
  }

  type CostAlert {
    id: ID!
    type: String!
    threshold: Float!
    currentValue: Float!
    message: String!
    resources: [Resource!]!
    timestamp: String!
  }

  type SecurityAlert {
    id: ID!
    severity: String!
    type: String!
    description: String!
    affectedResources: [Resource!]!
    mitigationSteps: [String!]!
    timestamp: String!
  }

  type GovernanceUpdate {
    id: ID!
    domain: String!
    type: String!
    description: String!
    impact: String!
    affectedPolicies: [Policy!]!
    timestamp: String!
  }

  type SystemStatus {
    healthy: Boolean!
    services: [ServiceStatus!]!
    timestamp: String!
  }

  type ServiceStatus {
    name: String!
    status: String!
    latency: Float!
    errorRate: Float!
  }

  # Input Types
  input PolicyFilter {
    category: String
    severity: String
    status: String
  }

  input ResourceFilter {
    type: String
    provider: String
    location: String
    tags: [TagInput!]
  }

  input TagInput {
    key: String!
    value: String!
  }

  input ComplianceFilter {
    status: String
    severity: String
    policyId: String
    resourceId: String
  }

  input TimeRange {
    start: String!
    end: String!
  }

  input CreatePolicyInput {
    name: String!
    description: String
    category: String!
    severity: String!
    rules: [PolicyRuleInput!]!
  }

  input PolicyRuleInput {
    condition: String!
    action: String!
    parameters: String
  }

  input UpdatePolicyInput {
    name: String
    description: String
    category: String
    severity: String
    status: String
    rules: [PolicyRuleInput!]
  }

  input CreateResourceInput {
    name: String!
    type: String!
    provider: String!
    location: String!
    tags: [TagInput!]
  }

  input UpdateResourceInput {
    name: String
    tags: [TagInput!]
    status: String
  }

  input CostBudgetInput {
    amount: Float!
    period: String!
    alertThresholds: [Float!]!
  }
`;

// Enhanced resolvers with subscription support
const resolvers = {
  Query: {
    policies: async (_, { filter }, context) => {
      // Implement filtered policy retrieval
      return mockPolicies(filter);
    },
    policy: async (_, { id }, context) => {
      return mockPolicy(id);
    },
    resources: async (_, { filter }, context) => {
      return mockResources(filter);
    },
    resource: async (_, { id }, context) => {
      return mockResource(id);
    },
    compliance: async (_, { filter }, context) => {
      return mockCompliance(filter);
    },
    metrics: async (_, __, context) => {
      return mockMetrics();
    },
    alerts: async (_, { severity }, context) => {
      return mockAlerts(severity);
    },
    costs: async (_, { timeRange }, context) => {
      return mockCostAnalysis(timeRange);
    },
  },

  Mutation: {
    createPolicy: async (_, { input }, context) => {
      const policy = {
        id: generateId(),
        ...input,
        status: 'Active',
        affectedResources: 0,
        complianceRate: 100.0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        tenant: mockTenant(),
      };
      
      // Publish policy change event
      await pubsub.publish(EVENTS.POLICY_CHANGED, {
        policyChanged: {
          type: 'POLICY',
          action: 'CREATED',
          policy,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return policy;
    },

    updatePolicy: async (_, { id, input }, context) => {
      const policy = {
        id,
        ...mockPolicy(id),
        ...input,
        updatedAt: new Date().toISOString(),
      };
      
      // Publish policy change event
      await pubsub.publish(EVENTS.POLICY_CHANGED, {
        policyChanged: {
          type: 'POLICY',
          action: 'UPDATED',
          policy,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return policy;
    },

    deletePolicy: async (_, { id }, context) => {
      const policy = mockPolicy(id);
      
      // Publish policy change event
      await pubsub.publish(EVENTS.POLICY_CHANGED, {
        policyChanged: {
          type: 'POLICY',
          action: 'DELETED',
          policy,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return true;
    },

    createResource: async (_, { input }, context) => {
      const resource = {
        id: generateId(),
        ...input,
        status: {
          state: 'Running',
          health: 'Healthy',
          lastChecked: new Date().toISOString(),
        },
        compliance: {
          compliant: true,
          violations: [],
          score: 100.0,
        },
        relationships: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        tenant: mockTenant(),
      };
      
      // Publish resource change event
      await pubsub.publish(EVENTS.RESOURCE_CREATED, {
        resourceChanged: {
          type: 'RESOURCE',
          action: 'CREATED',
          resource,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return resource;
    },

    updateResource: async (_, { id, input }, context) => {
      const resource = {
        ...mockResource(id),
        ...input,
        updatedAt: new Date().toISOString(),
      };
      
      // Publish resource change event
      await pubsub.publish(EVENTS.RESOURCE_UPDATED, {
        resourceChanged: {
          type: 'RESOURCE',
          action: 'UPDATED',
          resource,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return resource;
    },

    deleteResource: async (_, { id }, context) => {
      const resource = mockResource(id);
      
      // Publish resource change event
      await pubsub.publish(EVENTS.RESOURCE_DELETED, {
        resourceChanged: {
          type: 'RESOURCE',
          action: 'DELETED',
          resource,
          timestamp: new Date().toISOString(),
          triggeredBy: context.user?.id || 'system',
        },
      });
      
      return true;
    },

    triggerCompliance: async (_, { policyId, resourceIds }, context) => {
      const result = {
        id: generateId(),
        policyId,
        policy: mockPolicy(policyId),
        resourceId: resourceIds[0],
        resource: mockResource(resourceIds[0]),
        status: 'Non-Compliant',
        details: 'Resource violates policy requirements',
        severity: 'High',
        remediationSteps: ['Update resource configuration', 'Apply required tags'],
        autoRemediated: false,
        checkedAt: new Date().toISOString(),
      };
      
      // Publish compliance alert
      await pubsub.publish(EVENTS.COMPLIANCE_ALERT, {
        complianceAlert: {
          id: generateId(),
          severity: 'High',
          policyId,
          policy: mockPolicy(policyId),
          resourceId: resourceIds[0],
          resource: mockResource(resourceIds[0]),
          violation: 'Policy violation detected',
          timestamp: new Date().toISOString(),
        },
      });
      
      return result;
    },

    acknowledgeAlert: async (_, { alertId }, context) => {
      const alert = {
        ...mockAlert(alertId),
        acknowledged: true,
        acknowledgedBy: context.user?.id || 'system',
        acknowledgedAt: new Date().toISOString(),
      };
      
      return alert;
    },

    updateCostBudget: async (_, { budget }, context) => {
      const costBudget = {
        id: generateId(),
        ...budget,
        currentSpend: 5000,
        projectedSpend: 7500,
      };
      
      // Check if budget threshold exceeded
      if (costBudget.currentSpend > costBudget.amount * 0.8) {
        await pubsub.publish(EVENTS.COST_ALERT, {
          costAlert: {
            id: generateId(),
            type: 'BUDGET_THRESHOLD',
            threshold: costBudget.amount * 0.8,
            currentValue: costBudget.currentSpend,
            message: 'Cost budget threshold exceeded',
            resources: [],
            timestamp: new Date().toISOString(),
          },
        });
      }
      
      return costBudget;
    },
  },

  Subscription: {
    resourceChanged: {
      subscribe: (_, { resourceType }) => {
        return pubsub.asyncIterator([
          EVENTS.RESOURCE_CREATED,
          EVENTS.RESOURCE_UPDATED,
          EVENTS.RESOURCE_DELETED,
        ]);
      },
    },

    policyChanged: {
      subscribe: (_, { category }) => {
        return pubsub.asyncIterator([EVENTS.POLICY_CHANGED]);
      },
    },

    complianceAlert: {
      subscribe: (_, { severity }) => {
        return pubsub.asyncIterator([EVENTS.COMPLIANCE_ALERT]);
      },
    },

    costAlert: {
      subscribe: (_, { threshold }) => {
        return pubsub.asyncIterator([EVENTS.COST_ALERT]);
      },
    },

    securityAlert: {
      subscribe: (_, { severity }) => {
        return pubsub.asyncIterator([EVENTS.SECURITY_ALERT]);
      },
    },

    governanceUpdate: {
      subscribe: (_, { domain }) => {
        return pubsub.asyncIterator([EVENTS.GOVERNANCE_UPDATE]);
      },
    },

    metricsUpdate: {
      subscribe: async function* (_, { interval = 5000 }) {
        while (true) {
          yield { metricsUpdate: mockMetrics() };
          await new Promise(resolve => setTimeout(resolve, interval));
        }
      },
    },

    systemStatus: {
      subscribe: async function* () {
        while (true) {
          yield {
            systemStatus: {
              healthy: true,
              services: [
                { name: 'Core API', status: 'Running', latency: 25.5, errorRate: 0.1 },
                { name: 'GraphQL', status: 'Running', latency: 15.2, errorRate: 0.0 },
                { name: 'AI Engine', status: 'Running', latency: 150.8, errorRate: 0.5 },
              ],
              timestamp: new Date().toISOString(),
            },
          };
          await new Promise(resolve => setTimeout(resolve, 10000));
        }
      },
    },
  },
};

// Helper functions
function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

function mockTenant() {
  return {
    id: 'tenant-1',
    name: 'Enterprise Corp',
    tier: 'enterprise',
  };
}

function mockPolicy(id) {
  return {
    id,
    name: 'Sample Policy',
    description: 'Policy description',
    category: 'Security',
    severity: 'High',
    status: 'Active',
    rules: [
      {
        id: 'rule-1',
        condition: 'resource.encryption == true',
        action: 'allow',
        parameters: '{}',
      },
    ],
    affectedResources: 42,
    complianceRate: 95.5,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    tenant: mockTenant(),
  };
}

function mockPolicies(filter) {
  return [mockPolicy('1'), mockPolicy('2')];
}

function mockResource(id) {
  return {
    id,
    name: 'Sample Resource',
    type: 'VirtualMachine',
    provider: 'Azure',
    location: 'East US',
    tags: [
      { key: 'Environment', value: 'Production' },
      { key: 'Team', value: 'Platform' },
    ],
    status: {
      state: 'Running',
      health: 'Healthy',
      lastChecked: new Date().toISOString(),
    },
    compliance: {
      compliant: true,
      violations: [],
      score: 98.5,
    },
    cost: {
      daily: 50.0,
      monthly: 1500.0,
      forecast: 1550.0,
      currency: 'USD',
    },
    relationships: [],
    metrics: {
      cpu: 45.5,
      memory: 62.3,
      disk: 78.9,
      network: {
        inbound: 125.5,
        outbound: 89.3,
      },
    },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    tenant: mockTenant(),
  };
}

function mockResources(filter) {
  return [mockResource('1'), mockResource('2')];
}

function mockCompliance(filter) {
  return [
    {
      id: '1',
      policyId: '1',
      policy: mockPolicy('1'),
      resourceId: '1',
      resource: mockResource('1'),
      status: 'Compliant',
      details: null,
      severity: 'Low',
      remediationSteps: [],
      autoRemediated: false,
      checkedAt: new Date().toISOString(),
    },
  ];
}

function mockMetrics() {
  return {
    totalResources: 250,
    complianceScore: 95.5,
    costOptimizationScore: 88.2,
    securityScore: 92.8,
    operationalScore: 90.5,
    trends: {
      daily: [
        { timestamp: new Date().toISOString(), value: 95.5 },
      ],
      weekly: [
        { timestamp: new Date().toISOString(), value: 94.8 },
      ],
      monthly: [
        { timestamp: new Date().toISOString(), value: 93.2 },
      ],
    },
    byProvider: [
      { provider: 'Azure', resources: 150, cost: 15000.0, compliance: 96.5 },
      { provider: 'AWS', resources: 75, cost: 8500.0, compliance: 94.2 },
      { provider: 'GCP', resources: 25, cost: 3200.0, compliance: 93.8 },
    ],
  };
}

function mockAlerts(severity) {
  return [
    {
      id: '1',
      type: 'Security',
      severity: severity || 'High',
      title: 'Unauthorized access attempt',
      description: 'Multiple failed login attempts detected',
      resourceId: '1',
      resource: mockResource('1'),
      acknowledged: false,
      acknowledgedBy: null,
      acknowledgedAt: null,
      createdAt: new Date().toISOString(),
    },
  ];
}

function mockAlert(id) {
  return mockAlerts()[0];
}

function mockCostAnalysis(timeRange) {
  return {
    total: 25000.0,
    byService: [
      { service: 'Compute', cost: 10000.0, percentage: 40.0 },
      { service: 'Storage', cost: 5000.0, percentage: 20.0 },
      { service: 'Network', cost: 3000.0, percentage: 12.0 },
      { service: 'Database', cost: 7000.0, percentage: 28.0 },
    ],
    byRegion: [
      { region: 'East US', cost: 12000.0, percentage: 48.0 },
      { region: 'West Europe', cost: 8000.0, percentage: 32.0 },
      { region: 'Southeast Asia', cost: 5000.0, percentage: 20.0 },
    ],
    forecast: {
      nextDay: 850.0,
      nextWeek: 5950.0,
      nextMonth: 26500.0,
    },
    recommendations: [
      {
        id: 'rec-1',
        description: 'Right-size underutilized VMs',
        potentialSavings: 2000.0,
        effort: 'Low',
        resources: [mockResource('1')],
      },
    ],
  };
}

// Main server setup
async function startEnhancedGateway() {
  const app = express();
  const httpServer = http.createServer(app);

  // Create executable schema
  const schema = makeExecutableSchema({ typeDefs, resolvers });

  // Create WebSocket server for subscriptions
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/graphql',
  });

  // Set up WebSocket server with graphql-ws
  const serverCleanup = useServer(
    {
      schema,
      context: async (ctx) => {
        // Add authentication context for WebSocket connections
        return {
          user: await authenticateWebSocket(ctx),
          tenant: await getTenantFromToken(ctx),
        };
      },
      onConnect: async (ctx) => {
        console.log('Client connected:', ctx.connectionParams);
      },
      onDisconnect: async (ctx) => {
        console.log('Client disconnected');
      },
    },
    wsServer
  );

  // Create Apollo Server
  const server = new ApolloServer({
    schema,
    plugins: [
      ApolloServerPluginDrainHttpServer({ httpServer }),
      {
        async serverWillStart() {
          return {
            async drainServer() {
              await serverCleanup.dispose();
            },
          };
        },
      },
    ],
    formatError: (err) => {
      console.error('GraphQL Error:', err);
      return err;
    },
  });

  await server.start();

  // Set up Express middleware
  app.use(
    '/graphql',
    cors({
      origin: (process.env.ALLOWED_ORIGINS || 'http://localhost:3000')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean),
      credentials: true,
    }),
    bodyParser.json(),
    expressMiddleware(server, {
      context: async ({ req }) => {
        // Add authentication context for HTTP requests
        return {
          user: await authenticateHTTP(req),
          tenant: await getTenantFromRequest(req),
        };
      },
    })
  );

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({ 
      status: 'healthy',
      timestamp: new Date().toISOString(),
      subscriptions: 'enabled',
      websocket: 'active',
    });
  });

  // Metrics endpoint
  app.get('/metrics', (req, res) => {
    res.json({
      requests: 0,
      subscriptions: wsServer.clients.size,
      uptime: process.uptime(),
    });
  });

  const PORT = process.env.PORT || 4000;
  
  httpServer.listen(PORT, () => {
    console.log(`🚀 Enhanced GraphQL Gateway ready at http://localhost:${PORT}/graphql`);
    console.log(`🔌 WebSocket subscriptions ready at ws://localhost:${PORT}/graphql`);
    console.log(`📊 Health check at http://localhost:${PORT}/health`);
  });

  // Simulate real-time events
  if (process.env.ENABLE_MOCK_EVENTS === 'true') {
    startMockEventGenerator();
  }
}

// Authentication helpers
async function authenticateWebSocket(ctx) {
  const token = ctx.connectionParams?.authentication;
  if (!token) return null;
  
  // Implement token validation
  return { id: 'user-1', email: 'user@example.com' };
}

async function getTenantFromToken(ctx) {
  const token = ctx.connectionParams?.authentication;
  if (!token) return null;
  
  // Extract tenant from token
  return { id: 'tenant-1', name: 'Enterprise Corp' };
}

async function authenticateHTTP(req) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return null;
  
  // Implement token validation
  return { id: 'user-1', email: 'user@example.com' };
}

async function getTenantFromRequest(req) {
  const tenantHeader = req.headers['x-tenant-id'];
  if (!tenantHeader) return null;
  
  return { id: tenantHeader, name: 'Enterprise Corp' };
}

// Mock event generator for testing
function startMockEventGenerator() {
  setInterval(async () => {
    // Randomly publish different events
    const eventTypes = [
      async () => {
        await pubsub.publish(EVENTS.RESOURCE_UPDATED, {
          resourceChanged: {
            type: 'RESOURCE',
            action: 'UPDATED',
            resource: mockResource(generateId()),
            timestamp: new Date().toISOString(),
            triggeredBy: 'system',
          },
        });
      },
      async () => {
        await pubsub.publish(EVENTS.COMPLIANCE_ALERT, {
          complianceAlert: {
            id: generateId(),
            severity: ['Low', 'Medium', 'High', 'Critical'][Math.floor(Math.random() * 4)],
            policyId: '1',
            policy: mockPolicy('1'),
            resourceId: '1',
            resource: mockResource('1'),
            violation: 'Policy compliance check failed',
            timestamp: new Date().toISOString(),
          },
        });
      },
      async () => {
        await pubsub.publish(EVENTS.COST_ALERT, {
          costAlert: {
            id: generateId(),
            type: 'ANOMALY',
            threshold: 1000,
            currentValue: 1250,
            message: 'Unusual spending detected',
            resources: [mockResource('1')],
            timestamp: new Date().toISOString(),
          },
        });
      },
    ];
    
    const randomEvent = eventTypes[Math.floor(Math.random() * eventTypes.length)];
    await randomEvent();
  }, 10000); // Publish event every 10 seconds
}

// Start the enhanced gateway
startEnhancedGateway().catch((err) => {
  console.error('Failed to start enhanced gateway:', err);
  process.exit(1);
});

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

module.exports = { pubsub, EVENTS };