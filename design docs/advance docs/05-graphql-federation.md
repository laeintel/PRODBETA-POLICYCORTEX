# GraphQL Federation Architecture

## Table of Contents
1. [Federation Overview](#federation-overview)
2. [Schema Architecture](#schema-architecture)
3. [Resolvers and Data Sources](#resolvers-and-data-sources)
4. [Subscriptions and Real-time](#subscriptions-and-real-time)
5. [Query Planning](#query-planning)
6. [Caching Strategy](#caching-strategy)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Testing Federation](#testing-federation)

## Federation Overview

PolicyCortex uses Apollo Federation to create a unified GraphQL API that spans multiple microservices while maintaining service autonomy.

### Architecture Pattern

```
┌─────────────────────────────────────────────────────┐
│                Apollo Gateway                        │
│  ┌───────────────────────────────────────────────┐  │
│  │         Query Planning Engine                 │  │
│  │  • Schema composition                         │  │
│  │  • Query distribution                         │  │
│  │  • Response merging                           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Core API  │ │ AI Service  │ │ Azure API   │
    │  (Rust)     │ │  (Python)   │ │  (Python)   │
    │             │ │             │ │             │
    │ • Resources │ │ • Insights  │ │ • Live Data │
    │ • Policies  │ │ • Predict.  │ │ • Metrics   │
    │ • Events    │ │ • Convers.  │ │ • Config    │
    └─────────────┘ └─────────────┘ └─────────────┘
```

### Federation Benefits

1. **Service Autonomy**: Each service owns its schema and data
2. **Schema Composition**: Unified API without tight coupling
3. **Query Efficiency**: Single request for complex data needs
4. **Type Safety**: Strong typing across service boundaries
5. **Real-time Updates**: Subscriptions across federated services

## Schema Architecture

### Supergraph Schema

```graphql
# Gateway supergraph composition
schema
  @link(url: "https://specs.apollo.dev/link/v1.0")
  @link(url: "https://specs.apollo.dev/join/v0.3", for: EXECUTION)
{
  query: Query
  mutation: Mutation
  subscription: Subscription
}

directive @join__enumValue(graph: join__Graph!) on ENUM_VALUE
directive @join__field(graph: join__Graph!, requires: join__FieldSet, provides: join__FieldSet, type: String, external: Boolean, override: String, usedOverridden: Boolean) on FIELD_DEFINITION | INPUT_FIELD_DEFINITION
directive @join__graph(name: String!, url: String!) on ENUM_VALUE
directive @join__implements(graph: join__Graph!, interface: String!) on OBJECT | INTERFACE
directive @join__type(graph: join__Graph!, key: join__FieldSet, extension: Boolean, resolvable: Boolean, isInterfaceObject: Boolean) on OBJECT | INTERFACE | UNION | ENUM | INPUT_OBJECT | SCALAR
directive @join__unionMember(graph: join__Graph!, member: String!) on UNION

scalar join__FieldSet

enum join__Graph {
  CORE @join__graph(name: "core", url: "http://localhost:8080/graphql")
  AI @join__graph(name: "ai", url: "http://localhost:8081/graphql")
  AZURE @join__graph(name: "azure", url: "http://localhost:8082/graphql")
}
```

### Core Service Schema

```graphql
# Core service schema (core/graphql/schema.graphql)
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@extends", "@external", "@requires", "@provides"])

type Query {
  # Resource queries
  resources(filters: ResourceFilters): [Resource!]!
  resource(id: ID!): Resource
  
  # Policy queries
  policies(type: PolicyType): [Policy!]!
  policy(id: ID!): Policy
  
  # Compliance queries
  complianceStatus(resourceId: ID!): ComplianceStatus!
  complianceReport(filters: ComplianceFilters): ComplianceReport!
}

type Mutation {
  # Resource mutations
  createResource(input: CreateResourceInput!): Resource!
  updateResource(id: ID!, input: UpdateResourceInput!): Resource!
  deleteResource(id: ID!): Boolean!
  
  # Policy mutations
  createPolicy(input: CreatePolicyInput!): Policy!
  updatePolicy(id: ID!, input: UpdatePolicyInput!): Policy!
  evaluatePolicy(id: ID!, resourceId: ID!): PolicyEvaluation!
  
  # Action mutations
  executeAction(input: ExecuteActionInput!): ActionExecution!
  approveAction(id: ID!): ActionExecution!
  rollbackAction(id: ID!): ActionExecution!
}

type Subscription {
  # Real-time resource updates
  resourceUpdated(id: ID!): Resource!
  resourcesUpdated(filters: ResourceFilters): Resource!
  
  # Policy evaluation updates
  policyEvaluated: PolicyEvaluation!
  complianceChanged(resourceId: ID!): ComplianceStatus!
  
  # Action execution updates
  actionStatusChanged(id: ID!): ActionExecution!
}

# Core entities
type Resource @key(fields: "id") {
  id: ID!
  name: String!
  type: ResourceType!
  subscriptionId: String!
  resourceGroupName: String!
  location: String!
  tags: [Tag!]!
  properties: JSON!
  createdAt: DateTime!
  updatedAt: DateTime!
  
  # Federated fields (resolved by other services)
  insights: ResourceInsights! @external
  liveData: AzureResourceData! @external
  predictions: [CompliancePrediction!]! @external
}

type Policy @key(fields: "id") {
  id: ID!
  name: String!
  type: PolicyType!
  category: String!
  severity: Severity!
  definition: JSON!
  enabled: Boolean!
  createdAt: DateTime!
  updatedAt: DateTime!
  
  # Computed fields
  evaluations: [PolicyEvaluation!]!
  affectedResources: [Resource!]!
}

type PolicyEvaluation @key(fields: "id") {
  id: ID!
  policyId: ID!
  resourceId: ID!
  status: EvaluationStatus!
  result: EvaluationResult!
  evidence: [Evidence!]!
  recommendations: [String!]!
  evaluatedAt: DateTime!
  
  # Relations
  policy: Policy!
  resource: Resource!
}

# Enums and scalars
enum ResourceType {
  VIRTUAL_MACHINE
  STORAGE_ACCOUNT
  SQL_DATABASE
  APP_SERVICE
  FUNCTION_APP
  KEY_VAULT
  NETWORK_SECURITY_GROUP
  LOAD_BALANCER
}

enum PolicyType {
  SECURITY
  COMPLIANCE
  COST_OPTIMIZATION
  PERFORMANCE
  GOVERNANCE
}

enum Severity {
  CRITICAL
  HIGH
  MEDIUM
  LOW
  INFO
}

enum EvaluationStatus {
  PENDING
  EVALUATING
  COMPLETED
  FAILED
}

enum EvaluationResult {
  COMPLIANT
  NON_COMPLIANT
  WARNING
  UNKNOWN
}

scalar DateTime
scalar JSON

# Input types
input ResourceFilters {
  subscriptionId: String
  resourceGroupName: String
  resourceType: ResourceType
  location: String
  tags: [TagFilter!]
  complianceStatus: EvaluationResult
}

input TagFilter {
  key: String!
  value: String
}

input ComplianceFilters {
  subscriptionId: String
  resourceType: ResourceType
  policyType: PolicyType
  severity: Severity
  status: EvaluationResult
  dateRange: DateRangeInput
}

input DateRangeInput {
  startDate: DateTime!
  endDate: DateTime!
}

input CreateResourceInput {
  name: String!
  type: ResourceType!
  subscriptionId: String!
  resourceGroupName: String!
  location: String!
  properties: JSON!
  tags: [TagInput!]
}

input TagInput {
  key: String!
  value: String!
}

input CreatePolicyInput {
  name: String!
  type: PolicyType!
  category: String!
  severity: Severity!
  definition: JSON!
  enabled: Boolean = true
}

input ExecuteActionInput {
  resourceId: ID!
  actionType: String!
  parameters: JSON
  dryRun: Boolean = false
}

type Tag {
  key: String!
  value: String!
}

type Evidence {
  type: String!
  description: String!
  severity: Severity!
  data: JSON!
}

type ComplianceStatus {
  resourceId: ID!
  overallStatus: EvaluationResult!
  policyEvaluations: [PolicyEvaluation!]!
  score: Float!
  lastEvaluated: DateTime!
}

type ComplianceReport {
  summary: ComplianceSummary!
  evaluations: [PolicyEvaluation!]!
  trends: [ComplianceTrend!]!
  recommendations: [String!]!
}

type ComplianceSummary {
  totalResources: Int!
  compliantResources: Int!
  nonCompliantResources: Int!
  complianceScore: Float!
}

type ComplianceTrend {
  date: DateTime!
  score: Float!
  compliantCount: Int!
  nonCompliantCount: Int!
}

type ActionExecution {
  id: ID!
  resourceId: ID!
  actionType: String!
  status: ExecutionStatus!
  parameters: JSON
  result: JSON
  error: String
  dryRun: Boolean!
  createdAt: DateTime!
  updatedAt: DateTime!
}

enum ExecutionStatus {
  PENDING
  EXECUTING
  COMPLETED
  FAILED
  ROLLED_BACK
}
```

### AI Service Schema

```graphql
# AI service schema (backend/services/ai_engine/graphql/schema.graphql)
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@extends", "@external"])

type Resource @key(fields: "id") @extends {
  id: ID! @external
  
  # AI-generated insights
  insights: ResourceInsights!
  predictions: [CompliancePrediction!]!
}

type Query {
  # AI-specific queries
  generatePolicy(input: PolicyGenerationInput!): GeneratedPolicy!
  analyzeCompliance(resourceId: ID!): ComplianceAnalysis!
  optimizeCosts(input: CostOptimizationInput!): CostOptimization!
  conversationalQuery(query: String!): ConversationalResponse!
}

type Mutation {
  # AI interactions
  trainModel(input: TrainingInput!): TrainingJob!
  refinePolicy(policyId: ID!, feedback: String!): Policy!
  generateRecommendations(resourceId: ID!): [Recommendation!]!
}

type Subscription {
  # AI processing updates
  trainingProgress(jobId: ID!): TrainingProgress!
  insightsUpdated(resourceId: ID!): ResourceInsights!
  predictionUpdated(resourceId: ID!): CompliancePrediction!
}

type ResourceInsights {
  resourceId: ID!
  securityScore: Float!
  complianceScore: Float!
  costOptimizationScore: Float!
  performanceScore: Float!
  recommendations: [Recommendation!]!
  risks: [Risk!]!
  opportunities: [Opportunity!]!
  generatedAt: DateTime!
}

type CompliancePrediction {
  resourceId: ID!
  policyId: ID!
  predictedStatus: EvaluationResult!
  confidence: Float!
  timeHorizon: Int! # days
  factors: [PredictionFactor!]!
  mitigation: [String!]!
  generatedAt: DateTime!
}

type PredictionFactor {
  name: String!
  impact: Float! # -1.0 to 1.0
  description: String!
}

type GeneratedPolicy {
  name: String!
  type: PolicyType!
  category: String!
  definition: JSON!
  rationale: String!
  affectedResourceTypes: [ResourceType!]!
  estimatedImpact: PolicyImpact!
}

type PolicyImpact {
  securityImprovement: Float!
  complianceImprovement: Float!
  costImpact: Float!
  performanceImpact: Float!
}

type ComplianceAnalysis {
  resourceId: ID!
  currentStatus: EvaluationResult!
  gapAnalysis: [ComplianceGap!]!
  remediationPlan: [RemediationStep!]!
  riskAssessment: RiskAssessment!
}

type ComplianceGap {
  policyId: ID!
  description: String!
  severity: Severity!
  evidence: [Evidence!]!
}

type RemediationStep {
  order: Int!
  action: String!
  description: String!
  estimatedEffort: String!
  priority: Priority!
}

type RiskAssessment {
  overallRisk: RiskLevel!
  categories: [RiskCategory!]!
}

type RiskCategory {
  name: String!
  level: RiskLevel!
  description: String!
  mitigation: [String!]!
}

enum RiskLevel {
  VERY_LOW
  LOW
  MEDIUM
  HIGH
  VERY_HIGH
}

enum Priority {
  LOW
  MEDIUM
  HIGH
  CRITICAL
}

type CostOptimization {
  resourceId: ID!
  currentMonthlyCost: Float!
  optimizedMonthlyCost: Float!
  potentialSavings: Float!
  savingsPercentage: Float!
  recommendations: [CostRecommendation!]!
}

type CostRecommendation {
  type: String!
  description: String!
  impact: Float!
  effort: String!
  priority: Priority!
}

type ConversationalResponse {
  query: String!
  response: String!
  confidence: Float!
  sources: [String!]!
  suggestedActions: [String!]!
  relatedQuestions: [String!]!
}

type Recommendation {
  id: ID!
  type: RecommendationType!
  title: String!
  description: String!
  priority: Priority!
  category: String!
  estimatedImpact: String!
  implementationEffort: String!
  relatedPolicies: [ID!]!
}

enum RecommendationType {
  SECURITY_IMPROVEMENT
  COMPLIANCE_FIX
  COST_OPTIMIZATION
  PERFORMANCE_ENHANCEMENT
  GOVERNANCE_ALIGNMENT
}

type Risk {
  id: ID!
  type: String!
  title: String!
  description: String!
  severity: Severity!
  likelihood: Float!
  impact: Float!
  mitigation: [String!]!
}

type Opportunity {
  id: ID!
  type: String!
  title: String!
  description: String!
  value: Float!
  effort: String!
  timeframe: String!
}

type TrainingJob {
  id: ID!
  status: TrainingStatus!
  progress: Float!
  startedAt: DateTime!
  completedAt: DateTime
  error: String
}

type TrainingProgress {
  jobId: ID!
  status: TrainingStatus!
  progress: Float!
  currentEpoch: Int!
  totalEpochs: Int!
  loss: Float!
  accuracy: Float!
  estimatedTimeRemaining: Int! # seconds
}

enum TrainingStatus {
  PENDING
  RUNNING
  COMPLETED
  FAILED
  CANCELLED
}

# Input types
input PolicyGenerationInput {
  resourceType: ResourceType!
  category: String!
  requirements: [String!]!
  existingPolicies: [ID!]
  constraints: JSON
}

input CostOptimizationInput {
  resourceIds: [ID!]!
  targetSavingsPercentage: Float
  timeHorizon: Int # days
  constraints: [String!]
}

input TrainingInput {
  modelType: String!
  trainingData: JSON!
  hyperparameters: JSON
}
```

### Azure Service Schema

```graphql
# Azure service schema (backend/services/api_gateway/graphql/schema.graphql)
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@extends", "@external"])

type Resource @key(fields: "id") @extends {
  id: ID! @external
  
  # Live Azure data
  liveData: AzureResourceData!
}

type Query {
  # Azure-specific queries
  subscriptions: [AzureSubscription!]!
  resourceGroups(subscriptionId: String!): [AzureResourceGroup!]!
  azureMetrics(input: MetricsInput!): AzureMetrics!
  costAnalysis(input: CostAnalysisInput!): CostAnalysis!
  securityAlerts(subscriptionId: String!): [SecurityAlert!]!
}

type Mutation {
  # Azure operations
  syncSubscription(subscriptionId: String!): SyncResult!
  updateResourceTags(resourceId: ID!, tags: [TagInput!]!): Resource!
  enableDiagnostics(resourceId: ID!, settings: DiagnosticsSettings!): Boolean!
}

type Subscription {
  # Azure real-time updates
  azureResourceUpdated(subscriptionId: String!): AzureResourceData!
  metricsUpdated(resourceId: ID!): AzureMetrics!
  alertTriggered(subscriptionId: String!): SecurityAlert!
}

type AzureResourceData {
  resourceId: ID!
  azureResourceId: String!
  properties: JSON!
  configuration: JSON!
  metrics: AzureMetrics!
  diagnostics: DiagnosticsData!
  tags: [Tag!]!
  lastSynced: DateTime!
}

type AzureSubscription {
  id: String!
  displayName: String!
  state: SubscriptionState!
  tenantId: String!
  resourceGroups: [AzureResourceGroup!]!
}

type AzureResourceGroup {
  id: String!
  name: String!
  location: String!
  subscriptionId: String!
  resources: [Resource!]!
}

type AzureMetrics {
  resourceId: ID!
  timeGrain: String!
  metrics: [MetricData!]!
  aggregatedAt: DateTime!
}

type MetricData {
  name: String!
  displayName: String!
  unit: String!
  values: [MetricValue!]!
}

type MetricValue {
  timestamp: DateTime!
  value: Float!
  aggregationType: AggregationType!
}

enum AggregationType {
  AVERAGE
  MINIMUM
  MAXIMUM
  TOTAL
  COUNT
}

type DiagnosticsData {
  resourceId: ID!
  enabled: Boolean!
  settings: JSON!
  logs: [LogEntry!]!
  events: [DiagnosticEvent!]!
}

type LogEntry {
  timestamp: DateTime!
  level: LogLevel!
  category: String!
  message: String!
  properties: JSON
}

type DiagnosticEvent {
  timestamp: DateTime!
  eventType: String!
  severity: Severity!
  description: String!
  details: JSON!
}

enum LogLevel {
  ERROR
  WARNING
  INFORMATION
  VERBOSE
}

type CostAnalysis {
  subscriptionId: String!
  timeframe: Timeframe!
  totalCost: Float!
  costByService: [ServiceCost!]!
  costByResourceGroup: [ResourceGroupCost!]!
  costTrends: [CostTrend!]!
  forecast: CostForecast!
}

type ServiceCost {
  serviceName: String!
  cost: Float!
  percentage: Float!
  trend: Float! # percentage change
}

type ResourceGroupCost {
  resourceGroupName: String!
  cost: Float!
  percentage: Float!
  resources: [ResourceCost!]!
}

type ResourceCost {
  resourceId: ID!
  name: String!
  cost: Float!
  dailyAverage: Float!
}

type CostTrend {
  date: DateTime!
  cost: Float!
  cumulativeCost: Float!
}

type CostForecast {
  projectedMonthlyCost: Float!
  confidence: Float!
  factors: [ForecastFactor!]!
}

type ForecastFactor {
  name: String!
  impact: Float!
  description: String!
}

type SecurityAlert {
  id: ID!
  subscriptionId: String!
  resourceId: ID
  alertType: String!
  severity: AlertSeverity!
  status: AlertStatus!
  title: String!
  description: String!
  remediationSteps: [String!]!
  detectedAt: DateTime!
  updatedAt: DateTime!
}

enum AlertSeverity {
  INFORMATIONAL
  LOW
  MEDIUM
  HIGH
}

enum AlertStatus {
  ACTIVE
  RESOLVED
  DISMISSED
}

enum SubscriptionState {
  ENABLED
  DISABLED
  DELETED
  WARNED
}

type SyncResult {
  subscriptionId: String!
  success: Boolean!
  resourcesUpdated: Int!
  resourcesAdded: Int!
  resourcesRemoved: Int!
  errors: [String!]!
  syncedAt: DateTime!
}

enum Timeframe {
  LAST_HOUR
  LAST_24_HOURS
  LAST_7_DAYS
  LAST_30_DAYS
  LAST_90_DAYS
  CUSTOM
}

# Input types
input MetricsInput {
  resourceId: ID!
  metricNames: [String!]!
  timespan: TimespanInput!
  interval: String
  aggregation: AggregationType
}

input TimespanInput {
  start: DateTime!
  end: DateTime!
}

input CostAnalysisInput {
  subscriptionId: String!
  timeframe: Timeframe!
  granularity: Granularity!
  groupBy: [String!]
}

enum Granularity {
  DAILY
  MONTHLY
}

input DiagnosticsSettings {
  enabled: Boolean!
  logCategories: [String!]!
  retentionDays: Int!
  storageAccountId: String
}
```

## Resolvers and Data Sources

### Gateway Configuration

```javascript
// graphql/gateway.js
const { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } = require('@apollo/gateway');
const { ApolloServer } = require('apollo-server-express');
const express = require('express');
const cors = require('cors');

// Custom data source for authentication
class AuthenticatedDataSource extends RemoteGraphQLDataSource {
  willSendRequest({ request, context }) {
    // Forward authentication headers
    if (context.headers.authorization) {
      request.http.headers.set('authorization', context.headers.authorization);
    }
    
    // Add correlation ID for tracing
    if (context.correlationId) {
      request.http.headers.set('x-correlation-id', context.correlationId);
    }
    
    // Add tenant context
    if (context.tenantId) {
      request.http.headers.set('x-tenant-id', context.tenantId);
    }
  }

  didReceiveResponse({ response, request, context }) {
    // Log performance metrics
    console.log(`Service ${this.url} responded in ${response.http.headers.get('x-response-time')}ms`);
    return response;
  }

  didEncounterError(error, request, context) {
    // Enhanced error handling
    console.error(`Error from service ${this.url}:`, {
      error: error.message,
      correlationId: context.correlationId,
      query: request.query
    });
    
    // Transform service-specific errors
    if (error.extensions?.code === 'AZURE_API_ERROR') {
      error.extensions.code = 'EXTERNAL_SERVICE_ERROR';
      error.message = 'Azure service temporarily unavailable';
    }
    
    return error;
  }
}

// Gateway setup
const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: [
      { name: 'core', url: 'http://localhost:8080/graphql' },
      { name: 'ai', url: 'http://localhost:8081/graphql' },
      { name: 'azure', url: 'http://localhost:8082/graphql' }
    ],
  }),
  buildService({ name, url }) {
    return new AuthenticatedDataSource({ url });
  },
  experimental_pollInterval: 10000, // Poll for schema changes
});

// Apollo Server setup
const server = new ApolloServer({
  gateway,
  context: ({ req }) => ({
    headers: req.headers,
    correlationId: req.headers['x-correlation-id'] || generateCorrelationId(),
    tenantId: extractTenantId(req),
    user: extractUser(req)
  }),
  plugins: [
    // Query complexity analysis
    {
      requestDidStart() {
        return {
          didResolveOperation(requestContext) {
            const complexity = calculateQueryComplexity(requestContext.request.query);
            if (complexity > 1000) {
              throw new Error('Query too complex');
            }
          }
        };
      }
    },
    // Performance monitoring
    {
      requestDidStart() {
        return {
          willSendResponse(requestContext) {
            const responseTime = Date.now() - requestContext.request.http.body.timestamp;
            requestContext.response.http.headers.set('x-response-time', responseTime);
          }
        };
      }
    }
  ],
  subscriptions: {
    path: '/graphql',
    onConnect: (connectionParams, webSocket) => {
      // WebSocket authentication
      const token = connectionParams.authorization;
      if (!token) {
        throw new Error('Authentication required');
      }
      
      return {
        user: validateToken(token),
        correlationId: generateCorrelationId()
      };
    }
  },
  formatError: (error) => {
    // Error formatting and logging
    console.error('GraphQL Error:', {
      message: error.message,
      locations: error.locations,
      path: error.path,
      extensions: error.extensions
    });
    
    // Hide internal errors in production
    if (process.env.NODE_ENV === 'production') {
      if (error.extensions?.code === 'INTERNAL_ERROR') {
        return new Error('Internal server error');
      }
    }
    
    return error;
  }
});

// Express setup
const app = express();
app.use(cors());
app.use('/health', (req, res) => res.json({ status: 'healthy' }));

// Apply GraphQL middleware
server.applyMiddleware({ app, path: '/graphql' });

// Start server
const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
  console.log(`🚀 Federation gateway ready at http://localhost:${PORT}${server.graphqlPath}`);
});

function generateCorrelationId() {
  return require('crypto').randomBytes(16).toString('hex');
}

function extractTenantId(req) {
  return req.headers['x-tenant-id'] || 'default';
}

function extractUser(req) {
  // Extract user from JWT token
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return null;
  
  try {
    return jwt.verify(token, process.env.JWT_SECRET);
  } catch {
    return null;
  }
}

function calculateQueryComplexity(query) {
  // Simplified complexity calculation
  const depthLimit = 10;
  const complexityLimit = 1000;
  
  // Implementation would analyze AST and calculate complexity
  return 100; // Placeholder
}

function validateToken(token) {
  // WebSocket token validation
  try {
    return jwt.verify(token.replace('Bearer ', ''), process.env.JWT_SECRET);
  } catch {
    throw new Error('Invalid token');
  }
}
```

### Core Service Resolvers

```rust
// core/src/graphql/resolvers.rs
use async_graphql::{Context, Object, Result, Subscription, ID};
use futures_util::Stream;
use std::time::Duration;
use tokio_stream::StreamExt;

pub struct Query;
pub struct Mutation;
pub struct SubscriptionRoot;

#[Object]
impl Query {
    async fn resources(&self, ctx: &Context<'_>, filters: Option<ResourceFilters>) -> Result<Vec<Resource>> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        
        // Authorization check
        if !auth.can_read_resources() {
            return Err("Unauthorized".into());
        }
        
        let resources = if let Some(f) = filters {
            resource_service::get_resources_with_filters(db, f).await?
        } else {
            resource_service::get_all_resources(db).await?
        };
        
        Ok(resources)
    }
    
    async fn resource(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Resource>> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        
        let resource = resource_service::get_resource_by_id(db, &id).await?;
        
        if let Some(ref r) = resource {
            if !auth.can_read_resource(&r.subscription_id) {
                return Err("Unauthorized".into());
            }
        }
        
        Ok(resource)
    }
    
    async fn policies(&self, ctx: &Context<'_>, policy_type: Option<PolicyType>) -> Result<Vec<Policy>> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        
        if !auth.can_read_policies() {
            return Err("Unauthorized".into());
        }
        
        let policies = policy_service::get_policies_by_type(db, policy_type).await?;
        Ok(policies)
    }
    
    async fn compliance_status(&self, ctx: &Context<'_>, resource_id: ID) -> Result<ComplianceStatus> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        let cache = ctx.data::<CacheService>()?;
        
        // Check cache first
        let cache_key = format!("compliance_status:{}", resource_id);
        if let Some(cached) = cache.get(&cache_key).await? {
            return Ok(cached);
        }
        
        // Check authorization
        let resource = resource_service::get_resource_by_id(db, &resource_id).await?
            .ok_or("Resource not found")?;
            
        if !auth.can_read_resource(&resource.subscription_id) {
            return Err("Unauthorized".into());
        }
        
        // Calculate compliance status
        let status = compliance_service::calculate_status(db, &resource_id).await?;
        
        // Cache result for 5 minutes
        cache.set(&cache_key, &status, Duration::from_secs(300)).await?;
        
        Ok(status)
    }
}

#[Object]
impl Mutation {
    async fn create_resource(&self, ctx: &Context<'_>, input: CreateResourceInput) -> Result<Resource> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        let events = ctx.data::<EventStore>()?;
        
        if !auth.can_create_resources(&input.subscription_id) {
            return Err("Unauthorized".into());
        }
        
        // Create resource
        let resource = resource_service::create_resource(db, input).await?;
        
        // Emit event
        let event = ResourceCreatedEvent {
            resource_id: resource.id.clone(),
            subscription_id: resource.subscription_id.clone(),
            created_by: auth.user_id().to_string(),
        };
        events.append_event("resource_created", &event).await?;
        
        Ok(resource)
    }
    
    async fn execute_action(&self, ctx: &Context<'_>, input: ExecuteActionInput) -> Result<ActionExecution> {
        let db = ctx.data::<DatabasePool>()?;
        let auth = ctx.data::<AuthContext>()?;
        let action_service = ctx.data::<ActionService>()?;
        
        // Authorization and validation
        let resource = resource_service::get_resource_by_id(db, &input.resource_id).await?
            .ok_or("Resource not found")?;
            
        if !auth.can_execute_actions(&resource.subscription_id) {
            return Err("Unauthorized".into());
        }
        
        // Execute action
        let execution = if input.dry_run {
            action_service.dry_run_action(&input).await?
        } else {
            action_service.execute_action(&input).await?
        };
        
        Ok(execution)
    }
}

#[Subscription]
impl SubscriptionRoot {
    async fn resource_updated(&self, ctx: &Context<'_>, id: ID) -> Result<impl Stream<Item = Result<Resource>>> {
        let auth = ctx.data::<AuthContext>()?;
        let event_stream = ctx.data::<EventStream>()?;
        
        // Check initial authorization
        let db = ctx.data::<DatabasePool>()?;
        let resource = resource_service::get_resource_by_id(db, &id).await?
            .ok_or("Resource not found")?;
            
        if !auth.can_read_resource(&resource.subscription_id) {
            return Err("Unauthorized".into());
        }
        
        let stream = event_stream
            .filter_map(move |event| {
                match event {
                    Event::ResourceUpdated { resource_id, resource } if resource_id == id => {
                        Some(Ok(resource))
                    }
                    _ => None
                }
            })
            .take_while(|_| {
                // Re-check authorization periodically
                future::ready(true) // Simplified - should re-check auth
            });
            
        Ok(stream)
    }
    
    async fn compliance_changed(&self, ctx: &Context<'_>, resource_id: ID) -> Result<impl Stream<Item = Result<ComplianceStatus>>> {
        let auth = ctx.data::<AuthContext>()?;
        let compliance_stream = ctx.data::<ComplianceStream>()?;
        
        let stream = compliance_stream
            .filter_map(move |status| {
                if status.resource_id == resource_id {
                    Some(Ok(status))
                } else {
                    None
                }
            });
            
        Ok(stream)
    }
}

// Entity resolvers
#[Object]
impl Resource {
    // Fields resolved by this service
    async fn id(&self) -> &str { &self.id }
    async fn name(&self) -> &str { &self.name }
    async fn resource_type(&self) -> ResourceType { self.resource_type }
    
    // Computed fields
    async fn evaluations(&self, ctx: &Context<'_>) -> Result<Vec<PolicyEvaluation>> {
        let db = ctx.data::<DatabasePool>()?;
        policy_service::get_evaluations_for_resource(db, &self.id).await
    }
    
    // Federation: These fields are resolved by other services
    // The federation gateway will make additional requests to AI and Azure services
}

#[Object]
impl Policy {
    async fn id(&self) -> &str { &self.id }
    async fn name(&self) -> &str { &self.name }
    async fn policy_type(&self) -> PolicyType { self.policy_type }
    
    async fn affected_resources(&self, ctx: &Context<'_>) -> Result<Vec<Resource>> {
        let db = ctx.data::<DatabasePool>()?;
        resource_service::get_resources_affected_by_policy(db, &self.id).await
    }
}

// Context types
pub struct AuthContext {
    user_id: String,
    tenant_id: String,
    permissions: Vec<String>,
}

impl AuthContext {
    pub fn can_read_resources(&self) -> bool {
        self.permissions.contains(&"resources:read".to_string())
    }
    
    pub fn can_read_resource(&self, subscription_id: &str) -> bool {
        self.can_read_resources() && 
        (self.permissions.contains(&format!("subscription:{}:read", subscription_id)) ||
         self.permissions.contains(&"subscription:*:read".to_string()))
    }
    
    pub fn can_create_resources(&self, subscription_id: &str) -> bool {
        self.permissions.contains(&"resources:create".to_string()) &&
        (self.permissions.contains(&format!("subscription:{}:write", subscription_id)) ||
         self.permissions.contains(&"subscription:*:write".to_string()))
    }
    
    pub fn can_execute_actions(&self, subscription_id: &str) -> bool {
        self.permissions.contains(&"actions:execute".to_string()) &&
        (self.permissions.contains(&format!("subscription:{}:write", subscription_id)) ||
         self.permissions.contains(&"subscription:*:write".to_string()))
    }
    
    pub fn user_id(&self) -> &str { &self.user_id }
}
```

## Subscriptions and Real-time

### Subscription Infrastructure

```javascript
// graphql/subscriptions.js
const { PubSub } = require('graphql-subscriptions');
const { RedisPubSub } = require('graphql-redis-subscriptions');
const Redis = require('ioredis');

// Redis-based pub/sub for distributed subscriptions
const pubsub = new RedisPubSub({
  publisher: new Redis({
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    retryDelayOnFailover: 100,
    enableOfflineQueue: false,
    maxRetriesPerRequest: 3,
  }),
  subscriber: new Redis({
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    retryDelayOnFailover: 100,
    enableOfflineQueue: false,
    maxRetriesPerRequest: 3,
  }),
});

// Subscription channels
const CHANNELS = {
  RESOURCE_UPDATED: 'RESOURCE_UPDATED',
  COMPLIANCE_CHANGED: 'COMPLIANCE_CHANGED',
  ACTION_STATUS_CHANGED: 'ACTION_STATUS_CHANGED',
  AZURE_RESOURCE_UPDATED: 'AZURE_RESOURCE_UPDATED',
  INSIGHTS_UPDATED: 'INSIGHTS_UPDATED',
  TRAINING_PROGRESS: 'TRAINING_PROGRESS',
};

// Subscription manager
class SubscriptionManager {
  constructor() {
    this.activeSubscriptions = new Map();
    this.subscriptionFilters = new Map();
  }

  // Subscribe with filtering
  async subscribe(channel, filter = null, context) {
    const subscriptionId = this.generateSubscriptionId();
    
    // Store subscription context
    this.activeSubscriptions.set(subscriptionId, {
      channel,
      filter,
      context,
      createdAt: Date.now()
    });
    
    if (filter) {
      this.subscriptionFilters.set(subscriptionId, filter);
    }
    
    // Create filtered async iterator
    const asyncIterator = pubsub.asyncIterator(channel);
    
    return {
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          const { value, done } = await asyncIterator.next();
          
          if (done) {
            this.activeSubscriptions.delete(subscriptionId);
            this.subscriptionFilters.delete(subscriptionId);
            return { done: true };
          }
          
          // Apply filter if present
          if (filter && !this.matchesFilter(value, filter)) {
            return this.next(); // Skip this value
          }
          
          return { value, done: false };
        }
      })
    };
  }

  // Publish with fanout
  async publish(channel, payload) {
    // Add metadata
    const enrichedPayload = {
      ...payload,
      timestamp: Date.now(),
      channel
    };
    
    await pubsub.publish(channel, enrichedPayload);
    
    // Log subscription activity
    console.log(`Published to ${channel}:`, {
      subscriberCount: await this.getSubscriberCount(channel),
      payload: JSON.stringify(enrichedPayload).substring(0, 200)
    });
  }

  // Clean up expired subscriptions
  cleanup() {
    const now = Date.now();
    const maxAge = 30 * 60 * 1000; // 30 minutes
    
    for (const [id, subscription] of this.activeSubscriptions) {
      if (now - subscription.createdAt > maxAge) {
        this.activeSubscriptions.delete(id);
        this.subscriptionFilters.delete(id);
      }
    }
  }

  generateSubscriptionId() {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  matchesFilter(value, filter) {
    // Implementation depends on filter structure
    if (filter.resourceId && value.resourceId !== filter.resourceId) {
      return false;
    }
    
    if (filter.subscriptionId && value.subscriptionId !== filter.subscriptionId) {
      return false;
    }
    
    return true;
  }

  async getSubscriberCount(channel) {
    // Get subscriber count from Redis
    const subscribers = await pubsub.subscriber.pubsub('NUMSUB', channel);
    return subscribers[1] || 0;
  }
}

const subscriptionManager = new SubscriptionManager();

// Clean up periodically
setInterval(() => subscriptionManager.cleanup(), 5 * 60 * 1000);

module.exports = {
  pubsub,
  subscriptionManager,
  CHANNELS
};
```

### Real-time Event Processing

```rust
// core/src/events/subscription_handler.rs
use async_graphql::*;
use futures_util::StreamExt;
use redis::aio::Connection;
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdateEvent {
    pub resource_id: String,
    pub subscription_id: String,
    pub change_type: String,
    pub timestamp: i64,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceChangeEvent {
    pub resource_id: String,
    pub policy_id: String,
    pub old_status: String,
    pub new_status: String,
    pub timestamp: i64,
}

pub struct EventPublisher {
    redis_client: redis::Client,
}

impl EventPublisher {
    pub fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)?;
        Ok(Self { redis_client: client })
    }
    
    pub async fn publish_resource_update(&self, event: ResourceUpdateEvent) -> Result<()> {
        let mut conn = self.redis_client.get_async_connection().await?;
        let payload = serde_json::to_string(&event)?;
        
        redis::cmd("PUBLISH")
            .arg("RESOURCE_UPDATED")
            .arg(&payload)
            .query_async(&mut conn)
            .await?;
            
        // Also publish to specific resource channel for filtered subscriptions
        redis::cmd("PUBLISH")
            .arg(format!("RESOURCE_UPDATED:{}", event.resource_id))
            .arg(&payload)
            .query_async(&mut conn)
            .await?;
            
        Ok(())
    }
    
    pub async fn publish_compliance_change(&self, event: ComplianceChangeEvent) -> Result<()> {
        let mut conn = self.redis_client.get_async_connection().await?;
        let payload = serde_json::to_string(&event)?;
        
        redis::cmd("PUBLISH")
            .arg("COMPLIANCE_CHANGED")
            .arg(&payload)
            .query_async(&mut conn)
            .await?;
            
        // Resource-specific channel
        redis::cmd("PUBLISH")
            .arg(format!("COMPLIANCE_CHANGED:{}", event.resource_id))
            .arg(&payload)
            .query_async(&mut conn)
            .await?;
            
        Ok(())
    }
}

// Subscription stream handlers
pub struct SubscriptionStreams {
    event_publisher: EventPublisher,
}

impl SubscriptionStreams {
    pub fn new(redis_url: &str) -> Result<Self> {
        Ok(Self {
            event_publisher: EventPublisher::new(redis_url)?,
        })
    }
    
    // Resource monitoring task
    pub async fn start_resource_monitoring(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Check for resource changes
            if let Ok(changes) = self.detect_resource_changes().await {
                for change in changes {
                    let event = ResourceUpdateEvent {
                        resource_id: change.id,
                        subscription_id: change.subscription_id,
                        change_type: change.change_type,
                        timestamp: chrono::Utc::now().timestamp(),
                        data: change.data,
                    };
                    
                    if let Err(e) = self.event_publisher.publish_resource_update(event).await {
                        eprintln!("Failed to publish resource update: {}", e);
                    }
                }
            }
        }
    }
    
    // Compliance monitoring task
    pub async fn start_compliance_monitoring(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            if let Ok(changes) = self.detect_compliance_changes().await {
                for change in changes {
                    let event = ComplianceChangeEvent {
                        resource_id: change.resource_id,
                        policy_id: change.policy_id,
                        old_status: change.old_status,
                        new_status: change.new_status,
                        timestamp: chrono::Utc::now().timestamp(),
                    };
                    
                    if let Err(e) = self.event_publisher.publish_compliance_change(event).await {
                        eprintln!("Failed to publish compliance change: {}", e);
                    }
                }
            }
        }
    }
    
    async fn detect_resource_changes(&self) -> Result<Vec<ResourceChange>> {
        // Implementation would query database for changes
        // Compare current state with previous snapshots
        // Return detected changes
        Ok(vec![])
    }
    
    async fn detect_compliance_changes(&self) -> Result<Vec<ComplianceChange>> {
        // Implementation would re-evaluate policies
        // Compare current compliance status with previous status
        // Return compliance changes
        Ok(vec![])
    }
}

#[derive(Debug)]
struct ResourceChange {
    id: String,
    subscription_id: String,
    change_type: String,
    data: serde_json::Value,
}

#[derive(Debug)]
struct ComplianceChange {
    resource_id: String,
    policy_id: String,
    old_status: String,
    new_status: String,
}
```

## Query Planning

### Intelligent Query Distribution

```javascript
// graphql/query-planner.js
const { buildOperationContext } = require('@apollo/gateway');

class PolicyCortexQueryPlanner {
  constructor(gateway) {
    this.gateway = gateway;
    this.queryCache = new Map();
    this.performanceMetrics = new Map();
  }

  async planQuery(query, variables, context) {
    const queryHash = this.hashQuery(query, variables);
    
    // Check cache for repeated queries
    if (this.queryCache.has(queryHash)) {
      const cachedPlan = this.queryCache.get(queryHash);
      if (this.isCacheValid(cachedPlan)) {
        return this.optimizeCachedPlan(cachedPlan, context);
      }
    }
    
    // Analyze query complexity
    const complexity = this.analyzeComplexity(query);
    if (complexity.totalComplexity > 1000) {
      throw new Error(`Query too complex: ${complexity.totalComplexity}`);
    }
    
    // Plan query execution
    const plan = await this.createExecutionPlan(query, variables, context);
    
    // Cache plan
    this.queryCache.set(queryHash, {
      plan,
      createdAt: Date.now(),
      complexity,
      executionCount: 0
    });
    
    return plan;
  }

  analyzeComplexity(query) {
    const ast = parse(query);
    let totalComplexity = 0;
    let maxDepth = 0;
    let serviceQueries = new Set();

    visit(ast, {
      Field: {
        enter(node, key, parent, path, ancestors) {
          const depth = ancestors.filter(a => a.kind === 'Field').length;
          maxDepth = Math.max(maxDepth, depth);
          
          // Estimate field complexity
          const fieldComplexity = this.getFieldComplexity(node.name.value);
          totalComplexity += fieldComplexity;
          
          // Track which services will be queried
          const service = this.getServiceForField(node.name.value);
          if (service) {
            serviceQueries.add(service);
          }
        }
      }
    });

    return {
      totalComplexity,
      maxDepth,
      serviceCount: serviceQueries.size,
      services: Array.from(serviceQueries)
    };
  }

  async createExecutionPlan(query, variables, context) {
    // Parse query to understand data requirements
    const requirements = this.analyzeDataRequirements(query);
    
    // Optimize for common patterns
    const optimizations = this.identifyOptimizations(requirements);
    
    // Create execution plan
    const plan = {
      phases: [],
      parallelization: [],
      caching: [],
      optimizations
    };

    // Phase 1: Core data fetching
    if (requirements.needsResources) {
      plan.phases.push({
        service: 'core',
        priority: 1,
        fields: requirements.resourceFields,
        estimated_time: 50 // ms
      });
    }

    // Phase 2: Azure data (can be parallel with AI if no dependencies)
    if (requirements.needsAzureData) {
      const phase = {
        service: 'azure',
        priority: requirements.dependsOnCore ? 2 : 1,
        fields: requirements.azureFields,
        estimated_time: 200 // ms
      };
      
      if (!requirements.dependsOnCore) {
        plan.parallelization.push(['core', 'azure']);
      }
      
      plan.phases.push(phase);
    }

    // Phase 3: AI analysis (usually depends on core data)
    if (requirements.needsAI) {
      plan.phases.push({
        service: 'ai',
        priority: 3,
        fields: requirements.aiFields,
        estimated_time: 500, // ms
        dependencies: requirements.aiDependencies
      });
    }

    // Add caching strategies
    plan.caching = this.planCaching(requirements);

    return plan;
  }

  analyzeDataRequirements(query) {
    const ast = parse(query);
    const requirements = {
      needsResources: false,
      needsAzureData: false,
      needsAI: false,
      resourceFields: [],
      azureFields: [],
      aiFields: [],
      dependsOnCore: false,
      aiDependencies: []
    };

    visit(ast, {
      Field: {
        enter(node) {
          const fieldName = node.name.value;
          
          // Categorize fields by service
          if (this.isCoreField(fieldName)) {
            requirements.needsResources = true;
            requirements.resourceFields.push(fieldName);
          } else if (this.isAzureField(fieldName)) {
            requirements.needsAzureData = true;
            requirements.azureFields.push(fieldName);
            if (this.requiresCoreData(fieldName)) {
              requirements.dependsOnCore = true;
            }
          } else if (this.isAIField(fieldName)) {
            requirements.needsAI = true;
            requirements.aiFields.push(fieldName);
            
            // AI fields typically depend on core resource data
            if (this.requiresCoreData(fieldName)) {
              requirements.aiDependencies.push('core');
            }
          }
        }
      }
    });

    return requirements;
  }

  identifyOptimizations(requirements) {
    const optimizations = [];

    // Batch similar operations
    if (requirements.resourceFields.length > 5) {
      optimizations.push({
        type: 'BATCH_RESOURCES',
        description: 'Batch resource queries to reduce database roundtrips'
      });
    }

    // Prefetch commonly requested data
    if (requirements.needsAI && requirements.aiFields.includes('insights')) {
      optimizations.push({
        type: 'PREFETCH_INSIGHTS',
        description: 'Prefetch resource insights for better performance'
      });
    }

    // Use DataLoader for N+1 prevention
    if (this.hasNPlusOnePattern(requirements)) {
      optimizations.push({
        type: 'DATALOADER',
        description: 'Use DataLoader to prevent N+1 queries'
      });
    }

    return optimizations;
  }

  planCaching(requirements) {
    const caching = [];

    // Cache resource data (changes infrequently)
    if (requirements.needsResources) {
      caching.push({
        key: 'resources',
        ttl: 300, // 5 minutes
        invalidateOn: ['resource_updated']
      });
    }

    // Cache AI insights (expensive to compute)
    if (requirements.needsAI) {
      caching.push({
        key: 'ai_insights',
        ttl: 1800, // 30 minutes
        invalidateOn: ['resource_updated', 'policy_updated']
      });
    }

    // Cache Azure metrics (updates frequently but expensive)
    if (requirements.needsAzureData) {
      caching.push({
        key: 'azure_metrics',
        ttl: 60, // 1 minute
        invalidateOn: []
      });
    }

    return caching;
  }

  // Helper methods for field categorization
  isCoreField(fieldName) {
    const coreFields = [
      'resources', 'resource', 'policies', 'policy',
      'complianceStatus', 'evaluations', 'id', 'name', 'type'
    ];
    return coreFields.includes(fieldName);
  }

  isAzureField(fieldName) {
    const azureFields = [
      'liveData', 'metrics', 'azureMetrics', 'costAnalysis',
      'securityAlerts', 'diagnostics'
    ];
    return azureFields.includes(fieldName);
  }

  isAIField(fieldName) {
    const aiFields = [
      'insights', 'predictions', 'recommendations',
      'generatePolicy', 'analyzeCompliance', 'conversationalQuery'
    ];
    return aiFields.includes(fieldName);
  }

  requiresCoreData(fieldName) {
    const dependentFields = [
      'liveData', 'insights', 'predictions', 'recommendations'
    ];
    return dependentFields.includes(fieldName);
  }

  hasNPlusOnePattern(requirements) {
    // Detect patterns that might cause N+1 queries
    return requirements.resourceFields.includes('resources') &&
           (requirements.aiFields.length > 0 || requirements.azureFields.length > 0);
  }

  getFieldComplexity(fieldName) {
    const complexityMap = {
      // Simple fields
      'id': 1, 'name': 1, 'type': 1,
      
      // Medium complexity
      'resources': 10, 'policies': 10,
      'complianceStatus': 15,
      
      // High complexity
      'insights': 50, 'predictions': 40,
      'generatePolicy': 100, 'conversationalQuery': 80,
      
      // Very high complexity
      'costAnalysis': 200, 'analyzeCompliance': 150
    };
    
    return complexityMap[fieldName] || 5;
  }

  getServiceForField(fieldName) {
    if (this.isCoreField(fieldName)) return 'core';
    if (this.isAzureField(fieldName)) return 'azure';
    if (this.isAIField(fieldName)) return 'ai';
    return null;
  }

  hashQuery(query, variables) {
    const crypto = require('crypto');
    const content = query + JSON.stringify(variables || {});
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  isCacheValid(cachedPlan) {
    const maxAge = 60 * 60 * 1000; // 1 hour
    return Date.now() - cachedPlan.createdAt < maxAge;
  }

  optimizeCachedPlan(cachedPlan, context) {
    // Apply context-specific optimizations to cached plan
    const optimizedPlan = { ...cachedPlan.plan };
    
    // Update execution count for metrics
    cachedPlan.executionCount++;
    
    // Adjust priorities based on current load
    if (context.highPriority) {
      optimizedPlan.phases.forEach(phase => {
        phase.priority += 10;
      });
    }
    
    return optimizedPlan;
  }
}

module.exports = PolicyCortexQueryPlanner;
```

## Caching Strategy

### Multi-level Caching

```javascript
// graphql/cache-manager.js
const Redis = require('ioredis');
const { LRUCache } = require('lru-cache');

class FederatedCacheManager {
  constructor() {
    // L1 Cache: In-memory LRU cache
    this.l1Cache = new LRUCache({
      max: 10000,
      maxAge: 1000 * 60 * 5, // 5 minutes
      updateAgeOnGet: true,
    });
    
    // L2 Cache: Redis distributed cache
    this.l2Cache = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      db: 1, // Use different DB for caching
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
    });
    
    // Cache invalidation patterns
    this.invalidationPatterns = new Map();
    this.setupInvalidationPatterns();
    
    // Cache statistics
    this.stats = {
      l1Hits: 0,
      l1Misses: 0,
      l2Hits: 0,
      l2Misses: 0,
      invalidations: 0
    };
  }

  setupInvalidationPatterns() {
    // Resource updates invalidate related caches
    this.invalidationPatterns.set('resource_updated', [
      'resource:*',
      'compliance:*',
      'insights:*'
    ]);
    
    // Policy changes invalidate compliance data
    this.invalidationPatterns.set('policy_updated', [
      'compliance:*',
      'policy_evaluation:*'
    ]);
    
    // Azure data updates
    this.invalidationPatterns.set('azure_data_updated', [
      'azure_metrics:*',
      'azure_resource:*'
    ]);
  }

  async get(key, options = {}) {
    const startTime = Date.now();
    
    try {
      // Try L1 cache first
      let value = this.l1Cache.get(key);
      if (value !== undefined) {
        this.stats.l1Hits++;
        return this.deserializeValue(value);
      }
      
      this.stats.l1Misses++;
      
      // Try L2 cache (Redis)
      const redisValue = await this.l2Cache.get(key);
      if (redisValue !== null) {
        this.stats.l2Hits++;
        const deserializedValue = this.deserializeValue(redisValue);
        
        // Populate L1 cache
        this.l1Cache.set(key, redisValue);
        
        return deserializedValue;
      }
      
      this.stats.l2Misses++;
      return null;
      
    } catch (error) {
      console.error(`Cache get error for key ${key}:`, error);
      return null;
    } finally {
      // Record cache performance metrics
      this.recordMetrics('get', Date.now() - startTime);
    }
  }

  async set(key, value, ttl = 300) {
    const startTime = Date.now();
    
    try {
      const serializedValue = this.serializeValue(value);
      
      // Set in L1 cache
      this.l1Cache.set(key, serializedValue, ttl * 1000);
      
      // Set in L2 cache with TTL
      await this.l2Cache.setex(key, ttl, serializedValue);
      
      // Set cache metadata for invalidation
      await this.setCacheMetadata(key, {
        createdAt: Date.now(),
        ttl,
        tags: this.extractCacheTags(key)
      });
      
    } catch (error) {
      console.error(`Cache set error for key ${key}:`, error);
    } finally {
      this.recordMetrics('set', Date.now() - startTime);
    }
  }

  async invalidate(pattern) {
    const startTime = Date.now();
    
    try {
      this.stats.invalidations++;
      
      // Handle wildcard patterns
      if (pattern.includes('*')) {
        await this.invalidatePattern(pattern);
      } else {
        await this.invalidateSingle(pattern);
      }
      
      console.log(`Cache invalidated: ${pattern}`);
      
    } catch (error) {
      console.error(`Cache invalidation error for pattern ${pattern}:`, error);
    } finally {
      this.recordMetrics('invalidate', Date.now() - startTime);
    }
  }

  async invalidatePattern(pattern) {
    // Get all keys matching pattern
    const keys = await this.l2Cache.keys(pattern);
    
    if (keys.length > 0) {
      // Remove from L2 cache
      await this.l2Cache.del(...keys);
      
      // Remove from L1 cache
      keys.forEach(key => {
        this.l1Cache.delete(key);
      });
    }
    
    return keys.length;
  }

  async invalidateSingle(key) {
    // Remove from both caches
    this.l1Cache.delete(key);
    await this.l2Cache.del(key);
  }

  async invalidateByEvent(eventType) {
    const patterns = this.invalidationPatterns.get(eventType);
    if (patterns) {
      for (const pattern of patterns) {
        await this.invalidate(pattern);
      }
    }
  }

  // GraphQL-specific caching
  async cacheQuery(query, variables, result, ttl = 300) {
    const key = this.generateQueryKey(query, variables);
    await this.set(key, result, ttl);
  }

  async getCachedQuery(query, variables) {
    const key = this.generateQueryKey(query, variables);
    return await this.get(key);
  }

  generateQueryKey(query, variables) {
    const crypto = require('crypto');
    const content = query + JSON.stringify(variables || {});
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    return `query:${hash}`;
  }

  // Field-level caching for federation
  async cacheFieldResult(typename, id, fieldName, result, ttl = 300) {
    const key = `field:${typename}:${id}:${fieldName}`;
    await this.set(key, result, ttl);
  }

  async getCachedFieldResult(typename, id, fieldName) {
    const key = `field:${typename}:${id}:${fieldName}`;
    return await this.get(key);
  }

  // Entity caching
  async cacheEntity(typename, id, entity, ttl = 300) {
    const key = `entity:${typename}:${id}`;
    await this.set(key, entity, ttl);
  }

  async getCachedEntity(typename, id) {
    const key = `entity:${typename}:${id}`;
    return await this.get(key);
  }

  // Batch operations for DataLoader
  async batchGet(keys) {
    const results = await Promise.all(
      keys.map(key => this.get(key))
    );
    return results;
  }

  async batchSet(keyValuePairs, ttl = 300) {
    await Promise.all(
      keyValuePairs.map(({ key, value }) => this.set(key, value, ttl))
    );
  }

  // Cache warming
  async warmCache(warmingPlan) {
    console.log('Starting cache warming...');
    
    for (const item of warmingPlan) {
      try {
        switch (item.type) {
          case 'query':
            await this.warmQuery(item);
            break;
          case 'entity':
            await this.warmEntity(item);
            break;
          case 'field':
            await this.warmField(item);
            break;
        }
      } catch (error) {
        console.error(`Cache warming failed for item:`, item, error);
      }
    }
    
    console.log('Cache warming completed');
  }

  async warmQuery(item) {
    // Execute query and cache result
    const result = await this.executeQuery(item.query, item.variables);
    await this.cacheQuery(item.query, item.variables, result, item.ttl);
  }

  // Helper methods
  serializeValue(value) {
    return JSON.stringify(value);
  }

  deserializeValue(value) {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }

  extractCacheTags(key) {
    // Extract tags from cache key for better invalidation
    const parts = key.split(':');
    return parts.slice(0, -1); // Remove the last part (usually ID or hash)
  }

  async setCacheMetadata(key, metadata) {
    const metaKey = `meta:${key}`;
    await this.l2Cache.setex(metaKey, metadata.ttl, JSON.stringify(metadata));
  }

  recordMetrics(operation, duration) {
    // Record metrics for monitoring
    console.log(`Cache ${operation} took ${duration}ms`);
  }

  getStats() {
    return {
      ...this.stats,
      l1Size: this.l1Cache.size,
      l1HitRate: this.stats.l1Hits / (this.stats.l1Hits + this.stats.l1Misses),
      l2HitRate: this.stats.l2Hits / (this.stats.l2Hits + this.stats.l2Misses),
    };
  }

  async clearAll() {
    this.l1Cache.clear();
    await this.l2Cache.flushdb();
    console.log('All caches cleared');
  }
}

module.exports = FederatedCacheManager;
```

## Error Handling

### Comprehensive Error Management

```javascript
// graphql/error-handler.js
const { ApolloError, UserInputError, AuthenticationError, ForbiddenError } = require('apollo-server-express');

class GraphQLErrorHandler {
  constructor() {
    this.errorCodes = {
      // Authentication & Authorization
      'UNAUTHENTICATED': 'User authentication required',
      'FORBIDDEN': 'Insufficient permissions',
      'INVALID_TOKEN': 'Authentication token is invalid or expired',
      
      // Input Validation
      'INVALID_INPUT': 'Invalid input parameters',
      'VALIDATION_ERROR': 'Input validation failed',
      'MISSING_REQUIRED_FIELD': 'Required field is missing',
      
      // Business Logic
      'RESOURCE_NOT_FOUND': 'Requested resource does not exist',
      'POLICY_VIOLATION': 'Operation violates policy constraints',
      'COMPLIANCE_ERROR': 'Compliance check failed',
      
      // External Services
      'AZURE_API_ERROR': 'Azure service error',
      'AI_SERVICE_ERROR': 'AI service temporarily unavailable',
      'DATABASE_ERROR': 'Database operation failed',
      'CACHE_ERROR': 'Cache service error',
      
      // Rate Limiting & Capacity
      'RATE_LIMIT_EXCEEDED': 'Rate limit exceeded',
      'QUERY_TOO_COMPLEX': 'Query complexity exceeds limits',
      'TIMEOUT_ERROR': 'Operation timed out',
      
      // System Errors
      'INTERNAL_ERROR': 'Internal server error',
      'SERVICE_UNAVAILABLE': 'Service temporarily unavailable'
    };
  }

  formatError(error, context) {
    // Log error with correlation ID
    const correlationId = context?.correlationId || 'unknown';
    console.error(`[${correlationId}] GraphQL Error:`, {
      message: error.message,
      code: error.extensions?.code,
      path: error.path,
      locations: error.locations,
      source: error.source?.name,
      originalError: error.originalError?.message
    });

    // Determine error type and create appropriate response
    const formattedError = this.categorizeError(error, context);
    
    // Add correlation ID to response
    formattedError.extensions = {
      ...formattedError.extensions,
      correlationId
    };

    // In production, sanitize internal errors
    if (process.env.NODE_ENV === 'production') {
      return this.sanitizeError(formattedError);
    }

    return formattedError;
  }

  categorizeError(error, context) {
    // Handle known error types
    if (error instanceof AuthenticationError) {
      return this.createError('UNAUTHENTICATED', error.message);
    }
    
    if (error instanceof ForbiddenError) {
      return this.createError('FORBIDDEN', error.message);
    }
    
    if (error instanceof UserInputError) {
      return this.createError('INVALID_INPUT', error.message, error.extensions?.validationErrors);
    }

    // Handle service-specific errors
    if (error.originalError) {
      const original = error.originalError;
      
      if (original.name === 'AzureServiceError') {
        return this.createError('AZURE_API_ERROR', 'Azure service temporarily unavailable');
      }
      
      if (original.name === 'DatabaseError') {
        return this.createError('DATABASE_ERROR', 'Database operation failed');
      }
      
      if (original.name === 'TimeoutError') {
        return this.createError('TIMEOUT_ERROR', 'Operation timed out');
      }
    }

    // Handle GraphQL execution errors
    if (error.extensions?.code) {
      const code = error.extensions.code;
      
      if (this.errorCodes[code]) {
        return this.createError(code, this.errorCodes[code]);
      }
    }

    // Default to internal error
    return this.createError('INTERNAL_ERROR', 'An unexpected error occurred');
  }

  createError(code, message, extensions = {}) {
    return new ApolloError(message, code, {
      ...extensions,
      timestamp: new Date().toISOString()
    });
  }

  sanitizeError(error) {
    // Remove sensitive information in production
    const sanitized = { ...error };
    
    if (error.extensions?.code === 'INTERNAL_ERROR') {
      sanitized.message = 'Internal server error';
      delete sanitized.extensions.exception;
      delete sanitized.extensions.stacktrace;
    }
    
    return sanitized;
  }

  // Validation helpers
  validateRequiredFields(input, requiredFields) {
    const missing = requiredFields.filter(field => !input[field]);
    if (missing.length > 0) {
      throw new UserInputError('Missing required fields', {
        validationErrors: missing.map(field => ({
          field,
          message: `${field} is required`
        }))
      });
    }
  }

  validateResourceAccess(user, resource) {
    if (!user) {
      throw new AuthenticationError('Authentication required');
    }
    
    if (!this.canAccessResource(user, resource)) {
      throw new ForbiddenError('Insufficient permissions to access resource');
    }
  }

  canAccessResource(user, resource) {
    // Check if user has permission to access resource
    return user.permissions.includes(`subscription:${resource.subscriptionId}:read`) ||
           user.permissions.includes('subscription:*:read');
  }

  // Error recovery strategies
  async withErrorRecovery(operation, fallbackValue = null, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (this.isRetryableError(error) && attempt < maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        break;
      }
    }
    
    // If all retries failed, decide whether to return fallback or throw
    if (this.canUseFallback(lastError)) {
      console.warn('Using fallback value after error:', lastError.message);
      return fallbackValue;
    }
    
    throw lastError;
  }

  isRetryableError(error) {
    const retryableCodes = [
      'TIMEOUT_ERROR',
      'AZURE_API_ERROR',
      'DATABASE_ERROR',
      'SERVICE_UNAVAILABLE'
    ];
    
    return retryableCodes.includes(error.extensions?.code);
  }

  canUseFallback(error) {
    const fallbackableCodes = [
      'AZURE_API_ERROR',
      'AI_SERVICE_ERROR',
      'CACHE_ERROR'
    ];
    
    return fallbackableCodes.includes(error.extensions?.code);
  }

  // Circuit breaker for external services
  createCircuitBreaker(serviceName, options = {}) {
    const breaker = {
      name: serviceName,
      state: 'CLOSED', // CLOSED, OPEN, HALF_OPEN
      failureCount: 0,
      lastFailureTime: null,
      options: {
        failureThreshold: options.failureThreshold || 5,
        resetTimeout: options.resetTimeout || 60000, // 1 minute
        monitoringWindow: options.monitoringWindow || 300000 // 5 minutes
      }
    };

    return {
      async execute(operation) {
        if (breaker.state === 'OPEN') {
          if (Date.now() - breaker.lastFailureTime > breaker.options.resetTimeout) {
            breaker.state = 'HALF_OPEN';
          } else {
            throw new ApolloError(
              `Service ${serviceName} is currently unavailable`,
              'SERVICE_UNAVAILABLE'
            );
          }
        }

        try {
          const result = await operation();
          
          if (breaker.state === 'HALF_OPEN') {
            breaker.state = 'CLOSED';
            breaker.failureCount = 0;
          }
          
          return result;
        } catch (error) {
          breaker.failureCount++;
          breaker.lastFailureTime = Date.now();
          
          if (breaker.failureCount >= breaker.options.failureThreshold) {
            breaker.state = 'OPEN';
          }
          
          throw error;
        }
      },
      
      getState: () => breaker.state,
      getFailureCount: () => breaker.failureCount
    };
  }
}

module.exports = GraphQLErrorHandler;
```

## Performance Optimization

### Query Optimization Strategies

```javascript
// graphql/performance-optimizer.js
const DataLoader = require('dataloader');
const { PerformanceObserver, performance } = require('perf_hooks');

class GraphQLPerformanceOptimizer {
  constructor() {
    this.dataLoaders = new Map();
    this.performanceMetrics = [];
    this.queryCache = new Map();
    
    this.setupPerformanceObserver();
  }

  setupPerformanceObserver() {
    const obs = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name.startsWith('graphql-')) {
          this.recordMetric({
            name: entry.name,
            duration: entry.duration,
            startTime: entry.startTime,
            timestamp: Date.now()
          });
        }
      }
    });
    
    obs.observe({ entryTypes: ['measure'] });
  }

  // DataLoader factory for N+1 prevention
  createDataLoader(name, batchFunction, options = {}) {
    const loader = new DataLoader(batchFunction, {
      cache: true,
      maxBatchSize: options.maxBatchSize || 100,
      batchScheduleFn: callback => setTimeout(callback, options.batchDelay || 10),
      cacheKeyFn: key => typeof key === 'object' ? JSON.stringify(key) : key,
      ...options
    });

    this.dataLoaders.set(name, loader);
    return loader;
  }

  // Common DataLoaders for PolicyCortex
  createCommonDataLoaders() {
    // Resource loader
    const resourceLoader = this.createDataLoader(
      'resources',
      async (resourceIds) => {
        performance.mark('dataloader-resources-start');
        
        const resources = await this.batchLoadResources(resourceIds);
        
        performance.mark('dataloader-resources-end');
        performance.measure(
          'graphql-dataloader-resources',
          'dataloader-resources-start',
          'dataloader-resources-end'
        );
        
        return resources;
      }
    );

    // Policy evaluation loader
    const policyEvaluationLoader = this.createDataLoader(
      'policy-evaluations',
      async (resourceIds) => {
        return await this.batchLoadPolicyEvaluations(resourceIds);
      }
    );

    // Azure metrics loader
    const azureMetricsLoader = this.createDataLoader(
      'azure-metrics',
      async (resourceIds) => {
        return await this.batchLoadAzureMetrics(resourceIds);
      }
    );

    // AI insights loader
    const aiInsightsLoader = this.createDataLoader(
      'ai-insights',
      async (resourceIds) => {
        return await this.batchLoadAIInsights(resourceIds);
      }
    );

    return {
      resourceLoader,
      policyEvaluationLoader,
      azureMetricsLoader,
      aiInsightsLoader
    };
  }

  async batchLoadResources(resourceIds) {
    // Batch database query for resources
    const query = `
      SELECT * FROM resources 
      WHERE id = ANY($1)
      ORDER BY array_position($1, id)
    `;
    
    const result = await this.database.query(query, [resourceIds]);
    
    // Ensure order matches input order
    const resourceMap = new Map(result.rows.map(r => [r.id, r]));
    return resourceIds.map(id => resourceMap.get(id) || null);
  }

  async batchLoadPolicyEvaluations(resourceIds) {
    const query = `
      SELECT resource_id, json_agg(
        json_build_object(
          'id', id,
          'policy_id', policy_id,
          'status', status,
          'result', result,
          'evaluated_at', evaluated_at
        )
      ) as evaluations
      FROM policy_evaluations 
      WHERE resource_id = ANY($1)
      GROUP BY resource_id
    `;
    
    const result = await this.database.query(query, [resourceIds]);
    const evaluationMap = new Map(result.rows.map(r => [r.resource_id, r.evaluations]));
    
    return resourceIds.map(id => evaluationMap.get(id) || []);
  }

  async batchLoadAzureMetrics(resourceIds) {
    // Batch Azure API calls
    const batchSize = 20; // Azure API limits
    const batches = this.chunk(resourceIds, batchSize);
    const allMetrics = [];
    
    for (const batch of batches) {
      const batchMetrics = await this.azureClient.getBatchMetrics(batch);
      allMetrics.push(...batchMetrics);
    }
    
    const metricsMap = new Map(allMetrics.map(m => [m.resourceId, m]));
    return resourceIds.map(id => metricsMap.get(id) || null);
  }

  async batchLoadAIInsights(resourceIds) {
    // Batch AI service calls
    const batchInsights = await this.aiService.getBatchInsights(resourceIds);
    const insightsMap = new Map(batchInsights.map(i => [i.resourceId, i]));
    
    return resourceIds.map(id => insightsMap.get(id) || null);
  }

  // Query complexity analysis
  analyzeQueryComplexity(query, variables) {
    const ast = parse(query);
    let complexity = 0;
    let depth = 0;
    let fieldCount = 0;

    visit(ast, {
      Field: {
        enter(node, key, parent, path, ancestors) {
          fieldCount++;
          const currentDepth = ancestors.filter(a => a.kind === 'Field').length;
          depth = Math.max(depth, currentDepth);
          
          // Add complexity based on field type
          const fieldComplexity = this.getFieldComplexity(node.name.value);
          complexity += fieldComplexity;
          
          // Add complexity for list fields with arguments
          if (node.arguments && node.arguments.length > 0) {
            complexity += node.arguments.length * 2;
          }
        }
      }
    });

    return { complexity, depth, fieldCount };
  }

  // Query optimization suggestions
  optimizeQuery(query, variables) {
    const optimizations = [];
    const ast = parse(query);

    visit(ast, {
      Field: {
        enter(node) {
          const fieldName = node.name.value;
          
          // Suggest field selection optimization
          if (this.isExpensiveField(fieldName) && !this.hasFieldSelection(node)) {
            optimizations.push({
              type: 'FIELD_SELECTION',
              field: fieldName,
              suggestion: 'Consider selecting only needed subfields'
            });
          }
          
          // Suggest pagination for list fields
          if (this.isListField(fieldName) && !this.hasPagination(node)) {
            optimizations.push({
              type: 'PAGINATION',
              field: fieldName,
              suggestion: 'Consider adding pagination (first, after) to limit results'
            });
          }
          
          // Suggest caching for expensive operations
          if (this.isCacheableField(fieldName)) {
            optimizations.push({
              type: 'CACHING',
              field: fieldName,
              suggestion: 'This field can be cached for better performance'
            });
          }
        }
      }
    });

    return optimizations;
  }

  // Performance monitoring
  recordMetric(metric) {
    this.performanceMetrics.push(metric);
    
    // Keep only recent metrics
    const maxAge = 60 * 60 * 1000; // 1 hour
    const cutoff = Date.now() - maxAge;
    this.performanceMetrics = this.performanceMetrics.filter(m => m.timestamp > cutoff);
    
    // Log slow queries
    if (metric.duration > 1000) {
      console.warn(`Slow GraphQL operation: ${metric.name} took ${metric.duration}ms`);
    }
  }

  getPerformanceStats() {
    if (this.performanceMetrics.length === 0) {
      return { count: 0 };
    }

    const durations = this.performanceMetrics.map(m => m.duration);
    const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
    const sorted = durations.sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    
    return {
      count: this.performanceMetrics.length,
      average: Math.round(avg),
      p95: p95,
      p99: p99,
      min: Math.min(...durations),
      max: Math.max(...durations)
    };
  }

  // Query planning optimization
  optimizeQueryPlan(plan) {
    const optimizedPlan = { ...plan };
    
    // Parallelize independent operations
    optimizedPlan.parallelGroups = this.identifyParallelOperations(plan.phases);
    
    // Optimize field selection
    optimizedPlan.fieldOptimizations = this.optimizeFieldSelection(plan);
    
    // Add caching recommendations
    optimizedPlan.cachingStrategy = this.recommendCaching(plan);
    
    return optimizedPlan;
  }

  identifyParallelOperations(phases) {
    const parallelGroups = [];
    const dependencies = new Map();
    
    // Build dependency graph
    phases.forEach(phase => {
      dependencies.set(phase.service, phase.dependencies || []);
    });
    
    // Group phases that can run in parallel
    const remaining = new Set(phases.map(p => p.service));
    
    while (remaining.size > 0) {
      const ready = [];
      
      for (const service of remaining) {
        const deps = dependencies.get(service);
        if (deps.every(dep => !remaining.has(dep))) {
          ready.push(service);
        }
      }
      
      if (ready.length > 0) {
        parallelGroups.push(ready);
        ready.forEach(service => remaining.delete(service));
      } else {
        // Circular dependency or error
        break;
      }
    }
    
    return parallelGroups;
  }

  // Helper methods
  chunk(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  getFieldComplexity(fieldName) {
    const complexityMap = {
      'id': 1, 'name': 1, 'type': 1,
      'resources': 10, 'policies': 10,
      'complianceStatus': 15,
      'insights': 50, 'predictions': 40,
      'generatePolicy': 100, 'conversationalQuery': 80,
      'costAnalysis': 200, 'analyzeCompliance': 150
    };
    
    return complexityMap[fieldName] || 5;
  }

  isExpensiveField(fieldName) {
    const expensiveFields = [
      'insights', 'predictions', 'generatePolicy',
      'conversationalQuery', 'costAnalysis', 'analyzeCompliance'
    ];
    return expensiveFields.includes(fieldName);
  }

  isListField(fieldName) {
    const listFields = ['resources', 'policies', 'evaluations', 'recommendations'];
    return listFields.includes(fieldName);
  }

  isCacheableField(fieldName) {
    const cacheableFields = [
      'insights', 'predictions', 'azureMetrics',
      'complianceStatus', 'costAnalysis'
    ];
    return cacheableFields.includes(fieldName);
  }

  hasFieldSelection(node) {
    return node.selectionSet && node.selectionSet.selections.length > 0;
  }

  hasPagination(node) {
    const paginationArgs = ['first', 'last', 'after', 'before', 'offset', 'limit'];
    return node.arguments && node.arguments.some(arg => 
      paginationArgs.includes(arg.name.value)
    );
  }
}

module.exports = GraphQLPerformanceOptimizer;
```

## Testing Federation

### Comprehensive Testing Strategy

```javascript
// graphql/__tests__/federation.test.js
const { ApolloServer } = require('apollo-server-express');
const { ApolloGateway } = require('@apollo/gateway');
const { buildFederatedSchema } = require('@apollo/federation');
const gql = require('graphql-tag');

describe('GraphQL Federation', () => {
  let gateway;
  let server;
  let coreServer;
  let aiServer;
  let azureServer;

  beforeAll(async () => {
    // Start mock subgraph services
    coreServer = await startMockCoreService();
    aiServer = await startMockAIService();
    azureServer = await startMockAzureService();

    // Create gateway
    gateway = new ApolloGateway({
      serviceList: [
        { name: 'core', url: 'http://localhost:8080/graphql' },
        { name: 'ai', url: 'http://localhost:8081/graphql' },
        { name: 'azure', url: 'http://localhost:8082/graphql' }
      ]
    });

    // Create federated server
    server = new ApolloServer({
      gateway,
      subscriptions: false // Disable for testing
    });

    await server.start();
  });

  afterAll(async () => {
    await server.stop();
    await coreServer.stop();
    await aiServer.stop();
    await azureServer.stop();
  });

  describe('Schema Composition', () => {
    test('should compose schemas from all services', async () => {
      const { schema } = await gateway.load();
      expect(schema).toBeDefined();
      
      // Check that types from all services are present
      const typeMap = schema.getTypeMap();
      expect(typeMap.Resource).toBeDefined();
      expect(typeMap.ResourceInsights).toBeDefined();
      expect(typeMap.AzureResourceData).toBeDefined();
    });

    test('should handle entity resolution across services', async () => {
      const query = gql`
        query GetResourceWithInsights($id: ID!) {
          resource(id: $id) {
            id
            name
            type
            insights {
              securityScore
              recommendations {
                title
                priority
              }
            }
            liveData {
              metrics {
                name
                values {
                  timestamp
                  value
                }
              }
            }
          }
        }
      `;

      const result = await server.executeOperation({
        query,
        variables: { id: 'resource-123' }
      });

      expect(result.errors).toBeUndefined();
      expect(result.data.resource).toMatchObject({
        id: 'resource-123',
        name: expect.any(String),
        type: expect.any(String),
        insights: {
          securityScore: expect.any(Number),
          recommendations: expect.arrayContaining([
            expect.objectContaining({
              title: expect.any(String),
              priority: expect.any(String)
            })
          ])
        },
        liveData: {
          metrics: expect.arrayContaining([
            expect.objectContaining({
              name: expect.any(String),
              values: expect.any(Array)
            })
          ])
        }
      });
    });
  });

  describe('Query Planning', () => {
    test('should optimize query execution across services', async () => {
      const query = gql`
        query GetMultipleResourcesWithData {
          resources(filters: { resourceType: VIRTUAL_MACHINE }) {
            id
            name
            insights {
              securityScore
            }
            liveData {
              metrics {
                name
              }
            }
          }
        }
      `;

      const startTime = Date.now();
      const result = await server.executeOperation({ query });
      const duration = Date.now() - startTime;

      expect(result.errors).toBeUndefined();
      expect(result.data.resources).toBeDefined();
      expect(duration).toBeLessThan(2000); // Should complete within 2 seconds
    });

    test('should handle N+1 queries efficiently with DataLoader', async () => {
      const query = gql`
        query GetResourcesWithInsights {
          resources {
            id
            insights {
              securityScore
            }
          }
        }
      `;

      // Mock multiple resources
      const mockResources = Array.from({ length: 20 }, (_, i) => ({
        id: `resource-${i}`,
        name: `Resource ${i}`,
        type: 'VIRTUAL_MACHINE'
      }));

      // Set up mocks
      coreServer.setMockResponse('resources', mockResources);
      
      const result = await server.executeOperation({ query });
      
      expect(result.errors).toBeUndefined();
      expect(result.data.resources).toHaveLength(20);
      
      // Verify DataLoader batching - should make only one call to AI service
      expect(aiServer.getCallCount()).toBeLessThanOrEqual(2);
    });
  });

  describe('Error Handling', () => {
    test('should handle partial failures gracefully', async () => {
      const query = gql`
        query GetResourceWithFailingService($id: ID!) {
          resource(id: $id) {
            id
            name
            insights {
              securityScore
            }
          }
        }
      `;

      // Simulate AI service failure
      aiServer.simulateError('ResourceInsights', new Error('AI service unavailable'));

      const result = await server.executeOperation({
        query,
        variables: { id: 'resource-123' }
      });

      // Should have partial data and error
      expect(result.data.resource.id).toBe('resource-123');
      expect(result.data.resource.name).toBeDefined();
      expect(result.data.resource.insights).toBeNull();
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].message).toContain('AI service unavailable');
    });

    test('should propagate authentication errors', async () => {
      const query = gql`
        query GetProtectedResource($id: ID!) {
          resource(id: $id) {
            id
            name
          }
        }
      `;

      const result = await server.executeOperation({
        query,
        variables: { id: 'resource-123' },
        http: {
          headers: {
            authorization: 'Bearer invalid-token'
          }
        }
      });

      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].extensions.code).toBe('UNAUTHENTICATED');
    });
  });

  describe('Subscriptions', () => {
    test('should handle real-time updates across services', async (done) => {
      const subscription = gql`
        subscription ResourceUpdated($id: ID!) {
          resourceUpdated(id: $id) {
            id
            name
            insights {
              securityScore
            }
          }
        }
      `;

      const subscriptionServer = new ApolloServer({
        gateway,
        subscriptions: {
          path: '/subscriptions'
        }
      });

      const client = createSubscriptionClient();
      
      client.subscribe({
        query: subscription,
        variables: { id: 'resource-123' }
      }, {
        next: (result) => {
          expect(result.data.resourceUpdated.id).toBe('resource-123');
          expect(result.data.resourceUpdated.insights).toBeDefined();
          done();
        },
        error: done
      });

      // Trigger update
      setTimeout(() => {
        coreServer.triggerResourceUpdate('resource-123');
      }, 100);
    });
  });

  describe('Performance', () => {
    test('should handle concurrent queries efficiently', async () => {
      const query = gql`
        query GetResource($id: ID!) {
          resource(id: $id) {
            id
            insights {
              securityScore
            }
            liveData {
              metrics {
                name
              }
            }
          }
        }
      `;

      const concurrentQueries = Array.from({ length: 50 }, (_, i) =>
        server.executeOperation({
          query,
          variables: { id: `resource-${i}` }
        })
      );

      const startTime = Date.now();
      const results = await Promise.all(concurrentQueries);
      const duration = Date.now() - startTime;

      // All queries should succeed
      results.forEach(result => {
        expect(result.errors).toBeUndefined();
      });

      // Should handle 50 concurrent queries reasonably fast
      expect(duration).toBeLessThan(5000);
    });

    test('should cache repeated queries', async () => {
      const query = gql`
        query GetResourceInsights($id: ID!) {
          resource(id: $id) {
            insights {
              securityScore
              recommendations {
                title
              }
            }
          }
        }
      `;

      // First execution
      const result1 = await server.executeOperation({
        query,
        variables: { id: 'resource-123' }
      });

      const aiCallCountAfterFirst = aiServer.getCallCount();

      // Second execution (should be cached)
      const result2 = await server.executeOperation({
        query,
        variables: { id: 'resource-123' }
      });

      const aiCallCountAfterSecond = aiServer.getCallCount();

      expect(result1.data).toEqual(result2.data);
      expect(aiCallCountAfterSecond).toBe(aiCallCountAfterFirst); // No additional calls
    });
  });

  describe('Schema Evolution', () => {
    test('should handle service schema changes', async () => {
      // Add new field to core service
      coreServer.addField('Resource', 'newField', 'String');
      
      // Gateway should detect schema change
      await gateway.load();
      
      const query = gql`
        query GetResourceWithNewField($id: ID!) {
          resource(id: $id) {
            id
            name
            newField
          }
        }
      `;

      const result = await server.executeOperation({
        query,
        variables: { id: 'resource-123' }
      });

      expect(result.errors).toBeUndefined();
      expect(result.data.resource.newField).toBeDefined();
    });
  });
});

// Mock service helpers
async function startMockCoreService() {
  const typeDefs = gql`
    extend type Query {
      resource(id: ID!): Resource
      resources(filters: ResourceFilters): [Resource!]!
    }
    
    type Resource @key(fields: "id") {
      id: ID!
      name: String!
      type: ResourceType!
      subscriptionId: String!
    }
    
    enum ResourceType {
      VIRTUAL_MACHINE
      STORAGE_ACCOUNT
    }
    
    input ResourceFilters {
      resourceType: ResourceType
    }
  `;

  const resolvers = {
    Query: {
      resource: (_, { id }) => ({ id, name: `Resource ${id}`, type: 'VIRTUAL_MACHINE', subscriptionId: 'sub-123' }),
      resources: (_, { filters }) => {
        const mockResources = [
          { id: 'resource-1', name: 'Resource 1', type: 'VIRTUAL_MACHINE', subscriptionId: 'sub-123' },
          { id: 'resource-2', name: 'Resource 2', type: 'STORAGE_ACCOUNT', subscriptionId: 'sub-123' }
        ];
        
        if (filters?.resourceType) {
          return mockResources.filter(r => r.type === filters.resourceType);
        }
        
        return mockResources;
      }
    },
    Resource: {
      __resolveReference: (reference) => ({ ...reference, name: `Resource ${reference.id}` })
    }
  };

  const server = new ApolloServer({
    schema: buildFederatedSchema({ typeDefs, resolvers }),
    port: 8080
  });

  await server.start();
  return server;
}

async function startMockAIService() {
  // Similar mock setup for AI service...
  // Implementation details omitted for brevity
}

async function startMockAzureService() {
  // Similar mock setup for Azure service...
  // Implementation details omitted for brevity
}