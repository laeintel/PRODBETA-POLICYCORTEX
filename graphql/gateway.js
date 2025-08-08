const { ApolloServer } = require('@apollo/server');
const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');
const { startStandaloneServer } = require('@apollo/server/standalone');

async function startGateway() {
  // Create the gateway
  const gateway = new ApolloGateway({
    supergraphSdl: `
      @core(feature: "https://specs.apollo.dev/core/v0.1")
      @core(feature: "https://specs.apollo.dev/join/v0.1")
      @core(feature: "https://specs.apollo.dev/inaccessible/v0.1")
      {
        query: Query
        mutation: Mutation
      }

      type Query {
        policies: [Policy!]!
        policy(id: ID!): Policy
        resources: [Resource!]!
        resource(id: ID!): Resource
        compliance: [ComplianceResult!]!
        users: [User!]!
        organizations: [Organization!]!
      }

      type Mutation {
        createPolicy(input: CreatePolicyInput!): Policy!
        updatePolicy(id: ID!, input: UpdatePolicyInput!): Policy!
        deletePolicy(id: ID!): Boolean!
        createResource(input: CreateResourceInput!): Resource!
        checkCompliance(policyId: ID!, resourceId: ID!): ComplianceResult!
      }

      type Policy {
        id: ID!
        name: String!
        description: String
        category: String!
        severity: Severity!
        status: PolicyStatus!
        rules: [Rule!]!
        createdAt: String!
        updatedAt: String!
      }

      type Resource {
        id: ID!
        subscriptionId: String!
        resourceId: String!
        name: String!
        type: String!
        location: String!
        tags: String
        createdAt: String!
      }

      type ComplianceResult {
        id: ID!
        policyId: String!
        resourceId: String!
        status: ComplianceStatus!
        reason: String
        checkedAt: String!
      }

      type User {
        id: ID!
        email: String!
        name: String!
        role: UserRole!
        organizationId: String!
      }

      type Organization {
        id: ID!
        name: String!
        tier: Tier!
        users: [User!]!
        subscriptions: [Subscription!]!
      }

      type Subscription {
        id: ID!
        subscriptionId: String!
        name: String!
        resources: [Resource!]!
      }

      type Rule {
        id: ID!
        name: String!
        condition: String!
        action: String!
      }

      enum Severity {
        LOW
        MEDIUM
        HIGH
        CRITICAL
      }

      enum PolicyStatus {
        DRAFT
        ACTIVE
        ARCHIVED
      }

      enum ComplianceStatus {
        COMPLIANT
        NON_COMPLIANT
        EXEMPT
        NOT_APPLICABLE
      }

      enum UserRole {
        ADMIN
        ANALYST
        VIEWER
      }

      enum Tier {
        STARTER
        PROFESSIONAL
        ENTERPRISE
      }

      input CreatePolicyInput {
        name: String!
        description: String
        category: String!
        severity: Severity!
        rules: [RuleInput!]!
      }

      input UpdatePolicyInput {
        name: String
        description: String
        category: String
        severity: Severity
        status: PolicyStatus
      }

      input CreateResourceInput {
        subscriptionId: String!
        name: String!
        type: String!
        location: String!
        tags: String
      }

      input RuleInput {
        name: String!
        condition: String!
        action: String!
      }
    `,
    // In production, this would connect to actual subgraph services
    // For now, we'll use a mock implementation
  });

  // Create Apollo Server
  const server = new ApolloServer({
    gateway,
    subscriptions: false, // Subscriptions are not yet supported in Apollo Gateway
  });

  // Start the server
  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
    context: async ({ req }) => {
      // Add authentication context here
      return {
        token: req.headers.authorization || '',
      };
    },
  });

  console.log(`🚀 PolicyCortex GraphQL Gateway ready at ${url}`);
  console.log(`📊 Query endpoint: ${url}graphql`);
}

// Start the gateway
startGateway().catch((err) => {
  console.error('Failed to start gateway:', err);
  process.exit(1);
});