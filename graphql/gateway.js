/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const { buildSubgraphSchema } = require('@apollo/subgraph');
const { gql } = require('graphql-tag');

// Define the GraphQL schema
const typeDefs = gql`
  type Query {
    policies: [Policy!]!
    policy(id: ID!): Policy
    resources: [Resource!]!
    resource(id: ID!): Resource
    compliance: [ComplianceResult!]!
  }

  type Mutation {
    createPolicy(input: CreatePolicyInput!): Policy!
    updatePolicy(id: ID!, input: UpdatePolicyInput!): Policy!
    deletePolicy(id: ID!): Boolean!
  }

  type Policy {
    id: ID!
    name: String!
    description: String
    category: String!
    severity: String!
    status: String!
    createdAt: String!
    updatedAt: String!
  }

  type Resource {
    id: ID!
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
    status: String!
    reason: String
    checkedAt: String!
  }

  input CreatePolicyInput {
    name: String!
    description: String
    category: String!
    severity: String!
  }

  input UpdatePolicyInput {
    name: String
    description: String
    category: String
    severity: String
    status: String
  }
`;

// Define resolvers
const resolvers = {
  Query: {
    policies: () => [
      {
        id: '1',
        name: 'Require HTTPS',
        category: 'Security',
        severity: 'High',
        status: 'Active',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
      {
        id: '2',
        name: 'Tag Compliance',
        category: 'Governance',
        severity: 'Medium',
        status: 'Active',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
    ],
    policy: (_, { id }) => ({
      id,
      name: 'Sample Policy',
      category: 'Security',
      severity: 'High',
      status: 'Active',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }),
    resources: () => [
      {
        id: '1',
        name: 'Storage Account',
        type: 'Microsoft.Storage/storageAccounts',
        location: 'East US',
        createdAt: new Date().toISOString(),
      },
    ],
    resource: (_, { id }) => ({
      id,
      name: 'Sample Resource',
      type: 'Microsoft.Compute/virtualMachines',
      location: 'West US',
      createdAt: new Date().toISOString(),
    }),
    compliance: () => [
      {
        id: '1',
        policyId: '1',
        resourceId: '1',
        status: 'Compliant',
        checkedAt: new Date().toISOString(),
      },
    ],
  },
  Mutation: {
    createPolicy: (_, { input }) => ({
      id: Math.random().toString(36).substr(2, 9),
      ...input,
      status: 'Active',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }),
    updatePolicy: (_, { id, input }) => ({
      id,
      name: input.name || 'Updated Policy',
      category: input.category || 'Security',
      severity: input.severity || 'High',
      status: input.status || 'Active',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }),
    deletePolicy: () => true,
  },
};

async function startGateway() {
  // Create Apollo Server
  const server = new ApolloServer({
    typeDefs,
    resolvers,
    // Dev: disable CSRF prevention to ease local testing
    csrfPrevention: false,
  });

  // Start the server
  const { url } = await startStandaloneServer(server, {
    listen: { port: 4000 },
  });

  console.log(`🚀 PolicyCortex GraphQL Gateway ready at ${url}`);
}

// Start the gateway
startGateway().catch((err) => {
  console.error('Failed to start gateway:', err);
  process.exit(1);
});