/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { ApolloClient, InMemoryCache } from '@apollo/client'

export const client = new ApolloClient({
  // Prefer same-origin GraphQL path so nginx/Next rewrites can route appropriately
  uri: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT || process.env.NEXT_PUBLIC_GRAPHQL_URL || '/graphql',
  cache: new InMemoryCache(),
  // Ensure Apollo Server CSRF prevention passes in browsers
  // Forces a CORS preflight instead of a "simple" request
  headers: {
    'apollo-require-preflight': 'true',
  },
  credentials: 'include',
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
    },
  },
})