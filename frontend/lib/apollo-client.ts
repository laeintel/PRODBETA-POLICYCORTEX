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