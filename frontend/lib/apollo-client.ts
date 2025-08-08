// Apollo Client Configuration - Issue #37
// GraphQL client with caching, real-time subscriptions, and error handling

import { ApolloClient, InMemoryCache, createHttpLink, split, from } from '@apollo/client'
import { setContext } from '@apollo/client/link/context'
import { onError } from '@apollo/client/link/error'
import { RetryLink } from '@apollo/client/link/retry'
import { WebSocketLink } from '@apollo/client/link/ws'
import { getMainDefinition } from '@apollo/client/utilities'
import { createClient } from 'graphql-ws'

// HTTP Link for queries and mutations
const httpLink = createHttpLink({
  uri: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT || 'http://localhost:4000/graphql',
  credentials: 'same-origin',
})

// WebSocket Link for subscriptions
const wsLink = process.browser
  ? new WebSocketLink(
      createClient({
        url: process.env.NEXT_PUBLIC_WEBSOCKET_ENDPOINT || 'ws://localhost:4000/subscriptions',
        connectionParams: () => {
          const token = localStorage.getItem('auth-token')
          return {
            authorization: token ? `Bearer ${token}` : '',
            'x-client-version': '2.0.0',
          }
        },
        on: {
          connecting: () => console.log('🔗 Connecting to GraphQL WebSocket...'),
          opened: () => console.log('✅ Connected to GraphQL WebSocket'),
          closed: () => console.log('❌ Disconnected from GraphQL WebSocket'),
        },
      })
    )
  : null

// Auth Link - Add authorization token to requests
const authLink = setContext((_, { headers }) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth-token') : null
  
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : '',
      'x-client-name': 'PolicyCortex',
      'x-client-version': '2.0.0',
    },
  }
})

// Error Link - Global error handling
const errorLink = onError(({ graphQLErrors, networkError, operation, forward }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path, extensions }) => {
      console.error(
        `[GraphQL error]: Message: ${message}, Location: ${locations}, Path: ${path}`
      )
      
      // Handle specific error types
      if (extensions?.code === 'UNAUTHENTICATED') {
        // Clear auth token and redirect to login
        if (typeof window !== 'undefined') {
          localStorage.removeItem('auth-token')
          window.location.href = '/login'
        }
      } else if (extensions?.code === 'FORBIDDEN') {
        console.error('Insufficient permissions for operation:', operation.operationName)
      }
    })
  }

  if (networkError) {
    console.error(`[Network error]: ${networkError}`)
    
    // Handle specific network errors
    if (networkError.statusCode === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth-token')
        window.location.href = '/login'
      }
    }
  }
})

// Retry Link - Automatic retry for failed requests
const retryLink = new RetryLink({
  delay: {
    initial: 300,
    max: Infinity,
    jitter: true,
  },
  attempts: {
    max: 5,
    retryIf: (error, _operation) => !!error,
  },
})

// Split Link - Route queries/mutations to HTTP and subscriptions to WebSocket
const splitLink = wsLink
  ? split(
      ({ query }) => {
        const definition = getMainDefinition(query)
        return (
          definition.kind === 'OperationDefinition' &&
          definition.operation === 'subscription'
        )
      },
      wsLink,
      authLink.concat(httpLink)
    )
  : authLink.concat(httpLink)

// Combine all links
const link = from([errorLink, retryLink, splitLink])

// Advanced Cache Configuration
const cache = new InMemoryCache({
  typePolicies: {
    // Query-specific caching policies
    Query: {
      fields: {
        // Policies cache with merge strategy
        policies: {
          keyArgs: ['filters', 'search'],
          merge(existing = [], incoming) {
            return [...incoming]
          },
        },
        // Resources cache with pagination
        resources: {
          keyArgs: ['filters', 'search'],
          merge(existing = [], incoming, { args }) {
            if (args?.pagination?.offset === 0) {
              return incoming
            }
            return existing ? [...existing, ...incoming] : incoming
          },
        },
        // Reports with time-based cache
        reports: {
          keyArgs: ['type', 'dateRange'],
          merge(existing = [], incoming) {
            return [...incoming]
          },
        },
      },
    },
    
    // Policy type caching
    Policy: {
      fields: {
        // Auto-merge compliance updates
        compliance: {
          merge(existing, incoming) {
            return incoming
          },
        },
        // Predictions array with timestamp ordering
        predictions: {
          merge(existing = [], incoming) {
            const combined = [...existing, ...incoming]
            return combined
              .sort((a, b) => b.timestamp - a.timestamp)
              .slice(0, 100) // Keep last 100 predictions
          },
        },
      },
    },
    
    // Resource type caching
    Resource: {
      fields: {
        // Tags object merge
        tags: {
          merge(existing = {}, incoming) {
            return { ...existing, ...incoming }
          },
        },
        // Metadata deep merge
        metadata: {
          merge(existing = {}, incoming) {
            return { ...existing, ...incoming }
          },
        },
      },
    },
    
    // User preferences merge
    User: {
      fields: {
        preferences: {
          merge(existing = {}, incoming) {
            return { ...existing, ...incoming }
          },
        },
      },
    },
  },
  
  // Cache size and garbage collection
  resultCaching: true,
  canonizeResults: true,
})

// Apollo Client Configuration
export const apolloClient = new ApolloClient({
  link,
  cache,
  defaultOptions: {
    watchQuery: {
      errorPolicy: 'all',
      fetchPolicy: 'cache-and-network',
      notifyOnNetworkStatusChange: true,
    },
    query: {
      errorPolicy: 'all',
      fetchPolicy: 'cache-first',
    },
    mutate: {
      errorPolicy: 'all',
    },
  },
  connectToDevTools: process.env.NODE_ENV === 'development',
  name: 'PolicyCortex',
  version: '2.0.0',
})

// Cache persistence (optional)
if (typeof window !== 'undefined') {
  // Restore cache from localStorage
  const cacheData = localStorage.getItem('apollo-cache-persist')
  if (cacheData) {
    try {
      cache.restore(JSON.parse(cacheData))
    } catch (error) {
      console.warn('Failed to restore Apollo cache:', error)
    }
  }
  
  // Save cache to localStorage on updates
  let saveTimeout: NodeJS.Timeout
  apolloClient.onResetStore(() => {
    clearTimeout(saveTimeout)
    saveTimeout = setTimeout(() => {
      localStorage.setItem('apollo-cache-persist', JSON.stringify(cache.extract()))
    }, 1000)
  })
}

// Common GraphQL Operations
export const COMMON_FRAGMENTS = {
  POLICY_DETAILS: `
    fragment PolicyDetails on Policy {
      id
      name
      description
      category
      type
      enabled
      severity
      compliance
      lastModified
      createdBy
      tags
      aiConfidence
      autoRemediationAvailable
      rules {
        id
        condition
        action
        parameters
      }
      predictions {
        timestamp
        value
        confidence
        recommendation
      }
    }
  `,
  
  RESOURCE_DETAILS: `
    fragment ResourceDetails on Resource {
      id
      name
      type
      region
      resourceGroup
      subscription
      compliance
      riskLevel
      policies
      lastAssessed
      metadata
      tags
    }
  `,
  
  USER_DETAILS: `
    fragment UserDetails on User {
      id
      name
      email
      role
      avatar
      preferences {
        theme
        language
        notifications
        autoSave
        compactMode
      }
    }
  `,
}

// Query definitions for common operations
export const GET_POLICIES = `
  query GetPolicies($filters: PolicyFilters, $search: String) {
    policies(filters: $filters, search: $search) {
      ...PolicyDetails
    }
  }
  ${COMMON_FRAGMENTS.POLICY_DETAILS}
`

export const GET_RESOURCES = `
  query GetResources($filters: ResourceFilters, $search: String, $pagination: PaginationInput) {
    resources(filters: $filters, search: $search, pagination: $pagination) {
      ...ResourceDetails
    }
  }
  ${COMMON_FRAGMENTS.RESOURCE_DETAILS}
`

export const GET_COMPLIANCE_DASHBOARD = `
  query GetComplianceDashboard($dateRange: DateRange) {
    complianceDashboard(dateRange: $dateRange) {
      overallScore
      totalPolicies
      totalResources
      criticalIssues
      trends {
        date
        score
        issues
      }
      breakdown {
        category
        score
        resourceCount
        issueCount
      }
    }
  }
`

// Subscription definitions for real-time updates
export const POLICY_UPDATES = `
  subscription PolicyUpdates {
    policyUpdated {
      ...PolicyDetails
    }
  }
  ${COMMON_FRAGMENTS.POLICY_DETAILS}
`

export const RESOURCE_UPDATES = `
  subscription ResourceUpdates {
    resourceUpdated {
      ...ResourceDetails
    }
  }
  ${COMMON_FRAGMENTS.RESOURCE_DETAILS}
`

export const COMPLIANCE_UPDATES = `
  subscription ComplianceUpdates {
    complianceScoreUpdated {
      overallScore
      changedPolicies {
        id
        newScore
      }
      changedResources {
        id
        newScore
      }
      timestamp
    }
  }
`

// Mutation definitions
export const APPLY_AUTO_REMEDIATION = `
  mutation ApplyAutoRemediation($policyId: ID!) {
    applyAutoRemediation(policyId: $policyId) {
      success
      newCompliance
      message
      affectedResources {
        id
        newCompliance
      }
    }
  }
`

export const UPDATE_POLICY = `
  mutation UpdatePolicy($id: ID!, $input: PolicyUpdateInput!) {
    updatePolicy(id: $id, input: $input) {
      ...PolicyDetails
    }
  }
  ${COMMON_FRAGMENTS.POLICY_DETAILS}
`

export const CREATE_POLICY = `
  mutation CreatePolicy($input: PolicyCreateInput!) {
    createPolicy(input: $input) {
      ...PolicyDetails
    }
  }
  ${COMMON_FRAGMENTS.POLICY_DETAILS}
`

// Helper functions for Apollo Client usage
export const prefetchQuery = (query: string, variables?: any) => {
  return apolloClient.query({
    query: query,
    variables,
    fetchPolicy: 'cache-first',
  })
}

export const refetchQueries = (queryNames: string[]) => {
  return apolloClient.refetchQueries({
    include: queryNames,
  })
}

export const clearCache = () => {
  return apolloClient.clearStore()
}

// Performance monitoring
if (process.env.NODE_ENV === 'development') {
  apolloClient.onResetStore(() => {
    console.log('🧹 Apollo cache cleared')
  })
  
  // Log query performance
  apolloClient.setLink(
    from([
      new (class extends from([]).constructor {
        request(operation: any, forward: any) {
          const startTime = Date.now()
          return forward(operation).map((result: any) => {
            const duration = Date.now() - startTime
            console.log(
              `GraphQL ${operation.operationName}: ${duration}ms`,
              { variables: operation.variables }
            )
            return result
          })
        }
      })(),
      link,
    ])
  )
}