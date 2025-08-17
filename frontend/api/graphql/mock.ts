// GraphQL Mock Resolver - Returns 200 with empty arrays for demo
// This provides fallback responses when GraphQL gateway is unavailable
// Only active when DEMO_MODE is enabled

export const mockGraphQLResolver = async (query: string, variables?: any) => {
  // Only return mock data if demo mode is enabled
  if (process.env.DEMO_MODE !== 'true' && process.env.USE_MOCK_GRAPHQL !== 'true') {
    throw new Error('Mock GraphQL is disabled in production mode');
  }
  // Parse the query to determine the requested operation
  const operationMatch = query.match(/query\s+(\w+)|mutation\s+(\w+)/);
  const operation = operationMatch ? (operationMatch[1] || operationMatch[2]) : 'unknown';

  // Return mock responses based on operation
  const mockResponses: Record<string, any> = {
    // Metrics and governance queries
    GetMetrics: {
      data: {
        metrics: [],
        aggregatedMetrics: {
          totalResources: 0,
          complianceScore: 95,
          securityScore: 92,
          costOptimizationScore: 88
        }
      }
    },
    
    // Policy queries
    GetPolicies: {
      data: {
        policies: [],
        policyCount: 0
      }
    },
    
    // Resource queries
    GetResources: {
      data: {
        resources: [],
        resourceCount: 0
      }
    },
    
    // Correlation queries
    GetCorrelations: {
      data: {
        correlations: [],
        correlationAnalysis: {
          patterns: [],
          recommendations: []
        }
      }
    },
    
    // Predictions queries
    GetPredictions: {
      data: {
        predictions: [],
        complianceForecasts: []
      }
    },
    
    // Default fallback
    default: {
      data: {},
      message: "Mock response - GraphQL gateway unavailable"
    }
  };

  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 100));

  // Return appropriate mock response
  return mockResponses[operation] || mockResponses.default;
};

// Express middleware for GraphQL mock endpoint
export const graphQLMockMiddleware = async (req: any, res: any) => {
  // Check if mock mode is enabled
  if (!shouldUseMockGraphQL()) {
    res.status(503).json({
      errors: [{
        message: "Mock GraphQL is disabled in production mode",
        extensions: { code: "MOCK_DISABLED" }
      }]
    });
    return;
  }
  
  try {
    const { query, variables } = req.body;
    const response = await mockGraphQLResolver(query, variables);
    
    res.status(200).json(response);
  } catch (error) {
    res.status(200).json({
      data: {},
      errors: [{
        message: "Mock GraphQL resolver error - returning empty response",
        extensions: { code: "MOCK_MODE" }
      }]
    });
  }
};

// Utility to check if we should use mock mode
export const shouldUseMockGraphQL = (): boolean => {
  // Only use mock in demo mode or when explicitly enabled
  // In production, this should always be false unless DEMO_MODE is set
  return process.env.DEMO_MODE === 'true' || 
         process.env.USE_MOCK_GRAPHQL === 'true';
};