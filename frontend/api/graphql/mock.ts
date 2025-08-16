// GraphQL Mock Resolver - Returns 200 with empty arrays for demo
// This provides fallback responses when GraphQL gateway is unavailable

export const mockGraphQLResolver = async (query: string, variables?: any) => {
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
  return process.env.USE_MOCK_GRAPHQL === 'true' || 
         process.env.NODE_ENV === 'demo' ||
         !process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT;
};