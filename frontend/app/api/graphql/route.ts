/**
 * GraphQL API Route - Implements TD.MD hard-fail requirements
 * Returns 404 in real mode to prevent mock data leakage to production
 */

import { NextRequest, NextResponse } from 'next/server';
import { mockGraphQLResolver } from '../../../api/graphql/mock';

export async function POST(request: NextRequest) {
  // Check if we're in demo/mock mode
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  const useRealData = process.env.USE_REAL_DATA === 'true';
  
  // TD.MD requirement: Hard-fail in real mode - return 404, not fallback data
  if (!isDemoMode || useRealData) {
    return NextResponse.json(
      { 
        error: 'GraphQL disabled in real-data mode',
        message: 'GraphQL endpoint is not available when USE_REAL_DATA is true',
        hint: 'Configure real GraphQL backend or enable demo mode'
      },
      { status: 404 }
    );
  }
  
  try {
    // Only provide mock data in demo mode
    const body = await request.json();
    const { query, variables } = body;
    
    const response = await mockGraphQLResolver(query, variables);
    return NextResponse.json(response);
  } catch (error) {
    // In demo mode, return error but still with 200 status (GraphQL convention)
    return NextResponse.json({
      data: null,
      errors: [{
        message: error instanceof Error ? error.message : 'GraphQL resolver error',
        extensions: { 
          code: 'MOCK_MODE_ERROR',
          hint: 'This is a mock GraphQL endpoint active only in demo mode'
        }
      }]
    });
  }
}

// Also handle GET requests with informative message
export async function GET() {
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  const useRealData = process.env.USE_REAL_DATA === 'true';
  
  if (!isDemoMode || useRealData) {
    return NextResponse.json(
      { 
        error: 'GraphQL disabled in real-data mode',
        message: 'GraphQL endpoint requires POST method and is disabled when USE_REAL_DATA is true'
      },
      { status: 404 }
    );
  }
  
  return NextResponse.json({
    message: 'GraphQL endpoint (mock mode)',
    status: 'Available in demo mode only',
    method: 'Use POST method with query and variables'
  });
}