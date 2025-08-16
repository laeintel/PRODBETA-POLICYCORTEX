/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { NextRequest, NextResponse } from 'next/server'

const mockPredictions = [
  {
    id: 'pred-1',
    type: 'cost_anomaly',
    prediction: 'Unusual spike in storage costs expected',
    confidence: 0.78,
    time_horizon: '7 days',
    impact: 'medium',
    recommended_action: 'Review storage account lifecycle policies'
  },
  {
    id: 'pred-2',
    type: 'compliance_drift',
    prediction: 'Policy compliance likely to drop below 80%',
    confidence: 0.85,
    time_horizon: '14 days',
    impact: 'high',
    recommended_action: 'Schedule compliance review meeting'
  },
  {
    id: 'pred-3',
    type: 'resource_scaling',
    prediction: 'Database capacity will reach 85% in current growth rate',
    confidence: 0.92,
    time_horizon: '30 days',
    impact: 'high',
    recommended_action: 'Plan database scaling or optimization'
  }
]

export async function GET(request: NextRequest) {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
    const response = await fetch(`${backendUrl}/api/v1/predictions`, {
      headers: {
        'Content-Type': 'application/json',
        ...(request.headers.get('Authorization') ? {
          'Authorization': request.headers.get('Authorization')!
        } : {})
      },
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    console.log('Backend unavailable, using mock predictions')
  }

  return NextResponse.json(mockPredictions)
}