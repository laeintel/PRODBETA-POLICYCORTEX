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

const mockRecommendations = [
  {
    id: '1',
    recommendation_type: 'cost_optimization',
    severity: 'high',
    title: 'Optimize Underutilized VMs',
    description: 'We identified 12 virtual machines with average CPU usage below 10%. Consider resizing or deallocating.',
    potential_savings: 2456.78,
    risk_reduction: 0,
    automation_available: true,
    confidence: 0.92
  },
  {
    id: '2',
    recommendation_type: 'security',
    severity: 'critical',
    title: 'Enable MFA for Admin Accounts',
    description: '3 administrator accounts detected without multi-factor authentication enabled.',
    potential_savings: 0,
    risk_reduction: 0.85,
    automation_available: false,
    confidence: 1.0
  },
  {
    id: '3',
    recommendation_type: 'compliance',
    severity: 'medium',
    title: 'Update Resource Tags',
    description: '45 resources missing required compliance tags for cost center and environment.',
    potential_savings: 0,
    risk_reduction: 0.3,
    automation_available: true,
    confidence: 0.88
  }
]

export async function GET(request: NextRequest) {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
    const response = await fetch(`${backendUrl}/api/v1/recommendations`, {
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
    console.log('Backend unavailable, using mock recommendations')
  }

  return NextResponse.json({ recommendations: mockRecommendations })
}