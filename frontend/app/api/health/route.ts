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

export async function GET(request: NextRequest) {
  try {
    // Proxy to backend health endpoint - using Python API Gateway which is working
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || (process.env.IN_DOCKER === 'true' || process.env.DOCKER === 'true' ? 'http://api-gateway:8000' : 'http://localhost:8000')
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000) // 3 second timeout
    
    const response = await fetch(`${backendUrl}/health`, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
    }).catch((err) => {
      clearTimeout(timeoutId)
      throw err
    })
    
    clearTimeout(timeoutId)

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    // If backend is not available, return a degraded status with 200 OK
    // This prevents the 500 error in the frontend
    return NextResponse.json({
      status: 'degraded',
      version: '2.11.8',
      service: 'policycortex-frontend',
      azure_connected: false,
      data_mode: 'offline',
      backend_available: false,
      message: 'Running in frontend-only mode',
      error: error instanceof Error ? error.message : 'Backend unavailable'
    }, { status: 200 }) // Return 200 even when degraded to avoid console errors
  }
}