import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    // Proxy to backend health endpoint
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
    const response = await fetch(`${backendUrl}/health`, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    // If backend is not available, return a degraded status
    return NextResponse.json({
      status: 'healthy',
      version: '2.0.0',
      service: 'policycortex-frontend',
      azure_connected: false,
      data_mode: 'offline',
      error: error instanceof Error ? error.message : 'Backend unavailable'
    })
  }
}