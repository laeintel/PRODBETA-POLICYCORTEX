import { NextRequest, NextResponse } from 'next/server'

// Mock data for when backend is unavailable
const mockMetrics = {
  policies: {
    total: 45,
    active: 38,
    violations: 7,
    automated: 28,
    compliance_rate: 84.4,
    prediction_accuracy: 92.3
  },
  rbac: {
    users: 156,
    roles: 12,
    violations: 3,
    risk_score: 24.5,
    anomalies_detected: 2
  },
  costs: {
    current_spend: 45678.90,
    predicted_spend: 48234.56,
    savings_identified: 8456.78,
    optimization_rate: 18.5
  },
  network: {
    endpoints: 234,
    active_threats: 0,
    blocked_attempts: 45,
    latency_ms: 42
  },
  resources: {
    total: 567,
    optimized: 423,
    idle: 34,
    overprovisioned: 12
  },
  ai: {
    accuracy: 94.7,
    predictions_made: 1234,
    automations_executed: 89,
    learning_progress: 78.5
  }
}

export async function GET(request: NextRequest) {
  try {
    // Try to proxy to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
    const response = await fetch(`${backendUrl}/api/v1/metrics`, {
      headers: {
        'Content-Type': 'application/json',
        // Forward auth headers if present
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
    // Backend not available, use mock data
    console.log('Backend unavailable, using mock data')
  }

  // Return mock data when backend is unavailable
  return NextResponse.json(mockMetrics)
}