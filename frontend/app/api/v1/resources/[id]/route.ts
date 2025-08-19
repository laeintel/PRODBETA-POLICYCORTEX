import { NextRequest, NextResponse } from 'next/server'

// Minimal mock resource details for demo fallback
export async function GET(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params
  const resource = {
    id,
    name: `resource-${id}`,
    type: 'Microsoft.Compute/virtualMachines',
    location: 'eastus',
    tags: { Environment: 'Prod', Owner: 'FinOps' },
    status: { state: 'running', availability: 99.9, performance_score: 92 },
    health: { status: 'Healthy', issues: [], recommendations: [] },
    last_updated: new Date().toISOString()
  }
  return NextResponse.json(resource)
}

