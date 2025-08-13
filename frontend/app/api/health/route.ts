import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'policycortex-frontend',
    timestamp: new Date().toISOString(),
    mode: process.env.NODE_ENV,
  })
}