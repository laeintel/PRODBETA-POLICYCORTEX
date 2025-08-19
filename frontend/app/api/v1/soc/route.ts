import { NextRequest, NextResponse } from 'next/server'

// Minimal SOC mock endpoint to satisfy UI in demo
export async function GET(_req: NextRequest) {
  return NextResponse.json({ ok: true, generatedAt: new Date().toISOString() })
}

