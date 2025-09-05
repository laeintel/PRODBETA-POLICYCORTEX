import { NextResponse } from 'next/server';

export async function GET() {
  // Demo mode authentication bypass
  const demoUser = {
    id: 'demo-user',
    email: 'demo@policycortex.ai',
    name: 'Demo User',
    roles: ['admin'],
    tenantId: 'demo-tenant',
    authenticated: true,
  };

  return NextResponse.json(demoUser);
}

export async function POST() {
  // Handle demo login
  const demoSession = {
    user: {
      id: 'demo-user',
      email: 'demo@policycortex.ai',
      name: 'Demo User',
      roles: ['admin'],
    },
    token: 'demo-token-' + Date.now(),
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
  };

  return NextResponse.json(demoSession);
}