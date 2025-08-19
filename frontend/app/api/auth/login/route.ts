import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const { email, password } = await req.json()
    if (!email || !password) {
      return NextResponse.json({ error: 'Missing credentials' }, { status: 400 })
    }
    const resp = NextResponse.json({ ok: true })
    resp.cookies.set('auth-status', 'authenticated', { httpOnly: true, sameSite: 'lax', path: '/' })
    resp.cookies.set('auth-token', 'dev-token', { httpOnly: true, sameSite: 'lax', path: '/' })
    return resp
  } catch {
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 })
  }
}


