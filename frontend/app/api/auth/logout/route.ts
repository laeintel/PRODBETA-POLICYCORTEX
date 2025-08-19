import { NextResponse } from 'next/server'

export async function POST() {
  const resp = NextResponse.json({ ok: true })
  resp.cookies.set('auth-status', '', { httpOnly: true, sameSite: 'lax', path: '/', expires: new Date(0) })
  resp.cookies.set('auth-token', '', { httpOnly: true, sameSite: 'lax', path: '/', expires: new Date(0) })
  return resp
}


