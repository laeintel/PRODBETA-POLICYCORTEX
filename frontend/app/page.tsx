'use client'

import { useState, useEffect } from 'react'
import { Shield, Lock, KeyRound, Cpu, Fingerprint } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    // Check for demo mode and redirect
    if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
      router.push('/dashboard')
    }
  }, [])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })
      if (!resp.ok) throw new Error('Invalid credentials')
      window.location.href = '/dashboard'
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
      <div className="w-full max-w-md border border-gray-800 rounded-2xl bg-black/60 backdrop-blur-md p-6 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center"><Shield className="w-5 h-5 text-white" /></div>
          <div>
            <h1 className="text-xl font-bold text-white">POLICYCORTEX</h1>
            <p className="text-xs text-gray-400">Tactical Access Portal</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 mb-6 text-center">
          <div className="text-gray-400 text-xs">Zero Trust</div>
          <div className="text-gray-400 text-xs">MFA Ready</div>
          <div className="text-gray-400 text-xs">Encrypted</div>
        </div>

        {error && (
          <div className="mb-3 text-sm text-red-400 bg-red-900/20 border border-red-900/40 rounded p-2">{error}</div>
        )}

        <form onSubmit={handleLogin} className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-800 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-600"
              placeholder="you@company.com"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-9 px-3 py-2 rounded-lg bg-gray-900 border border-gray-800 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-600"
                placeholder="••••••••"
                required
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white text-sm font-medium disabled:opacity-70"
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <button className="py-2 text-xs bg-gray-800 hover:bg-gray-700 rounded-lg text-white flex items-center justify-center gap-2"><KeyRound className="w-4 h-4" /> SSO</button>
          <button className="py-2 text-xs bg-gray-800 hover:bg-gray-700 rounded-lg text-white flex items-center justify-center gap-2"><Fingerprint className="w-4 h-4" /> Passkey</button>
          <button onClick={() => router.push('/dashboard')} className="py-2 text-xs bg-gray-800 hover:bg-gray-700 rounded-lg text-white flex items-center justify-center gap-2"><Cpu className="w-4 h-4" /> Guest</button>
        </div>

        <p className="mt-4 text-[10px] text-gray-500">Unauthorized access is prohibited. Activity may be monitored and logged.</p>
      </div>
    </main>
  )
}

// Login page is the only export on root; redirect handled by middleware after authentication