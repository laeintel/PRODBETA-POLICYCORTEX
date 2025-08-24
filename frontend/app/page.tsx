'use client'

import { useState, useEffect } from 'react'
import { Shield, Lock } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const { login } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    // Check for demo mode and redirect
    if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
      router.push('/dashboard')
    }
  }, [])

  const handleLogin = async (e: React.MouseEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await login()
      router.push('/dashboard')
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-6">
      <div className="w-full max-w-md border border-gray-300 dark:border-gray-800 rounded-2xl bg-white dark:bg-gray-900/90 backdrop-blur-md p-6 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-gray-200 dark:bg-gray-800 flex items-center justify-center"><Shield className="w-5 h-5 text-gray-700 dark:text-white" /></div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">POLICYCORTEX</h1>
            <p className="text-xs text-gray-600 dark:text-gray-400">Tactical Access Portal</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 mb-6 text-center">
          <div className="text-gray-600 dark:text-gray-400 text-xs">Zero Trust</div>
          <div className="text-gray-600 dark:text-gray-400 text-xs">MFA Ready</div>
          <div className="text-gray-600 dark:text-gray-400 text-xs">Encrypted</div>
        </div>

        {error && (
          <div className="mb-3 text-sm text-red-400 bg-red-900/20 border border-red-900/40 rounded p-2">{error}</div>
        )}

        <div className="space-y-3">
          <button
            type="button"
            onClick={handleLogin}
            disabled={loading}
            className="w-full py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white text-sm font-medium disabled:opacity-70 flex items-center justify-center gap-2"
          >
            <Lock className="w-4 h-4" /> {loading ? 'Opening Microsoft sign-in…' : 'Sign in with Microsoft'}
          </button>
        </div>

        {/* No guest access; authentication required before UI */}

        <p className="mt-4 text-[10px] text-gray-500">Unauthorized access is prohibited. Activity may be monitored and logged.</p>
      </div>
    </main>
  )
}

// Login page is the only export on root; redirect handled by middleware after authentication