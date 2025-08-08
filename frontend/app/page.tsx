'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Shield, Brain, Zap, Globe, Lock, BarChart, Users, DollarSign, Server, Network, Activity, Sparkles, LogIn } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated, login, user, loading } = useAuth()

  const handleLogin = async () => {
    try {
      await login()
      // After successful login, redirect to dashboard
      router.push('/dashboard')
    } catch (error) {
      console.error('Login failed:', error)
    }
  }

  const handleGetStarted = () => {
    if (isAuthenticated) {
      router.push('/dashboard')
    } else {
      handleLogin()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h1 className="text-6xl font-bold text-white mb-4">
            PolicyCortex <span className="text-purple-400">v2</span>
          </h1>
          <p className="text-xl text-gray-300 mb-2">
            Complete AI-Powered Cloud Governance Suite
          </p>
          <p className="text-lg text-gray-400">
            RBAC • Cost Management • Policies • Network Security • Resource Optimization • Custom AI Training
          </p>
        </motion.div>


        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="text-center"
        >
          <div className="inline-flex gap-4">
            <button 
              onClick={handleGetStarted}
              disabled={loading}
              className="px-8 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <Sparkles className="w-5 h-5" />
              {isAuthenticated ? 'Go to Dashboard' : 'Get Started'}
            </button>
            <button 
              onClick={() => router.push('/features')}
              className="px-8 py-3 bg-white/10 text-white rounded-lg font-semibold backdrop-blur-md hover:bg-white/20 transition-colors border border-white/20"
            >
              Learn More
            </button>
            {!isAuthenticated ? (
              <button 
                onClick={handleLogin}
                disabled={loading}
                className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <LogIn className="w-5 h-5" />
                {loading ? 'Signing in...' : 'Login with Azure AD'}
              </button>
            ) : (
              <div className="px-8 py-3 bg-green-600/20 text-green-400 rounded-lg font-semibold flex items-center gap-2 border border-green-600/20">
                <Shield className="w-5 h-5" />
                Signed in as {user?.name || user?.username}
              </div>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mt-16 text-center text-gray-400"
        >
          <p>🚀 Complete AI-Powered Cloud Governance Suite</p>
          <p className="mt-2">
            RBAC • Cost Management • Policies • Network Security • Resource Optimization • Custom AI Training
          </p>
          <p className="mt-4 text-sm">
            Rust • Next.js 14 • GraphQL • WebAssembly • Blockchain • Quantum Computing
          </p>
        </motion.div>
      </div>
    </div>
  )
}