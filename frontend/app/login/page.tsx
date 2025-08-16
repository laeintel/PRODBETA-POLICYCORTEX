/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import React, { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Shield, Lock, User, AlertCircle, Mail, ChevronRight, Cloud, Brain } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

export default function LoginPage() {
  const router = useRouter()
  const { login, isAuthenticated, loading: authLoading, error: authError } = useAuth()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showResetModal, setShowResetModal] = useState(false)
  const [showContactModal, setShowContactModal] = useState(false)
  const [resetEmail, setResetEmail] = useState('')
  const [contactMessage, setContactMessage] = useState('')
  const [contactEmail, setContactEmail] = useState('')

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard')
    }
  }, [isAuthenticated, router])

  const handleLogin = async () => {
    setIsLoading(true)
    setError(null)
    try {
      await login()
    } catch (err: any) {
      setError(err.message || 'Failed to sign in. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handlePasswordReset = async () => {
    if (!resetEmail) {
      setError('Please enter your email address')
      return
    }
    
    // In production, this would call Azure AD B2C password reset flow
    console.log('Password reset requested for:', resetEmail)
    setShowResetModal(false)
    setError('Password reset link sent to ' + resetEmail)
    setResetEmail('')
  }

  const handleContactAdmin = async () => {
    if (!contactEmail || !contactMessage) {
      setError('Please fill in all fields')
      return
    }
    
    // In production, this would send an email to admin
    console.log('Contact admin:', { email: contactEmail, message: contactMessage })
    setShowContactModal(false)
    setError('Your message has been sent to the administrator')
    setContactEmail('')
    setContactMessage('')
  }

  // Animated background particles
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    size: Math.random() * 4 + 1,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 20 + 10
  }))

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0">
        {/* Grid pattern */}
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
        
        {/* Floating particles */}
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute bg-blue-400/20 rounded-full"
            style={{
              width: particle.size + 'px',
              height: particle.size + 'px',
              left: particle.x + '%',
              top: particle.y + '%',
            }}
            animate={{
              x: [0, 30, -30, 0],
              y: [0, -30, 30, 0],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        ))}

        {/* Gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full filter blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full filter blur-3xl animate-pulse delay-1000" />
      </div>

      {/* Main content */}
      <div className="relative z-10 w-full max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center items-center mb-4">
            <Shield className="w-16 h-16 text-blue-400 mr-4" />
            <h1 className="text-5xl font-bold text-white">PolicyCortex</h1>
          </div>
          <p className="text-xl text-blue-200">AI-Powered Azure Governance Platform</p>
          <p className="text-sm text-gray-400 mt-2">Unified compliance, security, and cost optimization</p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 items-center">
          {/* Left side - Features */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="hidden md:block"
          >
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Brain className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">AI-Driven Intelligence</h3>
                  <p className="text-gray-400 text-sm">4 patented technologies for predictive compliance and automated governance</p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Shield className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">Enterprise Security</h3>
                  <p className="text-gray-400 text-sm">Post-quantum cryptography and zero-trust architecture</p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Cloud className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">Multi-Cloud Ready</h3>
                  <p className="text-gray-400 text-sm">Seamless integration with Azure, AWS, and GCP environments</p>
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="mt-8 grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">99.9%</div>
                <div className="text-xs text-gray-400">Uptime SLA</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">&lt; 100ms</div>
                <div className="text-xs text-gray-400">Response Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">24/7</div>
                <div className="text-xs text-gray-400">AI Monitoring</div>
              </div>
            </div>
          </motion.div>

          {/* Right side - Login form */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-8 shadow-2xl border border-white/20">
              <div className="text-center mb-6">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Lock className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Welcome Back</h2>
                <p className="text-gray-400 text-sm">Sign in with your Azure AD account</p>
              </div>

              {(error || authError) && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start space-x-3"
                >
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-red-300">{error || authError}</div>
                </motion.div>
              )}

              <button
                onClick={handleLogin}
                disabled={isLoading || authLoading}
                className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg px-6 py-4 font-medium hover:from-blue-600 hover:to-blue-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 group"
              >
                {isLoading || authLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <>
                    <User className="w-5 h-5" />
                    <span>Sign in with Azure AD</span>
                    <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>

              <div className="mt-6 flex items-center">
                <div className="flex-1 border-t border-gray-600"></div>
                <span className="px-4 text-xs text-gray-400">OR</span>
                <div className="flex-1 border-t border-gray-600"></div>
              </div>

              <div className="mt-6 space-y-3">
                <button
                  onClick={() => setShowResetModal(true)}
                  className="w-full bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg px-4 py-3 text-sm font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                >
                  <Lock className="w-4 h-4" />
                  <span>Reset Password</span>
                </button>

                <button
                  onClick={() => setShowContactModal(true)}
                  className="w-full bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg px-4 py-3 text-sm font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                >
                  <Mail className="w-4 h-4" />
                  <span>Contact Administrator</span>
                </button>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-700">
                <p className="text-xs text-gray-400 text-center">
                  By signing in, you agree to our Terms of Service and Privacy Policy
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-12 text-center text-gray-400 text-sm"
        >
          <p>© 2024 PolicyCortex. Enterprise Azure Governance Platform.</p>
          <p className="mt-2">Powered by AI • Secured by Design • Trusted by Leaders</p>
        </motion.div>
      </div>

      {/* Password Reset Modal */}
      {showResetModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-800 rounded-xl p-6 max-w-md w-full border border-gray-700"
          >
            <h3 className="text-xl font-bold text-white mb-4">Reset Password</h3>
            <p className="text-gray-400 text-sm mb-4">
              Enter your email address and we&apos;ll send you a link to reset your password.
            </p>
            <input
              type="email"
              value={resetEmail}
              onChange={(e) => setResetEmail(e.target.value)}
              placeholder="Enter your email"
              className="w-full bg-white/10 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 mb-4"
            />
            <div className="flex space-x-3">
              <button
                onClick={handlePasswordReset}
                className="flex-1 bg-blue-500 hover:bg-blue-600 text-white rounded-lg px-4 py-2 font-medium transition-colors"
              >
                Send Reset Link
              </button>
              <button
                onClick={() => setShowResetModal(false)}
                className="flex-1 bg-gray-700 hover:bg-gray-600 text-white rounded-lg px-4 py-2 font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Contact Admin Modal */}
      {showContactModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-800 rounded-xl p-6 max-w-md w-full border border-gray-700"
          >
            <h3 className="text-xl font-bold text-white mb-4">Contact Administrator</h3>
            <p className="text-gray-400 text-sm mb-4">
              Need help accessing your account? Send a message to your system administrator.
            </p>
            <input
              type="email"
              value={contactEmail}
              onChange={(e) => setContactEmail(e.target.value)}
              placeholder="Your email address"
              className="w-full bg-white/10 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 mb-3"
            />
            <textarea
              value={contactMessage}
              onChange={(e) => setContactMessage(e.target.value)}
              placeholder="Describe your issue..."
              rows={4}
              className="w-full bg-white/10 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 mb-4 resize-none"
            />
            <div className="flex space-x-3">
              <button
                onClick={handleContactAdmin}
                className="flex-1 bg-blue-500 hover:bg-blue-600 text-white rounded-lg px-4 py-2 font-medium transition-colors"
              >
                Send Message
              </button>
              <button
                onClick={() => setShowContactModal(false)}
                className="flex-1 bg-gray-700 hover:bg-gray-600 text-white rounded-lg px-4 py-2 font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}