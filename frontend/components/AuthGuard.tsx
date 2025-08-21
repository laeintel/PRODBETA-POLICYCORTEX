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

import { useEffect, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '../contexts/AuthContext'
import { motion } from 'framer-motion'
import { Shield, LogIn, Loader2 } from 'lucide-react'

interface AuthGuardProps {
  children: React.ReactNode
  requireAuth?: boolean
}

export default function AuthGuard({ children, requireAuth = true }: AuthGuardProps) {
  const { isAuthenticated, login, loading: authLoading, user } = useAuth()
  const router = useRouter()
  const pathname = usePathname()
  const [checking, setChecking] = useState(true)
  const [loginAttempted, setLoginAttempted] = useState(false)

  useEffect(() => {
    // Allow a brief moment for auth to initialize
    const checkAuth = async () => {
      // Wait for auth to initialize
      await new Promise(resolve => setTimeout(resolve, 500))
      
      if (!requireAuth) {
        setChecking(false)
        return
      }

      // If not authenticated and auth is required
      if (!isAuthenticated && !authLoading) {
        // Store current path for redirect after login
        if (typeof window !== 'undefined') {
          sessionStorage.setItem('returnUrl', pathname)
        }
        setChecking(false)
      } else {
        setChecking(false)
      }
    }

    checkAuth()
  }, [isAuthenticated, authLoading, requireAuth, pathname])

  const handleLogin = async () => {
    setLoginAttempted(true)
    try {
      await login()
      // After successful login, redirect to stored URL or dashboard
      const returnUrl = sessionStorage.getItem('returnUrl') || '/dashboard'
      sessionStorage.removeItem('returnUrl')
      router.push(returnUrl)
    } catch (error) {
      console.error('Login failed:', error)
      setLoginAttempted(false)
    }
  }

  // Show loading state while checking authentication
  if (checking || authLoading) {
    return (
      <div className="min-h-screen bg-background dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary dark:text-purple-400 mx-auto mb-4 animate-spin" />
          <p className="text-foreground dark:text-white text-lg">Verifying authentication...</p>
        </div>
      </div>
    )
  }

  // If authentication is required but user is not authenticated
  if (requireAuth && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-background dark:bg-gray-900 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="bg-card/90 dark:bg-white/10 backdrop-blur-lg rounded-2xl p-8 max-w-md w-full mx-4 border border-border dark:border-white/20"
        >
          <div className="text-center">
            <Shield className="w-16 h-16 text-primary dark:text-purple-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-foreground dark:text-white mb-2">Authentication Required</h2>
            <p className="text-muted-foreground dark:text-gray-300 mb-6">
              Please sign in with your Azure AD account to access PolicyCortex
            </p>
            
            <button type="button"
              onClick={handleLogin}
              disabled={loginAttempted}
              className="w-full px-6 py-3 bg-primary dark:bg-purple-600 text-primary-foreground dark:text-white rounded-lg font-semibold hover:bg-primary/90 dark:hover:bg-purple-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loginAttempted ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  Sign in with Azure AD
                </>
              )}
            </button>

            <div className="mt-6 p-4 bg-muted/20 dark:bg-white/5 rounded-lg">
              <p className="text-sm text-muted-foreground dark:text-gray-400">
                <strong>Note:</strong> You need appropriate Azure AD permissions to access this application.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    )
  }

  // User is authenticated or auth is not required
  return <>{children}</>
}

// Higher-order component to wrap pages with auth guard
export function withAuth<P extends object>(
  Component: React.ComponentType<P>,
  requireAuth: boolean = true
) {
  return function AuthenticatedComponent(props: P) {
    return (
      <AuthGuard requireAuth={requireAuth}>
        <Component {...props} />
      </AuthGuard>
    )
  }
}