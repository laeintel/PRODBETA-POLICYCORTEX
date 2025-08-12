'use client'

import { Suspense } from 'react'
import DashboardContent from './DashboardContent'
import AuthGuard from './AuthGuard'

export default function DashboardWrapper() {
  return (
    <AuthGuard requireAuth={true}>
      <Suspense fallback={
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-purple-400 border-t-transparent rounded-full mx-auto mb-4 animate-spin" />
            <p className="text-white">Loading dashboard...</p>
          </div>
        </div>
      }>
        <DashboardContent />
      </Suspense>
    </AuthGuard>
  )
}