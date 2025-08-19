'use client'

import React from 'react'
import { usePathname } from 'next/navigation'
import TacticalSidebar from './TacticalSidebar'
import AuthGuard from './AuthGuard'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  // Hide sidebar on the public login root
  const isLogin = pathname === '/'

  if (isLogin) {
    return <>{children}</>
  }

  return (
    <AuthGuard requireAuth={true}>
      <div className="min-h-screen bg-gray-950 text-gray-100 flex">
        <TacticalSidebar />
        <div className="flex-1 flex flex-col">
          {children}
        </div>
      </div>
    </AuthGuard>
  )
}


