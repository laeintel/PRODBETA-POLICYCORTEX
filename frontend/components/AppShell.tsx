'use client'

import React from 'react'
import { usePathname } from 'next/navigation'
import SimplifiedNavigation from './SimplifiedNavigation'
import { ThemeProvider } from '@/contexts/ThemeContext'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  // Hide navigation on the public login root
  const isLogin = pathname === '/'
  
  // Simple demo mode - no complex auth
  if (isLogin) {
    return (
      <ThemeProvider>
        {children}
      </ThemeProvider>
    )
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen flex flex-col">
        <SimplifiedNavigation />
        <main 
          id="main-content"
          className="
          flex-1
          pt-16 
          lg:pl-64
          bg-gray-50 dark:bg-gray-900 
          transition-all duration-300
        ">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
            {children}
          </div>
        </main>
      </div>
    </ThemeProvider>
  )
}


