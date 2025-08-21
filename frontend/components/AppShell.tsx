'use client'

import React from 'react'
import { usePathname } from 'next/navigation'
import SimplifiedNavigation from './SimplifiedNavigation'
import AuthGuard from './AuthGuard'
import { ThemeProvider } from '@/contexts/ThemeContext'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  // Hide navigation on the public login root
  const isLogin = pathname === '/'
  
  // Check if demo mode is enabled
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

  if (isLogin) {
    return (
      <ThemeProvider>
        {children}
      </ThemeProvider>
    )
  }

  // In demo mode, skip auth guard
  if (isDemoMode) {
    return (
      <ThemeProvider>
        <SimplifiedNavigation />
        <div className="
          pt-16 lg:pt-[7.5rem] 
          lg:pl-64 xl:pl-72 2xl:pl-80 3xl:pl-96 4xl:pl-[28rem] 5xl:pl-[32rem]
          min-h-screen 
          bg-background dark:bg-gray-950 
          transition-all duration-300
        ">
          <div className="
            w-full 
            px-4 sm:px-6 lg:px-8 xl:px-12 2xl:px-16 3xl:px-20 4xl:px-24 5xl:px-32
            py-4 sm:py-6 lg:py-8 xl:py-10
            mx-auto
            max-w-full
            3xl:max-w-[2400px] 4xl:max-w-[3200px] 5xl:max-w-[3800px]
          ">
            {children}
          </div>
        </div>
      </ThemeProvider>
    )
  }

  return (
    <ThemeProvider>
      <AuthGuard requireAuth={true}>
        <>
          <SimplifiedNavigation />
          <div className="
            pt-16 lg:pt-[7.5rem] 
            lg:pl-64 xl:pl-72 2xl:pl-80 3xl:pl-96 4xl:pl-[28rem] 5xl:pl-[32rem]
            min-h-screen 
            bg-background dark:bg-gray-950 
            transition-all duration-300
          ">
            <div className="
              w-full 
              px-4 sm:px-6 lg:px-8 xl:px-12 2xl:px-16 3xl:px-20 4xl:px-24 5xl:px-32
              py-4 sm:py-6 lg:py-8 xl:py-10
              mx-auto
              max-w-full
              3xl:max-w-[2400px] 4xl:max-w-[3200px] 5xl:max-w-[3800px]
            ">
              {children}
            </div>
          </div>
        </>
      </AuthGuard>
    </ThemeProvider>
  )
}


