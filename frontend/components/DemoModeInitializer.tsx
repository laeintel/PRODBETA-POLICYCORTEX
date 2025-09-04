'use client'

import { useEffect } from 'react'

export default function DemoModeInitializer() {
  useEffect(() => {
    // Check if demo mode is enabled
    const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'
    
    if (isDemoMode && typeof window !== 'undefined') {
      // Initialize demo session immediately
      const initDemoSession = async () => {
        try {
          // Check if we already have a session
          const checkResponse = await fetch('/api/auth/demo', {
            method: 'GET',
            credentials: 'include'
          })
          
          const status = await checkResponse.json()
          
          // If not authenticated, create demo session
          if (!status.authenticated && status.demoModeEnabled) {
            const response = await fetch('/api/auth/demo', {
              method: 'POST',
              credentials: 'include'
            })
            
            if (response.ok) {
              console.log('✅ Demo mode session initialized')
              // Force a reload to ensure middleware picks up the cookies
              window.location.reload()
            }
          } else if (status.authenticated) {
            console.log('✅ Demo mode session already active')
          }
        } catch (error) {
          console.error('Failed to initialize demo session:', error)
        }
      }
      
      // Initialize immediately
      initDemoSession()
    }
  }, [])
  
  return null
}