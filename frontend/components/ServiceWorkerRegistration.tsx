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

import { useEffect } from 'react'

export function ServiceWorkerRegistration() {
  useEffect(() => {
    // Only run on client side
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator && window.location.hostname !== 'localhost') {
      window.addEventListener('load', () => {
        navigator.serviceWorker
          .register('/service-worker.js')
          .then((registration) => {
            console.log('SW registered:', registration)
            
            // Check for updates every 5 minutes
            setInterval(() => {
              registration.update()
            }, 5 * 60 * 1000)
            
            // Handle updates
            registration.addEventListener('updatefound', () => {
              const newWorker = registration.installing
              if (newWorker) {
                newWorker.addEventListener('statechange', () => {
                  if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                    // New update available
                    const event = new CustomEvent('sw-update-available')
                    window.dispatchEvent(event)
                  }
                })
              }
            })
          })
          .catch((error) => {
            console.error('SW registration failed:', error)
          })
      })
      
      // Handle controller change
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload()
      })
      
      // Request notification permission
      if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission()
      }
    }
  }, [])
  
  return null
}