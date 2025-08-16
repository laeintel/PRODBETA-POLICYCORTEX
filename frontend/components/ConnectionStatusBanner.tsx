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

export default function ConnectionStatusBanner() {
  const [status, setStatus] = useState<'ok' | 'degraded' | 'down'>('ok')
  const [message, setMessage] = useState<string>('')
  const [auth, setAuth] = useState<'ok' | 'unauth'>('ok')

  useEffect(() => {
    const check = async () => {
      try {
        // Check the health endpoint using Next.js API route proxy
        const res = await fetch('/api/health', { 
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        })
        const data = await res.json()
        
        if (data.azure_connected) {
          setStatus('ok')
          setMessage('')
        } else {
          setStatus('degraded')
          setMessage('Running in simulated mode. Azure connection not configured.')
        }
      } catch (err) {
        setStatus('down')
        setMessage('Backend service unavailable. Please start the backend with ./start-dev.bat')
      }
    }
    check()
    const id = setInterval(check, 30000) // Check every 30 seconds
    return () => clearInterval(id)
  }, [])

  const show = status !== 'ok' || auth === 'unauth'
  if (!show) return null

  return (
    <div className={`text-sm text-white px-4 py-2 ${status === 'down' ? 'bg-red-600/80' : 'bg-yellow-600/80'}`}>
      {message || (status === 'degraded' ? 'Some services are degraded.' : 'Service unavailable.')}
    </div>
  )
}


