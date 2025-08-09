'use client'

import { useEffect, useState } from 'react'

export default function ConnectionStatusBanner() {
  const [status, setStatus] = useState<'ok' | 'degraded' | 'down'>('ok')
  const [message, setMessage] = useState<string>('')

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/v1/policies', { cache: 'no-store' })
        if (!res.ok) throw new Error('Bad status')
        setStatus('ok')
        setMessage('')
      } catch {
        setStatus('down')
        setMessage('Azure connection unavailable. Please ensure az login and USE_REAL_AZURE=true.')
      }
    }
    check()
    const id = setInterval(check, 15000)
    return () => clearInterval(id)
  }, [])

  if (status === 'ok') return null

  return (
    <div className={`text-sm text-white px-4 py-2 ${status === 'down' ? 'bg-red-600/80' : 'bg-yellow-600/80'}`}>
      {message || (status === 'degraded' ? 'Some services are degraded.' : 'Service unavailable.')}
    </div>
  )
}


