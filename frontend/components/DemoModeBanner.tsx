'use client'

import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function DemoModeBanner() {
  const [demoMode, setDemoMode] = useState<boolean>(false)
  const [message, setMessage] = useState<string>('')

  useEffect(() => {
    // Default demo banner ON unless explicitly using real data
    const envDemo = process.env.NEXT_PUBLIC_USE_REAL_DATA !== 'true'
    setDemoMode(!!envDemo)
    if (envDemo) {
      setMessage('Simulated Mode: Read-only experience with mock data. Connect Azure to enable live data and remediation.')
    }
  }, [])

  if (!demoMode) return null

  return (
    <div className="bg-yellow-600/90 text-white text-sm px-4 py-2 flex items-center gap-2">
      <AlertTriangle className="w-4 h-4" />
      <span>{message}</span>
    </div>
  )
}


