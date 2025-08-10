'use client'

import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function DemoModeBanner() {
  const [demoMode, setDemoMode] = useState<boolean>(false)
  const [message, setMessage] = useState<string>('')

  useEffect(() => {
    const envDemo = process.env.NEXT_PUBLIC_DISABLE_DEEP === 'true'
    setDemoMode(!!envDemo)
    if (envDemo) {
      setMessage('Demo Mode: Some views use simulated data. Connect Azure to see live data and enable remediation.')
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


