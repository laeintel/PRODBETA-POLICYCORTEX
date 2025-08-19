/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function SecurityPage() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to Security Overview when accessing the main security page
    router.replace('/security/overview')
  }, [router])

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}