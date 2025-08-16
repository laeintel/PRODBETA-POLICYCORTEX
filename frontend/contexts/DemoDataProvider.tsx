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

import React, { createContext, useContext, useMemo } from 'react'

type DemoFlags = {
  demoMode: boolean
  useMockData: boolean
  disableDeep: boolean
}

const DemoDataContext = createContext<DemoFlags>({ demoMode: true, useMockData: true, disableDeep: false })

export function DemoDataProvider({ children }: { children: React.ReactNode }) {
  const flags = useMemo<DemoFlags>(() => {
    const useReal = process.env.NEXT_PUBLIC_USE_REAL_DATA === 'true'
    const mock = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true'
    const disableDeep = process.env.NEXT_PUBLIC_DISABLE_DEEP === 'true'
    return {
      demoMode: !useReal,
      useMockData: mock || !useReal || disableDeep,
      disableDeep,
    }
  }, [])

  return (
    <DemoDataContext.Provider value={flags}>{children}</DemoDataContext.Provider>
  )
}

export function useDemoDataFlags() {
  return useContext(DemoDataContext)
}


