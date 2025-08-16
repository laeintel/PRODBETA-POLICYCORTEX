/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

declare module 'react-window' {
  import * as React from 'react'
  export interface ListOnScrollProps {
    scrollDirection: 'forward' | 'backward'
    scrollOffset: number
    scrollUpdateWasRequested: boolean
  }
  export interface VariableSizeListProps {
    height: number
    width: number
    itemCount: number
    itemSize: (index: number) => number
    overscanCount?: number
    children: React.ComponentType<{ index: number; style: React.CSSProperties }>
  }
  export class VariableSizeList extends React.Component<VariableSizeListProps> {}
  export { VariableSizeList as List, VariableSizeList as FixedSizeList }
}


