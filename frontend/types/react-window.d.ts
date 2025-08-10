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


