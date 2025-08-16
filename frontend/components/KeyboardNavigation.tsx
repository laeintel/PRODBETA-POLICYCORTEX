"use client"

import { useEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'

interface KeyboardNavigationProps {
  children: React.ReactNode
  onKeyboardNav?: (direction: 'up' | 'down' | 'left' | 'right') => void
  focusOnMount?: boolean
  className?: string
}

export function KeyboardNavigation({ 
  children, 
  onKeyboardNav, 
  focusOnMount = false,
  className 
}: KeyboardNavigationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [focusedIndex, setFocusedIndex] = useState(0)

  useEffect(() => {
    if (focusOnMount && containerRef.current) {
      containerRef.current.focus()
    }
  }, [focusOnMount])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const focusableElements = containerRef.current?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )

    if (!focusableElements) return

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        setFocusedIndex(prev => Math.max(0, prev - 1))
        onKeyboardNav?.('up')
        break
      case 'ArrowDown':
        e.preventDefault()
        setFocusedIndex(prev => Math.min(focusableElements.length - 1, prev + 1))
        onKeyboardNav?.('down')
        break
      case 'ArrowLeft':
        e.preventDefault()
        onKeyboardNav?.('left')
        break
      case 'ArrowRight':
        e.preventDefault()
        onKeyboardNav?.('right')
        break
      case 'Enter':
      case ' ':
        const focused = focusableElements[focusedIndex] as HTMLElement
        if (focused && focused !== e.target) {
          e.preventDefault()
          focused.click()
        }
        break
    }
  }

  return (
    <div
      ref={containerRef}
      tabIndex={-1}
      onKeyDown={handleKeyDown}
      className={cn("outline-none", className)}
    >
      {children}
    </div>
  )
}

// Hook for global keyboard shortcuts
export function useKeyboardShortcuts(shortcuts: Record<string, () => void>) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.ctrlKey || e.metaKey 
        ? `${e.ctrlKey ? 'ctrl+' : 'cmd+'}${e.key.toLowerCase()}`
        : e.key.toLowerCase()

      if (shortcuts[key]) {
        e.preventDefault()
        shortcuts[key]()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [shortcuts])
}

// Focus trap component
interface FocusTrapProps {
  children: React.ReactNode
  active?: boolean
  onEscape?: () => void
}

export function FocusTrap({ children, active = true, onEscape }: FocusTrapProps) {
  const trapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!active) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onEscape) {
        onEscape()
        return
      }

      if (e.key === 'Tab') {
        const focusableElements = trapRef.current?.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )

        if (!focusableElements || focusableElements.length === 0) return

        const firstElement = focusableElements[0] as HTMLElement
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault()
            lastElement.focus()
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault()
            firstElement.focus()
          }
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [active, onEscape])

  return (
    <div ref={trapRef}>
      {children}
    </div>
  )
}

// Skip link for accessibility
export function SkipLink({ href = "#main-content", children = "Skip to main content" }) {
  return (
    <a
      href={href}
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-primary-foreground px-4 py-2 rounded-md z-50"
    >
      {children}
    </a>
  )
}