/**
 * Accessibility utilities and components for WCAG 2.1 AA compliance
 */

import * as React from 'react'
import { useEffect, useRef, useState } from 'react'

/**
 * Keyboard navigation hook for focus management
 */
export function useFocusTrap(isActive: boolean = true) {
  const containerRef = useRef<HTMLElement>(null)
  const [lastFocused, setLastFocused] = useState<HTMLElement | null>(null)

  useEffect(() => {
    if (!isActive || !containerRef.current) return

    const container = containerRef.current
    const focusableElements = container.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
    
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault()
          lastElement?.focus()
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault()
          firstElement?.focus()
        }
      }
    }

    // Store currently focused element
    setLastFocused(document.activeElement as HTMLElement)
    
    // Focus first element in trap
    firstElement?.focus()

    container.addEventListener('keydown', handleKeyDown)

    return () => {
      container.removeEventListener('keydown', handleKeyDown)
      // Restore focus when trap is deactivated
      lastFocused?.focus()
    }
  }, [isActive, lastFocused])

  return containerRef
}

/**
 * Skip navigation links for screen readers
 */
export function SkipLinks() {
  return (
    <div className="sr-only focus-within:not-sr-only" aria-label="Skip links">
      <ul className="flex flex-col space-y-2 p-4 bg-white dark:bg-gray-900 shadow-lg">
        <li>
          <a href="#main-content" className="skip-link focus:outline-none focus:ring-2 focus:ring-blue-500 p-2 block">Skip to main content</a>
        </li>
        <li>
          <a href="#main-navigation" className="skip-link focus:outline-none focus:ring-2 focus:ring-blue-500 p-2 block">Skip to navigation</a>
        </li>
        <li>
          <a href="#footer" className="skip-link focus:outline-none focus:ring-2 focus:ring-blue-500 p-2 block">Skip to footer</a>
        </li>
      </ul>
    </div>
  )
}

/**
 * Live region announcements for screen readers
 */
export function useAnnounce() {
  const [announcement, setAnnouncement] = useState('')
  const timeoutRef = useRef<NodeJS.Timeout>()

  const announce = (message: string, politeness: 'polite' | 'assertive' = 'polite') => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    // Set the announcement
    setAnnouncement(message)

    // Clear after 1 second to allow re-announcement of same message
    timeoutRef.current = setTimeout(() => {
      setAnnouncement('')
    }, 1000)
  }

  const LiveRegion: React.FC = () => {
    return (
      <>
        <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">{announcement}</div>
        <div role="alert" aria-live="assertive" aria-atomic="true" className="sr-only">{announcement}</div>
      </>
    )
  }

  return { announce, LiveRegion }
}

/**
 * Keyboard shortcuts manager
 */
export function useKeyboardShortcuts(shortcuts: Record<string, () => void>) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = `${e.ctrlKey ? 'ctrl+' : ''}${e.altKey ? 'alt+' : ''}${e.shiftKey ? 'shift+' : ''}${e.key.toLowerCase()}`
      
      if (shortcuts[key]) {
        e.preventDefault()
        shortcuts[key]()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [shortcuts])
}

/**
 * Focus visible indicator management
 */
export function useFocusVisible() {
  const [isKeyboardUser, setIsKeyboardUser] = useState(false)

  useEffect(() => {
    const handleMouseDown = () => setIsKeyboardUser(false)
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setIsKeyboardUser(true)
      }
    }

    document.addEventListener('mousedown', handleMouseDown)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('mousedown', handleMouseDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return isKeyboardUser
}

/**
 * ARIA properties helper
 */
export interface AriaProps {
  role?: string
  'aria-label'?: string
  'aria-labelledby'?: string
  'aria-describedby'?: string
  'aria-expanded'?: boolean
  'aria-selected'?: boolean
  'aria-checked'?: boolean | 'mixed'
  'aria-disabled'?: boolean
  'aria-hidden'?: boolean
  'aria-live'?: 'polite' | 'assertive' | 'off'
  'aria-atomic'?: boolean
  'aria-busy'?: boolean
  'aria-current'?: boolean | 'page' | 'step' | 'location' | 'date' | 'time'
  'aria-invalid'?: boolean | 'grammar' | 'spelling'
  'aria-required'?: boolean
  'aria-readonly'?: boolean
  'aria-controls'?: string
  'aria-owns'?: string
  'aria-flowto'?: string
  'aria-haspopup'?: boolean | 'menu' | 'listbox' | 'tree' | 'grid' | 'dialog'
  'aria-level'?: number
  'aria-multiline'?: boolean
  'aria-multiselectable'?: boolean
  'aria-orientation'?: 'horizontal' | 'vertical'
  'aria-placeholder'?: string
  'aria-pressed'?: boolean | 'mixed'
  'aria-sort'?: 'ascending' | 'descending' | 'none' | 'other'
  'aria-valuemax'?: number
  'aria-valuemin'?: number
  'aria-valuenow'?: number
  'aria-valuetext'?: string
}

/**
 * Build ARIA props for common patterns
 */
export const aria = {
  button: (label: string, pressed?: boolean): AriaProps => ({
    role: 'button',
    'aria-label': label,
    'aria-pressed': pressed,
  }),

  link: (label: string, current?: boolean): AriaProps => ({
    role: 'link',
    'aria-label': label,
    'aria-current': current ? 'page' : undefined,
  }),

  menu: (label: string, expanded: boolean): AriaProps => ({
    role: 'button',
    'aria-label': label,
    'aria-expanded': expanded,
    'aria-haspopup': 'menu',
  }),

  tab: (label: string, selected: boolean, controls: string): AriaProps => ({
    role: 'tab',
    'aria-label': label,
    'aria-selected': selected,
    'aria-controls': controls,
  }),

  tabpanel: (labelledby: string): AriaProps => ({
    role: 'tabpanel',
    'aria-labelledby': labelledby,
  }),

  dialog: (label: string, describedby?: string): AriaProps & { 'aria-modal'?: boolean } => ({
    role: 'dialog',
    'aria-label': label,
    'aria-describedby': describedby,
    'aria-modal': true,
  }),

  alert: (_message: string): AriaProps => ({
    role: 'alert',
    'aria-live': 'assertive',
    'aria-atomic': true,
  }),

  status: (_message: string): AriaProps => ({
    role: 'status',
    'aria-live': 'polite',
    'aria-atomic': true,
  }),

  progressbar: (label: string, value: number, max: number = 100): AriaProps => ({
    role: 'progressbar',
    'aria-label': label,
    'aria-valuenow': value,
    'aria-valuemin': 0,
    'aria-valuemax': max,
  }),

  combobox: (label: string, expanded: boolean, controls: string): AriaProps => ({
    role: 'combobox',
    'aria-label': label,
    'aria-expanded': expanded,
    'aria-controls': controls,
    'aria-haspopup': 'listbox',
  }),

  form: (label: string, invalid?: boolean, describedby?: string): AriaProps => ({
    'aria-label': label,
    'aria-invalid': invalid,
    'aria-describedby': describedby,
  }),
}

/**
 * Reduced motion preference
 */
export function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setPrefersReducedMotion(mediaQuery.matches)

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return prefersReducedMotion
}

/**
 * High contrast mode detection
 */
export function useHighContrast() {
  const [isHighContrast, setIsHighContrast] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-contrast: high)')
    setIsHighContrast(mediaQuery.matches)

    const handleChange = (e: MediaQueryListEvent) => {
      setIsHighContrast(e.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return isHighContrast
}

/**
 * Color scheme preference
 */
export function useColorScheme() {
  const [colorScheme, setColorScheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setColorScheme(mediaQuery.matches ? 'dark' : 'light')

    const handleChange = (e: MediaQueryListEvent) => {
      setColorScheme(e.matches ? 'dark' : 'light')
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return colorScheme
}

/**
 * Screen reader only class
 */
export const srOnly = 'absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0'

/**
 * Focus visible class
 */
export const focusVisible = 'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'

/**
 * Semantic HTML helpers
 */
export const semantic = {
  main: (props: any) => ({ ...props, role: 'main', id: 'main-content' }),
  nav: (label: string) => ({ role: 'navigation', 'aria-label': label, id: 'main-navigation' }),
  article: (label: string) => ({ role: 'article', 'aria-label': label }),
  section: (label: string) => ({ role: 'region', 'aria-label': label }),
  aside: (label: string) => ({ role: 'complementary', 'aria-label': label }),
  footer: () => ({ role: 'contentinfo', id: 'footer' }),
  header: () => ({ role: 'banner' }),
  search: () => ({ role: 'search' }),
}

/**
 * Accessibility testing helper
 */
export async function runAccessibilityTests() {
  if (typeof window === 'undefined') return

  try {
    // Dynamically import axe-core for testing
    const axe = await import('axe-core')
    const results = await (axe as any).default.run()
    
    if ((results as any).violations.length > 0) {
      console.group('🔴 Accessibility Violations')
      ;(results as any).violations.forEach((violation: any) => {
        console.error(
          `${violation.impact?.toUpperCase()}: ${violation.description}`,
          violation.nodes
        )
      })
      console.groupEnd()
    } else {
      console.log('✅ No accessibility violations found')
    }
    
    return results
  } catch (error) {
    console.warn('Could not run accessibility tests:', error)
  }
}


