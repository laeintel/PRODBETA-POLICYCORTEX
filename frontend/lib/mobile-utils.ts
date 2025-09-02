/**
 * Mobile-first utility functions for PolicyCortex
 * Provides responsive helpers and device detection
 */

/**
 * Check if the current device is mobile
 */
export const isMobile = (): boolean => {
  if (typeof window === 'undefined') return false
  return window.innerWidth < 768
}

/**
 * Check if the current device is tablet
 */
export const isTablet = (): boolean => {
  if (typeof window === 'undefined') return false
  const width = window.innerWidth
  return width >= 768 && width < 1024
}

/**
 * Check if the current device supports touch
 */
export const isTouchDevice = (): boolean => {
  if (typeof window === 'undefined') return false
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0
}

/**
 * Get the current viewport size
 */
export const getViewportSize = () => {
  if (typeof window === 'undefined') {
    return { width: 0, height: 0 }
  }
  return {
    width: window.innerWidth,
    height: window.innerHeight,
  }
}

/**
 * Lock body scroll (useful for modals/mobile menus)
 */
export const lockBodyScroll = (): void => {
  if (typeof document === 'undefined') return
  
  const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth
  document.body.style.overflow = 'hidden'
  document.body.style.paddingRight = `${scrollbarWidth}px`
  document.documentElement.style.overflow = 'hidden'
}

/**
 * Unlock body scroll
 */
export const unlockBodyScroll = (): void => {
  if (typeof document === 'undefined') return
  
  document.body.style.overflow = ''
  document.body.style.paddingRight = ''
  document.documentElement.style.overflow = ''
}

/**
 * Detect iOS device
 */
export const isIOS = (): boolean => {
  if (typeof window === 'undefined') return false
  
  return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
}

/**
 * Detect Android device
 */
export const isAndroid = (): boolean => {
  if (typeof window === 'undefined') return false
  return /Android/.test(navigator.userAgent)
}

/**
 * Get safe area insets for iOS devices
 */
export const getSafeAreaInsets = () => {
  if (typeof window === 'undefined') {
    return { top: 0, right: 0, bottom: 0, left: 0 }
  }
  
  const computedStyle = getComputedStyle(document.documentElement)
  return {
    top: parseInt(computedStyle.getPropertyValue('--inset-top') || '0'),
    right: parseInt(computedStyle.getPropertyValue('--inset-right') || '0'),
    bottom: parseInt(computedStyle.getPropertyValue('--inset-bottom') || '0'),
    left: parseInt(computedStyle.getPropertyValue('--inset-left') || '0'),
  }
}

/**
 * Debounce function for responsive events
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

/**
 * Throttle function for scroll events
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean = false
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * Format touch-friendly click handlers
 */
export const handleTouchClick = (
  callback: () => void,
  preventDefault = true
) => {
  return (e: React.MouseEvent | React.TouchEvent) => {
    if (preventDefault) {
      e.preventDefault()
    }
    callback()
  }
}

/**
 * Check if viewport matches media query
 */
export const matchesMediaQuery = (query: string): boolean => {
  if (typeof window === 'undefined') return false
  return window.matchMedia(query).matches
}

/**
 * Common breakpoint checks
 */
export const breakpoints = {
  sm: () => matchesMediaQuery('(min-width: 640px)'),
  md: () => matchesMediaQuery('(min-width: 768px)'),
  lg: () => matchesMediaQuery('(min-width: 1024px)'),
  xl: () => matchesMediaQuery('(min-width: 1280px)'),
  '2xl': () => matchesMediaQuery('(min-width: 1536px)'),
}

/**
 * Get current breakpoint
 */
export const getCurrentBreakpoint = (): string => {
  if (breakpoints['2xl']()) return '2xl'
  if (breakpoints.xl()) return 'xl'
  if (breakpoints.lg()) return 'lg'
  if (breakpoints.md()) return 'md'
  if (breakpoints.sm()) return 'sm'
  return 'xs'
}

/**
 * Format numbers for mobile display (shorter format)
 */
export const formatCompactNumber = (num: number): string => {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1)}M`
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`
  }
  return num.toString()
}

/**
 * Trap focus within an element (for accessibility)
 */
export const trapFocus = (element: HTMLElement) => {
  const focusableElements = element.querySelectorAll(
    'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select, [tabindex]:not([tabindex="-1"])'
  )
  
  const firstFocusable = focusableElements[0] as HTMLElement
  const lastFocusable = focusableElements[focusableElements.length - 1] as HTMLElement
  
  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return
    
    if (e.shiftKey) {
      if (document.activeElement === firstFocusable) {
        e.preventDefault()
        lastFocusable?.focus()
      }
    } else {
      if (document.activeElement === lastFocusable) {
        e.preventDefault()
        firstFocusable?.focus()
      }
    }
  }
  
  element.addEventListener('keydown', handleKeyDown)
  firstFocusable?.focus()
  
  return () => {
    element.removeEventListener('keydown', handleKeyDown)
  }
}