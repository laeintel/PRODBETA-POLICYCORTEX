'use client'

import { useState, useEffect } from 'react'
import { 
  isMobile, 
  isTablet, 
  isTouchDevice, 
  getCurrentBreakpoint,
  debounce 
} from '@/lib/mobile-utils'

interface MobileResponsiveState {
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
  isTouchDevice: boolean
  breakpoint: string
  viewportWidth: number
  viewportHeight: number
  orientation: 'portrait' | 'landscape'
}

/**
 * Custom hook for mobile-responsive behavior
 * Provides reactive state for responsive components
 */
export function useMobileResponsive(): MobileResponsiveState {
  const [state, setState] = useState<MobileResponsiveState>({
    isMobile: false,
    isTablet: false,
    isDesktop: false,
    isTouchDevice: false,
    breakpoint: 'xs',
    viewportWidth: 0,
    viewportHeight: 0,
    orientation: 'portrait',
  })

  useEffect(() => {
    const updateState = () => {
      const width = window.innerWidth
      const height = window.innerHeight
      const mobile = width < 768
      const tablet = width >= 768 && width < 1024
      const desktop = width >= 1024

      setState({
        isMobile: mobile,
        isTablet: tablet,
        isDesktop: desktop,
        isTouchDevice: isTouchDevice(),
        breakpoint: getCurrentBreakpoint(),
        viewportWidth: width,
        viewportHeight: height,
        orientation: width > height ? 'landscape' : 'portrait',
      })
    }

    // Initial state
    updateState()

    // Debounced resize handler
    const handleResize = debounce(updateState, 150)

    // Orientation change handler
    const handleOrientationChange = () => {
      // Small delay to ensure dimensions are updated
      setTimeout(updateState, 100)
    }

    // Add event listeners
    window.addEventListener('resize', handleResize)
    window.addEventListener('orientationchange', handleOrientationChange)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('orientationchange', handleOrientationChange)
    }
  }, [])

  return state
}

/**
 * Hook for managing mobile menu state
 */
export function useMobileMenu(defaultOpen = false) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const responsive = useMobileResponsive()

  // Close menu when switching to desktop
  useEffect(() => {
    if (responsive.isDesktop && isOpen) {
      setIsOpen(false)
    }
  }, [responsive.isDesktop, isOpen])

  // Lock body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
      document.documentElement.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
      document.documentElement.style.overflow = ''
    }

    return () => {
      document.body.style.overflow = ''
      document.documentElement.style.overflow = ''
    }
  }, [isOpen])

  const toggle = () => setIsOpen(prev => !prev)
  const open = () => setIsOpen(true)
  const close = () => setIsOpen(false)

  return {
    isOpen,
    toggle,
    open,
    close,
    shouldShow: responsive.isMobile || responsive.isTablet,
  }
}

/**
 * Hook for viewport-based visibility
 */
export function useViewportVisibility(
  threshold: { mobile?: boolean; tablet?: boolean; desktop?: boolean } = {}
) {
  const responsive = useMobileResponsive()
  
  const isVisible = 
    (threshold.mobile !== false && responsive.isMobile) ||
    (threshold.tablet !== false && responsive.isTablet) ||
    (threshold.desktop !== false && responsive.isDesktop)

  return isVisible
}

/**
 * Hook for responsive values based on viewport
 */
export function useResponsiveValue<T>(values: {
  mobile?: T
  tablet?: T
  desktop?: T
  default: T
}): T {
  const responsive = useMobileResponsive()

  if (responsive.isMobile && values.mobile !== undefined) {
    return values.mobile
  }
  if (responsive.isTablet && values.tablet !== undefined) {
    return values.tablet
  }
  if (responsive.isDesktop && values.desktop !== undefined) {
    return values.desktop
  }

  return values.default
}