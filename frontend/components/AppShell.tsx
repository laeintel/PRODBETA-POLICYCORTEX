'use client'

import React, { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import PersistentSidebar from './PersistentSidebar'
import { ThemeProvider } from '@/contexts/ThemeContext'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  
  // Load collapsed state from localStorage on mount
  useEffect(() => {
    const savedState = localStorage.getItem('sidebarCollapsed')
    if (savedState === 'true') {
      setSidebarCollapsed(true)
    }
  }, [])

  // Listen for changes to localStorage to sync the state
  useEffect(() => {
    const handleStorageChange = () => {
      const savedState = localStorage.getItem('sidebarCollapsed')
      setSidebarCollapsed(savedState === 'true')
    }

    window.addEventListener('storage', handleStorageChange)
    // Also listen for custom events within the same tab
    window.addEventListener('sidebarToggle', handleStorageChange)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('sidebarToggle', handleStorageChange)
    }
  }, [])

  return (
    <ThemeProvider>
      <div className="min-h-screen relative">
        {/* Persistent Sidebar - appears on ALL pages including login */}
        <PersistentSidebar />
        
        {/* Main Content Area - with responsive margin */}
        <div className={`transition-all duration-300 ${
          sidebarCollapsed 
            ? 'ml-0 sm:ml-16' // On mobile, no margin when collapsed
            : 'ml-0 sm:ml-64' // On mobile, no margin when expanded
        }`}>
          {/* Mobile overlay when sidebar is expanded */}
          {!sidebarCollapsed && (
            <div 
              className="sm:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
              onClick={() => {
                localStorage.setItem('sidebarCollapsed', 'true');
                window.dispatchEvent(new Event('sidebarToggle'));
              }}
            />
          )}
          
          {/* Main Content */}
          <main 
            id="main-content"
            className="
            min-h-screen
            bg-gray-50 dark:bg-gray-900 
            transition-all duration-300
          ">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
              {children}
            </div>
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}


