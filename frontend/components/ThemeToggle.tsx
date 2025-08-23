'use client'

import React from 'react'
import { Sun, Moon } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <button type="button"
      onClick={toggleTheme}
      className="
        relative p-2 rounded-lg transition-all duration-300
        bg-gray-800 dark:bg-gray-700 
        hover:bg-gray-700 dark:hover:bg-gray-600
        text-gray-400 dark:text-gray-300
        hover:text-yellow-500 dark:hover:text-blue-400
        focus:outline-none focus:ring-2 focus:ring-offset-2 
        focus:ring-blue-500 dark:focus:ring-yellow-500
        focus:ring-offset-gray-900 dark:focus:ring-offset-gray-800
      "
      aria-label="Toggle theme"
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      <div className="relative w-5 h-5">
        {/* Sun icon - visible in dark mode */}
        <Sun 
          className={`
            absolute inset-0 w-5 h-5 transition-all duration-300
            ${theme === 'dark' 
              ? 'opacity-100 rotate-0 scale-100' 
              : 'opacity-0 rotate-180 scale-0'
            }
          `}
        />
        
        {/* Moon icon - visible in light mode */}
        <Moon 
          className={`
            absolute inset-0 w-5 h-5 transition-all duration-300
            ${theme === 'light' 
              ? 'opacity-100 rotate-0 scale-100' 
              : 'opacity-0 -rotate-180 scale-0'
            }
          `}
        />
      </div>
    </button>
  )
}