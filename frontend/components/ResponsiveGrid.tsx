'use client'

import { ReactNode } from 'react'

interface ResponsiveGridProps {
  children: ReactNode
  className?: string
  variant?: 'cards' | 'metrics' | 'tables' | 'charts'
}

export default function ResponsiveGrid({ 
  children, 
  className = '', 
  variant = 'cards' 
}: ResponsiveGridProps) {
  
  const gridClasses = {
    cards: `
      grid gap-4 sm:gap-6 xl:gap-8 2xl:gap-10
      grid-cols-1 
      sm:grid-cols-2 
      lg:grid-cols-3 
      xl:grid-cols-4 
      2xl:grid-cols-5
      3xl:grid-cols-6
      4xl:grid-cols-8
      5xl:grid-cols-10
    `,
    metrics: `
      grid gap-3 sm:gap-4 xl:gap-6 2xl:gap-8
      grid-cols-2 
      sm:grid-cols-3 
      lg:grid-cols-4 
      xl:grid-cols-5 
      2xl:grid-cols-6
      3xl:grid-cols-8
      4xl:grid-cols-10
      5xl:grid-cols-12
    `,
    tables: `
      grid gap-4 sm:gap-6 xl:gap-8
      grid-cols-1 
      xl:grid-cols-2 
      3xl:grid-cols-3
      5xl:grid-cols-4
    `,
    charts: `
      grid gap-4 sm:gap-6 xl:gap-8 2xl:gap-10
      grid-cols-1 
      lg:grid-cols-2 
      2xl:grid-cols-3
      4xl:grid-cols-4
      5xl:grid-cols-5
    `
  }
  
  return (
    <div className={`${gridClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}

// Responsive container component for consistent max-widths
export function ResponsiveContainer({ 
  children, 
  className = '' 
}: { 
  children: ReactNode
  className?: string 
}) {
  return (
    <div className={`
      w-full 
      mx-auto
      px-4 sm:px-6 lg:px-8 xl:px-12 2xl:px-16 3xl:px-20 4xl:px-24 5xl:px-32
      max-w-full
      3xl:max-w-[2400px] 4xl:max-w-[3200px] 5xl:max-w-[3800px]
      ${className}
    `}>
      {children}
    </div>
  )
}

// Responsive text sizing utility
export function ResponsiveText({ 
  children, 
  variant = 'body',
  className = '' 
}: { 
  children: ReactNode
  variant?: 'title' | 'heading' | 'subheading' | 'body' | 'small'
  className?: string 
}) {
  const textClasses = {
    title: 'text-2xl sm:text-3xl lg:text-4xl xl:text-5xl 2xl:text-6xl 3xl:text-7xl font-bold',
    heading: 'text-xl sm:text-2xl lg:text-3xl xl:text-4xl 2xl:text-5xl 3xl:text-6xl font-semibold',
    subheading: 'text-lg sm:text-xl lg:text-2xl xl:text-3xl 2xl:text-4xl 3xl:text-5xl font-medium',
    body: 'text-sm sm:text-base lg:text-lg xl:text-xl 2xl:text-2xl 3xl:text-3xl',
    small: 'text-xs sm:text-sm lg:text-base xl:text-lg 2xl:text-xl 3xl:text-2xl'
  }
  
  return (
    <div className={`${textClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}