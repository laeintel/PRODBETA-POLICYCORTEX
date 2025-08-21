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
      grid gap-4 sm:gap-6
      grid-cols-1 
      sm:grid-cols-2 
      lg:grid-cols-3 
      xl:grid-cols-4 
      2xl:grid-cols-4
    `,
    metrics: `
      grid gap-3 sm:gap-4
      grid-cols-2 
      sm:grid-cols-3 
      lg:grid-cols-4 
      xl:grid-cols-6 
      2xl:grid-cols-6
    `,
    tables: `
      grid gap-4 sm:gap-6
      grid-cols-1 
      xl:grid-cols-2 
    `,
    charts: `
      grid gap-4 sm:gap-6
      grid-cols-1 
      lg:grid-cols-2 
      2xl:grid-cols-3
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
      px-4 sm:px-6 lg:px-8
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
    title: 'text-2xl sm:text-3xl lg:text-4xl font-bold',
    heading: 'text-xl sm:text-2xl lg:text-3xl font-semibold',
    subheading: 'text-lg sm:text-xl lg:text-2xl font-medium',
    body: 'text-sm sm:text-base lg:text-lg',
    small: 'text-xs sm:text-sm'
  }
  
  return (
    <div className={`${textClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}