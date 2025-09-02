import { cn } from '@/lib/utils'

interface PageContainerProps {
  children: React.ReactNode
  className?: string
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full'
  noPadding?: boolean
}

// Splunk-grade dense container with proper constraints
export default function PageContainer({ 
  children, 
  className,
  maxWidth = '2xl', // Default to 2xl for readable line length
  noPadding = false
}: PageContainerProps) {
  const maxWidthClasses = {
    'sm': 'max-w-screen-sm',
    'md': 'max-w-screen-md', 
    'lg': 'max-w-screen-lg',
    'xl': 'max-w-screen-xl',
    '2xl': 'max-w-screen-2xl',
    'full': 'max-w-full'
  }

  return (
    <div className={cn(
      'mx-auto w-full min-w-0', // min-w-0 prevents overflow
      maxWidthClasses[maxWidth],
      !noPadding && 'px-4 sm:px-6 lg:px-8',
      className
    )}>
      {children}
    </div>
  )
}