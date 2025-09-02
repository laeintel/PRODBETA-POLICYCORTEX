import { cn } from '@/lib/utils'

interface PageContainerProps {
  children: React.ReactNode
  className?: string
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl' | 'full'
  noPadding?: boolean
}

export default function PageContainer({ 
  children, 
  className,
  maxWidth = '7xl',
  noPadding = false
}: PageContainerProps) {
  const maxWidthClasses = {
    'sm': 'max-w-screen-sm',
    'md': 'max-w-screen-md', 
    'lg': 'max-w-screen-lg',
    'xl': 'max-w-screen-xl',
    '2xl': 'max-w-screen-2xl',
    '3xl': 'max-w-[1920px]',
    '4xl': 'max-w-[2048px]',
    '5xl': 'max-w-[2560px]',
    '6xl': 'max-w-[3072px]',
    '7xl': 'max-w-[3584px]',
    'full': 'max-w-full'
  }

  return (
    <div className={cn(
      'mx-auto w-full',
      maxWidthClasses[maxWidth],
      !noPadding && 'px-4 sm:px-6 lg:px-8',
      className
    )}>
      {children}
    </div>
  )
}