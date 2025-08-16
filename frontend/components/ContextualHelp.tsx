"use client"

import { useState, useEffect, useRef } from 'react'
import { HelpCircle, X, ChevronLeft, ChevronRight, Play, Pause } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { cn } from '@/lib/utils'

// Help tooltip component
interface HelpTooltipProps {
  content: string | React.ReactNode
  title?: string
  children: React.ReactNode
  side?: 'top' | 'right' | 'bottom' | 'left'
  className?: string
}

export function HelpTooltip({ 
  content, 
  title, 
  children, 
  side = 'top',
  className 
}: HelpTooltipProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className={cn("inline-flex items-center gap-1", className)}>
          {children}
          <HelpCircle className="h-4 w-4 text-muted-foreground hover:text-foreground cursor-help" />
        </div>
      </PopoverTrigger>
      <PopoverContent side={side} className="w-80">
        {title && (
          <div className="font-semibold text-sm mb-2">{title}</div>
        )}
        <div className="text-sm text-muted-foreground">
          {content}
        </div>
      </PopoverContent>
    </Popover>
  )
}

// Tour step interface
export interface TourStep {
  id: string
  target: string // CSS selector
  title: string
  content: string | React.ReactNode
  placement?: 'top' | 'right' | 'bottom' | 'left'
  action?: () => void
  beforeShow?: () => void
  afterShow?: () => void
}

// Guided tour component
interface GuidedTourProps {
  steps: TourStep[]
  isOpen: boolean
  onClose: () => void
  onComplete?: () => void
  autoPlay?: boolean
  autoPlayDelay?: number
  showProgress?: boolean
  showSkip?: boolean
}

export function GuidedTour({
  steps,
  isOpen,
  onClose,
  onComplete,
  autoPlay = false,
  autoPlayDelay = 3000,
  showProgress = true,
  showSkip = true
}: GuidedTourProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(autoPlay)
  const [highlightedElement, setHighlightedElement] = useState<HTMLElement | null>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const autoPlayRef = useRef<NodeJS.Timeout>()

  const step = steps[currentStep]
  const isLastStep = currentStep === steps.length - 1
  const isFirstStep = currentStep === 0

  // Auto-play functionality
  useEffect(() => {
    if (isAutoPlaying && isOpen && !isLastStep) {
      autoPlayRef.current = setTimeout(() => {
        nextStep()
      }, autoPlayDelay)
    }

    return () => {
      if (autoPlayRef.current) {
        clearTimeout(autoPlayRef.current)
      }
    }
  }, [currentStep, isAutoPlaying, isOpen, autoPlayDelay, isLastStep])

  // Highlight target element
  useEffect(() => {
    if (!isOpen || !step) return

    const targetElement = document.querySelector(step.target) as HTMLElement
    if (targetElement) {
      setHighlightedElement(targetElement)
      
      // Scroll element into view
      targetElement.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
        inline: 'center'
      })

      // Call beforeShow callback
      step.beforeShow?.()

      // Add highlight styles
      targetElement.style.position = 'relative'
      targetElement.style.zIndex = '1001'
      targetElement.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.5)'
      targetElement.style.borderRadius = '4px'

      // Call afterShow callback
      setTimeout(() => {
        step.afterShow?.()
      }, 100)
    }

    return () => {
      if (targetElement) {
        targetElement.style.position = ''
        targetElement.style.zIndex = ''
        targetElement.style.boxShadow = ''
        targetElement.style.borderRadius = ''
      }
    }
  }, [step, isOpen])

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case 'ArrowRight':
        case 'ArrowDown':
          if (!isLastStep) nextStep()
          break
        case 'ArrowLeft':
        case 'ArrowUp':
          if (!isFirstStep) previousStep()
          break
        case ' ':
          e.preventDefault()
          toggleAutoPlay()
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, currentStep, isLastStep, isFirstStep])

  const nextStep = () => {
    if (isLastStep) {
      completeTour()
    } else {
      step.action?.()
      setCurrentStep(prev => prev + 1)
    }
  }

  const previousStep = () => {
    if (!isFirstStep) {
      setCurrentStep(prev => prev - 1)
    }
  }

  const goToStep = (stepIndex: number) => {
    setCurrentStep(stepIndex)
  }

  const toggleAutoPlay = () => {
    setIsAutoPlaying(prev => !prev)
  }

  const completeTour = () => {
    onComplete?.()
    onClose()
  }

  const skipTour = () => {
    onClose()
  }

  if (!isOpen || !step) return null

  const getTooltipPosition = () => {
    if (!highlightedElement) return { top: '50%', left: '50%' }

    const rect = highlightedElement.getBoundingClientRect()
    const placement = step.placement || 'bottom'

    switch (placement) {
      case 'top':
        return {
          top: rect.top - 10,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, -100%)'
        }
      case 'right':
        return {
          top: rect.top + rect.height / 2,
          left: rect.right + 10,
          transform: 'translateY(-50%)'
        }
      case 'bottom':
        return {
          top: rect.bottom + 10,
          left: rect.left + rect.width / 2,
          transform: 'translateX(-50%)'
        }
      case 'left':
        return {
          top: rect.top + rect.height / 2,
          left: rect.left - 10,
          transform: 'translate(-100%, -50%)'
        }
      default:
        return {
          top: rect.bottom + 10,
          left: rect.left + rect.width / 2,
          transform: 'translateX(-50%)'
        }
    }
  }

  return (
    <>
      {/* Overlay */}
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black/50 z-1000"
        onClick={onClose}
      />

      {/* Tour tooltip */}
      <Card
        className="fixed z-1002 w-80 shadow-lg"
        style={getTooltipPosition()}
      >
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">{step.title}</CardTitle>
            <div className="flex items-center gap-1">
              {autoPlay && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleAutoPlay}
                  className="h-6 w-6"
                >
                  {isAutoPlaying ? (
                    <Pause className="h-3 w-3" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-6 w-6"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          {showProgress && (
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full h-2">
                <div
                  className="bg-primary rounded-full h-2 transition-all"
                  style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                />
              </div>
              <Badge variant="outline" className="text-xs">
                {currentStep + 1} of {steps.length}
              </Badge>
            </div>
          )}
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div className="text-sm text-muted-foreground">
            {step.content}
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={previousStep}
                disabled={isFirstStep}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>
              
              {showSkip && !isLastStep && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={skipTour}
                  className="text-muted-foreground"
                >
                  Skip Tour
                </Button>
              )}
            </div>
            
            <Button
              size="sm"
              onClick={nextStep}
            >
              {isLastStep ? 'Finish' : 'Next'}
              {!isLastStep && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          </div>
        </CardContent>
      </Card>
    </>
  )
}

// Help center component
interface HelpArticle {
  id: string
  title: string
  content: string
  category: string
  tags: string[]
  helpful?: number
  views?: number
}

interface HelpCenterProps {
  articles: HelpArticle[]
  categories: string[]
  onSearch?: (query: string) => void
  onRateArticle?: (articleId: string, helpful: boolean) => void
}

export function HelpCenter({ 
  articles, 
  categories, 
  onSearch, 
  onRateArticle 
}: HelpCenterProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedArticle, setSelectedArticle] = useState<HelpArticle | null>(null)

  const filteredArticles = articles.filter(article => {
    const matchesCategory = selectedCategory === 'all' || article.category === selectedCategory
    const matchesSearch = !searchQuery || 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    return matchesCategory && matchesSearch
  })

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    onSearch?.(query)
  }

  const handleRateArticle = (helpful: boolean) => {
    if (selectedArticle) {
      onRateArticle?.(selectedArticle.id, helpful)
    }
  }

  if (selectedArticle) {
    return (
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <Button
                variant="ghost"
                onClick={() => setSelectedArticle(null)}
                className="mb-2"
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Back to Help Center
              </Button>
              <CardTitle>{selectedArticle.title}</CardTitle>
            </div>
            <Badge variant="outline">{selectedArticle.category}</Badge>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div className="prose prose-sm max-w-none">
            {selectedArticle.content.split('\n').map((paragraph, index) => (
              <p key={index} className="mb-4">{paragraph}</p>
            ))}
          </div>
          
          <div className="border-t pt-4">
            <h4 className="font-medium mb-2">Was this helpful?</h4>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleRateArticle(true)}
              >
                👍 Yes
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleRateArticle(false)}
              >
                👎 No
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Help Center</CardTitle>
        
        {/* Search */}
        <div className="space-y-4">
          <input
            type="text"
            placeholder="Search help articles..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          />
          
          {/* Categories */}
          <div className="flex flex-wrap gap-2">
            <Button
              variant={selectedCategory === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory('all')}
            >
              All
            </Button>
            {categories.map(category => (
              <Button
                key={category}
                variant={selectedCategory === category ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory(category)}
              >
                {category}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          {filteredArticles.map(article => (
            <div
              key={article.id}
              className="border rounded-lg p-4 hover:bg-muted/50 cursor-pointer"
              onClick={() => setSelectedArticle(article)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-medium mb-1">{article.title}</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    {article.content.substring(0, 150)}...
                  </p>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {article.category}
                    </Badge>
                    {article.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                {(article.views || article.helpful) && (
                  <div className="text-xs text-muted-foreground text-right">
                    {article.views && <div>{article.views} views</div>}
                    {article.helpful && <div>👍 {article.helpful}</div>}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {filteredArticles.length === 0 && (
            <div className="text-center text-muted-foreground py-8">
              No articles found matching your search.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}