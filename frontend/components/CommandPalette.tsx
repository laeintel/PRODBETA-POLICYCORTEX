"use client"

import { useState, useEffect, useCallback, useMemo } from 'react'
import { 
  Search, Command, ArrowRight, Hash, User, Settings, 
  FileText, Database, Shield, Activity, Clock, Star 
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { useKeyboardShortcuts } from './KeyboardNavigation'

// Command item interface
export interface CommandItem {
  id: string
  title: string
  description?: string
  icon?: React.ReactNode
  category: string
  keywords?: string[]
  action: () => void
  shortcut?: string
  href?: string
  metadata?: Record<string, any>
}

// Command category interface
export interface CommandCategory {
  id: string
  title: string
  icon?: React.ReactNode
  priority?: number
}

// Recent command interface
interface RecentCommand {
  id: string
  timestamp: Date
  count: number
}

// Command palette props
interface CommandPaletteProps {
  commands: CommandItem[]
  categories: CommandCategory[]
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  placeholder?: string
  recentCommands?: RecentCommand[]
  onCommandExecute?: (command: CommandItem) => void
  maxResults?: number
  showRecent?: boolean
  showCategories?: boolean
}

export function CommandPalette({
  commands,
  categories,
  isOpen,
  onOpenChange,
  placeholder = "Type a command or search...",
  recentCommands = [],
  onCommandExecute,
  maxResults = 50,
  showRecent = true,
  showCategories = true
}: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [recentCommandsState, setRecentCommands] = useState<RecentCommand[]>(recentCommands)

  // Filter and sort commands
  const filteredCommands = useMemo(() => {
    if (!query.trim()) {
      // Show recent commands when no query
      if (showRecent && recentCommandsState.length > 0) {
        const recentIds = new Set(recentCommandsState.map(r => r.id))
        return commands
          .filter(cmd => recentIds.has(cmd.id))
          .sort((a, b) => {
            const aRecent = recentCommandsState.find(r => r.id === a.id)
            const bRecent = recentCommandsState.find(r => r.id === b.id)
            return (bRecent?.timestamp.getTime() || 0) - (aRecent?.timestamp.getTime() || 0)
          })
          .slice(0, 10)
      }
      return commands.slice(0, 20)
    }

    const lowerQuery = query.toLowerCase()
    
    return commands
      .map(command => {
        let score = 0
        
        // Title match (highest priority)
        if (command.title.toLowerCase().includes(lowerQuery)) {
          score += command.title.toLowerCase().indexOf(lowerQuery) === 0 ? 100 : 50
        }
        
        // Description match
        if (command.description?.toLowerCase().includes(lowerQuery)) {
          score += 20
        }
        
        // Keywords match
        if (command.keywords?.some(keyword => keyword.toLowerCase().includes(lowerQuery))) {
          score += 30
        }
        
        // Category match
        const category = categories.find(cat => cat.id === command.category)
        if (category?.title.toLowerCase().includes(lowerQuery)) {
          score += 10
        }
        
        // Recent usage boost
        const recentCommand = recentCommandsState.find(r => r.id === command.id)
        if (recentCommand) {
          score += Math.min(recentCommand.count * 5, 25)
        }
        
        return { command, score }
      })
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults)
      .map(({ command }) => command)
  }, [query, commands, categories, recentCommandsState, maxResults, showRecent])

  // Group commands by category
  const groupedCommands = useMemo(() => {
    if (!showCategories) {
      return [{ category: null, commands: filteredCommands }]
    }

    const groups = new Map<string, CommandItem[]>()
    
    filteredCommands.forEach(command => {
      const categoryId = command.category
      if (!groups.has(categoryId)) {
        groups.set(categoryId, [])
      }
      groups.get(categoryId)!.push(command)
    })

    return Array.from(groups.entries())
      .map(([categoryId, commands]) => ({
        category: categories.find(cat => cat.id === categoryId) || null,
        commands
      }))
      .sort((a, b) => (a.category?.priority || 0) - (b.category?.priority || 0))
  }, [filteredCommands, categories, showCategories])

  // Reset selection when commands change
  useEffect(() => {
    setSelectedIndex(0)
  }, [filteredCommands])

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1))
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex(prev => Math.max(prev - 1, 0))
          break
        case 'Enter':
          e.preventDefault()
          const selectedCommand = filteredCommands[selectedIndex]
          if (selectedCommand) {
            executeCommand(selectedCommand)
          }
          break
        case 'Escape':
          e.preventDefault()
          onOpenChange(false)
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, filteredCommands, selectedIndex, onOpenChange])

  // Execute command
  const executeCommand = useCallback((command: CommandItem) => {
    // Add to recent commands
    setRecentCommands(prev => {
      const existing = prev.find(r => r.id === command.id)
      const updated = prev.filter(r => r.id !== command.id)
      
      if (existing) {
        updated.unshift({
          ...existing,
          timestamp: new Date(),
          count: existing.count + 1
        })
      } else {
        updated.unshift({
          id: command.id,
          timestamp: new Date(),
          count: 1
        })
      }
      
      return updated.slice(0, 50) // Keep only 50 recent commands
    })

    // Execute the command
    command.action()
    onCommandExecute?.(command)
    
    // Close palette
    onOpenChange(false)
    setQuery('')
  }, [onCommandExecute, onOpenChange])

  // Reset state when dialog closes
  useEffect(() => {
    if (!isOpen) {
      setQuery('')
      setSelectedIndex(0)
    }
  }, [isOpen])

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl p-0 gap-0">
        <DialogHeader className="p-4 pb-2">
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={placeholder}
              className="border-0 focus-visible:ring-0 text-base"
              autoFocus
            />
          </div>
        </DialogHeader>

        <div className="max-h-96 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No commands found</p>
              {query && (
                <p className="text-sm mt-1">Try a different search term</p>
              )}
            </div>
          ) : (
            <div className="p-2">
              {groupedCommands.map(({ category, commands }, groupIndex) => (
                <div key={category?.id || 'uncategorized'}>
                  {category && showCategories && (
                    <div className="flex items-center gap-2 px-2 py-1 text-xs font-medium text-muted-foreground">
                      {category.icon}
                      {category.title}
                    </div>
                  )}
                  
                  {commands.map((command, commandIndex) => {
                    const globalIndex = groupedCommands
                      .slice(0, groupIndex)
                      .reduce((acc, group) => acc + group.commands.length, 0) + commandIndex
                    
                    const isSelected = globalIndex === selectedIndex
                    const isRecent = recentCommandsState.some(r => r.id === command.id)
                    
                    return (
                      <div
                        key={command.id}
                        onClick={() => executeCommand(command)}
                        className={cn(
                          "flex items-center gap-3 px-3 py-2 rounded-md cursor-pointer transition-colors",
                          isSelected && "bg-accent"
                        )}
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          {command.icon && (
                            <div className="text-muted-foreground">
                              {command.icon}
                            </div>
                          )}
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-medium truncate">
                                {command.title}
                              </span>
                              {isRecent && (
                                <Clock className="h-3 w-3 text-muted-foreground" />
                              )}
                            </div>
                            {command.description && (
                              <p className="text-sm text-muted-foreground truncate">
                                {command.description}
                              </p>
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          {command.shortcut && (
                            <Badge variant="outline" className="text-xs font-mono">
                              {command.shortcut}
                            </Badge>
                          )}
                          <ArrowRight className="h-3 w-3 text-muted-foreground" />
                        </div>
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer with shortcuts help */}
        <div className="border-t p-3 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Badge variant="outline" className="text-xs">↵</Badge>
              to select
            </span>
            <span className="flex items-center gap-1">
              <Badge variant="outline" className="text-xs">↑↓</Badge>
              to navigate
            </span>
          </div>
          <span className="flex items-center gap-1">
            <Badge variant="outline" className="text-xs">esc</Badge>
            to close
          </span>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Hook for command palette with keyboard shortcut
export function useCommandPalette(commands: CommandItem[], categories: CommandCategory[] = []) {
  const [isOpen, setIsOpen] = useState(false)

  // Global keyboard shortcut (Cmd+K / Ctrl+K)
  useKeyboardShortcuts({
    'cmd+k': () => setIsOpen(true),
    'ctrl+k': () => setIsOpen(true),
  })

  return {
    isOpen,
    openPalette: () => setIsOpen(true),
    closePalette: () => setIsOpen(false),
    CommandPalette: (props: Partial<CommandPaletteProps>) => (
      <CommandPalette
        commands={commands}
        categories={categories}
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        {...props}
      />
    )
  }
}

// Predefined command categories
export const DEFAULT_CATEGORIES: CommandCategory[] = [
  { id: 'navigation', title: 'Navigation', icon: <ArrowRight className="h-3 w-3" />, priority: 1 },
  { id: 'actions', title: 'Actions', icon: <Command className="h-3 w-3" />, priority: 2 },
  { id: 'search', title: 'Search', icon: <Search className="h-3 w-3" />, priority: 3 },
  { id: 'settings', title: 'Settings', icon: <Settings className="h-3 w-3" />, priority: 4 },
  { id: 'resources', title: 'Resources', icon: <Database className="h-3 w-3" />, priority: 5 },
  { id: 'security', title: 'Security', icon: <Shield className="h-3 w-3" />, priority: 6 },
  { id: 'monitoring', title: 'Monitoring', icon: <Activity className="h-3 w-3" />, priority: 7 },
]

// Example usage
export function CommandPaletteExample() {
  const exampleCommands: CommandItem[] = [
    {
      id: 'dashboard',
      title: 'Go to Dashboard',
      description: 'View the main dashboard',
      icon: <Activity className="h-4 w-4" />,
      category: 'navigation',
      shortcut: 'Ctrl+D',
      action: () => console.log('Navigate to dashboard'),
      keywords: ['home', 'overview']
    },
    {
      id: 'create-policy',
      title: 'Create New Policy',
      description: 'Create a new governance policy',
      icon: <FileText className="h-4 w-4" />,
      category: 'actions',
      shortcut: 'Ctrl+N',
      action: () => console.log('Create policy'),
      keywords: ['new', 'add', 'governance']
    },
    {
      id: 'search-resources',
      title: 'Search Resources',
      description: 'Find Azure resources',
      icon: <Database className="h-4 w-4" />,
      category: 'search',
      action: () => console.log('Search resources'),
      keywords: ['find', 'azure', 'vm', 'storage']
    }
  ]

  const { isOpen, openPalette, CommandPalette: Palette } = useCommandPalette(exampleCommands, DEFAULT_CATEGORIES)

  return (
    <div>
      <Button onClick={openPalette} className="flex items-center gap-2">
        <Search className="h-4 w-4" />
        Search... 
        <Badge variant="outline" className="ml-2">⌘K</Badge>
      </Button>
      
      <Palette />
    </div>
  )
}