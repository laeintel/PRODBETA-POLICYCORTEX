"use client"

import { useState, useCallback, useMemo } from 'react'
import { Check, X, Download, Trash2, Edit, Play, Pause, MoreHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

// Bulk action interface
export interface BulkAction {
  id: string
  label: string
  icon?: React.ReactNode
  variant?: 'default' | 'destructive' | 'outline'
  requiresConfirmation?: boolean
  confirmationTitle?: string
  confirmationMessage?: string
  disabled?: (selectedItems: any[]) => boolean
  action: (selectedItems: any[], options?: any) => Promise<void>
}

// Bulk selection hook
export function useBulkSelection<T extends { id: string }>(items: T[]) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const selectedItems = useMemo(() => 
    items.filter(item => selectedIds.has(item.id)),
    [items, selectedIds]
  )

  const isSelected = useCallback((id: string) => selectedIds.has(id), [selectedIds])

  const isAllSelected = useMemo(() => 
    items.length > 0 && items.every(item => selectedIds.has(item.id)),
    [items, selectedIds]
  )

  const isIndeterminate = useMemo(() => 
    selectedIds.size > 0 && selectedIds.size < items.length,
    [selectedIds.size, items.length]
  )

  const toggleItem = useCallback((id: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }, [])

  const toggleAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size === items.length) {
        return new Set()
      } else {
        return new Set(items.map(item => item.id))
      }
    })
  }, [items])

  const selectItems = useCallback((ids: string[]) => {
    setSelectedIds(new Set(ids))
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set())
  }, [])

  return {
    selectedIds,
    selectedItems,
    isSelected,
    isAllSelected,
    isIndeterminate,
    toggleItem,
    toggleAll,
    selectItems,
    clearSelection,
    selectedCount: selectedIds.size
  }
}

// Bulk action toolbar component
interface BulkActionToolbarProps<T> {
  selectedItems: T[]
  actions: BulkAction[]
  onClearSelection: () => void
  className?: string
}

export function BulkActionToolbar<T>({ 
  selectedItems, 
  actions, 
  onClearSelection,
  className 
}: BulkActionToolbarProps<T>) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionProgress, setExecutionProgress] = useState(0)
  const [confirmAction, setConfirmAction] = useState<BulkAction | null>(null)

  const availableActions = useMemo(() =>
    actions.filter(action => !action.disabled?.(selectedItems)),
    [actions, selectedItems]
  )

  const executeAction = async (action: BulkAction) => {
    if (action.requiresConfirmation) {
      setConfirmAction(action)
      return
    }

    await performAction(action)
  }

  const performAction = async (action: BulkAction) => {
    setIsExecuting(true)
    setExecutionProgress(0)

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setExecutionProgress(prev => Math.min(prev + 10, 90))
      }, 100)

      await action.action(selectedItems)
      
      clearInterval(progressInterval)
      setExecutionProgress(100)
      
      setTimeout(() => {
        onClearSelection()
        setExecutionProgress(0)
      }, 500)
    } catch (error) {
      console.error('Bulk action failed:', error)
      // Handle error (show toast, etc.)
    } finally {
      setIsExecuting(false)
      setConfirmAction(null)
    }
  }

  if (selectedItems.length === 0) return null

  return (
    <>
      <div className={cn(
        "flex items-center gap-3 p-3 bg-muted/50 border rounded-lg",
        className
      )}>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">
            {selectedItems.length} selected
          </Badge>
          
          {isExecuting && (
            <div className="flex items-center gap-2">
              <Progress value={executionProgress} className="w-20 h-2" />
              <span className="text-sm text-muted-foreground">
                {executionProgress}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-1">
          {availableActions.slice(0, 3).map(action => (
            <Button
              key={action.id}
              variant={action.variant || 'outline'}
              size="sm"
              onClick={() => executeAction(action)}
              disabled={isExecuting}
              className="flex items-center gap-1"
            >
              {action.icon}
              {action.label}
            </Button>
          ))}

          {availableActions.length > 3 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>More Actions</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {availableActions.slice(3).map(action => (
                  <DropdownMenuItem
                    key={action.id}
                    onClick={() => executeAction(action)}
                    disabled={isExecuting}
                  >
                    {action.icon}
                    {action.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}

          <Button
            variant="ghost"
            size="sm"
            onClick={onClearSelection}
            disabled={isExecuting}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Confirmation dialog */}
      <Dialog open={!!confirmAction} onOpenChange={() => setConfirmAction(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {confirmAction?.confirmationTitle || 'Confirm Action'}
            </DialogTitle>
            <DialogDescription>
              {confirmAction?.confirmationMessage || 
               `Are you sure you want to perform this action on ${selectedItems.length} items?`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmAction(null)}>
              Cancel
            </Button>
            <Button 
              variant={confirmAction?.variant || 'default'}
              onClick={() => confirmAction && performAction(confirmAction)}
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

// Selectable list item component
interface SelectableListItemProps {
  id: string
  isSelected: boolean
  onToggle: (id: string) => void
  children: React.ReactNode
  className?: string
}

export function SelectableListItem({
  id,
  isSelected,
  onToggle,
  children,
  className
}: SelectableListItemProps) {
  return (
    <div 
      className={cn(
        "flex items-center gap-3 p-3 border rounded-lg transition-colors hover:bg-muted/50",
        isSelected && "bg-muted/50 border-primary",
        className
      )}
    >
      <Checkbox
        checked={isSelected}
        onCheckedChange={() => onToggle(id)}
      />
      <div className="flex-1">
        {children}
      </div>
    </div>
  )
}

// Bulk select header component
interface BulkSelectHeaderProps {
  isAllSelected: boolean
  isIndeterminate: boolean
  onToggleAll: () => void
  label?: string
  className?: string
}

export function BulkSelectHeader({
  isAllSelected,
  isIndeterminate,
  onToggleAll,
  label = "Select all",
  className
}: BulkSelectHeaderProps) {
  return (
    <div className={cn("flex items-center gap-3 p-3 border-b", className)}>
      <Checkbox
        checked={isAllSelected}
        ref={(el) => {
          if (el) el.indeterminate = isIndeterminate
        }}
        onCheckedChange={onToggleAll}
      />
      <span className="text-sm font-medium">{label}</span>
    </div>
  )
}

// Example usage component
interface ExampleItem {
  id: string
  name: string
  status: 'active' | 'inactive'
  type: string
}

export function BulkActionsExample() {
  const [items] = useState<ExampleItem[]>([
    { id: '1', name: 'Item 1', status: 'active', type: 'resource' },
    { id: '2', name: 'Item 2', status: 'inactive', type: 'policy' },
    { id: '3', name: 'Item 3', status: 'active', type: 'resource' },
  ])

  const {
    selectedItems,
    isSelected,
    isAllSelected,
    isIndeterminate,
    toggleItem,
    toggleAll,
    clearSelection
  } = useBulkSelection(items)

  const actions: BulkAction[] = [
    {
      id: 'activate',
      label: 'Activate',
      icon: <Play className="h-4 w-4" />,
      variant: 'default',
      disabled: (items) => items.every(item => item.status === 'active'),
      action: async (items) => {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 2000))
        console.log('Activated items:', items)
      }
    },
    {
      id: 'deactivate',
      label: 'Deactivate',
      icon: <Pause className="h-4 w-4" />,
      variant: 'outline',
      disabled: (items) => items.every(item => item.status === 'inactive'),
      action: async (items) => {
        await new Promise(resolve => setTimeout(resolve, 2000))
        console.log('Deactivated items:', items)
      }
    },
    {
      id: 'delete',
      label: 'Delete',
      icon: <Trash2 className="h-4 w-4" />,
      variant: 'destructive',
      requiresConfirmation: true,
      confirmationTitle: 'Delete Items',
      confirmationMessage: 'This action cannot be undone. Are you sure you want to delete the selected items?',
      action: async (items) => {
        await new Promise(resolve => setTimeout(resolve, 2000))
        console.log('Deleted items:', items)
      }
    },
    {
      id: 'export',
      label: 'Export',
      icon: <Download className="h-4 w-4" />,
      variant: 'outline',
      action: async (items) => {
        await new Promise(resolve => setTimeout(resolve, 1000))
        console.log('Exported items:', items)
      }
    }
  ]

  return (
    <div className="space-y-4">
      <BulkActionToolbar
        selectedItems={selectedItems}
        actions={actions}
        onClearSelection={clearSelection}
      />

      <div className="border rounded-lg">
        <BulkSelectHeader
          isAllSelected={isAllSelected}
          isIndeterminate={isIndeterminate}
          onToggleAll={toggleAll}
        />

        <div className="divide-y">
          {items.map(item => (
            <SelectableListItem
              key={item.id}
              id={item.id}
              isSelected={isSelected(item.id)}
              onToggle={toggleItem}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium">{item.name}</h4>
                  <p className="text-sm text-muted-foreground">{item.type}</p>
                </div>
                <Badge variant={item.status === 'active' ? 'default' : 'secondary'}>
                  {item.status}
                </Badge>
              </div>
            </SelectableListItem>
          ))}
        </div>
      </div>
    </div>
  )
}