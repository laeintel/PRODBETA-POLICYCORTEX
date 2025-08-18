"use client"

import { useState, useCallback, useMemo } from 'react'
import { DndContext, DragEndEvent, DragOverlay, DragStartEvent, closestCenter } from '@dnd-kit/core'
import { SortableContext, arrayMove, rectSortingStrategy } from '@dnd-kit/sortable'
import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { GripVertical, Plus, Settings, Trash2, Copy, Eye, EyeOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
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
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { cn } from '@/lib/utils'

// Widget types and interfaces
export type WidgetType = 'chart' | 'metric' | 'table' | 'alert' | 'status' | 'progress' | 'custom'

export interface WidgetConfig {
  id: string
  type: WidgetType
  title: string
  description?: string
  size: 'small' | 'medium' | 'large' | 'xlarge'
  data?: any
  settings?: Record<string, any>
  visible?: boolean
  position?: { x: number; y: number }
}

export interface DashboardLayout {
  id: string
  name: string
  description?: string
  widgets: WidgetConfig[]
  settings?: {
    columns?: number
    gap?: number
    theme?: string
  }
}

// Widget size mappings
const WIDGET_SIZES = {
  small: 'col-span-1 row-span-1',
  medium: 'col-span-2 row-span-1',
  large: 'col-span-2 row-span-2',
  xlarge: 'col-span-4 row-span-2'
}

// Available widget templates
const WIDGET_TEMPLATES: Omit<WidgetConfig, 'id'>[] = [
  {
    type: 'metric',
    title: 'Compliance Score',
    description: 'Overall compliance percentage',
    size: 'small',
    settings: { format: 'percentage', color: 'green' }
  },
  {
    type: 'chart',
    title: 'Resource Trends',
    description: 'Resource usage over time',
    size: 'medium',
    settings: { chartType: 'line', timeRange: '7d' }
  },
  {
    type: 'table',
    title: 'Recent Violations',
    description: 'Latest policy violations',
    size: 'large',
    settings: { maxRows: 10, sortBy: 'timestamp' }
  },
  {
    type: 'alert',
    title: 'Critical Alerts',
    description: 'High priority security alerts',
    size: 'medium',
    settings: { severity: 'critical', maxAlerts: 5 }
  },
  {
    type: 'status',
    title: 'Service Health',
    description: 'System component status',
    size: 'small',
    settings: { showDetails: true }
  },
  {
    type: 'progress',
    title: 'Migration Progress',
    description: 'Cloud migration completion',
    size: 'medium',
    settings: { showPercentage: true, color: 'blue' }
  }
]

// Sortable widget component
interface SortableWidgetProps {
  widget: WidgetConfig
  onEdit: (widget: WidgetConfig) => void
  onDelete: (id: string) => void
  onDuplicate: (widget: WidgetConfig) => void
  onToggleVisibility: (id: string) => void
  isEditing?: boolean
}

function SortableWidget({ 
  widget, 
  onEdit, 
  onDelete, 
  onDuplicate, 
  onToggleVisibility,
  isEditing = false 
}: SortableWidgetProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: widget.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "group relative",
        WIDGET_SIZES[widget.size],
        !widget.visible && "opacity-50"
      )}
    >
      <Card className={cn(
        "h-full transition-all",
        isDragging && "shadow-lg rotate-3",
        !widget.visible && "bg-muted/50"
      )}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <CardTitle className="text-sm truncate">{widget.title}</CardTitle>
              {widget.description && (
                <p className="text-xs text-muted-foreground truncate">
                  {widget.description}
                </p>
              )}
            </div>
            
            {isEditing && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 opacity-0 group-hover:opacity-100"
                  {...attributes}
                  {...listeners}
                >
                  <GripVertical className="h-3 w-3" />
                </Button>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 opacity-0 group-hover:opacity-100"
                    >
                      <Settings className="h-3 w-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>Widget Actions</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => onEdit(widget)}>
                      <Settings className="h-3 w-3 mr-1" />
                      Edit Settings
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onDuplicate(widget)}>
                      <Copy className="h-3 w-3 mr-1" />
                      Duplicate
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onToggleVisibility(widget.id)}>
                      {widget.visible ? (
                        <>
                          <EyeOff className="h-3 w-3 mr-1" />
                          Hide
                        </>
                      ) : (
                        <>
                          <Eye className="h-3 w-3 mr-1" />
                          Show
                        </>
                      )}
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem 
                      onClick={() => onDelete(widget.id)}
                      className="text-destructive"
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            <Badge variant="outline" className="text-xs">
              {widget.type}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {widget.size}
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="pt-0">
          <WidgetRenderer widget={widget} />
        </CardContent>
      </Card>
    </div>
  )
}

// Widget renderer component
function WidgetRenderer({ widget }: { widget: WidgetConfig }) {
  switch (widget.type) {
    case 'metric':
      return (
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">92%</div>
          <div className="text-xs text-muted-foreground">Compliance Score</div>
        </div>
      )
    
    case 'chart':
      return (
        <div className="h-20 bg-muted rounded flex items-center justify-center">
          <span className="text-xs text-muted-foreground">Chart Placeholder</span>
        </div>
      )
    
    case 'table':
      return (
        <div className="space-y-1">
          {[1, 2, 3].map(i => (
            <div key={i} className="flex justify-between text-xs">
              <span>Item {i}</span>
              <Badge variant="outline">Status</Badge>
            </div>
          ))}
        </div>
      )
    
    case 'alert':
      return (
        <div className="space-y-1">
          <div className="text-xs text-destructive">🚨 Critical Alert</div>
          <div className="text-xs text-yellow-600">⚠️ Warning Alert</div>
          <div className="text-xs text-muted-foreground">ℹ️ Info Alert</div>
        </div>
      )
    
    case 'status':
      return (
        <div className="flex items-center justify-center">
          <div className="h-3 w-3 bg-green-500 rounded-full"></div>
          <span className="text-xs ml-2">Healthy</span>
        </div>
      )
    
    case 'progress':
      return (
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span>Progress</span>
            <span>78%</span>
          </div>
          <div className="h-2 bg-muted rounded">
            <div className="h-2 bg-primary rounded" style={{ width: '78%' }}></div>
          </div>
        </div>
      )
    
    default:
      return (
        <div className="h-16 bg-muted/50 rounded flex items-center justify-center">
          <span className="text-xs text-muted-foreground">Custom Widget</span>
        </div>
      )
  }
}

// Widget configuration dialog
interface WidgetConfigDialogProps {
  widget: WidgetConfig | null
  isOpen: boolean
  onClose: () => void
  onSave: (widget: WidgetConfig) => void
}

function WidgetConfigDialog({ widget, isOpen, onClose, onSave }: WidgetConfigDialogProps) {
  const [config, setConfig] = useState<WidgetConfig | null>(widget)

  const handleSave = () => {
    if (config) {
      onSave(config)
      onClose()
    }
  }

  if (!config) return null

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Configure Widget</DialogTitle>
          <DialogDescription>
            Customize the widget settings and appearance.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          <div>
            <Label htmlFor="title">Title</Label>
            <Input
              id="title"
              value={config.title}
              onChange={(e) => setConfig({ ...config, title: e.target.value })}
            />
          </div>
          
          <div>
            <Label htmlFor="description">Description</Label>
            <Input
              id="description"
              value={config.description || ''}
              onChange={(e) => setConfig({ ...config, description: e.target.value })}
            />
          </div>
          
          <div>
            <Label htmlFor="size">Size</Label>
            <Select 
              value={config.size} 
              onChange={(e) => setConfig({ ...config, size: e.target.value as WidgetConfig['size'] })}
            >
              <SelectItem value="small">Small (1x1)</SelectItem>
              <SelectItem value="medium">Medium (2x1)</SelectItem>
              <SelectItem value="large">Large (2x2)</SelectItem>
              <SelectItem value="xlarge">X-Large (4x2)</SelectItem>
            </Select>
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// Add widget dialog
interface AddWidgetDialogProps {
  isOpen: boolean
  onClose: () => void
  onAdd: (template: Omit<WidgetConfig, 'id'>) => void
}

function AddWidgetDialog({ isOpen, onClose, onAdd }: AddWidgetDialogProps) {
  const handleAddWidget = (template: Omit<WidgetConfig, 'id'>) => {
    onAdd(template)
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Add Widget</DialogTitle>
          <DialogDescription>
            Choose from available widget templates to add to your dashboard.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid grid-cols-2 gap-3 max-h-96 overflow-y-auto">
          {WIDGET_TEMPLATES.map((template, index) => (
            <Card 
              key={index}
              className="cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => handleAddWidget(template)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-sm">{template.title}</h4>
                  <Badge variant="outline" className="text-xs">
                    {template.type}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  {template.description}
                </p>
                <Badge variant="secondary" className="text-xs">
                  {template.size}
                </Badge>
              </CardContent>
            </Card>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Main dashboard builder component
interface DashboardBuilderProps {
  layout: DashboardLayout
  onLayoutChange: (layout: DashboardLayout) => void
  isEditing?: boolean
  onToggleEdit?: () => void
}

export function DashboardBuilder({ 
  layout, 
  onLayoutChange, 
  isEditing = false, 
  onToggleEdit 
}: DashboardBuilderProps) {
  const [activeWidget, setActiveWidget] = useState<string | null>(null)
  const [editingWidget, setEditingWidget] = useState<WidgetConfig | null>(null)
  const [showAddDialog, setShowAddDialog] = useState(false)

  const visibleWidgets = useMemo(() => 
    layout.widgets.filter(w => w.visible !== false),
    [layout.widgets]
  )

  const handleDragStart = (event: DragStartEvent) => {
    setActiveWidget(event.active.id as string)
  }

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    
    if (over && active.id !== over.id) {
      const oldIndex = layout.widgets.findIndex(w => w.id === active.id)
      const newIndex = layout.widgets.findIndex(w => w.id === over.id)
      
      const newWidgets = arrayMove(layout.widgets, oldIndex, newIndex)
      onLayoutChange({ ...layout, widgets: newWidgets })
    }
    
    setActiveWidget(null)
  }

  const handleAddWidget = (template: Omit<WidgetConfig, 'id'>) => {
    const newWidget: WidgetConfig = {
      ...template,
      id: `widget-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      visible: true
    }
    
    onLayoutChange({
      ...layout,
      widgets: [...layout.widgets, newWidget]
    })
  }

  const handleEditWidget = (widget: WidgetConfig) => {
    setEditingWidget(widget)
  }

  const handleSaveWidget = (updatedWidget: WidgetConfig) => {
    const newWidgets = layout.widgets.map(w => 
      w.id === updatedWidget.id ? updatedWidget : w
    )
    onLayoutChange({ ...layout, widgets: newWidgets })
    setEditingWidget(null)
  }

  const handleDeleteWidget = (id: string) => {
    const newWidgets = layout.widgets.filter(w => w.id !== id)
    onLayoutChange({ ...layout, widgets: newWidgets })
  }

  const handleDuplicateWidget = (widget: WidgetConfig) => {
    const duplicatedWidget: WidgetConfig = {
      ...widget,
      id: `widget-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      title: `${widget.title} (Copy)`
    }
    
    onLayoutChange({
      ...layout,
      widgets: [...layout.widgets, duplicatedWidget]
    })
  }

  const handleToggleVisibility = (id: string) => {
    const newWidgets = layout.widgets.map(w => 
      w.id === id ? { ...w, visible: !w.visible } : w
    )
    onLayoutChange({ ...layout, widgets: newWidgets })
  }

  return (
    <div className="space-y-4">
      {/* Dashboard header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">{layout.name}</h2>
          {layout.description && (
            <p className="text-muted-foreground">{layout.description}</p>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {isEditing && (
            <Button onClick={() => setShowAddDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Add Widget
            </Button>
          )}
          
          {onToggleEdit && (
            <Button variant="outline" onClick={onToggleEdit}>
              {isEditing ? 'Done' : 'Edit Dashboard'}
            </Button>
          )}
        </div>
      </div>

      {/* Widget grid */}
      <DndContext
        collisionDetection={closestCenter}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <SortableContext items={visibleWidgets.map(w => w.id)} strategy={rectSortingStrategy}>
          <div className={cn(
            "grid grid-cols-4 gap-4 auto-rows-min",
            layout.settings?.gap && `gap-${layout.settings.gap}`
          )}>
            {visibleWidgets.map(widget => (
              <SortableWidget
                key={widget.id}
                widget={widget}
                isEditing={isEditing}
                onEdit={handleEditWidget}
                onDelete={handleDeleteWidget}
                onDuplicate={handleDuplicateWidget}
                onToggleVisibility={handleToggleVisibility}
              />
            ))}
          </div>
        </SortableContext>
        
        <DragOverlay>
          {activeWidget ? (
            <div className="opacity-90">
              <SortableWidget
                widget={layout.widgets.find(w => w.id === activeWidget)!}
                onEdit={() => {}}
                onDelete={() => {}}
                onDuplicate={() => {}}
                onToggleVisibility={() => {}}
              />
            </div>
          ) : null}
        </DragOverlay>
      </DndContext>

      {/* Dialogs */}
      <AddWidgetDialog
        isOpen={showAddDialog}
        onClose={() => setShowAddDialog(false)}
        onAdd={handleAddWidget}
      />
      
      <WidgetConfigDialog
        widget={editingWidget}
        isOpen={!!editingWidget}
        onClose={() => setEditingWidget(null)}
        onSave={handleSaveWidget}
      />
    </div>
  )
}