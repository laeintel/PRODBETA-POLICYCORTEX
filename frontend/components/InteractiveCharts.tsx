"use client"

import { useState, useCallback, useMemo } from 'react'
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { Download, ZoomIn, ZoomOut, RefreshCw, Settings, Maximize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
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
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'

// Chart data types
export interface ChartDataPoint {
  [key: string]: any
}

export interface ChartConfig {
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'area' | 'radar'
  title: string
  data: ChartDataPoint[]
  xKey?: string
  yKey?: string
  valueKey?: string
  nameKey?: string
  colorPalette?: string[]
  showGrid?: boolean
  showLegend?: boolean
  showTooltip?: boolean
  height?: number
  animations?: boolean
  zoom?: boolean
}

// Color palettes
const COLOR_PALETTES = {
  default: ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'],
  blue: ['#1f77b4', '#aec7e8', '#c5dbf1', '#e5f1ff', '#f0f8ff'],
  green: ['#2ca02c', '#98df8a', '#c7e9c7', '#e5f5e5', '#f0fff0'],
  warm: ['#ff7f0e', '#ffbb78', '#ffd8b1', '#ffe8d1', '#fff5e8'],
  cool: ['#17becf', '#9edae5', '#c7e9f1', '#e0f3f8', '#f0f9fa'],
  purple: ['#9467bd', '#c5b0d5', '#d9c7dd', '#e8dde8', '#f4f0f4']
}

// Custom tooltip component
function CustomTooltip({ active, payload, label, formatValue, showPercentage }: any) {
  if (!active || !payload || !payload.length) return null

  return (
    <Card className="shadow-lg border">
      <CardContent className="p-3">
        <p className="font-medium mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 text-sm">
            <div 
              className="w-3 h-3 rounded" 
              style={{ backgroundColor: entry.color }}
            />
            <span>{entry.name}:</span>
            <span className="font-medium">
              {formatValue ? formatValue(entry.value) : entry.value}
              {showPercentage && '%'}
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

// Interactive chart component
interface InteractiveChartProps {
  config: ChartConfig
  onDataPointClick?: (data: ChartDataPoint) => void
  onExport?: (format: 'png' | 'svg' | 'pdf') => void
  formatValue?: (value: any) => string
  className?: string
}

export function InteractiveChart({ 
  config, 
  onDataPointClick, 
  onExport,
  formatValue,
  className 
}: InteractiveChartProps) {
  const [zoomLevel, setZoomLevel] = useState(1)
  const [showSettings, setShowSettings] = useState(false)
  const [chartConfig, setChartConfig] = useState(config)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const colors = chartConfig.colorPalette || COLOR_PALETTES.default

  const handleDataPointClick = useCallback((data: any) => {
    onDataPointClick?.(data)
  }, [onDataPointClick])

  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.2, 2))
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.2, 0.5))
  const handleResetZoom = () => setZoomLevel(1)

  const renderChart = () => {
    const baseProps = {
      data: chartConfig.data,
      onClick: handleDataPointClick,
      style: { transform: `scale(${zoomLevel})`, transformOrigin: 'center' }
    }

    switch (chartConfig.type) {
      case 'bar':
        return (
          <BarChart {...baseProps}>
            {chartConfig.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey={chartConfig.xKey} />
            <YAxis />
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Bar 
              dataKey={chartConfig.yKey || 'value'} 
              fill={colors[0]}
              animationDuration={chartConfig.animations ? 1000 : 0}
            />
          </BarChart>
        )

      case 'line':
        return (
          <LineChart {...baseProps}>
            {chartConfig.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey={chartConfig.xKey} />
            <YAxis />
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Line 
              type="monotone"
              dataKey={chartConfig.yKey} 
              stroke={colors[0]}
              strokeWidth={2}
              animationDuration={chartConfig.animations ? 1000 : 0}
            />
          </LineChart>
        )

      case 'pie':
        return (
          <PieChart {...baseProps}>
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} showPercentage />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Pie
              data={chartConfig.data}
              dataKey={chartConfig.valueKey || 'value'}
              nameKey={chartConfig.nameKey || 'name'}
              cx="50%"
              cy="50%"
              outerRadius={80}
              animationDuration={chartConfig.animations ? 1000 : 0}
            >
              {chartConfig.data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Pie>
          </PieChart>
        )

      case 'area':
        return (
          <AreaChart {...baseProps}>
            {chartConfig.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey={chartConfig.xKey} />
            <YAxis />
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Area
              type="monotone"
              dataKey={chartConfig.yKey || 'value'}
              stroke={colors[0]}
              fill={colors[0]}
              fillOpacity={0.3}
              animationDuration={chartConfig.animations ? 1000 : 0}
            />
          </AreaChart>
        )

      case 'scatter':
        return (
          <ScatterChart {...baseProps}>
            {chartConfig.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey={chartConfig.xKey} />
            <YAxis dataKey={chartConfig.yKey} />
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Scatter 
              data={chartConfig.data} 
              fill={colors[0]}
              animationDuration={chartConfig.animations ? 1000 : 0}
            />
          </ScatterChart>
        )

      case 'radar':
        return (
          <RadarChart {...baseProps}>
            <PolarGrid />
            <PolarAngleAxis dataKey={chartConfig.nameKey} />
            <PolarRadiusAxis />
            {chartConfig.showTooltip && (
              <Tooltip content={<CustomTooltip formatValue={formatValue} />} />
            )}
            {chartConfig.showLegend && <Legend />}
            <Radar
              dataKey={chartConfig.valueKey || 'value'}
              stroke={colors[0]}
              fill={colors[0]}
              fillOpacity={0.3}
              animationDuration={chartConfig.animations ? 1000 : 0}
            />
          </RadarChart>
        )

      default:
        return null
    }
  }

  const ChartWrapper = isFullscreen ? Dialog : Card

  return (
    <ChartWrapper className={className}>
      {isFullscreen ? (
        <DialogContent className="max-w-6xl h-[80vh]">
          <DialogHeader>
            <DialogTitle>{chartConfig.title}</DialogTitle>
          </DialogHeader>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              {renderChart() || <div>Chart type not supported</div>}
            </ResponsiveContainer>
          </div>
        </DialogContent>
      ) : (
        <>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{chartConfig.title}</CardTitle>
              <div className="flex items-center gap-1">
                {chartConfig.zoom && (
                  <>
                    <Button variant="ghost" size="icon" onClick={handleZoomOut}>
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={handleResetZoom}>
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={handleZoomIn}>
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                  </>
                )}
                
                <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
                  <DialogTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <Maximize2 className="h-4 w-4" />
                    </Button>
                  </DialogTrigger>
                </Dialog>

                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <Settings className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>Chart Settings</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <div className="p-2 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Show Grid</span>
                        <Switch
                          checked={chartConfig.showGrid}
                          onCheckedChange={(checked) => 
                            setChartConfig(prev => ({ ...prev, showGrid: checked }))
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Show Legend</span>
                        <Switch
                          checked={chartConfig.showLegend}
                          onCheckedChange={(checked) => 
                            setChartConfig(prev => ({ ...prev, showLegend: checked }))
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Animations</span>
                        <Switch
                          checked={chartConfig.animations}
                          onCheckedChange={(checked) => 
                            setChartConfig(prev => ({ ...prev, animations: checked }))
                          }
                        />
                      </div>
                    </div>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel>Color Palette</DropdownMenuLabel>
                    {Object.keys(COLOR_PALETTES).map(palette => (
                      <DropdownMenuItem
                        key={palette}
                        onClick={() => setChartConfig(prev => ({ ...prev, colorPalette: COLOR_PALETTES[palette as keyof typeof COLOR_PALETTES] }))}
                      >
                        <div className="flex items-center gap-2">
                          <div className="flex gap-1">
                            {COLOR_PALETTES[palette as keyof typeof COLOR_PALETTES].slice(0, 3).map((color, i) => (
                              <div
                                key={i}
                                className="w-3 h-3 rounded"
                                style={{ backgroundColor: color }}
                              />
                            ))}
                          </div>
                          <span className="capitalize">{palette}</span>
                        </div>
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>

                {onExport && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <Download className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuLabel>Export Chart</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem onClick={() => onExport('png')}>
                        Export as PNG
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onExport('svg')}>
                        Export as SVG
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onExport('pdf')}>
                        Export as PDF
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
              </div>
            </div>
            {zoomLevel !== 1 && (
              <Badge variant="outline" className="w-fit">
                Zoom: {Math.round(zoomLevel * 100)}%
              </Badge>
            )}
          </CardHeader>
          
          <CardContent>
            <ResponsiveContainer 
              width="100%" 
              height={chartConfig.height || 300}
            >
              {renderChart() || <div>Chart type not supported</div>}
            </ResponsiveContainer>
          </CardContent>
        </>
      )}
    </ChartWrapper>
  )
}

// Real-time chart component with automatic updates
interface RealTimeChartProps extends Omit<InteractiveChartProps, 'config'> {
  config: Omit<ChartConfig, 'data'>
  dataSource: () => Promise<ChartDataPoint[]>
  updateInterval?: number
  maxDataPoints?: number
}

export function RealTimeChart({
  config,
  dataSource,
  updateInterval = 5000,
  maxDataPoints = 50,
  ...props
}: RealTimeChartProps) {
  const [data, setData] = useState<ChartDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const updateData = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const newData = await dataSource()
      setData(prev => {
        const combined = [...prev, ...newData]
        return combined.slice(-maxDataPoints)
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update data')
    } finally {
      setIsLoading(false)
    }
  }, [dataSource, maxDataPoints])

  // Auto-update data
  useState(() => {
    updateData()
    const interval = setInterval(updateData, updateInterval)
    return () => clearInterval(interval)
  })

  const chartConfig = useMemo(() => ({
    ...config,
    data,
    title: `${config.title} ${isLoading ? '(Updating...)' : ''}`
  }), [config, data, isLoading])

  if (error) {
    return (
      <Card className="p-6 text-center">
        <p className="text-destructive mb-2">Failed to load chart data</p>
        <Button variant="outline" onClick={updateData}>
          Retry
        </Button>
      </Card>
    )
  }

  return (
    <InteractiveChart
      config={chartConfig}
      {...props}
    />
  )
}