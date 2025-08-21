// Type definitions for operations page components

export interface MetricCardProps {
  title: string
  value: string | number
  trend: string
  status: 'good' | 'warning' | 'critical' | 'neutral'
  icon: React.ElementType
}

export interface AlertItemProps {
  title: string
  severity: 'high' | 'medium' | 'low'
  time: string
  action: string
}

export interface ResourceRowProps {
  type: string
  count: number
  percentage: number
}

export interface AutomationRowProps {
  name: string
  status: 'running' | 'scheduled' | 'active' | 'idle'
  count: number
}

export interface ResourceCardProps {
  icon: React.ElementType
  type: string
  count: number
  status: string
  health: 'healthy' | 'warning' | 'critical'
}

export interface ResourceTableRowProps {
  name: string
  type: string
  region: string
  status: string
  cost: string
}

export interface OptimizationItemProps {
  resource: string
  issue: string
  recommendation: string
  savings: string
}

export interface MonitoringCardProps {
  title: string
  value: string | number
  period: string
  status: 'good' | 'warning' | 'critical' | 'neutral'
}

export interface UtilizationBarProps {
  label: string
  value: number
  threshold: number
}

export interface HealthItemProps {
  service: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  latency: string
}

export interface MonitorRowProps {
  name: string
  type: string
  frequency: string
  status: 'passing' | 'warning' | 'failing'
}

export interface AutomationCardProps {
  title: string
  value: string | number
  icon: React.ElementType
  status: 'active' | 'success' | 'good' | 'neutral'
}

export interface RunItemProps {
  workflow: string
  trigger: string
  status: 'success' | 'failed' | 'running'
  duration: string
  time: string
}

export interface PipelineItemProps {
  name: string
  branch: string
  status: 'running' | 'queued' | 'success' | 'failed'
  progress?: number
}

export interface JobItemProps {
  name: string
  schedule: string
  nextRun: string
  enabled: boolean
}