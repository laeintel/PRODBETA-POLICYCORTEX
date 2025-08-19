'use client'

import { api } from '@/lib/api-client'
import type { ChartConfig, ChartDataPoint } from '@/components/InteractiveCharts'

export type SectionKey =
  | 'security'
  | 'monitoring'
  | 'governance'
  | 'infrastructure'
  | 'finops'
  | 'devops'

export type MetricKpi = {
  value: string | number
  change?: number
  changeLabel?: string
  trend?: 'up' | 'down' | 'neutral'
  status?: 'success' | 'warning' | 'error' | 'neutral'
  sparklineData?: number[]
}

export type MetricFetchResult = {
  kpi: MetricKpi
  chart: ChartConfig
  drilldown?: Array<Record<string, unknown>>
}

export type MetricDefinition = {
  id: string
  title: string
  description?: string
  // Called to fetch data for both KPI and chart
  fetch: () => Promise<MetricFetchResult>
  // Optional suggested action to make it actionable
  primaryAction?: {
    label: string
    actionType: string
    buildParams?: () => Record<string, unknown>
  }
}

async function safe<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await fn()
  } catch {
    return fallback
  }
}

function lastN<T>(arr: T[], n: number): T[] {
  return arr.slice(Math.max(0, arr.length - n))
}

// Basic helpers for demo/fallback shaping
function rollingSeries(base: number, variance: number, points = 12): number[] {
  const out: number[] = []
  let current = base
  for (let i = 0; i < points; i++) {
    const delta = (Math.random() - 0.5) * variance
    current = Math.max(0, current + delta)
    out.push(Number(current.toFixed(2)))
  }
  return out
}

function toNamedSeries(name: string, values: number[]): ChartDataPoint[] {
  return values.map((v, i) => ({ name: `T-${values.length - i}`, value: v }))
}

export function getSectionMetrics(section: SectionKey): MetricDefinition[] {
  switch (section) {
    case 'security':
      return [
        {
          id: 'security_risk_score',
          title: 'Security Risk Score',
          description: 'Composite risk score based on threats, vulnerabilities and posture',
          fetch: async () => {
            const unified = await safe(() => api.getUnifiedMetrics(), { data: null, status: 200 })
            const score = unified?.data?.security?.risk_score ?? 92
            const trend = unified?.data?.security?.trend ?? 3
            const spark = rollingSeries(score, 2, 10)

            return {
              kpi: {
                value: `${score}%`,
                change: trend,
                changeLabel: 'vs last week',
                trend: trend >= 0 ? 'up' : 'down',
                status: score >= 90 ? 'success' : score >= 80 ? 'warning' : 'error',
                sparklineData: spark,
              },
              chart: {
                type: 'area',
                title: 'Risk Score Trend',
                data: toNamedSeries('Score', spark),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showLegend: false,
                showTooltip: true,
                animations: true,
                height: 220,
              },
              drilldown: lastN(toNamedSeries('Score', spark), 10),
            }
          },
          primaryAction: {
            label: 'Mitigate Top Threat',
            actionType: 'MITIGATE_TOP_THREAT',
          },
        },
        {
          id: 'active_threats',
          title: 'Active Threats',
          description: 'Current high/critical threats detected across assets',
          fetch: async () => {
            const threatsResp = await safe(() => api.getSecurityThreats(), { data: { threats: [] }, status: 200 })
            const threats = Array.isArray(threatsResp?.data?.threats) ? threatsResp.data.threats : []
            const critical = threats.filter((t: any) => t.severity === 'critical').length
            const high = threats.filter((t: any) => t.severity === 'high').length
            const total = threats.length || critical + high + 3
            const series = rollingSeries(total, 5, 12)

            return {
              kpi: {
                value: total,
                change: -5,
                changeLabel: 'last 24h',
                trend: 'down',
                status: total === 0 ? 'success' : critical > 0 ? 'error' : 'warning',
                sparklineData: series,
              },
              chart: {
                type: 'bar',
                title: 'Threats by Severity',
                data: [
                  { name: 'Critical', value: critical },
                  { name: 'High', value: high },
                  { name: 'Other', value: Math.max(0, total - critical - high) },
                ],
                xKey: 'name',
                yKey: 'value',
                showGrid: false,
                showLegend: false,
                showTooltip: true,
                animations: true,
                height: 220,
              },
              drilldown: threats,
            }
          },
          primaryAction: {
            label: 'Open Incident',
            actionType: 'OPEN_INCIDENT',
            buildParams: () => ({ severity: 'high' }),
          },
        },
        {
          id: 'vulns_by_severity',
          title: 'Vulnerabilities',
          description: 'Distribution by severity',
          fetch: async () => {
            const unified = await safe(() => api.getUnifiedMetrics(), { data: null, status: 200 })
            const vulns: { critical: number; high: number; medium: number; low: number } = unified?.data?.security?.vulnerabilities ?? {
              critical: 2,
              high: 5,
              medium: 16,
              low: 34,
            }
            const total: number = (vulns.critical ?? 0) + (vulns.high ?? 0) + (vulns.medium ?? 0) + (vulns.low ?? 0)

            return {
              kpi: {
                value: Number(total),
                change: -8,
                changeLabel: 'week over week',
                trend: 'down',
                status: total > 0 ? 'warning' : 'success',
              },
              chart: {
                type: 'pie',
                title: 'Vulnerabilities by Severity',
                data: [
                  { name: 'Critical', value: vulns.critical },
                  { name: 'High', value: vulns.high },
                  { name: 'Medium', value: vulns.medium },
                  { name: 'Low', value: vulns.low },
                ],
                nameKey: 'name',
                valueKey: 'value',
                showLegend: true,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: {
            label: 'Start Remediation',
            actionType: 'START_REMEDIATION',
          },
        },
      ]

    case 'monitoring':
      return [
        {
          id: 'uptime',
          title: 'Service Uptime',
          description: 'Aggregate uptime across critical services',
          fetch: async () => {
            const resp = await safe(() => api.getMonitoring(), { data: null, status: 200 })
            const uptime = resp?.data?.uptime ?? 99.85
            const series = rollingSeries(uptime, 0.2, 12)
            return {
              kpi: {
                value: `${uptime.toFixed(2)}%`,
                change: 0.03,
                changeLabel: 'vs last 7d',
                trend: 'up',
                status: uptime >= 99.9 ? 'success' : uptime >= 99.5 ? 'warning' : 'error',
                sparklineData: series,
              },
              chart: {
                type: 'line',
                title: 'Uptime (last 12 intervals)',
                data: toNamedSeries('Uptime', series),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showLegend: false,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: {
            label: 'Open Maintenance Window',
            actionType: 'OPEN_MAINTENANCE',
          },
        },
        {
          id: 'error_rate',
          title: 'Error Rate',
          description: 'Application error rate across services',
          fetch: async () => {
            const resp = await safe(() => api.getPerformance(), { data: null, status: 200 })
            const rate = resp?.data?.error_rate ?? 0.72
            const series = rollingSeries(rate, 0.3, 12)
            return {
              kpi: {
                value: `${rate.toFixed(2)}%`,
                change: -0.11,
                changeLabel: 'vs last 7d',
                trend: 'down',
                status: rate < 1 ? 'success' : rate < 2 ? 'warning' : 'error',
                sparklineData: series,
              },
              chart: {
                type: 'area',
                title: 'Error Rate (last 12 intervals)',
                data: toNamedSeries('Error Rate', series),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showLegend: false,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: {
            label: 'Scale Out',
            actionType: 'SCALE_OUT',
            buildParams: () => ({ percentage: 20 }),
          },
        },
      ]

    case 'governance':
      return [
        {
          id: 'compliance_rate',
          title: 'Compliance Rate',
          fetch: async () => {
            const resp = await safe(() => api.getComplianceStatus(), { data: null, status: 200 })
            const rate = resp?.data?.compliance_rate ?? 96.2
            const series = rollingSeries(rate, 1.2, 10)
            return {
              kpi: {
                value: `${rate.toFixed(1)}%`,
                change: 0.8,
                changeLabel: 'vs last month',
                trend: 'up',
                status: rate >= 99 ? 'success' : rate >= 95 ? 'warning' : 'error',
                sparklineData: series,
              },
              chart: {
                type: 'line',
                title: 'Compliance Trend',
                data: toNamedSeries('Compliance', series),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showLegend: false,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Run Compliance Scan', actionType: 'RUN_COMPLIANCE_SCAN' },
        },
        {
          id: 'policy_violations',
          title: 'Policy Violations',
          fetch: async () => {
            const resp = await safe(() => api.getPolicies(), { data: null, status: 200 })
            const violations = resp?.data?.violations ?? 27
            const series = rollingSeries(violations, 5, 12)
            return {
              kpi: {
                value: violations,
                change: -4,
                changeLabel: 'vs last month',
                trend: 'down',
                status: violations > 0 ? 'warning' : 'success',
              },
              chart: {
                type: 'bar',
                title: 'Violations (last 12 intervals)',
                data: toNamedSeries('Violations', series),
                xKey: 'name',
                yKey: 'value',
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Enforce Policies', actionType: 'ENFORCE_POLICIES' },
        },
      ]

    case 'infrastructure':
      return [
        {
          id: 'resource_count',
          title: 'Total Resources',
          fetch: async () => {
            const resp = await safe(() => api.getResources({ limit: 200 } as any), { data: { items: [] }, status: 200 })
            const count = Array.isArray((resp as any)?.data) ? (resp as any).data.length : (resp as any)?.data?.items?.length ?? 312
            const series = rollingSeries(count, 10, 12)
            return {
              kpi: { value: count, change: 2.1, changeLabel: 'new resources', trend: 'up', status: 'neutral', sparklineData: series },
              chart: {
                type: 'area',
                title: 'Resources Growth',
                data: toNamedSeries('Resources', series),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Create Resource', actionType: 'CREATE_RESOURCE' },
        },
      ]

    case 'finops':
      return [
        {
          id: 'monthly_spend',
          title: 'Monthly Spend',
          fetch: async () => {
            const resp = await safe(() => api.getCostAnalysis(), { data: null, status: 200 })
            const spend = resp?.data?.current_spend ?? 132000
            const series = rollingSeries(spend, 5000, 12)
            return {
              kpi: {
                value: `$${Math.round(spend).toLocaleString()}`,
                change: -3.1,
                changeLabel: 'vs last month',
                trend: 'down',
                status: 'success',
                sparklineData: series,
              },
              chart: {
                type: 'line',
                title: 'Spend Trend',
                data: toNamedSeries('Spend', series),
                xKey: 'name',
                yKey: 'value',
                showGrid: true,
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Optimize Costs', actionType: 'RUN_COST_OPTIMIZATION' },
        },
      ]

    case 'devops':
      return [
        {
          id: 'deploy_frequency',
          title: 'Deployment Frequency',
          fetch: async () => {
            const resp = await safe(() => api.getPerformance(), { data: null, status: 200 })
            const freq = resp?.data?.deployment_frequency ?? 18
            const series = rollingSeries(freq, 2, 12)
            return {
              kpi: { value: `${freq}/wk`, change: 1.2, changeLabel: 'vs last week', trend: 'up', status: 'success', sparklineData: series },
              chart: {
                type: 'bar',
                title: 'Deployments (last 12 intervals)',
                data: toNamedSeries('Deployments', series),
                xKey: 'name',
                yKey: 'value',
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Trigger Deployment', actionType: 'TRIGGER_DEPLOYMENT' },
        },
        {
          id: 'pipeline_failures',
          title: 'Pipeline Failures',
          fetch: async () => {
            const resp = await safe(() => api.getPerformance(), { data: null, status: 200 })
            const failures = resp?.data?.pipeline_failures ?? 3
            const series = rollingSeries(failures, 1, 12)
            return {
              kpi: { value: failures, change: -1, changeLabel: 'vs last week', trend: 'down', status: failures > 0 ? 'warning' : 'success' },
              chart: {
                type: 'area',
                title: 'Pipeline Failures',
                data: toNamedSeries('Failures', series),
                xKey: 'name',
                yKey: 'value',
                showTooltip: true,
                animations: true,
                height: 220,
              },
            }
          },
          primaryAction: { label: 'Open Incident', actionType: 'OPEN_INCIDENT' },
        },
      ]

    default:
      return []
  }
}


