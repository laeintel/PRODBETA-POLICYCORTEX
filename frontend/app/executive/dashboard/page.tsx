'use client'

import { useState } from 'react'
import {
  TrendingUp, TrendingDown, DollarSign, Shield, AlertTriangle,
  Users, Target, Award, BarChart3, Activity, Briefcase,
  CheckCircle, XCircle, Clock, PieChart, LineChart, ArrowUpRight
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  LineChart as RechartsLineChart,
  BarChart as RechartsBarChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'
import ViewToggle from '@/components/ViewToggle'

export default function ExecutiveDashboardPage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards')

  const kpiData = [
    { month: 'Jan', revenue: 4.2, cost: 1.8, compliance: 92, risk: 8 },
    { month: 'Feb', revenue: 4.5, cost: 1.7, compliance: 94, risk: 6 },
    { month: 'Mar', revenue: 4.8, cost: 1.6, compliance: 95, risk: 5 },
    { month: 'Apr', revenue: 5.1, cost: 1.5, compliance: 96, risk: 4 },
    { month: 'May', revenue: 5.4, cost: 1.4, compliance: 97, risk: 3 },
    { month: 'Jun', revenue: 5.8, cost: 1.3, compliance: 98, risk: 2 }
  ]

  const departmentPerformance = [
    { name: 'Engineering', score: 95, budget: 2.4, efficiency: 88 },
    { name: 'Operations', score: 89, budget: 1.8, efficiency: 92 },
    { name: 'Security', score: 92, budget: 1.2, efficiency: 85 },
    { name: 'Finance', score: 88, budget: 0.8, efficiency: 94 },
    { name: 'HR', score: 86, budget: 0.6, efficiency: 90 }
  ]

  const strategicMetrics = [
    { category: 'Digital Transformation', progress: 78, target: 95 },
    { category: 'Cloud Migration', progress: 85, target: 100 },
    { category: 'AI Adoption', progress: 62, target: 80 },
    { category: 'Automation', progress: 71, target: 90 },
    { category: 'Security Posture', progress: 94, target: 99 }
  ]

  return (
    <div className="min-h-screen bg-background">
      <div className="p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Executive Business KPIs
            </h1>
            <p className="text-muted-foreground mt-2">
              Real-time business performance metrics and strategic insights
            </p>
          </div>
          <ViewToggle view={view} onViewChange={setView} />
        </div>

        {view === 'cards' ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <MetricCard
                title="Annual Revenue"
                value="$31.8M"
                change={18.5}
                changeLabel="vs last year"
                trend="up"
                icon={<DollarSign className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="Cost Savings"
                value="$4.2M"
                change={24.3}
                changeLabel="YTD savings"
                trend="up"
                icon={<TrendingDown className="w-5 h-5 text-blue-500" />}
              />
              <MetricCard
                title="Compliance Score"
                value="98%"
                change={4.2}
                changeLabel="improvement"
                trend="up"
                icon={<Shield className="w-5 h-5 text-purple-500" />}
              />
              <MetricCard
                title="Risk Score"
                value="Low"
                status="success"
                icon={<AlertTriangle className="w-5 h-5 text-green-500" />}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="lg:col-span-2 bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Revenue & Cost Trends</h2>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={kpiData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="revenue" stackId="1" stroke="#10b981" fill="#10b981" name="Revenue ($M)" />
                      <Area type="monotone" dataKey="cost" stackId="1" stroke="#ef4444" fill="#ef4444" name="Cost ($M)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Strategic Goals</h2>
                <div className="space-y-4">
                  {strategicMetrics.map((metric) => (
                    <div key={metric.category}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{metric.category}</span>
                        <span className="font-medium">{metric.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                          style={{ width: `${metric.progress}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Department Performance</h2>
                <div className="space-y-3">
                  {departmentPerformance.map((dept) => (
                    <div key={dept.name} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Users className="w-5 h-5 text-blue-500" />
                        <div>
                          <div className="font-medium">{dept.name}</div>
                          <div className="text-sm text-muted-foreground">
                            Budget: ${dept.budget}M | Efficiency: {dept.efficiency}%
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold">{dept.score}</div>
                        <div className="text-xs text-muted-foreground">Score</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Key Business Metrics</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-5 h-5 text-green-600" />
                      <span className="text-sm font-medium">NPS Score</span>
                    </div>
                    <div className="text-2xl font-bold">72</div>
                    <div className="text-xs text-green-600">+8 points</div>
                  </div>
                  <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Award className="w-5 h-5 text-blue-600" />
                      <span className="text-sm font-medium">CSAT</span>
                    </div>
                    <div className="text-2xl font-bold">4.6/5</div>
                    <div className="text-xs text-blue-600">+0.3</div>
                  </div>
                  <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5 text-purple-600" />
                      <span className="text-sm font-medium">Uptime</span>
                    </div>
                    <div className="text-2xl font-bold">99.98%</div>
                    <div className="text-xs text-purple-600">Enterprise SLA</div>
                  </div>
                  <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="w-5 h-5 text-orange-600" />
                      <span className="text-sm font-medium">MTTR</span>
                    </div>
                    <div className="text-2xl font-bold">12m</div>
                    <div className="text-xs text-orange-600">-45% improved</div>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer title="Revenue vs Cost Analysis">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart data={kpiData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="revenue" stroke="#10b981" strokeWidth={2} name="Revenue ($M)" />
                  <Line type="monotone" dataKey="cost" stroke="#ef4444" strokeWidth={2} name="Cost ($M)" />
                </RechartsLineChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Compliance & Risk Trends">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={kpiData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="compliance" fill="#3b82f6" name="Compliance (%)" />
                  <Bar dataKey="risk" fill="#f59e0b" name="Risk Score" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Department Performance Matrix">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={departmentPerformance}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis />
                  <Radar name="Score" dataKey="score" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                  <Radar name="Efficiency" dataKey="efficiency" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Strategic Goals Progress">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={strategicMetrics} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="category" type="category" width={120} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="progress" fill="#8b5cf6" name="Progress (%)" />
                  <Bar dataKey="target" fill="#e5e7eb" name="Target (%)" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        )}
      </div>
    </div>
  )
}