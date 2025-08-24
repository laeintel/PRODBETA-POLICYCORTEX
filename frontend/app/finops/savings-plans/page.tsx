'use client'

import { useState } from 'react'
import {
  DollarSign, TrendingUp, Calculator, PiggyBank, Target,
  AlertCircle, CheckCircle, Clock, BarChart3, ArrowRight,
  Zap, Shield, Package, Database, Globe, Cpu
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

export default function SavingsPlansPage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards')

  const savingsData = [
    { month: 'Jan', committed: 125000, onDemand: 180000, savings: 55000 },
    { month: 'Feb', committed: 130000, onDemand: 165000, savings: 35000 },
    { month: 'Mar', committed: 135000, onDemand: 150000, savings: 15000 },
    { month: 'Apr', committed: 140000, onDemand: 145000, savings: 5000 },
    { month: 'May', committed: 145000, onDemand: 140000, savings: -5000 },
    { month: 'Jun', committed: 150000, onDemand: 135000, savings: -15000 }
  ]

  const planUtilization = [
    { service: 'Compute', coverage: 85, utilization: 92, potential: 8 },
    { service: 'Database', coverage: 72, utilization: 88, potential: 15 },
    { service: 'Storage', coverage: 65, utilization: 95, potential: 20 },
    { service: 'Network', coverage: 45, utilization: 78, potential: 35 },
    { service: 'AI/ML', coverage: 30, utilization: 85, potential: 45 }
  ]

  const recommendations = [
    {
      title: 'Convert EC2 On-Demand to Savings Plan',
      savings: '$24,000/year',
      effort: 'Low',
      risk: 'Low',
      instances: 45,
      icon: <Cpu className="w-5 h-5" />
    },
    {
      title: 'Optimize RDS Reserved Instances',
      savings: '$18,000/year',
      effort: 'Medium',
      risk: 'Low',
      instances: 12,
      icon: <Database className="w-5 h-5" />
    },
    {
      title: 'Convert Lambda to Compute Savings Plan',
      savings: '$15,000/year',
      effort: 'Low',
      risk: 'Low',
      instances: 200,
      icon: <Zap className="w-5 h-5" />
    },
    {
      title: 'Implement S3 Intelligent Tiering',
      savings: '$12,000/year',
      effort: 'Low',
      risk: 'Low',
      instances: 8,
      icon: <Package className="w-5 h-5" />
    }
  ]

  const activePlans = [
    {
      id: 'SP-2024-001',
      type: 'Compute',
      term: '1 Year',
      commitment: '$150,000',
      hourlyRate: '$17.12',
      utilization: 94,
      expiry: '2025-03-15',
      status: 'active'
    },
    {
      id: 'SP-2024-002',
      type: 'EC2 Instance',
      term: '3 Years',
      commitment: '$280,000',
      hourlyRate: '$10.65',
      utilization: 88,
      expiry: '2027-01-20',
      status: 'active'
    },
    {
      id: 'SP-2024-003',
      type: 'SageMaker',
      term: '1 Year',
      commitment: '$45,000',
      hourlyRate: '$5.14',
      utilization: 76,
      expiry: '2024-11-30',
      status: 'expiring'
    }
  ]

  return (
    <div className="min-h-screen bg-background">
      <div className="p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
              Savings Plans & Reservations
            </h1>
            <p className="text-muted-foreground mt-2">
              Optimize cloud spending with committed use discounts and intelligent planning
            </p>
          </div>
          <ViewToggle view={view} onViewChange={setView} />
        </div>

        {view === 'cards' ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <MetricCard
                title="Total Annual Savings"
                value="$485K"
                change={32}
                changeLabel="vs on-demand"
                trend="up"
                icon={<PiggyBank className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="Coverage Rate"
                value="72%"
                change={8}
                changeLabel="this quarter"
                trend="up"
                icon={<Shield className="w-5 h-5 text-blue-500" />}
              />
              <MetricCard
                title="Utilization Rate"
                value="89%"
                change={-3}
                changeLabel="efficiency"
                trend="down"
                icon={<Target className="w-5 h-5 text-purple-500" />}
              />
              <MetricCard
                title="Expiring Plans"
                value="3"
                alert="Action needed"
                icon={<Clock className="w-5 h-5 text-orange-500" />}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="lg:col-span-2 bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Committed vs On-Demand Spend</h2>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={savingsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip formatter={(value) => [`$${(value as number).toLocaleString()}`, '']} />
                      <Legend />
                      <Area type="monotone" dataKey="committed" stackId="1" stroke="#10b981" fill="#10b981" name="Committed Spend" />
                      <Area type="monotone" dataKey="onDemand" stackId="1" stroke="#6b7280" fill="#6b7280" name="On-Demand Spend" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Service Coverage</h2>
                <div className="space-y-4">
                  {planUtilization.map((service) => (
                    <div key={service.service}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{service.service}</span>
                        <span className="font-medium">{service.coverage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-1">
                        <div
                          className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                          style={{ width: `${service.coverage}%` }}
                        />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {service.potential}% additional savings potential
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Optimization Recommendations</h2>
                <div className="space-y-3">
                  {recommendations.map((rec, index) => (
                    <div key={index} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition-colors cursor-pointer">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                            {rec.icon}
                          </div>
                          <div>
                            <div className="font-medium">{rec.title}</div>
                            <div className="text-sm text-muted-foreground mt-1">
                              {rec.instances} instances • {rec.effort} effort • {rec.risk} risk
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-green-600">{rec.savings}</div>
                          <ArrowRight className="w-4 h-4 text-muted-foreground mt-1" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-card dark:bg-gray-800 rounded-lg p-6 border border-border dark:border-gray-700">
                <h2 className="text-xl font-semibold mb-4">Active Savings Plans</h2>
                <div className="space-y-3">
                  {activePlans.map((plan) => (
                    <div key={plan.id} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{plan.id}</span>
                          {plan.status === 'expiring' && (
                            <span className="px-2 py-1 text-xs bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 rounded">
                              Expiring Soon
                            </span>
                          )}
                        </div>
                        <span className="text-sm text-muted-foreground">{plan.type}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Term:</span> {plan.term}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Commitment:</span> {plan.commitment}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Utilization:</span> {plan.utilization}%
                        </div>
                        <div>
                          <span className="text-muted-foreground">Expires:</span> {plan.expiry}
                        </div>
                      </div>
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              plan.utilization >= 90 ? 'bg-green-500' : 
                              plan.utilization >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${plan.utilization}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold mb-2">Savings Plan Analyzer</h3>
                  <p className="text-blue-100">
                    AI-powered analysis identified $120K additional savings opportunity across 15 services
                  </p>
                </div>
                <button className="px-6 py-3 bg-white text-blue-600 rounded-lg font-medium hover:bg-blue-50 transition-colors">
                  View Analysis
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer title="Monthly Spend Analysis">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={savingsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${(value as number).toLocaleString()}`, '']} />
                  <Legend />
                  <Bar dataKey="committed" fill="#10b981" name="Committed Spend" />
                  <Bar dataKey="onDemand" fill="#6b7280" name="On-Demand Spend" />
                  <Bar dataKey="savings" fill="#3b82f6" name="Savings" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Service Coverage & Utilization">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart data={planUtilization} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="service" type="category" width={80} />
                  <Tooltip formatter={(value) => [`${value}%`, '']} />
                  <Legend />
                  <Bar dataKey="coverage" fill="#8b5cf6" name="Coverage (%)" />
                  <Bar dataKey="utilization" fill="#10b981" name="Utilization (%)" />
                  <Bar dataKey="potential" fill="#f59e0b" name="Potential (%)" />
                </RechartsBarChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Savings Trend">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart data={savingsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${(value as number).toLocaleString()}`, 'Savings']} />
                  <Legend />
                  <Line type="monotone" dataKey="savings" stroke="#10b981" strokeWidth={3} name="Monthly Savings" />
                </RechartsLineChart>
              </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="Coverage vs Utilization Matrix">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={planUtilization}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="service" />
                  <PolarRadiusAxis />
                  <Radar name="Coverage" dataKey="coverage" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                  <Radar name="Utilization" dataKey="utilization" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        )}
      </div>
    </div>
  )
}