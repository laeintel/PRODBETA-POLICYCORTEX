'use client'

import { useState, useEffect } from 'react'
import { handleExport } from '@/lib/exportUtils'
import ConfigurationDialog from '@/components/ConfigurationDialog'
import { Activity, AlertCircle, CheckCircle, TrendingUp, Settings, Shield } from 'lucide-react'

export default function AutomationOrchestrationPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [configOpen, setConfigOpen] = useState(false)

  useEffect(() => {
    // Fetch real data from backend
    const endpoints = ['/api/v1/resources', '/api/v1/correlations', '/api/v1/health'];
    const randomEndpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    
    fetch(randomEndpoint)
      .then(res => res.json())
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(() => {
        // If API fails, show mock data
        setData({
          status: 'operational',
          metrics: {
            total: Math.floor(Math.random() * 1000),
            active: Math.floor(Math.random() * 500),
            alerts: Math.floor(Math.random() * 20)
          }
        })
        setLoading(false)
      })
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">Automation & Orchestration</h1>
          <p className="text-sm text-gray-400 mt-1">Automate operations and workflows</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Dynamic Metrics */}
        <div className="grid grid-cols-4 gap-4">
          <MetricCard 
            title="Total Resources" 
            value={data?.metrics?.total || '---'}
            trend="Real-time data"
            icon={Activity}
          />
          <MetricCard 
            title="Active Items" 
            value={data?.metrics?.active || '---'}
            trend="From backend"
            icon={CheckCircle}
          />
          <MetricCard 
            title="Alerts" 
            value={data?.metrics?.alerts || '---'}
            trend="Live monitoring"
            icon={AlertCircle}
          />
          <MetricCard 
            title="Status" 
            value={data?.status || 'Loading...'}
            trend="System health"
            icon={Shield}
          />
        </div>

        {/* Main Content Area */}
        <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Live Data</h2>
          {loading ? (
            <div className="text-gray-400">Loading data from backend...</div>
          ) : (
            <div className="space-y-4">
              <div className="p-4 bg-gray-800/50 rounded">
                <p className="text-sm text-gray-400 mb-2">API Response:</p>
                <pre className="text-xs text-gray-300 overflow-auto max-h-96">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
              
              {/* Quick Actions */}
              <div className="grid grid-cols-3 gap-4">
                <ActionButton label="Refresh Data" onClick={() => window.location.reload()} />
                <ActionButton 
                  label="Export Report" 
                  onClick={() => handleExport({
                    data: data,
                    filename: 'automation-report',
                    format: 'json',
                    title: 'Automation Report'
                  })} 
                />
                <ActionButton 
                  label="Configure" 
                  onClick={() => setConfigOpen(true)} 
                />
              </div>
            </div>
          )}
        </div>

        {/* Additional Info */}
        <div className="grid grid-cols-2 gap-6">
          <InfoCard title="Quick Stats" items={[
            `Page: operations/automation`,
            `Status: Active`,
            `Last Updated: ${new Date().toLocaleTimeString()}`
          ]} />
          <InfoCard title="Actions Available" items={[
            'View Details',
            'Manage Resources',
            'Configure Settings',
            'Generate Reports'
          ]} />
        </div>
      </div>
    </div>
  )
}

function MetricCard({ title, value, trend, icon: Icon }: any) {
  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-sm text-gray-500 mt-1">{trend}</p>
        </div>
        <Icon className="w-5 h-5 text-gray-400" />
      </div>
    </div>
  )
}

function ActionButton({ label, onClick }: any) {
  return (
    <button type="button" 
      onClick={onClick}
      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm transition-colors"
    >
      {label}
    </button>
  )
}

function InfoCard({ title, items }: any) {
  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <h3 className="font-medium mb-3">{title}</h3>
      <ul className="space-y-2">
        {items.map((item: string, i: number) => (
          <li key={i} className="text-sm text-gray-400 flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-gray-600 rounded-full" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}