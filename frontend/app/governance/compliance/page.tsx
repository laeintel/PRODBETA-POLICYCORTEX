'use client'

import { useState, useEffect } from 'react'
import { Shield, AlertTriangle, CheckCircle, TrendingUp, Clock, Activity, GitBranch, Zap, BarChart3, Brain, LineChart, AlertCircle, Info, Calendar, Target } from 'lucide-react'

export default function CompliancePage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch real data from backend
    fetch('/api/v1/resources')
      .then(res => res.json())
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white">
      <div className="border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold">Compliance Management</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Monitor and manage compliance across all cloud resources
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-4 gap-4">
          <MetricCard 
            title="Compliance Score" 
            value="94%" 
            trend="+2% from last month"
            icon={CheckCircle}
            color="green"
          />
          <MetricCard 
            title="Active Policies" 
            value="127" 
            trend="12 newly added"
            icon={Shield}
            color="blue"
          />
          <MetricCard 
            title="Violations" 
            value="23" 
            trend="-5 from yesterday"
            icon={AlertTriangle}
            color="yellow"
          />
          <MetricCard 
            title="Resources Monitored" 
            value="1,847" 
            trend="+67 this week"
            icon={TrendingUp}
            color="purple"
          />
        </div>

        {/* Patent #4: Predictive Policy Compliance Engine */}
        <div className="bg-gradient-to-r from-purple-900/10 to-blue-900/10 rounded-lg border border-purple-500/30 p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <Brain className="w-6 h-6 text-purple-400" />
              <div>
                <h2 className="text-xl font-bold">Predictive Compliance Engine</h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">Patent #4: ML-Powered Drift Detection & Predictions</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-green-400">●</span>
              <span>99.2% Model Accuracy</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Drift Detection Panel */}
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-yellow-400" />
                Configuration Drift Detection
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-red-900/20 rounded-lg border border-red-500/30">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <div>
                      <p className="font-medium">Critical Drift Detected</p>
                      <p className="text-xs text-gray-400">Storage encryption disabled on 3 accounts</p>
                    </div>
                  </div>
                  <span className="text-xs text-red-400">2 min ago</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                  <div className="flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    <div>
                      <p className="font-medium">Policy Deviation</p>
                      <p className="text-xs text-gray-400">Network security group rules modified</p>
                    </div>
                  </div>
                  <span className="text-xs text-yellow-400">15 min ago</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-blue-900/20 rounded-lg border border-blue-500/30">
                  <div className="flex items-center gap-3">
                    <Info className="w-5 h-5 text-blue-400" />
                    <div>
                      <p className="font-medium">Configuration Change</p>
                      <p className="text-xs text-gray-400">Auto-scaling parameters updated</p>
                    </div>
                  </div>
                  <span className="text-xs text-blue-400">1 hour ago</span>
                </div>
              </div>

              <div className="mt-4 p-3 bg-gray-800 rounded-lg">
                <p className="text-xs text-gray-400 mb-2">Drift Statistics (Last 24h)</p>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <p className="text-2xl font-bold text-red-400">3</p>
                    <p className="text-xs text-gray-500">Critical</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-yellow-400">7</p>
                    <p className="text-xs text-gray-500">Warning</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-400">12</p>
                    <p className="text-xs text-gray-500">Info</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Predictive Timeline */}
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <LineChart className="w-5 h-5 text-green-400" />
                Predictive Compliance Timeline
              </h3>

              <div className="space-y-3">
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-700"></div>
                  
                  <div className="relative pl-10 pb-4">
                    <div className="absolute left-3 w-2 h-2 bg-green-400 rounded-full"></div>
                    <div className="bg-green-900/20 rounded-lg p-3 border border-green-500/30">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-medium text-green-400">Now</p>
                        <span className="text-xs bg-green-500/20 px-2 py-0.5 rounded">94% Compliant</span>
                      </div>
                      <p className="text-sm text-gray-400">Current compliance status</p>
                    </div>
                  </div>

                  <div className="relative pl-10 pb-4">
                    <div className="absolute left-3 w-2 h-2 bg-yellow-400 rounded-full"></div>
                    <div className="bg-yellow-900/20 rounded-lg p-3 border border-yellow-500/30">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-medium text-yellow-400">In 7 days</p>
                        <span className="text-xs bg-yellow-500/20 px-2 py-0.5 rounded">87% Predicted</span>
                      </div>
                      <p className="text-sm text-gray-400">5 certificates expiring</p>
                      <p className="text-xs text-yellow-400 mt-1">⚠️ Action Required</p>
                    </div>
                  </div>

                  <div className="relative pl-10 pb-4">
                    <div className="absolute left-3 w-2 h-2 bg-red-400 rounded-full"></div>
                    <div className="bg-red-900/20 rounded-lg p-3 border border-red-500/30">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-medium text-red-400">In 14 days</p>
                        <span className="text-xs bg-red-500/20 px-2 py-0.5 rounded">72% Risk</span>
                      </div>
                      <p className="text-sm text-gray-400">3 policies becoming obsolete</p>
                      <p className="text-xs text-red-400 mt-1">🚨 Critical: Update required</p>
                    </div>
                  </div>

                  <div className="relative pl-10">
                    <div className="absolute left-3 w-2 h-2 bg-blue-400 rounded-full"></div>
                    <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-500/30">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-medium text-blue-400">In 30 days</p>
                        <span className="text-xs bg-blue-500/20 px-2 py-0.5 rounded">Forecast</span>
                      </div>
                      <p className="text-sm text-gray-400">Predicted compliance: 91%</p>
                      <p className="text-xs text-blue-400 mt-1">With recommended actions applied</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-500/30">
                <p className="text-xs text-purple-400 mb-1">ML Model Confidence</p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-gray-700 rounded-full h-2">
                    <div className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                  </div>
                  <span className="text-sm font-medium">92%</span>
                </div>
              </div>
            </div>
          </div>

          {/* ML Model Performance Metrics */}
          <div className="mt-6 grid grid-cols-4 gap-4">
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
              <BarChart3 className="w-5 h-5 text-green-400 mx-auto mb-2" />
              <p className="text-2xl font-bold">99.2%</p>
              <p className="text-xs text-gray-400">Prediction Accuracy</p>
            </div>
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
              <Zap className="w-5 h-5 text-yellow-400 mx-auto mb-2" />
              <p className="text-2xl font-bold">&lt;100ms</p>
              <p className="text-xs text-gray-400">Inference Latency</p>
            </div>
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
              <Target className="w-5 h-5 text-blue-400 mx-auto mb-2" />
              <p className="text-2xl font-bold">&lt;2%</p>
              <p className="text-xs text-gray-400">False Positive Rate</p>
            </div>
            <div className="bg-white/5 dark:bg-gray-900/50 rounded-lg p-3 text-center">
              <Activity className="w-5 h-5 text-purple-400 mx-auto mb-2" />
              <p className="text-2xl font-bold">10k/s</p>
              <p className="text-xs text-gray-400">Training Throughput</p>
            </div>
          </div>
        </div>

        {/* Compliance Frameworks with drill-in */}
        <FrameworksDrillDown />

        {/* Recent Violations */}
        <RecentViolations />

        {/* Live Data Display */}
        {data && (
          <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-4">Resources from Backend</h2>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <pre>{JSON.stringify(data, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function MetricCard({ title, value, trend, icon: Icon, color }: {
  title: string
  value: string
  trend: string
  icon: React.ElementType
  color: 'green' | 'blue' | 'yellow' | 'purple'
}) {
  const colors = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400'
  }

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{trend}</p>
        </div>
        <Icon className={`w-5 h-5 ${colors[color]}`} />
      </div>
    </div>
  )
}

function FrameworksDrillDown() {
  const [selected, setSelected] = useState<string | null>(null)
  const [selectedControl, setSelectedControl] = useState<string | null>(null)
  const frameworks = [
    { name: 'ISO 27001', status: 'Compliant', coverage: '98%', controls: ['A.5 Information security policies', 'A.8 Asset management'] },
    { name: 'SOC 2', status: 'Compliant', coverage: '96%', controls: ['CC1.1 Control Environment', 'CC6.1 Logical Access'] },
    { name: 'HIPAA', status: 'Partial', coverage: '82%', controls: ['164.312(a)(1) Access Control', '164.312(c)(1) Integrity'] },
    { name: 'PCI DSS', status: 'Compliant', coverage: '94%', controls: ['Req 3 Protect Stored Data', 'Req 10 Track and Monitor'] },
  ]

  const violations = [
    { resource: 'stprod001', type: 'Storage', policy: 'Encryption at rest', severity: 'High', age: '2h' },
    { resource: 'vm-prod-001', type: 'VM', policy: 'Missing tags', severity: 'Medium', age: '5h' },
  ]

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Compliance Frameworks</h2>
      {!selected ? (
        <div className="grid grid-cols-2 gap-4">
          {frameworks.map(f => (
            <button type="button" key={f.name} onClick={() => setSelected(f.name)} className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg text-left hover:bg-gray-200 dark:hover:bg-gray-800">
              <h3 className="font-medium">{f.name}</h3>
              <p className={`text-sm ${f.status === 'Compliant' ? 'text-green-400' : 'text-yellow-400'}`}>{f.status}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Coverage: {f.coverage}</p>
            </button>
          ))}
        </div>
      ) : !selectedControl ? (
        <div>
          <button type="button" className="text-xs text-gray-600 dark:text-gray-400 mb-3" onClick={() => setSelected(null)}>← Back</button>
          <h3 className="font-medium mb-3">{selected} Controls</h3>
          <div className="space-y-2">
            {frameworks.find(f => f.name === selected)!.controls.map(c => (
              <button type="button" key={c} onClick={() => setSelectedControl(c)} className="w-full text-left p-3 bg-gray-100 dark:bg-gray-800/50 rounded hover:bg-gray-200 dark:hover:bg-gray-800">
                {c}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div>
          <button type="button" className="text-xs text-gray-600 dark:text-gray-400 mb-3" onClick={() => setSelectedControl(null)}>← Back to controls</button>
          <h3 className="font-medium mb-3">Violations for: {selectedControl}</h3>
          <div className="space-y-2">
            {violations.map(v => (
              <div key={v.resource} className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded">
                <div>
                  <p className="font-medium">{v.policy}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{v.resource} • {v.type}</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-sm ${v.severity === 'High' ? 'text-orange-400' : 'text-yellow-400'}`}>{v.severity}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{v.age}</span>
                  <button type="button" className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded">Auto-remediate</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function RecentViolations() {
  const items = [
    { policy: 'Data Encryption at Rest', resource: 'storage-account-prod-01', severity: 'High', age: '2 hours ago' },
    { policy: 'Network Security Group Rules', resource: 'vm-web-server-03', severity: 'Medium', age: '5 hours ago' },
    { policy: 'Access Control Policy', resource: 'keyvault-secrets-prod', severity: 'Critical', age: '1 day ago' },
  ]
  const map: any = { Critical: 'text-red-400', High: 'text-orange-400', Medium: 'text-yellow-400' }
  return (
    <div className="bg-white/50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Recent Violations</h2>
      <div className="space-y-3">
        {items.map(v => (
          <div key={v.resource} className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded">
            <div>
              <p className="font-medium">{v.policy}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{v.resource}</p>
            </div>
            <div className="flex items-center gap-4">
              <span className={`text-sm ${map[v.severity]}`}>{v.severity}</span>
              <span className="text-xs text-gray-500 dark:text-gray-400">{v.age}</span>
              <button type="button" className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded">Remediate</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}