'use client'

import { useState } from 'react'
import { FileText, Download, Calendar, TrendingUp, Shield, DollarSign, Users, Clock, Send, CheckCircle } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function ExecutiveReportsPage() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null)
  const [reportPeriod, setReportPeriod] = useState('monthly')
  const [scheduledReports, setScheduledReports] = useState<string[]>(['board-quarterly'])

  const reports = [
    {
      id: 'board-quarterly',
      name: 'Board Quarterly Report',
      description: 'Comprehensive governance and risk overview for board meetings',
      lastGenerated: '2024-01-30',
      frequency: 'Quarterly',
      recipients: ['board@company.com', 'ceo@company.com'],
      sections: ['Executive Summary', 'Risk Assessment', 'Compliance Status', 'Cost Optimization', 'Strategic Initiatives'],
      status: 'ready'
    },
    {
      id: 'cfo-monthly',
      name: 'CFO Monthly Cost Report',
      description: 'Detailed cloud spend analysis and optimization opportunities',
      lastGenerated: '2024-02-15',
      frequency: 'Monthly',
      recipients: ['cfo@company.com', 'finance@company.com'],
      sections: ['Cost Trends', 'Budget vs Actual', 'Optimization Savings', 'Forecasting', 'Department Breakdown'],
      status: 'ready'
    },
    {
      id: 'ciso-security',
      name: 'CISO Security Posture',
      description: 'Security metrics, incidents, and threat landscape analysis',
      lastGenerated: '2024-02-18',
      frequency: 'Weekly',
      recipients: ['ciso@company.com', 'security@company.com'],
      sections: ['Threat Overview', 'Incident Response', 'Vulnerability Status', 'Compliance Gaps', 'Risk Mitigation'],
      status: 'generating'
    },
    {
      id: 'cto-technical',
      name: 'CTO Technical Excellence',
      description: 'Engineering metrics, deployment velocity, and system reliability',
      lastGenerated: '2024-02-17',
      frequency: 'Bi-weekly',
      recipients: ['cto@company.com', 'engineering@company.com'],
      sections: ['Deployment Metrics', 'System Reliability', 'Performance KPIs', 'Technical Debt', 'Innovation Index'],
      status: 'ready'
    },
    {
      id: 'investor-annual',
      name: 'Investor Relations Annual',
      description: 'Year-end governance and compliance report for investors',
      lastGenerated: '2023-12-31',
      frequency: 'Annual',
      recipients: ['investors@company.com'],
      sections: ['Annual Performance', 'Risk Management', 'ESG Compliance', 'Strategic Outlook', 'Financial Impact'],
      status: 'scheduled'
    }
  ]

  const executiveSummary = {
    compliance: { score: 94, trend: 'up', target: 95 },
    risk: { score: 'Medium', count: 12, criticalCount: 2 },
    cost: { current: 485000, savings: 125000, trend: 'down' },
    security: { incidents: 3, blocked: 47, score: 89 },
    operations: { uptime: 99.97, deployments: 234, failures: 2 }
  }

  const keyMetrics = [
    { metric: 'Cloud Governance Score', value: 92, change: '+5%', status: 'green' },
    { metric: 'Policy Compliance Rate', value: 94, change: '+2%', status: 'green' },
    { metric: 'Cost Optimization Realized', value: 125000, change: '+18%', status: 'green' },
    { metric: 'Security Incidents Prevented', value: 47, change: '-23%', status: 'green' },
    { metric: 'Mean Time to Resolution', value: '1.2h', change: '-45%', status: 'green' },
    { metric: 'Automation Coverage', value: '78%', change: '+12%', status: 'yellow' }
  ]

  const boardHighlights = [
    {
      title: 'Cost Reduction Achievement',
      detail: '$1.5M annual savings identified, $1.2M realized',
      impact: 'Positive',
      action: 'Continue optimization initiatives'
    },
    {
      title: 'Compliance Improvement',
      detail: 'Zero critical violations for 6 consecutive months',
      impact: 'Positive',
      action: 'Maintain current controls'
    },
    {
      title: 'Security Posture Enhancement',
      detail: '67% reduction in security incidents YoY',
      impact: 'Positive',
      action: 'Expand AI threat detection'
    },
    {
      title: 'Operational Excellence',
      detail: '99.97% uptime, exceeding SLA by 0.47%',
      impact: 'Positive',
      action: 'Scale edge infrastructure'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Executive Reports
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Board-ready reports and executive dashboards
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Executive Summary</h3>
                <select 
                  value={reportPeriod}
                  onChange={(e) => setReportPeriod(e.target.value)}
                  className="px-3 py-1 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
                >
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                  <option value="quarterly">Quarterly</option>
                  <option value="annual">Annual</option>
                </select>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Compliance</span>
                    <TrendingUp className={`w-4 h-4 ${
                      executiveSummary.compliance.trend === 'up' ? 'text-green-500' : 'text-red-500'
                    }`} />
                  </div>
                  <div className="text-2xl font-bold">{executiveSummary.compliance.score}%</div>
                  <div className="text-xs text-gray-500">Target: {executiveSummary.compliance.target}%</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Risk Level</span>
                    <Shield className="w-4 h-4 text-orange-500" />
                  </div>
                  <div className="text-2xl font-bold">{executiveSummary.risk.score}</div>
                  <div className="text-xs text-gray-500">{executiveSummary.risk.count} risks ({executiveSummary.risk.criticalCount} critical)</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Cost Savings</span>
                    <DollarSign className="w-4 h-4 text-green-500" />
                  </div>
                  <div className="text-2xl font-bold">${(executiveSummary.cost.savings / 1000).toFixed(0)}K</div>
                  <div className="text-xs text-gray-500">This {reportPeriod}</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Security</span>
                    <Shield className="w-4 h-4 text-blue-500" />
                  </div>
                  <div className="text-2xl font-bold">{executiveSummary.security.score}%</div>
                  <div className="text-xs text-gray-500">{executiveSummary.security.blocked} threats blocked</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  </div>
                  <div className="text-2xl font-bold">{executiveSummary.operations.uptime}%</div>
                  <div className="text-xs text-gray-500">{executiveSummary.operations.deployments} deployments</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Incidents</span>
                    <Users className="w-4 h-4 text-purple-500" />
                  </div>
                  <div className="text-2xl font-bold">{executiveSummary.security.incidents}</div>
                  <div className="text-xs text-gray-500">This {reportPeriod}</div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium text-sm text-gray-700 dark:text-gray-300">Key Performance Indicators</h4>
                {keyMetrics.map((metric, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex-1">
                      <div className="font-medium">{metric.metric}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {typeof metric.value === 'number' && metric.value > 1000 
                          ? `$${(metric.value / 1000).toFixed(0)}K`
                          : typeof metric.value === 'number' && metric.value < 100
                          ? `${metric.value}%`
                          : metric.value}
                      </div>
                    </div>
                    <div className={`text-sm font-medium ${
                      metric.status === 'green' ? 'text-green-600' : 'text-yellow-600'
                    }`}>
                      {metric.change}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Board Highlights</h3>
              <div className="space-y-4">
                {boardHighlights.map((highlight, idx) => (
                  <div key={idx} className="border-l-4 border-blue-500 pl-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-medium">{highlight.title}</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{highlight.detail}</p>
                        <div className="flex items-center gap-4 mt-2">
                          <span className={`text-xs px-2 py-1 rounded ${
                            highlight.impact === 'Positive' 
                              ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                          }`}>
                            {highlight.impact} Impact
                          </span>
                          <span className="text-xs text-gray-500">Action: {highlight.action}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Available Reports</h3>
              <div className="space-y-3">
                {reports.map((report) => (
                  <div
                    key={report.id}
                    className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedReport === report.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
                    }`}
                    onClick={() => setSelectedReport(report.id)}
                  >
                    <div className="flex items-start justify-between mb-1">
                      <div className="font-medium text-sm">{report.name}</div>
                      <div className={`px-2 py-0.5 rounded text-xs font-medium ${
                        report.status === 'ready' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        report.status === 'generating' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400'
                      }`}>
                        {report.status === 'generating' ? 'GENERATING...' : report.status.toUpperCase()}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">{report.description}</div>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {report.frequency}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {report.lastGenerated}
                      </span>
                    </div>
                    
                    <div className="mt-3 flex gap-2">
                      <button className="flex-1 px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 transition-colors">
                        <Download className="w-3 h-3 inline mr-1" />
                        Download
                      </button>
                      <button className="flex-1 px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
                        <Send className="w-3 h-3 inline mr-1" />
                        Send
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Report Automation</h3>
              <p className="text-sm opacity-90 mb-4">
                Schedule automated reports for stakeholders with AI-generated insights
              </p>
              <button className="w-full px-4 py-2 bg-white text-indigo-600 rounded-lg hover:bg-gray-100 transition-colors">
                Configure Schedules
              </button>
            </div>
          </div>
        </div>

        <ChartContainer title="Governance Trends" className="mt-6">
          <div className="h-64 flex items-center justify-center text-gray-500">
            Interactive governance metrics visualization
          </div>
        </ChartContainer>
      </div>
    </div>
  )
}