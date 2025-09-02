'use client'

import { useState } from 'react'
import { AlertCircle, Clock, Users, TrendingUp, CheckCircle, AlertTriangle, XCircle, Filter } from 'lucide-react'

export default function IncidentsPage() {
  const [incidents] = useState([
    { id: 'INC-001', title: 'API Gateway High Latency', severity: 'high', status: 'in-progress', assignee: 'John Doe', created: '2 hours ago', sla: '2h remaining' },
    { id: 'INC-002', title: 'Database Connection Pool Exhausted', severity: 'critical', status: 'open', assignee: 'Sarah Admin', created: '30 minutes ago', sla: '30m remaining' },
    { id: 'INC-003', title: 'Storage Service Degradation', severity: 'medium', status: 'resolved', assignee: 'Mike Tech', created: '1 day ago', sla: 'Met' },
    { id: 'INC-004', title: 'Authentication Service Timeout', severity: 'high', status: 'in-progress', assignee: 'Lisa Dev', created: '4 hours ago', sla: 'Breached' },
    { id: 'INC-005', title: 'CDN Cache Miss Rate High', severity: 'low', status: 'open', assignee: 'Unassigned', created: '6 hours ago', sla: '18h remaining' }
  ])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      case 'high': return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30'
      case 'medium': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      case 'low': return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30'
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open': return <AlertCircle className="w-4 h-4" />
      case 'in-progress': return <Clock className="w-4 h-4" />
      case 'resolved': return <CheckCircle className="w-4 h-4" />
      case 'closed': return <XCircle className="w-4 h-4" />
      default: return <AlertTriangle className="w-4 h-4" />
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Incident Management</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">Track and resolve service incidents</p>
        </div>
        <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          Create Incident
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">2</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Critical</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">5</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">High Priority</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">8</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">In Progress</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">15</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Resolved Today</div>
        </div>
      </div>

      <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700">
        <div className="p-4 border-b border-border dark:border-gray-700">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold text-foreground dark:text-white">Active Incidents</h2>
            <button className="px-3 py-1 bg-muted text-foreground rounded hover:bg-accent transition-colors flex items-center gap-2">
              <Filter className="w-4 h-4" />
              Filter
            </button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900">
              <tr>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Incident ID</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Title</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Severity</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Assignee</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Created</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">SLA</th>
              </tr>
            </thead>
            <tbody>
              {incidents.map((incident) => (
                <tr key={incident.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4 font-medium text-foreground dark:text-white">{incident.id}</td>
                  <td className="p-4 text-foreground dark:text-gray-300">{incident.title}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getSeverityColor(incident.severity)}`}>
                      {incident.severity}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(incident.status)}
                      <span className="text-sm text-foreground dark:text-gray-300 capitalize">{incident.status.replace('-', ' ')}</span>
                    </div>
                  </td>
                  <td className="p-4 text-foreground dark:text-gray-300">{incident.assignee}</td>
                  <td className="p-4 text-muted-foreground dark:text-gray-400">{incident.created}</td>
                  <td className="p-4">
                    <span className={`text-sm font-medium ${
                      incident.sla === 'Breached' ? 'text-red-600 dark:text-red-400' :
                      incident.sla === 'Met' ? 'text-green-600 dark:text-green-400' :
                      'text-yellow-600 dark:text-yellow-400'
                    }`}>
                      {incident.sla}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}