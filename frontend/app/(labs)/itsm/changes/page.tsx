'use client'

import { useState } from 'react'
import { GitBranch, Calendar, Clock, CheckCircle, AlertCircle, XCircle, Users } from 'lucide-react'

export default function ChangesPage() {
  const [changes] = useState([
    { id: 'CHG-001', title: 'Database Schema Update', type: 'standard', status: 'approved', risk: 'medium', scheduledFor: '2024-01-15 02:00 UTC', requestor: 'Data Team' },
    { id: 'CHG-002', title: 'API Gateway Version Upgrade', type: 'emergency', status: 'in-progress', risk: 'high', scheduledFor: 'Immediate', requestor: 'Platform Team' },
    { id: 'CHG-003', title: 'Security Patch Deployment', type: 'standard', status: 'pending', risk: 'low', scheduledFor: '2024-01-16 04:00 UTC', requestor: 'Security Team' },
    { id: 'CHG-004', title: 'Network Configuration Update', type: 'normal', status: 'completed', risk: 'medium', scheduledFor: '2024-01-14 03:00 UTC', requestor: 'Network Team' },
    { id: 'CHG-005', title: 'Storage Capacity Expansion', type: 'standard', status: 'approved', risk: 'low', scheduledFor: '2024-01-17 05:00 UTC', requestor: 'Infrastructure' }
  ])

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      case 'medium': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30'
      case 'low': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Change Management</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">Plan and track infrastructure changes</p>
        </div>
        <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          Request Change
        </button>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">12</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Pending Approval</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">8</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Approved</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">3</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">In Progress</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">45</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Completed (30d)</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">2</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Failed</div>
        </div>
      </div>

      <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700">
        <div className="p-4 border-b border-border dark:border-gray-700">
          <h2 className="text-lg font-semibold text-foreground dark:text-white">Upcoming Changes</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900">
              <tr>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Change ID</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Title</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Type</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Risk</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Scheduled</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Requestor</th>
              </tr>
            </thead>
            <tbody>
              {changes.map((change) => (
                <tr key={change.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4 font-medium text-foreground dark:text-white">{change.id}</td>
                  <td className="p-4 text-foreground dark:text-gray-300">{change.title}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${
                      change.type === 'emergency' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      change.type === 'standard' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                    }`}>
                      {change.type}
                    </span>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${
                      change.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      change.status === 'approved' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      change.status === 'in-progress' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                    }`}>
                      {change.status.replace('-', ' ')}
                    </span>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getRiskColor(change.risk)}`}>
                      {change.risk}
                    </span>
                  </td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">{change.scheduledFor}</td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">{change.requestor}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}