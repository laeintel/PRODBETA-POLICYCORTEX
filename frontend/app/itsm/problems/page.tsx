'use client'

import { useState } from 'react'
import { AlertTriangle, TrendingUp, Users, Clock, Target, Activity } from 'lucide-react'

export default function ProblemsPage() {
  const [problems] = useState([
    { id: 'PRB-001', title: 'Recurring Database Deadlocks', category: 'Performance', status: 'investigating', priority: 'high', incidents: 12, rootCause: 'Under Investigation', owner: 'Database Team' },
    { id: 'PRB-002', title: 'Memory Leak in API Service', category: 'Application', status: 'identified', priority: 'critical', incidents: 8, rootCause: 'Identified', owner: 'Platform Team' },
    { id: 'PRB-003', title: 'Network Packet Loss - Region EU', category: 'Network', status: 'workaround', priority: 'medium', incidents: 5, rootCause: 'ISP Issue', owner: 'Network Team' },
    { id: 'PRB-004', title: 'SSL Certificate Renewal Failures', category: 'Security', status: 'resolved', priority: 'low', incidents: 3, rootCause: 'Process Gap', owner: 'Security Team' }
  ])

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Problem Management</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">Identify and resolve root causes of incidents</p>
        </div>
        <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          Create Problem Record
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">7</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Active Problems</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">4</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Under Investigation</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">28</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Related Incidents</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">12</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Resolved (30d)</div>
        </div>
      </div>

      <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700">
        <div className="p-4 border-b border-border dark:border-gray-700">
          <h2 className="text-lg font-semibold text-foreground dark:text-white">Problem Records</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900">
              <tr>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Problem ID</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Title</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Category</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Priority</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Incidents</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Root Cause</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Owner</th>
              </tr>
            </thead>
            <tbody>
              {problems.map((problem) => (
                <tr key={problem.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4 font-medium text-foreground dark:text-white">{problem.id}</td>
                  <td className="p-4 text-foreground dark:text-gray-300">{problem.title}</td>
                  <td className="p-4">
                    <span className="px-2 py-1 rounded text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">
                      {problem.category}
                    </span>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${
                      problem.status === 'resolved' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      problem.status === 'identified' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      problem.status === 'workaround' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
                    }`}>
                      {problem.status}
                    </span>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${
                      problem.priority === 'critical' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      problem.priority === 'high' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                      problem.priority === 'medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                    }`}>
                      {problem.priority}
                    </span>
                  </td>
                  <td className="p-4 text-center">
                    <span className="font-medium text-foreground dark:text-white">{problem.incidents}</span>
                  </td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">{problem.rootCause}</td>
                  <td className="p-4 text-sm text-foreground dark:text-gray-300">{problem.owner}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}