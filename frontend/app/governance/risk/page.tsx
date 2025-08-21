"use client"

import { useMemo } from 'react'

type Risk = { id: string; area: string; level: 'High'|'Medium'|'Low'; description: string; owners: string[] }

export default function RiskManagementPage() {
  const risks: Risk[] = useMemo(() => ([
    { id: 'R-001', area: 'Security', level: 'High', description: 'Unencrypted storage accounts in prod', owners: ['secops@company.com'] },
    { id: 'R-002', area: 'Cost', level: 'Medium', description: 'Oversized VMs in staging', owners: ['finops@company.com'] },
    { id: 'R-003', area: 'Operations', level: 'Low', description: 'Backup job intermittently failing', owners: ['sre@company.com'] },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Risk Register</h1>
        <p className="text-sm text-gray-400">Top risks with owners and mitigation</p>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50 border-b border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">ID</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Area</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Level</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Description</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Owners</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Action</th>
            </tr>
          </thead>
          <tbody>
            {risks.map(r => (
              <tr key={r.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="px-6 py-4 font-mono text-sm">{r.id}</td>
                <td className="px-6 py-4">{r.area}</td>
                <td className="px-6 py-4"><span className={`text-xs px-2 py-1 rounded ${r.level==='High'?'bg-red-900/40 text-red-400':r.level==='Medium'?'bg-yellow-900/40 text-yellow-400':'bg-green-900/40 text-green-400'}`}>{r.level}</span></td>
                <td className="px-6 py-4 text-sm text-gray-300">{r.description}</td>
                <td className="px-6 py-4 text-sm text-gray-400">{r.owners.join(', ')}</td>
                <td className="px-6 py-4"><button className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded">Mitigate</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}