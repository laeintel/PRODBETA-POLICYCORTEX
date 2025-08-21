"use client"

import { useMemo, useState } from 'react'

type Policy = { id: string; name: string; category: string; severity: 'High'|'Medium'|'Low'; assignments: number; nonCompliant: number }

export default function PolicyManagementPage() {
  const [selected, setSelected] = useState<Policy | null>(null)
  const policies: Policy[] = useMemo(() => ([
    { id: 'pol-001', name: 'Require Encryption at Rest', category: 'Security', severity: 'High', assignments: 23, nonCompliant: 12 },
    { id: 'pol-002', name: 'Require Resource Tags', category: 'Governance', severity: 'Medium', assignments: 41, nonCompliant: 7 },
    { id: 'pol-003', name: 'Restrict Public IPs', category: 'Network', severity: 'High', assignments: 12, nonCompliant: 2 },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Policy Management</h1>
        <p className="text-sm text-gray-400">Policies with assignments and violations drill-in</p>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50 border-b border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Policy</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Category</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Severity</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Assignments</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Non-compliant</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {policies.map(p => (
              <tr key={p.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="px-6 py-4 font-medium">{p.name}</td>
                <td className="px-6 py-4 text-gray-400 text-sm">{p.category}</td>
                <td className="px-6 py-4"><span className={`text-xs px-2 py-1 rounded ${p.severity==='High'?'bg-red-900/40 text-red-400':p.severity==='Medium'?'bg-yellow-900/40 text-yellow-400':'bg-green-900/40 text-green-400'}`}>{p.severity}</span></td>
                <td className="px-6 py-4">{p.assignments}</td>
                <td className="px-6 py-4">{p.nonCompliant}</td>
                <td className="px-6 py-4">
                  <button className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded" onClick={() => setSelected(p)}>View assignments</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-[520px] bg-gray-900 border-l border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">{selected.name}</h2>
              <p className="text-xs text-gray-400">Assignments • Non-compliant resources</p>
            </div>
            <button className="text-gray-400 hover:text-white text-2xl" onClick={() => setSelected(null)}>✕</button>
          </div>
          <div className="space-y-2">
            {Array.from({ length: selected.assignments }).slice(0, 8).map((_, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-gray-800/50 rounded">
                <div>
                  <p className="font-medium">/subscriptions/.../resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stprod00{i}</p>
                  <p className="text-xs text-gray-400">Scope: subscription • Region: East US</p>
                </div>
                <button className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded">Remediate</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}