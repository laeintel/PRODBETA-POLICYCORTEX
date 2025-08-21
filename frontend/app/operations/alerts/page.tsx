"use client"

import { useMemo, useState } from 'react'

type Alert = { id: string; title: string; severity: 'Critical'|'High'|'Medium'|'Low'; resource: string; age: string }

export default function AlertManagementPage() {
  const [selected, setSelected] = useState<Alert | null>(null)
  const alerts: Alert[] = useMemo(() => ([
    { id: 'A-001', title: 'CPU spike on prod-api-01', severity: 'High', resource: 'vm-001', age: '10m' },
    { id: 'A-002', title: 'Backup job failed', severity: 'Medium', resource: 'db-001', age: '1h' },
    { id: 'A-003', title: 'SSL cert near expiry', severity: 'Low', resource: 'api-gateway', age: '3d' },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Alerts</h1>
        <p className="text-sm text-gray-400">Open incidents and signals</p>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50 border-b border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">ID</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Title</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Severity</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Resource</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Age</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Action</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map(a => (
              <tr key={a.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="px-6 py-4 font-mono text-sm">{a.id}</td>
                <td className="px-6 py-4">{a.title}</td>
                <td className="px-6 py-4"><span className={`text-xs px-2 py-1 rounded ${a.severity==='Critical'?'bg-red-900/40 text-red-400':a.severity==='High'?'bg-orange-900/40 text-orange-400':a.severity==='Medium'?'bg-yellow-900/40 text-yellow-400':'bg-blue-900/40 text-blue-400'}`}>{a.severity}</span></td>
                <td className="px-6 py-4 text-sm text-gray-400">{a.resource}</td>
                <td className="px-6 py-4 text-sm text-gray-400">{a.age}</td>
                <td className="px-6 py-4"><button type="button" className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded" onClick={() => setSelected(a)}>View</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">{selected.title}</h2>
              <p className="text-xs text-gray-400">{selected.id} • {selected.resource}</p>
            </div>
            <button type="button" className="text-gray-400 hover:text-white text-2xl" onClick={() => setSelected(null)}>✕</button>
          </div>
          <div className="space-y-3">
            <Detail label="Severity" value={selected.severity} />
            <Detail label="Age" value={selected.age} />
            <Detail label="Recommended Action" value="Scale up VM or investigate noisy neighbor" />
            <button type="button" className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded">Create Incident</button>
            <button type="button" className="w-full py-2 bg-gray-800 hover:bg-gray-700 rounded">Silence 1h</button>
          </div>
        </div>
      )}
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-sm">{value}</p>
    </div>
  )
}