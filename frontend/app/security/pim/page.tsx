"use client"

import { useMemo, useState } from 'react'
import { Key, Clock, UserX } from 'lucide-react'

type Elevation = { user: string; role: string; requested: string; expires: string }

export default function PrivilegedIdentityManagementPage() {
  const [selected, setSelected] = useState<Elevation | null>(null)
  const elevations: Elevation[] = useMemo(() => ([
    { user: 'admin@company.com', role: 'Global Admin', requested: '2h ago', expires: '2h 15m' },
    { user: 'devops@company.com', role: 'Contributor', requested: '15m ago', expires: '45m' },
    { user: 'ops@company.com', role: 'Owner', requested: '5m ago', expires: '1h 10m' },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Privileged Identity Management (JIT)</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400">Active elevations and pending requests</p>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <Stat label="Eligible" value={127} />
        <Stat label="Active" value={8} />
        <Stat label="Pending" value={3} />
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-100 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">User</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Role</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Requested</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Expires</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {elevations.map((e) => (
              <tr key={`${e.user}-${e.role}`} className="border-b border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800/50">
                <td className="px-6 py-4">{e.user}</td>
                <td className="px-6 py-4">{e.role}</td>
                <td className="px-6 py-4 text-gray-600 dark:text-gray-400 text-sm">{e.requested}</td>
                <td className="px-6 py-4 text-yellow-600 dark:text-yellow-400 text-sm">{e.expires}</td>
                <td className="px-6 py-4">
                  <div className="flex gap-2">
                    <button type="button" className="text-xs px-3 py-1 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 rounded" onClick={() => setSelected(e)}>
                      View
                    </button>
                    <button type="button" className="text-xs px-3 py-1 bg-red-600 hover:bg-red-700 rounded inline-flex items-center gap-1">
                      <UserX className="w-3 h-3" /> Revoke
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">Elevation Details</h2>
              <p className="text-xs text-gray-600 dark:text-gray-400">{selected.user} • {selected.role}</p>
            </div>
            <button type="button" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-2xl" onClick={() => setSelected(null)}>✕</button>
          </div>
          <div className="space-y-3">
            <Detail label="User" value={selected.user} />
            <Detail label="Role" value={selected.role} />
            <Detail label="Requested" value={selected.requested} />
            <Detail label="Expires" value={selected.expires} />
            <button type="button" className="w-full py-2 bg-red-600 hover:bg-red-700 rounded">Revoke Access</button>
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <p className="text-xs text-gray-600 dark:text-gray-400">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-600 dark:text-gray-400">{label}</p>
      <p className="text-sm">{value}</p>
    </div>
  )
}