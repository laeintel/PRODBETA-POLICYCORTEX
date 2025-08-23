"use client"

import { useMemo, useState } from 'react'

type Deployment = { env: string; version: string; status: 'healthy'|'updating'|'failed'; time: string; changelog: string[] }

export default function DeploymentManagementPage() {
  const [selected, setSelected] = useState<Deployment | null>(null)
  const deployments: Deployment[] = useMemo(() => ([
    { env: 'Production', version: 'v2.1.0', status: 'healthy', time: '2 hours ago', changelog: ['feat: new policy engine', 'fix: audit log writer'] },
    { env: 'Staging', version: 'v2.1.0-rc.2', status: 'healthy', time: '1 day ago', changelog: ['chore: deps update'] },
    { env: 'Development', version: 'v2.2.0-dev', status: 'updating', time: '10 minutes ago', changelog: ['wip: correlation tuning'] },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Deployments</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400">Environment deployments with details</p>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-100/50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Environment</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Version</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Status</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Deployed</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Action</th>
            </tr>
          </thead>
          <tbody>
            {deployments.map(d => (
              <tr key={d.env} className="border-b border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800/50">
                <td className="px-6 py-4">{d.env}</td>
                <td className="px-6 py-4 font-mono text-sm">{d.version}</td>
                <td className="px-6 py-4"><span className={`text-xs px-2 py-1 rounded ${d.status==='healthy'?'bg-green-900/40 text-green-400':d.status==='updating'?'bg-blue-900/40 text-blue-400':'bg-red-900/40 text-red-400'}`}>{d.status}</span></td>
                <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-400">{d.time}</td>
                <td className="px-6 py-4"><button type="button" className="text-xs px-3 py-1 bg-gray-200 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded" onClick={() => setSelected(d)}>Details</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-[520px] bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">{selected.env} • {selected.version}</h2>
              <p className="text-xs text-gray-600 dark:text-gray-400">Status: {selected.status} • {selected.time}</p>
            </div>
            <button type="button" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-2xl" onClick={() => setSelected(null)}>✕</button>
          </div>
          <div className="space-y-3">
            <div className="text-xs text-gray-700 dark:text-gray-300 font-mono bg-gray-100 dark:bg-gray-900 p-3 rounded border border-gray-200 dark:border-gray-800">
              {selected.changelog.map((c, i) => `- ${c}${i < selected.changelog.length-1 ? '\n' : ''}`)}
            </div>
            <button type="button" className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded">Roll back</button>
            <button type="button" className="w-full py-2 bg-gray-200 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">Promote</button>
          </div>
        </div>
      )}
    </div>
  )
}