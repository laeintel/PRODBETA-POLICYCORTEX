"use client"

import { useMemo, useState } from 'react'
import { CheckCircle, XCircle } from 'lucide-react'

type Build = { id: string; branch: string; status: 'success'|'failed'; time: string; artifacts: string[] }

export default function BuildStatusPage() {
  const [selected, setSelected] = useState<Build | null>(null)
  const builds: Build[] = useMemo(() => ([
    { id: '#1234', branch: 'main', status: 'success', time: '5 minutes ago', artifacts: ['frontend-v2.1.0.tar.gz', 'api-v2.1.0.zip'] },
    { id: '#1233', branch: 'feature/auth', status: 'failed', time: '1 hour ago', artifacts: [] },
    { id: '#1232', branch: 'main', status: 'success', time: '2 hours ago', artifacts: ['docs-v2.1.0.pdf'] },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Builds</h1>
        <p className="text-sm text-gray-400">Recent builds and artifacts</p>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50 border-b border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Build</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Branch</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Status</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Time</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-400">Action</th>
            </tr>
          </thead>
          <tbody>
            {builds.map(b => (
              <tr key={b.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="px-6 py-4 font-mono text-sm">{b.id}</td>
                <td className="px-6 py-4">{b.branch}</td>
                <td className="px-6 py-4">{b.status === 'success' ? <span className="text-green-400 inline-flex items-center gap-1 text-xs"><CheckCircle className="w-3 h-3"/> success</span> : <span className="text-red-400 inline-flex items-center gap-1 text-xs"><XCircle className="w-3 h-3"/> failed</span>}</td>
                <td className="px-6 py-4 text-sm text-gray-400">{b.time}</td>
                <td className="px-6 py-4"><button type="button" className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded" onClick={() => setSelected(b)}>Details</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-[520px] bg-gray-900 border-l border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">Build {selected.id}</h2>
              <p className="text-xs text-gray-400">Branch: {selected.branch} • Status: {selected.status}</p>
            </div>
            <button type="button" className="text-gray-400 hover:text-white text-2xl" onClick={() => setSelected(null)}>✕</button>
          </div>
          <div className="space-y-3">
            <div className="text-xs text-gray-300 font-mono bg-gray-900 p-3 rounded border border-gray-800">{`[build] install deps\n[build] run tests\n[build] compile\n[build] package`}</div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Artifacts</p>
              {selected.artifacts.length === 0 ? (
                <p className="text-sm text-gray-500">No artifacts</p>
              ) : (
                <ul className="text-sm list-disc list-inside text-gray-300">
                  {selected.artifacts.map(a => (<li key={a}>{a}</li>))}
                </ul>
              )}
            </div>
            <button type="button" className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded">Download artifacts</button>
          </div>
        </div>
      )}
    </div>
  )
}