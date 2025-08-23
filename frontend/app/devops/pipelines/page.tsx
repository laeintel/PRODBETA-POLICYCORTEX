"use client"

import { useMemo, useState } from 'react'
import { GitBranch, CheckCircle, XCircle, Clock } from 'lucide-react'

type Pipeline = { name: string; branch: string; status: 'running'|'idle'|'failed'; lastRun: string; runs: Run[] }
type Run = { id: string; status: 'success'|'failed'|'running'; duration: string; time: string }

export default function CICDPipelinesPage() {
  const [selected, setSelected] = useState<Pipeline | null>(null)
  const [selectedRun, setSelectedRun] = useState<Run | null>(null)
  const pipelines: Pipeline[] = useMemo(() => ([
    {
      name: 'Main Branch CI/CD', branch: 'main', status: 'running', lastRun: '2m ago',
      runs: [
        { id: '#1240', status: 'running', duration: '3m 12s', time: '2m ago' },
        { id: '#1239', status: 'success', duration: '4m 03s', time: '1h ago' },
        { id: '#1238', status: 'failed', duration: '1m 18s', time: '3h ago' },
      ]
    },
    {
      name: 'PR Validation', branch: 'feature/auth', status: 'idle', lastRun: '1h ago',
      runs: [ { id: '#88', status: 'success', duration: '2m 44s', time: '1h ago' } ]
    },
  ]), [])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-white p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">CI/CD Pipelines</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400">Pipelines with drill-in run history</p>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-100 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-800">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Pipeline</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Branch</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Last Run</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Status</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-gray-600 dark:text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {pipelines.map(p => (
              <tr key={p.name} className="border-b border-gray-200 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                <td className="px-6 py-4 font-medium">{p.name}</td>
                <td className="px-6 py-4 text-gray-600 dark:text-gray-400 text-sm">{p.branch}</td>
                <td className="px-6 py-4 text-gray-600 dark:text-gray-400 text-sm">{p.lastRun}</td>
                <td className="px-6 py-4">
                  <StatusBadge status={p.status} />
                </td>
                <td className="px-6 py-4">
                  <button type="button" className="text-xs px-3 py-1 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 rounded" onClick={() => setSelected(p)}>View runs</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="fixed right-0 top-0 h-full w-[520px] bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 p-6 overflow-y-auto z-50">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-lg font-semibold">{selected.name}</h2>
              <p className="text-xs text-gray-600 dark:text-gray-400">Branch: {selected.branch}</p>
            </div>
            <button type="button" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-2xl" onClick={() => { setSelected(null); setSelectedRun(null) }}>✕</button>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {selected.runs.map(run => (
              <button type="button" key={run.id} onClick={() => setSelectedRun(run)} className="p-3 bg-gray-100 dark:bg-gray-800/60 rounded hover:bg-gray-200 dark:hover:bg-gray-800 text-left">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm">{run.id}</span>
                  <RunStatus status={run.status} />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">{run.time} • {run.duration}</p>
              </button>
            ))}
          </div>

          {selectedRun && (
            <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800/50 rounded">
              <h3 className="font-medium mb-2">Run Details {selectedRun.id}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Status: {selectedRun.status} • Duration: {selectedRun.duration}</p>
              <div className="text-xs text-gray-700 dark:text-gray-300 font-mono bg-gray-50 dark:bg-gray-900 p-3 rounded border border-gray-200 dark:border-gray-800">
{`[step] checkout@v3 ... ok\n[step] install deps ... ok\n[step] build ... ok\n[step] test ... ${selectedRun.status === 'failed' ? 'failed' : 'ok'}`}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: 'running'|'idle'|'failed' }) {
  const map = { running: 'text-blue-600 dark:text-blue-400', idle: 'text-gray-600 dark:text-gray-400', failed: 'text-red-600 dark:text-red-400' }
  return <span className={`text-sm font-medium ${map[status]}`}>{status}</span>
}

function RunStatus({ status }: { status: 'success'|'failed'|'running' }) {
  if (status === 'success') return <span className="text-green-600 dark:text-green-400 text-xs inline-flex items-center gap-1"><CheckCircle className="w-3 h-3"/> success</span>
  if (status === 'failed') return <span className="text-red-600 dark:text-red-400 text-xs inline-flex items-center gap-1"><XCircle className="w-3 h-3"/> failed</span>
  return <span className="text-blue-600 dark:text-blue-400 text-xs inline-flex items-center gap-1"><Clock className="w-3 h-3"/> running</span>
}