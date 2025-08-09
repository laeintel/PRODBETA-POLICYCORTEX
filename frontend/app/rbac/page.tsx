'use client'

import { useState } from 'react'
import AppLayout from '../../components/AppLayout'
import { useRbacAssignments } from '../../lib/azure-api'
import { Search, Users, Shield } from 'lucide-react'

export default function RbacPage() {
  const { assignments, loading } = useRbacAssignments()
  const [query, setQuery] = useState('')

  const rows = (assignments || []).filter(a => {
    const q = query.toLowerCase()
    return (
      a.principalName.toLowerCase().includes(q) ||
      a.roleName.toLowerCase().includes(q) ||
      a.scope.toLowerCase().includes(q)
    )
  })

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
              <Users className="w-6 h-6 text-green-400" />
              RBAC & Permissions
            </h1>
            <p className="text-gray-400">Review current role assignments and usage</p>
          </div>

          <div className="flex items-center gap-3 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by user, role, or scope..."
                className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
              />
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Principal</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Role</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Scope</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Created</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Last Used</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {(loading ? [] : rows).map(a => (
                    <tr key={a.id} className="hover:bg-white/5">
                      <td className="px-6 py-3 text-white text-sm">{a.principalName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.roleName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.principalType}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.scope}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.createdDate}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.lastUsed || '-'}</td>
                    </tr>
                  ))}
                  {(!loading && rows.length === 0) && (
                    <tr><td colSpan={6} className="px-6 py-6 text-sm text-gray-400">No assignments match your search.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}


