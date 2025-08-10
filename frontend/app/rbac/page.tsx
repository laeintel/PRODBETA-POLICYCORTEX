'use client'

import { useMemo, useState } from 'react'
import AppLayout from '../../components/AppLayout'
import { useRbacAssignments } from '../../lib/azure-api'
import { Search, Users, Shield, Filter, AlertTriangle, BarChart3 } from 'lucide-react'
import { ChartCard, RiskSurface } from '../../components/ChartCards'
import FilterBar from '../../components/FilterBar'

export default function RbacPage() {
  const { assignments, loading } = useRbacAssignments()
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<'all' | 'User' | 'ServicePrincipal'>('all')
  const [onlyPrivileged, setOnlyPrivileged] = useState(false)

  const rows = (assignments || []).filter(a => {
    const q = query.toLowerCase()
    return (
      a.principalName.toLowerCase().includes(q) ||
      a.roleName.toLowerCase().includes(q) ||
      a.scope.toLowerCase().includes(q)
    )
  }).filter(a => (typeFilter === 'all' ? true : a.principalType === typeFilter))

  // Basic privileged-role heuristic
  const privilegedRoles = new Set(['Owner','User Access Administrator','Contributor'])
  const filtered = onlyPrivileged ? rows.filter(a => privilegedRoles.has(a.roleName)) : rows

  const stats = useMemo(() => {
    const total = assignments?.length || 0
    const sp = (assignments || []).filter(a => a.principalType === 'ServicePrincipal').length
    const privileged = (assignments || []).filter(a => privilegedRoles.has(a.roleName)).length
    return { total, sp, privileged, users: total - sp }
  }, [assignments])

  const riskChartData = useMemo(() => (
    [
      { metric: 'Privilege', score: Math.min(100, (stats.privileged / Math.max(stats.total,1)) * 100) },
      { metric: 'Service Accts', score: Math.min(100, (stats.sp / Math.max(stats.total,1)) * 100) },
      { metric: 'Exposure', score: 50 },
      { metric: 'Usage', score: 35 },
      { metric: 'Scope', score: 40 },
    ]
  ), [stats])

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

          <div className="mb-4">
            <FilterBar
              facets={[
                { key: 'subscription', label: 'Subscription' },
                { key: 'resourceGroup', label: 'Resource Group' },
                { key: 'scope', label: 'Scope filter (contains)' },
                { key: 'role', label: 'Role' },
              ]}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            {/* Summary */}
            <div className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-400">Total Assignments</p>
                  <p className="text-2xl font-bold text-white">{stats.total}</p>
                </div>
                <Users className="w-5 h-5 text-green-400" />
              </div>
              <div className="grid grid-cols-3 gap-3 mt-3 text-center">
                <div>
                  <p className="text-sm text-purple-300">Users</p>
                  <p className="text-white font-semibold">{stats.users}</p>
                </div>
                <div>
                  <p className="text-sm text-purple-300">Service</p>
                  <p className="text-white font-semibold">{stats.sp}</p>
                </div>
                <div>
                  <p className="text-sm text-purple-300">Privileged</p>
                  <p className="text-white font-semibold">{stats.privileged}</p>
                </div>
              </div>
            </div>

            {/* Risk Surface */}
            <ChartCard title="RBAC Risk Surface" subtitle="Heuristic view">
              <RiskSurface data={riskChartData as any} />
            </ChartCard>

            {/* Filters */}
            <div className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
              <div className="flex items-center gap-2 mb-3">
                <Filter className="w-4 h-4 text-gray-400" />
                <p className="text-sm text-gray-300">Filters</p>
              </div>
              <div className="flex items-center gap-3">
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value as any)}
                  className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
                >
                  <option value="all">All Principals</option>
                  <option value="User">Users</option>
                  <option value="ServicePrincipal">Service Principals</option>
                </select>
                <label className="flex items-center gap-2 text-sm text-gray-300">
                  <input type="checkbox" className="accent-purple-500" checked={onlyPrivileged} onChange={(e)=>setOnlyPrivileged(e.target.checked)} />
                  Privileged only
                </label>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 mb-4">
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

          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden sticky top-2">
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
                  {(loading ? [] : filtered).map(a => (
                    <tr key={a.id} className="hover:bg-white/5 cursor-pointer" onClick={() => {
                      if (typeof window !== 'undefined') {
                        window.location.href = `/rbac/${encodeURIComponent(a.principalId)}`
                      }
                    }}>
                      <td className="px-6 py-3 text-white text-sm">{a.principalName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.roleName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.principalType}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.scope}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.createdDate}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.lastUsed || '-'}</td>
                    </tr>
                  ))}
                  {(!loading && filtered.length === 0) && (
                    <tr>
                      <td colSpan={6} className="px-6 py-6 text-sm text-gray-400">
                        <div className="flex items-center gap-2"><AlertTriangle className="w-4 h-4 text-yellow-400"/> No assignments found. Ensure Azure access is configured and try widening your filters.</div>
                      </td>
                    </tr>
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


