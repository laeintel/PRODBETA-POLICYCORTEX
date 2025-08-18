'use client';
import React, { useEffect, useState } from 'react'
import { api } from '../../../lib/api-client'
import { Shield, AlertTriangle, CheckCircle, Zap, BarChart3, FileText } from 'lucide-react'

export default function Page() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string|null>(null)

  useEffect(() => {
    const load = async () => {
      const resp = await api.getPoliciesDeep()
      if (resp.error) setError(resp.error); else setData(resp.data)
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div className="p-6 text-gray-300">Loading Policy Dashboard…</div>
  if (error) return <div className="p-6 text-red-400">{error}</div>

  const summary = data?.summary || {}
  const violations = data?.violations || []
  const recommendations = data?.recommendations || []

  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Policies" value={summary.totalPolicies || 0} icon={Shield} color="text-purple-400" />
        <Card title="Compliant" value={summary.compliant || 0} icon={CheckCircle} color="text-green-400" />
        <Card title="Violations" value={summary.violations || 0} icon={AlertTriangle} color="text-red-400" />
        <Card title="Automations" value={summary.automations || 0} icon={Zap} color="text-yellow-400" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Panel title="Recent Violations">
          <div className="space-y-3">
            {(violations as any[]).slice(0,8).map((v,i)=> (
              <div key={i} className="p-3 border border-white/10 rounded-lg flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">{v.policy || v.policy_name || 'Policy'}</div>
                  <div className="text-xs text-gray-400">{v.resource || v.resource_id}</div>
                </div>
                <span className="text-xs px-2 py-1 rounded bg-red-900/30 text-red-400">{v.severity || 'high'}</span>
              </div>
            ))}
            {violations.length === 0 && <div className="text-sm text-gray-400">No recent violations</div>}
          </div>
        </Panel>

        <Panel title="AI Recommendations">
          <div className="space-y-3">
            {(recommendations as any[]).slice(0,8).map((r,i)=> (
              <div key={i} className="p-3 border border-white/10 rounded-lg flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">{r.title || 'Recommendation'}</div>
                  <div className="text-xs text-gray-400">{r.description || ''}</div>
                </div>
                <button onClick={async ()=>{ await api.createAction('global','enforce_policies',{source:'policy-dashboard'}) }} className="px-3 py-1 bg-purple-600 text-white text-xs rounded">Run</button>
              </div>
            ))}
            {recommendations.length === 0 && <div className="text-sm text-gray-400">No recommendations</div>}
          </div>
        </Panel>
      </div>

      <Panel title="Compliance Trend">
        <div className="h-40 flex items-end gap-2">
          {Array.from({length: 24}).map((_,i)=>{
            const val = 60 + Math.round(Math.random()*40)
            return <div key={i} style={{height:`${val}%`}} className={`flex-1 bg-green-500/40`} />
          })}
        </div>
      </Panel>

      <div className="flex gap-3">
        <button onClick={()=> api.enforcePolicies()} className="px-4 py-2 bg-green-600 text-white rounded flex items-center gap-2"><BarChart3 className="w-4 h-4"/> Enforce Policies</button>
        <button onClick={()=> api.createAction('global','generate_policy_report')} className="px-4 py-2 bg-gray-800 border border-white/10 text-white rounded flex items-center gap-2"><FileText className="w-4 h-4"/> Generate Report</button>
      </div>
    </div>
  )
}

function Card({ title, value, icon:Icon, color }:{title:string; value:number|string; icon:any; color:string}){
  return (
    <div className="p-4 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-between">
      <div>
        <div className="text-sm text-gray-300">{title}</div>
        <div className="text-2xl font-bold text-white">{value}</div>
      </div>
      <Icon className={`w-6 h-6 ${color}`} />
    </div>
  )
}

function Panel({ title, children }:{title:string; children:React.ReactNode}){
  return (
    <div className="p-4 rounded-xl bg-white/10 backdrop-blur-md border border-white/20">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold text-white">{title}</div>
      </div>
      {children}
    </div>
  )
}