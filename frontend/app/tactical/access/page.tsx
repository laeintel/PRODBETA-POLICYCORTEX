'use client';
import React, { useEffect, useState } from 'react'
import { api } from '../../../lib/api-client'
import { Shield, Users, AlertTriangle } from 'lucide-react'

export default function Page(){
  const [data,setData]=useState<any>(null)
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState<string|null>(null)
  useEffect(()=>{(async()=>{ const resp = await api.getRbacDeep(); if(resp.error) setError(resp.error); else setData(resp.data); setLoading(false) })()},[])
  if(loading) return <div className="p-6 text-gray-300">Loading RBAC Analytics…</div>
  if(error) return <div className="p-6 text-red-400">{error}</div>
  const risk = data?.riskAnalysis||{}
  const assignments = data?.roleAssignments||[]
  const recs = data?.recommendations||[]
  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Privileged Accounts" value={risk.privilegedAccounts||0} icon={Shield} color="text-red-400"/>
        <Card title="High-Risk Assignments" value={risk.highRiskAssignments||0} icon={AlertTriangle} color="text-yellow-400"/>
        <Card title="Stale Assignments" value={risk.staleAssignments||0} icon={Users} color="text-gray-400"/>
        <Card title="Over-Privileged" value={(risk.overprivilegedIdentities||0)} icon={Users} color="text-orange-400"/>
      </div>
      <Panel title="Recent Role Assignments">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-gray-400"><th className="py-2">Principal</th><th>Role</th><th>Scope</th><th>Last Used</th></tr></thead>
            <tbody className="divide-y divide-white/10">
              {(assignments as any[]).slice(0,12).map((a:any,i:number)=> (
                <tr key={i} className="hover:bg-white/5">
                  <td className="py-2">{a.principalName||a.principal||'-'}</td>
                  <td>{a.roleName||a.role||'-'}</td>
                  <td>{a.scope||'-'}</td>
                  <td>{a.lastUsed||'-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel title="Recommendations">
        <ul className="list-disc list-inside text-sm text-gray-300">
          {(recs as any[]).slice(0,8).map((r,i)=> <li key={i}>{r}</li>)}
          {(!recs||recs.length===0)&& <li>No recommendations</li>}
        </ul>
      </Panel>
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
      <div className="flex items-center justify-between mb-3"><div className="text-sm font-semibold text-white">{title}</div></div>
      {children}
    </div>
  )
}