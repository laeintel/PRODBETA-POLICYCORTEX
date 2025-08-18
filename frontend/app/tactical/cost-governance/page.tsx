'use client';
import React, { useEffect, useState } from 'react'
import { api } from '../../../lib/api-client'
import { DollarSign, AlertTriangle, TrendingUp, Download } from 'lucide-react'

export default function Page(){
  const [data,setData]=useState<any>(null)
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState<string|null>(null)
  useEffect(()=>{(async()=>{ const resp=await api.getCostsDeep(); if(resp.error) setError(resp.error); else setData(resp.data); setLoading(false) })()},[])
  if(loading) return <div className="p-6 text-gray-300">Loading Cost Governance…</div>
  if(error) return <div className="p-6 text-red-400">{error}</div>
  const summary = data?.summary||{}
  const anomalies = data?.anomalies||[]
  const topTalkers = data?.topTalkers||[]
  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Current Spend" value={`$${(summary.current||0).toLocaleString()}`} icon={DollarSign} color="text-green-400"/>
        <Card title="Forecast" value={`$${(summary.forecast||0).toLocaleString()}`} icon={TrendingUp} color="text-yellow-400"/>
        <Card title="Anomalies" value={anomalies.length||0} icon={AlertTriangle} color="text-red-400"/>
        <Card title="Savings Identified" value={`$${(summary.savings||0).toLocaleString()}`} icon={DollarSign} color="text-blue-400"/>
      </div>
      <Panel title="Top Talker Cost Attribution">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-gray-400"><th className="py-2">Subscription</th><th>Name</th><th>Subnet</th><th>Region</th><th className="text-right">Flows</th></tr></thead>
            <tbody className="divide-y divide-white/10">
              {(topTalkers as any[]).slice(0,10).map((t,i)=> (
                <tr key={i} className="hover:bg-white/5">
                  <td className="py-2">{t.subscription||'-'}</td>
                  <td>{t.name||t.interface||'-'}</td>
                  <td>{t.subnet||'-'}</td>
                  <td>{t.region||'-'}</td>
                  <td className="text-right">{(t.flows||0).toLocaleString()}</td>
                </tr>
              ))}
              {topTalkers.length===0 && <tr><td className="py-3 text-gray-400" colSpan={5}>No data</td></tr>}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="flex gap-3">
        <button onClick={()=> api.createAction('global','optimize_costs')} className="px-4 py-2 bg-green-600 text-white rounded flex items-center gap-2"><TrendingUp className="w-4 h-4"/> Optimize</button>
        <button onClick={()=> api.createAction('global','export_cost_report')} className="px-4 py-2 bg-gray-800 border border-white/10 text-white rounded flex items-center gap-2"><Download className="w-4 h-4"/> Export Report</button>
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
      <div className="flex items-center justify-between mb-3"><div className="text-sm font-semibold text-white">{title}</div></div>
      {children}
    </div>
  )
}