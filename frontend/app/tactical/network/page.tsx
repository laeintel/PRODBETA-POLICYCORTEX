'use client';
import React, { useEffect, useState } from 'react'
import { api } from '../../../lib/api-client'
import { Network, AlertTriangle } from 'lucide-react'

export default function Page(){
  const [data,setData]=useState<any>(null)
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState<string|null>(null)
  useEffect(()=>{(async()=>{ const resp=await api.getCorrelations(); if(resp.error) setError(resp.error); else setData(resp.data); setLoading(false) })()},[])
  if(loading) return <div className="p-6 text-gray-300">Loading Network Insights…</div>
  if(error) return <div className="p-6 text-red-400">{error}</div>
  const problems = (data?.problems||[]).slice(0,10)
  const flows = (data?.flows||[]).slice(0,30)
  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {problems.map((p:any,i:number)=> (
          <div key={i} className="p-4 bg-white/10 rounded-xl border border-white/20">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-semibold text-white">{p.title||'Issue'}</div>
              <AlertTriangle className="w-4 h-4 text-yellow-400"/>
            </div>
            <div className="text-sm text-gray-300">{p.details||p.description||'Detected anomaly'}</div>
          </div>
        ))}
        {problems.length===0 && <div className="p-4 text-gray-400">No active problems detected</div>}
      </div>
      <div className="p-4 bg-white/10 rounded-xl border border-white/20">
        <div className="text-sm font-semibold text-white mb-3">Top Flows</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-gray-400"><th className="py-2">Source</th><th>Dest</th><th>Protocol</th><th className="text-right">Gb/s</th></tr></thead>
            <tbody className="divide-y divide-white/10">
              {flows.map((f:any,i:number)=> (
                <tr key={i} className="hover:bg-white/5">
                  <td className="py-2">{f.src||'-'}</td>
                  <td>{f.dst||'-'}</td>
                  <td>{f.proto||'-'}</td>
                  <td className="text-right">{f.rate||0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}