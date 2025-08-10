'use client'

import React from 'react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, AreaChart, Area, BarChart, Bar, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'

type Point = { name: string; value: number; [key: string]: any }

interface CardProps {
  title: string
  subtitle?: string
  onClick?: () => void
  children: React.ReactNode
}

export function ChartCard({ title, subtitle, onClick, children }: CardProps) {
  return (
    <div
      className="p-4 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 hover:bg-white/15 transition-all cursor-default"
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-white text-sm font-medium">{title}</h3>
          {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
        </div>
      </div>
      <div className="h-56">
        {children}
      </div>
    </div>
  )
}

export function ComplianceTrend({ data }: { data: Point[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ left: 0, right: 0, top: 8, bottom: 0 }}>
        <defs>
          <linearGradient id="compGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#34d399" stopOpacity={0.6}/>
            <stop offset="95%" stopColor="#34d399" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
        <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8 }} />
        <Area type="monotone" dataKey="value" stroke="#34d399" fill="url(#compGradient)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function CostTrend({ data }: { data: Point[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ left: 0, right: 0, top: 8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
        <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8 }} />
        <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function RiskSurface({ data }: { data: Array<{ metric: string; score: number }> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={data} outerRadius="80%">
        <PolarGrid stroke="#334155" />
        <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
        <PolarRadiusAxis stroke="#94a3b8" angle={30} domain={[0, 100]} />
        <Radar name="Risk" dataKey="score" stroke="#ef4444" fill="#ef4444" fillOpacity={0.35} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

export function ServiceCostBar({ data }: { data: Array<{ name: string; monthly: number }> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ left: 0, right: 0, top: 8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
        <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 8 }} />
        <Legend />
        <Bar dataKey="monthly" name="Monthly" fill="#a78bfa" />
      </BarChart>
    </ResponsiveContainer>
  )
}


