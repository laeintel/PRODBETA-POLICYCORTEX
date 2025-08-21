'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  Search, Filter, Download, RefreshCw, User, Shield, Key, 
  Activity, AlertCircle, CheckCircle, XCircle, Clock,
  ChevronRight, ChevronDown, Calendar, MapPin, Monitor,
  Database, Lock, Unlock, Settings, FileText, Trash2,
  UserPlus, UserMinus, Edit, Eye, Copy, Info, TrendingUp,
  ArrowUpRight, ArrowDownRight, BarChart3, Globe, X
} from 'lucide-react'
import { format, formatDistanceToNow, subDays, startOfDay, endOfDay } from 'date-fns'
import Link from 'next/link'

// Mock data for audit events
const generateMockAuditEvents = () => {
  const actions = [
    'LOGIN', 'LOGOUT', 'PERMISSION_GRANTED', 'PERMISSION_REVOKED', 
    'RESOURCE_CREATED', 'RESOURCE_MODIFIED', 'RESOURCE_DELETED',
    'POLICY_APPLIED', 'POLICY_VIOLATION', 'COMPLIANCE_CHECK',
    'ACCESS_DENIED', 'PASSWORD_CHANGED', 'MFA_ENABLED', 'MFA_DISABLED',
    'ROLE_ASSIGNED', 'ROLE_REMOVED', 'GROUP_JOINED', 'GROUP_LEFT',
    'API_KEY_CREATED', 'API_KEY_REVOKED', 'EXPORT_DATA', 'IMPORT_DATA'
  ]
  
  const users = [
    'john.doe@company.com', 'jane.smith@company.com', 'admin@company.com',
    'service.account@company.com', 'audit.bot@company.com', 'system@azure.com'
  ]
  
  const resources = [
    '/subscriptions/205b477d/resourceGroups/production',
    '/subscriptions/205b477d/vms/web-server-01',
    '/subscriptions/205b477d/storage/backups',
    '/policies/compliance/pci-dss',
    '/users/john.doe',
    '/groups/administrators',
    '/databases/customer-db',
    '/networks/vnet-main'
  ]
  
  const ipAddresses = [
    '192.168.1.100', '10.0.0.50', '172.16.0.10', 
    '52.255.128.14', '13.77.152.215', '40.78.195.137'
  ]
  
  const locations = [
    'Seattle, WA', 'New York, NY', 'London, UK', 
    'Tokyo, JP', 'Sydney, AU', 'Mumbai, IN'
  ]
  
  const events = []
  const now = new Date()
  
  for (let i = 0; i < 500; i++) {
    const timestamp = new Date(now.getTime() - Math.random() * 30 * 24 * 60 * 60 * 1000)
    const action = actions[Math.floor(Math.random() * actions.length)]
    const isSuccess = Math.random() > 0.1
    
    events.push({
      id: `evt-${i}`,
      timestamp,
      actor: users[Math.floor(Math.random() * users.length)],
      action,
      target: resources[Math.floor(Math.random() * resources.length)],
      result: (isSuccess ? 'SUCCESS' : 'FAILURE') as 'SUCCESS' | 'FAILURE',
      ipAddress: ipAddresses[Math.floor(Math.random() * ipAddresses.length)],
      location: locations[Math.floor(Math.random() * locations.length)],
      sessionId: `sess-${Math.floor(Math.random() * 100)}`,
      details: {
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        method: ['GET', 'POST', 'PUT', 'DELETE'][Math.floor(Math.random() * 4)],
        duration: Math.floor(Math.random() * 5000),
        changes: action.includes('MODIFIED') ? {
          before: { status: 'active', tier: 'standard' },
          after: { status: 'inactive', tier: 'premium' }
        } : null,
        riskScore: Math.floor(Math.random() * 100),
        complianceImpact: ['HIGH', 'MEDIUM', 'LOW', 'NONE'][Math.floor(Math.random() * 4)]
      }
    })
  }
  
  return events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
}

interface AuditEvent {
  id: string
  timestamp: Date
  actor: string
  action: string
  target: string
  result: 'SUCCESS' | 'FAILURE'
  ipAddress: string
  location: string
  sessionId: string
  details: {
    userAgent: string
    method: string
    duration: number
    changes: any
    riskScore: number
    complianceImpact: string
  }
}

export default function AuditTrailPage() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [filteredEvents, setFilteredEvents] = useState<AuditEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null)
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set())
  const [filters, setFilters] = useState({
    dateRange: '7d',
    actor: '',
    action: '',
    result: '',
    riskLevel: ''
  })
  const [viewMode, setViewMode] = useState<'timeline' | 'table' | 'graph'>('timeline')
  const [realTimeEnabled, setRealTimeEnabled] = useState(false)
  const [stats, setStats] = useState({
    totalEvents: 0,
    failureRate: 0,
    topActors: [] as any[],
    topActions: [] as any[],
    riskTrend: [] as any[]
  })

  // Load initial data
  useEffect(() => {
    const mockData = generateMockAuditEvents()
    setEvents(mockData)
    setFilteredEvents(mockData)
    calculateStats(mockData)
    setLoading(false)
  }, [])

  // Apply filters
  useEffect(() => {
    let filtered = [...events]
    
    // Date range filter
    if (filters.dateRange) {
      const now = new Date()
      let startDate = new Date()
      
      switch (filters.dateRange) {
        case '24h':
          startDate = subDays(now, 1)
          break
        case '7d':
          startDate = subDays(now, 7)
          break
        case '30d':
          startDate = subDays(now, 30)
          break
      }
      
      filtered = filtered.filter(e => e.timestamp >= startDate)
    }
    
    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(e =>
        e.actor.toLowerCase().includes(term) ||
        e.action.toLowerCase().includes(term) ||
        e.target.toLowerCase().includes(term) ||
        e.ipAddress.includes(term) ||
        e.location.toLowerCase().includes(term)
      )
    }
    
    // Other filters
    if (filters.actor) {
      filtered = filtered.filter(e => e.actor === filters.actor)
    }
    if (filters.action) {
      filtered = filtered.filter(e => e.action === filters.action)
    }
    if (filters.result) {
      filtered = filtered.filter(e => e.result === filters.result)
    }
    if (filters.riskLevel) {
      filtered = filtered.filter(e => {
        const risk = e.details.riskScore
        switch (filters.riskLevel) {
          case 'HIGH': return risk >= 70
          case 'MEDIUM': return risk >= 40 && risk < 70
          case 'LOW': return risk < 40
          default: return true
        }
      })
    }
    
    setFilteredEvents(filtered)
    calculateStats(filtered)
  }, [events, searchTerm, filters])

  // Simulate real-time events
  useEffect(() => {
    if (!realTimeEnabled) return
    
    const interval = setInterval(() => {
      const newEvent: AuditEvent = {
        id: `evt-${Date.now()}`,
        timestamp: new Date(),
        actor: 'realtime@monitor.com',
        action: 'RESOURCE_ACCESSED',
        target: '/live/monitoring',
        result: 'SUCCESS',
        ipAddress: '10.0.0.1',
        location: 'Seattle, WA',
        sessionId: 'live-session',
        details: {
          userAgent: 'Real-time Monitor',
          method: 'GET',
          duration: 100,
          changes: null,
          riskScore: Math.floor(Math.random() * 30),
          complianceImpact: 'LOW'
        }
      }
      
      setEvents(prev => [newEvent, ...prev])
    }, 5000)
    
    return () => clearInterval(interval)
  }, [realTimeEnabled])

  const calculateStats = (data: AuditEvent[]) => {
    const failures = data.filter(e => e.result === 'FAILURE').length
    const failureRate = data.length > 0 ? (failures / data.length) * 100 : 0
    
    // Top actors
    const actorCounts = data.reduce((acc, e) => {
      acc[e.actor] = (acc[e.actor] || 0) + 1
      return acc
    }, {} as Record<string, number>)
    const topActors = Object.entries(actorCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([actor, count]) => ({ actor, count }))
    
    // Top actions
    const actionCounts = data.reduce((acc, e) => {
      acc[e.action] = (acc[e.action] || 0) + 1
      return acc
    }, {} as Record<string, number>)
    const topActions = Object.entries(actionCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([action, count]) => ({ action, count }))
    
    // Risk trend (last 7 days)
    const riskTrend = []
    for (let i = 6; i >= 0; i--) {
      const date = subDays(new Date(), i)
      const dayEvents = data.filter(e => 
        e.timestamp >= startOfDay(date) && 
        e.timestamp <= endOfDay(date)
      )
      const avgRisk = dayEvents.length > 0
        ? dayEvents.reduce((sum, e) => sum + e.details.riskScore, 0) / dayEvents.length
        : 0
      riskTrend.push({
        date: format(date, 'MMM d'),
        risk: Math.round(avgRisk),
        events: dayEvents.length
      })
    }
    
    setStats({
      totalEvents: data.length,
      failureRate,
      topActors,
      topActions,
      riskTrend
    })
  }

  const toggleEventExpansion = (eventId: string) => {
    setExpandedEvents(prev => {
      const next = new Set(prev)
      if (next.has(eventId)) {
        next.delete(eventId)
      } else {
        next.add(eventId)
      }
      return next
    })
  }

  const getActionIcon = (action: string) => {
    if (action.includes('LOGIN')) return <User className="w-4 h-4" />
    if (action.includes('PERMISSION')) return <Shield className="w-4 h-4" />
    if (action.includes('RESOURCE')) return <Database className="w-4 h-4" />
    if (action.includes('POLICY')) return <FileText className="w-4 h-4" />
    if (action.includes('PASSWORD') || action.includes('MFA')) return <Key className="w-4 h-4" />
    if (action.includes('ROLE') || action.includes('GROUP')) return <UserPlus className="w-4 h-4" />
    if (action.includes('API')) return <Lock className="w-4 h-4" />
    return <Activity className="w-4 h-4" />
  }

  const getActionColor = (action: string, result: string) => {
    if (result === 'FAILURE') return 'text-red-600 dark:text-red-400'
    if (action.includes('DELETE') || action.includes('REVOKED')) return 'text-orange-600 dark:text-orange-400'
    if (action.includes('CREATED') || action.includes('GRANTED')) return 'text-green-600 dark:text-green-400'
    if (action.includes('MODIFIED')) return 'text-blue-600 dark:text-blue-400'
    return 'text-gray-600 dark:text-gray-400'
  }

  const exportData = () => {
    const csv = [
      ['Timestamp', 'Actor', 'Action', 'Target', 'Result', 'IP Address', 'Location', 'Risk Score'],
      ...filteredEvents.map(e => [
        e.timestamp.toISOString(),
        e.actor,
        e.action,
        e.target,
        e.result,
        e.ipAddress,
        e.location,
        e.details.riskScore.toString()
      ])
    ].map(row => row.join(',')).join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit-trail-${format(new Date(), 'yyyy-MM-dd')}.csv`
    a.click()
  }

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6 sm:mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground dark:text-white">
              Audit Trail
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground dark:text-gray-400 mt-1">
              Complete activity history with deep-drill analysis
            </p>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <button type="button"
              onClick={() => setRealTimeEnabled(!realTimeEnabled)}
              className={`
                px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg flex items-center gap-2 transition-all text-sm sm:text-base
                ${realTimeEnabled 
                  ? 'bg-green-600 text-white' 
                  : 'bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-700'
                }
              `}
            >
              <Activity className={`w-4 h-4 ${realTimeEnabled ? 'animate-pulse' : ''}`} />
              <span className="hidden sm:inline">Real-time</span>
            </button>
            
            <button type="button"
              onClick={exportData}
              className="px-3 sm:px-4 py-1.5 sm:py-2 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700 transition-colors flex items-center gap-2 text-sm sm:text-base"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Export</span>
            </button>
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3 sm:gap-4 mb-6">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-3 sm:p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs sm:text-sm text-muted-foreground dark:text-gray-400">Total Events</p>
              <p className="text-xl sm:text-2xl font-bold text-foreground dark:text-white">
                {stats.totalEvents.toLocaleString()}
              </p>
            </div>
            <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-3 sm:p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs sm:text-sm text-muted-foreground dark:text-gray-400">Failure Rate</p>
              <p className="text-xl sm:text-2xl font-bold text-red-600 dark:text-red-400">
                {stats.failureRate.toFixed(1)}%
              </p>
            </div>
            <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-3 sm:p-4 col-span-2 lg:col-span-1">
          <p className="text-xs sm:text-sm text-muted-foreground dark:text-gray-400 mb-2">Top Actor</p>
          <p className="text-sm sm:text-base font-medium text-foreground dark:text-white truncate">
            {stats.topActors[0]?.actor || 'N/A'}
          </p>
          <p className="text-xs text-muted-foreground dark:text-gray-400">
            {stats.topActors[0]?.count || 0} events
          </p>
        </div>
        
        <div className="bg-card dark:bg-gray-800 rounded-lg p-3 sm:p-4 col-span-2 lg:col-span-1 xl:col-span-2">
          <p className="text-xs sm:text-sm text-muted-foreground dark:text-gray-400 mb-2">Risk Trend</p>
          <div className="flex items-end gap-1 h-12">
            {stats.riskTrend.map((day, i) => (
              <div
                key={i}
                className="flex-1 bg-blue-600 dark:bg-blue-400 rounded-t"
                style={{ height: `${(day.risk / 100) * 100}%` }}
                title={`${day.date}: ${day.risk}% risk`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-card dark:bg-gray-800 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 sm:gap-4">
          <div className="xl:col-span-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground dark:text-gray-400" />
              <input
                type="text"
                placeholder="Search events..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-blue-500 text-sm"
              />
            </div>
          </div>
          
          <select
            value={filters.dateRange}
            onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
            className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-blue-500 text-sm"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="">All Time</option>
          </select>
          
          <select
            value={filters.result}
            onChange={(e) => setFilters(prev => ({ ...prev, result: e.target.value }))}
            className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-blue-500 text-sm"
          >
            <option value="">All Results</option>
            <option value="SUCCESS">Success</option>
            <option value="FAILURE">Failure</option>
          </select>
          
          <select
            value={filters.riskLevel}
            onChange={(e) => setFilters(prev => ({ ...prev, riskLevel: e.target.value }))}
            className="px-3 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-blue-500 text-sm"
          >
            <option value="">All Risk Levels</option>
            <option value="HIGH">High Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>
          
          <div className="flex gap-2">
            <button type="button"
              onClick={() => setViewMode('timeline')}
              className={`
                flex-1 px-3 py-2 rounded-lg transition-colors text-sm
                ${viewMode === 'timeline' 
                  ? 'bg-primary text-primary-foreground dark:bg-blue-600 dark:text-white' 
                  : 'bg-muted dark:bg-gray-700 text-muted-foreground dark:text-gray-300'
                }
              `}
            >
              Timeline
            </button>
            <button type="button"
              onClick={() => setViewMode('table')}
              className={`
                flex-1 px-3 py-2 rounded-lg transition-colors text-sm
                ${viewMode === 'table' 
                  ? 'bg-primary text-primary-foreground dark:bg-blue-600 dark:text-white' 
                  : 'bg-muted dark:bg-gray-700 text-muted-foreground dark:text-gray-300'
                }
              `}
            >
              Table
            </button>
          </div>
        </div>
      </div>

      {/* Events List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-primary dark:text-blue-400" />
          </div>
        ) : filteredEvents.length === 0 ? (
          <div className="text-center py-12">
            <Info className="w-12 h-12 mx-auto text-muted-foreground dark:text-gray-400 mb-4" />
            <p className="text-muted-foreground dark:text-gray-400">No events match your filters</p>
          </div>
        ) : viewMode === 'timeline' ? (
          // Timeline View
          <div className="relative">
            <div className="absolute left-4 sm:left-8 top-0 bottom-0 w-0.5 bg-border dark:bg-gray-700" />
            
            {filteredEvents.slice(0, 50).map((event, index) => {
              const isExpanded = expandedEvents.has(event.id)
              
              return (
                <div key={event.id} className="relative flex gap-4 sm:gap-8">
                  {/* Timeline dot */}
                  <div className="relative z-10">
                    <div className={`
                      w-8 h-8 rounded-full border-4 border-background dark:border-gray-900 flex items-center justify-center
                      ${event.result === 'SUCCESS' 
                        ? 'bg-green-600 dark:bg-green-500' 
                        : 'bg-red-600 dark:bg-red-500'
                      }
                    `}>
                      {event.result === 'SUCCESS' ? (
                        <CheckCircle className="w-4 h-4 text-white" />
                      ) : (
                        <XCircle className="w-4 h-4 text-white" />
                      )}
                    </div>
                  </div>
                  
                  {/* Event card */}
                  <div className="flex-1 mb-4">
                    <div
                      className="bg-card dark:bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-accent dark:hover:bg-gray-700 transition-colors"
                      onClick={() => toggleEventExpansion(event.id)}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <span className={getActionColor(event.action, event.result)}>
                              {getActionIcon(event.action)}
                            </span>
                            <span className="font-medium text-sm sm:text-base text-foreground dark:text-white">
                              {event.action.replace(/_/g, ' ')}
                            </span>
                            <span className="text-xs text-muted-foreground dark:text-gray-400">
                              {formatDistanceToNow(event.timestamp, { addSuffix: true })}
                            </span>
                          </div>
                          
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs sm:text-sm">
                            <div className="flex items-center gap-2">
                              <User className="w-3 h-3 text-muted-foreground dark:text-gray-400" />
                              <Link
                                href={`/security/rbac?user=${encodeURIComponent(event.actor)}`}
                                className="text-blue-600 dark:text-blue-400 hover:underline truncate"
                                onClick={(e) => e.stopPropagation()}
                              >
                                {event.actor}
                              </Link>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <Database className="w-3 h-3 text-muted-foreground dark:text-gray-400" />
                              <span className="text-muted-foreground dark:text-gray-300 truncate">
                                {event.target}
                              </span>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <MapPin className="w-3 h-3 text-muted-foreground dark:text-gray-400" />
                              <span className="text-muted-foreground dark:text-gray-300">
                                {event.location}
                              </span>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <Globe className="w-3 h-3 text-muted-foreground dark:text-gray-400" />
                              <span className="text-muted-foreground dark:text-gray-300">
                                {event.ipAddress}
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <button type="button" className="text-muted-foreground dark:text-gray-400">
                          {isExpanded ? (
                            <ChevronDown className="w-5 h-5" />
                          ) : (
                            <ChevronRight className="w-5 h-5" />
                          )}
                        </button>
                      </div>
                      
                      {/* Expanded details */}
                      {isExpanded && (
                        <div className="mt-4 pt-4 border-t border-border dark:border-gray-700">
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">Session ID</p>
                              <p className="font-mono text-xs">{event.sessionId}</p>
                            </div>
                            
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">Method</p>
                              <p>{event.details.method}</p>
                            </div>
                            
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">Duration</p>
                              <p>{event.details.duration}ms</p>
                            </div>
                            
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">Risk Score</p>
                              <div className="flex items-center gap-2">
                                <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full ${
                                      event.details.riskScore >= 70 ? 'bg-red-500' :
                                      event.details.riskScore >= 40 ? 'bg-yellow-500' :
                                      'bg-green-500'
                                    }`}
                                    style={{ width: `${event.details.riskScore}%` }}
                                  />
                                </div>
                                <span>{event.details.riskScore}%</span>
                              </div>
                            </div>
                            
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">Compliance Impact</p>
                              <span className={`
                                px-2 py-0.5 rounded text-xs font-medium
                                ${event.details.complianceImpact === 'HIGH' ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200' :
                                  event.details.complianceImpact === 'MEDIUM' ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200' :
                                  event.details.complianceImpact === 'LOW' ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' :
                                  'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
                                }
                              `}>
                                {event.details.complianceImpact}
                              </span>
                            </div>
                            
                            <div>
                              <p className="text-muted-foreground dark:text-gray-400 mb-1">User Agent</p>
                              <p className="text-xs font-mono truncate">{event.details.userAgent}</p>
                            </div>
                          </div>
                          
                          {event.details.changes && (
                            <div className="mt-4">
                              <p className="text-muted-foreground dark:text-gray-400 mb-2">Changes Made</p>
                              <div className="bg-muted dark:bg-gray-900 rounded p-3 font-mono text-xs">
                                <pre>{JSON.stringify(event.details.changes, null, 2)}</pre>
                              </div>
                            </div>
                          )}
                          
                          {/* Drill-down actions */}
                          <div className="mt-4 flex flex-wrap gap-2">
                            <Link
                              href={`/security/rbac?user=${encodeURIComponent(event.actor)}`}
                              className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
                              onClick={(e) => e.stopPropagation()}
                            >
                              View User Profile
                            </Link>
                            
                            <Link
                              href={`/audit?session=${event.sessionId}`}
                              className="px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded text-xs hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                              onClick={(e) => e.stopPropagation()}
                            >
                              View Session Events
                            </Link>
                            
                            <Link
                              href={`/operations/resources?filter=${encodeURIComponent(event.target)}`}
                              className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded text-xs hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
                              onClick={(e) => e.stopPropagation()}
                            >
                              View Resource
                            </Link>
                            
                            <button type="button"
                              className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded text-xs hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                              onClick={(e) => {
                                e.stopPropagation()
                                navigator.clipboard.writeText(JSON.stringify(event, null, 2))
                              }}
                            >
                              <Copy className="w-3 h-3 inline mr-1" />
                              Copy JSON
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          // Table View
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border dark:border-gray-700">
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Timestamp
                  </th>
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Actor
                  </th>
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Action
                  </th>
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Target
                  </th>
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Result
                  </th>
                  <th className="text-left p-2 text-xs sm:text-sm font-medium text-muted-foreground dark:text-gray-400">
                    Risk
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.slice(0, 50).map((event) => (
                  <tr
                    key={event.id}
                    className="border-b border-border dark:border-gray-700 hover:bg-accent dark:hover:bg-gray-800 cursor-pointer"
                    onClick={() => setSelectedEvent(event)}
                  >
                    <td className="p-2 text-xs sm:text-sm">
                      {format(event.timestamp, 'MMM d, HH:mm')}
                    </td>
                    <td className="p-2 text-xs sm:text-sm truncate max-w-[150px]">
                      {event.actor}
                    </td>
                    <td className="p-2 text-xs sm:text-sm">
                      <div className="flex items-center gap-1">
                        <span className={getActionColor(event.action, event.result)}>
                          {getActionIcon(event.action)}
                        </span>
                        <span>{event.action.replace(/_/g, ' ')}</span>
                      </div>
                    </td>
                    <td className="p-2 text-xs sm:text-sm truncate max-w-[200px]">
                      {event.target}
                    </td>
                    <td className="p-2 text-xs sm:text-sm">
                      <span className={`
                        px-2 py-0.5 rounded text-xs font-medium
                        ${event.result === 'SUCCESS' 
                          ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' 
                          : 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                        }
                      `}>
                        {event.result}
                      </span>
                    </td>
                    <td className="p-2 text-xs sm:text-sm">
                      <span className={`
                        ${event.details.riskScore >= 70 ? 'text-red-600 dark:text-red-400' :
                          event.details.riskScore >= 40 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-green-600 dark:text-green-400'
                        }
                      `}>
                        {event.details.riskScore}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {filteredEvents.length > 50 && (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground dark:text-gray-400">
              Showing 50 of {filteredEvents.length} events
            </p>
          </div>
        )}
      </div>
      
      {/* Event Detail Modal */}
      {selectedEvent && viewMode === 'table' && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedEvent(null)}
        >
          <div
            className="bg-card dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-foreground dark:text-white">Event Details</h2>
              <button type="button"
                onClick={() => setSelectedEvent(null)}
                className="text-muted-foreground dark:text-gray-400 hover:text-foreground dark:hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm text-muted-foreground dark:text-gray-400">Event ID</p>
                <p className="font-mono text-sm">{selectedEvent.id}</p>
              </div>
              
              <div>
                <p className="text-sm text-muted-foreground dark:text-gray-400">Timestamp</p>
                <p>{format(selectedEvent.timestamp, 'PPpp')}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground dark:text-gray-400">Actor</p>
                  <p>{selectedEvent.actor}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground dark:text-gray-400">Action</p>
                  <p>{selectedEvent.action}</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-muted-foreground dark:text-gray-400">Target Resource</p>
                <p className="font-mono text-sm">{selectedEvent.target}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground dark:text-gray-400">IP Address</p>
                  <p>{selectedEvent.ipAddress}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground dark:text-gray-400">Location</p>
                  <p>{selectedEvent.location}</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-muted-foreground dark:text-gray-400">Full Details</p>
                <pre className="bg-muted dark:bg-gray-900 rounded p-3 text-xs overflow-x-auto">
                  {JSON.stringify(selectedEvent, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}