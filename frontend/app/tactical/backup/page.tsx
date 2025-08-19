'use client';

import React, { useState, useEffect } from 'react';
import { 
  Archive, Database, HardDrive, Cloud, Shield, Clock, Calendar,
  CheckCircle, XCircle, AlertTriangle, Play, Pause, Download,
  Upload, RefreshCw, Settings, Trash2, Copy, History, Save,
  Server, Lock, Zap, Activity, TrendingUp, Timer, User, ChevronRight
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface Backup {
  id: string;
  name: string;
  type: 'full' | 'incremental' | 'differential' | 'snapshot';
  source: string;
  destination: string;
  status: 'completed' | 'in_progress' | 'failed' | 'scheduled' | 'cancelled';
  size: string;
  compressed: boolean;
  encrypted: boolean;
  retention: number; // days
  createdAt: string;
  completedAt?: string;
  duration?: number; // seconds
  schedule?: {
    frequency: 'hourly' | 'daily' | 'weekly' | 'monthly';
    time: string;
    nextRun: string;
  };
  verification: {
    status: 'verified' | 'pending' | 'failed';
    lastChecked?: string;
  };
  recovery: {
    rpo: number; // Recovery Point Objective in minutes
    rto: number; // Recovery Time Objective in minutes
    lastTested?: string;
  };
}

interface RecoveryPoint {
  id: string;
  backupId: string;
  name: string;
  timestamp: string;
  size: string;
  type: string;
  restorable: boolean;
  verified: boolean;
  dataIntegrity: number; // percentage
}

export default function BackupRecovery() {
  const [backups, setBackups] = useState<Backup[]>([]);
  const [recoveryPoints, setRecoveryPoints] = useState<RecoveryPoint[]>([]);
  const [selectedBackup, setSelectedBackup] = useState<Backup | null>(null);
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'backups' | 'recovery'>('backups');

  useEffect(() => {
    // Initialize with mock data
    setBackups([
      {
        id: 'BKP-001',
        name: 'Production Database Daily',
        type: 'full',
        source: 'PostgreSQL Production',
        destination: 'Azure Blob Storage',
        status: 'completed',
        size: '45.2 GB',
        compressed: true,
        encrypted: true,
        retention: 30,
        createdAt: 'Today 2:00 AM',
        completedAt: 'Today 3:30 AM',
        duration: 5400,
        schedule: {
          frequency: 'daily',
          time: '2:00 AM',
          nextRun: 'Tomorrow 2:00 AM'
        },
        verification: {
          status: 'verified',
          lastChecked: 'Today 4:00 AM'
        },
        recovery: {
          rpo: 1440, // 24 hours
          rto: 60, // 1 hour
          lastTested: '3 days ago'
        }
      },
      {
        id: 'BKP-002',
        name: 'Application Files Incremental',
        type: 'incremental',
        source: 'App Servers',
        destination: 'S3 Bucket',
        status: 'in_progress',
        size: '2.3 GB',
        compressed: true,
        encrypted: true,
        retention: 7,
        createdAt: '30 minutes ago',
        schedule: {
          frequency: 'hourly',
          time: 'Every hour',
          nextRun: 'In 30 minutes'
        },
        verification: {
          status: 'pending'
        },
        recovery: {
          rpo: 60,
          rto: 15,
          lastTested: 'Yesterday'
        }
      },
      {
        id: 'BKP-003',
        name: 'System Snapshot',
        type: 'snapshot',
        source: 'VM-PROD-01',
        destination: 'Azure Recovery Vault',
        status: 'scheduled',
        size: '120 GB',
        compressed: false,
        encrypted: true,
        retention: 14,
        createdAt: 'Scheduled',
        schedule: {
          frequency: 'weekly',
          time: 'Sunday 1:00 AM',
          nextRun: 'In 2 days'
        },
        verification: {
          status: 'verified',
          lastChecked: 'Last week'
        },
        recovery: {
          rpo: 10080, // 1 week
          rto: 30,
          lastTested: '2 weeks ago'
        }
      },
      {
        id: 'BKP-004',
        name: 'Configuration Backup',
        type: 'differential',
        source: 'Config Server',
        destination: 'Local NAS',
        status: 'failed',
        size: '450 MB',
        compressed: true,
        encrypted: false,
        retention: 90,
        createdAt: 'Yesterday 11:00 PM',
        schedule: {
          frequency: 'daily',
          time: '11:00 PM',
          nextRun: 'Today 11:00 PM'
        },
        verification: {
          status: 'failed'
        },
        recovery: {
          rpo: 1440,
          rto: 120,
          lastTested: 'Never'
        }
      },
      {
        id: 'BKP-005',
        name: 'User Data Archive',
        type: 'full',
        source: 'File Server',
        destination: 'Glacier Storage',
        status: 'completed',
        size: '890 GB',
        compressed: true,
        encrypted: true,
        retention: 365,
        createdAt: 'Last Month',
        completedAt: 'Last Month',
        duration: 28800,
        schedule: {
          frequency: 'monthly',
          time: '1st Sunday',
          nextRun: 'Next month'
        },
        verification: {
          status: 'verified',
          lastChecked: 'Last Month'
        },
        recovery: {
          rpo: 43200, // 30 days
          rto: 1440, // 24 hours
          lastTested: '2 months ago'
        }
      }
    ]);

    setRecoveryPoints([
      {
        id: 'RP-001',
        backupId: 'BKP-001',
        name: 'Database - Full Backup',
        timestamp: 'Today 3:30 AM',
        size: '45.2 GB',
        type: 'Full',
        restorable: true,
        verified: true,
        dataIntegrity: 100
      },
      {
        id: 'RP-002',
        backupId: 'BKP-001',
        name: 'Database - Previous Day',
        timestamp: 'Yesterday 2:00 AM',
        size: '44.8 GB',
        type: 'Full',
        restorable: true,
        verified: true,
        dataIntegrity: 100
      },
      {
        id: 'RP-003',
        backupId: 'BKP-002',
        name: 'App Files - Latest',
        timestamp: '30 minutes ago',
        size: '2.3 GB',
        type: 'Incremental',
        restorable: false,
        verified: false,
        dataIntegrity: 95
      },
      {
        id: 'RP-004',
        backupId: 'BKP-003',
        name: 'System Snapshot - Week 1',
        timestamp: 'Last Sunday',
        size: '118 GB',
        type: 'Snapshot',
        restorable: true,
        verified: true,
        dataIntegrity: 100
      }
    ]);
  }, []);

  const filteredBackups = backups.filter(backup => {
    if (filterType !== 'all' && backup.type !== filterType) return false;
    if (filterStatus !== 'all' && backup.status !== filterStatus) return false;
    return true;
  });

  const stats = {
    totalBackups: backups.length,
    successful: backups.filter(b => b.status === 'completed').length,
    failed: backups.filter(b => b.status === 'failed').length,
    totalSize: '1.2 TB',
    avgRPO: Math.round(backups.reduce((sum, b) => sum + b.recovery.rpo, 0) / backups.length / 60),
    avgRTO: Math.round(backups.reduce((sum, b) => sum + b.recovery.rto, 0) / backups.length)
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'completed': return 'text-green-500 bg-green-900/20';
      case 'in_progress': return 'text-blue-500 bg-blue-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'scheduled': return 'text-yellow-500 bg-yellow-900/20';
      case 'cancelled': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'full': return <Database className="w-4 h-4 text-blue-500" />;
      case 'incremental': return <Activity className="w-4 h-4 text-green-500" />;
      case 'differential': return <Copy className="w-4 h-4 text-yellow-500" />;
      case 'snapshot': return <Save className="w-4 h-4 text-purple-500" />;
      default: return <Archive className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Backup & Recovery Center</h1>
            <p className="text-sm text-gray-400 mt-1">Manage backups and recovery points</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setViewMode(viewMode === 'backups' ? 'recovery' : 'backups')}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2"
            >
              {viewMode === 'backups' ? <History className="w-4 h-4" /> : <Archive className="w-4 h-4" />}
              <span>{viewMode === 'backups' ? 'Recovery Points' : 'Backups'}</span>
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Upload className="w-4 h-4" />
              <span>New Backup</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <Archive className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Backups</p>
              <p className="text-xl font-bold">{stats.totalBackups}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Successful</p>
              <p className="text-xl font-bold">{stats.successful}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <XCircle className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-xs text-gray-400">Failed</p>
              <p className="text-xl font-bold text-red-500">{stats.failed}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <HardDrive className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-xs text-gray-400">Total Size</p>
              <p className="text-xl font-bold">{stats.totalSize}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-cyan-500" />
            <div>
              <p className="text-xs text-gray-400">Avg RPO</p>
              <p className="text-xl font-bold">{stats.avgRPO}h</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Timer className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Avg RTO</p>
              <p className="text-xl font-bold">{stats.avgRTO}m</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Filters */}
        <div className="flex items-center space-x-3 mb-6">
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Types</option>
            <option value="full">Full Backup</option>
            <option value="incremental">Incremental</option>
            <option value="differential">Differential</option>
            <option value="snapshot">Snapshot</option>
          </select>
          
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="in_progress">In Progress</option>
            <option value="failed">Failed</option>
            <option value="scheduled">Scheduled</option>
          </select>
        </div>

        {viewMode === 'backups' ? (
          /* Backups List */
          <div className="space-y-4">
            {filteredBackups.map(backup => (
              <div key={backup.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      {getTypeIcon(backup.type)}
                      <span className="text-xs text-gray-500 font-mono">{backup.id}</span>
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(backup.status)}`}>
                        {backup.status.toUpperCase()}
                      </span>
                      {backup.encrypted && (
                        <Lock className="w-3 h-3 text-green-500" />
                      )}
                      {backup.compressed && (
                        <Archive className="w-3 h-3 text-blue-500" />
                      )}
                    </div>
                    <h3 className="text-sm font-bold mb-1">{backup.name}</h3>
                    <div className="flex items-center space-x-6 text-xs text-gray-500">
                      <span>Source: {backup.source}</span>
                      <span>Destination: {backup.destination}</span>
                      <span>Size: {backup.size}</span>
                      <span>Retention: {backup.retention} days</span>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-gray-500" />
                </div>
                
                <div className="grid grid-cols-4 gap-4 pt-3 border-t border-gray-800">
                  <div>
                    <p className="text-xs text-gray-400">Schedule</p>
                    <p className="text-sm">{backup.schedule?.frequency} at {backup.schedule?.time}</p>
                    <p className="text-xs text-gray-500">Next: {backup.schedule?.nextRun}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Verification</p>
                    <p className={`text-sm ${
                      backup.verification.status === 'verified' ? 'text-green-500' :
                      backup.verification.status === 'failed' ? 'text-red-500' :
                      'text-yellow-500'
                    }`}>
                      {backup.verification.status.toUpperCase()}
                    </p>
                    {backup.verification.lastChecked && (
                      <p className="text-xs text-gray-500">{backup.verification.lastChecked}</p>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Recovery Objectives</p>
                    <p className="text-sm">RPO: {backup.recovery.rpo / 60}h | RTO: {backup.recovery.rto}m</p>
                    <p className="text-xs text-gray-500">Tested: {backup.recovery.lastTested || 'Never'}</p>
                  </div>
                  <div className="flex items-center justify-end space-x-2">
                    {backup.status === 'completed' && (
                      <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs">
                        Restore
                      </button>
                    )}
                    {backup.status === 'in_progress' && (
                      <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs">
                        Cancel
                      </button>
                    )}
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Recovery Points */
          <div className="grid grid-cols-2 gap-4">
            {recoveryPoints.map(point => (
              <div key={point.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <History className="w-4 h-4 text-blue-500" />
                    <span className="text-sm font-bold">{point.name}</span>
                  </div>
                  {point.verified && <Shield className="w-4 h-4 text-green-500" />}
                </div>
                
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Timestamp</span>
                    <span>{point.timestamp}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Size</span>
                    <span>{point.size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Type</span>
                    <span>{point.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Data Integrity</span>
                    <span className={point.dataIntegrity === 100 ? 'text-green-500' : 'text-yellow-500'}>
                      {point.dataIntegrity}%
                    </span>
                  </div>
                </div>
                
                <button
                  disabled={!point.restorable}
                  className={`mt-3 w-full px-3 py-1 rounded text-xs ${
                    point.restorable
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {point.restorable ? 'Restore' : 'Not Available'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}