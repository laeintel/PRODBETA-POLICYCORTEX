/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, CheckCircle, Clock, Download, Play, RefreshCw } from 'lucide-react';
import { useSSE } from '@/hooks/useSSE';
import { api } from '@/lib/api-client';

interface ActionDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  actionId?: string | null;
}

export function ActionDrawer({ isOpen, onClose, actionId }: ActionDrawerProps) {
  const [action, setAction] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('summary');
  const [isExecuting, setIsExecuting] = useState(false);
  const isSimMode = process.env.NEXT_PUBLIC_USE_REAL_DATA !== 'true'
  const lastFocusedRef = useRef<HTMLElement | null>(null)
  const headingRef = useRef<HTMLHeadingElement | null>(null)
  
  const { messages: events, isConnected } = useSSE(
    actionId ? `/api/v1/actions/${actionId}/events` : ''
  );

  useEffect(() => {
    if (actionId) {
      fetchActionDetails(actionId);
    }
  }, [actionId]);

  // Focus management and body scroll lock to ensure the drawer stays in focus
  useEffect(() => {
    if (isOpen) {
      lastFocusedRef.current = document.activeElement as HTMLElement
      document.body.style.overflow = 'hidden'
      setTimeout(() => headingRef.current?.focus(), 0)
    } else {
      document.body.style.overflow = ''
      lastFocusedRef.current?.focus?.()
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  const fetchActionDetails = async (id: string) => {
    try {
      const resp = await api.request<any>(`/api/v1/actions/${id}`)
      if (!resp.error) setAction(resp.data)
    } catch (error) {
      console.error('Failed to fetch action details:', error);
    }
  };

  const executeAction = async (dryRun: boolean = true) => {
    setIsExecuting(true);
    try {
      const resp = await api.createAction(action?.resource_id || 'global', action?.action_type || 'custom', { ...action, dry_run: dryRun })
      if (!resp.error) {
        setAction({ ...(action||{}), id: resp.data?.action_id || resp.data?.id })
      }
    } catch (error) {
      console.error('Failed to execute action:', error);
    } finally {
      setIsExecuting(false);
    }
  };

  const tabs = [
    { id: 'summary', label: 'Summary' },
    { id: 'preflight', label: 'Preflight' },
    { id: 'blast-radius', label: 'Blast Radius' },
    { id: 'approvals', label: 'Approvals' },
    { id: 'progress', label: 'Live Progress' },
    { id: 'evidence', label: 'Evidence' },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-[98] backdrop-blur-sm"
          />

          {/* Drawer (right-side, always on top) */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white dark:bg-gray-900 shadow-2xl z-[100] overflow-hidden"
            role="dialog"
            aria-modal="true"
            aria-labelledby="action-drawer-heading"
            tabIndex={-1}
          >
            {/* Header */}
            <div className="border-b border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 id="action-drawer-heading" ref={headingRef} className="text-2xl font-bold text-gray-900 dark:text-white" tabIndex={0}>
                    Action Execution
                  </h2>
                  {action && (
                    <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                      {action.type} - {action.resource_type}
                    </p>
                  )}
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
                  aria-label="Close action drawer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Status bar */}
              {isConnected && (
                <div className="mt-4 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Live updates active
                  </span>
                </div>
              )}
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200 dark:border-gray-800">
              <div className="flex overflow-x-auto">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-6 py-3 text-sm font-medium whitespace-nowrap ${
                      activeTab === tab.id
                        ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                    }`}
                    aria-selected={activeTab === tab.id}
                    role="tab"
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6" role="tabpanel">
              {activeTab === 'summary' && <SummaryTab action={action} />}
              {activeTab === 'preflight' && <PreflightTab action={action} />}
              {activeTab === 'blast-radius' && <BlastRadiusTab action={action} />}
              {activeTab === 'approvals' && <ApprovalsTab action={action} />}
              {activeTab === 'progress' && <ProgressTab events={events} />}
              {activeTab === 'evidence' && <EvidenceTab action={action} />}
            </div>

            {/* Actions */}
            <div className="border-t border-gray-200 dark:border-gray-800 p-6">
              <div className="flex gap-3">
                <button
                  onClick={() => executeAction(true)}
                  disabled={isExecuting}
                  className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
                >
                  <Play className="w-4 h-4 inline mr-2" />
                  Dry Run
                </button>
                <button
                  onClick={() => executeAction(false)}
                  disabled={true}
                  title={isSimMode ? 'Disabled in demo mode' : 'Temporarily disabled for demo'}
                  className="flex-1 px-4 py-2 bg-gray-400 text-white rounded-lg disabled:opacity-60 cursor-not-allowed"
                >
                  <CheckCircle className="w-4 h-4 inline mr-2" />
                  Execute (Disabled)
                </button>
              </div>
              {isSimMode && (
                <div className="mt-2 text-xs text-yellow-600 dark:text-yellow-400 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Writes are disabled in simulated mode. Connect Azure to enable remediation.
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function SummaryTab({ action }: { action: any }) {
  if (!action) return <div>Loading...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-3">Action Details</h3>
        <dl className="space-y-2">
          <div className="flex justify-between">
            <dt className="text-gray-600 dark:text-gray-400">Type</dt>
            <dd className="font-medium">{action.type || 'N/A'}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-600 dark:text-gray-400">Resource</dt>
            <dd className="font-medium">{action.resource_name || 'N/A'}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-600 dark:text-gray-400">Estimated Savings</dt>
            <dd className="font-medium text-green-600">${action.estimated_savings || 0}/month</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-600 dark:text-gray-400">Risk Level</dt>
            <dd className="font-medium">{action.risk_level || 'Low'}</dd>
          </div>
        </dl>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3">Description</h3>
        <p className="text-gray-700 dark:text-gray-300">
          {action.description || 'No description available'}
        </p>
      </div>
    </div>
  );
}

function PreflightTab({ action }: { action: any }) {
  return (
    <div className="space-y-4">
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
          <div>
            <h4 className="font-medium text-yellow-900 dark:text-yellow-100">
              Preflight Check
            </h4>
            <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
              Review changes before execution
            </p>
          </div>
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <pre className="text-sm text-gray-700 dark:text-gray-300 overflow-x-auto">
          {JSON.stringify(action?.preflight_diff || {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function BlastRadiusTab({ action }: { action: any }) {
  const affectedResources = action?.blast_radius || [];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Affected Resources</h3>
      {affectedResources.length > 0 ? (
        <div className="space-y-2">
          {affectedResources.map((resource: any, index: number) => (
            <div
              key={index}
              className="border border-gray-200 dark:border-gray-700 rounded-lg p-3"
            >
              <div className="font-medium">{resource.name}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {resource.type} - {resource.impact}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-600 dark:text-gray-400">No resources affected</p>
      )}
    </div>
  );
}

function ApprovalsTab({ action }: { action: any }) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Approval Status</h3>
      <div className="space-y-3">
        <ApprovalItem
          approver="Policy Engine"
          status="approved"
          timestamp="2 minutes ago"
        />
        <ApprovalItem
          approver="Security Scanner"
          status="approved"
          timestamp="1 minute ago"
        />
        <ApprovalItem
          approver="Manager Approval"
          status="pending"
          timestamp=""
        />
      </div>
    </div>
  );
}

function ApprovalItem({ approver, status, timestamp }: any) {
  const getStatusIcon = () => {
    switch (status) {
      case 'approved':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <X className="w-5 h-5 text-red-500" />;
    }
  };

  return (
    <div className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
      <div className="flex items-center gap-3">
        {getStatusIcon()}
        <div>
          <div className="font-medium">{approver}</div>
          {timestamp && (
            <div className="text-sm text-gray-600 dark:text-gray-400">{timestamp}</div>
          )}
        </div>
      </div>
      <span className={`text-sm font-medium ${
        status === 'approved' ? 'text-green-600' : 
        status === 'pending' ? 'text-yellow-600' : 'text-red-600'
      }`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    </div>
  );
}

function ProgressTab({ events }: { events: any[] }) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Live Progress</h3>
      <div className="space-y-2">
        {events.map((event, index) => (
          <div key={index} className="flex items-start gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full mt-2" />
            <div className="flex-1">
              <div className="font-medium">{event.message}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {new Date(event.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceTab({ action }: { action: any }) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Evidence & Artifacts</h3>
      <div className="space-y-3">
        <button className="w-full flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
          <span>Preflight Report</span>
          <Download className="w-4 h-4" />
        </button>
        <button className="w-full flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
          <span>Execution Log</span>
          <Download className="w-4 h-4" />
        </button>
        <button className="w-full flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
          <span>Compliance Report</span>
          <Download className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}