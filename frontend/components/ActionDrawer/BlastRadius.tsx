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

import React, { useState, useEffect } from 'react';
import { AlertCircle, Server, Database, Network, Users, Shield, Cloud } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BlastRadiusProps {
  actionId?: string;
}

interface AffectedResource {
  id: string;
  name: string;
  type: string;
  impact: 'direct' | 'indirect' | 'downstream';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  mitigations?: string[];
}

export function BlastRadius({ actionId }: BlastRadiusProps) {
  const [resources, setResources] = useState<AffectedResource[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'direct' | 'indirect' | 'downstream'>('all');

  useEffect(() => {
    if (actionId) {
      fetchBlastRadius();
    }
  }, [actionId]);

  const fetchBlastRadius = async () => {
    try {
      const response = await fetch(`/api/v1/actions/${actionId}/blast-radius`);
      const data = await response.json();
      setResources(data.resources || []);
    } catch (error) {
      console.error('Failed to fetch blast radius:', error);
    } finally {
      setLoading(false);
    }
  };

  const getResourceIcon = (type: string) => {
    const t = (type || '').toLowerCase()
    switch (t) {
      case 'vm':
      case 'compute':
        return <Server className="w-4 h-4" />;
      case 'database':
      case 'storage':
        return <Database className="w-4 h-4" />;
      case 'network':
        return <Network className="w-4 h-4" />;
      case 'identity':
      case 'rbac':
        return <Users className="w-4 h-4" />;
      case 'security':
        return <Shield className="w-4 h-4" />;
      default:
        return <Cloud className="w-4 h-4" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'direct':
        return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
      case 'indirect':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
      case 'downstream':
        return 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20';
    }
  };

  const getSeverityBadge = (severity: string) => {
    const colors = {
      low: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
      medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
      high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400',
      critical: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
    };
    return colors[severity as keyof typeof colors] || colors.low;
  };

  const filteredResources = resources.filter(r => 
    filter === 'all' || r.impact === filter
  );

  const stats = {
    total: resources.length,
    direct: resources.filter(r => r.impact === 'direct').length,
    indirect: resources.filter(r => r.impact === 'indirect').length,
    downstream: resources.filter(r => r.impact === 'downstream').length,
  };

  if (loading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded" />
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-24 bg-gray-200 dark:bg-gray-700 rounded" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Impact Summary */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-3 mb-3">
          <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <h3 className="font-medium">Impact Analysis</h3>
        </div>
        <div className="grid grid-cols-4 gap-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.total}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Total Resources
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {stats.direct}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Direct Impact
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
              {stats.indirect}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Indirect Impact
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {stats.downstream}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Downstream
            </div>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2">
        {['all', 'direct', 'indirect', 'downstream'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f as any)}
            className={cn(
              'px-3 py-1 text-sm rounded-lg transition-colors',
              filter === f
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'
            )}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f !== 'all' && (
              <span className="ml-1">
                ({stats[f as keyof typeof stats]})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Resource List */}
      <div className="space-y-3">
        {filteredResources.map((resource) => (
          <ResourceCard key={resource.id} resource={resource} getIcon={getResourceIcon} />
        ))}
      </div>

      {/* Visualization */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-3">Impact Visualization</h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <ImpactDiagram resources={filteredResources} />
        </div>
      </div>
    </div>
  );
}

function ResourceCard({ resource, getIcon }: { resource: AffectedResource; getIcon: (type: string) => JSX.Element }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={cn(
            'p-2 rounded-lg',
            resource.impact === 'direct' ? 'bg-red-50 dark:bg-red-900/20' :
            resource.impact === 'indirect' ? 'bg-yellow-50 dark:bg-yellow-900/20' :
            'bg-blue-50 dark:bg-blue-900/20'
          )}>
            {getIcon(resource.type)}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-gray-900 dark:text-white">
                {resource.name}
              </h4>
              <span className={cn(
                'px-2 py-0.5 text-xs rounded-full',
                getSeverityBadge(resource.severity)
              )}>
                {resource.severity}
              </span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {resource.description}
            </p>
            {resource.mitigations && resource.mitigations.length > 0 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-xs text-blue-600 dark:text-blue-400 mt-2 hover:underline"
              >
                {expanded ? 'Hide' : 'Show'} mitigations ({resource.mitigations.length})
              </button>
            )}
          </div>
        </div>
        <span className={cn(
          'px-2 py-1 text-xs rounded',
          getImpactColor(resource.impact)
        )}>
          {resource.impact}
        </span>
      </div>

      {expanded && resource.mitigations && (
        <div className="mt-3 pl-11">
          <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
            <h5 className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Mitigation Strategies
            </h5>
            <ul className="space-y-1">
              {resource.mitigations.map((mitigation, index) => (
                <li key={index} className="text-xs text-gray-600 dark:text-gray-400 flex items-start gap-1">
                  <span>•</span>
                  <span>{mitigation}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function ImpactDiagram({ resources }: { resources: AffectedResource[] }) {
  // Simple visualization - in production would use D3.js or similar
  return (
    <div className="flex items-center justify-center">
      <div className="relative">
        {/* Center - Action */}
        <div className="w-24 h-24 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
          Action
        </div>
        
        {/* Direct Impact Ring */}
        <div className="absolute -inset-8 border-2 border-red-300 dark:border-red-700 rounded-full" />
        
        {/* Indirect Impact Ring */}
        <div className="absolute -inset-16 border-2 border-yellow-300 dark:border-yellow-700 rounded-full" />
        
        {/* Downstream Impact Ring */}
        <div className="absolute -inset-24 border-2 border-blue-300 dark:border-blue-700 rounded-full" />
      </div>
    </div>
  );
}

function getSeverityBadge(severity: string) {
  const colors = {
    low: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
    high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400',
    critical: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
  };
  return colors[severity as keyof typeof colors] || colors.low;
}

function getImpactColor(impact: string) {
  switch (impact) {
    case 'direct':
      return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
    case 'indirect':
      return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
    case 'downstream':
      return 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20';
    default:
      return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20';
  }
}