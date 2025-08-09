'use client';

import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, XCircle, FileText, Code } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PreflightDiffProps {
  actionId?: string;
}

export function PreflightDiff({ actionId }: PreflightDiffProps) {
  const [diffData, setDiffData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'unified' | 'split'>('unified');

  useEffect(() => {
    if (actionId) {
      fetchPreflightDiff();
    }
  }, [actionId]);

  const fetchPreflightDiff = async () => {
    try {
      const response = await fetch(`/api/v1/actions/${actionId}/preflight`);
      const data = await response.json();
      setDiffData(data);
    } catch (error) {
      console.error('Failed to fetch preflight diff:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-12 bg-gray-200 dark:bg-gray-700 rounded" />
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Preflight Status */}
      <div className={cn(
        'p-4 rounded-lg border',
        diffData?.status === 'passed' ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' :
        diffData?.status === 'failed' ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' :
        'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
      )}>
        <div className="flex items-start gap-3">
          {diffData?.status === 'passed' ? (
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
          ) : diffData?.status === 'failed' ? (
            <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
          )}
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 dark:text-white">
              Preflight Check {diffData?.status === 'passed' ? 'Passed' : diffData?.status === 'failed' ? 'Failed' : 'In Progress'}
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {diffData?.message || 'Analyzing changes and validating prerequisites...'}
            </p>
          </div>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Configuration Changes</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('unified')}
            className={cn(
              'px-3 py-1 text-sm rounded',
              viewMode === 'unified' 
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'
            )}
          >
            Unified
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={cn(
              'px-3 py-1 text-sm rounded',
              viewMode === 'split'
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'
            )}
          >
            Split
          </button>
        </div>
      </div>

      {/* Diff Display */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        {diffData?.changes?.map((change: any, index: number) => (
          <DiffSection key={index} change={change} viewMode={viewMode} />
        ))}
      </div>

      {/* Validation Results */}
      {diffData?.validations && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Validation Results</h3>
          <div className="space-y-2">
            {diffData.validations.map((validation: any, index: number) => (
              <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                {validation.passed ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500" />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium">{validation.name}</p>
                  {validation.message && (
                    <p className="text-xs text-gray-600 dark:text-gray-400">{validation.message}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DiffSection({ change, viewMode }: { change: any; viewMode: string }) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-800"
      >
        <div className="flex items-center gap-3">
          <FileText className="w-4 h-4 text-gray-400" />
          <span className="font-medium text-sm">{change.resource}</span>
          <span className="text-xs text-gray-500">
            +{change.additions} -{change.deletions}
          </span>
        </div>
        <Code className={cn(
          'w-4 h-4 text-gray-400 transition-transform',
          isExpanded && 'rotate-90'
        )} />
      </button>

      {isExpanded && (
        <div className="bg-gray-900 text-gray-100 font-mono text-xs p-4 overflow-x-auto">
          {viewMode === 'unified' ? (
            <UnifiedDiff lines={change.diff} />
          ) : (
            <SplitDiff before={change.before} after={change.after} />
          )}
        </div>
      )}
    </div>
  );
}

function UnifiedDiff({ lines }: { lines: any[] }) {
  return (
    <div>
      {lines?.map((line: any, index: number) => (
        <div
          key={index}
          className={cn(
            'py-0.5',
            line.type === 'add' && 'bg-green-900/30 text-green-400',
            line.type === 'remove' && 'bg-red-900/30 text-red-400',
            line.type === 'context' && 'text-gray-400'
          )}
        >
          <span className="select-none pr-3 text-gray-600">
            {line.lineNumber}
          </span>
          <span className="select-none pr-3">
            {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
          </span>
          <span>{line.content}</span>
        </div>
      ))}
    </div>
  );
}

function SplitDiff({ before, after }: { before: string[]; after: string[] }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <div className="text-gray-500 pb-2 mb-2 border-b border-gray-700">Before</div>
        {before?.map((line: string, index: number) => (
          <div key={index} className="py-0.5 text-red-400">
            {line}
          </div>
        ))}
      </div>
      <div>
        <div className="text-gray-500 pb-2 mb-2 border-b border-gray-700">After</div>
        {after?.map((line: string, index: number) => (
          <div key={index} className="py-0.5 text-green-400">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}