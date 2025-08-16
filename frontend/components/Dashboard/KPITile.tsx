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

import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

export interface KPITileProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: number;
  changeLabel?: string;
  trend?: 'up' | 'down' | 'neutral';
  status?: 'success' | 'warning' | 'error' | 'neutral';
  deepLink?: string;
  icon?: React.ReactNode;
  loading?: boolean;
  sparklineData?: number[];
  onClick?: () => void;
}

export function KPITile({
  title,
  value,
  subtitle,
  change,
  changeLabel,
  trend = 'neutral',
  status = 'neutral',
  deepLink,
  icon,
  loading = false,
  sparklineData,
  onClick,
}: KPITileProps) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4" />;
      case 'down':
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <Minus className="w-4 h-4" />;
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getTrendColor = () => {
    if (status === 'success') return 'text-green-600 dark:text-green-400';
    if (status === 'warning') return 'text-yellow-600 dark:text-yellow-400';
    if (status === 'error') return 'text-red-600 dark:text-red-400';
    
    switch (trend) {
      case 'up':
        return change && change > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
      case 'down':
        return change && change < 0 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const content = (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        'relative p-6 rounded-2xl border border-gray-200/60 dark:border-white/10',
        'bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800',
        'shadow-[0_10px_30px_-10px_rgba(0,0,0,0.15)] hover:shadow-[0_20px_40px_-10px_rgba(0,0,0,0.25)]',
        'transition-all duration-300 backdrop-blur-sm',
        'before:content-[""] before:absolute before:inset-0 before:rounded-2xl before:bg-gradient-to-br before:from-white/40 before:to-transparent before:pointer-events-none',
        'after:content-[""] after:absolute after:-inset-px after:rounded-2xl after:bg-gradient-to-br after:from-white/10 after:to-transparent after:opacity-0 hover:after:opacity-100 after:pointer-events-none',
        onClick || deepLink ? 'cursor-pointer' : '',
        loading && 'animate-pulse'
      )}
      onClick={onClick}
    >
      {/* Status indicator */}
      {status !== 'neutral' && (
        <div className="absolute top-3 right-3">
          {getStatusIcon()}
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          {icon && (
            <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              {icon}
            </div>
          )}
          <div>
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              {title}
            </h3>
            {subtitle && (
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                {subtitle}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Value */}
      <div className="mb-4">
        {loading ? (
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-24" />
        ) : (
          <div className="text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white drop-shadow-sm">
            {value}
          </div>
        )}
      </div>

      {/* Sparkline */}
      {sparklineData && sparklineData.length > 0 && (
        <div className="mb-4 h-12">
          <Sparkline data={sparklineData} />
        </div>
      )}

      {/* Change indicator */}
      {change !== undefined && (
        <div className={cn('flex items-center gap-2 text-sm', getTrendColor())}>
          {getTrendIcon()}
          <span className="font-medium">
            {change > 0 ? '+' : ''}{change}%
          </span>
          {changeLabel && (
            <span className="text-gray-500 dark:text-gray-400">
              {changeLabel}
            </span>
          )}
        </div>
      )}

      {/* Deep link indicator */}
      {deepLink && (
        <div className="absolute bottom-2 right-2 opacity-70 hover:opacity-100 transition-opacity">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 7l5 5m0 0l-5 5m5-5H6"
            />
          </svg>
        </div>
      )}
    </motion.div>
  );

  if (deepLink) {
    return (
      <Link href={deepLink} passHref>
        {content}
      </Link>
    );
  }

  return content;
}

// Simple sparkline component
function Sparkline({ data }: { data: number[] }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min;
  const width = 100;
  const height = 40;
  
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="text-blue-500"
    >
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        points={points}
      />
    </svg>
  );
}