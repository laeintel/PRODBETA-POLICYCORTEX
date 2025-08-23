'use client';

import React from 'react';
import { Sparklines, SparklinesLine, SparklinesSpots } from 'react-sparklines';
import { ArrowUp, ArrowDown } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: React.ReactNode;
  sparklineData?: number[];
  onClick?: () => void;
  className?: string;
  status?: 'success' | 'warning' | 'error' | 'neutral';
  trend?: 'up' | 'down' | 'stable';
  alert?: string;
}

export default function MetricCard({
  title,
  value,
  change,
  changeLabel,
  icon,
  sparklineData,
  onClick,
  className = '',
  status = 'neutral',
  trend,
  alert
}: MetricCardProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'success':
        return 'border-green-200 dark:border-green-800 hover:border-green-300 dark:hover:border-green-700';
      case 'warning':
        return 'border-yellow-200 dark:border-yellow-800 hover:border-yellow-300 dark:hover:border-yellow-700';
      case 'error':
        return 'border-red-200 dark:border-red-800 hover:border-red-300 dark:hover:border-red-700';
      default:
        return 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600';
    }
  };

  const getTrendColor = () => {
    if (change === undefined) return 'text-gray-500 dark:text-gray-400';
    return change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
  };

  return (
    <div
      className={`
        bg-white dark:bg-gray-800 rounded-lg border-2 p-6 shadow-sm transition-all duration-200
        ${getStatusColor()}
        ${onClick ? 'cursor-pointer hover:shadow-md hover:scale-[1.02]' : ''}
        ${className}
      `}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-3">
            {icon && (
              <div className="flex-shrink-0">
                <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  {icon}
                </div>
              </div>
            )}
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                {title}
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {value}
              </p>
              {alert && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                  {alert}
                </p>
              )}
              {change !== undefined && (
                <div className={`flex items-center mt-1 text-sm ${getTrendColor()}`}>
                  {change >= 0 ? (
                    <ArrowUp className="h-4 w-4 mr-1" />
                  ) : (
                    <ArrowDown className="h-4 w-4 mr-1" />
                  )}
                  <span className="font-medium">
                    {Math.abs(change)}%
                  </span>
                  {changeLabel && (
                    <span className="ml-1 text-gray-500 dark:text-gray-400">
                      {changeLabel}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
        {sparklineData && sparklineData.length > 0 && (
          <div className="flex-shrink-0 ml-4">
            <div className="w-20 h-12">
              <Sparklines data={sparklineData} width={80} height={48}>
                <SparklinesLine 
                  color={change && change >= 0 ? '#10b981' : '#ef4444'} 
                  style={{ strokeWidth: 2, fill: 'none' }}
                />
                <SparklinesSpots 
                  size={2} 
                  style={{ fill: change && change >= 0 ? '#10b981' : '#ef4444' }}
                />
              </Sparklines>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}