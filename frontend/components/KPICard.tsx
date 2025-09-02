import React from 'react';
import {
  TrendingUp, TrendingDown, Minus, ArrowRight,
  ChevronRight, MoreVertical, ExternalLink
} from 'lucide-react';
import Link from 'next/link';

interface KPICardProps {
  title: string;
  metric: string | number;
  subtext?: string;
  change?: number;
  changeLabel?: string;
  trend?: 'up' | 'down' | 'stable';
  trendColor?: 'positive' | 'negative' | 'neutral';
  action?: {
    label: string;
    href?: string;
    onClick?: () => void;
  };
  icon?: React.ReactNode;
  sparklineData?: number[];
  variant?: 'default' | 'gradient' | 'outlined';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  footer?: React.ReactNode;
}

const sizeConfig = {
  sm: {
    padding: 'p-3',
    titleSize: 'text-xs',
    metricSize: 'text-xl',
    subtextSize: 'text-xs',
    iconSize: 'w-4 h-4',
  },
  md: {
    padding: 'p-4',
    titleSize: 'text-sm',
    metricSize: 'text-2xl',
    subtextSize: 'text-sm',
    iconSize: 'w-5 h-5',
  },
  lg: {
    padding: 'p-6',
    titleSize: 'text-base',
    metricSize: 'text-3xl',
    subtextSize: 'text-base',
    iconSize: 'w-6 h-6',
  },
};

export default function KPICard({
  title,
  metric,
  subtext,
  change,
  changeLabel,
  trend = 'stable',
  trendColor,
  action,
  icon,
  sparklineData,
  variant = 'default',
  size = 'md',
  className = '',
  footer,
}: KPICardProps) {
  const config = sizeConfig[size];
  
  // Determine trend color if not explicitly provided
  const getTrendColor = () => {
    if (trendColor) return trendColor;
    if (trend === 'up') return change && change >= 0 ? 'positive' : 'negative';
    if (trend === 'down') return change && change < 0 ? 'positive' : 'negative';
    return 'neutral';
  };
  
  const actualTrendColor = getTrendColor();
  
  const trendColorClasses = {
    positive: 'text-green-600 dark:text-green-400',
    negative: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  };
  
  const variantClasses = {
    default: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
    gradient: 'bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800',
    outlined: 'bg-transparent border-2 border-gray-300 dark:border-gray-600',
  };
  
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  
  // Generate simple sparkline
  const renderSparkline = () => {
    if (!sparklineData || sparklineData.length === 0) return null;
    
    const max = Math.max(...sparklineData);
    const min = Math.min(...sparklineData);
    const range = max - min || 1;
    const width = 100;
    const height = 30;
    
    const points = sparklineData.map((value, index) => {
      const x = (index / (sparklineData.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <svg 
        className="w-full h-8 mt-2" 
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={trendColorClasses[actualTrendColor]}
          points={points}
        />
      </svg>
    );
  };
  
  const cardContent = (
    <>
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon && (
            <div className={`${config.iconSize} text-gray-500 dark:text-gray-400`}>
              {icon}
            </div>
          )}
          <h3 className={`${config.titleSize} font-medium text-gray-600 dark:text-gray-400`}>
            {title}
          </h3>
        </div>
        {action && !action.href && (
          <button
            onClick={action.onClick}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
        )}
      </div>
      
      {/* Metric */}
      <div className="flex items-baseline gap-3">
        <div className={`${config.metricSize} font-bold text-gray-900 dark:text-white`}>
          {metric}
        </div>
        {change !== undefined && (
          <div className={`flex items-center gap-1 ${trendColorClasses[actualTrendColor]}`}>
            <TrendIcon className="w-4 h-4" />
            <span className="text-sm font-medium">
              {change > 0 ? '+' : ''}{change}%
            </span>
            {changeLabel && (
              <span className="text-xs opacity-75">
                {changeLabel}
              </span>
            )}
          </div>
        )}
      </div>
      
      {/* Subtext */}
      {subtext && (
        <p className={`${config.subtextSize} text-gray-500 dark:text-gray-400 mt-1`}>
          {subtext}
        </p>
      )}
      
      {/* Sparkline */}
      {sparklineData && renderSparkline()}
      
      {/* Action */}
      {action && (
        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
          {action.href ? (
            <Link
              href={action.href}
              className="flex items-center justify-between text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              <span>{action.label}</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          ) : (
            <button
              onClick={action.onClick}
              className="w-full flex items-center justify-between text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              <span>{action.label}</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          )}
        </div>
      )}
      
      {/* Footer */}
      {footer && (
        <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
          {footer}
        </div>
      )}
    </>
  );
  
  return (
    <div
      className={`
        ${config.padding}
        ${variantClasses[variant]}
        rounded-lg shadow-sm hover:shadow-md transition-shadow
        ${className}
      `.trim()}
    >
      {cardContent}
    </div>
  );
}

// Compound component for grid layouts
export function KPICardGrid({ 
  children, 
  columns = 4 
}: { 
  children: React.ReactNode;
  columns?: 1 | 2 | 3 | 4 | 5 | 6;
}) {
  const columnClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
    6: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6',
  };
  
  return (
    <div className={`grid ${columnClasses[columns]} gap-4`}>
      {children}
    </div>
  );
}