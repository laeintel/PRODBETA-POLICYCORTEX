'use client';

import React from 'react';
import { ComingSoonBanner } from './ComingSoonBanner';
import { Box, Database, FileText, Shield, Server, DollarSign, GitBranch, AlertCircle } from 'lucide-react';

interface EmptyStateWithBannerProps {
  title: string;
  description?: string;
  iconType?: 'resources' | 'policies' | 'security' | 'costs' | 'correlations' | 'predictions' | 'default';
  showComingSoon?: boolean;
  comingSoonFeature?: string;
  comingSoonDescription?: string;
  estimatedDate?: string;
}

export const EmptyStateWithBanner: React.FC<EmptyStateWithBannerProps> = ({
  title,
  description,
  iconType = 'default',
  showComingSoon = true,
  comingSoonFeature,
  comingSoonDescription,
  estimatedDate = 'Q1 2025'
}) => {
  const getIcon = () => {
    switch (iconType) {
      case 'resources':
        return <Server className="w-12 h-12 text-gray-400" />;
      case 'policies':
        return <Shield className="w-12 h-12 text-gray-400" />;
      case 'security':
        return <Shield className="w-12 h-12 text-gray-400" />;
      case 'costs':
        return <DollarSign className="w-12 h-12 text-gray-400" />;
      case 'correlations':
        return <GitBranch className="w-12 h-12 text-gray-400" />;
      case 'predictions':
        return <AlertCircle className="w-12 h-12 text-gray-400" />;
      default:
        return <Database className="w-12 h-12 text-gray-400" />;
    }
  };

  return (
    <div className="min-h-[400px] flex flex-col items-center justify-center p-8 space-y-6">
      {showComingSoon && comingSoonFeature && (
        <ComingSoonBanner
          feature={comingSoonFeature}
          description={comingSoonDescription}
          estimatedDate={estimatedDate}
          variant="inline"
        />
      )}
      
      <div className="text-center">
        <div className="flex justify-center mb-4">
          {getIcon()}
        </div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          {title}
        </h3>
        {description && (
          <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            {description}
          </p>
        )}
      </div>

      {/* Demo Mode Indicator */}
      <div className="text-xs text-gray-400 dark:text-gray-500">
        Demo Mode: No live data available
      </div>
    </div>
  );
};

// Pre-configured empty states for common views
export const EmptyStates = {
  Resources: () => (
    <EmptyStateWithBanner
      title="No Resources Found"
      description="Resources will appear here once they are discovered from your Azure subscriptions."
      iconType="resources"
      comingSoonFeature="Live Resource Discovery"
      comingSoonDescription="Real-time Azure resource synchronization and monitoring"
    />
  ),

  Policies: () => (
    <EmptyStateWithBanner
      title="No Policies Configured"
      description="Create policies to enforce governance and compliance standards."
      iconType="policies"
      comingSoonFeature="Policy Templates Marketplace"
      comingSoonDescription="Pre-built compliance templates for common standards"
    />
  ),

  SecurityFindings: () => (
    <EmptyStateWithBanner
      title="No Security Findings"
      description="Security issues and recommendations will appear here."
      iconType="security"
      comingSoonFeature="Azure Defender Integration"
      comingSoonDescription="Real-time security threat detection and remediation"
    />
  ),

  CostAnalysis: () => (
    <EmptyStateWithBanner
      title="No Cost Data Available"
      description="Cost optimization recommendations will appear here."
      iconType="costs"
      comingSoonFeature="FinOps Intelligence Engine"
      comingSoonDescription="AI-powered cost optimization and forecasting"
    />
  ),

  Correlations: () => (
    <EmptyStateWithBanner
      title="No Correlations Detected"
      description="Cross-domain patterns and insights will appear here."
      iconType="correlations"
      comingSoonFeature="Pattern Recognition Engine"
      comingSoonDescription="Advanced ML-based correlation detection"
      estimatedDate="Q2 2025"
    />
  ),

  Predictions: () => (
    <EmptyStateWithBanner
      title="No Predictions Available"
      description="Compliance drift predictions will appear here."
      iconType="predictions"
      comingSoonFeature="Predictive Analytics"
      comingSoonDescription="ML-powered compliance drift forecasting"
    />
  )
};