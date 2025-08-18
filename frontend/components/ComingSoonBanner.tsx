'use client';

import React from 'react';
import { Sparkles, Clock, ArrowRight } from 'lucide-react';

interface ComingSoonBannerProps {
  feature: string;
  description?: string;
  estimatedDate?: string;
  variant?: 'inline' | 'full';
}

export const ComingSoonBanner: React.FC<ComingSoonBannerProps> = ({ 
  feature, 
  description,
  estimatedDate,
  variant = 'inline' 
}) => {
  if (variant === 'full') {
    return (
      <div className="min-h-[400px] flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <Sparkles className="w-10 h-10 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
            {feature} Coming Soon
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {description || 'This feature is currently under development and will be available soon.'}
          </p>
          {estimatedDate && (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <Clock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm text-purple-700 dark:text-purple-300">
                Expected: {estimatedDate}
              </span>
            </div>
          )}
          <div className="mt-8">
            <button className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all">
              Get Notified
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
      <div className="flex items-center gap-3">
        <div className="flex-shrink-0">
          <Sparkles className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        </div>
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-purple-900 dark:text-purple-300">
            {feature} - Coming Soon
          </h3>
          {description && (
            <p className="text-xs text-purple-700 dark:text-purple-400 mt-1">
              {description}
            </p>
          )}
        </div>
        {estimatedDate && (
          <div className="flex items-center gap-1 text-xs text-purple-600 dark:text-purple-400">
            <Clock className="w-3 h-3" />
            <span>{estimatedDate}</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Feature-specific banners with pre-configured messages
export const FeatureBanners = {
  KnowledgeGraph: () => (
    <ComingSoonBanner
      feature="Knowledge Graph Visualization"
      description="Interactive graph view of resource relationships and dependencies"
      estimatedDate="Q1 2025"
      variant="full"
    />
  ),
  
  VoiceAssistant: () => (
    <ComingSoonBanner
      feature="Voice Assistant"
      description="Control PolicyCortex with natural voice commands"
      estimatedDate="Q2 2025"
      variant="inline"
    />
  ),
  
  AdvancedML: () => (
    <ComingSoonBanner
      feature="Advanced ML Models"
      description="Custom-trained models for your specific compliance needs"
      estimatedDate="Q1 2025"
      variant="inline"
    />
  ),
  
  MultiCloud: () => (
    <ComingSoonBanner
      feature="Multi-Cloud Support"
      description="Extend governance to AWS and GCP environments"
      estimatedDate="Q2 2025"
      variant="inline"
    />
  ),
  
  AutoRemediation: () => (
    <ComingSoonBanner
      feature="Automated Remediation"
      description="One-click fixes for compliance violations"
      estimatedDate="Available Now in Beta"
      variant="inline"
    />
  )
};