'use client';

import React from 'react';
import NextLink from 'next/link';
import { ArrowLeft, Activity } from 'lucide-react';

interface TacticalPageTemplateProps {
  title: string;
  subtitle?: string;
  icon?: React.ElementType;
  children?: React.ReactNode;
  backLink?: string;
}

export default function TacticalPageTemplate({ 
  title, 
  subtitle, 
  icon: Icon = Activity,
  children,
  backLink = '/tactical'
}: TacticalPageTemplateProps) {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <NextLink href={backLink} className="text-gray-400 hover:text-gray-200">
                <ArrowLeft className="w-5 h-5" />
              </NextLink>
              <div className="h-6 w-px bg-gray-700" />
              <div className="flex items-center space-x-3">
                <Icon className="w-6 h-6 text-green-500" />
                <div>
                  <h1 className="text-xl font-bold">{title}</h1>
                  {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="p-6">
        {children || (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center">
            <Icon className="w-16 h-16 text-gray-700 mx-auto mb-4" />
            <h2 className="text-2xl font-bold mb-2">{title}</h2>
            <p className="text-gray-400 mb-6">This section is currently being configured.</p>
            <div className="inline-flex items-center space-x-2 text-sm text-gray-500">
              <Activity className="w-4 h-4 animate-pulse" />
              <span>System initializing...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}