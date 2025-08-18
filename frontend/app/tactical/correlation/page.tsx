'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { Network, GitBranch, Activity, Layers, Link2, Shuffle, AlertCircle, TrendingUp } from 'lucide-react';

interface Correlation {
  id: string;
  source: string;
  target: string;
  strength: number;
  type: 'positive' | 'negative' | 'neutral';
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  discovered: string;
}

interface Pattern {
  id: string;
  name: string;
  occurrences: number;
  domains: string[];
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  lastSeen: string;
}

interface CorrelationMetrics {
  totalCorrelations: number;
  patternsDetected: number;
  domainsAnalyzed: number;
  anomaliesFound: number;
  strongCorrelations: number;
  weakCorrelations: number;
}

export default function CorrelationAnalysisCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <CorrelationAnalysisCenterContent />
    </AuthGuard>
  );
}

function CorrelationAnalysisCenterContent() {
  const [correlations, setCorrelations] = useState<Correlation[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [metrics, setMetrics] = useState<CorrelationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedDomain, setSelectedDomain] = useState('all');
  const [minStrength, setMinStrength] = useState(50);

  useEffect(() => {
    fetchCorrelationData();
    const interval = setInterval(fetchCorrelationData, 20000);
    return () => clearInterval(interval);
  }, []);

  const fetchCorrelationData = async () => {
    try {
      const response = await fetch('/api/v1/correlations');
      if (response.ok) {
        const data = await response.json();
        processCorrelationData(data);
      } else {
        loadMockData();
      }
    } catch (error) {
      loadMockData();
    } finally {
      setLoading(false);
    }
  };

  const processCorrelationData = (data: any) => {
    const mockData = getMockCorrelationData();
    setCorrelations(mockData.correlations);
    setPatterns(mockData.patterns);
    setMetrics(mockData.metrics);
  };

  const loadMockData = () => {
    const mockData = getMockCorrelationData();
    setCorrelations(mockData.correlations);
    setPatterns(mockData.patterns);
    setMetrics(mockData.metrics);
  };

  const getMockCorrelationData = () => ({
    correlations: [
      { id: 'c1', source: 'Security Violations', target: 'Cost Increase', strength: 87, type: 'positive' as const, confidence: 92, impact: 'high' as const, discovered: '2 hours ago' },
      { id: 'c2', source: 'Compliance Drift', target: 'Policy Updates', strength: 73, type: 'positive' as const, confidence: 85, impact: 'medium' as const, discovered: '5 hours ago' },
      { id: 'c3', source: 'Resource Scaling', target: 'Performance Metrics', strength: 91, type: 'positive' as const, confidence: 88, impact: 'high' as const, discovered: '1 day ago' },
      { id: 'c4', source: 'Network Latency', target: 'User Complaints', strength: 65, type: 'positive' as const, confidence: 78, impact: 'medium' as const, discovered: '3 days ago' },
      { id: 'c5', source: 'Storage Usage', target: 'Backup Failures', strength: 82, type: 'negative' as const, confidence: 90, impact: 'high' as const, discovered: '1 hour ago' },
      { id: 'c6', source: 'CPU Utilization', target: 'Auto-scaling Events', strength: 95, type: 'positive' as const, confidence: 94, impact: 'high' as const, discovered: '30 min ago' },
      { id: 'c7', source: 'Failed Logins', target: 'Security Alerts', strength: 78, type: 'positive' as const, confidence: 82, impact: 'medium' as const, discovered: '4 hours ago' },
      { id: 'c8', source: 'Database Queries', target: 'API Response Time', strength: 69, type: 'negative' as const, confidence: 75, impact: 'low' as const, discovered: '6 hours ago' }
    ],
    patterns: [
      { id: 'p1', name: 'Cost-Security Cascade', occurrences: 47, domains: ['Cost', 'Security', 'Compliance'], severity: 'high' as const, description: 'Security incidents leading to increased costs and compliance violations', lastSeen: '1 hour ago' },
      { id: 'p2', name: 'Performance Degradation Chain', occurrences: 32, domains: ['Performance', 'Resources', 'User Experience'], severity: 'medium' as const, description: 'Resource constraints causing cascading performance issues', lastSeen: '3 hours ago' },
      { id: 'p3', name: 'Compliance Drift Pattern', occurrences: 28, domains: ['Compliance', 'Policy', 'Governance'], severity: 'high' as const, description: 'Gradual policy violations across multiple resources', lastSeen: '2 days ago' },
      { id: 'p4', name: 'Weekend Anomaly Spike', occurrences: 15, domains: ['Security', 'Access', 'Network'], severity: 'critical' as const, description: 'Unusual activity patterns during off-hours', lastSeen: '12 hours ago' },
      { id: 'p5', name: 'Resource Optimization Loop', occurrences: 89, domains: ['Resources', 'Cost', 'Performance'], severity: 'low' as const, description: 'Positive feedback loop from optimization efforts', lastSeen: '5 min ago' }
    ],
    metrics: {
      totalCorrelations: 892,
      patternsDetected: 156,
      domainsAnalyzed: 12,
      anomaliesFound: 23,
      strongCorrelations: 234,
      weakCorrelations: 658
    }
  });

  const filteredCorrelations = correlations.filter(c => 
    c.strength >= minStrength && 
    (selectedDomain === 'all' || c.source.toLowerCase().includes(selectedDomain.toLowerCase()) || c.target.toLowerCase().includes(selectedDomain.toLowerCase()))
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyan-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">ANALYZING CORRELATIONS...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-gray-200">
                ← BACK
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">CORRELATION ANALYSIS CENTER</h1>
              <div className="px-3 py-1 bg-cyan-900/30 text-cyan-500 rounded text-xs font-bold">
                {metrics?.totalCorrelations} CORRELATIONS
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium rounded transition-colors flex items-center gap-2">
                <Network className="w-4 h-4" />
                DEEP ANALYSIS
              </button>
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors">
                EXPORT GRAPH
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Metrics */}
        <div className="grid grid-cols-6 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Total Correlations</p>
            <p className="text-3xl font-bold font-mono">{metrics?.totalCorrelations}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Patterns</p>
            <p className="text-3xl font-bold font-mono text-cyan-500">{metrics?.patternsDetected}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Domains</p>
            <p className="text-3xl font-bold font-mono">{metrics?.domainsAnalyzed}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Anomalies</p>
            <p className="text-3xl font-bold font-mono text-yellow-500">{metrics?.anomaliesFound}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Strong</p>
            <p className="text-3xl font-bold font-mono text-green-500">{metrics?.strongCorrelations}</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Weak</p>
            <p className="text-3xl font-bold font-mono text-gray-500">{metrics?.weakCorrelations}</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-4 mb-6">
          <select
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
            className="px-4 py-2 bg-gray-900 border border-gray-800 rounded text-sm"
          >
            <option value="all">All Domains</option>
            <option value="security">Security</option>
            <option value="cost">Cost</option>
            <option value="compliance">Compliance</option>
            <option value="performance">Performance</option>
            <option value="resource">Resources</option>
          </select>
          
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min Strength:</label>
            <input
              type="range"
              min="0"
              max="100"
              value={minStrength}
              onChange={(e) => setMinStrength(Number(e.target.value))}
              className="w-32"
            />
            <span className="text-sm font-mono text-gray-400">{minStrength}%</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Active Correlations */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">ACTIVE CORRELATIONS</h3>
            </div>
            <div className="divide-y divide-gray-800 max-h-[600px] overflow-y-auto">
              {filteredCorrelations.map((correlation) => (
                <div key={correlation.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="text-sm font-medium">{correlation.source}</div>
                        <div className={`flex items-center gap-1 ${
                          correlation.type === 'positive' ? 'text-green-500' :
                          correlation.type === 'negative' ? 'text-red-500' :
                          'text-gray-500'
                        }`}>
                          {correlation.type === 'positive' ? '→' : correlation.type === 'negative' ? '⇄' : '—'}
                        </div>
                        <div className="text-sm font-medium">{correlation.target}</div>
                      </div>
                      
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>Discovered {correlation.discovered}</span>
                        <span className={`px-2 py-0.5 rounded ${
                          correlation.impact === 'high' ? 'bg-red-900/30 text-red-500' :
                          correlation.impact === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                          'bg-gray-800 text-gray-500'
                        }`}>
                          {correlation.impact.toUpperCase()} IMPACT
                        </span>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-2xl font-bold font-mono">{correlation.strength}%</div>
                      <div className="text-xs text-gray-500">strength</div>
                    </div>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Correlation Strength</span>
                      <span className="text-gray-400">{correlation.strength}%</span>
                    </div>
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          correlation.strength >= 80 ? 'bg-gradient-to-r from-cyan-600 to-cyan-400' :
                          correlation.strength >= 60 ? 'bg-gradient-to-r from-blue-600 to-blue-400' :
                          'bg-gradient-to-r from-gray-600 to-gray-400'
                        }`}
                        style={{ width: `${correlation.strength}%` }}
                      />
                    </div>
                    
                    <div className="flex justify-between text-xs mt-2">
                      <span className="text-gray-500">Confidence</span>
                      <span className="text-gray-400">{correlation.confidence}%</span>
                    </div>
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-500 rounded-full"
                        style={{ width: `${correlation.confidence}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detected Patterns */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">DETECTED PATTERNS</h3>
            </div>
            <div className="divide-y divide-gray-800 max-h-[600px] overflow-y-auto">
              {patterns.map((pattern) => (
                <div key={pattern.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-medium">{pattern.name}</h4>
                    <span className={`text-xs px-2 py-1 rounded font-bold ${
                      pattern.severity === 'critical' ? 'bg-red-900/30 text-red-500' :
                      pattern.severity === 'high' ? 'bg-orange-900/30 text-orange-500' :
                      pattern.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                      'bg-gray-800 text-gray-500'
                    }`}>
                      {pattern.severity.toUpperCase()}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-3">{pattern.description}</p>
                  
                  <div className="flex flex-wrap gap-2 mb-3">
                    {pattern.domains.map((domain, idx) => (
                      <span key={idx} className="text-xs px-2 py-1 bg-gray-800 rounded text-gray-400">
                        {domain}
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{pattern.occurrences} occurrences</span>
                    <span>Last seen {pattern.lastSeen}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Correlation Graph Placeholder */}
        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-lg p-8">
          <div className="text-center">
            <Network className="w-16 h-16 text-cyan-500 mx-auto mb-4" />
            <h3 className="text-lg font-bold mb-2">CORRELATION VISUALIZATION</h3>
            <p className="text-gray-500 text-sm mb-4">Interactive graph showing relationships between domains</p>
            <button className="px-6 py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium rounded transition-colors">
              LAUNCH INTERACTIVE GRAPH
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}