'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { toast } from '@/hooks/useToast'
import { 
  Network, GitBranch, AlertTriangle, Activity, 
  Zap, Shield, TrendingUp, Download, Filter,
  Maximize2, Settings, RefreshCw, Info, Search,
  Layers, Target, Radio, Cpu, Clock
} from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96">Loading correlation graph...</div>
});

interface CorrelationNode {
  id: string;
  name: string;
  type: 'resource' | 'policy' | 'cost' | 'security' | 'compliance' | 'anomaly';
  risk: 'low' | 'medium' | 'high' | 'critical';
  value: number;
  metadata?: any;
}

interface CorrelationLink {
  source: string;
  target: string;
  strength: number;
  type: 'depends' | 'impacts' | 'similar' | 'inverse' | 'causal';
  bidirectional?: boolean;
}

export default function AICorrelationsPage() {
  const [selectedNode, setSelectedNode] = useState<CorrelationNode | null>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [correlationStrength, setCorrelationStrength] = useState(0.5);
  const [simulationMode, setSimulationMode] = useState(false);
  const [timeRange, setTimeRange] = useState('24h');
  const graphRef = useRef<any>();

  // Mock correlation data
  const graphData = useMemo(() => {
    const nodes: CorrelationNode[] = [
      { id: '1', name: 'Production Database', type: 'resource', risk: 'high', value: 95 },
      { id: '2', name: 'Compliance Policy #42', type: 'policy', risk: 'critical', value: 88 },
      { id: '3', name: 'Cost Anomaly Detected', type: 'cost', risk: 'medium', value: 72 },
      { id: '4', name: 'Security Group sg-123', type: 'security', risk: 'high', value: 85 },
      { id: '5', name: 'GDPR Compliance', type: 'compliance', risk: 'low', value: 92 },
      { id: '6', name: 'CPU Spike Pattern', type: 'anomaly', risk: 'medium', value: 67 },
      { id: '7', name: 'Load Balancer', type: 'resource', risk: 'low', value: 78 },
      { id: '8', name: 'Data Encryption Policy', type: 'policy', risk: 'low', value: 96 },
      { id: '9', name: 'Budget Threshold', type: 'cost', risk: 'high', value: 81 },
      { id: '10', name: 'WAF Rules', type: 'security', risk: 'medium', value: 74 },
      { id: '11', name: 'ISO 27001', type: 'compliance', risk: 'low', value: 89 },
      { id: '12', name: 'Memory Leak Detection', type: 'anomaly', risk: 'critical', value: 45 },
    ];

    const links: CorrelationLink[] = [
      { source: '1', target: '2', strength: 0.9, type: 'impacts' },
      { source: '1', target: '4', strength: 0.85, type: 'depends' },
      { source: '2', target: '5', strength: 0.95, type: 'similar' },
      { source: '3', target: '9', strength: 0.78, type: 'causal' },
      { source: '4', target: '10', strength: 0.82, type: 'similar' },
      { source: '6', target: '12', strength: 0.73, type: 'causal' },
      { source: '7', target: '1', strength: 0.88, type: 'depends', bidirectional: true },
      { source: '8', target: '5', strength: 0.91, type: 'impacts' },
      { source: '9', target: '3', strength: 0.69, type: 'inverse' },
      { source: '10', target: '11', strength: 0.77, type: 'impacts' },
      { source: '11', target: '8', strength: 0.84, type: 'similar' },
      { source: '12', target: '1', strength: 0.92, type: 'impacts' },
    ];

    // Filter links by strength
    const filteredLinks = links.filter(link => link.strength >= correlationStrength);

    // Filter nodes by type if needed
    const filteredNodes = filterType === 'all' 
      ? nodes 
      : nodes.filter(node => node.type === filterType);

    return { 
      nodes: filteredNodes, 
      links: filteredLinks 
    };
  }, [filterType, correlationStrength]);

  const nodeColor = (node: CorrelationNode) => {
    const colors = {
      resource: '#3b82f6',
      policy: '#8b5cf6',
      cost: '#f59e0b',
      security: '#ef4444',
      compliance: '#10b981',
      anomaly: '#ec4899'
    };
    return colors[node.type];
  };

  const getRiskColor = (risk: string) => {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#f97316',
      critical: '#ef4444'
    };
    return colors[risk as keyof typeof colors];
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold mb-2">AI Correlation Engine</h1>
            <p className="text-gray-400">Cross-domain pattern detection and risk propagation analysis</p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              type="button"
              className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              onClick={() => toast({ title: 'Refreshed', description: 'Correlation data refreshed' })}
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
            <button
              type="button"
              className="px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors flex items-center space-x-2"
              onClick={() => toast({ title: 'Export', description: 'Exporting correlation report...' })}
            >
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Network className="w-5 h-5 text-blue-500" />
              <span className="text-xs text-gray-400">Total Correlations</span>
            </div>
            <div className="text-2xl font-bold">247</div>
            <div className="text-xs text-green-500">+12% from last week</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle className="w-5 h-5 text-orange-500" />
              <span className="text-xs text-gray-400">Critical Patterns</span>
            </div>
            <div className="text-2xl font-bold">8</div>
            <div className="text-xs text-red-500">Requires attention</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              <span className="text-xs text-gray-400">Anomalies</span>
            </div>
            <div className="text-2xl font-bold">23</div>
            <div className="text-xs text-yellow-500">3 new detected</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <Shield className="w-5 h-5 text-green-500" />
              <span className="text-xs text-gray-400">Risk Score</span>
            </div>
            <div className="text-2xl font-bold">72</div>
            <div className="text-xs text-gray-400">Medium risk level</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Filter Type</label>
            <select 
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-gray-700 rounded px-3 py-1 text-sm"
            >
              <option value="all">All Types</option>
              <option value="resource">Resources</option>
              <option value="policy">Policies</option>
              <option value="cost">Cost</option>
              <option value="security">Security</option>
              <option value="compliance">Compliance</option>
              <option value="anomaly">Anomalies</option>
            </select>
          </div>
          
          <div className="flex-1">
            <label className="text-xs text-gray-400 block mb-1">
              Correlation Strength: {(correlationStrength * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={correlationStrength}
              onChange={(e) => setCorrelationStrength(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Time Range</label>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-gray-700 rounded px-3 py-1 text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>

          <button type="button"
            onClick={() => setSimulationMode(!simulationMode)}
            className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
              simulationMode ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>What-If Mode</span>
          </button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Correlation Graph */}
        <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center space-x-2">
              <GitBranch className="w-5 h-5" />
              <span>Correlation Network</span>
            </h2>
            <button
              type="button"
              className="p-2 hover:bg-gray-700 rounded transition-colors"
              onClick={() => toast({ title: 'Expanded', description: 'Opening full-screen view' })}
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
          
          <div className="relative h-[600px] bg-white dark:bg-gray-900 rounded-lg overflow-hidden">
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              nodeLabel="name"
              nodeColor={(node: any) => nodeColor(node as CorrelationNode)}
              nodeRelSize={6}
              linkWidth={(link: any) => link.strength * 3}
              linkDirectionalParticles={2}
              linkDirectionalParticleSpeed={0.005}
              onNodeClick={(node: any) => setSelectedNode(node)}
              enableNodeDrag={true}
              enableZoomInteraction={true}
              cooldownTicks={100}
            />
          </div>

          {/* Legend */}
          <div className="mt-4 flex flex-wrap gap-4 text-xs">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Resource</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span>Policy</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>Cost</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>Security</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Compliance</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-pink-500 rounded-full"></div>
              <span>Anomaly</span>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="space-y-6">
          {/* Selected Node Details */}
          {selectedNode ? (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Node Details</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-gray-400 mb-1">Name</p>
                  <p className="font-medium">{selectedNode.name}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Type</p>
                  <span className={`px-2 py-1 rounded text-xs bg-opacity-20 ${
                    selectedNode.type === 'resource' ? 'bg-blue-500 text-blue-400' :
                    selectedNode.type === 'policy' ? 'bg-purple-500 text-purple-400' :
                    selectedNode.type === 'cost' ? 'bg-yellow-500 text-yellow-400' :
                    selectedNode.type === 'security' ? 'bg-red-500 text-red-400' :
                    selectedNode.type === 'compliance' ? 'bg-green-500 text-green-400' :
                    'bg-pink-500 text-pink-400'
                  }`}>
                    {selectedNode.type}
                  </span>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Risk Level</p>
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: getRiskColor(selectedNode.risk) }}
                    ></div>
                    <span className="capitalize">{selectedNode.risk}</span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Impact Score</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                        style={{ width: `${selectedNode.value}%` }}
                      ></div>
                    </div>
                    <span className="text-sm">{selectedNode.value}%</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-center py-8 text-gray-500">
                <Info className="w-8 h-8 mx-auto mb-2" />
                <p className="text-sm">Select a node to view details</p>
              </div>
            </div>
          )}

          {/* Risk Propagation */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Radio className="w-5 h-5" />
              <span>Risk Propagation</span>
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Primary Impact</span>
                <span className="text-orange-500 font-medium">High</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Secondary Impact</span>
                <span className="text-yellow-500 font-medium">Medium</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Cascade Radius</span>
                <span className="font-medium">3 hops</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Affected Resources</span>
                <span className="font-medium">17</span>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-700">
                <p className="text-xs text-gray-400 mb-2">Propagation Timeline</p>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-xs">
                    <Clock className="w-3 h-3" />
                    <span>0-5 min: Critical systems</span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs">
                    <Clock className="w-3 h-3" />
                    <span>5-15 min: Dependent services</span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs">
                    <Clock className="w-3 h-3" />
                    <span>15-30 min: Secondary systems</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Anomaly Timeline */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Activity className="w-5 h-5" />
              <span>Anomaly Timeline</span>
            </h3>
            <div className="space-y-3">
              {[
                { time: '2 min ago', event: 'CPU spike detected', severity: 'high' },
                { time: '15 min ago', event: 'Policy violation', severity: 'critical' },
                { time: '1 hour ago', event: 'Cost threshold exceeded', severity: 'medium' },
                { time: '3 hours ago', event: 'Unusual traffic pattern', severity: 'low' },
              ].map((item, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className={`w-2 h-2 rounded-full mt-1.5 ${
                    item.severity === 'critical' ? 'bg-red-500' :
                    item.severity === 'high' ? 'bg-orange-500' :
                    item.severity === 'medium' ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}></div>
                  <div className="flex-1">
                    <p className="text-sm">{item.event}</p>
                    <p className="text-xs text-gray-400">{item.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* What-If Simulator */}
      {simulationMode && (
        <div className="mt-6 bg-gradient-to-r from-purple-900/20 to-pink-900/20 rounded-lg p-6 border border-purple-500/30">
          <h3 className="text-xl font-semibold mb-4 flex items-center space-x-2">
            <Cpu className="w-6 h-6 text-purple-400" />
            <span>What-If Scenario Simulator</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium mb-2">Simulate Event</label>
              <select className="w-full bg-gray-800 rounded-lg px-4 py-2">
                <option>Resource Failure</option>
                <option>Policy Change</option>
                <option>Security Breach</option>
                <option>Cost Spike</option>
                <option>Compliance Violation</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Impact Level</label>
              <select className="w-full bg-gray-800 rounded-lg px-4 py-2">
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
                <option>Critical</option>
              </select>
            </div>
          </div>
          <div className="mt-4">
            <label className="block text-sm font-medium mb-2">Target Resource</label>
            <input 
              type="text" 
              placeholder="Select or search for a resource..."
              className="w-full bg-gray-800 rounded-lg px-4 py-2"
            />
          </div>
          <button type="button" className="mt-4 px-6 py-2 bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors">
            Run Simulation
          </button>
        </div>
      )}
    </div>
  );
}