'use client';

import React, { useEffect, useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import QuantumLayout from '@/components/quantum/QuantumLayout';

// Dynamic imports for 3D components (to avoid SSR issues)
const AzureResourceMap = dynamic(
  () => import('@/components/quantum/AzureResourceMap'),
  { ssr: false }
);

const CorrelationConstellation = dynamic(
  () => import('@/components/quantum/CorrelationConstellation'),
  { ssr: false }
);

const ComplianceGrid = dynamic(
  () => import('@/components/quantum/ComplianceGrid'),
  { ssr: false }
);

const ThreatRadar = dynamic(
  () => import('@/components/quantum/ThreatRadar'),
  { ssr: false }
);

export default function QuantumDashboard() {
  const [selectedMetric, setSelectedMetric] = useState('compliance');
  const [realTimeData, setRealTimeData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Simulated real-time metrics
  const metrics = {
    resources: { value: 2847, change: '+12%', status: 'optimal' },
    compliance: { value: 98.7, change: '+2.3%', status: 'excellent' },
    threats: { value: 3, change: '-67%', status: 'low' },
    cost: { value: '$127,439', change: '-8%', status: 'optimized' },
    predictions: { value: 14, change: '+4', status: 'active' },
    correlations: { value: 892, change: '+127', status: 'processing' },
  };

  useEffect(() => {
    // Simulate connecting to real-time data stream
    const timer = setTimeout(() => setIsConnected(true), 1000);
    
    // Simulate real-time data updates
    const dataInterval = setInterval(() => {
      setRealTimeData({
        timestamp: new Date().toISOString(),
        cpuUsage: Math.random() * 100,
        memoryUsage: Math.random() * 100,
        networkTraffic: Math.random() * 1000,
        activeAlerts: Math.floor(Math.random() * 10),
      });
    }, 2000);

    return () => {
      clearTimeout(timer);
      clearInterval(dataInterval);
    };
  }, []);

  return (
    <QuantumLayout>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6 border border-quantum-blue/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-display font-bold bg-gradient-to-r from-quantum-blue to-neural-purple bg-clip-text text-transparent">
                NEURAL COMMAND CENTER
              </h1>
              <p className="text-photon-white/60 font-mono text-sm mt-1">
                Real-time Quantum Governance Overview
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-success-aurora' : 'bg-alert-solar'} pulse-quantum`} />
              <span className="text-sm font-mono text-photon-white/60">
                {isConnected ? 'LIVE' : 'CONNECTING...'}
              </span>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-6 gap-4">
            {Object.entries(metrics).map(([key, data], index) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
                onClick={() => setSelectedMetric(key)}
                className={`
                  relative p-4 rounded-xl cursor-pointer transition-all duration-300
                  ${selectedMetric === key 
                    ? 'glass-light border-2 border-quantum-blue neon-border' 
                    : 'glass border border-quantum-blue/10 hover:border-quantum-blue/30'}
                `}
              >
                <div className="text-xs font-mono text-photon-white/60 uppercase mb-1">
                  {key}
                </div>
                <div className="text-2xl font-bold font-display text-photon-white">
                  {data.value}
                </div>
                <div className="flex items-center justify-between mt-2">
                  <span className={`text-xs font-mono ${
                    data.change.startsWith('+') ? 'text-success-aurora' : 'text-alert-solar'
                  }`}>
                    {data.change}
                  </span>
                  <span className="text-xs font-mono text-quantum-blue">
                    {data.status}
                  </span>
                </div>
                {selectedMetric === key && (
                  <motion.div
                    layoutId="metric-indicator"
                    className="absolute inset-0 rounded-xl border-2 border-quantum-blue pointer-events-none"
                    style={{ boxShadow: '0 0 20px rgba(0, 212, 255, 0.5)' }}
                  />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Main Visualization Area */}
        <div className="grid grid-cols-2 gap-6">
          {/* 3D Azure Resource Map */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass rounded-2xl p-6 border border-quantum-blue/20"
          >
            <h2 className="text-xl font-display font-bold text-quantum-blue mb-4">
              AZURE RESOURCE MAP
            </h2>
            <div className="h-96 relative rounded-xl overflow-hidden bg-dark-matter/50">
              <AzureResourceMap />
              <div className="absolute top-4 right-4 glass px-3 py-2 rounded-lg">
                <p className="text-xs font-mono text-photon-white/60">RESOURCES</p>
                <p className="text-lg font-bold text-quantum-blue">2,847</p>
              </div>
            </div>
          </motion.div>

          {/* Correlation Constellation */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-2xl p-6 border border-quantum-blue/20"
          >
            <h2 className="text-xl font-display font-bold text-neural-purple mb-4">
              CORRELATION CONSTELLATION
            </h2>
            <div className="h-96 relative rounded-xl overflow-hidden bg-dark-matter/50">
              <CorrelationConstellation />
              <div className="absolute top-4 right-4 glass px-3 py-2 rounded-lg">
                <p className="text-xs font-mono text-photon-white/60">PATTERNS</p>
                <p className="text-lg font-bold text-neural-purple">892</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Bottom Visualizations */}
        <div className="grid grid-cols-3 gap-6">
          {/* Compliance Grid */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="glass rounded-2xl p-6 border border-quantum-blue/20"
          >
            <h2 className="text-lg font-display font-bold text-plasma-green mb-4">
              QUANTUM COMPLIANCE
            </h2>
            <div className="h-64 relative rounded-xl overflow-hidden bg-dark-matter/50">
              <ComplianceGrid />
            </div>
          </motion.div>

          {/* Threat Radar */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="glass rounded-2xl p-6 border border-quantum-blue/20"
          >
            <h2 className="text-lg font-display font-bold text-alert-nova mb-4">
              THREAT RADAR
            </h2>
            <div className="h-64 relative rounded-xl overflow-hidden bg-dark-matter/50">
              <ThreatRadar />
            </div>
          </motion.div>

          {/* Predictive Timeline */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="glass rounded-2xl p-6 border border-quantum-blue/20"
          >
            <h2 className="text-lg font-display font-bold text-quantum-cyan mb-4">
              PREDICTIVE TIMELINE
            </h2>
            <div className="h-64 relative">
              <div className="space-y-3">
                {[
                  { time: '+2h', event: 'Cost optimization window', confidence: 92 },
                  { time: '+6h', event: 'Security patch required', confidence: 87 },
                  { time: '+1d', event: 'Compliance drift predicted', confidence: 76 },
                  { time: '+3d', event: 'Resource scaling needed', confidence: 68 },
                ].map((prediction, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.8 + index * 0.1 }}
                    className="flex items-center justify-between p-3 glass-light rounded-lg hover:glass transition-all duration-300"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-mono text-quantum-cyan">
                        {prediction.time}
                      </span>
                      <span className="text-sm text-photon-white">
                        {prediction.event}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 h-1 bg-dark-matter rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${prediction.confidence}%` }}
                          transition={{ duration: 1, delay: 1 + index * 0.1 }}
                          className="h-full bg-gradient-to-r from-quantum-cyan to-quantum-blue"
                        />
                      </div>
                      <span className="text-xs font-mono text-photon-white/60">
                        {prediction.confidence}%
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Real-time Data Stream */}
        {realTimeData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-2xl p-4 border border-quantum-blue/20"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <p className="text-xs font-mono text-photon-white/60">CPU</p>
                  <p className="text-lg font-bold text-quantum-blue">
                    {realTimeData.cpuUsage.toFixed(1)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs font-mono text-photon-white/60">MEMORY</p>
                  <p className="text-lg font-bold text-neural-purple">
                    {realTimeData.memoryUsage.toFixed(1)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs font-mono text-photon-white/60">NETWORK</p>
                  <p className="text-lg font-bold text-plasma-green">
                    {realTimeData.networkTraffic.toFixed(0)} MB/s
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs font-mono text-photon-white/60">ALERTS</p>
                  <p className="text-lg font-bold text-alert-solar">
                    {realTimeData.activeAlerts}
                  </p>
                </div>
              </div>
              <div className="text-xs font-mono text-photon-white/40">
                {new Date(realTimeData.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </motion.div>
        )}
      </div>

      <style jsx>{`
        .glass {
          background: rgba(10, 14, 39, 0.6);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid rgba(0, 212, 255, 0.2);
        }

        .glass-light {
          background: rgba(240, 249, 255, 0.05);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
        }

        .neon-border {
          box-shadow: 
            0 0 10px rgba(0, 212, 255, 0.5),
            inset 0 0 10px rgba(0, 212, 255, 0.2);
        }

        .pulse-quantum {
          animation: pulse-quantum 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        @keyframes pulse-quantum {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        .text-quantum-blue { color: #00D4FF; }
        .text-neural-purple { color: #8B5CF6; }
        .text-plasma-green { color: #10F4B1; }
        .text-quantum-cyan { color: #00F5FF; }
        .text-photon-white { color: #F0F9FF; }
        .text-success-aurora { color: #00FF88; }
        .text-alert-solar { color: #FFB800; }
        .text-alert-nova { color: #FF0040; }
        .bg-dark-matter { background: #0A0E27; }
        .bg-success-aurora { background: #00FF88; }
        .border-quantum-blue { border-color: #00D4FF; }
      `}</style>
    </QuantumLayout>
  );
}