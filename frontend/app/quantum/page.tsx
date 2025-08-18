'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function QuantumHome() {
  const features = [
    {
      title: 'Neural Command Center',
      description: 'Real-time 3D visualization of your Azure infrastructure',
      icon: '⚡',
      link: '/quantum/dashboard',
      gradient: 'from-quantum-blue to-neural-purple',
    },
    {
      title: 'AI Conversation Nexus',
      description: 'Natural language interface with visual knowledge graphs',
      icon: '🧠',
      link: '/quantum/ai',
      gradient: 'from-neural-purple to-plasma-green',
    },
    {
      title: 'Resource Explorer',
      description: 'Navigate resources in 3D spatial environment',
      icon: '🌌',
      link: '/quantum/resources',
      gradient: 'from-plasma-green to-quantum-cyan',
    },
    {
      title: 'Security Operations',
      description: '360° threat detection and compliance monitoring',
      icon: '🛡️',
      link: '/quantum/security',
      gradient: 'from-alert-nova to-alert-solar',
    },
  ];

  return (
    <div className="min-h-screen bg-dark-void overflow-hidden relative">
      {/* Background effects */}
      <div className="fixed inset-0 grid-background opacity-20" />
      <div className="scanner-line" />
      
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 px-8 py-20"
      >
        <div className="max-w-7xl mx-auto text-center">
          <motion.h1
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="text-6xl font-display font-bold mb-4"
          >
            <span className="bg-gradient-to-r from-quantum-blue via-neural-purple to-plasma-green bg-clip-text text-transparent">
              POLICYCORTEX 2090
            </span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xl text-photon-white/60 font-mono mb-12"
          >
            QUANTUM GOVERNANCE INTERFACE • NEURAL AI SYSTEM • AZURE COMMAND CENTER
          </motion.p>
          
          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                whileHover={{ scale: 1.05, y: -10 }}
                className="group"
              >
                <Link href={feature.link}>
                  <div className="glass rounded-2xl p-6 border border-quantum-blue/20 hover:border-quantum-blue/50 transition-all duration-300 h-full">
                    <div className="text-4xl mb-4">{feature.icon}</div>
                    <h3 className={`text-lg font-display font-bold mb-2 bg-gradient-to-r ${feature.gradient} bg-clip-text text-transparent`}>
                      {feature.title}
                    </h3>
                    <p className="text-sm text-photon-white/60">
                      {feature.description}
                    </p>
                    <div className="mt-4 flex items-center justify-center">
                      <span className="text-xs font-mono text-quantum-blue group-hover:text-quantum-cyan transition-colors">
                        ENTER →
                      </span>
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
          
          {/* Status Bar */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-16 flex justify-center items-center space-x-8"
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-aurora rounded-full pulse-quantum" />
              <span className="text-xs font-mono text-photon-white/60">SYSTEM ONLINE</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-quantum-blue rounded-full pulse-quantum" />
              <span className="text-xs font-mono text-photon-white/60">AI ACTIVE</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-neural-purple rounded-full pulse-quantum" />
              <span className="text-xs font-mono text-photon-white/60">QUANTUM SYNC</span>
            </div>
          </motion.div>
        </div>
      </motion.div>

      <style jsx>{`
        .bg-dark-void {
          background: #000511;
        }
        
        .glass {
          background: rgba(10, 14, 39, 0.6);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
        }
        
        .grid-background {
          background-image: 
            linear-gradient(rgba(0, 212, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.1) 1px, transparent 1px);
          background-size: 50px 50px;
          animation: grid-move 20s linear infinite;
        }
        
        @keyframes grid-move {
          0% { background-position: 0 0; }
          100% { background-position: 50px 50px; }
        }
        
        .scanner-line {
          position: fixed;
          width: 100%;
          height: 2px;
          background: linear-gradient(90deg, transparent, #00D4FF, transparent);
          animation: scan 3s linear infinite;
        }
        
        @keyframes scan {
          0% { transform: translateY(-100vh); }
          100% { transform: translateY(100vh); }
        }
        
        .pulse-quantum {
          animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        .text-quantum-blue { color: #00D4FF; }
        .text-neural-purple { color: #8B5CF6; }
        .text-plasma-green { color: #10F4B1; }
        .text-quantum-cyan { color: #00F5FF; }
        .text-photon-white { color: #F0F9FF; }
        .text-success-aurora { color: #00FF88; }
        .text-alert-nova { color: #FF0040; }
        .text-alert-solar { color: #FFB800; }
        .bg-success-aurora { background: #00FF88; }
        .bg-quantum-blue { background: #00D4FF; }
        .bg-neural-purple { background: #8B5CF6; }
        .border-quantum-blue { border-color: #00D4FF; }
        
        .from-quantum-blue { --tw-gradient-from: #00D4FF; }
        .to-neural-purple { --tw-gradient-to: #8B5CF6; }
        .from-neural-purple { --tw-gradient-from: #8B5CF6; }
        .to-plasma-green { --tw-gradient-to: #10F4B1; }
        .from-plasma-green { --tw-gradient-from: #10F4B1; }
        .to-quantum-cyan { --tw-gradient-to: #00F5FF; }
        .from-alert-nova { --tw-gradient-from: #FF0040; }
        .to-alert-solar { --tw-gradient-to: #FFB800; }
        
        .font-display {
          font-family: 'Orbitron', 'Space Grotesk', sans-serif;
        }
        
        .font-mono {
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
      `}</style>
    </div>
  );
}