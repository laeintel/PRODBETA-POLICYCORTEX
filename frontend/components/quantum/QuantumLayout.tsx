'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';

const Canvas = dynamic(() => import('@react-three/fiber').then(mod => mod.Canvas), { ssr: false });
const Stars = dynamic(() => import('@react-three/drei').then(mod => mod.Stars), { ssr: false });
const OrbitControls = dynamic(() => import('@react-three/drei').then(mod => mod.OrbitControls), { ssr: false });
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface QuantumLayoutProps {
  children: React.ReactNode;
}

const QuantumLayout: React.FC<QuantumLayoutProps> = ({ children }) => {
  const pathname = usePathname();
  const [loading, setLoading] = useState(true);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const loadTimer = setTimeout(() => setLoading(false), 1500);
    return () => clearTimeout(loadTimer);
  }, []);

  const navItems = [
    { path: '/quantum/dashboard', label: 'COMMAND CENTER', icon: '⚡' },
    { path: '/quantum/ai', label: 'AI NEXUS', icon: '🧠' },
    { path: '/quantum/resources', label: 'RESOURCE MAP', icon: '🌌' },
    { path: '/quantum/security', label: 'SECURITY OPS', icon: '🛡️' },
    { path: '/quantum/analytics', label: 'ANALYTICS', icon: '📊' },
    { path: '/quantum/patents', label: 'TECH GALLERY', icon: '🚀' },
  ];

  return (
    <div className="quantum-container">
      {/* Background 3D Scene */}
      <div className="fixed inset-0 z-0">
        {!loading && Canvas && Stars && OrbitControls && (
          <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
          </Canvas>
        )}
      </div>

      {/* Grid Overlay */}
      <div className="fixed inset-0 z-1 grid-background opacity-20" />
      
      {/* Scanner Line */}
      <div className="scanner-line z-2" />

      {/* Loading Screen */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-void-black"
          >
            <div className="text-center">
              <div className="quantum-spinner mx-auto mb-4" />
              <h1 className="text-2xl font-mono text-quantum-blue neon-glow">
                INITIALIZING QUANTUM CORE
              </h1>
              <div className="mt-2 text-sm text-photon-white/60 font-mono">
                {Array.from({ length: 3 }, (_, i) => (
                  <motion.span
                    key={i}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 1.5, delay: i * 0.2, repeat: Infinity }}
                  >
                    .
                  </motion.span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top Bar */}
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: 'spring', stiffness: 100 }}
        className="fixed top-0 left-0 right-0 z-40 h-20 glass-dark border-b border-quantum-blue/20"
      >
        <div className="h-full px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-4">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              className="w-12 h-12 rounded-full bg-gradient-to-r from-quantum-blue to-neural-purple flex items-center justify-center"
            >
              <span className="text-2xl font-bold text-white">P</span>
            </motion.div>
            <div>
              <h1 className="text-xl font-display font-bold text-quantum-blue">
                POLICYCORTEX
              </h1>
              <p className="text-xs text-photon-white/60 font-mono">
                QUANTUM GOVERNANCE v2090
              </p>
            </div>
          </div>

          {/* Center Status */}
          <div className="flex items-center space-x-8">
            <div className="text-center">
              <p className="text-xs text-photon-white/60 font-mono">SYSTEM STATUS</p>
              <p className="text-sm font-bold text-success-aurora">OPTIMAL</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-photon-white/60 font-mono">QUANTUM TIME</p>
              <p className="text-sm font-mono text-quantum-blue">
                {time.toLocaleTimeString('en-US', { hour12: false })}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-photon-white/60 font-mono">RESOURCES</p>
              <p className="text-sm font-bold text-plasma-green">2,847</p>
            </div>
          </div>

          {/* User Section */}
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm font-medium text-photon-white">Commander</p>
              <p className="text-xs text-photon-white/60">Level 99</p>
            </div>
            <div className="w-10 h-10 rounded-full bg-gradient-to-r from-neural-purple to-plasma-green p-[2px]">
              <div className="w-full h-full rounded-full bg-dark-matter flex items-center justify-center">
                <span className="text-sm font-bold text-photon-white">AI</span>
              </div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Navigation Sidebar */}
      <motion.nav
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        transition={{ type: 'spring', stiffness: 100 }}
        className="fixed left-0 top-20 bottom-0 z-30 w-64 glass-dark border-r border-quantum-blue/20"
      >
        <div className="p-4 space-y-2">
          {navItems.map((item, index) => (
            <motion.div
              key={item.path}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link href={item.path}>
                <motion.div
                  whileHover={{ x: 10, backgroundColor: 'rgba(0, 212, 255, 0.1)' }}
                  whileTap={{ scale: 0.95 }}
                  className={`
                    relative px-4 py-3 rounded-lg cursor-pointer transition-all duration-300
                    ${pathname === item.path ? 'glass border-quantum-blue/50 neon-border' : 'hover:glass'}
                  `}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{item.icon}</span>
                    <div>
                      <p className="text-sm font-display font-medium text-photon-white">
                        {item.label}
                      </p>
                      {pathname === item.path && (
                        <p className="text-xs text-quantum-blue font-mono">ACTIVE</p>
                      )}
                    </div>
                  </div>
                  {pathname === item.path && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-quantum-blue to-neural-purple rounded-full"
                    />
                  )}
                </motion.div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* System Metrics */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-quantum-blue/20">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-photon-white/60">CPU USAGE</span>
                <span className="text-quantum-blue">47%</span>
              </div>
              <div className="h-1 bg-dark-matter rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '47%' }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-gradient-to-r from-quantum-blue to-neural-purple"
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-photon-white/60">QUANTUM SYNC</span>
                <span className="text-plasma-green">98%</span>
              </div>
              <div className="h-1 bg-dark-matter rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '98%' }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-gradient-to-r from-plasma-green to-quantum-blue"
                />
              </div>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Main Content Area */}
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="ml-64 mt-20 p-6 relative z-20"
      >
        <div className="relative">
          {children}
        </div>
      </motion.main>

      {/* Floating Action Button */}
      <motion.button
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 200, delay: 1 }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        className="fixed bottom-8 right-8 z-40 w-16 h-16 rounded-full bg-gradient-to-r from-quantum-blue to-neural-purple shadow-lg neon-border flex items-center justify-center cursor-pointer"
      >
        <span className="text-2xl text-white">🚀</span>
      </motion.button>

      {/* Notification Toast */}
      <motion.div
        initial={{ x: 400 }}
        animate={{ x: 0 }}
        transition={{ type: 'spring', stiffness: 100, delay: 2 }}
        className="fixed top-24 right-4 z-45 glass px-4 py-3 rounded-lg border-l-4 border-quantum-blue"
      >
        <p className="text-xs font-mono text-photon-white/60">SYSTEM NOTIFICATION</p>
        <p className="text-sm text-photon-white">Quantum core synchronized</p>
      </motion.div>

      <style jsx global>{`
        .quantum-container {
          min-height: 100vh;
          background: var(--void-black);
          color: var(--photon-white);
          font-family: var(--font-body);
        }

        .bg-void-black { background: var(--void-black); }
        .bg-dark-matter { background: var(--dark-matter); }
        .text-quantum-blue { color: var(--quantum-blue); }
        .text-neural-purple { color: var(--neural-purple); }
        .text-plasma-green { color: var(--plasma-green); }
        .text-photon-white { color: var(--photon-white); }
        .text-success-aurora { color: var(--success-aurora); }
        .border-quantum-blue { border-color: var(--quantum-blue); }
        
        .font-display { font-family: var(--font-display); }
        .font-mono { font-family: var(--font-mono); }

        /* Import quantum theme */
        @import '/styles/quantum-theme.css';
      `}</style>
    </div>
  );
};

export default QuantumLayout;