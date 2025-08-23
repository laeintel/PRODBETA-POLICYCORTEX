'use client';

import React, { useEffect, useState } from 'react';
import { Activity, Zap, Server, Gauge } from 'lucide-react';

interface PerformanceMetrics {
  fps: number;
  memory: number;
  loadTime: number;
  apiLatency: number;
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    memory: 0,
    loadTime: 0,
    apiLatency: 0
  });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Only show in development
    if (process.env.NODE_ENV !== 'development') return;

    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFPS = () => {
      const currentTime = performance.now();
      frameCount++;

      if (currentTime >= lastTime + 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        
        setMetrics(prev => ({
          ...prev,
          fps,
          memory: (performance as any).memory 
            ? Math.round((performance as any).memory.usedJSHeapSize / 1048576)
            : 0,
          loadTime: Math.round(performance.timing.loadEventEnd - performance.timing.navigationStart)
        }));

        frameCount = 0;
        lastTime = currentTime;
      }

      animationId = requestAnimationFrame(measureFPS);
    };

    animationId = requestAnimationFrame(measureFPS);

    // Measure API latency
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const startTime = performance.now();
      const response = await originalFetch(...args);
      const latency = Math.round(performance.now() - startTime);
      
      setMetrics(prev => ({
        ...prev,
        apiLatency: latency
      }));

      return response;
    };

    return () => {
      cancelAnimationFrame(animationId);
      window.fetch = originalFetch;
    };
  }, []);

  if (process.env.NODE_ENV !== 'development') return null;

  const getFPSColor = (fps: number) => {
    if (fps >= 50) return 'text-green-500';
    if (fps >= 30) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getLatencyColor = (latency: number) => {
    if (latency <= 100) return 'text-green-500';
    if (latency <= 300) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <>
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="fixed bottom-4 right-4 z-50 p-2 bg-gray-900 dark:bg-gray-800 text-white rounded-full shadow-lg hover:scale-110 transition-transform"
        aria-label="Toggle performance monitor"
      >
        <Activity className="w-5 h-5" />
      </button>

      {isVisible && (
        <div className="fixed bottom-16 right-4 z-50 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 min-w-[200px]">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Performance Monitor
          </h3>

          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                <Gauge className="w-3 h-3" />
                FPS
              </span>
              <span className={`font-mono ${getFPSColor(metrics.fps)}`}>
                {metrics.fps}
              </span>
            </div>

            {metrics.memory > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                  <Server className="w-3 h-3" />
                  Memory
                </span>
                <span className="font-mono text-gray-900 dark:text-white">
                  {metrics.memory} MB
                </span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Load Time
              </span>
              <span className="font-mono text-gray-900 dark:text-white">
                {metrics.loadTime} ms
              </span>
            </div>

            {metrics.apiLatency > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                  <Activity className="w-3 h-3" />
                  API Latency
                </span>
                <span className={`font-mono ${getLatencyColor(metrics.apiLatency)}`}>
                  {metrics.apiLatency} ms
                </span>
              </div>
            )}
          </div>

          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Dev mode only
            </div>
          </div>
        </div>
      )}
    </>
  );
}