'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Gauge, TrendingUp, Activity, Zap, Clock, BarChart3, AlertTriangle, CheckCircle } from 'lucide-react'
import { Line, Bar } from 'react-chartjs-2'
import AppLayout from '../../components/AppLayout'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

export default function PerformancePage() {
  const [performanceData, setPerformanceData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPerformanceData()
  }, [])

  const fetchPerformanceData = async () => {
    try {
      const response = await fetch('/api/v1/performance')
      const data = await response.json()
      setPerformanceData(data)
    } catch (error) {
      console.error('Failed to fetch performance data:', error)
    } finally {
      setLoading(false)
    }
  }

  const cpuChartData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    datasets: [
      {
        label: 'CPU Usage %',
        data: [45, 52, 68, 75, 62, 58],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  }

  const memoryChartData = {
    labels: ['VM1', 'VM2', 'VM3', 'VM4', 'VM5'],
    datasets: [
      {
        label: 'Memory Usage GB',
        data: [12, 19, 8, 15, 22],
        backgroundColor: 'rgba(34, 197, 94, 0.8)'
      }
    ]
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
          <Gauge className="w-8 h-8 mr-3 text-purple-500" />
          Performance Monitoring
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Real-time performance metrics and optimization insights
        </p>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8 text-blue-500" />
            <span className="text-sm bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-1 rounded-full">
              Normal
            </span>
          </div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Avg Response Time</h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">124ms</div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">-12% from last hour</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Zap className="w-8 h-8 text-yellow-500" />
            <span className="text-sm bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded-full">
              Warning
            </span>
          </div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">CPU Utilization</h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">78%</div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">+5% from baseline</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-green-500" />
            <CheckCircle className="w-5 h-5 text-green-500" />
          </div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Throughput</h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">2.4K</div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">requests/sec</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <Clock className="w-8 h-8 text-purple-500" />
            <span className="text-sm text-gray-500">99.98%</span>
          </div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Uptime</h3>
          <div className="text-3xl font-bold text-gray-900 dark:text-white">30d</div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">Since last incident</p>
        </motion.div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">CPU Usage Trend</h2>
          <Line data={cpuChartData} options={{
            responsive: true,
            plugins: {
              legend: { display: false }
            },
            scales: {
              y: {
                beginAtZero: true,
                max: 100
              }
            }
          }} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Memory Usage by VM</h2>
          <Bar data={memoryChartData} options={{
            responsive: true,
            plugins: {
              legend: { display: false }
            }
          }} />
        </motion.div>
      </div>

      {/* Performance Recommendations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Optimization Recommendations
        </h2>
        <div className="space-y-4">
          {[
            {
              title: 'Scale Up VM Size',
              description: 'VM "prod-api-01" is consistently at 90% CPU. Consider upgrading to D4s_v3.',
              impact: 'High',
              savings: '$120/month'
            },
            {
              title: 'Enable Auto-scaling',
              description: 'Configure auto-scaling for App Service to handle traffic spikes.',
              impact: 'Medium',
              savings: '$80/month'
            },
            {
              title: 'Optimize Database Queries',
              description: 'Slow queries detected in SQL Database. Add indexes to improve performance.',
              impact: 'High',
              savings: 'N/A'
            }
          ].map((rec, idx) => (
            <div key={idx} className="flex items-start justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 dark:text-white">{rec.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{rec.description}</p>
                <div className="flex items-center space-x-4 mt-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    rec.impact === 'High' ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' :
                    'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
                  }`}>
                    {rec.impact} Impact
                  </span>
                  {rec.savings !== 'N/A' && (
                    <span className="text-xs text-green-600 dark:text-green-400">
                      Save {rec.savings}
                    </span>
                  )}
                </div>
              </div>
              <button className="ml-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                Apply
              </button>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
    </AppLayout>
  )
}