'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, TrendingDown, AlertTriangle, DollarSign, 
  Cloud, Activity, PieChart, BarChart3, Target, Zap,
  AlertCircle, CheckCircle, XCircle, ArrowUpRight, ArrowDownRight
} from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart as RePieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar
} from 'recharts';

// Real-time cost tracking data
const realtimeCostData = [
  { time: '00:00', aws: 1250, azure: 980, gcp: 450, total: 2680 },
  { time: '04:00', aws: 1180, azure: 920, gcp: 430, total: 2530 },
  { time: '08:00', aws: 1450, azure: 1100, gcp: 520, total: 3070 },
  { time: '12:00', aws: 1680, azure: 1250, gcp: 580, total: 3510 },
  { time: '16:00', aws: 1820, azure: 1380, gcp: 620, total: 3820 },
  { time: '20:00', aws: 1590, azure: 1200, gcp: 550, total: 3340 },
  { time: 'Now', aws: 1720, azure: 1320, gcp: 590, total: 3630 },
];

// Cost by service breakdown
const serviceBreakdown = [
  { service: 'Compute', cost: 45000, percentage: 35, trend: 'up' },
  { service: 'Storage', cost: 28000, percentage: 22, trend: 'stable' },
  { service: 'Networking', cost: 15000, percentage: 12, trend: 'up' },
  { service: 'Databases', cost: 18000, percentage: 14, trend: 'down' },
  { service: 'AI/ML', cost: 12000, percentage: 9, trend: 'up' },
  { service: 'Other', cost: 10000, percentage: 8, trend: 'stable' },
];

// Waste detection metrics
const wasteMetrics = {
  idleResources: { count: 234, savings: 7332, percentage: 18 },
  overProvisioned: { count: 156, savings: 5421, percentage: 14 },
  untagged: { count: 892, cost: 12453, percentage: 8 },
  orphaned: { count: 67, savings: 2156, percentage: 5 },
};

// Cost anomalies
const anomalies = [
  { id: 1, service: 'AWS EC2', spike: 287, amount: 3450, time: '2 hours ago', severity: 'high' },
  { id: 2, service: 'Azure Storage', spike: 145, amount: 1230, time: '5 hours ago', severity: 'medium' },
  { id: 3, service: 'GCP BigQuery', spike: 92, amount: 890, time: '1 day ago', severity: 'low' },
];

// Forecast data
const forecastData = [
  { month: 'Jan', actual: 120000, forecast: 118000 },
  { month: 'Feb', actual: 125000, forecast: 123000 },
  { month: 'Mar', actual: 135000, forecast: 132000 },
  { month: 'Apr', actual: 142000, forecast: 140000 },
  { month: 'May', actual: 145832, forecast: 144000 },
  { month: 'Jun', actual: null, forecast: 148000 },
  { month: 'Jul', actual: null, forecast: 152000 },
];

export default function CostAnalyticsPage() {
  const [selectedPeriod, setSelectedPeriod] = useState('24h');
  const [refreshRate, setRefreshRate] = useState(5000);
  const [currentSpend, setCurrentSpend] = useState(145832);
  const [spendRate, setSpendRate] = useState(243.05);

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate real-time cost updates
      setCurrentSpend(prev => prev + Math.random() * 10);
      setSpendRate(240 + Math.random() * 10);
    }, refreshRate);

    return () => clearInterval(interval);
  }, [refreshRate]);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Cloud Cost Analytics</h1>
          <p className="text-gray-600 mt-1">Real-time cost tracking, forecasting, and optimization</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setSelectedPeriod('24h')}>24 Hours</Button>
          <Button variant="outline" onClick={() => setSelectedPeriod('7d')}>7 Days</Button>
          <Button variant="outline" onClick={() => setSelectedPeriod('30d')}>30 Days</Button>
          <Button className="bg-blue-600 hover:bg-blue-700">
            <Zap className="w-4 h-4 mr-2" />
            Optimize Now
          </Button>
        </div>
      </div>

      {/* Critical Alerts */}
      <Alert className="border-red-200 bg-red-50">
        <AlertCircle className="h-4 w-4 text-red-600" />
        <AlertDescription className="text-red-800">
          <strong>Cost Anomaly Detected:</strong> AWS EC2 costs spiked 287% in the last 2 hours. 
          Immediate action recommended to prevent budget overrun.
        </AlertDescription>
      </Alert>

      {/* Real-time Metrics Row */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Current Month Spend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${currentSpend.toLocaleString()}</div>
            <div className="flex items-center text-sm text-red-600 mt-1">
              <TrendingUp className="w-4 h-4 mr-1" />
              12% over budget
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Burn Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${spendRate.toFixed(2)}/hour</div>
            <div className="flex items-center text-sm text-green-600 mt-1">
              <TrendingDown className="w-4 h-4 mr-1" />
              5% below average
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Projected Monthly</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$152,000</div>
            <div className="flex items-center text-sm text-orange-600 mt-1">
              <AlertTriangle className="w-4 h-4 mr-1" />
              $12k over budget
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Potential Savings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">$17,065</div>
            <div className="text-sm text-gray-600 mt-1">
              From 457 optimizations
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-2 gap-6">
        {/* Real-time Spend Tracking */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Real-time Multi-Cloud Spend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={realtimeCostData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="aws" stackId="1" stroke="#FF9900" fill="#FF9900" />
                <Area type="monotone" dataKey="azure" stackId="1" stroke="#0078D4" fill="#0078D4" />
                <Area type="monotone" dataKey="gcp" stackId="1" stroke="#4285F4" fill="#4285F4" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cost Forecast */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="w-5 h-5 mr-2" />
              Cost Forecast & Prediction
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} />
                <Line type="monotone" dataKey="forecast" stroke="#3B82F6" strokeWidth={2} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Service Breakdown and Waste Detection */}
      <div className="grid grid-cols-3 gap-6">
        {/* Service Cost Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <PieChart className="w-5 h-5 mr-2" />
              Cost by Service
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <RePieChart>
                <Pie
                  data={serviceBreakdown}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ service, percentage }) => `${service}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="cost"
                >
                  {serviceBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </RePieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Waste Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-orange-500" />
              Waste Detection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-orange-50 rounded">
              <div>
                <div className="font-medium">Idle Resources</div>
                <div className="text-sm text-gray-600">{wasteMetrics.idleResources.count} resources</div>
              </div>
              <div className="text-right">
                <div className="font-bold text-orange-600">${wasteMetrics.idleResources.savings}</div>
                <div className="text-xs text-gray-500">potential savings</div>
              </div>
            </div>
            
            <div className="flex justify-between items-center p-3 bg-yellow-50 rounded">
              <div>
                <div className="font-medium">Over-provisioned</div>
                <div className="text-sm text-gray-600">{wasteMetrics.overProvisioned.count} instances</div>
              </div>
              <div className="text-right">
                <div className="font-bold text-yellow-600">${wasteMetrics.overProvisioned.savings}</div>
                <div className="text-xs text-gray-500">rightsizing savings</div>
              </div>
            </div>

            <div className="flex justify-between items-center p-3 bg-red-50 rounded">
              <div>
                <div className="font-medium">Untagged Resources</div>
                <div className="text-sm text-gray-600">{wasteMetrics.untagged.count} resources</div>
              </div>
              <div className="text-right">
                <div className="font-bold text-red-600">${wasteMetrics.untagged.cost}</div>
                <div className="text-xs text-gray-500">unallocated cost</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Cost Anomalies */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 text-red-500" />
              Cost Anomalies
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {anomalies.map(anomaly => (
              <div key={anomaly.id} className="flex items-center justify-between p-2 border rounded">
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    anomaly.severity === 'high' ? 'bg-red-500' :
                    anomaly.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                  }`} />
                  <div>
                    <div className="font-medium text-sm">{anomaly.service}</div>
                    <div className="text-xs text-gray-500">{anomaly.time}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="flex items-center text-red-600 font-bold">
                    <ArrowUpRight className="w-4 h-4" />
                    {anomaly.spike}%
                  </div>
                  <div className="text-xs text-gray-500">${anomaly.amount}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Optimization Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Zap className="w-5 h-5 mr-2 text-green-500" />
            AI-Powered Optimization Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Reserved Instances</span>
                <span className="text-green-600 font-bold">$4,250/mo</span>
              </div>
              <p className="text-sm text-gray-600">Convert 42 on-demand instances to RIs for 35% savings</p>
              <Button size="sm" className="mt-2 bg-green-600 hover:bg-green-700">Apply</Button>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Auto-scaling</span>
                <span className="text-blue-600 font-bold">$2,180/mo</span>
              </div>
              <p className="text-sm text-gray-600">Enable auto-scaling for 18 workloads to reduce idle time</p>
              <Button size="sm" className="mt-2 bg-blue-600 hover:bg-blue-700">Configure</Button>
            </div>

            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Storage Tiering</span>
                <span className="text-purple-600 font-bold">$1,890/mo</span>
              </div>
              <p className="text-sm text-gray-600">Move 2.5TB to archive tier for infrequently accessed data</p>
              <Button size="sm" className="mt-2 bg-purple-600 hover:bg-purple-700">Migrate</Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}