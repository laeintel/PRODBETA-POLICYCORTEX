'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Shield,
  Lock,
  Key,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  Eye,
  EyeOff,
  UserCheck,
  UserX,
  FileWarning,
  ShieldAlert,
  ShieldCheck,
  ShieldOff,
  Fingerprint,
  Scan,
  Bug,
  Activity,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Clock,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Settings,
  ChevronRight,
  AlertCircle,
  Zap
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RePieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

interface SecurityThreat {
  id: string;
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  affectedResources: number;
  detectedAt: Date;
  status: 'active' | 'investigating' | 'mitigated' | 'resolved';
  riskScore: number;
}

interface SecurityMetric {
  id: string;
  name: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  status: 'good' | 'warning' | 'critical';
  unit?: string;
}

interface ComplianceCheck {
  framework: string;
  score: number;
  passed: number;
  failed: number;
  total: number;
  lastAudit: Date;
}

interface VulnerabilityData {
  date: string;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

export default function SecurityDashboard() {
  const [loading, setLoading] = useState(true);
  const [threats, setThreats] = useState<SecurityThreat[]>([]);
  const [metrics, setMetrics] = useState<SecurityMetric[]>([]);
  const [complianceData, setComplianceData] = useState<ComplianceCheck[]>([]);
  const [vulnerabilityTrends, setVulnerabilityTrends] = useState<VulnerabilityData[]>([]);
  const [securityScore, setSecurityScore] = useState<number>(0);
  const [attackSurface, setAttackSurface] = useState<any[]>([]);
  const [threatDistribution, setThreatDistribution] = useState<any[]>([]);
  const [selectedView, setSelectedView] = useState<'overview' | 'threats' | 'compliance' | 'vulnerabilities'>('overview');

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock threats
      setThreats([
        {
          id: 'threat-001',
          type: 'Unauthorized Access',
          severity: 'critical',
          title: 'Multiple failed login attempts detected',
          description: 'Detected 150+ failed login attempts from multiple IPs targeting admin accounts',
          affectedResources: 5,
          detectedAt: new Date('2024-01-09T08:30:00'),
          status: 'investigating',
          riskScore: 95
        },
        {
          id: 'threat-002',
          type: 'Data Exfiltration',
          severity: 'high',
          title: 'Unusual data transfer activity',
          description: 'Large volume of data transfer to unknown external IP detected',
          affectedResources: 3,
          detectedAt: new Date('2024-01-09T07:15:00'),
          status: 'active',
          riskScore: 82
        },
        {
          id: 'threat-003',
          type: 'Malware',
          severity: 'medium',
          title: 'Suspicious process execution',
          description: 'Potentially malicious process detected on production server',
          affectedResources: 1,
          detectedAt: new Date('2024-01-09T06:45:00'),
          status: 'mitigated',
          riskScore: 65
        },
        {
          id: 'threat-004',
          type: 'Configuration Drift',
          severity: 'low',
          title: 'Security group rules modified',
          description: 'Unauthorized modification to security group rules detected',
          affectedResources: 8,
          detectedAt: new Date('2024-01-09T05:00:00'),
          status: 'resolved',
          riskScore: 35
        }
      ]);

      // Mock metrics
      setMetrics([
        {
          id: 'security-score',
          name: 'Security Score',
          value: 87,
          change: 3,
          trend: 'up',
          status: 'good',
          unit: '/100'
        },
        {
          id: 'active-threats',
          name: 'Active Threats',
          value: 12,
          change: -25,
          trend: 'down',
          status: 'warning'
        },
        {
          id: 'vulnerabilities',
          name: 'Open Vulnerabilities',
          value: 45,
          change: 15,
          trend: 'up',
          status: 'warning'
        },
        {
          id: 'patch-compliance',
          name: 'Patch Compliance',
          value: 92,
          change: 5,
          trend: 'up',
          status: 'good',
          unit: '%'
        },
        {
          id: 'mfa-adoption',
          name: 'MFA Adoption',
          value: 78,
          change: 12,
          trend: 'up',
          status: 'warning',
          unit: '%'
        },
        {
          id: 'encrypted-data',
          name: 'Data Encryption',
          value: 95,
          change: 2,
          trend: 'stable',
          status: 'good',
          unit: '%'
        }
      ]);

      // Mock compliance data
      setComplianceData([
        {
          framework: 'SOC 2 Type II',
          score: 94,
          passed: 112,
          failed: 7,
          total: 119,
          lastAudit: new Date('2024-01-05')
        },
        {
          framework: 'ISO 27001',
          score: 91,
          passed: 203,
          failed: 20,
          total: 223,
          lastAudit: new Date('2024-01-03')
        },
        {
          framework: 'NIST CSF',
          score: 88,
          passed: 95,
          failed: 13,
          total: 108,
          lastAudit: new Date('2024-01-07')
        },
        {
          framework: 'CIS Controls',
          score: 85,
          passed: 145,
          failed: 26,
          total: 171,
          lastAudit: new Date('2024-01-02')
        },
        {
          framework: 'PCI DSS',
          score: 96,
          passed: 287,
          failed: 12,
          total: 299,
          lastAudit: new Date('2024-01-08')
        }
      ]);

      // Mock vulnerability trends
      setVulnerabilityTrends([
        { date: 'Jan', critical: 5, high: 12, medium: 25, low: 45 },
        { date: 'Feb', critical: 3, high: 10, medium: 22, low: 42 },
        { date: 'Mar', critical: 4, high: 11, medium: 20, low: 40 },
        { date: 'Apr', critical: 2, high: 8, medium: 18, low: 38 },
        { date: 'May', critical: 3, high: 9, medium: 19, low: 35 },
        { date: 'Jun', critical: 1, high: 7, medium: 17, low: 32 },
        { date: 'Jul', critical: 2, high: 8, medium: 16, low: 30 },
        { date: 'Aug', critical: 1, high: 6, medium: 15, low: 28 },
        { date: 'Sep', critical: 2, high: 7, medium: 14, low: 25 },
        { date: 'Oct', critical: 3, high: 8, medium: 15, low: 23 }
      ]);

      // Mock attack surface
      setAttackSurface([
        { area: 'Network Exposure', current: 65, target: 40 },
        { area: 'Application Security', current: 75, target: 85 },
        { area: 'Identity & Access', current: 82, target: 90 },
        { area: 'Data Protection', current: 88, target: 95 },
        { area: 'Endpoint Security', current: 70, target: 80 },
        { area: 'Cloud Security', current: 78, target: 85 }
      ]);

      // Mock threat distribution
      setThreatDistribution([
        { name: 'Malware', value: 25, color: '#EF4444' },
        { name: 'Phishing', value: 20, color: '#F59E0B' },
        { name: 'Unauthorized Access', value: 18, color: '#3B82F6' },
        { name: 'DDoS', value: 15, color: '#8B5CF6' },
        { name: 'Data Breach', value: 12, color: '#10B981' },
        { name: 'Other', value: 10, color: '#6B7280' }
      ]);

      setSecurityScore(87);
      setLoading(false);
    }, 1000);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'high':
        return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Info className="h-5 w-5 text-gray-500" />;
    }
  };

  const getSecurityScoreColor = (score: number) => {
    if (score >= 80) return '#10B981';
    if (score >= 60) return '#F59E0B';
    return '#EF4444';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading security data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <Shield className="h-8 w-8 text-blue-600" />
              Security Dashboard
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Monitor security posture, threats, and compliance status
            </p>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4" />
              Incident Response
            </button>
            <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export Report
            </button>
          </div>
        </div>
      </div>

      {/* Security Score Card */}
      <Card className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
        <CardHeader>
          <CardTitle className="text-2xl">Overall Security Score</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <div className="relative w-32 h-32">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="rgba(255, 255, 255, 0.2)"
                    strokeWidth="12"
                    fill="none"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="#FFFFFF"
                    strokeWidth="12"
                    fill="none"
                    strokeDasharray={`${(securityScore / 100) * 352} 352`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-3xl font-bold">{securityScore}</div>
                    <div className="text-sm opacity-80">/100</div>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-lg font-semibold">Security Health: Good</div>
                <div className="text-sm opacity-90">Last assessment: 2 hours ago</div>
                <div className="flex items-center gap-2 text-sm">
                  <TrendingUp className="h-4 w-4" />
                  <span>+3 points from last week</span>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">12</div>
                <div className="text-sm opacity-80">Active Threats</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">45</div>
                <div className="text-sm opacity-80">Vulnerabilities</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">92%</div>
                <div className="text-sm opacity-80">Patch Compliance</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">4</div>
                <div className="text-sm opacity-80">Critical Issues</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* View Selector */}
      <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg w-fit">
        {(['overview', 'threats', 'compliance', 'vulnerabilities'] as const).map((view) => (
          <button
            key={view}
            onClick={() => setSelectedView(view)}
            className={`px-4 py-2 rounded-md font-medium transition-colors capitalize ${
              selectedView === view
                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {view}
          </button>
        ))}
      </div>

      {selectedView === 'overview' && (
        <>
          {/* Security Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
            {metrics.map((metric) => (
              <Card key={metric.id}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                      {metric.name}
                    </CardTitle>
                    {getStatusIcon(metric.status)}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {metric.value}{metric.unit || ''}
                  </div>
                  <div className={`flex items-center gap-1 text-sm mt-1 ${
                    metric.trend === 'up' && metric.name.includes('Threats') ? 'text-red-600 dark:text-red-400' :
                    metric.trend === 'up' ? 'text-green-600 dark:text-green-400' :
                    metric.trend === 'down' && metric.name.includes('Threats') ? 'text-green-600 dark:text-green-400' :
                    metric.trend === 'down' ? 'text-red-600 dark:text-red-400' :
                    'text-gray-600 dark:text-gray-400'
                  }`}>
                    {metric.trend === 'up' && <TrendingUp className="h-3 w-3" />}
                    {metric.trend === 'down' && <TrendingDown className="h-3 w-3" />}
                    {metric.change > 0 ? '+' : ''}{metric.change}%
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Attack Surface */}
            <Card>
              <CardHeader>
                <CardTitle>Attack Surface Analysis</CardTitle>
                <CardDescription>Current vs target security posture</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={attackSurface}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="area" className="text-xs" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Current"
                      dataKey="current"
                      stroke="#3B82F6"
                      fill="#3B82F6"
                      fillOpacity={0.6}
                    />
                    <Radar
                      name="Target"
                      dataKey="target"
                      stroke="#10B981"
                      fill="#10B981"
                      fillOpacity={0.3}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Threat Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Threat Distribution</CardTitle>
                <CardDescription>Types of security threats detected</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RePieChart>
                    <Pie
                      data={threatDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {threatDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => `${value}%`} />
                    <Legend />
                  </RePieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {selectedView === 'threats' && (
        <Card>
          <CardHeader>
            <CardTitle>Active Security Threats</CardTitle>
            <CardDescription>Real-time threat detection and monitoring</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {threats.map((threat) => (
                <div key={threat.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg ${
                        threat.severity === 'critical' ? 'bg-red-100 dark:bg-red-900/30' :
                        threat.severity === 'high' ? 'bg-orange-100 dark:bg-orange-900/30' :
                        threat.severity === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900/30' :
                        'bg-blue-100 dark:bg-blue-900/30'
                      }`}>
                        <ShieldAlert className={`h-5 w-5 ${
                          threat.severity === 'critical' ? 'text-red-600 dark:text-red-400' :
                          threat.severity === 'high' ? 'text-orange-600 dark:text-orange-400' :
                          threat.severity === 'medium' ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-blue-600 dark:text-blue-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold text-lg">{threat.title}</h4>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(threat.severity)}`}>
                            {threat.severity}
                          </span>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                            threat.status === 'active' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                            threat.status === 'investigating' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                            threat.status === 'mitigated' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                            'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          }`}>
                            {threat.status}
                          </span>
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-2">{threat.description}</p>
                        <div className="flex items-center gap-4 text-sm">
                          <div className="flex items-center gap-1">
                            <Bug className="h-4 w-4 text-gray-400" />
                            <span>Type: {threat.type}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Activity className="h-4 w-4 text-gray-400" />
                            <span>Risk Score: {threat.riskScore}/100</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4 text-gray-400" />
                            <span>{threat.detectedAt.toLocaleTimeString()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                        Affected Resources
                      </div>
                      <div className="text-2xl font-bold">{threat.affectedResources}</div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                      Investigate
                    </button>
                    <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                      View Details
                    </button>
                    <button className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800">
                      Create Ticket
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'compliance' && (
        <Card>
          <CardHeader>
            <CardTitle>Compliance Framework Status</CardTitle>
            <CardDescription>Security compliance across different frameworks</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {complianceData.map((framework) => (
                <div key={framework.framework} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-lg">{framework.framework}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Last audit: {framework.lastAudit.toLocaleDateString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold" style={{ color: getSecurityScoreColor(framework.score) }}>
                        {framework.score}%
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Compliance Score</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex justify-between text-sm mb-1">
                        <span>Progress</span>
                        <span>{framework.passed}/{framework.total} controls</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all"
                          style={{
                            width: `${(framework.passed / framework.total) * 100}%`,
                            backgroundColor: getSecurityScoreColor(framework.score)
                          }}
                        />
                      </div>
                    </div>
                    <div className="flex gap-4 text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-green-600 dark:text-green-400">
                          {framework.passed}
                        </div>
                        <div className="text-gray-600 dark:text-gray-400">Passed</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-red-600 dark:text-red-400">
                          {framework.failed}
                        </div>
                        <div className="text-gray-600 dark:text-gray-400">Failed</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {selectedView === 'vulnerabilities' && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Vulnerability Trends</CardTitle>
              <CardDescription>Monthly vulnerability detection and remediation</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={vulnerabilityTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="critical"
                    stackId="1"
                    stroke="#EF4444"
                    fill="#EF4444"
                    name="Critical"
                  />
                  <Area
                    type="monotone"
                    dataKey="high"
                    stackId="1"
                    stroke="#F59E0B"
                    fill="#F59E0B"
                    name="High"
                  />
                  <Area
                    type="monotone"
                    dataKey="medium"
                    stackId="1"
                    stroke="#FCD34D"
                    fill="#FCD34D"
                    name="Medium"
                  />
                  <Area
                    type="monotone"
                    dataKey="low"
                    stackId="1"
                    stroke="#3B82F6"
                    fill="#3B82F6"
                    name="Low"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="border-red-200 dark:border-red-800">
              <CardHeader>
                <CardTitle className="text-red-600 dark:text-red-400">Critical Vulnerabilities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">3</div>
                <div className="space-y-2">
                  <div className="text-sm">CVE-2024-0001: RCE in web server</div>
                  <div className="text-sm">CVE-2024-0002: SQL injection</div>
                  <div className="text-sm">CVE-2024-0003: Privilege escalation</div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-orange-200 dark:border-orange-800">
              <CardHeader>
                <CardTitle className="text-orange-600 dark:text-orange-400">High Priority</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">8</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Requires immediate attention within 7 days
                </div>
                <button className="mt-3 px-3 py-1 text-sm bg-orange-600 text-white rounded hover:bg-orange-700">
                  View All
                </button>
              </CardContent>
            </Card>

            <Card className="border-blue-200 dark:border-blue-800">
              <CardHeader>
                <CardTitle className="text-blue-600 dark:text-blue-400">Patch Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Available Patches</span>
                    <span className="font-semibold">24</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Applied Today</span>
                    <span className="font-semibold text-green-600 dark:text-green-400">12</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Scheduled</span>
                    <span className="font-semibold">8</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {/* Quick Actions */}
      <Card className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 border-gray-300 dark:border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-600" />
            Security Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Scan className="h-6 w-6 text-blue-600 mb-2" />
              <div className="font-medium">Run Security Scan</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Initiate vulnerability assessment
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Lock className="h-6 w-6 text-green-600 mb-2" />
              <div className="font-medium">Apply Patches</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Deploy security updates
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <FileWarning className="h-6 w-6 text-amber-600 mb-2" />
              <div className="font-medium">Review Policies</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Audit security policies
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <UserCheck className="h-6 w-6 text-purple-600 mb-2" />
              <div className="font-medium">Access Review</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Audit user permissions
              </div>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}