/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2026 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FileCheck,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  Clock,
  Download,
  RefreshCw,
  BarChart3,
  Target,
  Award,
  FileText,
  AlertCircle,
  ChevronRight,
  Search,
  Filter,
  Eye,
  Edit,
  Trash2,
  MoreVertical,
  Calendar,
  Globe,
  Server,
  Database,
  Cloud,
  Terminal,
  Bell,
  Zap,
  ChevronDown,
  X,
  Check,
  Info,
  ExternalLink,
  History,
  BookOpen,
  Clipboard,
  FileX,
  Plus,
  Settings,
  Users,
  Building,
  MapPin,
  Activity,
  PieChart,
  TrendingDown,
  Gauge
} from 'lucide-react'
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2'
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
  ArcElement,
  RadialLinearScale
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
  ArcElement,
  RadialLinearScale
)

interface ComplianceFramework {
  id: string
  name: string
  acronym: string
  score: number
  status: 'compliant' | 'non-compliant' | 'partial' | 'pending'
  controls: {
    total: number
    passed: number
    failed: number
    notApplicable: number
    pending: number
  }
  lastAudit: string
  nextAudit: string
  criticalFindings: number
  highFindings: number
  mediumFindings: number
  lowFindings: number
  recommendations: string[]
  auditor: string
  certificationDate?: string
  expirationDate?: string
  industry: string
  scope: string[]
  region: string
  assessmentType: 'internal' | 'external' | 'certification'
}

interface ComplianceControl {
  id: string
  controlId: string
  title: string
  description: string
  framework: string
  category: string
  domain: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'pass' | 'fail' | 'partial' | 'n/a' | 'pending'
  evidence: Evidence[]
  lastChecked: string
  nextCheck: string
  owner: string
  implementationGuidance: string
  testProcedure: string
  remediationPlan?: string
  riskRating: number
  businessImpact: string
  technicalRequirement: string
  automationLevel: 'manual' | 'semi-automated' | 'automated'
}

interface Evidence {
  id: string
  title: string
  type: 'document' | 'screenshot' | 'log' | 'configuration' | 'report'
  url: string
  uploadDate: string
  reviewer: string
  status: 'approved' | 'rejected' | 'pending'
}

interface AuditFinding {
  id: string
  framework: string
  control: string
  title: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'open' | 'in-progress' | 'resolved' | 'accepted-risk'
  discoveredDate: string
  dueDate: string
  assignee: string
  remediationEffort: string
  businessImpact: string
  riskScore: number
}

interface ComplianceMetrics {
  overallScore: number
  frameworksCount: number
  compliantFrameworks: number
  totalControls: number
  passingControls: number
  failingControls: number
  pendingControls: number
  totalFindings: number
  criticalFindings: number
  highFindings: number
  mediumFindings: number
  lowFindings: number
  openFindings: number
  overduefindings: number
  averageRemediationTime: number
  complianceTrend: number
  nextAuditDays: number
  certificationsExpiring: number
}

interface ComplianceReport {
  id: string
  title: string
  framework: string
  type: 'assessment' | 'audit' | 'certification' | 'monitoring'
  status: 'completed' | 'in-progress' | 'scheduled'
  generatedDate: string
  scope: string
  findings: number
  recommendations: number
  author: string
}

export default function ComplianceHubPage() {
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([])
  const [controls, setControls] = useState<ComplianceControl[]>([])
  const [findings, setFindings] = useState<AuditFinding[]>([])
  const [reports, setReports] = useState<ComplianceReport[]>([])
  const [metrics, setMetrics] = useState<ComplianceMetrics | null>(null)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'frameworks' | 'controls' | 'findings' | 'reports' | 'evidence' | 'analytics'>('overview')
  const [selectedFramework, setSelectedFramework] = useState('all')
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedTimeRange, setSelectedTimeRange] = useState('30d')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [realTimeData, setRealTimeData] = useState<any[]>([])

  useEffect(() => {
    loadAllData()
    const interval = autoRefresh ? setInterval(loadRealTimeData, 30000) : null
    return () => { if (interval) clearInterval(interval) }
  }, [autoRefresh])

  const loadAllData = () => {
    setLoading(true)
    setTimeout(() => {
      // Set metrics
      setMetrics({
        overallScore: 91,
        frameworksCount: 8,
        compliantFrameworks: 5,
        totalControls: 1247,
        passingControls: 1089,
        failingControls: 98,
        pendingControls: 60,
        totalFindings: 156,
        criticalFindings: 8,
        highFindings: 23,
        mediumFindings: 67,
        lowFindings: 58,
        openFindings: 89,
        overduefindings: 12,
        averageRemediationTime: 14,
        complianceTrend: 3.2,
        nextAuditDays: 45,
        certificationsExpiring: 2
      })
      
      setFrameworks([
        {
          id: 'fw-001',
          name: 'System and Organization Controls 2',
          acronym: 'SOC 2',
          score: 94,
          status: 'compliant',
          controls: {
            total: 156,
            passed: 147,
            failed: 5,
            notApplicable: 4,
            pending: 0
          },
          lastAudit: '2024-01-15',
          nextAudit: '2024-04-15',
          criticalFindings: 2,
          highFindings: 3,
          mediumFindings: 8,
          lowFindings: 4,
          recommendations: [
            'Implement automated log monitoring',
            'Update incident response procedures',
            'Enhance data encryption at rest'
          ],
          auditor: 'Deloitte & Touche LLP',
          certificationDate: '2024-01-30',
          expirationDate: '2025-01-30',
          industry: 'Technology',
          scope: ['Security', 'Availability', 'Confidentiality'],
          region: 'Global',
          assessmentType: 'external'
        },
        {
          id: 'fw-002',
          name: 'ISO/IEC 27001:2013',
          acronym: 'ISO 27001',
          score: 88,
          status: 'partial',
          controls: {
            total: 114,
            passed: 100,
            failed: 8,
            notApplicable: 6,
            pending: 0
          },
          lastAudit: '2023-12-20',
          nextAudit: '2024-03-20',
          criticalFindings: 3,
          highFindings: 5,
          mediumFindings: 12,
          lowFindings: 8,
          recommendations: [
            'Update risk assessment methodology',
            'Implement continuous monitoring',
            'Review access control policies'
          ],
          auditor: 'BSI Group',
          certificationDate: '2023-06-15',
          expirationDate: '2026-06-15',
          industry: 'Information Technology',
          scope: ['Information Security Management'],
          region: 'EMEA',
          assessmentType: 'certification'
        },
        {
          id: 'fw-003',
          name: 'Health Insurance Portability and Accountability Act',
          acronym: 'HIPAA',
          score: 96,
          status: 'compliant',
          controls: {
            total: 78,
            passed: 75,
            failed: 2,
            notApplicable: 1,
            pending: 0
          },
          lastAudit: '2024-01-05',
          nextAudit: '2024-07-05',
          criticalFindings: 0,
          highFindings: 1,
          mediumFindings: 3,
          lowFindings: 2,
          recommendations: [
            'Update BAA agreements',
            'Enhance PHI encryption'
          ],
          auditor: 'HHS Office for Civil Rights',
          industry: 'Healthcare',
          scope: ['Protected Health Information'],
          region: 'United States',
          assessmentType: 'internal'
        },
        {
          id: 'fw-004',
          name: 'Payment Card Industry Data Security Standard',
          acronym: 'PCI DSS',
          score: 92,
          status: 'compliant',
          controls: {
            total: 248,
            passed: 228,
            failed: 12,
            notApplicable: 8,
            pending: 0
          },
          lastAudit: '2024-01-10',
          nextAudit: '2024-04-10',
          criticalFindings: 1,
          highFindings: 4,
          mediumFindings: 15,
          lowFindings: 12,
          recommendations: [
            'Segment cardholder data environment',
            'Update firewall rules',
            'Implement file integrity monitoring'
          ],
          auditor: 'Trustwave',
          certificationDate: '2024-01-25',
          expirationDate: '2025-01-25',
          industry: 'Financial Services',
          scope: ['Cardholder Data Environment'],
          region: 'Global',
          assessmentType: 'external'
        },
        {
          id: 'fw-005',
          name: 'General Data Protection Regulation',
          acronym: 'GDPR',
          score: 89,
          status: 'partial',
          controls: {
            total: 99,
            passed: 88,
            failed: 7,
            notApplicable: 4,
            pending: 0
          },
          lastAudit: '2023-12-01',
          nextAudit: '2024-05-01',
          criticalFindings: 2,
          highFindings: 3,
          mediumFindings: 9,
          lowFindings: 6,
          recommendations: [
            'Update privacy notices',
            'Implement data retention policies',
            'Enhance consent management'
          ],
          auditor: 'Internal Audit Team',
          industry: 'Technology',
          scope: ['Personal Data Processing'],
          region: 'European Union',
          assessmentType: 'internal'
        },
        {
          id: 'fw-006',
          name: 'NIST Cybersecurity Framework',
          acronym: 'NIST CSF',
          score: 85,
          status: 'partial',
          controls: {
            total: 164,
            passed: 139,
            failed: 15,
            notApplicable: 10,
            pending: 0
          },
          lastAudit: '2024-01-20',
          nextAudit: '2024-04-20',
          criticalFindings: 4,
          highFindings: 8,
          mediumFindings: 18,
          lowFindings: 12,
          recommendations: [
            'Enhance threat intelligence capabilities',
            'Improve incident response automation',
            'Strengthen supply chain risk management'
          ],
          auditor: 'KPMG',
          industry: 'Technology',
          scope: ['Cybersecurity Framework'],
          region: 'United States',
          assessmentType: 'external'
        }
      ])

      setControls([
        {
          id: 'ctrl-001',
          controlId: 'CC6.1',
          title: 'Logical and Physical Access Controls',
          description: 'The entity implements logical and physical access controls to safeguard against unauthorized access to systems and data.',
          framework: 'SOC 2',
          category: 'Common Criteria',
          domain: 'Security',
          severity: 'critical',
          status: 'fail',
          evidence: [
            { id: 'e1', title: 'Access Control Policy', type: 'document', url: '/evidence/ac-policy.pdf', uploadDate: '2024-01-15', reviewer: 'security@company.com', status: 'approved' },
            { id: 'e2', title: 'User Access Review', type: 'report', url: '/evidence/user-access.xlsx', uploadDate: '2024-01-20', reviewer: 'audit@company.com', status: 'pending' }
          ],
          lastChecked: '2024-02-01',
          nextCheck: '2024-03-01',
          owner: 'security-team@company.com',
          implementationGuidance: 'Configure multi-factor authentication for all privileged accounts',
          testProcedure: 'Verify MFA is enabled for admin accounts and test authentication flow',
          remediationPlan: '1. Enable MFA for all admin accounts 2. Update access control policies 3. Conduct user training',
          riskRating: 9.2,
          businessImpact: 'High - Potential for unauthorized access to sensitive systems',
          technicalRequirement: 'Azure AD MFA, RBAC implementation',
          automationLevel: 'semi-automated'
        },
        {
          id: 'ctrl-002',
          controlId: 'A.12.1.1',
          title: 'Documented Operating Procedures',
          description: 'Operating procedures shall be documented and made available to all users who need them.',
          framework: 'ISO 27001',
          category: 'Operations Security',
          domain: 'Operations',
          severity: 'medium',
          status: 'partial',
          evidence: [
            { id: 'e3', title: 'Operating Procedures Manual', type: 'document', url: '/evidence/ops-manual.pdf', uploadDate: '2024-01-10', reviewer: 'ops@company.com', status: 'approved' },
            { id: 'e4', title: 'Procedure Training Records', type: 'report', url: '/evidence/training.xlsx', uploadDate: '2024-01-25', reviewer: 'hr@company.com', status: 'approved' }
          ],
          lastChecked: '2024-01-28',
          nextCheck: '2024-02-28',
          owner: 'operations@company.com',
          implementationGuidance: 'Ensure all operating procedures are documented and accessible to relevant personnel',
          testProcedure: 'Review documentation repository and verify procedures are up-to-date',
          riskRating: 5.5,
          businessImpact: 'Medium - Operational inefficiencies and compliance gaps',
          technicalRequirement: 'Document management system with version control',
          automationLevel: 'manual'
        }
      ])

      setFindings([
        {
          id: 'f-001',
          framework: 'SOC 2',
          control: 'CC6.1',
          title: 'Multi-Factor Authentication Not Enforced for Privileged Accounts',
          description: 'Critical finding: 15 privileged accounts do not have MFA enabled, creating significant security risk.',
          severity: 'critical',
          status: 'open',
          discoveredDate: '2024-02-01',
          dueDate: '2024-02-15',
          assignee: 'security-team@company.com',
          remediationEffort: '8 hours',
          businessImpact: 'High risk of unauthorized access to sensitive systems and data',
          riskScore: 9.5
        },
        {
          id: 'f-002',
          framework: 'ISO 27001',
          control: 'A.12.1.1',
          title: 'Outdated Operating Procedures Documentation',
          description: 'Several operating procedures have not been updated in over 12 months.',
          severity: 'medium',
          status: 'in-progress',
          discoveredDate: '2024-01-28',
          dueDate: '2024-03-15',
          assignee: 'operations@company.com',
          remediationEffort: '40 hours',
          businessImpact: 'Medium risk of operational inefficiencies and compliance gaps',
          riskScore: 6.2
        },
        {
          id: 'f-003',
          framework: 'PCI DSS',
          control: '2.2.4',
          title: 'Default Passwords Not Changed on Network Devices',
          description: 'Several network devices still have default administrative passwords.',
          severity: 'high',
          status: 'open',
          discoveredDate: '2024-02-05',
          dueDate: '2024-02-20',
          assignee: 'network-team@company.com',
          remediationEffort: '4 hours',
          businessImpact: 'High risk of unauthorized network access',
          riskScore: 8.7
        }
      ])

      setReports([
        {
          id: 'r-001',
          title: 'SOC 2 Type II Assessment Report',
          framework: 'SOC 2',
          type: 'audit',
          status: 'completed',
          generatedDate: '2024-01-30',
          scope: 'Security, Availability, Confidentiality',
          findings: 17,
          recommendations: 12,
          author: 'Deloitte & Touche LLP'
        },
        {
          id: 'r-002',
          title: 'ISO 27001 Surveillance Audit',
          framework: 'ISO 27001',
          type: 'certification',
          status: 'in-progress',
          generatedDate: '2024-02-15',
          scope: 'Information Security Management System',
          findings: 14,
          recommendations: 8,
          author: 'BSI Group'
        },
        {
          id: 'r-003',
          title: 'GDPR Compliance Assessment',
          framework: 'GDPR',
          type: 'assessment',
          status: 'completed',
          generatedDate: '2024-01-20',
          scope: 'Data Processing Activities',
          findings: 11,
          recommendations: 15,
          author: 'Internal Audit Team'
        }
      ])

      setRealTimeData(generateRealTimeData())
      setLoading(false)
    }, 1000)
  }

  const loadRealTimeData = () => {
    setRealTimeData(prev => {
      const newData = [...prev, {
        timestamp: new Date(),
        complianceScore: Math.floor(Math.random() * 5) + 88,
        newFindings: Math.floor(Math.random() * 3),
        resolvedFindings: Math.floor(Math.random() * 5)
      }]
      return newData.slice(-20)
    })
  }

  const generateRealTimeData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 3600000),
      complianceScore: Math.floor(Math.random() * 5) + 88,
      newFindings: Math.floor(Math.random() * 3),
      resolvedFindings: Math.floor(Math.random() * 5)
    }))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': case 'pass': case 'approved': case 'completed': case 'resolved': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'non-compliant': case 'fail': case 'rejected': case 'open': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'partial': case 'pending': case 'in-progress': case 'scheduled': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'n/a': case 'accepted-risk': return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-600/20 text-red-400 border-red-500/30'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-400'
    if (score >= 70) return 'text-yellow-400'
    return 'text-red-400'
  }

  const complianceTrendData = {
    labels: realTimeData.map(d => d.timestamp.toLocaleDateString()),
    datasets: [
      {
        label: 'Compliance Score',
        data: realTimeData.map(d => d.complianceScore),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4
      }
    ]
  }

  const frameworkScoreData = {
    labels: frameworks.map(f => f.acronym),
    datasets: [{
      data: frameworks.map(f => f.score),
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(59, 130, 246, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(156, 163, 175, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const findingsDistributionData = {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [{
      data: [
        metrics?.criticalFindings || 0,
        metrics?.highFindings || 0,
        metrics?.mediumFindings || 0,
        metrics?.lowFindings || 0
      ],
      backgroundColor: [
        'rgba(239, 68, 68, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(251, 191, 36, 0.8)',
        'rgba(59, 130, 246, 0.8)'
      ],
      borderWidth: 0
    }]
  }

  const filteredFrameworks = frameworks.filter(framework =>
    selectedFramework === 'all' || framework.id === selectedFramework
  )

  const filteredControls = controls.filter(control =>
    searchTerm === '' ||
    control.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    control.framework.toLowerCase().includes(searchTerm.toLowerCase()) ||
    control.controlId.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredFindings = findings.filter(finding =>
    searchTerm === '' ||
    finding.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    finding.framework.toLowerCase().includes(searchTerm.toLowerCase()) ||
    finding.control.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Compliance Hub...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-950 border-b border-gray-800 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <FileCheck className="w-8 h-8 text-green-500" />
              <div>
                <h1 className="text-2xl font-bold">Compliance Hub</h1>
                <p className="text-sm text-gray-500">Regulatory compliance and audit management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-400">COMPLIANCE MONITORING ACTIVE</span>
              </div>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded ${autoRefresh ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400'}`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </button>
              <select 
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
                <option value="1y">Last Year</option>
              </select>
              <button className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded transition-colors flex items-center space-x-2">
                <Plus className="w-4 h-4" />
                <span>New Assessment</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 flex space-x-6 border-t border-gray-800">
          {['overview', 'frameworks', 'controls', 'findings', 'reports', 'evidence', 'analytics'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-1 border-b-2 transition-colors capitalize ${selectedTab === tab
                  ? 'border-green-500 text-green-500'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      <div className="p-6">
        {selectedTab === 'overview' && metrics && (
          <>
            {/* Overview Metrics */}
            <div className="grid grid-cols-6 gap-4 mb-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Gauge className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Overall Score</span>
                </div>
                <p className={`text-2xl font-bold font-mono ${getScoreColor(metrics.overallScore)}`}>{metrics.overallScore}%</p>
                <p className="text-xs text-gray-500 mt-1">
                  {metrics.complianceTrend > 0 ? '↗' : '↘'} {Math.abs(metrics.complianceTrend)}% this month
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Shield className="w-5 h-5 text-blue-500" />
                  <span className="text-xs text-gray-500">Frameworks</span>
                </div>
                <p className="text-2xl font-bold font-mono">{metrics.compliantFrameworks}/{metrics.frameworksCount}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.compliantFrameworks / metrics.frameworksCount) * 100)}% compliant</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-xs text-gray-500">Controls</span>
                </div>
                <p className="text-2xl font-bold font-mono text-green-500">{metrics.passingControls}/{metrics.totalControls}</p>
                <p className="text-xs text-gray-500 mt-1">{Math.round((metrics.passingControls / metrics.totalControls) * 100)}% passing</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-xs text-gray-500">Critical Findings</span>
                </div>
                <p className="text-2xl font-bold font-mono text-red-500">{metrics.criticalFindings}</p>
                <p className="text-xs text-gray-500 mt-1">{metrics.openFindings} total open</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-gray-500">Next Audit</span>
                </div>
                <p className="text-2xl font-bold font-mono text-yellow-500">{metrics.nextAuditDays}</p>
                <p className="text-xs text-gray-500 mt-1">days remaining</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Award className="w-5 h-5 text-orange-500" />
                  <span className="text-xs text-gray-500">Certifications</span>
                </div>
                <p className="text-2xl font-bold font-mono text-orange-500">{metrics.certificationsExpiring}</p>
                <p className="text-xs text-gray-500 mt-1">expiring soon</p>
              </motion.div>
            </div>

            {/* Charts and Analytics */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              {/* Compliance Trend */}
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">COMPLIANCE SCORE TREND</h3>
                  <div className="flex items-center space-x-2">
                    <TrendingUp className={`w-4 h-4 ${metrics.complianceTrend > 0 ? 'text-green-500' : 'text-red-500'}`} />
                    <span className={`text-sm ${metrics.complianceTrend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {metrics.complianceTrend > 0 ? '+' : ''}{metrics.complianceTrend}%
                    </span>
                  </div>
                </div>
                <div className="h-64">
                  <Line data={complianceTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                        min: 80,
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Findings Distribution */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">FINDINGS BY SEVERITY</h3>
                <div className="h-64 flex items-center justify-center">
                  <Doughnut data={findingsDistributionData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { 
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 10 }
                        }
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Framework Status Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {frameworks.slice(0, 6).map((framework, index) => (
                <motion.div
                  key={framework.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:bg-gray-800/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{framework.acronym}</h3>
                      <p className="text-xs text-gray-400 mt-1">{framework.name}</p>
                      <p className="text-xs text-gray-500 mt-1">{framework.industry} • {framework.region}</p>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(framework.status)}`}>
                      {framework.status === 'compliant' ? 'Compliant' : 
                       framework.status === 'partial' ? 'Partial' : 'Non-Compliant'}
                    </span>
                  </div>

                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Score</span>
                      <span className={`text-xl font-bold ${getScoreColor(framework.score)}`}>
                        {framework.score}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          framework.score >= 90 ? 'bg-green-400' :
                          framework.score >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${framework.score}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="bg-gray-800 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Controls</p>
                      <p className="text-sm text-white">
                        {framework.controls.passed}/{framework.controls.total}
                      </p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-2">
                      <p className="text-xs text-gray-400">Findings</p>
                      <p className={`text-sm font-semibold ${
                        framework.criticalFindings > 0 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {framework.criticalFindings + framework.highFindings + framework.mediumFindings + framework.lowFindings}
                      </p>
                    </div>
                  </div>

                  <div className="border-t border-gray-700 pt-3">
                    <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                      <span>Next: {new Date(framework.nextAudit).toLocaleDateString()}</span>
                      <span>{framework.auditor}</span>
                    </div>
                    <button className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-sm transition-colors flex items-center justify-center space-x-1">
                      <FileText className="w-3 h-3" />
                      <span>View Details</span>
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'frameworks' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search frameworks..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  />
                </div>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Status</option>
                  <option value="compliant">Compliant</option>
                  <option value="partial">Partial</option>
                  <option value="non-compliant">Non-Compliant</option>
                </select>
                <select className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                  <option value="">All Industries</option>
                  <option value="technology">Technology</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="financial">Financial Services</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
              </div>
            </div>

            {/* Frameworks Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {filteredFrameworks.map((framework, index) => (
                <motion.div
                  key={framework.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-white">{framework.acronym}</h3>
                        <p className="text-sm text-gray-400 mt-1">{framework.name}</p>
                        <div className="flex items-center gap-4 mt-2">
                          <span className="text-xs text-gray-500">Industry: {framework.industry}</span>
                          <span className="text-xs text-gray-500">Region: {framework.region}</span>
                          <span className="text-xs text-gray-500">Type: {framework.assessmentType}</span>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(framework.status)}`}>
                        {framework.status === 'compliant' ? 'Compliant' : 
                         framework.status === 'partial' ? 'Partial' : 'Non-Compliant'}
                      </span>
                    </div>

                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Compliance Score</span>
                        <span className={`text-2xl font-bold ${getScoreColor(framework.score)}`}>
                          {framework.score}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full transition-all ${
                            framework.score >= 90 ? 'bg-green-400' :
                            framework.score >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                          }`}
                          style={{ width: `${framework.score}%` }}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-3 mb-6">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Total Controls</p>
                        <p className="text-lg font-semibold text-white">{framework.controls.total}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Passed</p>
                        <p className="text-lg font-semibold text-green-400">{framework.controls.passed}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Failed</p>
                        <p className="text-lg font-semibold text-red-400">{framework.controls.failed}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">N/A</p>
                        <p className="text-lg font-semibold text-gray-400">{framework.controls.notApplicable}</p>
                      </div>
                    </div>

                    <div className="mb-4">
                      <h4 className="text-sm font-semibold text-white mb-2">Findings Summary</h4>
                      <div className="grid grid-cols-4 gap-2">
                        <div className="bg-red-500/20 text-red-400 rounded p-2 text-center">
                          <p className="text-xs">Critical</p>
                          <p className="font-bold">{framework.criticalFindings}</p>
                        </div>
                        <div className="bg-orange-500/20 text-orange-400 rounded p-2 text-center">
                          <p className="text-xs">High</p>
                          <p className="font-bold">{framework.highFindings}</p>
                        </div>
                        <div className="bg-yellow-500/20 text-yellow-400 rounded p-2 text-center">
                          <p className="text-xs">Medium</p>
                          <p className="font-bold">{framework.mediumFindings}</p>
                        </div>
                        <div className="bg-blue-500/20 text-blue-400 rounded p-2 text-center">
                          <p className="text-xs">Low</p>
                          <p className="font-bold">{framework.lowFindings}</p>
                        </div>
                      </div>
                    </div>

                    <div className="border-t border-gray-700 pt-4">
                      <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                        <span>Auditor: {framework.auditor}</span>
                        <span>Next Audit: {new Date(framework.nextAudit).toLocaleDateString()}</span>
                      </div>
                      {framework.certificationDate && (
                        <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                          <span>Certified: {new Date(framework.certificationDate).toLocaleDateString()}</span>
                          <span>Expires: {framework.expirationDate ? new Date(framework.expirationDate).toLocaleDateString() : 'N/A'}</span>
                        </div>
                      )}
                      <div className="flex gap-2">
                        <button className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-sm">
                          View Report
                        </button>
                        <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                          Run Assessment
                        </button>
                        <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-white text-sm">
                          <MoreVertical className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'findings' && (
          <>
            {/* Findings Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <span className="text-2xl font-bold text-red-500">{findings.filter(f => f.severity === 'critical').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Critical Findings</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <span className="text-2xl font-bold text-orange-500">{findings.filter(f => f.severity === 'high').length}</span>
                </div>
                <p className="text-gray-400 text-sm">High Findings</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <span className="text-2xl font-bold text-yellow-500">{findings.filter(f => f.status === 'open').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Open Findings</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-2xl font-bold text-green-500">{findings.filter(f => f.status === 'resolved').length}</span>
                </div>
                <p className="text-gray-400 text-sm">Resolved</p>
              </div>
            </div>

            {/* Findings Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-400 uppercase">AUDIT FINDINGS</h3>
                <div className="flex items-center space-x-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                      type="text"
                      placeholder="Search findings..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-1.5 bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500 text-sm"
                    />
                  </div>
                  <button className="p-1.5 hover:bg-gray-800 rounded">
                    <Download className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Finding</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Framework</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Control</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Severity</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Due Date</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Assignee</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Risk Score</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {filteredFindings.map((finding) => (
                      <motion.tr
                        key={finding.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div>
                            <div className="font-medium text-white">{finding.title}</div>
                            <div className="text-xs text-gray-500 mt-1">{finding.description.substring(0, 80)}...</div>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-purple-500/20 text-purple-400 text-xs rounded">
                            {finding.framework}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <code className="text-xs bg-gray-800 px-2 py-1 rounded text-blue-400">{finding.control}</code>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border font-medium ${getSeverityColor(finding.severity)}`}>
                            {finding.severity.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded border ${getStatusColor(finding.status)}`}>
                            {finding.status.toUpperCase().replace('-', ' ')}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white">{new Date(finding.dueDate).toLocaleDateString()}</div>
                          <div className={`text-xs ${new Date(finding.dueDate) < new Date() ? 'text-red-400' : 'text-gray-500'}`}>
                            {new Date(finding.dueDate) < new Date() ? 'Overdue' : 'On track'}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-white">{finding.assignee.split('@')[0]}</div>
                          <div className="text-xs text-gray-500">{finding.remediationEffort}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className={`text-sm font-bold ${finding.riskScore > 8 ? 'text-red-400' : finding.riskScore > 6 ? 'text-orange-400' : 'text-yellow-400'}`}>
                            {finding.riskScore.toFixed(1)}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center space-x-1">
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Eye className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <Edit className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1 hover:bg-gray-700 rounded">
                              <MoreVertical className="w-4 h-4 text-gray-400" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'reports' && (
          <>
            {/* Reports Grid */}
            <div className="space-y-4">
              {reports.map((report, index) => (
                <motion.div
                  key={report.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-900 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-white">{report.title}</h3>
                        <p className="text-sm text-gray-400 mt-1">Framework: {report.framework}</p>
                        <p className="text-sm text-gray-500 mt-1">Scope: {report.scope}</p>
                        <p className="text-sm text-gray-500">Author: {report.author}</p>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`px-2 py-1 text-xs rounded border font-medium ${report.type === 'audit' ? 'bg-blue-500/20 text-blue-400 border-blue-500/30' : 
                          report.type === 'certification' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 
                          'bg-purple-500/20 text-purple-400 border-purple-500/30'}`}>
                          {report.type.toUpperCase()}
                        </span>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(report.status)}`}>
                          {report.status.toUpperCase().replace('-', ' ')}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mb-4">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Generated</p>
                        <p className="text-sm text-white">{new Date(report.generatedDate).toLocaleDateString()}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Findings</p>
                        <p className="text-sm font-semibold text-red-400">{report.findings}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Recommendations</p>
                        <p className="text-sm font-semibold text-yellow-400">{report.recommendations}</p>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-400 mb-1">Actions</p>
                        <div className="flex space-x-1">
                          <button className="p-1.5 bg-blue-600 hover:bg-blue-700 rounded text-white">
                            <Download className="w-3 h-3" />
                          </button>
                          <button className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded text-white">
                            <Eye className="w-3 h-3" />
                          </button>
                          <button className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded text-white">
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {selectedTab === 'analytics' && (
          <>
            {/* Analytics Dashboard */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Framework Scores */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">FRAMEWORK SCORES</h3>
                <div className="h-64 flex items-center justify-center">
                  <Bar data={frameworkScoreData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                        min: 0,
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>

              {/* Compliance Trend */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4">COMPLIANCE TREND</h3>
                <div className="h-64">
                  <Line data={complianceTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                      },
                      y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                        min: 80,
                        max: 100
                      }
                    }
                  }} />
                </div>
              </div>
            </div>

            {/* Detailed Analytics */}
            <div className="grid grid-cols-3 gap-6">
              {/* Top Failing Controls */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">TOP FAILING CONTROLS</h3>
                </div>
                <div className="p-4 space-y-3">
                  {controls.filter(c => c.status === 'fail').slice(0, 5).map((control, index) => (
                    <div key={control.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div>
                        <p className="text-sm font-medium text-white">{control.title.substring(0, 30)}...</p>
                        <p className="text-xs text-gray-400">{control.framework} • {control.controlId}</p>
                      </div>
                      <span className={`text-xs font-bold px-2 py-1 rounded ${getSeverityColor(control.severity)}`}>
                        {control.severity.toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Remediation Timeline */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">REMEDIATION TIMELINE</h3>
                </div>
                <div className="p-4 space-y-3">
                  {findings.filter(f => f.status !== 'resolved').slice(0, 5).map((finding, index) => (
                    <div key={finding.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                      <div>
                        <p className="text-sm font-medium text-white">{finding.title.substring(0, 25)}...</p>
                        <p className="text-xs text-gray-400">{finding.assignee.split('@')[0]}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-white">{new Date(finding.dueDate).toLocaleDateString()}</p>
                        <p className={`text-xs ${new Date(finding.dueDate) < new Date() ? 'text-red-400' : 'text-green-400'}`}>
                          {Math.ceil((new Date(finding.dueDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))} days
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Framework Performance */}
              <div className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">FRAMEWORK PERFORMANCE</h3>
                </div>
                <div className="p-4 space-y-3">
                  {frameworks.slice(0, 5).map((framework, index) => (
                    <div key={framework.id} className="p-2 bg-gray-800 rounded">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-white">{framework.acronym}</span>
                        <span className={`text-sm font-bold ${getScoreColor(framework.score)}`}>{framework.score}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${framework.score >= 90 ? 'bg-green-400' :
                            framework.score >= 70 ? 'bg-yellow-400' : 'bg-red-400'}`}
                          style={{ width: `${framework.score}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>{framework.controls.passed} passed</span>
                        <span>{framework.controls.failed} failed</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}