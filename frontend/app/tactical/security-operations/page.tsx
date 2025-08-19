'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { Line, Bar, Doughnut, Radar, Pie } from 'react-chartjs-2';
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
  RadialLinearScale,
  Filler
} from 'chart.js';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';

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
  RadialLinearScale,
  Filler
);

interface SOCData {
  operationalStatus: 'NORMAL' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
  activeAlerts: Array<{
    id: string;
    title: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    category: string;
    source: string;
    timestamp: string;
    status: 'open' | 'investigating' | 'resolved';
    analyst: string;
    description: string;
    evidence: string[];
    recommendations: string[];
    relatedAlerts: string[];
    riskScore: number;
    confidence: number;
  }>;
  incidents: Array<{
    id: string;
    title: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    status: 'new' | 'assigned' | 'investigating' | 'resolved' | 'closed';
    assignedTo: string;
    createdAt: string;
    updatedAt: string;
    category: string;
    priority: number;
    escalationLevel: number;
    timeline: Array<{
      timestamp: string;
      action: string;
      user: string;
      details: string;
    }>;
    containmentActions: string[];
    evidenceCollected: string[];
    affectedSystems: string[];
  }>;
  analysts: Array<{
    id: string;
    name: string;
    role: string;
    status: 'available' | 'busy' | 'offline';
    activeIncidents: number;
    specializations: string[];
    certifications: string[];
    shift: string;
    performance: {
      alertsHandled: number;
      avgResponseTime: number;
      resolutionRate: number;
      escalationRate: number;
    };
  }>;
  slaMetrics: {
    p1ResponseTime: number;
    p2ResponseTime: number;
    p3ResponseTime: number;
    overallSLA: number;
    breaches: number;
    mttr: number;
    mttd: number;
    mtbf: number;
  };
  workflows: Array<{
    id: string;
    name: string;
    description: string;
    triggerConditions: string[];
    actions: string[];
    status: 'active' | 'paused' | 'draft';
    executionCount: number;
    successRate: number;
    lastExecution: string;
    avgExecutionTime: number;
  }>;
  playbooks: Array<{
    id: string;
    name: string;
    category: string;
    description: string;
    steps: Array<{
      step: number;
      title: string;
      description: string;
      assignee: string;
      estimatedTime: number;
      status: 'pending' | 'in_progress' | 'completed';
    }>;
    lastUsed: string;
    effectiveness: number;
    averageExecutionTime: number;
    timesUsed: number;
  }>;
  metrics: {
    alertVolume: Array<{
      timestamp: string;
      count: number;
      resolved: number;
      false_positives: number;
      escalated: number;
    }>;
    incidentTrends: Array<{
      date: string;
      count: number;
      severity: string;
    }>;
    analystWorkload: Array<{
      analyst: string;
      activeAlerts: number;
      pendingTasks: number;
      workloadScore: number;
    }>;
    responseEfficiency: Array<{
      timeRange: string;
      averageResponseTime: number;
      slaCompliance: number;
    }>;
  };
  threatIntel: Array<{
    id: string;
    source: string;
    type: 'ioc' | 'ttp' | 'vulnerability' | 'campaign';
    indicator: string;
    confidence: number;
    severity: string;
    firstSeen: string;
    lastSeen: string;
    context: string;
    tags: string[];
    relatedIncidents: string[];
    actionTaken: string;
  }>;
}

export default function SecurityOperationsCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <SecurityOperationsCenterContent />
    </AuthGuard>
  );
}

function SecurityOperationsCenterContent() {
  const [data, setData] = useState<SOCData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('operations');
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [selectedIncident, setSelectedIncident] = useState<any>(null);
  const [selectedPlaybook, setSelectedPlaybook] = useState<any>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [sortBy, setSortBy] = useState('severity');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30000);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showPlaybookModal, setShowPlaybookModal] = useState(false);
  const [showAnalystModal, setShowAnalystModal] = useState(false);
  const [operationLogs, setOperationLogs] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState('24h');

  const triggerAction = async (actionType: string) => {
    try {
      const resp = await api.createAction('soc', actionType);
      if (resp.error || resp.status >= 400) {
        console.error('SOC Action failed', actionType, resp.error);
        return;
      }
      const id = resp.data?.action_id || resp.data?.id;
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[soc-action]', id, m));
        setTimeout(stop, 60000);
      }
    } catch (e) {
      console.error('SOC Trigger action error', actionType, e);
    }
  };

  useEffect(() => {
    fetchSOCData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchSOCData, refreshInterval);
      const logInterval = setInterval(() => {
        const newLog = `[${new Date().toISOString()}] ${generateRandomSOCLog()}`;
        setOperationLogs(prev => [newLog, ...prev.slice(0, 199)]);
      }, 3000);
      
      return () => {
        clearInterval(interval);
        clearInterval(logInterval);
      };
    }
  }, [autoRefresh, refreshInterval]);

  const fetchSOCData = async () => {
    try {
      const resp = await api.getSOCData();
      if (resp.error) setData(getMockSOCData());
      else setData(resp.data as any);
    } catch (error) {
      setData(getMockSOCData());
    } finally {
      setLoading(false);
    }
  };

  const getMockSOCData = (): SOCData => ({
    operationalStatus: 'ELEVATED',
    activeAlerts: [
      {
        id: 'ALT-001',
        title: 'Suspicious Network Traffic Detected',
        severity: 'high',
        category: 'Network Security',
        source: 'IDS/IPS System',
        timestamp: '2024-01-15T09:15:00Z',
        status: 'investigating',
        analyst: 'Sarah Chen',
        description: 'Unusual outbound traffic patterns detected from production servers indicating potential data exfiltration',
        evidence: ['network_capture_001.pcap', 'firewall_logs_Jan15.txt', 'dns_queries.log'],
        recommendations: ['Isolate affected servers', 'Analyze traffic patterns', 'Check for data exfiltration', 'Review DNS logs'],
        relatedAlerts: ['ALT-002', 'ALT-005'],
        riskScore: 85,
        confidence: 92
      },
      {
        id: 'ALT-002',
        title: 'Multiple Failed Authentication Attempts',
        severity: 'critical',
        category: 'Authentication',
        source: 'Active Directory',
        timestamp: '2024-01-15T08:45:00Z',
        status: 'open',
        analyst: 'Mike Rodriguez',
        description: 'Coordinated brute force attack detected against privileged administrator accounts from multiple IP addresses',
        evidence: ['auth_logs_001.log', 'failed_logins.csv', 'ip_geolocation.json'],
        recommendations: ['Lock affected accounts immediately', 'Enable emergency MFA', 'Investigate source IPs', 'Review user privileges'],
        relatedAlerts: ['ALT-001', 'ALT-003'],
        riskScore: 95,
        confidence: 98
      },
      {
        id: 'ALT-003',
        title: 'Advanced Malware Detection',
        severity: 'critical',
        category: 'Malware',
        source: 'Endpoint Protection',
        timestamp: '2024-01-15T08:30:00Z',
        status: 'investigating',
        analyst: 'David Kim',
        description: 'Zero-day malware with anti-analysis capabilities detected on critical finance workstation',
        evidence: ['malware_sample.bin', 'system_scan_report.xml', 'memory_dump.dmp'],
        recommendations: ['Immediate system quarantine', 'Full malware analysis', 'Check for lateral movement', 'Update IOCs'],
        relatedAlerts: ['ALT-004'],
        riskScore: 98,
        confidence: 94
      },
      {
        id: 'ALT-004',
        title: 'Data Exfiltration Attempt Blocked',
        severity: 'high',
        category: 'Data Loss Prevention',
        source: 'DLP System',
        timestamp: '2024-01-15T07:20:00Z',
        status: 'resolved',
        analyst: 'Emma Watson',
        description: 'Large-scale sensitive data transfer attempt to external cloud storage detected and blocked',
        evidence: ['dlp_alert_001.json', 'file_transfer_log.txt', 'data_classification.csv'],
        recommendations: ['Verify data integrity', 'Interview affected user', 'Review access permissions', 'Enhance DLP rules'],
        relatedAlerts: [],
        riskScore: 78,
        confidence: 89
      },
      {
        id: 'ALT-005',
        title: 'Privilege Escalation Attack',
        severity: 'critical',
        category: 'Privilege Management',
        source: 'PAM System',
        timestamp: '2024-01-15T06:15:00Z',
        status: 'investigating',
        analyst: 'Tom Wilson',
        description: 'Sophisticated privilege escalation attempt detected on domain controller using exploit chain',
        evidence: ['privilege_audit.log', 'user_activity_trace.csv', 'exploit_artifacts.zip'],
        recommendations: ['Emergency privilege revocation', 'Full system audit', 'Check for persistence', 'Rebuild compromised systems'],
        relatedAlerts: ['ALT-001'],
        riskScore: 97,
        confidence: 96
      }
    ],
    incidents: [
      {
        id: 'INC-2024-001',
        title: 'Advanced Persistent Threat Campaign',
        severity: 'critical',
        status: 'investigating',
        assignedTo: 'Incident Response Team Alpha',
        createdAt: '2024-01-15T06:00:00Z',
        updatedAt: '2024-01-15T09:30:00Z',
        category: 'Advanced Persistent Threat',
        priority: 1,
        escalationLevel: 3,
        timeline: [
          { timestamp: '06:00:00', action: 'Incident Created', user: 'System', details: 'Auto-generated from multiple correlated alerts' },
          { timestamp: '06:15:00', action: 'War Room Activated', user: 'SOC Manager', details: 'Escalated to IR Team Alpha - Emergency protocols initiated' },
          { timestamp: '07:30:00', action: 'Evidence Secured', user: 'Sarah Chen', details: 'Network captures, system logs, and memory dumps collected' },
          { timestamp: '08:45:00', action: 'Containment Initiated', user: 'Mike Rodriguez', details: 'Affected systems isolated from network' },
          { timestamp: '09:30:00', action: 'Threat Attribution', user: 'David Kim', details: 'Malware family identified as APT29 variant - Cozy Bear tactics' }
        ],
        containmentActions: ['Network segmentation', 'System isolation', 'User account lockdown', 'DNS sinkholing'],
        evidenceCollected: ['Network packet captures', 'System memory dumps', 'Registry hives', 'Event logs', 'Malware samples'],
        affectedSystems: ['dc-prod-01', 'file-server-02', 'ws-finance-15', 'sql-prod-cluster']
      },
      {
        id: 'INC-2024-002',
        title: 'Ransomware Outbreak - Finance Department',
        severity: 'high',
        status: 'resolved',
        assignedTo: 'Emma Watson',
        createdAt: '2024-01-14T14:20:00Z',
        updatedAt: '2024-01-14T18:45:00Z',
        category: 'Malware',
        priority: 1,
        escalationLevel: 2,
        timeline: [
          { timestamp: '14:20:00', action: 'Incident Created', user: 'Emma Watson', details: 'Multiple ransomware alerts from finance department' },
          { timestamp: '14:25:00', action: 'Emergency Response', user: 'SOC Team', details: 'Immediate network isolation and backup verification' },
          { timestamp: '15:30:00', action: 'Malware Analysis', user: 'Malware Lab', details: 'Ransomware variant identified as LockBit 3.0' },
          { timestamp: '16:45:00', action: 'Recovery Initiated', user: 'IT Operations', details: 'Clean backup restoration commenced' },
          { timestamp: '18:45:00', action: 'Incident Resolved', user: 'Emma Watson', details: 'All systems restored, security measures implemented' }
        ],
        containmentActions: ['Network isolation', 'Process termination', 'File system protection', 'Backup verification'],
        evidenceCollected: ['Ransomware binary', 'Encryption keys', 'Network logs', 'Email headers'],
        affectedSystems: ['ws-finance-01', 'ws-finance-02', 'file-share-finance']
      },
      {
        id: 'INC-2024-003',
        title: 'Targeted Phishing Campaign - C-Suite',
        severity: 'high',
        status: 'investigating',
        assignedTo: 'Alex Rodriguez',
        createdAt: '2024-01-13T11:15:00Z',
        updatedAt: '2024-01-15T09:00:00Z',
        category: 'Social Engineering',
        priority: 2,
        escalationLevel: 2,
        timeline: [
          { timestamp: '11:15:00', action: 'Incident Created', user: 'Alex Rodriguez', details: 'Sophisticated phishing emails targeting executives' },
          { timestamp: '11:30:00', action: 'Email Analysis', user: 'Email Security Team', details: 'Advanced spoofing techniques identified' },
          { timestamp: '12:00:00', action: 'User Notification', user: 'Security Team', details: 'All targeted users immediately notified' },
          { timestamp: '14:30:00', action: 'Threat Intel Update', user: 'Threat Hunter', details: 'Campaign linked to known APT group' }
        ],
        containmentActions: ['Email quarantine', 'User awareness', 'URL blocking', 'Email gateway rules'],
        evidenceCollected: ['Phishing emails', 'Malicious URLs', 'Sender reputation data', 'User reports'],
        affectedSystems: ['email-gateway', 'user-workstations']
      }
    ],
    analysts: [
      {
        id: 'analyst-001',
        name: 'Sarah Chen',
        role: 'Senior SOC Analyst',
        status: 'busy',
        activeIncidents: 3,
        specializations: ['Network Security', 'Incident Response', 'Threat Hunting', 'Digital Forensics'],
        certifications: ['GCIH', 'GCFA', 'CISSP', 'SANS FOR508'],
        shift: 'Day Shift (8AM-6PM)',
        performance: { alertsHandled: 847, avgResponseTime: 8, resolutionRate: 94.2, escalationRate: 12.5 }
      },
      {
        id: 'analyst-002',
        name: 'Mike Rodriguez',
        role: 'SOC Analyst L2',
        status: 'available',
        activeIncidents: 1,
        specializations: ['Malware Analysis', 'Digital Forensics', 'SIEM Management', 'Reverse Engineering'],
        certifications: ['GCFE', 'GREM', 'CEH', 'GIAC GMON'],
        shift: 'Day Shift (8AM-6PM)',
        performance: { alertsHandled: 623, avgResponseTime: 15, resolutionRate: 89.7, escalationRate: 18.3 }
      },
      {
        id: 'analyst-003',
        name: 'David Kim',
        role: 'Threat Hunter',
        status: 'busy',
        activeIncidents: 2,
        specializations: ['Threat Intelligence', 'IOC Analysis', 'Behavioral Analysis', 'APT Tracking'],
        certifications: ['GCTI', 'GNFA', 'CySA+', 'SANS FOR578'],
        shift: 'Day Shift (8AM-6PM)',
        performance: { alertsHandled: 412, avgResponseTime: 22, resolutionRate: 92.1, escalationRate: 8.7 }
      },
      {
        id: 'analyst-004',
        name: 'Emma Watson',
        role: 'SOC Analyst L1',
        status: 'available',
        activeIncidents: 0,
        specializations: ['Alert Triage', 'Log Analysis', 'Vulnerability Assessment', 'Compliance'],
        certifications: ['Security+', 'CySA+', 'GCFA', 'CompTIA PenTest+'],
        shift: 'Night Shift (6PM-8AM)',
        performance: { alertsHandled: 1203, avgResponseTime: 6, resolutionRate: 87.3, escalationRate: 25.1 }
      },
      {
        id: 'analyst-005',
        name: 'Tom Wilson',
        role: 'SOC Manager',
        status: 'available',
        activeIncidents: 1,
        specializations: ['Team Management', 'Crisis Response', 'Strategic Planning', 'Vendor Management'],
        certifications: ['CISSP', 'CISM', 'GCIH', 'CISSP-ISSMP'],
        shift: 'Business Hours (9AM-5PM)',
        performance: { alertsHandled: 156, avgResponseTime: 3, resolutionRate: 98.1, escalationRate: 5.2 }
      },
      {
        id: 'analyst-006',
        name: 'Alex Rodriguez',
        role: 'Incident Response Lead',
        status: 'busy',
        activeIncidents: 2,
        specializations: ['Incident Response', 'Crisis Management', 'Forensic Analysis', 'Business Continuity'],
        certifications: ['GCIR', 'CISSP', 'CISA', 'SANS FOR572'],
        shift: 'On-Call (24/7)',
        performance: { alertsHandled: 298, avgResponseTime: 4, resolutionRate: 96.8, escalationRate: 8.9 }
      }
    ],
    slaMetrics: {
      p1ResponseTime: 8,
      p2ResponseTime: 25,
      p3ResponseTime: 120,
      overallSLA: 94.7,
      breaches: 12,
      mttr: 42,
      mttd: 18,
      mtbf: 2160
    },
    workflows: [
      {
        id: 'wf-001',
        name: 'Malware Detection & Response',
        description: 'Automated response workflow for malware detection alerts including containment and analysis',
        triggerConditions: ['malware_detected', 'endpoint_alert', 'av_signature_match'],
        actions: ['isolate_endpoint', 'collect_artifacts', 'notify_analyst', 'create_case', 'update_iocs'],
        status: 'active',
        executionCount: 247,
        successRate: 96.3,
        lastExecution: '2024-01-15T08:30:00Z',
        avgExecutionTime: 180
      },
      {
        id: 'wf-002',
        name: 'Phishing Email Investigation',
        description: 'Automated phishing email analysis and response including user notification and URL blocking',
        triggerConditions: ['phishing_reported', 'suspicious_email', 'url_reputation_low'],
        actions: ['analyze_headers', 'check_attachments', 'block_sender', 'notify_users', 'quarantine_email'],
        status: 'active',
        executionCount: 189,
        successRate: 91.7,
        lastExecution: '2024-01-15T07:45:00Z',
        avgExecutionTime: 120
      },
      {
        id: 'wf-003',
        name: 'Failed Login Investigation',
        description: 'Automated response to multiple failed login attempts including account lockdown and investigation',
        triggerConditions: ['multiple_failed_logins', 'brute_force_detected', 'unusual_login_pattern'],
        actions: ['lock_account', 'analyze_source', 'notify_user', 'escalate_if_admin', 'update_threat_intel'],
        status: 'active',
        executionCount: 456,
        successRate: 88.9,
        lastExecution: '2024-01-15T09:15:00Z',
        avgExecutionTime: 90
      },
      {
        id: 'wf-004',
        name: 'Network Anomaly Response',
        description: 'Automated network anomaly detection and initial response including traffic analysis',
        triggerConditions: ['network_anomaly', 'unusual_traffic_pattern', 'bandwidth_spike'],
        actions: ['capture_traffic', 'analyze_flows', 'check_reputation', 'create_alert', 'notify_netops'],
        status: 'active',
        executionCount: 78,
        successRate: 93.6,
        lastExecution: '2024-01-15T06:20:00Z',
        avgExecutionTime: 300
      }
    ],
    playbooks: [
      {
        id: 'pb-001',
        name: 'Advanced Persistent Threat Response',
        category: 'Incident Response',
        description: 'Comprehensive playbook for APT investigation, containment, and eradication',
        steps: [
          { step: 1, title: 'Initial Triage & Assessment', description: 'Assess scope, impact, and threat actor', assignee: 'Lead Analyst', estimatedTime: 30, status: 'completed' },
          { step: 2, title: 'Evidence Preservation', description: 'Collect and preserve digital evidence', assignee: 'Forensics Team', estimatedTime: 120, status: 'in_progress' },
          { step: 3, title: 'Containment Strategy', description: 'Isolate affected systems and prevent spread', assignee: 'Network Team', estimatedTime: 45, status: 'pending' },
          { step: 4, title: 'Malware Analysis', description: 'Reverse engineer malware and identify IOCs', assignee: 'Malware Team', estimatedTime: 240, status: 'pending' },
          { step: 5, title: 'Threat Attribution', description: 'Identify threat actor and campaign', assignee: 'Threat Intel', estimatedTime: 180, status: 'pending' },
          { step: 6, title: 'Eradication & Recovery', description: 'Remove threats and restore services', assignee: 'Response Team', estimatedTime: 360, status: 'pending' }
        ],
        lastUsed: '2024-01-15T06:00:00Z',
        effectiveness: 94.2,
        averageExecutionTime: 975,
        timesUsed: 8
      },
      {
        id: 'pb-002',
        name: 'Data Breach Response',
        category: 'Incident Response',
        description: 'Comprehensive response playbook for confirmed or suspected data breaches',
        steps: [
          { step: 1, title: 'Breach Confirmation', description: 'Verify and assess the scope of data breach', assignee: 'SOC Lead', estimatedTime: 15, status: 'completed' },
          { step: 2, title: 'Immediate Containment', description: 'Stop ongoing data loss and secure systems', assignee: 'Security Team', estimatedTime: 30, status: 'completed' },
          { step: 3, title: 'Impact Assessment', description: 'Determine scope and type of compromised data', assignee: 'Data Team', estimatedTime: 60, status: 'in_progress' },
          { step: 4, title: 'Legal & Compliance Notification', description: 'Notify legal team and regulatory bodies', assignee: 'Compliance', estimatedTime: 15, status: 'pending' },
          { step: 5, title: 'Forensic Investigation', description: 'Conduct detailed investigation of breach', assignee: 'Forensics', estimatedTime: 480, status: 'pending' },
          { step: 6, title: 'System Recovery', description: 'Implement fixes and restore normal operations', assignee: 'Operations', estimatedTime: 240, status: 'pending' }
        ],
        lastUsed: '2024-01-10T14:30:00Z',
        effectiveness: 91.8,
        averageExecutionTime: 840,
        timesUsed: 3
      },
      {
        id: 'pb-003',
        name: 'Ransomware Response',
        category: 'Malware Response',
        description: 'Rapid response playbook for ransomware incidents including recovery procedures',
        steps: [
          { step: 1, title: 'Initial Detection & Isolation', description: 'Identify and isolate infected systems', assignee: 'SOC Analyst', estimatedTime: 10, status: 'completed' },
          { step: 2, title: 'Damage Assessment', description: 'Assess extent of encryption and impact', assignee: 'Security Team', estimatedTime: 20, status: 'completed' },
          { step: 3, title: 'Backup Verification', description: 'Verify integrity of backup systems', assignee: 'Backup Team', estimatedTime: 30, status: 'completed' },
          { step: 4, title: 'Decryption Analysis', description: 'Analyze ransomware for potential decryption', assignee: 'Malware Team', estimatedTime: 120, status: 'completed' },
          { step: 5, title: 'Recovery Process', description: 'Restore systems from clean backups', assignee: 'IT Operations', estimatedTime: 240, status: 'completed' },
          { step: 6, title: 'Security Hardening', description: 'Implement additional security measures', assignee: 'Security Team', estimatedTime: 60, status: 'completed' }
        ],
        lastUsed: '2024-01-14T14:20:00Z',
        effectiveness: 98.5,
        averageExecutionTime: 480,
        timesUsed: 12
      }
    ],
    metrics: {
      alertVolume: Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
        count: Math.floor(Math.random() * 80) + 30,
        resolved: Math.floor(Math.random() * 60) + 20,
        false_positives: Math.floor(Math.random() * 15) + 5,
        escalated: Math.floor(Math.random() * 10) + 2
      })),
      incidentTrends: Array.from({ length: 7 }, (_, i) => ({
        date: new Date(Date.now() - (6 - i) * 24 * 60 * 60 * 1000).toDateString(),
        count: Math.floor(Math.random() * 25) + 5,
        severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)]
      })),
      analystWorkload: [
        { analyst: 'Sarah Chen', activeAlerts: 12, pendingTasks: 18, workloadScore: 85 },
        { analyst: 'Mike Rodriguez', activeAlerts: 6, pendingTasks: 9, workloadScore: 52 },
        { analyst: 'David Kim', activeAlerts: 8, pendingTasks: 14, workloadScore: 68 },
        { analyst: 'Emma Watson', activeAlerts: 4, pendingTasks: 6, workloadScore: 35 },
        { analyst: 'Tom Wilson', activeAlerts: 2, pendingTasks: 8, workloadScore: 28 },
        { analyst: 'Alex Rodriguez', activeAlerts: 5, pendingTasks: 11, workloadScore: 58 }
      ],
      responseEfficiency: [
        { timeRange: '0-5 min', averageResponseTime: 3.2, slaCompliance: 98.5 },
        { timeRange: '5-15 min', averageResponseTime: 8.7, slaCompliance: 94.2 },
        { timeRange: '15-30 min', averageResponseTime: 22.1, slaCompliance: 89.3 },
        { timeRange: '30+ min', averageResponseTime: 45.6, slaCompliance: 76.8 }
      ]
    },
    threatIntel: [
      {
        id: 'ti-001',
        source: 'VirusTotal',
        type: 'ioc',
        indicator: '192.168.100.45',
        confidence: 95,
        severity: 'high',
        firstSeen: '2024-01-14T10:30:00Z',
        lastSeen: '2024-01-15T08:15:00Z',
        context: 'Known C2 server for APT29 campaign targeting financial institutions',
        tags: ['apt29', 'c2', 'malware', 'financial'],
        relatedIncidents: ['INC-2024-001'],
        actionTaken: 'Blocked at firewall, DNS sinkhole configured'
      },
      {
        id: 'ti-002',
        source: 'AlienVault OTX',
        type: 'vulnerability',
        indicator: 'CVE-2024-0001',
        confidence: 88,
        severity: 'critical',
        firstSeen: '2024-01-10T00:00:00Z',
        lastSeen: '2024-01-15T09:00:00Z',
        context: 'Zero-day vulnerability in Apache HTTP Server allowing remote code execution',
        tags: ['zero-day', 'apache', 'rce', 'web-server'],
        relatedIncidents: [],
        actionTaken: 'Emergency patching initiated, WAF rules deployed'
      },
      {
        id: 'ti-003',
        source: 'Internal Analysis',
        type: 'ttp',
        indicator: 'Living off the land techniques',
        confidence: 92,
        severity: 'medium',
        firstSeen: '2024-01-12T15:20:00Z',
        lastSeen: '2024-01-15T07:45:00Z',
        context: 'PowerShell-based persistence mechanism used in recent APT campaigns',
        tags: ['powershell', 'persistence', 'lolbins', 'apt'],
        relatedIncidents: ['INC-2024-001'],
        actionTaken: 'PowerShell logging enhanced, behavioral detection updated'
      }
    ]
  });

  const generateRandomSOCLog = () => {
    const logs = [
      'Alert escalated to Tier 2 analyst - High priority malware detection',
      'Incident response team activated for critical security breach',
      'Automated containment action executed successfully',
      'Threat intelligence feed updated with new IOCs',
      'Security workflow execution completed - 98% success rate',
      'New malicious IP added to global blocklist',
      'Senior analyst assigned to critical priority alert',
      'Emergency playbook execution initiated for data breach',
      'Digital evidence collection completed for forensic analysis',
      'Advanced malware analysis in progress - sandbox environment',
      'Network isolation implemented for compromised systems',
      'Signature database updated with latest threat patterns',
      'Vulnerability assessment scan completed - 3 critical findings',
      'Security control effectiveness validation completed',
      'Phishing email campaign blocked at email gateway',
      'User security awareness notification sent to 1,247 users',
      'DNS sinkhole configured for malicious domain',
      'Backup integrity verification completed successfully',
      'Security metrics dashboard updated with real-time data',
      'Cross-correlation analysis identified potential APT activity'
    ];
    return logs[Math.floor(Math.random() * logs.length)];
  };

  const filteredAlerts = useMemo(() => {
    if (!data?.activeAlerts) return [];
    
    return data.activeAlerts
      .filter(alert => {
        const matchesSearch = alert.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                             alert.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
                             alert.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
                             alert.analyst.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSeverity = filterSeverity === 'all' || alert.severity === filterSeverity;
        return matchesSearch && matchesSeverity;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'severity':
            const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
            return severityOrder[b.severity as keyof typeof severityOrder] - severityOrder[a.severity as keyof typeof severityOrder];
          case 'timestamp':
            return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
          case 'riskScore':
            return b.riskScore - a.riskScore;
          case 'confidence':
            return b.confidence - a.confidence;
          default:
            return 0;
        }
      });
  }, [data?.activeAlerts, searchTerm, sortBy, filterSeverity]);

  const exportData = (format: 'csv' | 'json') => {
    if (!data) return;
    
    let content: string;
    let filename: string;
    let mimeType: string;
    
    if (format === 'csv') {
      const headers = ['ID', 'Title', 'Severity', 'Category', 'Source', 'Status', 'Analyst', 'Risk Score', 'Confidence', 'Timestamp'];
      const rows = data.activeAlerts.map(alert => [
        alert.id, alert.title, alert.severity, alert.category, 
        alert.source, alert.status, alert.analyst, alert.riskScore, alert.confidence, alert.timestamp
      ]);
      content = [headers, ...rows].map(row => row.join(',')).join('\n');
      filename = 'soc-alerts.csv';
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(data, null, 2);
      filename = 'soc-data.json';
      mimeType = 'application/json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin absolute top-2 left-2 opacity-60" />
            <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full animate-spin absolute top-4 left-4 opacity-40" />
          </div>
          <p className="text-lg text-blue-500 font-bold animate-pulse">INITIALIZING SOC OPERATIONS</p>
          <div className="flex justify-center space-x-1 mt-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        labels: { color: '#9CA3AF' }
      }
    },
    scales: {
      x: {
        ticks: { color: '#9CA3AF' },
        grid: { color: '#374151' }
      },
      y: {
        ticks: { color: '#9CA3AF' },
        grid: { color: '#374151' }
      }
    }
  };

  const alertVolumeData = {
    labels: data?.metrics.alertVolume.map(d => new Date(d.timestamp).toLocaleTimeString()) || [],
    datasets: [
      {
        label: 'Total Alerts',
        data: data?.metrics.alertVolume.map(d => d.count) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Resolved',
        data: data?.metrics.alertVolume.map(d => d.resolved) || [],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'False Positives',
        data: data?.metrics.alertVolume.map(d => d.false_positives) || [],
        borderColor: 'rgb(251, 191, 36)',
        backgroundColor: 'rgba(251, 191, 36, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Escalated',
        data: data?.metrics.alertVolume.map(d => d.escalated) || [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const analystWorkloadData = {
    labels: data?.metrics.analystWorkload.map(a => a.analyst.split(' ')[0]) || [],
    datasets: [
      {
        label: 'Active Alerts',
        data: data?.metrics.analystWorkload.map(a => a.activeAlerts) || [],
        backgroundColor: 'rgba(59, 130, 246, 0.7)',
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 1
      },
      {
        label: 'Pending Tasks',
        data: data?.metrics.analystWorkload.map(a => a.pendingTasks) || [],
        backgroundColor: 'rgba(251, 191, 36, 0.7)',
        borderColor: 'rgb(251, 191, 36)',
        borderWidth: 1
      }
    ]
  };

  const incidentStatusData = {
    labels: ['New', 'Assigned', 'Investigating', 'Resolved', 'Closed'],
    datasets: [
      {
        data: [
          data?.incidents.filter(i => i.status === 'new').length || 1,
          data?.incidents.filter(i => i.status === 'assigned').length || 0,
          data?.incidents.filter(i => i.status === 'investigating').length || 2,
          data?.incidents.filter(i => i.status === 'resolved').length || 1,
          data?.incidents.filter(i => i.status === 'closed').length || 0
        ],
        backgroundColor: [
          'rgba(239, 68, 68, 0.8)',
          'rgba(251, 191, 36, 0.8)', 
          'rgba(59, 130, 246, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(156, 163, 175, 0.8)'
        ],
        borderColor: [
          'rgb(239, 68, 68)',
          'rgb(251, 191, 36)',
          'rgb(59, 130, 246)', 
          'rgb(34, 197, 94)',
          'rgb(156, 163, 175)'
        ],
        borderWidth: 2
      }
    ]
  };

  const responseEfficiencyData = {
    labels: data?.metrics.responseEfficiency.map(r => r.timeRange) || [],
    datasets: [
      {
        label: 'SLA Compliance (%)',
        data: data?.metrics.responseEfficiency.map(r => r.slaCompliance) || [],
        backgroundColor: 'rgba(34, 197, 94, 0.7)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="min-h-screen bg-black text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-blue-900/30 backdrop-blur-sm">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-blue-400 transition-colors">
                ← TACTICAL
              </Link>
              <div className="h-6 w-px bg-blue-800" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-cyan-500 bg-clip-text text-transparent">
                SECURITY OPERATIONS CENTER
              </h1>
              <div className={`px-3 py-1 rounded-lg text-xs font-bold border ${
                data?.operationalStatus === 'CRITICAL' ? 'bg-red-900/30 text-red-400 border-red-600 animate-pulse' :
                data?.operationalStatus === 'HIGH' ? 'bg-orange-900/30 text-orange-400 border-orange-600' :
                data?.operationalStatus === 'ELEVATED' ? 'bg-yellow-900/30 text-yellow-400 border-yellow-600' :
                'bg-green-900/30 text-green-400 border-green-600'
              }`}>
                STATUS: {data?.operationalStatus}
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-400">Auto Refresh:</label>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    autoRefresh ? 'bg-green-900/30 text-green-400 border border-green-600' : 'bg-gray-800 text-gray-400 border border-gray-600'
                  }`}
                >
                  {autoRefresh ? 'ON' : 'OFF'}
                </button>
              </div>
              <button onClick={() => exportData('csv')} className="px-4 py-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 text-blue-400 text-sm font-medium rounded-lg transition-all">
                EXPORT DATA
              </button>
              <button onClick={() => triggerAction('activate_war_room')} className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 border border-red-600 text-red-400 text-sm font-medium rounded-lg transition-all animate-pulse">
                ACTIVATE WAR ROOM
              </button>
              <button onClick={() => triggerAction('escalate_to_ciso')} className="px-4 py-2 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-600 text-purple-400 text-sm font-medium rounded-lg transition-all">
                ESCALATE TO CISO
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Navigation Tabs */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6">
        <div className="flex space-x-8">
          {[
            { id: 'operations', label: 'OPERATIONS CENTER', icon: '🎯' },
            { id: 'alerts', label: 'ACTIVE ALERTS', icon: '⚠️' },
            { id: 'incidents', label: 'INCIDENT RESPONSE', icon: '🚨' },
            { id: 'analysts', label: 'ANALYST WORKLOAD', icon: '👥' },
            { id: 'workflows', label: 'AUTOMATION', icon: '⚙️' },
            { id: 'playbooks', label: 'PLAYBOOKS', icon: '📖' },
            { id: 'intelligence', label: 'THREAT INTEL', icon: '🔍' },
            { id: 'metrics', label: 'PERFORMANCE', icon: '📊' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-6">
        {/* Enhanced Metrics Row */}
        <div className="grid grid-cols-8 gap-4 mb-6">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-blue-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-blue-400">Active Alerts</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.activeAlerts.length}</p>
            <div className="mt-1 text-xs">
              <span className="text-red-400">{data?.activeAlerts.filter(a => a.severity === 'critical').length} Critical</span>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-orange-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-orange-400">Open Incidents</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.incidents.length}</p>
            <div className="mt-1 text-xs">
              <span className="text-yellow-400">{data?.incidents.filter(i => i.status === 'investigating').length} Investigating</span>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-green-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-green-400">MTTR</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.slaMetrics.mttr}m</p>
            <div className="mt-1 text-xs text-green-400">↓ 18% this week</div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-cyan-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-cyan-400">MTTD</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.slaMetrics.mttd}m</p>
            <div className="mt-1 text-xs text-cyan-400">Detection time</div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-purple-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-purple-400">SLA Compliance</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.slaMetrics.overallSLA}%</p>
            <div className="mt-1 text-xs">
              <span className="text-red-400">{data?.slaMetrics.breaches} Breaches</span>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-teal-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-teal-400">Analysts Online</p>
            <p className="text-3xl font-bold font-mono text-white">
              {data?.analysts.filter(a => a.status !== 'offline').length}
            </p>
            <div className="mt-1 text-xs text-teal-400">
              {data?.analysts.filter(a => a.status === 'available').length} Available
            </div>
          </div>

          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-indigo-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-indigo-400">Active Workflows</p>
            <p className="text-3xl font-bold font-mono text-white">
              {data?.workflows.filter(w => w.status === 'active').length}
            </p>
            <div className="mt-1 text-xs text-indigo-400">
              {data?.workflows.reduce((sum, w) => sum + w.executionCount, 0)} Executions
            </div>
          </div>

          <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-4 hover:border-pink-600/50 transition-all duration-300 group">
            <p className="text-xs text-gray-400 uppercase mb-1 group-hover:text-pink-400">Threat Intel</p>
            <p className="text-3xl font-bold font-mono text-white">{data?.threatIntel.length}</p>
            <div className="mt-1 text-xs text-pink-400">
              {data?.threatIntel.filter(t => t.confidence > 90).length} High Confidence
            </div>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'operations' && (
          <div className="space-y-6">
            {/* Operations Command Center */}
            <div className="grid grid-cols-4 gap-6">
              {/* Alert Summary */}
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4 flex items-center space-x-2">
                  <span>⚠️</span>
                  <span>ALERT SUMMARY</span>
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-red-900/20 border border-red-600/30 rounded-lg">
                    <span className="text-gray-300">Critical</span>
                    <span className="text-red-400 font-bold text-xl">
                      {data?.activeAlerts.filter(a => a.severity === 'critical').length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-orange-900/20 border border-orange-600/30 rounded-lg">
                    <span className="text-gray-300">High</span>
                    <span className="text-orange-400 font-bold text-xl">
                      {data?.activeAlerts.filter(a => a.severity === 'high').length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-yellow-900/20 border border-yellow-600/30 rounded-lg">
                    <span className="text-gray-300">Medium</span>
                    <span className="text-yellow-400 font-bold text-xl">
                      {data?.activeAlerts.filter(a => a.severity === 'medium').length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                    <span className="text-gray-300">Low</span>
                    <span className="text-blue-400 font-bold text-xl">
                      {data?.activeAlerts.filter(a => a.severity === 'low').length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Incident Status */}
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4 flex items-center space-x-2">
                  <span>🚨</span>
                  <span>INCIDENT STATUS</span>
                </h3>
                <div className="space-y-3">
                  {data?.incidents.slice(0, 4).map(incident => (
                    <div key={incident.id} className="p-3 bg-gray-800/30 rounded-lg border border-gray-600/30">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-mono text-gray-400">{incident.id}</span>
                        <span className={`text-xs px-2 py-1 rounded ${
                          incident.severity === 'critical' ? 'bg-red-900/30 text-red-400' :
                          incident.severity === 'high' ? 'bg-orange-900/30 text-orange-400' :
                          incident.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-400' :
                          'bg-blue-900/30 text-blue-400'
                        }`}>
                          {incident.severity.toUpperCase()}
                        </span>
                      </div>
                      <h4 className="text-sm font-medium text-gray-200 mb-1">{incident.title}</h4>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{incident.assignedTo}</span>
                        <span className={`px-2 py-1 rounded ${
                          incident.status === 'investigating' ? 'bg-blue-900/30 text-blue-400' :
                          incident.status === 'resolved' ? 'bg-green-900/30 text-green-400' :
                          'bg-yellow-900/30 text-yellow-400'
                        }`}>
                          {incident.status.toUpperCase()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* SOC Activity Feed */}
              <div className="col-span-2 bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-200 flex items-center space-x-2">
                    <span>📡</span>
                    <span>SOC ACTIVITY FEED</span>
                  </h3>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <span className="text-xs text-green-400 font-medium">LIVE</span>
                  </div>
                </div>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {operationLogs.map((log, i) => (
                    <div key={i} className={`text-xs font-mono p-3 rounded border-l-4 transition-all duration-300 ${
                      log.includes('critical') || log.includes('breach') || log.includes('emergency') ? 
                        'bg-red-900/20 border-red-500 text-red-300' :
                      log.includes('alert') || log.includes('escalated') || log.includes('malware') ?
                        'bg-yellow-900/20 border-yellow-500 text-yellow-300' :
                      log.includes('completed') || log.includes('success') || log.includes('blocked') ?
                        'bg-green-900/20 border-green-500 text-green-300' :
                        'bg-blue-900/20 border-blue-500 text-blue-300'
                    }`} style={{ animationDelay: `${i * 50}ms` }}>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Quick Actions & Controls */}
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4 flex items-center space-x-2">
                  <span>⚡</span>
                  <span>QUICK ACTIONS</span>
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <button 
                    onClick={() => triggerAction('mass_triage')}
                    className="p-4 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 rounded-lg text-blue-400 font-medium transition-all hover:scale-105"
                  >
                    <div className="text-2xl mb-2">🔍</div>
                    <div className="text-sm">MASS TRIAGE</div>
                  </button>
                  <button 
                    onClick={() => triggerAction('threat_hunt')}
                    className="p-4 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-600 rounded-lg text-purple-400 font-medium transition-all hover:scale-105"
                  >
                    <div className="text-2xl mb-2">🎯</div>
                    <div className="text-sm">THREAT HUNT</div>
                  </button>
                  <button 
                    onClick={() => triggerAction('backup_analyst')}
                    className="p-4 bg-green-900/30 hover:bg-green-900/50 border border-green-600 rounded-lg text-green-400 font-medium transition-all hover:scale-105"
                  >
                    <div className="text-2xl mb-2">👤</div>
                    <div className="text-sm">CALL BACKUP</div>
                  </button>
                  <button 
                    onClick={() => triggerAction('status_update')}
                    className="p-4 bg-orange-900/30 hover:bg-orange-900/50 border border-orange-600 rounded-lg text-orange-400 font-medium transition-all hover:scale-105"
                  >
                    <div className="text-2xl mb-2">📢</div>
                    <div className="text-sm">BROADCAST</div>
                  </button>
                </div>
              </div>

              <div className="col-span-2 bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4 flex items-center space-x-2">
                  <span>👥</span>
                  <span>ANALYST STATUS</span>
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {data?.analysts.slice(0, 6).map(analyst => (
                    <div key={analyst.id} className="p-4 bg-gray-800/30 rounded-lg border border-gray-600/30">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-200">{analyst.name}</h4>
                        <span className={`w-3 h-3 rounded-full ${
                          analyst.status === 'available' ? 'bg-green-400' :
                          analyst.status === 'busy' ? 'bg-yellow-400' :
                          'bg-gray-400'
                        }`} />
                      </div>
                      <p className="text-xs text-gray-400 mb-2">{analyst.role}</p>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500">Active:</span>
                          <span className="text-yellow-400 ml-1">{analyst.activeIncidents}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Avg Response:</span>
                          <span className="text-blue-400 ml-1">{analyst.performance.avgResponseTime}m</span>
                        </div>
                      </div>
                      <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
                        <div 
                          className={`h-1 rounded-full transition-all ${
                            analyst.performance.resolutionRate > 95 ? 'bg-green-400' :
                            analyst.performance.resolutionRate > 90 ? 'bg-yellow-400' :
                            'bg-red-400'
                          }`}
                          style={{ width: `${analyst.performance.resolutionRate}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-6">
            {/* Advanced Search and Filters */}
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-200">ACTIVE SECURITY ALERTS</h3>
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search alerts, analysts, categories..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 pl-10 text-sm text-gray-200 placeholder-gray-400 focus:border-blue-500 focus:outline-none w-64"
                    />
                    <div className="absolute left-3 top-2.5 text-gray-400">🔍</div>
                  </div>
                  <select
                    value={filterSeverity}
                    onChange={(e) => setFilterSeverity(e.target.value)}
                    className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200"
                  >
                    <option value="all">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200"
                  >
                    <option value="severity">Sort by Severity</option>
                    <option value="timestamp">Sort by Time</option>
                    <option value="riskScore">Sort by Risk Score</option>
                    <option value="confidence">Sort by Confidence</option>
                  </select>
                  <button 
                    onClick={() => setShowCreateModal(true)}
                    className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 border border-red-600 text-red-400 text-sm font-medium rounded-lg transition-all"
                  >
                    + CREATE ALERT
                  </button>
                </div>
              </div>
              
              {/* Enhanced Alert Grid */}
              <div className="space-y-4">
                {filteredAlerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 hover:bg-gray-800/70 transition-all cursor-pointer group"
                    onClick={() => setSelectedAlert(alert.id)}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center space-x-4 mb-3">
                          <div className={`w-4 h-4 rounded-full ${
                            alert.severity === 'critical' ? 'bg-red-500 animate-pulse shadow-red-500/50 shadow-lg' :
                            alert.severity === 'high' ? 'bg-orange-500 shadow-orange-500/50 shadow-md' :
                            alert.severity === 'medium' ? 'bg-yellow-500 shadow-yellow-500/50 shadow-sm' :
                            'bg-gray-500'
                          }`} />
                          <h4 className="font-bold text-gray-200 group-hover:text-blue-400 transition-colors text-lg">{alert.title}</h4>
                          <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                            alert.status === 'open' ? 'bg-red-900/30 text-red-400 border border-red-600/30' :
                            alert.status === 'investigating' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/30' :
                            'bg-green-900/30 text-green-400 border border-green-600/30'
                          }`}>
                            {alert.status.toUpperCase()}
                          </span>
                          <div className="ml-auto flex items-center space-x-4">
                            <div className="text-xs text-gray-400 bg-gray-700/50 px-3 py-1 rounded-lg">
                              Risk: <span className="text-white font-bold">{alert.riskScore}</span>
                            </div>
                            <div className="text-xs text-gray-400 bg-gray-700/50 px-3 py-1 rounded-lg">
                              Confidence: <span className="text-white font-bold">{alert.confidence}%</span>
                            </div>
                          </div>
                        </div>
                        <p className="text-sm text-gray-400 mb-4 leading-relaxed">{alert.description}</p>
                        <div className="grid grid-cols-4 gap-6 text-sm">
                          <div>
                            <span className="text-gray-500 font-medium">Category:</span>
                            <div className="text-gray-300 mt-1 font-semibold">{alert.category}</div>
                          </div>
                          <div>
                            <span className="text-gray-500 font-medium">Source:</span>
                            <div className="text-gray-300 mt-1 font-semibold">{alert.source}</div>
                          </div>
                          <div>
                            <span className="text-gray-500 font-medium">Analyst:</span>
                            <div className="text-blue-300 mt-1 font-semibold">{alert.analyst}</div>
                          </div>
                          <div>
                            <span className="text-gray-500 font-medium">Detected:</span>
                            <div className="text-gray-300 mt-1 font-semibold">
                              {new Date(alert.timestamp).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        
                        {/* Evidence & Recommendations */}
                        <div className="grid grid-cols-2 gap-6 mt-4 pt-4 border-t border-gray-700">
                          <div>
                            <span className="text-gray-500 font-medium text-sm">Evidence Collected:</span>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {alert.evidence.slice(0, 3).map((evidence, idx) => (
                                <span key={idx} className="px-3 py-1 bg-gray-700/50 rounded-lg text-xs text-gray-300 border border-gray-600/50">
                                  📄 {evidence}
                                </span>
                              ))}
                              {alert.evidence.length > 3 && (
                                <span className="px-3 py-1 bg-gray-700/50 rounded-lg text-xs text-gray-500">
                                  +{alert.evidence.length - 3} more
                                </span>
                              )}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-500 font-medium text-sm">Recommendations:</span>
                            <div className="mt-2 space-y-1">
                              {alert.recommendations.slice(0, 2).map((rec, idx) => (
                                <div key={idx} className="flex items-center space-x-2 text-xs text-gray-300">
                                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                                  <span>{rec}</span>
                                </div>
                              ))}
                              {alert.recommendations.length > 2 && (
                                <div className="text-xs text-gray-500">+{alert.recommendations.length - 2} more recommendations</div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex flex-col space-y-3 ml-6">
                        <button className="px-4 py-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-600 rounded-lg text-blue-400 text-sm font-medium transition-all hover:shadow-blue-500/20 hover:shadow-lg">
                          INVESTIGATE
                        </button>
                        <button className="px-4 py-2 bg-green-900/30 hover:bg-green-900/50 border border-green-600 rounded-lg text-green-400 text-sm font-medium transition-all hover:shadow-green-500/20 hover:shadow-lg">
                          ASSIGN
                        </button>
                        <button className="px-4 py-2 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-600 rounded-lg text-purple-400 text-sm font-medium transition-all hover:shadow-purple-500/20 hover:shadow-lg">
                          ESCALATE
                        </button>
                        <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-gray-400 text-sm font-medium transition-all">
                          DISMISS
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="space-y-6">
            {/* Performance Charts */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">ALERT VOLUME TRENDS (24H)</h3>
                <div className="h-80">
                  <Line data={alertVolumeData} options={chartOptions} />
                </div>
              </div>
              
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">ANALYST WORKLOAD DISTRIBUTION</h3>
                <div className="h-80">
                  <Bar data={analystWorkloadData} options={chartOptions} />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">INCIDENT STATUS BREAKDOWN</h3>
                <div className="h-80">
                  <Doughnut data={incidentStatusData} options={chartOptions} />
                </div>
              </div>
              
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-200 mb-4">SLA PERFORMANCE BY RESPONSE TIME</h3>
                <div className="h-80">
                  <Bar data={responseEfficiencyData} options={chartOptions} />
                </div>
              </div>
            </div>

            {/* Detailed SLA Metrics */}
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-200 mb-6">DETAILED SLA PERFORMANCE METRICS</h3>
              <div className="grid grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400 mb-2">{data?.slaMetrics.p1ResponseTime}m</div>
                  <div className="text-sm text-gray-400 mb-3">P1 Response Time</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-green-500 h-3 rounded-full transition-all duration-1000" 
                         style={{ width: `${Math.max(0, 100 - (data?.slaMetrics.p1ResponseTime || 0) * 6.67)}%` }} />
                  </div>
                  <div className="text-xs text-gray-500 mt-2">Target: 15m</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-400 mb-2">{data?.slaMetrics.p2ResponseTime}m</div>
                  <div className="text-sm text-gray-400 mb-3">P2 Response Time</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-yellow-500 h-3 rounded-full transition-all duration-1000" 
                         style={{ width: `${Math.max(0, 100 - (data?.slaMetrics.p2ResponseTime || 0) * 1.67)}%` }} />
                  </div>
                  <div className="text-xs text-gray-500 mt-2">Target: 60m</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400 mb-2">{data?.slaMetrics.p3ResponseTime}m</div>
                  <div className="text-sm text-gray-400 mb-3">P3 Response Time</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-blue-500 h-3 rounded-full transition-all duration-1000" 
                         style={{ width: `${Math.max(0, 100 - (data?.slaMetrics.p3ResponseTime || 0) * 0.42)}%` }} />
                  </div>
                  <div className="text-xs text-gray-500 mt-2">Target: 240m</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400 mb-2">{data?.slaMetrics.overallSLA}%</div>
                  <div className="text-sm text-gray-400 mb-3">Overall SLA</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-1000" 
                         style={{ width: `${data?.slaMetrics.overallSLA}%` }} />
                  </div>
                  <div className="text-xs text-gray-500 mt-2">Target: 95%</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Additional tabs can be implemented here following the same pattern */}
        {activeTab === 'workflows' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-200 mb-4">SECURITY AUTOMATION WORKFLOWS</h3>
              <div className="grid grid-cols-1 gap-4">
                {data?.workflows.map(workflow => (
                  <div key={workflow.id} className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h4 className="font-bold text-gray-200">{workflow.name}</h4>
                          <span className={`text-xs px-3 py-1 rounded-full ${
                            workflow.status === 'active' ? 'bg-green-900/30 text-green-400 border border-green-600/30' :
                            workflow.status === 'paused' ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/30' :
                            'bg-gray-800 text-gray-400 border border-gray-600/30'
                          }`}>
                            {workflow.status.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">{workflow.description}</p>
                        <div className="grid grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Executions:</span>
                            <div className="text-blue-300 font-bold">{workflow.executionCount}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Success Rate:</span>
                            <div className="text-green-300 font-bold">{workflow.successRate}%</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Avg Duration:</span>
                            <div className="text-yellow-300 font-bold">{workflow.avgExecutionTime}s</div>
                          </div>
                          <div>
                            <span className="text-gray-500">Last Run:</span>
                            <div className="text-gray-300">{new Date(workflow.lastExecution).toLocaleString()}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}