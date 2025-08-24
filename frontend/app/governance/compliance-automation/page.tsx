'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, AlertTriangle, CheckCircle, XCircle, Clock,
  FileText, Activity, Lock, RefreshCw, Zap, TrendingUp,
  Database, GitBranch, Users, Settings, AlertCircle
} from 'lucide-react';
import { useButtonActions } from '@/lib/button-actions';
import { toast } from 'react-hot-toast';

// Compliance frameworks with real-time status
const complianceFrameworks = [
  { 
    name: 'NIST 800-53', 
    controls: 110, 
    passed: 95, 
    failed: 15, 
    coverage: 86.4,
    lastAudit: '2 hours ago',
    nextAudit: 'In 5 days',
    trend: 'improving'
  },
  { 
    name: 'CIS Azure', 
    controls: 92, 
    passed: 78, 
    failed: 14, 
    coverage: 84.8,
    lastAudit: '4 hours ago',
    nextAudit: 'In 3 days',
    trend: 'stable'
  },
  { 
    name: 'ISO 27001', 
    controls: 114, 
    passed: 102, 
    failed: 12, 
    coverage: 89.5,
    lastAudit: '1 day ago',
    nextAudit: 'In 7 days',
    trend: 'improving'
  },
  { 
    name: 'HIPAA', 
    controls: 45, 
    passed: 43, 
    failed: 2, 
    coverage: 95.6,
    lastAudit: '6 hours ago',
    nextAudit: 'Tomorrow',
    trend: 'stable'
  },
  { 
    name: 'PCI DSS', 
    controls: 78, 
    passed: 71, 
    failed: 7, 
    coverage: 91.0,
    lastAudit: '12 hours ago',
    nextAudit: 'In 4 days',
    trend: 'declining'
  },
];

// Configuration drift items
const configDrifts = [
  {
    id: 1,
    resource: 'prod-web-sg',
    type: 'Security Group',
    drift: 'Port 22 opened',
    severity: 'critical',
    detected: '15 minutes ago',
    autoRemediate: true
  },
  {
    id: 2,
    resource: 's3-backup-bucket',
    type: 'S3 Bucket',
    drift: 'Encryption disabled',
    severity: 'high',
    detected: '1 hour ago',
    autoRemediate: true
  },
  {
    id: 3,
    resource: 'db-prod-instance',
    type: 'RDS Database',
    drift: 'Public access enabled',
    severity: 'critical',
    detected: '2 hours ago',
    autoRemediate: false
  },
  {
    id: 4,
    resource: 'app-load-balancer',
    type: 'Load Balancer',
    drift: 'SSL certificate expired',
    severity: 'high',
    detected: '3 hours ago',
    autoRemediate: true
  },
];

// Policy violations
const policyViolations = [
  {
    id: 1,
    policy: 'Require Encryption at Rest',
    resources: 23,
    severity: 'high',
    compliance: 'NIST, ISO 27001',
    remediationTime: '< 1 hour'
  },
  {
    id: 2,
    policy: 'Enforce MFA for Admin',
    resources: 8,
    severity: 'critical',
    compliance: 'CIS, PCI DSS',
    remediationTime: 'Immediate'
  },
  {
    id: 3,
    policy: 'Tag Resources',
    resources: 156,
    severity: 'medium',
    compliance: 'Internal',
    remediationTime: '< 2 hours'
  },
];

export default function ComplianceAutomationPage() {
  const router = useRouter();
  const actions = useButtonActions(router);
  const [autoRemediateEnabled, setAutoRemediateEnabled] = useState(true);
  const [continuousMonitoring, setContinuousMonitoring] = useState(true);
  const [complianceScore, setComplianceScore] = useState(87.5);
  const [remediationProgress, setRemediationProgress] = useState(0);

  useEffect(() => {
    // Simulate remediation progress
    if (remediationProgress < 100) {
      const timer = setTimeout(() => {
        setRemediationProgress(prev => Math.min(prev + 5, 100));
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [remediationProgress]);

  const handleAutoRemediate = (driftId: number) => {
    setRemediationProgress(10);
    // Simulate remediation
    setTimeout(() => {
      setComplianceScore(prev => Math.min(prev + 0.5, 100));
    }, 2000);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Compliance Automation</h1>
          <p className="text-gray-600 mt-1">Continuous compliance monitoring and automated remediation</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant={continuousMonitoring ? "default" : "outline"}
            onClick={() => setContinuousMonitoring(!continuousMonitoring)}
          >
            <Activity className="w-4 h-4 mr-2" />
            {continuousMonitoring ? 'Monitoring Active' : 'Enable Monitoring'}
          </Button>
          <Button 
            variant={autoRemediateEnabled ? "default" : "outline"}
            onClick={() => setAutoRemediateEnabled(!autoRemediateEnabled)}
            className="bg-green-600 hover:bg-green-700"
          >
            <Zap className="w-4 h-4 mr-2" />
            {autoRemediateEnabled ? 'Auto-Remediation ON' : 'Auto-Remediation OFF'}
          </Button>
        </div>
      </div>

      {/* Critical Alert */}
      {configDrifts.filter(d => d.severity === 'critical').length > 0 && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            <strong>Critical Configuration Drift Detected:</strong> {configDrifts.filter(d => d.severity === 'critical').length} critical 
            security configurations have drifted from baseline. Immediate action required.
          </AlertDescription>
        </Alert>
      )}

      {/* Compliance Score Overview */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Overall Compliance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{complianceScore}%</div>
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <Progress value={complianceScore} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Controls Passed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">389</div>
            <div className="text-sm text-gray-600">Out of 439 total</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Active Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-600">50</div>
            <div className="text-sm text-gray-600">187 auto-remediated today</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Time to Compliance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">4.2h</div>
            <div className="text-sm text-gray-600">Average remediation time</div>
          </CardContent>
        </Card>
      </div>

      {/* Compliance Frameworks Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            Multi-Framework Compliance Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {complianceFrameworks.map((framework) => (
              <div key={framework.name} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-medium text-lg">{framework.name}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      framework.trend === 'improving' ? 'bg-green-100 text-green-700' :
                      framework.trend === 'declining' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {framework.trend}
                    </span>
                  </div>
                  <div className="flex gap-6 text-sm text-gray-600">
                    <span>Controls: {framework.controls}</span>
                    <span className="text-green-600">Passed: {framework.passed}</span>
                    <span className="text-red-600">Failed: {framework.failed}</span>
                    <span>Last Audit: {framework.lastAudit}</span>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-2xl font-bold">{framework.coverage}%</div>
                    <div className="text-xs text-gray-500">Coverage</div>
                  </div>
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => actions.runAudit(framework.name)}
                  >
                    Audit Now
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-6">
        {/* Configuration Drift Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <RefreshCw className="w-5 h-5 mr-2 text-orange-500" />
              Configuration Drift Detection
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {configDrifts.map((drift) => (
                <div key={drift.id} className="flex items-center justify-between p-3 border rounded">
                  <div className="flex items-start gap-3">
                    <div className={`w-2 h-2 rounded-full mt-2 ${
                      drift.severity === 'critical' ? 'bg-red-500' :
                      drift.severity === 'high' ? 'bg-orange-500' :
                      'bg-yellow-500'
                    }`} />
                    <div>
                      <div className="font-medium">{drift.resource}</div>
                      <div className="text-sm text-gray-600">{drift.type} • {drift.drift}</div>
                      <div className="text-xs text-gray-500">{drift.detected}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {drift.autoRemediate ? (
                      <Button 
                        size="sm" 
                        className="bg-green-600 hover:bg-green-700"
                        onClick={() => handleAutoRemediate(drift.id)}
                      >
                        <Zap className="w-3 h-3 mr-1" />
                        Fix Now
                      </Button>
                    ) : (
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => {
                          toast(`Reviewing drift for ${drift.resource}`, { icon: '🔍' });
                          router.push(`/governance/drift-detection?resource=${drift.id}`);
                        }}
                      >
                        Review
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {remediationProgress > 0 && remediationProgress < 100 && (
              <div className="mt-4 p-3 bg-blue-50 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Auto-remediation in progress...</span>
                  <span className="text-sm">{remediationProgress}%</span>
                </div>
                <Progress value={remediationProgress} />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Policy Violations */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-red-500" />
              Policy Violations Requiring Action
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {policyViolations.map((violation) => (
                <div key={violation.id} className="p-3 border rounded">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <div className="font-medium">{violation.policy}</div>
                      <div className="text-sm text-gray-600 mt-1">
                        Affects {violation.resources} resources • {violation.compliance}
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded ${
                      violation.severity === 'critical' ? 'bg-red-100 text-red-700' :
                      violation.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {violation.severity}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-500">
                      Est. remediation: {violation.remediationTime}
                    </span>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => actions.remediateIssue(`policy-${violation.id}`)}
                    >
                      Remediate
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Automated Evidence Collection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <FileText className="w-5 h-5 mr-2 text-blue-500" />
            Automated Evidence Collection
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded">
              <div className="flex items-center justify-between mb-2">
                <Database className="w-8 h-8 text-blue-600" />
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="font-medium">Configuration Snapshots</div>
              <div className="text-sm text-gray-600 mt-1">2,847 configs captured</div>
              <div className="text-xs text-gray-500 mt-2">Last: 2 minutes ago</div>
            </div>

            <div className="p-4 bg-green-50 rounded">
              <div className="flex items-center justify-between mb-2">
                <GitBranch className="w-8 h-8 text-green-600" />
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="font-medium">Change Logs</div>
              <div className="text-sm text-gray-600 mt-1">15,234 changes tracked</div>
              <div className="text-xs text-gray-500 mt-2">Real-time collection</div>
            </div>

            <div className="p-4 bg-purple-50 rounded">
              <div className="flex items-center justify-between mb-2">
                <Users className="w-8 h-8 text-purple-600" />
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="font-medium">Access Reviews</div>
              <div className="text-sm text-gray-600 mt-1">892 permissions audited</div>
              <div className="text-xs text-gray-500 mt-2">Next review: Tomorrow</div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-gray-50 rounded flex items-center justify-between">
            <div>
              <div className="font-medium">Audit Package Ready</div>
              <div className="text-sm text-gray-600">All evidence compiled for SOC 2 Type II audit</div>
            </div>
            <Button 
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => actions.downloadEvidence()}
            >
              <FileText className="w-4 h-4 mr-2" />
              Download Evidence
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}