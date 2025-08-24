'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  GitBranch, Shield, Zap, CheckCircle, XCircle, AlertTriangle,
  Terminal, Code, Package, PlayCircle, Pause, RefreshCw,
  Lock, Cpu, Cloud, Database, Users, FileCode
} from 'lucide-react';

// Pipeline status data
const pipelineStatus = [
  {
    id: 1,
    name: 'main',
    commit: 'feat: Add payment gateway',
    author: 'sarah.chen',
    status: 'running',
    stage: 'security-scan',
    compliance: 92,
    violations: 3,
    startTime: '2 minutes ago'
  },
  {
    id: 2,
    name: 'feature/user-auth',
    commit: 'fix: Session timeout issue',
    author: 'mike.johnson',
    status: 'failed',
    stage: 'policy-check',
    compliance: 78,
    violations: 8,
    startTime: '15 minutes ago'
  },
  {
    id: 3,
    name: 'hotfix/security-patch',
    commit: 'security: Patch CVE-2024-1234',
    author: 'alex.kumar',
    status: 'passed',
    stage: 'deploy',
    compliance: 100,
    violations: 0,
    startTime: '1 hour ago'
  },
];

// Policy gates
const policyGates = [
  {
    name: 'Pre-commit',
    enabled: true,
    rules: 12,
    blockers: 3,
    passed: 892,
    failed: 45,
    avgTime: '0.3s'
  },
  {
    name: 'Build',
    enabled: true,
    rules: 24,
    blockers: 8,
    passed: 756,
    failed: 23,
    avgTime: '2.1s'
  },
  {
    name: 'Pre-deploy',
    enabled: true,
    rules: 38,
    blockers: 15,
    passed: 623,
    failed: 12,
    avgTime: '5.4s'
  },
  {
    name: 'Runtime',
    enabled: true,
    rules: 18,
    blockers: 5,
    passed: 1245,
    failed: 8,
    avgTime: 'continuous'
  },
];

// Security findings
const securityFindings = [
  {
    id: 1,
    type: 'Secret Exposed',
    severity: 'critical',
    file: 'config/database.yml',
    line: 23,
    description: 'AWS access key found in code',
    autoFix: true
  },
  {
    id: 2,
    type: 'Vulnerable Dependency',
    severity: 'high',
    file: 'package.json',
    line: 45,
    description: 'lodash@4.17.19 has known vulnerability',
    autoFix: true
  },
  {
    id: 3,
    type: 'Insecure Configuration',
    severity: 'medium',
    file: 'terraform/main.tf',
    line: 89,
    description: 'S3 bucket missing encryption',
    autoFix: true
  },
];

export default function PolicyAsCodePage() {
  const [selectedPipeline, setSelectedPipeline] = useState(pipelineStatus[0]);
  const [autoRemediationEnabled, setAutoRemediationEnabled] = useState(true);
  const [blockOnViolation, setBlockOnViolation] = useState(true);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Policy as Code & DevSecOps</h1>
          <p className="text-gray-600 mt-1">Automated security and compliance in CI/CD pipelines</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant={blockOnViolation ? "default" : "outline"}
            onClick={() => setBlockOnViolation(!blockOnViolation)}
            className="bg-red-600 hover:bg-red-700"
          >
            <Lock className="w-4 h-4 mr-2" />
            {blockOnViolation ? 'Blocking Enabled' : 'Blocking Disabled'}
          </Button>
          <Button 
            variant={autoRemediationEnabled ? "default" : "outline"}
            onClick={() => setAutoRemediationEnabled(!autoRemediationEnabled)}
            className="bg-green-600 hover:bg-green-700"
          >
            <Zap className="w-4 h-4 mr-2" />
            {autoRemediationEnabled ? 'Auto-Fix ON' : 'Auto-Fix OFF'}
          </Button>
        </div>
      </div>

      {/* Security Alert */}
      {securityFindings.filter(f => f.severity === 'critical').length > 0 && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">
            <strong>Critical Security Issue:</strong> Secret exposed in code. 
            {autoRemediationEnabled ? ' Auto-remediation initiated.' : ' Manual intervention required.'}
          </AlertDescription>
        </Alert>
      )}

      {/* Pipeline Overview */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Active Pipelines</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{pipelineStatus.length}</div>
            <div className="text-sm text-gray-600 mt-1">
              {pipelineStatus.filter(p => p.status === 'running').length} running
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Policy Gates</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">
              {policyGates.filter(g => g.enabled).length}/{policyGates.length}
            </div>
            <div className="text-sm text-gray-600 mt-1">Active gates</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Compliance Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">94.2%</div>
            <div className="text-sm text-gray-600 mt-1">Last 24 hours</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Auto-Fixed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-600">287</div>
            <div className="text-sm text-gray-600 mt-1">Issues today</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-2 gap-6">
        {/* Pipeline Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <GitBranch className="w-5 h-5 mr-2" />
              CI/CD Pipeline Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {pipelineStatus.map((pipeline) => (
                <div 
                  key={pipeline.id} 
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedPipeline.id === pipeline.id ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedPipeline(pipeline)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Code className="w-4 h-4 text-gray-500" />
                        <span className="font-medium">{pipeline.name}</span>
                        <span className={`text-xs px-2 py-1 rounded ${
                          pipeline.status === 'running' ? 'bg-blue-100 text-blue-700' :
                          pipeline.status === 'failed' ? 'bg-red-100 text-red-700' :
                          'bg-green-100 text-green-700'
                        }`}>
                          {pipeline.status}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {pipeline.commit} • by {pipeline.author}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Stage: {pipeline.stage} • {pipeline.startTime}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold">{pipeline.compliance}%</div>
                      <div className="text-xs text-gray-500">compliance</div>
                      {pipeline.violations > 0 && (
                        <div className="text-xs text-red-600 mt-1">
                          {pipeline.violations} violations
                        </div>
                      )}
                    </div>
                  </div>
                  {pipeline.status === 'running' && (
                    <div className="mt-3 flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
                      </div>
                      <PlayCircle className="w-4 h-4 text-blue-500 animate-spin" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Security Findings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="w-5 h-5 mr-2" />
              Security Findings & Auto-Remediation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {securityFindings.map((finding) => (
                <div key={finding.id} className="p-3 border rounded">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${
                        finding.severity === 'critical' ? 'bg-red-500' :
                        finding.severity === 'high' ? 'bg-orange-500' :
                        'bg-yellow-500'
                      }`} />
                      <span className="font-medium">{finding.type}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded ${
                      finding.severity === 'critical' ? 'bg-red-100 text-red-700' :
                      finding.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {finding.severity}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {finding.file}:{finding.line}
                  </div>
                  <div className="text-sm mt-1">{finding.description}</div>
                  {finding.autoFix && (
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-xs text-green-600">Auto-fix available</span>
                      <Button size="sm" className="h-7 text-xs bg-green-600 hover:bg-green-700">
                        <Zap className="w-3 h-3 mr-1" />
                        Apply Fix
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Policy Gates Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Lock className="w-5 h-5 mr-2" />
            Policy Gates Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            {policyGates.map((gate) => (
              <div key={gate.name} className="p-4 border rounded-lg">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="font-medium">{gate.name}</div>
                    <div className="text-sm text-gray-600">{gate.rules} rules</div>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${
                    gate.enabled ? 'bg-green-500' : 'bg-gray-300'
                  }`} />
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Blockers:</span>
                    <span className="font-medium text-red-600">{gate.blockers}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Passed:</span>
                    <span className="font-medium text-green-600">{gate.passed}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Failed:</span>
                    <span className="font-medium text-red-600">{gate.failed}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Time:</span>
                    <span className="font-medium">{gate.avgTime}</span>
                  </div>
                </div>
                <Button size="sm" variant="outline" className="w-full mt-3">
                  Configure
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Developer Experience */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Terminal className="w-5 h-5 mr-2" />
            Developer-Friendly Feedback
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-900 text-gray-100 p-4 rounded font-mono text-sm">
            <div className="text-green-400">$ git push origin feature/payment-gateway</div>
            <div className="mt-2">🔍 PolicyCortex Security Scan Started...</div>
            <div className="mt-1 text-yellow-400">⚠️  Warning: Potential security issue detected</div>
            <div className="mt-2 pl-4">
              <div>File: config/database.yml:23</div>
              <div>Issue: AWS access key exposed in configuration</div>
              <div>Severity: CRITICAL</div>
            </div>
            <div className="mt-2 text-green-400">✨ Auto-remediation available!</div>
            <div className="pl-4">
              <div>→ Move secret to environment variable</div>
              <div>→ Add to .gitignore</div>
              <div>→ Rotate compromised key</div>
            </div>
            <div className="mt-2">Would you like to apply fixes? [Y/n]</div>
            <div className="mt-1 text-green-400">✅ Fixes applied successfully!</div>
            <div className="mt-2">📊 Compliance Score: 94% (3 minor issues remaining)</div>
            <div>🚀 Pipeline continues with enhanced security...</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}