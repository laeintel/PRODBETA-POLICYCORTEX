'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Cloud, Shield, Lock, GitBranch, Settings, Database,
  CheckCircle, XCircle, AlertTriangle, Zap, Globe,
  Users, Key, FileCode, Terminal, Eye, Link
} from 'lucide-react';
import { useButtonActions } from '@/lib/button-actions';
import { toast } from 'react-hot-toast';

// Multi-cloud policy mappings
const unifiedPolicies = [
  {
    id: 1,
    name: 'Encryption at Rest',
    description: 'All storage must be encrypted using AES-256',
    aws: { enabled: true, syntax: 'AWS::S3::BucketEncryption', compliance: 98 },
    azure: { enabled: true, syntax: 'Microsoft.Storage/encryption', compliance: 95 },
    gcp: { enabled: true, syntax: 'google.storage.bucket.encryption', compliance: 92 },
    status: 'active',
    violations: 23,
    lastUpdated: '2 hours ago'
  },
  {
    id: 2,
    name: 'Network Segmentation',
    description: 'Production and development must be isolated',
    aws: { enabled: true, syntax: 'AWS::EC2::SecurityGroup', compliance: 89 },
    azure: { enabled: true, syntax: 'Microsoft.Network/networkSecurityGroups', compliance: 91 },
    gcp: { enabled: false, syntax: 'google.compute.firewall', compliance: 0 },
    status: 'active',
    violations: 45,
    lastUpdated: '1 day ago'
  },
  {
    id: 3,
    name: 'Identity & Access',
    description: 'Enforce MFA and least privilege access',
    aws: { enabled: true, syntax: 'AWS::IAM::Policy', compliance: 94 },
    azure: { enabled: true, syntax: 'Microsoft.Authorization/roleDefinitions', compliance: 96 },
    gcp: { enabled: true, syntax: 'google.iam.policy', compliance: 93 },
    status: 'active',
    violations: 12,
    lastUpdated: '4 hours ago'
  },
];

// Cross-platform visibility data
const platformInventory = {
  aws: { 
    accounts: 12, 
    regions: 8, 
    resources: 1243, 
    compliant: 1089,
    violations: 154,
    services: ['EC2', 'S3', 'RDS', 'Lambda', 'EKS']
  },
  azure: { 
    subscriptions: 8, 
    regions: 6, 
    resources: 987, 
    compliant: 892,
    violations: 95,
    services: ['VMs', 'Storage', 'SQL', 'AKS', 'Functions']
  },
  gcp: { 
    projects: 4, 
    regions: 4, 
    resources: 456, 
    compliant: 412,
    violations: 44,
    services: ['Compute', 'Storage', 'BigQuery', 'GKE', 'Cloud Run']
  },
};

// Policy conflicts
const policyConflicts = [
  {
    id: 1,
    type: 'Definition Mismatch',
    policy: 'Data Residency',
    platforms: ['AWS', 'Azure'],
    description: 'AWS allows US-East, Azure restricts to US-West',
    severity: 'high',
    resolution: 'Align to most restrictive (Azure)'
  },
  {
    id: 2,
    type: 'Implementation Gap',
    policy: 'Backup Retention',
    platforms: ['Azure', 'GCP'],
    description: 'Azure: 30 days, GCP: Not configured',
    severity: 'medium',
    resolution: 'Configure GCP to match Azure'
  },
];

export default function UnifiedPoliciesPage() {
  const router = useRouter();
  const actions = useButtonActions(router);
  const [selectedCloud, setSelectedCloud] = useState('all');
  const [policyLanguage, setPolicyLanguage] = useState('unified');

  const totalResources = Object.values(platformInventory).reduce((sum, p) => sum + p.resources, 0);
  const totalCompliant = Object.values(platformInventory).reduce((sum, p) => sum + p.compliant, 0);
  const totalViolations = Object.values(platformInventory).reduce((sum, p) => sum + p.violations, 0);
  const complianceRate = ((totalCompliant / totalResources) * 100).toFixed(1);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">Unified Multi-Cloud Governance</h1>
          <p className="text-gray-600 mt-1">Single policy framework across AWS, Azure, and GCP</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline"
            onClick={() => actions.exportPolicies()}
          >
            <FileCode className="w-4 h-4 mr-2" />
            Export Policies
          </Button>
          <Button 
            className="bg-blue-600 hover:bg-blue-700"
            onClick={() => actions.syncPolicies()}
          >
            <Zap className="w-4 h-4 mr-2" />
            Sync All Clouds
          </Button>
        </div>
      </div>

      {/* Policy Conflicts Alert */}
      {policyConflicts.length > 0 && (
        <Alert className="border-orange-200 bg-orange-50">
          <AlertTriangle className="h-4 w-4 text-orange-600" />
          <AlertDescription className="text-orange-800">
            <strong>Policy Conflicts Detected:</strong> {policyConflicts.length} inconsistencies found 
            between cloud platforms. Review and align policies to maintain unified governance.
          </AlertDescription>
        </Alert>
      )}

      {/* Cross-Platform Overview */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Total Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{totalResources.toLocaleString()}</div>
            <div className="text-sm text-gray-600 mt-1">Across 3 clouds</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Compliance Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">{complianceRate}%</div>
            <div className="text-sm text-gray-600 mt-1">{totalCompliant} compliant</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Policy Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-600">{totalViolations}</div>
            <div className="text-sm text-gray-600 mt-1">Needs attention</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Unified Policies</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{unifiedPolicies.length}</div>
            <div className="text-sm text-gray-600 mt-1">Active policies</div>
          </CardContent>
        </Card>
      </div>

      {/* Platform Inventory */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Globe className="w-5 h-5 mr-2" />
            Multi-Cloud Resource Inventory
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-6">
            {/* AWS */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Cloud className="w-6 h-6 text-orange-500" />
                  <span className="font-semibold text-lg">AWS</span>
                </div>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Accounts:</span>
                  <span className="font-medium">{platformInventory.aws.accounts}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Regions:</span>
                  <span className="font-medium">{platformInventory.aws.regions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Resources:</span>
                  <span className="font-medium">{platformInventory.aws.resources}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Compliance:</span>
                  <span className="font-medium text-green-600">
                    {((platformInventory.aws.compliant / platformInventory.aws.resources) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-500">Top Services:</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {platformInventory.aws.services.map(service => (
                    <span key={service} className="text-xs px-2 py-1 bg-orange-100 text-orange-700 rounded">
                      {service}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Azure */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Cloud className="w-6 h-6 text-blue-500" />
                  <span className="font-semibold text-lg">Azure</span>
                </div>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Subscriptions:</span>
                  <span className="font-medium">{platformInventory.azure.subscriptions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Regions:</span>
                  <span className="font-medium">{platformInventory.azure.regions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Resources:</span>
                  <span className="font-medium">{platformInventory.azure.resources}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Compliance:</span>
                  <span className="font-medium text-green-600">
                    {((platformInventory.azure.compliant / platformInventory.azure.resources) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-500">Top Services:</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {platformInventory.azure.services.map(service => (
                    <span key={service} className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                      {service}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* GCP */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Cloud className="w-6 h-6 text-green-500" />
                  <span className="font-semibold text-lg">GCP</span>
                </div>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Projects:</span>
                  <span className="font-medium">{platformInventory.gcp.projects}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Regions:</span>
                  <span className="font-medium">{platformInventory.gcp.regions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Resources:</span>
                  <span className="font-medium">{platformInventory.gcp.resources}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Compliance:</span>
                  <span className="font-medium text-green-600">
                    {((platformInventory.gcp.compliant / platformInventory.gcp.resources) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-500">Top Services:</div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {platformInventory.gcp.services.map(service => (
                    <span key={service} className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">
                      {service}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Unified Policy Management */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            Unified Policy Framework
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="policies" className="w-full">
            <TabsList>
              <TabsTrigger value="policies">Active Policies</TabsTrigger>
              <TabsTrigger value="conflicts">Conflicts</TabsTrigger>
              <TabsTrigger value="translation">Policy Translation</TabsTrigger>
            </TabsList>

            <TabsContent value="policies" className="space-y-4">
              {unifiedPolicies.map((policy) => (
                <div key={policy.id} className="p-4 border rounded-lg">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <div className="font-semibold text-lg">{policy.name}</div>
                      <div className="text-sm text-gray-600">{policy.description}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      {policy.violations > 0 && (
                        <span className="text-sm px-2 py-1 bg-red-100 text-red-700 rounded">
                          {policy.violations} violations
                        </span>
                      )}
                      <span className="text-sm px-2 py-1 bg-green-100 text-green-700 rounded">
                        {policy.status}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="flex items-center justify-between p-2 bg-orange-50 rounded">
                      <div className="flex items-center gap-2">
                        <Cloud className="w-4 h-4 text-orange-500" />
                        <span className="text-sm font-medium">AWS</span>
                      </div>
                      {policy.aws.enabled ? (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{policy.aws.compliance}%</span>
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        </div>
                      ) : (
                        <XCircle className="w-4 h-4 text-gray-400" />
                      )}
                    </div>

                    <div className="flex items-center justify-between p-2 bg-blue-50 rounded">
                      <div className="flex items-center gap-2">
                        <Cloud className="w-4 h-4 text-blue-500" />
                        <span className="text-sm font-medium">Azure</span>
                      </div>
                      {policy.azure.enabled ? (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{policy.azure.compliance}%</span>
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        </div>
                      ) : (
                        <XCircle className="w-4 h-4 text-gray-400" />
                      )}
                    </div>

                    <div className="flex items-center justify-between p-2 bg-green-50 rounded">
                      <div className="flex items-center gap-2">
                        <Cloud className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium">GCP</span>
                      </div>
                      {policy.gcp.enabled ? (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{policy.gcp.compliance}%</span>
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        </div>
                      ) : (
                        <XCircle className="w-4 h-4 text-gray-400" />
                      )}
                    </div>
                  </div>

                  <div className="flex justify-between items-center mt-3 pt-3 border-t">
                    <span className="text-xs text-gray-500">Last updated: {policy.lastUpdated}</span>
                    <div className="flex gap-2">
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => actions.viewPolicyDetails(`unified-${policy.id}`)}
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        View Details
                      </Button>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => {
                          toast(`Configuring ${policy.name}`, { icon: '⚙️' });
                          router.push(`/governance/policies/configure?id=${policy.id}`);
                        }}
                      >
                        <Settings className="w-3 h-3 mr-1" />
                        Configure
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="conflicts" className="space-y-4">
              {policyConflicts.map((conflict) => (
                <div key={conflict.id} className="p-4 border rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className={`w-5 h-5 ${
                          conflict.severity === 'high' ? 'text-red-500' :
                          conflict.severity === 'medium' ? 'text-orange-500' :
                          'text-yellow-500'
                        }`} />
                        <span className="font-semibold">{conflict.type}</span>
                      </div>
                      <div className="text-sm text-gray-600 mb-1">
                        Policy: <span className="font-medium">{conflict.policy}</span>
                      </div>
                      <div className="text-sm text-gray-600 mb-2">
                        Platforms: {conflict.platforms.join(' ↔ ')}
                      </div>
                      <div className="text-sm mb-2">{conflict.description}</div>
                      <div className="text-sm text-green-600">
                        Recommended: {conflict.resolution}
                      </div>
                    </div>
                    <Button 
                      size="sm" 
                      className="bg-green-600 hover:bg-green-700"
                      onClick={() => actions.alignPolicies()}
                    >
                      <Link className="w-3 h-3 mr-1" />
                      Align Policies
                    </Button>
                  </div>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="translation" className="space-y-4">
              <div className="p-4 bg-gray-50 rounded">
                <div className="mb-4">
                  <label className="text-sm font-medium">Policy Definition (Natural Language)</label>
                  <textarea 
                    className="w-full mt-2 p-3 border rounded"
                    rows={3}
                    placeholder="Example: All databases must be encrypted at rest using AES-256 and have automated backups enabled"
                  />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-sm font-medium mb-2 flex items-center">
                      <Cloud className="w-4 h-4 mr-1 text-orange-500" />
                      AWS Translation
                    </div>
                    <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
{`{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Action": "rds:CreateDBInstance",
    "Resource": "*",
    "Condition": {
      "Bool": {
        "rds:StorageEncrypted": false
      }
    }
  }]
}`}
                    </pre>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-2 flex items-center">
                      <Cloud className="w-4 h-4 mr-1 text-blue-500" />
                      Azure Translation
                    </div>
                    <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
{`{
  "if": {
    "field": "Microsoft.Sql/servers",
    "exists": "true"
  },
  "then": {
    "effect": "deny",
    "details": {
      "encryption": "required"
    }
  }
}`}
                    </pre>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-2 flex items-center">
                      <Cloud className="w-4 h-4 mr-1 text-green-500" />
                      GCP Translation
                    </div>
                    <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
{`constraint: sql.restrictAuthorizedNetworks
params:
  mode: "deny"
  encryptionRequired: true
  backupConfiguration:
    enabled: true
    startTime: "03:00"`}
                    </pre>
                  </div>
                </div>
                <Button 
                  className="mt-4 bg-blue-600 hover:bg-blue-700"
                  onClick={() => actions.deployPolicy()}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Deploy to All Clouds
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}