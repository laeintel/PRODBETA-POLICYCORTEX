'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Link,
  Shield,
  Lock,
  Hash,
  CheckCircle,
  AlertTriangle,
  Clock,
  FileText,
  Download,
  Search,
  Filter,
  RefreshCw,
  Zap,
  Award,
  TrendingUp,
  Eye,
  Key,
  Server,
  Database
} from 'lucide-react';

interface BlockchainRecord {
  blockNumber: number;
  hash: string;
  previousHash: string;
  timestamp: Date;
  eventType: 'compliance' | 'policy_change' | 'access_grant' | 'configuration' | 'audit' | 'incident';
  data: {
    action: string;
    resource: string;
    user: string;
    details: any;
    complianceFramework?: string;
    riskScore?: number;
  };
  signature: string;
  verified: boolean;
  merkleRoot: string;
}

interface ComplianceProof {
  id: string;
  framework: string;
  timestamp: Date;
  blockHash: string;
  status: 'valid' | 'expired' | 'pending';
  controls: number;
  passed: number;
  auditor?: string;
  certificate?: string;
}

interface SmartContract {
  id: string;
  name: string;
  description: string;
  type: 'governance' | 'compliance' | 'access' | 'automation';
  status: 'active' | 'paused' | 'deprecated';
  executionCount: number;
  lastExecution?: Date;
  conditions: string[];
  actions: string[];
}

export default function BlockchainAuditChain() {
  const router = useRouter();
  const [blockchainRecords, setBlockchainRecords] = useState<BlockchainRecord[]>([]);
  const [complianceProofs, setComplianceProofs] = useState<ComplianceProof[]>([]);
  const [smartContracts, setSmartContracts] = useState<SmartContract[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<BlockchainRecord | null>(null);
  const [filter, setFilter] = useState<'all' | 'compliance' | 'policy' | 'access'>('all');
  const [verifying, setVerifying] = useState<string | null>(null);

  useEffect(() => {
    // Generate blockchain records
    const records: BlockchainRecord[] = [
      {
        blockNumber: 1247,
        hash: '0x7f9fade1c0d57a7af66ab4ead79fade1c0d57a7af66ab4ead7c2df6a2e4f8b23',
        previousHash: '0x6e8dd63c3f4e9c8a4f9e495a6b8fe4b8c5a9d7e3f2a1b0c9d8e7f6a5b4c3d2e1f0',
        timestamp: new Date(Date.now() - 300000),
        eventType: 'compliance',
        data: {
          action: 'Compliance Validation',
          resource: 'Production Environment',
          user: 'automated-scanner',
          details: { framework: 'HIPAA', score: 94, controls: 198 },
          complianceFramework: 'HIPAA',
          riskScore: 6
        },
        signature: '0x3045022100...',
        verified: true,
        merkleRoot: '0x9c22ff5f21f0b6b0e646d3c2be5f8e4d3a2e1b0c9d8e7f6a5b4c3d2e1f0'
      },
      {
        blockNumber: 1246,
        hash: '0x6e8dd63c3f4e9c8a4f9e495a6b8fe4b8c5a9d7e3f2a1b0c9d8e7f6a5b4c3d2e1f0',
        previousHash: '0x5d7cc52b2e3f3d2c1b0a9f8e7d6c5b4a3928e7d6c5b4a3f2e1d0c9b8a7968574',
        timestamp: new Date(Date.now() - 600000),
        eventType: 'policy_change',
        data: {
          action: 'Policy Update',
          resource: 'S3 Bucket Encryption Policy',
          user: 'admin@company.com',
          details: { 
            change: 'Enforced AES-256 encryption',
            approval: 'CAB-2024-0142',
            impact: 'All S3 buckets'
          }
        },
        signature: '0x3045022100...',
        verified: true,
        merkleRoot: '0x8b11ee4d20e5c7f5a3d2b1a0f9e8d7c6b5a4938271d6c5b4a3f2e1d0'
      },
      {
        blockNumber: 1245,
        hash: '0x5d7cc52b2e3f3d2c1b0a9f8e7d6c5b4a3928e7d6c5b4a3f2e1d0c9b8a7968574',
        previousHash: '0x4c6bb41a1d2e2c1a0908d7b6a5948362716b5a4938271605a4b3c2d1e0f90807',
        timestamp: new Date(Date.now() - 900000),
        eventType: 'access_grant',
        data: {
          action: 'Privileged Access Grant',
          resource: 'Production Database',
          user: 'john.doe@company.com',
          details: {
            role: 'Database Administrator',
            duration: '4 hours',
            justification: 'Emergency patch deployment',
            approver: 'manager@company.com'
          }
        },
        signature: '0x3045022100...',
        verified: true,
        merkleRoot: '0x7a00dd3c10d46e2a8e9e7c5b4a3827160504938271605a4b3c2d1e0'
      },
      {
        blockNumber: 1244,
        hash: '0x4c6bb41a1d2e2c1a0908d7b6a5948362716b5a4938271605a4b3c2d1e0f90807',
        previousHash: '0x3b5aa30901c1b009f7a6948251605938160493a271504b3a2c1d0e0f90706f5',
        timestamp: new Date(Date.now() - 1200000),
        eventType: 'audit',
        data: {
          action: 'External Audit Completed',
          resource: 'Entire Infrastructure',
          user: 'external-auditor@audit-firm.com',
          details: {
            type: 'SOC 2 Type II',
            findings: 3,
            status: 'Passed with observations',
            reportId: 'SOC2-2024-Q4'
          }
        },
        signature: '0x3045022100...',
        verified: true,
        merkleRoot: '0x6900cc2b00c35d1909d6a5837150493816049271504a3b2c1d0e0f'
      }
    ];

    setBlockchainRecords(records);

    // Generate compliance proofs
    setComplianceProofs([
      {
        id: 'proof-1',
        framework: 'HIPAA',
        timestamp: new Date(Date.now() - 86400000),
        blockHash: '0x7f9fade1c0d57a7af66ab4ead79fade1c0d57a7af66ab4ead7c2df6a2e4f8b23',
        status: 'valid',
        controls: 198,
        passed: 186,
        auditor: 'Deloitte',
        certificate: 'HIPAA-2024-CERT-0923'
      },
      {
        id: 'proof-2',
        framework: 'SOC 2 Type II',
        timestamp: new Date(Date.now() - 172800000),
        blockHash: '0x4c6bb41a1d2e2c1a0908d7b6a5948362716b5a4938271605a4b3c2d1e0f90807',
        status: 'valid',
        controls: 156,
        passed: 153,
        auditor: 'PwC',
        certificate: 'SOC2-2024-CERT-0142'
      },
      {
        id: 'proof-3',
        framework: 'ISO 27001',
        timestamp: new Date(Date.now() - 259200000),
        blockHash: '0x3b5aa30901c1b009f7a6948251605938160493a271504b3a2c1d0e0f90706f5',
        status: 'pending',
        controls: 114,
        passed: 108
      }
    ]);

    // Generate smart contracts
    setSmartContracts([
      {
        id: 'contract-1',
        name: 'Auto-Remediation Contract',
        description: 'Automatically fix non-compliant resources',
        type: 'automation',
        status: 'active',
        executionCount: 342,
        lastExecution: new Date(Date.now() - 3600000),
        conditions: [
          'If S3 bucket is public',
          'If encryption is disabled',
          'If versioning is not enabled'
        ],
        actions: [
          'Block public access',
          'Enable AES-256 encryption',
          'Enable versioning'
        ]
      },
      {
        id: 'contract-2',
        name: 'Compliance Enforcement',
        description: 'Enforce compliance policies across all resources',
        type: 'compliance',
        status: 'active',
        executionCount: 1847,
        lastExecution: new Date(Date.now() - 1800000),
        conditions: [
          'Resource violates policy',
          'Non-compliant configuration detected',
          'Missing required tags'
        ],
        actions: [
          'Send alert to owner',
          'Apply remediation if available',
          'Log to audit chain'
        ]
      },
      {
        id: 'contract-3',
        name: 'Access Governance',
        description: 'Manage privileged access requests',
        type: 'access',
        status: 'active',
        executionCount: 89,
        lastExecution: new Date(Date.now() - 7200000),
        conditions: [
          'PIM request submitted',
          'Manager approval received',
          'Risk score below threshold'
        ],
        actions: [
          'Grant temporary access',
          'Set expiration timer',
          'Record in blockchain'
        ]
      }
    ]);
  }, []);

  const verifyBlock = async (blockHash: string) => {
    setVerifying(blockHash);
    // Simulate verification
    setTimeout(() => {
      setVerifying(null);
      alert('Block verified successfully! Cryptographic proof is valid.');
    }, 2000);
  };

  const getEventIcon = (type: BlockchainRecord['eventType']) => {
    switch (type) {
      case 'compliance': return <Shield className="h-5 w-5 text-green-500" />;
      case 'policy_change': return <FileText className="h-5 w-5 text-blue-500" />;
      case 'access_grant': return <Key className="h-5 w-5 text-yellow-500" />;
      case 'configuration': return <Settings className="h-5 w-5 text-purple-500" />;
      case 'audit': return <Award className="h-5 w-5 text-indigo-500" />;
      case 'incident': return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default: return <Database className="h-5 w-5 text-gray-500" />;
    }
  };

  const filteredRecords = blockchainRecords.filter(record => {
    if (filter === 'all') return true;
    if (filter === 'compliance') return record.eventType === 'compliance' || record.eventType === 'audit';
    if (filter === 'policy') return record.eventType === 'policy_change' || record.eventType === 'configuration';
    if (filter === 'access') return record.eventType === 'access_grant';
    return true;
  });

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Link className="h-10 w-10 text-indigo-600" />
            Blockchain Audit Chain
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Immutable compliance evidence with cryptographic proof
          </p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2">
            <Download className="h-5 w-5" />
            Export Audit Report
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Blocks</p>
              <p className="text-2xl font-bold">1,247</p>
            </div>
            <Link className="h-8 w-8 text-indigo-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Verified Proofs</p>
              <p className="text-2xl font-bold">{complianceProofs.filter(p => p.status === 'valid').length}</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Smart Contracts</p>
              <p className="text-2xl font-bold">{smartContracts.length}</p>
            </div>
            <Zap className="h-8 w-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Chain Status</p>
              <p className="text-2xl font-bold text-green-600">Valid</p>
            </div>
            <Shield className="h-8 w-8 text-green-500" />
          </div>
        </div>
      </div>

      {/* Compliance Proofs */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Award className="h-6 w-6 text-indigo-600" />
          Cryptographic Compliance Proofs
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {complianceProofs.map((proof) => (
            <div
              key={proof.id}
              className="border dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold">{proof.framework}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  proof.status === 'valid' 
                    ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'
                    : proof.status === 'expired'
                    ? 'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'
                    : 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300'
                }`}>
                  {proof.status.toUpperCase()}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Controls</span>
                  <span className="font-medium">{proof.passed}/{proof.controls}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Block Hash</span>
                  <span className="font-mono text-xs">{proof.blockHash.slice(0, 10)}...</span>
                </div>
                {proof.auditor && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Auditor</span>
                    <span className="font-medium">{proof.auditor}</span>
                  </div>
                )}
                {proof.certificate && (
                  <div className="pt-2 border-t dark:border-gray-700">
                    <p className="text-xs text-gray-500">Certificate: {proof.certificate}</p>
                  </div>
                )}
              </div>
              <button
                onClick={() => verifyBlock(proof.blockHash)}
                className="mt-3 w-full px-3 py-1 bg-indigo-600 text-white rounded-md text-sm hover:bg-indigo-700"
              >
                Verify Proof
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Smart Contracts */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Zap className="h-6 w-6 text-yellow-600" />
          Active Smart Contracts
        </h2>
        <div className="space-y-3">
          {smartContracts.map((contract) => (
            <div
              key={contract.id}
              className="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="font-semibold">{contract.name}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      contract.status === 'active'
                        ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'
                        : 'bg-gray-100 dark:bg-gray-900/50 text-gray-700 dark:text-gray-300'
                    }`}>
                      {contract.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {contract.description}
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <span>Executions: <span className="font-medium">{contract.executionCount}</span></span>
                    {contract.lastExecution && (
                      <span>Last: {new Date(contract.lastExecution).toLocaleString()}</span>
                    )}
                  </div>
                </div>
                <button className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                  View Details
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Blockchain Records */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Immutable Audit Trail</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1 rounded-md text-sm ${
                filter === 'all' 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('compliance')}
              className={`px-3 py-1 rounded-md text-sm ${
                filter === 'compliance' 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              Compliance
            </button>
            <button
              onClick={() => setFilter('policy')}
              className={`px-3 py-1 rounded-md text-sm ${
                filter === 'policy' 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              Policy
            </button>
            <button
              onClick={() => setFilter('access')}
              className={`px-3 py-1 rounded-md text-sm ${
                filter === 'access' 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              Access
            </button>
          </div>
        </div>

        <div className="space-y-3">
          {filteredRecords.map((record) => (
            <div
              key={record.blockNumber}
              className="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
              onClick={() => setSelectedRecord(record)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  {getEventIcon(record.eventType)}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-1">
                      <span className="font-mono text-sm">Block #{record.blockNumber}</span>
                      <span className="text-sm text-gray-500">
                        {new Date(record.timestamp).toLocaleString()}
                      </span>
                      {record.verified && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                    </div>
                    <h3 className="font-semibold">{record.data.action}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Resource: {record.data.resource} • User: {record.data.user}
                    </p>
                    <div className="flex items-center gap-4 mt-2 text-xs">
                      <span className="font-mono">Hash: {record.hash.slice(0, 20)}...</span>
                      <span className="font-mono">Previous: {record.previousHash.slice(0, 20)}...</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    verifyBlock(record.hash);
                  }}
                  disabled={verifying === record.hash}
                  className="px-3 py-1 bg-indigo-600 text-white rounded-md text-sm hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2"
                >
                  {verifying === record.hash ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    <>
                      <Eye className="h-4 w-4" />
                      Verify
                    </>
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Add missing import
import { Settings } from 'lucide-react';