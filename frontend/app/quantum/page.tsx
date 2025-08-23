'use client'

import { useState, useEffect } from 'react'
import { Shield, Lock, Key, AlertTriangle, CheckCircle, XCircle, Cpu, Zap, Globe, Database, RefreshCw, TrendingUp } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function QuantumSafeSecretsPage() {
  const [activeTab, setActiveTab] = useState('overview')
  const [migrationProgress, setMigrationProgress] = useState(0)
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('kyber1024')
  
  useEffect(() => {
    const timer = setInterval(() => {
      setMigrationProgress(prev => Math.min(prev + Math.random() * 5, 100))
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  const quantumAlgorithms = [
    {
      id: 'kyber1024',
      name: 'Kyber-1024',
      type: 'Key Encapsulation',
      security: 'Level 5',
      status: 'NIST Approved',
      performance: '99.8%',
      quantumResistant: true
    },
    {
      id: 'dilithium5',
      name: 'Dilithium5',
      type: 'Digital Signature',
      security: 'Level 5',
      status: 'NIST Approved',
      performance: '98.5%',
      quantumResistant: true
    },
    {
      id: 'sphincs',
      name: 'SPHINCS+',
      type: 'Hash-Based Signature',
      security: 'Level 5',
      status: 'NIST Approved',
      performance: '97.2%',
      quantumResistant: true
    },
    {
      id: 'falcon1024',
      name: 'Falcon-1024',
      type: 'Lattice Signature',
      security: 'Level 5',
      status: 'NIST Finalist',
      performance: '99.1%',
      quantumResistant: true
    }
  ]

  const secrets = [
    {
      id: 'db-master-key',
      name: 'Database Master Key',
      type: 'AES-256',
      quantum: 'Kyber-1024',
      status: 'migrated',
      lastRotated: '2 hours ago',
      risk: 'low'
    },
    {
      id: 'api-signing-cert',
      name: 'API Signing Certificate',
      type: 'RSA-4096',
      quantum: 'Dilithium5',
      status: 'migrating',
      lastRotated: '1 day ago',
      risk: 'medium'
    },
    {
      id: 'service-mesh-tls',
      name: 'Service Mesh TLS',
      type: 'ECDSA-P256',
      quantum: 'Pending',
      status: 'vulnerable',
      lastRotated: '3 days ago',
      risk: 'high'
    },
    {
      id: 'backup-encryption',
      name: 'Backup Encryption Key',
      type: 'AES-256',
      quantum: 'Kyber-1024',
      status: 'migrated',
      lastRotated: '1 hour ago',
      risk: 'low'
    }
  ]

  const quantumThreats = [
    {
      threat: 'Harvest Now, Decrypt Later',
      impact: 'Critical',
      timeline: '5-10 years',
      mitigation: 'Immediate migration to PQC'
    },
    {
      threat: 'Shor\'s Algorithm',
      impact: 'Critical',
      timeline: '10-15 years',
      mitigation: 'Replace RSA/ECC with lattice-based'
    },
    {
      threat: 'Grover\'s Algorithm',
      impact: 'High',
      timeline: '15-20 years',
      mitigation: 'Double symmetric key sizes'
    }
  ]

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Quantum-Safe Secrets"
                value="67%"
                subtitle="156/234 migrated"
                trend="up"
                icon={<Shield className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="Vulnerable Secrets"
                value="78"
                subtitle="Requires immediate action"
                alert="High risk"
                icon={<AlertTriangle className="w-5 h-5 text-red-500" />}
              />
              <MetricCard
                title="Migration Rate"
                value="12/day"
                subtitle="Est. completion: 6 days"
                trend="up"
                icon={<RefreshCw className="w-5 h-5 text-blue-500" />}
              />
              <MetricCard
                title="Quantum Readiness"
                value="Level 4"
                subtitle="NIST PQC Compliant"
                icon={<Cpu className="w-5 h-5 text-purple-500" />}
              />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Post-Quantum Cryptography Migration</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Overall Progress</span>
                  <span className="text-sm font-medium">{migrationProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div 
                    className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${migrationProgress}%` }}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4 mt-6">
                  <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">156</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Quantum-Safe</div>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">78</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Vulnerable</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Quantum Threat Timeline</h3>
              <div className="space-y-3">
                {quantumThreats.map((threat, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex-1">
                      <div className="font-medium">{threat.threat}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">{threat.mitigation}</div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-medium ${
                        threat.impact === 'Critical' ? 'text-red-600' : 'text-orange-600'
                      }`}>{threat.impact}</div>
                      <div className="text-xs text-gray-500">{threat.timeline}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )

      case 'algorithms':
        return (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">NIST Post-Quantum Algorithms</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {quantumAlgorithms.map((algo) => (
                  <div 
                    key={algo.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedAlgorithm === algo.id 
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20' 
                        : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
                    }`}
                    onClick={() => setSelectedAlgorithm(algo.id)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold">{algo.name}</h4>
                        <div className="text-sm text-gray-600 dark:text-gray-400">{algo.type}</div>
                      </div>
                      {algo.quantumResistant && (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      )}
                    </div>
                    <div className="space-y-2 mt-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Security Level</span>
                        <span className="font-medium">{algo.security}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Status</span>
                        <span className={`font-medium ${
                          algo.status === 'NIST Approved' ? 'text-green-600' : 'text-yellow-600'
                        }`}>{algo.status}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Performance</span>
                        <span className="font-medium">{algo.performance}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <ChartContainer title="Algorithm Performance Comparison">
              <div className="h-64 flex items-center justify-center text-gray-500">
                Performance benchmarks visualization
              </div>
            </ChartContainer>
          </div>
        )

      case 'secrets':
        return (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Secret Migration Status</h3>
                <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                  Bulk Migrate
                </button>
              </div>
              
              <div className="space-y-3">
                {secrets.map((secret) => (
                  <div key={secret.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Key className={`w-5 h-5 ${
                        secret.status === 'migrated' ? 'text-green-500' :
                        secret.status === 'migrating' ? 'text-yellow-500' : 'text-red-500'
                      }`} />
                      <div>
                        <div className="font-medium">{secret.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {secret.type} → {secret.quantum}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className={`text-sm font-medium ${
                          secret.risk === 'low' ? 'text-green-600' :
                          secret.risk === 'medium' ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {secret.risk.toUpperCase()} RISK
                        </div>
                        <div className="text-xs text-gray-500">Rotated {secret.lastRotated}</div>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                        secret.status === 'migrated' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        secret.status === 'migrating' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {secret.status.toUpperCase()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Automated Key Rotation</h3>
              <p className="text-sm opacity-90 mb-4">
                Enable quantum-safe key rotation policies to automatically migrate secrets
              </p>
              <button className="px-4 py-2 bg-white text-purple-600 rounded-lg hover:bg-gray-100 transition-colors">
                Configure Rotation Policy
              </button>
            </div>
          </div>
        )

      case 'compliance':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard
                title="NIST Compliance"
                value="92%"
                subtitle="PQC Standards Met"
                icon={<CheckCircle className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="CNSA 2.0"
                value="88%"
                subtitle="NSA Requirements"
                icon={<Shield className="w-5 h-5 text-blue-500" />}
              />
              <MetricCard
                title="ISO/IEC 23837"
                value="95%"
                subtitle="Quantum-Safe Standards"
                icon={<Globe className="w-5 h-5 text-purple-500" />}
              />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Compliance Requirements</h3>
              <div className="space-y-4">
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-green-900 dark:text-green-100">FIPS 203: ML-KEM</h4>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        Module-Lattice-Based Key-Encapsulation Mechanism Standard
                      </p>
                    </div>
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  </div>
                </div>
                
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-green-900 dark:text-green-100">FIPS 204: ML-DSA</h4>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        Module-Lattice-Based Digital Signature Algorithm
                      </p>
                    </div>
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  </div>
                </div>
                
                <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium text-yellow-900 dark:text-yellow-100">FIPS 205: SLH-DSA</h4>
                      <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                        Stateless Hash-Based Digital Signature Algorithm - In Progress
                      </p>
                    </div>
                    <AlertTriangle className="w-6 h-6 text-yellow-600" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg">
              <Lock className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Quantum-Safe Secrets Management
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Post-quantum cryptography migration and key management
              </p>
            </div>
          </div>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto">
          {['overview', 'algorithms', 'secrets', 'compliance'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                activeTab === tab
                  ? 'bg-purple-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {renderContent()}

        <div className="mt-8 p-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Quantum Computing Threat Level</h3>
              <p className="text-sm opacity-90 mt-1">
                Current quantum computer capability: 1,000+ qubits • Threat timeline: 5-10 years
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">ELEVATED</div>
              <div className="text-xs opacity-75">Immediate action recommended</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}