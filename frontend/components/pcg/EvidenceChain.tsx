'use client'

import React from 'react'
import { Shield, CheckCircle, AlertCircle, Hash, Link2, FileText, Calendar } from 'lucide-react'

interface EvidenceLink {
  id: string
  timestamp: string
  action: string
  actor: string
  hash: string
  verified: boolean
  details?: string
}

interface EvidenceChainProps {
  links: EvidenceLink[]
  integrityStatus: 'verified' | 'unverified' | 'compromised'
  onVerifyChain?: () => void
  onExportEvidence?: () => void
}

export default function EvidenceChain({
  links,
  integrityStatus,
  onVerifyChain,
  onExportEvidence
}: EvidenceChainProps) {
  const getIntegrityColor = () => {
    switch (integrityStatus) {
      case 'verified':
        return 'bg-green-100 border-green-500 text-green-800'
      case 'unverified':
        return 'bg-yellow-100 border-yellow-500 text-yellow-800'
      case 'compromised':
        return 'bg-red-100 border-red-500 text-red-800'
      default:
        return 'bg-gray-100 border-gray-500 text-gray-800'
    }
  }

  const getIntegrityIcon = () => {
    switch (integrityStatus) {
      case 'verified':
        return <CheckCircle className="h-5 w-5" />
      case 'compromised':
        return <AlertCircle className="h-5 w-5" />
      default:
        return <Shield className="h-5 w-5" />
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <Link2 className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Evidence Chain</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">Cryptographically verified audit trail</p>
            </div>
          </div>
          <div className={`px-4 py-2 rounded-full flex items-center gap-2 ${getIntegrityColor()}`}>
            {getIntegrityIcon()}
            <span className="font-semibold text-sm">{integrityStatus.toUpperCase()}</span>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onVerifyChain}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg font-medium text-sm hover:bg-purple-700 transition-colors flex items-center gap-2"
          >
            <Shield className="h-4 w-4" />
            Verify Chain Integrity
          </button>
          <button
            onClick={onExportEvidence}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
          >
            <FileText className="h-4 w-4" />
            Export Evidence
          </button>
        </div>
      </div>

      <div className="p-6">
        <div className="space-y-4">
          {links.map((link, index) => (
            <div key={link.id} className="relative">
              {index < links.length - 1 && (
                <div className="absolute left-6 top-12 w-0.5 h-16 bg-gray-300 dark:bg-gray-600" />
              )}
              
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  link.verified 
                    ? 'bg-green-100 dark:bg-green-900/30' 
                    : 'bg-yellow-100 dark:bg-yellow-900/30'
                }`}>
                  {link.verified 
                    ? <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
                    : <AlertCircle className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
                  }
                </div>
                
                <div className="flex-1 bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-gray-100">{link.action}</h3>
                      <div className="flex items-center gap-4 mt-1 text-sm text-gray-500 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {link.timestamp}
                        </span>
                        <span>by {link.actor}</span>
                      </div>
                    </div>
                    {link.verified && (
                      <span className="text-xs bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300 px-2 py-1 rounded-full">
                        Verified
                      </span>
                    )}
                  </div>
                  
                  {link.details && (
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{link.details}</p>
                  )}
                  
                  <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-500">
                    <Hash className="h-3 w-3" />
                    <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded font-mono">
                      {link.hash}
                    </code>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}