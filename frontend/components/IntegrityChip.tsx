'use client'

import { useState, useEffect } from 'react'
import { Shield, ShieldCheck, ShieldAlert, Loader2 } from 'lucide-react'

interface VerificationResult {
  chain_integrity: boolean
  merkle_proof_valid: boolean
  signature_valid: boolean
}

export function IntegrityChip({ 
  verify 
}: { 
  verify: () => Promise<VerificationResult> 
}) {
  const [state, setState] = useState<'idle' | 'checking' | 'ok' | 'fail'>('idle')
  const [details, setDetails] = useState<VerificationResult | null>(null)

  useEffect(() => {
    setState('checking')
    verify()
      .then(result => {
        setDetails(result)
        const isValid = result.chain_integrity && 
                       result.merkle_proof_valid && 
                       result.signature_valid
        setState(isValid ? 'ok' : 'fail')
      })
      .catch(() => {
        setState('fail')
        setDetails(null)
      })
  }, [verify])

  const getStyles = () => {
    switch (state) {
      case 'ok':
        return 'bg-emerald-100 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
      case 'fail':
        return 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800'
      default:
        return 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-400 border-neutral-200 dark:border-neutral-700'
    }
  }

  const getIcon = () => {
    switch (state) {
      case 'ok':
        return <ShieldCheck className="w-3.5 h-3.5" />
      case 'fail':
        return <ShieldAlert className="w-3.5 h-3.5" />
      case 'checking':
        return <Loader2 className="w-3.5 h-3.5 animate-spin" />
      default:
        return <Shield className="w-3.5 h-3.5" />
    }
  }

  const getLabel = () => {
    switch (state) {
      case 'ok':
        return 'Integrity OK'
      case 'fail':
        return 'Integrity FAIL'
      case 'checking':
        return 'Checking…'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className="relative group">
      <span 
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium border ${getStyles()}`}
        role="status"
        aria-label={getLabel()}
      >
        {getIcon()}
        {getLabel()}
      </span>
      
      {/* Tooltip with details on hover */}
      {details && state !== 'checking' && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
          <div className="bg-gray-900 dark:bg-gray-800 text-white text-xs rounded-lg p-2 whitespace-nowrap">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className={details.chain_integrity ? 'text-green-400' : 'text-red-400'}>●</span>
                <span>Chain: {details.chain_integrity ? 'Valid' : 'Invalid'}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={details.merkle_proof_valid ? 'text-green-400' : 'text-red-400'}>●</span>
                <span>Merkle: {details.merkle_proof_valid ? 'Valid' : 'Invalid'}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={details.signature_valid ? 'text-green-400' : 'text-red-400'}>●</span>
                <span>Signature: {details.signature_valid ? 'Valid' : 'Invalid'}</span>
              </div>
            </div>
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full">
              <div className="border-8 border-transparent border-t-gray-900 dark:border-t-gray-800"></div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}