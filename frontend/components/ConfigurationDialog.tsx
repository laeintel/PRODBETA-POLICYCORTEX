'use client'

import { useState } from 'react'
import { Settings, X, Save, RefreshCw } from 'lucide-react'
import { toast } from '@/hooks/useToast'

interface ConfigurationDialogProps {
  isOpen: boolean
  onClose: () => void
  title: string
  configType: 'notifications' | 'automation' | 'security' | 'devops' | 'operations' | 'ai'
  currentConfig?: any
  onSave?: (config: any) => void
}

export default function ConfigurationDialog({
  isOpen,
  onClose,
  title,
  configType,
  currentConfig = {},
  onSave
}: ConfigurationDialogProps) {
  const [config, setConfig] = useState(currentConfig)
  
  if (!isOpen) return null
  
  const handleSave = () => {
    if (onSave) {
      onSave(config)
    }
    toast({
      title: 'Configuration saved',
      description: `${title} settings have been updated`
    })
    onClose()
  }
  
  const handleReset = () => {
    setConfig(currentConfig)
    toast({
      title: 'Configuration reset',
      description: 'Settings reverted to last saved state'
    })
  }
  
  const renderConfigFields = () => {
    switch (configType) {
      case 'notifications':
        return (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">Email Notifications</label>
              <input
                type="checkbox"
                checked={config.emailEnabled ?? true}
                onChange={(e) => setConfig({ ...config, emailEnabled: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm text-gray-400">Enable email notifications</span>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Notification Frequency</label>
              <select
                value={config.frequency ?? 'immediate'}
                onChange={(e) => setConfig({ ...config, frequency: e.target.value })}
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                <option value="immediate">Immediate</option>
                <option value="hourly">Hourly Digest</option>
                <option value="daily">Daily Digest</option>
                <option value="weekly">Weekly Summary</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Severity Filter</label>
              <select
                value={config.severity ?? 'all'}
                onChange={(e) => setConfig({ ...config, severity: e.target.value })}
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical Only</option>
                <option value="high">High & Critical</option>
                <option value="medium">Medium & Above</option>
              </select>
            </div>
          </>
        )
        
      case 'automation':
        return (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">Auto-Remediation</label>
              <input
                type="checkbox"
                checked={config.autoRemediate ?? false}
                onChange={(e) => setConfig({ ...config, autoRemediate: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm text-gray-400">Enable automatic remediation</span>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Approval Required</label>
              <input
                type="checkbox"
                checked={config.requireApproval ?? true}
                onChange={(e) => setConfig({ ...config, requireApproval: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm text-gray-400">Require approval for actions</span>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Retry Attempts</label>
              <input
                type="number"
                value={config.retryAttempts ?? 3}
                onChange={(e) => setConfig({ ...config, retryAttempts: parseInt(e.target.value) })}
                min="0"
                max="10"
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
          </>
        )
        
      case 'security':
        return (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">MFA Enforcement</label>
              <select
                value={config.mfaPolicy ?? 'required'}
                onChange={(e) => setConfig({ ...config, mfaPolicy: e.target.value })}
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                <option value="required">Required for All</option>
                <option value="admins">Admins Only</option>
                <option value="optional">Optional</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Session Timeout (minutes)</label>
              <input
                type="number"
                value={config.sessionTimeout ?? 30}
                onChange={(e) => setConfig({ ...config, sessionTimeout: parseInt(e.target.value) })}
                min="5"
                max="1440"
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Password Policy</label>
              <select
                value={config.passwordPolicy ?? 'strong'}
                onChange={(e) => setConfig({ ...config, passwordPolicy: e.target.value })}
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                <option value="strong">Strong (12+ chars, mixed)</option>
                <option value="medium">Medium (8+ chars)</option>
                <option value="basic">Basic (6+ chars)</option>
              </select>
            </div>
          </>
        )
        
      case 'ai':
        return (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">AI Model</label>
              <select
                value={config.model ?? 'gpt4'}
                onChange={(e) => setConfig({ ...config, model: e.target.value })}
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                <option value="gpt4">GPT-4 (Most Accurate)</option>
                <option value="gpt35">GPT-3.5 (Faster)</option>
                <option value="local">Local Model (Private)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Confidence Threshold</label>
              <input
                type="number"
                value={config.confidenceThreshold ?? 0.8}
                onChange={(e) => setConfig({ ...config, confidenceThreshold: parseFloat(e.target.value) })}
                min="0"
                max="1"
                step="0.1"
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Auto-Suggestions</label>
              <input
                type="checkbox"
                checked={config.autoSuggest ?? true}
                onChange={(e) => setConfig({ ...config, autoSuggest: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm text-gray-400">Enable AI suggestions</span>
            </div>
          </>
        )
        
      default:
        return (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">Refresh Interval (seconds)</label>
              <input
                type="number"
                value={config.refreshInterval ?? 30}
                onChange={(e) => setConfig({ ...config, refreshInterval: parseInt(e.target.value) })}
                min="5"
                max="300"
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Data Retention (days)</label>
              <input
                type="number"
                value={config.dataRetention ?? 90}
                onChange={(e) => setConfig({ ...config, dataRetention: parseInt(e.target.value) })}
                min="1"
                max="365"
                className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
          </>
        )
    }
  }
  
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-lg max-w-md w-full">
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-blue-400" />
            <h2 className="text-lg font-semibold">{title} Configuration</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-800 rounded"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {renderConfigFields()}
        </div>
        
        <div className="flex items-center justify-between p-4 border-t border-gray-800">
          <button
            onClick={handleReset}
            className="px-4 py-2 text-gray-400 hover:text-white flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
          
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}