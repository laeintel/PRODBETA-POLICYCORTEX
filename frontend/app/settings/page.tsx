'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import AppLayout from '../../components/AppLayout'
import { 
  User, 
  Shield, 
  Bell, 
  Globe, 
  Key, 
  Database,
  Moon,
  Sun,
  Save,
  RefreshCw,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('profile')
  const [saved, setSaved] = useState(false)
  const [darkMode, setDarkMode] = useState(true)
  
  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'integrations', label: 'Integrations', icon: Globe },
    { id: 'api', label: 'API Keys', icon: Key },
    { id: 'data', label: 'Data & Privacy', icon: Database }
  ]

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
            <p className="text-gray-400">Manage your PolicyCortex preferences and configuration</p>
          </div>

          {/* Tabs */}
          <div className="flex space-x-1 mb-8 bg-white/5 p-1 rounded-lg">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                    activeTab === tab.id
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{tab.label}</span>
                </button>
              )
            })}
          </div>

          {/* Content */}
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6"
          >
            {activeTab === 'profile' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">Profile Settings</h2>
                
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Display Name
                    </label>
                    <input
                      type="text"
                      defaultValue="Administrator"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Email
                    </label>
                    <input
                      type="email"
                      defaultValue="admin@policycortex.com"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Organization
                    </label>
                    <input
                      type="text"
                      defaultValue="AeoliTech"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Role
                    </label>
                    <select className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-purple-400">
                      <option>Global Administrator</option>
                      <option>Security Administrator</option>
                      <option>Compliance Officer</option>
                      <option>Cost Manager</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4">
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-gray-300">Theme</span>
                    <button
                      onClick={() => setDarkMode(!darkMode)}
                      className="flex items-center gap-2 px-3 py-1 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
                    >
                      {darkMode ? <Moon className="w-4 h-4 text-purple-400" /> : <Sun className="w-4 h-4 text-yellow-400" />}
                      <span className="text-sm text-white">{darkMode ? 'Dark' : 'Light'}</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">Security Settings</h2>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">Two-Factor Authentication</h3>
                      <p className="text-sm text-gray-400">Add an extra layer of security to your account</p>
                    </div>
                    <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                      Enable
                    </button>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">Session Timeout</h3>
                      <p className="text-sm text-gray-400">Automatically sign out after inactivity</p>
                    </div>
                    <select className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white">
                      <option>15 minutes</option>
                      <option>30 minutes</option>
                      <option>1 hour</option>
                      <option>4 hours</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">API Access</h3>
                      <p className="text-sm text-gray-400">Manage programmatic access to PolicyCortex</p>
                    </div>
                    <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                      Configure
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">Notification Preferences</h2>
                
                <div className="space-y-4">
                  {[
                    { title: 'Policy Violations', description: 'Alert when policies are violated' },
                    { title: 'Cost Alerts', description: 'Notify when spending exceeds thresholds' },
                    { title: 'Security Threats', description: 'Immediate alerts for security issues' },
                    { title: 'Resource Optimization', description: 'Suggestions for resource improvements' },
                    { title: 'Compliance Reports', description: 'Weekly compliance summary' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <h3 className="text-white font-medium">{item.title}</h3>
                        <p className="text-sm text-gray-400">{item.description}</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" defaultChecked className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'integrations' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">Azure Integrations</h2>
                
                <div className="space-y-4">
                  <div className="p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                    <div className="flex items-center gap-3 mb-2">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <h3 className="text-white font-medium">Azure Subscription Connected</h3>
                    </div>
                    <p className="text-sm text-gray-300">Subscription ID: 205b477d-17e7-4b3b-92c1-32cf02626b78</p>
                    <p className="text-sm text-gray-400 mt-1">Last sync: 2 minutes ago</p>
                    <button className="mt-3 flex items-center gap-2 px-3 py-1 bg-green-600/20 text-green-400 rounded-lg hover:bg-green-600/30 transition-colors">
                      <RefreshCw className="w-4 h-4" />
                      <span className="text-sm">Sync Now</span>
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-white/5 rounded-lg">
                      <h3 className="text-white font-medium mb-2">Azure Policy</h3>
                      <p className="text-sm text-green-400">Connected</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <h3 className="text-white font-medium mb-2">Azure AD</h3>
                      <p className="text-sm text-green-400">Connected</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <h3 className="text-white font-medium mb-2">Cost Management</h3>
                      <p className="text-sm text-green-400">Connected</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <h3 className="text-white font-medium mb-2">Network Security</h3>
                      <p className="text-sm text-green-400">Connected</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'api' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">API Keys</h2>
                
                <div className="p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg mb-4">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-yellow-400" />
                    <p className="text-sm text-yellow-300">Keep your API keys secure. Never share them publicly.</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-white font-medium">Production API Key</h3>
                      <span className="text-xs text-green-400 px-2 py-1 bg-green-900/20 rounded">Active</span>
                    </div>
                    <p className="text-sm text-gray-400 mb-3">Created on: January 15, 2025</p>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 px-3 py-2 bg-black/30 rounded text-sm text-gray-300">
                        pk_live_********************************
                      </code>
                      <button className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                        Regenerate
                      </button>
                    </div>
                  </div>
                </div>
                
                <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                  + Create New API Key
                </button>
              </div>
            )}

            {activeTab === 'data' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-4">Data & Privacy</h2>
                
                <div className="space-y-4">
                  <div className="p-4 bg-white/5 rounded-lg">
                    <h3 className="text-white font-medium mb-2">Data Retention</h3>
                    <p className="text-sm text-gray-400 mb-3">How long should we keep your governance data?</p>
                    <select className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white">
                      <option>30 days</option>
                      <option>90 days</option>
                      <option>1 year</option>
                      <option>Indefinite</option>
                    </select>
                  </div>
                  
                  <div className="p-4 bg-white/5 rounded-lg">
                    <h3 className="text-white font-medium mb-2">Export Data</h3>
                    <p className="text-sm text-gray-400 mb-3">Download all your PolicyCortex data</p>
                    <button disabled className="px-4 py-2 bg-gray-400 text-white rounded-lg disabled:opacity-60 cursor-not-allowed" title="Disabled in demo">
                      Export as JSON (Disabled)
                    </button>
                  </div>
                  
                  <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <h3 className="text-red-400 font-medium mb-2">Danger Zone</h3>
                    <p className="text-sm text-gray-400 mb-3">Permanently delete all data and close account</p>
                    <button className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
                      Delete Account
                    </button>
                  </div>
                </div>
              </div>
            )}
          </motion.div>

          {/* Save Button */}
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              {saved ? (
                <>
                  <CheckCircle className="w-5 h-5" />
                  <span>Saved Successfully</span>
                </>
              ) : (
                <>
                  <Save className="w-5 h-5" />
                  <span>Save Changes</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}