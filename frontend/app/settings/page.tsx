'use client';

import React, { useState } from 'react';
import { 
  User, Bell, Key, Palette, Globe, Shield, Link, Database, 
  Download, Monitor, Moon, Sun, ChevronRight, Save, Lock,
  Smartphone, Mail, AlertCircle, CheckCircle, Activity,
  Settings as SettingsIcon, LogOut, RefreshCw, Trash2
} from 'lucide-react';
import { toast } from '@/hooks/useToast'

interface SettingSection {
  id: string;
  title: string;
  icon: React.ElementType;
  description: string;
}

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState('profile');
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState({
    email: true,
    sms: false,
    inApp: true,
    critical: true,
    warnings: true,
    info: false
  });
  const [mfaEnabled, setMfaEnabled] = useState(false);
  const [language, setLanguage] = useState('en');
  const [dataRetention, setDataRetention] = useState('90');

  const sections: SettingSection[] = [
    { id: 'profile', title: 'User Profile', icon: User, description: 'Manage your personal information' },
    { id: 'notifications', title: 'Notifications', icon: Bell, description: 'Configure alert preferences' },
    { id: 'api', title: 'API Keys', icon: Key, description: 'Manage API access tokens' },
    { id: 'appearance', title: 'Appearance', icon: Palette, description: 'Customize interface theme' },
    { id: 'language', title: 'Language & Region', icon: Globe, description: 'Set language and locale' },
    { id: 'security', title: 'Security', icon: Shield, description: 'Security and authentication' },
    { id: 'integrations', title: 'Integrations', icon: Link, description: 'Connect external services' },
    { id: 'data', title: 'Data & Privacy', icon: Database, description: 'Data retention and privacy' },
    { id: 'export', title: 'Export & Import', icon: Download, description: 'Backup and restore settings' },
    { id: 'diagnostics', title: 'System Diagnostics', icon: Activity, description: 'System health and logs' }
  ];

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'profile':
        return <ProfileSection />;
      case 'notifications':
        return <NotificationSection notifications={notifications} setNotifications={setNotifications} />;
      case 'api':
        return <APIKeysSection />;
      case 'appearance':
        return <AppearanceSection darkMode={darkMode} setDarkMode={setDarkMode} />;
      case 'language':
        return <LanguageSection language={language} setLanguage={setLanguage} />;
      case 'security':
        return <SecuritySection mfaEnabled={mfaEnabled} setMfaEnabled={setMfaEnabled} />;
      case 'integrations':
        return <IntegrationsSection />;
      case 'data':
        return <DataPrivacySection dataRetention={dataRetention} setDataRetention={setDataRetention} />;
      case 'export':
        return <ExportImportSection />;
      case 'diagnostics':
        return <DiagnosticsSection />;
      default:
        return <ProfileSection />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-gray-800 border-r border-gray-700 min-h-screen">
          <div className="p-6">
            <h1 className="text-2xl font-bold mb-6 flex items-center space-x-2">
              <SettingsIcon className="w-6 h-6" />
              <span>Settings</span>
            </h1>
            <nav className="space-y-1">
              {sections.map((section) => {
                const Icon = section.icon;
                return (
                  <button type="button"
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                      activeSection === section.id
                        ? 'bg-blue-600 text-white'
                        : 'hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className="w-5 h-5" />
                      <div>
                        <p className="font-medium">{section.title}</p>
                        <p className="text-xs opacity-75">{section.description}</p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-8">
          {renderSectionContent()}
        </div>
      </div>
    </div>
  );
}

// Profile Section Component
function ProfileSection() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">User Profile</h2>
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-start space-x-6 mb-6">
          <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-3xl font-bold">
            JD
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-semibold mb-2">John Doe</h3>
            <p className="text-gray-400 mb-1">john.doe@company.com</p>
            <p className="text-gray-400 text-sm">Cloud Administrator</p>
            <button
              type="button"
              className="mt-3 px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
              onClick={() => toast({ title: 'Change avatar', description: 'Avatar change coming soon' })}
            >
              Change Avatar
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">First Name</label>
            <input type="text" defaultValue="John" className="w-full bg-gray-700 rounded px-4 py-2" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Last Name</label>
            <input type="text" defaultValue="Doe" className="w-full bg-gray-700 rounded px-4 py-2" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Email</label>
            <input type="email" defaultValue="john.doe@company.com" className="w-full bg-gray-700 rounded px-4 py-2" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Phone</label>
            <input type="tel" defaultValue="+1 555-123-4567" className="w-full bg-gray-700 rounded px-4 py-2" />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Department</label>
            <select className="w-full bg-gray-700 rounded px-4 py-2">
              <option>IT Operations</option>
              <option>Security</option>
              <option>DevOps</option>
              <option>Finance</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Role</label>
            <select className="w-full bg-gray-700 rounded px-4 py-2">
              <option>Administrator</option>
              <option>Analyst</option>
              <option>Viewer</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end mt-6 space-x-4">
          <button
            type="button"
            className="px-6 py-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
            onClick={() => toast({ title: 'Cancelled', description: 'No changes were saved' })}
          >
            Cancel
          </button>
          <button
            type="button"
            className="px-6 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors flex items-center space-x-2"
            onClick={() => toast({ title: 'Saved', description: 'Profile updated successfully' })}
          >
            <Save className="w-4 h-4" />
            <span>Save Changes</span>
          </button>
        </div>
      </div>
    </div>
  );
}

// Notification Section Component
function NotificationSection({ notifications, setNotifications }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Notification Preferences</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Notification Channels</h3>
        <div className="space-y-4">
          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Mail className="w-5 h-5 text-gray-400" />
              <div>
                <p className="font-medium">Email Notifications</p>
                <p className="text-sm text-gray-400">Receive alerts via email</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.email}
              onChange={(e) => setNotifications({...notifications, email: e.target.checked})}
              className="w-5 h-5"
            />
          </label>

          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Smartphone className="w-5 h-5 text-gray-400" />
              <div>
                <p className="font-medium">SMS Notifications</p>
                <p className="text-sm text-gray-400">Receive critical alerts via SMS</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.sms}
              onChange={(e) => setNotifications({...notifications, sms: e.target.checked})}
              className="w-5 h-5"
            />
          </label>

          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Bell className="w-5 h-5 text-gray-400" />
              <div>
                <p className="font-medium">In-App Notifications</p>
                <p className="text-sm text-gray-400">Show notifications in the application</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.inApp}
              onChange={(e) => setNotifications({...notifications, inApp: e.target.checked})}
              className="w-5 h-5"
            />
          </label>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Alert Types</h3>
        <div className="space-y-4">
          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <div>
                <p className="font-medium">Critical Alerts</p>
                <p className="text-sm text-gray-400">Security breaches, system failures</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.critical}
              onChange={(e) => setNotifications({...notifications, critical: e.target.checked})}
              className="w-5 h-5"
            />
          </label>

          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-yellow-400" />
              <div>
                <p className="font-medium">Warning Alerts</p>
                <p className="text-sm text-gray-400">Policy violations, threshold breaches</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.warnings}
              onChange={(e) => setNotifications({...notifications, warnings: e.target.checked})}
              className="w-5 h-5"
            />
          </label>

          <label className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-blue-400" />
              <div>
                <p className="font-medium">Informational Alerts</p>
                <p className="text-sm text-gray-400">System updates, reports</p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={notifications.info}
              onChange={(e) => setNotifications({...notifications, info: e.target.checked})}
              className="w-5 h-5"
            />
          </label>
        </div>
      </div>
    </div>
  );
}

// API Keys Section Component
function APIKeysSection() {
  const apiKeys = [
    { id: 1, name: 'Production API', key: 'sk-...4567', created: '2024-01-15', lastUsed: '2 hours ago', status: 'active' },
    { id: 2, name: 'Development API', key: 'sk-...8901', created: '2024-02-01', lastUsed: '5 days ago', status: 'active' },
    { id: 3, name: 'Test Integration', key: 'sk-...2345', created: '2023-12-20', lastUsed: 'Never', status: 'inactive' }
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">API Key Management</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-lg font-semibold">Active API Keys</h3>
            <p className="text-sm text-gray-400">Manage your API access tokens</p>
          </div>
          <button
            type="button"
            className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors flex items-center space-x-2"
            onClick={() => toast({ title: 'API key', description: 'Generating new API key...' })}
          >
            <Key className="w-4 h-4" />
            <span>Generate New Key</span>
          </button>
        </div>

        <div className="space-y-4">
          {apiKeys.map((apiKey) => (
            <div key={apiKey.id} className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold mb-1">{apiKey.name}</h4>
                  <p className="text-sm text-gray-400 font-mono">{apiKey.key}</p>
                  <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                    <span>Created: {apiKey.created}</span>
                    <span>Last used: {apiKey.lastUsed}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs rounded ${
                    apiKey.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-gray-600 text-gray-400'
                  }`}>
                    {apiKey.status}
                  </span>
                  <button
                    type="button"
                    className="p-2 hover:bg-gray-600 rounded transition-colors"
                    onClick={() => toast({ title: 'Rotated', description: 'API key rotated' })}
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                  <button
                    type="button"
                    className="p-2 hover:bg-red-600 rounded transition-colors"
                    onClick={() => toast({ title: 'Deleted', description: 'API key revoked' })}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-yellow-400">Security Notice</p>
            <p className="text-sm text-gray-300 mt-1">
              Keep your API keys secure. Never share them publicly or commit them to version control.
              Rotate keys regularly and revoke unused keys immediately.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Appearance Section Component
function AppearanceSection({ darkMode, setDarkMode }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Appearance Settings</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Theme</h3>
        <div className="grid grid-cols-2 gap-4">
          <button type="button"
            onClick={() => setDarkMode(false)}
            className={`p-4 rounded-lg border-2 transition-all ${
              !darkMode ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-gray-500'
            }`}
          >
            <Sun className="w-6 h-6 mb-2" />
            <p className="font-medium">Light Mode</p>
            <p className="text-sm text-gray-400">Bright interface for daytime</p>
          </button>
          
          <button type="button"
            onClick={() => setDarkMode(true)}
            className={`p-4 rounded-lg border-2 transition-all ${
              darkMode ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-gray-500'
            }`}
          >
            <Moon className="w-6 h-6 mb-2" />
            <p className="font-medium">Dark Mode</p>
            <p className="text-sm text-gray-400">Easy on the eyes</p>
          </button>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Accent Color</h3>
        <div className="flex space-x-4">
          {['blue', 'purple', 'green', 'orange', 'red', 'pink'].map((color) => (
            <button type="button"
              key={color}
              className={`w-12 h-12 rounded-lg bg-${color}-500 hover:scale-110 transition-transform`}
              style={{ backgroundColor: getColorHex(color) }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Language Section Component
function LanguageSection({ language, setLanguage }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Language & Region</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">Language</label>
            <select 
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full bg-gray-700 rounded px-4 py-2"
            >
              <option value="en">English</option>
              <option value="es">Español</option>
              <option value="fr">Français</option>
              <option value="de">Deutsch</option>
              <option value="ja">日本語</option>
              <option value="zh">中文</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Region</label>
            <select className="w-full bg-gray-700 rounded px-4 py-2">
              <option>United States</option>
              <option>Europe</option>
              <option>Asia Pacific</option>
              <option>Latin America</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Timezone</label>
            <select className="w-full bg-gray-700 rounded px-4 py-2">
              <option>UTC-08:00 Pacific Time</option>
              <option>UTC-05:00 Eastern Time</option>
              <option>UTC+00:00 GMT</option>
              <option>UTC+01:00 Central European Time</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Date Format</label>
            <select className="w-full bg-gray-700 rounded px-4 py-2">
              <option>MM/DD/YYYY</option>
              <option>DD/MM/YYYY</option>
              <option>YYYY-MM-DD</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

// Security Section Component
function SecuritySection({ mfaEnabled, setMfaEnabled }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Security Settings</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Two-Factor Authentication</h3>
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="font-medium">Enable 2FA</p>
            <p className="text-sm text-gray-400">Add an extra layer of security to your account</p>
          </div>
          <button type="button"
            onClick={() => setMfaEnabled(!mfaEnabled)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full ${
              mfaEnabled ? 'bg-blue-600' : 'bg-gray-600'
            }`}
          >
            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
              mfaEnabled ? 'translate-x-6' : 'translate-x-1'
            }`} />
          </button>
        </div>
        
        {mfaEnabled && (
          <div className="bg-gray-700 rounded p-4">
            <p className="text-sm text-green-400 mb-2">✓ 2FA is enabled</p>
            <button type="button" className="text-sm text-blue-400 hover:underline">Configure authenticator app</button>
          </div>
        )}
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Active Sessions</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
            <div className="flex items-center space-x-3">
              <Monitor className="w-5 h-5 text-gray-400" />
              <div>
                <p className="font-medium">Windows - Chrome</p>
                <p className="text-sm text-gray-400">Current session</p>
              </div>
            </div>
            <span className="text-xs text-green-400">Active now</span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
            <div className="flex items-center space-x-3">
              <Smartphone className="w-5 h-5 text-gray-400" />
              <div>
                <p className="font-medium">iPhone - Safari</p>
                <p className="text-sm text-gray-400">Seattle, WA</p>
              </div>
            </div>
            <button type="button" className="text-xs text-red-400 hover:underline">Revoke</button>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Password</h3>
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
          onClick={() => toast({ title: 'Password', description: 'Password change flow coming soon' })}
        >
          Change Password
        </button>
      </div>
    </div>
  );
}

// Integrations Section Component
function IntegrationsSection() {
  const integrations = [
    { name: 'Azure DevOps', status: 'connected', icon: '🔷' },
    { name: 'GitHub', status: 'connected', icon: '🐙' },
    { name: 'Slack', status: 'disconnected', icon: '💬' },
    { name: 'Jira', status: 'disconnected', icon: '📋' },
    { name: 'ServiceNow', status: 'connected', icon: '🎯' },
    { name: 'PagerDuty', status: 'disconnected', icon: '🚨' }
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Integrations</h2>
      
      <div className="grid grid-cols-2 gap-4">
        {integrations.map((integration) => (
          <div key={integration.name} className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{integration.icon}</span>
                <div>
                  <h4 className="font-semibold">{integration.name}</h4>
                  <p className="text-sm text-gray-400">
                    {integration.status === 'connected' ? 'Connected' : 'Not connected'}
                  </p>
                </div>
              </div>
              <button
                type="button"
                className={`px-3 py-1 rounded text-sm ${
                integration.status === 'connected'
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-blue-600 hover:bg-blue-700'
              } transition-colors`}
                onClick={() => toast({ title: 'Integration', description: `${integration.name}: ${integration.status === 'connected' ? 'Disconnecting' : 'Connecting'}...` })}
              >
                {integration.status === 'connected' ? 'Disconnect' : 'Connect'}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Data Privacy Section Component
function DataPrivacySection({ dataRetention, setDataRetention }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Data & Privacy</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Data Retention</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Retention Period</label>
            <select 
              value={dataRetention}
              onChange={(e) => setDataRetention(e.target.value)}
              className="w-full bg-gray-700 rounded px-4 py-2"
            >
              <option value="30">30 days</option>
              <option value="60">60 days</option>
              <option value="90">90 days</option>
              <option value="180">180 days</option>
              <option value="365">1 year</option>
              <option value="0">Forever</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Auto-delete old data</label>
            <input type="checkbox" className="w-5 h-5" defaultChecked />
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Privacy Options</h3>
        <div className="space-y-3">
          <label className="flex items-center space-x-3">
            <input type="checkbox" className="w-5 h-5" defaultChecked />
            <span>Share usage analytics to improve the product</span>
          </label>
          <label className="flex items-center space-x-3">
            <input type="checkbox" className="w-5 h-5" />
            <span>Allow personalized recommendations</span>
          </label>
          <label className="flex items-center space-x-3">
            <input type="checkbox" className="w-5 h-5" defaultChecked />
            <span>Enable audit logging</span>
          </label>
        </div>
      </div>
    </div>
  );
}

// Export Import Section Component
function ExportImportSection() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Export & Import Settings</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Export Data</h3>
        <p className="text-gray-400 mb-4">Download your settings and data for backup or migration</p>
        <div className="space-y-3">
          <button
            type="button"
            className="w-full px-4 py-3 bg-gray-700 rounded hover:bg-gray-600 transition-colors text-left"
            onClick={() => toast({ title: 'Export', description: 'Exporting settings...' })}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Export Settings</p>
                <p className="text-sm text-gray-400">Download all configuration settings</p>
              </div>
              <Download className="w-5 h-5" />
            </div>
          </button>
          
          <button
            type="button"
            className="w-full px-4 py-3 bg-gray-700 rounded hover:bg-gray-600 transition-colors text-left"
            onClick={() => toast({ title: 'Export', description: 'Exporting audit logs...' })}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Export Audit Logs</p>
                <p className="text-sm text-gray-400">Download activity history</p>
              </div>
              <Download className="w-5 h-5" />
            </div>
          </button>
          
          <button
            type="button"
            className="w-full px-4 py-3 bg-gray-700 rounded hover:bg-gray-600 transition-colors text-left"
            onClick={() => toast({ title: 'Export', description: 'Exporting full backup...' })}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Export Full Backup</p>
                <p className="text-sm text-gray-400">Complete data export</p>
              </div>
              <Download className="w-5 h-5" />
            </div>
          </button>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Import Data</h3>
        <p className="text-gray-400 mb-4">Restore settings from a backup file</p>
        <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
          <Download className="w-12 h-12 mx-auto mb-3 text-gray-500" />
          <p className="text-gray-400 mb-2">Drop files here or click to browse</p>
          <button
            type="button"
            className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
            onClick={() => toast({ title: 'Import', description: 'Opening file picker...' })}
          >
            Choose File
          </button>
        </div>
      </div>
    </div>
  );
}

// Diagnostics Section Component
function DiagnosticsSection() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">System Diagnostics</h2>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span>API Connection</span>
            <span className="flex items-center space-x-2 text-green-400">
              <CheckCircle className="w-4 h-4" />
              <span>Healthy</span>
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Database Connection</span>
            <span className="flex items-center space-x-2 text-green-400">
              <CheckCircle className="w-4 h-4" />
              <span>Connected</span>
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Cache Service</span>
            <span className="flex items-center space-x-2 text-green-400">
              <CheckCircle className="w-4 h-4" />
              <span>Active</span>
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Background Jobs</span>
            <span className="flex items-center space-x-2 text-yellow-400">
              <AlertCircle className="w-4 h-4" />
              <span>3 pending</span>
            </span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>CPU Usage</span>
              <span>42%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full" style={{ width: '42%' }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Memory Usage</span>
              <span>68%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '68%' }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Storage Usage</span>
              <span>31%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-green-500 h-2 rounded-full" style={{ width: '31%' }}></div>
            </div>
          </div>
        </div>
        
        <button
          type="button"
          className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
          onClick={() => toast({ title: 'Diagnostics', description: 'Downloading diagnostic report...' })}
        >
          Download Diagnostic Report
        </button>
      </div>
    </div>
  );
}

// Helper function for color hex values
function getColorHex(color: string): string {
  const colors: { [key: string]: string } = {
    blue: '#3b82f6',
    purple: '#8b5cf6',
    green: '#10b981',
    orange: '#f97316',
    red: '#ef4444',
    pink: '#ec4899'
  };
  return colors[color] || '#3b82f6';
}