'use client';

import { Card } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Settings, Bell, Shield, Globe, Database, Key, User, Mail, Save } from 'lucide-react';
import { useState } from 'react';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      slack: false,
      teams: true,
      webhooks: false
    },
    security: {
      mfa: true,
      sessionTimeout: 30,
      ipWhitelist: false,
      auditLogs: true
    },
    compliance: {
      autoRemediation: false,
      driftDetection: true,
      continuousScanning: true,
      reportGeneration: true
    },
    integration: {
      azureSync: true,
      awsSync: false,
      gcpSync: false,
      githubSync: true
    }
  });

  const handleToggle = (category: string, setting: string) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category as keyof typeof prev],
        [setting]: !prev[category as keyof typeof prev][setting as keyof typeof prev[category as keyof typeof prev]]
      }
    }));
  };

  const handleSave = () => {
    // Save settings to backend
    console.log('Saving settings:', settings);
    // Show success message
    alert('Settings saved successfully!');
  };

  return (
    <div className="container mx-auto p-6 space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Settings className="h-8 w-8" />
            Settings
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Configure your PolicyCortex platform preferences
          </p>
        </div>
        <Button onClick={handleSave} className="flex items-center gap-2">
          <Save className="h-4 w-4" />
          Save Changes
        </Button>
      </div>

      {/* Notifications Settings */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Bell className="h-5 w-5 text-blue-500" />
          <h2 className="text-xl font-semibold">Notifications</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Email Notifications</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Receive alerts and updates via email
              </p>
            </div>
            <Switch
              checked={settings.notifications.email}
              onCheckedChange={() => handleToggle('notifications', 'email')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Slack Integration</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Send notifications to Slack channels
              </p>
            </div>
            <Switch
              checked={settings.notifications.slack}
              onCheckedChange={() => handleToggle('notifications', 'slack')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Microsoft Teams</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Send notifications to Teams channels
              </p>
            </div>
            <Switch
              checked={settings.notifications.teams}
              onCheckedChange={() => handleToggle('notifications', 'teams')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Webhooks</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Send notifications to custom webhook endpoints
              </p>
            </div>
            <Switch
              checked={settings.notifications.webhooks}
              onCheckedChange={() => handleToggle('notifications', 'webhooks')}
            />
          </div>
        </div>
      </Card>

      {/* Security Settings */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Shield className="h-5 w-5 text-green-500" />
          <h2 className="text-xl font-semibold">Security</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Multi-Factor Authentication</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Require MFA for all users
              </p>
            </div>
            <Switch
              checked={settings.security.mfa}
              onCheckedChange={() => handleToggle('security', 'mfa')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Session Timeout</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Auto-logout after {settings.security.sessionTimeout} minutes of inactivity
              </p>
            </div>
            <input
              type="number"
              value={settings.security.sessionTimeout}
              onChange={(e) => setSettings(prev => ({
                ...prev,
                security: { ...prev.security, sessionTimeout: parseInt(e.target.value) }
              }))}
              className="w-20 px-3 py-1 border rounded-md dark:bg-gray-800"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">IP Whitelist</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Restrict access to specific IP addresses
              </p>
            </div>
            <Switch
              checked={settings.security.ipWhitelist}
              onCheckedChange={() => handleToggle('security', 'ipWhitelist')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Audit Logs</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Track all user actions and system events
              </p>
            </div>
            <Switch
              checked={settings.security.auditLogs}
              onCheckedChange={() => handleToggle('security', 'auditLogs')}
            />
          </div>
        </div>
      </Card>

      {/* Compliance Settings */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Database className="h-5 w-5 text-purple-500" />
          <h2 className="text-xl font-semibold">Compliance & Governance</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Auto-Remediation</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Automatically fix compliance violations
              </p>
            </div>
            <Switch
              checked={settings.compliance.autoRemediation}
              onCheckedChange={() => handleToggle('compliance', 'autoRemediation')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Drift Detection</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Monitor configuration drift from baseline
              </p>
            </div>
            <Switch
              checked={settings.compliance.driftDetection}
              onCheckedChange={() => handleToggle('compliance', 'driftDetection')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Continuous Scanning</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Continuously scan resources for compliance
              </p>
            </div>
            <Switch
              checked={settings.compliance.continuousScanning}
              onCheckedChange={() => handleToggle('compliance', 'continuousScanning')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Automated Reports</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Generate compliance reports automatically
              </p>
            </div>
            <Switch
              checked={settings.compliance.reportGeneration}
              onCheckedChange={() => handleToggle('compliance', 'reportGeneration')}
            />
          </div>
        </div>
      </Card>

      {/* Integration Settings */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Globe className="h-5 w-5 text-indigo-500" />
          <h2 className="text-xl font-semibold">Integrations</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Azure Sync</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Sync resources from Azure subscriptions
              </p>
            </div>
            <Switch
              checked={settings.integration.azureSync}
              onCheckedChange={() => handleToggle('integration', 'azureSync')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">AWS Sync</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Sync resources from AWS accounts
              </p>
            </div>
            <Switch
              checked={settings.integration.awsSync}
              onCheckedChange={() => handleToggle('integration', 'awsSync')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">GCP Sync</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Sync resources from Google Cloud projects
              </p>
            </div>
            <Switch
              checked={settings.integration.gcpSync}
              onCheckedChange={() => handleToggle('integration', 'gcpSync')}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">GitHub Integration</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Create PRs for remediation fixes
              </p>
            </div>
            <Switch
              checked={settings.integration.githubSync}
              onCheckedChange={() => handleToggle('integration', 'githubSync')}
            />
          </div>
        </div>
      </Card>

      {/* Account Settings */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <User className="h-5 w-5 text-orange-500" />
          <h2 className="text-xl font-semibold">Account</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Profile Information</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Update your name and contact details
              </p>
            </div>
            <Button variant="outline" size="sm">
              Edit Profile
            </Button>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Change Password</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Update your account password
              </p>
            </div>
            <Button variant="outline" size="sm">
              Change
            </Button>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">API Keys</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Manage API keys for programmatic access
              </p>
            </div>
            <Button variant="outline" size="sm">
              Manage Keys
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}