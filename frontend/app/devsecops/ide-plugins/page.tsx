'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Code,
  Download,
  Star,
  Users,
  CheckCircle,
  AlertTriangle,
  Settings,
  Plus,
  RefreshCw,
  ExternalLink,
  Brain,
  Target,
  Activity,
  TrendingUp,
  BarChart3,
  Shield,
  Zap,
  FileCode,
  Terminal,
  Search,
  Filter,
  Eye,
  GitBranch,
  Bug,
  Lock,
  Database,
  Globe
} from 'lucide-react';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface IDEPlugin {
  id: string;
  name: string;
  description: string;
  ide: 'vscode' | 'intellij' | 'eclipse' | 'sublime' | 'atom' | 'vim';
  category: 'security' | 'compliance' | 'policy' | 'scanning' | 'reporting';
  version: string;
  author: string;
  downloads: number;
  rating: number;
  reviews: number;
  lastUpdated: Date;
  features: string[];
  supportedLanguages: string[];
  integrations: string[];
  installCommand: string;
  marketplaceUrl: string;
  documentationUrl: string;
  status: 'published' | 'beta' | 'deprecated' | 'in-development';
  usageStats: {
    activeUsers: number;
    dailyScans: number;
    issuesFound: number;
    falsePositives: number;
  };
}

interface PluginUsage {
  pluginId: string;
  pluginName: string;
  users: number;
  scans: number;
  findings: number;
  date: Date;
}

export default function IDEPluginsPage() {
  const router = useRouter();
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [plugins, setPlugins] = useState<IDEPlugin[]>([]);
  const [usage, setUsage] = useState<PluginUsage[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlugin, setSelectedPlugin] = useState<string | null>(null);
  const [filterIDE, setFilterIDE] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');

  useEffect(() => {
    loadPluginData();
  }, []);

  const loadPluginData = async () => {
    // Load ML predictions for plugin adoption
    const adoptionPrediction = await MLPredictionEngine.predictCostSpike('ide-plugin-adoption');
    setPredictions([adoptionPrediction]);

    // Mock IDE plugins
    const mockPlugins: IDEPlugin[] = [
      {
        id: 'plugin-001',
        name: 'PolicyCortex Security Scanner',
        description: 'Real-time security scanning and policy validation directly in your editor',
        ide: 'vscode',
        category: 'security',
        version: '2.4.1',
        author: 'PolicyCortex Team',
        downloads: 15420,
        rating: 4.8,
        reviews: 342,
        lastUpdated: new Date(Date.now() - 86400000),
        features: [
          'Real-time SAST scanning',
          'Policy violation detection',
          'Inline remediation suggestions',
          'Security hotspot highlighting',
          'Custom rule configuration'
        ],
        supportedLanguages: ['JavaScript', 'TypeScript', 'Python', 'Java', 'C#', 'Go'],
        integrations: ['SonarQube', 'Checkmarx', 'Veracode', 'Azure DevOps'],
        installCommand: 'ext install policycortex.security-scanner',
        marketplaceUrl: 'https://marketplace.visualstudio.com/items?itemName=policycortex.security-scanner',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/vscode',
        status: 'published',
        usageStats: {
          activeUsers: 3240,
          dailyScans: 12580,
          issuesFound: 892,
          falsePositives: 23
        }
      },
      {
        id: 'plugin-002',
        name: 'Compliance Checker for IntelliJ',
        description: 'Automated compliance validation for enterprise development workflows',
        ide: 'intellij',
        category: 'compliance',
        version: '1.8.3',
        author: 'PolicyCortex Team',
        downloads: 8730,
        rating: 4.6,
        reviews: 156,
        lastUpdated: new Date(Date.now() - 259200000),
        features: [
          'GDPR compliance checking',
          'SOX compliance validation',
          'HIPAA data protection',
          'Custom compliance frameworks',
          'Automated documentation'
        ],
        supportedLanguages: ['Java', 'Kotlin', 'Scala', 'Groovy'],
        integrations: ['Jenkins', 'GitLab CI', 'Bamboo'],
        installCommand: 'Install from JetBrains Plugin Repository',
        marketplaceUrl: 'https://plugins.jetbrains.com/plugin/policycortex-compliance',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/intellij',
        status: 'published',
        usageStats: {
          activeUsers: 1840,
          dailyScans: 6420,
          issuesFound: 234,
          falsePositives: 12
        }
      },
      {
        id: 'plugin-003',
        name: 'Policy-as-Code Assistant',
        description: 'Intelligent assistance for writing and testing policy definitions',
        ide: 'vscode',
        category: 'policy',
        version: '1.2.0',
        author: 'PolicyCortex Team',
        downloads: 5680,
        rating: 4.7,
        reviews: 89,
        lastUpdated: new Date(Date.now() - 172800000),
        features: [
          'Rego syntax highlighting',
          'Policy template library',
          'Test case generation',
          'Policy simulation',
          'Live policy validation'
        ],
        supportedLanguages: ['Rego', 'Sentinel', 'YAML', 'JSON'],
        integrations: ['Open Policy Agent', 'Terraform', 'Kubernetes'],
        installCommand: 'ext install policycortex.policy-assistant',
        marketplaceUrl: 'https://marketplace.visualstudio.com/items?itemName=policycortex.policy-assistant',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/policy-assistant',
        status: 'published',
        usageStats: {
          activeUsers: 1520,
          dailyScans: 2340,
          issuesFound: 156,
          falsePositives: 8
        }
      },
      {
        id: 'plugin-004',
        name: 'Infrastructure Security Lint',
        description: 'Security linting for Infrastructure as Code templates and configurations',
        ide: 'vscode',
        category: 'security',
        version: '0.9.2-beta',
        author: 'PolicyCortex Team',
        downloads: 2340,
        rating: 4.5,
        reviews: 34,
        lastUpdated: new Date(Date.now() - 432000000),
        features: [
          'Terraform security scanning',
          'ARM template validation',
          'CloudFormation checks',
          'Kubernetes YAML analysis',
          'Security misconfiguration detection'
        ],
        supportedLanguages: ['HCL', 'YAML', 'JSON'],
        integrations: ['Terraform', 'Azure Resource Manager', 'AWS CloudFormation'],
        installCommand: 'ext install policycortex.infra-security',
        marketplaceUrl: 'https://marketplace.visualstudio.com/items?itemName=policycortex.infra-security',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/infra-security',
        status: 'beta',
        usageStats: {
          activeUsers: 680,
          dailyScans: 1240,
          issuesFound: 89,
          falsePositives: 15
        }
      },
      {
        id: 'plugin-005',
        name: 'DevSecOps Dashboard',
        description: 'Centralized dashboard for security metrics and pipeline status in Eclipse',
        ide: 'eclipse',
        category: 'reporting',
        version: '1.5.1',
        author: 'PolicyCortex Team',
        downloads: 3450,
        rating: 4.3,
        reviews: 67,
        lastUpdated: new Date(Date.now() - 518400000),
        features: [
          'Security metrics dashboard',
          'Pipeline status monitoring',
          'Vulnerability tracking',
          'Compliance reporting',
          'Team collaboration tools'
        ],
        supportedLanguages: ['Java', 'C++', 'Python'],
        integrations: ['Jenkins', 'GitHub Actions', 'SonarQube'],
        installCommand: 'Install from Eclipse Marketplace',
        marketplaceUrl: 'https://marketplace.eclipse.org/content/policycortex-devsecops',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/eclipse',
        status: 'published',
        usageStats: {
          activeUsers: 920,
          dailyScans: 1580,
          issuesFound: 123,
          falsePositives: 19
        }
      },
      {
        id: 'plugin-006',
        name: 'Secret Scanner Pro',
        description: 'Advanced secret detection and credential management for development environments',
        ide: 'intellij',
        category: 'security',
        version: '3.1.0',
        author: 'PolicyCortex Team',
        downloads: 7820,
        rating: 4.9,
        reviews: 203,
        lastUpdated: new Date(Date.now() - 345600000),
        features: [
          'Real-time secret detection',
          'Custom pattern recognition',
          'Safe credential storage',
          'Git pre-commit hooks',
          'Team secret sharing'
        ],
        supportedLanguages: ['All languages'],
        integrations: ['Azure Key Vault', 'AWS Secrets Manager', 'HashiCorp Vault'],
        installCommand: 'Install from JetBrains Plugin Repository',
        marketplaceUrl: 'https://plugins.jetbrains.com/plugin/policycortex-secrets',
        documentationUrl: 'https://docs.policycortex.com/ide-plugins/secret-scanner',
        status: 'published',
        usageStats: {
          activeUsers: 2150,
          dailyScans: 8940,
          issuesFound: 345,
          falsePositives: 7
        }
      }
    ];

    setPlugins(mockPlugins);

    // Mock usage data
    const mockUsage: PluginUsage[] = mockPlugins.map(plugin => ({
      pluginId: plugin.id,
      pluginName: plugin.name,
      users: plugin.usageStats.activeUsers,
      scans: plugin.usageStats.dailyScans,
      findings: plugin.usageStats.issuesFound,
      date: new Date()
    }));

    setUsage(mockUsage);
    setLoading(false);
  };

  const getIDEIcon = (ide: string) => {
    // Using generic Code icon for all IDEs since we don't have specific icons
    return <Code className="h-5 w-5" />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400';
      case 'beta': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400';
      case 'deprecated': return 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400';
      case 'in-development': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'security': return <Shield className="h-5 w-5" />;
      case 'compliance': return <CheckCircle className="h-5 w-5" />;
      case 'policy': return <FileCode className="h-5 w-5" />;
      case 'scanning': return <Search className="h-5 w-5" />;
      case 'reporting': return <BarChart3 className="h-5 w-5" />;
      default: return <Code className="h-5 w-5" />;
    }
  };

  const filteredPlugins = plugins.filter(plugin => {
    const ideMatch = filterIDE === 'all' || plugin.ide === filterIDE;
    const categoryMatch = filterCategory === 'all' || plugin.category === filterCategory;
    return ideMatch && categoryMatch;
  });

  const metrics = [
    {
      id: 'total-downloads',
      title: 'Total Downloads',
      value: plugins.reduce((sum, p) => sum + p.downloads, 0).toLocaleString(),
      change: 24.8,
      trend: 'up' as const,
      sparklineData: [35000, 38000, 42000, 45000, 47000, plugins.reduce((sum, p) => sum + p.downloads, 0)],
      alert: `${plugins.filter(p => p.status === 'beta').length} in beta`
    },
    {
      id: 'active-users',
      title: 'Active Users',
      value: plugins.reduce((sum, p) => sum + p.usageStats.activeUsers, 0).toLocaleString(),
      change: 18.3,
      trend: 'up' as const,
      sparklineData: [8500, 9200, 9800, 10100, 10300, plugins.reduce((sum, p) => sum + p.usageStats.activeUsers, 0)]
    },
    {
      id: 'daily-scans',
      title: 'Daily Scans',
      value: plugins.reduce((sum, p) => sum + p.usageStats.dailyScans, 0).toLocaleString(),
      change: 31.7,
      trend: 'up' as const,
      sparklineData: [25000, 27000, 29000, 31000, 32000, plugins.reduce((sum, p) => sum + p.usageStats.dailyScans, 0)]
    },
    {
      id: 'avg-rating',
      title: 'Average Rating',
      value: `${(plugins.reduce((sum, p) => sum + p.rating, 0) / plugins.length || 0).toFixed(1)}★`,
      change: 2.1,
      trend: 'up' as const,
      sparklineData: [4.4, 4.5, 4.6, 4.6, 4.7, plugins.reduce((sum, p) => sum + p.rating, 0) / plugins.length || 0]
    }
  ];

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Code className="h-10 w-10 text-green-600" />
            IDE Plugin Hub
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            VS Code, IntelliJ, and Eclipse plugins for inline policy checks and security scanning
          </p>
        </div>
        <div className="flex gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/devsecops/ide-plugins/developer')}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
          >
            <Plus className="h-5 w-5" />
            Develop Plugin
          </button>
          <button
            onClick={() => loadPluginData()}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* AI Predictions Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'medium' && (
        <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-green-600 dark:text-green-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-green-900 dark:text-green-100">
                Plugin Adoption Insight
              </h3>
              <p className="text-green-700 dark:text-green-300 mt-1">
                AI predicts 40% increase in IDE plugin adoption over the next quarter
              </p>
              <button className="mt-2 px-3 py-1 bg-green-600 text-white rounded-md text-sm hover:bg-green-700">
                View Adoption Trends
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.id}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            trend={metric.trend}
            sparklineData={metric.sparklineData}
            alert={metric.alert}
          />
        ))}
      </div>

      {view === 'cards' ? (
        <>
          {/* Filters */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4 mb-6">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <Filter className="h-5 w-5 text-gray-500" />
                <span className="font-medium">Filters:</span>
              </div>
              <select
                value={filterIDE}
                onChange={(e) => setFilterIDE(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All IDEs</option>
                <option value="vscode">VS Code</option>
                <option value="intellij">IntelliJ</option>
                <option value="eclipse">Eclipse</option>
                <option value="sublime">Sublime Text</option>
                <option value="atom">Atom</option>
                <option value="vim">Vim</option>
              </select>
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="px-3 py-1 border dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm"
              >
                <option value="all">All Categories</option>
                <option value="security">Security</option>
                <option value="compliance">Compliance</option>
                <option value="policy">Policy</option>
                <option value="scanning">Scanning</option>
                <option value="reporting">Reporting</option>
              </select>
              <span className="text-sm text-gray-500">
                {filteredPlugins.length} of {plugins.length} plugins
              </span>
            </div>
          </div>

          {/* Plugin Gallery */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {filteredPlugins.map((plugin) => (
              <div
                key={plugin.id}
                className={`bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer ${
                  selectedPlugin === plugin.id ? 'ring-2 ring-green-500' : ''
                }`}
                onClick={() => setSelectedPlugin(selectedPlugin === plugin.id ? null : plugin.id)}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        {getCategoryIcon(plugin.category)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-lg">{plugin.name}</h3>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-sm text-gray-600 dark:text-gray-400">v{plugin.version}</span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(plugin.status)}`}>
                            {plugin.status}
                          </span>
                          <div className="flex items-center gap-1">
                            <Star className="h-4 w-4 text-yellow-500 fill-current" />
                            <span className="text-sm font-medium">{plugin.rating}</span>
                            <span className="text-sm text-gray-500">({plugin.reviews})</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {getIDEIcon(plugin.ide)}
                      <span className="text-sm font-medium capitalize">{plugin.ide}</span>
                    </div>
                  </div>

                  <p className="text-gray-600 dark:text-gray-400 mb-4">{plugin.description}</p>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-sm">
                      <span className="text-gray-500">Downloads:</span>
                      <span className="font-semibold ml-2">{plugin.downloads.toLocaleString()}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-500">Active Users:</span>
                      <span className="font-semibold ml-2">{plugin.usageStats.activeUsers.toLocaleString()}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-500">Daily Scans:</span>
                      <span className="font-semibold ml-2">{plugin.usageStats.dailyScans.toLocaleString()}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-500">Issues Found:</span>
                      <span className="font-semibold ml-2 text-orange-600">{plugin.usageStats.issuesFound}</span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {plugin.features.slice(0, 3).map((feature, idx) => (
                      <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded text-xs">
                        {feature}
                      </span>
                    ))}
                    {plugin.features.length > 3 && (
                      <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded text-xs">
                        +{plugin.features.length - 3} more
                      </span>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        window.open(plugin.marketplaceUrl, '_blank');
                      }}
                      className="flex-1 px-3 py-2 bg-green-600 text-white rounded-md text-sm hover:bg-green-700 flex items-center justify-center gap-2"
                    >
                      <Download className="h-4 w-4" />
                      Install
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        window.open(plugin.documentationUrl, '_blank');
                      }}
                      className="px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center gap-2"
                    >
                      <ExternalLink className="h-4 w-4" />
                      Docs
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedPlugin(selectedPlugin === plugin.id ? null : plugin.id);
                      }}
                      className="px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  </div>

                  {selectedPlugin === plugin.id && (
                    <div className="mt-4 pt-4 border-t dark:border-gray-700">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium mb-2">Features:</h4>
                          <ul className="text-sm space-y-1">
                            {plugin.features.map((feature, idx) => (
                              <li key={idx} className="flex items-center gap-2">
                                <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                                {feature}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="space-y-3">
                          <div>
                            <h4 className="font-medium mb-2">Supported Languages:</h4>
                            <div className="flex flex-wrap gap-1">
                              {plugin.supportedLanguages.map((lang, idx) => (
                                <span key={idx} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                                  {lang}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2">Integrations:</h4>
                            <div className="flex flex-wrap gap-1">
                              {plugin.integrations.map((integration, idx) => (
                                <span key={idx} className="px-2 py-1 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded text-xs">
                                  {integration}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2">Install Command:</h4>
                            <code className="text-xs bg-gray-900 text-green-400 p-2 rounded block">
                              {plugin.installCommand}
                            </code>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Quick Install Commands */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Terminal className="h-6 w-6 text-purple-600" />
              Quick Install Commands
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-medium mb-2">VS Code Extensions</h3>
                <div className="space-y-2">
                  {plugins.filter(p => p.ide === 'vscode').map((plugin) => (
                    <div key={plugin.id} className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                      <div className="text-sm font-medium mb-1">{plugin.name}</div>
                      <code className="text-xs text-green-600 dark:text-green-400">
                        {plugin.installCommand}
                      </code>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-medium mb-2">IntelliJ Plugins</h3>
                <div className="space-y-2">
                  {plugins.filter(p => p.ide === 'intellij').map((plugin) => (
                    <div key={plugin.id} className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                      <div className="text-sm font-medium mb-1">{plugin.name}</div>
                      <code className="text-xs text-green-600 dark:text-green-400">
                        {plugin.installCommand}
                      </code>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Visualization Mode */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer
              title="Plugin Adoption by IDE"
              onDrillIn={() => router.push('/devsecops/ide-plugins/analytics')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Plugin adoption by IDE visualization</p>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer
              title="Usage Trends by Category"
              onDrillIn={() => router.push('/devsecops/ide-plugins/trends')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Category usage trend visualization</p>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* Plugin Performance Analytics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-purple-600" />
              Plugin Performance Analytics
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Most Downloaded</h3>
                <div className="space-y-2">
                  {plugins
                    .sort((a, b) => b.downloads - a.downloads)
                    .slice(0, 3)
                    .map((plugin) => (
                      <div key={plugin.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{plugin.name}</span>
                        <span className="text-blue-600 dark:text-blue-400 font-semibold text-sm">
                          {plugin.downloads.toLocaleString()}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Highest Rated</h3>
                <div className="space-y-2">
                  {plugins
                    .sort((a, b) => b.rating - a.rating)
                    .slice(0, 3)
                    .map((plugin) => (
                      <div key={plugin.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{plugin.name}</span>
                        <span className="text-yellow-600 dark:text-yellow-400 font-semibold text-sm flex items-center gap-1">
                          {plugin.rating}
                          <Star className="h-3 w-3 fill-current" />
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">Most Active</h3>
                <div className="space-y-2">
                  {plugins
                    .sort((a, b) => b.usageStats.activeUsers - a.usageStats.activeUsers)
                    .slice(0, 3)
                    .map((plugin) => (
                      <div key={plugin.id} className="flex justify-between items-center">
                        <span className="text-sm truncate">{plugin.name}</span>
                        <span className="text-green-600 dark:text-green-400 font-semibold text-sm">
                          {plugin.usageStats.activeUsers.toLocaleString()}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-3">IDE Distribution</h3>
                <div className="space-y-2">
                  {Object.entries(
                    plugins.reduce((acc, plugin) => {
                      acc[plugin.ide] = (acc[plugin.ide] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  )
                  .sort(([,a], [,b]) => b - a)
                  .map(([ide, count]) => (
                    <div key={ide} className="flex justify-between items-center">
                      <span className="text-sm capitalize">{ide}</span>
                      <span className="font-semibold text-sm">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}