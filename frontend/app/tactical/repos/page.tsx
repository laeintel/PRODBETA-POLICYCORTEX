'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { 
  GitBranch, Star, Eye, GitFork, Users, Code, Shield, Zap, AlertTriangle,
  Calendar, Clock, Activity, TrendingUp, TrendingDown, RefreshCw, Plus,
  Search, Filter, Download, Upload, Settings, Edit, Trash2, MoreHorizontal,
  FileText, Database, Server, Terminal, Package, Award, Target, BarChart3,
  GitCommit, GitMerge, GitPullRequest, CheckCircle, XCircle, Hash, Archive
} from 'lucide-react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';

interface Repository {
  id: string;
  name: string;
  fullName: string;
  description: string;
  language: string;
  visibility: 'public' | 'private' | 'internal';
  stars: number;
  forks: number;
  watchers: number;
  size: number; // in KB
  openIssues: number;
  openPrs: number;
  branches: number;
  tags: number;
  contributors: number;
  lastCommit: string;
  lastActivity: string;
  defaultBranch: string;
  license: string;
  topics: string[];
  codeHealth: number;
  security: number;
  coverage: number;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  commits: {
    today: number;
    thisWeek: number;
    thisMonth: number;
  };
  pullRequests: {
    merged: number;
    open: number;
    closed: number;
  };
  deployments: number;
  isArchived: boolean;
  isTemplate: boolean;
  hasWiki: boolean;
  hasPages: boolean;
  hasIssues: boolean;
  hasProjects: boolean;
  hasDiscussions: boolean;
}

interface RepoMetrics {
  totalRepos: number;
  activeRepos: number;
  archivedRepos: number;
  privateRepos: number;
  publicRepos: number;
  totalStars: number;
  totalForks: number;
  totalCommitsToday: number;
  totalOpenIssues: number;
  totalOpenPrs: number;
  averageCodeHealth: number;
  criticalVulnerabilities: number;
  languageDistribution: Record<string, number>;
}

const mockRepositories: Repository[] = [
  {
    id: '1',
    name: 'policycortex-core',
    fullName: 'policycortex/policycortex-core',
    description: 'Core governance engine for PolicyCortex platform with advanced AI-driven compliance analysis',
    language: 'Rust',
    visibility: 'private',
    stars: 234,
    forks: 42,
    watchers: 89,
    size: 45230,
    openIssues: 12,
    openPrs: 4,
    branches: 23,
    tags: 18,
    contributors: 15,
    lastCommit: '2024-01-20T14:30:00Z',
    lastActivity: '2024-01-20T14:45:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['governance', 'rust', 'ai', 'compliance', 'policy'],
    codeHealth: 92,
    security: 96,
    coverage: 87,
    vulnerabilities: { critical: 0, high: 1, medium: 3, low: 8 },
    commits: { today: 8, thisWeek: 34, thisMonth: 156 },
    pullRequests: { merged: 234, open: 4, closed: 12 },
    deployments: 45,
    isArchived: false,
    isTemplate: false,
    hasWiki: true,
    hasPages: true,
    hasIssues: true,
    hasProjects: true,
    hasDiscussions: true
  },
  {
    id: '2',
    name: 'frontend-dashboard',
    fullName: 'policycortex/frontend-dashboard',
    description: 'Next.js dashboard for PolicyCortex with modern UI and real-time analytics',
    language: 'TypeScript',
    visibility: 'private',
    stars: 89,
    forks: 23,
    watchers: 45,
    size: 28450,
    openIssues: 8,
    openPrs: 3,
    branches: 15,
    tags: 12,
    contributors: 8,
    lastCommit: '2024-01-20T11:15:00Z',
    lastActivity: '2024-01-20T13:20:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['nextjs', 'typescript', 'dashboard', 'react', 'tailwind'],
    codeHealth: 89,
    security: 94,
    coverage: 82,
    vulnerabilities: { critical: 0, high: 0, medium: 2, low: 5 },
    commits: { today: 5, thisWeek: 28, thisMonth: 124 },
    pullRequests: { merged: 156, open: 3, closed: 8 },
    deployments: 67,
    isArchived: false,
    isTemplate: false,
    hasWiki: true,
    hasPages: false,
    hasIssues: true,
    hasProjects: true,
    hasDiscussions: false
  },
  {
    id: '3',
    name: 'ai-engine',
    fullName: 'policycortex/ai-engine',
    description: 'Machine learning models and AI inference engine for policy analysis and predictions',
    language: 'Python',
    visibility: 'private',
    stars: 156,
    forks: 31,
    watchers: 67,
    size: 78920,
    openIssues: 15,
    openPrs: 6,
    branches: 18,
    tags: 24,
    contributors: 12,
    lastCommit: '2024-01-19T16:45:00Z',
    lastActivity: '2024-01-20T09:30:00Z',
    defaultBranch: 'main',
    license: 'Apache-2.0',
    topics: ['ai', 'machine-learning', 'python', 'tensorflow', 'policy-analysis'],
    codeHealth: 85,
    security: 91,
    coverage: 79,
    vulnerabilities: { critical: 1, high: 2, medium: 5, low: 12 },
    commits: { today: 2, thisWeek: 15, thisMonth: 89 },
    pullRequests: { merged: 187, open: 6, closed: 23 },
    deployments: 34,
    isArchived: false,
    isTemplate: false,
    hasWiki: true,
    hasPages: false,
    hasIssues: true,
    hasProjects: true,
    hasDiscussions: true
  },
  {
    id: '4',
    name: 'infrastructure',
    fullName: 'policycortex/infrastructure',
    description: 'Terraform configurations and infrastructure as code for Azure cloud deployment',
    language: 'HCL',
    visibility: 'private',
    stars: 45,
    forks: 12,
    watchers: 34,
    size: 15670,
    openIssues: 4,
    openPrs: 2,
    branches: 8,
    tags: 15,
    contributors: 6,
    lastCommit: '2024-01-18T14:20:00Z',
    lastActivity: '2024-01-19T10:15:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['terraform', 'infrastructure', 'azure', 'iac', 'devops'],
    codeHealth: 95,
    security: 98,
    coverage: 0,
    vulnerabilities: { critical: 0, high: 0, medium: 1, low: 2 },
    commits: { today: 0, thisWeek: 3, thisMonth: 24 },
    pullRequests: { merged: 89, open: 2, closed: 5 },
    deployments: 12,
    isArchived: false,
    isTemplate: true,
    hasWiki: true,
    hasPages: false,
    hasIssues: true,
    hasProjects: false,
    hasDiscussions: false
  },
  {
    id: '5',
    name: 'api-gateway',
    fullName: 'policycortex/api-gateway',
    description: 'GraphQL federation gateway providing unified API access across microservices',
    language: 'JavaScript',
    visibility: 'private',
    stars: 67,
    forks: 18,
    watchers: 29,
    size: 12340,
    openIssues: 6,
    openPrs: 1,
    branches: 12,
    tags: 9,
    contributors: 7,
    lastCommit: '2024-01-20T08:45:00Z',
    lastActivity: '2024-01-20T12:30:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['graphql', 'api-gateway', 'microservices', 'nodejs', 'federation'],
    codeHealth: 91,
    security: 93,
    coverage: 85,
    vulnerabilities: { critical: 0, high: 0, medium: 1, low: 4 },
    commits: { today: 3, thisWeek: 18, thisMonth: 67 },
    pullRequests: { merged: 123, open: 1, closed: 7 },
    deployments: 28,
    isArchived: false,
    isTemplate: false,
    hasWiki: false,
    hasPages: false,
    hasIssues: true,
    hasProjects: true,
    hasDiscussions: false
  },
  {
    id: '6',
    name: 'mobile-app',
    fullName: 'policycortex/mobile-app',
    description: 'React Native mobile application for PolicyCortex governance on-the-go',
    language: 'TypeScript',
    visibility: 'private',
    stars: 34,
    forks: 8,
    watchers: 19,
    size: 23450,
    openIssues: 11,
    openPrs: 3,
    branches: 9,
    tags: 6,
    contributors: 4,
    lastCommit: '2024-01-17T15:30:00Z',
    lastActivity: '2024-01-19T11:45:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['react-native', 'mobile', 'ios', 'android', 'governance'],
    codeHealth: 78,
    security: 89,
    coverage: 72,
    vulnerabilities: { critical: 0, high: 1, medium: 4, low: 7 },
    commits: { today: 0, thisWeek: 8, thisMonth: 45 },
    pullRequests: { merged: 67, open: 3, closed: 12 },
    deployments: 18,
    isArchived: false,
    isTemplate: false,
    hasWiki: false,
    hasPages: false,
    hasIssues: true,
    hasProjects: true,
    hasDiscussions: false
  },
  {
    id: '7',
    name: 'docs-site',
    fullName: 'policycortex/docs-site',
    description: 'Official documentation website built with Docusaurus for PolicyCortex platform',
    language: 'MDX',
    visibility: 'public',
    stars: 78,
    forks: 45,
    watchers: 123,
    size: 8920,
    openIssues: 3,
    openPrs: 2,
    branches: 6,
    tags: 8,
    contributors: 23,
    lastCommit: '2024-01-20T13:15:00Z',
    lastActivity: '2024-01-20T13:15:00Z',
    defaultBranch: 'main',
    license: 'CC-BY-4.0',
    topics: ['documentation', 'docusaurus', 'mdx', 'website', 'public'],
    codeHealth: 94,
    security: 99,
    coverage: 0,
    vulnerabilities: { critical: 0, high: 0, medium: 0, low: 1 },
    commits: { today: 4, thisWeek: 12, thisMonth: 56 },
    pullRequests: { merged: 89, open: 2, closed: 3 },
    deployments: 34,
    isArchived: false,
    isTemplate: false,
    hasWiki: false,
    hasPages: true,
    hasIssues: true,
    hasProjects: false,
    hasDiscussions: true
  },
  {
    id: '8',
    name: 'legacy-migration',
    fullName: 'policycortex/legacy-migration',
    description: 'Migration tools and scripts for transitioning from legacy governance systems',
    language: 'Python',
    visibility: 'private',
    stars: 12,
    forks: 3,
    watchers: 8,
    size: 5430,
    openIssues: 2,
    openPrs: 0,
    branches: 4,
    tags: 3,
    contributors: 3,
    lastCommit: '2024-01-15T10:20:00Z',
    lastActivity: '2024-01-16T14:30:00Z',
    defaultBranch: 'main',
    license: 'MIT',
    topics: ['migration', 'legacy', 'scripts', 'python', 'etl'],
    codeHealth: 82,
    security: 95,
    coverage: 68,
    vulnerabilities: { critical: 0, high: 0, medium: 1, low: 3 },
    commits: { today: 0, thisWeek: 0, thisMonth: 8 },
    pullRequests: { merged: 23, open: 0, closed: 2 },
    deployments: 5,
    isArchived: true,
    isTemplate: false,
    hasWiki: true,
    hasPages: false,
    hasIssues: true,
    hasProjects: false,
    hasDiscussions: false
  }
];

export default function ReposPage() {
  const [repositories, setRepositories] = useState<Repository[]>(mockRepositories);
  const [selectedRepo, setSelectedRepo] = useState<Repository | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [languageFilter, setLanguageFilter] = useState<string>('all');
  const [visibilityFilter, setVisibilityFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'stars' | 'lastActivity' | 'size'>('lastActivity');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [showArchived, setShowArchived] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Calculate metrics
  const metrics: RepoMetrics = useMemo(() => {
    const activeRepos = repositories.filter(r => !r.isArchived);
    const archivedRepos = repositories.filter(r => r.isArchived);
    const privateRepos = repositories.filter(r => r.visibility === 'private');
    const publicRepos = repositories.filter(r => r.visibility === 'public');
    
    const languageDistribution = repositories.reduce((acc, repo) => {
      acc[repo.language] = (acc[repo.language] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalRepos: repositories.length,
      activeRepos: activeRepos.length,
      archivedRepos: archivedRepos.length,
      privateRepos: privateRepos.length,
      publicRepos: publicRepos.length,
      totalStars: repositories.reduce((acc, r) => acc + r.stars, 0),
      totalForks: repositories.reduce((acc, r) => acc + r.forks, 0),
      totalCommitsToday: repositories.reduce((acc, r) => acc + r.commits.today, 0),
      totalOpenIssues: repositories.reduce((acc, r) => acc + r.openIssues, 0),
      totalOpenPrs: repositories.reduce((acc, r) => acc + r.openPrs, 0),
      averageCodeHealth: repositories.reduce((acc, r) => acc + r.codeHealth, 0) / repositories.length,
      criticalVulnerabilities: repositories.reduce((acc, r) => acc + r.vulnerabilities.critical, 0),
      languageDistribution
    };
  }, [repositories]);

  // Auto-refresh simulation
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      setRepositories(prevRepos => 
        prevRepos.map(repo => ({
          ...repo,
          commits: {
            ...repo.commits,
            today: repo.commits.today + Math.floor(Math.random() * 2)
          }
        }))
      );
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const filteredAndSortedRepos = useMemo(() => {
    let filtered = repositories.filter(repo => {
      const matchesSearch = repo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           repo.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           repo.topics.some(topic => topic.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesLanguage = languageFilter === 'all' || repo.language === languageFilter;
      const matchesVisibility = visibilityFilter === 'all' || repo.visibility === visibilityFilter;
      const matchesArchived = showArchived || !repo.isArchived;
      
      return matchesSearch && matchesLanguage && matchesVisibility && matchesArchived;
    });

    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'stars':
          comparison = a.stars - b.stars;
          break;
        case 'lastActivity':
          comparison = new Date(a.lastActivity).getTime() - new Date(b.lastActivity).getTime();
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [repositories, searchTerm, languageFilter, visibilityFilter, showArchived, sortBy, sortOrder]);

  const getLanguageColor = (language: string) => {
    const colors: Record<string, string> = {
      'Rust': 'bg-orange-500',
      'TypeScript': 'bg-blue-500',
      'JavaScript': 'bg-yellow-500',
      'Python': 'bg-green-500',
      'HCL': 'bg-purple-500',
      'MDX': 'bg-pink-500',
      'Go': 'bg-cyan-500',
      'Java': 'bg-red-500',
    };
    return colors[language] || 'bg-gray-500';
  };

  const formatSize = (sizeInKB: number) => {
    if (sizeInKB < 1024) return `${sizeInKB} KB`;
    if (sizeInKB < 1024 * 1024) return `${(sizeInKB / 1024).toFixed(1)} MB`;
    return `${(sizeInKB / (1024 * 1024)).toFixed(1)} GB`;
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return 'Just now';
  };

  const getHealthColor = (health: number) => {
    if (health >= 90) return 'text-green-400';
    if (health >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <TacticalPageTemplate
      title="Git Repositories"
      subtitle="Repository Management & Analytics Dashboard"
      icon={GitBranch}
    >
      <div className="space-y-6">
        {/* Metrics Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-gray-300">Total Repos</span>
              </div>
              <span className="text-xs text-blue-400 bg-blue-400/10 px-2 py-1 rounded-full">
                {metrics.activeRepos} active
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{metrics.totalRepos}</span>
              <span className="text-sm text-gray-400">{metrics.archivedRepos} archived</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Star className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-medium text-gray-300">Total Stars</span>
              </div>
              <TrendingUp className="w-4 h-4 text-green-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{metrics.totalStars.toLocaleString()}</span>
              <span className="text-sm text-green-400">+12 today</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <GitCommit className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium text-gray-300">Commits Today</span>
              </div>
              <Activity className="w-4 h-4 text-green-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{metrics.totalCommitsToday}</span>
              <span className="text-sm text-green-400">across {metrics.activeRepos} repos</span>
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-gray-300">Code Health</span>
              </div>
              <Award className="w-4 h-4 text-blue-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{Math.round(metrics.averageCodeHealth)}%</span>
              <span className={`text-sm ${metrics.criticalVulnerabilities > 0 ? 'text-red-400' : 'text-green-400'}`}>
                {metrics.criticalVulnerabilities} critical issues
              </span>
            </div>
          </div>
        </div>

        {/* Language Distribution & Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">Language Distribution</h3>
              <Code className="w-4 h-4 text-gray-400" />
            </div>
            <div className="space-y-2">
              {Object.entries(metrics.languageDistribution).map(([language, count]) => (
                <div key={language} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${getLanguageColor(language)}`} />
                    <span className="text-sm text-gray-300">{language}</span>
                  </div>
                  <span className="text-sm font-medium text-white">{count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-black border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">Repository Activity</h3>
              <BarChart3 className="w-4 h-4 text-gray-400" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Open Issues</span>
                <span className="text-sm font-medium text-white">{metrics.totalOpenIssues}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Open PRs</span>
                <span className="text-sm font-medium text-white">{metrics.totalOpenPrs}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Total Forks</span>
                <span className="text-sm font-medium text-white">{metrics.totalForks}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Private/Public</span>
                <span className="text-sm font-medium text-white">{metrics.privateRepos}/{metrics.publicRepos}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-black border border-gray-800 rounded-xl p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search and Filters */}
            <div className="flex-1 flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search repositories..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                />
              </div>
              
              <select
                value={languageFilter}
                onChange={(e) => setLanguageFilter(e.target.value)}
                className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Languages</option>
                {Object.keys(metrics.languageDistribution).map(lang => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>

              <select
                value={visibilityFilter}
                onChange={(e) => setVisibilityFilter(e.target.value)}
                className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Visibility</option>
                <option value="public">Public</option>
                <option value="private">Private</option>
                <option value="internal">Internal</option>
              </select>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowArchived(!showArchived)}
                className={`px-4 py-2 border rounded-lg transition-colors ${
                  showArchived 
                    ? 'bg-yellow-600/20 border-yellow-500/30 text-yellow-400' 
                    : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Show Archived
              </button>
              
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors ${
                  autoRefresh 
                    ? 'bg-green-600/20 border-green-500/30 text-green-400' 
                    : 'bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                Auto Refresh
              </button>

              <div className="flex border border-gray-700 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 transition-colors ${
                    viewMode === 'grid' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 transition-colors ${
                    viewMode === 'list' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <FileText className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Repository Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredAndSortedRepos.map((repo) => (
              <div key={repo.id} className={`bg-black border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors ${repo.isArchived ? 'opacity-60' : ''}`}>
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-white truncate">{repo.name}</h3>
                      {repo.isTemplate && <Award className="w-4 h-4 text-yellow-400" />}
                      {repo.isArchived && <Archive className="w-4 h-4 text-gray-400" />}
                    </div>
                    <p className="text-sm text-gray-400 line-clamp-2">{repo.description}</p>
                  </div>
                  <div className="flex items-center gap-1 ml-2">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      repo.visibility === 'public' ? 'bg-green-500/20 text-green-400' :
                      repo.visibility === 'private' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {repo.visibility}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-4 mb-3 text-sm text-gray-400">
                  <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded-full ${getLanguageColor(repo.language)}`} />
                    <span>{repo.language}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Star className="w-3 h-3" />
                    <span>{repo.stars}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <GitFork className="w-3 h-3" />
                    <span>{repo.forks}</span>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Code Health:</span>
                    <span className={getHealthColor(repo.codeHealth)}>{repo.codeHealth}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Coverage:</span>
                    <span className="text-gray-300">{repo.coverage}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Size:</span>
                    <span className="text-gray-300">{formatSize(repo.size)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Last Activity:</span>
                    <span className="text-gray-300">{formatTimeAgo(repo.lastActivity)}</span>
                  </div>
                </div>

                {/* Vulnerability Summary */}
                {(repo.vulnerabilities.critical > 0 || repo.vulnerabilities.high > 0) && (
                  <div className="mb-4 p-2 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <div className="flex items-center gap-2 text-sm">
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                      <span className="text-red-300">
                        {repo.vulnerabilities.critical + repo.vulnerabilities.high} security issues
                      </span>
                    </div>
                  </div>
                )}

                {/* Topics */}
                <div className="flex flex-wrap gap-1 mb-4">
                  {repo.topics.slice(0, 3).map((topic) => (
                    <span key={topic} className="px-2 py-1 bg-gray-800 text-xs text-gray-300 rounded">
                      {topic}
                    </span>
                  ))}
                  {repo.topics.length > 3 && (
                    <span className="px-2 py-1 bg-gray-800 text-xs text-gray-400 rounded">
                      +{repo.topics.length - 3}
                    </span>
                  )}
                </div>

                {/* Activity Stats */}
                <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
                  <div className="text-center">
                    <div className="font-bold text-white">{repo.commits.today}</div>
                    <div className="text-gray-400">Commits Today</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-white">{repo.openIssues}</div>
                    <div className="text-gray-400">Open Issues</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-white">{repo.openPrs}</div>
                    <div className="text-gray-400">Open PRs</div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedRepo(repo)}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    <Eye className="w-4 h-4" />
                    Details
                  </button>
                  
                  <button className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                    <GitBranch className="w-4 h-4" />
                  </button>
                  
                  <button className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-black border border-gray-800 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Repository</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Language</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Visibility</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Stars</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Issues</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">PRs</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Health</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Last Activity</th>
                    <th className="text-left p-4 text-sm font-medium text-gray-300">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSortedRepos.map((repo) => (
                    <tr key={repo.id} className={`border-b border-gray-800 hover:bg-gray-900/50 ${repo.isArchived ? 'opacity-60' : ''}`}>
                      <td className="p-4">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-white">{repo.name}</span>
                            {repo.isTemplate && <Award className="w-4 h-4 text-yellow-400" />}
                            {repo.isArchived && <Archive className="w-4 h-4 text-gray-400" />}
                          </div>
                          <div className="text-sm text-gray-400 truncate max-w-md">{repo.description}</div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full ${getLanguageColor(repo.language)}`} />
                          <span className="text-sm text-gray-300">{repo.language}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          repo.visibility === 'public' ? 'bg-green-500/20 text-green-400' :
                          repo.visibility === 'private' ? 'bg-red-500/20 text-red-400' :
                          'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {repo.visibility}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-gray-300">{repo.stars}</td>
                      <td className="p-4 text-sm text-gray-300">{repo.openIssues}</td>
                      <td className="p-4 text-sm text-gray-300">{repo.openPrs}</td>
                      <td className="p-4">
                        <span className={`text-sm font-medium ${getHealthColor(repo.codeHealth)}`}>
                          {repo.codeHealth}%
                        </span>
                      </td>
                      <td className="p-4 text-sm text-gray-300">{formatTimeAgo(repo.lastActivity)}</td>
                      <td className="p-4">
                        <div className="flex gap-2">
                          <button
                            onClick={() => setSelectedRepo(repo)}
                            className="p-1 text-blue-400 hover:text-blue-300"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="p-1 text-green-400 hover:text-green-300">
                            <GitBranch className="w-4 h-4" />
                          </button>
                          <button className="p-1 text-gray-400 hover:text-gray-300">
                            <Settings className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Repository Details Modal */}
        {selectedRepo && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-black border border-gray-800 rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className={`w-6 h-6 rounded-full ${getLanguageColor(selectedRepo.language)}`} />
                    <h2 className="text-xl font-bold text-white">{selectedRepo.name}</h2>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      selectedRepo.visibility === 'public' ? 'bg-green-500/20 text-green-400' :
                      selectedRepo.visibility === 'private' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {selectedRepo.visibility}
                    </span>
                    {selectedRepo.isTemplate && <Award className="w-5 h-5 text-yellow-400" />}
                    {selectedRepo.isArchived && <Archive className="w-5 h-5 text-gray-400" />}
                  </div>
                  <button
                    onClick={() => setSelectedRepo(null)}
                    className="text-gray-400 hover:text-white"
                  >
                    <XCircle className="w-6 h-6" />
                  </button>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-6">
                    {/* Description */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-2">Description</h3>
                      <p className="text-gray-300">{selectedRepo.description}</p>
                    </div>

                    {/* Metrics */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Repository Metrics</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{selectedRepo.stars}</div>
                          <div className="text-xs text-gray-400">Stars</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{selectedRepo.forks}</div>
                          <div className="text-xs text-gray-400">Forks</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{selectedRepo.watchers}</div>
                          <div className="text-xs text-gray-400">Watchers</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{selectedRepo.contributors}</div>
                          <div className="text-xs text-gray-400">Contributors</div>
                        </div>
                      </div>
                    </div>

                    {/* Security & Quality */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Security & Quality</h3>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Code Health</span>
                            <span className={`text-sm font-medium ${getHealthColor(selectedRepo.codeHealth)}`}>
                              {selectedRepo.codeHealth}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                selectedRepo.codeHealth >= 90 ? 'bg-green-600' :
                                selectedRepo.codeHealth >= 70 ? 'bg-yellow-600' : 'bg-red-600'
                              }`}
                              style={{ width: `${selectedRepo.codeHealth}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Security Score</span>
                            <span className="text-sm text-white">{selectedRepo.security}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-purple-600 h-2 rounded-full"
                              style={{ width: `${selectedRepo.security}%` }}
                            />
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <span className="text-sm text-gray-400">Test Coverage</span>
                            <span className="text-sm text-white">{selectedRepo.coverage}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${selectedRepo.coverage}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Vulnerabilities */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Security Vulnerabilities</h3>
                      <div className="grid grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-lg font-bold text-red-400">{selectedRepo.vulnerabilities.critical}</div>
                          <div className="text-xs text-gray-400">Critical</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-orange-400">{selectedRepo.vulnerabilities.high}</div>
                          <div className="text-xs text-gray-400">High</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-yellow-400">{selectedRepo.vulnerabilities.medium}</div>
                          <div className="text-xs text-gray-400">Medium</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-bold text-blue-400">{selectedRepo.vulnerabilities.low}</div>
                          <div className="text-xs text-gray-400">Low</div>
                        </div>
                      </div>
                    </div>

                    {/* Topics */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Topics</h3>
                      <div className="flex flex-wrap gap-2">
                        {selectedRepo.topics.map((topic) => (
                          <span key={topic} className="px-3 py-1 bg-gray-800 text-sm text-gray-300 rounded-full">
                            {topic}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    {/* Repository Info */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Repository Info</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="text-xs text-gray-400">Full Name</div>
                          <div className="text-sm text-white">{selectedRepo.fullName}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Primary Language</div>
                          <div className="text-sm text-white">{selectedRepo.language}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Default Branch</div>
                          <div className="text-sm text-white">{selectedRepo.defaultBranch}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">License</div>
                          <div className="text-sm text-white">{selectedRepo.license}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Size</div>
                          <div className="text-sm text-white">{formatSize(selectedRepo.size)}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Last Commit</div>
                          <div className="text-sm text-white">{formatTimeAgo(selectedRepo.lastCommit)}</div>
                        </div>
                      </div>
                    </div>

                    {/* Activity Stats */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Activity Statistics</h3>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Commits (Today)</span>
                          <span className="text-sm text-white">{selectedRepo.commits.today}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Commits (This Week)</span>
                          <span className="text-sm text-white">{selectedRepo.commits.thisWeek}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Commits (This Month)</span>
                          <span className="text-sm text-white">{selectedRepo.commits.thisMonth}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Open Issues</span>
                          <span className="text-sm text-white">{selectedRepo.openIssues}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Open PRs</span>
                          <span className="text-sm text-white">{selectedRepo.openPrs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Branches</span>
                          <span className="text-sm text-white">{selectedRepo.branches}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-400">Tags</span>
                          <span className="text-sm text-white">{selectedRepo.tags}</span>
                        </div>
                      </div>
                    </div>

                    {/* Features */}
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-white mb-3">Repository Features</h3>
                      <div className="space-y-2">
                        {[
                          { key: 'hasWiki', label: 'Wiki', enabled: selectedRepo.hasWiki },
                          { key: 'hasPages', label: 'GitHub Pages', enabled: selectedRepo.hasPages },
                          { key: 'hasIssues', label: 'Issues', enabled: selectedRepo.hasIssues },
                          { key: 'hasProjects', label: 'Projects', enabled: selectedRepo.hasProjects },
                          { key: 'hasDiscussions', label: 'Discussions', enabled: selectedRepo.hasDiscussions },
                        ].map(({ key, label, enabled }) => (
                          <div key={key} className="flex items-center justify-between">
                            <span className="text-sm text-gray-300">{label}</span>
                            {enabled ? (
                              <CheckCircle className="w-4 h-4 text-green-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-gray-500" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="space-y-2">
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                        <GitBranch className="w-4 h-4" />
                        View Repository
                      </button>
                      
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors">
                        <GitCommit className="w-4 h-4" />
                        View Commits
                      </button>
                      
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                        <GitPullRequest className="w-4 h-4" />
                        View Pull Requests
                      </button>
                      
                      <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                        <Settings className="w-4 h-4" />
                        Repository Settings
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </TacticalPageTemplate>
  );
}