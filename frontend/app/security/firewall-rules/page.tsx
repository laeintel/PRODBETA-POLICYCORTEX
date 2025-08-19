'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, AlertTriangle, CheckCircle, Clock, 
  Plus, Search, Filter, Download, Upload, RotateCcw,
  Eye, EyeOff, Copy, Trash2, Edit, Settings, 
  Calendar, Map, Activity, Lock, Unlock,
  FileText, Globe, Zap, Network, Server, Target
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Badge } from '../../../components/ui/badge';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../../../components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Progress } from '../../../components/ui/progress';
import { Textarea } from '../../../components/ui/textarea';
import { Label } from '../../../components/ui/label';

interface FirewallRule {
  id: string;
  name: string;
  description: string;
  ruleType: 'NAT' | 'Network' | 'Application' | 'DNAT';
  action: 'allow' | 'deny' | 'drop';
  priority: number;
  status: 'enabled' | 'disabled' | 'pending' | 'error';
  sourceType: 'IP' | 'FQDN' | 'Service Tag' | 'IP Group';
  destinationType: 'IP' | 'FQDN' | 'Service Tag' | 'IP Group';
  source: string[];
  destination: string[];
  sourcePort: string;
  destinationPort: string;
  protocol: 'TCP' | 'UDP' | 'ICMP' | 'Any';
  firewallPolicy: string;
  ruleCollection: string;
  createdDate: string;
  lastModified: string;
  lastHit?: string;
  hitCount: number;
  threatIntelligence: boolean;
  logging: boolean;
  tags: { [key: string]: string };
}

interface FirewallMetrics {
  totalRules: number;
  enabledRules: number;
  blockedConnections: number;
  allowedConnections: number;
  ruleHitRate: number;
  threatIntelBlocks: number;
  topBlockedPorts: { port: string; count: number }[];
  topThreatCategories: { category: string; count: number }[];
}

export default function FirewallRulesPage() {
  const [firewallRules, setFirewallRules] = useState<FirewallRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [actionFilter, setActionFilter] = useState<string>('all');
  const [selectedRules, setSelectedRules] = useState<string[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedRule, setSelectedRule] = useState<FirewallRule | null>(null);

  const [metrics, setMetrics] = useState<FirewallMetrics>({
    totalRules: 0,
    enabledRules: 0,
    blockedConnections: 0,
    allowedConnections: 0,
    ruleHitRate: 0,
    threatIntelBlocks: 0,
    topBlockedPorts: [],
    topThreatCategories: []
  });

  // Mock data generation
  useEffect(() => {
    const generateMockFirewallRules = (): FirewallRule[] => {
      const ruleTypes: FirewallRule['ruleType'][] = ['NAT', 'Network', 'Application', 'DNAT'];
      const actions: FirewallRule['action'][] = ['allow', 'deny', 'drop'];
      const statuses: FirewallRule['status'][] = ['enabled', 'disabled', 'pending', 'error'];
      const protocols: FirewallRule['protocol'][] = ['TCP', 'UDP', 'ICMP', 'Any'];
      const sourceTypes: FirewallRule['sourceType'][] = ['IP', 'FQDN', 'Service Tag', 'IP Group'];
      const destinationTypes: FirewallRule['destinationType'][] = ['IP', 'FQDN', 'Service Tag', 'IP Group'];
      const firewallPolicies = ['policy-prod-01', 'policy-staging-02', 'policy-dev-03'];
      const ruleCollections = ['app-rules', 'network-rules', 'nat-rules', 'threat-rules'];

      const commonPorts = ['80', '443', '22', '3389', '53', '25', '110', '143', '993', '995'];
      const commonSources = ['10.0.0.0/8', '192.168.0.0/16', '172.16.0.0/12', '0.0.0.0/0'];
      const commonDestinations = ['10.0.1.0/24', '10.0.2.0/24', '192.168.1.0/24', 'www.example.com'];

      return Array.from({ length: 60 }, (_, i) => ({
        id: `fw-rule-${i + 1}`,
        name: `firewall-rule-${i + 1}`,
        description: `Firewall rule for ${['web traffic', 'API access', 'database connections', 'management', 'external services'][Math.floor(Math.random() * 5)]}`,
        ruleType: ruleTypes[Math.floor(Math.random() * ruleTypes.length)],
        action: actions[Math.floor(Math.random() * actions.length)],
        priority: 100 + (i * 10),
        status: statuses[Math.floor(Math.random() * statuses.length)],
        sourceType: sourceTypes[Math.floor(Math.random() * sourceTypes.length)],
        destinationType: destinationTypes[Math.floor(Math.random() * destinationTypes.length)],
        source: [commonSources[Math.floor(Math.random() * commonSources.length)]],
        destination: [commonDestinations[Math.floor(Math.random() * commonDestinations.length)]],
        sourcePort: Math.random() > 0.5 ? '*' : commonPorts[Math.floor(Math.random() * commonPorts.length)],
        destinationPort: commonPorts[Math.floor(Math.random() * commonPorts.length)],
        protocol: protocols[Math.floor(Math.random() * protocols.length)],
        firewallPolicy: firewallPolicies[Math.floor(Math.random() * firewallPolicies.length)],
        ruleCollection: ruleCollections[Math.floor(Math.random() * ruleCollections.length)],
        createdDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        lastModified: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        lastHit: Math.random() > 0.3 ? new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString() : undefined,
        hitCount: Math.floor(Math.random() * 100000),
        threatIntelligence: Math.random() > 0.7,
        logging: Math.random() > 0.2,
        tags: {
          environment: ['prod', 'staging', 'dev'][Math.floor(Math.random() * 3)],
          team: ['Security', 'DevOps', 'Infrastructure'][Math.floor(Math.random() * 3)],
          criticality: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)]
        }
      }));
    };

    setTimeout(() => {
      const mockRules = generateMockFirewallRules();
      setFirewallRules(mockRules);

      // Calculate metrics
      const enabledRules = mockRules.filter(r => r.status === 'enabled').length;
      const allowedConnections = mockRules.filter(r => r.action === 'allow').reduce((sum, r) => sum + r.hitCount, 0);
      const blockedConnections = mockRules.filter(r => r.action !== 'allow').reduce((sum, r) => sum + r.hitCount, 0);
      const threatIntelBlocks = mockRules.filter(r => r.threatIntelligence).reduce((sum, r) => sum + r.hitCount, 0);
      const rulesWithHits = mockRules.filter(r => r.hitCount > 0).length;
      const ruleHitRate = (rulesWithHits / mockRules.length) * 100;

      // Generate top blocked ports
      const portCounts: { [key: string]: number } = {};
      mockRules.filter(r => r.action !== 'allow').forEach(r => {
        portCounts[r.destinationPort] = (portCounts[r.destinationPort] || 0) + r.hitCount;
      });
      const topBlockedPorts = Object.entries(portCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([port, count]) => ({ port, count }));

      // Generate threat categories
      const threatCategories = [
        { category: 'Malware', count: Math.floor(Math.random() * 1000) + 500 },
        { category: 'Botnet', count: Math.floor(Math.random() * 800) + 300 },
        { category: 'Phishing', count: Math.floor(Math.random() * 600) + 200 },
        { category: 'C&C', count: Math.floor(Math.random() * 400) + 100 },
        { category: 'Suspicious', count: Math.floor(Math.random() * 300) + 50 }
      ];

      setMetrics({
        totalRules: mockRules.length,
        enabledRules,
        blockedConnections,
        allowedConnections,
        ruleHitRate: Math.round(ruleHitRate),
        threatIntelBlocks,
        topBlockedPorts,
        topThreatCategories: threatCategories
      });

      setLoading(false);
    }, 1000);
  }, []);

  const filteredRules = firewallRules.filter(rule => {
    const matchesSearch = rule.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         rule.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         rule.source.some(s => s.toLowerCase().includes(searchTerm.toLowerCase())) ||
                         rule.destination.some(d => d.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesStatus = statusFilter === 'all' || rule.status === statusFilter;
    const matchesType = typeFilter === 'all' || rule.ruleType === typeFilter;
    const matchesAction = actionFilter === 'all' || rule.action === actionFilter;
    return matchesSearch && matchesStatus && matchesType && matchesAction;
  });

  const getStatusColor = (status: FirewallRule['status']) => {
    switch (status) {
      case 'enabled': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'disabled': return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
      case 'pending': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'error': return 'bg-red-500/10 text-red-400 border-red-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getActionColor = (action: FirewallRule['action']) => {
    switch (action) {
      case 'allow': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'deny': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'drop': return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const toggleRuleSelection = (ruleId: string) => {
    setSelectedRules(prev => 
      prev.includes(ruleId) 
        ? prev.filter(id => id !== ruleId)
        : [...prev, ruleId]
    );
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white p-6">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-800 rounded w-1/4 mb-6"></div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-32 bg-gray-800 rounded"></div>
              ))}
            </div>
            <div className="h-96 bg-gray-800 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-red-500/10 rounded-lg">
              <Shield className="h-8 w-8 text-red-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Firewall Rules</h1>
              <p className="text-gray-400">Manage firewall policies and network security rules</p>
            </div>
          </div>
          <Button 
            onClick={() => setShowCreateDialog(true)}
            className="bg-red-600 hover:bg-red-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Rule
          </Button>
        </motion.div>

        {/* Metrics Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Rules</p>
                  <p className="text-2xl font-bold text-white">{metrics.totalRules}</p>
                </div>
                <Shield className="h-8 w-8 text-red-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Activity className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">{metrics.enabledRules} Enabled</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Rule Hit Rate</p>
                  <p className="text-2xl font-bold text-white">{metrics.ruleHitRate}%</p>
                </div>
                <Target className="h-8 w-8 text-blue-400" />
              </div>
              <div className="mt-2">
                <Progress value={metrics.ruleHitRate} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Blocked Connections</p>
                  <p className="text-2xl font-bold text-red-400">{metrics.blockedConnections.toLocaleString()}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Shield className="h-4 w-4 text-red-400" />
                <span className="text-sm text-red-400">Security Events</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Threat Intel Blocks</p>
                  <p className="text-2xl font-bold text-orange-400">{metrics.threatIntelBlocks.toLocaleString()}</p>
                </div>
                <Zap className="h-8 w-8 text-orange-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-orange-400" />
                <span className="text-sm text-orange-400">AI Protection</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Filters and Search */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex flex-col sm:flex-row gap-4 items-center justify-between"
        >
          <div className="flex gap-4 items-center flex-1">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <Input
                placeholder="Search firewall rules..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 bg-gray-900 border-gray-700"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="enabled">Enabled</SelectItem>
                <SelectItem value="disabled">Disabled</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="NAT">NAT</SelectItem>
                <SelectItem value="Network">Network</SelectItem>
                <SelectItem value="Application">Application</SelectItem>
                <SelectItem value="DNAT">DNAT</SelectItem>
              </SelectContent>
            </Select>
            <Select value={actionFilter} onValueChange={setActionFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Action" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Actions</SelectItem>
                <SelectItem value="allow">Allow</SelectItem>
                <SelectItem value="deny">Deny</SelectItem>
                <SelectItem value="drop">Drop</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" className="border-gray-700">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="outline" className="border-gray-700">
              <Upload className="h-4 w-4 mr-2" />
              Import
            </Button>
          </div>
        </motion.div>

        {/* Bulk Actions */}
        <AnimatePresence>
          {selectedRules.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-red-600/10 border border-red-600/20 rounded-lg p-4"
            >
              <div className="flex items-center justify-between">
                <span className="text-red-400">
                  {selectedRules.length} rule{selectedRules.length > 1 ? 's' : ''} selected
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="border-red-600 text-red-400">
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Selected
                  </Button>
                  <Button size="sm" variant="outline" className="border-orange-600 text-orange-400">
                    <Lock className="h-4 w-4 mr-2" />
                    Disable Selected
                  </Button>
                  <Button size="sm" variant="outline" className="border-gray-600 text-gray-400">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete Selected
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Firewall Rules Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900/50 rounded-lg border border-gray-800 overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800/50">
                <tr>
                  <th className="text-left p-4 font-medium text-gray-300">
                    <input
                      type="checkbox"
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedRules(filteredRules.map(r => r.id));
                        } else {
                          setSelectedRules([]);
                        }
                      }}
                      checked={selectedRules.length === filteredRules.length}
                      className="rounded"
                    />
                  </th>
                  <th className="text-left p-4 font-medium text-gray-300">Rule Name</th>
                  <th className="text-left p-4 font-medium text-gray-300">Type</th>
                  <th className="text-left p-4 font-medium text-gray-300">Action</th>
                  <th className="text-left p-4 font-medium text-gray-300">Status</th>
                  <th className="text-left p-4 font-medium text-gray-300">Source → Destination</th>
                  <th className="text-left p-4 font-medium text-gray-300">Protocol/Port</th>
                  <th className="text-left p-4 font-medium text-gray-300">Hit Count</th>
                  <th className="text-left p-4 font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredRules.map((rule, index) => (
                    <motion.tr
                      key={rule.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-t border-gray-800 hover:bg-gray-800/30"
                    >
                      <td className="p-4">
                        <input
                          type="checkbox"
                          checked={selectedRules.includes(rule.id)}
                          onChange={() => toggleRuleSelection(rule.id)}
                          className="rounded"
                        />
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-red-500/10 rounded">
                            <Shield className="h-4 w-4 text-red-400" />
                          </div>
                          <div>
                            <div className="font-medium text-white">{rule.name}</div>
                            <div className="text-sm text-gray-400">Priority: {rule.priority}</div>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className="border-red-500/20 text-red-400">
                          {rule.ruleType}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <Badge className={getActionColor(rule.action)}>
                          {rule.action}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <Badge className={getStatusColor(rule.status)}>
                          {rule.status}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <div className="text-gray-300 font-mono text-sm">
                          <div>{rule.source[0]} → {rule.destination[0]}</div>
                          {rule.threatIntelligence && (
                            <div className="text-xs text-orange-400 flex items-center mt-1">
                              <Zap className="h-3 w-3 mr-1" />
                              Threat Intel
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="text-gray-300 font-mono text-sm">
                          <div>{rule.protocol}</div>
                          <div className="text-xs text-gray-500">
                            {rule.sourcePort !== '*' && `${rule.sourcePort} → `}{rule.destinationPort}
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="text-gray-300">
                          <div className="font-medium">{rule.hitCount.toLocaleString()}</div>
                          <div className="text-xs text-gray-500">
                            {rule.lastHit ? `Last: ${new Date(rule.lastHit).toLocaleDateString()}` : 'No hits'}
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setSelectedRule(rule)}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyToClipboard(rule.id)}
                            className="h-8 w-8 p-0"
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0"
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0 text-red-400 hover:text-red-300"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Rule Details Dialog */}
        <Dialog open={selectedRule !== null} onOpenChange={() => setSelectedRule(null)}>
          <DialogContent className="bg-gray-900 border-gray-800 max-w-4xl">
            <DialogHeader>
              <DialogTitle className="text-white flex items-center space-x-2">
                <Shield className="h-5 w-5 text-red-400" />
                <span>Firewall Rule: {selectedRule?.name}</span>
              </DialogTitle>
            </DialogHeader>
            
            {selectedRule && (
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4 bg-gray-800">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="traffic">Traffic Flow</TabsTrigger>
                  <TabsTrigger value="analytics">Analytics</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Rule ID</Label>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{selectedRule.id}</span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => copyToClipboard(selectedRule.id)}
                          className="h-6 w-6 p-0"
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Type</Label>
                      <Badge variant="outline" className="border-red-500/20 text-red-400">
                        {selectedRule.ruleType}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Action</Label>
                      <Badge className={getActionColor(selectedRule.action)}>
                        {selectedRule.action}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Status</Label>
                      <Badge className={getStatusColor(selectedRule.status)}>
                        {selectedRule.status}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Priority</Label>
                      <span className="text-white">{selectedRule.priority}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Protocol</Label>
                      <span className="text-white">{selectedRule.protocol}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Firewall Policy</Label>
                      <span className="text-white">{selectedRule.firewallPolicy}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Rule Collection</Label>
                      <span className="text-white">{selectedRule.ruleCollection}</span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Description</Label>
                    <p className="text-white">{selectedRule.description}</p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Created</Label>
                      <span className="text-white">{new Date(selectedRule.createdDate).toLocaleString()}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Last Modified</Label>
                      <span className="text-white">{new Date(selectedRule.lastModified).toLocaleString()}</span>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="traffic" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Source</Label>
                      <div className="p-3 bg-gray-800/50 rounded border border-gray-700">
                        <div className="text-sm text-gray-400 mb-1">Type: {selectedRule.sourceType}</div>
                        <div className="space-y-1">
                          {selectedRule.source.map((src, index) => (
                            <div key={index} className="font-mono text-sm text-white">{src}</div>
                          ))}
                        </div>
                        <div className="text-sm text-gray-400 mt-2">Port: {selectedRule.sourcePort}</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Destination</Label>
                      <div className="p-3 bg-gray-800/50 rounded border border-gray-700">
                        <div className="text-sm text-gray-400 mb-1">Type: {selectedRule.destinationType}</div>
                        <div className="space-y-1">
                          {selectedRule.destination.map((dest, index) => (
                            <div key={index} className="font-mono text-sm text-white">{dest}</div>
                          ))}
                        </div>
                        <div className="text-sm text-gray-400 mt-2">Port: {selectedRule.destinationPort}</div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="analytics" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Hit Count</Label>
                      <span className="text-2xl font-bold text-white">{selectedRule.hitCount.toLocaleString()}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Last Hit</Label>
                      <span className="text-white">
                        {selectedRule.lastHit ? new Date(selectedRule.lastHit).toLocaleString() : 'Never'}
                      </span>
                    </div>
                  </div>
                  <div className="text-center py-8">
                    <Activity className="h-12 w-12 text-gray-500 mx-auto mb-2" />
                    <p className="text-gray-400">Detailed analytics would be displayed here</p>
                    <p className="text-sm text-gray-500">Traffic patterns, trending data, and performance metrics</p>
                  </div>
                </TabsContent>
                
                <TabsContent value="settings" className="space-y-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-gray-400">Threat Intelligence</Label>
                        <p className="text-sm text-gray-500">Enable AI-powered threat detection</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        {selectedRule.threatIntelligence ? (
                          <>
                            <CheckCircle className="h-4 w-4 text-green-400" />
                            <span className="text-green-400">Enabled</span>
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="h-4 w-4 text-yellow-400" />
                            <span className="text-yellow-400">Disabled</span>
                          </>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-gray-400">Logging</Label>
                        <p className="text-sm text-gray-500">Log rule matches for auditing</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        {selectedRule.logging ? (
                          <>
                            <CheckCircle className="h-4 w-4 text-green-400" />
                            <span className="text-green-400">Enabled</span>
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="h-4 w-4 text-yellow-400" />
                            <span className="text-yellow-400">Disabled</span>
                          </>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex space-x-2 pt-4">
                      <Button className="bg-red-600 hover:bg-red-700">
                        <Edit className="h-4 w-4 mr-2" />
                        Edit Rule
                      </Button>
                      <Button variant="outline" className="border-orange-600 text-orange-400">
                        <Lock className="h-4 w-4 mr-2" />
                        Disable Rule
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>

        {/* Create Rule Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent className="bg-gray-900 border-gray-800">
            <DialogHeader>
              <DialogTitle className="text-white">Create Firewall Rule</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Rule Name</Label>
                <Input placeholder="Enter rule name" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="space-y-2">
                <Label>Description</Label>
                <Textarea placeholder="Describe the purpose of this rule" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Rule Type</Label>
                  <Select>
                    <SelectTrigger className="bg-gray-800 border-gray-700">
                      <SelectValue placeholder="Select rule type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="network">Network</SelectItem>
                      <SelectItem value="application">Application</SelectItem>
                      <SelectItem value="nat">NAT</SelectItem>
                      <SelectItem value="dnat">DNAT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Action</Label>
                  <Select>
                    <SelectTrigger className="bg-gray-800 border-gray-700">
                      <SelectValue placeholder="Select action" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="allow">Allow</SelectItem>
                      <SelectItem value="deny">Deny</SelectItem>
                      <SelectItem value="drop">Drop</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Source</Label>
                  <Input placeholder="10.0.0.0/8 or example.com" className="bg-gray-800 border-gray-700" />
                </div>
                
                <div className="space-y-2">
                  <Label>Destination</Label>
                  <Input placeholder="10.0.1.0/24 or api.example.com" className="bg-gray-800 border-gray-700" />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Protocol</Label>
                  <Select>
                    <SelectTrigger className="bg-gray-800 border-gray-700">
                      <SelectValue placeholder="Select protocol" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="tcp">TCP</SelectItem>
                      <SelectItem value="udp">UDP</SelectItem>
                      <SelectItem value="icmp">ICMP</SelectItem>
                      <SelectItem value="any">Any</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Destination Port</Label>
                  <Input placeholder="80, 443, 22 or *" className="bg-gray-800 border-gray-700" />
                </div>
              </div>
              
              <div className="flex space-x-2 pt-4">
                <Button className="bg-red-600 hover:bg-red-700 flex-1">
                  Create Rule
                </Button>
                <Button variant="outline" onClick={() => setShowCreateDialog(false)} className="border-gray-700">
                  Cancel
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}


