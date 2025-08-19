'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, Shield, AlertTriangle, CheckCircle, Clock, 
  Plus, Search, Filter, Download, Upload, RotateCcw,
  Eye, EyeOff, Copy, Trash2, Edit, Settings, 
  Calendar, Map, Activity, Lock, Unlock,
  FileText, Globe, Zap, Network, Server
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

interface SecurityGroup {
  id: string;
  name: string;
  description: string;
  type: 'NSG' | 'ASG' | 'WAF' | 'Custom';
  status: 'active' | 'inactive' | 'updating' | 'error';
  location: string;
  resourceGroup: string;
  subscription: string;
  vnet: string;
  subnet: string;
  associatedResources: number;
  inboundRules: SecurityRule[];
  outboundRules: SecurityRule[];
  createdDate: string;
  lastModified: string;
  compliance: number;
  riskScore: number;
  tags: { [key: string]: string };
}

interface SecurityRule {
  id: string;
  name: string;
  priority: number;
  direction: 'inbound' | 'outbound';
  access: 'allow' | 'deny';
  protocol: 'TCP' | 'UDP' | 'ICMP' | 'Any';
  sourcePortRange: string;
  destinationPortRange: string;
  sourceAddressPrefix: string;
  destinationAddressPrefix: string;
  description: string;
  lastUsed?: string;
  hitCount: number;
}

interface SecurityGroupMetrics {
  totalGroups: number;
  activeGroups: number;
  highRiskGroups: number;
  complianceViolations: number;
  averageCompliance: number;
  rulesCount: number;
  associatedResources: number;
  recentChanges: number;
}

export default function SecurityGroupsPage() {
  const [securityGroups, setSecurityGroups] = useState<SecurityGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<SecurityGroup | null>(null);

  const [metrics, setMetrics] = useState<SecurityGroupMetrics>({
    totalGroups: 0,
    activeGroups: 0,
    highRiskGroups: 0,
    complianceViolations: 0,
    averageCompliance: 0,
    rulesCount: 0,
    associatedResources: 0,
    recentChanges: 0
  });

  // Mock data generation
  useEffect(() => {
    const generateMockSecurityGroups = (): SecurityGroup[] => {
      const groupTypes: SecurityGroup['type'][] = ['NSG', 'ASG', 'WAF', 'Custom'];
      const statuses: SecurityGroup['status'][] = ['active', 'inactive', 'updating', 'error'];
      const locations = ['East US 2', 'West Europe', 'Asia Pacific', 'Canada Central'];
      const vnets = ['vnet-prod-01', 'vnet-staging-02', 'vnet-dev-03', 'vnet-mgmt-04'];
      const subnets = ['subnet-web', 'subnet-api', 'subnet-db', 'subnet-mgmt'];

      const generateMockRules = (direction: 'inbound' | 'outbound', count: number): SecurityRule[] => {
        return Array.from({ length: count }, (_, i) => ({
          id: `rule-${direction}-${i + 1}`,
          name: `${direction}-rule-${i + 1}`,
          priority: 100 + (i * 10),
          direction,
          access: Math.random() > 0.8 ? 'deny' : 'allow',
          protocol: ['TCP', 'UDP', 'ICMP', 'Any'][Math.floor(Math.random() * 4)] as any,
          sourcePortRange: direction === 'inbound' ? '*' : '80,443,8080',
          destinationPortRange: direction === 'outbound' ? '*' : '80,443,22,3389',
          sourceAddressPrefix: direction === 'inbound' ? '0.0.0.0/0' : '10.0.0.0/16',
          destinationAddressPrefix: direction === 'outbound' ? '0.0.0.0/0' : '10.0.0.0/16',
          description: `${direction} rule for ${['web', 'api', 'database', 'management'][Math.floor(Math.random() * 4)]} traffic`,
          lastUsed: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
          hitCount: Math.floor(Math.random() * 10000)
        }));
      };

      return Array.from({ length: 40 }, (_, i) => ({
        id: `sg-${i + 1}`,
        name: `security-group-${i + 1}`,
        description: `Security group for ${['web tier', 'application tier', 'database tier', 'management'][Math.floor(Math.random() * 4)]}`,
        type: groupTypes[Math.floor(Math.random() * groupTypes.length)],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        location: locations[Math.floor(Math.random() * locations.length)],
        resourceGroup: `rg-security-${Math.floor(Math.random() * 5) + 1}`,
        subscription: 'PolicyCortex-Prod',
        vnet: vnets[Math.floor(Math.random() * vnets.length)],
        subnet: subnets[Math.floor(Math.random() * subnets.length)],
        associatedResources: Math.floor(Math.random() * 50) + 1,
        inboundRules: generateMockRules('inbound', Math.floor(Math.random() * 8) + 2),
        outboundRules: generateMockRules('outbound', Math.floor(Math.random() * 6) + 1),
        createdDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        lastModified: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        compliance: Math.floor(Math.random() * 40) + 60,
        riskScore: Math.floor(Math.random() * 100),
        tags: {
          environment: ['prod', 'staging', 'dev'][Math.floor(Math.random() * 3)],
          team: ['Security', 'DevOps', 'Infrastructure'][Math.floor(Math.random() * 3)],
          criticality: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)]
        }
      }));
    };

    setTimeout(() => {
      const mockGroups = generateMockSecurityGroups();
      setSecurityGroups(mockGroups);

      // Calculate metrics
      const activeGroups = mockGroups.filter(g => g.status === 'active').length;
      const highRiskGroups = mockGroups.filter(g => g.riskScore > 70).length;
      const complianceViolations = mockGroups.filter(g => g.compliance < 80).length;
      const avgCompliance = mockGroups.reduce((sum, g) => sum + g.compliance, 0) / mockGroups.length;
      const totalRules = mockGroups.reduce((sum, g) => sum + g.inboundRules.length + g.outboundRules.length, 0);
      const totalResources = mockGroups.reduce((sum, g) => sum + g.associatedResources, 0);
      const recentChanges = mockGroups.filter(g => 
        new Date(g.lastModified).getTime() > Date.now() - 7 * 24 * 60 * 60 * 1000
      ).length;

      setMetrics({
        totalGroups: mockGroups.length,
        activeGroups,
        highRiskGroups,
        complianceViolations,
        averageCompliance: Math.round(avgCompliance),
        rulesCount: totalRules,
        associatedResources: totalResources,
        recentChanges
      });

      setLoading(false);
    }, 1000);
  }, []);

  const filteredGroups = securityGroups.filter(group => {
    const matchesSearch = group.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         group.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         group.vnet.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || group.status === statusFilter;
    const matchesType = typeFilter === 'all' || group.type === typeFilter;
    return matchesSearch && matchesStatus && matchesType;
  });

  const getStatusColor = (status: SecurityGroup['status']) => {
    switch (status) {
      case 'active': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'inactive': return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
      case 'updating': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'error': return 'bg-red-500/10 text-red-400 border-red-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore > 70) return 'text-red-400';
    if (riskScore > 40) return 'text-yellow-400';
    return 'text-green-400';
  };

  const toggleGroupSelection = (groupId: string) => {
    setSelectedGroups(prev => 
      prev.includes(groupId) 
        ? prev.filter(id => id !== groupId)
        : [...prev, groupId]
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
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <Users className="h-8 w-8 text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Security Groups</h1>
              <p className="text-gray-400">Manage network security groups and access control</p>
            </div>
          </div>
          <Button 
            onClick={() => setShowCreateDialog(true)}
            className="bg-purple-600 hover:bg-purple-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Group
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
                  <p className="text-gray-400 text-sm">Total Groups</p>
                  <p className="text-2xl font-bold text-white">{metrics.totalGroups}</p>
                </div>
                <Users className="h-8 w-8 text-purple-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Activity className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">{metrics.activeGroups} Active</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Compliance Score</p>
                  <p className="text-2xl font-bold text-white">{metrics.averageCompliance}%</p>
                </div>
                <Shield className="h-8 w-8 text-blue-400" />
              </div>
              <div className="mt-2">
                <Progress value={metrics.averageCompliance} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">High Risk Groups</p>
                  <p className="text-2xl font-bold text-red-400">{metrics.highRiskGroups}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-red-400" />
                <span className="text-sm text-red-400">Requires Review</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Security Rules</p>
                  <p className="text-2xl font-bold text-white">{metrics.rulesCount}</p>
                </div>
                <Lock className="h-8 w-8 text-orange-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Server className="h-4 w-4 text-orange-400" />
                <span className="text-sm text-orange-400">{metrics.associatedResources} Resources</span>
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
                placeholder="Search security groups..."
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
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
                <SelectItem value="updating">Updating</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="NSG">Network Security Group</SelectItem>
                <SelectItem value="ASG">Application Security Group</SelectItem>
                <SelectItem value="WAF">Web Application Firewall</SelectItem>
                <SelectItem value="Custom">Custom</SelectItem>
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
          {selectedGroups.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-purple-600/10 border border-purple-600/20 rounded-lg p-4"
            >
              <div className="flex items-center justify-between">
                <span className="text-purple-400">
                  {selectedGroups.length} group{selectedGroups.length > 1 ? 's' : ''} selected
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="border-purple-600 text-purple-400">
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Selected
                  </Button>
                  <Button size="sm" variant="outline" className="border-red-600 text-red-400">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete Selected
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Security Groups Table */}
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
                          setSelectedGroups(filteredGroups.map(g => g.id));
                        } else {
                          setSelectedGroups([]);
                        }
                      }}
                      checked={selectedGroups.length === filteredGroups.length}
                      className="rounded"
                    />
                  </th>
                  <th className="text-left p-4 font-medium text-gray-300">Group Name</th>
                  <th className="text-left p-4 font-medium text-gray-300">Type</th>
                  <th className="text-left p-4 font-medium text-gray-300">Status</th>
                  <th className="text-left p-4 font-medium text-gray-300">VNet/Subnet</th>
                  <th className="text-left p-4 font-medium text-gray-300">Rules</th>
                  <th className="text-left p-4 font-medium text-gray-300">Resources</th>
                  <th className="text-left p-4 font-medium text-gray-300">Risk Score</th>
                  <th className="text-left p-4 font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredGroups.map((group, index) => (
                    <motion.tr
                      key={group.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-t border-gray-800 hover:bg-gray-800/30"
                    >
                      <td className="p-4">
                        <input
                          type="checkbox"
                          checked={selectedGroups.includes(group.id)}
                          onChange={() => toggleGroupSelection(group.id)}
                          className="rounded"
                        />
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-purple-500/10 rounded">
                            <Users className="h-4 w-4 text-purple-400" />
                          </div>
                          <div>
                            <div className="font-medium text-white">{group.name}</div>
                            <div className="text-sm text-gray-400">{group.description}</div>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className="border-purple-500/20 text-purple-400">
                          {group.type}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <Badge className={getStatusColor(group.status)}>
                          {group.status}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <div className="text-gray-300">
                          <div>{group.vnet}</div>
                          <div className="text-sm text-gray-500">{group.subnet}</div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="text-gray-300">
                          <div className="flex items-center space-x-2">
                            <span className="text-green-400">↓{group.inboundRules.length}</span>
                            <span className="text-blue-400">↑{group.outboundRules.length}</span>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          <Server className="h-4 w-4 text-orange-400" />
                          <span className="text-white">{group.associatedResources}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className={`text-sm font-medium ${getRiskColor(group.riskScore)}`}>
                          {group.riskScore}/100
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setSelectedGroup(group)}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyToClipboard(group.id)}
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

        {/* Security Group Details Dialog */}
        <Dialog open={selectedGroup !== null} onOpenChange={() => setSelectedGroup(null)}>
          <DialogContent className="bg-gray-900 border-gray-800 max-w-4xl">
            <DialogHeader>
              <DialogTitle className="text-white flex items-center space-x-2">
                <Users className="h-5 w-5 text-purple-400" />
                <span>Security Group: {selectedGroup?.name}</span>
              </DialogTitle>
            </DialogHeader>
            
            {selectedGroup && (
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4 bg-gray-800">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="inbound">Inbound Rules</TabsTrigger>
                  <TabsTrigger value="outbound">Outbound Rules</TabsTrigger>
                  <TabsTrigger value="resources">Resources</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Group ID</Label>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{selectedGroup.id}</span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => copyToClipboard(selectedGroup.id)}
                          className="h-6 w-6 p-0"
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Type</Label>
                      <Badge variant="outline" className="border-purple-500/20 text-purple-400">
                        {selectedGroup.type}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Status</Label>
                      <Badge className={getStatusColor(selectedGroup.status)}>
                        {selectedGroup.status}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Location</Label>
                      <span className="text-white">{selectedGroup.location}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Virtual Network</Label>
                      <span className="text-white">{selectedGroup.vnet}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Subnet</Label>
                      <span className="text-white">{selectedGroup.subnet}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Risk Score</Label>
                      <span className={getRiskColor(selectedGroup.riskScore)}>
                        {selectedGroup.riskScore}/100
                      </span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Compliance</Label>
                      <span className="text-white">{selectedGroup.compliance}%</span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Description</Label>
                    <p className="text-white">{selectedGroup.description}</p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Tags</Label>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedGroup.tags).map(([key, value]) => (
                        <Badge key={key} variant="outline" className="border-gray-700">
                          {key}: {value}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="inbound" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Inbound Security Rules</Label>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {selectedGroup.inboundRules.map((rule) => (
                        <div key={rule.id} className="p-3 bg-gray-800/50 rounded border border-gray-700">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <Badge className={rule.access === 'allow' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}>
                                {rule.access}
                              </Badge>
                              <span className="font-medium">{rule.name}</span>
                            </div>
                            <span className="text-sm text-gray-400">Priority: {rule.priority}</span>
                          </div>
                          <div className="mt-2 text-sm text-gray-300">
                            <div>Protocol: {rule.protocol}</div>
                            <div>Source: {rule.sourceAddressPrefix}:{rule.sourcePortRange}</div>
                            <div>Destination: {rule.destinationAddressPrefix}:{rule.destinationPortRange}</div>
                            <div className="text-gray-500">Hits: {rule.hitCount.toLocaleString()}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="outbound" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Outbound Security Rules</Label>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {selectedGroup.outboundRules.map((rule) => (
                        <div key={rule.id} className="p-3 bg-gray-800/50 rounded border border-gray-700">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <Badge className={rule.access === 'allow' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}>
                                {rule.access}
                              </Badge>
                              <span className="font-medium">{rule.name}</span>
                            </div>
                            <span className="text-sm text-gray-400">Priority: {rule.priority}</span>
                          </div>
                          <div className="mt-2 text-sm text-gray-300">
                            <div>Protocol: {rule.protocol}</div>
                            <div>Source: {rule.sourceAddressPrefix}:{rule.sourcePortRange}</div>
                            <div>Destination: {rule.destinationAddressPrefix}:{rule.destinationPortRange}</div>
                            <div className="text-gray-500">Hits: {rule.hitCount.toLocaleString()}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="resources" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Associated Resources</Label>
                    <div className="text-center py-8">
                      <Server className="h-12 w-12 text-gray-500 mx-auto mb-2" />
                      <p className="text-gray-400">{selectedGroup.associatedResources} resources associated</p>
                      <p className="text-sm text-gray-500">Resources list would be displayed here</p>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>

        {/* Create Security Group Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent className="bg-gray-900 border-gray-800">
            <DialogHeader>
              <DialogTitle className="text-white">Create Security Group</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Group Name</Label>
                <Input placeholder="Enter group name" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="space-y-2">
                <Label>Description</Label>
                <Textarea placeholder="Describe the purpose of this security group" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="space-y-2">
                <Label>Type</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select group type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="nsg">Network Security Group</SelectItem>
                    <SelectItem value="asg">Application Security Group</SelectItem>
                    <SelectItem value="waf">Web Application Firewall</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label>Virtual Network</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select virtual network" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="vnet-prod-01">vnet-prod-01</SelectItem>
                    <SelectItem value="vnet-staging-02">vnet-staging-02</SelectItem>
                    <SelectItem value="vnet-dev-03">vnet-dev-03</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex space-x-2 pt-4">
                <Button className="bg-purple-600 hover:bg-purple-700 flex-1">
                  Create Security Group
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


