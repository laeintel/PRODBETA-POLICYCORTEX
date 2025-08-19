'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Key, Shield, AlertTriangle, CheckCircle, Clock, 
  Plus, Search, Filter, Download, Upload, RotateCcw,
  Eye, EyeOff, Copy, Trash2, Edit, Settings, 
  Calendar, Map, Activity, Users, Lock, Unlock
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

interface EncryptionKey {
  id: string;
  name: string;
  type: 'AES-256' | 'RSA-4096' | 'ECC-P384' | 'Kyber-1024';
  status: 'active' | 'expired' | 'expiring' | 'revoked' | 'pending';
  purpose: 'data-encryption' | 'key-encryption' | 'signing' | 'authentication' | 'tls';
  createdDate: string;
  expiryDate: string;
  lastUsed: string;
  usageCount: number;
  keyVault: string;
  resourceGroup: string;
  subscription: string;
  owner: string;
  compliance: number;
  location: string;
  size: number;
  algorithm: string;
  rotationStatus: 'current' | 'scheduled' | 'overdue';
  permissions: string[];
  tags: { [key: string]: string };
}

interface KeyMetrics {
  totalKeys: number;
  activeKeys: number;
  expiringKeys: number;
  revokedKeys: number;
  complianceScore: number;
  rotationCompliance: number;
  usageStats: {
    daily: number;
    weekly: number;
    monthly: number;
  };
}

export default function EncryptionKeysPage() {
  const [keys, setKeys] = useState<EncryptionKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedKey, setSelectedKey] = useState<EncryptionKey | null>(null);
  const [showKeyValue, setShowKeyValue] = useState<{[key: string]: boolean}>({});

  const [metrics, setMetrics] = useState<KeyMetrics>({
    totalKeys: 0,
    activeKeys: 0,
    expiringKeys: 0,
    revokedKeys: 0,
    complianceScore: 0,
    rotationCompliance: 0,
    usageStats: { daily: 0, weekly: 0, monthly: 0 }
  });

  // Mock data generation
  useEffect(() => {
    const generateMockKeys = (): EncryptionKey[] => {
      const keyTypes: EncryptionKey['type'][] = ['AES-256', 'RSA-4096', 'ECC-P384', 'Kyber-1024'];
      const statuses: EncryptionKey['status'][] = ['active', 'expired', 'expiring', 'revoked', 'pending'];
      const purposes: EncryptionKey['purpose'][] = ['data-encryption', 'key-encryption', 'signing', 'authentication', 'tls'];
      const keyVaults = ['prod-kv-east', 'dev-kv-west', 'staging-kv-central', 'backup-kv-south'];
      const owners = ['Security Team', 'DevOps Team', 'Data Team', 'Infrastructure Team'];
      const locations = ['East US 2', 'West Europe', 'Asia Pacific', 'Canada Central'];

      return Array.from({ length: 50 }, (_, i) => ({
        id: `key-${i + 1}`,
        name: `encryption-key-${i + 1}`,
        type: keyTypes[Math.floor(Math.random() * keyTypes.length)],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        purpose: purposes[Math.floor(Math.random() * purposes.length)],
        createdDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        expiryDate: new Date(Date.now() + Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        lastUsed: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        usageCount: Math.floor(Math.random() * 10000),
        keyVault: keyVaults[Math.floor(Math.random() * keyVaults.length)],
        resourceGroup: `rg-security-${Math.floor(Math.random() * 5) + 1}`,
        subscription: 'PolicyCortex-Prod',
        owner: owners[Math.floor(Math.random() * owners.length)],
        compliance: Math.floor(Math.random() * 40) + 60,
        location: locations[Math.floor(Math.random() * locations.length)],
        size: [2048, 3072, 4096][Math.floor(Math.random() * 3)],
        algorithm: keyTypes[Math.floor(Math.random() * keyTypes.length)],
        rotationStatus: ['current', 'scheduled', 'overdue'][Math.floor(Math.random() * 3)] as 'current' | 'scheduled' | 'overdue',
        permissions: ['read', 'encrypt', 'decrypt', 'sign', 'verify'].slice(0, Math.floor(Math.random() * 5) + 1),
        tags: {
          environment: ['prod', 'staging', 'dev'][Math.floor(Math.random() * 3)],
          team: owners[Math.floor(Math.random() * owners.length)].replace(' Team', ''),
          criticality: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)]
        }
      }));
    };

    setTimeout(() => {
      const mockKeys = generateMockKeys();
      setKeys(mockKeys);

      // Calculate metrics
      const activeKeys = mockKeys.filter(k => k.status === 'active').length;
      const expiringKeys = mockKeys.filter(k => k.status === 'expiring').length;
      const revokedKeys = mockKeys.filter(k => k.status === 'revoked').length;
      const avgCompliance = mockKeys.reduce((sum, k) => sum + k.compliance, 0) / mockKeys.length;
      const rotationCompliant = mockKeys.filter(k => k.rotationStatus === 'current').length;

      setMetrics({
        totalKeys: mockKeys.length,
        activeKeys,
        expiringKeys,
        revokedKeys,
        complianceScore: Math.round(avgCompliance),
        rotationCompliance: Math.round((rotationCompliant / mockKeys.length) * 100),
        usageStats: {
          daily: Math.floor(Math.random() * 1000) + 500,
          weekly: Math.floor(Math.random() * 5000) + 2000,
          monthly: Math.floor(Math.random() * 20000) + 10000
        }
      });

      setLoading(false);
    }, 1000);
  }, []);

  const filteredKeys = keys.filter(key => {
    const matchesSearch = key.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         key.keyVault.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         key.owner.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || key.status === statusFilter;
    const matchesType = typeFilter === 'all' || key.type === typeFilter;
    return matchesSearch && matchesStatus && matchesType;
  });

  const getStatusColor = (status: EncryptionKey['status']) => {
    switch (status) {
      case 'active': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'expired': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'expiring': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'revoked': return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
      case 'pending': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getRotationStatusColor = (status: string) => {
    switch (status) {
      case 'current': return 'text-green-400';
      case 'scheduled': return 'text-blue-400';
      case 'overdue': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const toggleKeySelection = (keyId: string) => {
    setSelectedKeys(prev => 
      prev.includes(keyId) 
        ? prev.filter(id => id !== keyId)
        : [...prev, keyId]
    );
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const toggleKeyVisibility = (keyId: string) => {
    setShowKeyValue(prev => ({ ...prev, [keyId]: !prev[keyId] }));
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
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Key className="h-8 w-8 text-blue-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Encryption Keys</h1>
              <p className="text-gray-400">Manage cryptographic keys and certificates</p>
            </div>
          </div>
          <Button 
            onClick={() => setShowCreateDialog(true)}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Key
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
                  <p className="text-gray-400 text-sm">Total Keys</p>
                  <p className="text-2xl font-bold text-white">{metrics.totalKeys}</p>
                </div>
                <Key className="h-8 w-8 text-blue-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Activity className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">Active Management</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Compliance Score</p>
                  <p className="text-2xl font-bold text-white">{metrics.complianceScore}%</p>
                </div>
                <Shield className="h-8 w-8 text-green-400" />
              </div>
              <div className="mt-2">
                <Progress value={metrics.complianceScore} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Expiring Soon</p>
                  <p className="text-2xl font-bold text-yellow-400">{metrics.expiringKeys}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-yellow-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Clock className="h-4 w-4 text-yellow-400" />
                <span className="text-sm text-yellow-400">Requires Attention</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Rotation Health</p>
                  <p className="text-2xl font-bold text-white">{metrics.rotationCompliance}%</p>
                </div>
                <RotateCcw className="h-8 w-8 text-blue-400" />
              </div>
              <div className="mt-2">
                <Progress value={metrics.rotationCompliance} className="h-2" />
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
                placeholder="Search encryption keys..."
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
                <SelectItem value="expired">Expired</SelectItem>
                <SelectItem value="expiring">Expiring</SelectItem>
                <SelectItem value="revoked">Revoked</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Key Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="AES-256">AES-256</SelectItem>
                <SelectItem value="RSA-4096">RSA-4096</SelectItem>
                <SelectItem value="ECC-P384">ECC-P384</SelectItem>
                <SelectItem value="Kyber-1024">Kyber-1024</SelectItem>
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
          {selectedKeys.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-blue-600/10 border border-blue-600/20 rounded-lg p-4"
            >
              <div className="flex items-center justify-between">
                <span className="text-blue-400">
                  {selectedKeys.length} key{selectedKeys.length > 1 ? 's' : ''} selected
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="border-blue-600 text-blue-400">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Rotate Selected
                  </Button>
                  <Button size="sm" variant="outline" className="border-red-600 text-red-400">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Revoke Selected
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Keys Table */}
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
                          setSelectedKeys(filteredKeys.map(k => k.id));
                        } else {
                          setSelectedKeys([]);
                        }
                      }}
                      checked={selectedKeys.length === filteredKeys.length}
                      className="rounded"
                    />
                  </th>
                  <th className="text-left p-4 font-medium text-gray-300">Key Name</th>
                  <th className="text-left p-4 font-medium text-gray-300">Type</th>
                  <th className="text-left p-4 font-medium text-gray-300">Status</th>
                  <th className="text-left p-4 font-medium text-gray-300">Purpose</th>
                  <th className="text-left p-4 font-medium text-gray-300">Key Vault</th>
                  <th className="text-left p-4 font-medium text-gray-300">Expiry</th>
                  <th className="text-left p-4 font-medium text-gray-300">Rotation</th>
                  <th className="text-left p-4 font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredKeys.map((key, index) => (
                    <motion.tr
                      key={key.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-t border-gray-800 hover:bg-gray-800/30"
                    >
                      <td className="p-4">
                        <input
                          type="checkbox"
                          checked={selectedKeys.includes(key.id)}
                          onChange={() => toggleKeySelection(key.id)}
                          className="rounded"
                        />
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-blue-500/10 rounded">
                            <Key className="h-4 w-4 text-blue-400" />
                          </div>
                          <div>
                            <div className="font-medium text-white">{key.name}</div>
                            <div className="text-sm text-gray-400">{key.id}</div>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className="border-blue-500/20 text-blue-400">
                          {key.type}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <Badge className={getStatusColor(key.status)}>
                          {key.status}
                        </Badge>
                      </td>
                      <td className="p-4 text-gray-300">{key.purpose}</td>
                      <td className="p-4 text-gray-300">{key.keyVault}</td>
                      <td className="p-4">
                        <div className="text-gray-300">
                          {new Date(key.expiryDate).toLocaleDateString()}
                        </div>
                        <div className="text-xs text-gray-500">
                          {Math.ceil((new Date(key.expiryDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24))} days
                        </div>
                      </td>
                      <td className="p-4">
                        <div className={`flex items-center space-x-1 ${getRotationStatusColor(key.rotationStatus)}`}>
                          <RotateCcw className="h-4 w-4" />
                          <span className="text-sm capitalize">{key.rotationStatus}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setSelectedKey(key)}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyToClipboard(key.id)}
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

        {/* Key Details Dialog */}
        <Dialog open={selectedKey !== null} onOpenChange={() => setSelectedKey(null)}>
          <DialogContent className="bg-gray-900 border-gray-800 max-w-2xl">
            <DialogHeader>
              <DialogTitle className="text-white flex items-center space-x-2">
                <Key className="h-5 w-5 text-blue-400" />
                <span>Key Details: {selectedKey?.name}</span>
              </DialogTitle>
            </DialogHeader>
            
            {selectedKey && (
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4 bg-gray-800">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="permissions">Permissions</TabsTrigger>
                  <TabsTrigger value="usage">Usage</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Key ID</Label>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{selectedKey.id}</span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => copyToClipboard(selectedKey.id)}
                          className="h-6 w-6 p-0"
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Type</Label>
                      <Badge variant="outline" className="border-blue-500/20 text-blue-400">
                        {selectedKey.type}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Status</Label>
                      <Badge className={getStatusColor(selectedKey.status)}>
                        {selectedKey.status}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Purpose</Label>
                      <span className="text-white">{selectedKey.purpose}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Key Vault</Label>
                      <span className="text-white">{selectedKey.keyVault}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Location</Label>
                      <span className="text-white">{selectedKey.location}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Created</Label>
                      <span className="text-white">{new Date(selectedKey.createdDate).toLocaleDateString()}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Expires</Label>
                      <span className="text-white">{new Date(selectedKey.expiryDate).toLocaleDateString()}</span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Tags</Label>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedKey.tags).map(([key, value]) => (
                        <Badge key={key} variant="outline" className="border-gray-700">
                          {key}: {value}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="permissions" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Granted Permissions</Label>
                    <div className="space-y-2">
                      {selectedKey.permissions.map((permission) => (
                        <div key={permission} className="flex items-center space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-400" />
                          <span className="text-white capitalize">{permission}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="usage" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Usage Count</Label>
                      <span className="text-2xl font-bold text-white">{selectedKey.usageCount.toLocaleString()}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Last Used</Label>
                      <span className="text-white">{new Date(selectedKey.lastUsed).toLocaleString()}</span>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="settings" className="space-y-4">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Auto Rotation</Label>
                      <div className="flex items-center space-x-2">
                        <span className={getRotationStatusColor(selectedKey.rotationStatus)}>
                          {selectedKey.rotationStatus}
                        </span>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <Button className="bg-blue-600 hover:bg-blue-700">
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Rotate Now
                      </Button>
                      <Button variant="outline" className="border-red-600 text-red-400">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Revoke Key
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>

        {/* Create Key Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent className="bg-gray-900 border-gray-800">
            <DialogHeader>
              <DialogTitle className="text-white">Create New Encryption Key</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Key Name</Label>
                <Input placeholder="Enter key name" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="space-y-2">
                <Label>Key Type</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select key type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="aes-256">AES-256</SelectItem>
                    <SelectItem value="rsa-4096">RSA-4096</SelectItem>
                    <SelectItem value="ecc-p384">ECC-P384</SelectItem>
                    <SelectItem value="kyber-1024">Kyber-1024 (Post-Quantum)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label>Purpose</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select purpose" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="data-encryption">Data Encryption</SelectItem>
                    <SelectItem value="key-encryption">Key Encryption</SelectItem>
                    <SelectItem value="signing">Digital Signing</SelectItem>
                    <SelectItem value="authentication">Authentication</SelectItem>
                    <SelectItem value="tls">TLS/SSL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label>Key Vault</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select key vault" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="prod-kv-east">prod-kv-east</SelectItem>
                    <SelectItem value="dev-kv-west">dev-kv-west</SelectItem>
                    <SelectItem value="staging-kv-central">staging-kv-central</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex space-x-2 pt-4">
                <Button className="bg-blue-600 hover:bg-blue-700 flex-1">
                  Create Key
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


