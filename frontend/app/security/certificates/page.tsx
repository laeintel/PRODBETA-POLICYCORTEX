'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Award, Shield, AlertTriangle, CheckCircle, Clock, 
  Plus, Search, Filter, Download, Upload, RotateCcw,
  Eye, EyeOff, Copy, Trash2, Edit, Settings, 
  Calendar, Map, Activity, Users, Lock, Unlock,
  FileText, Globe, Zap, RefreshCw
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

interface Certificate {
  id: string;
  name: string;
  commonName: string;
  type: 'SSL/TLS' | 'Code Signing' | 'Client Auth' | 'Email' | 'Root CA' | 'Intermediate CA';
  status: 'valid' | 'expired' | 'expiring' | 'revoked' | 'pending' | 'renewed';
  issuer: string;
  subject: string;
  serialNumber: string;
  thumbprint: string;
  algorithm: 'RSA-2048' | 'RSA-4096' | 'ECC-P256' | 'ECC-P384' | 'Ed25519';
  keyUsage: string[];
  issuedDate: string;
  expiryDate: string;
  daysUntilExpiry: number;
  autoRenewal: boolean;
  keyVault: string;
  resourceGroup: string;
  subscription: string;
  domain: string;
  subjectAlternativeNames: string[];
  certificateChain: string[];
  revocationReason?: string;
  lastValidation: string;
  validationStatus: 'valid' | 'warning' | 'error';
  tags: { [key: string]: string };
}

interface CertificateMetrics {
  totalCertificates: number;
  validCertificates: number;
  expiringCertificates: number;
  expiredCertificates: number;
  complianceScore: number;
  autoRenewalEnabled: number;
  validationErrors: number;
  averageDaysToExpiry: number;
}

export default function CertificatesPage() {
  const [certificates, setCertificates] = useState<Certificate[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [selectedCertificates, setSelectedCertificates] = useState<string[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedCertificate, setSelectedCertificate] = useState<Certificate | null>(null);

  const [metrics, setMetrics] = useState<CertificateMetrics>({
    totalCertificates: 0,
    validCertificates: 0,
    expiringCertificates: 0,
    expiredCertificates: 0,
    complianceScore: 0,
    autoRenewalEnabled: 0,
    validationErrors: 0,
    averageDaysToExpiry: 0
  });

  // Mock data generation
  useEffect(() => {
    const generateMockCertificates = (): Certificate[] => {
      const certTypes: Certificate['type'][] = ['SSL/TLS', 'Code Signing', 'Client Auth', 'Email', 'Root CA', 'Intermediate CA'];
      const statuses: Certificate['status'][] = ['valid', 'expired', 'expiring', 'revoked', 'pending', 'renewed'];
      const algorithms: Certificate['algorithm'][] = ['RSA-2048', 'RSA-4096', 'ECC-P256', 'ECC-P384', 'Ed25519'];
      const issuers = ['DigiCert', 'Let\'s Encrypt', 'GlobalSign', 'Sectigo', 'Internal CA', 'Entrust'];
      const domains = ['policycortex.com', 'api.policycortex.com', 'staging.policycortex.com', 'dev.policycortex.com'];
      const keyVaults = ['prod-kv-east', 'dev-kv-west', 'staging-kv-central', 'backup-kv-south'];

      return Array.from({ length: 50 }, (_, i) => {
        const expiryDate = new Date(Date.now() + (Math.random() * 365 - 30) * 24 * 60 * 60 * 1000);
        const daysUntilExpiry = Math.ceil((expiryDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
        
        return {
          id: `cert-${i + 1}`,
          name: `certificate-${i + 1}`,
          commonName: `${domains[Math.floor(Math.random() * domains.length)]}`,
          type: certTypes[Math.floor(Math.random() * certTypes.length)],
          status: daysUntilExpiry < 0 ? 'expired' : 
                 daysUntilExpiry < 30 ? 'expiring' : 
                 statuses[Math.floor(Math.random() * statuses.length)],
          issuer: issuers[Math.floor(Math.random() * issuers.length)],
          subject: `CN=${domains[Math.floor(Math.random() * domains.length)]}, O=PolicyCortex, C=US`,
          serialNumber: Math.random().toString(36).substring(2, 15).toUpperCase(),
          thumbprint: Array.from({ length: 40 }, () => Math.floor(Math.random() * 16).toString(16)).join('').toUpperCase(),
          algorithm: algorithms[Math.floor(Math.random() * algorithms.length)],
          keyUsage: ['Digital Signature', 'Key Encipherment', 'Data Encipherment'].slice(0, Math.floor(Math.random() * 3) + 1),
          issuedDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
          expiryDate: expiryDate.toISOString(),
          daysUntilExpiry,
          autoRenewal: Math.random() > 0.3,
          keyVault: keyVaults[Math.floor(Math.random() * keyVaults.length)],
          resourceGroup: `rg-security-${Math.floor(Math.random() * 5) + 1}`,
          subscription: 'PolicyCortex-Prod',
          domain: domains[Math.floor(Math.random() * domains.length)],
          subjectAlternativeNames: [`*.${domains[Math.floor(Math.random() * domains.length)]}`, `www.${domains[Math.floor(Math.random() * domains.length)]}`],
          certificateChain: ['Root CA', 'Intermediate CA', 'End Entity'],
          lastValidation: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
          validationStatus: ['valid', 'warning', 'error'][Math.floor(Math.random() * 3)] as 'valid' | 'warning' | 'error',
          tags: {
            environment: ['prod', 'staging', 'dev'][Math.floor(Math.random() * 3)],
            team: ['Security', 'DevOps', 'Infrastructure'][Math.floor(Math.random() * 3)],
            criticality: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)]
          }
        };
      });
    };

    setTimeout(() => {
      const mockCerts = generateMockCertificates();
      setCertificates(mockCerts);

      // Calculate metrics
      const validCerts = mockCerts.filter(c => c.status === 'valid').length;
      const expiringCerts = mockCerts.filter(c => c.status === 'expiring').length;
      const expiredCerts = mockCerts.filter(c => c.status === 'expired').length;
      const autoRenewalEnabled = mockCerts.filter(c => c.autoRenewal).length;
      const validationErrors = mockCerts.filter(c => c.validationStatus === 'error').length;
      const avgDaysToExpiry = mockCerts
        .filter(c => c.daysUntilExpiry > 0)
        .reduce((sum, c) => sum + c.daysUntilExpiry, 0) / mockCerts.filter(c => c.daysUntilExpiry > 0).length;

      setMetrics({
        totalCertificates: mockCerts.length,
        validCertificates: validCerts,
        expiringCertificates: expiringCerts,
        expiredCertificates: expiredCerts,
        complianceScore: Math.round(((validCerts + expiringCerts) / mockCerts.length) * 100),
        autoRenewalEnabled: Math.round((autoRenewalEnabled / mockCerts.length) * 100),
        validationErrors,
        averageDaysToExpiry: Math.round(avgDaysToExpiry)
      });

      setLoading(false);
    }, 1000);
  }, []);

  const filteredCertificates = certificates.filter(cert => {
    const matchesSearch = cert.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cert.commonName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cert.issuer.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || cert.status === statusFilter;
    const matchesType = typeFilter === 'all' || cert.type === typeFilter;
    return matchesSearch && matchesStatus && matchesType;
  });

  const getStatusColor = (status: Certificate['status']) => {
    switch (status) {
      case 'valid': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'expired': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'expiring': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'revoked': return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
      case 'pending': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'renewed': return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getValidationStatusColor = (status: string) => {
    switch (status) {
      case 'valid': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const toggleCertificateSelection = (certId: string) => {
    setSelectedCertificates(prev => 
      prev.includes(certId) 
        ? prev.filter(id => id !== certId)
        : [...prev, certId]
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
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Award className="h-8 w-8 text-green-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Digital Certificates</h1>
              <p className="text-gray-400">Manage SSL/TLS certificates and digital identities</p>
            </div>
          </div>
          <Button 
            onClick={() => setShowCreateDialog(true)}
            className="bg-green-600 hover:bg-green-700"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Certificate
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
                  <p className="text-gray-400 text-sm">Total Certificates</p>
                  <p className="text-2xl font-bold text-white">{metrics.totalCertificates}</p>
                </div>
                <Award className="h-8 w-8 text-green-400" />
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
                <Shield className="h-8 w-8 text-blue-400" />
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
                  <p className="text-2xl font-bold text-yellow-400">{metrics.expiringCertificates}</p>
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
                  <p className="text-gray-400 text-sm">Auto Renewal</p>
                  <p className="text-2xl font-bold text-white">{metrics.autoRenewalEnabled}%</p>
                </div>
                <RefreshCw className="h-8 w-8 text-purple-400" />
              </div>
              <div className="mt-2">
                <Progress value={metrics.autoRenewalEnabled} className="h-2" />
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
                placeholder="Search certificates..."
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
                <SelectItem value="valid">Valid</SelectItem>
                <SelectItem value="expired">Expired</SelectItem>
                <SelectItem value="expiring">Expiring</SelectItem>
                <SelectItem value="revoked">Revoked</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="renewed">Renewed</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="SSL/TLS">SSL/TLS</SelectItem>
                <SelectItem value="Code Signing">Code Signing</SelectItem>
                <SelectItem value="Client Auth">Client Auth</SelectItem>
                <SelectItem value="Email">Email</SelectItem>
                <SelectItem value="Root CA">Root CA</SelectItem>
                <SelectItem value="Intermediate CA">Intermediate CA</SelectItem>
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
          {selectedCertificates.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-green-600/10 border border-green-600/20 rounded-lg p-4"
            >
              <div className="flex items-center justify-between">
                <span className="text-green-400">
                  {selectedCertificates.length} certificate{selectedCertificates.length > 1 ? 's' : ''} selected
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="border-green-600 text-green-400">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Renew Selected
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

        {/* Certificates Table */}
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
                          setSelectedCertificates(filteredCertificates.map(c => c.id));
                        } else {
                          setSelectedCertificates([]);
                        }
                      }}
                      checked={selectedCertificates.length === filteredCertificates.length}
                      className="rounded"
                    />
                  </th>
                  <th className="text-left p-4 font-medium text-gray-300">Certificate</th>
                  <th className="text-left p-4 font-medium text-gray-300">Type</th>
                  <th className="text-left p-4 font-medium text-gray-300">Status</th>
                  <th className="text-left p-4 font-medium text-gray-300">Issuer</th>
                  <th className="text-left p-4 font-medium text-gray-300">Expiry</th>
                  <th className="text-left p-4 font-medium text-gray-300">Auto Renewal</th>
                  <th className="text-left p-4 font-medium text-gray-300">Validation</th>
                  <th className="text-left p-4 font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredCertificates.map((cert, index) => (
                    <motion.tr
                      key={cert.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-t border-gray-800 hover:bg-gray-800/30"
                    >
                      <td className="p-4">
                        <input
                          type="checkbox"
                          checked={selectedCertificates.includes(cert.id)}
                          onChange={() => toggleCertificateSelection(cert.id)}
                          className="rounded"
                        />
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-green-500/10 rounded">
                            <Award className="h-4 w-4 text-green-400" />
                          </div>
                          <div>
                            <div className="font-medium text-white">{cert.commonName}</div>
                            <div className="text-sm text-gray-400">{cert.name}</div>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className="border-green-500/20 text-green-400">
                          {cert.type}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <Badge className={getStatusColor(cert.status)}>
                          {cert.status}
                        </Badge>
                      </td>
                      <td className="p-4 text-gray-300">{cert.issuer}</td>
                      <td className="p-4">
                        <div className="text-gray-300">
                          {new Date(cert.expiryDate).toLocaleDateString()}
                        </div>
                        <div className={`text-xs ${cert.daysUntilExpiry < 30 ? 'text-red-400' : cert.daysUntilExpiry < 90 ? 'text-yellow-400' : 'text-gray-500'}`}>
                          {cert.daysUntilExpiry > 0 ? `${cert.daysUntilExpiry} days` : 'Expired'}
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          {cert.autoRenewal ? (
                            <>
                              <CheckCircle className="h-4 w-4 text-green-400" />
                              <span className="text-sm text-green-400">Enabled</span>
                            </>
                          ) : (
                            <>
                              <AlertTriangle className="h-4 w-4 text-yellow-400" />
                              <span className="text-sm text-yellow-400">Disabled</span>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="p-4">
                        <div className={`flex items-center space-x-1 ${getValidationStatusColor(cert.validationStatus)}`}>
                          {cert.validationStatus === 'valid' && <CheckCircle className="h-4 w-4" />}
                          {cert.validationStatus === 'warning' && <AlertTriangle className="h-4 w-4" />}
                          {cert.validationStatus === 'error' && <AlertTriangle className="h-4 w-4" />}
                          <span className="text-sm capitalize">{cert.validationStatus}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center space-x-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setSelectedCertificate(cert)}
                            className="h-8 w-8 p-0"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyToClipboard(cert.thumbprint)}
                            className="h-8 w-8 p-0"
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0"
                          >
                            <RefreshCw className="h-4 w-4" />
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

        {/* Certificate Details Dialog */}
        <Dialog open={selectedCertificate !== null} onOpenChange={() => setSelectedCertificate(null)}>
          <DialogContent className="bg-gray-900 border-gray-800 max-w-3xl">
            <DialogHeader>
              <DialogTitle className="text-white flex items-center space-x-2">
                <Award className="h-5 w-5 text-green-400" />
                <span>Certificate Details: {selectedCertificate?.commonName}</span>
              </DialogTitle>
            </DialogHeader>
            
            {selectedCertificate && (
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-5 bg-gray-800">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="subject">Subject</TabsTrigger>
                  <TabsTrigger value="chain">Chain</TabsTrigger>
                  <TabsTrigger value="validation">Validation</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Common Name</Label>
                      <span className="text-white">{selectedCertificate.commonName}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Type</Label>
                      <Badge variant="outline" className="border-green-500/20 text-green-400">
                        {selectedCertificate.type}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Status</Label>
                      <Badge className={getStatusColor(selectedCertificate.status)}>
                        {selectedCertificate.status}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Algorithm</Label>
                      <span className="text-white">{selectedCertificate.algorithm}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Issuer</Label>
                      <span className="text-white">{selectedCertificate.issuer}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Serial Number</Label>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{selectedCertificate.serialNumber}</span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => copyToClipboard(selectedCertificate.serialNumber)}
                          className="h-6 w-6 p-0"
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Issued Date</Label>
                      <span className="text-white">{new Date(selectedCertificate.issuedDate).toLocaleDateString()}</span>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Expiry Date</Label>
                      <span className="text-white">{new Date(selectedCertificate.expiryDate).toLocaleDateString()}</span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Thumbprint</Label>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-sm break-all">{selectedCertificate.thumbprint}</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(selectedCertificate.thumbprint)}
                        className="h-6 w-6 p-0"
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="subject" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Subject</Label>
                    <div className="text-white font-mono text-sm">{selectedCertificate.subject}</div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Subject Alternative Names</Label>
                    <div className="space-y-1">
                      {selectedCertificate.subjectAlternativeNames.map((san, index) => (
                        <div key={index} className="text-white text-sm">{san}</div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-400">Key Usage</Label>
                    <div className="flex flex-wrap gap-2">
                      {selectedCertificate.keyUsage.map((usage) => (
                        <Badge key={usage} variant="outline" className="border-gray-700">
                          {usage}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="chain" className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-400">Certificate Chain</Label>
                    <div className="space-y-2">
                      {selectedCertificate.certificateChain.map((cert, index) => (
                        <div key={index} className="flex items-center space-x-2">
                          <Award className="h-4 w-4 text-green-400" />
                          <span className="text-white">{cert}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="validation" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Validation Status</Label>
                      <div className={`flex items-center space-x-1 ${getValidationStatusColor(selectedCertificate.validationStatus)}`}>
                        {selectedCertificate.validationStatus === 'valid' && <CheckCircle className="h-4 w-4" />}
                        {selectedCertificate.validationStatus === 'warning' && <AlertTriangle className="h-4 w-4" />}
                        {selectedCertificate.validationStatus === 'error' && <AlertTriangle className="h-4 w-4" />}
                        <span className="capitalize">{selectedCertificate.validationStatus}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-gray-400">Last Validation</Label>
                      <span className="text-white">{new Date(selectedCertificate.lastValidation).toLocaleString()}</span>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="settings" className="space-y-4">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-gray-400">Auto Renewal</Label>
                      <div className="flex items-center space-x-2">
                        {selectedCertificate.autoRenewal ? (
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
                    <div className="flex space-x-2">
                      <Button className="bg-green-600 hover:bg-green-700">
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Renew Now
                      </Button>
                      <Button variant="outline" className="border-red-600 text-red-400">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Revoke Certificate
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>

        {/* Create Certificate Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent className="bg-gray-900 border-gray-800">
            <DialogHeader>
              <DialogTitle className="text-white">Request New Certificate</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Common Name</Label>
                <Input placeholder="example.com" className="bg-gray-800 border-gray-700" />
              </div>
              
              <div className="space-y-2">
                <Label>Certificate Type</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select certificate type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ssl-tls">SSL/TLS</SelectItem>
                    <SelectItem value="code-signing">Code Signing</SelectItem>
                    <SelectItem value="client-auth">Client Authentication</SelectItem>
                    <SelectItem value="email">Email</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label>Key Algorithm</Label>
                <Select>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue placeholder="Select algorithm" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rsa-2048">RSA-2048</SelectItem>
                    <SelectItem value="rsa-4096">RSA-4096</SelectItem>
                    <SelectItem value="ecc-p256">ECC-P256</SelectItem>
                    <SelectItem value="ecc-p384">ECC-P384</SelectItem>
                    <SelectItem value="ed25519">Ed25519</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label>Subject Alternative Names</Label>
                <Textarea 
                  placeholder="*.example.com&#10;www.example.com&#10;api.example.com"
                  className="bg-gray-800 border-gray-700"
                />
              </div>
              
              <div className="flex space-x-2 pt-4">
                <Button className="bg-green-600 hover:bg-green-700 flex-1">
                  Request Certificate
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


