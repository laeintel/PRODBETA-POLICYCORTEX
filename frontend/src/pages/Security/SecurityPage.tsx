import React, { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useFilter } from '../../contexts/FilterContext'
import GlobalFilterPanel from '../../components/Filters/GlobalFilterPanel'
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  LinearProgress,
  Stack,
  IconButton,
  Tooltip,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar
} from '@mui/material'
import {
  SecurityOutlined,
  RefreshOutlined,
  ShieldOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ErrorOutlined,
  VpnKeyOutlined,
  LockOutlined,
  VisibilityOutlined,
  BugReportOutlined,
  PolicyOutlined,
  SecurityOutlined as NetworkSecurityOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface SecurityIssue {
  id: string
  title: string
  severity: 'High' | 'Medium' | 'Low'
  category: string
  resourceName: string
  resourceType: string
  description: string
  recommendation: string
  status: 'Open' | 'In Progress' | 'Resolved'
  detectedDate: string
}

interface SecurityData {
  summary: {
    totalIssues: number
    highSeverityIssues: number
    mediumSeverityIssues: number
    lowSeverityIssues: number
    resolvedIssues: number
    securityScore: number
    complianceScore: number
  }
  issues: SecurityIssue[]
  securityCenters: Array<{
    name: string
    status: string
    lastScanned: string
    findings: number
  }>
  recommendations: Array<{
    title: string
    impact: string
    effort: string
    category: string
  }>
  data_source: string
}

const SecurityPage = () => {
  const navigate = useNavigate()
  const { applyFilters } = useFilter()
  const [securityData, setSecurityData] = useState<SecurityData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Apply filters to security issues
  const filteredIssues = useMemo(() => {
    if (!securityData?.issues?.length) return []
    
    return applyFilters(securityData.issues.map(issue => ({
      ...issue,
      name: issue.title || issue.resourceName || issue.id, // Add required name property
      subscription: '/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595', // Mock subscription
      resourceGroup: issue.resourceName?.includes('dev') ? 'rg-policycortex-dev' : 'rg-policycortex-prod',
      type: issue.resourceType || '',
      location: 'East US' // Mock location
    })))
  }, [securityData?.issues, applyFilters])

  const fetchSecurityData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.get('/api/v1/security/overview').catch(() => null)
      
      if (response) {
        setSecurityData(response.data)
      } else {
        // Generate mock security data
        setSecurityData(generateMockSecurityData())
      }
    } catch (err: any) {
      console.error('Error fetching security data:', err)
      setError('Failed to load security data')
      setSecurityData(generateMockSecurityData())
    } finally {
      setLoading(false)
    }
  }

  const generateMockSecurityData = (): SecurityData => {
    const mockIssues: SecurityIssue[] = [
      {
        id: 'sec-001',
        title: 'Storage account allows public access',
        severity: 'High',
        category: 'Data Protection',
        resourceName: 'stpolicycortexdev',
        resourceType: 'Microsoft.Storage/storageAccounts',
        description: 'Storage account has public blob access enabled',
        recommendation: 'Disable public blob access and use private endpoints',
        status: 'Open',
        detectedDate: '2024-08-01T10:30:00Z'
      },
      {
        id: 'sec-002',
        title: 'VM missing security patches',
        severity: 'Medium',
        category: 'Patch Management',
        resourceName: 'vm-policycortex-dev',
        resourceType: 'Microsoft.Compute/virtualMachines',
        description: '15 critical security updates are missing',
        recommendation: 'Enable automatic patching and install pending updates',
        status: 'In Progress',
        detectedDate: '2024-07-28T14:20:00Z'
      },
      {
        id: 'sec-003',
        title: 'Network Security Group allows unrestricted SSH',
        severity: 'High',
        category: 'Network Security',
        resourceName: 'nsg-dev-subnet',
        resourceType: 'Microsoft.Network/networkSecurityGroups',
        description: 'SSH access is allowed from any source IP (0.0.0.0/0)',
        recommendation: 'Restrict SSH access to specific IP ranges or use bastion host',
        status: 'Open',
        detectedDate: '2024-08-02T09:15:00Z'
      },
      {
        id: 'sec-004',
        title: 'Key Vault access policy too permissive',
        severity: 'Medium',
        category: 'Identity & Access',
        resourceName: 'kv-policycortex-dev',
        resourceType: 'Microsoft.KeyVault/vaults',
        description: 'Multiple users have unnecessary secret management permissions',
        recommendation: 'Apply principle of least privilege to Key Vault access policies',
        status: 'Open',
        detectedDate: '2024-07-30T11:45:00Z'
      },
      {
        id: 'sec-005',
        title: 'Container registry allows anonymous pulls',
        severity: 'Low',
        category: 'Container Security',
        resourceName: 'crpolicycortexdev',
        resourceType: 'Microsoft.ContainerRegistry/registries',
        description: 'Anonymous pull access is enabled for container registry',
        recommendation: 'Disable anonymous pulls and use authentication',
        status: 'Resolved',
        detectedDate: '2024-07-25T16:30:00Z'
      }
    ]

    return {
      summary: {
        totalIssues: mockIssues.length,
        highSeverityIssues: mockIssues.filter(i => i.severity === 'High').length,
        mediumSeverityIssues: mockIssues.filter(i => i.severity === 'Medium').length,
        lowSeverityIssues: mockIssues.filter(i => i.severity === 'Low').length,
        resolvedIssues: mockIssues.filter(i => i.status === 'Resolved').length,
        securityScore: 78,
        complianceScore: 82
      },
      issues: mockIssues,
      securityCenters: [
        {
          name: 'Microsoft Defender for Cloud',
          status: 'Active',
          lastScanned: '2024-08-05T08:00:00Z',
          findings: 3
        },
        {
          name: 'Azure Security Center',
          status: 'Active',
          lastScanned: '2024-08-05T06:30:00Z',
          findings: 2
        },
        {
          name: 'Policy Compliance Scanner',
          status: 'Active',
          lastScanned: '2024-08-05T07:15:00Z',
          findings: 1
        }
      ],
      recommendations: [
        {
          title: 'Enable Azure Sentinel for SIEM',
          impact: 'High',
          effort: 'Medium',
          category: 'Monitoring'
        },
        {
          title: 'Implement Just-In-Time VM Access',
          impact: 'High',
          effort: 'Low',
          category: 'Access Control'
        },
        {
          title: 'Enable disk encryption for all VMs',
          impact: 'Medium',
          effort: 'Low',
          category: 'Data Protection'
        }
      ],
      data_source: 'mock-security-data'
    }
  }

  useEffect(() => {
    fetchSecurityData()
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'High': return 'error'
      case 'Medium': return 'warning'
      case 'Low': return 'info'
      default: return 'default'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Open': return 'error'
      case 'In Progress': return 'warning'
      case 'Resolved': return 'success'
      default: return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Open': return <ErrorOutlined />
      case 'In Progress': return <WarningOutlined />
      case 'Resolved': return <CheckCircleOutlined />
      default: return <ErrorOutlined />
    }
  }

  const handleComplianceClick = () => {
    navigate('/security/compliance')
  }

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <>
      <Helmet>
        <title>Security - PolicyCortex</title>
        <meta name="description" content="Security and compliance overview" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityOutlined />
            Security
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchSecurityData}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<PolicyOutlined />}
              onClick={handleComplianceClick}
            >
              Compliance
            </Button>
          </Stack>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Global Filter Panel */}
        <GlobalFilterPanel
          availableResourceGroups={securityData?.issues?.map(i => i.resourceName?.includes('dev') ? 'rg-policycortex-dev' : 'rg-policycortex-prod').filter(Boolean) || []}
          availableResourceTypes={securityData?.issues?.map(i => i.resourceType).filter(Boolean) || []}
        />

        {securityData && (
          <>
            {/* Security Score Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <ShieldOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{securityData?.summary?.securityScore || 0}%</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Security Score
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={securityData?.summary?.securityScore || 0} 
                      sx={{ mt: 2 }}
                      color={securityData?.summary?.securityScore >= 80 ? 'success' : securityData?.summary?.securityScore >= 60 ? 'warning' : 'error'}
                    />
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <PolicyOutlined color="info" />
                      <Box>
                        <Typography variant="h4">{securityData?.summary?.complianceScore || 0}%</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Compliance Score
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={securityData?.summary?.complianceScore || 0} 
                      sx={{ mt: 2 }}
                      color={securityData?.summary?.complianceScore >= 80 ? 'success' : securityData?.summary?.complianceScore >= 60 ? 'warning' : 'error'}
                    />
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <BugReportOutlined color="warning" />
                      <Box>
                        <Typography variant="h4">{securityData?.summary?.totalIssues || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Security Issues
                        </Typography>
                      </Box>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Chip 
                        label={`${securityData?.summary?.highSeverityIssues || 0} High`} 
                        size="small" 
                        color="error" 
                        variant="outlined"
                      />
                      <Chip 
                        label={`${securityData?.summary?.mediumSeverityIssues || 0} Med`} 
                        size="small" 
                        color="warning" 
                        variant="outlined"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <CheckCircleOutlined color="success" />
                      <Box>
                        <Typography variant="h4">{securityData?.summary?.resolvedIssues || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Resolved Issues
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              {/* Security Issues Table */}
              <Grid item xs={12} lg={8}>
                <Paper sx={{ overflow: 'hidden' }}>
                  <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                    <Typography variant="h6">Security Issues</Typography>
                  </Box>
                  
                  <TableContainer sx={{ maxHeight: 500 }}>
                    <Table stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Issue</TableCell>
                          <TableCell>Resource</TableCell>
                          <TableCell align="center">Severity</TableCell>
                          <TableCell align="center">Status</TableCell>
                          <TableCell align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredIssues.map((issue) => (
                          <TableRow key={issue.id} hover>
                            <TableCell>
                              <Box>
                                <Typography variant="subtitle2">{issue.title}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {issue.category} • {new Date(issue.detectedDate).toLocaleDateString()}
                                </Typography>
                              </Box>
                            </TableCell>
                            
                            <TableCell>
                              <Box>
                                <Typography variant="body2">{issue.resourceName}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {issue.resourceType.split('/').pop()}
                                </Typography>
                              </Box>
                            </TableCell>
                            
                            <TableCell align="center">
                              <Chip
                                label={issue.severity}
                                color={getSeverityColor(issue.severity) as any}
                                size="small"
                              />
                            </TableCell>
                            
                            <TableCell align="center">
                              <Chip
                                icon={getStatusIcon(issue.status)}
                                label={issue.status}
                                color={getStatusColor(issue.status) as any}
                                size="small"
                                variant="outlined"
                              />
                            </TableCell>
                            
                            <TableCell align="center">
                              <Tooltip title="View Details">
                                <IconButton size="small" color="primary">
                                  <VisibilityOutlined />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>
              </Grid>

              {/* Sidebar with Security Centers and Recommendations */}
              <Grid item xs={12} lg={4}>
                <Stack spacing={3}>
                  {/* Security Centers */}
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Security Centers</Typography>
                    <List dense>
                      {(securityData?.securityCenters || []).map((center, index) => (
                        <ListItem key={index}>
                          <ListItemAvatar>
                            <Avatar sx={{ bgcolor: 'success.main', width: 32, height: 32 }}>
                              <NetworkSecurityOutlined />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={center.name}
                            secondary={`${center.findings} findings • ${center.status}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>

                  {/* Security Recommendations */}
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Top Recommendations</Typography>
                    <List dense>
                      {(securityData?.recommendations || []).map((rec, index) => (
                        <ListItem key={index}>
                          <ListItemAvatar>
                            <Avatar sx={{ bgcolor: 'info.main', width: 32, height: 32 }}>
                              <LockOutlined />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={rec.title}
                            secondary={`${rec.impact} impact • ${rec.effort} effort`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                </Stack>
              </Grid>
            </Grid>

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {securityData?.data_source || 'security-management'} • Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default SecurityPage