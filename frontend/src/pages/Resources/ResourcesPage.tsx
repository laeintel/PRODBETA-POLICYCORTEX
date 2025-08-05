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
  Divider,
  IconButton,
  Tooltip
} from '@mui/material'
import {
  CloudOutlined,
  RefreshOutlined,
  CheckCircleOutlined,
  ErrorOutlined,
  AttachMoneyOutlined,
  SecurityOutlined,
  StorageOutlined,
  ComputerOutlined,
  NetworkPingOutlined,
  VisibilityOutlined,
  InventoryOutlined,
  AccountTreeOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface Resource {
  id: string
  name: string
  type: string
  resourceGroup: string
  location: string
  status: string
  compliance: {
    status: string
    policiesApplied: number
    violations: number
  }
  cost: {
    dailyCost: number
    monthlyCost: number
    currency: string
  }
  tags: Record<string, string>
  createdTime: string
  lastModified: string
}

interface ResourceData {
  resources: Resource[]
  summary: {
    total: number
    running: number
    stopped: number
    compliant: number
    nonCompliant: number
    totalMonthlyCost: number
  }
  resourceGroups: string[]
  data_source: string
}

const ResourcesPage = () => {
  const navigate = useNavigate()
  const { applyFilters } = useFilter()
  const [resourceData, setResourceData] = useState<ResourceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Apply filters to resources
  const filteredResources = useMemo(() => {
    if (!resourceData?.resources) return []
    
    return applyFilters(resourceData.resources.map(resource => ({
      ...resource,
      subscription: resource.id.split('/')[2], // Extract subscription from ID
      type: resource.type
    })))
  }, [resourceData?.resources, applyFilters])

  const fetchResourceData = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get('/api/v1/resources')
      setResourceData(response.data)
    } catch (err: any) {
      console.error('Error fetching resource data:', err)
      setError('Failed to load resource data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchResourceData()
  }, [])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
  }

  const getResourceIcon = (type: string) => {
    const lowerType = type.toLowerCase()
    if (lowerType.includes('storage')) return <StorageOutlined />
    if (lowerType.includes('compute') || lowerType.includes('vm')) return <ComputerOutlined />
    if (lowerType.includes('network')) return <NetworkPingOutlined />
    if (lowerType.includes('security')) return <SecurityOutlined />
    return <CloudOutlined />
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running': return 'success'
      case 'stopped': return 'error'
      case 'available': return 'info'
      default: return 'default'
    }
  }

  const getComplianceColor = (status: string) => {
    return status === 'Compliant' ? 'success' : 'error'
  }

  const handleResourceClick = (resourceId: string) => {
    const resourceName = resourceId.split('/').pop()
    navigate(`/resources/${resourceName}`)
  }

  const handleInventoryClick = () => {
    navigate('/resources/inventory')
  }

  const handleTopologyClick = () => {
    navigate('/resources/topology')
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
        <title>Resources - PolicyCortex</title>
        <meta name="description" content="Manage Azure resources and inventory" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CloudOutlined />
            Resources
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchResourceData}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<InventoryOutlined />}
              onClick={handleInventoryClick}
            >
              Inventory
            </Button>
            <Button
              variant="outlined"
              startIcon={<AccountTreeOutlined />}
              onClick={handleTopologyClick}
            >
              Topology
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
          availableResourceGroups={resourceData?.resourceGroups || []}
          availableResourceTypes={resourceData?.resources?.map(r => r.type) || []}
        />

        {resourceData && (
          <>
            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <CloudOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{filteredResources.length}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Resources
                        </Typography>
                      </Box>
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
                        <Typography variant="h4">{resourceData.summary.running}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Running
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <SecurityOutlined color="info" />
                      <Box>
                        <Typography variant="h4">
                          {Math.round((resourceData.summary.compliant / resourceData.summary.total) * 100)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Compliance Rate
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <AttachMoneyOutlined color="secondary" />
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency(resourceData.summary.totalMonthlyCost)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Monthly Cost
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Resources Table */}
            <Paper sx={{ overflow: 'hidden' }}>
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="h6">Resource Inventory</Typography>
              </Box>
              
              <TableContainer sx={{ maxHeight: 600 }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Resource</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Resource Group</TableCell>
                      <TableCell>Location</TableCell>
                      <TableCell align="center">Status</TableCell>
                      <TableCell align="center">Compliance</TableCell>
                      <TableCell align="right">Monthly Cost</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredResources.map((resource) => (
                      <TableRow 
                        key={resource.id} 
                        hover 
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': { backgroundColor: 'action.hover' }
                        }}
                        onClick={() => handleResourceClick(resource.id)}
                      >
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getResourceIcon(resource.type)}
                            <Box>
                              <Typography variant="subtitle2">{resource.name}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                Created: {new Date(resource.createdTime).toLocaleDateString()}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2">
                            {resource.type.split('/').pop()}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2">{resource.resourceGroup}</Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2">{resource.location}</Typography>
                        </TableCell>
                        
                        <TableCell align="center">
                          <Chip
                            label={resource.status}
                            color={getStatusColor(resource.status) as any}
                            size="small"
                          />
                        </TableCell>
                        
                        <TableCell align="center">
                          <Chip
                            icon={resource.compliance.status === 'Compliant' ? <CheckCircleOutlined /> : <ErrorOutlined />}
                            label={resource.compliance.status}
                            color={getComplianceColor(resource.compliance.status) as any}
                            size="small"
                          />
                        </TableCell>
                        
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight="medium">
                            {formatCurrency(resource.cost.monthlyCost)}
                          </Typography>
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

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {resourceData.data_source} • 
                Resource Groups: {resourceData.resourceGroups.length} • 
                Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default ResourcesPage