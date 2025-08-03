import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ResourceDetailModal from '../../components/ResourceDetailModal'
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Badge,
  Tooltip,
  Stack,
  Tabs,
  Tab,
  Divider,
  TextField,
  InputAdornment
} from '@mui/material'
import {
  PolicyOutlined,
  SecurityOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ArrowBackOutlined,
  FilterListOutlined,
  SearchOutlined,
  VisibilityOutlined,
  GetAppOutlined,
  RefreshOutlined,
  InfoOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface PolicyResource {
  id: string
  name: string
  type: string
  status: string
  location: string
  resourceGroup: string
  policyDefinitionAction: string
  timestamp: string
  complianceReasonCode: string
  subscriptionId: string
}

interface PolicyDetails {
  id: string
  name: string
  displayName: string
  description: string
  type: string
  category: string
  effect: string
  scope: string
  policyDefinitionId: string
  compliance: {
    status: string
    compliancePercentage: number
    resourceCount: number
    compliantResources: number
    nonCompliantResources: number
    exemptResources: number
    lastEvaluated: string
  }
  resources: PolicyResource[]
  parameters: Record<string, any>
  metadata: {
    assignedBy: string
    source: string
    createdOn: string
    updatedOn: string
    data_source?: string
  }
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`policy-tabpanel-${index}`}
      aria-labelledby={`policy-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  )
}

const PolicyDetailsPage = () => {
  const { id: policyId } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [policy, setPolicy] = useState<PolicyDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tabValue, setTabValue] = useState(0)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [resourceModalOpen, setResourceModalOpen] = useState(false)
  const [selectedResourceId, setSelectedResourceId] = useState('')
  const [selectedResourceName, setSelectedResourceName] = useState('')

  const fetchPolicyDetails = async () => {
    if (!policyId) return
    
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get(`/api/v1/policies/${policyId}`)
      
      if (response.data) {
        setPolicy(response.data)
      }
    } catch (err: any) {
      console.error('Error fetching policy details:', err)
      setError('Failed to load policy details')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPolicyDetails()
  }, [policyId])

  const getComplianceColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'compliant': return 'success'
      case 'noncompliant': return 'error'
      default: return 'warning'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString()
  }

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const handleResourceClick = (resourceId: string, resourceName: string) => {
    setSelectedResourceId(resourceId)
    setSelectedResourceName(resourceName)
    setResourceModalOpen(true)
  }

  const handleResourceModalClose = () => {
    setResourceModalOpen(false)
    setSelectedResourceId('')
    setSelectedResourceName('')
  }

  const filteredResources = policy?.resources?.filter(resource => {
    const matchesFilter = filterStatus === 'all' || 
      (filterStatus === 'compliant' && resource.status === 'Compliant') ||
      (filterStatus === 'noncompliant' && resource.status === 'NonCompliant')
    
    const matchesSearch = !searchTerm || 
      resource.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.resourceGroup.toLowerCase().includes(searchTerm.toLowerCase())
    
    return matchesFilter && matchesSearch
  }) || []

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error || !policy) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error || 'Policy not found'}
        </Alert>
        <Button
          variant="outlined"
          startIcon={<ArrowBackOutlined />}
          onClick={() => navigate('/policies')}
        >
          Back to Policies
        </Button>
      </Box>
    )
  }

  return (
    <>
      <Helmet>
        <title>{policy.displayName} - Policy Details - PolicyCortex</title>
        <meta name="description" content={`View compliance details for ${policy.displayName}`} />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <IconButton onClick={() => navigate('/policies')} size="large">
            <ArrowBackOutlined />
          </IconButton>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PolicyOutlined />
              {policy.displayName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {policy.name} • {policy.category} • {policy.type}
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={<RefreshOutlined />}
            onClick={fetchPolicyDetails}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <PolicyOutlined color="primary" />
                  <Box>
                    <Typography variant="h4">{policy.compliance.resourceCount}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Resources
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CheckCircleOutlined color="success" />
                  <Box>
                    <Typography variant="h4">{policy.compliance.compliantResources}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Compliant
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <WarningOutlined color="error" />
                  <Box>
                    <Typography variant="h4">{policy.compliance.nonCompliantResources}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Non-Compliant
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Chip
                    label={policy.compliance.status}
                    color={getComplianceColor(policy.compliance.status) as any}
                    icon={policy.compliance.status === 'Compliant' ? <CheckCircleOutlined /> : <WarningOutlined />}
                  />
                  <Box>
                    <Typography variant="h4">{policy.compliance.compliancePercentage}%</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Compliance Rate
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="Resource Compliance" />
            <Tab label="Policy Definition" />
            <Tab label="Parameters & Metadata" />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            {/* Resource Compliance Tab */}
            <Box sx={{ mb: 3 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Search resources..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <SearchOutlined />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={3}>
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Filter by Status"
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    SelectProps={{ native: true }}
                  >
                    <option value="all">All Resources</option>
                    <option value="compliant">Compliant Only</option>
                    <option value="noncompliant">Non-Compliant Only</option>
                  </TextField>
                </Grid>
                <Grid item xs={12} md={5}>
                  <Typography variant="body2" color="text.secondary">
                    Showing {filteredResources.length} of {policy.compliance.resourceCount} resources
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Resource Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Resource Group</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Compliance Status</TableCell>
                    <TableCell>Last Evaluated</TableCell>
                    <TableCell>Reason</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredResources.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} align="center">
                        <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                          No resources found matching your criteria
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredResources.map((resource) => (
                      <TableRow 
                        key={resource.id} 
                        hover
                        onClick={() => handleResourceClick(resource.id, resource.name)}
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: 'action.hover'
                          }
                        }}
                      >
                        <TableCell>
                          <Typography variant="subtitle2">
                            {resource.name}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {resource.type}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2">
                            {resource.resourceGroup}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="body2">
                            {resource.location}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Chip
                            label={resource.status}
                            size="small"
                            color={getComplianceColor(resource.status) as any}
                            icon={resource.status === 'Compliant' ? <CheckCircleOutlined /> : <WarningOutlined />}
                          />
                        </TableCell>
                        
                        <TableCell>
                          <Typography variant="caption">
                            {formatDateTime(resource.timestamp)}
                          </Typography>
                        </TableCell>
                        
                        <TableCell>
                          {resource.complianceReasonCode ? (
                            <Tooltip title={resource.complianceReasonCode}>
                              <IconButton size="small">
                                <InfoOutlined />
                              </IconButton>
                            </Tooltip>
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              -
                            </Typography>
                          )}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {/* Policy Definition Tab */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Policy Description
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {policy.description}
                    </Typography>
                    
                    <Divider sx={{ my: 2 }} />
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Effect
                        </Typography>
                        <Typography variant="body1">
                          {policy.effect}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Category
                        </Typography>
                        <Typography variant="body1">
                          {policy.category}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Scope
                        </Typography>
                        <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                          {policy.scope}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Policy Definition ID
                        </Typography>
                        <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                          {policy.policyDefinitionId}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Quick Actions
                    </Typography>
                    <Stack spacing={2}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<GetAppOutlined />}
                        onClick={() => console.log('Export compliance report')}
                      >
                        Export Report
                      </Button>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<VisibilityOutlined />}
                        onClick={() => console.log('View in Azure Portal')}
                      >
                        View in Azure Portal
                      </Button>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {/* Parameters & Metadata Tab */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Metadata
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Assigned By
                        </Typography>
                        <Typography variant="body1">
                          {policy.metadata.assignedBy}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Source
                        </Typography>
                        <Typography variant="body1">
                          {policy.metadata.source}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Created On
                        </Typography>
                        <Typography variant="body1">
                          {formatDate(policy.metadata.createdOn)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Updated On
                        </Typography>
                        <Typography variant="body1">
                          {formatDate(policy.metadata.updatedOn)}
                        </Typography>
                      </Grid>
                      {policy.metadata.data_source && (
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Data Source
                          </Typography>
                          <Typography variant="body2">
                            {policy.metadata.data_source}
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Parameters
                    </Typography>
                    {Object.keys(policy.parameters).length === 0 ? (
                      <Typography variant="body2" color="text.secondary">
                        No custom parameters configured for this policy.
                      </Typography>
                    ) : (
                      <Box>
                        {Object.entries(policy.parameters).map(([key, value]) => (
                          <Box key={key} sx={{ mb: 1 }}>
                            <Typography variant="subtitle2" color="text.secondary">
                              {key}
                            </Typography>
                            <Typography variant="body1">
                              {JSON.stringify(value)}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Paper>

        {/* Data Source Info */}
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Last evaluated: {formatDateTime(policy.compliance.lastEvaluated)} • 
            Data source: {policy.metadata.data_source || 'Azure Policy Insights'}
          </Typography>
        </Box>
      </Box>

      {/* Resource Detail Modal */}
      <ResourceDetailModal
        open={resourceModalOpen}
        onClose={handleResourceModalClose}
        resourceId={selectedResourceId}
        resourceName={selectedResourceName}
      />
    </>
  )
}

export default PolicyDetailsPage