import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Stack,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material'
import {
  CloudOutlined,
  ArrowBackOutlined,
  CheckCircleOutlined,
  ErrorOutlined,
  AttachMoneyOutlined,
  SecurityOutlined,
  InfoOutlined,
  LocalOfferOutlined,
  HistoryOutlined,
  TrendingUpOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface ResourceDetails {
  id: string
  name: string
  type: string
  resourceGroup: string
  location: string
  status: string
  properties: {
    provisioningState: string
    sku: string
    tier: string
  }
  compliance: {
    status: string
    policiesApplied: number
    violations: any[]
    lastEvaluated: string
  }
  cost: {
    dailyCost: number
    monthlyCost: number
    currency: string
    costTrend: string
  }
  tags: Record<string, string>
  createdTime: string
  lastModified: string
}

const ResourceDetailsPage = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [resource, setResource] = useState<ResourceDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState(0)

  const fetchResourceDetails = async () => {
    if (!id) return
    
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get(`/api/v1/resources/${id}`)
      setResource(response.data)
    } catch (err: any) {
      console.error('Error fetching resource details:', err)
      setError('Failed to load resource details')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchResourceDetails()
  }, [id])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
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

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error || !resource) {
    return (
      <Box sx={{ p: 3 }}>
        <Button
          startIcon={<ArrowBackOutlined />}
          onClick={() => navigate('/resources')}
          sx={{ mb: 2 }}
        >
          Back to Resources
        </Button>
        <Alert severity="error">
          {error || 'Resource not found'}
        </Alert>
      </Box>
    )
  }

  return (
    <>
      <Helmet>
        <title>{resource.name} - Resource Details - PolicyCortex</title>
        <meta name="description" content={`Details for ${resource.name} resource`} />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ mb: 3 }}>
          <Button
            startIcon={<ArrowBackOutlined />}
            onClick={() => navigate('/resources')}
            sx={{ mb: 2 }}
          >
            Back to Resources
          </Button>
          
          <Stack direction="row" alignItems="center" spacing={2}>
            <CloudOutlined sx={{ fontSize: 40 }} color="primary" />
            <Box>
              <Typography variant="h4">{resource.name}</Typography>
              <Typography variant="body1" color="text.secondary">
                {resource.type.split('/').pop()} in {resource.resourceGroup}
              </Typography>
            </Box>
            <Box sx={{ ml: 'auto' }}>
              <Chip
                label={resource.status}
                color={getStatusColor(resource.status) as any}
                sx={{ mr: 1 }}
              />
              <Chip
                icon={resource.compliance.status === 'Compliant' ? <CheckCircleOutlined /> : <ErrorOutlined />}
                label={resource.compliance.status}
                color={getComplianceColor(resource.compliance.status) as any}
              />
            </Box>
          </Stack>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <AttachMoneyOutlined color="primary" />
                  <Box>
                    <Typography variant="h6">
                      {formatCurrency(resource.cost.monthlyCost)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Monthly Cost
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
                    <Typography variant="h6">{resource.compliance.policiesApplied}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Policies Applied
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
                  <TrendingUpOutlined color="secondary" />
                  <Box>
                    <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                      {resource.cost.costTrend}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Cost Trend
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
                  <InfoOutlined color="success" />
                  <Box>
                    <Typography variant="h6">{resource.properties.provisioningState}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Provisioning State
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Detailed Information Tabs */}
        <Paper>
          <Tabs value={activeTab} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab label="Overview" />
            <Tab label="Properties" />
            <Tab label="Compliance" />
            <Tab label="Cost Analysis" />
            <Tab label="Tags" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {/* Overview Tab */}
            {activeTab === 0 && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Basic Information</Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon><InfoOutlined /></ListItemIcon>
                      <ListItemText
                        primary="Resource ID"
                        secondary={resource.id}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CloudOutlined /></ListItemIcon>
                      <ListItemText
                        primary="Resource Type"
                        secondary={resource.type}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><InfoOutlined /></ListItemIcon>
                      <ListItemText
                        primary="Location"
                        secondary={resource.location}
                      />
                    </ListItem>
                  </List>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Timeline</Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon><HistoryOutlined /></ListItemIcon>
                      <ListItemText
                        primary="Created Time"
                        secondary={new Date(resource.createdTime).toLocaleString()}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><HistoryOutlined /></ListItemIcon>
                      <ListItemText
                        primary="Last Modified"
                        secondary={new Date(resource.lastModified).toLocaleString()}
                      />
                    </ListItem>
                  </List>
                </Grid>
              </Grid>
            )}

            {/* Properties Tab */}
            {activeTab === 1 && (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Property</TableCell>
                      <TableCell>Value</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Provisioning State</TableCell>
                      <TableCell>{resource.properties.provisioningState}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>SKU</TableCell>
                      <TableCell>{resource.properties.sku}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Tier</TableCell>
                      <TableCell>{resource.properties.tier}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            )}

            {/* Compliance Tab */}
            {activeTab === 2 && (
              <Box>
                <Alert severity={resource.compliance.status === 'Compliant' ? 'success' : 'error'} sx={{ mb: 3 }}>
                  Resource is {resource.compliance.status.toLowerCase()} with {resource.compliance.policiesApplied} policies applied.
                </Alert>
                <Typography variant="body2" color="text.secondary">
                  Last evaluated: {new Date(resource.compliance.lastEvaluated).toLocaleString()}
                </Typography>
              </Box>
            )}

            {/* Cost Analysis Tab */}
            {activeTab === 3 && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Cost Breakdown</Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Daily Cost"
                        secondary={formatCurrency(resource.cost.dailyCost, resource.cost.currency)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Monthly Cost"
                        secondary={formatCurrency(resource.cost.monthlyCost, resource.cost.currency)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Cost Trend"
                        secondary={resource.cost.costTrend}
                      />
                    </ListItem>
                  </List>
                </Grid>
              </Grid>
            )}

            {/* Tags Tab */}
            {activeTab === 4 && (
              <Box>
                <Typography variant="h6" gutterBottom>Resource Tags</Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {Object.entries(resource.tags).map(([key, value]) => (
                    <Chip
                      key={key}
                      icon={<LocalOfferOutlined />}
                      label={`${key}: ${value}`}
                      variant="outlined"
                    />
                  ))}
                </Stack>
              </Box>
            )}
          </Box>
        </Paper>
      </Box>
    </>
  )
}

export default ResourceDetailsPage