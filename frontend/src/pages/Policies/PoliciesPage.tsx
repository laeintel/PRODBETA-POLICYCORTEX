import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
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
  Stack
} from '@mui/material'
import {
  PolicyOutlined,
  SecurityOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  VisibilityOutlined,
  RefreshOutlined,
  TrendingUpOutlined,
  AccountTreeOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface Policy {
  id: string
  name: string
  displayName: string
  description: string
  type: string
  category: string
  effect: string
  compliance: {
    status: string
    compliancePercentage: number
    resourceCount: number
    compliantResources: number
    nonCompliantResources: number
  }
  scope: string
  createdOn: string
  updatedOn: string
  metadata: {
    assignedBy: string
    source: string
  }
}

interface PoliciesSummary {
  total: number
  compliant: number
  nonCompliant: number
  exempt: number
}

const PoliciesPage = () => {
  const navigate = useNavigate()
  const [policies, setPolicies] = useState<Policy[]>([])
  const [summary, setSummary] = useState<PoliciesSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPolicies = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get('/api/v1/policies')
      
      if (response.data) {
        setPolicies(response.data.policies || [])
        setSummary(response.data.summary || null)
      }
    } catch (err: any) {
      console.error('Error fetching policies:', err)
      setError('Failed to load policies data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPolicies()
  }, [])

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
        <title>Policies - PolicyCortex</title>
        <meta name="description" content="Manage Azure policies and compliance" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PolicyOutlined />
            Azure Policies
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshOutlined />}
            onClick={fetchPolicies}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Summary Cards */}
        {summary && (
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <AccountTreeOutlined color="primary" />
                    <Box>
                      <Typography variant="h4">{summary.total}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Policies
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
                      <Typography variant="h4">{summary.compliant}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Compliant
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
                    <WarningOutlined color="error" />
                    <Box>
                      <Typography variant="h4">{summary.nonCompliant}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Non-Compliant
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
                    <TrendingUpOutlined color="info" />
                    <Box>
                      <Typography variant="h4">
                        {summary.total > 0 ? Math.round((summary.compliant / summary.total) * 100) : 0}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Compliance Rate
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Policies Table */}
        <Paper sx={{ overflow: 'hidden' }}>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h6">Policy Assignments</Typography>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Policy Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Effect</TableCell>
                  <TableCell>Compliance</TableCell>
                  <TableCell>Resources</TableCell>
                  <TableCell>Updated</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {policies.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                        No policies found
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  policies.map((policy) => (
                    <TableRow 
                      key={policy.id} 
                      hover 
                      sx={{ 
                        cursor: 'pointer',
                        '&:hover': {
                          backgroundColor: 'action.hover'
                        }
                      }}
                      onClick={() => navigate(`/policies/${policy.id}`)}
                    >
                      <TableCell>
                        <Box>
                          <Typography variant="subtitle2">
                            {policy.displayName}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {policy.name}
                          </Typography>
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Chip 
                          label={policy.type} 
                          size="small" 
                          variant="outlined"
                          color="primary"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Chip 
                          label={policy.category} 
                          size="small" 
                          icon={<SecurityOutlined />}
                          color="secondary"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Chip 
                          label={policy.effect} 
                          size="small" 
                          variant="outlined"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Stack spacing={1}>
                          <Chip
                            label={policy.compliance.status}
                            size="small"
                            color={getComplianceColor(policy.compliance.status) as any}
                            icon={policy.compliance.status === 'Compliant' ? <CheckCircleOutlined /> : <WarningOutlined />}
                          />
                          <Box sx={{ width: '100px' }}>
                            <LinearProgress
                              variant="determinate"
                              value={policy.compliance.compliancePercentage}
                              color={getComplianceColor(policy.compliance.status) as any}
                            />
                            <Typography variant="caption">
                              {policy.compliance.compliancePercentage}%
                            </Typography>
                          </Box>
                        </Stack>
                      </TableCell>
                      
                      <TableCell>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Tooltip title={`${policy.compliance.compliantResources} compliant`}>
                            <Badge badgeContent={policy.compliance.compliantResources} color="success">
                              <CheckCircleOutlined fontSize="small" />
                            </Badge>
                          </Tooltip>
                          <Tooltip title={`${policy.compliance.nonCompliantResources} non-compliant`}>
                            <Badge badgeContent={policy.compliance.nonCompliantResources} color="error">
                              <WarningOutlined fontSize="small" />
                            </Badge>
                          </Tooltip>
                        </Stack>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="caption">
                          {formatDate(policy.updatedOn)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Tooltip title="View Details">
                          <IconButton 
                            size="small" 
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/policies/${policy.id}`)
                            }}
                          >
                            <VisibilityOutlined />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>

        {/* Data Source Info */}
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Data source: Live Azure subscription • Last updated: {new Date().toLocaleString()}
          </Typography>
        </Box>
      </Box>
    </>
  )
}

export default PoliciesPage