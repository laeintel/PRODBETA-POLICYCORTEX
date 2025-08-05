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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Button,
  Stack,
  Divider
} from '@mui/material'
import {
  AttachMoneyOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  RefreshOutlined,
  RecommendOutlined,
  CloudOutlined,
  StorageOutlined,
  AccountTreeOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface CostData {
  current: {
    dailyCost: number
    monthlyCost: number
    currency: string
    billingPeriod: string
  }
  forecast: {
    nextMonthEstimate: number
    trend: string
    confidence: number
  }
  breakdown: {
    byService: Array<{
      service: string
      cost: number
      percentage: number
    }>
    byResourceGroup: Array<{
      resourceGroup: string
      cost: number
      percentage: number
    }>
    bySubscription?: Array<{
      subscription_id: string
      subscription_name: string
      total_cost: number
      currency: string
    }>
  }
  recommendations: Array<{
    type: string
    description: string
    estimatedSavings: number
    resource: string
  }>
  data_source: string
}

const CostsPage = () => {
  const navigate = useNavigate()
  const [costData, setCostData] = useState<CostData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCostData = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get('/api/v1/costs/overview')
      setCostData(response.data)
    } catch (err: any) {
      console.error('Error fetching cost data:', err)
      setError('Failed to load cost data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCostData()
  }, [])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
  }

  const getTrendIcon = (trend: string) => {
    switch (trend.toLowerCase()) {
      case 'increasing':
        return <TrendingUpOutlined color="error" />
      case 'decreasing':
        return <TrendingDownOutlined color="success" />
      default:
        return <TrendingUpOutlined color="info" />
    }
  }

  const getRecommendationColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'right-sizing':
        return 'warning'
      case 'reserved instances':
        return 'info'
      case 'storage optimization':
        return 'success'
      default:
        return 'primary'
    }
  }

  const handleServiceClick = (serviceName: string) => {
    navigate(`/costs/analysis?service=${encodeURIComponent(serviceName)}`)
  }

  const handleResourceGroupClick = (resourceGroupName: string) => {
    navigate(`/costs/analysis?resourceGroup=${encodeURIComponent(resourceGroupName)}`)
  }

  const handleSubscriptionClick = (subscriptionId: string) => {
    navigate(`/costs/analysis?subscription=${encodeURIComponent(subscriptionId)}`)
  }

  const handleViewAnalysis = () => {
    navigate('/costs/analysis')
  }

  const handleViewBudgets = () => {
    navigate('/costs/budgets')
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
        <title>Cost Management - PolicyCortex</title>
        <meta name="description" content="Manage Azure costs and budgets" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AttachMoneyOutlined />
            Cost Management
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshOutlined />}
            onClick={fetchCostData}
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

        {costData && (
          <>
            {/* Cost Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card 
                  sx={{ 
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                  onClick={handleViewAnalysis}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <AttachMoneyOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency(costData?.current?.monthlyCost || 0)}
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
                      <AttachMoneyOutlined color="secondary" />
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency(costData?.current?.dailyCost || 0)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Daily Average
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
                      {getTrendIcon(costData?.forecast?.trend || 'stable')}
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency(costData?.forecast?.nextMonthEstimate || 0)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Next Month Forecast
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card 
                  sx={{ 
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                  onClick={handleViewBudgets}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <RecommendOutlined color="info" />
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency((costData?.recommendations || []).reduce((sum, rec) => sum + rec.estimatedSavings, 0))}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Potential Savings
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              {/* Cost by Service */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ overflow: 'hidden' }}>
                  <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CloudOutlined />
                      Cost by Service
                    </Typography>
                  </Box>
                  <TableContainer sx={{ maxHeight: 400 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Service</TableCell>
                          <TableCell align="right">Cost</TableCell>
                          <TableCell align="right">%</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(costData?.breakdown?.byService || []).map((service, index) => (
                          <TableRow 
                            key={index} 
                            hover 
                            sx={{ 
                              cursor: 'pointer',
                              '&:hover': { backgroundColor: 'action.hover' }
                            }}
                            onClick={() => handleServiceClick(service.service)}
                          >
                            <TableCell>{service.service}</TableCell>
                            <TableCell align="right">
                              {formatCurrency(service.cost)}
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={service.percentage}
                                  sx={{ width: 60, height: 6 }}
                                />
                                {service.percentage}%
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>
              </Grid>

              {/* Cost by Resource Group */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ overflow: 'hidden' }}>
                  <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AccountTreeOutlined />
                      Cost by Resource Group
                    </Typography>
                  </Box>
                  <TableContainer sx={{ maxHeight: 400 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Resource Group</TableCell>
                          <TableCell align="right">Cost</TableCell>
                          <TableCell align="right">%</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(costData?.breakdown?.byResourceGroup || []).map((rg, index) => (
                          <TableRow 
                            key={index} 
                            hover 
                            sx={{ 
                              cursor: 'pointer',
                              '&:hover': { backgroundColor: 'action.hover' }
                            }}
                            onClick={() => handleResourceGroupClick(rg.resourceGroup)}
                          >
                            <TableCell>
                              <Typography variant="body2" noWrap>
                                {rg.resourceGroup}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              {formatCurrency(rg.cost)}
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={rg.percentage}
                                  sx={{ width: 60, height: 6 }}
                                />
                                {rg.percentage}%
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Paper>
              </Grid>

              {/* Cost Recommendations */}
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <RecommendOutlined />
                    Cost Optimization Recommendations
                  </Typography>
                  <Grid container spacing={2}>
                    {(costData?.recommendations || []).map((rec, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Stack spacing={2}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <Chip
                                  label={rec.type}
                                  color={getRecommendationColor(rec.type) as any}
                                  size="small"
                                />
                                <Typography variant="h6" color="success.main">
                                  {formatCurrency(rec.estimatedSavings)}
                                </Typography>
                              </Box>
                              <Typography variant="body2">
                                {rec.description}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Resource: {rec.resource}
                              </Typography>
                            </Stack>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Paper>
              </Grid>

              {/* Subscription Breakdown (if available) */}
              {costData?.breakdown?.bySubscription && (
                <Grid item xs={12}>
                  <Paper sx={{ overflow: 'hidden' }}>
                    <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <StorageOutlined />
                        Cost by Subscription
                      </Typography>
                    </Box>
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Subscription</TableCell>
                            <TableCell>ID</TableCell>
                            <TableCell align="right">Cost</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {(costData?.breakdown?.bySubscription || []).map((sub, index) => (
                            <TableRow 
                              key={index} 
                              hover 
                              sx={{ 
                                cursor: 'pointer',
                                '&:hover': { backgroundColor: 'action.hover' }
                              }}
                              onClick={() => handleSubscriptionClick(sub.subscription_id)}
                            >
                              <TableCell>{sub.subscription_name}</TableCell>
                              <TableCell>
                                <Typography variant="caption">
                                  {sub.subscription_id.substring(0, 8)}...
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                {formatCurrency(sub.total_cost, sub.currency)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Paper>
                </Grid>
              )}
            </Grid>

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {costData?.data_source || 'cost-management'} • Billing period: {costData?.current?.billingPeriod || 'Current month'}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default CostsPage