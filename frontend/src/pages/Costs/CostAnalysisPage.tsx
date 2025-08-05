import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Divider,
  LinearProgress,
  ButtonGroup
} from '@mui/material'
import {
  AnalyticsOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  RefreshOutlined,
  FilterListOutlined,
  CloudOutlined,
  StorageOutlined,
  AccountTreeOutlined,
  AttachMoneyOutlined,
  DateRangeOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface CostData {
  historical_data: Array<{
    month: string
    date: string
    total_cost: number
    daily_average: number
    growth_rate: number
  }>
  trend_analysis: {
    overall_trend: string
    average_monthly_growth: number
    peak_month: any
    lowest_month: any
  }
  projections: {
    next_month: number
    next_quarter: number
    annual_projection: number
  }
  data_source: string
}

interface OverviewData {
  current: {
    dailyCost: number
    monthlyCost: number
    currency: string
    billingPeriod: string
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
}

const CostAnalysisPage = () => {
  const [searchParams] = useSearchParams()
  const [trendData, setTrendData] = useState<CostData | null>(null)
  const [overviewData, setOverviewData] = useState<OverviewData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [timeRange, setTimeRange] = useState('12months')
  const [viewType, setViewType] = useState('trends')

  // Get filter from URL params (when clicking from main cost page)
  const serviceFilter = searchParams.get('service')
  const resourceGroupFilter = searchParams.get('resourceGroup')
  const subscriptionFilter = searchParams.get('subscription')

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Fetch both trend data and overview data
      const [trendsResponse, overviewResponse] = await Promise.all([
        apiClient.get('/api/v1/costs/trends').catch(() => null),
        apiClient.get('/api/v1/costs/overview')
      ])
      
      if (trendsResponse?.data) {
        setTrendData(trendsResponse.data)
      } else {
        // Generate mock trend data if API not available
        console.log('Trends API not available, using mock data')
        setTrendData(generateMockTrendData())
      }
      
      setOverviewData(overviewResponse.data)
    } catch (err: any) {
      console.error('Error fetching cost analysis data:', err)
      setError('Failed to load cost analysis data')
      // Set mock data for demonstration
      if (!trendData) {
        setTrendData(generateMockTrendData())
      }
    } finally {
      setLoading(false)
    }
  }

  const generateMockTrendData = (): CostData => {
    const months = []
    const currentDate = new Date()
    
    for (let i = 11; i >= 0; i--) {
      const monthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() - i, 1)
      const baseCost = 1200 + (i * 30) + (Math.random() * 200)
      
      months.push({
        month: monthDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
        date: monthDate.toISOString().substring(0, 7),
        total_cost: Math.round(baseCost),
        daily_average: Math.round(baseCost / 30 * 100) / 100,
        growth_rate: i > 0 ? Math.round((Math.random() - 0.5) * 10 * 100) / 100 : 0
      })
    }
    
    return {
      historical_data: months,
      trend_analysis: {
        overall_trend: 'increasing',
        average_monthly_growth: 3.2,
        peak_month: months[months.length - 1],
        lowest_month: months[0]
      },
      projections: {
        next_month: months[months.length - 1].total_cost * 1.05,
        next_quarter: months.slice(-3).reduce((sum, m) => sum + m.total_cost, 0) * 1.1,
        annual_projection: months.reduce((sum, m) => sum + m.total_cost, 0) * 1.08
      },
      data_source: 'mock-trend-data'
    }
  }

  useEffect(() => {
    fetchData()
  }, [timeRange])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
  }

  const getTrendIcon = (trend: string) => {
    return trend === 'increasing' ? 
      <TrendingUpOutlined color="error" /> : 
      <TrendingDownOutlined color="success" />
  }

  const getFilteredData = () => {
    if (!overviewData) return null

    let filteredServices = overviewData.breakdown.byService
    let filteredResourceGroups = overviewData.breakdown.byResourceGroup

    if (serviceFilter) {
      filteredServices = filteredServices.filter(s => 
        s.service.toLowerCase().includes(serviceFilter.toLowerCase())
      )
    }

    if (resourceGroupFilter) {
      filteredResourceGroups = filteredResourceGroups.filter(rg => 
        rg.resourceGroup.toLowerCase().includes(resourceGroupFilter.toLowerCase())
      )
    }

    return { filteredServices, filteredResourceGroups }
  }

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  const filteredData = getFilteredData()

  return (
    <>
      <Helmet>
        <title>Cost Analysis - PolicyCortex</title>
        <meta name="description" content="Analyze Azure costs and trends" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AnalyticsOutlined />
            Cost Analysis
            {serviceFilter && (
              <Chip 
                label={`Service: ${serviceFilter}`} 
                color="primary" 
                size="small" 
                sx={{ ml: 2 }}
              />
            )}
            {resourceGroupFilter && (
              <Chip 
                label={`RG: ${resourceGroupFilter}`} 
                color="secondary" 
                size="small" 
                sx={{ ml: 1 }}
              />
            )}
          </Typography>
          <Stack direction="row" spacing={2}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <MenuItem value="3months">3 Months</MenuItem>
                <MenuItem value="6months">6 Months</MenuItem>
                <MenuItem value="12months">12 Months</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchData}
              disabled={loading}
            >
              Refresh
            </Button>
          </Stack>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* View Toggle */}
        <Box sx={{ mb: 3 }}>
          <ButtonGroup variant="outlined">
            <Button 
              variant={viewType === 'trends' ? 'contained' : 'outlined'}
              onClick={() => setViewType('trends')}
              startIcon={<TrendingUpOutlined />}
            >
              Trends
            </Button>
            <Button 
              variant={viewType === 'breakdown' ? 'contained' : 'outlined'}
              onClick={() => setViewType('breakdown')}
              startIcon={<FilterListOutlined />}
            >
              Breakdown
            </Button>
            <Button 
              variant={viewType === 'forecast' ? 'contained' : 'outlined'}
              onClick={() => setViewType('forecast')}
              startIcon={<DateRangeOutlined />}
            >
              Forecast
            </Button>
          </ButtonGroup>
        </Box>

        {/* Trends View */}
        {viewType === 'trends' && trendData && (
          <Grid container spacing={3}>
            {/* Trend Summary Cards */}
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {getTrendIcon(trendData?.trend_analysis?.overall_trend || 'stable')}
                    <Box>
                      <Typography variant="h6">
                        {trendData?.trend_analysis?.overall_trend ? 
                          (trendData?.trend_analysis?.overall_trend?.charAt(0).toUpperCase() + 
                           trendData?.trend_analysis?.overall_trend?.slice(1)) : 
                          'Stable'
                        }
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Overall Trend
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <TrendingUpOutlined color="info" />
                    <Box>
                      <Typography variant="h6">
                        {trendData?.trend_analysis?.average_monthly_growth || 0}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Avg Monthly Growth
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <AttachMoneyOutlined color="primary" />
                    <Box>
                      <Typography variant="h6">
                        {trendData?.trend_analysis?.peak_month ? 
                          formatCurrency(trendData?.trend_analysis?.peak_month?.total_cost || 0) : 
                          '$0.00'
                        }
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Peak Month Cost
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Historical Trend Table */}
            <Grid item xs={12}>
              <Paper>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6">Historical Cost Trends</Typography>
                </Box>
                <TableContainer sx={{ maxHeight: 500 }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Month</TableCell>
                        <TableCell align="right">Total Cost</TableCell>
                        <TableCell align="right">Daily Average</TableCell>
                        <TableCell align="right">Growth Rate</TableCell>
                        <TableCell align="right">Trend</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(trendData?.historical_data || []).map((month, index) => (
                        <TableRow key={index}>
                          <TableCell>{month.month}</TableCell>
                          <TableCell align="right">
                            {formatCurrency(month.total_cost)}
                          </TableCell>
                          <TableCell align="right">
                            {formatCurrency(month.daily_average)}
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`${month.growth_rate > 0 ? '+' : ''}${month.growth_rate}%`}
                              color={month.growth_rate > 0 ? 'error' : 'success'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell align="right">
                            <LinearProgress
                              variant="determinate"
                              value={Math.min(Math.abs(month.growth_rate) * 10, 100)}
                              color={month.growth_rate > 0 ? 'error' : 'success'}
                              sx={{ width: 80 }}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          </Grid>
        )}

        {/* Breakdown View */}
        {viewType === 'breakdown' && filteredData && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CloudOutlined />
                    Detailed Service Analysis
                  </Typography>
                </Box>
                <TableContainer sx={{ maxHeight: 500 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Service</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">% of Total</TableCell>
                        <TableCell align="right">Trend</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredData.filteredServices.map((service, index) => (
                        <TableRow key={index} hover>
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
                          <TableCell align="right">
                            <Chip
                              label="Stable"
                              color="info"
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AccountTreeOutlined />
                    Resource Group Analysis
                  </Typography>
                </Box>
                <TableContainer sx={{ maxHeight: 500 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Resource Group</TableCell>
                        <TableCell align="right">Cost</TableCell>
                        <TableCell align="right">% of Total</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredData.filteredResourceGroups.map((rg, index) => (
                        <TableRow key={index} hover>
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
          </Grid>
        )}

        {/* Forecast View */}
        {viewType === 'forecast' && trendData && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Next Month</Typography>
                  <Typography variant="h4" color="primary">
                    {formatCurrency(trendData?.projections?.next_month || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Projected cost for next month
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Next Quarter</Typography>
                  <Typography variant="h4" color="secondary">
                    {formatCurrency(trendData?.projections?.next_quarter || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Projected cost for next 3 months
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Annual Projection</Typography>
                  <Typography variant="h4" color="info">
                    {formatCurrency(trendData?.projections?.annual_projection || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Projected annual cost
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Data Source Info */}
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Data source: {trendData?.data_source || 'cost-analysis'} • 
            Last updated: {new Date().toLocaleString()}
          </Typography>
        </Box>
      </Box>
    </>
  )
}

export default CostAnalysisPage