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
  Alert,
  CircularProgress,
  LinearProgress,
  Stack,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar
} from '@mui/material'
import {
  AnalyticsOutlined,
  RefreshOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  AttachMoneyOutlined,
  SecurityOutlined,
  CloudOutlined,
  PolicyOutlined,
  AssessmentOutlined,
  TimelineOutlined,
  PieChartOutlined,
  BarChartOutlined,
  ShowChartOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface AnalyticsMetric {
  name: string
  value: number
  change: number
  trend: 'up' | 'down' | 'stable'
  period: string
}

interface AnalyticsData {
  summary: {
    totalResources: number
    totalCost: number
    complianceRate: number
    securityScore: number
    costTrend: number
    resourceTrend: number
  }
  metrics: AnalyticsMetric[]
  costAnalysis: {
    dailySpend: number
    monthlySpend: number
    projectedMonthlySpend: number
    topCostResources: Array<{
      name: string
      cost: number
      percentage: number
    }>
  }
  complianceAnalysis: {
    totalPolicies: number
    compliantResources: number
    nonCompliantResources: number
    topViolations: Array<{
      policy: string
      violations: number
    }>
  }
  usagePatterns: Array<{
    category: string
    usage: number
    efficiency: number
    recommendation: string
  }>
  insights: Array<{
    title: string
    description: string
    impact: 'high' | 'medium' | 'low'
    category: string
  }>
  data_source: string
}

const AnalyticsPage = () => {
  const navigate = useNavigate()  
  const { applyFilters } = useFilter()
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.get('/api/v1/analytics/overview').catch(() => null)
      
      if (response) {
        setAnalyticsData(response.data)
      } else {
        // Generate mock analytics data
        setAnalyticsData(generateMockAnalyticsData())
      }
    } catch (err: any) {
      console.error('Error fetching analytics data:', err)
      setError('Failed to load analytics data')
      setAnalyticsData(generateMockAnalyticsData())
    } finally {
      setLoading(false)
    }
  }

  const generateMockAnalyticsData = (): AnalyticsData => {
    return {
      summary: {
        totalResources: 73,
        totalCost: 2847.32,
        complianceRate: 78.5,
        securityScore: 82.3,
        costTrend: -5.2,
        resourceTrend: 8.7
      },
      metrics: [
        {
          name: 'Resource Utilization',
          value: 67.8,
          change: 4.2,
          trend: 'up',
          period: 'last 7 days'
        },
        {
          name: 'Cost Efficiency',
          value: 74.5,
          change: -2.1,
          trend: 'down',
          period: 'last 30 days'
        },
        {
          name: 'Security Compliance',
          value: 85.2,
          change: 3.7,
          trend: 'up',
          period: 'last 7 days'
        },
        {
          name: 'Policy Adherence',
          value: 91.4,
          change: 1.8,
          trend: 'up',
          period: 'last 30 days'
        }
      ],
      costAnalysis: {
        dailySpend: 94.91,
        monthlySpend: 2847.32,
        projectedMonthlySpend: 3156.78,
        topCostResources: [
          { name: 'Azure Kubernetes Service', cost: 1203.45, percentage: 42.3 },
          { name: 'Virtual Machines', cost: 876.23, percentage: 30.8 },
          { name: 'Storage Accounts', cost: 432.18, percentage: 15.2 },
          { name: 'Application Gateway', cost: 335.46, percentage: 11.7 }
        ]
      },
      complianceAnalysis: {
        totalPolicies: 28,
        compliantResources: 57,
        nonCompliantResources: 16,
        topViolations: [
          { policy: 'Storage account should use HTTPS', violations: 8 },
          { policy: 'VM should have disk encryption', violations: 5 },
          { policy: 'Network security groups should restrict access', violations: 3 }
        ]
      },
      usagePatterns: [
        {
          category: 'Compute',
          usage: 78.5,
          efficiency: 65.2,
          recommendation: 'Right-size underutilized VMs'
        },
        {
          category: 'Storage',
          usage: 45.8,
          efficiency: 82.1,
          recommendation: 'Archive infrequently accessed data'
        },
        {
          category: 'Network',
          usage: 34.2,
          efficiency: 71.6,
          recommendation: 'Optimize bandwidth allocation'
        }
      ],
      insights: [
        {
          title: 'Cost optimization opportunity identified',
          description: 'You can save approximately $456/month by right-sizing 12 underutilized virtual machines',
          impact: 'high',
          category: 'Cost Optimization'
        },
        {
          title: 'Security compliance improving',
          description: 'Security score increased by 3.7% this week due to implemented recommendations',
          impact: 'medium',
          category: 'Security'
        },
        {
          title: 'Resource sprawl detected',
          description: '8 orphaned resources found that are no longer needed and can be decommissioned',
          impact: 'medium',
          category: 'Resource Management'
        }
      ],
      data_source: 'mock-analytics-data'
    }
  }

  useEffect(() => {
    fetchAnalyticsData()
  }, [])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUpOutlined color="success" />
      case 'down': return <TrendingDownOutlined color="error" />
      default: return <ShowChartOutlined color="info" />
    }
  }

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up': return 'success'
      case 'down': return 'error'
      default: return 'info'
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'error'
      case 'medium': return 'warning'
      case 'low': return 'info'
      default: return 'default'
    }
  }

  const handleReportsClick = () => {
    navigate('/analytics/reports')
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
        <title>Analytics - PolicyCortex</title>
        <meta name="description" content="View analytics and insights" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AnalyticsOutlined />
            Analytics
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchAnalyticsData}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<AssessmentOutlined />}
              onClick={handleReportsClick}
            >
              Reports
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
          availableResourceGroups={['rg-policycortex-dev', 'rg-policycortex-prod', 'rg-policycortex-shared']}
          availableResourceTypes={['Analytics', 'Metrics', 'Reports', 'Insights']}
        />

        {analyticsData && (
          <>
            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <CloudOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{analyticsData?.summary?.totalResources || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Resources
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                          {getTrendIcon(analyticsData?.summary?.resourceTrend >= 0 ? 'up' : 'down')}
                          <Typography variant="caption" color={analyticsData?.summary?.resourceTrend >= 0 ? 'success.main' : 'error.main'}>
                            {Math.abs(analyticsData?.summary?.resourceTrend || 0)}% this month
                          </Typography>
                        </Box>
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
                        <Typography variant="h4">{formatCurrency(analyticsData?.summary?.totalCost || 0)}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Monthly Cost
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                          {getTrendIcon(analyticsData?.summary?.costTrend >= 0 ? 'up' : 'down')}
                          <Typography variant="caption" color={analyticsData?.summary?.costTrend >= 0 ? 'error.main' : 'success.main'}>
                            {Math.abs(analyticsData?.summary?.costTrend || 0)}% this month
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <PolicyOutlined color="info" />
                      <Box>
                        <Typography variant="h4">{analyticsData?.summary?.complianceRate || 0}%</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Compliance Rate
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={analyticsData?.summary?.complianceRate || 0} 
                      sx={{ mt: 2 }}
                      color={analyticsData?.summary?.complianceRate >= 80 ? 'success' : analyticsData?.summary?.complianceRate >= 60 ? 'warning' : 'error'}
                    />
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <SecurityOutlined color="warning" />
                      <Box>
                        <Typography variant="h4">{analyticsData?.summary?.securityScore || 0}%</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Security Score
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={analyticsData?.summary?.securityScore || 0} 
                      sx={{ mt: 2 }}
                      color={analyticsData?.summary?.securityScore >= 80 ? 'success' : analyticsData?.summary?.securityScore >= 60 ? 'warning' : 'error'}
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              {/* Key Metrics */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TimelineOutlined />
                    Key Metrics
                  </Typography>
                  <Stack spacing={2}>
                    {(analyticsData?.metrics || []).map((metric, index) => (
                      <Box key={index}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="subtitle2">{metric.name}</Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="h6">{metric.value}%</Typography>
                            <Chip
                              size="small"
                              icon={getTrendIcon(metric.trend)}
                              label={`${metric.change >= 0 ? '+' : ''}${metric.change}%`}
                              color={getTrendColor(metric.trend) as any}
                              variant="outlined"
                            />
                          </Box>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={metric.value} 
                          color={metric.value >= 80 ? 'success' : metric.value >= 60 ? 'warning' : 'error'}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          {metric.period}
                        </Typography>
                        {index < analyticsData.metrics.length - 1 && <Divider sx={{ mt: 2 }} />}
                      </Box>
                    ))}
                  </Stack>
                </Paper>
              </Grid>

              {/* Cost Analysis */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PieChartOutlined />
                    Cost Analysis
                  </Typography>
                  <Stack spacing={2}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Daily Spend:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {formatCurrency(analyticsData?.costAnalysis?.dailySpend || 0)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Monthly Spend:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {formatCurrency(analyticsData?.costAnalysis?.monthlySpend || 0)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Projected Monthly:</Typography>
                      <Typography variant="body2" fontWeight="medium" color="warning.main">
                        {formatCurrency(analyticsData?.costAnalysis?.projectedMonthlySpend || 0)}
                      </Typography>
                    </Box>
                    
                    <Divider />
                    
                    <Typography variant="subtitle2">Top Cost Resources</Typography>
                    {(analyticsData?.costAnalysis?.topCostResources || []).map((resource, index) => (
                      <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">{resource.name}</Typography>
                        <Box sx={{ textAlign: 'right' }}>
                          <Typography variant="body2" fontWeight="medium">
                            {formatCurrency(resource.cost)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {resource.percentage}%
                          </Typography>
                        </Box>
                      </Box>
                    ))}
                  </Stack>
                </Paper>
              </Grid>

              {/* Usage Patterns */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <BarChartOutlined />
                    Usage Patterns
                  </Typography>
                  <Stack spacing={3}>
                    {(analyticsData?.usagePatterns || []).map((pattern, index) => (
                      <Box key={index}>
                        <Typography variant="subtitle2" gutterBottom>{pattern.category}</Typography>
                        <Box sx={{ mb: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography variant="caption">Usage: {pattern.usage}%</Typography>
                            <Typography variant="caption">Efficiency: {pattern.efficiency}%</Typography>
                          </Box>
                          <LinearProgress variant="determinate" value={pattern.usage} sx={{ mb: 0.5 }} />
                          <LinearProgress variant="determinate" value={pattern.efficiency} color="success" />
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          💡 {pattern.recommendation}
                        </Typography>
                      </Box>
                    ))}
                  </Stack>
                </Paper>
              </Grid>

              {/* Insights */}
              <Grid item xs={12} lg={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>AI Insights</Typography>
                  <List dense>
                    {(analyticsData?.insights || []).map((insight, index) => (
                      <ListItem key={index} sx={{ px: 0 }}>
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: `${getImpactColor(insight.impact)}.main`, width: 32, height: 32 }}>
                            <AssessmentOutlined fontSize="small" />
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={insight.title}
                          secondary={insight.description}
                          primaryTypographyProps={{ variant: 'subtitle2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                        <Chip
                          label={insight.impact}
                          size="small"
                          color={getImpactColor(insight.impact) as any}
                          variant="outlined"
                        />
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              </Grid>
            </Grid>

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {analyticsData?.data_source || 'analytics-engine'} • Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default AnalyticsPage