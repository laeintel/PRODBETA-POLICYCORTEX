import { useState } from 'react'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  useTheme,
  Skeleton,
} from '@mui/material'
import {
  DashboardOutlined,
  PolicyOutlined,
  CloudOutlined,
  AttachMoneyOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  WarningOutlined,
  CheckCircleOutlined,
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { useQuery } from '@tanstack/react-query'
import { useAuth } from '@/hooks/useAuth'
import { motion } from 'framer-motion'

// Mock data for demonstration
const mockDashboardData = {
  overview: {
    totalResources: 1247,
    totalPolicies: 89,
    complianceScore: 87.5,
    monthlyCost: 15420.50,
    costTrend: 5.2,
    alertCount: 12,
    recommendations: 24,
  },
  recentAlerts: [
    {
      id: '1',
      title: 'High CPU Usage Detected',
      severity: 'warning',
      time: '2 minutes ago',
    },
    {
      id: '2',
      title: 'Storage Account Misconfigured',
      severity: 'error',
      time: '15 minutes ago',
    },
    {
      id: '3',
      title: 'Policy Compliance Improved',
      severity: 'success',
      time: '1 hour ago',
    },
  ],
}

const DashboardPage = () => {
  const theme = useTheme()
  const { user } = useAuth()

  // Mock query for dashboard data
  const { data: dashboardData, isLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: async () => {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      return mockDashboardData
    },
  })

  const StatCard = ({ title, value, icon, trend, color = 'primary' }: any) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card
        sx={{
          height: '100%',
          background: `linear-gradient(135deg, ${theme.palette[color].main}15 0%, ${theme.palette[color].main}05 100%)`,
          border: `1px solid ${theme.palette[color].main}30`,
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {title}
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: theme.palette[color].main }}>
                {isLoading ? <Skeleton width={100} /> : value}
              </Typography>
              {trend && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  {trend > 0 ? (
                    <TrendingUpOutlined color="success" sx={{ fontSize: 16, mr: 0.5 }} />
                  ) : (
                    <TrendingDownOutlined color="error" sx={{ fontSize: 16, mr: 0.5 }} />
                  )}
                  <Typography variant="body2" color={trend > 0 ? 'success.main' : 'error.main'}>
                    {Math.abs(trend)}%
                  </Typography>
                </Box>
              )}
            </Box>
            <Box
              sx={{
                width: 60,
                height: 60,
                borderRadius: '50%',
                backgroundColor: theme.palette[color].main,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
              }}
            >
              {icon}
            </Box>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  )

  const AlertCard = ({ alert }: any) => {
    const getSeverityColor = (severity: string) => {
      switch (severity) {
        case 'error': return 'error'
        case 'warning': return 'warning'
        case 'success': return 'success'
        default: return 'info'
      }
    }

    const getSeverityIcon = (severity: string) => {
      switch (severity) {
        case 'error': return <WarningOutlined />
        case 'warning': return <WarningOutlined />
        case 'success': return <CheckCircleOutlined />
        default: return <WarningOutlined />
      }
    }

    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          p: 2,
          borderLeft: `4px solid ${theme.palette[getSeverityColor(alert.severity)].main}`,
          backgroundColor: theme.palette.background.paper,
          mb: 1,
          borderRadius: 1,
        }}
      >
        <Box sx={{ mr: 2, color: theme.palette[getSeverityColor(alert.severity)].main }}>
          {getSeverityIcon(alert.severity)}
        </Box>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {alert.title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {alert.time}
          </Typography>
        </Box>
      </Box>
    )
  }

  return (
    <>
      <Helmet>
        <title>Dashboard - PolicyCortex</title>
        <meta name="description" content="PolicyCortex Dashboard - Overview of your Azure governance" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
            Welcome back, {user?.firstName || 'User'}!
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Here's an overview of your Azure governance and compliance status.
          </Typography>
        </Box>

        {/* Stats Grid */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Resources"
              value={dashboardData?.overview.totalResources.toLocaleString()}
              icon={<CloudOutlined />}
              color="primary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Active Policies"
              value={dashboardData?.overview.totalPolicies}
              icon={<PolicyOutlined />}
              color="secondary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Compliance Score"
              value={`${dashboardData?.overview.complianceScore}%`}
              icon={<CheckCircleOutlined />}
              color="success"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Monthly Cost"
              value={`$${dashboardData?.overview.monthlyCost.toLocaleString()}`}
              icon={<AttachMoneyOutlined />}
              trend={dashboardData?.overview.costTrend}
              color="warning"
            />
          </Grid>
        </Grid>

        {/* Content Grid */}
        <Grid container spacing={3}>
          {/* Recent Alerts */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <WarningOutlined sx={{ mr: 1 }} />
                    Recent Alerts
                  </Typography>
                  <Box>
                    {isLoading ? (
                      Array.from({ length: 3 }).map((_, index) => (
                        <Skeleton key={index} variant="rectangular" height={60} sx={{ mb: 1 }} />
                      ))
                    ) : (
                      dashboardData?.recentAlerts.map((alert) => (
                        <AlertCard key={alert.id} alert={alert} />
                      ))
                    )}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          {/* Quick Actions */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <DashboardOutlined sx={{ mr: 1 }} />
                    Quick Actions
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Paper
                        sx={{
                          p: 2,
                          textAlign: 'center',
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: theme.palette.action.hover,
                          },
                        }}
                      >
                        <PolicyOutlined sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                        <Typography variant="body2">Create Policy</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Paper
                        sx={{
                          p: 2,
                          textAlign: 'center',
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: theme.palette.action.hover,
                          },
                        }}
                      >
                        <CloudOutlined sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                        <Typography variant="body2">View Resources</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Paper
                        sx={{
                          p: 2,
                          textAlign: 'center',
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: theme.palette.action.hover,
                          },
                        }}
                      >
                        <AttachMoneyOutlined sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                        <Typography variant="body2">Cost Analysis</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Paper
                        sx={{
                          p: 2,
                          textAlign: 'center',
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: theme.palette.action.hover,
                          },
                        }}
                      >
                        <CheckCircleOutlined sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                        <Typography variant="body2">Compliance</Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      </Box>
    </>
  )
}

export default DashboardPage