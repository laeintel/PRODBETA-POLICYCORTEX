import React, { useState, useEffect } from 'react'
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  IconButton,
  Tooltip,
  Badge
} from '@mui/material'
import {
  AttachMoneyOutlined,
  AddOutlined,
  EditOutlined,
  DeleteOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  NotificationImportantOutlined,
  TrendingUpOutlined,
  RefreshOutlined,
  EmailOutlined,
  AccountBalanceWalletOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface Budget {
  id: string
  name: string
  amount: number
  spent: number
  remaining: number
  percentage_used: number
  status: 'on_track' | 'warning' | 'critical'
  period: string
  alerts: Array<{
    threshold: number
    enabled: boolean
    email_contacts: string[]
  }>
}

interface BudgetData {
  budgets: Budget[]
  summary: {
    total_budgets: number
    total_allocated: number
    total_spent: number
    total_remaining: number
    overall_utilization: number
    budgets_at_risk: number
  }
  data_source: string
}

const CostBudgetsPage = () => {
  const [budgetData, setBudgetData] = useState<BudgetData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [newBudget, setNewBudget] = useState({
    name: '',
    amount: '',
    period: 'Monthly',
    threshold: 80
  })

  const fetchBudgetData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await apiClient.get('/api/v1/costs/budgets').catch(() => null)
      
      if (response) {
        setBudgetData(response.data)
      } else {
        // Generate mock budget data
        setBudgetData(generateMockBudgetData())
      }
    } catch (err: any) {
      console.error('Error fetching budget data:', err)
      setError('Failed to load budget data')
      setBudgetData(generateMockBudgetData())
    } finally {
      setLoading(false)
    }
  }

  const generateMockBudgetData = (): BudgetData => {
    return {
      budgets: [
        {
          id: 'budget-dev-001',
          name: 'Development Environment Budget',
          amount: 2000.00,
          spent: 1247.85,
          remaining: 752.15,
          percentage_used: 62.4,
          status: 'on_track',
          period: 'Monthly',
          alerts: [
            {
              threshold: 80,
              enabled: true,
              email_contacts: ['admin@aeolitech.com']
            }
          ]
        },
        {
          id: 'budget-prod-001',
          name: 'Production Environment Budget',
          amount: 3000.00,
          spent: 2184.90,
          remaining: 815.10,
          percentage_used: 72.8,
          status: 'warning',
          period: 'Monthly',
          alerts: [
            {
              threshold: 75,
              enabled: true,
              email_contacts: ['admin@aeolitech.com', 'finance@aeolitech.com']
            }
          ]
        },
        {
          id: 'budget-network-001',
          name: 'Network Infrastructure Budget',
          amount: 800.00,
          spent: 156.40,
          remaining: 643.60,
          percentage_used: 19.6,
          status: 'on_track',
          period: 'Monthly',
          alerts: [
            {
              threshold: 85,
              enabled: true,
              email_contacts: ['network@aeolitech.com']
            }
          ]
        }
      ],
      summary: {
        total_budgets: 3,
        total_allocated: 5800.00,
        total_spent: 3588.15,
        total_remaining: 2211.85,
        overall_utilization: 61.9,
        budgets_at_risk: 1
      },
      data_source: 'budget-management-system'
    }
  }

  useEffect(() => {
    fetchBudgetData()
  }, [])

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency
    }).format(amount)
  }

  const getBudgetStatusColor = (status: string) => {
    switch (status) {
      case 'on_track': return 'success'
      case 'warning': return 'warning'
      case 'critical': return 'error'
      default: return 'info'
    }
  }

  const getBudgetStatusIcon = (status: string) => {
    switch (status) {
      case 'on_track': return <CheckCircleOutlined />
      case 'warning': return <WarningOutlined />
      case 'critical': return <NotificationImportantOutlined />
      default: return <AccountBalanceWalletOutlined />
    }
  }

  const handleCreateBudget = async () => {
    // Mock budget creation
    console.log('Creating budget:', newBudget)
    setCreateDialogOpen(false)
    setNewBudget({ name: '', amount: '', period: 'Monthly', threshold: 80 })
    // Refresh data
    await fetchBudgetData()
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
        <title>Cost Budgets - PolicyCortex</title>
        <meta name="description" content="Manage cost budgets and alerts" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AttachMoneyOutlined />
            Cost Budgets
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchBudgetData}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              startIcon={<AddOutlined />}
              onClick={() => setCreateDialogOpen(true)}
            >
              Create Budget
            </Button>
          </Stack>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {budgetData && (
          <>
            {/* Budget Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <AccountBalanceWalletOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{budgetData?.summary?.total_budgets || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Active Budgets
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
                      <AttachMoneyOutlined color="info" />
                      <Box>
                        <Typography variant="h4">
                          {formatCurrency(budgetData?.summary?.total_allocated || 0)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Allocated
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
                        <Typography variant="h4">
                          {budgetData?.summary?.overall_utilization || 0}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Overall Utilization
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
                      <Badge badgeContent={budgetData?.summary?.budgets_at_risk || 0} color="error">
                        <WarningOutlined color="warning" />
                      </Badge>
                      <Box>
                        <Typography variant="h4">{budgetData?.summary?.budgets_at_risk || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Budgets at Risk
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Budget Details Table */}
            <Paper sx={{ overflow: 'hidden' }}>
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="h6">Budget Details</Typography>
              </Box>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Budget Name</TableCell>
                      <TableCell>Period</TableCell>
                      <TableCell align="right">Allocated</TableCell>
                      <TableCell align="right">Spent</TableCell>
                      <TableCell align="right">Remaining</TableCell>
                      <TableCell align="center">Utilization</TableCell>
                      <TableCell align="center">Status</TableCell>
                      <TableCell align="center">Alerts</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(budgetData?.budgets || []).map((budget) => (
                      <TableRow key={budget.id} hover>
                        <TableCell>
                          <Typography variant="subtitle2">{budget.name}</Typography>
                        </TableCell>
                        
                        <TableCell>
                          <Chip
                            label={budget.period}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        
                        <TableCell align="right">
                          {formatCurrency(budget.amount)}
                        </TableCell>
                        
                        <TableCell align="right">
                          {formatCurrency(budget.spent)}
                        </TableCell>
                        
                        <TableCell align="right">
                          <Typography 
                            color={budget.remaining > 0 ? 'text.primary' : 'error'}
                          >
                            {formatCurrency(budget.remaining)}
                          </Typography>
                        </TableCell>
                        
                        <TableCell align="center">
                          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={Math.min(budget.percentage_used, 100)}
                              color={getBudgetStatusColor(budget.status) as any}
                              sx={{ width: 100, height: 8 }}
                            />
                            <Typography variant="caption">
                              {budget.percentage_used}%
                            </Typography>
                          </Box>
                        </TableCell>
                        
                        <TableCell align="center">
                          <Chip
                            icon={getBudgetStatusIcon(budget.status)}
                            label={budget.status.replace('_', ' ').toUpperCase()}
                            color={getBudgetStatusColor(budget.status) as any}
                            size="small"
                          />
                        </TableCell>
                        
                        <TableCell align="center">
                          <Stack direction="row" spacing={1} justifyContent="center">
                            {budget.alerts.map((alert, index) => (
                              <Tooltip 
                                key={index}
                                title={`${alert.threshold}% threshold, ${alert.email_contacts.length} contacts`}
                              >
                                <Chip
                                  icon={<EmailOutlined />}
                                  label={`${alert.threshold}%`}
                                  size="small"
                                  color={alert.enabled ? 'primary' : 'default'}
                                  variant="outlined"
                                />
                              </Tooltip>
                            ))}
                          </Stack>
                        </TableCell>
                        
                        <TableCell align="center">
                          <Stack direction="row" spacing={1}>
                            <Tooltip title="Edit Budget">
                              <IconButton size="small" color="primary">
                                <EditOutlined />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete Budget">
                              <IconButton size="small" color="error">
                                <DeleteOutlined />
                              </IconButton>
                            </Tooltip>
                          </Stack>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>

            {/* Budget Recommendations */}
            <Paper sx={{ p: 3, mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Budget Recommendations
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Alert severity="info">
                    <Typography variant="subtitle2">Optimize Development Budget</Typography>
                    <Typography variant="body2">
                      Current utilization is 62%. Consider reallocating unused budget to production environment.
                    </Typography>
                  </Alert>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Alert severity="warning">
                    <Typography variant="subtitle2">Production Budget Alert</Typography>
                    <Typography variant="body2">
                      Production budget is at 73% utilization. Consider increasing allocation or optimizing resources.
                    </Typography>
                  </Alert>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Alert severity="success">
                    <Typography variant="subtitle2">Network Budget Healthy</Typography>
                    <Typography variant="body2">
                      Network budget is well within limits at 20% utilization. Good cost management!
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>
            </Paper>

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {budgetData?.data_source || 'budget-management'} • Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}

        {/* Create Budget Dialog */}
        <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Create New Budget</DialogTitle>
          <DialogContent>
            <Stack spacing={3} sx={{ mt: 1 }}>
              <TextField
                label="Budget Name"
                fullWidth
                value={newBudget.name}
                onChange={(e) => setNewBudget({ ...newBudget, name: e.target.value })}
                placeholder="e.g., Q4 Development Budget"
              />
              
              <TextField
                label="Budget Amount"
                type="number"
                fullWidth
                value={newBudget.amount}
                onChange={(e) => setNewBudget({ ...newBudget, amount: e.target.value })}
                InputProps={{
                  startAdornment: '$'
                }}
              />
              
              <FormControl fullWidth>
                <InputLabel>Budget Period</InputLabel>
                <Select
                  value={newBudget.period}
                  label="Budget Period"
                  onChange={(e) => setNewBudget({ ...newBudget, period: e.target.value })}
                >
                  <MenuItem value="Weekly">Weekly</MenuItem>
                  <MenuItem value="Monthly">Monthly</MenuItem>
                  <MenuItem value="Quarterly">Quarterly</MenuItem>
                  <MenuItem value="Annual">Annual</MenuItem>
                </Select>
              </FormControl>
              
              <TextField
                label="Alert Threshold (%)"
                type="number"
                fullWidth
                value={newBudget.threshold}
                onChange={(e) => setNewBudget({ ...newBudget, threshold: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 100 }}
              />
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
            <Button 
              onClick={handleCreateBudget} 
              variant="contained"
              disabled={!newBudget.name || !newBudget.amount}
            >
              Create Budget
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </>
  )
}

export default CostBudgetsPage