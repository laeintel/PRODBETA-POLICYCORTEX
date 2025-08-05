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
  GroupOutlined,
  PersonOutlined,
  AdminPanelSettingsOutlined,
  RefreshOutlined,
  VisibilityOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  SupervisorAccountOutlined,
  KeyOutlined,
  AssignmentIndOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface RoleAssignment {
  id: string
  displayName: string
  roleDefinitionName: string
  roleName: string
  scope: string
  principalType: 'User' | 'Group' | 'ServicePrincipal'
  principalName: string
  principalId: string
  assignmentType: 'Direct' | 'Inherited'
  createdOn: string
}

interface RBACData {
  roleAssignments: RoleAssignment[]
  summary: {
    totalAssignments: number
    uniqueUsers: number
    uniqueRoles: number
    inheritedAssignments: number
    directAssignments: number
    privilegedRoles: number
  }
  topRoles: Array<{
    roleName: string
    assignmentCount: number
  }>
  recentActivity: Array<{
    action: string
    principalName: string
    roleName: string
    timestamp: string
  }>
  data_source: string
}

const RBACPage = () => {
  const navigate = useNavigate()
  const { applyFilters } = useFilter()
  const [rbacData, setRbacData] = useState<RBACData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Apply filters to role assignments
  const filteredAssignments = useMemo(() => {
    if (!rbacData?.roleAssignments?.length) return []
    
    return applyFilters(rbacData.roleAssignments.map(assignment => ({
      ...assignment,
      subscription: assignment.scope?.split('/')[2] || '',
      resourceGroup: assignment.scope?.includes('/resourceGroups/') ? 
        assignment.scope.split('/resourceGroups/')[1]?.split('/')[0] || '' : '',
      type: assignment.principalType || '',
      location: 'Global' // RBAC assignments are typically global
    })))
  }, [rbacData?.roleAssignments, applyFilters])

  const fetchRBACData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.get('/api/v1/rbac/assignments').catch(() => null)
      
      if (response) {
        setRbacData(response.data)
      } else {
        // Generate mock RBAC data
        setRbacData(generateMockRBACData())
      }
    } catch (err: any) {
      console.error('Error fetching RBAC data:', err)
      setError('Failed to load RBAC data')
      setRbacData(generateMockRBACData())
    } finally {
      setLoading(false)
    }
  }

  const generateMockRBACData = (): RBACData => {
    const mockAssignments: RoleAssignment[] = [
      {
        id: 'rbac-001',
        displayName: 'John Doe',
        roleDefinitionName: 'Owner',
        roleName: 'Owner',
        scope: '/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595',
        principalType: 'User',
        principalName: 'john.doe@aeolitech.com',
        principalId: 'user-001',
        assignmentType: 'Direct',
        createdOn: '2024-01-15T10:30:00Z'
      },
      {
        id: 'rbac-002',
        displayName: 'DevOps Team',
        roleDefinitionName: 'Contributor',
        roleName: 'Contributor',
        scope: '/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev',
        principalType: 'Group',
        principalName: 'DevOps-Team',
        principalId: 'group-001',
        assignmentType: 'Direct',
        createdOn: '2024-02-10T14:20:00Z'
      },
      {
        id: 'rbac-003',
        displayName: 'PolicyCortex App',
        roleDefinitionName: 'Reader',
        roleName: 'Reader',
        scope: '/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-prod',
        principalType: 'ServicePrincipal',
        principalName: 'PolicyCortex-SP',
        principalId: 'sp-001',
        assignmentType: 'Direct',
        createdOn: '2024-03-05T09:15:00Z'
      },
      {
        id: 'rbac-004',
        displayName: 'Security Team',
        roleDefinitionName: 'Security Admin',
        roleName: 'Security Administrator',
        scope: '/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595',
        principalType: 'Group',
        principalName: 'Security-Team',
        principalId: 'group-002',
        assignmentType: 'Inherited',
        createdOn: '2024-01-20T16:45:00Z'
      }
    ]

    return {
      roleAssignments: mockAssignments,
      summary: {
        totalAssignments: mockAssignments.length,
        uniqueUsers: mockAssignments.filter(a => a.principalType === 'User').length,
        uniqueRoles: new Set(mockAssignments.map(a => a.roleName)).size,
        inheritedAssignments: mockAssignments.filter(a => a.assignmentType === 'Inherited').length,
        directAssignments: mockAssignments.filter(a => a.assignmentType === 'Direct').length,
        privilegedRoles: mockAssignments.filter(a => ['Owner', 'Contributor', 'Security Administrator'].includes(a.roleName)).length
      },
      topRoles: [
        { roleName: 'Reader', assignmentCount: 8 },
        { roleName: 'Contributor', assignmentCount: 5 },
        { roleName: 'Owner', assignmentCount: 2 },
        { roleName: 'Security Administrator', assignmentCount: 1 }
      ],
      recentActivity: [
        {
          action: 'Role Assigned',
          principalName: 'john.doe@aeolitech.com',
          roleName: 'Owner',
          timestamp: '2024-08-01T10:30:00Z'
        },
        {
          action: 'Role Removed',
          principalName: 'DevOps-Team',
          roleName: 'Storage Account Contributor',
          timestamp: '2024-07-28T15:20:00Z'
        }
      ],
      data_source: 'mock-rbac-data'
    }
  }

  useEffect(() => {
    fetchRBACData()
  }, [])

  const getPrincipalIcon = (type: string) => {
    switch (type) {
      case 'User': return <PersonOutlined />
      case 'Group': return <GroupOutlined />
      case 'ServicePrincipal': return <AdminPanelSettingsOutlined />
      default: return <PersonOutlined />
    }
  }

  const getRoleColor = (roleName: string) => {
    if (roleName.includes('Owner')) return 'error'
    if (roleName.includes('Contributor')) return 'warning'
    if (roleName.includes('Admin')) return 'secondary'
    return 'default'
  }

  const getAssignmentTypeColor = (type: string) => {
    return type === 'Direct' ? 'primary' : 'default'
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
        <title>RBAC - Role-Based Access Control - PolicyCortex</title>
        <meta name="description" content="Manage Azure role-based access control and permissions" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityOutlined />
            Role-Based Access Control
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshOutlined />}
            onClick={fetchRBACData}
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

        {/* Global Filter Panel */}
        <GlobalFilterPanel
          availableResourceGroups={rbacData?.roleAssignments?.map(a => a.scope?.includes('/resourceGroups/') ? 
            a.scope.split('/resourceGroups/')[1]?.split('/')[0] || '' : '').filter(Boolean) || []}
          availableResourceTypes={rbacData?.roleAssignments?.map(a => a.principalType).filter(Boolean) || []}
        />

        {rbacData && (
          <>
            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <AssignmentIndOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{rbacData?.summary?.totalAssignments || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Assignments
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
                      <PersonOutlined color="info" />
                      <Box>
                        <Typography variant="h4">{rbacData?.summary?.uniqueUsers || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Unique Users
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
                      <KeyOutlined color="secondary" />
                      <Box>
                        <Typography variant="h4">{rbacData?.summary?.uniqueRoles || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Unique Roles
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
                      <SupervisorAccountOutlined color="warning" />
                      <Box>
                        <Typography variant="h4">{rbacData?.summary?.privilegedRoles || 0}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Privileged Roles
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              {/* Role Assignments Table */}
              <Grid item xs={12} lg={8}>
                <Paper sx={{ overflow: 'hidden' }}>
                  <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                    <Typography variant="h6">Role Assignments</Typography>
                  </Box>
                  
                  <TableContainer sx={{ maxHeight: 500 }}>
                    <Table stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Principal</TableCell>
                          <TableCell>Role</TableCell>
                          <TableCell>Scope</TableCell>
                          <TableCell align="center">Type</TableCell>
                          <TableCell align="center">Assignment</TableCell>
                          <TableCell align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredAssignments.map((assignment) => (
                          <TableRow key={assignment.id} hover>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                                  {getPrincipalIcon(assignment.principalType)}
                                </Avatar>
                                <Box>
                                  <Typography variant="subtitle2">{assignment.displayName}</Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {assignment.principalName}
                                  </Typography>
                                </Box>
                              </Box>
                            </TableCell>
                            
                            <TableCell>
                              <Chip
                                label={assignment.roleName}
                                color={getRoleColor(assignment.roleName) as any}
                                size="small"
                              />
                            </TableCell>
                            
                            <TableCell>
                              <Typography variant="body2" sx={{ maxWidth: 200 }} noWrap>
                                {assignment.scope.split('/').slice(-2).join('/')}
                              </Typography>
                            </TableCell>
                            
                            <TableCell align="center">
                              <Chip
                                label={assignment.principalType}
                                variant="outlined"
                                size="small"
                              />
                            </TableCell>
                            
                            <TableCell align="center">
                              <Chip
                                label={assignment.assignmentType}
                                color={getAssignmentTypeColor(assignment.assignmentType) as any}
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

              {/* Sidebar with Top Roles and Recent Activity */}
              <Grid item xs={12} lg={4}>
                <Stack spacing={3}>
                  {/* Top Roles */}
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Top Roles</Typography>
                    <List dense>
                      {(rbacData?.topRoles || []).map((role, index) => (
                        <ListItem key={index}>
                          <ListItemAvatar>
                            <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32 }}>
                              <KeyOutlined />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={role.roleName}
                            secondary={`${role.assignmentCount} assignments`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>

                  {/* Recent Activity */}
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Recent Activity</Typography>
                    <List dense>
                      {(rbacData?.recentActivity || []).map((activity, index) => (
                        <ListItem key={index}>
                          <ListItemAvatar>
                            <Avatar sx={{ bgcolor: 'info.main', width: 32, height: 32 }}>
                              <SecurityOutlined />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={`${activity.action}: ${activity.roleName}`}
                            secondary={`${activity.principalName} • ${new Date(activity.timestamp).toLocaleDateString()}`}
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
                Data source: {rbacData?.data_source || 'rbac-management'} • Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default RBACPage