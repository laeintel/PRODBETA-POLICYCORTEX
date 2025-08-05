import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
  Divider,
  Collapse,
  useTheme,
  alpha,
} from '@mui/material'
import {
  DashboardOutlined,
  PolicyOutlined,
  CloudOutlined,
  AttachMoneyOutlined,
  ChatOutlined,
  NotificationsOutlined,
  SettingsOutlined,
  ExpandLess,
  ExpandMore,
  AnalyticsOutlined,
  SecurityOutlined,
  ReportOutlined,
  AccountTreeOutlined,
} from '@mui/icons-material'
import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'
import { motion } from 'framer-motion'

interface SidebarProps {
  open: boolean
  width: number
  onToggle: () => void
  variant?: 'permanent' | 'temporary'
}

interface MenuItem {
  id: string
  label: string
  icon: React.ElementType
  path?: string
  children?: MenuItem[]
  permission?: string
  badge?: string | number
}

const menuItems: MenuItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: DashboardOutlined,
    path: '/dashboard',
  },
  {
    id: 'policies',
    label: 'Policies',
    icon: PolicyOutlined,
    children: [
      {
        id: 'policies-overview',
        label: 'Overview',
        icon: PolicyOutlined,
        path: '/policies',
      },
      {
        id: 'policies-create',
        label: 'Create Policy',
        icon: PolicyOutlined,
        path: '/policies/create',
        permission: 'policies:create',
      },
      {
        id: 'policies-templates',
        label: 'Templates',
        icon: PolicyOutlined,
        path: '/policies/templates',
      },
    ],
  },
  {
    id: 'resources',
    label: 'Resources',
    icon: CloudOutlined,
    children: [
      {
        id: 'resources-overview',
        label: 'Overview',
        icon: CloudOutlined,
        path: '/resources',
      },
      {
        id: 'resources-inventory',
        label: 'Inventory',
        icon: CloudOutlined,
        path: '/resources/inventory',
      },
      {
        id: 'resources-topology',
        label: 'Topology',
        icon: AccountTreeOutlined,
        path: '/resources/topology',
      },
    ],
  },
  {
    id: 'costs',
    label: 'Cost Management',
    icon: AttachMoneyOutlined,
    children: [
      {
        id: 'costs-overview',
        label: 'Overview',
        icon: AttachMoneyOutlined,
        path: '/costs',
      },
      {
        id: 'costs-analysis',
        label: 'Cost Analysis',
        icon: AnalyticsOutlined,
        path: '/costs/analysis',
      },
      {
        id: 'costs-budgets',
        label: 'Budgets',
        icon: AttachMoneyOutlined,
        path: '/costs/budgets',
      },
    ],
  },
  {
    id: 'conversation',
    label: 'AI Assistant',
    icon: ChatOutlined,
    path: '/conversation',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: AnalyticsOutlined,
    children: [
      {
        id: 'analytics-insights',
        label: 'Insights',
        icon: AnalyticsOutlined,
        path: '/analytics',
      },
      {
        id: 'analytics-reports',
        label: 'Reports',
        icon: ReportOutlined,
        path: '/analytics/reports',
      },
    ],
  },
  {
    id: 'security',
    label: 'Security',
    icon: SecurityOutlined,
    children: [
      {
        id: 'security-overview',
        label: 'Overview',
        icon: SecurityOutlined,
        path: '/security',
      },
      {
        id: 'security-compliance',
        label: 'Compliance',
        icon: SecurityOutlined,
        path: '/security/compliance',
      },
    ],
  },
  {
    id: 'rbac',
    label: 'RBAC',
    icon: SecurityOutlined,
    path: '/rbac',
  },
  {
    id: 'notifications',
    label: 'Notifications',
    icon: NotificationsOutlined,
    path: '/notifications',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: SettingsOutlined,
    path: '/settings',
  },
]

export const Sidebar = ({ open, width, onToggle, variant = 'permanent' }: SidebarProps) => {
  const theme = useTheme()
  const location = useLocation()
  const navigate = useNavigate()
  const { hasPermission } = useAuth()
  const [expandedItems, setExpandedItems] = useState<string[]>(['policies', 'resources', 'costs'])

  const handleItemClick = (item: MenuItem) => {
    if (item.children) {
      toggleExpanded(item.id)
    } else if (item.path) {
      navigate(item.path)
      if (variant === 'temporary') {
        onToggle()
      }
    }
  }

  const toggleExpanded = (itemId: string) => {
    setExpandedItems(prev =>
      prev.includes(itemId)
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    )
  }

  const isItemActive = (item: MenuItem) => {
    if (item.path) {
      return location.pathname === item.path || location.pathname.startsWith(item.path + '/')
    }
    return false
  }

  const hasItemPermission = (item: MenuItem) => {
    if (!item.permission) return true
    return hasPermission(item.permission)
  }

  const renderMenuItem = (item: MenuItem, level = 0) => {
    if (!hasItemPermission(item)) {
      return null
    }

    const isActive = isItemActive(item)
    const isExpanded = expandedItems.includes(item.id)
    const hasChildren = item.children && item.children.length > 0

    return (
      <Box key={item.id}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => handleItemClick(item)}
            sx={{
              pl: 2 + level * 2,
              py: 1,
              mx: 1,
              borderRadius: 1,
              backgroundColor: isActive ? alpha(theme.palette.primary.main, 0.1) : 'transparent',
              color: isActive ? theme.palette.primary.main : theme.palette.text.primary,
              '&:hover': {
                backgroundColor: isActive
                  ? alpha(theme.palette.primary.main, 0.15)
                  : alpha(theme.palette.primary.main, 0.05),
              },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 40,
                color: isActive ? theme.palette.primary.main : theme.palette.text.secondary,
              }}
            >
              <item.icon />
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              sx={{
                '& .MuiTypography-root': {
                  fontWeight: isActive ? 600 : 400,
                  fontSize: level > 0 ? '0.875rem' : '1rem',
                },
              }}
            />
            {item.badge && (
              <Box
                sx={{
                  backgroundColor: theme.palette.error.main,
                  color: 'white',
                  borderRadius: '50%',
                  minWidth: 20,
                  height: 20,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                }}
              >
                {item.badge}
              </Box>
            )}
            {hasChildren && (
              isExpanded ? <ExpandLess /> : <ExpandMore />
            )}
          </ListItemButton>
        </ListItem>

        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => renderMenuItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </Box>
    )
  }

  const drawerContent = (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: theme.palette.background.paper,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          minHeight: 64,
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mr: 2,
          }}
        >
          <Typography
            variant="h6"
            sx={{
              color: 'white',
              fontWeight: 'bold',
              fontSize: '1.25rem',
            }}
          >
            PC
          </Typography>
        </Box>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            color: theme.palette.text.primary,
          }}
        >
          PolicyCortex
        </Typography>
      </Box>

      {/* Navigation */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', py: 1 }}>
        <List>
          {menuItems.map(item => renderMenuItem(item))}
        </List>
      </Box>

      {/* Footer */}
      <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          v1.0.0
        </Typography>
      </Box>
    </Box>
  )

  return (
    <Drawer
      variant={variant}
      open={open}
      onClose={onToggle}
      sx={{
        width: width,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: width,
          boxSizing: 'border-box',
          border: 'none',
          boxShadow: theme.shadows[2],
        },
      }}
    >
      {drawerContent}
    </Drawer>
  )
}