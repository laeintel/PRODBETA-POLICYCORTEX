import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Box,
  Badge,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  ListItemIcon,
  ListItemText,
  useTheme,
  alpha,
} from '@mui/material'
import {
  MenuOutlined,
  NotificationsOutlined,
  AccountCircleOutlined,
  SettingsOutlined,
  LogoutOutlined,
  DarkModeOutlined,
  LightModeOutlined,
  SearchOutlined,
} from '@mui/icons-material'
import { useState } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { useTheme as useAppTheme } from '@/hooks/useTheme'
import { useNotifications } from '@/hooks/useNotifications'
import { SearchBar } from '../UI/SearchBar'
import { NotificationPanel } from '../Notifications/NotificationPanel'
import { UserAvatar } from '../UI/UserAvatar'

interface HeaderProps {
  onMenuClick: () => void
}

export const Header = ({ onMenuClick }: HeaderProps) => {
  const theme = useTheme()
  const { theme: appTheme, toggleTheme } = useAppTheme() || { theme: 'light', toggleTheme: () => {} }
  const { user, logout } = useAuth()
  const { unreadCount } = useNotifications()
  
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null)
  const [notificationAnchor, setNotificationAnchor] = useState<null | HTMLElement>(null)
  const [searchOpen, setSearchOpen] = useState(false)

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget)
  }

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null)
  }

  const handleNotificationOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget)
  }

  const handleNotificationClose = () => {
    setNotificationAnchor(null)
  }

  const handleLogout = async () => {
    handleUserMenuClose()
    await logout()
  }

  const handleSettings = () => {
    handleUserMenuClose()
    // Navigate to settings page
    window.location.href = '/settings'
  }

  const handleProfile = () => {
    handleUserMenuClose()
    // Navigate to profile page
    window.location.href = '/profile'
  }

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        backgroundColor: theme.palette.background.paper,
        borderBottom: `1px solid ${theme.palette.divider}`,
        color: theme.palette.text.primary,
      }}
    >
      <Toolbar sx={{ px: { xs: 2, md: 3 } }}>
        {/* Menu Button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuOutlined />
        </IconButton>

        {/* Logo/Title */}
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <Typography
            variant="h6"
            component="div"
            sx={{
              fontWeight: 600,
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
              WebkitTextFillColor: 'transparent',
            }}
          >
            PolicyCortex
          </Typography>
        </Box>

        {/* Search Bar */}
        <Box sx={{ flexGrow: 1, mx: 3, display: { xs: 'none', md: 'block' } }}>
          <SearchBar />
        </Box>

        {/* Mobile Search Button */}
        <IconButton
          color="inherit"
          onClick={() => setSearchOpen(true)}
          sx={{ display: { xs: 'flex', md: 'none' }, mr: 1 }}
        >
          <SearchOutlined />
        </IconButton>

        {/* Actions */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Theme Toggle */}
          <IconButton
            color="inherit"
            onClick={toggleTheme}
            title={`Switch to ${appTheme === 'light' ? 'dark' : 'light'} mode`}
          >
            {appTheme === 'light' ? <DarkModeOutlined /> : <LightModeOutlined />}
          </IconButton>

          {/* Notifications */}
          <IconButton
            color="inherit"
            onClick={handleNotificationOpen}
            title="Notifications"
          >
            <Badge badgeContent={unreadCount} color="error">
              <NotificationsOutlined />
            </Badge>
          </IconButton>

          {/* User Menu */}
          <IconButton
            color="inherit"
            onClick={handleUserMenuOpen}
            title="User menu"
            sx={{ p: 0.5 }}
          >
            <UserAvatar
              user={user}
              size={32}
              sx={{
                border: `2px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                '&:hover': {
                  border: `2px solid ${theme.palette.primary.main}`,
                },
              }}
            />
          </IconButton>
        </Box>

        {/* User Menu */}
        <Menu
          anchorEl={userMenuAnchor}
          open={Boolean(userMenuAnchor)}
          onClose={handleUserMenuClose}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          PaperProps={{
            sx: {
              mt: 1,
              minWidth: 200,
              '& .MuiMenuItem-root': {
                px: 2,
                py: 1,
              },
            },
          }}
        >
          <Box sx={{ px: 2, py: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              {user?.displayName || user?.firstName || 'User'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {user?.email}
            </Typography>
          </Box>

          <Divider />

          <MenuItem onClick={handleProfile}>
            <ListItemIcon>
              <AccountCircleOutlined fontSize="small" />
            </ListItemIcon>
            <ListItemText>Profile</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleSettings}>
            <ListItemIcon>
              <SettingsOutlined fontSize="small" />
            </ListItemIcon>
            <ListItemText>Settings</ListItemText>
          </MenuItem>

          <Divider />

          <MenuItem onClick={handleLogout}>
            <ListItemIcon>
              <LogoutOutlined fontSize="small" />
            </ListItemIcon>
            <ListItemText>Logout</ListItemText>
          </MenuItem>
        </Menu>

        {/* Notification Panel */}
        <NotificationPanel
          anchorEl={notificationAnchor}
          open={Boolean(notificationAnchor)}
          onClose={handleNotificationClose}
        />
      </Toolbar>
    </AppBar>
  )
}