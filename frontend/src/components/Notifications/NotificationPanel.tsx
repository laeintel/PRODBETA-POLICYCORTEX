import {
  Menu,
  MenuProps,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Divider,
  Button,
  Chip,
  useTheme,
  alpha,
} from '@mui/material'
import {
  NotificationsOutlined,
  MarkEmailReadOutlined,
  DeleteOutlined,
  ErrorOutline,
  WarningOutlined,
  InfoOutlined,
  CheckCircleOutlined,
} from '@mui/icons-material'
import { useNotifications } from '@/hooks/useNotifications'
import { Notification } from '@/types'
import { formatDistanceToNow } from 'date-fns'

interface NotificationPanelProps extends Omit<MenuProps, 'children'> {
  onClose: () => void
}

export const NotificationPanel = ({ onClose, ...props }: NotificationPanelProps) => {
  const theme = useTheme()
  const {
    notifications,
    unreadCount,
    isLoading,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    getRecentNotifications,
  } = useNotifications()

  const recentNotifications = getRecentNotifications(10)

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <ErrorOutline color="error" />
      case 'warning':
        return <WarningOutlined color="warning" />
      case 'success':
        return <CheckCircleOutlined color="success" />
      default:
        return <InfoOutlined color="info" />
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return theme.palette.error.main
      case 'warning':
        return theme.palette.warning.main
      case 'success':
        return theme.palette.success.main
      default:
        return theme.palette.info.main
    }
  }

  const handleNotificationClick = (notification: Notification) => {
    if (!notification.isRead) {
      markAsRead(notification.id)
    }
    
    if (notification.actionUrl) {
      window.location.href = notification.actionUrl
    }
    
    onClose()
  }

  const handleMarkAllAsRead = () => {
    markAllAsRead()
  }

  const handleDeleteNotification = (notificationId: string, event: React.MouseEvent) => {
    event.stopPropagation()
    deleteNotification(notificationId)
  }

  return (
    <Menu
      {...props}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: 400,
          maxHeight: 500,
          '& .MuiList-root': {
            py: 0,
          },
        },
      }}
    >
      {/* Header */}
      <Box sx={{ px: 2, py: 1.5, backgroundColor: alpha(theme.palette.primary.main, 0.05) }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <NotificationsOutlined />
            Notifications
          </Typography>
          {unreadCount > 0 && (
            <Chip
              label={unreadCount}
              size="small"
              color="primary"
              sx={{ minWidth: 24, height: 20 }}
            />
          )}
        </Box>
        
        {unreadCount > 0 && (
          <Button
            size="small"
            startIcon={<MarkEmailReadOutlined />}
            onClick={handleMarkAllAsRead}
            sx={{ mt: 1 }}
          >
            Mark all as read
          </Button>
        )}
      </Box>

      <Divider />

      {/* Notifications List */}
      {isLoading ? (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Loading notifications...
          </Typography>
        </Box>
      ) : recentNotifications.length > 0 ? (
        <List sx={{ maxHeight: 400, overflow: 'auto' }}>
          {recentNotifications.map((notification) => (
            <ListItem
              key={notification.id}
              button
              onClick={() => handleNotificationClick(notification)}
              sx={{
                borderLeft: `4px solid ${getSeverityColor(notification.severity)}`,
                backgroundColor: notification.isRead 
                  ? 'transparent' 
                  : alpha(theme.palette.primary.main, 0.02),
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.05),
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {getSeverityIcon(notification.severity)}
              </ListItemIcon>
              
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: notification.isRead ? 400 : 600,
                        flex: 1,
                      }}
                    >
                      {notification.title}
                    </Typography>
                    {!notification.isRead && (
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: theme.palette.primary.main,
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {notification.message}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatDistanceToNow(new Date(notification.createdAt), { addSuffix: true })}
                    </Typography>
                  </Box>
                }
              />
              
              <IconButton
                size="small"
                onClick={(e) => handleDeleteNotification(notification.id, e)}
                sx={{ ml: 1 }}
              >
                <DeleteOutlined fontSize="small" />
              </IconButton>
            </ListItem>
          ))}
        </List>
      ) : (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <NotificationsOutlined sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          <Typography variant="body2" color="text.secondary">
            No notifications yet
          </Typography>
        </Box>
      )}

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 1 }}>
        <Button
          fullWidth
          variant="text"
          onClick={() => {
            onClose()
            window.location.href = '/notifications'
          }}
        >
          View All Notifications
        </Button>
      </Box>
    </Menu>
  )
}