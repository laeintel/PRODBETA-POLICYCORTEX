import { Box, Typography, Paper } from '@mui/material'
import { NotificationsOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const NotificationsPage = () => {
  return (
    <>
      <Helmet>
        <title>Notifications - PolicyCortex</title>
        <meta name="description" content="View all notifications" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <NotificationsOutlined />
          Notifications
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Notification Center
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain all notifications and alerts.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default NotificationsPage