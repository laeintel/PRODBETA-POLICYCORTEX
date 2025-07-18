import { Box, Typography, Paper } from '@mui/material'
import { AccountCircleOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ProfilePage = () => {
  return (
    <>
      <Helmet>
        <title>Profile - PolicyCortex</title>
        <meta name="description" content="User profile and account settings" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccountCircleOutlined />
          Profile
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            User Profile
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain user profile and account settings.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ProfilePage