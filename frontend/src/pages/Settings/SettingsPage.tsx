import { Box, Typography, Paper } from '@mui/material'
import { SettingsOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const SettingsPage = () => {
  return (
    <>
      <Helmet>
        <title>Settings - PolicyCortex</title>
        <meta name="description" content="Application settings and preferences" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsOutlined />
          Settings
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Application Settings
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain application settings and preferences.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default SettingsPage