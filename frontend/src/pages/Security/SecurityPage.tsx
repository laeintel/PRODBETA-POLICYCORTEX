import { Box, Typography, Paper } from '@mui/material'
import { SecurityOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const SecurityPage = () => {
  return (
    <>
      <Helmet>
        <title>Security - PolicyCortex</title>
        <meta name="description" content="Security and compliance overview" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SecurityOutlined />
          Security
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Security Overview
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain security and compliance information.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default SecurityPage