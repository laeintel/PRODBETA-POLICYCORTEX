import { Box, Typography, Paper } from '@mui/material'
import { PolicyOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const PoliciesPage = () => {
  return (
    <>
      <Helmet>
        <title>Policies - PolicyCortex</title>
        <meta name="description" content="Manage Azure policies and compliance" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PolicyOutlined />
          Policies
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Policy Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the policy management interface.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default PoliciesPage