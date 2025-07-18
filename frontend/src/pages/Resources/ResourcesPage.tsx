import { Box, Typography, Paper } from '@mui/material'
import { CloudOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ResourcesPage = () => {
  return (
    <>
      <Helmet>
        <title>Resources - PolicyCortex</title>
        <meta name="description" content="Manage Azure resources and inventory" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CloudOutlined />
          Resources
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resource Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the resource management interface.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ResourcesPage