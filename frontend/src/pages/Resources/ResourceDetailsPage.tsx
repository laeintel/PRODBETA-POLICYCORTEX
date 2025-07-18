import { Box, Typography, Paper } from '@mui/material'
import { CloudOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ResourceDetailsPage = () => {
  return (
    <>
      <Helmet>
        <title>Resource Details - PolicyCortex</title>
        <meta name="description" content="View resource details and configuration" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CloudOutlined />
          Resource Details
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resource Information
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain detailed resource information.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ResourceDetailsPage