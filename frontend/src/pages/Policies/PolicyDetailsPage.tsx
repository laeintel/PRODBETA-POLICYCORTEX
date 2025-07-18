import { Box, Typography, Paper } from '@mui/material'
import { PolicyOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const PolicyDetailsPage = () => {
  return (
    <>
      <Helmet>
        <title>Policy Details - PolicyCortex</title>
        <meta name="description" content="View policy details and compliance" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PolicyOutlined />
          Policy Details
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Policy Information
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain detailed policy information.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default PolicyDetailsPage