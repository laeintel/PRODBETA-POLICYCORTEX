import { Box, Typography, Paper } from '@mui/material'
import { PolicyOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const PolicyTemplatesPage = () => {
  return (
    <>
      <Helmet>
        <title>Policy Templates - PolicyCortex</title>
        <meta name="description" content="Browse and use policy templates" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PolicyOutlined />
          Policy Templates
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Template Library
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain policy templates.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default PolicyTemplatesPage