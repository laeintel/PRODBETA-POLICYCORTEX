import { Box, Typography, Paper } from '@mui/material'
import { AttachMoneyOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const CostsPage = () => {
  return (
    <>
      <Helmet>
        <title>Cost Management - PolicyCortex</title>
        <meta name="description" content="Manage Azure costs and budgets" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AttachMoneyOutlined />
          Cost Management
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Cost Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the cost management interface.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default CostsPage