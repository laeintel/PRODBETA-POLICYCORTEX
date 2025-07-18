import { Box, Typography, Paper } from '@mui/material'
import { AttachMoneyOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const CostBudgetsPage = () => {
  return (
    <>
      <Helmet>
        <title>Cost Budgets - PolicyCortex</title>
        <meta name="description" content="Manage cost budgets and alerts" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AttachMoneyOutlined />
          Cost Budgets
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Budget Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain budget management features.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default CostBudgetsPage