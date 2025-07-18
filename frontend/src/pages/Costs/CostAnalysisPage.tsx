import { Box, Typography, Paper } from '@mui/material'
import { AnalyticsOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const CostAnalysisPage = () => {
  return (
    <>
      <Helmet>
        <title>Cost Analysis - PolicyCortex</title>
        <meta name="description" content="Analyze Azure costs and trends" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AnalyticsOutlined />
          Cost Analysis
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Cost Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain cost analysis and trends.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default CostAnalysisPage