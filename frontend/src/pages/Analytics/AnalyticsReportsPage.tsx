import { Box, Typography, Paper } from '@mui/material'
import { ReportOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const AnalyticsReportsPage = () => {
  return (
    <>
      <Helmet>
        <title>Analytics Reports - PolicyCortex</title>
        <meta name="description" content="View analytics reports and insights" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ReportOutlined />
          Analytics Reports
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Reports and Insights
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain analytics reports and insights.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default AnalyticsReportsPage