import { Box, Typography, Paper } from '@mui/material'
import { AnalyticsOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const AnalyticsPage = () => {
  return (
    <>
      <Helmet>
        <title>Analytics - PolicyCortex</title>
        <meta name="description" content="View analytics and insights" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AnalyticsOutlined />
          Analytics
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Analytics Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain analytics and insights.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default AnalyticsPage