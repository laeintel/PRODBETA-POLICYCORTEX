import { Box, Typography, Paper } from '@mui/material'
import { ShieldOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const CompliancePage = () => {
  return (
    <>
      <Helmet>
        <title>Compliance - PolicyCortex</title>
        <meta name="description" content="View compliance status and reports" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShieldOutlined />
          Compliance
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Compliance Status
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain compliance status and reports.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default CompliancePage