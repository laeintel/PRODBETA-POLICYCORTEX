import { Box, Typography, Paper } from '@mui/material'
import { AccountTreeOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ResourceTopologyPage = () => {
  return (
    <>
      <Helmet>
        <title>Resource Topology - PolicyCortex</title>
        <meta name="description" content="View resource topology and dependencies" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccountTreeOutlined />
          Resource Topology
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resource Topology
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain resource topology visualization.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ResourceTopologyPage