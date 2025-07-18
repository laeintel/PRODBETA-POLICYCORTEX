import { Box, Typography, Paper } from '@mui/material'
import { InventoryOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ResourceInventoryPage = () => {
  return (
    <>
      <Helmet>
        <title>Resource Inventory - PolicyCortex</title>
        <meta name="description" content="View complete resource inventory" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <InventoryOutlined />
          Resource Inventory
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resource Inventory
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the complete resource inventory.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ResourceInventoryPage