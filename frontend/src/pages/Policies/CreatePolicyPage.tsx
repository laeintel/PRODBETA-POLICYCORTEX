import { Box, Typography, Paper } from '@mui/material'
import { AddOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const CreatePolicyPage = () => {
  return (
    <>
      <Helmet>
        <title>Create Policy - PolicyCortex</title>
        <meta name="description" content="Create a new Azure policy" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AddOutlined />
          Create Policy
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Policy Creation
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the policy creation form.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default CreatePolicyPage