import { Box, Typography, Button, Paper, useTheme } from '@mui/material'
import { HomeOutlined, SearchOutlined } from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

const NotFoundPage = () => {
  const theme = useTheme()
  const navigate = useNavigate()

  const handleGoHome = () => {
    navigate('/dashboard')
  }

  const handleGoBack = () => {
    navigate(-1)
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 3,
        background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.background.paper} 100%)`,
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Paper
          elevation={8}
          sx={{
            p: 6,
            textAlign: 'center',
            maxWidth: 500,
            borderRadius: 3,
          }}
        >
          {/* 404 Illustration */}
          <Box
            sx={{
              width: 200,
              height: 200,
              margin: '0 auto',
              mb: 3,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}20 0%, ${theme.palette.primary.main}05 100%)`,
              borderRadius: '50%',
              border: `2px solid ${theme.palette.primary.main}30`,
            }}
          >
            <Typography
              variant="h1"
              sx={{
                fontSize: '4rem',
                fontWeight: 'bold',
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                WebkitTextFillColor: 'transparent',
              }}
            >
              404
            </Typography>
          </Box>

          <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
            Page Not Found
          </Typography>

          <Typography variant="body1" color="text.secondary" paragraph>
            The page you're looking for doesn't exist or has been moved.
          </Typography>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
            Don't worry, it happens to the best of us. Let's get you back on track.
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              startIcon={<HomeOutlined />}
              onClick={handleGoHome}
              sx={{
                minWidth: 140,
                borderRadius: 2,
                py: 1.5,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                '&:hover': {
                  background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
                },
              }}
            >
              Go Home
            </Button>

            <Button
              variant="outlined"
              onClick={handleGoBack}
              sx={{
                minWidth: 140,
                borderRadius: 2,
                py: 1.5,
              }}
            >
              Go Back
            </Button>
          </Box>

          <Box sx={{ mt: 4, pt: 3, borderTop: `1px solid ${theme.palette.divider}` }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Need help? Try these popular pages:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                size="small"
                variant="text"
                onClick={() => navigate('/dashboard')}
                sx={{ textTransform: 'none' }}
              >
                Dashboard
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => navigate('/policies')}
                sx={{ textTransform: 'none' }}
              >
                Policies
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => navigate('/resources')}
                sx={{ textTransform: 'none' }}
              >
                Resources
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => navigate('/costs')}
                sx={{ textTransform: 'none' }}
              >
                Costs
              </Button>
            </Box>
          </Box>
        </Paper>
      </motion.div>
    </Box>
  )
}

export default NotFoundPage