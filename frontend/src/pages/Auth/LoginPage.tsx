import { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Container,
  Paper,
  useTheme,
  Stack,
} from '@mui/material'
import { LoginOutlined, DarkModeOutlined, LightModeOutlined } from '@mui/icons-material'
import { useAuth } from '@/hooks/useAuth'
import { useTheme as useAppTheme } from '@/hooks/useTheme'
import { Helmet } from 'react-helmet-async'
import { motion } from 'framer-motion'
import { env } from '@/config/environment'

const LoginPage = () => {
  console.log('🚀 LoginPage component rendering')
  const theme = useTheme()
  const { theme: appTheme, toggleTheme } = useAppTheme() || { theme: 'light', toggleTheme: () => {} }
  const { login, loginPopup, isLoading, error } = useAuth()
  const [loginMethod, setLoginMethod] = useState<'redirect' | 'popup'>('redirect')
  
  console.log('🚀 LoginPage state:', { loginMethod, isLoading, error })
  console.log('🚀 Auth functions available:', { 
    loginExists: !!login, 
    loginPopupExists: !!loginPopup 
  })

  const handleLogin = async () => {
    console.log('🚀 handleLogin called with method:', loginMethod)
    console.log('🚀 isLoading:', isLoading)
    console.log('🚀 login function:', login)
    console.log('🚀 loginPopup function:', loginPopup)
    try {
      if (loginMethod === 'popup') {
        console.log('🚀 Calling loginPopup...')
        await loginPopup()
      } else {
        console.log('🚀 Calling login (redirect)...')
        await login()
      }
    } catch (error) {
      console.error('🚀 Login error in handleLogin:', error)
    }
  }

  const handleThemeToggle = () => {
    toggleTheme()
  }

  return (
    <>
      <Helmet>
        <title>Login - PolicyCortex</title>
        <meta name="description" content="Login to PolicyCortex - AI-Powered Azure Governance Intelligence Platform" />
      </Helmet>

      <Container maxWidth="sm">
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            py: 4,
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ width: '100%' }}
          >
            <Card
              elevation={8}
              sx={{
                borderRadius: 3,
                background: theme.palette.mode === 'dark' 
                  ? 'linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%)'
                  : 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
                border: theme.palette.mode === 'dark' 
                  ? '1px solid rgba(255, 255, 255, 0.1)'
                  : '1px solid rgba(0, 0, 0, 0.05)',
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Box textAlign="center" mb={4}>
                  {/* Theme Toggle Button */}
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
                    <Button
                      size="small"
                      onClick={handleThemeToggle}
                      startIcon={appTheme === 'dark' ? <LightModeOutlined /> : <DarkModeOutlined />}
                      sx={{ minWidth: 'auto' }}
                    >
                      {appTheme === 'dark' ? 'Light' : 'Dark'}
                    </Button>
                  </Box>

                  {/* Logo/Brand */}
                  <Box
                    sx={{
                      width: 80,
                      height: 80,
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto',
                      mb: 3,
                      boxShadow: theme.shadows[8],
                    }}
                  >
                    <Typography
                      variant="h4"
                      sx={{
                        color: 'white',
                        fontWeight: 'bold',
                        fontSize: '2rem',
                      }}
                    >
                      PC
                    </Typography>
                  </Box>

                  <Typography
                    variant="h4"
                    gutterBottom
                    sx={{
                      fontWeight: 600,
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      color: 'transparent',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    PolicyCortex
                  </Typography>

                  <Typography
                    variant="body1"
                    color="text.secondary"
                    sx={{ mb: 3 }}
                  >
                    AI-Powered Azure Governance Intelligence Platform
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Secure login with Azure Active Directory
                  </Typography>
                </Box>

                {error && (
                  <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                  </Alert>
                )}

                <Stack spacing={2}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={(e) => {
                      console.log('🚀 Button click event triggered!', e)
                      handleLogin()
                    }}
                    disabled={isLoading}
                    startIcon={isLoading ? <CircularProgress size={20} /> : <LoginOutlined />}
                    sx={{
                      py: 1.5,
                      borderRadius: 2,
                      textTransform: 'none',
                      fontSize: '1rem',
                      fontWeight: 500,
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                      boxShadow: theme.shadows[4],
                      '&:hover': {
                        background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
                        boxShadow: theme.shadows[8],
                      },
                    }}
                  >
                    {isLoading ? 'Signing in...' : 'Sign in with Azure AD'}
                  </Button>

                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant={loginMethod === 'redirect' ? 'contained' : 'outlined'}
                      size="small"
                      onClick={() => setLoginMethod('redirect')}
                      sx={{ flex: 1, textTransform: 'none' }}
                    >
                      Redirect
                    </Button>
                    <Button
                      variant={loginMethod === 'popup' ? 'contained' : 'outlined'}
                      size="small"
                      onClick={() => setLoginMethod('popup')}
                      sx={{ flex: 1, textTransform: 'none' }}
                    >
                      Popup
                    </Button>
                  </Box>
                </Stack>

                <Box mt={4}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      backgroundColor: theme.palette.mode === 'dark' 
                        ? 'rgba(255, 255, 255, 0.02)'
                        : 'rgba(0, 0, 0, 0.02)',
                      borderRadius: 2,
                    }}
                  >
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong>Features:</strong>
                    </Typography>
                    <Typography variant="body2" color="text.secondary" component="ul" sx={{ pl: 2, m: 0 }}>
                      <li>Real-time Azure resource monitoring</li>
                      <li>AI-powered policy recommendations</li>
                      <li>Cost optimization insights</li>
                      <li>Compliance tracking & reporting</li>
                      <li>Interactive conversation with AI</li>
                    </Typography>
                  </Paper>
                </Box>

                <Box mt={3} textAlign="center">
                  <Typography variant="body2" color="text.secondary">
                    Version {env.APP_VERSION}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Box>
      </Container>
    </>
  )
}

export default LoginPage
export { LoginPage }