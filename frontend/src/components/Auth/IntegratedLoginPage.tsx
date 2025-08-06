/**
 * Enhanced Login Page with Zero-Configuration Authentication
 * Implements automatic organization detection and multi-method authentication
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material'
import {
  Email as EmailIcon,
  Business as BusinessIcon,
  Security as SecurityIcon,
  VpnKey as VpnKeyIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  CloudDone as CloudDoneIcon
} from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'
import useIntegratedAuth from '../../hooks/useIntegratedAuth'

const IntegratedLoginPage: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const {
    isAuthenticated,
    isLoading,
    error,
    organizationConfig,
    isDetectingOrganization,
    detectOrganization,
    login,
    clearError
  } = useIntegratedAuth()

  const [activeStep, setActiveStep] = useState(0)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [authCode, setAuthCode] = useState('')
  const [emailError, setEmailError] = useState('')

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      const from = location.state?.from?.pathname || '/'
      navigate(from, { replace: true })
    }
  }, [isAuthenticated, navigate, location])

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!email) {
      setEmailError('Email is required')
      return
    }

    if (!validateEmail(email)) {
      setEmailError('Please enter a valid email address')
      return
    }

    setEmailError('')
    clearError()

    try {
      const orgConfig = await detectOrganization(email)
      setActiveStep(1)
    } catch (e) {
      console.error('Organization detection failed:', e)
    }
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    clearError()

    if (!organizationConfig) {
      return
    }

    try {
      let loginRequest: any = { email }

      // Route to appropriate authentication method
      switch (organizationConfig.authentication_method) {
        case 'azure_ad':
          // In a real implementation, this would redirect to Azure AD
          // For demo, we'll simulate with auth_code
          loginRequest.auth_code = authCode || 'demo-azure-code'
          break
        case 'internal':
          if (!password) {
            setEmailError('Password is required')
            return
          }
          loginRequest.password = password
          break
        case 'saml':
          // SAML would typically involve redirects
          loginRequest.saml_response = 'demo-saml-response'
          break
        default:
          loginRequest.password = password || 'demo-password'
      }

      await login(loginRequest)
      
    } catch (e) {
      console.error('Login failed:', e)
    }
  }

  const handleBack = () => {
    setActiveStep(0)
    clearError()
  }

  const getOrganizationTypeColor = (type: string) => {
    switch (type) {
      case 'enterprise': return 'success'
      case 'professional': return 'primary'
      case 'starter': return 'secondary'
      case 'trial': return 'warning'
      default: return 'default'
    }
  }

  const getAuthMethodIcon = (method: string) => {
    switch (method) {
      case 'azure_ad': return <CloudDoneIcon />
      case 'saml': return <SecurityIcon />
      case 'oauth2': return <VpnKeyIcon />
      case 'ldap': return <BusinessIcon />
      default: return <VpnKeyIcon />
    }
  }

  const renderFeatureList = (features: Record<string, boolean>) => {
    const enabledFeatures = Object.entries(features)
      .filter(([_, enabled]) => enabled)
      .map(([feature, _]) => feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))

    return (
      <List dense>
        {enabledFeatures.slice(0, 6).map((feature, index) => (
          <ListItem key={index} sx={{ py: 0.5 }}>
            <ListItemIcon sx={{ minWidth: 30 }}>
              <CheckCircleIcon color="success" fontSize="small" />
            </ListItemIcon>
            <ListItemText primary={feature} />
          </ListItem>
        ))}
        {enabledFeatures.length > 6 && (
          <ListItem sx={{ py: 0.5 }}>
            <ListItemText 
              primary={`+${enabledFeatures.length - 6} more features`}
              sx={{ fontStyle: 'italic', color: 'text.secondary' }}
            />
          </ListItem>
        )}
      </List>
    )
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        p: 2
      }}
    >
      <Card
        sx={{
          maxWidth: 800,
          width: '100%',
          boxShadow: 3,
          borderRadius: 2
        }}
      >
        <CardContent sx={{ p: 4 }}>
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              Welcome to PolicyCortex
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Enterprise-grade policy compliance made simple
            </Typography>
          </Box>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={clearError}>
              {error}
            </Alert>
          )}

          <Stepper activeStep={activeStep} orientation="vertical">
            {/* Step 1: Email Detection */}
            <Step>
              <StepLabel>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <EmailIcon fontSize="small" />
                  Enter Your Email
                </Box>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  We'll automatically detect your organization and configure everything for you.
                </Typography>
                
                <form onSubmit={handleEmailSubmit}>
                  <TextField
                    fullWidth
                    type="email"
                    label="Work Email Address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    error={!!emailError}
                    helperText={emailError}
                    disabled={isDetectingOrganization}
                    sx={{ mb: 3 }}
                  />
                  
                  <Button
                    type="submit"
                    variant="contained"
                    fullWidth
                    size="large"
                    disabled={isDetectingOrganization || !email}
                    startIcon={isDetectingOrganization ? <CircularProgress size={20} /> : <BusinessIcon />}
                  >
                    {isDetectingOrganization ? 'Detecting Organization...' : 'Continue'}
                  </Button>
                </form>
              </StepContent>
            </Step>

            {/* Step 2: Organization Configuration & Authentication */}
            <Step>
              <StepLabel>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SecurityIcon fontSize="small" />
                  Sign In
                </Box>
              </StepLabel>
              <StepContent>
                {organizationConfig && (
                  <>
                    {/* Organization Info */}
                    <Paper elevation={1} sx={{ p: 3, mb: 3, bgcolor: 'grey.50' }}>
                      <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>
                            Organization Detected
                          </Typography>
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="body1" fontWeight="bold">
                              {organizationConfig.organization_name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {organizationConfig.domain}
                            </Typography>
                          </Box>
                          
                          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                            <Chip 
                              label={organizationConfig.organization_type}
                              color={getOrganizationTypeColor(organizationConfig.organization_type) as any}
                              size="small"
                            />
                            <Chip
                              label={organizationConfig.authentication_method.replace('_', ' ').toUpperCase()}
                              variant="outlined"
                              size="small"
                              icon={getAuthMethodIcon(organizationConfig.authentication_method)}
                            />
                          </Box>

                          {organizationConfig.sso_enabled && (
                            <Alert severity="info" sx={{ mb: 2 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <InfoIcon fontSize="small" />
                                Single Sign-On (SSO) is enabled for your organization
                              </Box>
                            </Alert>
                          )}
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>
                            Available Features
                          </Typography>
                          {renderFeatureList(organizationConfig.features)}
                        </Grid>
                      </Grid>
                    </Paper>

                    {/* Authentication Form */}
                    <form onSubmit={handleLogin}>
                      {organizationConfig.authentication_method === 'azure_ad' && (
                        <>
                          <Alert severity="info" sx={{ mb: 2 }}>
                            In production, you would be redirected to your organization's Azure AD login.
                            For this demo, click "Sign In with Azure AD" to simulate the process.
                          </Alert>
                          <TextField
                            fullWidth
                            label="Azure AD Auth Code (Demo)"
                            value={authCode}
                            onChange={(e) => setAuthCode(e.target.value)}
                            placeholder="Leave empty for demo"
                            sx={{ mb: 2 }}
                          />
                        </>
                      )}

                      {organizationConfig.authentication_method === 'internal' && (
                        <TextField
                          fullWidth
                          type="password"
                          label="Password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          required
                          sx={{ mb: 2 }}
                        />
                      )}

                      {organizationConfig.authentication_method === 'saml' && (
                        <Alert severity="info" sx={{ mb: 2 }}>
                          In production, you would be redirected to your organization's SAML identity provider.
                          For this demo, click "Sign In with SAML" to simulate the process.
                        </Alert>
                      )}

                      <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                          onClick={handleBack}
                          disabled={isLoading}
                        >
                          Back
                        </Button>
                        <Button
                          type="submit"
                          variant="contained"
                          size="large"
                          disabled={isLoading}
                          startIcon={isLoading ? <CircularProgress size={20} /> : getAuthMethodIcon(organizationConfig.authentication_method)}
                          sx={{ flex: 1 }}
                        >
                          {isLoading ? 'Signing In...' : `Sign In${organizationConfig.sso_enabled ? ' with SSO' : ''}`}
                        </Button>
                      </Box>
                    </form>
                  </>
                )}
              </StepContent>
            </Step>
          </Stepper>

          {/* Demo Information */}
          <Divider sx={{ my: 4 }} />
          <Alert severity="info">
            <Typography variant="body2">
              <strong>Demo Mode:</strong> This is a demonstration of zero-configuration authentication.
              Try emails like: admin@microsoft.com, user@google.com, test@startup.com, or trial@company.org
              to see different organization types and authentication methods.
            </Typography>
          </Alert>
        </CardContent>
      </Card>
    </Box>
  )
}

export default IntegratedLoginPage