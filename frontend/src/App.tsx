import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { CssBaseline, Box } from '@mui/material'
import { useIsAuthenticated } from '@azure/msal-react'
import { Helmet } from 'react-helmet-async'
import { useTheme as useAppTheme } from '@/hooks/useTheme'
import { useTheme } from '@mui/material/styles'
import { useAuth } from '@/hooks/useAuth'
import { AuthenticatedTemplate, UnauthenticatedTemplate } from '@azure/msal-react'
import { Layout } from '@/components/Layout/Layout'
import { LoginPage } from '@/pages/Auth/LoginPage'
import { LoadingScreen } from '@/components/UI/LoadingScreen'
import { ErrorBoundary } from '@/components/UI/ErrorBoundary'
import { WebSocketProvider } from '@/providers/WebSocketProvider'
import { NotificationProvider } from '@/providers/NotificationProvider'
import { AppRoutes } from '@/routes/AppRoutes'
import { useAuthStatus } from '@/hooks/useAuthStatus'

function App() {
  const isAuthenticated = useIsAuthenticated()
  const theme = useTheme()
  const { theme: appTheme } = useAppTheme() || { theme: 'light' }
  const { initialize, isLoading } = useAuth()
  const { isReady } = useAuthStatus()

  useEffect(() => {
    initialize()
  }, [initialize])

  if (isLoading || !isReady) {
    return <LoadingScreen />
  }

  return (
    <ErrorBoundary>
      <Helmet>
        <title>PolicyCortex - Azure Governance Intelligence</title>
        <meta name="description" content="AI-Powered Azure Governance Intelligence Platform" />
      </Helmet>
      
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          backgroundColor: theme.palette.background.default,
          color: theme.palette.text.primary,
        }}
      >
        <AuthenticatedTemplate>
          <NotificationProvider>
            <WebSocketProvider>
              <Layout>
                <AppRoutes />
              </Layout>
            </WebSocketProvider>
          </NotificationProvider>
        </AuthenticatedTemplate>

        <UnauthenticatedTemplate>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="*" element={<Navigate to="/login" replace />} />
          </Routes>
        </UnauthenticatedTemplate>
      </Box>
    </ErrorBoundary>
  )
}

export default App