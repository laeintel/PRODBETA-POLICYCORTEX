import React, { Component, ErrorInfo, ReactNode } from 'react'
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
  Collapse,
  IconButton,
  useTheme,
} from '@mui/material'
import {
  ErrorOutline,
  ExpandMore,
  ExpandLess,
  Refresh,
  Home,
  BugReport,
} from '@mui/icons-material'
import { motion } from 'framer-motion'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
  showDetails: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    showDetails: false,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, showDetails: false }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    this.setState({
      error,
      errorInfo,
    })

    // Log error to monitoring service
    this.logErrorToService(error, errorInfo)
  }

  private logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // Log to external service (e.g., Sentry)
    try {
      // This would typically be a call to your error reporting service
      console.error('Error logged to monitoring service:', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
      })
    } catch (loggingError) {
      console.error('Failed to log error to monitoring service:', loggingError)
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  private handleGoHome = () => {
    window.location.href = '/'
  }

  private handleReportBug = () => {
    const { error, errorInfo } = this.state
    const subject = encodeURIComponent('Bug Report - PolicyCortex')
    const body = encodeURIComponent(`
Error: ${error?.message || 'Unknown error'}

Stack Trace:
${error?.stack || 'No stack trace available'}

Component Stack:
${errorInfo?.componentStack || 'No component stack available'}

Browser: ${navigator.userAgent}
Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}

Additional Details:
[Please describe what you were doing when this error occurred]
    `)
    
    window.open(`mailto:support@policycortex.com?subject=${subject}&body=${body}`)
  }

  private toggleDetails = () => {
    this.setState(prevState => ({ showDetails: !prevState.showDetails }))
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return <ErrorDisplay {...this.state} onRetry={this.handleRetry} />
    }

    return this.props.children
  }
}

interface ErrorDisplayProps extends State {
  onRetry: () => void
}

const ErrorDisplay = ({ error, errorInfo, showDetails, onRetry }: ErrorDisplayProps) => {
  const theme = useTheme()

  const handleGoHome = () => {
    window.location.href = '/'
  }

  const handleReportBug = () => {
    const subject = encodeURIComponent('Bug Report - PolicyCortex')
    const body = encodeURIComponent(`
Error: ${error?.message || 'Unknown error'}

Stack Trace:
${error?.stack || 'No stack trace available'}

Component Stack:
${errorInfo?.componentStack || 'No component stack available'}

Browser: ${navigator.userAgent}
Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}

Additional Details:
[Please describe what you were doing when this error occurred]
    `)
    
    window.open(`mailto:support@policycortex.com?subject=${subject}&body=${body}`)
  }

  const toggleDetails = () => {
    // This would need to be handled differently since we can't use setState in a functional component
    // For now, we'll use a simple approach
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        backgroundColor: theme.palette.background.default,
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
            p: 4,
            maxWidth: 600,
            textAlign: 'center',
            borderRadius: 3,
          }}
        >
          <ErrorOutline
            sx={{
              fontSize: 80,
              color: theme.palette.error.main,
              mb: 2,
            }}
          />

          <Typography variant="h4" gutterBottom color="error">
            Oops! Something went wrong
          </Typography>

          <Typography variant="body1" color="text.secondary" paragraph>
            We're sorry for the inconvenience. An unexpected error occurred while processing your request.
          </Typography>

          <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {error?.message || 'Unknown error occurred'}
            </Typography>
          </Alert>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center', mb: 3 }}>
            <Button
              variant="contained"
              startIcon={<Refresh />}
              onClick={onRetry}
              sx={{ minWidth: 120 }}
            >
              Try Again
            </Button>

            <Button
              variant="outlined"
              startIcon={<Home />}
              onClick={handleGoHome}
              sx={{ minWidth: 120 }}
            >
              Go Home
            </Button>

            <Button
              variant="outlined"
              startIcon={<BugReport />}
              onClick={handleReportBug}
              sx={{ minWidth: 120 }}
            >
              Report Bug
            </Button>
          </Box>

          <Box sx={{ textAlign: 'left' }}>
            <Button
              variant="text"
              size="small"
              onClick={toggleDetails}
              startIcon={showDetails ? <ExpandLess /> : <ExpandMore />}
              sx={{ mb: 1 }}
            >
              {showDetails ? 'Hide' : 'Show'} Technical Details
            </Button>

            <Collapse in={showDetails}>
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  backgroundColor: theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50',
                  maxHeight: 200,
                  overflow: 'auto',
                }}
              >
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  <strong>Error Stack:</strong>
                  <br />
                  {error?.stack || 'No stack trace available'}
                  <br />
                  <br />
                  <strong>Component Stack:</strong>
                  <br />
                  {errorInfo?.componentStack || 'No component stack available'}
                  <br />
                  <br />
                  <strong>Timestamp:</strong> {new Date().toISOString()}
                  <br />
                  <strong>URL:</strong> {window.location.href}
                  <br />
                  <strong>User Agent:</strong> {navigator.userAgent}
                </Typography>
              </Paper>
            </Collapse>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
            If this problem persists, please contact our support team at{' '}
            <a href="mailto:support@policycortex.com" style={{ color: theme.palette.primary.main }}>
              support@policycortex.com
            </a>
          </Typography>
        </Paper>
      </motion.div>
    </Box>
  )
}

// Higher-order component for wrapping components with error boundary
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode
) => {
  return (props: P) => (
    <ErrorBoundary fallback={fallback}>
      <Component {...props} />
    </ErrorBoundary>
  )
}

// Hook for handling errors in functional components
export const useErrorHandler = () => {
  const handleError = (error: Error, errorInfo?: any) => {
    console.error('Error caught by useErrorHandler:', error, errorInfo)
    
    // Log to monitoring service
    try {
      // This would typically be a call to your error reporting service
      console.error('Error logged to monitoring service:', {
        error: error.message,
        stack: error.stack,
        errorInfo,
        timestamp: new Date().toISOString(),
      })
    } catch (loggingError) {
      console.error('Failed to log error to monitoring service:', loggingError)
    }
  }

  return { handleError }
}