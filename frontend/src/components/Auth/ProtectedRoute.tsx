import { ReactNode } from 'react'
import { Box, Typography, Paper, Button } from '@mui/material'
import { LockOutlined, HomeOutlined } from '@mui/icons-material'
import { useAuth } from '@/hooks/useAuth'
import { useNavigate } from 'react-router-dom'

interface ProtectedRouteProps {
  children: ReactNode
  permission?: string
  role?: string
  requireAll?: boolean
  fallback?: ReactNode
}

export const ProtectedRoute = ({
  children,
  permission,
  role,
  requireAll = false,
  fallback,
}: ProtectedRouteProps) => {
  const { isAuthenticated, hasPermission, hasRole } = useAuth()
  const navigate = useNavigate()

  // Check if user is authenticated
  if (!isAuthenticated) {
    return fallback || <div>Not authenticated</div>
  }

  // Check permissions
  const hasRequiredPermission = permission ? hasPermission(permission) : true
  const hasRequiredRole = role ? hasRole(role) : true

  let hasAccess = true

  if (requireAll) {
    // User must have all specified permissions and roles
    hasAccess = hasRequiredPermission && hasRequiredRole
  } else {
    // User must have at least one of the specified permissions or roles
    hasAccess = hasRequiredPermission || hasRequiredRole
  }

  if (!hasAccess) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          p: 3,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            textAlign: 'center',
            maxWidth: 400,
          }}
        >
          <LockOutlined
            sx={{
              fontSize: 64,
              color: 'error.main',
              mb: 2,
            }}
          />
          
          <Typography variant="h5" gutterBottom>
            Access Denied
          </Typography>
          
          <Typography variant="body1" color="text.secondary" paragraph>
            You don't have permission to access this resource.
          </Typography>
          
          {permission && (
            <Typography variant="body2" color="text.secondary" paragraph>
              Required permission: <strong>{permission}</strong>
            </Typography>
          )}
          
          {role && (
            <Typography variant="body2" color="text.secondary" paragraph>
              Required role: <strong>{role}</strong>
            </Typography>
          )}
          
          <Button
            variant="contained"
            startIcon={<HomeOutlined />}
            onClick={() => navigate('/dashboard')}
            sx={{ mt: 2 }}
          >
            Go to Dashboard
          </Button>
        </Paper>
      </Box>
    )
  }

  return <>{children}</>
}