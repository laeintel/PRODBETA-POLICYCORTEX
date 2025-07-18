import {
  Box,
  Typography,
  Link,
  useTheme,
  Container,
  Stack,
  Divider,
} from '@mui/material'
import { env } from '@/config/environment'

export const Footer = () => {
  const theme = useTheme()
  const currentYear = new Date().getFullYear()

  return (
    <Box
      component="footer"
      sx={{
        backgroundColor: theme.palette.background.paper,
        borderTop: `1px solid ${theme.palette.divider}`,
        py: 2,
        mt: 'auto',
      }}
    >
      <Container maxWidth="xl">
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          alignItems={{ xs: 'center', md: 'flex-start' }}
          justifyContent="space-between"
          spacing={2}
        >
          {/* Left side - Copyright */}
          <Box sx={{ textAlign: { xs: 'center', md: 'left' } }}>
            <Typography variant="body2" color="text.secondary">
              © {currentYear} PolicyCortex. All rights reserved.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              AI-Powered Azure Governance Intelligence Platform
            </Typography>
          </Box>

          {/* Right side - Links and Version */}
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            alignItems="center"
            spacing={2}
            sx={{ textAlign: { xs: 'center', md: 'right' } }}
          >
            <Stack direction="row" spacing={2}>
              <Link
                href="/privacy"
                color="text.secondary"
                variant="body2"
                sx={{ textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
              >
                Privacy Policy
              </Link>
              <Link
                href="/terms"
                color="text.secondary"
                variant="body2"
                sx={{ textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
              >
                Terms of Service
              </Link>
              <Link
                href="/support"
                color="text.secondary"
                variant="body2"
                sx={{ textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
              >
                Support
              </Link>
            </Stack>
            
            <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', sm: 'block' } }} />
            
            <Typography variant="body2" color="text.secondary">
              Version {env.APP_VERSION}
            </Typography>
          </Stack>
        </Stack>
      </Container>
    </Box>
  )
}