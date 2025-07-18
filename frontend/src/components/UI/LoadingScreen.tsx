import { Box, CircularProgress, Typography, useTheme } from '@mui/material'
import { motion } from 'framer-motion'

interface LoadingScreenProps {
  message?: string
  size?: number
  fullScreen?: boolean
}

export const LoadingScreen = ({ 
  message = 'Loading...', 
  size = 40, 
  fullScreen = true 
}: LoadingScreenProps) => {
  const theme = useTheme()

  const containerSx = fullScreen ? {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.palette.background.default,
    zIndex: 9999,
  } : {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '200px',
    flex: 1,
  }

  return (
    <Box sx={containerSx}>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          {/* Logo */}
          <Box
            sx={{
              width: 60,
              height: 60,
              borderRadius: '50%',
              background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 1,
              boxShadow: theme.shadows[4],
            }}
          >
            <Typography
              variant="h6"
              sx={{
                color: 'white',
                fontWeight: 'bold',
                fontSize: '1.25rem',
              }}
            >
              PC
            </Typography>
          </Box>

          {/* Loading Spinner */}
          <CircularProgress
            size={size}
            sx={{
              color: theme.palette.primary.main,
            }}
          />

          {/* Loading Message */}
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{
              fontWeight: 500,
              textAlign: 'center',
            }}
          >
            {message}
          </Typography>

          {/* Animated Dots */}
          <Box sx={{ display: 'flex', gap: 0.5, mt: 1 }}>
            {[0, 1, 2].map((index) => (
              <motion.div
                key={index}
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  delay: index * 0.2,
                }}
              >
                <Box
                  sx={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    backgroundColor: theme.palette.primary.main,
                  }}
                />
              </motion.div>
            ))}
          </Box>
        </Box>
      </motion.div>
    </Box>
  )
}