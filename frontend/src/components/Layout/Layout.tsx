import { useState, ReactNode } from 'react'
import { Box, useTheme, useMediaQuery } from '@mui/material'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { Footer } from './Footer'
import { useLayoutStore } from '@/store/layoutStore'

interface LayoutProps {
  children: ReactNode
}

export const Layout = ({ children }: LayoutProps) => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const { sidebarOpen, sidebarWidth, toggleSidebar } = useLayoutStore()

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        width={sidebarWidth}
        onToggle={toggleSidebar}
        variant={isMobile ? 'temporary' : 'permanent'}
      />

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          marginLeft: isMobile ? 0 : sidebarOpen ? `${sidebarWidth}px` : '0px',
          transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        {/* Header */}
        <Header onMenuClick={toggleSidebar} />

        {/* Page Content */}
        <Box
          sx={{
            flexGrow: 1,
            p: 3,
            backgroundColor: theme.palette.background.default,
            minHeight: 'calc(100vh - 64px - 60px)', // Subtract header and footer height
          }}
        >
          {children}
        </Box>

        {/* Footer */}
        <Footer />
      </Box>
    </Box>
  )
}