'use client'

import { ResourceDashboard } from '@/components/resources/ResourceDashboard'
import { Toaster } from 'react-hot-toast'

export default function ResourcesV2Page() {
  return (
    <>
      <ResourceDashboard />
      <Toaster 
        position="bottom-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </>
  )
}