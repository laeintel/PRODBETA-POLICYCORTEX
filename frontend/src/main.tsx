import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { HelmetProvider } from 'react-helmet-async'
import { Toaster } from 'react-hot-toast'
import { MsalProvider } from '@azure/msal-react'
import { ThemeProvider } from '@/providers/ThemeProvider'
import { initializeMsal } from '@/utils/initializeMsal'
import App from './App'
import './index.css'

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        // Don't retry on 401 or 403 errors
        if ((error as any)?.response?.status === 401 || (error as any)?.response?.status === 403) {
          return false
        }
        return failureCount < 2
      },
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
    },
    mutations: {
      retry: false,
    },
  },
})

// Loading component
const LoadingApp = () => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh',
    backgroundColor: '#f5f5f5'
  }}>
    <div style={{ textAlign: 'center' }}>
      <h2>Initializing PolicyCortex...</h2>
      <p>Loading configuration...</p>
    </div>
  </div>
)

// Initialize and render app
const renderApp = async () => {
  const root = ReactDOM.createRoot(document.getElementById('root')!)
  
  // Show loading screen
  root.render(<LoadingApp />)
  
  try {
    // Initialize MSAL
    const msalInstance = await initializeMsal()
    
    // Render the app with MSAL
    root.render(
      <React.StrictMode>
        <MsalProvider instance={msalInstance}>
          <QueryClientProvider client={queryClient}>
            <HelmetProvider>
              <BrowserRouter>
                <ThemeProvider>
                  <App />
                  <Toaster
                    position="top-right"
                    toastOptions={{
                      duration: 4000,
                      style: {
                        background: '#363636',
                        color: '#fff',
                      },
                    }}
                  />
                </ThemeProvider>
              </BrowserRouter>
            </HelmetProvider>
            <ReactQueryDevtools initialIsOpen={false} />
          </QueryClientProvider>
        </MsalProvider>
      </React.StrictMode>
    )
  } catch (error) {
    console.error('Failed to initialize app:', error)
    root.render(
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        backgroundColor: '#f5f5f5',
        padding: '20px'
      }}>
        <div style={{ textAlign: 'center', maxWidth: '600px' }}>
          <h2 style={{ color: '#d32f2f' }}>Initialization Error</h2>
          <p>Failed to initialize the application. Please check the console for details.</p>
          <pre style={{ 
            textAlign: 'left', 
            background: '#f0f0f0', 
            padding: '10px',
            borderRadius: '4px',
            overflow: 'auto'
          }}>
            {error instanceof Error ? error.message : String(error)}
          </pre>
          <button 
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Reload Page
          </button>
        </div>
      </div>
    )
  }
}

// Start the app
renderApp()