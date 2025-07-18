import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { createTheme } from '@mui/material/styles'
import { HelmetProvider } from 'react-helmet-async'
import { MsalProvider } from '@azure/msal-react'
import { ThemeProvider as CustomThemeProvider } from '@/providers/ThemeProvider'
import { NotificationProvider } from '@/providers/NotificationProvider'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Mock MSAL instance (since MSAL is fully mocked in setup.ts)
const msalInstance = {} as any
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
})
const theme = createTheme()

interface AllTheProvidersProps {
  children: React.ReactNode
}

const AllTheProviders = ({ children }: AllTheProvidersProps) => {
  return (
    <HelmetProvider>
      <MsalProvider instance={msalInstance}>
        <BrowserRouter>
          <QueryClientProvider client={queryClient}>
            <ThemeProvider theme={theme}>
              <CustomThemeProvider>
                <NotificationProvider>
                  {children}
                </NotificationProvider>
              </CustomThemeProvider>
            </ThemeProvider>
          </QueryClientProvider>
        </BrowserRouter>
      </MsalProvider>
    </HelmetProvider>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything
export * from '@testing-library/react'
export { customRender as render }