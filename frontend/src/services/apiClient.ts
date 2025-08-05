import axios, { AxiosInstance, AxiosResponse, AxiosError, InternalAxiosRequestConfig } from 'axios'
import { apiConfig, httpStatus } from '@/config/api'
import { ApiResponse, ApiError } from '@/types'
import { env } from '@/config/environment'
import toast from 'react-hot-toast'

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: apiConfig.baseURL,
  timeout: apiConfig.timeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Add request ID for tracking
    config.headers['X-Request-ID'] = crypto.randomUUID()
    
    // Add timestamp
    config.headers['X-Timestamp'] = new Date().toISOString()
    
    // Add client info
    config.headers['X-Client-Version'] = env.APP_VERSION
    
    // Log request in development
    if (env.NODE_ENV === 'development') {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data)
    }
    
    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response in development
    if (env.NODE_ENV === 'development') {
      console.log(`[API] ${response.status} ${response.config.url}`, response.data)
    }
    
    return response
  },
  (error: AxiosError) => {
    const { response, config } = error
    
    // Log error in development (less verbose for expected 404s)
    if (env.NODE_ENV === 'development') {
      const developmentEndpoints = ['/api/v1/costs/trends', '/api/v1/costs/budgets', '/api/v1/costs/details', '/api/v1/rbac/assignments']
      const isDevEndpoint = developmentEndpoints.some(endpoint => config?.url?.includes(endpoint))
      
      if (response?.status === 404 && isDevEndpoint) {
        console.log(`[API] ${response.status} ${config?.url} (expected - using fallback data)`)
      } else {
        console.error(`[API] ${response?.status} ${config?.url}`, error)
      }
    }
    
    // Handle different error types
    if (response) {
      const { status, data } = response
      
      switch (status) {
        case httpStatus.UNAUTHORIZED:
          // Handle unauthorized access
          handleUnauthorizedError()
          break
          
        case httpStatus.FORBIDDEN:
          // Handle forbidden access
          handleForbiddenError()
          break
          
        case httpStatus.NOT_FOUND:
          // Handle not found
          handleNotFoundError(config?.url || '')
          break
          
        case httpStatus.UNPROCESSABLE_ENTITY:
          // Handle validation errors
          handleValidationError(data)
          break
          
        case httpStatus.TOO_MANY_REQUESTS:
          // Handle rate limiting
          handleRateLimitError()
          break
          
        case httpStatus.INTERNAL_SERVER_ERROR:
          // Handle server errors
          handleServerError()
          break
          
        case httpStatus.BAD_GATEWAY:
        case httpStatus.SERVICE_UNAVAILABLE:
        case httpStatus.GATEWAY_TIMEOUT:
          // Handle service unavailable
          handleServiceUnavailableError()
          break
          
        default:
          // Handle other errors
          handleGenericError(status, data)
      }
    } else if (error.code === 'ECONNABORTED') {
      // Handle timeout
      handleTimeoutError()
    } else if (error.code === 'ERR_NETWORK') {
      // Handle network errors
      handleNetworkError()
    } else {
      // Handle other errors
      handleGenericError(0, error.message)
    }
    
    return Promise.reject(createApiError(error))
  }
)

// Error handlers
const handleUnauthorizedError = () => {
  toast.error('Your session has expired. Please log in again.')
  // Redirect to login or trigger logout
  window.location.href = '/login'
}

const handleForbiddenError = () => {
  toast.error('You do not have permission to access this resource.')
}

const handleNotFoundError = (url: string) => {
  // Suppress toast notifications and console warnings for known development endpoints
  const developmentEndpoints = ['/api/v1/costs/trends', '/api/v1/costs/budgets', '/api/v1/costs/details', '/api/v1/rbac/assignments']
  const isDevEndpoint = developmentEndpoints.some(endpoint => url?.includes(endpoint))
  
  if (!isDevEndpoint) {
    toast.error('The requested resource was not found.')
    console.warn(`[API] Not found: ${url}`)
  }
}

const handleValidationError = (data: any) => {
  const message = data?.message || 'Please check your input and try again.'
  toast.error(message)
}

const handleRateLimitError = () => {
  toast.error('Too many requests. Please wait a moment and try again.')
}

const handleServerError = () => {
  toast.error('A server error occurred. Please try again later.')
}

const handleServiceUnavailableError = () => {
  toast.error('Service is temporarily unavailable. Please try again later.')
}

const handleTimeoutError = () => {
  toast.error('Request timed out. Please check your connection and try again.')
}

const handleNetworkError = () => {
  toast.error('Network error. Please check your connection and try again.')
}

const handleGenericError = (status: number, data: any) => {
  const message = data?.message || data || 'An unexpected error occurred.'
  toast.error(message)
  console.error(`[API] Error ${status}:`, message)
}

// Create standardized API error
const createApiError = (error: AxiosError): ApiError => {
  const { response } = error
  const errorData = response?.data as any
  const errorInfo = error as any
  
  return {
    message: errorData?.message || errorInfo?.message || 'An unexpected error occurred',
    code: errorData?.code || errorInfo?.code || 'UNKNOWN_ERROR',
    details: errorData?.details || errorInfo?.details || {},
    timestamp: new Date().toISOString(),
  }
}

// Helper functions for common operations
export const apiHelpers = {
  // Get with error handling
  async get<T>(url: string, config?: any): Promise<T> {
    const response = await apiClient.get<T>(url, config)
    return response.data
  },

  // Post with error handling
  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await apiClient.post<T>(url, data, config)
    return response.data
  },

  // Put with error handling
  async put<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await apiClient.put<T>(url, data, config)
    return response.data
  },

  // Patch with error handling
  async patch<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await apiClient.patch<T>(url, data, config)
    return response.data
  },

  // Delete with error handling
  async delete<T>(url: string, config?: any): Promise<T> {
    const response = await apiClient.delete<T>(url, config)
    return response.data
  },

  // Upload file
  async uploadFile<T>(url: string, file: File, onProgress?: (progress: number) => void): Promise<T> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await apiClient.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })

    return response.data
  },

  // Download file
  async downloadFile(url: string, filename?: string): Promise<void> {
    const response = await apiClient.get(url, {
      responseType: 'blob',
    })

    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  },
}

// Add token to requests
export const setAuthToken = (token: string | null) => {
  if (token) {
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`
  } else {
    delete apiClient.defaults.headers.common['Authorization']
  }
}

// Add request retry mechanism
export const setupRetry = () => {
  let retryCount = 0
  
  apiClient.interceptors.response.use(
    (response) => {
      retryCount = 0
      return response
    },
    async (error) => {
      const { config, response } = error
      
      // Only retry on server errors or network errors
      const shouldRetry = (
        !response || 
        response.status >= 500 || 
        error.code === 'ECONNABORTED' || 
        error.code === 'ERR_NETWORK'
      )
      
      if (shouldRetry && retryCount < apiConfig.retryAttempts) {
        retryCount++
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, apiConfig.retryDelay * retryCount))
        
        return apiClient(config)
      }
      
      return Promise.reject(error)
    }
  )
}

// Initialize retry mechanism
setupRetry()

export { apiClient }
export default apiClient