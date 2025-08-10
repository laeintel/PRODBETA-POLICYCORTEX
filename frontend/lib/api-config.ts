// API configuration with environment-based URL resolution
export function getApiUrl(path: string): string {
  const envBase = process.env.NEXT_PUBLIC_API_URL
  if (envBase) return `${envBase}${path}`

  if (typeof window !== 'undefined') {
    // If running frontend on 3005 (docker), backend is at 8085 on host
    if (window.location.port === '3005') return `http://localhost:8085${path}`
    // Default: let Next.js middleware proxy relative paths
    return path
  }
  // Server-side rendering: keep relative, middleware rewrites
  return path
}

export function getHealthUrl(): string {
  const envBase = process.env.NEXT_PUBLIC_API_URL
  if (envBase) return `${envBase}/health`
  if (typeof window !== 'undefined' && window.location.port === '3005') {
    return 'http://localhost:8085/health'
  }
  return '/health'
}