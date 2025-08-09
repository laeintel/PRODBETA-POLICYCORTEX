// API configuration with environment-based URL resolution
export function getApiUrl(path: string): string {
  // In browser, check if we're in Docker container (port 3005)
  if (typeof window !== 'undefined') {
    const port = window.location.port;
    
    // If accessing via Docker frontend port 3005
    if (port === '3005') {
      // Use the backend port directly
      return `http://localhost:8085${path}`;
    }
    
    // Default to relative URL for local dev
    return path;
  }
  
  // Server-side: use relative URLs
  return path;
}

export function getHealthUrl(): string {
  if (typeof window !== 'undefined' && window.location.port === '3005') {
    return 'http://localhost:8085/health';
  }
  return '/health';
}