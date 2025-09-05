/**
 * Unified API fetch helper as per TD.MD requirements
 * Single fetch helper with no-store cache and consistent error handling
 */

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081';
  const url = path.startsWith('http') ? path : `${baseUrl}${path}`;
  
  const response = await fetch(url, {
    cache: 'no-store', // Prevent Next.js caching as per TD.MD
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    
    // Handle 503 Service Unavailable with configuration hints
    if (response.status === 503) {
      let errorData;
      try {
        errorData = JSON.parse(text);
      } catch {
        errorData = { message: text || 'Service unavailable' };
      }
      
      throw {
        status: 503,
        message: errorData.message || 'Service unavailable',
        hint: errorData.hint || 'Check configuration in docs/REVAMP/REAL_MODE_SETUP.md',
        error: errorData.error,
        isConfigError: true,
      };
    }
    
    // Standard error handling
    throw new Error(`API ${response.status}: ${text || response.statusText}`);
  }

  return response.json() as Promise<T>;
}

/**
 * Helper to check if error is a configuration error (503 with hints)
 */
export function isConfigError(error: any): error is { 
  status: number; 
  message: string; 
  hint: string;
  error?: string;
  isConfigError: true;
} {
  return error?.isConfigError === true;
}

/**
 * Helper to format configuration error for display
 */
export function formatConfigError(error: any): {
  title: string;
  message: string;
  hint?: string;
} {
  if (isConfigError(error)) {
    return {
      title: 'Configuration Required',
      message: error.message,
      hint: error.hint
    };
  }
  
  return {
    title: 'Error',
    message: error?.message || 'An unexpected error occurred'
  };
}