// Real API connection helper - connects to Azure real data server on port 8084
export const REAL_API_BASE = process.env.NEXT_PUBLIC_REAL_API_BASE || 'http://localhost:8084';

export async function real<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${REAL_API_BASE}${path}`, { 
    cache: 'no-store', 
    ...init 
  });
  
  if (!res.ok) {
    throw new Error(`Real API ${res.status}: ${res.statusText}`);
  }
  
  return res.json() as Promise<T>;
}

// Helper to check if we're in real data mode
export function isRealMode(): boolean {
  return process.env.NEXT_PUBLIC_USE_REAL_DATA === 'true' || 
         process.env.USE_REAL_DATA === 'true';
}

// Get the appropriate API base URL
export function getApiBase(): string {
  return isRealMode() ? REAL_API_BASE : process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
}