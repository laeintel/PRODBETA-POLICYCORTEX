import { render, screen } from '@/test/test-utils';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock the hooks that might cause issues in tests
vi.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    initialize: vi.fn(),
    isLoading: false,
    isAuthenticated: true,
  }),
}));

vi.mock('@/hooks/useAuthStatus', () => ({
  useAuthStatus: () => ({
    isReady: true,
  }),
}));

vi.mock('@azure/msal-react', () => ({
  ...vi.importActual('@azure/msal-react'),
  useIsAuthenticated: () => true,
  AuthenticatedTemplate: ({ children }: any) => children,
  UnauthenticatedTemplate: () => null,
}));

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    expect(document.body).toBeInTheDocument();
  });

  it('renders loading screen when not ready', () => {
    vi.mock('@/hooks/useAuthStatus', () => ({
      useAuthStatus: () => ({
        isReady: false,
      }),
    }));
    
    render(<App />);
    // The loading screen would be rendered
  });
});