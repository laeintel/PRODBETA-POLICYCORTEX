import '@testing-library/jest-dom'
import { vi } from 'vitest'
import { TextEncoder, TextDecoder } from 'util'

// Polyfill for TextEncoder/TextDecoder
global.TextEncoder = TextEncoder
global.TextDecoder = TextDecoder

// Mock crypto API for MSAL
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: (arr: any) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256)
      }
      return arr
    },
    randomUUID: () => '12345678-1234-1234-1234-123456789012',
    subtle: {
      digest: vi.fn(),
      generateKey: vi.fn(),
      sign: vi.fn(),
      verify: vi.fn(),
      encrypt: vi.fn(),
      decrypt: vi.fn(),
    },
  },
})

// Mock MSAL library
vi.mock('@azure/msal-browser', () => ({
  PublicClientApplication: vi.fn().mockImplementation(() => ({
    initialize: vi.fn().mockResolvedValue(undefined),
    handleRedirectPromise: vi.fn().mockResolvedValue(null),
    acquireTokenSilent: vi.fn().mockResolvedValue({ accessToken: 'mock-token' }),
    acquireTokenPopup: vi.fn().mockResolvedValue({ accessToken: 'mock-token' }),
    loginPopup: vi.fn().mockResolvedValue({}),
    logout: vi.fn().mockResolvedValue(undefined),
    getAllAccounts: vi.fn().mockReturnValue([]),
    getAccountByUsername: vi.fn().mockReturnValue(null),
    addEventCallback: vi.fn().mockReturnValue('mock-callback-id'),
    removeEventCallback: vi.fn(),
  })),
  InteractionRequiredAuthError: class InteractionRequiredAuthError extends Error {},
  EventType: {
    LOGIN_SUCCESS: 'LOGIN_SUCCESS',
    LOGIN_FAILURE: 'LOGIN_FAILURE',
    LOGOUT_SUCCESS: 'LOGOUT_SUCCESS',
    ACQUIRE_TOKEN_SUCCESS: 'ACQUIRE_TOKEN_SUCCESS',
    ACQUIRE_TOKEN_FAILURE: 'ACQUIRE_TOKEN_FAILURE',
  },
  InteractionType: {
    Popup: 'popup',
    Redirect: 'redirect',
    Silent: 'silent',
  },
  BrowserAuthError: class BrowserAuthError extends Error {},
}))

// Mock @azure/msal-react
vi.mock('@azure/msal-react', () => ({
  MsalProvider: ({ children }: any) => children,
  useMsal: () => ({
    instance: {
      acquireTokenSilent: vi.fn().mockResolvedValue({ accessToken: 'mock-token' }),
      getAllAccounts: vi.fn().mockReturnValue([]),
    },
    accounts: [],
    inProgress: 'none',
  }),
  useIsAuthenticated: () => false,
  useMsalAuthentication: () => ({
    login: vi.fn(),
    result: null,
    error: null,
  }),
  AuthenticatedTemplate: ({ children }: any) => children,
  UnauthenticatedTemplate: ({ children }: any) => children,
  useAccount: () => null,
  useMsalProvider: () => ({
    instance: {
      acquireTokenSilent: vi.fn().mockResolvedValue({ accessToken: 'mock-token' }),
    },
  }),
}))

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock window.location
Object.defineProperty(window, 'location', {
  writable: true,
  value: {
    href: 'http://localhost:3000',
    origin: 'http://localhost:3000',
    protocol: 'http:',
    host: 'localhost:3000',
    hostname: 'localhost',
    port: '3000',
    pathname: '/',
    search: '',
    hash: '',
    assign: vi.fn(),
    replace: vi.fn(),
    reload: vi.fn(),
  },
})

// Mock navigator
Object.defineProperty(navigator, 'userAgent', {
  writable: true,
  value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
})

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock clipboard API
Object.defineProperty(navigator, 'clipboard', {
  writable: true,
  value: {
    readText: vi.fn().mockResolvedValue(''),
    writeText: vi.fn().mockResolvedValue(undefined),
  },
})


// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
}

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// Mock sessionStorage
const sessionStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
}

Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock,
})

// Mock fetch
global.fetch = vi.fn()

// Mock URL.createObjectURL
Object.defineProperty(URL, 'createObjectURL', {
  writable: true,
  value: vi.fn().mockReturnValue('blob:http://localhost:3000/12345678-1234-1234-1234-123456789012'),
})

Object.defineProperty(URL, 'revokeObjectURL', {
  writable: true,
  value: vi.fn(),
})

// Mock MutationObserver
global.MutationObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn(),
}))

// Mock performance API
Object.defineProperty(window, 'performance', {
  writable: true,
  value: {
    now: vi.fn().mockReturnValue(Date.now()),
    mark: vi.fn(),
    measure: vi.fn(),
    getEntriesByType: vi.fn().mockReturnValue([]),
    getEntriesByName: vi.fn().mockReturnValue([]),
  },
})

// Mock console methods to avoid noise in tests
const originalConsole = global.console
global.console = {
  ...originalConsole,
  warn: vi.fn(),
  error: vi.fn(),
  log: vi.fn(),
  info: vi.fn(),
  debug: vi.fn(),
}

// Restore console for debugging if needed
export const restoreConsole = () => {
  global.console = originalConsole
}

// Mock socket.io-client
vi.mock('socket.io-client', () => ({
  io: vi.fn().mockReturnValue({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    connected: false,
    id: 'mock-socket-id',
  }),
}))

// Mock react-hot-toast
vi.mock('react-hot-toast', () => ({
  default: {
    success: vi.fn(),
    error: vi.fn(),
    loading: vi.fn(),
    dismiss: vi.fn(),
  },
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    loading: vi.fn(),
    dismiss: vi.fn(),
  },
  Toaster: vi.fn(() => null),
}))

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: vi.fn(({ children }) => children),
    span: vi.fn(({ children }) => children),
    p: vi.fn(({ children }) => children),
    h1: vi.fn(({ children }) => children),
    h2: vi.fn(({ children }) => children),
    h3: vi.fn(({ children }) => children),
    h4: vi.fn(({ children }) => children),
    h5: vi.fn(({ children }) => children),
    h6: vi.fn(({ children }) => children),
    button: vi.fn(({ children }) => children),
    a: vi.fn(({ children }) => children),
    img: vi.fn(({ children }) => children),
    section: vi.fn(({ children }) => children),
    article: vi.fn(({ children }) => children),
    aside: vi.fn(({ children }) => children),
    nav: vi.fn(({ children }) => children),
    header: vi.fn(({ children }) => children),
    footer: vi.fn(({ children }) => children),
    main: vi.fn(({ children }) => children),
  },
  AnimatePresence: vi.fn(({ children }) => children),
}))

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: vi.fn(),
    useLocation: vi.fn(() => ({ pathname: '/' })),
    useParams: vi.fn(() => ({})),
    useSearchParams: vi.fn(() => [new URLSearchParams(), vi.fn()]),
    Navigate: vi.fn(({ to }) => `Navigate to ${to}`),
    Link: vi.fn(({ children, to }) => `Link to ${to}: ${children}`),
    NavLink: vi.fn(({ children, to }) => `NavLink to ${to}: ${children}`),
  }
})

// Mock environment variables
vi.mock('@/config/environment', () => ({
  env: {
    NODE_ENV: 'test',
    APP_NAME: 'PolicyCortex',
    APP_VERSION: '1.0.0',
    API_BASE_URL: 'http://localhost:8000/api',
    WS_URL: 'ws://localhost:8000/ws',
    AZURE_CLIENT_ID: 'test-client-id',
    AZURE_TENANT_ID: 'test-tenant-id',
    AZURE_REDIRECT_URI: 'http://localhost:3000',
    ENABLE_ANALYTICS: false,
    ENABLE_NOTIFICATIONS: true,
    ENABLE_WEBSOCKET: true,
    ENABLE_PWA: true,
    ENABLE_DARK_MODE: true,
    ENABLE_DEBUG: false,
  },
  isDevelopment: false,
  isProduction: false,
  isTest: true,
  features: {
    analytics: false,
    notifications: true,
    websocket: true,
    pwa: true,
    darkMode: true,
    debug: false,
  },
  storageKeys: {
    theme: 'policycortex_theme',
    user: 'policycortex_user',
    settings: 'policycortex_settings',
    preferences: 'policycortex_preferences',
    tokens: 'policycortex_tokens',
    cache: 'policycortex_cache',
    notifications: 'policycortex_notifications',
    dashboard: 'policycortex_dashboard',
    filters: 'policycortex_filters',
    viewState: 'policycortex_view_state',
  },
}))

// Mock WebSocketProvider
vi.mock('@/providers/WebSocketProvider', () => ({
  WebSocketProvider: ({ children }: any) => children,
  useWebSocket: () => ({
    connected: false,
    connect: vi.fn(),
    disconnect: vi.fn(),
    send: vi.fn(),
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
  }),
}))

// Mock NotificationProvider
vi.mock('@/providers/NotificationProvider', () => ({
  NotificationProvider: ({ children }: any) => children,
  useNotifications: () => ({
    notifications: [],
    addNotification: vi.fn(),
    removeNotification: vi.fn(),
    clearNotifications: vi.fn(),
  }),
}))

// Mock ThemeProvider
vi.mock('@/providers/ThemeProvider', () => ({
  ThemeProvider: ({ children }: any) => children,
  useTheme: () => ({
    theme: 'light',
    toggleTheme: vi.fn(),
    setTheme: vi.fn(),
  }),
}))

// Clean up after each test
afterEach(() => {
  vi.clearAllMocks()
  localStorageMock.clear()
  sessionStorageMock.clear()
})