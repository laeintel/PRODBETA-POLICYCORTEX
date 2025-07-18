import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { User } from '@/types'
import { storageKeys } from '@/config/environment'

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  token: string | null
  refreshToken: string | null
  lastLoginAt: string | null
  sessionTimeout: number | null
}

interface AuthActions {
  setUser: (user: User | null) => void
  setAuthenticated: (isAuthenticated: boolean) => void
  setToken: (token: string | null) => void
  setRefreshToken: (refreshToken: string | null) => void
  setLastLoginAt: (lastLoginAt: string | null) => void
  setSessionTimeout: (timeout: number | null) => void
  logout: () => void
  updateUser: (updates: Partial<User>) => void
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  token: null,
  refreshToken: null,
  lastLoginAt: null,
  sessionTimeout: null,
}

export const useAuthStore = create<AuthState & AuthActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      setUser: (user) => {
        set({ user })
        if (user) {
          set({ isAuthenticated: true, lastLoginAt: new Date().toISOString() })
        }
      },

      setAuthenticated: (isAuthenticated) => {
        set({ isAuthenticated })
        if (!isAuthenticated) {
          set({ user: null, token: null, refreshToken: null })
        }
      },

      setToken: (token) => {
        set({ token })
      },

      setRefreshToken: (refreshToken) => {
        set({ refreshToken })
      },

      setLastLoginAt: (lastLoginAt) => {
        set({ lastLoginAt })
      },

      setSessionTimeout: (sessionTimeout) => {
        set({ sessionTimeout })
      },

      logout: () => {
        set(initialState)
      },

      updateUser: (updates) => {
        const { user } = get()
        if (user) {
          set({ user: { ...user, ...updates } })
        }
      },
    }),
    {
      name: storageKeys.user,
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        lastLoginAt: state.lastLoginAt,
        sessionTimeout: state.sessionTimeout,
      }),
    }
  )
)