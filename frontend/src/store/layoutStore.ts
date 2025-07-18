import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { storageKeys } from '@/config/environment'

interface LayoutState {
  sidebarOpen: boolean
  sidebarWidth: number
  sidebarCollapsed: boolean
  headerHeight: number
  footerHeight: number
  breadcrumbs: BreadcrumbItem[]
  pageTitle: string
  showBreadcrumbs: boolean
  compactMode: boolean
}

interface BreadcrumbItem {
  label: string
  href?: string
  icon?: string
}

interface LayoutActions {
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setSidebarWidth: (width: number) => void
  toggleSidebarCollapsed: () => void
  setSidebarCollapsed: (collapsed: boolean) => void
  setBreadcrumbs: (breadcrumbs: BreadcrumbItem[]) => void
  setPageTitle: (title: string) => void
  setShowBreadcrumbs: (show: boolean) => void
  setCompactMode: (compact: boolean) => void
}

const initialState: LayoutState = {
  sidebarOpen: true,
  sidebarWidth: 280,
  sidebarCollapsed: false,
  headerHeight: 64,
  footerHeight: 60,
  breadcrumbs: [],
  pageTitle: '',
  showBreadcrumbs: true,
  compactMode: false,
}

export const useLayoutStore = create<LayoutState & LayoutActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      toggleSidebar: () => {
        set(state => ({ sidebarOpen: !state.sidebarOpen }))
      },

      setSidebarOpen: (open) => {
        set({ sidebarOpen: open })
      },

      setSidebarWidth: (width) => {
        set({ sidebarWidth: Math.max(240, Math.min(400, width)) })
      },

      toggleSidebarCollapsed: () => {
        set(state => ({ sidebarCollapsed: !state.sidebarCollapsed }))
      },

      setSidebarCollapsed: (collapsed) => {
        set({ sidebarCollapsed: collapsed })
      },

      setBreadcrumbs: (breadcrumbs) => {
        set({ breadcrumbs })
      },

      setPageTitle: (title) => {
        set({ pageTitle: title })
      },

      setShowBreadcrumbs: (show) => {
        set({ showBreadcrumbs: show })
      },

      setCompactMode: (compact) => {
        set({ compactMode: compact })
      },
    }),
    {
      name: storageKeys.settings,
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        sidebarWidth: state.sidebarWidth,
        sidebarCollapsed: state.sidebarCollapsed,
        showBreadcrumbs: state.showBreadcrumbs,
        compactMode: state.compactMode,
      }),
    }
  )
)