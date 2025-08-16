/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

/**
 * Internationalization (i18n) framework for multi-language support
 */

import { createContext, useContext, useState, useEffect } from 'react'

// Supported locales
export const LOCALES = {
  'en-US': { name: 'English (US)', flag: '🇺🇸', dir: 'ltr' },
  'en-GB': { name: 'English (UK)', flag: '🇬🇧', dir: 'ltr' },
  'es-ES': { name: 'Español', flag: '🇪🇸', dir: 'ltr' },
  'fr-FR': { name: 'Français', flag: '🇫🇷', dir: 'ltr' },
  'de-DE': { name: 'Deutsch', flag: '🇩🇪', dir: 'ltr' },
  'it-IT': { name: 'Italiano', flag: '🇮🇹', dir: 'ltr' },
  'pt-BR': { name: 'Português (Brasil)', flag: '🇧🇷', dir: 'ltr' },
  'ja-JP': { name: '日本語', flag: '🇯🇵', dir: 'ltr' },
  'zh-CN': { name: '中文 (简体)', flag: '🇨🇳', dir: 'ltr' },
  'ko-KR': { name: '한국어', flag: '🇰🇷', dir: 'ltr' },
  'ar-SA': { name: 'العربية', flag: '🇸🇦', dir: 'rtl' },
  'he-IL': { name: 'עברית', flag: '🇮🇱', dir: 'rtl' },
} as const

export type Locale = keyof typeof LOCALES

// Translation dictionary type
export interface TranslationDict {
  [key: string]: string | TranslationDict
}

// Translations storage
const translations: Record<Locale, TranslationDict> = {
  'en-US': {
    common: {
      loading: 'Loading...',
      error: 'An error occurred',
      retry: 'Retry',
      cancel: 'Cancel',
      save: 'Save',
      delete: 'Delete',
      edit: 'Edit',
      create: 'Create',
      search: 'Search',
      filter: 'Filter',
      export: 'Export',
      import: 'Import',
      yes: 'Yes',
      no: 'No',
    },
    navigation: {
      dashboard: 'Dashboard',
      policies: 'Policies',
      resources: 'Resources',
      compliance: 'Compliance',
      security: 'Security',
      costs: 'Costs',
      settings: 'Settings',
      help: 'Help',
      logout: 'Logout',
    },
    dashboard: {
      title: 'Governance Dashboard',
      subtitle: 'Real-time cloud governance metrics',
      metrics: {
        policies: 'Active Policies',
        violations: 'Policy Violations',
        resources: 'Total Resources',
        compliance: 'Compliance Score',
        costs: 'Monthly Spend',
        savings: 'Potential Savings',
      },
    },
    policies: {
      title: 'Policy Management',
      create: 'Create Policy',
      edit: 'Edit Policy',
      delete: 'Delete Policy',
      activate: 'Activate',
      deactivate: 'Deactivate',
      fields: {
        name: 'Policy Name',
        description: 'Description',
        category: 'Category',
        severity: 'Severity',
        rules: 'Rules',
        compliance: 'Compliance Frameworks',
      },
      severity: {
        critical: 'Critical',
        high: 'High',
        medium: 'Medium',
        low: 'Low',
      },
    },
    approvals: {
      title: 'Approval Requests',
      pending: 'Pending Approvals',
      approved: 'Approved',
      rejected: 'Rejected',
      approve: 'Approve',
      reject: 'Reject',
      comment: 'Add Comment',
      breakGlass: 'Emergency Access',
      justification: 'Justification',
    },
    errors: {
      unauthorized: 'You are not authorized to perform this action',
      notFound: 'Resource not found',
      serverError: 'Server error. Please try again later',
      validationError: 'Please check your input and try again',
      networkError: 'Network error. Please check your connection',
    },
  },
  'es-ES': {
    common: {
      loading: 'Cargando...',
      error: 'Ocurrió un error',
      retry: 'Reintentar',
      cancel: 'Cancelar',
      save: 'Guardar',
      delete: 'Eliminar',
      edit: 'Editar',
      create: 'Crear',
      search: 'Buscar',
      filter: 'Filtrar',
      export: 'Exportar',
      import: 'Importar',
      yes: 'Sí',
      no: 'No',
    },
    navigation: {
      dashboard: 'Panel de Control',
      policies: 'Políticas',
      resources: 'Recursos',
      compliance: 'Cumplimiento',
      security: 'Seguridad',
      costs: 'Costos',
      settings: 'Configuración',
      help: 'Ayuda',
      logout: 'Cerrar Sesión',
    },
    dashboard: {
      title: 'Panel de Gobernanza',
      subtitle: 'Métricas de gobernanza en la nube en tiempo real',
      metrics: {
        policies: 'Políticas Activas',
        violations: 'Violaciones de Política',
        resources: 'Recursos Totales',
        compliance: 'Puntuación de Cumplimiento',
        costs: 'Gasto Mensual',
        savings: 'Ahorros Potenciales',
      },
    },
    // ... more Spanish translations
  },
  'fr-FR': {
    common: {
      loading: 'Chargement...',
      error: 'Une erreur est survenue',
      retry: 'Réessayer',
      cancel: 'Annuler',
      save: 'Enregistrer',
      delete: 'Supprimer',
      edit: 'Modifier',
      create: 'Créer',
      search: 'Rechercher',
      filter: 'Filtrer',
      export: 'Exporter',
      import: 'Importer',
      yes: 'Oui',
      no: 'Non',
    },
    // ... more French translations
  },
  // ... other locale translations (stubbed for brevity)
} as any

// ICU Message Format support
export function formatMessage(
  message: string,
  values: Record<string, any> = {},
  locale: Locale = 'en-US'
): string {
  let formatted = message

  // Replace placeholders {key} with values
  Object.entries(values).forEach(([key, value]) => {
    const placeholder = new RegExp(`\\{${key}\\}`, 'g')
    formatted = formatted.replace(placeholder, String(value))
  })

  // Handle pluralization {count, plural, one {# item} other {# items}}
  const pluralRegex = /\{(\w+),\s*plural,\s*([^}]+)\}/g
  formatted = formatted.replace(pluralRegex, (match, key, rules) => {
    const count = (values as any)[key]
    const rulesObj = parsePluralRules(rules)
    
    if (count === 0 && rulesObj.zero) {
      return rulesObj.zero.replace('#', count)
    } else if (count === 1 && rulesObj.one) {
      return rulesObj.one.replace('#', count)
    } else {
      return (rulesObj.other || rulesObj.many || '').replace('#', count)
    }
  })

  // Handle select {gender, select, male {He} female {She} other {They}}
  const selectRegex = /\{(\w+),\s*select,\s*([^}]+)\}/g
  formatted = formatted.replace(selectRegex, (match, key, options) => {
    const value = (values as any)[key]
    const optionsObj = parseSelectOptions(options)
    return optionsObj[value] || optionsObj.other || ''
  })

  return formatted
}

function parsePluralRules(rules: string): Record<string, string> {
  const result: Record<string, string> = {}
  const matches = rules.matchAll(/(\w+)\s*\{([^}]+)\}/g)
  
  for (const match of matches) {
    result[(match as any)[1]] = (match as any)[2]
  }
  
  return result
}

function parseSelectOptions(options: string): Record<string, string> {
  const result: Record<string, string> = {}
  const matches = options.matchAll(/(\w+)\s*\{([^}]+)\}/g)
  
  for (const match of matches) {
    result[(match as any)[1]] = (match as any)[2]
  }
  
  return result
}

// Number formatting
export function formatNumber(
  value: number,
  locale: Locale = 'en-US',
  options?: Intl.NumberFormatOptions
): string {
  return new Intl.NumberFormat(locale, options).format(value)
}

// Currency formatting
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  locale: Locale = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value)
}

// Date formatting
export function formatDate(
  date: Date | string,
  locale: Locale = 'en-US',
  options?: Intl.DateTimeFormatOptions
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat(locale, options).format(dateObj)
}

// Relative time formatting
export function formatRelativeTime(
  date: Date | string,
  locale: Locale = 'en-US'
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })
  
  const diff = (dateObj.getTime() - Date.now()) / 1000
  const absDiff = Math.abs(diff)
  
  if (absDiff < 60) return rtf.format(Math.round(diff), 'second')
  if (absDiff < 3600) return rtf.format(Math.round(diff / 60), 'minute')
  if (absDiff < 86400) return rtf.format(Math.round(diff / 3600), 'hour')
  if (absDiff < 2592000) return rtf.format(Math.round(diff / 86400), 'day')
  if (absDiff < 31536000) return rtf.format(Math.round(diff / 2592000), 'month')
  return rtf.format(Math.round(diff / 31536000), 'year')
}

// i18n Context
interface I18nContextType {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: string, values?: Record<string, any>) => string
  dir: 'ltr' | 'rtl'
  formatNumber: typeof formatNumber
  formatCurrency: typeof formatCurrency
  formatDate: typeof formatDate
  formatRelativeTime: typeof formatRelativeTime
}

const I18nContext = createContext<I18nContextType | null>(null)

// i18n Provider
export function I18nProvider({ 
  children, 
  defaultLocale = 'en-US' 
}: { 
  children: React.ReactNode
  defaultLocale?: Locale 
}) {
  const [locale, setLocaleState] = useState<Locale>(defaultLocale)

  useEffect(() => {
    // Only access browser APIs on client side
    if (typeof window !== 'undefined') {
      // Detect browser locale
      const browserLocale = navigator.language as Locale
      if (LOCALES[browserLocale]) {
        setLocaleState(browserLocale)
      }

      // Load saved locale from localStorage
      const savedLocale = localStorage.getItem('locale') as Locale
      if (savedLocale && LOCALES[savedLocale]) {
        setLocaleState(savedLocale)
      }
    }
  }, [])

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale)
    
    // Only access browser APIs on client side
    if (typeof window !== 'undefined') {
      localStorage.setItem('locale', newLocale)
      
      // Update document direction for RTL languages
      document.documentElement.dir = LOCALES[newLocale].dir
      document.documentElement.lang = newLocale
    }
  }

  const t = (key: string, values?: Record<string, any>): string => {
    const keys = key.split('.')
    let translation: any = translations[locale] || translations['en-US']
    
    for (const k of keys) {
      translation = translation?.[k]
      if (!translation) break
    }
    
    if (typeof translation === 'string') {
      return formatMessage(translation, values, locale)
    }
    
    // Fallback to English
    translation = translations['en-US']
    for (const k of keys) {
      translation = translation?.[k]
      if (!translation) break
    }
    
    if (typeof translation === 'string') {
      return formatMessage(translation, values, 'en-US')
    }
    
    // Return key if no translation found
    return key
  }

  const value: I18nContextType = {
    locale,
    setLocale,
    t,
    dir: LOCALES[locale].dir,
    formatNumber: (value: number, localeOverride?: Locale, options?: Intl.NumberFormatOptions) => 
      formatNumber(value, localeOverride || locale, options),
    formatCurrency: (value: number, currency?: string, localeOverride?: Locale) => 
      formatCurrency(value, currency, localeOverride || locale),
    formatDate: (date: Date | string, localeOverride?: Locale, options?: Intl.DateTimeFormatOptions) => 
      formatDate(date, localeOverride || locale, options),
    formatRelativeTime: (date: Date | string, localeOverride?: Locale) => 
      formatRelativeTime(date, localeOverride || locale),
  }

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  )
}

// useI18n hook
export function useI18n() {
  const context = useContext(I18nContext)
  
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider')
  }
  
  return context
}

// Language selector component
export function LanguageSelector() {
  const { locale, setLocale } = useI18n()

  return (
    <select
      value={locale}
      onChange={(e) => setLocale(e.target.value as Locale)}
      className="px-3 py-1 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      aria-label="Select language"
    >
      {Object.entries(LOCALES).map(([key, value]) => (
        <option key={key} value={key}>
          {value.flag} {value.name}
        </option>
      ))}
    </select>
  )
}

// Load translations dynamically
export async function loadTranslations(locale: Locale): Promise<TranslationDict> {
  try {
    // In production, load from CDN or API
    const response = await fetch(`/locales/${locale}.json`)
    const data = await response.json()
    translations[locale] = data
    return data
  } catch (error) {
    console.error(`Failed to load translations for ${locale}:`, error)
    return translations['en-US']
  }
}

// Preload translations for better performance
export async function preloadTranslations(locales: Locale[]) {
  await Promise.all(locales.map(loadTranslations))
}

// Export types
export type { I18nContextType }


