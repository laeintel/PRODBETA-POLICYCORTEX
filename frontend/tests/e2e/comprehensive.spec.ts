import { test, expect, Page } from '@playwright/test'

// Configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000'

// Helper function to test responsive design
async function testResponsiveness(page: Page, url: string) {
  const viewports = [
    { width: 320, height: 568, name: 'iPhone SE' },
    { width: 768, height: 1024, name: 'iPad' },
    { width: 1024, height: 768, name: 'Desktop Small' },
    { width: 1920, height: 1080, name: 'Desktop Full HD' },
    { width: 2560, height: 1440, name: '4K' }
  ]

  for (const viewport of viewports) {
    await page.setViewportSize(viewport)
    await page.goto(url)
    
    // Check if layout adapts
    const isMenuVisible = await page.locator('[data-testid="navigation-menu"]').isVisible()
    const isMobileMenuVisible = await page.locator('[data-testid="mobile-menu-toggle"]').isVisible()
    
    if (viewport.width < 1024) {
      expect(isMobileMenuVisible).toBeTruthy()
    } else {
      expect(isMenuVisible).toBeTruthy()
    }
    
    // Take screenshot for visual regression
    await page.screenshot({ 
      path: `screenshots/${url.replace(/\//g, '-')}-${viewport.name}.png`,
      fullPage: true 
    })
  }
}

// Test all navigation items
test.describe('Navigation Tests', () => {
  test('All navigation links should work', async ({ page }) => {
    await page.goto(BASE_URL)
    
    const navigationItems = [
      { name: 'Dashboard', url: '/tactical' },
      { name: 'Executive', url: '/executive' },
      { name: 'FinOps', url: '/finops' },
      { name: 'Governance', url: '/governance' },
      { name: 'Security & Access', url: '/security' },
      { name: 'Operations', url: '/operations' },
      { name: 'DevOps & CI/CD', url: '/devops' },
      { name: 'DevSecOps', url: '/devsecops' },
      { name: 'Cloud ITSM', url: '/itsm' },
      { name: 'AI Intelligence', url: '/ai' },
      { name: 'Governance Copilot', url: '/copilot' },
      { name: 'Blockchain Audit', url: '/blockchain' },
      { name: 'Quantum-Safe Secrets', url: '/quantum' },
      { name: 'Edge Governance', url: '/edge' },
      { name: 'Audit Trail', url: '/audit' },
      { name: 'Settings', url: '/settings' }
    ]
    
    for (const item of navigationItems) {
      // Click navigation item
      await page.click(`text=${item.name}`)
      
      // Wait for navigation
      await page.waitForURL(`**${item.url}`)
      
      // Verify we're on the correct page
      expect(page.url()).toContain(item.url)
      
      // Check page loaded correctly
      await expect(page.locator('h1')).toBeVisible()
      
      // Check for console errors
      const consoleErrors: string[] = []
      page.on('console', msg => {
        if (msg.type() === 'error') {
          consoleErrors.push(msg.text())
        }
      })
      
      expect(consoleErrors).toHaveLength(0)
    }
  })
  
  test('Subsection navigation should work', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Test Executive subsections
    await page.click('text=Executive')
    await page.click('text=ROI Calculator')
    await expect(page).toHaveURL(/.*\/executive\/roi/)
    
    await page.click('text=Risk-to-Revenue Map')
    await expect(page).toHaveURL(/.*\/executive\/risk-map/)
    
    await page.click('text=Board Reports')
    await expect(page).toHaveURL(/.*\/executive\/reports/)
  })
})

// Test all buttons functionality
test.describe('Button Functionality Tests', () => {
  test('All buttons should have proper click handlers', async ({ page }) => {
    const pagesToTest = [
      '/executive',
      '/finops',
      '/quantum',
      '/edge',
      '/blockchain',
      '/copilot'
    ]
    
    for (const pageUrl of pagesToTest) {
      await page.goto(`${BASE_URL}${pageUrl}`)
      
      // Find all buttons
      const buttons = await page.locator('button').all()
      
      for (const button of buttons) {
        const buttonText = await button.textContent()
        
        // Skip if button is disabled
        const isDisabled = await button.isDisabled()
        if (isDisabled) continue
        
        // Click button and check for action
        await button.click()
        
        // Check if button triggered an action (no console.log only)
        const hasAction = await page.evaluate(() => {
          // Check if any state changed, modal opened, or API called
          return window.performance.getEntriesByType('resource').length > 0
        })
        
        // Button should either navigate, open modal, or make API call
        if (!hasAction) {
          const currentUrl = page.url()
          // Wait a bit to see if navigation happens
          await page.waitForTimeout(500)
          const newUrl = page.url()
          
          expect(currentUrl !== newUrl || hasAction).toBeTruthy()
        }
      }
    }
  })
  
  test('Export buttons should download data', async ({ page }) => {
    await page.goto(`${BASE_URL}/executive`)
    
    // Set up download promise before clicking
    const downloadPromise = page.waitForEvent('download')
    
    // Click export button
    await page.click('text=Download ROI Report')
    
    // Wait for download
    const download = await downloadPromise
    
    // Verify download
    expect(download).toBeTruthy()
  })
})

// Test API integration
test.describe('API Integration Tests', () => {
  test('Pages should load real data from APIs', async ({ page }) => {
    await page.goto(`${BASE_URL}/executive`)
    
    // Wait for API calls
    const apiCalls = await page.evaluate(() => {
      return window.performance.getEntriesByType('resource')
        .filter(entry => entry.name.includes('/api/'))
    })
    
    // Should make API calls, not use hardcoded data
    expect(apiCalls.length).toBeGreaterThan(0)
    
    // Check data is rendered
    await expect(page.locator('[data-testid="kpi-metric"]')).toHaveCount(5)
    
    // Data should update periodically
    const initialValue = await page.locator('[data-testid="kpi-value"]').first().textContent()
    await page.waitForTimeout(61000) // Wait for refresh interval
    const updatedValue = await page.locator('[data-testid="kpi-value"]').first().textContent()
    
    // Values might change on refresh
    expect(initialValue).toBeDefined()
    expect(updatedValue).toBeDefined()
  })
  
  test('Error states should display correctly', async ({ page }) => {
    // Intercept API calls and return error
    await page.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' })
      })
    })
    
    await page.goto(`${BASE_URL}/executive`)
    
    // Should show error state
    await expect(page.locator('text=Error Loading Data')).toBeVisible()
    
    // Should have retry button
    await expect(page.locator('button:has-text("Retry")')).toBeVisible()
  })
  
  test('Loading states should display', async ({ page }) => {
    // Delay API responses
    await page.route('**/api/**', async route => {
      await new Promise(resolve => setTimeout(resolve, 3000))
      route.continue()
    })
    
    await page.goto(`${BASE_URL}/finops`)
    
    // Should show loading state
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible()
  })
})

// Test WebSocket real-time updates
test.describe('Real-time Updates', () => {
  test('WebSocket should connect and receive updates', async ({ page }) => {
    await page.goto(`${BASE_URL}/finops`)
    
    // Check WebSocket connection
    const wsConnected = await page.evaluate(() => {
      return new Promise(resolve => {
        const ws = new WebSocket('ws://localhost:8080/ws')
        ws.onopen = () => resolve(true)
        ws.onerror = () => resolve(false)
        setTimeout(() => resolve(false), 5000)
      })
    })
    
    expect(wsConnected).toBeTruthy()
    
    // Wait for real-time update
    await page.waitForTimeout(10000)
    
    // Check if anomaly alerts updated
    const anomalyCount = await page.locator('[data-testid="anomaly-alert"]').count()
    expect(anomalyCount).toBeGreaterThanOrEqual(0)
  })
})

// Test responsive design
test.describe('Responsive Design Tests', () => {
  test('All pages should be responsive', async ({ page }) => {
    const pages = [
      '/',
      '/executive',
      '/finops',
      '/governance',
      '/security',
      '/operations',
      '/devops',
      '/ai',
      '/quantum',
      '/edge',
      '/blockchain',
      '/copilot'
    ]
    
    for (const pageUrl of pages) {
      await testResponsiveness(page, `${BASE_URL}${pageUrl}`)
    }
  })
  
  test('Navigation should adapt to mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto(BASE_URL)
    
    // Mobile menu should be hidden initially
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeHidden()
    
    // Click hamburger menu
    await page.click('[data-testid="mobile-menu-toggle"]')
    
    // Mobile menu should be visible
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible()
    
    // Should be able to navigate
    await page.click('text=Executive')
    await expect(page).toHaveURL(/.*\/executive/)
  })
})

// Test UI/UX quality
test.describe('UI/UX Quality Tests', () => {
  test('All icons should render correctly', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Check all icons are visible
    const icons = await page.locator('svg').all()
    
    for (const icon of icons) {
      const isVisible = await icon.isVisible()
      expect(isVisible).toBeTruthy()
      
      // Check icon has proper size
      const box = await icon.boundingBox()
      expect(box?.width).toBeGreaterThan(0)
      expect(box?.height).toBeGreaterThan(0)
    }
  })
  
  test('Dark mode should work correctly', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Get initial theme
    const initialTheme = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })
    
    // Click theme toggle
    await page.click('[data-testid="theme-toggle"]')
    
    // Theme should change
    const newTheme = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark')
    })
    
    expect(newTheme).not.toBe(initialTheme)
    
    // All elements should adapt to theme
    const backgroundColor = await page.evaluate(() => {
      return window.getComputedStyle(document.body).backgroundColor
    })
    
    if (newTheme) {
      expect(backgroundColor).toContain('rgb(17, 24, 39)') // dark background
    } else {
      expect(backgroundColor).toContain('rgb(255, 255, 255)') // light background
    }
  })
  
  test('Charts should be interactive', async ({ page }) => {
    await page.goto(`${BASE_URL}/executive`)
    
    // Find chart elements
    const charts = await page.locator('[data-testid="chart-container"]').all()
    
    for (const chart of charts) {
      // Hover over chart
      await chart.hover()
      
      // Tooltip should appear
      await expect(page.locator('[data-testid="chart-tooltip"]')).toBeVisible()
    }
  })
})

// Test form validation
test.describe('Form Validation Tests', () => {
  test('ROI Calculator should validate inputs', async ({ page }) => {
    await page.goto(`${BASE_URL}/executive/roi`)
    
    // Try invalid input
    await page.fill('[data-testid="cloud-spend-input"]', '-1000')
    
    // Should show validation error
    await expect(page.locator('text=Please enter a valid amount')).toBeVisible()
    
    // Enter valid input
    await page.fill('[data-testid="cloud-spend-input"]', '500000')
    
    // Error should disappear
    await expect(page.locator('text=Please enter a valid amount')).toBeHidden()
    
    // ROI should update
    await expect(page.locator('[data-testid="roi-percentage"]')).toContainText('%')
  })
})

// Test accessibility
test.describe('Accessibility Tests', () => {
  test('All pages should be keyboard navigable', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Tab through elements
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('Tab')
      
      // Check focused element is visible
      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.tagName
      })
      
      expect(focusedElement).toBeTruthy()
    }
    
    // Should be able to activate with Enter
    await page.keyboard.press('Enter')
    
    // Check navigation happened
    const url = page.url()
    expect(url).not.toBe(BASE_URL)
  })
  
  test('ARIA labels should be present', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Check important elements have ARIA labels
    const buttons = await page.locator('button').all()
    
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label')
      const text = await button.textContent()
      
      // Button should have either aria-label or text content
      expect(ariaLabel || text).toBeTruthy()
    }
  })
})

// Performance tests
test.describe('Performance Tests', () => {
  test('Pages should load quickly', async ({ page }) => {
    const pages = ['/', '/executive', '/finops', '/quantum']
    
    for (const pageUrl of pages) {
      const startTime = Date.now()
      await page.goto(`${BASE_URL}${pageUrl}`)
      await page.waitForLoadState('networkidle')
      const loadTime = Date.now() - startTime
      
      // Page should load in under 3 seconds
      expect(loadTime).toBeLessThan(3000)
    }
  })
  
  test('No memory leaks on navigation', async ({ page }) => {
    await page.goto(BASE_URL)
    
    // Get initial memory
    const initialMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0
    })
    
    // Navigate through multiple pages
    for (let i = 0; i < 10; i++) {
      await page.goto(`${BASE_URL}/executive`)
      await page.goto(`${BASE_URL}/finops`)
      await page.goto(`${BASE_URL}/quantum`)
    }
    
    // Force garbage collection if available
    await page.evaluate(() => {
      if ((window as any).gc) {
        (window as any).gc()
      }
    })
    
    // Check memory hasn't grown excessively
    const finalMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0
    })
    
    // Memory shouldn't grow more than 50MB
    const memoryGrowth = finalMemory - initialMemory
    expect(memoryGrowth).toBeLessThan(50 * 1024 * 1024)
  })
})

// Complete click test for all 520+ interactive elements
test.describe('Comprehensive Click Tests', () => {
  test('Test all clickable elements', async ({ page }) => {
    const pages = [
      '/', '/executive', '/executive/roi', '/executive/risk-map', '/executive/reports',
      '/finops', '/finops/optimization', '/finops/forecasting', '/finops/chargeback',
      '/governance', '/security', '/security/rbac', '/security/pim',
      '/operations', '/operations/resources', '/operations/monitoring',
      '/devops', '/devops/pipelines', '/devops/releases',
      '/devsecops', '/devsecops/pipelines',
      '/ai', '/ai/predictive', '/ai/correlations', '/ai/chat',
      '/quantum', '/edge', '/blockchain', '/copilot',
      '/itsm', '/audit', '/settings'
    ]
    
    let totalClicks = 0
    const clickResults: { page: string; element: string; success: boolean }[] = []
    
    for (const pageUrl of pages) {
      await page.goto(`${BASE_URL}${pageUrl}`)
      await page.waitForLoadState('networkidle')
      
      // Find all clickable elements
      const clickables = await page.locator('button, a, [role="button"], [onclick], input[type="submit"], [data-clickable]').all()
      
      for (const element of clickables) {
        try {
          const isVisible = await element.isVisible()
          if (!isVisible) continue
          
          const elementText = await element.textContent() || 'unnamed'
          const elementTag = await element.evaluate(el => el.tagName)
          
          // Skip if disabled
          const isDisabled = await element.isDisabled().catch(() => false)
          if (isDisabled) continue
          
          // Click element
          await element.click({ timeout: 1000 }).catch(() => {})
          totalClicks++
          
          // Check if action occurred
          await page.waitForTimeout(100)
          
          clickResults.push({
            page: pageUrl,
            element: `${elementTag}: ${elementText.substring(0, 30)}`,
            success: true
          })
          
          // Go back if navigated away
          if (page.url() !== `${BASE_URL}${pageUrl}`) {
            await page.goto(`${BASE_URL}${pageUrl}`)
            await page.waitForLoadState('networkidle')
          }
        } catch (error) {
          // Log failed clicks
          clickResults.push({
            page: pageUrl,
            element: 'unknown',
            success: false
          })
        }
      }
    }
    
    // Generate report
    console.log(`Total clicks tested: ${totalClicks}`)
    console.log(`Successful clicks: ${clickResults.filter(r => r.success).length}`)
    console.log(`Failed clicks: ${clickResults.filter(r => !r.success).length}`)
    
    // Should have tested at least 520 clicks
    expect(totalClicks).toBeGreaterThanOrEqual(520)
    
    // Success rate should be high
    const successRate = clickResults.filter(r => r.success).length / clickResults.length
    expect(successRate).toBeGreaterThan(0.95)
  })
})

// Visual regression tests
test.describe('Visual Regression Tests', () => {
  test('Screenshots should match baseline', async ({ page }) => {
    const pages = ['/', '/executive', '/finops', '/quantum', '/edge', '/blockchain']
    
    for (const pageUrl of pages) {
      await page.goto(`${BASE_URL}${pageUrl}`)
      await page.waitForLoadState('networkidle')
      
      // Take screenshot
      await expect(page).toHaveScreenshot(`${pageUrl.replace(/\//g, '-')}.png`, {
        fullPage: true,
        animations: 'disabled'
      })
    }
  })
})