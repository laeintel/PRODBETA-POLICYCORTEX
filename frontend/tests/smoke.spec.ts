/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { test, expect } from '@playwright/test'

const baseURL = process.env.BASE_URL || 'http://localhost:3000'

test.describe('MVP smoke', () => {
  test('home loads and shows demo banner', async ({ page }) => {
    await page.goto(baseURL)
    await expect(page.locator('text=PolicyCortex')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('text=Simulated Mode').first()).toBeVisible()
  })

  test('dashboard loads', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`)
    await expect(page.locator('text=Governance Dashboard')).toBeVisible()
  })

  test('policies page loads', async ({ page }) => {
    await page.goto(`${baseURL}/policies`)
    await expect(page.locator('text=Policies')).toBeVisible()
  })
})
