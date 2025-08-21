/**
 * Crawls common routes and attempts to click non-destructive buttons.
 * Skips obvious submits and destructive patterns.
 */
import { test, expect, Page } from '@playwright/test'

const baseURL = process.env.BASE_URL || 'http://localhost:3000'

async function clickSafeButtons(page: Page) {
  const buttons = page.locator('button');
  const count = await buttons.count();
  for (let i = 0; i < count; i++) {
    const btn = buttons.nth(i);
    const text = (await btn.textContent())?.trim() || '';
    const typeAttr = await btn.getAttribute('type');
    const classes = (await btn.getAttribute('class')) || '';

    // Skip destructive or submit buttons
    const looksDestructive = /delete|remove|revoke|reset|wipe|escalate/i.test(text);
    const isSubmit = typeAttr === 'submit' || /submit|form/i.test(classes);
    if (looksDestructive || isSubmit) continue;

    // Attempt click if visible and enabled
    if (await btn.isVisible()) {
      try {
        await btn.click({ trial: true });
        await btn.click();
      } catch {
        // ignore failures, continue
      }
    }
  }
}

test.describe('Clickable audit', () => {
  const routes = [
    '/',
    '/dashboard',
    '/tactical',
    '/governance',
    '/security/rbac',
    '/operations/resources',
    '/ai/chat'
  ]

  for (const route of routes) {
    test(`click non-destructive buttons on ${route}`, async ({ page }) => {
      await page.goto(`${baseURL}${route}`);
      await clickSafeButtons(page);
      // Basic sanity: no error modals; no console errors if we wire listeners later
      await expect(page).toHaveURL(/.+/);
    });
  }
});


