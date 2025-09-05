import { test, expect } from '@playwright/test';

const BASE = process.env.BASE_URL || 'http://localhost:3000';

test('Audit page verifies seeded hash and shows proof', async ({ page }) => {
  await page.goto(`${BASE}/audit`);
  await expect(page.getByText('Audit Verification')).toBeVisible();
  await page.getByLabel('Evidence hash').fill('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  await page.getByRole('button', { name: /verify/i }).click();
  await expect(page.getByText('✔ Verified')).toBeVisible();
  await page.getByText('View Merkle proof').click();
  await expect(page.locator('pre')).toContainText('bbbbbbbbbbbbbbbb'); // proof contains sibling hash from seeded pair
});