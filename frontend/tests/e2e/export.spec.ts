import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

const BASE = process.env.BASE_URL || 'http://localhost:3000';

test('Export Evidence downloads artifact JSON and contains hash/root/proof', async ({ page }) => {
  await page.goto(`${BASE}/audit`);
  await page.getByLabel('Evidence hash').fill('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  await page.getByRole('button', { name: /verify/i }).click();
  await expect(page.getByText('✔ Verified')).toBeVisible();
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: /Export Evidence/i }).click(),
  ]);
  const file = await download.path();
  expect(file).toBeTruthy();
  const txt = await fs.readFile(file!, 'utf8');
  const j = JSON.parse(txt);
  expect(j).toHaveProperty('contentHash', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  expect(j).toHaveProperty('merkleRoot');
  expect(Array.isArray(j.proof)).toBeTruthy();
});