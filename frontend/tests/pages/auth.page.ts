/**
 * Authentication Page Object Model
 */

import { Page, expect } from '@playwright/test';
import { selectors } from '../fixtures/test-fixtures';

export class AuthPage {
  constructor(private page: Page) {}
  
  async goto() {
    await this.page.goto('/auth/login');
    await this.page.waitForLoadState('networkidle');
  }
  
  async login(email: string, password: string) {
    await this.fillEmail(email);
    await this.fillPassword(password);
    await this.clickLogin();
  }
  
  async fillEmail(email: string) {
    const emailInput = this.page.locator(selectors.forms.email);
    await emailInput.fill(email);
  }
  
  async fillPassword(password: string) {
    const passwordInput = this.page.locator(selectors.forms.password);
    await passwordInput.fill(password);
  }
  
  async clickLogin() {
    const loginButton = this.page.locator(selectors.forms.submit);
    await Promise.all([
      this.page.waitForResponse(resp => 
        resp.url().includes('/api/auth') && resp.status() === 200
      ),
      loginButton.click()
    ]);
  }
  
  async loginWithAzure() {
    const azureButton = this.page.locator('[data-testid="azure-login"]');
    await azureButton.click();
    
    // Handle Azure AD login flow
    await this.page.waitForURL(/login\.microsoftonline\.com/);
    await this.page.fill('input[name="loginfmt"]', process.env.AZURE_TEST_EMAIL || 'test@example.com');
    await this.page.click('input[type="submit"]');
    
    await this.page.waitForSelector('input[name="passwd"]');
    await this.page.fill('input[name="passwd"]', process.env.AZURE_TEST_PASSWORD || 'TestPassword123!');
    await this.page.click('input[type="submit"]');
    
    // Handle MFA if required
    if (await this.page.locator('text="Verify your identity"').isVisible({ timeout: 5000 }).catch(() => false)) {
      // Handle MFA verification
      console.log('MFA required - handling verification');
    }
    
    // Wait for redirect back to app
    await this.page.waitForURL(/localhost:3000|policycortex/);
  }
  
  async logout() {
    const userMenu = this.page.locator(selectors.navigation.userMenu);
    await userMenu.click();
    
    const logoutButton = this.page.locator('[data-testid="logout-button"]');
    await logoutButton.click();
    
    await this.page.waitForURL('/auth/login');
  }
  
  async verifyLoggedIn() {
    await expect(this.page).toHaveURL(/dashboard|home/);
    await expect(this.page.locator(selectors.navigation.userMenu)).toBeVisible();
  }
  
  async verifyLoggedOut() {
    await expect(this.page).toHaveURL('/auth/login');
    await expect(this.page.locator(selectors.forms.email)).toBeVisible();
  }
  
  async verifyError(message: string) {
    const errorElement = this.page.locator(selectors.forms.error);
    await expect(errorElement).toContainText(message);
  }
  
  async register(email: string, password: string, confirmPassword: string) {
    await this.page.goto('/auth/register');
    
    await this.page.fill('[data-testid="register-email"]', email);
    await this.page.fill('[data-testid="register-password"]', password);
    await this.page.fill('[data-testid="register-confirm-password"]', confirmPassword);
    
    const registerButton = this.page.locator('[data-testid="register-submit"]');
    await registerButton.click();
  }
  
  async requestPasswordReset(email: string) {
    await this.page.goto('/auth/forgot-password');
    
    await this.page.fill('[data-testid="reset-email"]', email);
    const resetButton = this.page.locator('[data-testid="reset-submit"]');
    await resetButton.click();
    
    await expect(this.page.locator(selectors.forms.success)).toContainText('Password reset email sent');
  }
  
  async verifyMFASetup() {
    await expect(this.page.locator('[data-testid="mfa-qr-code"]')).toBeVisible();
    await expect(this.page.locator('[data-testid="mfa-secret"]')).toBeVisible();
  }
  
  async enterMFACode(code: string) {
    await this.page.fill('[data-testid="mfa-code"]', code);
    await this.page.click('[data-testid="mfa-verify"]');
  }
  
  async checkSessionValidity() {
    const response = await this.page.request.get('/api/auth/session');
    return response.ok();
  }
  
  async measureLoginPerformance() {
    const startTime = Date.now();
    await this.login('test@example.com', 'TestPassword123!');
    const endTime = Date.now();
    return endTime - startTime;
  }
}