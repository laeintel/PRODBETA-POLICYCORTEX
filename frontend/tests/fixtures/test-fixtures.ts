/**
 * PolicyCortex E2E Test Fixtures
 * Common fixtures and page objects for testing
 */

import { test as base, expect } from '@playwright/test';
import { AuthPage } from '../pages/auth.page';
import { DashboardPage } from '../pages/dashboard.page';
import { ITSMPage } from '../pages/itsm.page';
import { AIPage } from '../pages/ai.page';
import { GovernancePage } from '../pages/governance.page';

// Define custom fixtures
type PolicyCortexFixtures = {
  authPage: AuthPage;
  dashboardPage: DashboardPage;
  itsmPage: ITSMPage;
  aiPage: AIPage;
  governancePage: GovernancePage;
  authenticatedContext: void;
};

// Extend base test with our fixtures
export const test = base.extend<PolicyCortexFixtures>({
  // Page object fixtures
  authPage: async ({ page }, use) => {
    await use(new AuthPage(page));
  },
  
  dashboardPage: async ({ page }, use) => {
    await use(new DashboardPage(page));
  },
  
  itsmPage: async ({ page }, use) => {
    await use(new ITSMPage(page));
  },
  
  aiPage: async ({ page }, use) => {
    await use(new AIPage(page));
  },
  
  governancePage: async ({ page }, use) => {
    await use(new GovernancePage(page));
  },
  
  // Authenticated context fixture
  authenticatedContext: async ({ page, context }, use) => {
    // Set up authentication state
    await context.addCookies([
      {
        name: 'auth-token',
        value: process.env.TEST_AUTH_TOKEN || 'test-token',
        domain: 'localhost',
        path: '/',
        httpOnly: true,
        secure: false,
        sameSite: 'Lax'
      }
    ]);
    
    // Add Azure MSAL tokens to localStorage
    await page.addInitScript(() => {
      localStorage.setItem('msal.idtoken', JSON.stringify({
        secret: 'test-id-token',
        expiresOn: Date.now() + 3600000
      }));
      localStorage.setItem('msal.accesstoken', JSON.stringify({
        secret: 'test-access-token',
        expiresOn: Date.now() + 3600000
      }));
    });
    
    await use();
  }
});

export { expect };

// Test data helpers
export const testData = {
  users: {
    admin: {
      email: 'admin@policycortex.test',
      password: 'TestAdmin123!',
      role: 'admin'
    },
    user: {
      email: 'user@policycortex.test',
      password: 'TestUser123!',
      role: 'user'
    },
    readonly: {
      email: 'readonly@policycortex.test',
      password: 'TestReadOnly123!',
      role: 'readonly'
    }
  },
  
  resources: {
    vm: {
      name: 'test-vm-001',
      type: 'Microsoft.Compute/virtualMachines',
      location: 'eastus',
      tags: {
        environment: 'test',
        owner: 'e2e-tests'
      }
    },
    storage: {
      name: 'teststorage001',
      type: 'Microsoft.Storage/storageAccounts',
      location: 'westus',
      sku: 'Standard_LRS'
    }
  },
  
  policies: {
    tagging: {
      name: 'require-environment-tag',
      description: 'Require environment tag on all resources',
      effect: 'deny'
    },
    location: {
      name: 'allowed-locations',
      description: 'Restrict resource locations',
      allowedLocations: ['eastus', 'westus']
    }
  }
};

// Common test selectors
export const selectors = {
  navigation: {
    mainMenu: '[data-testid="main-menu"]',
    sidebar: '[data-testid="sidebar"]',
    breadcrumb: '[data-testid="breadcrumb"]',
    userMenu: '[data-testid="user-menu"]',
    themeToggle: '[data-testid="theme-toggle"]'
  },
  
  dashboard: {
    metricCard: '[data-testid="metric-card"]',
    chart: '[data-testid="chart-container"]',
    viewToggle: '[data-testid="view-toggle"]',
    exportButton: '[data-testid="export-button"]',
    refreshButton: '[data-testid="refresh-button"]'
  },
  
  forms: {
    input: 'input[type="text"]',
    email: 'input[type="email"]',
    password: 'input[type="password"]',
    submit: 'button[type="submit"]',
    cancel: 'button[data-testid="cancel"]',
    error: '[data-testid="error-message"]',
    success: '[data-testid="success-message"]'
  },
  
  table: {
    header: 'thead',
    row: 'tbody tr',
    cell: 'td',
    sortButton: '[data-testid="sort-button"]',
    filterInput: '[data-testid="filter-input"]',
    pagination: '[data-testid="pagination"]'
  },
  
  modal: {
    container: '[role="dialog"]',
    title: '[data-testid="modal-title"]',
    body: '[data-testid="modal-body"]',
    closeButton: '[data-testid="modal-close"]',
    confirmButton: '[data-testid="modal-confirm"]'
  }
};

// Performance thresholds
export const performanceThresholds = {
  pageLoad: 3000, // 3 seconds
  apiResponse: 1000, // 1 second
  chartRender: 500, // 500ms
  searchResponse: 200, // 200ms
  
  // Patent-specific thresholds
  patents: {
    correlation: 100, // Patent #1: <100ms
    conversation: 1000, // Patent #2: <1s for NLP
    unified: 500, // Patent #3: <500ms
    prediction: 100 // Patent #4: <100ms
  }
};

// Accessibility test helpers
export const a11yConfig = {
  rules: {
    // Disable some rules that may have false positives
    'color-contrast': { enabled: false }, // We'll test this separately
    'heading-order': { enabled: true },
    'landmark-one-main': { enabled: true },
    'page-has-heading-one': { enabled: true },
    'region': { enabled: true },
    'button-name': { enabled: true },
    'link-name': { enabled: true },
    'image-alt': { enabled: true }
  }
};

// Visual regression config
export const visualConfig = {
  threshold: 0.2, // 20% difference threshold
  animations: 'disabled',
  clip: undefined,
  fullPage: false,
  mask: [
    // Mask dynamic content
    '[data-testid="timestamp"]',
    '[data-testid="random-id"]',
    '.loading-spinner'
  ],
  maxDiffPixels: 100,
  maxDiffPixelRatio: 0.1
};