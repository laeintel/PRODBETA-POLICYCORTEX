import { test, expect } from '@playwright/test';

test.describe('PolicyCortex Comprehensive Navigation Tests', () => {
  
  test.beforeEach(async ({ page }) => {
    // Navigate to the main dashboard
    await page.goto('http://localhost:3000/tactical');
    await page.waitForLoadState('networkidle');
  });

  test.describe('Main Dashboard Navigation', () => {
    test('should navigate to all main sections from dashboard cards', async ({ page }) => {
      // Test Governance card navigation
      await page.click('text=Governance & Compliance');
      await expect(page).toHaveURL(/\/governance/);
      await page.goBack();

      // Test Security card navigation
      await page.click('text=Security & Access Management');
      await expect(page).toHaveURL(/\/security/);
      await page.goBack();

      // Test Operations card navigation
      await page.click('text=Operations & Monitoring');
      await expect(page).toHaveURL(/\/operations/);
      await page.goBack();

      // Test DevOps card navigation
      await page.click('text=DevOps & CI/CD');
      await expect(page).toHaveURL(/\/devops/);
      await page.goBack();

      // Test AI Intelligence card navigation
      await page.click('text=AI Intelligence Hub');
      await expect(page).toHaveURL(/\/ai/);
      await page.goBack();

      // Test Audit Trail card navigation
      await page.click('text=Audit Trail & History');
      await expect(page).toHaveURL(/\/audit/);
    });

    test('should expand and navigate through quick access items', async ({ page }) => {
      // Expand Governance quick access
      await page.click('button:has-text("Quick Access"):near(:text("Governance"))');
      await page.click('text=Policies & Compliance');
      await expect(page).toHaveURL(/\/governance\/compliance/);
      await page.goBack();

      // Expand Security quick access
      await page.click('button:has-text("Quick Access"):near(:text("Security"))');
      await page.click('text=Role Management (RBAC)');
      await expect(page).toHaveURL(/\/security\/rbac/);
    });

    test('should navigate via system metrics cards', async ({ page }) => {
      await page.click('text=CPU Usage');
      await expect(page).toHaveURL(/\/operations\/monitoring/);
      await page.goBack();

      await page.click('text=Storage');
      await expect(page).toHaveURL(/\/operations\/resources/);
    });

    test('should navigate via quick actions', async ({ page }) => {
      await page.click('button:has-text("Security Scan")');
      await expect(page).toHaveURL(/\/security/);
      await page.goBack();

      await page.click('button:has-text("Restart Services")');
      await expect(page).toHaveURL(/\/operations\/automation/);
    });
  });

  test.describe('Sidebar Navigation', () => {
    test('should navigate through all sidebar menu items', async ({ page }) => {
      // Test main menu items
      const menuItems = [
        { text: 'Dashboard', url: '/tactical' },
        { text: 'Governance', url: '/governance' },
        { text: 'Security & Access', url: '/security' },
        { text: 'Operations', url: '/operations' },
        { text: 'DevOps & CI/CD', url: '/devops' },
        { text: 'AI Intelligence', url: '/ai' },
        { text: 'Audit Trail', url: '/audit' },
        { text: 'Settings', url: '/settings' }
      ];

      for (const item of menuItems) {
        await page.click(`text=${item.text}`);
        await expect(page).toHaveURL(new RegExp(item.url));
      }
    });

    test('should expand and navigate sub-menu items', async ({ page }) => {
      // Expand Security menu
      await page.click('text=Security & Access');
      await expect(page).toHaveURL(/\/security/);
      
      // Navigate to sub-items
      await page.click('text=Identity & Access (IAM)');
      await expect(page).toHaveURL(/\/security.*tab=iam/);

      await page.click('text=Privileged Identity (PIM)');
      await expect(page).toHaveURL(/\/security\/pim/);

      await page.click('text=Zero Trust Policies');
      await expect(page).toHaveURL(/\/security\/zero-trust/);
    });

    test('should maintain sidebar context when navigating', async ({ page }) => {
      // Navigate to Governance
      await page.click('text=Governance');
      
      // Check if Governance is highlighted
      const governanceItem = page.locator('text=Governance').first();
      await expect(governanceItem).toHaveClass(/bg-primary/);
      
      // Sub-sections should be visible
      await expect(page.locator('text=Policies & Compliance')).toBeVisible();
      await expect(page.locator('text=Risk Management')).toBeVisible();
    });
  });

  test.describe('Governance Section Navigation', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('http://localhost:3000/governance');
    });

    test('should navigate through governance tabs', async ({ page }) => {
      // Test tab navigation
      await page.click('text=Risk Management');
      await expect(page).toHaveURL(/tab=risk/);

      await page.click('text=Cost Optimization');
      await expect(page).toHaveURL(/tab=cost/);

      await page.click('text=Policy Templates');
      await expect(page).toHaveURL(/tab=policies/);
    });

    test('should navigate via governance cards', async ({ page }) => {
      await page.click('div:has-text("Compliance Score"):has-text("94%")');
      await expect(page).toHaveURL(/\/governance.*tab=compliance/);

      await page.click('div:has-text("Risk Assessment"):has-text("View Risks")');
      await expect(page).toHaveURL(/tab=risk/);
    });
  });

  test.describe('Security Section Navigation', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('http://localhost:3000/security');
    });

    test('should navigate through security tabs', async ({ page }) => {
      await page.click('text=RBAC');
      await expect(page).toHaveURL(/tab=rbac/);

      await page.click('text=PIM');
      await expect(page).toHaveURL(/tab=pim/);

      await page.click('text=Conditional Access');
      await expect(page).toHaveURL(/tab=conditional-access/);

      await page.click('text=Zero Trust');
      await expect(page).toHaveURL(/tab=zero-trust/);
    });

    test('should navigate via security cards', async ({ page }) => {
      await page.click('div:has-text("Identity & Access Management")');
      await expect(page).toHaveURL(/tab=iam/);

      await page.click('div:has-text("Privileged Identity Management")');
      await expect(page).toHaveURL(/tab=pim/);
    });
  });

  test.describe('Operations Section Navigation', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('http://localhost:3000/operations');
    });

    test('should navigate through operations cards', async ({ page }) => {
      await page.click('div:has-text("Resource Management"):has-text("342 Resources")');
      await expect(page).toHaveURL(/\/operations\/resources/);
      await page.goBack();

      await page.click('div:has-text("Real-time Monitoring")');
      await expect(page).toHaveURL(/\/operations\/monitoring/);
      await page.goBack();

      await page.click('div:has-text("Automation Workflows")');
      await expect(page).toHaveURL(/\/operations\/automation/);
    });

    test('should navigate via quick actions', async ({ page }) => {
      await page.click('button:has-text("View All Resources")');
      await expect(page).toHaveURL(/\/operations\/resources/);

      await page.goBack();
      await page.click('button:has-text("Configure Alerts")');
      await expect(page).toHaveURL(/\/operations\/alerts/);
    });
  });

  test.describe('DevOps Section Navigation', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('http://localhost:3000/devops');
    });

    test('should navigate through DevOps cards', async ({ page }) => {
      await page.click('div:has-text("CI/CD Pipelines"):has-text("42 Active")');
      await expect(page).toHaveURL(/\/devops\/pipelines/);
      await page.goBack();

      await page.click('div:has-text("Release Management")');
      await expect(page).toHaveURL(/\/devops\/releases/);
      await page.goBack();

      await page.click('div:has-text("Deployment History")');
      await expect(page).toHaveURL(/\/devops\/deployments/);
    });
  });

  test.describe('AI Section Navigation', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('http://localhost:3000/ai');
    });

    test('should navigate through AI feature cards', async ({ page }) => {
      await page.click('div:has-text("Predictive Compliance Engine")');
      await expect(page).toHaveURL(/\/ai\/predictive/);
      await page.goBack();

      await page.click('div:has-text("Cross-Domain Correlation Analysis")');
      await expect(page).toHaveURL(/\/ai\/correlations/);
      await page.goBack();

      await page.click('div:has-text("Conversational AI Interface")');
      await expect(page).toHaveURL(/\/ai\/chat/);
      await page.goBack();

      await page.click('div:has-text("Unified Platform Metrics")');
      await expect(page).toHaveURL(/\/ai\/unified/);
    });

    test('should display patent badges', async ({ page }) => {
      await expect(page.locator('text=Patent #1')).toBeVisible();
      await expect(page.locator('text=Patent #2')).toBeVisible();
      await expect(page.locator('text=Patent #3')).toBeVisible();
      await expect(page.locator('text=Patent #4')).toBeVisible();
    });
  });

  test.describe('Alert and Activity Navigation', () => {
    test('should navigate from alert items', async ({ page }) => {
      // Click on an alert to open modal
      await page.click('text=Database Connection Failure');
      await expect(page.locator('text=Affected Resources')).toBeVisible();
      
      // Click escalate to navigate
      await page.click('button:has-text("Escalate")');
      await expect(page).toHaveURL(/\/operations\/notifications/);
    });

    test('should navigate from recent activities', async ({ page }) => {
      await page.click('text=Policy PCI-DSS-2024 updated');
      await expect(page).toHaveURL(/\/governance\/policies/);
      await page.goBack();

      await page.click('text=Deployment to production completed');
      await expect(page).toHaveURL(/\/devops\/deployments/);
    });

    test('should navigate from cost summary', async ({ page }) => {
      await page.click('button:has-text("View Cost Details")');
      await expect(page).toHaveURL(/\/governance\/cost/);
    });
  });

  test.describe('Back Navigation', () => {
    test('should navigate back to Command Center from all sections', async ({ page }) => {
      const sections = [
        '/governance',
        '/security',
        '/operations',
        '/devops',
        '/ai'
      ];

      for (const section of sections) {
        await page.goto(`http://localhost:3000${section}`);
        await page.click('button:has-text("Back to Command Center")');
        await expect(page).toHaveURL(/\/tactical/);
      }
    });
  });

  test.describe('Mobile Navigation', () => {
    test.beforeEach(async ({ page }) => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
    });

    test('should toggle mobile menu', async ({ page }) => {
      // Menu should be hidden initially
      await expect(page.locator('nav').first()).not.toBeVisible();
      
      // Open menu
      await page.click('button[aria-label="Toggle menu"]');
      await expect(page.locator('nav').first()).toBeVisible();
      
      // Navigate via mobile menu
      await page.click('text=Security & Access');
      await expect(page).toHaveURL(/\/security/);
    });
  });

  test.describe('Command Palette Navigation', () => {
    test('should navigate via command palette', async ({ page }) => {
      // Open command palette
      await page.click('text=Quick Actions');
      await expect(page.locator('text=Type a command or search')).toBeVisible();
      
      // Click a quick action
      await page.click('text=Chat with AI');
      await expect(page).toHaveURL(/\/ai\/chat/);
    });
  });

  test.describe('Settings Navigation', () => {
    test('should navigate to settings page', async ({ page }) => {
      await page.click('button:has-text("Settings")');
      await expect(page).toHaveURL(/\/settings/);
    });
  });
});

test.describe('Navigation Performance', () => {
  test('should load pages quickly', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('http://localhost:3000/tactical');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000); // Should load in under 3 seconds
  });

  test('should handle rapid navigation', async ({ page }) => {
    await page.goto('http://localhost:3000/tactical');
    
    // Rapidly navigate between sections
    const sections = ['/governance', '/security', '/operations', '/devops', '/ai'];
    
    for (let i = 0; i < 3; i++) {
      for (const section of sections) {
        await page.goto(`http://localhost:3000${section}`);
        await expect(page).toHaveURL(new RegExp(section));
      }
    }
  });
});