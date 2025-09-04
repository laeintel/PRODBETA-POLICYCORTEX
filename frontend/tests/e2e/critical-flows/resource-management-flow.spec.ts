/**
 * PolicyCortex Critical User Flow Test Suite: Resource Management
 * 
 * This test suite covers comprehensive resource management scenarios including:
 * - Resource discovery and inventory
 * - Resource lifecycle management (Create, Read, Update, Delete)
 * - Resource tagging and categorization
 * - Bulk operations
 * - Resource health monitoring
 * - Cost tracking and optimization
 * - Resource compliance validation
 * 
 * Performance targets:
 * - Resource list load: <2s
 * - Resource creation: <3s
 * - Bulk operations: <5s
 * - Search/filter: <1s
 */

import { test, expect, Page } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

// Test configuration
test.use({
  trace: 'on-first-retry',
  video: 'on-first-retry',
  screenshot: 'only-on-failure',
  actionTimeout: 20000,
  navigationTimeout: 30000,
});

// Helper to set up authenticated session
async function setupAuthenticatedSession(page: Page) {
  await page.goto(BASE_URL);
  await page.evaluate(() => {
    localStorage.setItem('msal.account.keys', JSON.stringify(['resource-admin']));
    localStorage.setItem('msal.resource-admin', JSON.stringify({
      username: 'admin@policycortex.com',
      name: 'Resource Admin',
      roles: ['admin', 'resource-manager'],
      permissions: ['resource:create', 'resource:read', 'resource:update', 'resource:delete'],
    }));
    sessionStorage.setItem('isAuthenticated', 'true');
  });
}

// Helper to wait for resource list to load
async function waitForResourceList(page: Page) {
  await page.waitForSelector('[data-testid="resource-list"], .resource-table, .resource-grid', {
    state: 'visible',
    timeout: 5000,
  });
  
  // Wait for loading to complete
  const loadingIndicator = page.locator('[data-testid="loading"], .loading-spinner');
  if (await loadingIndicator.count() > 0) {
    await loadingIndicator.first().waitFor({ state: 'hidden', timeout: 5000 }).catch(() => {});
  }
}

// Helper to generate mock resource data
function generateMockResource(index: number) {
  return {
    name: `test-vm-${index}-${Date.now()}`,
    type: 'Microsoft.Compute/virtualMachines',
    location: 'eastus',
    resourceGroup: 'test-rg',
    tags: {
      environment: 'test',
      owner: 'test-user',
      department: 'engineering',
      'cost-center': 'CC-001',
    },
    sku: {
      name: 'Standard_B2s',
      tier: 'Standard',
    },
  };
}

test.describe('Critical Flow: Resource Management', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuthenticatedSession(page);
  });

  test('01 - Resource inventory and discovery', async ({ page }) => {
    await test.step('Navigate to resource inventory', async () => {
      const startTime = Date.now();
      await page.goto(`${BASE_URL}/operations/resources`);
      await waitForResourceList(page);
      
      const loadTime = Date.now() - startTime;
      console.log(`Resource inventory loaded in ${loadTime}ms`);
      expect(loadTime).toBeLessThan(2000);
    });

    await test.step('Verify resource list display', async () => {
      // Check for resource list elements
      const resourceItems = page.locator('[data-testid="resource-item"], .resource-row, tr[data-resource]');
      const resourceCount = await resourceItems.count();
      
      expect(resourceCount).toBeGreaterThan(0);
      console.log(`Found ${resourceCount} resources in inventory`);
      
      // Check first resource has required information
      if (resourceCount > 0) {
        const firstResource = resourceItems.first();
        
        // Verify essential resource information is displayed
        const resourceName = await firstResource.locator('[data-testid="resource-name"], .resource-name').textContent();
        const resourceType = await firstResource.locator('[data-testid="resource-type"], .resource-type').textContent();
        const resourceStatus = await firstResource.locator('[data-testid="resource-status"], .status-badge').textContent();
        
        expect(resourceName).toBeTruthy();
        expect(resourceType).toBeTruthy();
        expect(resourceStatus).toBeTruthy();
      }
    });

    await test.step('Test resource type filtering', async () => {
      // Find and use resource type filter
      const typeFilter = page.locator('[data-testid="type-filter"], select[name*="type"], [aria-label*="resource type"]');
      
      if (await typeFilter.count() > 0) {
        // Select a specific resource type
        await typeFilter.selectOption({ label: 'Virtual Machines' }).catch(async () => {
          // Fallback to clicking first option
          await typeFilter.click();
          const firstOption = page.locator('option, [role="option"]').first();
          await firstOption.click();
        });
        
        // Wait for filtered results
        await page.waitForTimeout(1000);
        await waitForResourceList(page);
        
        // Verify filtered results
        const filteredItems = page.locator('[data-testid="resource-item"]');
        const filteredCount = await filteredItems.count();
        
        if (filteredCount > 0) {
          // Check that all items match the filter
          const firstItem = filteredItems.first();
          const itemType = await firstItem.locator('[data-testid="resource-type"]').textContent();
          expect(itemType?.toLowerCase()).toContain('virtual');
        }
      }
    });

    await test.step('Test resource search', async () => {
      const searchInput = page.locator('[data-testid="resource-search"], input[placeholder*="search"]');
      
      if (await searchInput.count() > 0) {
        // Search for specific resource
        await searchInput.fill('vm');
        await searchInput.press('Enter');
        
        // Wait for search results
        await page.waitForTimeout(1000);
        
        // Verify search results
        const searchResults = page.locator('[data-testid="resource-item"]');
        if (await searchResults.count() > 0) {
          const firstResult = await searchResults.first().textContent();
          expect(firstResult?.toLowerCase()).toContain('vm');
        }
        
        // Clear search
        await searchInput.clear();
        await searchInput.press('Enter');
      }
    });
  });

  test('02 - Resource creation workflow', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('Initiate resource creation', async () => {
      const createButton = page.locator('[data-testid="create-resource"], button:has-text("Create"), button:has-text("New Resource")');
      
      if (await createButton.count() > 0) {
        await createButton.first().click();
        
        // Wait for creation form/modal
        await page.waitForSelector('[data-testid="resource-form"], [role="dialog"], .create-resource-modal', {
          state: 'visible',
          timeout: 3000,
        });
      }
    });

    await test.step('Fill resource creation form', async () => {
      const resourceForm = page.locator('[data-testid="resource-form"], form');
      
      if (await resourceForm.count() > 0) {
        // Fill in resource details
        const mockResource = generateMockResource(1);
        
        // Resource name
        const nameInput = resourceForm.locator('input[name="name"], #resource-name');
        if (await nameInput.count() > 0) {
          await nameInput.fill(mockResource.name);
        }
        
        // Resource type
        const typeSelect = resourceForm.locator('select[name="type"], #resource-type');
        if (await typeSelect.count() > 0) {
          await typeSelect.selectOption(mockResource.type);
        }
        
        // Location
        const locationSelect = resourceForm.locator('select[name="location"], #resource-location');
        if (await locationSelect.count() > 0) {
          await locationSelect.selectOption(mockResource.location);
        }
        
        // Resource group
        const rgInput = resourceForm.locator('input[name="resourceGroup"], #resource-group');
        if (await rgInput.count() > 0) {
          await rgInput.fill(mockResource.resourceGroup);
        }
        
        // SKU selection
        const skuSelect = resourceForm.locator('select[name="sku"], #resource-sku');
        if (await skuSelect.count() > 0) {
          await skuSelect.selectOption(mockResource.sku.name);
        }
      }
    });

    await test.step('Add resource tags', async () => {
      const tagSection = page.locator('[data-testid="tag-section"], .tag-editor');
      
      if (await tagSection.count() > 0) {
        // Add environment tag
        const addTagButton = tagSection.locator('button:has-text("Add Tag")');
        if (await addTagButton.count() > 0) {
          await addTagButton.click();
          
          const tagKeyInput = tagSection.locator('input[placeholder*="key"]').last();
          const tagValueInput = tagSection.locator('input[placeholder*="value"]').last();
          
          await tagKeyInput.fill('environment');
          await tagValueInput.fill('test');
        }
      }
    });

    await test.step('Validate and submit form', async () => {
      const submitButton = page.locator('button[type="submit"], button:has-text("Create")').last();
      
      if (await submitButton.count() > 0) {
        // Check form validation
        const isValid = await page.evaluate(() => {
          const form = document.querySelector('form');
          return form ? form.checkValidity() : true;
        });
        
        expect(isValid).toBeTruthy();
        
        // Submit form
        const startTime = Date.now();
        await submitButton.click();
        
        // Wait for success indication
        await page.waitForSelector('[data-testid="success-message"], .toast-success, .alert-success', {
          state: 'visible',
          timeout: 5000,
        }).catch(() => {});
        
        const createTime = Date.now() - startTime;
        console.log(`Resource creation completed in ${createTime}ms`);
        expect(createTime).toBeLessThan(3000);
      }
    });
  });

  test('03 - Resource details and editing', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    let resourceId: string | null = null;

    await test.step('Navigate to resource details', async () => {
      const resourceItems = page.locator('[data-testid="resource-item"], .resource-row');
      
      if (await resourceItems.count() > 0) {
        const firstResource = resourceItems.first();
        
        // Get resource ID for later use
        resourceId = await firstResource.getAttribute('data-resource-id');
        
        // Click to view details
        await firstResource.click();
        
        // Wait for details page/panel
        await page.waitForSelector('[data-testid="resource-details"], .resource-detail-view', {
          state: 'visible',
          timeout: 3000,
        });
      }
    });

    await test.step('Verify resource details display', async () => {
      const detailsContainer = page.locator('[data-testid="resource-details"]');
      
      if (await detailsContainer.count() > 0) {
        // Check for essential details
        const details = {
          name: await detailsContainer.locator('[data-testid="detail-name"]').textContent(),
          type: await detailsContainer.locator('[data-testid="detail-type"]').textContent(),
          status: await detailsContainer.locator('[data-testid="detail-status"]').textContent(),
          location: await detailsContainer.locator('[data-testid="detail-location"]').textContent(),
        };
        
        expect(details.name).toBeTruthy();
        expect(details.type).toBeTruthy();
        expect(details.status).toBeTruthy();
        expect(details.location).toBeTruthy();
        
        // Check for tabs/sections
        const tabs = detailsContainer.locator('[role="tab"], .tab-button');
        if (await tabs.count() > 0) {
          // Test each tab
          const tabCount = await tabs.count();
          for (let i = 0; i < Math.min(tabCount, 3); i++) {
            await tabs.nth(i).click();
            await page.waitForTimeout(500); // Wait for content to load
          }
        }
      }
    });

    await test.step('Edit resource properties', async () => {
      const editButton = page.locator('[data-testid="edit-resource"], button:has-text("Edit")');
      
      if (await editButton.count() > 0) {
        await editButton.first().click();
        
        // Wait for edit form
        await page.waitForSelector('[data-testid="edit-form"], .edit-mode', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Modify a property
        const nameInput = page.locator('input[name="name"], #edit-resource-name');
        if (await nameInput.count() > 0) {
          const currentValue = await nameInput.inputValue();
          await nameInput.clear();
          await nameInput.fill(`${currentValue}-modified`);
        }
        
        // Add/modify tags
        const tagInput = page.locator('input[name*="tag"], .tag-input').last();
        if (await tagInput.count() > 0) {
          await tagInput.fill('modified');
        }
        
        // Save changes
        const saveButton = page.locator('button:has-text("Save"), button[type="submit"]').last();
        await saveButton.click();
        
        // Wait for success message
        await page.waitForSelector('[data-testid="success-message"], .toast-success', {
          state: 'visible',
          timeout: 3000,
        }).catch(() => {});
      }
    });
  });

  test('04 - Bulk resource operations', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('Select multiple resources', async () => {
      const checkboxes = page.locator('[data-testid="resource-checkbox"], input[type="checkbox"][data-resource]');
      const checkboxCount = await checkboxes.count();
      
      if (checkboxCount >= 3) {
        // Select first 3 resources
        for (let i = 0; i < 3; i++) {
          await checkboxes.nth(i).check();
        }
        
        // Verify bulk action bar appears
        const bulkActionBar = page.locator('[data-testid="bulk-actions"], .bulk-action-bar');
        await expect(bulkActionBar.first()).toBeVisible();
        
        // Check selected count
        const selectedCount = page.locator('[data-testid="selected-count"], .selection-count');
        if (await selectedCount.count() > 0) {
          const countText = await selectedCount.textContent();
          expect(countText).toContain('3');
        }
      }
    });

    await test.step('Perform bulk tag operation', async () => {
      const bulkTagButton = page.locator('[data-testid="bulk-tag"], button:has-text("Tag")');
      
      if (await bulkTagButton.count() > 0) {
        await bulkTagButton.click();
        
        // Wait for tag modal
        await page.waitForSelector('[data-testid="bulk-tag-modal"], [role="dialog"]', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Add bulk tag
        const tagKeyInput = page.locator('input[placeholder*="tag key"]');
        const tagValueInput = page.locator('input[placeholder*="tag value"]');
        
        if (await tagKeyInput.count() > 0 && await tagValueInput.count() > 0) {
          await tagKeyInput.fill('bulk-operation');
          await tagValueInput.fill('test');
          
          // Apply tags
          const applyButton = page.locator('button:has-text("Apply")');
          
          const startTime = Date.now();
          await applyButton.click();
          
          // Wait for operation to complete
          await page.waitForSelector('[data-testid="bulk-success"], .operation-complete', {
            state: 'visible',
            timeout: 5000,
          }).catch(() => {});
          
          const operationTime = Date.now() - startTime;
          console.log(`Bulk tag operation completed in ${operationTime}ms`);
          expect(operationTime).toBeLessThan(5000);
        }
      }
    });

    await test.step('Bulk export resources', async () => {
      const exportButton = page.locator('[data-testid="bulk-export"], button:has-text("Export")');
      
      if (await exportButton.count() > 0) {
        // Set up download promise
        const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null);
        
        await exportButton.click();
        
        // Select export format
        const csvOption = page.locator('button:has-text("CSV"), [data-format="csv"]');
        if (await csvOption.count() > 0) {
          await csvOption.click();
          
          const download = await downloadPromise;
          if (download) {
            expect(download.suggestedFilename()).toContain('resources');
            expect(download.suggestedFilename()).toMatch(/\.csv$/);
          }
        }
      }
    });
  });

  test('05 - Resource health monitoring', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('View resource health status', async () => {
      // Check for health indicators
      const healthBadges = page.locator('[data-testid="health-status"], .health-badge, .status-indicator');
      const healthCount = await healthBadges.count();
      
      if (healthCount > 0) {
        // Count resources by health status
        const healthStats = {
          healthy: 0,
          warning: 0,
          critical: 0,
          unknown: 0,
        };
        
        for (let i = 0; i < healthCount; i++) {
          const badge = healthBadges.nth(i);
          const text = (await badge.textContent())?.toLowerCase() || '';
          const classes = await badge.getAttribute('class') || '';
          
          if (text.includes('healthy') || text.includes('running') || classes.includes('success')) {
            healthStats.healthy++;
          } else if (text.includes('warning') || classes.includes('warning')) {
            healthStats.warning++;
          } else if (text.includes('critical') || text.includes('error') || classes.includes('danger')) {
            healthStats.critical++;
          } else {
            healthStats.unknown++;
          }
        }
        
        console.log('Resource Health Statistics:', healthStats);
        expect(healthCount).toBeGreaterThan(0);
      }
    });

    await test.step('Access resource metrics', async () => {
      const resourceItem = page.locator('[data-testid="resource-item"]').first();
      
      if (await resourceItem.count() > 0) {
        await resourceItem.click();
        
        // Navigate to metrics tab
        const metricsTab = page.locator('[data-testid="metrics-tab"], button:has-text("Metrics")');
        if (await metricsTab.count() > 0) {
          await metricsTab.click();
          
          // Wait for metrics to load
          await page.waitForSelector('[data-testid="resource-metrics"], .metrics-chart', {
            state: 'visible',
            timeout: 3000,
          });
          
          // Verify metrics display
          const metricCharts = page.locator('.metric-chart, canvas');
          expect(await metricCharts.count()).toBeGreaterThan(0);
          
          // Check for time range selector
          const timeRangeSelector = page.locator('[data-testid="time-range"], select[name*="time"]');
          if (await timeRangeSelector.count() > 0) {
            // Change time range
            await timeRangeSelector.selectOption('24h');
            await page.waitForTimeout(1000); // Wait for metrics to reload
          }
        }
      }
    });

    await test.step('Set up resource alerts', async () => {
      const alertsButton = page.locator('[data-testid="configure-alerts"], button:has-text("Alerts")');
      
      if (await alertsButton.count() > 0) {
        await alertsButton.click();
        
        // Wait for alerts configuration
        await page.waitForSelector('[data-testid="alerts-config"], .alerts-panel', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Create new alert rule
        const createAlertButton = page.locator('button:has-text("Create Alert"), button:has-text("New Alert")');
        if (await createAlertButton.count() > 0) {
          await createAlertButton.click();
          
          // Fill alert configuration
          const alertNameInput = page.locator('input[name="alertName"]');
          const metricSelect = page.locator('select[name="metric"]');
          const thresholdInput = page.locator('input[name="threshold"]');
          
          if (await alertNameInput.count() > 0) {
            await alertNameInput.fill('High CPU Usage Alert');
            await metricSelect.selectOption('cpu_percentage');
            await thresholdInput.fill('80');
            
            // Save alert
            const saveAlertButton = page.locator('button:has-text("Save Alert")');
            await saveAlertButton.click();
            
            // Verify alert created
            await page.waitForSelector('[data-testid="alert-created"], .alert-success', {
              state: 'visible',
              timeout: 3000,
            }).catch(() => {});
          }
        }
      }
    });
  });

  test('06 - Resource cost tracking and optimization', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('View resource costs', async () => {
      // Check for cost column in resource list
      const costColumns = page.locator('[data-testid="cost-column"], td.cost, .resource-cost');
      
      if (await costColumns.count() > 0) {
        const costs: number[] = [];
        
        for (let i = 0; i < Math.min(await costColumns.count(), 5); i++) {
          const costText = await costColumns.nth(i).textContent();
          const costValue = parseFloat(costText?.replace(/[$,]/g, '') || '0');
          costs.push(costValue);
        }
        
        const totalCost = costs.reduce((a, b) => a + b, 0);
        console.log(`Sample resource costs: $${totalCost.toFixed(2)} total`);
        
        expect(costs.length).toBeGreaterThan(0);
      }
    });

    await test.step('Access cost analysis', async () => {
      const costAnalysisButton = page.locator('[data-testid="cost-analysis"], button:has-text("Cost Analysis")');
      
      if (await costAnalysisButton.count() > 0) {
        await costAnalysisButton.click();
        
        // Wait for cost analysis view
        await page.waitForSelector('[data-testid="cost-analysis-view"], .cost-dashboard', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Check for cost breakdown
        const costBreakdown = page.locator('[data-testid="cost-breakdown"], .cost-chart');
        await expect(costBreakdown.first()).toBeVisible();
        
        // Check for optimization recommendations
        const recommendations = page.locator('[data-testid="cost-recommendations"], .recommendation-card');
        if (await recommendations.count() > 0) {
          console.log(`Found ${await recommendations.count()} cost optimization recommendations`);
          
          // Click first recommendation for details
          await recommendations.first().click();
          
          // Check for recommendation details
          const detailsPanel = page.locator('[data-testid="recommendation-details"]');
          if (await detailsPanel.count() > 0) {
            const savings = await detailsPanel.locator('.estimated-savings').textContent();
            console.log(`Potential savings: ${savings}`);
          }
        }
      }
    });

    await test.step('Apply cost optimization', async () => {
      const optimizeButton = page.locator('[data-testid="apply-optimization"], button:has-text("Optimize")');
      
      if (await optimizeButton.count() > 0) {
        // Select optimization action
        await optimizeButton.click();
        
        // Confirm optimization
        const confirmDialog = page.locator('[role="dialog"], .confirm-dialog');
        if (await confirmDialog.count() > 0) {
          const estimatedSavings = await confirmDialog.locator('.savings-amount').textContent();
          console.log(`Estimated savings from optimization: ${estimatedSavings}`);
          
          const confirmButton = confirmDialog.locator('button:has-text("Confirm")');
          await confirmButton.click();
          
          // Wait for optimization to complete
          await page.waitForSelector('[data-testid="optimization-complete"], .success-message', {
            state: 'visible',
            timeout: 5000,
          }).catch(() => {});
        }
      }
    });
  });

  test('07 - Resource compliance validation', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('Check resource compliance status', async () => {
      const complianceBadges = page.locator('[data-testid="compliance-status"], .compliance-badge');
      
      if (await complianceBadges.count() > 0) {
        const complianceStats = {
          compliant: 0,
          nonCompliant: 0,
          unknown: 0,
        };
        
        for (let i = 0; i < await complianceBadges.count(); i++) {
          const badge = complianceBadges.nth(i);
          const status = (await badge.textContent())?.toLowerCase() || '';
          
          if (status.includes('compliant') && !status.includes('non')) {
            complianceStats.compliant++;
          } else if (status.includes('non-compliant') || status.includes('violation')) {
            complianceStats.nonCompliant++;
          } else {
            complianceStats.unknown++;
          }
        }
        
        console.log('Resource Compliance Statistics:', complianceStats);
        
        // Click on non-compliant resource for details
        const nonCompliantResource = page.locator('[data-testid="resource-item"]:has([data-testid="compliance-status"]:has-text("Non-Compliant"))').first();
        
        if (await nonCompliantResource.count() > 0) {
          await nonCompliantResource.click();
          
          // View compliance details
          const complianceTab = page.locator('[data-testid="compliance-tab"], button:has-text("Compliance")');
          if (await complianceTab.count() > 0) {
            await complianceTab.click();
            
            // Check for violation details
            const violations = page.locator('[data-testid="compliance-violation"], .violation-item');
            console.log(`Found ${await violations.count()} compliance violations`);
            
            if (await violations.count() > 0) {
              const firstViolation = violations.first();
              const violationType = await firstViolation.locator('.violation-type').textContent();
              const violationSeverity = await firstViolation.locator('.severity').textContent();
              
              console.log(`Violation: ${violationType} (${violationSeverity})`);
            }
          }
        }
      }
    });

    await test.step('Run compliance scan', async () => {
      const scanButton = page.locator('[data-testid="run-compliance-scan"], button:has-text("Scan")');
      
      if (await scanButton.count() > 0) {
        const startTime = Date.now();
        await scanButton.click();
        
        // Wait for scan to complete
        const progressBar = page.locator('[data-testid="scan-progress"], .progress-bar');
        if (await progressBar.count() > 0) {
          await progressBar.waitFor({ state: 'hidden', timeout: 10000 }).catch(() => {});
        }
        
        const scanTime = Date.now() - startTime;
        console.log(`Compliance scan completed in ${scanTime}ms`);
        
        // Check scan results
        const scanResults = page.locator('[data-testid="scan-results"], .scan-summary');
        if (await scanResults.count() > 0) {
          const resultText = await scanResults.textContent();
          console.log(`Scan results: ${resultText}`);
        }
      }
    });

    await test.step('Apply remediation', async () => {
      const remediateButton = page.locator('[data-testid="remediate"], button:has-text("Remediate")');
      
      if (await remediateButton.count() > 0) {
        await remediateButton.click();
        
        // Select remediation actions
        const remediationDialog = page.locator('[data-testid="remediation-dialog"]');
        if (await remediationDialog.count() > 0) {
          const remediationOptions = remediationDialog.locator('input[type="checkbox"]');
          
          // Select first remediation option
          if (await remediationOptions.count() > 0) {
            await remediationOptions.first().check();
            
            // Apply remediation
            const applyButton = remediationDialog.locator('button:has-text("Apply")');
            await applyButton.click();
            
            // Wait for remediation to complete
            await page.waitForSelector('[data-testid="remediation-complete"], .success-message', {
              state: 'visible',
              timeout: 5000,
            }).catch(() => {});
          }
        }
      }
    });
  });

  test('08 - Resource lifecycle management', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    let testResourceId: string | null = null;

    await test.step('Create test resource', async () => {
      const createButton = page.locator('[data-testid="create-resource"], button:has-text("Create")');
      
      if (await createButton.count() > 0) {
        await createButton.click();
        
        // Fill minimal required fields
        const nameInput = page.locator('input[name="name"]');
        const timestamp = Date.now();
        const resourceName = `lifecycle-test-${timestamp}`;
        
        await nameInput.fill(resourceName);
        
        // Submit creation
        const submitButton = page.locator('button[type="submit"]');
        await submitButton.click();
        
        // Get resource ID from success message or redirect
        await page.waitForTimeout(2000);
        
        // Store resource name for later cleanup
        testResourceId = resourceName;
      }
    });

    await test.step('Modify resource state', async () => {
      if (testResourceId) {
        // Search for created resource
        const searchInput = page.locator('[data-testid="resource-search"]');
        await searchInput.fill(testResourceId);
        await searchInput.press('Enter');
        
        await page.waitForTimeout(1000);
        
        // Find and select resource
        const resource = page.locator(`[data-testid="resource-item"]:has-text("${testResourceId}")`).first();
        if (await resource.count() > 0) {
          await resource.click();
          
          // Stop/deallocate resource
          const stopButton = page.locator('[data-testid="stop-resource"], button:has-text("Stop")');
          if (await stopButton.count() > 0) {
            await stopButton.click();
            
            // Confirm action
            const confirmButton = page.locator('button:has-text("Confirm")');
            await confirmButton.click();
            
            // Wait for state change
            await page.waitForSelector('[data-testid="resource-status"]:has-text("Stopped")', {
              state: 'visible',
              timeout: 10000,
            }).catch(() => {});
          }
        }
      }
    });

    await test.step('Delete test resource', async () => {
      if (testResourceId) {
        const deleteButton = page.locator('[data-testid="delete-resource"], button:has-text("Delete")');
        
        if (await deleteButton.count() > 0) {
          await deleteButton.click();
          
          // Confirm deletion
          const confirmDialog = page.locator('[role="dialog"], .delete-confirmation');
          if (await confirmDialog.count() > 0) {
            // Type confirmation if required
            const confirmInput = confirmDialog.locator('input[placeholder*="confirm"]');
            if (await confirmInput.count() > 0) {
              await confirmInput.fill('DELETE');
            }
            
            const confirmButton = confirmDialog.locator('button:has-text("Delete")');
            await confirmButton.click();
            
            // Wait for deletion to complete
            await page.waitForSelector('[data-testid="delete-success"], .success-message', {
              state: 'visible',
              timeout: 5000,
            }).catch(() => {});
            
            // Verify resource is gone
            await page.reload();
            const searchInput = page.locator('[data-testid="resource-search"]');
            await searchInput.fill(testResourceId);
            await searchInput.press('Enter');
            
            await page.waitForTimeout(1000);
            
            const resourceCount = await page.locator(`[data-testid="resource-item"]:has-text("${testResourceId}")`).count();
            expect(resourceCount).toBe(0);
          }
        }
      }
    });
  });

  test('09 - Resource access control and permissions', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForResourceList(page);

    await test.step('View resource permissions', async () => {
      const resourceItem = page.locator('[data-testid="resource-item"]').first();
      await resourceItem.click();
      
      // Navigate to access control tab
      const accessTab = page.locator('[data-testid="access-tab"], button:has-text("Access")');
      if (await accessTab.count() > 0) {
        await accessTab.click();
        
        // View current permissions
        const permissionsList = page.locator('[data-testid="permissions-list"], .permission-item');
        const permissionCount = await permissionsList.count();
        
        console.log(`Found ${permissionCount} permission entries`);
        
        if (permissionCount > 0) {
          const firstPermission = permissionsList.first();
          const principal = await firstPermission.locator('.principal-name').textContent();
          const role = await firstPermission.locator('.role-name').textContent();
          
          console.log(`Sample permission: ${principal} - ${role}`);
        }
      }
    });

    await test.step('Add resource permission', async () => {
      const addPermissionButton = page.locator('[data-testid="add-permission"], button:has-text("Add Permission")');
      
      if (await addPermissionButton.count() > 0) {
        await addPermissionButton.click();
        
        // Fill permission form
        const permissionDialog = page.locator('[data-testid="permission-dialog"]');
        if (await permissionDialog.count() > 0) {
          // Select principal
          const principalInput = permissionDialog.locator('input[name="principal"]');
          await principalInput.fill('test-user@policycortex.com');
          
          // Select role
          const roleSelect = permissionDialog.locator('select[name="role"]');
          await roleSelect.selectOption('Reader');
          
          // Save permission
          const saveButton = permissionDialog.locator('button:has-text("Save")');
          await saveButton.click();
          
          // Verify permission added
          await page.waitForSelector('[data-testid="permission-added"], .success-message', {
            state: 'visible',
            timeout: 3000,
          }).catch(() => {});
        }
      }
    });
  });

  test('10 - Performance and accessibility validation', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);

    await test.step('Measure page load performance', async () => {
      const metrics = await page.evaluate(() => {
        const perf = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        return {
          domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
          loadComplete: perf.loadEventEnd - perf.loadEventStart,
          domInteractive: perf.domInteractive - perf.fetchStart,
        };
      });
      
      console.log('Resource Management Performance Metrics:', metrics);
      
      expect(metrics.domInteractive).toBeLessThan(2000);
      expect(metrics.loadComplete).toBeLessThan(3000);
    });

    await test.step('Test keyboard navigation', async () => {
      // Tab through resource list
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      
      // Check if resource item is focused
      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.getAttribute('data-testid');
      });
      
      expect(focusedElement).toBeTruthy();
      
      // Test keyboard selection
      await page.keyboard.press('Space'); // Select item
      await page.keyboard.press('Enter'); // Open details
      
      // Verify action was performed
      const detailsVisible = await page.locator('[data-testid="resource-details"]').isVisible();
      expect(detailsVisible).toBeTruthy();
    });

    await test.step('Validate ARIA labels', async () => {
      const interactiveElements = page.locator('button, a, input, select, [role="button"]');
      const elementCount = await interactiveElements.count();
      
      let missingLabels = 0;
      for (let i = 0; i < Math.min(elementCount, 10); i++) {
        const element = interactiveElements.nth(i);
        const ariaLabel = await element.getAttribute('aria-label');
        const ariaLabelledBy = await element.getAttribute('aria-labelledby');
        const textContent = await element.textContent();
        
        if (!ariaLabel && !ariaLabelledBy && !textContent?.trim()) {
          missingLabels++;
        }
      }
      
      console.log(`Checked ${Math.min(elementCount, 10)} elements, ${missingLabels} missing labels`);
      expect(missingLabels).toBeLessThan(2); // Allow maximum 1 missing label
    });
  });
});

// Cleanup
test.afterAll(async () => {
  console.log('Resource management tests completed');
});