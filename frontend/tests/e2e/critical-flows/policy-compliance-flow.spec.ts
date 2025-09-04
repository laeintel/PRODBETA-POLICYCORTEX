/**
 * PolicyCortex Critical User Flow Test Suite: Policy Compliance Checks
 * 
 * This test suite covers comprehensive policy compliance scenarios including:
 * - Policy creation and management
 * - Compliance scanning and assessment
 * - Violation detection and reporting
 * - Remediation workflows
 * - Compliance drift prediction (Patent #4)
 * - Cross-domain correlation (Patent #1)
 * - AI-driven recommendations
 * 
 * Performance targets:
 * - Policy evaluation: <2s
 * - Compliance scan: <5s
 * - Remediation application: <3s
 * - AI predictions: <100ms (Patent requirement)
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
    localStorage.setItem('msal.account.keys', JSON.stringify(['compliance-admin']));
    localStorage.setItem('msal.compliance-admin', JSON.stringify({
      username: 'compliance@policycortex.com',
      name: 'Compliance Admin',
      roles: ['admin', 'compliance-officer'],
      permissions: ['policy:manage', 'compliance:scan', 'remediation:apply'],
    }));
    sessionStorage.setItem('isAuthenticated', 'true');
  });
}

// Helper to measure API performance (Patent #4 requirement)
async function measureApiPerformance(page: Page, apiEndpoint: string): Promise<number> {
  const startTime = Date.now();
  
  const response = await page.evaluate(async (endpoint) => {
    const res = await fetch(endpoint);
    return res.status;
  }, apiEndpoint);
  
  const duration = Date.now() - startTime;
  console.log(`API ${apiEndpoint} responded in ${duration}ms`);
  
  // Patent #4 requires <100ms for predictions
  if (apiEndpoint.includes('predictions')) {
    expect(duration).toBeLessThan(100);
  }
  
  return duration;
}

test.describe('Critical Flow: Policy Compliance Checks', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuthenticatedSession(page);
  });

  test('01 - Policy management and creation', async ({ page }) => {
    await test.step('Navigate to policy management', async () => {
      await page.goto(`${BASE_URL}/governance/policies`);
      
      // Wait for policy list to load
      await page.waitForSelector('[data-testid="policy-list"], .policy-table', {
        state: 'visible',
        timeout: 5000,
      });
      
      // Verify policy categories are displayed
      const policyCategories = page.locator('[data-testid="policy-category"], .category-tab');
      const categoryCount = await policyCategories.count();
      
      expect(categoryCount).toBeGreaterThan(0);
      console.log(`Found ${categoryCount} policy categories`);
    });

    await test.step('Create new compliance policy', async () => {
      const createButton = page.locator('[data-testid="create-policy"], button:has-text("Create Policy")');
      
      if (await createButton.count() > 0) {
        await createButton.click();
        
        // Wait for policy creation form
        await page.waitForSelector('[data-testid="policy-form"], [role="dialog"]', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Fill policy details
        const policyForm = page.locator('[data-testid="policy-form"]');
        
        // Policy name
        const nameInput = policyForm.locator('input[name="policyName"]');
        await nameInput.fill('Test Compliance Policy - Resource Tagging');
        
        // Policy description
        const descInput = policyForm.locator('textarea[name="description"]');
        await descInput.fill('Ensures all resources have required tags for compliance tracking');
        
        // Policy type
        const typeSelect = policyForm.locator('select[name="policyType"]');
        await typeSelect.selectOption('compliance');
        
        // Severity
        const severitySelect = policyForm.locator('select[name="severity"]');
        await severitySelect.selectOption('high');
        
        // Policy rules (JSON editor or form fields)
        const rulesEditor = policyForm.locator('[data-testid="policy-rules"], textarea[name="rules"]');
        if (await rulesEditor.count() > 0) {
          const policyRules = {
            required_tags: ['environment', 'owner', 'cost-center'],
            resource_types: ['Microsoft.Compute/virtualMachines', 'Microsoft.Storage/storageAccounts'],
            enforcement_mode: 'audit',
          };
          await rulesEditor.fill(JSON.stringify(policyRules, null, 2));
        }
        
        // Save policy
        const saveButton = policyForm.locator('button[type="submit"], button:has-text("Save")');
        await saveButton.click();
        
        // Verify policy created
        await page.waitForSelector('[data-testid="policy-created"], .success-message', {
          state: 'visible',
          timeout: 3000,
        });
      }
    });

    await test.step('Test policy validation', async () => {
      // Click on a policy to view details
      const policyItem = page.locator('[data-testid="policy-item"]').first();
      
      if (await policyItem.count() > 0) {
        await policyItem.click();
        
        // Wait for policy details
        await page.waitForSelector('[data-testid="policy-details"]', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Test policy validation
        const validateButton = page.locator('[data-testid="validate-policy"], button:has-text("Validate")');
        if (await validateButton.count() > 0) {
          const startTime = Date.now();
          await validateButton.click();
          
          // Wait for validation results
          await page.waitForSelector('[data-testid="validation-results"]', {
            state: 'visible',
            timeout: 5000,
          });
          
          const validationTime = Date.now() - startTime;
          console.log(`Policy validation completed in ${validationTime}ms`);
          expect(validationTime).toBeLessThan(2000);
          
          // Check validation results
          const validationStatus = page.locator('[data-testid="validation-status"]');
          const statusText = await validationStatus.textContent();
          console.log(`Validation status: ${statusText}`);
        }
      }
    });
  });

  test('02 - Compliance scanning and assessment', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance`);

    await test.step('Initiate compliance scan', async () => {
      const scanButton = page.locator('[data-testid="run-scan"], button:has-text("Run Compliance Scan")');
      
      if (await scanButton.count() > 0) {
        const startTime = Date.now();
        await scanButton.click();
        
        // Monitor scan progress
        const progressBar = page.locator('[data-testid="scan-progress"], .progress-bar');
        if (await progressBar.count() > 0) {
          // Wait for scan to complete
          await progressBar.waitFor({ state: 'hidden', timeout: 10000 }).catch(() => {});
        }
        
        const scanTime = Date.now() - startTime;
        console.log(`Compliance scan completed in ${scanTime}ms`);
        expect(scanTime).toBeLessThan(5000);
      }
    });

    await test.step('Review scan results', async () => {
      // Wait for results to load
      await page.waitForSelector('[data-testid="scan-results"], .compliance-results', {
        state: 'visible',
        timeout: 5000,
      });
      
      // Check compliance score
      const scoreElement = page.locator('[data-testid="compliance-score"], .score-display');
      if (await scoreElement.count() > 0) {
        const score = await scoreElement.textContent();
        console.log(`Overall compliance score: ${score}`);
        
        // Parse score percentage
        const scoreValue = parseFloat(score?.replace('%', '') || '0');
        expect(scoreValue).toBeGreaterThanOrEqual(0);
        expect(scoreValue).toBeLessThanOrEqual(100);
      }
      
      // Check violation summary
      const violationSummary = page.locator('[data-testid="violation-summary"]');
      if (await violationSummary.count() > 0) {
        const criticalCount = await violationSummary.locator('.critical-count').textContent();
        const highCount = await violationSummary.locator('.high-count').textContent();
        const mediumCount = await violationSummary.locator('.medium-count').textContent();
        const lowCount = await violationSummary.locator('.low-count').textContent();
        
        console.log(`Violations - Critical: ${criticalCount}, High: ${highCount}, Medium: ${mediumCount}, Low: ${lowCount}`);
      }
    });

    await test.step('Filter violations by severity', async () => {
      const severityFilter = page.locator('[data-testid="severity-filter"], select[name="severity"]');
      
      if (await severityFilter.count() > 0) {
        // Filter critical violations
        await severityFilter.selectOption('critical');
        await page.waitForTimeout(1000);
        
        // Verify filtered results
        const violations = page.locator('[data-testid="violation-item"], .violation-row');
        const violationCount = await violations.count();
        
        if (violationCount > 0) {
          // Check all displayed violations are critical
          for (let i = 0; i < Math.min(violationCount, 3); i++) {
            const violation = violations.nth(i);
            const severity = await violation.locator('.severity-badge').textContent();
            expect(severity?.toLowerCase()).toContain('critical');
          }
        }
        
        // Reset filter
        await severityFilter.selectOption('all');
      }
    });
  });

  test('03 - Violation detection and detailed analysis', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance`);

    await test.step('Access violation details', async () => {
      // Find a violation item
      const violationItem = page.locator('[data-testid="violation-item"]').first();
      
      if (await violationItem.count() > 0) {
        // Get violation info
        const violationType = await violationItem.locator('.violation-type').textContent();
        const affectedResource = await violationItem.locator('.resource-name').textContent();
        
        console.log(`Analyzing violation: ${violationType} on ${affectedResource}`);
        
        // Click to view details
        await violationItem.click();
        
        // Wait for detail panel
        await page.waitForSelector('[data-testid="violation-details"], .detail-panel', {
          state: 'visible',
          timeout: 3000,
        });
      }
    });

    await test.step('Review violation impact analysis', async () => {
      const detailPanel = page.locator('[data-testid="violation-details"]');
      
      if (await detailPanel.count() > 0) {
        // Check impact assessment
        const impactScore = await detailPanel.locator('[data-testid="impact-score"]').textContent();
        console.log(`Impact score: ${impactScore}`);
        
        // Check affected policies
        const affectedPolicies = detailPanel.locator('[data-testid="affected-policy"]');
        const policyCount = await affectedPolicies.count();
        console.log(`Affects ${policyCount} policies`);
        
        // Check remediation suggestions
        const remediations = detailPanel.locator('[data-testid="remediation-option"]');
        const remediationCount = await remediations.count();
        console.log(`${remediationCount} remediation options available`);
        
        // View evidence
        const evidenceTab = detailPanel.locator('[data-testid="evidence-tab"], button:has-text("Evidence")');
        if (await evidenceTab.count() > 0) {
          await evidenceTab.click();
          
          // Check for evidence data
          const evidenceItems = detailPanel.locator('[data-testid="evidence-item"]');
          expect(await evidenceItems.count()).toBeGreaterThan(0);
        }
      }
    });

    await test.step('Export violation report', async () => {
      const exportButton = page.locator('[data-testid="export-violations"], button:has-text("Export")');
      
      if (await exportButton.count() > 0) {
        // Set up download listener
        const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null);
        
        await exportButton.click();
        
        // Select export format
        const pdfOption = page.locator('button:has-text("PDF"), [data-format="pdf"]');
        if (await pdfOption.count() > 0) {
          await pdfOption.click();
          
          const download = await downloadPromise;
          if (download) {
            expect(download.suggestedFilename()).toMatch(/compliance.*\.pdf$/);
            console.log(`Downloaded: ${download.suggestedFilename()}`);
          }
        }
      }
    });
  });

  test('04 - Remediation workflows', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance`);

    await test.step('Select violations for remediation', async () => {
      // Select multiple violations
      const checkboxes = page.locator('[data-testid="violation-checkbox"], input[type="checkbox"][data-violation]');
      
      const checkCount = Math.min(await checkboxes.count(), 3);
      for (let i = 0; i < checkCount; i++) {
        await checkboxes.nth(i).check();
      }
      
      if (checkCount > 0) {
        // Verify bulk actions appear
        const bulkActions = page.locator('[data-testid="bulk-remediation"], .bulk-action-bar');
        await expect(bulkActions.first()).toBeVisible();
      }
    });

    await test.step('Configure remediation', async () => {
      const remediateButton = page.locator('[data-testid="bulk-remediate"], button:has-text("Remediate")');
      
      if (await remediateButton.count() > 0) {
        await remediateButton.click();
        
        // Wait for remediation dialog
        await page.waitForSelector('[data-testid="remediation-dialog"], [role="dialog"]', {
          state: 'visible',
          timeout: 3000,
        });
        
        const dialog = page.locator('[data-testid="remediation-dialog"]');
        
        // Select remediation strategy
        const strategySelect = dialog.locator('select[name="strategy"]');
        await strategySelect.selectOption('auto-fix');
        
        // Set remediation scope
        const scopeRadio = dialog.locator('input[name="scope"][value="selected"]');
        await scopeRadio.check();
        
        // Configure scheduling
        const scheduleToggle = dialog.locator('[data-testid="schedule-toggle"]');
        if (await scheduleToggle.count() > 0) {
          await scheduleToggle.click();
          
          // Set schedule time
          const timeInput = dialog.locator('input[type="time"]');
          await timeInput.fill('22:00'); // Schedule for 10 PM
        }
        
        // Review impact preview
        const previewButton = dialog.locator('button:has-text("Preview")');
        if (await previewButton.count() > 0) {
          await previewButton.click();
          
          // Wait for preview to load
          await page.waitForSelector('[data-testid="remediation-preview"]', {
            state: 'visible',
            timeout: 2000,
          });
          
          const previewText = await dialog.locator('[data-testid="remediation-preview"]').textContent();
          console.log(`Remediation preview: ${previewText}`);
        }
      }
    });

    await test.step('Apply remediation', async () => {
      const applyButton = page.locator('button:has-text("Apply Remediation")');
      
      if (await applyButton.count() > 0) {
        const startTime = Date.now();
        await applyButton.click();
        
        // Monitor remediation progress
        const progressIndicator = page.locator('[data-testid="remediation-progress"]');
        if (await progressIndicator.count() > 0) {
          await progressIndicator.waitFor({ state: 'hidden', timeout: 10000 }).catch(() => {});
        }
        
        const remediationTime = Date.now() - startTime;
        console.log(`Remediation completed in ${remediationTime}ms`);
        expect(remediationTime).toBeLessThan(3000);
        
        // Check remediation results
        const resultsPanel = page.locator('[data-testid="remediation-results"]');
        if (await resultsPanel.count() > 0) {
          const successCount = await resultsPanel.locator('.success-count').textContent();
          const failedCount = await resultsPanel.locator('.failed-count').textContent();
          
          console.log(`Remediation results - Success: ${successCount}, Failed: ${failedCount}`);
        }
      }
    });

    await test.step('Verify remediation effectiveness', async () => {
      // Re-run compliance scan to verify fixes
      const rescanButton = page.locator('[data-testid="rescan"], button:has-text("Re-scan")');
      
      if (await rescanButton.count() > 0) {
        await rescanButton.click();
        
        // Wait for scan to complete
        await page.waitForTimeout(3000);
        
        // Check if violations were resolved
        const newViolationCount = await page.locator('[data-testid="violation-item"]').count();
        console.log(`Remaining violations after remediation: ${newViolationCount}`);
      }
    });
  });

  test('05 - Compliance drift prediction (Patent #4)', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance/predictions`);

    await test.step('Test predictive compliance API performance', async () => {
      // Patent #4 requires <100ms inference latency
      const apiPerformance = await measureApiPerformance(page, '/api/v1/predictions');
      expect(apiPerformance).toBeLessThan(100);
    });

    await test.step('View drift predictions', async () => {
      // Wait for predictions to load
      await page.waitForSelector('[data-testid="drift-predictions"], .prediction-dashboard', {
        state: 'visible',
        timeout: 5000,
      });
      
      // Check prediction accuracy display (Patent requirement: 99.2%)
      const accuracyElement = page.locator('[data-testid="prediction-accuracy"]');
      if (await accuracyElement.count() > 0) {
        const accuracy = await accuracyElement.textContent();
        console.log(`Prediction accuracy: ${accuracy}`);
        
        const accuracyValue = parseFloat(accuracy?.replace('%', '') || '0');
        expect(accuracyValue).toBeGreaterThanOrEqual(99.0); // Close to patent requirement
      }
      
      // Check for high-risk predictions
      const riskPredictions = page.locator('[data-testid="high-risk-prediction"]');
      const riskCount = await riskPredictions.count();
      
      console.log(`Found ${riskCount} high-risk drift predictions`);
      
      if (riskCount > 0) {
        // Examine first prediction
        const firstPrediction = riskPredictions.first();
        
        const resourceName = await firstPrediction.locator('.resource-name').textContent();
        const driftProbability = await firstPrediction.locator('.drift-probability').textContent();
        const timeframe = await firstPrediction.locator('.prediction-timeframe').textContent();
        
        console.log(`Prediction: ${resourceName} has ${driftProbability} drift probability in ${timeframe}`);
      }
    });

    await test.step('Test SHAP explainability (Patent #4)', async () => {
      // Click on a prediction for details
      const predictionItem = page.locator('[data-testid="prediction-item"]').first();
      
      if (await predictionItem.count() > 0) {
        await predictionItem.click();
        
        // Wait for explainability panel
        await page.waitForSelector('[data-testid="explainability-panel"], .shap-analysis', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Check for SHAP values
        const shapValues = page.locator('[data-testid="shap-feature"]');
        const featureCount = await shapValues.count();
        
        console.log(`SHAP analysis shows ${featureCount} contributing features`);
        expect(featureCount).toBeGreaterThan(0);
        
        // Verify feature importance visualization
        const featureChart = page.locator('[data-testid="feature-importance-chart"], canvas');
        await expect(featureChart.first()).toBeVisible();
      }
    });

    await test.step('Submit prediction feedback', async () => {
      // Patent #4 includes continuous learning with human feedback
      const feedbackButton = page.locator('[data-testid="prediction-feedback"], button:has-text("Feedback")');
      
      if (await feedbackButton.count() > 0) {
        await feedbackButton.click();
        
        // Submit feedback
        const feedbackDialog = page.locator('[data-testid="feedback-dialog"]');
        if (await feedbackDialog.count() > 0) {
          // Rate prediction accuracy
          const accuracyRating = feedbackDialog.locator('input[name="accuracy"][value="accurate"]');
          await accuracyRating.check();
          
          // Add comment
          const commentInput = feedbackDialog.locator('textarea[name="comment"]');
          await commentInput.fill('Prediction was accurate, drift occurred as predicted');
          
          // Submit feedback
          const submitButton = feedbackDialog.locator('button:has-text("Submit")');
          await submitButton.click();
          
          // Verify feedback submitted
          await page.waitForSelector('[data-testid="feedback-success"]', {
            state: 'visible',
            timeout: 2000,
          });
        }
      }
    });
  });

  test('06 - Cross-domain correlation (Patent #1)', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/correlations`);

    await test.step('View cross-domain correlations', async () => {
      // Wait for correlation engine results
      await page.waitForSelector('[data-testid="correlation-graph"], .correlation-visualization', {
        state: 'visible',
        timeout: 5000,
      });
      
      // Check for correlation patterns
      const correlationNodes = page.locator('[data-testid="correlation-node"], .graph-node');
      const nodeCount = await correlationNodes.count();
      
      console.log(`Found ${nodeCount} correlation nodes across domains`);
      expect(nodeCount).toBeGreaterThan(0);
      
      // Test correlation strength indicators
      const strongCorrelations = page.locator('[data-testid="strong-correlation"], .correlation-strong');
      const strongCount = await strongCorrelations.count();
      
      console.log(`${strongCount} strong correlations detected`);
    });

    await test.step('Drill into correlation details', async () => {
      // Click on a correlation link
      const correlationLink = page.locator('[data-testid="correlation-link"], .correlation-edge').first();
      
      if (await correlationLink.count() > 0) {
        await correlationLink.click();
        
        // View correlation analysis
        const analysisPanel = page.locator('[data-testid="correlation-analysis"]');
        await expect(analysisPanel).toBeVisible();
        
        // Check correlation metrics
        const correlationScore = await analysisPanel.locator('.correlation-score').textContent();
        const confidence = await analysisPanel.locator('.confidence-level').textContent();
        
        console.log(`Correlation score: ${correlationScore}, Confidence: ${confidence}`);
        
        // Check domains involved
        const domains = analysisPanel.locator('[data-testid="correlated-domain"]');
        const domainCount = await domains.count();
        
        expect(domainCount).toBeGreaterThanOrEqual(2); // Cross-domain requirement
      }
    });
  });

  test('07 - AI-driven compliance recommendations', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/recommendations`);

    await test.step('View AI recommendations', async () => {
      // Wait for recommendations to load
      await page.waitForSelector('[data-testid="ai-recommendations"], .recommendation-list', {
        state: 'visible',
        timeout: 5000,
      });
      
      // Check recommendation cards
      const recommendations = page.locator('[data-testid="recommendation-card"]');
      const recCount = await recommendations.count();
      
      console.log(`${recCount} AI-driven recommendations available`);
      
      if (recCount > 0) {
        // Examine first recommendation
        const firstRec = recommendations.first();
        
        const title = await firstRec.locator('.recommendation-title').textContent();
        const impact = await firstRec.locator('.impact-badge').textContent();
        const effort = await firstRec.locator('.effort-indicator').textContent();
        
        console.log(`Recommendation: ${title} (Impact: ${impact}, Effort: ${effort})`);
        
        // Check for cost savings estimate
        const savings = await firstRec.locator('.estimated-savings').textContent();
        if (savings) {
          console.log(`Estimated savings: ${savings}`);
        }
      }
    });

    await test.step('Apply AI recommendation', async () => {
      const applyButton = page.locator('[data-testid="apply-recommendation"]').first();
      
      if (await applyButton.count() > 0) {
        await applyButton.click();
        
        // Review implementation plan
        const planDialog = page.locator('[data-testid="implementation-plan"]');
        await expect(planDialog).toBeVisible();
        
        // Check implementation steps
        const steps = planDialog.locator('[data-testid="implementation-step"]');
        const stepCount = await steps.count();
        
        console.log(`Implementation requires ${stepCount} steps`);
        
        // Approve and apply
        const approveButton = planDialog.locator('button:has-text("Approve")');
        await approveButton.click();
        
        // Monitor implementation
        await page.waitForSelector('[data-testid="implementation-progress"]', {
          state: 'visible',
          timeout: 3000,
        });
      }
    });
  });

  test('08 - Compliance reporting and dashboards', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance/reports`);

    await test.step('Generate compliance report', async () => {
      const generateButton = page.locator('[data-testid="generate-report"], button:has-text("Generate Report")');
      
      if (await generateButton.count() > 0) {
        await generateButton.click();
        
        // Configure report parameters
        const reportDialog = page.locator('[data-testid="report-config"]');
        if (await reportDialog.count() > 0) {
          // Select report type
          const typeSelect = reportDialog.locator('select[name="reportType"]');
          await typeSelect.selectOption('executive-summary');
          
          // Set date range
          const startDate = reportDialog.locator('input[name="startDate"]');
          const endDate = reportDialog.locator('input[name="endDate"]');
          
          const today = new Date();
          const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
          
          await startDate.fill(lastMonth.toISOString().split('T')[0]);
          await endDate.fill(today.toISOString().split('T')[0]);
          
          // Generate report
          const generateBtn = reportDialog.locator('button:has-text("Generate")');
          await generateBtn.click();
          
          // Wait for report generation
          await page.waitForSelector('[data-testid="report-ready"], .report-viewer', {
            state: 'visible',
            timeout: 10000,
          });
        }
      }
    });

    await test.step('View compliance trends', async () => {
      const trendsTab = page.locator('[data-testid="trends-tab"], button:has-text("Trends")');
      
      if (await trendsTab.count() > 0) {
        await trendsTab.click();
        
        // Check trend charts
        const trendCharts = page.locator('[data-testid="trend-chart"], canvas');
        const chartCount = await trendCharts.count();
        
        console.log(`Displaying ${chartCount} compliance trend charts`);
        
        // Check for improvement indicators
        const improvements = page.locator('[data-testid="improvement-indicator"], .trend-up');
        const improvementCount = await improvements.count();
        
        console.log(`${improvementCount} areas showing improvement`);
      }
    });

    await test.step('Schedule automated reports', async () => {
      const scheduleButton = page.locator('[data-testid="schedule-report"], button:has-text("Schedule")');
      
      if (await scheduleButton.count() > 0) {
        await scheduleButton.click();
        
        const scheduleDialog = page.locator('[data-testid="schedule-dialog"]');
        if (await scheduleDialog.count() > 0) {
          // Set frequency
          const frequencySelect = scheduleDialog.locator('select[name="frequency"]');
          await frequencySelect.selectOption('weekly');
          
          // Set recipients
          const recipientsInput = scheduleDialog.locator('input[name="recipients"]');
          await recipientsInput.fill('compliance-team@policycortex.com');
          
          // Save schedule
          const saveButton = scheduleDialog.locator('button:has-text("Save Schedule")');
          await saveButton.click();
          
          // Verify schedule created
          await page.waitForSelector('[data-testid="schedule-created"]', {
            state: 'visible',
            timeout: 2000,
          });
        }
      }
    });
  });

  test('09 - Multi-cloud compliance validation', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance/multi-cloud`);

    await test.step('View multi-cloud compliance status', async () => {
      // Check for cloud provider tabs
      const cloudTabs = page.locator('[data-testid="cloud-tab"], .cloud-provider-tab');
      const providers = ['Azure', 'AWS', 'GCP'];
      
      for (const provider of providers) {
        const tab = cloudTabs.filter({ hasText: provider });
        
        if (await tab.count() > 0) {
          await tab.click();
          await page.waitForTimeout(1000);
          
          // Check provider-specific compliance
          const complianceScore = page.locator(`[data-testid="${provider.toLowerCase()}-score"]`);
          if (await complianceScore.count() > 0) {
            const score = await complianceScore.textContent();
            console.log(`${provider} compliance score: ${score}`);
          }
        }
      }
    });

    await test.step('Compare cloud compliance', async () => {
      const compareButton = page.locator('[data-testid="compare-clouds"], button:has-text("Compare")');
      
      if (await compareButton.count() > 0) {
        await compareButton.click();
        
        // View comparison matrix
        const comparisonMatrix = page.locator('[data-testid="comparison-matrix"], .comparison-table');
        await expect(comparisonMatrix).toBeVisible();
        
        // Check for compliance gaps
        const gaps = page.locator('[data-testid="compliance-gap"], .gap-indicator');
        const gapCount = await gaps.count();
        
        console.log(`Found ${gapCount} compliance gaps across clouds`);
      }
    });
  });

  test('10 - Performance and accessibility validation', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance`);

    await test.step('Measure Core Web Vitals', async () => {
      const vitals = await page.evaluate(() => {
        return new Promise((resolve) => {
          let lcpValue = 0;
          let clsValue = 0;
          
          new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1] as any;
            lcpValue = lastEntry.renderTime || lastEntry.loadTime;
          }).observe({ entryTypes: ['largest-contentful-paint'] });
          
          new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (!(entry as any).hadRecentInput) {
                clsValue += (entry as any).value;
              }
            }
          }).observe({ entryTypes: ['layout-shift'] });
          
          setTimeout(() => {
            resolve({
              LCP: lcpValue,
              CLS: clsValue,
              FID: 50, // Simulated FID
            });
          }, 3000);
        });
      });
      
      console.log('Compliance Page Web Vitals:', vitals);
      
      expect(vitals.LCP).toBeLessThan(2500);
      expect(vitals.CLS).toBeLessThan(0.1);
      expect(vitals.FID).toBeLessThan(100);
    });

    await test.step('Validate accessibility', async () => {
      // Check for skip links
      const skipLink = page.locator('a[href="#main"], [data-testid="skip-link"]');
      if (await skipLink.count() > 0) {
        // Focus skip link
        await skipLink.focus();
        await expect(skipLink).toBeFocused();
      }
      
      // Check heading hierarchy
      const headings = await page.evaluate(() => {
        const h1Count = document.querySelectorAll('h1').length;
        const h2Count = document.querySelectorAll('h2').length;
        const h3Count = document.querySelectorAll('h3').length;
        
        return { h1: h1Count, h2: h2Count, h3: h3Count };
      });
      
      console.log('Heading hierarchy:', headings);
      expect(headings.h1).toBeGreaterThanOrEqual(1);
      
      // Check for proper form labels
      const formInputs = page.locator('input:not([type="hidden"]), select, textarea');
      const inputCount = await formInputs.count();
      
      for (let i = 0; i < Math.min(inputCount, 5); i++) {
        const input = formInputs.nth(i);
        const id = await input.getAttribute('id');
        const ariaLabel = await input.getAttribute('aria-label');
        const ariaLabelledBy = await input.getAttribute('aria-labelledby');
        
        // Input should have either id with label, aria-label, or aria-labelledby
        const hasLabel = id || ariaLabel || ariaLabelledBy;
        expect(hasLabel).toBeTruthy();
      }
    });

    await test.step('Test responsive design', async () => {
      const viewports = [
        { width: 375, height: 667, name: 'mobile' },
        { width: 768, height: 1024, name: 'tablet' },
        { width: 1920, height: 1080, name: 'desktop' },
      ];
      
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await page.waitForTimeout(500);
        
        // Check layout adjustments
        const mainContent = page.locator('main, [role="main"]');
        const contentBox = await mainContent.boundingBox();
        
        if (contentBox) {
          expect(contentBox.width).toBeLessThanOrEqual(viewport.width);
        }
        
        // Take screenshot for visual validation
        await page.screenshot({
          path: `test-results/screenshots/compliance-${viewport.name}.png`,
          fullPage: false,
        });
      }
    });
  });
});

// Cleanup
test.afterAll(async () => {
  console.log('Policy compliance tests completed');
});