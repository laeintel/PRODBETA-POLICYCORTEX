/**
 * Governance Page Object Model
 */

import { Page, expect } from '@playwright/test';

export class GovernancePage {
  constructor(private page: Page) {}
  
  async goto(section?: string) {
    const path = section ? `/governance/${section}` : '/governance';
    await this.page.goto(path);
    await this.page.waitForLoadState('networkidle');
  }
  
  // Compliance Management
  async viewComplianceDashboard() {
    await this.goto('compliance');
    
    // Wait for compliance data to load
    await this.page.waitForSelector('[data-testid="compliance-score"]');
    
    const complianceScore = await this.page.locator('[data-testid="compliance-score"]').textContent();
    return parseFloat(complianceScore || '0');
  }
  
  async checkComplianceStatus(framework: string) {
    const frameworkCard = this.page.locator(`[data-testid="framework-${framework}"]`);
    const status = await frameworkCard.locator('[data-testid="compliance-status"]').textContent();
    const score = await frameworkCard.locator('[data-testid="compliance-percentage"]').textContent();
    
    return { status, score: parseFloat(score || '0') };
  }
  
  async viewNonCompliantResources(framework?: string) {
    if (framework) {
      const filterDropdown = this.page.locator('[data-testid="framework-filter"]');
      await filterDropdown.selectOption(framework);
    }
    
    const viewButton = this.page.locator('[data-testid="view-non-compliant"]');
    await viewButton.click();
    
    // Wait for results
    await this.page.waitForSelector('[data-testid="non-compliant-list"]');
    
    const resources = await this.page.locator('[data-testid="non-compliant-resource"]').allTextContents();
    return resources;
  }
  
  async remediateResource(resourceId: string, remediationType: 'auto' | 'manual') {
    const resourceRow = this.page.locator(`[data-testid="resource-${resourceId}"]`);
    const remediateButton = resourceRow.locator('[data-testid="remediate-button"]');
    await remediateButton.click();
    
    if (remediationType === 'auto') {
      const autoRemediateButton = this.page.locator('[data-testid="auto-remediate"]');
      await autoRemediateButton.click();
      
      // Wait for remediation to complete
      await this.page.waitForSelector('[data-testid="remediation-success"]');
    } else {
      const manualSteps = await this.page.locator('[data-testid="manual-steps"]').textContent();
      return manualSteps;
    }
  }
  
  // Policy Management
  async createPolicy(data: {
    name: string;
    description: string;
    type: 'preventive' | 'detective' | 'corrective';
    rules: string[];
    effect: 'allow' | 'deny' | 'audit';
  }) {
    await this.goto('policies');
    
    const createButton = this.page.locator('[data-testid="create-policy"]');
    await createButton.click();
    
    // Fill policy form
    await this.page.fill('[data-testid="policy-name"]', data.name);
    await this.page.fill('[data-testid="policy-description"]', data.description);
    await this.page.selectOption('[data-testid="policy-type"]', data.type);
    await this.page.selectOption('[data-testid="policy-effect"]', data.effect);
    
    // Add rules
    for (const rule of data.rules) {
      await this.page.click('[data-testid="add-rule"]');
      await this.page.fill('[data-testid="rule-input"]:last-child', rule);
    }
    
    const saveButton = this.page.locator('[data-testid="save-policy"]');
    await saveButton.click();
    
    // Verify policy created
    await expect(this.page.locator('[data-testid="policy-created"]')).toBeVisible();
  }
  
  async assignPolicy(policyName: string, scope: string[]) {
    const policyRow = this.page.locator(`[data-testid="policy-${policyName}"]`);
    const assignButton = policyRow.locator('[data-testid="assign-button"]');
    await assignButton.click();
    
    // Select scope
    for (const s of scope) {
      const scopeCheckbox = this.page.locator(`[data-testid="scope-${s}"]`);
      await scopeCheckbox.check();
    }
    
    const applyButton = this.page.locator('[data-testid="apply-assignment"]');
    await applyButton.click();
  }
  
  async evaluatePolicy(policyName: string) {
    const policyRow = this.page.locator(`[data-testid="policy-${policyName}"]`);
    const evaluateButton = policyRow.locator('[data-testid="evaluate-button"]');
    await evaluateButton.click();
    
    // Wait for evaluation results
    await this.page.waitForSelector('[data-testid="evaluation-results"]');
    
    const results = await this.page.locator('[data-testid="evaluation-data"]').textContent();
    return JSON.parse(results || '{}');
  }
  
  // Cost Management
  async viewCostAnalysis(timeRange: '30d' | '90d' | '12m') {
    await this.goto('cost');
    
    const timeSelector = this.page.locator('[data-testid="time-range"]');
    await timeSelector.selectOption(timeRange);
    
    // Wait for cost data to load
    await this.page.waitForSelector('[data-testid="cost-summary"]');
    
    const totalCost = await this.page.locator('[data-testid="total-cost"]').textContent();
    const trend = await this.page.locator('[data-testid="cost-trend"]').textContent();
    
    return { 
      totalCost: parseFloat(totalCost?.replace(/[^0-9.-]+/g, '') || '0'),
      trend 
    };
  }
  
  async setCostBudget(data: {
    name: string;
    amount: number;
    scope: string;
    alertThresholds: number[];
  }) {
    const setBudgetButton = this.page.locator('[data-testid="set-budget"]');
    await setBudgetButton.click();
    
    // Fill budget form
    await this.page.fill('[data-testid="budget-name"]', data.name);
    await this.page.fill('[data-testid="budget-amount"]', data.amount.toString());
    await this.page.selectOption('[data-testid="budget-scope"]', data.scope);
    
    // Set alert thresholds
    for (const threshold of data.alertThresholds) {
      await this.page.click('[data-testid="add-threshold"]');
      await this.page.fill('[data-testid="threshold-value"]:last-child', threshold.toString());
    }
    
    const saveButton = this.page.locator('[data-testid="save-budget"]');
    await saveButton.click();
  }
  
  async viewCostRecommendations() {
    const recommendationsTab = this.page.locator('[data-testid="cost-recommendations-tab"]');
    await recommendationsTab.click();
    
    // Wait for recommendations to load
    await this.page.waitForSelector('[data-testid="recommendations-list"]');
    
    const recommendations = await this.page.locator('[data-testid="cost-recommendation"]').all();
    const results = [];
    
    for (const rec of recommendations) {
      const title = await rec.locator('[data-testid="rec-title"]').textContent();
      const savings = await rec.locator('[data-testid="rec-savings"]').textContent();
      results.push({ title, savings });
    }
    
    return results;
  }
  
  // Risk Assessment
  async viewRiskDashboard() {
    await this.goto('risk');
    
    // Wait for risk data to load
    await this.page.waitForSelector('[data-testid="risk-matrix"]');
    
    const overallRisk = await this.page.locator('[data-testid="overall-risk-score"]').textContent();
    return parseFloat(overallRisk || '0');
  }
  
  async assessResourceRisk(resourceId: string) {
    const searchInput = this.page.locator('[data-testid="resource-search"]');
    await searchInput.fill(resourceId);
    await searchInput.press('Enter');
    
    // Wait for risk assessment
    await this.page.waitForSelector('[data-testid="risk-assessment-result"]');
    
    const riskLevel = await this.page.locator('[data-testid="risk-level"]').textContent();
    const riskFactors = await this.page.locator('[data-testid="risk-factors"]').allTextContents();
    
    return { riskLevel, riskFactors };
  }
  
  async createRiskMitigationPlan(riskId: string, actions: string[]) {
    const riskRow = this.page.locator(`[data-testid="risk-${riskId}"]`);
    const mitigateButton = riskRow.locator('[data-testid="mitigate-button"]');
    await mitigateButton.click();
    
    // Add mitigation actions
    for (const action of actions) {
      await this.page.click('[data-testid="add-action"]');
      await this.page.fill('[data-testid="action-input"]:last-child', action);
    }
    
    const savePlanButton = this.page.locator('[data-testid="save-mitigation-plan"]');
    await savePlanButton.click();
  }
  
  // Audit Trail
  async viewAuditLogs(filters?: {
    startDate?: string;
    endDate?: string;
    user?: string;
    action?: string;
    resource?: string;
  }) {
    await this.goto('audit');
    
    // Apply filters if provided
    if (filters) {
      if (filters.startDate) {
        await this.page.fill('[data-testid="start-date"]', filters.startDate);
      }
      if (filters.endDate) {
        await this.page.fill('[data-testid="end-date"]', filters.endDate);
      }
      if (filters.user) {
        await this.page.fill('[data-testid="user-filter"]', filters.user);
      }
      if (filters.action) {
        await this.page.selectOption('[data-testid="action-filter"]', filters.action);
      }
      if (filters.resource) {
        await this.page.fill('[data-testid="resource-filter"]', filters.resource);
      }
      
      const applyFiltersButton = this.page.locator('[data-testid="apply-filters"]');
      await applyFiltersButton.click();
    }
    
    // Wait for logs to load
    await this.page.waitForSelector('[data-testid="audit-logs-table"]');
    
    const logs = await this.page.locator('[data-testid="audit-log-entry"]').allTextContents();
    return logs;
  }
  
  async exportAuditReport(format: 'pdf' | 'csv' | 'json') {
    const exportButton = this.page.locator('[data-testid="export-audit-report"]');
    await exportButton.click();
    
    const formatOption = this.page.locator(`[data-testid="export-${format}"]`);
    const downloadPromise = this.page.waitForEvent('download');
    await formatOption.click();
    
    const download = await downloadPromise;
    return download;
  }
  
  // Regulatory Compliance
  async viewRegulatoryFrameworks() {
    await this.goto('regulatory');
    
    const frameworks = await this.page.locator('[data-testid="regulatory-framework"]').all();
    const results = [];
    
    for (const framework of frameworks) {
      const name = await framework.locator('[data-testid="framework-name"]').textContent();
      const status = await framework.locator('[data-testid="framework-status"]').textContent();
      const coverage = await framework.locator('[data-testid="framework-coverage"]').textContent();
      results.push({ name, status, coverage });
    }
    
    return results;
  }
  
  async generateComplianceReport(framework: string) {
    const frameworkCard = this.page.locator(`[data-testid="framework-${framework}"]`);
    const reportButton = frameworkCard.locator('[data-testid="generate-report"]');
    await reportButton.click();
    
    // Wait for report generation
    await this.page.waitForSelector('[data-testid="report-ready"]', { timeout: 30000 });
    
    const downloadButton = this.page.locator('[data-testid="download-report"]');
    const downloadPromise = this.page.waitForEvent('download');
    await downloadButton.click();
    
    const download = await downloadPromise;
    return download;
  }
  
  // Verification methods
  async verifyPolicyEnforcement(policyName: string) {
    const policyRow = this.page.locator(`[data-testid="policy-${policyName}"]`);
    const status = await policyRow.locator('[data-testid="enforcement-status"]').textContent();
    
    expect(status).toBe('Enforced');
    
    const violations = await policyRow.locator('[data-testid="violation-count"]').textContent();
    return parseInt(violations || '0');
  }
  
  async verifyComplianceImprovement(initialScore: number) {
    const currentScore = await this.viewComplianceDashboard();
    expect(currentScore).toBeGreaterThan(initialScore);
    
    return currentScore - initialScore;
  }
}