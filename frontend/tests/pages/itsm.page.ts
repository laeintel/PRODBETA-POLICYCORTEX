/**
 * ITSM Page Object Model
 */

import { Page, expect } from '@playwright/test';

export class ITSMPage {
  constructor(private page: Page) {}
  
  async goto(module?: string) {
    const path = module ? `/itsm/${module}` : '/itsm';
    await this.page.goto(path);
    await this.page.waitForLoadState('networkidle');
  }
  
  async navigateToModule(module: string) {
    const moduleCard = this.page.locator(`[data-testid="itsm-module-${module}"]`);
    await moduleCard.click();
    await this.page.waitForURL(new RegExp(`/itsm/${module}`));
  }
  
  // Incident Management
  async createIncident(data: {
    title: string;
    description: string;
    priority: 'critical' | 'high' | 'medium' | 'low';
    category: string;
  }) {
    await this.navigateToModule('incidents');
    
    const createButton = this.page.locator('[data-testid="create-incident"]');
    await createButton.click();
    
    // Fill incident form
    await this.page.fill('[data-testid="incident-title"]', data.title);
    await this.page.fill('[data-testid="incident-description"]', data.description);
    await this.page.selectOption('[data-testid="incident-priority"]', data.priority);
    await this.page.selectOption('[data-testid="incident-category"]', data.category);
    
    // Submit
    const submitButton = this.page.locator('[data-testid="submit-incident"]');
    await submitButton.click();
    
    // Wait for success
    await expect(this.page.locator('[data-testid="incident-created"]')).toBeVisible();
  }
  
  async updateIncidentStatus(incidentId: string, status: string) {
    const incidentRow = this.page.locator(`[data-testid="incident-${incidentId}"]`);
    const statusDropdown = incidentRow.locator('[data-testid="status-dropdown"]');
    await statusDropdown.selectOption(status);
    
    // Confirm update
    await this.page.waitForResponse(resp => 
      resp.url().includes(`/api/itsm/incidents/${incidentId}`) && resp.status() === 200
    );
  }
  
  // Change Management
  async createChangeRequest(data: {
    title: string;
    description: string;
    type: 'standard' | 'normal' | 'emergency';
    impactAnalysis: string;
    rollbackPlan: string;
  }) {
    await this.navigateToModule('changes');
    
    const createButton = this.page.locator('[data-testid="create-change"]');
    await createButton.click();
    
    // Fill change request form
    await this.page.fill('[data-testid="change-title"]', data.title);
    await this.page.fill('[data-testid="change-description"]', data.description);
    await this.page.selectOption('[data-testid="change-type"]', data.type);
    await this.page.fill('[data-testid="impact-analysis"]', data.impactAnalysis);
    await this.page.fill('[data-testid="rollback-plan"]', data.rollbackPlan);
    
    // Submit for approval
    const submitButton = this.page.locator('[data-testid="submit-change"]');
    await submitButton.click();
  }
  
  async approveChangeRequest(changeId: string, comments?: string) {
    const changeRow = this.page.locator(`[data-testid="change-${changeId}"]`);
    const approveButton = changeRow.locator('[data-testid="approve-button"]');
    await approveButton.click();
    
    if (comments) {
      await this.page.fill('[data-testid="approval-comments"]', comments);
    }
    
    const confirmButton = this.page.locator('[data-testid="confirm-approval"]');
    await confirmButton.click();
  }
  
  // Service Catalog
  async requestService(serviceName: string, parameters: Record<string, string>) {
    await this.navigateToModule('catalog');
    
    const serviceCard = this.page.locator(`[data-testid="service-${serviceName}"]`);
    await serviceCard.click();
    
    // Fill service request form
    for (const [key, value] of Object.entries(parameters)) {
      await this.page.fill(`[data-testid="param-${key}"]`, value);
    }
    
    const requestButton = this.page.locator('[data-testid="request-service"]');
    await requestButton.click();
    
    // Verify request submitted
    await expect(this.page.locator('[data-testid="request-submitted"]')).toBeVisible();
  }
  
  // Resource Inventory
  async searchResources(query: string) {
    await this.navigateToModule('inventory');
    
    const searchInput = this.page.locator('[data-testid="resource-search"]');
    await searchInput.fill(query);
    await searchInput.press('Enter');
    
    // Wait for results
    await this.page.waitForSelector('[data-testid="resource-results"]');
  }
  
  async viewResourceDetails(resourceId: string) {
    const resourceRow = this.page.locator(`[data-testid="resource-${resourceId}"]`);
    await resourceRow.click();
    
    // Wait for details panel
    await expect(this.page.locator('[data-testid="resource-details"]')).toBeVisible();
  }
  
  async updateResourceTags(resourceId: string, tags: Record<string, string>) {
    await this.viewResourceDetails(resourceId);
    
    const editTagsButton = this.page.locator('[data-testid="edit-tags"]');
    await editTagsButton.click();
    
    // Clear existing tags
    const clearButton = this.page.locator('[data-testid="clear-tags"]');
    await clearButton.click();
    
    // Add new tags
    for (const [key, value] of Object.entries(tags)) {
      await this.page.click('[data-testid="add-tag"]');
      await this.page.fill('[data-testid="tag-key-input"]', key);
      await this.page.fill('[data-testid="tag-value-input"]', value);
    }
    
    const saveButton = this.page.locator('[data-testid="save-tags"]');
    await saveButton.click();
  }
  
  // Performance Monitoring
  async checkServiceHealth(serviceName: string) {
    await this.navigateToModule('monitoring');
    
    const serviceCard = this.page.locator(`[data-testid="service-health-${serviceName}"]`);
    const status = await serviceCard.locator('[data-testid="health-status"]').textContent();
    
    return status;
  }
  
  async viewPerformanceMetrics(timeRange: '1h' | '24h' | '7d' | '30d') {
    await this.navigateToModule('monitoring');
    
    const timeSelector = this.page.locator('[data-testid="time-range-selector"]');
    await timeSelector.selectOption(timeRange);
    
    // Wait for metrics to load
    await this.page.waitForResponse(resp => 
      resp.url().includes('/api/itsm/metrics') && resp.status() === 200
    );
  }
  
  // Workflow Automation
  async createAutomationWorkflow(data: {
    name: string;
    trigger: string;
    conditions: string[];
    actions: string[];
  }) {
    await this.navigateToModule('automation');
    
    const createButton = this.page.locator('[data-testid="create-workflow"]');
    await createButton.click();
    
    // Configure workflow
    await this.page.fill('[data-testid="workflow-name"]', data.name);
    await this.page.selectOption('[data-testid="workflow-trigger"]', data.trigger);
    
    // Add conditions
    for (const condition of data.conditions) {
      await this.page.click('[data-testid="add-condition"]');
      await this.page.fill('[data-testid="condition-input"]', condition);
    }
    
    // Add actions
    for (const action of data.actions) {
      await this.page.click('[data-testid="add-action"]');
      await this.page.selectOption('[data-testid="action-select"]', action);
    }
    
    const saveButton = this.page.locator('[data-testid="save-workflow"]');
    await saveButton.click();
  }
  
  async toggleWorkflowStatus(workflowId: string, enabled: boolean) {
    const workflowRow = this.page.locator(`[data-testid="workflow-${workflowId}"]`);
    const toggleSwitch = workflowRow.locator('[data-testid="workflow-toggle"]');
    
    const currentState = await toggleSwitch.getAttribute('aria-checked') === 'true';
    
    if (currentState !== enabled) {
      await toggleSwitch.click();
    }
  }
  
  // Knowledge Base
  async searchKnowledgeBase(query: string) {
    await this.navigateToModule('knowledge');
    
    const searchInput = this.page.locator('[data-testid="kb-search"]');
    await searchInput.fill(query);
    await searchInput.press('Enter');
    
    // Wait for results
    await this.page.waitForSelector('[data-testid="kb-results"]');
  }
  
  async createKnowledgeArticle(data: {
    title: string;
    content: string;
    category: string;
    tags: string[];
  }) {
    await this.navigateToModule('knowledge');
    
    const createButton = this.page.locator('[data-testid="create-article"]');
    await createButton.click();
    
    // Fill article form
    await this.page.fill('[data-testid="article-title"]', data.title);
    await this.page.fill('[data-testid="article-content"]', data.content);
    await this.page.selectOption('[data-testid="article-category"]', data.category);
    
    // Add tags
    for (const tag of data.tags) {
      await this.page.fill('[data-testid="tag-input"]', tag);
      await this.page.press('[data-testid="tag-input"]', 'Enter');
    }
    
    const publishButton = this.page.locator('[data-testid="publish-article"]');
    await publishButton.click();
  }
  
  // Verification methods
  async verifyModuleAccess(module: string) {
    await this.navigateToModule(module);
    await expect(this.page).toHaveURL(new RegExp(`/itsm/${module}`));
    
    const moduleHeader = this.page.locator('[data-testid="module-header"]');
    await expect(moduleHeader).toBeVisible();
  }
  
  async verifyIncidentCreated(incidentTitle: string) {
    const incident = this.page.locator(`[data-testid="incident-title"]:has-text("${incidentTitle}")`);
    await expect(incident).toBeVisible();
  }
  
  async verifyWorkflowExecuted(workflowId: string) {
    const executionLog = this.page.locator(`[data-testid="workflow-execution-${workflowId}"]`);
    await expect(executionLog).toBeVisible();
    
    const status = await executionLog.locator('[data-testid="execution-status"]').textContent();
    expect(status).toBe('Success');
  }
}