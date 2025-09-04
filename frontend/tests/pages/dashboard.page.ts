/**
 * Dashboard Page Object Model
 */

import { Page, expect } from '@playwright/test';
import { selectors } from '../fixtures/test-fixtures';

export class DashboardPage {
  constructor(private page: Page) {}
  
  async goto(section?: string) {
    const path = section ? `/dashboard/${section}` : '/dashboard';
    await this.page.goto(path);
    await this.page.waitForLoadState('networkidle');
  }
  
  async waitForDataLoad() {
    // Wait for data to load
    await this.page.waitForResponse(resp => 
      resp.url().includes('/api/v1/metrics') && resp.status() === 200,
      { timeout: 10000 }
    );
    
    // Wait for charts to render
    await this.page.waitForSelector(selectors.dashboard.chart, { state: 'visible' });
  }
  
  async toggleView(mode: 'card' | 'visualization') {
    const viewToggle = this.page.locator(selectors.dashboard.viewToggle);
    const currentMode = await viewToggle.getAttribute('data-mode');
    
    if (currentMode !== mode) {
      await viewToggle.click();
      await this.page.waitForTimeout(300); // Wait for animation
    }
  }
  
  async getMetricValue(metricName: string): Promise<string> {
    const metricCard = this.page.locator(`${selectors.dashboard.metricCard}:has-text("${metricName}")`);
    const value = await metricCard.locator('[data-testid="metric-value"]').textContent();
    return value || '';
  }
  
  async verifyMetricTrend(metricName: string, trend: 'up' | 'down' | 'stable') {
    const metricCard = this.page.locator(`${selectors.dashboard.metricCard}:has-text("${metricName}")`);
    const trendIndicator = metricCard.locator('[data-testid="trend-indicator"]');
    await expect(trendIndicator).toHaveAttribute('data-trend', trend);
  }
  
  async exportData(format: 'csv' | 'json') {
    const exportButton = this.page.locator(selectors.dashboard.exportButton);
    await exportButton.click();
    
    const formatOption = this.page.locator(`[data-testid="export-${format}"]`);
    const downloadPromise = this.page.waitForEvent('download');
    await formatOption.click();
    
    const download = await downloadPromise;
    return download;
  }
  
  async refreshData() {
    const refreshButton = this.page.locator(selectors.dashboard.refreshButton);
    await refreshButton.click();
    await this.waitForDataLoad();
  }
  
  async openChartFullscreen(chartTitle: string) {
    const chart = this.page.locator(`${selectors.dashboard.chart}:has-text("${chartTitle}")`);
    const fullscreenButton = chart.locator('[data-testid="fullscreen-button"]');
    await fullscreenButton.click();
    
    await expect(this.page.locator('[data-testid="fullscreen-modal"]')).toBeVisible();
  }
  
  async closeFullscreen() {
    const closeButton = this.page.locator('[data-testid="fullscreen-close"]');
    await closeButton.click();
    await expect(this.page.locator('[data-testid="fullscreen-modal"]')).toBeHidden();
  }
  
  async drillIntoChart(chartTitle: string, dataPoint: string) {
    const chart = this.page.locator(`${selectors.dashboard.chart}:has-text("${chartTitle}")`);
    const point = chart.locator(`[data-testid="data-point-${dataPoint}"]`);
    await point.click();
    
    // Wait for drill-in view
    await this.page.waitForSelector('[data-testid="drill-in-view"]');
  }
  
  async navigateToSection(section: string) {
    const navItem = this.page.locator(`${selectors.navigation.sidebar} a:has-text("${section}")`);
    await navItem.click();
    await this.page.waitForURL(new RegExp(section.toLowerCase()));
  }
  
  async searchDashboard(query: string) {
    const searchInput = this.page.locator('[data-testid="dashboard-search"]');
    await searchInput.fill(query);
    await searchInput.press('Enter');
    
    // Wait for search results
    await this.page.waitForResponse(resp => 
      resp.url().includes('/api/search') && resp.status() === 200
    );
  }
  
  async verifyCardLayout() {
    const cards = await this.page.locator(selectors.dashboard.metricCard).count();
    expect(cards).toBeGreaterThan(0);
    
    // Verify responsive layout
    const firstCard = this.page.locator(selectors.dashboard.metricCard).first();
    const cardBox = await firstCard.boundingBox();
    expect(cardBox?.width).toBeGreaterThan(200);
  }
  
  async verifyVisualizationLayout() {
    const charts = await this.page.locator(selectors.dashboard.chart).count();
    expect(charts).toBeGreaterThan(0);
    
    // Verify charts are rendered
    for (let i = 0; i < charts; i++) {
      const chart = this.page.locator(selectors.dashboard.chart).nth(i);
      await expect(chart.locator('canvas, svg')).toBeVisible();
    }
  }
  
  async measureDashboardLoadTime(): Promise<number> {
    const startTime = Date.now();
    await this.goto();
    await this.waitForDataLoad();
    const endTime = Date.now();
    return endTime - startTime;
  }
  
  async verifyMetricUpdate(metricName: string, expectedValue: string) {
    const actualValue = await this.getMetricValue(metricName);
    expect(actualValue).toBe(expectedValue);
  }
  
  async interactWithFilter(filterName: string, value: string) {
    const filter = this.page.locator(`[data-testid="filter-${filterName}"]`);
    await filter.selectOption(value);
    await this.waitForDataLoad();
  }
  
  async verifyDataFreshness() {
    const timestamp = this.page.locator('[data-testid="last-updated"]');
    const timeText = await timestamp.textContent();
    
    // Parse and verify timestamp is recent (within last 5 minutes)
    const lastUpdate = new Date(timeText || '');
    const now = new Date();
    const diffMinutes = (now.getTime() - lastUpdate.getTime()) / 60000;
    expect(diffMinutes).toBeLessThan(5);
  }
  
  async checkAccessibility() {
    // Check for proper ARIA labels
    const charts = this.page.locator(selectors.dashboard.chart);
    const chartsCount = await charts.count();
    
    for (let i = 0; i < chartsCount; i++) {
      const chart = charts.nth(i);
      await expect(chart).toHaveAttribute('aria-label', /.+/);
    }
    
    // Check keyboard navigation
    await this.page.keyboard.press('Tab');
    const focusedElement = await this.page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();
  }
}