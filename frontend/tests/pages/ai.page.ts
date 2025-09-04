/**
 * AI Page Object Model
 */

import { Page, expect } from '@playwright/test';
import { performanceThresholds } from '../fixtures/test-fixtures';

export class AIPage {
  constructor(private page: Page) {}
  
  async goto(feature?: string) {
    const path = feature ? `/ai/${feature}` : '/ai';
    await this.page.goto(path);
    await this.page.waitForLoadState('networkidle');
  }
  
  // Conversational AI (Patent #2)
  async sendMessage(message: string) {
    const chatInput = this.page.locator('[data-testid="chat-input"]');
    await chatInput.fill(message);
    
    const sendButton = this.page.locator('[data-testid="send-message"]');
    
    // Measure response time for patent validation
    const startTime = Date.now();
    await sendButton.click();
    
    // Wait for AI response
    await this.page.waitForSelector('[data-testid="ai-response"]:last-child', {
      timeout: 5000
    });
    
    const responseTime = Date.now() - startTime;
    return { responseTime };
  }
  
  async verifyIntentClassification(message: string, expectedIntent: string) {
    await this.sendMessage(message);
    
    // Check intent classification metadata
    const responseMetadata = this.page.locator('[data-testid="response-metadata"]:last-child');
    const intent = await responseMetadata.locator('[data-testid="intent"]').textContent();
    
    expect(intent).toBe(expectedIntent);
  }
  
  async verifyEntityExtraction(message: string, expectedEntities: Record<string, string>) {
    await this.sendMessage(message);
    
    const responseMetadata = this.page.locator('[data-testid="response-metadata"]:last-child');
    const entities = await responseMetadata.locator('[data-testid="entities"]').textContent();
    
    const parsedEntities = JSON.parse(entities || '{}');
    for (const [key, value] of Object.entries(expectedEntities)) {
      expect(parsedEntities[key]).toBe(value);
    }
  }
  
  async translateNaturalLanguageToPolicy(query: string) {
    await this.goto('policy-translator');
    
    const input = this.page.locator('[data-testid="nl-input"]');
    await input.fill(query);
    
    const translateButton = this.page.locator('[data-testid="translate-button"]');
    await translateButton.click();
    
    // Wait for policy generation
    const policyOutput = this.page.locator('[data-testid="policy-output"]');
    await expect(policyOutput).toBeVisible();
    
    const policyJson = await policyOutput.textContent();
    return JSON.parse(policyJson || '{}');
  }
  
  // Cross-Domain Correlation (Patent #1)
  async analyzeCorrelations(domains: string[]) {
    await this.goto('correlations');
    
    // Select domains for analysis
    for (const domain of domains) {
      const checkbox = this.page.locator(`[data-testid="domain-${domain}"]`);
      await checkbox.check();
    }
    
    const analyzeButton = this.page.locator('[data-testid="analyze-correlations"]');
    
    // Measure correlation analysis time
    const startTime = Date.now();
    await analyzeButton.click();
    
    // Wait for results
    await this.page.waitForSelector('[data-testid="correlation-results"]', {
      timeout: 5000
    });
    
    const analysisTime = Date.now() - startTime;
    
    // Verify performance meets patent requirement (<100ms)
    expect(analysisTime).toBeLessThan(performanceThresholds.patents.correlation);
    
    const results = await this.page.locator('[data-testid="correlation-data"]').textContent();
    return { analysisTime, results: JSON.parse(results || '{}') };
  }
  
  async viewCorrelationGraph() {
    const graphContainer = this.page.locator('[data-testid="correlation-graph"]');
    await expect(graphContainer).toBeVisible();
    
    // Verify graph is interactive
    const node = graphContainer.locator('[data-testid="graph-node"]:first-child');
    await node.click();
    
    // Check if details panel appears
    await expect(this.page.locator('[data-testid="node-details"]')).toBeVisible();
  }
  
  // Predictive Compliance (Patent #4)
  async getPredictions(resourceId?: string) {
    await this.goto('predictions');
    
    const startTime = Date.now();
    
    if (resourceId) {
      // Get predictions for specific resource
      const searchInput = this.page.locator('[data-testid="resource-search"]');
      await searchInput.fill(resourceId);
      await searchInput.press('Enter');
    }
    
    // Wait for predictions to load
    await this.page.waitForSelector('[data-testid="prediction-results"]', {
      timeout: 5000
    });
    
    const predictionTime = Date.now() - startTime;
    
    // Verify performance meets patent requirement (<100ms)
    expect(predictionTime).toBeLessThan(performanceThresholds.patents.prediction);
    
    const predictions = await this.page.locator('[data-testid="predictions-data"]').textContent();
    return { predictionTime, predictions: JSON.parse(predictions || '[]') };
  }
  
  async verifyPredictionAccuracy() {
    const accuracyMetric = this.page.locator('[data-testid="accuracy-metric"]');
    const accuracy = parseFloat(await accuracyMetric.textContent() || '0');
    
    // Verify 99.2% accuracy as per patent claim
    expect(accuracy).toBeGreaterThanOrEqual(99.2);
  }
  
  async getRiskScore(resourceId: string) {
    const response = await this.page.request.get(`/api/v1/predictions/risk-score/${resourceId}`);
    const data = await response.json();
    
    return data.riskScore;
  }
  
  async submitFeedback(predictionId: string, feedback: 'correct' | 'incorrect', notes?: string) {
    const predictionRow = this.page.locator(`[data-testid="prediction-${predictionId}"]`);
    const feedbackButton = predictionRow.locator('[data-testid="feedback-button"]');
    await feedbackButton.click();
    
    // Select feedback type
    const feedbackOption = this.page.locator(`[data-testid="feedback-${feedback}"]`);
    await feedbackOption.click();
    
    if (notes) {
      await this.page.fill('[data-testid="feedback-notes"]', notes);
    }
    
    const submitButton = this.page.locator('[data-testid="submit-feedback"]');
    await submitButton.click();
    
    // Verify feedback submitted
    await expect(this.page.locator('[data-testid="feedback-success"]')).toBeVisible();
  }
  
  // Unified Platform (Patent #3)
  async getUnifiedMetrics() {
    await this.goto('unified-platform');
    
    const startTime = Date.now();
    
    // Load unified dashboard
    await this.page.waitForSelector('[data-testid="unified-metrics"]', {
      timeout: 5000
    });
    
    const loadTime = Date.now() - startTime;
    
    // Verify performance meets patent requirement (<500ms)
    expect(loadTime).toBeLessThan(performanceThresholds.patents.unified);
    
    const metrics = await this.page.locator('[data-testid="unified-data"]').textContent();
    return { loadTime, metrics: JSON.parse(metrics || '{}') };
  }
  
  async verifyAIEngineIntegration() {
    // Check all AI features are accessible
    const features = ['chat', 'correlations', 'predictions', 'unified-platform'];
    
    for (const feature of features) {
      const featureCard = this.page.locator(`[data-testid="ai-feature-${feature}"]`);
      await expect(featureCard).toBeVisible();
      
      // Verify feature is operational
      const status = await featureCard.locator('[data-testid="feature-status"]').textContent();
      expect(status).toBe('operational');
    }
  }
  
  // SHAP Explainability
  async viewFeatureImportance(predictionId: string) {
    const predictionRow = this.page.locator(`[data-testid="prediction-${predictionId}"]`);
    const explainButton = predictionRow.locator('[data-testid="explain-button"]');
    await explainButton.click();
    
    // Wait for SHAP visualization
    await expect(this.page.locator('[data-testid="shap-chart"]')).toBeVisible();
    
    const features = await this.page.locator('[data-testid="feature-importance"]').allTextContents();
    return features;
  }
  
  // Recommendation Engine
  async getRecommendations(context: string) {
    await this.goto('recommendations');
    
    const contextInput = this.page.locator('[data-testid="context-input"]');
    await contextInput.fill(context);
    
    const getRecommendationsButton = this.page.locator('[data-testid="get-recommendations"]');
    await getRecommendationsButton.click();
    
    // Wait for recommendations
    await this.page.waitForSelector('[data-testid="recommendations-list"]');
    
    const recommendations = await this.page.locator('[data-testid="recommendation-item"]').allTextContents();
    return recommendations;
  }
  
  async applyRecommendation(recommendationId: string) {
    const recommendation = this.page.locator(`[data-testid="recommendation-${recommendationId}"]`);
    const applyButton = recommendation.locator('[data-testid="apply-button"]');
    await applyButton.click();
    
    // Confirm application
    const confirmButton = this.page.locator('[data-testid="confirm-apply"]');
    await confirmButton.click();
    
    // Wait for success
    await expect(this.page.locator('[data-testid="recommendation-applied"]')).toBeVisible();
  }
  
  // Performance validation
  async validateConversationAccuracy() {
    const testCases = [
      { message: "Show me non-compliant resources", expectedIntent: "compliance_query" },
      { message: "Create a backup policy for VMs", expectedIntent: "policy_creation" },
      { message: "What's my cloud spend this month?", expectedIntent: "cost_query" }
    ];
    
    let correctCount = 0;
    
    for (const testCase of testCases) {
      await this.sendMessage(testCase.message);
      
      const responseMetadata = this.page.locator('[data-testid="response-metadata"]:last-child');
      const intent = await responseMetadata.locator('[data-testid="intent"]').textContent();
      
      if (intent === testCase.expectedIntent) {
        correctCount++;
      }
    }
    
    const accuracy = (correctCount / testCases.length) * 100;
    
    // Patent #2 requires 95% intent classification accuracy
    expect(accuracy).toBeGreaterThanOrEqual(95);
    
    return accuracy;
  }
}