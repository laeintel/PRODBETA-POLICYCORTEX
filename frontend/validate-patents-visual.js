const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Create screenshots directory
const screenshotsDir = path.join(__dirname, 'patent-screenshots');
if (!fs.existsSync(screenshotsDir)) {
  fs.mkdirSync(screenshotsDir, { recursive: true });
}

async function validatePatents() {
  const browser = await chromium.launch({ 
    headless: false,
    args: ['--start-maximized']
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  console.log('🚀 Starting Patent Implementation Validation...\n');
  
  // Helper function to take screenshot
  async function captureScreen(name) {
    const screenshotPath = path.join(screenshotsDir, `${name}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`📸 Screenshot saved: ${name}.png`);
    return screenshotPath;
  }
  
  // Helper function to check element visibility
  async function checkElement(selector, description) {
    try {
      const element = await page.waitForSelector(selector, { timeout: 5000 });
      if (element) {
        console.log(`✅ Found: ${description}`);
        return true;
      }
    } catch (e) {
      console.log(`❌ Missing: ${description}`);
      return false;
    }
  }
  
  const testResults = {
    patent1: { passed: 0, failed: 0, features: [] },
    patent2: { passed: 0, failed: 0, features: [] },
    patent3: { passed: 0, failed: 0, features: [] },
    patent4: { passed: 0, failed: 0, features: [] }
  };
  
  try {
    // ====== PATENT 1: Cross-Domain Correlation Engine ======
    console.log('\n📋 PATENT 1: Cross-Domain Correlation Engine\n');
    console.log('Navigating to Correlations page...');
    
    await page.goto('http://localhost:3000/ai/correlations', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent1-correlations-main');
    
    // Check for correlation visualization
    const hasCorrelationViz = await checkElement(
      'canvas, svg, [class*="graph"], [class*="correlation"], [id*="correlation"]',
      'Correlation Visualization Graph'
    );
    testResults.patent1.features.push({ 
      name: 'Correlation Graph Visualization',
      status: hasCorrelationViz 
    });
    
    // Check for correlation types
    const correlationTypes = [
      'Direct', 'Indirect', 'Security', 'Cost', 'Performance', 'Compliance', 'Risk'
    ];
    
    for (const type of correlationTypes) {
      const found = await page.locator(`text=/${type}/i`).first().isVisible().catch(() => false);
      if (found) {
        console.log(`  ✓ Correlation type: ${type}`);
        testResults.patent1.passed++;
      }
    }
    
    // Look for What-If Analysis
    const whatIfButton = await page.locator('button:has-text("What"), button:has-text("Scenario"), button:has-text("Simulate")').first();
    if (await whatIfButton.isVisible().catch(() => false)) {
      console.log('  ✓ What-If Analysis capability found');
      testResults.patent1.features.push({ name: 'What-If Analysis', status: true });
      await whatIfButton.click();
      await page.waitForTimeout(1000);
      await captureScreen('patent1-what-if-analysis');
    } else {
      testResults.patent1.features.push({ name: 'What-If Analysis', status: false });
    }
    
    // ====== PATENT 2: Conversational Governance Intelligence ======
    console.log('\n📋 PATENT 2: Conversational Governance Intelligence\n');
    console.log('Navigating to Chat page...');
    
    await page.goto('http://localhost:3000/ai/chat', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent2-chat-interface');
    
    // Check for chat interface elements
    const hasChatInput = await checkElement(
      'input[type="text"], textarea, [contenteditable="true"]',
      'Chat Input Field'
    );
    testResults.patent2.features.push({ 
      name: 'Chat Input Interface',
      status: hasChatInput 
    });
    
    // Try to send a message
    const chatInput = await page.locator('input[type="text"], textarea').first();
    const sendButton = await page.locator('button').filter({ hasText: /send|submit|→|enter/i }).first();
    
    if (await chatInput.isVisible() && await sendButton.isVisible()) {
      console.log('Testing conversational AI...');
      await chatInput.fill('Show me compliance status for all resources');
      await captureScreen('patent2-chat-query-typed');
      
      await sendButton.click();
      await page.waitForTimeout(3000);
      await captureScreen('patent2-chat-response');
      
      // Check for response
      const hasResponse = await page.locator('.message, .chat-message, .response, [class*="message"]').count() > 0;
      testResults.patent2.features.push({ 
        name: 'Chat Response Generation',
        status: hasResponse 
      });
    }
    
    // ====== PATENT 3: Unified AI-Driven Platform ======
    console.log('\n📋 PATENT 3: Unified AI-Driven Platform\n');
    console.log('Navigating to Dashboard...');
    
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent3-dashboard-overview');
    
    // Check for unified metrics
    const metricCards = await page.locator('.card, [class*="card"], [class*="metric"]').count();
    console.log(`  Found ${metricCards} metric cards`);
    testResults.patent3.features.push({ 
      name: 'Unified Metrics Dashboard',
      status: metricCards > 0 
    });
    
    // Check governance sections
    const governanceSections = ['Security', 'Compliance', 'Cost', 'Operations', 'DevOps'];
    for (const section of governanceSections) {
      const hasSection = await page.locator(`text=/${section}/i`).first().isVisible().catch(() => false);
      if (hasSection) {
        console.log(`  ✓ Governance domain: ${section}`);
        testResults.patent3.passed++;
      }
    }
    
    // Navigate to Governance pages
    await page.goto('http://localhost:3000/governance', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent3-governance-main');
    
    // Check sub-pages
    const governancePages = [
      { path: '/governance/compliance', name: 'Compliance' },
      { path: '/governance/policies', name: 'Policies' },
      { path: '/governance/risk', name: 'Risk' },
      { path: '/governance/cost', name: 'Cost' }
    ];
    
    for (const govPage of governancePages) {
      await page.goto(`http://localhost:3000${govPage.path}`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1500);
      await captureScreen(`patent3-${govPage.name.toLowerCase()}`);
      console.log(`  ✓ ${govPage.name} page loaded`);
    }
    
    // ====== PATENT 4: Predictive Policy Compliance ======
    console.log('\n📋 PATENT 4: Predictive Policy Compliance\n');
    console.log('Navigating to AI/Predictions...');
    
    await page.goto('http://localhost:3000/ai', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent4-ai-main');
    
    // Check for prediction elements
    const predictionKeywords = ['predict', 'forecast', 'drift', 'anomaly', 'compliance', 'risk'];
    let foundPredictions = 0;
    
    for (const keyword of predictionKeywords) {
      const found = await page.locator(`text=/${keyword}/i`).first().isVisible().catch(() => false);
      if (found) {
        console.log(`  ✓ Prediction feature: ${keyword}`);
        foundPredictions++;
      }
    }
    
    testResults.patent4.features.push({ 
      name: 'Predictive Analytics',
      status: foundPredictions > 0 
    });
    
    // Check Risk page for drift detection
    await page.goto('http://localhost:3000/governance/risk', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await captureScreen('patent4-risk-drift-detection');
    
    const hasDriftDetection = await page.locator('text=/drift|deviation|anomal/i').first().isVisible().catch(() => false);
    testResults.patent4.features.push({ 
      name: 'Drift Detection',
      status: hasDriftDetection 
    });
    
    // ====== OVERALL UI VALIDATION ======
    console.log('\n📋 OVERALL UI VALIDATION\n');
    
    // Check navigation menu
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await captureScreen('main-landing-page');
    
    // Check for navigation elements
    const navItems = await page.locator('nav a, [role="navigation"] a, aside a').count();
    console.log(`  Found ${navItems} navigation items`);
    
    // Check theme toggle
    const themeToggle = await page.locator('[aria-label*="theme"], button:has-text("theme"), .theme-toggle').first();
    if (await themeToggle.isVisible().catch(() => false)) {
      console.log('  ✓ Theme toggle found');
      await themeToggle.click();
      await page.waitForTimeout(1000);
      await captureScreen('theme-switched');
    }
    
    // Check for errors on main pages
    const mainPages = [
      '/',
      '/dashboard',
      '/ai',
      '/governance',
      '/security',
      '/devops',
      '/operations'
    ];
    
    console.log('\n🔍 Checking for errors on main pages...\n');
    
    for (const pagePath of mainPages) {
      await page.goto(`http://localhost:3000${pagePath}`, { waitUntil: 'networkidle' });
      const errors = await page.locator('text=/error|failed|exception|undefined|null/i').count();
      if (errors > 0) {
        console.log(`  ⚠️ Potential issues on ${pagePath} (${errors} error indicators)`);
        await captureScreen(`errors-${pagePath.replace(/\//g, '-')}`);
      } else {
        console.log(`  ✓ ${pagePath} - No errors detected`);
      }
    }
    
  } catch (error) {
    console.error('Error during validation:', error);
    await captureScreen('error-state');
  } finally {
    // Generate summary report
    console.log('\n' + '='.repeat(60));
    console.log('📊 VALIDATION SUMMARY');
    console.log('='.repeat(60) + '\n');
    
    console.log('Patent 1 (Cross-Domain Correlation):');
    testResults.patent1.features.forEach(f => {
      console.log(`  ${f.status ? '✅' : '❌'} ${f.name}`);
    });
    
    console.log('\nPatent 2 (Conversational Governance):');
    testResults.patent2.features.forEach(f => {
      console.log(`  ${f.status ? '✅' : '❌'} ${f.name}`);
    });
    
    console.log('\nPatent 3 (Unified Platform):');
    testResults.patent3.features.forEach(f => {
      console.log(`  ${f.status ? '✅' : '❌'} ${f.name}`);
    });
    
    console.log('\nPatent 4 (Predictive Compliance):');
    testResults.patent4.features.forEach(f => {
      console.log(`  ${f.status ? '✅' : '❌'} ${f.name}`);
    });
    
    // Save results to file
    fs.writeFileSync(
      path.join(__dirname, 'patent-validation-results.json'),
      JSON.stringify(testResults, null, 2)
    );
    
    console.log('\n📁 Screenshots saved to:', screenshotsDir);
    console.log('📄 Results saved to: patent-validation-results.json\n');
    
    await browser.close();
  }
}

// Run validation
validatePatents().catch(console.error);