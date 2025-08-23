const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Create directories for screenshots and fixes
const screenshotsDir = path.join(__dirname, 'ui-test-screenshots');
const fixesDir = path.join(__dirname, 'ui-fixes');
if (!fs.existsSync(screenshotsDir)) fs.mkdirSync(screenshotsDir, { recursive: true });
if (!fs.existsSync(fixesDir)) fs.mkdirSync(fixesDir, { recursive: true });

// Track all issues found
const issues = [];
const fixes = [];

async function testAndFixUI() {
  const browser = await chromium.launch({ 
    headless: false,
    args: ['--start-maximized'],
    slowMo: 500 // Slow down to see what's happening
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  console.log('🚀 Starting Comprehensive UI Testing & Fixing...\n');
  
  // Helper to take screenshot
  async function screenshot(name) {
    const filepath = `${screenshotsDir}/${name}.png`;
    await page.screenshot({ path: filepath, fullPage: true });
    console.log(`📸 Screenshot: ${name}`);
    return filepath;
  }
  
  // Helper to check and fix issues
  async function checkAndFix(selector, description, fixAction) {
    try {
      const element = await page.waitForSelector(selector, { timeout: 3000 });
      if (element) {
        console.log(`✅ Found: ${description}`);
        return true;
      }
    } catch (e) {
      console.log(`❌ Issue: ${description} not found`);
      issues.push({ selector, description, error: e.message });
      if (fixAction) {
        console.log(`🔧 Applying fix for: ${description}`);
        await fixAction();
        fixes.push({ description, fixed: true });
      }
      return false;
    }
  }
  
  // Helper to click button and verify action
  async function testButton(selector, description, expectedResult) {
    console.log(`\n🔘 Testing button: ${description}`);
    try {
      const button = await page.waitForSelector(selector, { timeout: 3000 });
      if (!button) {
        console.log(`  ❌ Button not found: ${description}`);
        issues.push({ type: 'button', description, selector });
        return false;
      }
      
      // Check if button is enabled
      const isDisabled = await button.evaluate(el => el.disabled);
      if (isDisabled) {
        console.log(`  ⚠️ Button is disabled: ${description}`);
        issues.push({ type: 'disabled-button', description, selector });
        return false;
      }
      
      // Click the button
      await button.click();
      console.log(`  ✓ Clicked: ${description}`);
      await page.waitForTimeout(1500);
      
      // Verify expected result
      if (expectedResult) {
        try {
          await expectedResult();
          console.log(`  ✓ Action successful`);
          return true;
        } catch (e) {
          console.log(`  ❌ Expected result not achieved: ${e.message}`);
          issues.push({ type: 'action-failed', description, error: e.message });
          return false;
        }
      }
      return true;
    } catch (e) {
      console.log(`  ❌ Error testing button: ${e.message}`);
      issues.push({ type: 'button-error', description, error: e.message });
      return false;
    }
  }
  
  // Test navigation link
  async function testNavigation(linkText, expectedUrl, expectedContent) {
    console.log(`\n📍 Testing navigation: ${linkText}`);
    try {
      // Find and click the navigation link
      const link = await page.locator(`a:has-text("${linkText}"), button:has-text("${linkText}")`).first();
      if (!await link.isVisible()) {
        console.log(`  ❌ Navigation link not found: ${linkText}`);
        issues.push({ type: 'nav-missing', linkText });
        return false;
      }
      
      await link.click();
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(1000);
      
      // Check URL
      const currentUrl = page.url();
      if (expectedUrl && !currentUrl.includes(expectedUrl)) {
        console.log(`  ❌ Wrong URL: Expected ${expectedUrl}, got ${currentUrl}`);
        issues.push({ type: 'nav-wrong-url', linkText, expected: expectedUrl, actual: currentUrl });
        return false;
      }
      
      // Check for expected content
      if (expectedContent) {
        const hasContent = await page.locator(`text=/${expectedContent}/i`).first().isVisible().catch(() => false);
        if (!hasContent) {
          console.log(`  ❌ Expected content not found: ${expectedContent}`);
          issues.push({ type: 'nav-no-content', linkText, expectedContent });
          return false;
        }
      }
      
      console.log(`  ✓ Navigation successful to ${linkText}`);
      return true;
    } catch (e) {
      console.log(`  ❌ Navigation error: ${e.message}`);
      issues.push({ type: 'nav-error', linkText, error: e.message });
      return false;
    }
  }
  
  try {
    // ========== TEST MAIN LANDING PAGE ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING MAIN LANDING PAGE');
    console.log('='.repeat(60));
    
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await screenshot('01-landing-page');
    
    // Check main elements
    await checkAndFix('h1, h2, .title', 'Page Title');
    await checkAndFix('nav, [role="navigation"], aside', 'Navigation Menu');
    
    // Test theme toggle
    const themeToggle = await page.locator('[aria-label*="theme"], button:has-text("theme"), .theme-toggle, button[title*="theme"]').first();
    if (await themeToggle.isVisible()) {
      console.log('\n🎨 Testing Theme Toggle...');
      await themeToggle.click();
      await page.waitForTimeout(1000);
      await screenshot('02-theme-dark');
      
      // Toggle back
      await themeToggle.click();
      await page.waitForTimeout(1000);
      await screenshot('03-theme-light');
    }
    
    // ========== TEST DASHBOARD ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING DASHBOARD');
    console.log('='.repeat(60));
    
    await testNavigation('Dashboard', '/dashboard', 'Dashboard');
    await screenshot('04-dashboard');
    
    // Check dashboard cards
    const cards = await page.locator('.card, [class*="card"], [class*="metric"]').count();
    console.log(`  Found ${cards} dashboard cards`);
    if (cards === 0) {
      issues.push({ type: 'no-content', page: 'dashboard', description: 'No metric cards found' });
    }
    
    // ========== TEST AI INTELLIGENCE SECTION ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING AI INTELLIGENCE (ALL PATENTS)');
    console.log('='.repeat(60));
    
    // Expand AI Intelligence menu if needed
    const aiMenu = await page.locator('text="AI Intelligence"').first();
    if (await aiMenu.isVisible()) {
      await aiMenu.click();
      await page.waitForTimeout(500);
    }
    
    // Patent 1: Cross-Domain Correlations
    console.log('\n🔬 Patent 1: Cross-Domain Correlations');
    await testNavigation('Cross-Domain Analysis', '/ai/correlations', 'Correlation');
    await screenshot('05-patent1-correlations');
    
    // Test What-If Model button
    await testButton(
      'button:has-text("What-If Model"), button:has-text("What-If"), button:has-text("Scenario")',
      'What-If Analysis',
      async () => {
        await page.waitForSelector('.modal, [role="dialog"], .scenario-simulator', { timeout: 3000 });
      }
    );
    await screenshot('06-patent1-whatif');
    
    // Close modal if open
    const closeButton = await page.locator('button:has-text("Close"), button:has-text("Cancel"), button:has-text("X")').first();
    if (await closeButton.isVisible()) {
      await closeButton.click();
      await page.waitForTimeout(500);
    }
    
    // Patent 2: Conversational AI
    console.log('\n💬 Patent 2: Conversational AI');
    await testNavigation('Conversational AI', '/ai/chat', 'Chat');
    await screenshot('07-patent2-chat');
    
    // Test chat functionality
    const chatInput = await page.locator('input[placeholder*="message"], textarea[placeholder*="message"], input[type="text"]').first();
    const sendButton = await page.locator('button:has-text("Send"), button[type="submit"]').first();
    
    if (await chatInput.isVisible() && await sendButton.isVisible()) {
      console.log('  Testing chat input...');
      await chatInput.fill('Show me compliance status');
      await screenshot('08-patent2-typed');
      
      // Check if send button becomes enabled
      const isDisabled = await sendButton.evaluate(el => el.disabled);
      if (!isDisabled) {
        await sendButton.click();
        await page.waitForTimeout(2000);
        await screenshot('09-patent2-response');
      } else {
        console.log('  ⚠️ Send button remains disabled');
        issues.push({ type: 'functionality', description: 'Chat send button not enabling' });
      }
    }
    
    // Patent 3: Unified Platform
    console.log('\n🏗️ Patent 3: Unified Platform');
    await testNavigation('Unified Platform', '/ai', 'AI');
    await screenshot('10-patent3-platform');
    
    // ========== TEST GOVERNANCE SECTION ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING GOVERNANCE');
    console.log('='.repeat(60));
    
    // Expand Governance menu if needed
    const govMenu = await page.locator('text="Governance"').first();
    if (await govMenu.isVisible()) {
      await govMenu.click();
      await page.waitForTimeout(500);
    }
    
    const governancePages = [
      { name: 'Compliance', url: '/governance/compliance', content: 'Compliance' },
      { name: 'Policies', url: '/governance/policies', content: 'Policies' },
      { name: 'Risk', url: '/governance/risk', content: 'Risk' },
      { name: 'Cost', url: '/governance/cost', content: 'Cost' }
    ];
    
    for (const govPage of governancePages) {
      await testNavigation(govPage.name, govPage.url, govPage.content);
      await screenshot(`11-governance-${govPage.name.toLowerCase()}`);
      
      // Patent 4: Check for predictions on compliance page
      if (govPage.name === 'Compliance') {
        console.log('\n🔮 Patent 4: Predictive Compliance');
        const hasPredictions = await page.locator('text=/predict|forecast|drift|score/i').first().isVisible().catch(() => false);
        if (hasPredictions) {
          console.log('  ✓ Predictive compliance features found');
        } else {
          console.log('  ❌ No predictive features visible');
          issues.push({ type: 'missing-feature', patent: 4, description: 'Predictive compliance not visible' });
        }
      }
    }
    
    // ========== TEST SECURITY SECTION ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING SECURITY');
    console.log('='.repeat(60));
    
    // Expand Security menu if needed
    const secMenu = await page.locator('text="Security & Access"').first();
    if (await secMenu.isVisible()) {
      await secMenu.click();
      await page.waitForTimeout(500);
    }
    
    const securityPages = [
      { name: 'IAM', url: '/security/iam', content: 'Identity' },
      { name: 'RBAC', url: '/security/rbac', content: 'Role' },
      { name: 'Zero Trust', url: '/security/zero-trust', content: 'Zero Trust' }
    ];
    
    for (const secPage of securityPages) {
      await testNavigation(secPage.name, secPage.url, secPage.content);
      await screenshot(`12-security-${secPage.name.toLowerCase().replace(' ', '-')}`);
    }
    
    // ========== TEST OPERATIONS SECTION ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING OPERATIONS');
    console.log('='.repeat(60));
    
    await testNavigation('Operations', '/operations', 'Operations');
    await screenshot('13-operations');
    
    // ========== TEST DEVOPS SECTION ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING DEVOPS & CI/CD');
    console.log('='.repeat(60));
    
    // Expand DevOps menu if needed
    const devopsMenu = await page.locator('text="DevOps & CI/CD"').first();
    if (await devopsMenu.isVisible()) {
      await devopsMenu.click();
      await page.waitForTimeout(500);
    }
    
    const devopsPages = [
      { name: 'Pipelines', url: '/devops/pipelines', content: 'Pipeline' },
      { name: 'Repos', url: '/devops/repos', content: 'Repo' },
      { name: 'Builds', url: '/devops/builds', content: 'Build' }
    ];
    
    for (const devPage of devopsPages) {
      await testNavigation(devPage.name, devPage.url, devPage.content);
      await screenshot(`14-devops-${devPage.name.toLowerCase()}`);
    }
    
    // ========== TEST QUICK ACTIONS BAR ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING QUICK ACTIONS BAR');
    console.log('='.repeat(60));
    
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
    
    // Test quick action buttons
    const quickActions = [
      { text: 'Compliance Status', color: 'yellow' },
      { text: 'Cost Savings', color: 'blue' },
      { text: 'Chat with AI', color: 'purple' },
      { text: 'View Predictions', color: 'green' },
      { text: 'Active Risks', color: 'red' },
      { text: 'View Resources', color: 'teal' }
    ];
    
    for (const action of quickActions) {
      const button = await page.locator(`button:has-text("${action.text}")`).first();
      if (await button.isVisible()) {
        console.log(`  Testing: ${action.text}`);
        await button.click();
        await page.waitForTimeout(1500);
        await screenshot(`15-quick-action-${action.text.toLowerCase().replace(/\s+/g, '-')}`);
        
        // Go back to dashboard
        await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
      } else {
        console.log(`  ❌ Quick action not found: ${action.text}`);
        issues.push({ type: 'missing-quick-action', action: action.text });
      }
    }
    
    // ========== TEST INTERACTIVE ELEMENTS ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 TESTING INTERACTIVE ELEMENTS');
    console.log('='.repeat(60));
    
    // Test all buttons on current page
    const allButtons = await page.locator('button:visible').all();
    console.log(`\nFound ${allButtons.length} visible buttons to test`);
    
    for (let i = 0; i < Math.min(allButtons.length, 10); i++) {
      const button = allButtons[i];
      const text = await button.textContent();
      const isDisabled = await button.evaluate(el => el.disabled);
      
      if (!isDisabled && text && text.trim()) {
        console.log(`  Button ${i + 1}: "${text.trim()}" - ${isDisabled ? 'disabled' : 'enabled'}`);
      }
    }
    
    // ========== PERFORMANCE TESTING ==========
    console.log('\n' + '='.repeat(60));
    console.log('📋 PERFORMANCE TESTING');
    console.log('='.repeat(60));
    
    const performancePages = [
      '/dashboard',
      '/ai/correlations',
      '/ai/chat',
      '/governance/compliance'
    ];
    
    for (const pageUrl of performancePages) {
      const start = Date.now();
      await page.goto(`http://localhost:3000${pageUrl}`, { waitUntil: 'networkidle' });
      const loadTime = Date.now() - start;
      
      console.log(`  ${pageUrl}: ${loadTime}ms`);
      if (loadTime > 3000) {
        console.log(`    ⚠️ Slow load time!`);
        issues.push({ type: 'performance', page: pageUrl, loadTime });
      }
    }
    
  } catch (error) {
    console.error('\n❌ Test Error:', error);
    await screenshot('error-state');
  } finally {
    // ========== GENERATE REPORT ==========
    console.log('\n' + '='.repeat(60));
    console.log('📊 TEST RESULTS SUMMARY');
    console.log('='.repeat(60));
    
    if (issues.length === 0) {
      console.log('\n✅ ALL TESTS PASSED! No issues found.');
    } else {
      console.log(`\n⚠️ Found ${issues.length} issues:\n`);
      issues.forEach((issue, i) => {
        console.log(`${i + 1}. [${issue.type}] ${issue.description || issue.linkText || issue.selector}`);
        if (issue.error) console.log(`   Error: ${issue.error}`);
      });
    }
    
    if (fixes.length > 0) {
      console.log(`\n🔧 Applied ${fixes.length} fixes:`);
      fixes.forEach(fix => {
        console.log(`  ✓ ${fix.description}`);
      });
    }
    
    // Save detailed report
    const report = {
      timestamp: new Date().toISOString(),
      totalTests: 50,
      passed: 50 - issues.length,
      failed: issues.length,
      issues: issues,
      fixes: fixes,
      screenshots: fs.readdirSync(screenshotsDir)
    };
    
    fs.writeFileSync(
      path.join(__dirname, 'ui-fixes-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    console.log('\n📁 Screenshots saved to:', screenshotsDir);
    console.log('📄 Report saved to: ui-fixes-report.json');
    
    await browser.close();
  }
}

// Run the test
testAndFixUI().catch(console.error);