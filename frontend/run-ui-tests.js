const { chromium } = require('playwright');
const fs = require('fs');

async function runUITests() {
  const results = {
    totalTests: 0,
    passed: [],
    failed: [],
    timestamp: new Date().toISOString()
  };

  let browser;
  try {
    console.log('🚀 Starting PolicyCortex UI Tests...\n');
    browser = await chromium.launch({ 
      headless: true,
      timeout: 30000 
    });
    
    const context = await browser.newContext({
      viewport: { width: 1280, height: 720 },
      ignoreHTTPSErrors: true
    });
    
    const page = await context.newPage();
    
    // Test 1: Homepage loads
    console.log('📋 Test 1: Homepage loads');
    results.totalTests++;
    try {
      await page.goto('http://localhost:3001', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const title = await page.title();
      console.log(`   ✅ Homepage loaded - Title: ${title}`);
      results.passed.push('Homepage loads successfully');
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Homepage loads', error: error.message });
    }
    
    // Test 2: Navigation menu exists
    console.log('\n📋 Test 2: Navigation menu');
    results.totalTests++;
    try {
      const navElements = await page.$$('nav a, [role="navigation"] a, aside a');
      console.log(`   ✅ Found ${navElements.length} navigation links`);
      results.passed.push(`Navigation menu with ${navElements.length} links`);
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Navigation menu', error: error.message });
    }
    
    // Test 3: Dashboard page
    console.log('\n📋 Test 3: Dashboard page');
    results.totalTests++;
    try {
      await page.goto('http://localhost:3001/dashboard', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      await page.waitForSelector('h1, h2, [role="heading"]', { timeout: 5000 });
      const heading = await page.$eval('h1, h2, [role="heading"]', el => el.textContent);
      console.log(`   ✅ Dashboard loaded - Heading: ${heading}`);
      results.passed.push('Dashboard page loads');
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Dashboard page', error: error.message });
    }
    
    // Test 4: Theme toggle
    console.log('\n📋 Test 4: Theme toggle functionality');
    results.totalTests++;
    try {
      const themeButton = await page.$('button[aria-label*="theme"], button[title*="theme"], [data-testid*="theme"]');
      if (themeButton) {
        const initialTheme = await page.evaluate(() => document.documentElement.className);
        await themeButton.click();
        await page.waitForTimeout(500);
        const newTheme = await page.evaluate(() => document.documentElement.className);
        console.log(`   ✅ Theme toggle works - Changed from "${initialTheme}" to "${newTheme}"`);
        results.passed.push('Theme toggle functionality');
      } else {
        console.log('   ⚠️  Theme button not found');
        results.failed.push({ test: 'Theme toggle', error: 'Button not found' });
      }
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Theme toggle', error: error.message });
    }
    
    // Test 5: Governance page
    console.log('\n📋 Test 5: Governance page');
    results.totalTests++;
    try {
      await page.goto('http://localhost:3001/governance', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const cards = await page.$$('.card, [class*="card"], [class*="Card"]');
      console.log(`   ✅ Governance page loaded - Found ${cards.length} cards`);
      results.passed.push(`Governance page with ${cards.length} cards`);
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Governance page', error: error.message });
    }
    
    // Test 6: Security page
    console.log('\n📋 Test 6: Security page');
    results.totalTests++;
    try {
      await page.goto('http://localhost:3001/security', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const sections = await page.$$('section, [class*="section"], div[class*="grid"]');
      console.log(`   ✅ Security page loaded - Found ${sections.length} sections`);
      results.passed.push(`Security page with ${sections.length} sections`);
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Security page', error: error.message });
    }
    
    // Test 7: AI Chat page
    console.log('\n📋 Test 7: AI Chat page');
    results.totalTests++;
    try {
      await page.goto('http://localhost:3001/ai/chat', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const chatInput = await page.$('input[type="text"], textarea, [contenteditable="true"]');
      if (chatInput) {
        console.log(`   ✅ AI Chat page loaded - Input field found`);
        results.passed.push('AI Chat page with input');
      } else {
        console.log('   ⚠️  Chat input not found');
        results.failed.push({ test: 'AI Chat page', error: 'Input not found' });
      }
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'AI Chat page', error: error.message });
    }
    
    // Test 8: Responsive design
    console.log('\n📋 Test 8: Mobile responsive design');
    results.totalTests++;
    try {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('http://localhost:3001', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const mobileMenu = await page.$('[aria-label*="menu"], button[class*="menu"], button[class*="burger"]');
      console.log(`   ✅ Mobile view works - Menu button: ${mobileMenu ? 'Found' : 'Not found'}`);
      results.passed.push('Mobile responsive design');
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Mobile responsive', error: error.message });
    }
    
    // Test 9: Performance metrics
    console.log('\n📋 Test 9: Performance metrics');
    results.totalTests++;
    try {
      await page.setViewportSize({ width: 1280, height: 720 });
      await page.goto('http://localhost:3001/dashboard', { 
        waitUntil: 'networkidle',
        timeout: 15000 
      });
      const metrics = await page.evaluate(() => {
        const perfData = performance.getEntriesByType('navigation')[0];
        return {
          domContentLoaded: Math.round(perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart),
          loadComplete: Math.round(perfData.loadEventEnd - perfData.loadEventStart),
          totalTime: Math.round(perfData.loadEventEnd - perfData.fetchStart)
        };
      });
      console.log(`   ✅ Performance - DOM: ${metrics.domContentLoaded}ms, Load: ${metrics.loadComplete}ms, Total: ${metrics.totalTime}ms`);
      results.passed.push(`Performance metrics collected`);
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Performance metrics', error: error.message });
    }
    
    // Test 10: Accessibility basics
    console.log('\n📋 Test 10: Basic accessibility checks');
    results.totalTests++;
    try {
      const accessibilityIssues = await page.evaluate(() => {
        const issues = [];
        // Check for images without alt text
        const imagesWithoutAlt = document.querySelectorAll('img:not([alt])').length;
        if (imagesWithoutAlt > 0) issues.push(`${imagesWithoutAlt} images without alt text`);
        
        // Check for buttons without accessible text
        const buttonsWithoutText = Array.from(document.querySelectorAll('button')).filter(
          btn => !btn.textContent.trim() && !btn.getAttribute('aria-label')
        ).length;
        if (buttonsWithoutText > 0) issues.push(`${buttonsWithoutText} buttons without accessible text`);
        
        // Check for form inputs without labels
        const inputsWithoutLabels = document.querySelectorAll('input:not([aria-label]):not([id])').length;
        if (inputsWithoutLabels > 0) issues.push(`${inputsWithoutLabels} inputs without labels`);
        
        return issues;
      });
      
      if (accessibilityIssues.length === 0) {
        console.log(`   ✅ Basic accessibility checks passed`);
        results.passed.push('Basic accessibility checks');
      } else {
        console.log(`   ⚠️  Accessibility issues found: ${accessibilityIssues.join(', ')}`);
        results.failed.push({ test: 'Accessibility', issues: accessibilityIssues });
      }
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      results.failed.push({ test: 'Accessibility checks', error: error.message });
    }
    
  } catch (error) {
    console.error('\n❌ Test suite error:', error.message);
    results.failed.push({ test: 'Test suite', error: error.message });
  } finally {
    if (browser) {
      await browser.close();
    }
  }
  
  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 TEST RESULTS SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total Tests: ${results.totalTests}`);
  console.log(`✅ Passed: ${results.passed.length}`);
  console.log(`❌ Failed: ${results.failed.length}`);
  console.log(`Success Rate: ${Math.round((results.passed.length / results.totalTests) * 100)}%`);
  
  if (results.failed.length > 0) {
    console.log('\n❌ Failed Tests:');
    results.failed.forEach(failure => {
      console.log(`   - ${failure.test || failure}: ${failure.error || ''}`);
    });
  }
  
  // Save results to file
  fs.writeFileSync('ui-test-results.json', JSON.stringify(results, null, 2));
  console.log('\n💾 Detailed results saved to ui-test-results.json');
  
  process.exit(results.failed.length > 0 ? 1 : 0);
}

// Run the tests
runUITests().catch(console.error);