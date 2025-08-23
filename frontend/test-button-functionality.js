const { chromium } = require('playwright');

async function testButtonFunctionality() {
  console.log('🚀 Starting Button Functionality Test...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  let passedTests = 0;
  let failedTests = 0;
  const issues = [];

  try {
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');

    console.log('\n📌 Testing Navigation Buttons...');
    
    // Test main navigation buttons
    const navButtons = [
      { selector: 'text=Dashboard', expectedUrl: '/tactical' },
      { selector: 'text=Governance', expectedUrl: '/governance' },
      { selector: 'text=Security & Access', expectedUrl: '/security' },
      { selector: 'text=Operations', expectedUrl: '/operations' },
      { selector: 'text=DevOps & CI/CD', expectedUrl: '/devops' },
      { selector: 'text=AI Intelligence', expectedUrl: '/ai' },
    ];

    for (const nav of navButtons) {
      try {
        await page.click(nav.selector);
        await page.waitForTimeout(500);
        const currentUrl = page.url();
        if (currentUrl.includes(nav.expectedUrl)) {
          console.log(`✅ ${nav.selector} navigation works`);
          passedTests++;
        } else {
          console.log(`❌ ${nav.selector} navigation failed - Expected: ${nav.expectedUrl}, Got: ${currentUrl}`);
          issues.push(`Navigation to ${nav.expectedUrl} not working`);
          failedTests++;
        }
      } catch (e) {
        console.log(`❌ Could not click ${nav.selector}: ${e.message}`);
        issues.push(`Button ${nav.selector} not clickable`);
        failedTests++;
      }
    }

    console.log('\n📌 Testing Quick Action Buttons...');
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');

    // Test Quick Actions bar buttons
    const quickActions = [
      { selector: 'text=Compliance Status', desc: 'Compliance Quick Action' },
      { selector: 'text=Cost Savings', desc: 'Cost Savings Quick Action' },
      { selector: 'text=Chat with AI', desc: 'AI Chat Quick Action' },
      { selector: 'text=View Predictions', desc: 'Predictions Quick Action' },
      { selector: 'text=Active Risks', desc: 'Risks Quick Action' },
      { selector: 'text=View Resources', desc: 'Resources Quick Action' },
    ];

    for (const action of quickActions) {
      try {
        const button = await page.locator(action.selector).first();
        if (await button.isVisible()) {
          await button.click();
          await page.waitForTimeout(500);
          console.log(`✅ ${action.desc} is clickable`);
          passedTests++;
        } else {
          console.log(`❌ ${action.desc} not visible`);
          issues.push(`${action.desc} not visible`);
          failedTests++;
        }
      } catch (e) {
        console.log(`❌ ${action.desc} failed: ${e.message}`);
        issues.push(`${action.desc} not working`);
        failedTests++;
      }
    }

    console.log('\n📌 Testing Dashboard Action Buttons...');
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');

    // Test dashboard quick action buttons
    const dashboardButtons = [
      { selector: 'button:has-text("AI Assistant")', desc: 'AI Assistant button' },
      { selector: 'button:has-text("View Policies")', desc: 'View Policies button' },
      { selector: 'button:has-text("Optimize Costs")', desc: 'Optimize Costs button' },
      { selector: 'button:has-text("Check Alerts")', desc: 'Check Alerts button' },
    ];

    for (const btn of dashboardButtons) {
      try {
        const button = await page.locator(btn.selector).first();
        if (await button.isVisible()) {
          await button.click();
          await page.waitForTimeout(500);
          console.log(`✅ ${btn.desc} is clickable`);
          passedTests++;
          await page.goto('http://localhost:3000/dashboard');
        } else {
          console.log(`❌ ${btn.desc} not found`);
          issues.push(`${btn.desc} not found`);
          failedTests++;
        }
      } catch (e) {
        console.log(`❌ ${btn.desc} failed: ${e.message}`);
        issues.push(`${btn.desc} not working`);
        failedTests++;
      }
    }

    console.log('\n📌 Testing Cloud Integration Buttons...');
    
    // Test cloud integration buttons
    const cloudButtons = await page.locator('button:has-text("Connect"), button:has-text("Refresh"), button:has-text("Manage")').all();
    for (let i = 0; i < cloudButtons.length; i++) {
      try {
        await cloudButtons[i].click();
        console.log(`✅ Cloud integration button ${i + 1} is clickable`);
        passedTests++;
        await page.waitForTimeout(500);
      } catch (e) {
        console.log(`❌ Cloud integration button ${i + 1} failed`);
        issues.push(`Cloud integration button ${i + 1} not working`);
        failedTests++;
      }
    }

    console.log('\n📌 Testing Theme Toggle...');
    
    // Test theme toggle
    try {
      const themeToggle = await page.locator('[aria-label="Toggle theme"]').first();
      if (await themeToggle.isVisible()) {
        const initialBg = await page.evaluate(() => {
          return window.getComputedStyle(document.body).backgroundColor;
        });
        
        await themeToggle.click();
        await page.waitForTimeout(500);
        
        const newBg = await page.evaluate(() => {
          return window.getComputedStyle(document.body).backgroundColor;
        });
        
        if (initialBg !== newBg) {
          console.log('✅ Theme toggle works');
          passedTests++;
        } else {
          console.log('❌ Theme toggle not changing colors');
          issues.push('Theme toggle not working');
          failedTests++;
        }
      } else {
        console.log('❌ Theme toggle not found');
        issues.push('Theme toggle not visible');
        failedTests++;
      }
    } catch (e) {
      console.log(`❌ Theme toggle test failed: ${e.message}`);
      issues.push('Theme toggle error');
      failedTests++;
    }

    console.log('\n📌 Testing DevOps Page...');
    
    // Test the new tactical/devops page
    await page.goto('http://localhost:3000/tactical/devops');
    await page.waitForLoadState('networkidle');
    
    const devOpsButtons = [
      { selector: 'button:has-text("Deploy to Production")', desc: 'Deploy button' },
      { selector: 'button:has-text("Pipeline Settings")', desc: 'Settings button' },
    ];

    for (const btn of devOpsButtons) {
      try {
        const button = await page.locator(btn.selector).first();
        if (await button.isVisible()) {
          await button.click();
          console.log(`✅ ${btn.desc} on DevOps page is clickable`);
          passedTests++;
        } else {
          console.log(`❌ ${btn.desc} on DevOps page not found`);
          issues.push(`${btn.desc} on DevOps page not found`);
          failedTests++;
        }
      } catch (e) {
        console.log(`❌ ${btn.desc} on DevOps page failed: ${e.message}`);
        issues.push(`${btn.desc} on DevOps page not working`);
        failedTests++;
      }
    }

  } catch (error) {
    console.error('Test execution error:', error);
    issues.push(`Test execution error: ${error.message}`);
  } finally {
    console.log('\n' + '='.repeat(60));
    console.log('📊 TEST RESULTS SUMMARY');
    console.log('='.repeat(60));
    console.log(`✅ Passed: ${passedTests}`);
    console.log(`❌ Failed: ${failedTests}`);
    console.log(`📈 Success Rate: ${((passedTests / (passedTests + failedTests)) * 100).toFixed(1)}%`);
    
    if (issues.length > 0) {
      console.log('\n⚠️  Issues Found:');
      issues.forEach((issue, idx) => {
        console.log(`${idx + 1}. ${issue}`);
      });
    }

    await browser.close();
  }
}

testButtonFunctionality().catch(console.error);