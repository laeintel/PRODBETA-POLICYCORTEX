const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function testUIUX() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500 
  });
  
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    colorScheme: 'light'
  });
  
  const page = await context.newPage();
  
  console.log('🔍 Starting UI/UX Analysis...\n');
  
  const issues = [];
  const fixes = [];
  
  try {
    // Test 1: Home page load
    console.log('1. Testing home page...');
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Check for theme toggle
    const themeToggle = await page.locator('[aria-label="Toggle theme"]').count();
    if (themeToggle === 0) {
      issues.push('❌ Theme toggle button not found or missing aria-label');
      fixes.push('Add aria-label="Toggle theme" to theme toggle button');
    } else {
      console.log('✅ Theme toggle found');
      
      // Test theme switching
      await page.click('[aria-label="Toggle theme"]');
      await page.waitForTimeout(500);
      
      const isDark = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark');
      });
      
      console.log(`✅ Theme switched to: ${isDark ? 'dark' : 'light'}`);
      
      // Switch back
      await page.click('[aria-label="Toggle theme"]');
      await page.waitForTimeout(500);
    }
    
    // Test 2: Navigation
    console.log('\n2. Testing navigation...');
    const navItems = [
      { name: 'Dashboard', path: '/dashboard' },
      { name: 'AI', path: '/ai' },
      { name: 'Governance', path: '/governance' },
      { name: 'Security', path: '/security' },
      { name: 'DevOps', path: '/devops' },
      { name: 'Operations', path: '/operations' }
    ];
    
    for (const item of navItems) {
      try {
        console.log(`   Navigating to ${item.name}...`);
        await page.goto(`http://localhost:3000${item.path}`);
        await page.waitForLoadState('networkidle');
        
        // Check if page loads without errors
        const title = await page.title();
        console.log(`   ✅ ${item.name} loaded (title: ${title})`);
        
        // Check theme consistency
        const bgColor = await page.evaluate(() => {
          const body = document.body;
          return window.getComputedStyle(body).backgroundColor;
        });
        
        // Take screenshot for review
        await page.screenshot({ 
          path: `C:\\Users\\leona\\Screenshots\\playwright\\${item.name.toLowerCase()}.png` 
        });
        
      } catch (error) {
        issues.push(`❌ Error loading ${item.name}: ${error.message}`);
      }
    }
    
    // Test 3: Responsive design
    console.log('\n3. Testing responsive design...');
    const viewports = [
      { name: 'Mobile', width: 375, height: 667 },
      { name: 'Tablet', width: 768, height: 1024 },
      { name: 'Desktop', width: 1920, height: 1080 }
    ];
    
    await page.goto('http://localhost:3000/dashboard');
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);
      
      // Check if navigation is properly responsive
      const isMenuVisible = await page.locator('.mobile-menu-button').isVisible().catch(() => false);
      
      if (viewport.name === 'Mobile' && !isMenuVisible) {
        issues.push('❌ Mobile menu button not visible on small screens');
        fixes.push('Add responsive mobile menu button for small screens');
      }
      
      console.log(`   ✅ ${viewport.name} viewport tested (${viewport.width}x${viewport.height})`);
    }
    
    // Test 4: Patent features visibility
    console.log('\n4. Testing patent features UI...');
    await page.goto('http://localhost:3000/ai');
    await page.setViewportSize({ width: 1280, height: 720 });
    
    const patentFeatures = [
      { selector: 'text=/correlation/i', name: 'Cross-Domain Correlation' },
      { selector: 'text=/conversation/i', name: 'Conversational AI' },
      { selector: 'text=/predictive/i', name: 'Predictive Compliance' },
      { selector: 'text=/unified/i', name: 'Unified Platform' }
    ];
    
    for (const feature of patentFeatures) {
      const found = await page.locator(feature.selector).count() > 0;
      if (!found) {
        issues.push(`⚠️ Patent feature "${feature.name}" not prominently displayed`);
        fixes.push(`Add clear UI element for ${feature.name} feature`);
      } else {
        console.log(`   ✅ ${feature.name} feature found in UI`);
      }
    }
    
    // Test 5: Accessibility
    console.log('\n5. Testing accessibility...');
    await page.goto('http://localhost:3000/dashboard');
    
    // Check for proper heading hierarchy
    const headings = await page.evaluate(() => {
      const h1Count = document.querySelectorAll('h1').length;
      const h2Count = document.querySelectorAll('h2').length;
      return { h1Count, h2Count };
    });
    
    if (headings.h1Count === 0) {
      issues.push('❌ No H1 heading found on dashboard');
      fixes.push('Add proper H1 heading to dashboard page');
    }
    
    // Check for alt text on images
    const imagesWithoutAlt = await page.evaluate(() => {
      const images = Array.from(document.querySelectorAll('img'));
      return images.filter(img => !img.alt).length;
    });
    
    if (imagesWithoutAlt > 0) {
      issues.push(`⚠️ ${imagesWithoutAlt} images without alt text`);
      fixes.push('Add alt text to all images for accessibility');
    }
    
    // Test 6: Performance
    console.log('\n6. Testing performance...');
    const startTime = Date.now();
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
    const loadTime = Date.now() - startTime;
    
    if (loadTime > 3000) {
      issues.push(`⚠️ Dashboard load time is ${loadTime}ms (should be < 3000ms)`);
      fixes.push('Optimize dashboard loading performance');
    } else {
      console.log(`   ✅ Dashboard loads in ${loadTime}ms`);
    }
    
  } catch (error) {
    console.error('Error during testing:', error);
  }
  
  // Generate report
  console.log('\n' + '='.repeat(50));
  console.log('📊 UI/UX TEST REPORT');
  console.log('='.repeat(50));
  
  if (issues.length === 0) {
    console.log('\n🎉 No critical issues found!');
  } else {
    console.log('\n🔴 Issues Found:');
    issues.forEach(issue => console.log(`   ${issue}`));
    
    console.log('\n🔧 Recommended Fixes:');
    fixes.forEach(fix => console.log(`   • ${fix}`));
  }
  
  // Save report
  const report = {
    timestamp: new Date().toISOString(),
    issues,
    fixes,
    summary: {
      totalIssues: issues.length,
      critical: issues.filter(i => i.includes('❌')).length,
      warnings: issues.filter(i => i.includes('⚠️')).length
    }
  };
  
  fs.writeFileSync(
    path.join(__dirname, 'ui-ux-report.json'),
    JSON.stringify(report, null, 2)
  );
  
  console.log('\n📁 Report saved to scripts/ui-ux-report.json');
  
  await browser.close();
  
  return report;
}

// Run the test
testUIUX().then(report => {
  if (report.summary.critical > 0) {
    console.log('\n⚠️ Critical issues need immediate attention!');
    process.exit(1);
  }
  process.exit(0);
}).catch(error => {
  console.error('Test failed:', error);
  process.exit(1);
});