const { chromium } = require('playwright');

async function testOperationsTheme() {
  console.log('🔍 Testing Operations Pages Theme Issue...\n');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  try {
    // First, navigate to dashboard and set to light theme
    console.log('1. Setting up light theme from dashboard...');
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Check if we can find theme toggle and ensure light mode
    const themeToggle = await page.locator('[aria-label="Toggle theme"]').first();
    if (await themeToggle.isVisible()) {
      // Ensure we're in light mode
      const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
      if (isDark) {
        await themeToggle.click();
        await page.waitForTimeout(500);
        console.log('   ✓ Switched to light theme');
      } else {
        console.log('   ✓ Already in light theme');
      }
    }

    // Test all operations pages
    const operationsPages = [
      '/operations',
      '/operations/resources', 
      '/operations/monitoring',
      '/operations/automation',
      '/operations/notifications',
      '/operations/alerts'
    ];

    console.log('\n2. Testing Operations pages in light theme:');
    for (const path of operationsPages) {
      await page.goto(`http://localhost:3000${path}`);
      await page.waitForLoadState('networkidle');
      
      // Take screenshot for visual inspection
      await page.screenshot({ 
        path: `operations-light-${path.replace(/\//g, '-')}.png`,
        fullPage: false 
      });
      
      // Check actual background colors
      const bgColors = await page.evaluate(() => {
        const body = document.querySelector('body');
        const main = document.querySelector('main') || document.querySelector('div');
        return {
          bodyBg: window.getComputedStyle(body).backgroundColor,
          mainBg: main ? window.getComputedStyle(main).backgroundColor : null,
          isDark: document.documentElement.classList.contains('dark'),
          bodyClasses: body.className,
          htmlClasses: document.documentElement.className
        };
      });
      
      console.log(`\n   ${path}:`);
      console.log(`   - HTML classes: ${bgColors.htmlClasses}`);
      console.log(`   - Body BG: ${bgColors.bodyBg}`);
      console.log(`   - Main BG: ${bgColors.mainBg}`);
      console.log(`   - Dark mode: ${bgColors.isDark}`);
      
      // Check if it's incorrectly dark
      if (bgColors.bodyBg === 'rgb(0, 0, 0)' || bgColors.bodyBg === 'rgb(17, 24, 39)') {
        console.log(`   ❌ Page showing dark theme in light mode!`);
      } else {
        console.log(`   ✓ Theme appears correct`);
      }
    }

    // Now test with dark theme
    console.log('\n3. Switching to dark theme...');
    await page.goto('http://localhost:3000/dashboard');
    const toggle = await page.locator('[aria-label="Toggle theme"]').first();
    await toggle.click();
    await page.waitForTimeout(500);

    console.log('\n4. Testing Operations pages in dark theme:');
    for (const path of operationsPages) {
      await page.goto(`http://localhost:3000${path}`);
      await page.waitForLoadState('networkidle');
      
      const bgColors = await page.evaluate(() => {
        const body = document.querySelector('body');
        return {
          bodyBg: window.getComputedStyle(body).backgroundColor,
          isDark: document.documentElement.classList.contains('dark')
        };
      });
      
      console.log(`   ${path}: BG=${bgColors.bodyBg}, Dark=${bgColors.isDark}`);
    }

    // Test settings page
    console.log('\n5. Testing Settings page:');
    await page.goto('http://localhost:3000/settings');
    await page.waitForLoadState('networkidle');
    
    const settingsBg = await page.evaluate(() => {
      const body = document.querySelector('body');
      return {
        bodyBg: window.getComputedStyle(body).backgroundColor,
        isDark: document.documentElement.classList.contains('dark')
      };
    });
    
    console.log(`   Settings: BG=${settingsBg.bodyBg}, Dark=${settingsBg.isDark}`);
    
    // Check for any hardcoded dark backgrounds
    console.log('\n6. Checking for hardcoded dark backgrounds...');
    const elementsWithDarkBg = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const darkElements = [];
      
      elements.forEach(el => {
        const classes = el.className;
        if (typeof classes === 'string') {
          if (classes.includes('bg-black') || 
              classes.includes('bg-gray-950') ||
              classes.includes('from-black') || 
              classes.includes('to-black')) {
            darkElements.push({
              tag: el.tagName,
              classes: classes.substring(0, 100),
              id: el.id
            });
          }
        }
      });
      
      return darkElements;
    });
    
    if (elementsWithDarkBg.length > 0) {
      console.log('   Found elements with hardcoded dark backgrounds:');
      elementsWithDarkBg.forEach(el => {
        console.log(`   - ${el.tag}${el.id ? '#' + el.id : ''}: ${el.classes}`);
      });
    }

  } catch (error) {
    console.error('Error during testing:', error);
  } finally {
    console.log('\n✅ Test complete. Check screenshots for visual confirmation.');
    await browser.close();
  }
}

testOperationsTheme().catch(console.error);