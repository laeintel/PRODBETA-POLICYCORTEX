const { chromium } = require('playwright');

async function testVisualThemeCheck() {
  console.log('🎨 Visual Theme Verification Test\n');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  try {
    // Test in light mode
    console.log('Testing Light Mode Appearance:');
    console.log('=' .repeat(50));
    
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Ensure light mode
    const themeToggle = await page.locator('[aria-label="Toggle theme"]').first();
    const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
    if (isDark && await themeToggle.isVisible()) {
      await themeToggle.click();
      await page.waitForTimeout(500);
    }

    // Check operations page visual elements
    await page.goto('http://localhost:3000/operations');
    await page.waitForLoadState('networkidle');
    
    const lightModeCheck = await page.evaluate(() => {
      const results = {};
      
      // Check main container
      const mainDiv = document.querySelector('div.min-h-screen');
      if (mainDiv) {
        const styles = window.getComputedStyle(mainDiv);
        results.mainBg = styles.backgroundColor;
        results.mainColor = styles.color;
        results.mainClasses = mainDiv.className;
      }
      
      // Check header
      const header = document.querySelector('div.border-b');
      if (header) {
        const styles = window.getComputedStyle(header);
        results.headerBg = styles.backgroundColor;
        results.headerBorder = styles.borderColor;
      }
      
      // Check tabs
      const tabs = document.querySelector('div.flex.gap-6');
      if (tabs) {
        const tabButton = tabs.querySelector('button');
        if (tabButton) {
          const styles = window.getComputedStyle(tabButton);
          results.tabColor = styles.color;
        }
      }
      
      // Check any cards
      const card = document.querySelector('.bg-white, .dark\\:bg-gray-800');
      if (card) {
        const styles = window.getComputedStyle(card);
        results.cardBg = styles.backgroundColor;
      }
      
      return results;
    });
    
    console.log('\nLight Mode - Operations Page:');
    console.log('Main Background:', lightModeCheck.mainBg);
    console.log('Main Text Color:', lightModeCheck.mainColor);
    console.log('Header Background:', lightModeCheck.headerBg);
    console.log('Tab Text Color:', lightModeCheck.tabColor);
    console.log('Card Background:', lightModeCheck.cardBg);
    
    // Switch to dark mode
    console.log('\n' + '=' .repeat(50));
    console.log('Testing Dark Mode Appearance:');
    console.log('=' .repeat(50));
    
    await page.goto('http://localhost:3000/dashboard');
    await themeToggle.click();
    await page.waitForTimeout(500);
    
    await page.goto('http://localhost:3000/operations');
    await page.waitForLoadState('networkidle');
    
    const darkModeCheck = await page.evaluate(() => {
      const results = {};
      
      // Check main container
      const mainDiv = document.querySelector('div.min-h-screen');
      if (mainDiv) {
        const styles = window.getComputedStyle(mainDiv);
        results.mainBg = styles.backgroundColor;
        results.mainColor = styles.color;
        results.mainClasses = mainDiv.className;
      }
      
      // Check if dark mode is applied
      results.htmlHasDark = document.documentElement.classList.contains('dark');
      
      // Check computed styles
      const body = document.body;
      const bodyStyles = window.getComputedStyle(body);
      results.bodyBg = bodyStyles.backgroundColor;
      results.bodyColor = bodyStyles.color;
      
      // Find any elements that might be staying light
      const allElements = document.querySelectorAll('*');
      const lightElements = [];
      
      allElements.forEach(el => {
        const bg = window.getComputedStyle(el).backgroundColor;
        // Check for white or very light backgrounds
        if (bg === 'rgb(255, 255, 255)' || bg === 'rgb(249, 250, 251)' || bg === 'rgb(243, 244, 246)') {
          const classes = el.className || '';
          if (!classes.includes('dark:') && el.offsetWidth > 100 && el.offsetHeight > 50) {
            lightElements.push({
              tag: el.tagName,
              classes: classes.substring(0, 50),
              bg: bg
            });
          }
        }
      });
      
      results.lightElements = lightElements.slice(0, 5); // First 5 problematic elements
      
      return results;
    });
    
    console.log('\nDark Mode - Operations Page:');
    console.log('HTML has dark class:', darkModeCheck.htmlHasDark);
    console.log('Body Background:', darkModeCheck.bodyBg);
    console.log('Body Text Color:', darkModeCheck.bodyColor);
    console.log('Main Background:', darkModeCheck.mainBg);
    console.log('Main Text Color:', darkModeCheck.mainColor);
    
    if (darkModeCheck.lightElements && darkModeCheck.lightElements.length > 0) {
      console.log('\n⚠️  Elements still showing light backgrounds in dark mode:');
      darkModeCheck.lightElements.forEach(el => {
        console.log(`  - ${el.tag}: ${el.classes} (${el.bg})`);
      });
    }
    
    // Test Settings page specifically
    console.log('\n' + '=' .repeat(50));
    console.log('Testing Settings Page:');
    console.log('=' .repeat(50));
    
    await page.goto('http://localhost:3000/settings');
    await page.waitForLoadState('networkidle');
    
    const settingsCheck = await page.evaluate(() => {
      const mainDiv = document.querySelector('div.min-h-screen');
      const body = document.body;
      
      return {
        isDark: document.documentElement.classList.contains('dark'),
        bodyBg: window.getComputedStyle(body).backgroundColor,
        mainBg: mainDiv ? window.getComputedStyle(mainDiv).backgroundColor : null,
        mainClasses: mainDiv ? mainDiv.className : null
      };
    });
    
    console.log('Dark mode active:', settingsCheck.isDark);
    console.log('Body Background:', settingsCheck.bodyBg);
    console.log('Main Background:', settingsCheck.mainBg);
    console.log('Main Classes:', settingsCheck.mainClasses);
    
    // Visual confirmation
    console.log('\n📸 Taking screenshots for visual confirmation...');
    
    // Light mode screenshots
    await page.goto('http://localhost:3000/dashboard');
    const toggle = await page.locator('[aria-label="Toggle theme"]').first();
    const currentlyDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
    if (currentlyDark) {
      await toggle.click();
      await page.waitForTimeout(500);
    }
    
    await page.goto('http://localhost:3000/operations');
    await page.screenshot({ path: 'operations-light-mode.png', fullPage: false });
    
    await page.goto('http://localhost:3000/settings');
    await page.screenshot({ path: 'settings-light-mode.png', fullPage: false });
    
    // Dark mode screenshots
    await page.goto('http://localhost:3000/dashboard');
    await toggle.click();
    await page.waitForTimeout(500);
    
    await page.goto('http://localhost:3000/operations');
    await page.screenshot({ path: 'operations-dark-mode.png', fullPage: false });
    
    await page.goto('http://localhost:3000/settings');
    await page.screenshot({ path: 'settings-dark-mode.png', fullPage: false });
    
    console.log('✅ Screenshots saved for manual inspection');

  } catch (error) {
    console.error('Error during testing:', error);
  } finally {
    await browser.close();
  }
}

testVisualThemeCheck().catch(console.error);