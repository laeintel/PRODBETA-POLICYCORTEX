const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

async function browseAndAnalyze() {
  console.log('🚀 Launching Playwright to actually browse the application...\n');
  
  const browser = await chromium.launch({ 
    headless: false,  // Show the browser so we can see it
    slowMo: 1000      // Slow down actions so they're visible
  });
  
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: {
      dir: 'C:\\Users\\leona\\Videos\\playwright-recordings',
      size: { width: 1280, height: 720 }
    }
  });
  
  const page = await context.newPage();
  
  try {
    // Actually navigate and see the app
    console.log('📱 Opening PolicyCortex application...');
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Take a screenshot of the home page
    await page.screenshot({ 
      path: 'C:\\Users\\leona\\Screenshots\\playwright\\home-actual.png',
      fullPage: true 
    });
    console.log('📸 Screenshot taken: home-actual.png');
    
    // Check what's actually visible
    const visibleElements = await page.evaluate(() => {
      const elements = {
        hasThemeToggle: !!document.querySelector('[aria-label*="theme"]'),
        hasNavigation: !!document.querySelector('nav'),
        backgroundColor: window.getComputedStyle(document.body).backgroundColor,
        textColor: window.getComputedStyle(document.body).color,
        h1Text: document.querySelector('h1')?.textContent || 'No H1 found',
        buttonCount: document.querySelectorAll('button').length,
        linkCount: document.querySelectorAll('a').length
      };
      return elements;
    });
    
    console.log('\n📊 What I actually see:');
    console.log('   Theme toggle exists:', visibleElements.hasThemeToggle);
    console.log('   Navigation exists:', visibleElements.hasNavigation);
    console.log('   Background color:', visibleElements.backgroundColor);
    console.log('   Text color:', visibleElements.textColor);
    console.log('   Main heading:', visibleElements.h1Text);
    console.log('   Number of buttons:', visibleElements.buttonCount);
    console.log('   Number of links:', visibleElements.linkCount);
    
    // Try to click on theme toggle if it exists
    const themeToggle = page.locator('button').filter({ hasText: /theme|dark|light/i }).first();
    const toggleExists = await themeToggle.count() > 0;
    
    if (toggleExists) {
      console.log('\n🌓 Found theme toggle, clicking it...');
      await themeToggle.click();
      await page.waitForTimeout(1000);
      
      // Check if theme actually changed
      const newBgColor = await page.evaluate(() => 
        window.getComputedStyle(document.body).backgroundColor
      );
      console.log('   New background color:', newBgColor);
      
      await page.screenshot({ 
        path: 'C:\\Users\\leona\\Screenshots\\playwright\\theme-switched.png',
        fullPage: true 
      });
      console.log('📸 Screenshot taken: theme-switched.png');
    } else {
      console.log('\n❌ No theme toggle found!');
    }
    
    // Navigate to Dashboard
    console.log('\n📍 Navigating to Dashboard...');
    await page.goto('http://localhost:3000/dashboard');
    await page.waitForLoadState('networkidle');
    
    await page.screenshot({ 
      path: 'C:\\Users\\leona\\Screenshots\\playwright\\dashboard-actual.png',
      fullPage: true 
    });
    console.log('📸 Screenshot taken: dashboard-actual.png');
    
    // Check dashboard content
    const dashboardContent = await page.evaluate(() => {
      return {
        title: document.querySelector('h1')?.textContent || document.querySelector('h2')?.textContent || 'No title',
        hasCharts: document.querySelectorAll('svg').length,
        hasCards: document.querySelectorAll('[class*="card"], [class*="Card"]').length,
        hasMetrics: document.body.textContent?.includes('metric') || document.body.textContent?.includes('KPI')
      };
    });
    
    console.log('\n📊 Dashboard Analysis:');
    console.log('   Title:', dashboardContent.title);
    console.log('   Number of charts:', dashboardContent.hasCharts);
    console.log('   Number of cards:', dashboardContent.hasCards);
    console.log('   Has metrics:', dashboardContent.hasMetrics);
    
    // Navigate to AI page to check patent features
    console.log('\n🤖 Navigating to AI page...');
    await page.goto('http://localhost:3000/ai');
    await page.waitForLoadState('networkidle');
    
    await page.screenshot({ 
      path: 'C:\\Users\\leona\\Screenshots\\playwright\\ai-page-actual.png',
      fullPage: true 
    });
    console.log('📸 Screenshot taken: ai-page-actual.png');
    
    // Check for patent features
    const patentFeatures = await page.evaluate(() => {
      const pageText = document.body.textContent || '';
      return {
        hasPatent1: pageText.includes('Cross-Domain') || pageText.includes('Correlation'),
        hasPatent2: pageText.includes('Conversational'),
        hasPatent3: pageText.includes('Unified'),
        hasPatent4: pageText.includes('Predictive'),
        hasPatentBadges: document.querySelectorAll('[class*="Patent"], [class*="patent"]').length
      };
    });
    
    console.log('\n🏆 Patent Features Visibility:');
    console.log('   Patent #1 (Cross-Domain):', patentFeatures.hasPatent1);
    console.log('   Patent #2 (Conversational):', patentFeatures.hasPatent2);
    console.log('   Patent #3 (Unified):', patentFeatures.hasPatent3);
    console.log('   Patent #4 (Predictive):', patentFeatures.hasPatent4);
    console.log('   Patent badges found:', patentFeatures.hasPatentBadges);
    
    // Test responsive design
    console.log('\n📱 Testing responsive design...');
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    await page.screenshot({ 
      path: 'C:\\Users\\leona\\Screenshots\\playwright\\mobile-view.png',
      fullPage: false 
    });
    console.log('📸 Screenshot taken: mobile-view.png');
    
    const mobileCheck = await page.evaluate(() => {
      return {
        hasHamburgerMenu: !!document.querySelector('[class*="menu"], [class*="Menu"], [aria-label*="menu"]'),
        isNavigationVisible: !!document.querySelector('nav') && window.getComputedStyle(document.querySelector('nav')).display !== 'none'
      };
    });
    
    console.log('\n📱 Mobile View Analysis:');
    console.log('   Has hamburger menu:', mobileCheck.hasHamburgerMenu);
    console.log('   Navigation visible:', mobileCheck.isNavigationVisible);
    
  } catch (error) {
    console.error('❌ Error during browsing:', error.message);
  } finally {
    // Close the video and browser
    await context.close();
    await browser.close();
    
    console.log('\n' + '='.repeat(50));
    console.log('✅ ACTUAL BROWSING COMPLETE');
    console.log('='.repeat(50));
    console.log('\n📁 Screenshots saved to: C:\\Users\\leona\\Screenshots\\playwright\\');
    console.log('📹 Video saved to: C:\\Users\\leona\\Videos\\playwright-recordings\\');
    console.log('\nYou can now view the actual screenshots to see what your app really looks like!');
  }
}

// Run the actual browsing
browseAndAnalyze().catch(console.error);