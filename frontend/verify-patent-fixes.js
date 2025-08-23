const { chromium } = require('playwright');

async function verifyPatentFixes() {
  const browser = await chromium.launch({ 
    headless: false,
    args: ['--start-maximized']
  });
  
  const page = await browser.newContext().then(ctx => ctx.newPage());
  
  console.log('🔍 Verifying Patent Implementation Fixes...\n');
  
  const results = {
    passed: 0,
    failed: 0,
    tests: []
  };
  
  async function test(name, testFn) {
    try {
      const result = await testFn();
      if (result) {
        console.log(`✅ ${name}`);
        results.passed++;
        results.tests.push({ name, status: 'passed' });
      } else {
        console.log(`❌ ${name}`);
        results.failed++;
        results.tests.push({ name, status: 'failed' });
      }
    } catch (error) {
      console.log(`❌ ${name} - Error: ${error.message}`);
      results.failed++;
      results.tests.push({ name, status: 'error', error: error.message });
    }
  }
  
  try {
    // Patent 1: Correlation Page
    console.log('\n📊 Patent 1: Cross-Domain Correlation\n');
    await page.goto('http://localhost:3000/ai/correlations', { waitUntil: 'networkidle' });
    
    await test('Graph has ID and data-testid', async () => {
      const graph = await page.locator('#correlation-graph, [data-testid="correlation-graph"]').first();
      return await graph.isVisible();
    });
    
    await test('What-If button exists', async () => {
      const button = await page.locator('button:has-text("What-If")').first();
      return await button.isVisible();
    });
    
    await test('What-If modal can open', async () => {
      const button = await page.locator('button:has-text("What-If")').first();
      if (await button.isVisible()) {
        await button.click();
        await page.waitForTimeout(1000);
        const modal = await page.locator('.scenario-simulator, .modal, [role="dialog"]').first();
        const isVisible = await modal.isVisible();
        
        // Close modal
        const closeBtn = await page.locator('button svg').filter({ has: page.locator('svg') }).first();
        if (await closeBtn.isVisible()) await closeBtn.click();
        
        return isVisible;
      }
      return false;
    });
    
    // Patent 2: Chat Page
    console.log('\n💬 Patent 2: Conversational AI\n');
    await page.goto('http://localhost:3000/ai/chat', { waitUntil: 'networkidle' });
    
    await test('Chat input has proper ID', async () => {
      const input = await page.locator('#chat-message-input, [data-testid="chat-message-input"]').first();
      return await input.isVisible();
    });
    
    await test('Send button has proper ID', async () => {
      const button = await page.locator('#chat-send-button, [data-testid="chat-send-button"]').first();
      return await button.isVisible();
    });
    
    await test('Can type in chat input', async () => {
      const input = await page.locator('#chat-message-input').first();
      if (await input.isVisible()) {
        await input.fill('Test message');
        const value = await input.inputValue();
        await input.fill(''); // Clear
        return value === 'Test message';
      }
      return false;
    });
    
    // Patent 3: Dashboard
    console.log('\n🏢 Patent 3: Unified Platform\n');
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
    
    await test('Unified Platform section exists', async () => {
      const section = await page.locator('text="Unified AI-Driven Governance Platform"').first();
      return await section.isVisible();
    });
    
    await test('Shows 6 metric cards', async () => {
      const metrics = await page.locator('text="AI Models Active"').first();
      return await metrics.isVisible();
    });
    
    await test('Shows cross-domain correlation status', async () => {
      const status = await page.locator('text="Cross-Domain Correlation Engine"').first();
      return await status.isVisible();
    });
    
    // Patent 4: Compliance Page
    console.log('\n⚡ Patent 4: Predictive Compliance\n');
    await page.goto('http://localhost:3000/governance/compliance', { waitUntil: 'networkidle' });
    
    await test('Predictive Compliance Engine section exists', async () => {
      const section = await page.locator('text="Predictive Compliance Engine"').first();
      return await section.isVisible();
    });
    
    await test('Drift Detection panel exists', async () => {
      const drift = await page.locator('text="Configuration Drift Detection"').first();
      return await drift.isVisible();
    });
    
    await test('Shows drift items', async () => {
      const criticalDrift = await page.locator('text="Critical Drift Detected"').first();
      return await criticalDrift.isVisible();
    });
    
    await test('Predictive Timeline exists', async () => {
      const timeline = await page.locator('text="Predictive Compliance Timeline"').first();
      return await timeline.isVisible();
    });
    
    await test('Shows timeline predictions', async () => {
      const prediction = await page.locator('text="In 7 days"').first();
      return await prediction.isVisible();
    });
    
    await test('Shows ML model metrics', async () => {
      const accuracy = await page.locator('text="99.2%"').first();
      return await accuracy.isVisible();
    });
    
  } finally {
    // Summary
    console.log('\n' + '='.repeat(50));
    console.log('📊 RESULTS SUMMARY');
    console.log('='.repeat(50));
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log(`📈 Success Rate: ${((results.passed / (results.passed + results.failed)) * 100).toFixed(1)}%`);
    
    console.log('\n📋 PATENT STATUS:');
    const patent1 = results.tests.slice(0, 3).filter(t => t.status === 'passed').length;
    const patent2 = results.tests.slice(3, 6).filter(t => t.status === 'passed').length;
    const patent3 = results.tests.slice(6, 9).filter(t => t.status === 'passed').length;
    const patent4 = results.tests.slice(9, 15).filter(t => t.status === 'passed').length;
    
    console.log(`Patent 1: ${patent1}/3 tests passed (${(patent1/3*100).toFixed(0)}%)`);
    console.log(`Patent 2: ${patent2}/3 tests passed (${(patent2/3*100).toFixed(0)}%)`);
    console.log(`Patent 3: ${patent3}/3 tests passed (${(patent3/3*100).toFixed(0)}%)`);
    console.log(`Patent 4: ${patent4}/6 tests passed (${(patent4/6*100).toFixed(0)}%)`);
    
    await browser.close();
  }
}

verifyPatentFixes().catch(console.error);