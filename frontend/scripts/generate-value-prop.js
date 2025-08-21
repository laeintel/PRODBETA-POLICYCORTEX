/*
  Generates DOCX and PDF for the value proposition document.
  Outputs to: ../docs/value-proposition.docx and ../docs/value-proposition.pdf
*/
const fs = require('fs');
const path = require('path');

async function generate() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const docsDir = path.join(repoRoot, 'docs');
  const outDocx = path.join(docsDir, 'value-proposition.docx');
  const outPdf = path.join(docsDir, 'value-proposition.pdf');

  fs.mkdirSync(docsDir, { recursive: true });

  const html = renderHtml();

  // Create DOCX from HTML (handle Buffer/Blob/Promise returns across versions)
  const HTMLDocx = require('html-docx-js-typescript');
  let docxOut = HTMLDocx.asBlob(html);
  if (docxOut && typeof docxOut.then === 'function') {
    docxOut = await docxOut;
  }
  let docxBuffer;
  if (docxOut && typeof docxOut.arrayBuffer === 'function') {
    const ab = await docxOut.arrayBuffer();
    docxBuffer = Buffer.from(ab);
  } else if (Buffer.isBuffer(docxOut)) {
    docxBuffer = docxOut;
  } else if (docxOut instanceof Uint8Array) {
    docxBuffer = Buffer.from(docxOut);
  } else if (docxOut && docxOut.buffer) {
    docxBuffer = Buffer.from(docxOut.buffer);
  } else {
    docxBuffer = Buffer.from(String(docxOut || ''));
  }
  fs.writeFileSync(outDocx, docxBuffer);

  // Create PDF from HTML using Puppeteer
  const puppeteer = require('puppeteer');
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.pdf({
    path: outPdf,
    format: 'A4',
    printBackground: true,
    margin: { top: '20mm', right: '18mm', bottom: '20mm', left: '18mm' },
  });
  await browser.close();

  console.log('Generated:', outDocx);
  console.log('Generated:', outPdf);
}

function renderHtml() {
  const css = `
    body { font-family: -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; color: #0f172a; line-height: 1.55; }
    h1 { font-size: 28px; margin: 0 0 12px; color: #0b1220; }
    h2 { font-size: 20px; margin: 22px 0 8px; color: #0b1220; }
    h3 { font-size: 16px; margin: 18px 0 6px; color: #0b1220; }
    p { margin: 10px 0; }
    ul { margin: 8px 0 12px 22px; }
    li { margin: 6px 0; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }
    .muted { color: #475569; }
    .tagline { font-style: italic; color: #0b1220; background: #f8fafc; padding: 10px 12px; border-left: 3px solid #0ea5e9; border-radius: 4px; }
    .small { font-size: 12px; color: #64748b; }
  `;

  const body = `
    <h1>PolicyCortex — Value Proposition</h1>
    <p class="small muted">Generated automatically from current understanding of the application’s capabilities.</p>

    <h2>Core value proposition</h2>
    <p>
      A single, AI-driven control plane that unifies cloud cost, security access (RBAC/IAM), compliance, and operations into one actionable workflow layer.
      It doesn’t just visualize risk and spend; it correlates, prioritizes, and turns findings into guided fixes (JIT, SoD, de-provision, policy remediations) with measurable savings and risk reduction.
    </p>

    <h2>Why companies care</h2>
    <ul>
      <li><b>Cost:</b> Cut cloud spend 8–15% via anomaly detection and optimization guidance.</li>
      <li><b>Risk:</b> Reduce breach and audit exposure with least-privilege, SoD checks, and continuous controls.</li>
      <li><b>Speed:</b> Shorter MTTR and faster approvals through AI triage, correlations, and predictive insights.</li>
      <li><b>Audit/compliance:</b> Fewer hours and lower fees with drill-downs, evidence capture, and report-ready posture.</li>
    </ul>

    <h2>What makes it different</h2>
    <ul>
      <li><b>AI-native remediation (not just dashboards):</b> Predictive and correlation views drive “what to fix next” and “why” (<code>app/ai/predictive/page.tsx</code>, <code>app/ai/correlations/page.tsx</code>); in-product assistant for triage/actions (<code>app/ai/chat/page.tsx</code>).</li>
      <li><b>End-to-end, not siloed:</b> Cost anomalies, compliance violations, IAM/RBAC over-entitlements, and ops health live in one place:
        <ul>
          <li>Cost: <code>components/cost/CostAnomalyDeepDrill.tsx</code></li>
          <li>Compliance: <code>components/compliance/ComplianceDeepDrill.tsx</code></li>
          <li>Access governance: <code>components/rbac/DeepDrillDashboard.tsx</code>, <code>app/security/rbac/page.tsx</code>, <code>app/security/iam/page.tsx</code></li>
          <li>Ops/monitoring: <code>app/operations/monitoring/page.tsx</code>, <code>app/operations/resources/page.tsx</code></li>
        </ul>
      </li>
      <li><b>Actionability built in:</b> JIT access, SoD conflict workflows, and permission removal flows are first-class UI actions, not FYIs.</li>
      <li><b>Production readiness focus:</b> Guardrails and reliability scaffolding (e2e click sweeps, button verifiers, scripted audits) reduce UI regressions (<code>scripts/verify-clickables.js</code>, <code>frontend/tests/clicks.spec.ts</code>). Security-first headers/CSP/middleware patterns.</li>
    </ul>

    <h2>Who buys and why (primary ICPs)</h2>
    <ul>
      <li><b>FinOps + Platform Engineering:</b> unify spend visibility with fix-paths.</li>
      <li><b>Security/IAM:</b> least-privilege at scale, SoD and JIT automation.</li>
      <li><b>Compliance/GRC:</b> continuous evidence and faster audits.</li>
      <li><b>SRE/Operations:</b> faster incident triage and remediation guidance.</li>
    </ul>

    <h2>Competitive angle</h2>
    <ul>
      <li><b>Versus point tools:</b> correlates across cost, access, compliance, and ops to compress handoffs (find → decide → fix).</li>
      <li><b>Versus generic BI:</b> ships domain-specific actions and guardrails, not just charts.</li>
      <li><b>Versus “AI overlays”:</b> AI drives grounded, workflow-safe actions tied to real RBAC/compliance objects, not chat summaries.</li>
    </ul>

    <h2>Proof in code (feature → module map)</h2>
    <ul>
      <li>Cost anomalies deep-drill: <code>components/cost/CostAnomalyDeepDrill.tsx</code></li>
      <li>Compliance deep-drill and evidence: <code>components/compliance/ComplianceDeepDrill.tsx</code></li>
      <li>RBAC/IAM deep-drill, JIT/SoD actions: <code>components/rbac/DeepDrillDashboard.tsx</code>, <code>app/security/rbac/page.tsx</code>, <code>app/security/iam/page.tsx</code></li>
      <li>Predictive/correlation analytics: <code>app/ai/predictive/page.tsx</code>, <code>app/ai/correlations/page.tsx</code></li>
      <li>AI assistant for triage: <code>app/ai/chat/page.tsx</code></li>
      <li>Ops monitoring/resources: <code>app/operations/monitoring/page.tsx</code>, <code>app/operations/resources/page.tsx</code></li>
      <li>Reliability/security scaffolding: <code>scripts/verify-clickables.js</code>, <code>frontend/tests/clicks.spec.ts</code></li>
    </ul>

    <h2>Tagline</h2>
    <p class="tagline">Turn cloud risk and spend into guided, auditable fixes—one AI control plane for FinOps, SecOps, and Compliance.</p>
  `;

  return `<!doctype html>
  <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>PolicyCortex — Value Proposition</title>
      <style>${css}</style>
    </head>
    <body>${body}</body>
  </html>`;
}

generate().catch(err => {
  console.error(err);
  process.exit(1);
});


