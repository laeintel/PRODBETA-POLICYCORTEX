/*
  Generates DOCX and PDF for Pricing & Valuation.
  Outputs to: ../docs/pricing-and-valuation.docx and ../docs/pricing-and-valuation.pdf
*/
const fs = require('fs');
const path = require('path');

async function generate() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const docsDir = path.join(repoRoot, 'docs');
  const outDocx = path.join(docsDir, 'pricing-and-valuation.docx');
  const outPdf = path.join(docsDir, 'pricing-and-valuation.pdf');

  fs.mkdirSync(docsDir, { recursive: true });

  const html = renderHtml();

  // DOCX from HTML with cross-version handling
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

  // PDF via Puppeteer
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
    table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; }
    th, td { border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }
    th { background: #f8fafc; }
    .muted { color: #475569; }
    .tagline { font-style: italic; color: #0b1220; background: #f8fafc; padding: 10px 12px; border-left: 3px solid #0ea5e9; border-radius: 4px; }
    .small { font-size: 12px; color: #64748b; }
  `;

  const body = `
    <h1>PolicyCortex — Pricing & Valuation</h1>
    <p class="small muted">Multicloud control plane for cost, RBAC/IAM, compliance, and operations. Value-based pricing guidance and valuation snapshots.</p>

    <h2>Pricing strategy overview</h2>
    <p>
      Price on measurable business value captured and scope managed. Use a platform base fee plus usage bands across controlled cloud spend and governed assets/identities, with add-ons for predictive AI and access automation.
    </p>

    <h2>Packaging</h2>
    <ul>
      <li><b>Platform Base</b>: FinOps + SecOps + Compliance workspace, multicloud connectors.</li>
      <li><b>Usage Metrics</b>: Cloud spend under management (CSUM), governed resources/assets, managed identities/users, policy/control count.</li>
      <li><b>Add-ons</b>: Predictive/Correlations AI, SoD/JIT automation, Auditor Workspace, Private Inference / Dedicated VPC, Premium SLA.</li>
    </ul>

    <h2>Recommended pricing levers</h2>
    <ul>
      <li>Base platform fee per tenant (size-tiered).</li>
      <li>CSUM bands (e.g., $0–$1M, $1–$5M, $5–$20M, $20M+).</li>
      <li>Per-governed resource/identity bands (keep bracketed to avoid per-unit nickel-and-diming).</li>
      <li>Add-on multipliers: Predictive/Correlations +15–30%; SoD/JIT +20–35%; Auditor +10–15%; Private/VPC +20–30%.</li>
    </ul>

    <h2>Target ACVs by segment (multicloud)</h2>
    <table>
      <tr><th>Segment</th><th>Base + Usage</th><th>Typical ACV</th></tr>
      <tr><td>SMB</td><td>$15k–$25k base + usage</td><td>$50k–$80k</td></tr>
      <tr><td>Mid‑market</td><td>$60k–$100k base + usage</td><td>$180k–$260k</td></tr>
      <tr><td>Enterprise</td><td>$150k–$250k base + usage</td><td>$450k–$700k</td></tr>
    </table>

    <h2>Value capture rationale</h2>
    <ul>
      <li>Cost savings: 8–15% of cloud spend (FinOps anomaly detection, optimization guidance).</li>
      <li>Risk reduction: least‑privilege, SoD conflict prevention, JIT reduce high‑impact exposure.</li>
      <li>Audit efficiency: 15–30% reduction in audit effort/fees via evidence and continuous controls.</li>
      <li>Ops velocity: faster MTTR/approvals from AI triage and correlations.</li>
    </ul>

    <h2>Sample tiering</h2>
    <table>
      <tr><th>Tier</th><th>CSUM</th><th>Resources</th><th>Identities</th><th>Price (annual)</th></tr>
      <tr><td>Growth</td><td>$0–$1M</td><td>≤5k</td><td>≤2k</td><td>$60k</td></tr>
      <tr><td>Scale</td><td>$1–$5M</td><td>≤20k</td><td>≤10k</td><td>$200k</td></tr>
      <tr><td>Enterprise</td><td>$5–$20M</td><td>≤100k</td><td>≤50k</td><td>$480k</td></tr>
      <tr><td>Elite</td><td>$20M+</td><td>200k+</td><td>100k+</td><td>$700k+</td></tr>
    </table>

    <h2>Enterprise uplift & terms</h2>
    <ul>
      <li>Uplifts: SSO/SCIM, private inference, dedicated VPC, premium SLA (99.9%+), data residency.</li>
      <li>Multi‑year incentives: 2‑year (−7%), 3‑year (−12%) if prepaid annually.</li>
      <li>Volume/CSUM discounts when customers onboard additional cloud accounts or business units.</li>
    </ul>

    <h2>ARR and valuation snapshots</h2>
    <ul>
      <li>10 customers (mixed): ARR ≈ $2.05M → EV ≈ $16M–$25M (8–12× ARR).</li>
      <li>100 customers (mixed): ARR ≈ $20.5M → EV ≈ $125M–$250M.</li>
      <li>Enterprise‑leaning: 10 cust ARR ≈ $3.52M → EV $25M–$42M; 100 cust ARR ≈ $35.2M → EV $210M–$420M.</li>
    </ul>

    <h2>10‑year scenarios</h2>
    <ul>
      <li>Conservative: ARR $60M → EV $300M–$480M (5–8×).</li>
      <li>Base: ARR $150M → EV $900M–$1.5B (6–10×).</li>
      <li>Upside: ARR $300M → EV $2.1B–$3.6B (7–12×).</li>
    </ul>

    <h2>Operational guidance</h2>
    <ul>
      <li>Start with base + CSUM + resource bands; attach SoD/JIT and Predictive as add‑ons.</li>
      <li>Quote ROI with a savings/risk model; target 4–7× ROI for procurement alignment.</li>
      <li>Offer land‑and‑expand: begin with one cloud, expand to multicloud for uplift.</li>
    </ul>

    <h2>Tagline</h2>
    <p class="tagline">Price the correlated value: one AI control plane that reduces spend, risk, and audit burden across all clouds.</p>
  `;

  return `<!doctype html>
  <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>PolicyCortex — Pricing & Valuation</title>
      <style>${css}</style>
    </head>
    <body>${body}</body>
  </html>`;
}

generate().catch(err => {
  console.error(err);
  process.exit(1);
});


