const fs = require('fs');
const path = require('path');

const cwd = process.cwd();
const isFrontend = /[\\\/]frontend$/i.test(cwd);
const repoRoot = isFrontend ? path.resolve(cwd, '..') : cwd;
const FRONTEND_ROOT = path.join(repoRoot, 'frontend');
const INVENTORY = path.join(FRONTEND_ROOT, 'clickable-inventory.json');
const DOCS = path.join(repoRoot, 'docs', 'botton-wired.md');

function main() {
  if (!fs.existsSync(INVENTORY)) {
    console.error('Inventory missing. Run: node scripts/audit-clickables.js');
    process.exit(2);
  }
  const data = JSON.parse(fs.readFileSync(INVENTORY, 'utf8'));
  const include = (code) => (
    /onClick\s*=/.test(code) && (
      /toast\(/.test(code) || /router\.push\(/.test(code) || /setShowHistory\(/.test(code) || /console\.log\(/.test(code)
    )
  );
  const items = data.items.filter(i => include(i.code));
  const byFile = new Map();
  for (const it of items) {
    const arr = byFile.get(it.file) || [];
    arr.push(it);
    byFile.set(it.file, arr);
  }
  const lines = [];
  lines.push('# Buttons wired with onClick or navigation');
  lines.push('');
  lines.push('Generated from clickable-inventory.json');
  for (const file of Array.from(byFile.keys()).sort()) {
    lines.push('');
    lines.push(`## ${file}`);
    const arr = byFile.get(file).sort((a,b)=>a.line-b.line);
    for (const it of arr) {
      lines.push(`- ${file}:${it.line} :: ${it.code}`);
    }
  }
  fs.mkdirSync(path.dirname(DOCS), { recursive: true });
  fs.writeFileSync(DOCS, lines.join('\n'));
  console.log('Wrote', DOCS, 'with', items.length, 'entries');
}

main();


