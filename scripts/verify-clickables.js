/*
  Verification script: fails if native <button> is found without onClick and without explicit type
*/
const fs = require('fs');
const path = require('path');

// Robust root detection: allow running from repo root or frontend
const cwd = process.cwd();
const isFrontendCwd = /[\\\/]frontend$/i.test(cwd);
const repoRoot = isFrontendCwd ? path.resolve(cwd, '..') : cwd;
const FRONTEND_ROOT = path.join(repoRoot, 'frontend');
const INVENTORY = path.join(FRONTEND_ROOT, 'clickable-inventory.json');

function main() {
  if (!fs.existsSync(INVENTORY)) {
    console.error('Inventory not found. Run: node scripts/audit-clickables.js');
    process.exit(2);
  }
  const data = JSON.parse(fs.readFileSync(INVENTORY, 'utf8'));
  const violations = data.items.filter(item =>
    item.elementType === 'button' && !item.hasOnClick && !item.hasTypeAttr
  );

  if (violations.length > 0) {
    console.error(`Found ${violations.length} button(s) missing both onClick and type attributes:`);
    for (const v of violations.slice(0, 50)) {
      console.error(`- ${v.file}:${v.line} :: ${v.code}`);
    }
    if (violations.length > 50) console.error(`...and ${violations.length - 50} more.`);
    process.exit(1);
  }

  console.log('Clickable verification passed.');
}

main();


