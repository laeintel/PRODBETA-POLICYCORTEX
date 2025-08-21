const fs = require('fs');
const path = require('path');

const cwd = process.cwd();
const isFrontendCwd = /[\\\/]frontend$/i.test(cwd);
const repoRoot = isFrontendCwd ? path.resolve(cwd, '..') : cwd;
const INVENTORY = path.join(repoRoot, 'frontend', 'clickable-inventory.json');

function main() {
  const data = JSON.parse(fs.readFileSync(INVENTORY, 'utf8'));
  const byFile = new Map();
  for (const item of data.items) {
    if (item.elementType === 'button' && !item.hasOnClick) {
      const arr = byFile.get(item.file) || [];
      arr.push(item);
      byFile.set(item.file, arr);
    }
  }
  const result = Array.from(byFile.entries())
    .map(([file, items]) => ({ file, count: items.length }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 15);
  console.log(JSON.stringify(result, null, 2));
}

main();


