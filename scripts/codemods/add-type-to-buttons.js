/*
  Codemod: Add type="button" to native <button> tags missing a type attribute.
  - Scans the frontend directory
  - Updates files in-place by default
  - Skips buttons that already have type= attribute
*/
const fs = require('fs');
const path = require('path');

const DRY_RUN = process.argv.includes('--dry');
const cwd = process.cwd();
const isFrontendCwd = /[\\\/]frontend$/i.test(cwd);
const repoRoot = isFrontendCwd ? path.resolve(cwd, '..') : cwd;
const FRONTEND_ROOT = path.join(repoRoot, 'frontend');

const IGNORE_DIRS = new Set(['node_modules', '.git', '.next', 'dist', 'build', 'out', 'coverage']);
const EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx', '.vue', '.svelte', '.html']);

function listFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  let files = [];
  for (const entry of entries) {
    if (IGNORE_DIRS.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files = files.concat(listFiles(full));
    } else if (EXTENSIONS.has(path.extname(entry.name))) {
      files.push(full);
    }
  }
  return files;
}

function transform(content) {
  // Add type="button" to <button> tags that do not already declare a type= attribute
  // Regex explanation:
  //  - <button\b matches the tag start
  //  - (?![^>]*\btype\s*=) ensures no type attribute appears before the closing '>'
  return content.replace(/<button\b(?![^>]*\btype\s*=)/gis, '<button type="button"');
}

function main() {
  if (!fs.existsSync(FRONTEND_ROOT)) {
    console.error('Frontend directory not found at', FRONTEND_ROOT);
    process.exit(1);
  }
  const files = listFiles(FRONTEND_ROOT);
  let changed = 0;
  for (const file of files) {
    const before = fs.readFileSync(file, 'utf8');
    const after = transform(before);
    if (after !== before) {
      changed++;
      if (!DRY_RUN) fs.writeFileSync(file, after);
      console.log((DRY_RUN ? '[dry] ' : '') + 'Updated', path.relative(repoRoot, file));
    }
  }
  console.log(`${DRY_RUN ? 'Would update' : 'Updated'} ${changed} file(s).`);
}

main();


