/*
  Button and Clickable Inventory Script
  - Recursively scans the provided directory (default: frontend)
  - Finds button-like elements and click handlers
  - Outputs a JSON inventory to frontend/clickable-inventory.json
*/

const fs = require('fs');
const path = require('path');

// Robust root detection: allow running from repo root or frontend
const cwd = process.cwd();
const isFrontendCwd = /[\\\/]frontend$/i.test(cwd);
const repoRoot = isFrontendCwd ? path.resolve(cwd, '..') : cwd;
const FRONTEND_ROOT = path.join(repoRoot, 'frontend');
const ROOT = process.argv[2] || FRONTEND_ROOT;
const OUTPUT = path.join(FRONTEND_ROOT, 'clickable-inventory.json');

const IGNORE_DIRS = new Set([
	'node_modules', '.git', '.next', 'dist', 'build', 'out', 'coverage'
]);

const EXTENSIONS = new Set([
	'.ts', '.tsx', '.js', '.jsx', '.vue', '.svelte', '.html'
]);

/**
 * Recursively list files under a directory.
 */
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

// Patterns to detect button-like elements and click handlers
const PATTERNS = [
	{ key: 'onClick', regex: /onClick\s*=\s*\{/ },
	{ key: 'buttonTag', regex: /<button\b[^>]*>/i },
	{ key: 'inputButton', regex: /<input\b[^>]*type\s*=\s*['\"](button|submit|reset)['\"][^>]*>/i },
	{ key: 'aRoleButton', regex: /<a\b[^>]*role\s*=\s*['\"]button['\"][^>]*>/i },
	{ key: 'roleButton', regex: /role\s*=\s*['\"]button['\"]/i },
	{ key: 'addEventListener', regex: /addEventListener\s*\(\s*['\"]click['\"]/i },
	{ key: 'divWithOnClick', regex: /<div\b[^>]*onClick\s*=/i },
	{ key: 'spanWithOnClick', regex: /<span\b[^>]*onClick\s*=/i },
	{ key: 'customButton', regex: /<(?:(?:Icon)?Button|MenuItem|DropdownItem|ListItemButton)\b[^>]*>/ },
	{ key: 'radixItem', regex: /data-\[state=.*?\]|data-\[disabled\]/ },
];

const USE_CLIENT_REGEX = /^\s*['\"]use client['\"];?/m;
const FULLSCREEN_OVERLAY_REGEX = /className\s*=\s*{?['\"][^'\"]*\bfixed\b[^'\"]*\binset-0\b[^'\"]*['\"]/i;

function analyzeLine(line) {
	const info = {};
	for (const p of PATTERNS) {
		if (p.regex.test(line)) info[p.key] = true;
	}
	// Quick attribute checks
	info.hasOnClick = /onClick\s*=/.test(line);
	info.disabled = /\bdisabled\b|aria-disabled\s*=\s*['\"][^'\"]+['\"]/i.test(line);
	const classMatch = line.match(/className\s*=\s*{?['\"]([^'\"]+)['\"]/);
	info.classes = classMatch ? classMatch[1] : undefined;
	if (info.classes && /pointer-events-none/.test(info.classes)) {
		info.pointerEventsNone = true;
	}
	const typeMatch = line.match(/\btype\s*=\s*{?['\"]([^'\"]+)['\"]/i);
	info.hasTypeAttr = Boolean(typeMatch);
	info.typeValue = typeMatch ? typeMatch[1] : undefined;
	return info;
}

function buildInventory(filePath) {
	const content = fs.readFileSync(filePath, 'utf8');
	const lines = content.split(/\r?\n/);
	const entries = [];
  const hasUseClient = USE_CLIENT_REGEX.test(content);
  const hasFullscreenOverlay = FULLSCREEN_OVERLAY_REGEX.test(content);
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		let matched = false;
		for (const p of PATTERNS) {
			if (p.regex.test(line)) { matched = true; break; }
		}
		if (!matched) continue;
		const analysis = analyzeLine(line);
		const elementType =
			analysis.buttonTag ? 'button' :
			analysis.inputButton ? 'input' :
			analysis.aRoleButton ? 'a(role=button)' :
			analysis.divWithOnClick ? 'div(onClick)' :
			analysis.spanWithOnClick ? 'span(onClick)' :
			analysis.customButton ? 'CustomButtonLike' :
			analysis.addEventListener ? 'addEventListener(click)' :
			analysis.roleButton ? 'role=button' : 'unknown';

		entries.push({
			file: path.relative(process.cwd(), filePath).replace(/\\/g, '/'),
			line: i + 1,
			code: line.trim().slice(0, 400),
			elementType,
			hasOnClick: analysis.hasOnClick || false,
			disabled: analysis.disabled || false,
			pointerEventsNone: analysis.pointerEventsNone || false,
			classes: analysis.classes,
			hasUseClient,
			hasFullscreenOverlay,
			hasTypeAttr: analysis.hasTypeAttr || false,
			typeValue: analysis.typeValue,
		});
	}
	return entries;
}

function main() {
	if (!fs.existsSync(ROOT)) {
		console.error(`Directory not found: ${ROOT}`);
		process.exit(1);
	}
	const files = listFiles(ROOT);
	const inventory = [];
	for (const file of files) {
		try {
			const entries = buildInventory(file);
			inventory.push(...entries);
		} catch (err) {
			// Ignore parse errors; continue
		}
	}

	const summary = inventory.reduce((acc, e) => {
		acc.total += 1;
		acc.byType[e.elementType] = (acc.byType[e.elementType] || 0) + 1;
		if (e.elementType === 'button' && !e.hasOnClick) acc.buttonsWithoutOnClick += 1;
		if (e.pointerEventsNone) acc.pointerEventsNone += 1;
		return acc;
	}, { total: 0, buttonsWithoutOnClick: 0, pointerEventsNone: 0, byType: {} });

	const output = { generatedAt: new Date().toISOString(), root: path.relative(process.cwd(), ROOT), summary, items: inventory };

	fs.writeFileSync(OUTPUT, JSON.stringify(output, null, 2));
	console.log(`Wrote clickable inventory to ${OUTPUT}`);
}

main();


