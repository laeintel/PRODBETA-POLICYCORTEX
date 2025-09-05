/**
 * PolicyCortex Offline Merkle Verifier (T05)
 * Standalone verification without API calls
 * 
 * Usage:
 *   node verify.mjs --file ./artifact.json
 *   node verify.mjs --hash <64hex> --merkleRoot <64hex> --proof <hex,hex,...>
 */

import { createHash } from 'crypto';
import fs from 'node:fs';

function hexToBuf(h) {
  if (!/^[a-f0-9]{64}$/i.test(h)) throw new Error(`Invalid hex: ${h}`);
  return Buffer.from(h, 'hex');
}

function hashPair(a, b) {
  // Canonical pair (matches core): order lexicographically so we don't need side flags
  const [left, right] = Buffer.compare(a, b) <= 0 ? [a, b] : [b, a];
  const h = createHash('sha256');
  h.update(left);
  h.update(right);
  return h.digest();
}

function verifyFromParts(hashHex, merkleRootHex, proofHexList) {
  let cur = hexToBuf(hashHex);
  for (const ph of proofHexList) {
    cur = hashPair(cur, hexToBuf(ph));
  }
  const rootHex = cur.toString('hex');
  return { ok: rootHex === merkleRootHex.toLowerCase(), computed: rootHex };
}

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {};
  for (let i = 0; i < args.length; i++) {
    const k = args[i];
    const v = args[i + 1];
    if (k === '--file') out.file = v;
    if (k === '--hash') out.hash = v;
    if (k === '--merkleRoot') out.merkleRoot = v;
    if (k === '--proof') out.proof = v;
  }
  return out;
}

const a = parseArgs();

if (a.file) {
  // Verify from artifact file
  const raw = fs.readFileSync(a.file, 'utf8');
  const j = JSON.parse(raw);
  const hash = j.contentHash || j.hash;
  const merkleRoot = j.merkleRoot;
  const proof = j.proof || [];
  
  if (!hash || !merkleRoot) {
    console.error('Artifact missing contentHash/merkleRoot');
    process.exit(2);
  }
  
  const res = verifyFromParts(hash, merkleRoot, proof);
  console.log(JSON.stringify({ 
    verified: res.ok, 
    computedMerkleRoot: res.computed,
    providedMerkleRoot: merkleRoot,
    contentHash: hash
  }, null, 2));
  
  if (res.ok) {
    console.log('\n✅ Verification SUCCESSFUL - Evidence integrity confirmed');
  } else {
    console.log('\n❌ Verification FAILED - Evidence may have been tampered');
  }
  
  process.exit(res.ok ? 0 : 1);
  
} else if (a.hash && a.merkleRoot && a.proof) {
  // Verify from command-line arguments
  const proof = a.proof.split(',').map(s => s.trim()).filter(Boolean);
  const res = verifyFromParts(a.hash, a.merkleRoot, proof);
  
  console.log(JSON.stringify({ 
    verified: res.ok, 
    computedMerkleRoot: res.computed,
    providedMerkleRoot: a.merkleRoot,
    contentHash: a.hash
  }, null, 2));
  
  if (res.ok) {
    console.log('\n✅ Verification SUCCESSFUL - Evidence integrity confirmed');
  } else {
    console.log('\n❌ Verification FAILED - Evidence may have been tampered');
  }
  
  process.exit(res.ok ? 0 : 1);
  
} else {
  // Show usage
  console.log('PolicyCortex Offline Merkle Verifier');
  console.log('=====================================');
  console.log('');
  console.log('Usage:');
  console.log('  node verify.mjs --file <artifact.json>');
  console.log('  node verify.mjs --hash <hex> --merkleRoot <hex> --proof <hex,hex,...>');
  console.log('');
  console.log('Examples:');
  console.log('  # Verify an exported artifact file');
  console.log('  node verify.mjs --file ./evidence-export.json');
  console.log('');
  console.log('  # Verify using command-line parameters');
  console.log('  node verify.mjs \\');
  console.log('    --hash aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \\');
  console.log('    --merkleRoot 3b7546ed79e3e5a7907381b093c5a182cbf364c5dd0443dfa956c8cca271cc33 \\');
  console.log('    --proof bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb');
  console.log('');
  console.log('This tool verifies Merkle proofs offline without any API calls.');
  console.log('Exit code 0 = verified, 1 = failed, 2 = error');
  
  process.exit(2);
}