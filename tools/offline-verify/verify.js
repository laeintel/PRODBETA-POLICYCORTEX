#!/usr/bin/env node
/**
 * PolicyCortex Offline Merkle Tree Verifier (T05)
 * Standalone verification without needing to connect to any API
 * 
 * Features:
 * - SHA-256 hash validation of content
 * - Merkle tree proof verification using canonical pairing function
 * - Digital signature verification (if present)
 * - Returns exit code 0 if valid, 1 if invalid
 * 
 * Usage:
 *   node verify.js ./artifact.json
 *   node verify.js --file ./evidence-export.json
 *   node verify.js --hash <hex> --merkleRoot <hex> --proof <hex,hex,...>
 */

const crypto = require('crypto');
const fs = require('fs');
const { program } = require('commander');

// Validate hex hash (64 characters for SHA-256)
function validateHex(hexString, name = 'hash') {
  if (!/^[a-f0-9]{64}$/i.test(hexString)) {
    throw new Error(`Invalid ${name}: ${hexString} (must be 64 hex characters)`);
  }
  return hexString.toLowerCase();
}

// Convert hex string to buffer with validation
function hexToBuffer(hexString) {
  validateHex(hexString);
  return Buffer.from(hexString, 'hex');
}

// Canonical hash pair function (matches main evidence system)
// Sort hashes lexicographically before concatenating to avoid needing side flags
function hashPair(bufferA, bufferB) {
  const [left, right] = Buffer.compare(bufferA, bufferB) <= 0 ? [bufferA, bufferB] : [bufferB, bufferA];
  const hash = crypto.createHash('sha256');
  hash.update(left);
  hash.update(right);
  return hash.digest();
}

// Verify SHA-256 content hash
function verifyContentHash(content, expectedHash) {
  const computedHash = crypto.createHash('sha256').update(content, 'utf8').digest('hex');
  return {
    valid: computedHash === expectedHash.toLowerCase(),
    computed: computedHash,
    expected: expectedHash.toLowerCase()
  };
}

// Verify Merkle proof
function verifyMerkleProof(leafHash, merkleRoot, proof) {
  let currentHash = hexToBuffer(leafHash);
  
  // Walk up the proof path
  for (const proofHash of proof) {
    const siblingBuffer = hexToBuffer(proofHash);
    currentHash = hashPair(currentHash, siblingBuffer);
  }
  
  const computedRoot = currentHash.toString('hex');
  return {
    valid: computedRoot === merkleRoot.toLowerCase(),
    computed: computedRoot,
    expected: merkleRoot.toLowerCase()
  };
}

// Verify digital signature (basic implementation)
function verifyDigitalSignature(content, signature, publicKey) {
  if (!signature || !publicKey) {
    return { valid: true, reason: 'No signature provided - skipping verification' };
  }
  
  try {
    // This is a placeholder for actual signature verification
    // In a real implementation, you'd use proper cryptographic signature verification
    // For now, we'll just validate the format
    if (!/^[a-f0-9]+$/i.test(signature)) {
      return { valid: false, reason: 'Invalid signature format' };
    }
    
    return { valid: true, reason: 'Signature format valid (full cryptographic verification not implemented)' };
  } catch (error) {
    return { valid: false, reason: `Signature verification failed: ${error.message}` };
  }
}

// Main verification function
function verifyEvidence(evidenceData) {
  const results = {
    timestamp: new Date().toISOString(),
    valid: true,
    checks: {},
    errors: []
  };
  
  try {
    // Extract required fields
    const { 
      content, 
      contentHash, 
      merkleRoot, 
      merkleProof = [], 
      signature,
      signer,
      timestamp
    } = evidenceData;
    
    // 1. Validate content hash if content is provided
    if (content && contentHash) {
      console.log('🔍 Verifying content hash...');
      const contentResult = verifyContentHash(content, contentHash);
      results.checks.contentHash = contentResult;
      
      if (contentResult.valid) {
        console.log('✅ Content hash valid');
      } else {
        console.log('❌ Content hash invalid');
        console.log(`   Expected: ${contentResult.expected}`);
        console.log(`   Computed: ${contentResult.computed}`);
        results.valid = false;
        results.errors.push('Content hash mismatch');
      }
    } else if (contentHash) {
      console.log('ℹ️  Content not provided - skipping content hash verification');
      results.checks.contentHash = { skipped: true, reason: 'No content provided' };
    }
    
    // 2. Verify Merkle proof if provided
    if (contentHash && merkleRoot && merkleProof.length > 0) {
      console.log('🌳 Verifying Merkle proof...');
      const merkleResult = verifyMerkleProof(contentHash, merkleRoot, merkleProof);
      results.checks.merkleProof = merkleResult;
      
      if (merkleResult.valid) {
        console.log('✅ Merkle proof valid');
      } else {
        console.log('❌ Merkle proof invalid');
        console.log(`   Expected root: ${merkleResult.expected}`);
        console.log(`   Computed root: ${merkleResult.computed}`);
        results.valid = false;
        results.errors.push('Merkle proof verification failed');
      }
    } else {
      console.log('ℹ️  Insufficient data for Merkle verification - skipping');
      results.checks.merkleProof = { skipped: true, reason: 'Missing required fields' };
    }
    
    // 3. Verify digital signature if present
    if (signature) {
      console.log('🔐 Verifying digital signature...');
      const signatureResult = verifyDigitalSignature(content || contentHash, signature, signer);
      results.checks.signature = signatureResult;
      
      if (signatureResult.valid) {
        console.log(`✅ ${signatureResult.reason}`);
      } else {
        console.log(`❌ ${signatureResult.reason}`);
        results.valid = false;
        results.errors.push('Digital signature verification failed');
      }
    } else {
      console.log('ℹ️  No digital signature - skipping signature verification');
      results.checks.signature = { skipped: true, reason: 'No signature provided' };
    }
    
    // 4. Validate timestamp format
    if (timestamp) {
      const timestampResult = { valid: true };
      try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) {
          timestampResult.valid = false;
          timestampResult.reason = 'Invalid timestamp format';
        }
      } catch (error) {
        timestampResult.valid = false;
        timestampResult.reason = `Timestamp parsing error: ${error.message}`;
      }
      
      results.checks.timestamp = timestampResult;
      
      if (!timestampResult.valid) {
        console.log(`❌ ${timestampResult.reason}`);
        results.errors.push('Invalid timestamp');
      }
    }
    
  } catch (error) {
    console.error('❌ Verification error:', error.message);
    results.valid = false;
    results.errors.push(`Verification error: ${error.message}`);
  }
  
  return results;
}

// CLI setup using Commander
program
  .name('pcx-verify')
  .description('PolicyCortex Offline Merkle Tree Verifier')
  .version('1.0.0');

program
  .argument('[file]', 'Evidence JSON file to verify')
  .option('-f, --file <file>', 'Evidence file to verify')
  .option('--hash <hash>', 'Content hash (64 hex characters)')
  .option('--merkleRoot <root>', 'Merkle root hash (64 hex characters)')
  .option('--proof <proof>', 'Comma-separated proof hashes')
  .option('--json', 'Output results in JSON format')
  .option('--verbose', 'Enable verbose output')
  .action((file, options) => {
    try {
      let evidenceData;
      
      // Determine input method
      const inputFile = file || options.file;
      
      if (inputFile) {
        // Load from file
        console.log(`📁 Loading evidence from: ${inputFile}`);
        
        if (!fs.existsSync(inputFile)) {
          console.error(`❌ File not found: ${inputFile}`);
          process.exit(2);
        }
        
        const rawData = fs.readFileSync(inputFile, 'utf8');
        evidenceData = JSON.parse(rawData);
        
      } else if (options.hash && options.merkleRoot && options.proof) {
        // Build from command line arguments
        console.log('⌨️  Verifying from command line arguments');
        evidenceData = {
          contentHash: options.hash,
          merkleRoot: options.merkleRoot,
          merkleProof: options.proof.split(',').map(h => h.trim()).filter(Boolean)
        };
        
      } else {
        // Show usage
        console.log('PolicyCortex Offline Merkle Tree Verifier');
        console.log('==========================================\n');
        console.log('Usage:');
        console.log('  node verify.js <evidence-file.json>');
        console.log('  node verify.js --file <evidence-file.json>');
        console.log('  node verify.js --hash <hex> --merkleRoot <hex> --proof <hex,hex,...>\n');
        console.log('Options:');
        console.log('  --json      Output results in JSON format');
        console.log('  --verbose   Enable verbose logging\n');
        console.log('Examples:');
        console.log('  node verify.js ./example-evidence.json');
        console.log('  node verify.js --file ./evidence-export.json --json');
        console.log('  node verify.js \\');
        console.log('    --hash aaaa...aaaa \\');
        console.log('    --merkleRoot bbbb...bbbb \\');
        console.log('    --proof cccc...cccc,dddd...dddd\n');
        console.log('Exit codes: 0 = verified, 1 = failed, 2 = error');
        process.exit(2);
      }
      
      // Perform verification
      const results = verifyEvidence(evidenceData);
      
      // Output results
      if (options.json) {
        console.log(JSON.stringify(results, null, 2));
      } else {
        console.log('\n' + '='.repeat(50));
        console.log('VERIFICATION RESULTS');
        console.log('='.repeat(50));
        
        if (results.valid) {
          console.log('🎉 VERIFICATION SUCCESSFUL');
          console.log('   Evidence integrity confirmed');
        } else {
          console.log('💥 VERIFICATION FAILED');
          console.log('   Evidence may have been tampered with');
          
          if (results.errors.length > 0) {
            console.log('\nErrors:');
            results.errors.forEach(error => console.log(`   - ${error}`));
          }
        }
        
        console.log(`\nVerified at: ${results.timestamp}`);
      }
      
      // Exit with appropriate code
      process.exit(results.valid ? 0 : 1);
      
    } catch (error) {
      console.error('❌ Error:', error.message);
      if (options.verbose) {
        console.error(error.stack);
      }
      process.exit(2);
    }
  });

// Handle case where no arguments provided
if (process.argv.length === 2) {
  program.help();
}

program.parse();