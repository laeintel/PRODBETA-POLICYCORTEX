# PolicyCortex Offline Merkle Tree Verifier

A standalone tool for verifying PolicyCortex evidence chain integrity without requiring any API connection. This tool validates exported evidence using SHA-256 hashing and Merkle tree proofs.

## Features

- ✅ **Content Hash Verification**: Validates SHA-256 hash of evidence content
- 🌳 **Merkle Proof Verification**: Rebuilds and verifies Merkle tree proofs using canonical pairing
- 🔐 **Digital Signature Support**: Basic digital signature validation (format checking)
- 📁 **File and CLI Input**: Accepts evidence files or command-line parameters
- 🎯 **Exit Codes**: Returns standard exit codes for automated workflows
- 📊 **JSON Output**: Machine-readable JSON output option

## Installation

### Prerequisites

- Node.js >= 14.0.0

### Setup

```bash
cd tools/offline-verify
npm install
```

### Global Installation (Optional)

```bash
npm install -g .
# Then use as: pcx-verify <options>
```

## Usage

### Basic Usage

```bash
# Verify from evidence file
node verify.js evidence-export.json

# Verify with explicit file flag
node verify.js --file ./example-evidence.json

# Get JSON output
node verify.js --file ./evidence.json --json

# Verbose output
node verify.js --file ./evidence.json --verbose
```

### Command Line Arguments

```bash
# Verify using command-line parameters
node verify.js \
  --hash aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --merkleRoot 3b7546ed79e3e5a7907381b093c5a182cbf364c5dd0443dfa956c8cca271cc33 \
  --proof bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
```

## Input File Format

The verifier expects JSON files with the following structure:

```json
{
  "content": "Raw evidence data to be hashed",
  "contentHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "merkleRoot": "3b7546ed79e3e5a7907381b093c5a182cbf364c5dd0443dfa956c8cca271cc33",
  "merkleProof": [
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  ],
  "signature": "abcd1234...",
  "signer": "pcx-signer",
  "timestamp": "2024-09-05T10:30:00.000Z"
}
```

### Required Fields

- **contentHash**: SHA-256 hash (64 hex characters) of the evidence content
- **merkleRoot**: Root hash of the Merkle tree containing this evidence

### Optional Fields

- **content**: Raw evidence data (if provided, will be hashed and compared to contentHash)
- **merkleProof**: Array of sibling hashes for Merkle tree verification
- **signature**: Digital signature of the evidence (basic format validation)
- **signer**: Identity of the signer
- **timestamp**: ISO 8601 timestamp

## How Merkle Verification Works

The verifier implements the same canonical pairing function as the main PolicyCortex evidence system:

1. **Canonical Ordering**: Hash pairs are sorted lexicographically before concatenation
2. **No Side Flags**: The sorting eliminates the need for left/right indicators
3. **SHA-256 Hashing**: Uses SHA-256 for all hash operations
4. **Proof Walking**: Walks up the Merkle tree using the provided proof path

### Example Verification Process

```
Given:
- Leaf Hash:   aaaa...aaaa
- Proof:       [bbbb...bbbb, cccc...cccc]
- Merkle Root: 3b75...cc33

Step 1: Hash(sort(aaaa...aaaa, bbbb...bbbb)) = dddd...dddd
Step 2: Hash(sort(dddd...dddd, cccc...cccc)) = 3b75...cc33
Result: ✅ Computed root matches expected root
```

## Output Formats

### Standard Output

```
🔍 Verifying content hash...
✅ Content hash valid
🌳 Verifying Merkle proof...
✅ Merkle proof valid
🔐 Verifying digital signature...
✅ Signature format valid (full cryptographic verification not implemented)

==================================================
VERIFICATION RESULTS
==================================================
🎉 VERIFICATION SUCCESSFUL
   Evidence integrity confirmed

Verified at: 2024-09-05T10:30:45.123Z
```

### JSON Output (`--json` flag)

```json
{
  "timestamp": "2024-09-05T10:30:45.123Z",
  "valid": true,
  "checks": {
    "contentHash": {
      "valid": true,
      "computed": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "expected": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    "merkleProof": {
      "valid": true,
      "computed": "3b7546ed79e3e5a7907381b093c5a182cbf364c5dd0443dfa956c8cca271cc33",
      "expected": "3b7546ed79e3e5a7907381b093c5a182cbf364c5dd0443dfa956c8cca271cc33"
    },
    "signature": {
      "valid": true,
      "reason": "Signature format valid (full cryptographic verification not implemented)"
    }
  },
  "errors": []
}
```

## Exit Codes

- **0**: Verification successful - evidence is valid
- **1**: Verification failed - evidence may have been tampered with
- **2**: Error - invalid input or system error

## Examples

### Successful Verification

```bash
$ node verify.js example-evidence.json
📁 Loading evidence from: example-evidence.json
🔍 Verifying content hash...
✅ Content hash valid
🌳 Verifying Merkle proof...
✅ Merkle proof valid
🔐 Verifying digital signature...
✅ Signature format valid

==================================================
VERIFICATION RESULTS
==================================================
🎉 VERIFICATION SUCCESSFUL
   Evidence integrity confirmed

$ echo $?
0
```

### Failed Verification

```bash
$ node verify.js tampered-evidence.json
📁 Loading evidence from: tampered-evidence.json
🔍 Verifying content hash...
❌ Content hash invalid
   Expected: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
   Computed: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

==================================================
VERIFICATION RESULTS
==================================================
💥 VERIFICATION FAILED
   Evidence may have been tampered with

Errors:
   - Content hash mismatch

$ echo $?
1
```

## Integration with CI/CD

```bash
#!/bin/bash
# verification-script.sh

# Verify evidence file
node verify.js evidence-export.json
if [ $? -eq 0 ]; then
    echo "✅ Evidence verification passed"
else
    echo "❌ Evidence verification failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **"Invalid hex hash" Error**
   - Ensure all hashes are exactly 64 hexadecimal characters
   - Check for extra whitespace or invalid characters

2. **"File not found" Error**
   - Verify the evidence file path is correct
   - Ensure the file exists and is readable

3. **"Verification FAILED" Result**
   - This indicates potential tampering with the evidence
   - Double-check the evidence file integrity
   - Ensure you're using the correct evidence export

### Debug Mode

Use the `--verbose` flag to see detailed error information:

```bash
node verify.js --file evidence.json --verbose
```

## Security Considerations

- The tool only validates hash integrity and Merkle proofs
- Digital signature verification is basic format checking only
- For production use, implement full cryptographic signature verification
- Always verify evidence files from trusted sources

## Technical Details

### Cryptographic Functions

- **Hash Function**: SHA-256
- **Merkle Tree**: Binary tree with canonical node ordering
- **Signature**: Format validation only (placeholder for full implementation)

### Dependencies

- **commander**: CLI argument parsing
- **crypto**: Node.js built-in cryptographic functions
- **fs**: File system operations

## Contributing

When extending the verifier:

1. Maintain compatibility with the canonical pairing function
2. Add comprehensive test cases for new features
3. Update documentation for any new command-line options
4. Follow the existing error handling patterns

## License

MIT License - see LICENSE file for details

## Version History

- **v1.0.0**: Initial release with full verification capabilities
  - Content hash validation
  - Merkle proof verification
  - Digital signature format checking
  - JSON output support
  - Comprehensive CLI interface