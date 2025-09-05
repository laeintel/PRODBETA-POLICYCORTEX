/**
 * PolicyCortex Evidence & Events Server (T04/T05)
 * Node.js implementation for events and Merkle verification
 */

const express = require('express');
const cors = require('cors');
const crypto = require('crypto');

const app = express();
app.use(cors());
app.use(express.json());

// In-memory stores (for demo; production would use PostgreSQL)
const events = [];
const evidenceStore = new Map();
let nextEventId = 1;

// Helper functions for Merkle tree
function hexToBuffer(hex) {
  if (!/^[a-f0-9]{64}$/i.test(hex)) {
    throw new Error(`Invalid hex hash: ${hex}`);
  }
  return Buffer.from(hex, 'hex');
}

function bufferToHex(buffer) {
  return buffer.toString('hex');
}

function hashPair(a, b) {
  const [left, right] = Buffer.compare(a, b) <= 0 ? [a, b] : [b, a];
  const hash = crypto.createHash('sha256');
  hash.update(left);
  hash.update(right);
  return hash.digest();
}

function merkleRootAndProof(leaves, targetIndex) {
  if (leaves.length === 0) {
    throw new Error('Merkle tree needs at least one leaf');
  }
  if (targetIndex >= leaves.length) {
    throw new Error('Target index out of bounds');
  }

  const proof = [];
  let idx = targetIndex;
  let level = [...leaves];

  while (level.length > 1) {
    const nextLevel = [];
    
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) {
        nextLevel.push(hashPair(level[i], level[i + 1]));
      } else {
        nextLevel.push(hashPair(level[i], level[i]));
      }
    }

    if (level.length > 1) {
      const pairBase = idx - (idx % 2);
      const siblingIdx = pairBase === idx ? idx + 1 : idx - 1;
      const sibling = siblingIdx < level.length ? level[siblingIdx] : level[idx];
      proof.push(sibling);
      idx = Math.floor(idx / 2);
    }

    level = nextLevel;
  }

  return { root: level[0], proof };
}

// --- Event Store Endpoints (T04) ---

app.get('/health', (req, res) => {
  res.json({
    ok: true,
    db_ok: true,
    time: new Date().toISOString()
  });
});

app.post('/api/v1/events', (req, res) => {
  const { payload } = req.body;
  
  const event = {
    id: nextEventId++,
    ts: new Date().toISOString(),
    payload: payload || req.body
  };
  
  events.push(event);
  
  res.json({
    id: event.id,
    ts: event.ts
  });
});

app.get('/api/v1/events', (req, res) => {
  res.json(events.map(e => ({
    id: e.id,
    ts: e.ts,
    payload: e.payload
  })));
});

app.get('/api/v1/events/replay', (req, res) => {
  res.json(events.map(e => e.payload));
});

// --- Evidence & Merkle Endpoints (T05) ---

// Store evidence
app.post('/api/v1/evidence', (req, res) => {
  const { contentHash, signer, metadata } = req.body;
  
  if (!contentHash || !/^[a-f0-9]{64}$/i.test(contentHash)) {
    return res.status(400).json({ error: 'Invalid content hash' });
  }
  
  const evidence = {
    contentHash: contentHash.toLowerCase(),
    timestamp: new Date(),
    signer: signer || 'pcx-signer',
    metadata: metadata || {}
  };
  
  evidenceStore.set(evidence.contentHash, evidence);
  
  res.json({
    id: evidence.contentHash,
    timestamp: evidence.timestamp.toISOString()
  });
});

// Verify hash with Merkle proof
app.get('/api/v1/verify/:hash', (req, res) => {
  const hashHex = req.params.hash.toLowerCase();
  
  if (!/^[a-f0-9]{64}$/i.test(hashHex)) {
    return res.status(400).json({ error: 'Invalid hash format' });
  }
  
  const evidence = evidenceStore.get(hashHex);
  
  if (!evidence) {
    return res.json({
      verified: false,
      merkleRoot: '',
      proof: [],
      day: ''
    });
  }
  
  // Get all evidence for the same day
  const targetDay = evidence.timestamp.toISOString().split('T')[0];
  const dayEvidence = Array.from(evidenceStore.values())
    .filter(e => e.timestamp.toISOString().startsWith(targetDay))
    .sort((a, b) => a.contentHash.localeCompare(b.contentHash));
  
  const targetIndex = dayEvidence.findIndex(e => e.contentHash === hashHex);
  
  if (targetIndex === -1) {
    return res.json({
      verified: false,
      merkleRoot: '',
      proof: [],
      day: ''
    });
  }
  
  try {
    const leaves = dayEvidence.map(e => hexToBuffer(e.contentHash));
    const { root, proof } = merkleRootAndProof(leaves, targetIndex);
    
    res.json({
      verified: true,
      merkleRoot: bufferToHex(root),
      proof: proof.map(bufferToHex),
      day: targetDay
    });
  } catch (error) {
    console.error('Merkle verification error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Export evidence with Merkle proof
app.post('/api/v1/evidence/export', (req, res) => {
  const { contentHashes, day } = req.body;
  
  let evidenceToExport;
  
  if (contentHashes && Array.isArray(contentHashes)) {
    evidenceToExport = contentHashes
      .map(h => evidenceStore.get(h.toLowerCase()))
      .filter(Boolean);
  } else if (day) {
    evidenceToExport = Array.from(evidenceStore.values())
      .filter(e => e.timestamp.toISOString().startsWith(day));
  } else {
    evidenceToExport = Array.from(evidenceStore.values());
  }
  
  if (evidenceToExport.length === 0) {
    return res.json({
      exports: [],
      merkleRoot: '',
      totalCount: 0
    });
  }
  
  // Sort for deterministic Merkle tree
  evidenceToExport.sort((a, b) => a.contentHash.localeCompare(b.contentHash));
  
  try {
    const leaves = evidenceToExport.map(e => hexToBuffer(e.contentHash));
    let level = [...leaves];
    
    while (level.length > 1) {
      const nextLevel = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) {
          nextLevel.push(hashPair(level[i], level[i + 1]));
        } else {
          nextLevel.push(hashPair(level[i], level[i]));
        }
      }
      level = nextLevel;
    }
    
    const merkleRoot = bufferToHex(level[0]);
    
    // Generate individual proofs
    const exports = evidenceToExport.map((e, idx) => {
      const { proof } = merkleRootAndProof(leaves, idx);
      return {
        artifactRef: `evidence://${e.contentHash}`,
        contentHash: e.contentHash,
        signer: e.signer,
        timestamp: e.timestamp.toISOString(),
        merkleRoot,
        proof: proof.map(bufferToHex)
      };
    });
    
    res.json({
      exports,
      merkleRoot,
      totalCount: exports.length,
      exportedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error('Export error:', error);
    res.status(500).json({ error: 'Export failed' });
  }
});

// Seed some test data
function seedTestData() {
  // Add some test events
  const testEvents = [
    {
      type: 'PredictionIssued',
      ruleId: 'AZ-NSG-OPEN-22',
      etaDays: 7,
      confidence: 0.82,
      repo: 'org/infra',
      fixBranch: 'pcx/autofix/AZ-NSG-OPEN-22',
      timestamp: new Date().toISOString()
    },
    {
      type: 'ChainVerified',
      hash: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      verified: true,
      merkleRoot: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      timestamp: new Date().toISOString()
    }
  ];
  
  testEvents.forEach(event => {
    events.push({
      id: nextEventId++,
      ts: new Date().toISOString(),
      payload: event
    });
  });
  
  // Add some test evidence
  const testEvidence = [
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
  ];
  
  testEvidence.forEach(hash => {
    evidenceStore.set(hash, {
      contentHash: hash,
      timestamp: new Date(),
      signer: 'pcx-signer',
      metadata: { test: true }
    });
  });
  
  console.log(`Seeded ${testEvents.length} events and ${testEvidence.length} evidence entries`);
}

// Start server
const PORT = process.env.EVIDENCE_PORT || 8081;

app.listen(PORT, () => {
  console.log(`PolicyCortex Evidence Server listening on http://localhost:${PORT}`);
  console.log('Endpoints:');
  console.log('  - GET  /health');
  console.log('  - POST /api/v1/events');
  console.log('  - GET  /api/v1/events');
  console.log('  - GET  /api/v1/events/replay');
  console.log('  - POST /api/v1/evidence');
  console.log('  - GET  /api/v1/verify/:hash');
  console.log('  - POST /api/v1/evidence/export');
  
  // Seed test data
  seedTestData();
});

module.exports = app;