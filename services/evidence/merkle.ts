/**
 * PolicyCortex Merkle Tree Implementation (T05)
 * SHA-256 based Merkle tree for evidence verification
 */

import { createHash } from 'crypto';

export type Hash32 = Buffer;

/**
 * Convert hex string to Buffer
 */
export function hexToHash(hex: string): Hash32 {
  if (!/^[a-f0-9]{64}$/i.test(hex)) {
    throw new Error(`Invalid hex hash: ${hex}`);
  }
  return Buffer.from(hex, 'hex');
}

/**
 * Convert Buffer to hex string
 */
export function hashToHex(hash: Hash32): string {
  return hash.toString('hex');
}

/**
 * Hash two buffers together using canonical pairing
 * (order lexicographically to avoid needing side flags in proof)
 */
export function hashPair(a: Hash32, b: Hash32): Hash32 {
  const [left, right] = Buffer.compare(a, b) <= 0 ? [a, b] : [b, a];
  const hasher = createHash('sha256');
  hasher.update(left);
  hasher.update(right);
  return hasher.digest();
}

/**
 * Hash a string to get its SHA-256
 */
export function hashString(data: string): Hash32 {
  return createHash('sha256').update(data, 'utf8').digest();
}

/**
 * Build Merkle root and proof for verification
 */
export function merkleRootAndProof(
  leaves: Hash32[], 
  targetIndex: number
): { root: Hash32; proof: Hash32[] } {
  if (leaves.length === 0) {
    throw new Error('Merkle tree needs at least one leaf');
  }
  if (targetIndex >= leaves.length) {
    throw new Error('Target index out of bounds');
  }

  const proof: Hash32[] = [];
  let idx = targetIndex;
  let level = [...leaves];

  while (level.length > 1) {
    const nextLevel: Hash32[] = [];
    
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) {
        // Pair exists
        nextLevel.push(hashPair(level[i], level[i + 1]));
      } else {
        // Odd leaf: hash with itself (idempotent)
        nextLevel.push(hashPair(level[i], level[i]));
      }
    }

    // Record sibling for proof
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

/**
 * Verify a Merkle proof
 */
export function verifyMerkleProof(
  leaf: Hash32,
  merkleRoot: Hash32,
  proof: Hash32[]
): boolean {
  let current = leaf;
  
  for (const sibling of proof) {
    current = hashPair(current, sibling);
  }
  
  return Buffer.compare(current, merkleRoot) === 0;
}

/**
 * Build Merkle tree from string data
 */
export function buildMerkleTree(data: string[]): {
  leaves: Hash32[];
  root: Hash32;
} {
  if (data.length === 0) {
    throw new Error('Cannot build Merkle tree from empty data');
  }

  const leaves = data.map(hashString);
  let level = [...leaves];

  while (level.length > 1) {
    const nextLevel: Hash32[] = [];
    
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) {
        nextLevel.push(hashPair(level[i], level[i + 1]));
      } else {
        nextLevel.push(hashPair(level[i], level[i]));
      }
    }
    
    level = nextLevel;
  }

  return { leaves, root: level[0] };
}

/**
 * Evidence storage interface
 */
export interface EvidenceEntry {
  contentHash: string;
  timestamp: Date;
  signer?: string;
  metadata?: Record<string, any>;
}

/**
 * Build Merkle tree for a day's evidence
 */
export function buildEvidenceMerkleTree(
  entries: EvidenceEntry[]
): {
  root: string;
  leaves: string[];
} {
  if (entries.length === 0) {
    throw new Error('No evidence entries provided');
  }

  // Sort entries by content hash for deterministic ordering
  const sorted = [...entries].sort((a, b) => 
    a.contentHash.localeCompare(b.contentHash)
  );

  const leaves = sorted.map(e => e.contentHash);
  const leafHashes = leaves.map(hexToHash);
  
  let level = [...leafHashes];
  while (level.length > 1) {
    const nextLevel: Hash32[] = [];
    
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) {
        nextLevel.push(hashPair(level[i], level[i + 1]));
      } else {
        nextLevel.push(hashPair(level[i], level[i]));
      }
    }
    
    level = nextLevel;
  }

  return {
    root: hashToHex(level[0]),
    leaves
  };
}

/**
 * Generate proof for specific evidence hash
 */
export function generateEvidenceProof(
  entries: EvidenceEntry[],
  targetHash: string
): {
  verified: boolean;
  merkleRoot: string;
  proof: string[];
  day?: string;
} {
  const sorted = [...entries].sort((a, b) => 
    a.contentHash.localeCompare(b.contentHash)
  );

  const targetIndex = sorted.findIndex(e => 
    e.contentHash.toLowerCase() === targetHash.toLowerCase()
  );

  if (targetIndex === -1) {
    return {
      verified: false,
      merkleRoot: '',
      proof: []
    };
  }

  const leaves = sorted.map(e => hexToHash(e.contentHash));
  const { root, proof } = merkleRootAndProof(leaves, targetIndex);

  return {
    verified: true,
    merkleRoot: hashToHex(root),
    proof: proof.map(hashToHex),
    day: sorted[targetIndex].timestamp.toISOString().split('T')[0]
  };
}