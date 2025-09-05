export type Hash = string; // 64-hex SHA-256

export interface ArtifactRef {
  uri: string;            // blob://pcx-artifacts/... or s3://...
  contentHash: Hash;      // sha256 hex
  signer?: string;        // key id or subject
  timestamp?: string;     // ISO
}

export interface MerkleProof {
  root: Hash;
  path: string[]; // sibling hashes from leaf→root
}

export interface EvidenceExport {
  artifactRef: string;
  contentHash: Hash;
  signer: string;
  timestamp: string; // ISO
  merkleRoot: Hash;
  proof: string[];
}

export type ToolArtifact =
  | { type: 'EvidenceExport'; payload: EvidenceExport }
  | { type: 'GenericArtifact'; payload: ArtifactRef };

export function isSha256Hex(s: string): boolean {
  return /^[a-f0-9]{64}$/.test(s);
}