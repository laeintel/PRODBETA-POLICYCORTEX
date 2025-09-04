use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey, Verifier};
use rand::rngs::OsRng;

/// Maximum events per block
const MAX_EVENTS_PER_BLOCK: usize = 100;
/// Maximum time between blocks (1 hour in seconds)
const MAX_BLOCK_INTERVAL_SECS: i64 = 3600;

/// Evidence structure representing a single audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub resource_id: String,
    pub policy_id: String,
    pub compliance_status: ComplianceStatus,
    pub actor: String,
    pub details: serde_json::Value,
    pub metadata: HashMap<String, String>,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    Warning,
    Error,
    Pending,
}

/// Block in the hash chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index: u64,
    pub timestamp: DateTime<Utc>,
    pub previous_hash: String,
    pub merkle_root: String,
    pub events: Vec<Evidence>,
    pub nonce: u64,
    pub signature: String,
    pub signer_public_key: String,
    pub block_hash: String,
}

/// Merkle tree proof for evidence verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub evidence_hash: String,
    pub merkle_path: Vec<MerkleNode>,
    pub block_index: u64,
    pub merkle_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    pub hash: String,
    pub position: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Position {
    Left,
    Right,
}

/// Chain verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainVerificationResult {
    pub is_valid: bool,
    pub total_blocks: u64,
    pub total_events: usize,
    pub invalid_blocks: Vec<u64>,
    pub verification_timestamp: DateTime<Utc>,
    pub chain_hash: String,
}

/// Chain status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStatus {
    pub total_blocks: u64,
    pub total_events: usize,
    pub last_block_timestamp: DateTime<Utc>,
    pub last_block_hash: String,
    pub chain_start_timestamp: DateTime<Utc>,
    pub pending_events: usize,
}

/// Main hash chain implementation
pub struct HashChain {
    blocks: Arc<RwLock<Vec<Block>>>,
    pending_events: Arc<RwLock<Vec<Evidence>>>,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    last_block_time: Arc<RwLock<DateTime<Utc>>>,
}

impl HashChain {
    /// Create a new hash chain
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();

        let genesis_block = Self::create_genesis_block(&signing_key);
        
        Self {
            blocks: Arc::new(RwLock::new(vec![genesis_block.clone()])),
            pending_events: Arc::new(RwLock::new(Vec::new())),
            signing_key,
            verifying_key,
            last_block_time: Arc::new(RwLock::new(genesis_block.timestamp)),
        }
    }

    /// Create the genesis block
    fn create_genesis_block(signing_key: &SigningKey) -> Block {
        let timestamp = Utc::now();
        let mut block = Block {
            index: 0,
            timestamp,
            previous_hash: "0".repeat(64),
            merkle_root: "0".repeat(64),
            events: Vec::new(),
            nonce: 0,
            signature: String::new(),
            signer_public_key: hex::encode(signing_key.verifying_key().as_bytes()),
            block_hash: String::new(),
        };

        // Calculate block hash
        block.block_hash = Self::calculate_block_hash(&block);
        
        // Sign the block
        let signature = signing_key.sign(block.block_hash.as_bytes());
        block.signature = hex::encode(signature.to_bytes());

        block
    }

    /// Add evidence to the chain
    pub async fn add_evidence(&self, mut evidence: Evidence) -> Result<String, String> {
        // Calculate evidence hash
        evidence.hash = Self::calculate_evidence_hash(&evidence);

        // Add to pending events
        let mut pending = self.pending_events.write().await;
        pending.push(evidence.clone());

        // Check if we should create a new block
        let should_create_block = {
            let last_time = self.last_block_time.read().await;
            let time_elapsed = Utc::now().signed_duration_since(*last_time).num_seconds();
            
            pending.len() >= MAX_EVENTS_PER_BLOCK || time_elapsed >= MAX_BLOCK_INTERVAL_SECS
        };

        if should_create_block {
            drop(pending); // Release lock before creating block
            self.create_block().await?;
        }

        Ok(evidence.hash)
    }

    /// Create a new block from pending events
    pub async fn create_block(&self) -> Result<Block, String> {
        let mut pending = self.pending_events.write().await;
        if pending.is_empty() {
            return Err("No pending events to create block".to_string());
        }

        let events: Vec<Evidence> = pending.drain(..).collect();
        drop(pending);

        let blocks = self.blocks.read().await;
        let last_block = blocks.last().ok_or("No genesis block found")?;
        let index = last_block.index + 1;
        let previous_hash = last_block.block_hash.clone();
        drop(blocks);

        // Calculate Merkle root
        let merkle_root = Self::calculate_merkle_root(&events);

        let mut block = Block {
            index,
            timestamp: Utc::now(),
            previous_hash,
            merkle_root,
            events,
            nonce: 0,
            signature: String::new(),
            signer_public_key: hex::encode(self.verifying_key.as_bytes()),
            block_hash: String::new(),
        };

        // Calculate block hash
        block.block_hash = Self::calculate_block_hash(&block);

        // Sign the block
        let signature = self.signing_key.sign(block.block_hash.as_bytes());
        block.signature = hex::encode(signature.to_bytes());

        // Add block to chain
        let mut blocks = self.blocks.write().await;
        blocks.push(block.clone());

        // Update last block time
        let mut last_time = self.last_block_time.write().await;
        *last_time = block.timestamp;

        Ok(block)
    }

    /// Calculate hash for evidence
    fn calculate_evidence_hash(evidence: &Evidence) -> String {
        let mut hasher = Sha256::new();
        hasher.update(evidence.id.as_bytes());
        hasher.update(evidence.timestamp.to_rfc3339().as_bytes());
        hasher.update(evidence.event_type.as_bytes());
        hasher.update(evidence.resource_id.as_bytes());
        hasher.update(evidence.policy_id.as_bytes());
        hasher.update(format!("{:?}", evidence.compliance_status).as_bytes());
        hasher.update(evidence.actor.as_bytes());
        hasher.update(evidence.details.to_string().as_bytes());
        
        hex::encode(hasher.finalize())
    }

    /// Calculate Merkle root from events
    fn calculate_merkle_root(events: &[Evidence]) -> String {
        if events.is_empty() {
            return "0".repeat(64);
        }

        let mut hashes: Vec<String> = events.iter().map(|e| e.hash.clone()).collect();

        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for i in (0..hashes.len()).step_by(2) {
                let left = &hashes[i];
                let right = if i + 1 < hashes.len() {
                    &hashes[i + 1]
                } else {
                    &hashes[i]
                };

                let mut hasher = Sha256::new();
                hasher.update(left.as_bytes());
                hasher.update(right.as_bytes());
                next_level.push(hex::encode(hasher.finalize()));
            }
            
            hashes = next_level;
        }

        hashes[0].clone()
    }

    /// Calculate block hash
    fn calculate_block_hash(block: &Block) -> String {
        let mut hasher = Sha256::new();
        hasher.update(block.index.to_string().as_bytes());
        hasher.update(block.timestamp.to_rfc3339().as_bytes());
        hasher.update(block.previous_hash.as_bytes());
        hasher.update(block.merkle_root.as_bytes());
        hasher.update(block.nonce.to_string().as_bytes());
        
        hex::encode(hasher.finalize())
    }

    /// Verify the entire chain
    pub async fn verify_chain(&self) -> ChainVerificationResult {
        let blocks = self.blocks.read().await;
        let mut invalid_blocks = Vec::new();
        let mut total_events = 0;

        for (i, block) in blocks.iter().enumerate() {
            // Skip genesis block previous hash check
            if i > 0 {
                let prev_block = &blocks[i - 1];
                if block.previous_hash != prev_block.block_hash {
                    invalid_blocks.push(block.index);
                    continue;
                }
            }

            // Verify block hash
            let calculated_hash = Self::calculate_block_hash(block);
            if calculated_hash != block.block_hash {
                invalid_blocks.push(block.index);
                continue;
            }

            // Verify Merkle root
            if !block.events.is_empty() {
                let calculated_root = Self::calculate_merkle_root(&block.events);
                if calculated_root != block.merkle_root {
                    invalid_blocks.push(block.index);
                    continue;
                }
            }

            // Verify signature
            if let Ok(public_key_bytes) = hex::decode(&block.signer_public_key) {
                if let Ok(verifying_key) = VerifyingKey::from_bytes(&public_key_bytes.try_into().unwrap_or_default()) {
                    if let Ok(signature_bytes) = hex::decode(&block.signature) {
                        if let Ok(signature) = Signature::from_bytes(&signature_bytes.try_into().unwrap_or_default()) {
                            if verifying_key.verify(block.block_hash.as_bytes(), &signature).is_err() {
                                invalid_blocks.push(block.index);
                            }
                        }
                    }
                }
            }

            total_events += block.events.len();
        }

        let chain_hash = if let Some(last_block) = blocks.last() {
            last_block.block_hash.clone()
        } else {
            "0".repeat(64)
        };

        ChainVerificationResult {
            is_valid: invalid_blocks.is_empty(),
            total_blocks: blocks.len() as u64,
            total_events,
            invalid_blocks,
            verification_timestamp: Utc::now(),
            chain_hash,
        }
    }

    /// Get chain status
    pub async fn get_status(&self) -> ChainStatus {
        let blocks = self.blocks.read().await;
        let pending = self.pending_events.read().await;
        
        let total_events: usize = blocks.iter().map(|b| b.events.len()).sum();
        let first_block = blocks.first().unwrap();
        let last_block = blocks.last().unwrap();

        ChainStatus {
            total_blocks: blocks.len() as u64,
            total_events,
            last_block_timestamp: last_block.timestamp,
            last_block_hash: last_block.block_hash.clone(),
            chain_start_timestamp: first_block.timestamp,
            pending_events: pending.len(),
        }
    }

    /// Generate Merkle proof for evidence
    pub async fn generate_merkle_proof(&self, evidence_hash: &str) -> Result<MerkleProof, String> {
        let blocks = self.blocks.read().await;
        
        for block in blocks.iter() {
            if let Some(evidence) = block.events.iter().find(|e| e.hash == evidence_hash) {
                let evidence_index = block.events.iter().position(|e| e.hash == evidence_hash).unwrap();
                let merkle_path = Self::build_merkle_path(&block.events, evidence_index);
                
                return Ok(MerkleProof {
                    evidence_hash: evidence_hash.to_string(),
                    merkle_path,
                    block_index: block.index,
                    merkle_root: block.merkle_root.clone(),
                });
            }
        }
        
        Err("Evidence not found in chain".to_string())
    }

    /// Build Merkle path for proof
    fn build_merkle_path(events: &[Evidence], target_index: usize) -> Vec<MerkleNode> {
        let mut path = Vec::new();
        let mut hashes: Vec<String> = events.iter().map(|e| e.hash.clone()).collect();
        let mut index = target_index;

        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for i in (0..hashes.len()).step_by(2) {
                if i == index || i + 1 == index {
                    let sibling_index = if i == index { i + 1 } else { i };
                    if sibling_index < hashes.len() {
                        path.push(MerkleNode {
                            hash: hashes[sibling_index].clone(),
                            position: if i == index { Position::Right } else { Position::Left },
                        });
                    }
                    index = i / 2;
                }

                let left = &hashes[i];
                let right = if i + 1 < hashes.len() {
                    &hashes[i + 1]
                } else {
                    &hashes[i]
                };

                let mut hasher = Sha256::new();
                hasher.update(left.as_bytes());
                hasher.update(right.as_bytes());
                next_level.push(hex::encode(hasher.finalize()));
            }
            
            hashes = next_level;
        }

        path
    }

    /// Verify Merkle proof
    pub fn verify_merkle_proof(proof: &MerkleProof) -> bool {
        let mut current_hash = proof.evidence_hash.clone();
        
        for node in &proof.merkle_path {
            let mut hasher = Sha256::new();
            
            match node.position {
                Position::Left => {
                    hasher.update(node.hash.as_bytes());
                    hasher.update(current_hash.as_bytes());
                }
                Position::Right => {
                    hasher.update(current_hash.as_bytes());
                    hasher.update(node.hash.as_bytes());
                }
            }
            
            current_hash = hex::encode(hasher.finalize());
        }
        
        current_hash == proof.merkle_root
    }

    /// Export chain to JSON
    pub async fn export_chain(&self) -> Result<String, String> {
        let blocks = self.blocks.read().await;
        serde_json::to_string_pretty(&*blocks).map_err(|e| e.to_string())
    }

    /// Get block by index
    pub async fn get_block(&self, index: u64) -> Option<Block> {
        let blocks = self.blocks.read().await;
        blocks.iter().find(|b| b.index == index).cloned()
    }

    /// Get blocks in range
    pub async fn get_blocks_range(&self, start: u64, end: u64) -> Vec<Block> {
        let blocks = self.blocks.read().await;
        blocks
            .iter()
            .filter(|b| b.index >= start && b.index <= end)
            .cloned()
            .collect()
    }
}

/// Hex encoding/decoding module
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hash_chain_creation() {
        let chain = HashChain::new();
        let status = chain.get_status().await;
        
        assert_eq!(status.total_blocks, 1); // Genesis block
        assert_eq!(status.total_events, 0);
    }

    #[tokio::test]
    async fn test_evidence_addition() {
        let chain = HashChain::new();
        
        let evidence = Evidence {
            id: "test-001".to_string(),
            timestamp: Utc::now(),
            event_type: "PolicyCheck".to_string(),
            resource_id: "vm-001".to_string(),
            policy_id: "policy-001".to_string(),
            compliance_status: ComplianceStatus::Compliant,
            actor: "system".to_string(),
            details: serde_json::json!({"check": "success"}),
            metadata: HashMap::new(),
            hash: String::new(),
        };
        
        let hash = chain.add_evidence(evidence).await.unwrap();
        assert!(!hash.is_empty());
    }

    #[tokio::test]
    async fn test_chain_verification() {
        let chain = HashChain::new();
        
        // Add some evidence
        for i in 0..5 {
            let evidence = Evidence {
                id: format!("test-{:03}", i),
                timestamp: Utc::now(),
                event_type: "PolicyCheck".to_string(),
                resource_id: format!("vm-{:03}", i),
                policy_id: "policy-001".to_string(),
                compliance_status: ComplianceStatus::Compliant,
                actor: "system".to_string(),
                details: serde_json::json!({"check": "success"}),
                metadata: HashMap::new(),
                hash: String::new(),
            };
            
            chain.add_evidence(evidence).await.unwrap();
        }
        
        // Force block creation
        chain.create_block().await.unwrap();
        
        // Verify chain
        let result = chain.verify_chain().await;
        assert!(result.is_valid);
        assert_eq!(result.invalid_blocks.len(), 0);
    }
}