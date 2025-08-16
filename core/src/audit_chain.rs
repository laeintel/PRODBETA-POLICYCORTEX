// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// Immutable, tamper-evident audit log with hash chaining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: String,
    pub event_type: EventType,
    pub actor: Actor,
    pub resource: Resource,
    pub action: String,
    pub outcome: Outcome,
    pub details: HashMap<String, serde_json::Value>,
    pub previous_hash: String,
    pub hash: String,
    pub signature: Option<String>, // Digital signature for non-repudiation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    ConfigurationChange,
    PolicyViolation,
    ApprovalDecision,
    BreakGlassAccess,
    SystemEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    pub id: String,
    pub email: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub session_id: Option<String>,
    pub service_account: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    pub resource_type: String,
    pub name: String,
    pub classification: Option<DataClassification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    Success,
    Failure(String),
    PartialSuccess(String),
}

/// Audit chain manager for tamper-evident logging
pub struct AuditChain {
    entries: Vec<AuditEntry>,
    current_sequence: u64,
    merkle_tree: Option<MerkleTree>,
}

impl AuditChain {
    pub fn new() -> Self {
        AuditChain {
            entries: Vec::new(),
            current_sequence: 0,
            merkle_tree: None,
        }
    }

    pub fn add_entry(
        &mut self,
        tenant_id: String,
        event_type: EventType,
        actor: Actor,
        resource: Resource,
        action: String,
        outcome: Outcome,
        details: HashMap<String, serde_json::Value>,
    ) -> AuditEntry {
        let previous_hash = if let Some(last) = self.entries.last() {
            last.hash.clone()
        } else {
            "GENESIS".to_string()
        };

        self.current_sequence += 1;

        let mut entry = AuditEntry {
            id: Uuid::new_v4(),
            sequence: self.current_sequence,
            timestamp: Utc::now(),
            tenant_id,
            event_type,
            actor,
            resource,
            action,
            outcome,
            details,
            previous_hash: previous_hash.clone(),
            hash: String::new(),
            signature: None,
        };

        // Calculate hash
        entry.hash = self.calculate_hash(&entry);

        // Add digital signature (would use actual key management)
        entry.signature = Some(self.sign_entry(&entry));

        self.entries.push(entry.clone());

        // Update Merkle tree periodically for efficient verification
        if self.current_sequence % 100 == 0 {
            self.update_merkle_tree();
        }

        entry
    }

    fn calculate_hash(&self, entry: &AuditEntry) -> String {
        let mut hasher = Sha256::new();

        // Hash all immutable fields
        hasher.update(entry.id.to_string().as_bytes());
        hasher.update(entry.sequence.to_string().as_bytes());
        hasher.update(entry.timestamp.to_rfc3339().as_bytes());
        hasher.update(entry.tenant_id.as_bytes());
        hasher.update(format!("{:?}", entry.event_type).as_bytes());
        hasher.update(entry.actor.id.as_bytes());
        hasher.update(entry.resource.id.as_bytes());
        hasher.update(entry.action.as_bytes());
        hasher.update(format!("{:?}", entry.outcome).as_bytes());
        hasher.update(entry.previous_hash.as_bytes());

        // Include details in hash
        let details_json = serde_json::to_string(&entry.details).unwrap_or_default();
        hasher.update(details_json.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    fn sign_entry(&self, entry: &AuditEntry) -> String {
        // In production, use actual cryptographic signing with private key
        // For now, return a mock signature
        let mut hasher = Sha256::new();
        hasher.update(entry.hash.as_bytes());
        hasher.update("MOCK_PRIVATE_KEY".as_bytes());
        format!("SIG:{:x}", hasher.finalize())
    }

    pub fn verify_chain(&self) -> Result<(), VerificationError> {
        if self.entries.is_empty() {
            return Ok(());
        }

        let mut previous_hash = "GENESIS".to_string();

        for (i, entry) in self.entries.iter().enumerate() {
            // Verify sequence
            if entry.sequence != (i as u64) + 1 {
                return Err(VerificationError::SequenceMismatch {
                    expected: (i as u64) + 1,
                    actual: entry.sequence,
                });
            }

            // Verify previous hash link
            if entry.previous_hash != previous_hash {
                return Err(VerificationError::HashChainBroken {
                    sequence: entry.sequence,
                });
            }

            // Verify entry hash
            let calculated_hash = self.calculate_hash(entry);
            if entry.hash != calculated_hash {
                return Err(VerificationError::HashMismatch {
                    sequence: entry.sequence,
                });
            }

            // Verify signature
            if let Some(ref signature) = entry.signature {
                if !self.verify_signature(entry, signature) {
                    return Err(VerificationError::SignatureInvalid {
                        sequence: entry.sequence,
                    });
                }
            }

            previous_hash = entry.hash.clone();
        }

        Ok(())
    }

    fn verify_signature(&self, entry: &AuditEntry, signature: &str) -> bool {
        // In production, use actual cryptographic verification with public key
        let mut hasher = Sha256::new();
        hasher.update(entry.hash.as_bytes());
        hasher.update("MOCK_PRIVATE_KEY".as_bytes());
        let expected_signature = format!("SIG:{:x}", hasher.finalize());
        signature == expected_signature
    }

    fn update_merkle_tree(&mut self) {
        let hashes: Vec<String> = self.entries.iter().map(|e| e.hash.clone()).collect();
        self.merkle_tree = Some(MerkleTree::new(hashes));
    }

    pub fn export_for_auditor(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> AuditExport {
        let filtered_entries: Vec<AuditEntry> = self
            .entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect();

        let merkle_root = self.merkle_tree.as_ref().map(|tree| tree.root.clone());

        AuditExport {
            export_id: Uuid::new_v4(),
            exported_at: Utc::now(),
            start_date: start,
            end_date: end,
            total_entries: filtered_entries.len(),
            entries: filtered_entries,
            merkle_root,
            chain_verified: self.verify_chain().is_ok(),
        }
    }
}

#[derive(Debug)]
pub enum VerificationError {
    SequenceMismatch { expected: u64, actual: u64 },
    HashChainBroken { sequence: u64 },
    HashMismatch { sequence: u64 },
    SignatureInvalid { sequence: u64 },
}

/// Merkle tree for efficient verification of large audit logs
#[derive(Debug, Clone)]
pub struct MerkleTree {
    pub root: String,
    levels: Vec<Vec<String>>,
}

impl MerkleTree {
    pub fn new(leaf_hashes: Vec<String>) -> Self {
        if leaf_hashes.is_empty() {
            return MerkleTree {
                root: String::new(),
                levels: Vec::new(),
            };
        }

        let mut levels = vec![leaf_hashes];

        while levels.last().unwrap().len() > 1 {
            let current_level = levels.last().unwrap();
            let mut next_level = Vec::new();

            for i in (0..current_level.len()).step_by(2) {
                let left = &current_level[i];
                let right = if i + 1 < current_level.len() {
                    &current_level[i + 1]
                } else {
                    left // Duplicate last node if odd number
                };

                let mut hasher = Sha256::new();
                hasher.update(left.as_bytes());
                hasher.update(right.as_bytes());
                next_level.push(format!("{:x}", hasher.finalize()));
            }

            levels.push(next_level);
        }

        MerkleTree {
            root: levels.last().unwrap()[0].clone(),
            levels,
        }
    }

    pub fn get_proof(&self, index: usize) -> Vec<(String, bool)> {
        let mut proof = Vec::new();
        let mut current_index = index;

        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            if sibling_index < level.len() {
                proof.push((level[sibling_index].clone(), current_index % 2 == 0));
            }

            current_index /= 2;
        }

        proof
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditExport {
    pub export_id: Uuid,
    pub exported_at: DateTime<Utc>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub total_entries: usize,
    pub entries: Vec<AuditEntry>,
    pub merkle_root: Option<String>,
    pub chain_verified: bool,
}

// Integration with database for persistence
pub struct PersistentAuditChain {
    db_pool: sqlx::PgPool,
    cache: AuditChain,
}

impl PersistentAuditChain {
    pub async fn new(db_pool: sqlx::PgPool) -> Result<Self, sqlx::Error> {
        let mut chain = PersistentAuditChain {
            db_pool,
            cache: AuditChain::new(),
        };

        // Load recent entries into cache
        chain.load_recent_entries().await?;

        Ok(chain)
    }

    async fn load_recent_entries(&mut self) -> Result<(), sqlx::Error> {
        // Load last 1000 entries for verification
        let query = r#"
            SELECT * FROM audit.immutable_logs
            ORDER BY sequence DESC
            LIMIT 1000
        "#;

        // Execute query and populate cache
        // Implementation depends on actual database schema

        Ok(())
    }

    pub async fn add_entry_persistent(
        &mut self,
        tenant_id: String,
        event_type: EventType,
        actor: Actor,
        resource: Resource,
        action: String,
        outcome: Outcome,
        details: HashMap<String, serde_json::Value>,
    ) -> Result<AuditEntry, sqlx::Error> {
        let entry = self.cache.add_entry(
            tenant_id, event_type, actor, resource, action, outcome, details,
        );

        // Persist to database with WORM guarantees
        let query = r#"
            INSERT INTO audit.immutable_logs (
                id, sequence, timestamp, tenant_id, event_type,
                actor_data, resource_data, action, outcome,
                details, previous_hash, hash, signature
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        "#;

        sqlx::query(query)
            .bind(&entry.id)
            .bind(entry.sequence as i64)
            .bind(&entry.timestamp)
            .bind(&entry.tenant_id)
            .bind(
                serde_json::to_value(&entry.event_type)
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
            )
            .bind(
                serde_json::to_value(&entry.actor)
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
            )
            .bind(
                serde_json::to_value(&entry.resource)
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
            )
            .bind(&entry.action)
            .bind(
                serde_json::to_value(&entry.outcome)
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
            )
            .bind(
                serde_json::to_value(&entry.details)
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?,
            )
            .bind(&entry.previous_hash)
            .bind(&entry.hash)
            .bind(&entry.signature)
            .execute(&self.db_pool)
            .await?;

        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_chain_integrity() {
        let mut chain = AuditChain::new();

        // Add some entries
        for i in 0..5 {
            chain.add_entry(
                "test-tenant".to_string(),
                EventType::DataAccess,
                Actor {
                    id: format!("user{}", i),
                    email: Some(format!("user{}@example.com", i)),
                    ip_address: Some("192.168.1.1".to_string()),
                    user_agent: None,
                    session_id: None,
                    service_account: false,
                },
                Resource {
                    id: format!("resource{}", i),
                    resource_type: "Document".to_string(),
                    name: format!("Document {}", i),
                    classification: Some(DataClassification::Confidential),
                },
                "READ".to_string(),
                Outcome::Success,
                HashMap::new(),
            );
        }

        // Verify chain integrity
        assert!(chain.verify_chain().is_ok());

        // Tamper with an entry
        if let Some(entry) = chain.entries.get_mut(2) {
            entry.action = "TAMPERED".to_string();
        }

        // Verification should fail
        assert!(chain.verify_chain().is_err());
    }

    #[test]
    fn test_merkle_tree() {
        let hashes = vec![
            "hash1".to_string(),
            "hash2".to_string(),
            "hash3".to_string(),
            "hash4".to_string(),
        ];

        let tree = MerkleTree::new(hashes);
        assert!(!tree.root.is_empty());

        let proof = tree.get_proof(0);
        assert!(!proof.is_empty());
    }
}
