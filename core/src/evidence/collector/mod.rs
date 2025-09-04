use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;

/// Evidence severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum EvidenceSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Evidence source types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    PolicyEngine,
    ComplianceScanner,
    SecurityCenter,
    CostOptimizer,
    AnomalyDetector,
    ManualReview,
    AutomatedCheck,
}

/// Cryptographic evidence metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoMetadata {
    pub signature: String,
    pub public_key: String,
    pub algorithm: String,
    pub ntp_timestamp: Option<String>,
}

/// Evidence record structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub source: EvidenceSource,
    pub severity: EvidenceSeverity,
    pub resource_id: String,
    pub resource_type: String,
    pub subscription_id: String,
    pub tenant_id: String,
    pub policy_id: String,
    pub policy_name: String,
    pub check_result: CheckResult,
    pub evidence_data: HashMap<String, serde_json::Value>,
    pub hash: String,
    pub previous_hash: Option<String>,
    pub crypto_metadata: Option<CryptoMetadata>,
    pub immutable: bool,
    pub block_height: Option<u64>,
}

/// Check result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub compliant: bool,
    pub message: String,
    pub details: HashMap<String, serde_json::Value>,
    pub recommendations: Vec<String>,
    pub evidence_artifacts: Vec<EvidenceArtifact>,
}

/// Evidence artifact (screenshots, logs, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceArtifact {
    pub artifact_type: String,
    pub content_hash: String,
    pub storage_url: String,
    pub size_bytes: u64,
    pub mime_type: String,
}

/// Evidence collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    pub batch_size: usize,
    pub flush_interval_secs: u64,
    pub enable_ntp_sync: bool,
    pub enable_crypto_signing: bool,
    pub retention_days: u32,
    pub compression_enabled: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        CollectorConfig {
            batch_size: 100,
            flush_interval_secs: 60,
            enable_ntp_sync: true,
            enable_crypto_signing: true,
            retention_days: 2557, // 7 years for compliance
            compression_enabled: true,
        }
    }
}

/// Main Evidence Collector service
pub struct EvidenceCollector {
    config: CollectorConfig,
    buffer: Arc<RwLock<Vec<EvidenceRecord>>>,
    keypair: Option<Keypair>,
    last_hash: Arc<RwLock<String>>,
    block_height: Arc<RwLock<u64>>,
}

impl EvidenceCollector {
    /// Create new evidence collector
    pub fn new(config: CollectorConfig) -> Self {
        let keypair = if config.enable_crypto_signing {
            let mut csprng = OsRng {};
            Some(Keypair::generate(&mut csprng))
        } else {
            None
        };

        EvidenceCollector {
            config,
            buffer: Arc::new(RwLock::new(Vec::new())),
            keypair,
            last_hash: Arc::new(RwLock::new(String::from("genesis"))),
            block_height: Arc::new(RwLock::new(0)),
        }
    }

    /// Collect evidence from a policy check
    pub async fn collect_evidence(
        &self,
        source: EvidenceSource,
        severity: EvidenceSeverity,
        resource_id: String,
        resource_type: String,
        subscription_id: String,
        tenant_id: String,
        policy_id: String,
        policy_name: String,
        check_result: CheckResult,
        evidence_data: HashMap<String, serde_json::Value>,
    ) -> Result<EvidenceRecord, String> {
        // Get timestamp with NTP verification if enabled
        let timestamp = if self.config.enable_ntp_sync {
            self.get_ntp_timestamp().await?
        } else {
            Utc::now()
        };

        // Get previous hash
        let previous_hash = {
            let hash = self.last_hash.read().await;
            Some(hash.clone())
        };

        // Create evidence record
        let mut record = EvidenceRecord {
            id: Uuid::new_v4(),
            timestamp,
            source,
            severity,
            resource_id,
            resource_type,
            subscription_id,
            tenant_id,
            policy_id,
            policy_name,
            check_result,
            evidence_data,
            hash: String::new(),
            previous_hash,
            crypto_metadata: None,
            immutable: true,
            block_height: None,
        };

        // Calculate hash
        record.hash = self.calculate_hash(&record)?;

        // Sign if crypto signing is enabled
        if self.config.enable_crypto_signing {
            record.crypto_metadata = Some(self.sign_evidence(&record)?);
        }

        // Update last hash
        {
            let mut hash = self.last_hash.write().await;
            *hash = record.hash.clone();
        }

        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            buffer.push(record.clone());

            // Check if we should flush
            if buffer.len() >= self.config.batch_size {
                self.flush_buffer().await?;
            }
        }

        Ok(record)
    }

    /// Calculate SHA3-256 hash of evidence
    fn calculate_hash(&self, record: &EvidenceRecord) -> Result<String, String> {
        let mut hasher = Sha3_256::new();
        
        // Hash key fields in deterministic order
        hasher.update(record.id.to_string().as_bytes());
        hasher.update(record.timestamp.to_rfc3339().as_bytes());
        hasher.update(format!("{:?}", record.source).as_bytes());
        hasher.update(format!("{:?}", record.severity).as_bytes());
        hasher.update(record.resource_id.as_bytes());
        hasher.update(record.policy_id.as_bytes());
        hasher.update(format!("{}", record.check_result.compliant).as_bytes());
        
        if let Some(ref prev_hash) = record.previous_hash {
            hasher.update(prev_hash.as_bytes());
        }

        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Sign evidence with Ed25519
    fn sign_evidence(&self, record: &EvidenceRecord) -> Result<CryptoMetadata, String> {
        let keypair = self.keypair.as_ref()
            .ok_or_else(|| "Crypto signing not enabled".to_string())?;

        let message = format!("{}{}", record.hash, record.timestamp.to_rfc3339());
        let signature = keypair.sign(message.as_bytes());

        Ok(CryptoMetadata {
            signature: hex::encode(signature.to_bytes()),
            public_key: hex::encode(keypair.public.to_bytes()),
            algorithm: "Ed25519".to_string(),
            ntp_timestamp: Some(record.timestamp.to_rfc3339()),
        })
    }

    /// Get NTP-verified timestamp
    async fn get_ntp_timestamp(&self) -> Result<DateTime<Utc>, String> {
        // In production, this would query NTP servers
        // For MVP, we'll use system time with a verification flag
        Ok(Utc::now())
    }

    /// Flush evidence buffer to storage
    async fn flush_buffer(&self) -> Result<(), String> {
        let mut buffer = self.buffer.write().await;
        
        if buffer.is_empty() {
            return Ok(());
        }

        // Increment block height
        let block_height = {
            let mut height = self.block_height.write().await;
            *height += 1;
            *height
        };

        // Update block height for all records
        for record in buffer.iter_mut() {
            record.block_height = Some(block_height);
        }

        // Here we would persist to append-only storage
        // For now, we'll log the action
        tracing::info!(
            "Flushing {} evidence records to block {}",
            buffer.len(),
            block_height
        );

        // Clear buffer after successful flush
        buffer.clear();
        
        Ok(())
    }

    /// Verify evidence integrity
    pub async fn verify_evidence(&self, record: &EvidenceRecord) -> Result<bool, String> {
        // Verify hash
        let calculated_hash = self.calculate_hash(record)?;
        if calculated_hash != record.hash {
            return Ok(false);
        }

        // Verify signature if present
        if let Some(ref crypto_meta) = record.crypto_metadata {
            if self.config.enable_crypto_signing {
                return self.verify_signature(record, crypto_meta);
            }
        }

        Ok(true)
    }

    /// Verify cryptographic signature
    fn verify_signature(
        &self,
        record: &EvidenceRecord,
        crypto_meta: &CryptoMetadata,
    ) -> Result<bool, String> {
        let public_key_bytes = hex::decode(&crypto_meta.public_key)
            .map_err(|e| format!("Failed to decode public key: {}", e))?;
        
        let public_key = PublicKey::from_bytes(&public_key_bytes)
            .map_err(|e| format!("Invalid public key: {}", e))?;
        
        let signature_bytes = hex::decode(&crypto_meta.signature)
            .map_err(|e| format!("Failed to decode signature: {}", e))?;
        
        let signature = Signature::from_bytes(&signature_bytes)
            .map_err(|e| format!("Invalid signature: {}", e))?;
        
        let message = format!("{}{}", record.hash, record.timestamp.to_rfc3339());
        
        Ok(public_key.verify(message.as_bytes(), &signature).is_ok())
    }

    /// Get collector statistics
    pub async fn get_stats(&self) -> CollectorStats {
        let buffer_size = self.buffer.read().await.len();
        let current_block = *self.block_height.read().await;

        CollectorStats {
            buffer_size,
            current_block_height: current_block,
            crypto_enabled: self.config.enable_crypto_signing,
            ntp_enabled: self.config.enable_ntp_sync,
        }
    }
}

/// Collector statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorStats {
    pub buffer_size: usize,
    pub current_block_height: u64,
    pub crypto_enabled: bool,
    pub ntp_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evidence_collection() {
        let config = CollectorConfig::default();
        let collector = EvidenceCollector::new(config);

        let mut evidence_data = HashMap::new();
        evidence_data.insert("test_key".to_string(), serde_json::json!("test_value"));

        let check_result = CheckResult {
            compliant: true,
            message: "Test check passed".to_string(),
            details: HashMap::new(),
            recommendations: vec![],
            evidence_artifacts: vec![],
        };

        let record = collector
            .collect_evidence(
                EvidenceSource::PolicyEngine,
                EvidenceSeverity::Low,
                "resource-123".to_string(),
                "VirtualMachine".to_string(),
                "sub-123".to_string(),
                "tenant-123".to_string(),
                "policy-123".to_string(),
                "Test Policy".to_string(),
                check_result,
                evidence_data,
            )
            .await
            .unwrap();

        assert!(!record.hash.is_empty());
        assert!(record.immutable);
        assert!(collector.verify_evidence(&record).await.unwrap());
    }

    #[tokio::test]
    async fn test_hash_chain() {
        let config = CollectorConfig::default();
        let collector = EvidenceCollector::new(config);

        let check_result = CheckResult {
            compliant: true,
            message: "Test".to_string(),
            details: HashMap::new(),
            recommendations: vec![],
            evidence_artifacts: vec![],
        };

        // Collect first evidence
        let record1 = collector
            .collect_evidence(
                EvidenceSource::PolicyEngine,
                EvidenceSeverity::Low,
                "res-1".to_string(),
                "VM".to_string(),
                "sub-1".to_string(),
                "tenant-1".to_string(),
                "pol-1".to_string(),
                "Policy 1".to_string(),
                check_result.clone(),
                HashMap::new(),
            )
            .await
            .unwrap();

        // Collect second evidence
        let record2 = collector
            .collect_evidence(
                EvidenceSource::PolicyEngine,
                EvidenceSeverity::Low,
                "res-2".to_string(),
                "VM".to_string(),
                "sub-1".to_string(),
                "tenant-1".to_string(),
                "pol-2".to_string(),
                "Policy 2".to_string(),
                check_result,
                HashMap::new(),
            )
            .await
            .unwrap();

        // Verify chain linkage
        assert_eq!(record2.previous_hash, Some(record1.hash.clone()));
    }
}