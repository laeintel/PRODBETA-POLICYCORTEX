// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

/// Evidence Pipeline for collecting, signing, and storing compliance artifacts
/// Provides cryptographically signed evidence for audits and compliance verification
pub struct EvidencePipeline {
    evidence_store: Arc<RwLock<HashMap<Uuid, Evidence>>>,
    signing_keys: Arc<RwLock<SigningKeys>>,
    verification_cache: Arc<RwLock<HashMap<String, VerificationResult>>>,
    storage_backend: StorageBackend,
    db_pool: Option<sqlx::PgPool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: Uuid,
    pub evidence_type: EvidenceType,
    pub source: EvidenceSource,
    pub subject: String,
    pub description: String,
    pub data: serde_json::Value,
    pub hash: String,
    pub signature: String,
    pub signing_key_id: String,
    pub chain_of_custody: Vec<CustodyEntry>,
    pub metadata: EvidenceMetadata,
    pub tenant_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub verification_status: VerificationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    ComplianceReport,
    AuditLog,
    SecurityScan,
    PolicyEvaluation,
    AccessReview,
    ChangeRecord,
    IncidentReport,
    TestResult,
    Certification,
    Screenshot,
    Configuration,
    Approval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub system: String,
    pub component: String,
    pub version: String,
    pub environment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustodyEntry {
    pub timestamp: DateTime<Utc>,
    pub actor: String,
    pub action: String,
    pub location: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceMetadata {
    pub compliance_frameworks: Vec<String>,
    pub controls: Vec<String>,
    pub tags: Vec<String>,
    pub severity: Option<String>,
    pub confidence: f32,
    pub automated: bool,
    pub review_required: bool,
    pub related_evidence: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerificationStatus {
    Verified,
    Failed,
    Pending,
    Expired,
    Tampered,
}

#[derive(Debug, Clone)]
struct SigningKeys {
    active_key_id: String,
    keys: HashMap<String, SigningKey>,
}

#[derive(Debug, Clone)]
struct SigningKey {
    id: String,
    algorithm: SigningAlgorithm,
    public_key: Vec<u8>,
    private_key: Vec<u8>,
    created_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    revoked: bool,
}

#[derive(Debug, Clone)]
enum SigningAlgorithm {
    RS256,
    ES256,
    EdDSA,
}

#[derive(Debug, Clone)]
struct VerificationResult {
    evidence_id: Uuid,
    status: VerificationStatus,
    verified_at: DateTime<Utc>,
    details: String,
}

#[derive(Debug, Clone)]
enum StorageBackend {
    Local(String),
    AzureBlob(AzureBlobConfig),
    S3(S3Config),
}

#[derive(Debug, Clone)]
struct AzureBlobConfig {
    account_name: String,
    container_name: String,
    sas_token: Option<String>,
}

#[derive(Debug, Clone)]
struct S3Config {
    bucket_name: String,
    region: String,
}

impl EvidencePipeline {
    pub async fn new(
        storage_backend: StorageBackend,
        db_pool: Option<sqlx::PgPool>,
    ) -> Result<Self, String> {
        let signing_keys = Self::initialize_signing_keys().await?;

        Ok(Self {
            evidence_store: Arc::new(RwLock::new(HashMap::new())),
            signing_keys: Arc::new(RwLock::new(signing_keys)),
            verification_cache: Arc::new(RwLock::new(HashMap::new())),
            storage_backend,
            db_pool,
        })
    }

    /// Collect and sign evidence
    pub async fn collect_evidence(
        &self,
        evidence_type: EvidenceType,
        subject: String,
        data: serde_json::Value,
        metadata: EvidenceMetadata,
        tenant_id: String,
        actor: String,
    ) -> Result<Evidence, String> {
        // Generate evidence ID
        let id = Uuid::new_v4();

        // Calculate hash of the data
        let data_str = serde_json::to_string(&data).map_err(|e| e.to_string())?;
        let hash = self.calculate_hash(&data_str);

        // Sign the evidence
        let signature = self.sign_evidence(&hash).await?;

        // Get signing key ID
        let signing_keys = self.signing_keys.read().await;
        let signing_key_id = signing_keys.active_key_id.clone();
        drop(signing_keys);

        // Create initial custody entry
        let custody_entry = CustodyEntry {
            timestamp: Utc::now(),
            actor: actor.clone(),
            action: "Created".to_string(),
            location: self.get_storage_location(&id),
            hash: hash.clone(),
        };

        // Create evidence record
        let evidence = Evidence {
            id,
            evidence_type,
            source: EvidenceSource {
                system: "PolicyCortex".to_string(),
                component: "EvidencePipeline".to_string(),
                version: "2.0.0".to_string(),
                environment: std::env::var("ENVIRONMENT")
                    .unwrap_or_else(|_| "production".to_string()),
            },
            subject,
            description: format!("Evidence collected at {}", Utc::now()),
            data: data.clone(),
            hash,
            signature,
            signing_key_id,
            chain_of_custody: vec![custody_entry],
            metadata,
            tenant_id,
            created_at: Utc::now(),
            expires_at: None,
            verification_status: VerificationStatus::Verified,
        };

        // Store evidence
        self.store_evidence(evidence.clone()).await?;

        // Persist to storage backend
        self.persist_to_storage(&evidence).await?;

        // Log to database if available
        if let Some(ref pool) = self.db_pool {
            self.persist_to_database(pool, &evidence).await?;
        }

        info!("Evidence {} collected and signed", evidence.id);

        Ok(evidence)
    }

    /// Verify evidence integrity
    pub async fn verify_evidence(&self, evidence_id: Uuid) -> Result<VerificationStatus, String> {
        // Check cache first
        let cache = self.verification_cache.read().await;
        if let Some(result) = cache.get(&evidence_id.to_string()) {
            if result.verified_at > Utc::now() - chrono::Duration::minutes(5) {
                return Ok(result.status.clone());
            }
        }
        drop(cache);

        // Get evidence
        let store = self.evidence_store.read().await;
        let evidence = store
            .get(&evidence_id)
            .ok_or_else(|| "Evidence not found".to_string())?
            .clone();
        drop(store);

        // Verify hash
        let data_str = serde_json::to_string(&evidence.data).map_err(|e| e.to_string())?;
        let calculated_hash = self.calculate_hash(&data_str);

        if calculated_hash != evidence.hash {
            warn!("Evidence {} hash mismatch", evidence_id);
            return Ok(VerificationStatus::Tampered);
        }

        // Verify signature
        let signature_valid = self
            .verify_signature(
                &evidence.hash,
                &evidence.signature,
                &evidence.signing_key_id,
            )
            .await?;

        if !signature_valid {
            warn!("Evidence {} signature invalid", evidence_id);
            return Ok(VerificationStatus::Failed);
        }

        // Check expiration
        if let Some(expires_at) = evidence.expires_at {
            if expires_at < Utc::now() {
                return Ok(VerificationStatus::Expired);
            }
        }

        // Verify chain of custody
        for (i, entry) in evidence.chain_of_custody.iter().enumerate() {
            if i > 0 {
                let prev_entry = &evidence.chain_of_custody[i - 1];
                if entry.timestamp < prev_entry.timestamp {
                    warn!(
                        "Evidence {} chain of custody timeline violation",
                        evidence_id
                    );
                    return Ok(VerificationStatus::Tampered);
                }
            }
        }

        // Cache verification result
        let result = VerificationResult {
            evidence_id,
            status: VerificationStatus::Verified,
            verified_at: Utc::now(),
            details: "All verification checks passed".to_string(),
        };

        let mut cache = self.verification_cache.write().await;
        cache.insert(evidence_id.to_string(), result);

        Ok(VerificationStatus::Verified)
    }

    /// Add custody entry to evidence
    pub async fn add_custody_entry(
        &self,
        evidence_id: Uuid,
        actor: String,
        action: String,
    ) -> Result<(), String> {
        let mut store = self.evidence_store.write().await;
        let evidence = store
            .get_mut(&evidence_id)
            .ok_or_else(|| "Evidence not found".to_string())?;

        // Calculate new hash including custody chain
        let custody_str =
            serde_json::to_string(&evidence.chain_of_custody).map_err(|e| e.to_string())?;
        let new_hash = self.calculate_hash(&format!("{}{}", evidence.hash, custody_str));

        let entry = CustodyEntry {
            timestamp: Utc::now(),
            actor,
            action,
            location: self.get_storage_location(&evidence_id),
            hash: new_hash,
        };

        evidence.chain_of_custody.push(entry);

        Ok(())
    }

    /// Export evidence for audit
    pub async fn export_evidence(
        &self,
        evidence_ids: Vec<Uuid>,
        format: ExportFormat,
    ) -> Result<Vec<u8>, String> {
        let mut evidence_list = Vec::new();

        let store = self.evidence_store.read().await;
        for id in evidence_ids {
            if let Some(evidence) = store.get(&id) {
                evidence_list.push(evidence.clone());
            }
        }
        drop(store);

        match format {
            ExportFormat::Json => {
                let json =
                    serde_json::to_string_pretty(&evidence_list).map_err(|e| e.to_string())?;
                Ok(json.into_bytes())
            }
            ExportFormat::Csv => {
                // TODO: Implement CSV export
                Err("CSV export not yet implemented".to_string())
            }
            ExportFormat::Pdf => {
                // TODO: Implement PDF export with signatures
                Err("PDF export not yet implemented".to_string())
            }
        }
    }

    /// Archive old evidence
    pub async fn archive_evidence(&self, older_than: DateTime<Utc>) -> Result<u32, String> {
        let mut store = self.evidence_store.write().await;
        let mut archived_count = 0;

        let evidence_to_archive: Vec<Uuid> = store
            .iter()
            .filter(|(_, e)| e.created_at < older_than)
            .map(|(id, _)| *id)
            .collect();

        for id in evidence_to_archive {
            if let Some(evidence) = store.remove(&id) {
                // Archive to long-term storage
                self.archive_to_storage(&evidence).await?;
                archived_count += 1;
            }
        }

        info!("Archived {} evidence records", archived_count);

        Ok(archived_count)
    }

    /// Search evidence by criteria
    pub async fn search_evidence(&self, criteria: SearchCriteria) -> Result<Vec<Evidence>, String> {
        let store = self.evidence_store.read().await;
        let mut results = Vec::new();

        for evidence in store.values() {
            if self.matches_criteria(evidence, &criteria) {
                results.push(evidence.clone());
            }
        }

        // Sort by creation date (newest first)
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(results)
    }

    // Helper methods

    fn calculate_hash(&self, data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn sign_evidence(&self, hash: &str) -> Result<String, String> {
        // HMAC-SHA256 signature using an environment-provided secret
        let secret = std::env::var("EVIDENCE_SIGNING_SECRET")
            .map_err(|_| "EVIDENCE_SIGNING_SECRET not configured".to_string())?;
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<sha2::Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).map_err(|e| e.to_string())?;
        mac.update(hash.as_bytes());
        let sig = mac.finalize().into_bytes();
        Ok(general_purpose::STANDARD.encode(sig))
    }

    async fn verify_signature(
        &self,
        hash: &str,
        signature: &str,
        key_id: &str,
    ) -> Result<bool, String> {
        let secret = std::env::var("EVIDENCE_SIGNING_SECRET")
            .map_err(|_| "EVIDENCE_SIGNING_SECRET not configured".to_string())?;
        let decoded = general_purpose::STANDARD
            .decode(signature)
            .map_err(|e| e.to_string())?;
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<sha2::Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).map_err(|e| e.to_string())?;
        mac.update(hash.as_bytes());
        Ok(mac.verify_slice(&decoded).is_ok())
    }

    async fn initialize_signing_keys() -> Result<SigningKeys, String> {
        // TODO: Load or generate actual signing keys
        // For now, create dummy keys

        let key_id = Uuid::new_v4().to_string();
        let key = SigningKey {
            id: key_id.clone(),
            algorithm: SigningAlgorithm::RS256,
            public_key: vec![0; 32],  // Dummy key
            private_key: vec![0; 32], // Dummy key
            created_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::days(365)),
            revoked: false,
        };

        let mut keys = HashMap::new();
        keys.insert(key_id.clone(), key);

        Ok(SigningKeys {
            active_key_id: key_id,
            keys,
        })
    }

    fn get_storage_location(&self, evidence_id: &Uuid) -> String {
        match &self.storage_backend {
            StorageBackend::Local(path) => format!("{}/{}.json", path, evidence_id),
            StorageBackend::AzureBlob(config) => {
                format!(
                    "https://{}.blob.core.windows.net/{}/{}.json",
                    config.account_name, config.container_name, evidence_id
                )
            }
            StorageBackend::S3(config) => {
                format!("s3://{}/{}.json", config.bucket_name, evidence_id)
            }
        }
    }

    async fn store_evidence(&self, evidence: Evidence) -> Result<(), String> {
        let mut store = self.evidence_store.write().await;
        store.insert(evidence.id, evidence);
        Ok(())
    }

    async fn persist_to_storage(&self, evidence: &Evidence) -> Result<(), String> {
        // TODO: Implement actual storage backend persistence
        info!("Persisting evidence {} to storage backend", evidence.id);
        Ok(())
    }

    async fn persist_to_database(
        &self,
        pool: &sqlx::PgPool,
        evidence: &Evidence,
    ) -> Result<(), String> {
        info!("Persisting evidence {} to database", evidence.id);
        sqlx::query(
            r#"INSERT INTO evidence_store (
                id, evidence_type, source, subject, description, data, hash, signature,
                signing_key_id, chain_of_custody, metadata, tenant_id, created_at, expires_at, verification_status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12, $13, $14, $15
            ) ON CONFLICT (id) DO NOTHING"#
        )
        .bind(evidence.id)
        .bind(format!("{:?}", evidence.evidence_type))
        .bind(serde_json::to_value(&evidence.source).map_err(|e| e.to_string())?)
        .bind(&evidence.subject)
        .bind(&evidence.description)
        .bind(&evidence.data)
        .bind(&evidence.hash)
        .bind(&evidence.signature)
        .bind(&evidence.signing_key_id)
        .bind(serde_json::to_value(&evidence.chain_of_custody).map_err(|e| e.to_string())?)
        .bind(serde_json::to_value(&evidence.metadata).map_err(|e| e.to_string())?)
        .bind(&evidence.tenant_id)
        .bind(evidence.created_at)
        .bind(evidence.expires_at)
        .bind(format!("{:?}", evidence.verification_status))
        .execute(pool)
        .await
        .map_err(|e| format!("Failed to persist evidence: {}", e))?;
        Ok(())
    }

    async fn archive_to_storage(&self, evidence: &Evidence) -> Result<(), String> {
        // TODO: Implement archival to long-term storage
        info!("Archiving evidence {} to long-term storage", evidence.id);
        Ok(())
    }

    fn matches_criteria(&self, evidence: &Evidence, criteria: &SearchCriteria) -> bool {
        // Check evidence type
        if let Some(ref evidence_type) = criteria.evidence_type {
            if !matches_evidence_type(&evidence.evidence_type, evidence_type) {
                return false;
            }
        }

        // Check date range
        if let Some(ref from) = criteria.from_date {
            if evidence.created_at < *from {
                return false;
            }
        }

        if let Some(ref to) = criteria.to_date {
            if evidence.created_at > *to {
                return false;
            }
        }

        // Check tenant
        if let Some(ref tenant_id) = criteria.tenant_id {
            if evidence.tenant_id != *tenant_id {
                return false;
            }
        }

        // Check subject
        if let Some(ref subject) = criteria.subject_contains {
            if !evidence.subject.contains(subject) {
                return false;
            }
        }

        // Check compliance frameworks
        if let Some(ref frameworks) = criteria.compliance_frameworks {
            if !frameworks
                .iter()
                .any(|f| evidence.metadata.compliance_frameworks.contains(f))
            {
                return false;
            }
        }

        // Check verification status
        if let Some(ref status) = criteria.verification_status {
            if evidence.verification_status != *status {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone)]
pub struct SearchCriteria {
    pub evidence_type: Option<EvidenceType>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
    pub tenant_id: Option<String>,
    pub subject_contains: Option<String>,
    pub compliance_frameworks: Option<Vec<String>>,
    pub verification_status: Option<VerificationStatus>,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Pdf,
}

fn matches_evidence_type(actual: &EvidenceType, expected: &EvidenceType) -> bool {
    std::mem::discriminant(actual) == std::mem::discriminant(expected)
}
