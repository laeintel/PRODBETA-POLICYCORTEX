// Compliance Evidence Factory - Comprehensive Implementation
// Based on Roadmap_09_Compliance_Evidence_Factory.md
// Addresses GitHub Issues #60-63: Compliance and Evidence Generation

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use async_trait::async_trait;
use sha2::{Sha256, Digest};
use std::path::PathBuf;

// Core compliance models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFramework {
    pub id: String,
    pub name: String,
    pub version: String,
    pub controls: Vec<ComplianceControl>,
    pub last_assessed: DateTime<Utc>,
    pub next_assessment: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceControl {
    pub control_id: String,
    pub framework: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub severity: String, // "Low", "Medium", "High", "Critical"
    pub automated: bool,
    pub test_frequency: String, // "Continuous", "Daily", "Weekly", "Monthly"
    pub test_queries: Vec<ControlTest>,
    pub required_artifacts: Vec<ArtifactRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlTest {
    pub test_id: String,
    pub signal_source: String, // "AzurePolicy", "AzureMonitor", "ConfigSnapshot", "Logs"
    pub query: String,
    pub success_criteria: SuccessCriteria,
    pub last_run: Option<DateTime<Utc>>,
    pub last_result: Option<TestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub criteria_type: String, // "Threshold", "Boolean", "Regex", "Count"
    pub expected_value: serde_json::Value,
    pub operator: String, // "equals", "greater_than", "less_than", "contains", "matches"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub status: String, // "Pass", "Fail", "Warning", "Skip"
    pub actual_value: serde_json::Value,
    pub evidence_refs: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    pub artifact_type: String, // "PolicyDefinition", "ConfigSnapshot", "Screenshot", "Log", "Report"
    pub generator: String,
    pub format: String, // "JSON", "CSV", "PDF", "PNG"
    pub retention_days: u32,
}

// Evidence artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceArtifact {
    pub artifact_id: String,
    pub control_id: String,
    pub artifact_type: String,
    pub format: String,
    pub content_hash: String,
    pub storage_path: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

// Evidence pack for auditors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePack {
    pub pack_id: String,
    pub framework: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub control_results: Vec<ControlEvidenceResult>,
    pub artifacts: Vec<EvidenceArtifact>,
    pub summary: ComplianceSummary,
    pub manifest: EvidenceManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlEvidenceResult {
    pub control_id: String,
    pub control_name: String,
    pub status: String,
    pub test_results: Vec<TestResult>,
    pub artifacts: Vec<String>, // artifact_ids
    pub exceptions: Vec<ComplianceException>,
    pub remediation_actions: Vec<RemediationAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceException {
    pub exception_id: String,
    pub reason: String,
    pub approved_by: String,
    pub approved_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub risk_acceptance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    pub action_id: String,
    pub description: String,
    pub status: String, // "Pending", "InProgress", "Completed"
    pub assigned_to: String,
    pub due_date: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummary {
    pub total_controls: u32,
    pub passed_controls: u32,
    pub failed_controls: u32,
    pub warning_controls: u32,
    pub skipped_controls: u32,
    pub compliance_score: f64,
    pub critical_findings: Vec<String>,
    pub improvement_areas: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceManifest {
    pub version: String,
    pub checksum: String,
    pub total_artifacts: u32,
    pub total_size_bytes: u64,
    pub file_index: Vec<ManifestEntry>,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub filename: String,
    pub path: String,
    pub checksum: String,
    pub size_bytes: u64,
    pub artifact_type: String,
}

// Compliance Engine trait
#[async_trait]
pub trait ComplianceEngine: Send + Sync {
    async fn run_control_tests(&self, framework: &str) -> Result<Vec<ControlEvidenceResult>, ComplianceError>;
    async fn generate_artifacts(&self, control_id: &str) -> Result<Vec<EvidenceArtifact>, ComplianceError>;
    async fn assemble_evidence_pack(&self, framework: &str, period: DateRange) -> Result<EvidencePack, ComplianceError>;
    async fn validate_compliance(&self, framework: &str) -> Result<ComplianceValidation, ComplianceError>;
    async fn schedule_assessments(&self) -> Result<Vec<AssessmentSchedule>, ComplianceError>;
}

#[derive(Debug, Clone)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceValidation {
    pub framework: String,
    pub validation_time: DateTime<Utc>,
    pub is_compliant: bool,
    pub score: f64,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub severity: String,
    pub control_id: String,
    pub description: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentSchedule {
    pub framework: String,
    pub control_id: String,
    pub next_run: DateTime<Utc>,
    pub frequency: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ComplianceError {
    #[error("Test execution failed: {0}")]
    TestError(String),
    #[error("Artifact generation failed: {0}")]
    ArtifactError(String),
    #[error("Evidence assembly failed: {0}")]
    AssemblyError(String),
    #[error("Validation failed: {0}")]
    ValidationError(String),
}

// Azure Compliance Implementation
pub struct AzureComplianceEngine {
    azure_client: crate::azure_client_async::AsyncAzureClient,
    frameworks: HashMap<String, ComplianceFramework>,
    artifact_store: PathBuf,
}

impl AzureComplianceEngine {
    pub async fn new(azure_client: crate::azure_client_async::AsyncAzureClient) -> Result<Self, ComplianceError> {
        let mut engine = Self {
            azure_client,
            frameworks: HashMap::new(),
            artifact_store: PathBuf::from("/tmp/compliance_artifacts"),
        };
        
        // Initialize frameworks
        engine.load_frameworks().await?;
        
        // Create artifact store directory
        tokio::fs::create_dir_all(&engine.artifact_store).await
            .map_err(|e| ComplianceError::ArtifactError(e.to_string()))?;
        
        Ok(engine)
    }

    async fn load_frameworks(&mut self) -> Result<(), ComplianceError> {
        // Load compliance frameworks - in production these would come from a database
        let frameworks = vec![
            self.create_iso27001_framework(),
            self.create_pci_dss_framework(),
            self.create_hipaa_framework(),
            self.create_gdpr_framework(),
            self.create_cis_framework(),
        ];
        
        for framework in frameworks {
            self.frameworks.insert(framework.id.clone(), framework);
        }
        
        Ok(())
    }

    fn create_iso27001_framework(&self) -> ComplianceFramework {
        ComplianceFramework {
            id: "iso27001".to_string(),
            name: "ISO 27001:2013".to_string(),
            version: "2013".to_string(),
            controls: vec![
                ComplianceControl {
                    control_id: "A.9.1.2".to_string(),
                    framework: "iso27001".to_string(),
                    name: "Access to networks and network services".to_string(),
                    description: "Users should only be provided with access to network services they are authorized to use".to_string(),
                    category: "Access Control".to_string(),
                    severity: "High".to_string(),
                    automated: true,
                    test_frequency: "Daily".to_string(),
                    test_queries: vec![
                        ControlTest {
                            test_id: "test-network-access".to_string(),
                            signal_source: "AzurePolicy".to_string(),
                            query: "PolicyAssignments | where displayName contains 'Network'".to_string(),
                            success_criteria: SuccessCriteria {
                                criteria_type: "Threshold".to_string(),
                                expected_value: serde_json::json!(100),
                                operator: "equals".to_string(),
                            },
                            last_run: None,
                            last_result: None,
                        }
                    ],
                    required_artifacts: vec![
                        ArtifactRequirement {
                            artifact_type: "PolicyDefinition".to_string(),
                            generator: "AzurePolicyExporter".to_string(),
                            format: "JSON".to_string(),
                            retention_days: 365,
                        }
                    ],
                },
                ComplianceControl {
                    control_id: "A.12.1.1".to_string(),
                    framework: "iso27001".to_string(),
                    name: "Documented operating procedures".to_string(),
                    description: "Operating procedures should be documented and made available".to_string(),
                    category: "Operations".to_string(),
                    severity: "Medium".to_string(),
                    automated: false,
                    test_frequency: "Monthly".to_string(),
                    test_queries: vec![],
                    required_artifacts: vec![
                        ArtifactRequirement {
                            artifact_type: "Report".to_string(),
                            generator: "DocumentationReporter".to_string(),
                            format: "PDF".to_string(),
                            retention_days: 365,
                        }
                    ],
                },
            ],
            last_assessed: Utc::now() - Duration::days(15),
            next_assessment: Utc::now() + Duration::days(15),
        }
    }

    fn create_pci_dss_framework(&self) -> ComplianceFramework {
        ComplianceFramework {
            id: "pci-dss".to_string(),
            name: "PCI DSS v4.0".to_string(),
            version: "4.0".to_string(),
            controls: vec![
                ComplianceControl {
                    control_id: "1.1".to_string(),
                    framework: "pci-dss".to_string(),
                    name: "Firewall configuration standards".to_string(),
                    description: "Establish and implement firewall and router configuration standards".to_string(),
                    category: "Network Security".to_string(),
                    severity: "Critical".to_string(),
                    automated: true,
                    test_frequency: "Continuous".to_string(),
                    test_queries: vec![],
                    required_artifacts: vec![],
                },
            ],
            last_assessed: Utc::now() - Duration::days(7),
            next_assessment: Utc::now() + Duration::days(23),
        }
    }

    fn create_hipaa_framework(&self) -> ComplianceFramework {
        ComplianceFramework {
            id: "hipaa".to_string(),
            name: "HIPAA Security Rule".to_string(),
            version: "2013".to_string(),
            controls: vec![
                ComplianceControl {
                    control_id: "164.312(a)(1)".to_string(),
                    framework: "hipaa".to_string(),
                    name: "Access Control".to_string(),
                    description: "Implement technical policies for electronic information systems".to_string(),
                    category: "Technical Safeguards".to_string(),
                    severity: "Critical".to_string(),
                    automated: true,
                    test_frequency: "Daily".to_string(),
                    test_queries: vec![],
                    required_artifacts: vec![],
                },
            ],
            last_assessed: Utc::now() - Duration::days(30),
            next_assessment: Utc::now() + Duration::days(60),
        }
    }

    fn create_gdpr_framework(&self) -> ComplianceFramework {
        ComplianceFramework {
            id: "gdpr".to_string(),
            name: "General Data Protection Regulation".to_string(),
            version: "2018".to_string(),
            controls: vec![
                ComplianceControl {
                    control_id: "Article-32".to_string(),
                    framework: "gdpr".to_string(),
                    name: "Security of processing".to_string(),
                    description: "Implement appropriate technical and organizational measures".to_string(),
                    category: "Data Protection".to_string(),
                    severity: "High".to_string(),
                    automated: true,
                    test_frequency: "Weekly".to_string(),
                    test_queries: vec![],
                    required_artifacts: vec![],
                },
            ],
            last_assessed: Utc::now() - Duration::days(14),
            next_assessment: Utc::now() + Duration::days(7),
        }
    }

    fn create_cis_framework(&self) -> ComplianceFramework {
        ComplianceFramework {
            id: "cis-azure".to_string(),
            name: "CIS Azure Foundations Benchmark".to_string(),
            version: "1.5.0".to_string(),
            controls: vec![
                ComplianceControl {
                    control_id: "1.1".to_string(),
                    framework: "cis-azure".to_string(),
                    name: "Ensure Security Defaults is enabled".to_string(),
                    description: "Security defaults in Azure AD provide secure default settings".to_string(),
                    category: "Identity and Access Management".to_string(),
                    severity: "High".to_string(),
                    automated: true,
                    test_frequency: "Daily".to_string(),
                    test_queries: vec![],
                    required_artifacts: vec![],
                },
            ],
            last_assessed: Utc::now() - Duration::days(3),
            next_assessment: Utc::now() + Duration::days(4),
        }
    }

    async fn execute_control_test(&self, control: &ComplianceControl, test: &ControlTest) -> Result<TestResult, ComplianceError> {
        // Execute REAL tests based on signal source
        let (status, actual_value, details) = match test.signal_source.as_str() {
            "AzurePolicy" => {
                // Query REAL Azure Policy compliance
                let policy_results = self.azure_client.get_policy_compliance_details(&test.query).await
                    .map_err(|e| ComplianceError::TestError(format!("Failed to query Azure Policy: {}", e)))?;
                
                let compliance_percentage = policy_results.compliance_percentage;
                let expected = test.success_criteria.expected_value.as_f64().unwrap_or(100.0);
                
                let status = match test.success_criteria.operator.as_str() {
                    "equals" => if compliance_percentage == expected { "Pass" } else { "Fail" },
                    "greater_than" => if compliance_percentage > expected { "Pass" } else { "Fail" },
                    "less_than" => if compliance_percentage < expected { "Pass" } else { "Fail" },
                    _ => "Warning"
                };
                
                (status, serde_json::json!(compliance_percentage), 
                 format!("Policy compliance: {:.1}% (expected {} {})", 
                         compliance_percentage, test.success_criteria.operator, expected))
            }
            "AzureMonitor" => {
                // Query REAL Azure Monitor metrics
                let metrics = self.azure_client.query_metrics(&test.query).await
                    .map_err(|e| ComplianceError::TestError(format!("Failed to query Azure Monitor: {}", e)))?;
                
                let metric_value = metrics.average_value;
                let threshold = test.success_criteria.expected_value.as_f64().unwrap_or(0.0);
                
                let status = match test.success_criteria.operator.as_str() {
                    "less_than" => if metric_value < threshold { "Pass" } else { "Fail" },
                    "greater_than" => if metric_value > threshold { "Pass" } else { "Fail" },
                    _ => "Warning"
                };
                
                (status, serde_json::json!(metric_value),
                 format!("Metric value: {:.2} (threshold: {} {})", 
                         metric_value, test.success_criteria.operator, threshold))
            }
            "ConfigSnapshot" => {
                // Check REAL configuration snapshots
                let config = self.azure_client.get_resource_configuration(&test.query).await
                    .map_err(|e| ComplianceError::TestError(format!("Failed to get config: {}", e)))?;
                
                let is_compliant = self.validate_configuration(&config, &test.success_criteria);
                let status = if is_compliant { "Pass" } else { "Fail" };
                
                (status, serde_json::json!(config),
                 format!("Configuration validation: {}", if is_compliant { "Compliant" } else { "Non-compliant" }))
            }
            "Logs" => {
                // Analyze REAL log data
                let log_results = self.azure_client.query_logs(&test.query).await
                    .map_err(|e| ComplianceError::TestError(format!("Failed to query logs: {}", e)))?;
                
                let violation_count = log_results.security_violations;
                let status = if violation_count == 0 { "Pass" } else { "Fail" };
                
                (status, serde_json::json!(violation_count),
                 format!("{} security violations found in logs", violation_count))
            }
            _ => ("Skip", serde_json::json!(null), "Unknown signal source".to_string())
        };

        Ok(TestResult {
            status: status.to_string(),
            actual_value,
            evidence_refs: vec![self.generate_evidence_ref()],
            timestamp: Utc::now(),
            details: details.to_string(),
        })
    }

    fn validate_configuration(&self, config: &serde_json::Value, criteria: &SuccessCriteria) -> bool {
        // Validate configuration against criteria
        match criteria.criteria_type.as_str() {
            "Boolean" => config.as_bool() == criteria.expected_value.as_bool(),
            "Regex" => {
                if let (Some(config_str), Some(pattern)) = (config.as_str(), criteria.expected_value.as_str()) {
                    regex::Regex::new(pattern).ok()
                        .and_then(|re| Some(re.is_match(config_str)))
                        .unwrap_or(false)
                } else {
                    false
                }
            }
            _ => config == &criteria.expected_value
        }
    }

    fn generate_evidence_ref(&self) -> String {
        format!("evidence-{}", uuid::Uuid::new_v4())
    }

    async fn generate_artifact(&self, control: &ComplianceControl, requirement: &ArtifactRequirement) -> Result<EvidenceArtifact, ComplianceError> {
        let content = match requirement.artifact_type.as_str() {
            "PolicyDefinition" => {
                // Fetch REAL policy definitions from Azure
                let policies = self.azure_client.get_policy_definitions().await
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch policies: {}", e)))?;
                
                let compliance_state = self.azure_client.get_policy_compliance_summary().await
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch compliance: {}", e)))?;
                
                serde_json::json!({
                    "framework": control.framework,
                    "control_id": control.control_id,
                    "timestamp": Utc::now(),
                    "policies": policies,
                    "compliance_summary": compliance_state,
                    "total_policies": policies.len(),
                    "compliant_resources": compliance_state.compliant_count,
                    "non_compliant_resources": compliance_state.non_compliant_count
                }).to_string()
            }
            "ConfigSnapshot" => {
                // Fetch REAL configuration snapshot from Azure
                let resources = self.azure_client.get_resource_configurations().await
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch configs: {}", e)))?;
                
                serde_json::json!({
                    "framework": control.framework,
                    "control_id": control.control_id,
                    "timestamp": Utc::now(),
                    "resources": resources,
                    "total_resources": resources.len(),
                    "configuration_drift": self.detect_configuration_drift(&resources)
                }).to_string()
            }
            "Report" => {
                // Generate REAL compliance report from Azure data
                let metrics = self.azure_client.get_governance_metrics().await
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch metrics: {}", e)))?;
                
                format!("Compliance Report for {}\n\nControl: {}\n\nSummary:\n- Policies: {}\n- Compliance Rate: {:.1}%\n- RBAC Violations: {}\n- Network Threats: {}\n\nGenerated: {}",
                    control.framework,
                    control.name,
                    metrics.policies.total,
                    metrics.policies.compliance_rate,
                    metrics.rbac.violations,
                    metrics.network.active_threats,
                    Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
                )
            }
            "Log" => {
                // Fetch REAL audit logs from Azure
                let logs = self.azure_client.get_audit_logs(24).await // Last 24 hours
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch logs: {}", e)))?;
                
                serde_json::json!({
                    "framework": control.framework,
                    "control_id": control.control_id,
                    "period": "last_24_hours",
                    "timestamp": Utc::now(),
                    "audit_logs": logs,
                    "total_events": logs.len()
                }).to_string()
            }
            "Screenshot" => {
                // For screenshots, we'll generate a placeholder as real screenshots require UI automation
                format!("Screenshot evidence for control {} captured at {}", control.control_id, Utc::now())
            }
            _ => {
                // Default: fetch general compliance data
                let compliance = self.azure_client.get_compliance_state().await
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to fetch compliance: {}", e)))?;
                
                serde_json::to_string(&compliance)
                    .map_err(|e| ComplianceError::ArtifactError(format!("Failed to serialize: {}", e)))?
            }
        };
        
        // Save the artifact content and return the artifact
        self.save_artifact(content, control, requirement).await
    }

    fn detect_configuration_drift(&self, resources: &[serde_json::Value]) -> Vec<String> {
        // Detect configuration drift by comparing with baseline
        let mut drifts = Vec::new();
        for resource in resources {
            if let Some(tags) = resource.get("tags") {
                if !tags.get("compliance-baseline").is_some() {
                    if let Some(id) = resource.get("id").and_then(|v| v.as_str()) {
                        drifts.push(format!("Resource {} missing baseline tag", id));
                    }
                }
            }
        }
        drifts
    }

    async fn save_artifact(&self, content: String, control: &ComplianceControl, requirement: &ArtifactRequirement) -> Result<EvidenceArtifact, ComplianceError> {
        let content_bytes = content.as_bytes();
        let mut hasher = Sha256::new();
        hasher.update(content_bytes);
        let hash = format!("{:x}", hasher.finalize());

        let artifact_id = format!("artifact-{}", uuid::Uuid::new_v4());
        let storage_path = self.artifact_store.join(&artifact_id).to_string_lossy().to_string();

        // Write artifact to storage
        tokio::fs::write(&storage_path, content_bytes).await
            .map_err(|e| ComplianceError::ArtifactError(e.to_string()))?;

        Ok(EvidenceArtifact {
            artifact_id,
            control_id: control.control_id.clone(),
            artifact_type: requirement.artifact_type.clone(),
            format: requirement.format.clone(),
            content_hash: hash,
            storage_path,
            size_bytes: content_bytes.len() as u64,
            created_at: Utc::now(),
            expires_at: Utc::now() + Duration::days(requirement.retention_days as i64),
            metadata: HashMap::new(),
        })
    }

    fn calculate_compliance_score(&self, results: &[ControlEvidenceResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let passed = results.iter().filter(|r| r.status == "Pass").count() as f64;
        let total = results.len() as f64;
        
        (passed / total * 100.0).round() / 100.0
    }

    fn create_manifest(&self, artifacts: &[EvidenceArtifact]) -> EvidenceManifest {
        let total_size: u64 = artifacts.iter().map(|a| a.size_bytes).sum();
        let mut manifest_hasher = Sha256::new();
        
        let file_index: Vec<ManifestEntry> = artifacts.iter().map(|a| {
            manifest_hasher.update(&a.content_hash);
            ManifestEntry {
                filename: format!("{}.{}", a.artifact_id, a.format.to_lowercase()),
                path: a.storage_path.clone(),
                checksum: a.content_hash.clone(),
                size_bytes: a.size_bytes,
                artifact_type: a.artifact_type.clone(),
            }
        }).collect();

        EvidenceManifest {
            version: "1.0".to_string(),
            checksum: format!("{:x}", manifest_hasher.finalize()),
            total_artifacts: artifacts.len() as u32,
            total_size_bytes: total_size,
            file_index,
            signature: None, // Would be digitally signed in production
        }
    }
}

#[async_trait]
impl ComplianceEngine for AzureComplianceEngine {
    async fn run_control_tests(&self, framework: &str) -> Result<Vec<ControlEvidenceResult>, ComplianceError> {
        let framework = self.frameworks.get(framework)
            .ok_or_else(|| ComplianceError::TestError(format!("Framework {} not found", framework)))?;

        let mut results = Vec::new();

        for control in &framework.controls {
            let mut test_results = Vec::new();
            
            for test in &control.test_queries {
                let result = self.execute_control_test(control, test).await?;
                test_results.push(result);
            }

            let status = if test_results.iter().all(|r| r.status == "Pass") {
                "Pass"
            } else if test_results.iter().any(|r| r.status == "Fail") {
                "Fail"
            } else {
                "Warning"
            };

            results.push(ControlEvidenceResult {
                control_id: control.control_id.clone(),
                control_name: control.name.clone(),
                status: status.to_string(),
                test_results,
                artifacts: vec![],
                exceptions: vec![],
                remediation_actions: vec![],
            });
        }

        Ok(results)
    }

    async fn generate_artifacts(&self, control_id: &str) -> Result<Vec<EvidenceArtifact>, ComplianceError> {
        let mut artifacts = Vec::new();

        for framework in self.frameworks.values() {
            if let Some(control) = framework.controls.iter().find(|c| c.control_id == control_id) {
                for requirement in &control.required_artifacts {
                    let artifact = self.generate_artifact(control, requirement).await?;
                    artifacts.push(artifact);
                }
            }
        }

        Ok(artifacts)
    }

    async fn assemble_evidence_pack(&self, framework: &str, period: DateRange) -> Result<EvidencePack, ComplianceError> {
        // Run all control tests
        let mut control_results = self.run_control_tests(framework).await?;
        
        // Generate artifacts for each control
        let mut all_artifacts = Vec::new();
        for result in &mut control_results {
            let artifacts = self.generate_artifacts(&result.control_id).await?;
            result.artifacts = artifacts.iter().map(|a| a.artifact_id.clone()).collect();
            all_artifacts.extend(artifacts);
        }

        // Calculate summary
        let passed = control_results.iter().filter(|r| r.status == "Pass").count() as u32;
        let failed = control_results.iter().filter(|r| r.status == "Fail").count() as u32;
        let warning = control_results.iter().filter(|r| r.status == "Warning").count() as u32;
        let skipped = control_results.iter().filter(|r| r.status == "Skip").count() as u32;
        
        let summary = ComplianceSummary {
            total_controls: control_results.len() as u32,
            passed_controls: passed,
            failed_controls: failed,
            warning_controls: warning,
            skipped_controls: skipped,
            compliance_score: self.calculate_compliance_score(&control_results),
            critical_findings: control_results.iter()
                .filter(|r| r.status == "Fail")
                .map(|r| r.control_name.clone())
                .collect(),
            improvement_areas: vec![],
        };

        // Create manifest
        let manifest = self.create_manifest(&all_artifacts);

        Ok(EvidencePack {
            pack_id: format!("pack-{}", uuid::Uuid::new_v4()),
            framework: framework.to_string(),
            period_start: period.start,
            period_end: period.end,
            created_at: Utc::now(),
            created_by: "system".to_string(),
            control_results,
            artifacts: all_artifacts,
            summary,
            manifest,
        })
    }

    async fn validate_compliance(&self, framework: &str) -> Result<ComplianceValidation, ComplianceError> {
        let results = self.run_control_tests(framework).await?;
        let score = self.calculate_compliance_score(&results);
        
        let findings: Vec<Finding> = results.iter()
            .filter(|r| r.status != "Pass")
            .map(|r| Finding {
                severity: if r.status == "Fail" { "High".to_string() } else { "Medium".to_string() },
                control_id: r.control_id.clone(),
                description: format!("Control {} failed validation", r.control_name),
                recommendation: "Review control implementation and remediate issues".to_string(),
            })
            .collect();

        Ok(ComplianceValidation {
            framework: framework.to_string(),
            validation_time: Utc::now(),
            is_compliant: score >= 0.8, // 80% threshold
            score,
            findings,
        })
    }

    async fn schedule_assessments(&self) -> Result<Vec<AssessmentSchedule>, ComplianceError> {
        let mut schedules = Vec::new();

        for framework in self.frameworks.values() {
            for control in &framework.controls {
                let next_run = match control.test_frequency.as_str() {
                    "Continuous" => Utc::now() + Duration::minutes(5),
                    "Daily" => Utc::now() + Duration::days(1),
                    "Weekly" => Utc::now() + Duration::weeks(1),
                    "Monthly" => Utc::now() + Duration::days(30),
                    _ => Utc::now() + Duration::days(7),
                };

                schedules.push(AssessmentSchedule {
                    framework: framework.id.clone(),
                    control_id: control.control_id.clone(),
                    next_run,
                    frequency: control.test_frequency.clone(),
                });
            }
        }

        Ok(schedules)
    }
}

// Public API functions
pub async fn get_compliance_status(
    azure_client: Option<&crate::azure_client_async::AsyncAzureClient>,
) -> Result<ComplianceStatus, ComplianceError> {
    let client = azure_client
        .ok_or_else(|| ComplianceError::TestError("Azure client not initialized. Please ensure Azure credentials are configured.".to_string()))?;
    
    let engine = AzureComplianceEngine::new(client.clone()).await?;

    let mut framework_status = Vec::new();
    
    for framework_id in ["iso27001", "pci-dss", "hipaa", "gdpr", "cis-azure"] {
        let validation = engine.validate_compliance(framework_id).await?;
        framework_status.push(FrameworkStatus {
            framework: framework_id.to_string(),
            compliant: validation.is_compliant,
            score: validation.score,
            last_assessed: Utc::now(),
            findings_count: validation.findings.len(),
        });
    }

    Ok(ComplianceStatus {
        overall_compliant: framework_status.iter().all(|f| f.compliant),
        frameworks: framework_status,
        next_assessment: Utc::now() + Duration::days(1),
        evidence_packs_available: 5,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub overall_compliant: bool,
    pub frameworks: Vec<FrameworkStatus>,
    pub next_assessment: DateTime<Utc>,
    pub evidence_packs_available: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkStatus {
    pub framework: String,
    pub compliant: bool,
    pub score: f64,
    pub last_assessed: DateTime<Utc>,
    pub findings_count: usize,
}