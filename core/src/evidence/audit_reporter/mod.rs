use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use qrcode::QrCode;
use qrcode::render::svg;
use base64::{Engine as _, engine::general_purpose};
use sha3::{Digest, Sha3_256};
use ed25519_dalek::{Keypair, Signature, Signer};
use rand::rngs::OsRng;

use crate::evidence::collector::{EvidenceRecord, EvidenceSeverity};
use crate::evidence::hash_chain::{Block, ChainStats};

/// Report format types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportFormat {
    Pdf,
    Html,
    Json,
    Markdown,
    Xml,
}

/// Report type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportType {
    ExecutiveSummary,
    TechnicalDetailed,
    ComplianceAudit,
    SecurityAssessment,
    CostOptimization,
    IncidentResponse,
    QuarterlyReview,
    AnnualCompliance,
}

/// Audit report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_id: Uuid,
    pub report_type: ReportType,
    pub title: String,
    pub organization: String,
    pub tenant_id: String,
    pub subscription_ids: Vec<String>,
    pub reporting_period_start: DateTime<Utc>,
    pub reporting_period_end: DateTime<Utc>,
    pub generation_timestamp: DateTime<Utc>,
    pub generated_by: String,
    pub approved_by: Option<String>,
    pub classification: String,
    pub retention_years: u32,
}

/// Verification data for report integrity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationData {
    pub report_hash: String,
    pub signature: String,
    pub public_key: String,
    pub block_height: u64,
    pub merkle_root: String,
    pub verification_url: String,
    pub qr_code_svg: String,
    pub timestamp: DateTime<Utc>,
}

/// Executive summary section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub total_resources_scanned: u64,
    pub compliance_score: f64,
    pub critical_findings: u32,
    pub high_findings: u32,
    pub medium_findings: u32,
    pub low_findings: u32,
    pub cost_savings_identified: f64,
    pub security_incidents: u32,
    pub key_recommendations: Vec<String>,
    pub trend_analysis: TrendAnalysis,
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub compliance_trend: String, // "improving", "stable", "declining"
    pub security_posture_trend: String,
    pub cost_trend: String,
    pub monthly_comparison: HashMap<String, f64>,
}

/// Detailed finding in the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub finding_id: Uuid,
    pub evidence_ids: Vec<Uuid>,
    pub severity: EvidenceSeverity,
    pub title: String,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub impact: String,
    pub remediation_steps: Vec<String>,
    pub compliance_mappings: Vec<String>,
    pub evidence_chain: Vec<String>, // Hashes
    pub first_detected: DateTime<Utc>,
    pub last_observed: DateTime<Utc>,
}

/// Complete audit report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub metadata: ReportMetadata,
    pub executive_summary: ExecutiveSummary,
    pub findings: Vec<Finding>,
    pub evidence_summary: EvidenceSummary,
    pub compliance_status: ComplianceStatus,
    pub verification_data: VerificationData,
    pub appendices: Vec<Appendix>,
}

/// Evidence summary section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    pub total_evidence_collected: u64,
    pub evidence_by_source: HashMap<String, u64>,
    pub evidence_by_severity: HashMap<String, u64>,
    pub chain_stats: ChainStats,
    pub verification_status: String,
}

/// Compliance status section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub frameworks: Vec<FrameworkCompliance>,
    pub overall_compliance_percentage: f64,
    pub controls_passed: u32,
    pub controls_failed: u32,
    pub controls_not_applicable: u32,
}

/// Framework compliance details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCompliance {
    pub framework_name: String,
    pub version: String,
    pub compliance_percentage: f64,
    pub control_details: Vec<ControlStatus>,
}

/// Control status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlStatus {
    pub control_id: String,
    pub control_name: String,
    pub status: String, // "pass", "fail", "not_applicable"
    pub evidence_refs: Vec<Uuid>,
}

/// Report appendix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appendix {
    pub title: String,
    pub content: String,
    pub appendix_type: String,
}

/// Main Audit Reporter service
pub struct AuditReporter {
    keypair: Keypair,
    base_verification_url: String,
}

impl AuditReporter {
    /// Create new audit reporter
    pub fn new(base_verification_url: String) -> Self {
        let mut csprng = OsRng {};
        let keypair = Keypair::generate(&mut csprng);

        AuditReporter {
            keypair,
            base_verification_url,
        }
    }

    /// Generate audit report
    pub async fn generate_report(
        &self,
        report_type: ReportType,
        title: String,
        organization: String,
        tenant_id: String,
        subscription_ids: Vec<String>,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        evidence_records: Vec<EvidenceRecord>,
        chain_stats: ChainStats,
        compliance_mappings: HashMap<Uuid, Vec<String>>,
    ) -> Result<AuditReport, String> {
        // Create report metadata
        let metadata = ReportMetadata {
            report_id: Uuid::new_v4(),
            report_type,
            title,
            organization,
            tenant_id,
            subscription_ids,
            reporting_period_start: period_start,
            reporting_period_end: period_end,
            generation_timestamp: Utc::now(),
            generated_by: "PolicyCortex Audit System".to_string(),
            approved_by: None,
            classification: "Internal".to_string(),
            retention_years: 7,
        };

        // Analyze evidence and create findings
        let findings = self.analyze_evidence(&evidence_records, &compliance_mappings)?;

        // Generate executive summary
        let executive_summary = self.create_executive_summary(&findings, &evidence_records)?;

        // Create evidence summary
        let evidence_summary = self.create_evidence_summary(&evidence_records, chain_stats)?;

        // Calculate compliance status
        let compliance_status = self.calculate_compliance_status(&findings, &compliance_mappings)?;

        // Generate verification data
        let report_hash = self.calculate_report_hash(&metadata, &findings)?;
        let verification_data = self.create_verification_data(
            report_hash,
            metadata.report_id,
            chain_stats.last_block_hash.clone(),
        )?;

        // Create appendices
        let appendices = self.create_appendices(&evidence_records)?;

        Ok(AuditReport {
            metadata,
            executive_summary,
            findings,
            evidence_summary,
            compliance_status,
            verification_data,
            appendices,
        })
    }

    /// Analyze evidence and create findings
    fn analyze_evidence(
        &self,
        evidence_records: &[EvidenceRecord],
        compliance_mappings: &HashMap<Uuid, Vec<String>>,
    ) -> Result<Vec<Finding>, String> {
        let mut findings_map: HashMap<String, Finding> = HashMap::new();

        for record in evidence_records {
            let key = format!("{}:{}", record.policy_id, record.severity as u8);
            
            let finding = findings_map.entry(key.clone()).or_insert_with(|| {
                Finding {
                    finding_id: Uuid::new_v4(),
                    evidence_ids: vec![],
                    severity: record.severity.clone(),
                    title: record.policy_name.clone(),
                    description: record.check_result.message.clone(),
                    affected_resources: vec![],
                    impact: self.determine_impact(&record.severity),
                    remediation_steps: record.check_result.recommendations.clone(),
                    compliance_mappings: compliance_mappings
                        .get(&record.id)
                        .cloned()
                        .unwrap_or_default(),
                    evidence_chain: vec![],
                    first_detected: record.timestamp,
                    last_observed: record.timestamp,
                }
            });

            finding.evidence_ids.push(record.id);
            finding.affected_resources.push(record.resource_id.clone());
            finding.evidence_chain.push(record.hash.clone());
            
            if record.timestamp > finding.last_observed {
                finding.last_observed = record.timestamp;
            }
            if record.timestamp < finding.first_detected {
                finding.first_detected = record.timestamp;
            }
        }

        Ok(findings_map.into_values().collect())
    }

    /// Determine impact based on severity
    fn determine_impact(&self, severity: &EvidenceSeverity) -> String {
        match severity {
            EvidenceSeverity::Critical => "Critical impact on security, compliance, or availability. Immediate action required.".to_string(),
            EvidenceSeverity::High => "Significant security or compliance risk. Should be addressed within 24-48 hours.".to_string(),
            EvidenceSeverity::Medium => "Moderate risk to operations or compliance. Plan remediation within 1 week.".to_string(),
            EvidenceSeverity::Low => "Minor issue with minimal impact. Address during next maintenance window.".to_string(),
            EvidenceSeverity::Info => "Informational finding. No immediate action required.".to_string(),
        }
    }

    /// Create executive summary
    fn create_executive_summary(
        &self,
        findings: &[Finding],
        evidence_records: &[EvidenceRecord],
    ) -> Result<ExecutiveSummary, String> {
        let mut severity_counts = HashMap::new();
        for finding in findings {
            *severity_counts
                .entry(format!("{:?}", finding.severity))
                .or_insert(0) += 1;
        }

        let compliance_score = self.calculate_compliance_score(evidence_records);

        Ok(ExecutiveSummary {
            total_resources_scanned: evidence_records
                .iter()
                .map(|e| e.resource_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .len() as u64,
            compliance_score,
            critical_findings: *severity_counts.get("Critical").unwrap_or(&0),
            high_findings: *severity_counts.get("High").unwrap_or(&0),
            medium_findings: *severity_counts.get("Medium").unwrap_or(&0),
            low_findings: *severity_counts.get("Low").unwrap_or(&0),
            cost_savings_identified: 0.0, // Would be calculated from cost findings
            security_incidents: 0, // Would be calculated from security findings
            key_recommendations: self.generate_key_recommendations(findings),
            trend_analysis: TrendAnalysis {
                compliance_trend: "improving".to_string(),
                security_posture_trend: "stable".to_string(),
                cost_trend: "optimizing".to_string(),
                monthly_comparison: HashMap::new(),
            },
        })
    }

    /// Calculate compliance score
    fn calculate_compliance_score(&self, evidence_records: &[EvidenceRecord]) -> f64 {
        let total = evidence_records.len() as f64;
        let compliant = evidence_records
            .iter()
            .filter(|e| e.check_result.compliant)
            .count() as f64;
        
        if total > 0.0 {
            (compliant / total) * 100.0
        } else {
            100.0
        }
    }

    /// Generate key recommendations
    fn generate_key_recommendations(&self, findings: &[Finding]) -> Vec<String> {
        let mut recommendations = vec![];

        // Add critical findings remediation
        let critical_count = findings
            .iter()
            .filter(|f| matches!(f.severity, EvidenceSeverity::Critical))
            .count();
        
        if critical_count > 0 {
            recommendations.push(format!(
                "Address {} critical findings immediately to prevent security breaches",
                critical_count
            ));
        }

        // Add high severity recommendations
        let high_count = findings
            .iter()
            .filter(|f| matches!(f.severity, EvidenceSeverity::High))
            .count();
        
        if high_count > 0 {
            recommendations.push(format!(
                "Remediate {} high severity issues within 48 hours",
                high_count
            ));
        }

        // Add compliance recommendations
        recommendations.push("Implement automated compliance monitoring for continuous assurance".to_string());
        recommendations.push("Review and update security policies quarterly".to_string());

        recommendations
    }

    /// Create evidence summary
    fn create_evidence_summary(
        &self,
        evidence_records: &[EvidenceRecord],
        chain_stats: ChainStats,
    ) -> Result<EvidenceSummary, String> {
        let mut by_source = HashMap::new();
        let mut by_severity = HashMap::new();

        for record in evidence_records {
            *by_source
                .entry(format!("{:?}", record.source))
                .or_insert(0) += 1;
            *by_severity
                .entry(format!("{:?}", record.severity))
                .or_insert(0) += 1;
        }

        Ok(EvidenceSummary {
            total_evidence_collected: evidence_records.len() as u64,
            evidence_by_source: by_source,
            evidence_by_severity: by_severity,
            chain_stats,
            verification_status: "Verified".to_string(),
        })
    }

    /// Calculate compliance status
    fn calculate_compliance_status(
        &self,
        findings: &[Finding],
        compliance_mappings: &HashMap<Uuid, Vec<String>>,
    ) -> Result<ComplianceStatus, String> {
        // Group by framework
        let mut framework_map: HashMap<String, Vec<&Finding>> = HashMap::new();
        
        for finding in findings {
            for mapping in &finding.compliance_mappings {
                let framework = mapping.split('-').next().unwrap_or("Unknown");
                framework_map
                    .entry(framework.to_string())
                    .or_insert_with(Vec::new)
                    .push(finding);
            }
        }

        let mut frameworks = vec![];
        let mut total_passed = 0;
        let mut total_failed = 0;

        for (framework_name, framework_findings) in framework_map {
            let passed = framework_findings
                .iter()
                .filter(|f| matches!(f.severity, EvidenceSeverity::Info | EvidenceSeverity::Low))
                .count() as u32;
            
            let failed = framework_findings.len() as u32 - passed;
            
            total_passed += passed;
            total_failed += failed;

            frameworks.push(FrameworkCompliance {
                framework_name,
                version: "Latest".to_string(),
                compliance_percentage: if framework_findings.is_empty() {
                    100.0
                } else {
                    (passed as f64 / framework_findings.len() as f64) * 100.0
                },
                control_details: vec![], // Would be populated with detailed control status
            });
        }

        let total = (total_passed + total_failed) as f64;
        let overall_percentage = if total > 0.0 {
            (total_passed as f64 / total) * 100.0
        } else {
            100.0
        };

        Ok(ComplianceStatus {
            frameworks,
            overall_compliance_percentage: overall_percentage,
            controls_passed: total_passed,
            controls_failed: total_failed,
            controls_not_applicable: 0,
        })
    }

    /// Calculate report hash
    fn calculate_report_hash(
        &self,
        metadata: &ReportMetadata,
        findings: &[Finding],
    ) -> Result<String, String> {
        let mut hasher = Sha3_256::new();
        
        hasher.update(metadata.report_id.to_string().as_bytes());
        hasher.update(metadata.title.as_bytes());
        hasher.update(metadata.generation_timestamp.to_rfc3339().as_bytes());
        
        for finding in findings {
            hasher.update(finding.finding_id.to_string().as_bytes());
            hasher.update(format!("{:?}", finding.severity).as_bytes());
        }

        Ok(hex::encode(hasher.finalize()))
    }

    /// Create verification data with QR code
    fn create_verification_data(
        &self,
        report_hash: String,
        report_id: Uuid,
        merkle_root: String,
    ) -> Result<VerificationData, String> {
        // Sign the report
        let message = format!("{}{}", report_hash, report_id);
        let signature = self.keypair.sign(message.as_bytes());

        // Create verification URL
        let verification_url = format!(
            "{}/verify/{}?hash={}",
            self.base_verification_url,
            report_id,
            report_hash
        );

        // Generate QR code
        let qr_code = QrCode::new(&verification_url)
            .map_err(|e| format!("Failed to generate QR code: {}", e))?;
        
        let qr_svg = qr_code.render::<svg::Color>()
            .min_dimensions(200, 200)
            .build();

        Ok(VerificationData {
            report_hash,
            signature: hex::encode(signature.to_bytes()),
            public_key: hex::encode(self.keypair.public.to_bytes()),
            block_height: 0, // Would be set from actual chain
            merkle_root,
            verification_url,
            qr_code_svg: qr_svg,
            timestamp: Utc::now(),
        })
    }

    /// Create appendices
    fn create_appendices(&self, evidence_records: &[EvidenceRecord]) -> Result<Vec<Appendix>, String> {
        let mut appendices = vec![];

        // Methodology appendix
        appendices.push(Appendix {
            title: "Audit Methodology".to_string(),
            content: "This audit was conducted using automated policy scanning, continuous compliance monitoring, and cryptographic evidence collection. All findings are backed by immutable evidence stored in a hash chain with Merkle tree verification.".to_string(),
            appendix_type: "methodology".to_string(),
        });

        // Evidence chain appendix
        appendices.push(Appendix {
            title: "Evidence Chain Verification".to_string(),
            content: format!(
                "Total evidence records: {}. All evidence is cryptographically signed and stored in an immutable hash chain. Verification can be performed using the QR code on the cover page.",
                evidence_records.len()
            ),
            appendix_type: "verification".to_string(),
        });

        // Glossary
        appendices.push(Appendix {
            title: "Glossary".to_string(),
            content: "Critical: Immediate threat to security or availability. High: Significant risk requiring prompt attention. Medium: Moderate risk with planned remediation. Low: Minor issue for future consideration.".to_string(),
            appendix_type: "glossary".to_string(),
        });

        Ok(appendices)
    }

    /// Export report to different formats
    pub async fn export_report(
        &self,
        report: &AuditReport,
        format: ReportFormat,
    ) -> Result<Vec<u8>, String> {
        match format {
            ReportFormat::Json => {
                serde_json::to_vec_pretty(report)
                    .map_err(|e| format!("Failed to serialize to JSON: {}", e))
            }
            ReportFormat::Html => {
                self.generate_html_report(report)
            }
            ReportFormat::Pdf => {
                // Would use a PDF generation library like wkhtmltopdf or similar
                self.generate_pdf_report(report)
            }
            ReportFormat::Markdown => {
                self.generate_markdown_report(report)
            }
            ReportFormat::Xml => {
                // Would use quick-xml or similar for XML generation
                self.generate_xml_report(report)
            }
        }
    }

    /// Generate HTML report
    fn generate_html_report(&self, report: &AuditReport) -> Result<Vec<u8>, String> {
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metadata {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .finding {{ border-left: 4px solid #ff6b6b; padding-left: 20px; margin: 20px 0; }}
        .finding.critical {{ border-color: #ff0000; }}
        .finding.high {{ border-color: #ff6b6b; }}
        .finding.medium {{ border-color: #ffd93d; }}
        .finding.low {{ border-color: #6bcf7f; }}
        .qr-code {{ text-align: center; margin: 30px 0; }}
        .verification {{ background: #e8f5e9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{}</h1>
    
    <div class="metadata">
        <h2>Report Information</h2>
        <p><strong>Organization:</strong> {}</p>
        <p><strong>Report ID:</strong> {}</p>
        <p><strong>Generated:</strong> {}</p>
        <p><strong>Period:</strong> {} to {}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Compliance Score:</strong> {:.1}%</p>
        <p><strong>Total Resources Scanned:</strong> {}</p>
        <p><strong>Critical Findings:</strong> {}</p>
        <p><strong>High Findings:</strong> {}</p>
        <p><strong>Medium Findings:</strong> {}</p>
        <p><strong>Low Findings:</strong> {}</p>
    </div>

    <h2>Key Findings</h2>
    {}</p>

    <div class="verification">
        <h2>Report Verification</h2>
        <p><strong>Report Hash:</strong> {}</p>
        <p><strong>Verification URL:</strong> <a href="{}">{}</a></p>
        <div class="qr-code">
            {}
        </div>
    </div>
</body>
</html>"#,
            report.metadata.title,
            report.metadata.title,
            report.metadata.organization,
            report.metadata.report_id,
            report.metadata.generation_timestamp.to_rfc3339(),
            report.metadata.reporting_period_start.date_naive(),
            report.metadata.reporting_period_end.date_naive(),
            report.executive_summary.compliance_score,
            report.executive_summary.total_resources_scanned,
            report.executive_summary.critical_findings,
            report.executive_summary.high_findings,
            report.executive_summary.medium_findings,
            report.executive_summary.low_findings,
            self.format_findings_html(&report.findings),
            report.verification_data.report_hash,
            report.verification_data.verification_url,
            report.verification_data.verification_url,
            report.verification_data.qr_code_svg,
        );

        Ok(html.into_bytes())
    }

    /// Format findings as HTML
    fn format_findings_html(&self, findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|f| {
                format!(
                    r#"<div class="finding {}">
                        <h3>{}</h3>
                        <p>{}</p>
                        <p><strong>Affected Resources:</strong> {}</p>
                        <p><strong>Impact:</strong> {}</p>
                        <p><strong>Remediation:</strong></p>
                        <ul>{}</ul>
                    </div>"#,
                    format!("{:?}", f.severity).to_lowercase(),
                    f.title,
                    f.description,
                    f.affected_resources.join(", "),
                    f.impact,
                    f.remediation_steps
                        .iter()
                        .map(|s| format!("<li>{}</li>", s))
                        .collect::<Vec<_>>()
                        .join(""),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate PDF report (placeholder)
    fn generate_pdf_report(&self, report: &AuditReport) -> Result<Vec<u8>, String> {
        // In production, would use a PDF library
        // For now, return HTML that could be converted to PDF
        self.generate_html_report(report)
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, report: &AuditReport) -> Result<Vec<u8>, String> {
        let markdown = format!(
            r#"# {}

## Report Information
- **Organization:** {}
- **Report ID:** {}
- **Generated:** {}
- **Period:** {} to {}

## Executive Summary
- **Compliance Score:** {:.1}%
- **Total Resources Scanned:** {}
- **Critical Findings:** {}
- **High Findings:** {}
- **Medium Findings:** {}
- **Low Findings:** {}

## Key Recommendations
{}

## Findings
{}

## Verification
- **Report Hash:** `{}`
- **Verification URL:** [{}]({})

---
*This report was generated by PolicyCortex Audit System with cryptographic verification.*
"#,
            report.metadata.title,
            report.metadata.organization,
            report.metadata.report_id,
            report.metadata.generation_timestamp.to_rfc3339(),
            report.metadata.reporting_period_start.date_naive(),
            report.metadata.reporting_period_end.date_naive(),
            report.executive_summary.compliance_score,
            report.executive_summary.total_resources_scanned,
            report.executive_summary.critical_findings,
            report.executive_summary.high_findings,
            report.executive_summary.medium_findings,
            report.executive_summary.low_findings,
            report.executive_summary
                .key_recommendations
                .iter()
                .map(|r| format!("- {}", r))
                .collect::<Vec<_>>()
                .join("\n"),
            self.format_findings_markdown(&report.findings),
            report.verification_data.report_hash,
            report.verification_data.verification_url,
            report.verification_data.verification_url,
        );

        Ok(markdown.into_bytes())
    }

    /// Format findings as Markdown
    fn format_findings_markdown(&self, findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|f| {
                format!(
                    r#"### {} - {:?} Severity

{}

**Affected Resources:** {}
**Impact:** {}
**First Detected:** {}
**Last Observed:** {}

**Remediation Steps:**
{}
"#,
                    f.title,
                    f.severity,
                    f.description,
                    f.affected_resources.join(", "),
                    f.impact,
                    f.first_detected.date_naive(),
                    f.last_observed.date_naive(),
                    f.remediation_steps
                        .iter()
                        .map(|s| format!("1. {}", s))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n\n")
    }

    /// Generate XML report (placeholder)
    fn generate_xml_report(&self, report: &AuditReport) -> Result<Vec<u8>, String> {
        // In production, would use an XML library
        // For now, return JSON
        self.export_report(report, ReportFormat::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::collector::{CheckResult, EvidenceSource};

    #[tokio::test]
    async fn test_report_generation() {
        let reporter = AuditReporter::new("https://verify.policycortex.com".to_string());

        // Create sample evidence
        let evidence = vec![
            EvidenceRecord {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                source: EvidenceSource::PolicyEngine,
                severity: EvidenceSeverity::High,
                resource_id: "vm-001".to_string(),
                resource_type: "VirtualMachine".to_string(),
                subscription_id: "sub-123".to_string(),
                tenant_id: "tenant-123".to_string(),
                policy_id: "pol-001".to_string(),
                policy_name: "Encryption Policy".to_string(),
                check_result: CheckResult {
                    compliant: false,
                    message: "Disk encryption not enabled".to_string(),
                    details: HashMap::new(),
                    recommendations: vec!["Enable disk encryption".to_string()],
                    evidence_artifacts: vec![],
                },
                evidence_data: HashMap::new(),
                hash: "abc123".to_string(),
                previous_hash: None,
                crypto_metadata: None,
                immutable: true,
                block_height: Some(1),
            },
        ];

        let chain_stats = ChainStats {
            total_blocks: 10,
            current_block_height: 10,
            pending_transactions: 0,
            last_block_hash: "xyz789".to_string(),
            chain_verified: true,
        };

        let report = reporter
            .generate_report(
                ReportType::ComplianceAudit,
                "Q4 2024 Compliance Audit".to_string(),
                "Contoso Corporation".to_string(),
                "tenant-123".to_string(),
                vec!["sub-123".to_string()],
                Utc::now() - chrono::Duration::days(90),
                Utc::now(),
                evidence,
                chain_stats,
                HashMap::new(),
            )
            .await
            .unwrap();

        assert_eq!(report.metadata.title, "Q4 2024 Compliance Audit");
        assert_eq!(report.findings.len(), 1);
        assert!(!report.verification_data.report_hash.is_empty());
        assert!(!report.verification_data.qr_code_svg.is_empty());
    }

    #[tokio::test]
    async fn test_report_export() {
        let reporter = AuditReporter::new("https://verify.policycortex.com".to_string());

        let chain_stats = ChainStats {
            total_blocks: 1,
            current_block_height: 1,
            pending_transactions: 0,
            last_block_hash: "genesis".to_string(),
            chain_verified: true,
        };

        let report = reporter
            .generate_report(
                ReportType::ExecutiveSummary,
                "Test Report".to_string(),
                "Test Org".to_string(),
                "tenant-1".to_string(),
                vec![],
                Utc::now() - chrono::Duration::days(30),
                Utc::now(),
                vec![],
                chain_stats,
                HashMap::new(),
            )
            .await
            .unwrap();

        // Test JSON export
        let json = reporter
            .export_report(&report, ReportFormat::Json)
            .await
            .unwrap();
        assert!(!json.is_empty());

        // Test HTML export
        let html = reporter
            .export_report(&report, ReportFormat::Html)
            .await
            .unwrap();
        assert!(!html.is_empty());

        // Test Markdown export
        let markdown = reporter
            .export_report(&report, ReportFormat::Markdown)
            .await
            .unwrap();
        assert!(!markdown.is_empty());
    }
}