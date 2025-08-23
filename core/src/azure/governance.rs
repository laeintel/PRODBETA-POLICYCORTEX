// Azure Governance Integration
// Provides Policy, Compliance, and Security Center data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

use super::client::AzureClient;
use super::api_versions;

/// Azure Policy Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDefinition {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub policy_type: String,
    pub properties: PolicyProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyProperties {
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "policyType")]
    pub policy_type: Option<String>,
    pub mode: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub parameters: Option<serde_json::Value>,
    #[serde(rename = "policyRule")]
    pub policy_rule: Option<serde_json::Value>,
}

/// Policy Assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAssignment {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub assignment_type: String,
    pub location: Option<String>,
    pub properties: PolicyAssignmentProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAssignmentProperties {
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "policyDefinitionId")]
    pub policy_definition_id: Option<String>,
    pub scope: Option<String>,
    #[serde(rename = "notScopes")]
    pub not_scopes: Option<Vec<String>>,
    pub parameters: Option<serde_json::Value>,
    #[serde(rename = "enforcementMode")]
    pub enforcement_mode: Option<String>,
}

/// Compliance State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceState {
    #[serde(rename = "@odata.context")]
    pub odata_context: Option<String>,
    #[serde(rename = "@odata.count")]
    pub odata_count: Option<i32>,
    pub value: Vec<ComplianceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    #[serde(rename = "resourceId")]
    pub resource_id: String,
    #[serde(rename = "policyAssignmentId")]
    pub policy_assignment_id: String,
    #[serde(rename = "policyDefinitionId")]
    pub policy_definition_id: Option<String>,
    #[serde(rename = "complianceState")]
    pub compliance_state: String,
    #[serde(rename = "subscriptionId")]
    pub subscription_id: Option<String>,
    #[serde(rename = "resourceType")]
    pub resource_type: Option<String>,
    #[serde(rename = "resourceLocation")]
    pub resource_location: Option<String>,
    #[serde(rename = "resourceGroup")]
    pub resource_group: Option<String>,
    #[serde(rename = "policyAssignmentName")]
    pub policy_assignment_name: Option<String>,
    #[serde(rename = "policyDefinitionName")]
    pub policy_definition_name: Option<String>,
    #[serde(rename = "isCompliant")]
    pub is_compliant: Option<bool>,
}

/// Security Assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAssessment {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub assessment_type: String,
    pub properties: SecurityAssessmentProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAssessmentProperties {
    #[serde(rename = "resourceDetails")]
    pub resource_details: serde_json::Value,
    pub status: AssessmentStatus,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "additionalData")]
    pub additional_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentStatus {
    pub code: String,
    pub cause: Option<String>,
    pub description: Option<String>,
}

/// Regulatory Compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryCompliance {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub compliance_type: String,
    pub properties: RegulatoryComplianceProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryComplianceProperties {
    pub state: String,
    #[serde(rename = "passedControls")]
    pub passed_controls: Option<i32>,
    #[serde(rename = "failedControls")]
    pub failed_controls: Option<i32>,
    #[serde(rename = "skippedControls")]
    pub skipped_controls: Option<i32>,
    #[serde(rename = "unsupportedControls")]
    pub unsupported_controls: Option<i32>,
}

/// Azure Governance service
pub struct GovernanceService {
    client: AzureClient,
}

impl GovernanceService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get all policy definitions
    pub async fn get_policy_definitions(&self) -> Result<Vec<PolicyDefinition>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Authorization/policyDefinitions",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::POLICY).await
    }

    /// Get all policy assignments
    pub async fn get_policy_assignments(&self) -> Result<Vec<PolicyAssignment>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Authorization/policyAssignments",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::POLICY).await
    }

    /// Get compliance state
    pub async fn get_compliance_state(&self) -> Result<ComplianceState> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.PolicyInsights/policyStates/latest/queryResults",
            self.client.config.subscription_id
        );

        self.client.get_management(&path, "2019-10-01").await
    }

    /// Get security assessments
    pub async fn get_security_assessments(&self) -> Result<Vec<SecurityAssessment>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Security/assessments",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::SECURITY_CENTER).await
    }

    /// Get regulatory compliance
    pub async fn get_regulatory_compliance(&self) -> Result<Vec<RegulatoryCompliance>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Security/regulatoryComplianceStandards",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, api_versions::SECURITY_CENTER).await
    }

    /// Get compliance summary
    pub async fn get_compliance_summary(&self) -> Result<ComplianceSummary> {
        info!("Fetching compliance summary from Azure");

        let compliance_state = self.get_compliance_state().await?;
        let assessments = self.get_security_assessments().await.unwrap_or_default();
        let regulatory = self.get_regulatory_compliance().await.unwrap_or_default();

        // Calculate compliance metrics
        let total_resources = compliance_state.value.len();
        let compliant_resources = compliance_state.value.iter()
            .filter(|r| r.compliance_state == "Compliant")
            .count();
        let non_compliant_resources = compliance_state.value.iter()
            .filter(|r| r.compliance_state == "NonCompliant")
            .count();

        // Security score calculation
        let healthy_assessments = assessments.iter()
            .filter(|a| a.properties.status.code == "Healthy")
            .count();
        let total_assessments = assessments.len();
        let security_score = if total_assessments > 0 {
            (healthy_assessments as f64 / total_assessments as f64) * 100.0
        } else {
            0.0
        };

        // Policy violations by type
        let mut violations_by_type = HashMap::new();
        for result in &compliance_state.value {
            if result.compliance_state == "NonCompliant" {
                let resource_type = result.resource_type.clone().unwrap_or_else(|| "Unknown".to_string());
                *violations_by_type.entry(resource_type).or_insert(0) += 1;
            }
        }

        Ok(ComplianceSummary {
            total_resources,
            compliant_resources,
            non_compliant_resources,
            compliance_percentage: if total_resources > 0 {
                (compliant_resources as f64 / total_resources as f64) * 100.0
            } else {
                0.0
            },
            security_score,
            total_policies: 0, // Will be updated with policy definitions count
            active_violations: non_compliant_resources,
            violations_by_type,
            regulatory_standards: regulatory.len(),
        })
    }

    /// Get policy violations
    pub async fn get_policy_violations(&self) -> Result<Vec<PolicyViolation>> {
        let compliance_state = self.get_compliance_state().await?;
        
        let violations: Vec<PolicyViolation> = compliance_state.value
            .into_iter()
            .filter(|r| r.compliance_state == "NonCompliant")
            .map(|r| PolicyViolation {
                resource_id: r.resource_id,
                policy_name: r.policy_definition_name.unwrap_or_else(|| "Unknown".to_string()),
                policy_assignment: r.policy_assignment_name.unwrap_or_else(|| "Unknown".to_string()),
                resource_type: r.resource_type.unwrap_or_else(|| "Unknown".to_string()),
                resource_group: r.resource_group.unwrap_or_else(|| "Unknown".to_string()),
                location: r.resource_location.unwrap_or_else(|| "Unknown".to_string()),
                severity: "Medium".to_string(), // Default severity
                detected_at: Utc::now(),
            })
            .collect();

        Ok(violations)
    }
}

#[derive(Debug, Serialize)]
pub struct ComplianceSummary {
    pub total_resources: usize,
    pub compliant_resources: usize,
    pub non_compliant_resources: usize,
    pub compliance_percentage: f64,
    pub security_score: f64,
    pub total_policies: usize,
    pub active_violations: usize,
    pub violations_by_type: HashMap<String, usize>,
    pub regulatory_standards: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyViolation {
    pub resource_id: String,
    pub policy_name: String,
    pub policy_assignment: String,
    pub resource_type: String,
    pub resource_group: String,
    pub location: String,
    pub severity: String,
    pub detected_at: DateTime<Utc>,
}