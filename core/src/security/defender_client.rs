// Microsoft Defender for Cloud integration
// Provides security posture management and threat protection

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use reqwest::Client;
use anyhow::{Result, Context};

/// Defender for Cloud client for security operations
pub struct DefenderClient {
    client: Client,
    subscription_id: String,
    tenant_id: String,
    access_token: String,
    base_url: String,
}

/// Security alert from Defender
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub id: String,
    pub name: String,
    pub alert_type: String,
    pub severity: AlertSeverity,
    pub status: AlertStatus,
    pub description: String,
    pub remediation_steps: Vec<String>,
    pub resource_id: String,
    pub time_generated: DateTime<Utc>,
    pub vendor_name: String,
    pub alert_uri: String,
    pub compromised_entity: Option<String>,
    pub attack_vector: Option<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    High,
    Medium,
    Low,
    Informational,
}

/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertStatus {
    Active,
    Resolved,
    Dismissed,
    InProgress,
}

/// Azure Security Benchmark (ASB) assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASBAssessment {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub status: ComplianceStatus,
    pub description: String,
    pub remediation_description: String,
    pub category: String,
    pub severity: String,
    pub user_impact: String,
    pub implementation_effort: String,
    pub threats: Vec<String>,
    pub preview: bool,
    pub assessment_type: String,
}

/// Compliance status for assessments
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStatus {
    Healthy,
    Unhealthy,
    NotApplicable,
    Unknown,
}

/// Secure Score data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureScore {
    pub score: f64,
    pub max_score: f64,
    pub percentage: f64,
    pub weight: i32,
    pub controls: Vec<SecureScoreControl>,
}

/// Secure Score control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureScoreControl {
    pub id: String,
    pub display_name: String,
    pub score: f64,
    pub max_score: f64,
    pub healthy_resource_count: i32,
    pub unhealthy_resource_count: i32,
    pub not_applicable_resource_count: i32,
    pub description: String,
    pub remediation_description: String,
    pub actions: Vec<String>,
}

/// Regulatory compliance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryCompliance {
    pub id: String,
    pub name: String,
    pub compliance_standard: String,
    pub control_id: String,
    pub control_name: String,
    pub description: String,
    pub passed_assessments: i32,
    pub failed_assessments: i32,
    pub skipped_assessments: i32,
    pub compliance_score: f64,
}

/// Threat intelligence indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIndicator {
    pub id: String,
    pub indicator_type: String,
    pub threat_type: Vec<String>,
    pub description: String,
    pub confidence: i32,
    pub severity: AlertSeverity,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub source: String,
    pub pattern: String,
    pub kill_chain_phases: Vec<String>,
}

impl DefenderClient {
    /// Create a new Defender client
    pub async fn new(
        subscription_id: String,
        tenant_id: String,
    ) -> Result<Self> {
        let client = Client::new();
        let access_token = Self::get_access_token(&client, &tenant_id).await?;
        
        Ok(Self {
            client,
            subscription_id,
            tenant_id,
            access_token,
            base_url: "https://management.azure.com".to_string(),
        })
    }
    
    /// Get access token for Azure Management API
    async fn get_access_token(client: &Client, tenant_id: &str) -> Result<String> {
        // In production, use proper Azure authentication
        // This is a placeholder for managed identity or service principal auth
        // You would typically use azure_identity crate here
        
        // For now, return a placeholder
        // TODO: Implement proper authentication
        Ok("placeholder_token".to_string())
    }
    
    /// Fetch security alerts
    pub async fn get_security_alerts(&self) -> Result<Vec<SecurityAlert>> {
        let url = format!(
            "{}/subscriptions/{}/providers/Microsoft.Security/alerts?api-version=2022-01-01",
            self.base_url, self.subscription_id
        );
        
        let response = self.client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to fetch security alerts")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch alerts: {}", response.status());
        }
        
        let data: serde_json::Value = response.json().await?;
        let alerts = self.parse_alerts(data)?;
        
        Ok(alerts)
    }
    
    /// Parse alerts from API response
    fn parse_alerts(&self, data: serde_json::Value) -> Result<Vec<SecurityAlert>> {
        let mut alerts = Vec::new();
        
        if let Some(value_array) = data["value"].as_array() {
            for item in value_array {
                let alert = SecurityAlert {
                    id: item["id"].as_str().unwrap_or("").to_string(),
                    name: item["name"].as_str().unwrap_or("").to_string(),
                    alert_type: item["properties"]["alertType"].as_str().unwrap_or("").to_string(),
                    severity: self.parse_severity(item["properties"]["severity"].as_str().unwrap_or("Low")),
                    status: self.parse_status(item["properties"]["status"].as_str().unwrap_or("Active")),
                    description: item["properties"]["description"].as_str().unwrap_or("").to_string(),
                    remediation_steps: self.parse_string_array(&item["properties"]["remediationSteps"]),
                    resource_id: item["properties"]["resourceId"].as_str().unwrap_or("").to_string(),
                    time_generated: DateTime::parse_from_rfc3339(
                        item["properties"]["timeGeneratedUtc"].as_str().unwrap_or("2024-01-01T00:00:00Z")
                    ).unwrap_or_default().with_timezone(&Utc),
                    vendor_name: item["properties"]["vendorName"].as_str().unwrap_or("").to_string(),
                    alert_uri: item["properties"]["alertUri"].as_str().unwrap_or("").to_string(),
                    compromised_entity: item["properties"]["compromisedEntity"].as_str().map(String::from),
                    attack_vector: item["properties"]["attackVector"].as_str().map(String::from),
                };
                alerts.push(alert);
            }
        }
        
        Ok(alerts)
    }
    
    /// Get Azure Security Benchmark assessments
    pub async fn get_asb_assessments(&self) -> Result<Vec<ASBAssessment>> {
        let url = format!(
            "{}/subscriptions/{}/providers/Microsoft.Security/assessments?api-version=2021-06-01",
            self.base_url, self.subscription_id
        );
        
        let response = self.client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to fetch ASB assessments")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch assessments: {}", response.status());
        }
        
        let data: serde_json::Value = response.json().await?;
        let assessments = self.parse_assessments(data)?;
        
        Ok(assessments)
    }
    
    /// Parse ASB assessments from API response
    fn parse_assessments(&self, data: serde_json::Value) -> Result<Vec<ASBAssessment>> {
        let mut assessments = Vec::new();
        
        if let Some(value_array) = data["value"].as_array() {
            for item in value_array {
                let assessment = ASBAssessment {
                    id: item["id"].as_str().unwrap_or("").to_string(),
                    name: item["name"].as_str().unwrap_or("").to_string(),
                    display_name: item["properties"]["displayName"].as_str().unwrap_or("").to_string(),
                    status: self.parse_compliance_status(
                        item["properties"]["status"]["code"].as_str().unwrap_or("Unknown")
                    ),
                    description: item["properties"]["description"].as_str().unwrap_or("").to_string(),
                    remediation_description: item["properties"]["remediationDescription"]
                        .as_str().unwrap_or("").to_string(),
                    category: item["properties"]["category"].as_str().unwrap_or("").to_string(),
                    severity: item["properties"]["severity"].as_str().unwrap_or("").to_string(),
                    user_impact: item["properties"]["userImpact"].as_str().unwrap_or("").to_string(),
                    implementation_effort: item["properties"]["implementationEffort"]
                        .as_str().unwrap_or("").to_string(),
                    threats: self.parse_string_array(&item["properties"]["threats"]),
                    preview: item["properties"]["preview"].as_bool().unwrap_or(false),
                    assessment_type: item["properties"]["assessmentType"].as_str().unwrap_or("").to_string(),
                };
                assessments.push(assessment);
            }
        }
        
        Ok(assessments)
    }
    
    /// Get Secure Score
    pub async fn get_secure_score(&self) -> Result<SecureScore> {
        let url = format!(
            "{}/subscriptions/{}/providers/Microsoft.Security/secureScores/ascScore?api-version=2020-01-01",
            self.base_url, self.subscription_id
        );
        
        let response = self.client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to fetch secure score")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch secure score: {}", response.status());
        }
        
        let data: serde_json::Value = response.json().await?;
        let secure_score = self.parse_secure_score(data)?;
        
        Ok(secure_score)
    }
    
    /// Parse Secure Score from API response
    fn parse_secure_score(&self, data: serde_json::Value) -> Result<SecureScore> {
        let score = SecureScore {
            score: data["properties"]["score"]["current"].as_f64().unwrap_or(0.0),
            max_score: data["properties"]["score"]["max"].as_f64().unwrap_or(100.0),
            percentage: data["properties"]["score"]["percentage"].as_f64().unwrap_or(0.0),
            weight: data["properties"]["weight"].as_i64().unwrap_or(0) as i32,
            controls: Vec::new(), // Would need separate API call for controls
        };
        
        Ok(score)
    }
    
    /// Get regulatory compliance data
    pub async fn get_regulatory_compliance(&self) -> Result<Vec<RegulatoryCompliance>> {
        let url = format!(
            "{}/subscriptions/{}/providers/Microsoft.Security/regulatoryComplianceStandards?api-version=2019-01-01",
            self.base_url, self.subscription_id
        );
        
        let response = self.client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to fetch regulatory compliance")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch compliance data: {}", response.status());
        }
        
        let data: serde_json::Value = response.json().await?;
        let compliance = self.parse_regulatory_compliance(data)?;
        
        Ok(compliance)
    }
    
    /// Parse regulatory compliance from API response
    fn parse_regulatory_compliance(&self, data: serde_json::Value) -> Result<Vec<RegulatoryCompliance>> {
        let mut compliance_list = Vec::new();
        
        if let Some(value_array) = data["value"].as_array() {
            for item in value_array {
                let compliance = RegulatoryCompliance {
                    id: item["id"].as_str().unwrap_or("").to_string(),
                    name: item["name"].as_str().unwrap_or("").to_string(),
                    compliance_standard: item["properties"]["complianceStandard"]
                        .as_str().unwrap_or("").to_string(),
                    control_id: item["properties"]["controlId"].as_str().unwrap_or("").to_string(),
                    control_name: item["properties"]["controlName"].as_str().unwrap_or("").to_string(),
                    description: item["properties"]["description"].as_str().unwrap_or("").to_string(),
                    passed_assessments: item["properties"]["passedAssessments"].as_i64().unwrap_or(0) as i32,
                    failed_assessments: item["properties"]["failedAssessments"].as_i64().unwrap_or(0) as i32,
                    skipped_assessments: item["properties"]["skippedAssessments"].as_i64().unwrap_or(0) as i32,
                    compliance_score: item["properties"]["complianceScore"].as_f64().unwrap_or(0.0),
                };
                compliance_list.push(compliance);
            }
        }
        
        Ok(compliance_list)
    }
    
    // Helper methods
    fn parse_severity(&self, severity: &str) -> AlertSeverity {
        match severity.to_lowercase().as_str() {
            "high" => AlertSeverity::High,
            "medium" => AlertSeverity::Medium,
            "low" => AlertSeverity::Low,
            _ => AlertSeverity::Informational,
        }
    }
    
    fn parse_status(&self, status: &str) -> AlertStatus {
        match status.to_lowercase().as_str() {
            "active" => AlertStatus::Active,
            "resolved" => AlertStatus::Resolved,
            "dismissed" => AlertStatus::Dismissed,
            "inprogress" => AlertStatus::InProgress,
            _ => AlertStatus::Active,
        }
    }
    
    fn parse_compliance_status(&self, status: &str) -> ComplianceStatus {
        match status.to_lowercase().as_str() {
            "healthy" => ComplianceStatus::Healthy,
            "unhealthy" => ComplianceStatus::Unhealthy,
            "notapplicable" => ComplianceStatus::NotApplicable,
            _ => ComplianceStatus::Unknown,
        }
    }
    
    fn parse_string_array(&self, value: &serde_json::Value) -> Vec<String> {
        if let Some(array) = value.as_array() {
            array.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_severity() {
        let client = DefenderClient {
            client: Client::new(),
            subscription_id: "test".to_string(),
            tenant_id: "test".to_string(),
            access_token: "test".to_string(),
            base_url: "test".to_string(),
        };
        
        assert_eq!(client.parse_severity("High"), AlertSeverity::High);
        assert_eq!(client.parse_severity("medium"), AlertSeverity::Medium);
        assert_eq!(client.parse_severity("LOW"), AlertSeverity::Low);
        assert_eq!(client.parse_severity("unknown"), AlertSeverity::Informational);
    }
    
    #[test]
    fn test_parse_status() {
        let client = DefenderClient {
            client: Client::new(),
            subscription_id: "test".to_string(),
            tenant_id: "test".to_string(),
            access_token: "test".to_string(),
            base_url: "test".to_string(),
        };
        
        assert_eq!(client.parse_status("Active"), AlertStatus::Active);
        assert_eq!(client.parse_status("resolved"), AlertStatus::Resolved);
        assert_eq!(client.parse_status("DISMISSED"), AlertStatus::Dismissed);
    }
}