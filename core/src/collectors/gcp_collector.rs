use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AuditLog, CloudCollector, CloudPolicy, CloudProvider, CloudResource, ComplianceStatus, ResourceType};

/// GCP resource collector for multi-cloud parity (stub implementation)
#[derive(Clone)]
pub struct GcpCollector;

impl GcpCollector {
    pub async fn new(_project: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
}

#[async_trait]
impl CloudCollector for GcpCollector {
    async fn collect_resources(&self) -> Result<Vec<CloudResource>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    async fn collect_policies(&self) -> Result<Vec<CloudPolicy>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    async fn collect_audit_logs(&self) -> Result<Vec<AuditLog>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    async fn normalize_data(
        &self,
        data: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        Ok(data)
    }
}
