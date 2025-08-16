// Integration Tests for One-Click Automated Remediation System
// Tests end-to-end workflows including approval, notification, and rollback

pub mod remediation_workflow_tests;
pub mod notification_integration_tests;
pub mod approval_workflow_tests;
pub mod validation_engine_tests;
pub mod performance_tests;

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::Utc;
use std::collections::HashMap;

// Test utilities and mock implementations
pub struct TestContext {
    pub test_id: String,
    pub cleanup_tasks: Vec<Box<dyn Fn() + Send + Sync>>,
}

impl TestContext {
    pub fn new() -> Self {
        Self {
            test_id: Uuid::new_v4().to_string(),
            cleanup_tasks: Vec::new(),
        }
    }

    pub async fn cleanup(&mut self) {
        for task in &self.cleanup_tasks {
            task();
        }
        self.cleanup_tasks.clear();
    }
}

// Mock implementations for testing
pub struct MockAzureClient;
pub struct MockNotificationService;
pub struct MockTemplateStore;

impl MockAzureClient {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn get_resource(&self, resource_id: &str) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "id": resource_id,
            "type": "Microsoft.Storage/storageAccounts",
            "properties": {
                "encryption": {
                    "services": {
                        "blob": { "enabled": false }
                    }
                }
            }
        }))
    }
    
    pub async fn apply_template(&self, _template: &serde_json::Value) -> Result<String, String> {
        Ok("deployment-123".to_string())
    }
}

impl MockNotificationService {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn send_notification(&self, _recipient: &str, _message: &str) -> Result<(), String> {
        // Mock successful notification
        Ok(())
    }
}

impl MockTemplateStore {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn get_template(&self, template_id: &str) -> Result<serde_json::Value, String> {
        match template_id {
            "enable-storage-encryption" => Ok(serde_json::json!({
                "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "parameters": {
                    "storageAccountName": {
                        "type": "string"
                    }
                },
                "resources": [{
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2021-04-01",
                    "name": "[parameters('storageAccountName')]",
                    "properties": {
                        "encryption": {
                            "services": {
                                "blob": { "enabled": true },
                                "file": { "enabled": true }
                            }
                        }
                    }
                }]
            })),
            _ => Err(format!("Template {} not found", template_id))
        }
    }
}

// Test data builders
pub struct RemediationRequestBuilder {
    request: policycortex_core::remediation::RemediationRequest,
}

impl RemediationRequestBuilder {
    pub fn new() -> Self {
        Self {
            request: policycortex_core::remediation::RemediationRequest {
                request_id: Uuid::new_v4(),
                violation_id: "test-violation".to_string(),
                resource_id: "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/teststorage".to_string(),
                resource_type: "Microsoft.Storage/storageAccounts".to_string(),
                policy_id: "storage-encryption-policy".to_string(),
                remediation_type: policycortex_core::remediation::RemediationType::Encryption,
                parameters: HashMap::new(),
                requested_by: "test-user@company.com".to_string(),
                requested_at: Utc::now(),
                approval_required: true,
                auto_rollback: true,
                rollback_window_minutes: 60,
            }
        }
    }
    
    pub fn with_high_risk(mut self) -> Self {
        self.request.approval_required = true;
        self.request.parameters.insert("risk_level".to_string(), serde_json::Value::String("high".to_string()));
        self
    }
    
    pub fn with_auto_rollback(mut self, enabled: bool) -> Self {
        self.request.auto_rollback = enabled;
        self
    }
    
    pub fn build(self) -> policycortex_core::remediation::RemediationRequest {
        self.request
    }
}

// Test result tracking
#[derive(Debug)]
pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub failures: Vec<String>,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: Vec::new(),
        }
    }
    
    pub fn record_pass(&mut self) {
        self.passed += 1;
    }
    
    pub fn record_failure(&mut self, error: String) {
        self.failed += 1;
        self.failures.push(error);
    }
    
    pub fn record_skip(&mut self) {
        self.skipped += 1;
    }
    
    pub fn success_rate(&self) -> f64 {
        let total = self.passed + self.failed;
        if total == 0 {
            0.0
        } else {
            self.passed as f64 / total as f64 * 100.0
        }
    }
}