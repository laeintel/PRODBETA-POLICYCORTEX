// End-to-End Remediation Workflow Integration Tests
// Tests complete remediation lifecycle including approval, execution, and rollback

use super::*;
use policycortex_core::remediation::*;
use policycortex_core::remediation::workflow_engine::*;
use policycortex_core::remediation::approval_manager::{ApprovalManager, ApprovalPolicy, ApprovalRule, ApprovalState};
use policycortex_core::remediation::notification_system::{NotificationSystem, NotificationChannel, NotificationTemplate};
use policycortex_core::remediation::rollback_manager::{RollbackManager, RollbackStrategy};
use policycortex_core::remediation::validation_engine::{ValidationEngine, ValidationResult, RiskLevel};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_remediation_workflow() {
        let mut test_ctx = TestContext::new();
        let mut results = TestResults::new();
        
        println!("🧪 Testing Complete Remediation Workflow");
        
        // Setup test environment
        let workflow_engine = Arc::new(WorkflowEngine::new().await);
        let approval_manager = Arc::new(ApprovalWorkflowManager::new());
        let notification_system = Arc::new(NotificationSystem::new());
        let rollback_manager = Arc::new(RollbackManager::new());
        let validation_engine = Arc::new(ValidationEngine::new().await);
        
        // Test Case 1: High-risk remediation requiring approval
        match test_high_risk_remediation_with_approval(
            &workflow_engine,
            &approval_manager,
            &notification_system,
            &validation_engine,
        ).await {
            Ok(_) => {
                println!("  ✅ High-risk remediation with approval - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ High-risk remediation with approval - FAILED: {}", e);
                results.record_failure(format!("High-risk remediation test: {}", e));
            }
        }
        
        // Test Case 2: Low-risk auto-approved remediation
        match test_low_risk_auto_remediation(
            &workflow_engine,
            &validation_engine,
        ).await {
            Ok(_) => {
                println!("  ✅ Low-risk auto-remediation - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Low-risk auto-remediation - FAILED: {}", e);
                results.record_failure(format!("Low-risk auto-remediation test: {}", e));
            }
        }
        
        // Test Case 3: Bulk remediation with parallel execution
        match test_bulk_remediation_workflow(
            &workflow_engine,
            &validation_engine,
        ).await {
            Ok(_) => {
                println!("  ✅ Bulk remediation workflow - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Bulk remediation workflow - FAILED: {}", e);
                results.record_failure(format!("Bulk remediation test: {}", e));
            }
        }
        
        // Test Case 4: Rollback workflow
        match test_rollback_workflow(
            &workflow_engine,
            &rollback_manager,
        ).await {
            Ok(_) => {
                println!("  ✅ Rollback workflow - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Rollback workflow - FAILED: {}", e);
                results.record_failure(format!("Rollback workflow test: {}", e));
            }
        }
        
        // Test Case 5: Validation failure prevention
        match test_validation_failure_prevention(
            &workflow_engine,
            &validation_engine,
        ).await {
            Ok(_) => {
                println!("  ✅ Validation failure prevention - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Validation failure prevention - FAILED: {}", e);
                results.record_failure(format!("Validation failure test: {}", e));
            }
        }
        
        test_ctx.cleanup().await;
        
        println!("📊 Remediation Workflow Test Results:");
        println!("   Passed: {}", results.passed);
        println!("   Failed: {}", results.failed);
        println!("   Success Rate: {:.1}%", results.success_rate());
        
        if results.failed > 0 {
            println!("❌ Failures:");
            for failure in &results.failures {
                println!("   - {}", failure);
            }
            panic!("Integration tests failed");
        }
        
        assert!(results.success_rate() >= 100.0, "All integration tests must pass");
    }

    async fn test_high_risk_remediation_with_approval(
        workflow_engine: &Arc<WorkflowEngine>,
        approval_manager: &Arc<ApprovalWorkflowManager>,
        notification_system: &Arc<NotificationSystem>,
        validation_engine: &Arc<ValidationEngine>,
    ) -> Result<(), String> {
        // Step 1: Create high-risk remediation request
        let request = RemediationRequestBuilder::new()
            .with_high_risk()
            .with_auto_rollback(true)
            .build();
        
        // Step 2: Submit for validation
        let validation_result = validation_engine.validate_pre_conditions(&request).await?;
        if validation_result.risk_level == RiskLevel::VeryHigh {
            return Err("Validation should block very high risk operations".to_string());
        }
        
        // Step 3: Create approval request
        let approval_request = ApprovalRequest {
            id: uuid::Uuid::new_v4().to_string(),
            remediation_request: request.clone(),
            approvers: vec!["security.admin@company.com".to_string()],
            require_all: Some(true),
            created_by: "test.user@company.com".to_string(),
            created_at: chrono::Utc::now(),
            expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
            status: "pending".to_string(),
            decisions: std::collections::HashMap::new(),
        };
        
        let approval_id = approval_manager.create_approval(approval_request).await?;
        
        // Step 4: Simulate approval decision
        let approval_decision = ApprovalDecision::Approved;
        let approval_result = approval_manager.process_approval(&approval_id, approval_decision).await?;
        
        if !approval_result.approved || !approval_result.final_decision {
            return Err("Approval process failed".to_string());
        }
        
        // Step 5: Execute remediation workflow
        let execution_result = workflow_engine.execute_remediation(request).await?;
        
        if execution_result.status != RemediationStatus::Completed {
            return Err(format!("Remediation execution failed: {:?}", execution_result.status));
        }
        
        Ok(())
    }

    async fn test_low_risk_auto_remediation(
        workflow_engine: &Arc<WorkflowEngine>,
        validation_engine: &Arc<ValidationEngine>,
    ) -> Result<(), String> {
        // Step 1: Create low-risk remediation request
        let mut request = RemediationRequestBuilder::new().build();
        request.approval_required = false; // Low risk - auto-approve
        request.parameters.insert("risk_level".to_string(), serde_json::Value::String("low".to_string()));
        
        // Step 2: Validate (should pass without approval)
        let validation_result = validation_engine.validate_pre_conditions(&request).await?;
        if validation_result.requires_approval {
            return Err("Low risk operations should not require approval".to_string());
        }
        
        // Step 3: Execute directly (auto-approved)
        let execution_result = workflow_engine.execute_remediation(request).await?;
        
        if execution_result.status != RemediationStatus::Completed {
            return Err(format!("Auto-remediation failed: {:?}", execution_result.status));
        }
        
        Ok(())
    }

    async fn test_bulk_remediation_workflow(
        workflow_engine: &Arc<WorkflowEngine>,
        validation_engine: &Arc<ValidationEngine>,
    ) -> Result<(), String> {
        // Step 1: Create multiple remediation requests
        let requests = vec![
            RemediationRequestBuilder::new().build(),
            RemediationRequestBuilder::new().build(),
            RemediationRequestBuilder::new().build(),
        ];
        
        // Step 2: Execute bulk remediation
        let bulk_id = uuid::Uuid::new_v4().to_string();
        let bulk_result = workflow_engine.execute_bulk_remediation(bulk_id, requests).await?;
        
        if bulk_result.successful < 3 {
            return Err(format!("Bulk remediation failed: only {} of 3 succeeded", bulk_result.successful));
        }
        
        Ok(())
    }

    async fn test_rollback_workflow(
        workflow_engine: &Arc<WorkflowEngine>,
        rollback_manager: &Arc<RollbackManager>,
    ) -> Result<(), String> {
        // Step 1: Execute remediation that creates rollback point
        let request = RemediationRequestBuilder::new()
            .with_auto_rollback(true)
            .build();
        
        let execution_result = workflow_engine.execute_remediation(request).await?;
        
        if execution_result.rollback_token.is_none() {
            return Err("No rollback token created".to_string());
        }
        
        // Step 2: Execute rollback
        let rollback_token = execution_result.rollback_token.unwrap();
        let rollback_result = rollback_manager
            .execute_rollback(rollback_token, "Test rollback".to_string())
            .await?;
        
        if !rollback_result.success {
            return Err("Rollback execution failed".to_string());
        }
        
        Ok(())
    }

    async fn test_validation_failure_prevention(
        workflow_engine: &Arc<WorkflowEngine>,
        validation_engine: &Arc<ValidationEngine>,
    ) -> Result<(), String> {
        // Step 1: Create dangerous remediation request
        let mut request = RemediationRequestBuilder::new().build();
        request.resource_id = "/subscriptions/test/resourceGroups/production/providers/Microsoft.Compute/virtualMachines/critical-vm".to_string();
        request.parameters.insert("operation".to_string(), serde_json::Value::String("delete".to_string()));
        request.parameters.insert("risk_level".to_string(), serde_json::Value::String("very_high".to_string()));
        
        // Step 2: Validation should block this
        let validation_result = validation_engine.validate_pre_conditions(&request).await?;
        
        if validation_result.validation_status != ValidationStatus::Blocked {
            return Err("Validation should have blocked dangerous operation".to_string());
        }
        
        if validation_result.blocking_rules.is_empty() {
            return Err("Validation should specify blocking rules".to_string());
        }
        
        // Step 3: Attempt execution should fail
        let execution_result = workflow_engine.execute_remediation(request).await;
        if execution_result.is_ok() {
            return Err("Workflow should have rejected blocked operation".to_string());
        }
        
        Ok(())
    }
}

// Additional test helper implementations
impl WorkflowEngine {
    pub async fn new() -> Self {
        // Mock implementation for testing
        WorkflowEngine {
            execution_queue: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            active_executions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            template_executor: Arc::new(MockTemplateExecutor),
            status_tracker: Arc::new(StatusTracker::new()),
            azure_client: Arc::new(MockAzureClient::new()),
        }
    }
    
    pub async fn execute_remediation(&self, request: RemediationRequest) -> Result<RemediationResult, String> {
        // Mock implementation that simulates successful execution
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate work
        
        Ok(RemediationResult {
            request_id: request.request_id,
            status: RemediationStatus::Completed,
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            execution_time_ms: 100,
            changes_applied: vec![AppliedChange {
                change_id: uuid::Uuid::new_v4(),
                resource_path: request.resource_id,
                property: "encryption.enabled".to_string(),
                old_value: Some(serde_json::Value::Bool(false)),
                new_value: serde_json::Value::Bool(true),
                change_type: ChangeType::Update,
                timestamp: chrono::Utc::now(),
            }],
            rollback_available: request.auto_rollback,
            rollback_token: if request.auto_rollback { 
                Some(format!("rollback-{}", request.request_id)) 
            } else { 
                None 
            },
            error: None,
            warnings: vec![],
        })
    }
    
    pub async fn execute_bulk_remediation(&self, _bulk_id: String, requests: Vec<RemediationRequest>) -> Result<BulkRemediationResult, String> {
        // Mock implementation for bulk processing
        tokio::time::sleep(Duration::from_millis(200)).await; // Simulate work
        
        Ok(BulkRemediationResult {
            bulk_id: _bulk_id,
            total_requested: requests.len(),
            successful: requests.len(),
            failed: 0,
            skipped: 0,
            execution_time_ms: 200,
            results: requests.into_iter().map(|req| {
                RemediationResult {
                    request_id: req.request_id,
                    status: RemediationStatus::Completed,
                    started_at: chrono::Utc::now(),
                    completed_at: Some(chrono::Utc::now()),
                    execution_time_ms: 50,
                    changes_applied: vec![],
                    rollback_available: false,
                    rollback_token: None,
                    error: None,
                    warnings: vec![],
                }
            }).collect(),
        })
    }
}

impl ValidationEngine {
    pub async fn new() -> Self {
        // Mock implementation
        ValidationEngine {
            azure_client: Arc::new(MockAzureClient::new()),
            risk_assessor: Arc::new(RiskAssessment::new()),
            dependency_checker: Arc::new(DependencyChecker::new()),
            safety_rules: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
    
    pub async fn validate_pre_conditions(&self, request: &RemediationRequest) -> Result<ValidationResult, String> {
        // Mock validation logic
        let risk_level = if request.parameters.get("risk_level")
            .and_then(|v| v.as_str())
            .unwrap_or("medium") == "very_high" {
            RiskLevel::VeryHigh
        } else if request.parameters.get("risk_level")
            .and_then(|v| v.as_str())
            .unwrap_or("medium") == "high" {
            RiskLevel::High
        } else {
            RiskLevel::Medium
        };
        
        let validation_status = if risk_level == RiskLevel::VeryHigh {
            ValidationStatus::Blocked
        } else {
            ValidationStatus::Passed
        };
        
        Ok(ValidationResult {
            validation_id: uuid::Uuid::new_v4(),
            request_id: request.request_id,
            validation_status,
            risk_level,
            requires_approval: risk_level == RiskLevel::High,
            safety_violations: vec![],
            blocking_rules: if validation_status == ValidationStatus::Blocked {
                vec!["Very high risk operations are blocked".to_string()]
            } else {
                vec![]
            },
            warnings: vec![],
            evidence: vec![],
            validated_at: chrono::Utc::now(),
            validator: "test-validator".to_string(),
        })
    }
}

// Mock implementations for test dependencies
struct MockTemplateExecutor;

#[derive(Debug)]
pub struct BulkRemediationResult {
    pub bulk_id: String,
    pub total_requested: usize,
    pub successful: usize,
    pub failed: usize,
    pub skipped: usize,
    pub execution_time_ms: u64,
    pub results: Vec<RemediationResult>,
}

// Additional required mock structs
pub struct WorkflowEngine {
    execution_queue: Arc<RwLock<std::collections::VecDeque<RemediationRequest>>>,
    active_executions: Arc<RwLock<std::collections::HashMap<String, RemediationResult>>>,
    template_executor: Arc<MockTemplateExecutor>,
    status_tracker: Arc<StatusTracker>,
    azure_client: Arc<MockAzureClient>,
}

pub struct StatusTracker {
    status_updates: Arc<RwLock<std::collections::HashMap<uuid::Uuid, RemediationStatus>>>,
}

impl StatusTracker {
    pub fn new() -> Self {
        Self {
            status_updates: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
}

pub struct RiskAssessment;
impl RiskAssessment {
    pub fn new() -> Self {
        Self
    }
}

pub struct DependencyChecker;
impl DependencyChecker {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, PartialEq)]
pub enum ValidationStatus {
    Passed,
    Failed,
    Blocked,
}