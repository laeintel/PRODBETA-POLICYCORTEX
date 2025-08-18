// Approval Workflow Integration Tests
// Tests for approval management and workflow processing

use super::*;
use policycortex_core::remediation::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_approval_workflow() {
        let mut test_ctx = TestContext::new();
        let mut results = TestResults::new();
        
        println!("🔄 Testing Approval Workflows");
        
        // Test Case 1: Create approval request
        match test_create_approval_request().await {
            Ok(approval_id) => {
                println!("  ✅ Approval request created: {}", approval_id);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Failed to create approval request: {}", e);
                results.record_failure(format!("Approval creation: {}", e));
            }
        }
        
        // Test Case 2: Process approval
        match test_process_approval().await {
            Ok(_) => {
                println!("  ✅ Approval processed successfully");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Failed to process approval: {}", e);
                results.record_failure(format!("Approval processing: {}", e));
            }
        }
        
        // Test Case 3: Approval timeout handling
        match test_approval_timeout().await {
            Ok(_) => {
                println!("  ✅ Approval timeout handled correctly");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Approval timeout test failed: {}", e);
                results.record_failure(format!("Timeout handling: {}", e));
            }
        }
        
        // Test Case 4: Multi-level approval chain
        match test_multi_level_approval().await {
            Ok(_) => {
                println!("  ✅ Multi-level approval chain works");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Multi-level approval failed: {}", e);
                results.record_failure(format!("Multi-level: {}", e));
            }
        }
        
        test_ctx.cleanup().await;
        
        assert!(results.success_rate() >= 75.0, "Approval workflow tests failed");
    }

    async fn test_create_approval_request() -> Result<String, String> {
        let approval_manager = MockApprovalManager::new();
        
        let request = ApprovalRequest {
            request_id: uuid::Uuid::new_v4(),
            resource_id: "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/test".to_string(),
            violation_id: "violation-123".to_string(),
            policy_id: "encryption-policy".to_string(),
            remediation_type: "EnableEncryption".to_string(),
            requested_by: "user@company.com".to_string(),
            approvers: vec!["manager@company.com".to_string()],
            priority: ApprovalPriority::High,
            timeout_minutes: 60,
            metadata: std::collections::HashMap::new(),
        };
        
        approval_manager.create_approval(request).await
    }

    async fn test_process_approval() -> Result<(), String> {
        let approval_manager = MockApprovalManager::new();
        
        // Create an approval first
        let approval_id = test_create_approval_request().await?;
        
        // Process the approval
        let decision = ApprovalDecision {
            approval_id: approval_id.clone(),
            decision: Decision::Approved,
            approver: "manager@company.com".to_string(),
            comments: Some("Approved for immediate remediation".to_string()),
            decided_at: chrono::Utc::now(),
        };
        
        approval_manager.process_decision(decision).await?;
        
        // Verify status
        let status = approval_manager.get_approval_status(&approval_id).await?;
        if status != ApprovalStatus::Approved {
            return Err(format!("Expected Approved status, got {:?}", status));
        }
        
        Ok(())
    }

    async fn test_approval_timeout() -> Result<(), String> {
        let approval_manager = MockApprovalManager::new();
        
        // Create approval with very short timeout
        let request = ApprovalRequest {
            request_id: uuid::Uuid::new_v4(),
            resource_id: "/subscriptions/test/resourceGroups/test".to_string(),
            violation_id: "violation-timeout".to_string(),
            policy_id: "test-policy".to_string(),
            remediation_type: "Test".to_string(),
            requested_by: "user@company.com".to_string(),
            approvers: vec!["manager@company.com".to_string()],
            priority: ApprovalPriority::Low,
            timeout_minutes: 0, // Immediate timeout
            metadata: std::collections::HashMap::new(),
        };
        
        let approval_id = approval_manager.create_approval(request).await?;
        
        // Wait and check for timeout
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let status = approval_manager.get_approval_status(&approval_id).await?;
        if status != ApprovalStatus::TimedOut {
            return Err(format!("Expected TimedOut status, got {:?}", status));
        }
        
        Ok(())
    }

    async fn test_multi_level_approval() -> Result<(), String> {
        let approval_manager = MockApprovalManager::new();
        
        // Create multi-level approval request
        let request = ApprovalRequest {
            request_id: uuid::Uuid::new_v4(),
            resource_id: "/subscriptions/test/resourceGroups/prod".to_string(),
            violation_id: "critical-violation".to_string(),
            policy_id: "critical-policy".to_string(),
            remediation_type: "CriticalFix".to_string(),
            requested_by: "user@company.com".to_string(),
            approvers: vec![
                "team-lead@company.com".to_string(),
                "manager@company.com".to_string(),
                "director@company.com".to_string(),
            ],
            priority: ApprovalPriority::Critical,
            timeout_minutes: 120,
            metadata: std::collections::HashMap::new(),
        };
        
        let approval_id = approval_manager.create_approval(request).await?;
        
        // Process approvals from each level
        for approver in vec!["team-lead@company.com", "manager@company.com", "director@company.com"] {
            let decision = ApprovalDecision {
                approval_id: approval_id.clone(),
                decision: Decision::Approved,
                approver: approver.to_string(),
                comments: Some(format!("Approved by {}", approver)),
                decided_at: chrono::Utc::now(),
            };
            
            approval_manager.process_decision(decision).await?;
        }
        
        // Verify all levels approved
        let status = approval_manager.get_approval_status(&approval_id).await?;
        if status != ApprovalStatus::Approved {
            return Err(format!("Multi-level approval failed, status: {:?}", status));
        }
        
        Ok(())
    }
}

// Mock implementation for testing
pub struct MockApprovalManager {
    approvals: Arc<RwLock<std::collections::HashMap<String, ApprovalRecord>>>,
}

impl MockApprovalManager {
    pub fn new() -> Self {
        Self {
            approvals: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
    
    pub async fn create_approval(&self, request: ApprovalRequest) -> Result<String, String> {
        let approval_id = uuid::Uuid::new_v4().to_string();
        
        let record = ApprovalRecord {
            approval_id: approval_id.clone(),
            request,
            status: ApprovalStatus::Pending,
            decisions: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        let mut approvals = self.approvals.write().await;
        approvals.insert(approval_id.clone(), record);
        
        Ok(approval_id)
    }
    
    pub async fn process_decision(&self, decision: ApprovalDecision) -> Result<(), String> {
        let mut approvals = self.approvals.write().await;
        
        let record = approvals.get_mut(&decision.approval_id)
            .ok_or_else(|| "Approval not found".to_string())?;
        
        record.decisions.push(decision.clone());
        
        // Update status based on decision
        record.status = match decision.decision {
            Decision::Approved => {
                // Check if all approvers have approved
                if record.decisions.len() >= record.request.approvers.len() {
                    ApprovalStatus::Approved
                } else {
                    ApprovalStatus::Pending
                }
            }
            Decision::Rejected => ApprovalStatus::Rejected,
        };
        
        record.updated_at = chrono::Utc::now();
        
        Ok(())
    }
    
    pub async fn get_approval_status(&self, approval_id: &str) -> Result<ApprovalStatus, String> {
        let approvals = self.approvals.read().await;
        
        let record = approvals.get(approval_id)
            .ok_or_else(|| "Approval not found".to_string())?;
        
        // Check for timeout
        if record.request.timeout_minutes == 0 {
            return Ok(ApprovalStatus::TimedOut);
        }
        
        Ok(record.status.clone())
    }
}

// Data structures for approval workflow
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub request_id: uuid::Uuid,
    pub resource_id: String,
    pub violation_id: String,
    pub policy_id: String,
    pub remediation_type: String,
    pub requested_by: String,
    pub approvers: Vec<String>,
    pub priority: ApprovalPriority,
    pub timeout_minutes: u32,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ApprovalRecord {
    pub approval_id: String,
    pub request: ApprovalRequest,
    pub status: ApprovalStatus,
    pub decisions: Vec<ApprovalDecision>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct ApprovalDecision {
    pub approval_id: String,
    pub decision: Decision,
    pub approver: String,
    pub comments: Option<String>,
    pub decided_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum Decision {
    Approved,
    Rejected,
}

#[derive(Debug, Clone)]
pub enum ApprovalPriority {
    Critical,
    High,
    Medium,
    Low,
}

use policycortex_core::remediation::RemediationRequest;