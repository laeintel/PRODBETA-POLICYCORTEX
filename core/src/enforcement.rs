use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use crate::policy_engine::{Policy, PolicyEvaluation, Resource, Severity, Violation};

/// Enforcement action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementAction {
    pub id: String,
    pub policy_id: String,
    pub resource_id: String,
    pub action_type: EnforcementType,
    pub status: EnforcementStatus,
    pub priority: Priority,
    pub parameters: HashMap<String, Value>,
    pub approval_required: bool,
    pub approved_by: Option<String>,
    pub executed_by: Option<String>,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub executed_at: Option<DateTime<Utc>>,
    pub result: Option<EnforcementResult>,
}

/// Enforcement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementType {
    /// Prevent resource creation/modification
    Block,
    /// Automatically fix non-compliance
    AutoRemediate,
    /// Delete non-compliant resource
    Delete,
    /// Quarantine resource
    Quarantine,
    /// Apply tags
    ApplyTags(HashMap<String, String>),
    /// Modify configuration
    ModifyConfig(HashMap<String, Value>),
    /// Stop/disable resource
    Disable,
    /// Send notification
    Notify(Vec<String>),
    /// Execute webhook
    Webhook(String),
    /// Custom enforcement
    Custom(String),
}

/// Enforcement status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnforcementStatus {
    Pending,
    AwaitingApproval,
    Scheduled,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    RolledBack,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Medium = 2,
    Low = 3,
}

/// Enforcement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementResult {
    pub success: bool,
    pub message: String,
    pub changes_made: Vec<Change>,
    pub rollback_available: bool,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Change made during enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub field: String,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
    pub change_type: ChangeType,
}

/// Change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
}

/// Enforcement path definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementPath {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Vec<EnforcementStep>,
    pub rollback_steps: Vec<EnforcementStep>,
    pub conditions: Vec<EnforcementCondition>,
    pub metadata: HashMap<String, Value>,
}

/// Enforcement step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementStep {
    pub id: String,
    pub name: String,
    pub action: EnforcementType,
    pub parameters: HashMap<String, Value>,
    pub timeout_seconds: u64,
    pub retry_count: u32,
    pub continue_on_error: bool,
}

/// Enforcement condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementCondition {
    pub id: String,
    pub condition_type: ConditionType,
    pub parameters: HashMap<String, Value>,
}

/// Condition types for enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    SeverityThreshold(Severity),
    ResourceType(Vec<String>),
    TimeWindow { start: String, end: String },
    ApprovalRequired,
    Custom(String),
}

/// Enforcement engine trait
#[async_trait]
pub trait EnforcementEngine: Send + Sync {
    /// Create enforcement action from policy evaluation
    async fn create_enforcement_action(
        &self,
        evaluation: &PolicyEvaluation,
        policy: &Policy,
        resource: &Resource,
    ) -> Result<EnforcementAction, String>;

    /// Execute enforcement action
    async fn execute_enforcement(
        &self,
        action: &EnforcementAction,
        resource: &Resource,
    ) -> Result<EnforcementResult, String>;

    /// Rollback enforcement action
    async fn rollback_enforcement(
        &self,
        action: &EnforcementAction,
    ) -> Result<EnforcementResult, String>;

    /// Get enforcement path for violations
    async fn get_enforcement_path(
        &self,
        violations: &[Violation],
        resource: &Resource,
    ) -> Result<EnforcementPath, String>;

    /// Schedule enforcement action
    async fn schedule_enforcement(
        &self,
        action: &EnforcementAction,
        scheduled_at: DateTime<Utc>,
    ) -> Result<(), String>;

    /// Cancel enforcement action
    async fn cancel_enforcement(&self, action_id: &str) -> Result<(), String>;

    /// Get enforcement history
    async fn get_enforcement_history(
        &self,
        resource_id: Option<&str>,
        policy_id: Option<&str>,
    ) -> Result<Vec<EnforcementAction>, String>;
}

/// Default enforcement engine implementation
pub struct DefaultEnforcementEngine {
    enforcer: Box<dyn Enforcer>,
    approval_service: Option<Box<dyn ApprovalService>>,
}

impl DefaultEnforcementEngine {
    pub fn new(
        enforcer: Box<dyn Enforcer>,
        approval_service: Option<Box<dyn ApprovalService>>,
    ) -> Self {
        Self {
            enforcer,
            approval_service,
        }
    }

    /// Determine enforcement type based on violations
    fn determine_enforcement_type(&self, violations: &[Violation]) -> EnforcementType {
        // Check for critical violations
        let has_critical = violations.iter().any(|v| v.severity == Severity::Critical);

        if has_critical {
            // Critical violations require immediate action
            EnforcementType::Block
        } else {
            // Non-critical violations can be auto-remediated
            EnforcementType::AutoRemediate
        }
    }

    /// Check if approval is required
    fn requires_approval(&self, action: &EnforcementType, resource: &Resource) -> bool {
        match action {
            EnforcementType::Delete | EnforcementType::Disable => true,
            EnforcementType::ModifyConfig(_) if resource.provider == "Production" => true,
            _ => false,
        }
    }
}

#[async_trait]
impl EnforcementEngine for DefaultEnforcementEngine {
    async fn create_enforcement_action(
        &self,
        evaluation: &PolicyEvaluation,
        policy: &Policy,
        resource: &Resource,
    ) -> Result<EnforcementAction, String> {
        if evaluation.compliant {
            return Err("Resource is compliant, no enforcement needed".to_string());
        }

        let action_type = self.determine_enforcement_type(&evaluation.violations);
        let approval_required = self.requires_approval(&action_type, resource);

        let action = EnforcementAction {
            id: Uuid::new_v4().to_string(),
            policy_id: policy.id.clone(),
            resource_id: resource.id.clone(),
            action_type,
            status: if approval_required {
                EnforcementStatus::AwaitingApproval
            } else {
                EnforcementStatus::Pending
            },
            priority: self.determine_priority(&evaluation.violations),
            parameters: HashMap::new(),
            approval_required,
            approved_by: None,
            executed_by: None,
            created_at: Utc::now(),
            scheduled_at: None,
            executed_at: None,
            result: None,
        };

        Ok(action)
    }

    async fn execute_enforcement(
        &self,
        action: &EnforcementAction,
        resource: &Resource,
    ) -> Result<EnforcementResult, String> {
        // Check approval if required
        if action.approval_required && action.approved_by.is_none() {
            return Err("Approval required for this enforcement action".to_string());
        }

        let start = std::time::Instant::now();

        // Execute the enforcement
        let result = self.enforcer.enforce(&action.action_type, resource).await?;

        Ok(EnforcementResult {
            success: result.success,
            message: result.message,
            changes_made: result.changes,
            rollback_available: true,
            error: result.error,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn rollback_enforcement(
        &self,
        action: &EnforcementAction,
    ) -> Result<EnforcementResult, String> {
        if action
            .result
            .as_ref()
            .map_or(true, |r| !r.rollback_available)
        {
            return Err("Rollback not available for this action".to_string());
        }

        // Execute rollback
        let result = self.enforcer.rollback(&action.id).await?;

        Ok(result)
    }

    async fn get_enforcement_path(
        &self,
        violations: &[Violation],
        resource: &Resource,
    ) -> Result<EnforcementPath, String> {
        let mut steps = Vec::new();
        let mut rollback_steps = Vec::new();

        // Create enforcement steps based on violations
        for violation in violations {
            let step = EnforcementStep {
                id: Uuid::new_v4().to_string(),
                name: format!("Fix violation: {}", violation.rule_id),
                action: self.determine_enforcement_type(&[violation.clone()]),
                parameters: violation.details.clone(),
                timeout_seconds: 300,
                retry_count: 3,
                continue_on_error: violation.severity != Severity::Critical,
            };

            steps.push(step.clone());

            // Create corresponding rollback step
            rollback_steps.push(EnforcementStep {
                id: Uuid::new_v4().to_string(),
                name: format!("Rollback: {}", step.name),
                action: EnforcementType::Custom("rollback".to_string()),
                parameters: HashMap::new(),
                timeout_seconds: 300,
                retry_count: 1,
                continue_on_error: true,
            });
        }

        // Reverse rollback steps
        rollback_steps.reverse();

        Ok(EnforcementPath {
            id: Uuid::new_v4().to_string(),
            name: format!("Enforcement path for {}", resource.id),
            description: format!(
                "Auto-generated enforcement path for {} violations",
                violations.len()
            ),
            steps,
            rollback_steps,
            conditions: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    async fn schedule_enforcement(
        &self,
        _action: &EnforcementAction,
        _scheduled_at: DateTime<Utc>,
    ) -> Result<(), String> {
        // In production, persist to scheduler
        Ok(())
    }

    async fn cancel_enforcement(&self, _action_id: &str) -> Result<(), String> {
        // In production, cancel in scheduler
        Ok(())
    }

    async fn get_enforcement_history(
        &self,
        _resource_id: Option<&str>,
        _policy_id: Option<&str>,
    ) -> Result<Vec<EnforcementAction>, String> {
        // In production, query from database
        Ok(Vec::new())
    }

    fn determine_priority(&self, violations: &[Violation]) -> Priority {
        violations
            .iter()
            .map(|v| match v.severity {
                Severity::Critical => Priority::Critical,
                Severity::High => Priority::High,
                Severity::Medium => Priority::Medium,
                Severity::Low => Priority::Low,
                Severity::Info => Priority::Low,
            })
            .min()
            .unwrap_or(Priority::Low)
    }
}

/// Enforcer trait for executing enforcement actions
#[async_trait]
pub trait Enforcer: Send + Sync {
    async fn enforce(
        &self,
        action: &EnforcementType,
        resource: &Resource,
    ) -> Result<EnforcerResult, String>;

    async fn rollback(&self, action_id: &str) -> Result<EnforcementResult, String>;
}

/// Enforcer result
#[derive(Debug)]
pub struct EnforcerResult {
    pub success: bool,
    pub message: String,
    pub changes: Vec<Change>,
    pub error: Option<String>,
}

/// Default enforcer implementation
pub struct DefaultEnforcer;

#[async_trait]
impl Enforcer for DefaultEnforcer {
    async fn enforce(
        &self,
        action: &EnforcementType,
        _resource: &Resource,
    ) -> Result<EnforcerResult, String> {
        match action {
            EnforcementType::Block => Ok(EnforcerResult {
                success: true,
                message: "Resource blocked".to_string(),
                changes: Vec::new(),
                error: None,
            }),
            EnforcementType::AutoRemediate => Ok(EnforcerResult {
                success: true,
                message: "Resource auto-remediated".to_string(),
                changes: vec![Change {
                    field: "compliance".to_string(),
                    old_value: Some(Value::Bool(false)),
                    new_value: Some(Value::Bool(true)),
                    change_type: ChangeType::Modified,
                }],
                error: None,
            }),
            EnforcementType::ApplyTags(tags) => {
                let changes = tags
                    .iter()
                    .map(|(k, v)| Change {
                        field: format!("tags.{}", k),
                        old_value: None,
                        new_value: Some(Value::String(v.clone())),
                        change_type: ChangeType::Added,
                    })
                    .collect();

                Ok(EnforcerResult {
                    success: true,
                    message: format!("Applied {} tags", tags.len()),
                    changes,
                    error: None,
                })
            }
            _ => Ok(EnforcerResult {
                success: false,
                message: "Enforcement type not implemented".to_string(),
                changes: Vec::new(),
                error: Some("Not implemented".to_string()),
            }),
        }
    }

    async fn rollback(&self, _action_id: &str) -> Result<EnforcementResult, String> {
        Ok(EnforcementResult {
            success: true,
            message: "Rollback completed".to_string(),
            changes_made: Vec::new(),
            rollback_available: false,
            error: None,
            duration_ms: 100,
        })
    }
}

/// Approval service trait
#[async_trait]
pub trait ApprovalService: Send + Sync {
    async fn request_approval(&self, action: &EnforcementAction) -> Result<String, String>;
    async fn check_approval(&self, approval_id: &str) -> Result<ApprovalStatus, String>;
}

/// Approval status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved(String), // Approver ID
    Rejected(String), // Reason
    Expired,
}
