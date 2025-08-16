// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// One-Click Remediation System
// Patent 3: Unified AI-Driven Cloud Governance Platform

pub mod workflow_engine;
pub mod approval_manager;
pub mod rollback_manager;
pub mod bulk_remediation;
pub mod arm_executor;
pub mod template_library;
pub mod status_tracker;
pub mod validation_engine;
pub mod notification_system;
pub mod quick_fixes;

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationRequest {
    pub request_id: Uuid,
    pub violation_id: String,
    pub resource_id: String,
    pub resource_type: String,
    pub policy_id: String,
    pub remediation_type: RemediationType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub requested_by: String,
    pub requested_at: DateTime<Utc>,
    pub approval_required: bool,
    pub auto_rollback: bool,
    pub rollback_window_minutes: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationType {
    AutoFix,
    PolicyEnforcement,
    ConfigurationUpdate,
    AccessControl,
    NetworkSecurity,
    Encryption,
    Tagging,
    CostOptimization,
    PerformanceTuning,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationResult {
    pub request_id: Uuid,
    pub status: RemediationStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub execution_time_ms: u64,
    pub changes_applied: Vec<AppliedChange>,
    pub rollback_available: bool,
    pub rollback_token: Option<String>,
    pub error: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RemediationStatus {
    Pending,
    AwaitingApproval,
    Approved,
    InProgress,
    Completed,
    Failed,
    RolledBack,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedChange {
    pub change_id: Uuid,
    pub resource_path: String,
    pub property: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub change_type: ChangeType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Create,
    Update,
    Delete,
    Replace,
    PolicyAssignment,
    RoleAssignment,
    NetworkRule,
    ConfigSetting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationTemplate {
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub violation_types: Vec<String>,
    pub resource_types: Vec<String>,
    pub arm_template: Option<serde_json::Value>,
    pub powershell_script: Option<String>,
    pub azure_cli_commands: Vec<String>,
    pub validation_rules: Vec<ValidationRule>,
    pub rollback_template: Option<serde_json::Value>,
    pub success_criteria: SuccessCriteria,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_id: String,
    pub rule_type: ValidationType,
    pub condition: String,
    pub error_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    PreCondition,
    PostCondition,
    ResourceState,
    PolicyCompliance,
    DependencyCheck,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub compliance_check: bool,
    pub health_check: bool,
    pub performance_check: bool,
    pub custom_validations: Vec<String>,
    pub min_success_percentage: f64,
}

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self {
            compliance_check: true,
            health_check: true,
            performance_check: false,
            custom_validations: Vec::new(),
            min_success_percentage: 100.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkRemediationRequest {
    pub bulk_id: Uuid,
    pub violation_pattern: ViolationPattern,
    pub resources: Vec<String>,
    pub remediation_template: String,
    pub parallel_execution: bool,
    pub max_parallel: usize,
    pub stop_on_error: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPattern {
    pub policy_ids: Vec<String>,
    pub resource_types: Vec<String>,
    pub tags: HashMap<String, String>,
    pub severity: Vec<String>,
    pub age_days: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationWorkflow {
    pub workflow_id: Uuid,
    pub name: String,
    pub steps: Vec<WorkflowStep>,
    pub triggers: Vec<WorkflowTrigger>,
    pub approval_gates: Vec<ApprovalGate>,
    pub rollback_strategy: RollbackStrategy,
    pub notification_config: NotificationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub step_id: String,
    pub name: String,
    pub action: RemediationAction,
    pub conditions: Vec<StepCondition>,
    pub retry_policy: RetryPolicy,
    pub timeout_seconds: u64,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationAction {
    ApplyTemplate(String),
    RunScript(String),
    CallAPI(APICall),
    WaitForApproval,
    ValidateCompliance,
    NotifyStakeholders,
    CreateSnapshot,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APICall {
    pub method: String,
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub body: Option<serde_json::Value>,
    pub expected_status: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepCondition {
    pub condition_type: ConditionType,
    pub expression: String,
    pub on_true: String,  // next step id
    pub on_false: String, // next step id
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    ResourceState,
    PolicyCompliance,
    TimeWindow,
    ApprovalStatus,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff_seconds: u64,
    pub exponential_backoff: bool,
    pub retry_on_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowTrigger {
    PolicyViolation(String),
    Schedule(String), // cron expression
    Manual,
    APIWebhook,
    EventGrid(String),
    AlertRule(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalGate {
    pub gate_id: String,
    pub name: String,
    pub approvers: Vec<Approver>,
    pub approval_type: ApprovalType,
    pub timeout_hours: u64,
    pub auto_approve_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Approver {
    pub approver_type: ApproverType,
    pub identifier: String, // email, group id, etc
    pub notification_method: NotificationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApproverType {
    User,
    Group,
    ServicePrincipal,
    Role,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalType {
    SingleApprover,
    AllApprovers,
    MinimumApprovers(u32),
    Percentage(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationMethod {
    Email,
    Teams,
    Slack,
    Webhook,
    ServiceNow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStrategy {
    pub strategy_type: RollbackType,
    pub trigger_conditions: Vec<RollbackTrigger>,
    pub checkpoint_interval_seconds: u64,
    pub max_rollback_time_seconds: u64,
    pub preserve_logs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackType {
    Automatic,
    Manual,
    Checkpoint,
    Snapshot,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackTrigger {
    OnError,
    OnTimeout,
    OnValidationFailure,
    OnPerformanceDegradation,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub channels: Vec<NotificationChannel>,
    pub events: Vec<NotificationEvent>,
    pub include_details: bool,
    pub include_logs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub channel_type: NotificationMethod,
    pub destination: String,
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationEvent {
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowFailed,
    ApprovalRequired,
    ApprovalReceived,
    RollbackInitiated,
    RollbackCompleted,
    StepCompleted,
    StepFailed,
}

pub trait RemediationEngine {
    async fn execute(&self, request: RemediationRequest) -> Result<RemediationResult, String>;
    async fn validate(&self, request: &RemediationRequest) -> Result<bool, String>;
    async fn rollback(&self, rollback_token: String) -> Result<RemediationResult, String>;
    async fn get_status(&self, request_id: Uuid) -> Result<RemediationStatus, String>;
}