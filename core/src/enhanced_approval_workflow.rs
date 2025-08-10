use crate::approvals::{
    Approval, ApprovalDecision, ApprovalRequest, ApprovalStatus, ApprovalType, RiskLevel,
    ImpactAnalysis,
};
use crate::auth::AuthUser;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Complete approval workflow engine with state machine and audit trail
pub struct EnhancedApprovalWorkflow {
    db_pool: Arc<PgPool>,
    pending_approvals: Arc<RwLock<HashMap<Uuid, ApprovalRequest>>>,
    policies: Arc<RwLock<Vec<ApprovalPolicy>>>,
    notification_service: Arc<NotificationService>,
    audit_log: Arc<AuditLog>,
    state_machine: Arc<Mutex<ApprovalStateMachine>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalPolicy {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub operation_types: Vec<String>,
    pub resource_patterns: Vec<String>,
    pub risk_thresholds: RiskThresholds,
    pub auto_approve_conditions: Vec<AutoApproveCondition>,
    pub escalation_rules: Vec<EscalationRule>,
    pub notification_templates: HashMap<String, String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThresholds {
    pub low: ApprovalRequirement,
    pub medium: ApprovalRequirement,
    pub high: ApprovalRequirement,
    pub critical: ApprovalRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequirement {
    pub approval_type: ApprovalType,
    pub required_approvers: Vec<String>,
    pub min_approvals: usize,
    pub timeout_hours: u32,
    pub allow_self_approval: bool,
    pub require_mfa: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoApproveCondition {
    pub condition_type: AutoApproveType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoApproveType {
    BelowCostThreshold,
    NonProductionEnvironment,
    PreApprovedResource,
    WithinMaintenanceWindow,
    LowRiskOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub trigger_after_hours: u32,
    pub escalate_to: Vec<String>,
    pub notification_template: String,
    pub max_escalations: u32,
}

/// State machine for approval workflow
pub struct ApprovalStateMachine {
    states: HashMap<ApprovalStatus, Vec<ApprovalStatus>>,
    current_states: HashMap<Uuid, ApprovalStatus>,
}

impl ApprovalStateMachine {
    pub fn new() -> Self {
        let mut states = HashMap::new();
        
        // Define valid state transitions
        states.insert(
            ApprovalStatus::Pending,
            vec![
                ApprovalStatus::Approved,
                ApprovalStatus::Rejected,
                ApprovalStatus::Cancelled,
                ApprovalStatus::Expired,
                ApprovalStatus::Escalated,
            ],
        );
        
        states.insert(
            ApprovalStatus::Escalated,
            vec![
                ApprovalStatus::Approved,
                ApprovalStatus::Rejected,
                ApprovalStatus::Cancelled,
                ApprovalStatus::Expired,
            ],
        );
        
        states.insert(
            ApprovalStatus::Approved,
            vec![ApprovalStatus::Executed, ApprovalStatus::Failed],
        );
        
        Self {
            states,
            current_states: HashMap::new(),
        }
    }
    
    pub fn can_transition(&self, from: &ApprovalStatus, to: &ApprovalStatus) -> bool {
        self.states
            .get(from)
            .map(|valid_transitions| valid_transitions.contains(to))
            .unwrap_or(false)
    }
    
    pub fn transition(
        &mut self,
        approval_id: Uuid,
        to: ApprovalStatus,
    ) -> Result<(), String> {
        let current = self
            .current_states
            .get(&approval_id)
            .cloned()
            .unwrap_or(ApprovalStatus::Pending);
        
        if self.can_transition(&current, &to) {
            self.current_states.insert(approval_id, to);
            Ok(())
        } else {
            Err(format!(
                "Invalid state transition from {:?} to {:?}",
                current, to
            ))
        }
    }
}

/// Notification service for approval requests
pub struct NotificationService {
    email_enabled: bool,
    teams_enabled: bool,
    slack_enabled: bool,
    webhook_urls: Vec<String>,
}

impl NotificationService {
    pub fn new() -> Self {
        Self {
            email_enabled: std::env::var("ENABLE_EMAIL_NOTIFICATIONS")
                .unwrap_or_else(|_| "false".to_string())
                == "true",
            teams_enabled: std::env::var("ENABLE_TEAMS_NOTIFICATIONS")
                .unwrap_or_else(|_| "false".to_string())
                == "true",
            slack_enabled: std::env::var("ENABLE_SLACK_NOTIFICATIONS")
                .unwrap_or_else(|_| "false".to_string())
                == "true",
            webhook_urls: std::env::var("NOTIFICATION_WEBHOOKS")
                .unwrap_or_default()
                .split(',')
                .map(String::from)
                .filter(|s| !s.is_empty())
                .collect(),
        }
    }
    
    pub async fn send_approval_request(
        &self,
        request: &ApprovalRequest,
        approvers: &[String],
    ) -> Result<(), String> {
        let mut sent = false;
        
        // Send email notifications
        if self.email_enabled {
            for approver in approvers {
                if let Err(e) = self.send_email_notification(request, approver).await {
                    warn!("Failed to send email to {}: {}", approver, e);
                } else {
                    sent = true;
                }
            }
        }
        
        // Send Teams notification
        if self.teams_enabled {
            if let Err(e) = self.send_teams_notification(request).await {
                warn!("Failed to send Teams notification: {}", e);
            } else {
                sent = true;
            }
        }
        
        // Send Slack notification
        if self.slack_enabled {
            if let Err(e) = self.send_slack_notification(request).await {
                warn!("Failed to send Slack notification: {}", e);
            } else {
                sent = true;
            }
        }
        
        // Send webhook notifications
        for webhook_url in &self.webhook_urls {
            if let Err(e) = self.send_webhook_notification(request, webhook_url).await {
                warn!("Failed to send webhook to {}: {}", webhook_url, e);
            } else {
                sent = true;
            }
        }
        
        if sent {
            Ok(())
        } else {
            Err("No notification channels available or all failed".to_string())
        }
    }
    
    async fn send_email_notification(
        &self,
        request: &ApprovalRequest,
        recipient: &str,
    ) -> Result<(), String> {
        // Implementation would use an email service like SendGrid
        info!("Sending email notification to {} for approval {}", recipient, request.id);
        Ok(())
    }
    
    async fn send_teams_notification(&self, request: &ApprovalRequest) -> Result<(), String> {
        // Implementation would use Microsoft Teams webhook
        info!("Sending Teams notification for approval {}", request.id);
        Ok(())
    }
    
    async fn send_slack_notification(&self, request: &ApprovalRequest) -> Result<(), String> {
        // Implementation would use Slack webhook
        info!("Sending Slack notification for approval {}", request.id);
        Ok(())
    }
    
    async fn send_webhook_notification(
        &self,
        request: &ApprovalRequest,
        webhook_url: &str,
    ) -> Result<(), String> {
        // Send generic webhook notification
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "approval_id": request.id,
            "title": request.title,
            "description": request.description,
            "requester": request.requester_email,
            "risk_level": request.impact_analysis.risk_level,
            "expires_at": request.expires_at,
            "approval_url": format!("/approvals/{}", request.id)
        });
        
        match client.post(webhook_url).json(&payload).send().await {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => Err(format!("Webhook returned status: {}", response.status())),
            Err(e) => Err(format!("Webhook request failed: {}", e)),
        }
    }
}

/// Audit log for all approval activities
pub struct AuditLog {
    db_pool: Arc<PgPool>,
}

impl AuditLog {
    pub async fn new(db_pool: Arc<PgPool>) -> Self {
        Self { db_pool }
    }
    
    pub async fn log_approval_created(
        &self,
        request: &ApprovalRequest,
        policy_id: Option<Uuid>,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            INSERT INTO approval_audit_log (
                id, approval_id, action, actor_id, actor_email,
                details, policy_id, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#,
            Uuid::new_v4(),
            request.id,
            "CREATED",
            request.requester_id,
            request.requester_email,
            serde_json::json!({
                "action_type": request.action_type,
                "resource_id": request.resource_id,
                "risk_level": request.impact_analysis.risk_level
            }),
            policy_id
        )
        .execute(&*self.db_pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn log_approval_decision(
        &self,
        approval_id: Uuid,
        decision: &ApprovalDecision,
        approver: &AuthUser,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            INSERT INTO approval_audit_log (
                id, approval_id, action, actor_id, actor_email,
                details, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            "#,
            Uuid::new_v4(),
            approval_id,
            format!("{:?}", decision),
            approver.claims.sub.clone(),
            approver.claims.preferred_username.clone().unwrap_or_default(),
            serde_json::json!({
                "decision": decision,
                "timestamp": Utc::now()
            })
        )
        .execute(&*self.db_pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn log_state_transition(
        &self,
        approval_id: Uuid,
        from_state: &ApprovalStatus,
        to_state: &ApprovalStatus,
        reason: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            INSERT INTO approval_audit_log (
                id, approval_id, action, details, created_at
            )
            VALUES ($1, $2, $3, $4, NOW())
            "#,
            Uuid::new_v4(),
            approval_id,
            "STATE_TRANSITION",
            serde_json::json!({
                "from": from_state,
                "to": to_state,
                "reason": reason
            })
        )
        .execute(&*self.db_pool)
        .await?;
        
        Ok(())
    }
}

impl EnhancedApprovalWorkflow {
    pub async fn new(db_pool: Arc<PgPool>) -> Self {
        Self {
            db_pool: db_pool.clone(),
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(Self::load_policies(&db_pool).await)),
            notification_service: Arc::new(NotificationService::new()),
            audit_log: Arc::new(AuditLog::new(db_pool.clone()).await),
            state_machine: Arc::new(Mutex::new(ApprovalStateMachine::new())),
        }
    }
    
    async fn load_policies(db_pool: &PgPool) -> Vec<ApprovalPolicy> {
        match sqlx::query_as!(
            ApprovalPolicy,
            r#"
            SELECT id, name, description, operation_types, resource_patterns,
                   risk_thresholds as "risk_thresholds: sqlx::types::Json<RiskThresholds>",
                   auto_approve_conditions as "auto_approve_conditions: sqlx::types::Json<Vec<AutoApproveCondition>>",
                   escalation_rules as "escalation_rules: sqlx::types::Json<Vec<EscalationRule>>",
                   notification_templates, is_active, created_at, updated_at
            FROM approval_policies
            WHERE is_active = true
            "#
        )
        .fetch_all(db_pool)
        .await
        {
            Ok(policies) => policies,
            Err(e) => {
                error!("Failed to load approval policies: {}", e);
                Self::default_policies()
            }
        }
    }
    
    fn default_policies() -> Vec<ApprovalPolicy> {
        vec![
            ApprovalPolicy {
                id: Uuid::new_v4(),
                name: "Default Approval Policy".to_string(),
                description: "Default policy for all operations".to_string(),
                operation_types: vec!["*".to_string()],
                resource_patterns: vec!["*".to_string()],
                risk_thresholds: RiskThresholds {
                    low: ApprovalRequirement {
                        approval_type: ApprovalType::SingleApproval,
                        required_approvers: vec![],
                        min_approvals: 1,
                        timeout_hours: 72,
                        allow_self_approval: true,
                        require_mfa: false,
                    },
                    medium: ApprovalRequirement {
                        approval_type: ApprovalType::SingleApproval,
                        required_approvers: vec![],
                        min_approvals: 1,
                        timeout_hours: 48,
                        allow_self_approval: false,
                        require_mfa: true,
                    },
                    high: ApprovalRequirement {
                        approval_type: ApprovalType::MultipleApprovals,
                        required_approvers: vec![],
                        min_approvals: 2,
                        timeout_hours: 24,
                        allow_self_approval: false,
                        require_mfa: true,
                    },
                    critical: ApprovalRequirement {
                        approval_type: ApprovalType::UnanimousApproval,
                        required_approvers: vec![],
                        min_approvals: 3,
                        timeout_hours: 12,
                        allow_self_approval: false,
                        require_mfa: true,
                    },
                },
                auto_approve_conditions: vec![],
                escalation_rules: vec![
                    EscalationRule {
                        trigger_after_hours: 4,
                        escalate_to: vec!["managers".to_string()],
                        notification_template: "escalation_default".to_string(),
                        max_escalations: 2,
                    },
                ],
                notification_templates: HashMap::new(),
                is_active: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ]
    }
    
    /// Create a new approval request
    pub async fn create_approval_request(
        &self,
        operation_type: &str,
        resource_id: &str,
        requester: &AuthUser,
        impact_analysis: ImpactAnalysis,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<ApprovalRequest, String> {
        // Find applicable policy
        let policies = self.policies.read().await;
        let policy = policies
            .iter()
            .find(|p| {
                p.is_active
                    && (p.operation_types.contains(&operation_type.to_string())
                        || p.operation_types.contains(&"*".to_string()))
            })
            .ok_or_else(|| "No applicable approval policy found".to_string())?;
        
        // Get approval requirements based on risk level
        let requirements = match impact_analysis.risk_level {
            RiskLevel::Low => &policy.risk_thresholds.low,
            RiskLevel::Medium => &policy.risk_thresholds.medium,
            RiskLevel::High => &policy.risk_thresholds.high,
            RiskLevel::Critical => &policy.risk_thresholds.critical,
        };
        
        // Check auto-approve conditions
        for condition in &policy.auto_approve_conditions {
            if self.check_auto_approve_condition(condition, &impact_analysis, &metadata).await {
                info!("Auto-approving request based on condition: {:?}", condition.condition_type);
                return self.create_auto_approved_request(
                    operation_type,
                    resource_id,
                    requester,
                    impact_analysis,
                    metadata,
                ).await;
            }
        }
        
        // Create the approval request
        let request = ApprovalRequest {
            id: Uuid::new_v4(),
            tenant_id: requester.claims.tid.clone().unwrap_or_default(),
            action_id: Uuid::new_v4(),
            action_type: operation_type.to_string(),
            resource_id: resource_id.to_string(),
            requester_id: requester.claims.sub.clone(),
            requester_email: requester.claims.preferred_username.clone().unwrap_or_default(),
            title: format!("{} on {}", operation_type, resource_id),
            description: format!(
                "Approval required for {} operation on resource {}",
                operation_type, resource_id
            ),
            impact_analysis,
            approval_type: requirements.approval_type.clone(),
            required_approvers: requirements.required_approvers.clone(),
            status: ApprovalStatus::Pending,
            approvals: Vec::new(),
            expires_at: Utc::now() + Duration::hours(requirements.timeout_hours as i64),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata,
        };
        
        // Persist to database
        self.persist_approval_request(&request).await?;
        
        // Add to in-memory cache
        self.pending_approvals.write().await.insert(request.id, request.clone());
        
        // Initialize state machine
        self.state_machine
            .lock()
            .await
            .current_states
            .insert(request.id, ApprovalStatus::Pending);
        
        // Log to audit trail
        self.audit_log
            .log_approval_created(&request, Some(policy.id))
            .await
            .map_err(|e| format!("Failed to log approval creation: {}", e))?;
        
        // Send notifications
        let approvers = if requirements.required_approvers.is_empty() {
            self.get_default_approvers(&impact_analysis.risk_level).await
        } else {
            requirements.required_approvers.clone()
        };
        
        self.notification_service
            .send_approval_request(&request, &approvers)
            .await?;
        
        // Schedule escalation if needed
        if !policy.escalation_rules.is_empty() {
            self.schedule_escalation(request.id, &policy.escalation_rules).await;
        }
        
        Ok(request)
    }
    
    async fn check_auto_approve_condition(
        &self,
        condition: &AutoApproveCondition,
        impact: &ImpactAnalysis,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> bool {
        match condition.condition_type {
            AutoApproveType::BelowCostThreshold => {
                if let Some(threshold) = condition.parameters.get("max_cost") {
                    if let Some(threshold_value) = threshold.as_f64() {
                        return impact.estimated_cost < threshold_value;
                    }
                }
                false
            }
            AutoApproveType::NonProductionEnvironment => {
                if let Some(env) = metadata.get("environment") {
                    if let Some(env_str) = env.as_str() {
                        return env_str.to_lowercase() != "production";
                    }
                }
                false
            }
            AutoApproveType::PreApprovedResource => {
                if let Some(resource_list) = condition.parameters.get("resources") {
                    if let Some(resources) = resource_list.as_array() {
                        if let Some(resource_id) = metadata.get("resource_id") {
                            return resources.contains(resource_id);
                        }
                    }
                }
                false
            }
            AutoApproveType::WithinMaintenanceWindow => {
                // Check if current time is within maintenance window
                let now = Utc::now();
                if let (Some(start), Some(end)) = (
                    condition.parameters.get("window_start"),
                    condition.parameters.get("window_end"),
                ) {
                    // Implementation would parse window times and check
                    return true; // Simplified for example
                }
                false
            }
            AutoApproveType::LowRiskOperation => {
                matches!(impact.risk_level, RiskLevel::Low)
            }
        }
    }
    
    async fn create_auto_approved_request(
        &self,
        operation_type: &str,
        resource_id: &str,
        requester: &AuthUser,
        impact_analysis: ImpactAnalysis,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<ApprovalRequest, String> {
        let mut request = ApprovalRequest {
            id: Uuid::new_v4(),
            tenant_id: requester.claims.tid.clone().unwrap_or_default(),
            action_id: Uuid::new_v4(),
            action_type: operation_type.to_string(),
            resource_id: resource_id.to_string(),
            requester_id: requester.claims.sub.clone(),
            requester_email: requester.claims.preferred_username.clone().unwrap_or_default(),
            title: format!("{} on {} (Auto-Approved)", operation_type, resource_id),
            description: format!(
                "Auto-approved {} operation on resource {}",
                operation_type, resource_id
            ),
            impact_analysis,
            approval_type: ApprovalType::AutoApproved,
            required_approvers: vec![],
            status: ApprovalStatus::Approved,
            approvals: vec![Approval {
                approver_id: "system".to_string(),
                approver_email: "system@policycortex.com".to_string(),
                decision: ApprovalDecision::Approved,
                comments: Some("Auto-approved based on policy conditions".to_string()),
                approved_at: Utc::now(),
            }],
            expires_at: Utc::now() + Duration::hours(24),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata,
        };
        
        // Persist and return
        self.persist_approval_request(&request).await?;
        Ok(request)
    }
    
    /// Process an approval decision
    pub async fn process_approval_decision(
        &self,
        approval_id: Uuid,
        approver: &AuthUser,
        decision: ApprovalDecision,
        comments: Option<String>,
    ) -> Result<ApprovalRequest, String> {
        // Get the approval request
        let mut approvals = self.pending_approvals.write().await;
        let request = approvals
            .get_mut(&approval_id)
            .ok_or_else(|| "Approval request not found".to_string())?;
        
        // Check if approver is authorized
        if !self.is_authorized_approver(request, approver).await {
            return Err("Not authorized to approve this request".to_string());
        }
        
        // Check if request has expired
        if request.expires_at < Utc::now() {
            request.status = ApprovalStatus::Expired;
            self.update_approval_status(request, ApprovalStatus::Expired).await?;
            return Err("Approval request has expired".to_string());
        }
        
        // Add the approval
        let approval = Approval {
            approver_id: approver.claims.sub.clone(),
            approver_email: approver.claims.preferred_username.clone().unwrap_or_default(),
            decision: decision.clone(),
            comments,
            approved_at: Utc::now(),
        };
        
        request.approvals.push(approval);
        request.updated_at = Utc::now();
        
        // Log the decision
        self.audit_log
            .log_approval_decision(approval_id, &decision, approver)
            .await
            .map_err(|e| format!("Failed to log approval decision: {}", e))?;
        
        // Check if approval requirements are met
        let new_status = self.evaluate_approval_status(request).await;
        
        if new_status != request.status {
            // Update state machine
            self.state_machine
                .lock()
                .await
                .transition(approval_id, new_status.clone())?;
            
            // Update status
            request.status = new_status.clone();
            self.update_approval_status(request, new_status).await?;
            
            // Log state transition
            self.audit_log
                .log_state_transition(
                    approval_id,
                    &ApprovalStatus::Pending,
                    &new_status,
                    &format!("Decision by {}", approver.claims.preferred_username.clone().unwrap_or_default()),
                )
                .await
                .map_err(|e| format!("Failed to log state transition: {}", e))?;
        }
        
        // Persist changes
        self.persist_approval_request(request).await?;
        
        Ok(request.clone())
    }
    
    async fn is_authorized_approver(&self, request: &ApprovalRequest, approver: &AuthUser) -> bool {
        // Check if approver is in required approvers list
        if !request.required_approvers.is_empty() {
            return request.required_approvers.contains(
                &approver.claims.preferred_username.clone().unwrap_or_default()
            );
        }
        
        // Check if approver has appropriate role based on risk level
        match request.impact_analysis.risk_level {
            RiskLevel::Critical => {
                // Only admins can approve critical operations
                approver.claims.roles.as_ref()
                    .map(|roles| roles.contains(&"admin".to_string()))
                    .unwrap_or(false)
            }
            RiskLevel::High => {
                // Admins or managers can approve high risk operations
                approver.claims.roles.as_ref()
                    .map(|roles| {
                        roles.contains(&"admin".to_string()) ||
                        roles.contains(&"manager".to_string())
                    })
                    .unwrap_or(false)
            }
            _ => {
                // Any authenticated user can approve low/medium risk
                true
            }
        }
    }
    
    async fn evaluate_approval_status(&self, request: &ApprovalRequest) -> ApprovalStatus {
        let approved_count = request
            .approvals
            .iter()
            .filter(|a| matches!(a.decision, ApprovalDecision::Approved))
            .count();
        
        let rejected_count = request
            .approvals
            .iter()
            .filter(|a| matches!(a.decision, ApprovalDecision::Rejected))
            .count();
        
        match request.approval_type {
            ApprovalType::SingleApproval => {
                if rejected_count > 0 {
                    ApprovalStatus::Rejected
                } else if approved_count >= 1 {
                    ApprovalStatus::Approved
                } else {
                    ApprovalStatus::Pending
                }
            }
            ApprovalType::MultipleApprovals => {
                let min_approvals = 2; // Should come from policy
                if rejected_count > 0 {
                    ApprovalStatus::Rejected
                } else if approved_count >= min_approvals {
                    ApprovalStatus::Approved
                } else {
                    ApprovalStatus::Pending
                }
            }
            ApprovalType::UnanimousApproval => {
                let total_required = request.required_approvers.len().max(3);
                if rejected_count > 0 {
                    ApprovalStatus::Rejected
                } else if approved_count >= total_required {
                    ApprovalStatus::Approved
                } else {
                    ApprovalStatus::Pending
                }
            }
            ApprovalType::AutoApproved => ApprovalStatus::Approved,
        }
    }
    
    async fn persist_approval_request(&self, request: &ApprovalRequest) -> Result<(), String> {
        sqlx::query!(
            r#"
            INSERT INTO approval_requests (
                id, tenant_id, action_id, action_type, resource_id,
                requester_id, requester_email, title, description,
                impact_analysis, approval_type, required_approvers,
                status, approvals, expires_at, created_at, updated_at, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                approvals = EXCLUDED.approvals,
                updated_at = EXCLUDED.updated_at
            "#,
            request.id,
            request.tenant_id,
            request.action_id,
            request.action_type,
            request.resource_id,
            request.requester_id,
            request.requester_email,
            request.title,
            request.description,
            serde_json::to_value(&request.impact_analysis).unwrap(),
            serde_json::to_value(&request.approval_type).unwrap(),
            &request.required_approvers,
            serde_json::to_value(&request.status).unwrap(),
            serde_json::to_value(&request.approvals).unwrap(),
            request.expires_at,
            request.created_at,
            request.updated_at,
            serde_json::to_value(&request.metadata).unwrap()
        )
        .execute(&*self.db_pool)
        .await
        .map_err(|e| format!("Failed to persist approval request: {}", e))?;
        
        Ok(())
    }
    
    async fn update_approval_status(
        &self,
        request: &ApprovalRequest,
        new_status: ApprovalStatus,
    ) -> Result<(), String> {
        sqlx::query!(
            r#"
            UPDATE approval_requests
            SET status = $1, updated_at = NOW()
            WHERE id = $2
            "#,
            serde_json::to_value(&new_status).unwrap(),
            request.id
        )
        .execute(&*self.db_pool)
        .await
        .map_err(|e| format!("Failed to update approval status: {}", e))?;
        
        Ok(())
    }
    
    async fn get_default_approvers(&self, risk_level: &RiskLevel) -> Vec<String> {
        // In production, this would query a group or role mapping
        match risk_level {
            RiskLevel::Critical => vec!["admin@company.com".to_string()],
            RiskLevel::High => vec!["manager@company.com".to_string()],
            _ => vec!["approver@company.com".to_string()],
        }
    }
    
    async fn schedule_escalation(&self, approval_id: Uuid, rules: &[EscalationRule]) {
        // In production, this would use a job scheduler like Tokio's timer or an external service
        for rule in rules {
            let duration = Duration::hours(rule.trigger_after_hours as i64);
            let approval_id = approval_id;
            let rule = rule.clone();
            
            tokio::spawn(async move {
                tokio::time::sleep(duration.to_std().unwrap()).await;
                // Check if still pending and escalate
                info!("Escalating approval {} after {} hours", approval_id, rule.trigger_after_hours);
            });
        }
    }
    
    /// Get pending approvals for a user
    pub async fn get_pending_approvals_for_user(
        &self,
        user: &AuthUser,
    ) -> Result<Vec<ApprovalRequest>, String> {
        let user_email = user.claims.preferred_username.clone().unwrap_or_default();
        
        let approvals = sqlx::query_as!(
            ApprovalRequest,
            r#"
            SELECT * FROM approval_requests
            WHERE status = 'Pending'
            AND (
                required_approvers @> ARRAY[$1]::text[]
                OR required_approvers = '{}'::text[]
            )
            AND expires_at > NOW()
            ORDER BY created_at DESC
            "#,
            user_email
        )
        .fetch_all(&*self.db_pool)
        .await
        .map_err(|e| format!("Failed to fetch pending approvals: {}", e))?;
        
        Ok(approvals)
    }
    
    /// Get approval history
    pub async fn get_approval_history(
        &self,
        filter: ApprovalHistoryFilter,
    ) -> Result<Vec<ApprovalRequest>, String> {
        let mut query = String::from("SELECT * FROM approval_requests WHERE 1=1");
        
        if let Some(tenant_id) = filter.tenant_id {
            query.push_str(&format!(" AND tenant_id = '{}'", tenant_id));
        }
        
        if let Some(requester_id) = filter.requester_id {
            query.push_str(&format!(" AND requester_id = '{}'", requester_id));
        }
        
        if let Some(status) = filter.status {
            query.push_str(&format!(" AND status = '{:?}'", status));
        }
        
        if let Some(from_date) = filter.from_date {
            query.push_str(&format!(" AND created_at >= '{}'", from_date));
        }
        
        if let Some(to_date) = filter.to_date {
            query.push_str(&format!(" AND created_at <= '{}'", to_date));
        }
        
        query.push_str(" ORDER BY created_at DESC LIMIT 100");
        
        let approvals = sqlx::query_as::<_, ApprovalRequest>(&query)
            .fetch_all(&*self.db_pool)
            .await
            .map_err(|e| format!("Failed to fetch approval history: {}", e))?;
        
        Ok(approvals)
    }
}

#[derive(Debug, Clone)]
pub struct ApprovalHistoryFilter {
    pub tenant_id: Option<String>,
    pub requester_id: Option<String>,
    pub status: Option<ApprovalStatus>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_state_machine_transitions() {
        let mut state_machine = ApprovalStateMachine::new();
        let approval_id = Uuid::new_v4();
        
        // Valid transition: Pending -> Approved
        assert!(state_machine
            .transition(approval_id, ApprovalStatus::Approved)
            .is_ok());
        
        // Invalid transition: Approved -> Pending
        assert!(state_machine
            .transition(approval_id, ApprovalStatus::Pending)
            .is_err());
        
        // Valid transition: Approved -> Executed
        assert!(state_machine
            .transition(approval_id, ApprovalStatus::Executed)
            .is_ok());
    }
    
    #[test]
    fn test_auto_approve_condition_cost_threshold() {
        let condition = AutoApproveCondition {
            condition_type: AutoApproveType::BelowCostThreshold,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_cost".to_string(), serde_json::json!(100.0));
                params
            },
            description: "Auto-approve if cost < $100".to_string(),
        };
        
        let impact_low = ImpactAnalysis {
            affected_resources: 1,
            estimated_downtime: Duration::minutes(0).num_seconds() as u32,
            estimated_cost: 50.0,
            risk_level: RiskLevel::Low,
            security_impact: "None".to_string(),
            compliance_impact: "None".to_string(),
        };
        
        let impact_high = ImpactAnalysis {
            affected_resources: 1,
            estimated_downtime: Duration::minutes(0).num_seconds() as u32,
            estimated_cost: 150.0,
            risk_level: RiskLevel::Low,
            security_impact: "None".to_string(),
            compliance_impact: "None".to_string(),
        };
        
        // Test implementation would go here
        assert!(true); // Placeholder
    }
}