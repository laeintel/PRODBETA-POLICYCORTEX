// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use super::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use chrono::{Duration, DateTime, Utc};
use serde::{Serialize, Deserialize};

// Public alias for backward compatibility
pub type ApprovalManager = ApprovalWorkflowManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub remediation_request: RemediationRequest,
    pub approvers: Vec<String>,
    pub require_all: Option<bool>,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub status: String,
    pub decisions: HashMap<String, bool>,
}


pub struct ApprovalWorkflowManager {
    pending_approvals: Arc<RwLock<HashMap<String, PendingApproval>>>,
    approval_policies: Arc<RwLock<HashMap<String, ApprovalPolicy>>>,
    notification_service: Arc<NotificationService>,
    audit_log: Arc<RwLock<Vec<ApprovalAuditEntry>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    pub approval_id: String,
    pub workflow_id: Uuid,
    pub resource_id: String,
    pub remediation_type: RemediationType,
    pub risk_level: RiskLevel,
    pub requested_by: String,
    pub requested_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub approval_gate: ApprovalGate,
    pub approvers_responded: HashMap<String, ApprovalResponse>,
    pub context: ApprovalContext,
    pub status: ApprovalStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalContext {
    pub changes_summary: String,
    pub affected_resources: Vec<String>,
    pub estimated_impact: ImpactAssessment,
    pub compliance_implications: Vec<String>,
    pub rollback_available: bool,
    pub supporting_documents: Vec<DocumentLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub downtime_expected: bool,
    pub downtime_minutes: Option<u32>,
    pub users_affected: u32,
    pub services_affected: Vec<String>,
    pub cost_impact: CostImpact,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostImpact {
    pub one_time_cost: f64,
    pub monthly_cost_change: f64,
    pub annual_cost_change: f64,
    pub currency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentLink {
    pub document_type: DocumentType,
    pub title: String,
    pub url: String,
    pub uploaded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    ChangeRequest,
    RiskAssessment,
    ComplianceReport,
    TestResults,
    Diagram,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResponse {
    pub approver_id: String,
    pub approver_name: String,
    pub decision: ApprovalDecision,
    pub responded_at: DateTime<Utc>,
    pub comments: Option<String>,
    pub conditions: Vec<ApprovalCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approved,
    Rejected,
    ApprovedWithConditions,
    Deferred,
    Delegated(String), // delegated to another approver
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalCondition {
    pub condition_type: String,
    pub description: String,
    pub must_complete_by: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    InProgress,
    Approved,
    Rejected,
    Expired,
    Cancelled,
    AutoApproved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub resource_types: Vec<String>,
    pub risk_thresholds: RiskThresholds,
    pub auto_approval_rules: Vec<AutoApprovalRule>,
    pub escalation_rules: Vec<EscalationRule>,
    pub emergency_override: EmergencyOverride,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThresholds {
    pub low_risk_auto_approve: bool,
    pub medium_risk_approvers: u32,
    pub high_risk_approvers: u32,
    pub critical_risk_approvers: u32,
    pub critical_requires_ciso: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoApprovalRule {
    pub rule_id: String,
    pub condition: String,
    pub applies_to: Vec<String>, // resource types
    pub max_auto_approvals_per_day: Option<u32>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub trigger: EscalationTrigger,
    pub escalate_to: Vec<String>, // approver IDs
    pub notification_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationTrigger {
    NoResponseAfterHours(u32),
    RejectionCount(u32),
    CriticalRisk,
    ComplianceViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyOverride {
    pub enabled: bool,
    pub authorized_users: Vec<String>,
    pub requires_justification: bool,
    pub notify_ciso: bool,
    pub audit_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalAuditEntry {
    pub entry_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub approval_id: String,
    pub action: AuditAction,
    pub actor: String,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    ApprovalRequested,
    ApprovalGranted,
    ApprovalRejected,
    ApprovalExpired,
    ApprovalCancelled,
    ApprovalEscalated,
    EmergencyOverride,
}

pub struct NotificationService {
    email_client: Option<EmailClient>,
    teams_client: Option<TeamsClient>,
    slack_client: Option<SlackClient>,
}

struct EmailClient;
struct TeamsClient;
struct SlackClient;

impl ApprovalWorkflowManager {
    pub fn new() -> Self {
        Self {
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
            approval_policies: Arc::new(RwLock::new(Self::init_policies())),
            notification_service: Arc::new(NotificationService::new()),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn init_policies() -> HashMap<String, ApprovalPolicy> {
        let mut policies = HashMap::new();
        
        // Standard approval policy
        policies.insert("standard".to_string(), ApprovalPolicy {
            policy_id: "standard".to_string(),
            name: "Standard Approval Policy".to_string(),
            description: "Default approval policy for remediation actions".to_string(),
            resource_types: vec!["*".to_string()],
            risk_thresholds: RiskThresholds {
                low_risk_auto_approve: true,
                medium_risk_approvers: 1,
                high_risk_approvers: 2,
                critical_risk_approvers: 3,
                critical_requires_ciso: true,
            },
            auto_approval_rules: vec![
                AutoApprovalRule {
                    rule_id: "low-risk-auto".to_string(),
                    condition: "risk_level == 'Low' && downtime_expected == false".to_string(),
                    applies_to: vec!["*".to_string()],
                    max_auto_approvals_per_day: Some(50),
                    enabled: true,
                },
            ],
            escalation_rules: vec![
                EscalationRule {
                    trigger: EscalationTrigger::NoResponseAfterHours(4),
                    escalate_to: vec!["manager".to_string()],
                    notification_message: "Approval request requires urgent attention".to_string(),
                },
            ],
            emergency_override: EmergencyOverride {
                enabled: true,
                authorized_users: vec!["admin".to_string(), "ciso".to_string()],
                requires_justification: true,
                notify_ciso: true,
                audit_required: true,
            },
        });
        
        policies
    }

    pub async fn request_approval(&self, request: ApprovalRequest) -> Result<String, String> {
        let approval_id = Uuid::new_v4().to_string();
        
        // Determine approval requirements based on risk
        let approval_gate = self.determine_approval_gate(&request).await?;
        
        // Check for auto-approval
        if self.check_auto_approval(&request, &approval_gate).await? {
            self.auto_approve(&approval_id, &request).await?;
            return Ok(approval_id);
        }
        
        let pending_approval = PendingApproval {
            approval_id: approval_id.clone(),
            workflow_id: request.remediation_request.request_id,
            resource_id: request.remediation_request.resource_id.clone(),
            remediation_type: request.remediation_request.remediation_type.clone(),
            risk_level: RiskLevel::Medium, // Default risk level
            requested_by: request.created_by.clone(),
            requested_at: Utc::now(),
            expires_at: Utc::now() + Duration::hours(24),
            approval_gate,
            approvers_responded: HashMap::new(),
            context: ApprovalContext {
                changes_summary: "Automated remediation request".to_string(),
                affected_resources: vec![request.remediation_request.resource_id.clone()],
                estimated_impact: ImpactAssessment {
                    downtime_expected: false,
                    downtime_minutes: None,
                    users_affected: 0,
                    services_affected: vec![],
                    cost_impact: CostImpact {
                        one_time_cost: 0.0,
                        monthly_cost_change: 0.0,
                        annual_cost_change: 0.0,
                        currency: "USD".to_string(),
                    },
                    risk_score: 0.1,
                },
                compliance_implications: vec![],
                rollback_available: true,
                supporting_documents: vec![],
            },
            status: ApprovalStatus::Pending,
        };
        
        // Store pending approval
        self.pending_approvals.write().await.insert(approval_id.clone(), pending_approval.clone());
        
        // Send notifications
        self.send_approval_notifications(&pending_approval).await?;
        
        // Log audit entry
        self.log_audit(ApprovalAuditEntry {
            entry_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            approval_id: approval_id.clone(),
            action: AuditAction::ApprovalRequested,
            actor: request.created_by.clone(),
            details: serde_json::json!({
                "resource": request.remediation_request.resource_id,
                "remediation_type": format!("{:?}", request.remediation_request.remediation_type),
                "risk_level": "medium",
            }),
        }).await;
        
        Ok(approval_id)
    }

    async fn determine_approval_gate(&self, request: &ApprovalRequest) -> Result<ApprovalGate, String> {
        let policies = self.approval_policies.read().await;
        let policy = policies.get("standard").ok_or("No approval policy found")?;
        
        let risk_level = RiskLevel::Medium; // Default risk level
        let approvers = match risk_level {
            RiskLevel::Low => vec![],
            RiskLevel::Medium => vec![Approver {
                approver_type: ApproverType::Role,
                identifier: "CloudAdmin".to_string(),
                notification_method: NotificationMethod::Email,
            }],
            RiskLevel::High => vec![
                Approver {
                    approver_type: ApproverType::Role,
                    identifier: "CloudAdmin".to_string(),
                    notification_method: NotificationMethod::Email,
                },
                Approver {
                    approver_type: ApproverType::User,
                    identifier: "manager@company.com".to_string(),
                    notification_method: NotificationMethod::Teams,
                },
            ],
            RiskLevel::Critical => vec![
                Approver {
                    approver_type: ApproverType::Role,
                    identifier: "CloudAdmin".to_string(),
                    notification_method: NotificationMethod::Email,
                },
                Approver {
                    approver_type: ApproverType::User,
                    identifier: "ciso@company.com".to_string(),
                    notification_method: NotificationMethod::Email,
                },
                Approver {
                    approver_type: ApproverType::Group,
                    identifier: "SecurityTeam".to_string(),
                    notification_method: NotificationMethod::Teams,
                },
            ],
        };
        
        Ok(ApprovalGate {
            gate_id: Uuid::new_v4().to_string(),
            name: format!("{:?} Risk Approval", risk_level),
            approvers,
            approval_type: match risk_level {
                RiskLevel::Critical => ApprovalType::AllApprovers,
                RiskLevel::High => ApprovalType::MinimumApprovers(2),
                _ => ApprovalType::SingleApprover,
            },
            timeout_hours: 24,
            auto_approve_conditions: vec![],
        })
    }

    async fn check_auto_approval(&self, request: &ApprovalRequest, gate: &ApprovalGate) -> Result<bool, String> {
        let default_risk = RiskLevel::Medium;
        if matches!(default_risk, RiskLevel::Low) && gate.approvers.is_empty() {
            return Ok(true);
        }
        
        let policies = self.approval_policies.read().await;
        if let Some(policy) = policies.get("standard") {
            for rule in &policy.auto_approval_rules {
                if rule.enabled && self.evaluate_auto_approval_rule(rule, request).await {
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }

    async fn evaluate_auto_approval_rule(&self, rule: &AutoApprovalRule, _request: &ApprovalRequest) -> bool {
        // Simplified evaluation - in production would use a proper expression evaluator
        rule.enabled
    }

    async fn auto_approve(&self, approval_id: &str, request: &ApprovalRequest) -> Result<(), String> {
        self.log_audit(ApprovalAuditEntry {
            entry_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            approval_id: approval_id.to_string(),
            action: AuditAction::ApprovalGranted,
            actor: "System".to_string(),
            details: serde_json::json!({
                "auto_approved": true,
                "reason": "Low risk auto-approval",
            }),
        }).await;
        
        Ok(())
    }

    async fn send_approval_notifications(&self, approval: &PendingApproval) -> Result<(), String> {
        self.notification_service.notify_approvers(approval).await
    }

    async fn log_audit(&self, entry: ApprovalAuditEntry) {
        self.audit_log.write().await.push(entry);
    }

    pub async fn submit_approval_response(&self, approval_id: &str, response: ApprovalResponse) -> Result<ApprovalOutcome, String> {
        let mut approvals = self.pending_approvals.write().await;
        
        if let Some(approval) = approvals.get_mut(approval_id) {
            approval.approvers_responded.insert(response.approver_id.clone(), response.clone());
            
            // Check if approval requirements are met
            let outcome = self.evaluate_approval_outcome(approval).await?;
            
            match outcome {
                ApprovalOutcome::Approved => {
                    approval.status = ApprovalStatus::Approved;
                },
                ApprovalOutcome::Rejected => {
                    approval.status = ApprovalStatus::Rejected;
                },
                _ => {}
            }
            
            self.log_audit(ApprovalAuditEntry {
                entry_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                approval_id: approval_id.to_string(),
                action: match response.decision {
                    ApprovalDecision::Approved => AuditAction::ApprovalGranted,
                    ApprovalDecision::Rejected => AuditAction::ApprovalRejected,
                    _ => AuditAction::ApprovalRequested,
                },
                actor: response.approver_id.clone(),
                details: serde_json::to_value(&response).unwrap(),
            }).await;
            
            Ok(outcome)
        } else {
            Err("Approval not found".to_string())
        }
    }

    async fn evaluate_approval_outcome(&self, approval: &PendingApproval) -> Result<ApprovalOutcome, String> {
        let total_approvers = approval.approval_gate.approvers.len();
        let responses = &approval.approvers_responded;
        
        let approved_count = responses.values()
            .filter(|r| matches!(r.decision, ApprovalDecision::Approved | ApprovalDecision::ApprovedWithConditions))
            .count();
        
        let rejected_count = responses.values()
            .filter(|r| matches!(r.decision, ApprovalDecision::Rejected))
            .count();
        
        match approval.approval_gate.approval_type {
            ApprovalType::SingleApprover => {
                if approved_count >= 1 {
                    Ok(ApprovalOutcome::Approved)
                } else if rejected_count >= 1 {
                    Ok(ApprovalOutcome::Rejected)
                } else {
                    Ok(ApprovalOutcome::Pending)
                }
            },
            ApprovalType::AllApprovers => {
                if approved_count == total_approvers {
                    Ok(ApprovalOutcome::Approved)
                } else if rejected_count > 0 {
                    Ok(ApprovalOutcome::Rejected)
                } else {
                    Ok(ApprovalOutcome::Pending)
                }
            },
            ApprovalType::MinimumApprovers(min) => {
                if approved_count >= min as usize {
                    Ok(ApprovalOutcome::Approved)
                } else if rejected_count > (total_approvers - min as usize) {
                    Ok(ApprovalOutcome::Rejected)
                } else {
                    Ok(ApprovalOutcome::Pending)
                }
            },
            ApprovalType::Percentage(pct) => {
                let required = ((total_approvers as f64) * pct).ceil() as usize;
                if approved_count >= required {
                    Ok(ApprovalOutcome::Approved)
                } else if rejected_count > (total_approvers - required) {
                    Ok(ApprovalOutcome::Rejected)
                } else {
                    Ok(ApprovalOutcome::Pending)
                }
            },
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalOutcome {
    Approved,
    Rejected,
    Pending,
    TimedOut,
    AutoApproved,
}

impl NotificationService {
    fn new() -> Self {
        Self {
            email_client: None,
            teams_client: None,
            slack_client: None,
        }
    }

    async fn notify_approvers(&self, _approval: &PendingApproval) -> Result<(), String> {
        // In production, would send actual notifications
        Ok(())
    }
}