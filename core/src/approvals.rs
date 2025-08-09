use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ApprovalType {
    SingleApprover,
    AllApprovers,
    MinimumApprovers(usize),
    Hierarchical,
    EmergencyBreakGlass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub tenant_id: String,
    pub action_id: Uuid,
    pub action_type: String,
    pub resource_id: String,
    pub requester_id: String,
    pub requester_email: String,
    pub title: String,
    pub description: String,
    pub impact_analysis: ImpactAnalysis,
    pub approval_type: ApprovalType,
    pub required_approvers: Vec<String>,
    pub status: ApprovalStatus,
    pub approvals: Vec<Approval>,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Approval {
    pub id: Uuid,
    pub approver_id: String,
    pub approver_email: String,
    pub decision: ApprovalDecision,
    pub comments: Option<String>,
    pub conditions: Vec<ApprovalCondition>,
    pub approved_at: DateTime<Utc>,
    pub signature: Option<String>, // Digital signature for non-repudiation
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ApprovalDecision {
    Approved,
    Rejected,
    ApprovedWithConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalCondition {
    pub condition_type: String,
    pub description: String,
    pub must_be_met_before: Option<DateTime<Utc>>,
    pub is_met: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub affected_resources: Vec<String>,
    pub risk_level: RiskLevel,
    pub estimated_duration: Duration,
    pub is_reversible: bool,
    pub rollback_plan: Option<String>,
    pub business_impact: String,
    pub compliance_impact: Vec<ComplianceImpact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceImpact {
    pub framework: String,
    pub controls: Vec<String>,
    pub impact_description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalPolicy {
    pub id: Uuid,
    pub tenant_id: String,
    pub name: String,
    pub resource_pattern: String, // Regex pattern for matching resources
    pub action_types: Vec<String>,
    pub approval_type: ApprovalType,
    pub approver_groups: Vec<ApproverGroup>,
    pub auto_approve_conditions: Vec<AutoApproveCondition>,
    pub sod_rules: Vec<SeparationOfDutyRule>,
    pub escalation_policy: Option<EscalationPolicy>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproverGroup {
    pub name: String,
    pub members: Vec<String>,
    pub minimum_approvals: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoApproveCondition {
    pub condition_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationOfDutyRule {
    pub name: String,
    pub conflicting_roles: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub escalation_after: Duration,
    pub escalate_to: Vec<String>,
    pub max_escalations: usize,
}

// Emergency break-glass for critical situations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakGlassAccess {
    pub id: Uuid,
    pub tenant_id: String,
    pub requester_id: String,
    pub justification: String,
    pub emergency_type: EmergencyType,
    pub accessed_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub resources_accessed: Vec<String>,
    pub actions_taken: Vec<String>,
    pub audit_trail: Vec<AuditEntry>,
    pub post_incident_review_required: bool,
    pub review_status: Option<ReviewStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyType {
    SecurityIncident,
    ServiceOutage,
    DataBreach,
    ComplianceViolation,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewStatus {
    pub reviewed_by: String,
    pub reviewed_at: DateTime<Utc>,
    pub findings: String,
    pub actions_taken: String,
    pub policy_updates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub details: String,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

pub struct ApprovalEngine {
    policies: Vec<ApprovalPolicy>,
}

impl ApprovalEngine {
    pub fn new() -> Self {
        ApprovalEngine {
            policies: Vec::new(),
        }
    }

    pub fn add_policy(&mut self, policy: ApprovalPolicy) {
        self.policies.push(policy);
    }

    pub fn requires_approval(&self, tenant_id: &str, action_type: &str, resource_id: &str) -> Option<ApprovalPolicy> {
        self.policies.iter()
            .filter(|p| p.is_active && p.tenant_id == tenant_id)
            .filter(|p| p.action_types.contains(&action_type.to_string()))
            .find(|p| {
                // Match resource pattern
                if let Ok(re) = regex::Regex::new(&p.resource_pattern) {
                    re.is_match(resource_id)
                } else {
                    false
                }
            })
            .cloned()
    }

    pub fn create_approval_request(
        &self,
        policy: &ApprovalPolicy,
        action_id: Uuid,
        action_type: String,
        resource_id: String,
        requester_id: String,
        requester_email: String,
        impact_analysis: ImpactAnalysis,
    ) -> ApprovalRequest {
        let required_approvers = self.get_required_approvers(policy, &requester_id);
        
        ApprovalRequest {
            id: Uuid::new_v4(),
            tenant_id: policy.tenant_id.clone(),
            action_id,
            action_type,
            resource_id,
            requester_id,
            requester_email,
            title: format!("Approval required for {} on {}", action_type, resource_id),
            description: format!("Impact: {:?}, Reversible: {}", impact_analysis.risk_level, impact_analysis.is_reversible),
            impact_analysis,
            approval_type: policy.approval_type.clone(),
            required_approvers,
            status: ApprovalStatus::Pending,
            approvals: Vec::new(),
            expires_at: Utc::now() + Duration::hours(24),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn get_required_approvers(&self, policy: &ApprovalPolicy, requester_id: &str) -> Vec<String> {
        let mut approvers = Vec::new();
        
        for group in &policy.approver_groups {
            // Apply SoD rules - exclude conflicting roles
            let eligible_approvers: Vec<String> = group.members.iter()
                .filter(|member| *member != requester_id) // Can't approve own request
                .filter(|member| self.check_sod_compliance(policy, requester_id, member))
                .cloned()
                .collect();
            
            approvers.extend(eligible_approvers);
        }
        
        approvers
    }

    fn check_sod_compliance(&self, policy: &ApprovalPolicy, requester_id: &str, approver_id: &str) -> bool {
        // Check separation of duty rules
        for rule in &policy.sod_rules {
            // This would check against actual role assignments
            // For now, return true (compliant)
            return true;
        }
        true
    }

    pub fn process_approval(
        &self,
        request: &mut ApprovalRequest,
        approver_id: String,
        approver_email: String,
        decision: ApprovalDecision,
        comments: Option<String>,
    ) -> Result<ApprovalStatus, String> {
        // Check if approver is authorized
        if !request.required_approvers.contains(&approver_id) {
            return Err("Not authorized to approve this request".to_string());
        }

        // Check if already approved by this user
        if request.approvals.iter().any(|a| a.approver_id == approver_id) {
            return Err("Already processed by this approver".to_string());
        }

        // Add approval
        let approval = Approval {
            id: Uuid::new_v4(),
            approver_id,
            approver_email,
            decision: decision.clone(),
            comments,
            conditions: Vec::new(),
            approved_at: Utc::now(),
            signature: None, // Would be generated using digital signature
        };

        request.approvals.push(approval);
        request.updated_at = Utc::now();

        // Check if approval requirements are met
        match &request.approval_type {
            ApprovalType::SingleApprover => {
                if decision == ApprovalDecision::Approved {
                    request.status = ApprovalStatus::Approved;
                } else {
                    request.status = ApprovalStatus::Rejected;
                }
            }
            ApprovalType::AllApprovers => {
                let approved_count = request.approvals.iter()
                    .filter(|a| a.decision == ApprovalDecision::Approved)
                    .count();
                
                if approved_count == request.required_approvers.len() {
                    request.status = ApprovalStatus::Approved;
                } else if request.approvals.iter().any(|a| a.decision == ApprovalDecision::Rejected) {
                    request.status = ApprovalStatus::Rejected;
                }
            }
            ApprovalType::MinimumApprovers(min) => {
                let approved_count = request.approvals.iter()
                    .filter(|a| a.decision == ApprovalDecision::Approved)
                    .count();
                
                if approved_count >= *min {
                    request.status = ApprovalStatus::Approved;
                }
            }
            _ => {}
        }

        Ok(request.status.clone())
    }

    pub fn create_break_glass_access(
        &self,
        tenant_id: String,
        requester_id: String,
        justification: String,
        emergency_type: EmergencyType,
        duration_hours: i64,
    ) -> BreakGlassAccess {
        BreakGlassAccess {
            id: Uuid::new_v4(),
            tenant_id,
            requester_id,
            justification,
            emergency_type,
            accessed_at: Utc::now(),
            expires_at: Utc::now() + Duration::hours(duration_hours),
            resources_accessed: Vec::new(),
            actions_taken: Vec::new(),
            audit_trail: vec![
                AuditEntry {
                    timestamp: Utc::now(),
                    action: "BREAK_GLASS_ACTIVATED".to_string(),
                    details: justification.clone(),
                    ip_address: None,
                    user_agent: None,
                }
            ],
            post_incident_review_required: true,
            review_status: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_workflow() {
        let mut engine = ApprovalEngine::new();
        
        let policy = ApprovalPolicy {
            id: Uuid::new_v4(),
            tenant_id: "test-tenant".to_string(),
            name: "Production Changes".to_string(),
            resource_pattern: "prod-.*".to_string(),
            action_types: vec!["DELETE".to_string(), "MODIFY".to_string()],
            approval_type: ApprovalType::MinimumApprovers(2),
            approver_groups: vec![
                ApproverGroup {
                    name: "Admins".to_string(),
                    members: vec!["admin1".to_string(), "admin2".to_string(), "admin3".to_string()],
                    minimum_approvals: 2,
                }
            ],
            auto_approve_conditions: Vec::new(),
            sod_rules: Vec::new(),
            escalation_policy: None,
            is_active: true,
        };
        
        engine.add_policy(policy.clone());
        
        // Check if approval is required
        let requires = engine.requires_approval("test-tenant", "DELETE", "prod-database");
        assert!(requires.is_some());
        
        // Create approval request
        let impact = ImpactAnalysis {
            affected_resources: vec!["prod-database".to_string()],
            risk_level: RiskLevel::High,
            estimated_duration: Duration::hours(1),
            is_reversible: false,
            rollback_plan: None,
            business_impact: "High".to_string(),
            compliance_impact: Vec::new(),
        };
        
        let mut request = engine.create_approval_request(
            &policy,
            Uuid::new_v4(),
            "DELETE".to_string(),
            "prod-database".to_string(),
            "user1".to_string(),
            "user1@example.com".to_string(),
            impact,
        );
        
        // Process approvals
        let _ = engine.process_approval(
            &mut request,
            "admin1".to_string(),
            "admin1@example.com".to_string(),
            ApprovalDecision::Approved,
            Some("Looks good".to_string()),
        );
        
        assert_eq!(request.status, ApprovalStatus::Pending);
        
        let _ = engine.process_approval(
            &mut request,
            "admin2".to_string(),
            "admin2@example.com".to_string(),
            ApprovalDecision::Approved,
            None,
        );
        
        assert_eq!(request.status, ApprovalStatus::Approved);
    }
}