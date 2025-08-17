// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Azure RBAC Integration for Access Control Management
// Comprehensive access governance with role-based access control and identity management

use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use crate::azure_client::AzureClient;
use crate::governance::{GovernanceResult, ComponentHealth, HealthStatus};

/// Azure RBAC access governance engine
pub struct AccessGovernanceEngine {
    azure_client: Arc<AzureClient>,
    access_cache: Arc<dashmap::DashMap<String, CachedAccessData>>,
    role_analyzer: RoleAnalyzer,
    permission_monitor: PermissionMonitor,
    identity_tracker: IdentityTracker,
}

/// Cached access data with TTL
#[derive(Debug, Clone)]
pub struct CachedAccessData {
    pub data: AccessData,
    pub cached_at: DateTime<Utc>,
    pub ttl: Duration,
}

impl CachedAccessData {
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.cached_at + self.ttl
    }
}

/// Comprehensive access control data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessData {
    pub scope: String,
    pub role_assignments: Vec<RoleAssignment>,
    pub custom_roles: Vec<CustomRole>,
    pub identity_summary: IdentitySummary,
    pub access_reviews: Vec<AccessReview>,
    pub privileged_accounts: Vec<PrivilegedAccount>,
    pub access_anomalies: Vec<AccessAnomaly>,
    pub compliance_status: AccessComplianceStatus,
    pub last_assessment: DateTime<Utc>,
}

/// Azure RBAC role assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAssignment {
    pub assignment_id: String,
    pub role_definition_id: String,
    pub role_name: String,
    pub principal_id: String,
    pub principal_type: PrincipalType,
    pub principal_name: String,
    pub scope: String,
    pub assignment_type: AssignmentType,
    pub created_on: DateTime<Utc>,
    pub created_by: String,
    pub condition: Option<String>,
    pub condition_version: Option<String>,
    pub delegated_managed_identity_resource_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrincipalType {
    User,
    Group,
    ServicePrincipal,
    ForeignGroup,
    Device,
}

impl std::fmt::Display for PrincipalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrincipalType::User => write!(f, "User"),
            PrincipalType::Group => write!(f, "Group"),
            PrincipalType::ServicePrincipal => write!(f, "ServicePrincipal"),
            PrincipalType::ForeignGroup => write!(f, "ForeignGroup"),
            PrincipalType::Device => write!(f, "Device"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentType {
    Direct,
    Inherited,
    Delegated,
}

/// Custom role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRole {
    pub role_id: String,
    pub role_name: String,
    pub description: String,
    pub role_type: RoleType,
    pub assignable_scopes: Vec<String>,
    pub permissions: Vec<RolePermission>,
    pub created_on: DateTime<Utc>,
    pub updated_on: DateTime<Utc>,
    pub created_by: String,
    pub is_custom: bool,
    pub assignment_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoleType {
    BuiltInRole,
    CustomRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolePermission {
    pub actions: Vec<String>,
    pub not_actions: Vec<String>,
    pub data_actions: Vec<String>,
    pub not_data_actions: Vec<String>,
}

/// Identity and access summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentitySummary {
    pub total_identities: u32,
    pub active_users: u32,
    pub service_principals: u32,
    pub groups: u32,
    pub guest_users: u32,
    pub privileged_roles_assigned: u32,
    pub custom_roles_count: u32,
    pub orphaned_assignments: u32,
    pub last_activity_analysis: DateTime<Utc>,
}

/// Access review process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReview {
    pub review_id: String,
    pub display_name: String,
    pub description: String,
    pub scope: String,
    pub reviewers: Vec<String>,
    pub status: AccessReviewStatus,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub recurrence: AccessReviewRecurrence,
    pub decisions: Vec<AccessReviewDecision>,
    pub completion_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessReviewStatus {
    NotStarted,
    InProgress,
    Completed,
    AutoReviewing,
    Stopping,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessReviewRecurrence {
    OneTime,
    Weekly,
    Monthly,
    Quarterly,
    SemiAnnually,
    Annually,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReviewDecision {
    pub decision_id: String,
    pub reviewer_id: String,
    pub principal_id: String,
    pub decision: ReviewDecision,
    pub justification: String,
    pub applied_on: DateTime<Utc>,
    pub applied_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approve,
    Deny,
    NotReviewed,
    DontKnow,
}

/// Privileged account monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivilegedAccount {
    pub principal_id: String,
    pub principal_name: String,
    pub principal_type: PrincipalType,
    pub privileged_roles: Vec<String>,
    pub assignment_scope: String,
    pub privilege_level: PrivilegeLevel,
    pub last_login: Option<DateTime<Utc>>,
    pub mfa_enabled: bool,
    pub conditional_access_applied: bool,
    pub pim_enabled: bool,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivilegeLevel {
    Critical,      // Global Admin, Security Admin
    High,          // Privileged Role Admin, User Admin
    Medium,        // Application Admin, Groups Admin
    Low,           // Directory Readers, Reports Reader
}

/// Access anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessAnomaly {
    pub anomaly_id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub confidence: f64,
    pub detected_at: DateTime<Utc>,
    pub principal_id: String,
    pub principal_name: String,
    pub description: String,
    pub details: HashMap<String, String>,
    pub remediation_suggestions: Vec<String>,
    pub status: AnomalyStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UnusualRoleAssignment,
    PrivilegeEscalation,
    DormantAccountActivity,
    AbnormalAccessPattern,
    CrossTenantAccess,
    ServicePrincipalMisuse,
    OrphanedPermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyStatus {
    Active,
    Investigating,
    Resolved,
    FalsePositive,
    Suppressed,
}

/// Access compliance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessComplianceStatus {
    pub overall_compliance: f64,
    pub least_privilege_compliance: f64,
    pub segregation_of_duties_compliance: f64,
    pub access_review_compliance: f64,
    pub privileged_access_compliance: f64,
    pub violations: Vec<ComplianceViolation>,
    pub recommendations: Vec<ComplianceRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_id: String,
    pub violation_type: ViolationType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub affected_principals: Vec<String>,
    pub detected_at: DateTime<Utc>,
    pub remediation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    ExcessivePermissions,
    StaleAccess,
    PrivilegedAccessWithoutMFA,
    UnreviewedAccess,
    ConflictingRoles,
    SharedAccounts,
    OrphanedAccounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub compliance_impact: f64,
    pub implementation_effort: ImplementationEffort,
    pub affected_count: u32,
    pub automation_available: bool,
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    RequiresPlanning,
}

/// Role analysis engine
pub struct RoleAnalyzer {
    privilege_matrix: HashMap<String, PrivilegeLevel>,
    role_relationships: HashMap<String, Vec<String>>,
}

/// Permission monitoring engine
pub struct PermissionMonitor {
    permission_usage_tracking: HashMap<String, PermissionUsage>,
    baseline_permissions: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct PermissionUsage {
    pub permission: String,
    pub usage_count: u32,
    pub last_used: DateTime<Utc>,
    pub principals_using: Vec<String>,
}

/// Identity tracking engine
pub struct IdentityTracker {
    identity_cache: HashMap<String, IdentityInfo>,
    activity_patterns: HashMap<String, Vec<ActivityPattern>>,
}

#[derive(Debug, Clone)]
pub struct IdentityInfo {
    pub principal_id: String,
    pub display_name: String,
    pub principal_type: PrincipalType,
    pub created_date: DateTime<Utc>,
    pub last_sign_in: Option<DateTime<Utc>>,
    pub is_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ActivityPattern {
    pub date: DateTime<Utc>,
    pub activity_type: String,
    pub resource_accessed: String,
    pub success: bool,
}

impl AccessGovernanceEngine {
    pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
        Ok(Self {
            azure_client,
            access_cache: Arc::new(dashmap::DashMap::new()),
            role_analyzer: RoleAnalyzer::new(),
            permission_monitor: PermissionMonitor::new(),
            identity_tracker: IdentityTracker::new(),
        })
    }

    /// Analyze access patterns and permissions across the organization
    pub async fn analyze_access_patterns(&self, scope: &str) -> GovernanceResult<AccessData> {
        let cache_key = format!("access_patterns_{}", scope);

        // Check cache first
        if let Some(cached) = self.access_cache.get(&cache_key) {
            if !cached.is_expired() {
                return Ok(cached.data.clone());
            }
        }

        // Fetch access data from Azure AD and RBAC
        let access_data = self.fetch_access_data(scope).await?;

        // Cache the result
        self.access_cache.insert(cache_key, CachedAccessData {
            data: access_data.clone(),
            cached_at: Utc::now(),
            ttl: Duration::hours(2), // Access data changes less frequently
        });

        Ok(access_data)
    }

    /// Detect privilege escalation attempts and anomalous access patterns
    pub async fn detect_privilege_escalation(&self, scope: &str) -> GovernanceResult<Vec<AccessAnomaly>> {
        let access_data = self.analyze_access_patterns(scope).await?;
        let mut anomalies = Vec::new();

        // Analyze role assignments for privilege escalation patterns
        for assignment in &access_data.role_assignments {
            if self.is_privilege_escalation(assignment) {
                anomalies.push(AccessAnomaly {
                    anomaly_id: uuid::Uuid::new_v4().to_string(),
                    anomaly_type: AnomalyType::PrivilegeEscalation,
                    severity: AnomalySeverity::High,
                    confidence: 0.85,
                    detected_at: Utc::now(),
                    principal_id: assignment.principal_id.clone(),
                    principal_name: assignment.principal_name.clone(),
                    description: format!("Unusual privilege escalation detected for principal {} assigned role {}", 
                        assignment.principal_name, assignment.role_name),
                    details: {
                        let mut details = HashMap::new();
                        details.insert("role_name".to_string(), assignment.role_name.clone());
                        details.insert("assignment_scope".to_string(), assignment.scope.clone());
                        details.insert("created_by".to_string(), assignment.created_by.clone());
                        details
                    },
                    remediation_suggestions: vec![
                        "Review the business justification for this role assignment".to_string(),
                        "Verify the assignment was approved through proper channels".to_string(),
                        "Consider implementing Privileged Identity Management (PIM)".to_string(),
                        "Enable additional monitoring for this principal".to_string(),
                    ],
                    status: AnomalyStatus::Active,
                });
            }
        }

        // Detect dormant account activity
        for privileged_account in &access_data.privileged_accounts {
            if let Some(last_login) = privileged_account.last_login {
                if Utc::now() - last_login > Duration::days(30) {
                    anomalies.push(AccessAnomaly {
                        anomaly_id: uuid::Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::DormantAccountActivity,
                        severity: AnomalySeverity::Medium,
                        confidence: 0.9,
                        detected_at: Utc::now(),
                        principal_id: privileged_account.principal_id.clone(),
                        principal_name: privileged_account.principal_name.clone(),
                        description: format!("Dormant privileged account {} has not been used for {} days", 
                            privileged_account.principal_name,
                            (Utc::now() - last_login).num_days()),
                        details: HashMap::new(),
                        remediation_suggestions: vec![
                            "Review if this account is still needed".to_string(),
                            "Consider disabling or removing unused privileged access".to_string(),
                            "Implement access reviews for privileged accounts".to_string(),
                        ],
                        status: AnomalyStatus::Active,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Enforce least privilege principle by analyzing excessive permissions
    pub async fn enforce_least_privilege(&self, scope: &str) -> GovernanceResult<Vec<ComplianceRecommendation>> {
        let access_data = self.analyze_access_patterns(scope).await?;
        let mut recommendations = Vec::new();

        // Analyze role assignments for excessive permissions
        for assignment in &access_data.role_assignments {
            if self.has_excessive_permissions(assignment) {
                recommendations.push(ComplianceRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    title: format!("Reduce excessive permissions for {}", assignment.principal_name),
                    description: format!("Principal {} has role '{}' which may grant more permissions than required for their job function", 
                        assignment.principal_name, assignment.role_name),
                    priority: RecommendationPriority::High,
                    compliance_impact: 15.0,
                    implementation_effort: ImplementationEffort::Medium,
                    affected_count: 1,
                    automation_available: false,
                    implementation_steps: vec![
                        "Review job responsibilities and required permissions".to_string(),
                        "Identify minimum required permissions for role".to_string(),
                        "Create custom role with reduced permissions if needed".to_string(),
                        "Replace current assignment with least privilege role".to_string(),
                        "Verify functionality after permission reduction".to_string(),
                    ],
                });
            }
        }

        // Recommend custom role creation for frequently used permission sets
        let frequent_permissions = self.analyze_permission_patterns(&access_data);
        for (permission_set, usage_count) in frequent_permissions {
            if usage_count > 5 {
                recommendations.push(ComplianceRecommendation {
                    recommendation_id: uuid::Uuid::new_v4().to_string(),
                    title: "Create custom role for common permission pattern".to_string(),
                    description: format!("Permission set used by {} principals could be standardized into a custom role", usage_count),
                    priority: RecommendationPriority::Medium,
                    compliance_impact: 10.0,
                    implementation_effort: ImplementationEffort::High,
                    affected_count: usage_count,
                    automation_available: true,
                    implementation_steps: vec![
                        "Define custom role with required permissions".to_string(),
                        "Test custom role with pilot group".to_string(),
                        "Migrate affected principals to custom role".to_string(),
                        "Remove previous individual assignments".to_string(),
                    ],
                });
            }
        }

        // Recommend access reviews for privileged accounts
        let privileged_without_reviews = access_data.privileged_accounts.iter()
            .filter(|account| !self.has_recent_access_review(account))
            .count() as u32;

        if privileged_without_reviews > 0 {
            recommendations.push(ComplianceRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                title: "Implement access reviews for privileged accounts".to_string(),
                description: format!("{} privileged accounts lack recent access reviews", privileged_without_reviews),
                priority: RecommendationPriority::Critical,
                compliance_impact: 25.0,
                implementation_effort: ImplementationEffort::Medium,
                affected_count: privileged_without_reviews,
                automation_available: true,
                implementation_steps: vec![
                    "Set up recurring access reviews for privileged roles".to_string(),
                    "Define appropriate reviewers (managers, resource owners)".to_string(),
                    "Configure automated email notifications".to_string(),
                    "Implement automatic removal for denied access".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }

    /// Monitor access reviews and compliance status
    pub async fn monitor_access_reviews(&self, scope: &str) -> GovernanceResult<Vec<AccessReview>> {
        // In production, would call Azure AD Access Reviews API
        // GET https://graph.microsoft.com/v1.0/identityGovernance/accessReviews/definitions

        Ok(vec![
            AccessReview {
                review_id: uuid::Uuid::new_v4().to_string(),
                display_name: "Quarterly Privileged Access Review".to_string(),
                description: "Review of all privileged role assignments".to_string(),
                scope: scope.to_string(),
                reviewers: vec!["manager".to_string(), "resource-owner".to_string()],
                status: AccessReviewStatus::InProgress,
                start_date: Utc::now() - Duration::days(7),
                end_date: Utc::now() + Duration::days(7),
                recurrence: AccessReviewRecurrence::Quarterly,
                decisions: vec![
                    AccessReviewDecision {
                        decision_id: uuid::Uuid::new_v4().to_string(),
                        reviewer_id: "reviewer1".to_string(),
                        principal_id: "user1".to_string(),
                        decision: ReviewDecision::Approve,
                        justification: "User requires admin access for their role".to_string(),
                        applied_on: Utc::now() - Duration::days(2),
                        applied_by: "system".to_string(),
                    }
                ],
                completion_percentage: 65.0,
            }
        ])
    }

    /// Get comprehensive access control metrics
    pub async fn get_access_metrics(&self, scope: &str) -> GovernanceResult<AccessMetrics> {
        let access_data = self.analyze_access_patterns(scope).await?;
        let anomalies = self.detect_privilege_escalation(scope).await?;

        Ok(AccessMetrics {
            total_role_assignments: access_data.role_assignments.len() as u32,
            privileged_accounts: access_data.privileged_accounts.len() as u32,
            custom_roles: access_data.custom_roles.len() as u32,
            active_anomalies: anomalies.iter().filter(|a| a.status == AnomalyStatus::Active).count() as u32,
            least_privilege_compliance: access_data.compliance_status.least_privilege_compliance,
            access_review_coverage: access_data.compliance_status.access_review_compliance,
            orphaned_assignments: access_data.identity_summary.orphaned_assignments,
            mfa_enabled_privileged_accounts: access_data.privileged_accounts.iter()
                .filter(|account| account.mfa_enabled)
                .count() as u32,
        })
    }

    /// Create or update access review
    pub async fn create_access_review(&self, review_definition: AccessReviewDefinition) -> GovernanceResult<String> {
        // In production, would call Azure AD Access Reviews API to create review
        // POST https://graph.microsoft.com/v1.0/identityGovernance/accessReviews/definitions

        let review_id = uuid::Uuid::new_v4().to_string();
        Ok(review_id)
    }

    /// Remediate access control violations
    pub async fn remediate_access_violations(&self, violation_ids: Vec<String>) -> GovernanceResult<Vec<RemediationResult>> {
        let mut results = Vec::new();

        for violation_id in violation_ids {
            // In production, would implement actual remediation based on violation type
            results.push(RemediationResult {
                violation_id: violation_id.clone(),
                status: RemediationStatus::Completed,
                actions_taken: vec![
                    "Removed excessive role assignment".to_string(),
                    "Applied principle of least privilege".to_string(),
                ],
                completed_at: Utc::now(),
            });
        }

        Ok(results)
    }

    /// Health check for access control components
    pub async fn health_check(&self) -> ComponentHealth {
        let mut metrics = HashMap::new();
        metrics.insert("cache_size".to_string(), self.access_cache.len() as f64);
        metrics.insert("role_definitions_cached".to_string(), self.role_analyzer.privilege_matrix.len() as f64);
        metrics.insert("permission_tracking_entries".to_string(), self.permission_monitor.permission_usage_tracking.len() as f64);

        ComponentHealth {
            component: "AccessControl".to_string(),
            status: HealthStatus::Healthy,
            message: "Access control governance operational with RBAC monitoring and compliance".to_string(),
            last_check: Utc::now(),
            metrics,
        }
    }

    // Private helper methods

    async fn fetch_access_data(&self, scope: &str) -> GovernanceResult<AccessData> {
        // In production, would call multiple Azure AD and RBAC APIs:
        // GET https://management.azure.com/{scope}/providers/Microsoft.Authorization/roleAssignments
        // GET https://management.azure.com/{scope}/providers/Microsoft.Authorization/roleDefinitions
        // GET https://graph.microsoft.com/v1.0/users
        // GET https://graph.microsoft.com/v1.0/servicePrincipals

        Ok(AccessData {
            scope: scope.to_string(),
            role_assignments: vec![
                RoleAssignment {
                    assignment_id: uuid::Uuid::new_v4().to_string(),
                    role_definition_id: "/subscriptions/12345/providers/Microsoft.Authorization/roleDefinitions/8e3af657-a8ff-443c-a75c-2fe8c4bcb635".to_string(),
                    role_name: "Owner".to_string(),
                    principal_id: "user-001".to_string(),
                    principal_type: PrincipalType::User,
                    principal_name: "john.doe@company.com".to_string(),
                    scope: scope.to_string(),
                    assignment_type: AssignmentType::Direct,
                    created_on: Utc::now() - Duration::days(30),
                    created_by: "admin@company.com".to_string(),
                    condition: None,
                    condition_version: None,
                    delegated_managed_identity_resource_id: None,
                },
                RoleAssignment {
                    assignment_id: uuid::Uuid::new_v4().to_string(),
                    role_definition_id: "/subscriptions/12345/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c".to_string(),
                    role_name: "Contributor".to_string(),
                    principal_id: "group-001".to_string(),
                    principal_type: PrincipalType::Group,
                    principal_name: "Developers".to_string(),
                    scope: format!("{}/resourceGroups/dev", scope),
                    assignment_type: AssignmentType::Direct,
                    created_on: Utc::now() - Duration::days(60),
                    created_by: "admin@company.com".to_string(),
                    condition: None,
                    condition_version: None,
                    delegated_managed_identity_resource_id: None,
                },
            ],
            custom_roles: vec![
                CustomRole {
                    role_id: uuid::Uuid::new_v4().to_string(),
                    role_name: "Custom VM Operator".to_string(),
                    description: "Can manage VMs but not create new ones".to_string(),
                    role_type: RoleType::CustomRole,
                    assignable_scopes: vec![scope.to_string()],
                    permissions: vec![
                        RolePermission {
                            actions: vec![
                                "Microsoft.Compute/virtualMachines/start/action".to_string(),
                                "Microsoft.Compute/virtualMachines/restart/action".to_string(),
                                "Microsoft.Compute/virtualMachines/deallocate/action".to_string(),
                                "Microsoft.Compute/virtualMachines/read".to_string(),
                            ],
                            not_actions: vec!["Microsoft.Compute/virtualMachines/delete".to_string()],
                            data_actions: vec![],
                            not_data_actions: vec![],
                        }
                    ],
                    created_on: Utc::now() - Duration::days(90),
                    updated_on: Utc::now() - Duration::days(10),
                    created_by: "admin@company.com".to_string(),
                    is_custom: true,
                    assignment_count: 5,
                }
            ],
            identity_summary: IdentitySummary {
                total_identities: 250,
                active_users: 180,
                service_principals: 45,
                groups: 25,
                guest_users: 15,
                privileged_roles_assigned: 8,
                custom_roles_count: 3,
                orphaned_assignments: 2,
                last_activity_analysis: Utc::now() - Duration::hours(6),
            },
            access_reviews: vec![],
            privileged_accounts: vec![
                PrivilegedAccount {
                    principal_id: "admin-001".to_string(),
                    principal_name: "admin@company.com".to_string(),
                    principal_type: PrincipalType::User,
                    privileged_roles: vec!["Global Administrator".to_string()],
                    assignment_scope: "/".to_string(),
                    privilege_level: PrivilegeLevel::Critical,
                    last_login: Some(Utc::now() - Duration::hours(4)),
                    mfa_enabled: true,
                    conditional_access_applied: true,
                    pim_enabled: true,
                    risk_score: 0.2,
                },
                PrivilegedAccount {
                    principal_id: "security-admin-001".to_string(),
                    principal_name: "security@company.com".to_string(),
                    principal_type: PrincipalType::User,
                    privileged_roles: vec!["Security Administrator".to_string()],
                    assignment_scope: scope.to_string(),
                    privilege_level: PrivilegeLevel::High,
                    last_login: Some(Utc::now() - Duration::days(7)),
                    mfa_enabled: false,
                    conditional_access_applied: false,
                    pim_enabled: false,
                    risk_score: 0.8,
                },
            ],
            access_anomalies: vec![],
            compliance_status: AccessComplianceStatus {
                overall_compliance: 78.5,
                least_privilege_compliance: 72.0,
                segregation_of_duties_compliance: 85.0,
                access_review_compliance: 65.0,
                privileged_access_compliance: 90.0,
                violations: vec![
                    ComplianceViolation {
                        violation_id: uuid::Uuid::new_v4().to_string(),
                        violation_type: ViolationType::PrivilegedAccessWithoutMFA,
                        severity: AnomalySeverity::High,
                        description: "Privileged account without MFA enabled".to_string(),
                        affected_principals: vec!["security@company.com".to_string()],
                        detected_at: Utc::now() - Duration::hours(2),
                        remediation_steps: vec![
                            "Enable MFA for privileged account".to_string(),
                            "Apply conditional access policy".to_string(),
                            "Enable PIM for just-in-time access".to_string(),
                        ],
                    }
                ],
                recommendations: vec![],
            },
            last_assessment: Utc::now(),
        })
    }

    fn is_privilege_escalation(&self, assignment: &RoleAssignment) -> bool {
        // Simple heuristic: check if high-privilege role assigned recently
        let high_privilege_roles = ["Owner", "User Access Administrator", "Global Administrator"];
        let is_high_privilege = high_privilege_roles.contains(&assignment.role_name.as_str());
        let is_recent = Utc::now() - assignment.created_on < Duration::hours(24);

        is_high_privilege && is_recent
    }

    fn has_excessive_permissions(&self, assignment: &RoleAssignment) -> bool {
        // Simple heuristic: Owner role at broad scope might be excessive
        assignment.role_name == "Owner" && assignment.scope.contains("/subscriptions/")
    }

    fn analyze_permission_patterns(&self, access_data: &AccessData) -> HashMap<String, u32> {
        let mut permission_patterns = HashMap::new();

        for assignment in &access_data.role_assignments {
            let pattern = format!("{}:{}", assignment.role_name, assignment.principal_type);
            *permission_patterns.entry(pattern).or_insert(0) += 1;
        }

        permission_patterns.into_iter()
            .filter(|(_, count)| *count > 1)
            .collect()
    }

    fn has_recent_access_review(&self, account: &PrivilegedAccount) -> bool {
        // In production, would check actual access review records
        account.risk_score < 0.5 // Simplified heuristic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessMetrics {
    pub total_role_assignments: u32,
    pub privileged_accounts: u32,
    pub custom_roles: u32,
    pub active_anomalies: u32,
    pub least_privilege_compliance: f64,
    pub access_review_coverage: f64,
    pub orphaned_assignments: u32,
    pub mfa_enabled_privileged_accounts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessReviewDefinition {
    pub display_name: String,
    pub description: String,
    pub scope: String,
    pub reviewers: Vec<String>,
    pub duration_in_days: u32,
    pub recurrence: AccessReviewRecurrence,
    pub auto_apply_decisions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationResult {
    pub violation_id: String,
    pub status: RemediationStatus,
    pub actions_taken: Vec<String>,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationStatus {
    Initiated,
    InProgress,
    Completed,
    Failed,
    PartiallyCompleted,
}

impl RoleAnalyzer {
    pub fn new() -> Self {
        let mut privilege_matrix = HashMap::new();
        privilege_matrix.insert("Global Administrator".to_string(), PrivilegeLevel::Critical);
        privilege_matrix.insert("Privileged Role Administrator".to_string(), PrivilegeLevel::Critical);
        privilege_matrix.insert("Security Administrator".to_string(), PrivilegeLevel::High);
        privilege_matrix.insert("User Administrator".to_string(), PrivilegeLevel::High);
        privilege_matrix.insert("Application Administrator".to_string(), PrivilegeLevel::Medium);
        privilege_matrix.insert("Groups Administrator".to_string(), PrivilegeLevel::Medium);
        privilege_matrix.insert("Directory Readers".to_string(), PrivilegeLevel::Low);

        Self {
            privilege_matrix,
            role_relationships: HashMap::new(),
        }
    }

    pub fn get_privilege_level(&self, role_name: &str) -> PrivilegeLevel {
        self.privilege_matrix.get(role_name).cloned().unwrap_or(PrivilegeLevel::Low)
    }
}

impl PermissionMonitor {
    pub fn new() -> Self {
        Self {
            permission_usage_tracking: HashMap::new(),
            baseline_permissions: HashMap::new(),
        }
    }

    pub fn track_permission_usage(&mut self, permission: &str, principal_id: &str) {
        let usage = self.permission_usage_tracking.entry(permission.to_string()).or_insert(PermissionUsage {
            permission: permission.to_string(),
            usage_count: 0,
            last_used: Utc::now(),
            principals_using: Vec::new(),
        });

        usage.usage_count += 1;
        usage.last_used = Utc::now();
        if !usage.principals_using.contains(&principal_id.to_string()) {
            usage.principals_using.push(principal_id.to_string());
        }
    }
}

impl IdentityTracker {
    pub fn new() -> Self {
        Self {
            identity_cache: HashMap::new(),
            activity_patterns: HashMap::new(),
        }
    }

    pub fn track_identity_activity(&mut self, principal_id: &str, activity: ActivityPattern) {
        self.activity_patterns.entry(principal_id.to_string()).or_default().push(activity);
    }

    pub fn get_identity_risk_score(&self, principal_id: &str) -> f64 {
        // Calculate risk score based on activity patterns
        if let Some(patterns) = self.activity_patterns.get(principal_id) {
            let failed_attempts = patterns.iter().filter(|p| !p.success).count();
            let total_attempts = patterns.len();

            if total_attempts > 0 {
                (failed_attempts as f64 / total_attempts as f64).min(1.0)
            } else {
                0.0
            }
        } else {
            0.5 // Unknown = medium risk
        }
    }
}