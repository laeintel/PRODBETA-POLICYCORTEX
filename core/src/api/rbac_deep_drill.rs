use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    azure::client::AzureClient,
    errors::AppError,
    state::AppState,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct UserPermissionDetail {
    pub user_id: String,
    pub display_name: String,
    pub email: String,
    pub department: Option<String>,
    pub job_title: Option<String>,
    pub permissions: Vec<PermissionDetail>,
    pub roles: Vec<RoleDetail>,
    pub groups: Vec<GroupMembership>,
    pub risk_score: f32,
    pub over_provisioning_score: f32,
    pub last_sign_in: Option<DateTime<Utc>>,
    pub account_enabled: bool,
    pub recommendations: Vec<PermissionRecommendation>,
    pub access_patterns: AccessPatternAnalysis,
    pub compliance_status: ComplianceStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PermissionDetail {
    pub permission_id: String,
    pub permission_name: String,
    pub resource_type: String,
    pub resource_id: String,
    pub resource_name: String,
    pub scope: String,
    pub actions: Vec<String>,
    pub not_actions: Vec<String>,
    pub data_actions: Vec<String>,
    pub not_data_actions: Vec<String>,
    pub assigned_date: DateTime<Utc>,
    pub assigned_by: String,
    pub assignment_type: AssignmentType,
    pub last_used: Option<DateTime<Utc>>,
    pub usage_count_30d: u32,
    pub usage_count_90d: u32,
    pub is_high_privilege: bool,
    pub is_custom: bool,
    pub risk_level: RiskLevel,
    pub usage_pattern: UsagePattern,
    pub similar_users_have_this: f32, // percentage
    pub removal_impact: RemovalImpact,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoleDetail {
    pub role_id: String,
    pub role_name: String,
    pub role_type: RoleType,
    pub is_builtin: bool,
    pub assigned_date: DateTime<Utc>,
    pub assigned_by: String,
    pub scope: String,
    pub permissions_count: u32,
    pub high_risk_permissions: Vec<String>,
    pub last_activity: Option<DateTime<Utc>>,
    pub usage_frequency: UsageFrequency,
    pub justification: Option<String>,
    pub expiry_date: Option<DateTime<Utc>>,
    pub is_eligible: bool,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GroupMembership {
    pub group_id: String,
    pub group_name: String,
    pub membership_type: MembershipType,
    pub joined_date: DateTime<Utc>,
    pub added_by: String,
    pub permissions_inherited: u32,
    pub nested_groups: Vec<String>,
    pub is_dynamic: bool,
    pub dynamic_rule: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AccessPatternAnalysis {
    pub typical_access_times: Vec<TimeRange>,
    pub typical_locations: Vec<Location>,
    pub typical_devices: Vec<Device>,
    pub unusual_activities: Vec<UnusualActivity>,
    pub access_velocity: f32,
    pub failed_attempts_30d: u32,
    pub mfa_usage_rate: f32,
    pub conditional_access_compliance: f32,
    pub privileged_operations_count: u32,
    pub data_access_volume: DataVolume,
    pub service_usage: HashMap<String, ServiceUsage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PermissionRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: RecommendationType,
    pub title: String,
    pub description: String,
    pub impact: ImpactLevel,
    pub confidence: f32,
    pub affected_permissions: Vec<String>,
    pub suggested_action: SuggestedAction,
    pub estimated_risk_reduction: f32,
    pub similar_users_implemented: u32,
    pub auto_remediation_available: bool,
    pub requires_approval: bool,
    pub approval_workflow_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub is_compliant: bool,
    pub violations: Vec<ComplianceViolation>,
    pub certifications: Vec<Certification>,
    pub last_review_date: Option<DateTime<Utc>>,
    pub next_review_date: Option<DateTime<Utc>>,
    pub reviewer: Option<String>,
    pub attestation_status: AttestationStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_id: String,
    pub policy_id: String,
    pub policy_name: String,
    pub violation_type: String,
    pub severity: Severity,
    pub detected_date: DateTime<Utc>,
    pub remediation_deadline: Option<DateTime<Utc>>,
    pub remediation_steps: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnusualActivity {
    pub activity_id: String,
    pub activity_type: String,
    pub timestamp: DateTime<Utc>,
    pub risk_score: f32,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub detection_method: String,
    pub is_investigated: bool,
    pub investigation_notes: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PermissionUsageHistory {
    pub permission_id: String,
    pub daily_usage: Vec<DailyUsage>,
    pub weekly_trends: WeeklyTrends,
    pub monthly_summary: MonthlySummary,
    pub peak_usage_times: Vec<PeakTime>,
    pub resource_access_patterns: Vec<ResourceAccessPattern>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DailyUsage {
    pub date: DateTime<Utc>,
    pub usage_count: u32,
    pub unique_resources_accessed: u32,
    pub operations_performed: Vec<String>,
    pub success_rate: f32,
    pub average_duration_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OverProvisioningAnalysis {
    pub user_id: String,
    pub analysis_date: DateTime<Utc>,
    pub over_provisioned_permissions: Vec<OverProvisionedPermission>,
    pub unused_permissions: Vec<UnusedPermission>,
    pub redundant_permissions: Vec<RedundantPermission>,
    pub peer_comparison: PeerComparison,
    pub optimization_potential: OptimizationPotential,
    pub recommended_permission_set: Vec<String>,
    pub estimated_risk_reduction: f32,
    pub implementation_plan: ImplementationPlan,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OverProvisionedPermission {
    pub permission_id: String,
    pub permission_name: String,
    pub reason: OverProvisioningReason,
    pub evidence: Vec<String>,
    pub last_legitimate_use: Option<DateTime<Utc>>,
    pub alternative_permission: Option<String>,
    pub removal_risk: RiskLevel,
    pub affected_workflows: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PeerComparison {
    pub peer_group: String,
    pub peer_count: u32,
    pub common_permissions: Vec<String>,
    pub unique_permissions: Vec<String>,
    pub permission_overlap_percentage: f32,
    pub risk_score_percentile: f32,
    pub usage_pattern_similarity: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JustInTimeAccessRequest {
    pub request_id: String,
    pub user_id: String,
    pub requested_permissions: Vec<String>,
    pub justification: String,
    pub requested_duration: std::time::Duration,
    pub approval_status: ApprovalStatus,
    pub approvers: Vec<String>,
    pub activation_time: Option<DateTime<Utc>>,
    pub expiration_time: Option<DateTime<Utc>>,
    pub business_context: BusinessContext,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RbacDeepDrillQuery {
    pub drill_type: DrillType,
    pub include_usage_history: bool,
    pub include_peer_analysis: bool,
    pub include_recommendations: bool,
    pub time_range_days: Option<u32>,
    pub risk_threshold: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrillType {
    UserPermissions,
    OverProvisioning,
    UnusedPermissions,
    HighRiskPermissions,
    ComplianceViolations,
    AccessPatterns,
    JustInTimeAccess,
    PermissionLineage,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentType {
    Direct,
    RoleBased,
    GroupInherited,
    Dynamic,
    JustInTime,
    Conditional,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
    None,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsagePattern {
    Daily,
    Weekly,
    Monthly,
    Occasional,
    Rare,
    Never,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationType {
    RemovePermission,
    DowngradePermission,
    ConvertToJustInTime,
    AddConditionalAccess,
    EnableMfa,
    ReviewGroupMembership,
    CertifyAccess,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverProvisioningReason {
    NeverUsed,
    NotUsedInDays(u32),
    RedundantWithOtherPermission,
    ExcessiveForRole,
    PolicyViolation,
    PeerGroupMismatch,
    ExpiredNeed,
}

// API Handlers

pub async fn get_user_permission_details(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Query(params): Query<RbacDeepDrillQuery>,
) -> Result<Json<UserPermissionDetail>, AppError> {
    // Comprehensive user permission analysis
    let user_detail = analyze_user_permissions(&state, &user_id, &params).await?;
    Ok(Json(user_detail))
}

pub async fn get_permission_usage_history(
    State(state): State<Arc<AppState>>,
    Path((user_id, permission_id)): Path<(String, String)>,
) -> Result<Json<PermissionUsageHistory>, AppError> {
    let history = fetch_permission_usage_history(&state, &user_id, &permission_id).await?;
    Ok(Json(history))
}

pub async fn get_over_provisioning_analysis(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> Result<Json<OverProvisioningAnalysis>, AppError> {
    let analysis = analyze_over_provisioning(&state, &user_id).await?;
    Ok(Json(analysis))
}

pub async fn recommend_permission_removal(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<PermissionRecommendation>>, AppError> {
    let recommendations = generate_removal_recommendations(&state, &user_id).await?;
    Ok(Json(recommendations))
}

pub async fn get_permission_lineage(
    State(state): State<Arc<AppState>>,
    Path((user_id, permission_id)): Path<(String, String)>,
) -> Result<Json<PermissionLineage>, AppError> {
    let lineage = trace_permission_lineage(&state, &user_id, &permission_id).await?;
    Ok(Json(lineage))
}

pub async fn get_role_analysis(
    State(state): State<Arc<AppState>>,
    Path(role_id): Path<String>,
) -> Result<Json<RoleAnalysis>, AppError> {
    let analysis = analyze_role_usage(&state, &role_id).await?;
    Ok(Json(analysis))
}

pub async fn request_just_in_time_access(
    State(state): State<Arc<AppState>>,
    Json(request): Json<JustInTimeAccessRequest>,
) -> Result<Json<JustInTimeAccessResponse>, AppError> {
    let response = process_jit_request(&state, request).await?;
    Ok(Json(response))
}

pub async fn export_rbac_analysis(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExportParams>,
) -> Result<Json<ExportResult>, AppError> {
    let result = export_analysis_data(&state, &params).await?;
    Ok(Json(result))
}

// Supporting structures

#[derive(Debug, Serialize, Deserialize)]
pub struct PermissionLineage {
    pub permission_id: String,
    pub assignment_chain: Vec<AssignmentNode>,
    pub inheritance_path: Vec<InheritanceNode>,
    pub effective_scope: String,
    pub original_assignor: String,
    pub assignment_history: Vec<AssignmentEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoleAnalysis {
    pub role_id: String,
    pub role_name: String,
    pub assigned_users: Vec<UserSummary>,
    pub permission_distribution: PermissionDistribution,
    pub usage_statistics: UsageStatistics,
    pub risk_assessment: RiskAssessment,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JustInTimeAccessResponse {
    pub request_id: String,
    pub status: ApprovalStatus,
    pub activation_token: Option<String>,
    pub activation_url: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub approval_required: bool,
    pub approvers_notified: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportParams {
    pub format: ExportFormat,
    pub include_sensitive: bool,
    pub time_range_days: Option<u32>,
    pub user_ids: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportResult {
    pub export_id: String,
    pub download_url: String,
    pub expires_at: DateTime<Utc>,
    pub size_bytes: u64,
    pub record_count: u32,
}

// Implementation helpers (stubs for now)

async fn analyze_user_permissions(
    state: &AppState,
    user_id: &str,
    params: &RbacDeepDrillQuery,
) -> Result<UserPermissionDetail, AppError> {
    // Implementation would fetch and analyze user permissions from Azure AD
    todo!("Implement comprehensive user permission analysis")
}

async fn fetch_permission_usage_history(
    state: &AppState,
    user_id: &str,
    permission_id: &str,
) -> Result<PermissionUsageHistory, AppError> {
    // Implementation would fetch usage history from audit logs
    todo!("Implement permission usage history fetching")
}

async fn analyze_over_provisioning(
    state: &AppState,
    user_id: &str,
) -> Result<OverProvisioningAnalysis, AppError> {
    // Implementation would analyze over-provisioning patterns
    todo!("Implement over-provisioning analysis")
}

async fn generate_removal_recommendations(
    state: &AppState,
    user_id: &str,
) -> Result<Vec<PermissionRecommendation>, AppError> {
    // Implementation would generate ML-based recommendations
    todo!("Implement permission removal recommendations")
}

async fn trace_permission_lineage(
    state: &AppState,
    user_id: &str,
    permission_id: &str,
) -> Result<PermissionLineage, AppError> {
    // Implementation would trace how permission was assigned
    todo!("Implement permission lineage tracing")
}

async fn analyze_role_usage(
    state: &AppState,
    role_id: &str,
) -> Result<RoleAnalysis, AppError> {
    // Implementation would analyze role usage patterns
    todo!("Implement role usage analysis")
}

async fn process_jit_request(
    state: &AppState,
    request: JustInTimeAccessRequest,
) -> Result<JustInTimeAccessResponse, AppError> {
    // Implementation would process JIT access request
    todo!("Implement JIT access request processing")
}

async fn export_analysis_data(
    state: &AppState,
    params: &ExportParams,
) -> Result<ExportResult, AppError> {
    // Implementation would export analysis data
    todo!("Implement analysis data export")
}

// Additional supporting types
#[derive(Debug, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Location {
    pub country: String,
    pub region: String,
    pub city: String,
    pub ip_range: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Device {
    pub device_id: String,
    pub device_type: String,
    pub os: String,
    pub is_compliant: bool,
    pub is_managed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataVolume {
    pub reads_gb: f64,
    pub writes_gb: f64,
    pub downloads_gb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceUsage {
    pub access_count: u32,
    pub unique_operations: u32,
    pub data_volume: DataVolume,
    pub peak_times: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleType {
    BuiltIn,
    Custom,
    ApplicationSpecific,
    Delegated,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MembershipType {
    Direct,
    Dynamic,
    Nested,
    Transitive,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageFrequency {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
    None,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemovalImpact {
    pub severity: ImpactLevel,
    pub affected_workflows: Vec<String>,
    pub dependent_users: Vec<String>,
    pub business_impact: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
    None,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SuggestedAction {
    pub action_type: String,
    pub description: String,
    pub automated: bool,
    pub requires_approval: bool,
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Certification {
    pub certification_id: String,
    pub name: String,
    pub issued_date: DateTime<Utc>,
    pub expiry_date: Option<DateTime<Utc>>,
    pub certifier: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttestationStatus {
    Attested,
    Pending,
    Expired,
    Rejected,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WeeklyTrends {
    pub average_daily_usage: f32,
    pub peak_day: String,
    pub trend_direction: TrendDirection,
    pub growth_rate: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonthlySummary {
    pub total_usage: u32,
    pub unique_days_used: u32,
    pub average_daily_usage: f32,
    pub peak_usage: u32,
    pub trend: TrendDirection,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PeakTime {
    pub hour: u8,
    pub day_of_week: String,
    pub usage_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceAccessPattern {
    pub resource_id: String,
    pub resource_type: String,
    pub access_frequency: u32,
    pub operations: Vec<String>,
    pub data_volume: DataVolume,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnusedPermission {
    pub permission_id: String,
    pub permission_name: String,
    pub assigned_date: DateTime<Utc>,
    pub days_unused: u32,
    pub removal_recommendation: RemovalRecommendation,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RedundantPermission {
    pub permission_id: String,
    pub redundant_with: Vec<String>,
    pub reason: String,
    pub keep_recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationPotential {
    pub current_permission_count: u32,
    pub optimal_permission_count: u32,
    pub reduction_percentage: f32,
    pub risk_reduction: f32,
    pub compliance_improvement: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImplementationPlan {
    pub phases: Vec<ImplementationPhase>,
    pub estimated_duration: std::time::Duration,
    pub rollback_plan: String,
    pub success_metrics: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImplementationPhase {
    pub phase_number: u8,
    pub name: String,
    pub actions: Vec<String>,
    pub duration: std::time::Duration,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemovalRecommendation {
    pub should_remove: bool,
    pub confidence: f32,
    pub alternative: Option<String>,
    pub impact_assessment: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
    Activated,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BusinessContext {
    pub project: Option<String>,
    pub ticket_id: Option<String>,
    pub urgency: UrgencyLevel,
    pub business_justification: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrgencyLevel {
    Emergency,
    High,
    Normal,
    Low,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AssignmentNode {
    pub node_type: String,
    pub node_id: String,
    pub node_name: String,
    pub assignment_date: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InheritanceNode {
    pub source_type: String,
    pub source_id: String,
    pub source_name: String,
    pub inheritance_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AssignmentEvent {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub actor: String,
    pub changes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserSummary {
    pub user_id: String,
    pub display_name: String,
    pub email: String,
    pub last_activity: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PermissionDistribution {
    pub total_permissions: u32,
    pub high_risk_count: u32,
    pub medium_risk_count: u32,
    pub low_risk_count: u32,
    pub categories: HashMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub total_uses_30d: u32,
    pub unique_users_30d: u32,
    pub average_uses_per_user: f32,
    pub peak_usage_time: String,
    pub trend: TrendDirection,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_name: String,
    pub severity: Severity,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_id: String,
    pub opportunity_type: String,
    pub description: String,
    pub potential_savings: f32,
    pub implementation_effort: EffortLevel,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Json,
    Csv,
    Excel,
    Pdf,
}