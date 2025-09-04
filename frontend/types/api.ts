/**
 * API Type Definitions for PolicyCortex
 * These types enforce strict type safety across the application
 */

// Base types
export interface TimestampData {
  created_at: string;
  updated_at: string;
}

export interface BaseEntity extends TimestampData {
  id: string;
  name: string;
  description?: string;
}

// Dashboard types
export interface DashboardMetrics {
  compliance_score: number;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  active_resources: number;
  policy_violations: number;
  cost_optimization_savings: number;
  security_score: number;
  timestamp: string;
}

export interface Alert extends BaseEntity {
  severity: 'Info' | 'Warning' | 'Error' | 'Critical';
  status: 'Active' | 'Acknowledged' | 'Resolved';
  resource_id?: string;
  resource_type?: string;
  message: string;
  metadata?: Record<string, unknown>;
}

export interface Activity extends TimestampData {
  id: string;
  user: string;
  action: string;
  resource?: string;
  result: 'Success' | 'Failed' | 'Pending';
  metadata?: Record<string, unknown>;
}

// Governance types
export interface ComplianceStatus {
  overall_score: number;
  framework: string;
  controls_passed: number;
  controls_failed: number;
  controls_total: number;
  last_assessment: string;
  next_assessment?: string;
  details: ComplianceControl[];
}

export interface ComplianceControl {
  id: string;
  name: string;
  status: 'Pass' | 'Fail' | 'Not Applicable';
  description: string;
  remediation?: string;
  evidence?: string[];
}

export interface PolicyViolation extends BaseEntity {
  policy_id: string;
  policy_name: string;
  resource_id: string;
  resource_type: string;
  violation_type: string;
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  detected_at: string;
  remediation_status: 'Pending' | 'In Progress' | 'Completed' | 'Failed';
  remediation_steps?: string[];
}

export interface RiskAssessment {
  overall_risk_score: number;
  risk_categories: RiskCategory[];
  recommendations: RiskRecommendation[];
  last_updated: string;
}

export interface RiskCategory {
  name: string;
  score: number;
  level: 'Low' | 'Medium' | 'High' | 'Critical';
  findings: number;
  mitigated: number;
}

export interface RiskRecommendation {
  id: string;
  title: string;
  description: string;
  priority: 'Low' | 'Medium' | 'High' | 'Critical';
  estimated_impact: number;
  effort_level: 'Low' | 'Medium' | 'High';
  category: string;
}

export interface CostSummary {
  current_month_spend: number;
  projected_month_spend: number;
  last_month_spend: number;
  year_to_date_spend: number;
  budget_limit?: number;
  budget_utilization?: number;
  cost_by_service: CostByService[];
  cost_trends: CostTrend[];
  optimization_opportunities: CostOptimization[];
}

export interface CostByService {
  service: string;
  cost: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
}

export interface CostTrend {
  date: string;
  cost: number;
  forecast?: number;
}

export interface CostOptimization {
  id: string;
  title: string;
  potential_savings: number;
  effort: 'Low' | 'Medium' | 'High';
  impact: 'Low' | 'Medium' | 'High';
  recommendations: string[];
}

export interface Policy extends BaseEntity {
  type: 'Security' | 'Compliance' | 'Cost' | 'Performance' | 'Custom';
  enabled: boolean;
  scope: string[];
  conditions: PolicyCondition[];
  actions: PolicyAction[];
  assignments: PolicyAssignment[];
  last_evaluated?: string;
  evaluation_count?: number;
}

export interface PolicyCondition {
  field: string;
  operator: 'equals' | 'notEquals' | 'contains' | 'greaterThan' | 'lessThan';
  value: string | number | boolean;
}

export interface PolicyAction {
  type: 'deny' | 'audit' | 'append' | 'modify' | 'remediate';
  details?: Record<string, unknown>;
}

export interface PolicyAssignment {
  scope: string;
  excluded_scopes?: string[];
  parameters?: Record<string, unknown>;
}

// Security types
export interface IAMUser extends BaseEntity {
  email: string;
  status: 'Active' | 'Inactive' | 'Locked' | 'Suspended';
  roles: string[];
  last_login?: string;
  mfa_enabled: boolean;
  created_by?: string;
  department?: string;
  manager?: string;
}

export interface RBACRole extends BaseEntity {
  permissions: Permission[];
  assignable_scopes: string[];
  assigned_users: number;
  builtin: boolean;
  category: 'Security' | 'Operations' | 'Development' | 'Custom';
}

export interface Permission {
  action: string;
  resource: string;
  effect: 'Allow' | 'Deny';
  conditions?: Record<string, unknown>;
}

export interface PIMRequest extends BaseEntity {
  requester: string;
  role: string;
  resource: string;
  justification: string;
  status: 'Pending' | 'Approved' | 'Denied' | 'Expired' | 'Active';
  requested_at: string;
  approved_by?: string;
  approved_at?: string;
  expires_at?: string;
  activation_duration?: number;
}

export interface ConditionalAccessPolicy extends BaseEntity {
  enabled: boolean;
  users: ConditionalAccessAssignment;
  applications: ConditionalAccessAssignment;
  conditions: ConditionalAccessConditions;
  grant_controls?: GrantControls;
  session_controls?: SessionControls;
}

export interface ConditionalAccessAssignment {
  include?: string[];
  exclude?: string[];
}

export interface ConditionalAccessConditions {
  sign_in_risk_levels?: ('low' | 'medium' | 'high')[];
  client_app_types?: string[];
  platforms?: string[];
  locations?: string[];
  device_states?: string[];
}

export interface GrantControls {
  operator: 'AND' | 'OR';
  built_in_controls?: string[];
  custom_authentication_factors?: string[];
  terms_of_use?: string[];
}

export interface SessionControls {
  application_enforced_restrictions?: boolean;
  cloud_app_security?: string;
  persistent_browser_session?: boolean;
  sign_in_frequency?: number;
}

export interface ZeroTrustStatus {
  overall_score: number;
  identity_score: number;
  device_score: number;
  network_score: number;
  application_score: number;
  data_score: number;
  visibility_score: number;
  automation_score: number;
  recommendations: string[];
  last_assessment: string;
}

export interface Entitlement extends BaseEntity {
  resource_type: string;
  resource_id: string;
  principal_type: 'User' | 'Group' | 'ServicePrincipal';
  principal_id: string;
  permissions: string[];
  granted_at: string;
  granted_by: string;
  expires_at?: string;
  justification?: string;
}

export interface AccessReview extends BaseEntity {
  status: 'NotStarted' | 'InProgress' | 'Completed' | 'Canceled';
  scope: string;
  reviewers: string[];
  start_date: string;
  end_date: string;
  recurrence?: string;
  decisions_made: number;
  decisions_pending: number;
  auto_apply_decisions: boolean;
  recommendations_enabled: boolean;
}

// Operations types
export interface Resource extends BaseEntity {
  type: string;
  subscription_id: string;
  resource_group: string;
  location: string;
  status: 'Running' | 'Stopped' | 'Failed' | 'Updating' | 'Deleting';
  tags: Record<string, string>;
  properties?: Record<string, unknown>;
  cost_per_month?: number;
  compliance_status?: 'Compliant' | 'NonCompliant' | 'Unknown';
}

export interface MonitoringMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  aggregation: 'Average' | 'Sum' | 'Min' | 'Max' | 'Count';
  dimensions?: Record<string, string>;
}

export interface AutomationWorkflow extends BaseEntity {
  trigger_type: 'Schedule' | 'Event' | 'Manual' | 'Webhook';
  trigger_config: Record<string, unknown>;
  actions: WorkflowAction[];
  status: 'Active' | 'Inactive' | 'Failed' | 'Running';
  last_run?: string;
  next_run?: string;
  run_count: number;
  success_count: number;
  failure_count: number;
}

export interface WorkflowAction {
  type: string;
  parameters: Record<string, unknown>;
  on_failure?: 'stop' | 'continue' | 'retry';
  retry_count?: number;
  timeout?: number;
}

export interface Notification extends BaseEntity {
  type: 'Info' | 'Warning' | 'Error' | 'Success';
  channel: 'Email' | 'SMS' | 'Webhook' | 'InApp';
  recipient: string;
  subject?: string;
  message: string;
  sent_at?: string;
  delivered: boolean;
  read?: boolean;
  metadata?: Record<string, unknown>;
}

// DevOps types
export interface Pipeline extends BaseEntity {
  project: string;
  repository: string;
  branch: string;
  status: 'Success' | 'Failed' | 'Running' | 'Queued' | 'Canceled';
  stages: PipelineStage[];
  triggered_by: string;
  started_at?: string;
  completed_at?: string;
  duration?: number;
  commit_id?: string;
  commit_message?: string;
}

export interface PipelineStage {
  name: string;
  status: 'Success' | 'Failed' | 'Running' | 'Pending' | 'Skipped';
  jobs: PipelineJob[];
  started_at?: string;
  completed_at?: string;
}

export interface PipelineJob {
  name: string;
  status: 'Success' | 'Failed' | 'Running' | 'Pending' | 'Skipped';
  steps: number;
  logs_url?: string;
  artifacts?: string[];
}

export interface Release extends BaseEntity {
  version: string;
  pipeline_id: string;
  environment: string;
  status: 'Deployed' | 'Failed' | 'InProgress' | 'Pending' | 'Canceled';
  deployed_by: string;
  deployed_at?: string;
  artifacts: Artifact[];
  approvals?: ReleaseApproval[];
  rollback_available: boolean;
}

export interface ReleaseApproval {
  approver: string;
  status: 'Pending' | 'Approved' | 'Rejected';
  comments?: string;
  approved_at?: string;
}

export interface Artifact extends BaseEntity {
  version: string;
  type: 'Container' | 'Package' | 'Binary' | 'Configuration';
  repository: string;
  size: number;
  hash?: string;
  download_url?: string;
  uploaded_by: string;
  uploaded_at: string;
  tags?: string[];
}

export interface Deployment extends BaseEntity {
  release_id: string;
  environment: string;
  status: 'Success' | 'Failed' | 'InProgress' | 'Pending' | 'RolledBack';
  strategy: 'BlueGreen' | 'Canary' | 'Rolling' | 'Recreate';
  instances: DeploymentInstance[];
  started_at: string;
  completed_at?: string;
  health_check_status?: 'Healthy' | 'Unhealthy' | 'Unknown';
}

export interface DeploymentInstance {
  id: string;
  status: 'Running' | 'Failed' | 'Pending' | 'Terminated';
  health: 'Healthy' | 'Unhealthy' | 'Unknown';
  version: string;
  started_at: string;
}

export interface Build extends BaseEntity {
  number: string;
  branch: string;
  commit_id: string;
  status: 'Success' | 'Failed' | 'Running' | 'Queued' | 'Canceled';
  triggered_by: string;
  started_at: string;
  completed_at?: string;
  duration?: number;
  test_results?: TestResults;
  artifacts?: string[];
  logs_url?: string;
}

export interface TestResults {
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  coverage?: number;
}

export interface Repository extends BaseEntity {
  url: string;
  default_branch: string;
  language: string;
  size: number;
  last_commit?: Commit;
  branches: number;
  contributors: number;
  open_pull_requests: number;
  visibility: 'Public' | 'Private' | 'Internal';
}

export interface Commit {
  id: string;
  message: string;
  author: string;
  timestamp: string;
  files_changed: number;
  additions: number;
  deletions: number;
}

// AI types
export interface PredictiveCompliance {
  predictions: CompliancePrediction[];
  confidence_score: number;
  model_version: string;
  last_updated: string;
}

export interface CompliancePrediction {
  resource_id: string;
  resource_type: string;
  prediction_type: 'Drift' | 'Violation' | 'Failure';
  probability: number;
  timeframe: string;
  impact: 'Low' | 'Medium' | 'High' | 'Critical';
  recommended_actions: string[];
  contributing_factors: Factor[];
}

export interface Factor {
  name: string;
  weight: number;
  trend: 'Increasing' | 'Decreasing' | 'Stable';
}

export interface Correlation {
  id: string;
  type: 'Security-Cost' | 'Compliance-Performance' | 'Risk-Availability' | 'Custom';
  strength: number;
  entities: CorrelatedEntity[];
  insights: string[];
  recommendations: string[];
  confidence: number;
  discovered_at: string;
}

export interface CorrelatedEntity {
  type: string;
  id: string;
  name: string;
  metrics: Record<string, number>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    intent?: string;
    entities?: Record<string, string>;
    confidence?: number;
    actions?: string[];
  };
}

export interface ChatResponse {
  message: ChatMessage;
  suggested_actions?: string[];
  related_resources?: string[];
  follow_up_questions?: string[];
}

export interface UnifiedMetrics {
  governance: GovernanceMetrics;
  security: SecurityMetrics;
  operations: OperationsMetrics;
  devops: DevOpsMetrics;
  ai: AIMetrics;
  timestamp: string;
}

export interface GovernanceMetrics {
  compliance_score: number;
  policy_compliance_rate: number;
  risk_score: number;
  cost_optimization_potential: number;
  policy_violations: number;
}

export interface SecurityMetrics {
  security_score: number;
  zero_trust_score: number;
  privileged_access_requests: number;
  conditional_access_coverage: number;
  mfa_adoption_rate: number;
}

export interface OperationsMetrics {
  resource_utilization: number;
  availability: number;
  performance_score: number;
  automation_coverage: number;
  incident_rate: number;
}

export interface DevOpsMetrics {
  deployment_frequency: number;
  lead_time: number;
  mttr: number;
  change_failure_rate: number;
  pipeline_success_rate: number;
}

export interface AIMetrics {
  prediction_accuracy: number;
  correlation_discoveries: number;
  automation_suggestions: number;
  chat_interactions: number;
  model_confidence: number;
}

// Health check types
export interface AzureHealthStatus {
  connected: boolean;
  subscription_id?: string;
  tenant_id?: string;
  resource_count?: number;
  last_sync?: string;
  errors?: string[];
}

// API Response wrapper types
export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}