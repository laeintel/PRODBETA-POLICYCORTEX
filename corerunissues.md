  fn check_hierarchical_approval(&self, _request: &ApprovalRequest) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
315 |     async fn get_applicable_policy(
    |              ^^^^^^^^^^^^^^^^^^^^^
...
328 |     fn get_approval_requirements(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
341 |     async fn check_auto_approve_conditions(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
363 |     async fn create_auto_approved_request(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
414 |     async fn execute_approved_operation(&self, request: &ApprovalRequest) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
442 |     async fn schedule_escalation(&self, request: &ApprovalRequest, policy: &ApprovalPolicy) {
    |              ^^^^^^^^^^^^^^^^^^^
...
460 |     fn generate_signature(&self, user: &AuthUser, decision: &ApprovalDecision) -> String {
    |        ^^^^^^^^^^^^^^^^^^
...
471 |     async fn is_admin(&self, user: &AuthUser) -> bool {
    |              ^^^^^^^^
...
480 |     fn default_policies() -> Vec<ApprovalPolicy> {
    |        ^^^^^^^^^^^^^^^^

warning: associated items `new` and `notify_approvers` are never used
   --> src\approval_workflow.rs:534:8
    |
533 | impl NotificationService {
    | ------------------------ associated items in this implementation
534 |     fn new() -> Self {
    |        ^^^
...
542 |     async fn notify_approvers(&self, request: &ApprovalRequest) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^

warning: associated items `new`, `log_approval_requested`, `log_approval_decision`, and `log_approval_cancelled` are never used
   --> src\approval_workflow.rs:553:14
    |
552 | impl AuditService {
    | ----------------- associated items in this implementation
553 |     async fn new() -> Self {
    |              ^^^
...
557 |     async fn log_approval_requested(&self, request: &ApprovalRequest) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^^^^^^^
...
566 |     async fn log_approval_decision(
    |              ^^^^^^^^^^^^^^^^^^^^^
...
579 |     async fn log_approval_cancelled(
    |              ^^^^^^^^^^^^^^^^^^^^^^

warning: multiple variants are never constructed
  --> src\error.rs:19:5
   |
17 | pub enum ApiError {
   |          -------- variants in this enum
18 |     // Authentication & Authorization
19 |     Unauthorized(String),
   |     ^^^^^^^^^^^^
20 |     Forbidden(String),
21 |     InvalidToken(String),
   |     ^^^^^^^^^^^^
...
24 |     BadRequest(String),
   |     ^^^^^^^^^^
25 |     InvalidInput(String),
26 |     MissingParameter(String),
   |     ^^^^^^^^^^^^^^^^
...
30 |     Conflict(String),
   |     ^^^^^^^^
...
33 |     AzureError(String),
   |     ^^^^^^^^^^
...
43 |     TooManyRequests(String),
   |     ^^^^^^^^^^^^^^^
44 |     ServiceUnavailable(String),
   |     ^^^^^^^^^^^^^^^^^^
   |
   = note: `ApiError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: function `unwrap_or_internal` is never used
   --> src\error.rs:172:8
    |
172 | pub fn unwrap_or_internal<T>(result: Result<T, impl std::fmt::Display>, context: &str) -> ApiResult<T> {
    |        ^^^^^^^^^^^^^^^^^^

warning: function `require_param` is never used
   --> src\error.rs:177:8
    |
177 | pub fn require_param<T>(param: Option<T>, name: &str) -> ApiResult<T> {
    |        ^^^^^^^^^^^^^

warning: function `validate_input` is never used
   --> src\error.rs:182:8
    |
182 | pub fn validate_input<T>(result: Result<T, impl std::fmt::Display>, context: &str) -> ApiResult<T> {
    |        ^^^^^^^^^^^^^^

warning: multiple associated functions are never used
   --> src\validation.rs:20:12
    |
18  | impl Validator {
    | -------------- associated functions in this implementation
19  |     /// Validate UUID format
20  |     pub fn validate_uuid(input: &str, field_name: &str) -> ApiResult<Uuid> {
    |            ^^^^^^^^^^^^^
...
26  |     pub fn validate_email(email: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^
...
81  |     pub fn validate_tenant_id(tenant_id: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^^^^^
...
87  |     pub fn validate_subscription_id(subscription_id: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
...
123 |     pub fn validate_json_payload<T>(payload: &T, max_size_bytes: usize) -> ApiResult<()>
    |            ^^^^^^^^^^^^^^^^^^^^^
...
141 |     pub fn validate_date_range(start_date: &str, end_date: &str) -> ApiResult<(chrono::DateTime<chrono::Utc>, chrono::DateTime...
    |            ^^^^^^^^^^^^^^^^^^^
...
163 |     pub fn validate_pagination(page: Option<u32>, limit: Option<u32>) -> ApiResult<(u32, u32)> {
    |            ^^^^^^^^^^^^^^^^^^^
...
179 |     pub fn validate_action_type(action_type: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^^^^^^^
...
199 |     pub fn validate_search_query(query: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^^^^^^^^
...
221 |     pub fn validate_ip_address(ip: &str) -> ApiResult<std::net::IpAddr> {
    |            ^^^^^^^^^^^^^^^^^^^
...
227 |     pub fn validate_port(port: u16) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^
...
235 |     pub fn validate_url(url: &str) -> ApiResult<()> {
    |            ^^^^^^^^^^^^
...
242 |     pub fn validate_webhook_payload(payload: &serde_json::Value) -> ApiResult<()> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: fields `resource_id`, `policy_id`, `reason`, and `expires_days` are never read
   --> src\validation.rs:268:9
    |
267 | pub struct CreateExceptionValidated {
    |            ------------------------ fields in this struct
268 |     pub resource_id: String,
    |         ^^^^^^^^^^^
269 |     pub policy_id: String,
    |         ^^^^^^^^^
270 |     pub reason: String,
    |         ^^^^^^
271 |     pub expires_days: Option<u32>,
    |         ^^^^^^^^^^^^
    |
    = note: `CreateExceptionValidated` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: method `validate` is never used
   --> src\validation.rs:275:12
    |
274 | impl CreateExceptionValidated {
    | ----------------------------- method in this implementation
275 |     pub fn validate(self) -> ApiResult<Self> {
    |            ^^^^^^^^

warning: fields `action_type`, `resource_id`, and `params` are never read
   --> src\validation.rs:295:9
    |
294 | pub struct ExecuteActionValidated {
    |            ---------------------- fields in this struct
295 |     pub action_type: String,
    |         ^^^^^^^^^^^
296 |     pub resource_id: String,
    |         ^^^^^^^^^^^
297 |     pub params: HashMap<String, serde_json::Value>,
    |         ^^^^^^
    |
    = note: `ExecuteActionValidated` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: method `validate` is never used
   --> src\validation.rs:301:12
    |
300 | impl ExecuteActionValidated {
    | --------------------------- method in this implementation
301 |     pub fn validate(self) -> ApiResult<Self> {
    |            ^^^^^^^^

warning: struct `ApprovalEngine` is never constructed
   --> src\approvals.rs:193:12
    |
193 | pub struct ApprovalEngine {
    |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\approvals.rs:198:12
    |
197 | impl ApprovalEngine {
    | ------------------- associated items in this implementation
198 |     pub fn new() -> Self {
    |            ^^^
...
204 |     pub fn add_policy(&mut self, policy: ApprovalPolicy) {
    |            ^^^^^^^^^^
...
208 |     pub fn requires_approval(
    |            ^^^^^^^^^^^^^^^^^
...
229 |     pub fn create_approval_request(
    |            ^^^^^^^^^^^^^^^^^^^^^^^
...
266 |     fn get_required_approvers(&self, policy: &ApprovalPolicy, requester_id: &str) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
285 |     fn check_sod_compliance(
    |        ^^^^^^^^^^^^^^^^^^^^
...
300 |     pub fn process_approval(
    |            ^^^^^^^^^^^^^^^^
...
380 |     pub fn create_break_glass_access(
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `AuditChain` is never constructed
  --> src\audit_chain.rs:80:12
   |
80 | pub struct AuditChain {
   |            ^^^^^^^^^^

warning: multiple associated items are never used
   --> src\audit_chain.rs:87:12
    |
86  | impl AuditChain {
    | --------------- associated items in this implementation
87  |     pub fn new() -> Self {
    |            ^^^
...
95  |     pub fn add_entry(
    |            ^^^^^^^^^
...
145 |     fn calculate_hash(&self, entry: &AuditEntry) -> String {
    |        ^^^^^^^^^^^^^^
...
167 |     fn sign_entry(&self, entry: &AuditEntry) -> String {
    |        ^^^^^^^^^^
...
176 |     pub fn verify_chain(&self) -> Result<(), VerificationError> {
    |            ^^^^^^^^^^^^
...
222 |     fn verify_signature(&self, entry: &AuditEntry, signature: &str) -> bool {
    |        ^^^^^^^^^^^^^^^^
...
231 |     fn update_merkle_tree(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^
...
236 |     pub fn export_for_auditor(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> AuditExport {
    |            ^^^^^^^^^^^^^^^^^^

warning: enum `VerificationError` is never used
   --> src\audit_chain.rs:260:10
    |
260 | pub enum VerificationError {
    |          ^^^^^^^^^^^^^^^^^

warning: struct `MerkleTree` is never constructed
   --> src\audit_chain.rs:269:12
    |
269 | pub struct MerkleTree {
    |            ^^^^^^^^^^

warning: associated items `new` and `get_proof` are never used
   --> src\audit_chain.rs:275:12
    |
274 | impl MerkleTree {
    | --------------- associated items in this implementation
275 |     pub fn new(leaf_hashes: Vec<String>) -> Self {
    |            ^^^
...
312 |     pub fn get_proof(&self, index: usize) -> Vec<(String, bool)> {
    |            ^^^^^^^^^

warning: struct `PersistentAuditChain` is never constructed
   --> src\audit_chain.rs:347:12
    |
347 | pub struct PersistentAuditChain {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `load_recent_entries`, and `add_entry_persistent` are never used
   --> src\audit_chain.rs:353:18
    |
352 | impl PersistentAuditChain {
    | ------------------------- associated items in this implementation
353 |     pub async fn new(db_pool: sqlx::PgPool) -> Result<Self, sqlx::Error> {
    |                  ^^^
...
365 |     async fn load_recent_entries(&mut self) -> Result<(), sqlx::Error> {
    |              ^^^^^^^^^^^^^^^^^^^
...
379 |     pub async fn add_entry_persistent(
    |                  ^^^^^^^^^^^^^^^^^^^^

warning: fields `use`, `x5t`, `x5c`, and `alg` are never read
  --> src\auth.rs:85:9
   |
83 | pub struct Jwk {
   |            --- fields in this struct
84 |     pub kty: String,
85 |     pub r#use: Option<String>,
   |         ^^^^^
86 |     pub kid: String,
87 |     pub x5t: Option<String>,
   |         ^^^
...
90 |     pub x5c: Option<Vec<String>>,
   |         ^^^
91 |     pub alg: Option<String>,
   |         ^^^
   |
   = note: `Jwk` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: variant `InsufficientPermissions` is never constructed
   --> src\auth.rs:355:5
    |
344 | pub enum AuthError {
    |          --------- variant in this enum
...
355 |     InsufficientPermissions,
    |     ^^^^^^^^^^^^^^^^^^^^^^^
    |
    = note: `AuthError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: struct `RequirePermissions` is never constructed
   --> src\auth.rs:424:12
    |
424 | pub struct RequirePermissions {
    |            ^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `check` are never used
   --> src\auth.rs:429:12
    |
428 | impl RequirePermissions {
    | ----------------------- associated items in this implementation
429 |     pub fn new(scopes: Vec<&str>) -> Self {
    |            ^^^
...
435 |     pub fn check(&self, user: &AuthUser) -> Result<(), AuthError> {
    |            ^^^^^

warning: method `http_client` is never used
  --> src\azure_client.rs:97:12
   |
74 | impl AzureClient {
   | ---------------- method in this implementation
...
97 |     pub fn http_client(&self) -> &HttpClient {
   |            ^^^^^^^^^^^

warning: field `retry_attempts` is never read
  --> src\azure_client_async.rs:38:9
   |
32 | pub struct AzureClientConfig {
   |            ----------------- field in this struct
...
38 |     pub retry_attempts: u32,
   |         ^^^^^^^^^^^^^^
   |
   = note: `AzureClientConfig` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: field `policy_definition_id` is never read
    --> src\azure_client_async.rs:1372:9
     |
1368 | pub struct PolicyState {
     |            ----------- field in this struct
...
1372 |     pub policy_definition_id: String,
     |         ^^^^^^^^^^^^^^^^^^^^
     |
     = note: `PolicyState` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: fields `max_connections`, `connection_timeout_ms`, and `retry_attempts` are never read
  --> src\cache.rs:35:9
   |
32 | pub struct CacheConfig {
   |            ----------- fields in this struct
...
35 |     pub max_connections: u32,
   |         ^^^^^^^^^^^^^^^
36 |     pub connection_timeout_ms: u64,
   |         ^^^^^^^^^^^^^^^^^^^^^
37 |     pub retry_attempts: u32,
   |         ^^^^^^^^^^^^^^
   |
   = note: `CacheConfig` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: multiple methods are never used
   --> src\cache.rs:94:18
    |
53  | impl CacheManager {
    | ----------------- methods in this implementation
...
94  |     pub async fn get_warm<T>(
    |                  ^^^^^^^^
...
120 |     pub async fn get_cold<T>(
    |                  ^^^^^^^^
...
146 |     pub async fn get_smart<T>(
    |                  ^^^^^^^^^
...
217 |     pub async fn get_batch<T>(
    |                  ^^^^^^^^^
...
248 |     pub async fn set_batch<T>(
    |                  ^^^^^^^^^
...
273 |     pub async fn invalidate_pattern(
    |                  ^^^^^^^^^^^^^^^^^^
...
291 |     pub async fn invalidate_by_tags(
    |                  ^^^^^^^^^^^^^^^^^^
...
306 |     pub async fn get_stats(
    |                  ^^^^^^^^^

warning: variants `Frequent`, `Occasional`, and `Rare` are never constructed
   --> src\cache.rs:417:5
    |
415 | pub enum CacheAccessPattern {
    |          ------------------ variants in this enum
416 |     RealTime,   // < 30 seconds (compliance violations, alerts)
417 |     Frequent,   // 5 minutes (policies, resources, costs)
    |     ^^^^^^^^
418 |     Occasional, // 30 minutes (reports, analytics)
    |     ^^^^^^^^^^
419 |     Rare,       // 2+ hours (historical data, configurations)
    |     ^^^^
    |
    = note: `CacheAccessPattern` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: struct `CacheStats` is never constructed
   --> src\cache.rs:423:12
    |
423 | pub struct CacheStats {
    |            ^^^^^^^^^^

warning: associated functions `policy_compliance`, `resource_data`, `cost_analysis`, `rbac_assignments`, `security_alerts`, and `compliance_score` are never used
   --> src\cache.rs:437:12
    |
432 | impl CacheKeys {
    | -------------- associated functions in this implementation
...
437 |     pub fn policy_compliance(policy_id: &str) -> String {
    |            ^^^^^^^^^^^^^^^^^
...
441 |     pub fn resource_data(resource_id: &str) -> String {
    |            ^^^^^^^^^^^^^
...
445 |     pub fn cost_analysis(subscription_id: &str, date: &str) -> String {
    |            ^^^^^^^^^^^^^
...
449 |     pub fn rbac_assignments(tenant_id: &str) -> String {
    |            ^^^^^^^^^^^^^^^^
...
453 |     pub fn security_alerts(tenant_id: &str) -> String {
    |            ^^^^^^^^^^^^^^^
...
457 |     pub fn compliance_score(tenant_id: &str) -> String {
    |            ^^^^^^^^^^^^^^^^

warning: trait `ChangeManagementSystem` is never used
   --> src\change_management.rs:133:11
    |
133 | pub trait ChangeManagementSystem: Send + Sync {
    |           ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ServiceNowIntegration` is never constructed
   --> src\change_management.rs:148:12
    |
148 | pub struct ServiceNowIntegration {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `map_to_servicenow`, `map_category`, `map_priority`, `map_risk`, and `map_impact` are never used
   --> src\change_management.rs:156:12
    |
155 | impl ServiceNowIntegration {
    | -------------------------- associated items in this implementation
156 |     pub fn new(instance_url: String, username: String, password: String) -> Self {
    |            ^^^
...
165 |     fn map_to_servicenow(&self, request: &ChangeRequest) -> serde_json::Value {
    |        ^^^^^^^^^^^^^^^^^
...
184 |     fn map_category(&self, category: &ChangeCategory) -> &str {
    |        ^^^^^^^^^^^^
...
193 |     fn map_priority(&self, priority: &Priority) -> i32 {
    |        ^^^^^^^^^^^^
...
202 |     fn map_risk(&self, risk: &RiskLevel) -> i32 {
    |        ^^^^^^^^
...
212 |     fn map_impact(&self, impact: &Impact) -> i32 {
    |        ^^^^^^^^^^

warning: struct `JiraIntegration` is never constructed
   --> src\change_management.rs:459:12
    |
459 | pub struct JiraIntegration {
    |            ^^^^^^^^^^^^^^^

warning: associated items `new` and `map_to_jira` are never used
   --> src\change_management.rs:467:12
    |
466 | impl JiraIntegration {
    | -------------------- associated items in this implementation
467 |     pub fn new(base_url: String, api_token: String, project_key: String) -> Self {
    |            ^^^
...
476 |     fn map_to_jira(&self, request: &ChangeRequest) -> serde_json::Value {
    |        ^^^^^^^^^^^

warning: struct `ChangeManagementOrchestrator` is never constructed
   --> src\change_management.rs:702:12
    |
702 | pub struct ChangeManagementOrchestrator {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `submit_change`, `is_in_freeze_window`, `validate_change`, `calculate_risk_score`, and `auto_approve_standard_changes` are never used
   --> src\change_management.rs:708:12
    |
707 | impl ChangeManagementOrchestrator {
    | --------------------------------- associated items in this implementation
708 |     pub fn new(system: Box<dyn ChangeManagementSystem>) -> Self {
    |            ^^^
...
715 |     pub async fn submit_change(&self, mut request: ChangeRequest) -> Result<String, String> {
    |                  ^^^^^^^^^^^^^
...
737 |     async fn is_in_freeze_window(&self, request: &ChangeRequest) -> Result<bool, String> {
    |              ^^^^^^^^^^^^^^^^^^^
...
752 |     fn validate_change(&self, request: &ChangeRequest) -> Result<(), String> {
    |        ^^^^^^^^^^^^^^^
...
774 |     fn calculate_risk_score(&self, request: &ChangeRequest) -> i32 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
805 |     pub async fn auto_approve_standard_changes(&self, id: &str) -> Result<(), String> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `update_progress`, and `record_error` are never used
  --> src\checkpoint.rs:20:12
   |
19 | impl IngestionCheckpoint {
   | ------------------------ associated items in this implementation
20 |     pub fn new(source: String) -> Self {
   |            ^^^
...
32 |     pub fn update_progress(&mut self, record_id: Option<String>, etag: Option<String>) {
   |            ^^^^^^^^^^^^^^^
...
45 |     pub fn record_error(&mut self, error: String) {
   |            ^^^^^^^^^^^^

warning: struct `CheckpointManager` is never constructed
  --> src\checkpoint.rs:51:12
   |
51 | pub struct CheckpointManager {
   |            ^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\checkpoint.rs:57:18
    |
56  | impl CheckpointManager {
    | ---------------------- associated items in this implementation
57  |     pub async fn new(redis_url: &str, namespace: &str) -> Result<Self, RedisError> {
    |                  ^^^
...
67  |     fn get_key(&self, source: &str) -> String {
    |        ^^^^^^^
...
71  |     pub async fn save_checkpoint(&mut self, checkpoint: &IngestionCheckpoint) -> Result<(), RedisError> {
    |                  ^^^^^^^^^^^^^^^
...
87  |     pub async fn load_checkpoint(&mut self, source: &str) -> Result<Option<IngestionCheckpoint>, RedisError> {
    |                  ^^^^^^^^^^^^^^^
...
110 |     pub async fn delete_checkpoint(&mut self, source: &str) -> Result<(), RedisError> {
    |                  ^^^^^^^^^^^^^^^^^
...
117 |     pub async fn list_checkpoints(&mut self) -> Result<HashMap<String, IngestionCheckpoint>, RedisError> {
    |                  ^^^^^^^^^^^^^^^^
...
135 |     pub async fn get_stale_checkpoints(&mut self, stale_after_hours: i64) -> Result<Vec<String>, RedisError> {
    |                  ^^^^^^^^^^^^^^^^^^^^^

warning: constant `RESOURCE_GRAPH` is never used
   --> src\checkpoint.rs:160:15
    |
160 |     pub const RESOURCE_GRAPH: &str = "azure_resource_graph";
    |               ^^^^^^^^^^^^^^

warning: constant `POLICY_COMPLIANCE` is never used
   --> src\checkpoint.rs:161:15
    |
161 |     pub const POLICY_COMPLIANCE: &str = "azure_policy_compliance";
    |               ^^^^^^^^^^^^^^^^^

warning: constant `COST_MANAGEMENT` is never used
   --> src\checkpoint.rs:162:15
    |
162 |     pub const COST_MANAGEMENT: &str = "azure_cost_management";
    |               ^^^^^^^^^^^^^^^

warning: constant `ACTIVITY_LOG` is never used
   --> src\checkpoint.rs:163:15
    |
163 |     pub const ACTIVITY_LOG: &str = "azure_activity_log";
    |               ^^^^^^^^^^^^

warning: constant `DEFENDER_ALERTS` is never used
   --> src\checkpoint.rs:164:15
    |
164 |     pub const DEFENDER_ALERTS: &str = "azure_defender_alerts";
    |               ^^^^^^^^^^^^^^^

warning: constant `ADVISOR_RECOMMENDATIONS` is never used
   --> src\checkpoint.rs:165:15
    |
165 |     pub const ADVISOR_RECOMMENDATIONS: &str = "azure_advisor";
    |               ^^^^^^^^^^^^^^^^^^^^^^^

warning: constant `MONITOR_METRICS` is never used
   --> src\checkpoint.rs:166:15
    |
166 |     pub const MONITOR_METRICS: &str = "azure_monitor_metrics";
    |               ^^^^^^^^^^^^^^^

warning: constant `KEYVAULT_EVENTS` is never used
   --> src\checkpoint.rs:167:15
    |
167 |     pub const KEYVAULT_EVENTS: &str = "azure_keyvault_events";
    |               ^^^^^^^^^^^^^^^

warning: struct `AwsCollector` is never constructed
  --> src\collectors\aws_collector.rs:15:12
   |
15 | pub struct AwsCollector {
   |            ^^^^^^^^^^^^

warning: associated function `new` is never used
  --> src\collectors\aws_collector.rs:20:18
   |
19 | impl AwsCollector {
   | ----------------- associated function in this implementation
20 |     pub async fn new(region: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
   |                  ^^^

warning: trait `CloudCollector` is never used
   --> src\collectors\aws_collector.rs:118:11
    |
118 | pub trait CloudCollector: Send + Sync {
    |           ^^^^^^^^^^^^^^

warning: struct `GcpCollector` is never constructed
  --> src\collectors\gcp_collector.rs:17:12
   |
17 | pub struct GcpCollector;
   |            ^^^^^^^^^^^^

warning: associated function `new` is never used
  --> src\collectors\gcp_collector.rs:20:18
   |
19 | impl GcpCollector {
   | ----------------- associated function in this implementation
20 |     pub async fn new(_project: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
   |                  ^^^

warning: trait `ComplianceEngine` is never used
   --> src\compliance\mod.rs:173:11
    |
173 | pub trait ComplianceEngine: Send + Sync {
    |           ^^^^^^^^^^^^^^^^

warning: struct `DateRange` is never constructed
   --> src\compliance\mod.rs:195:12
    |
195 | pub struct DateRange {
    |            ^^^^^^^^^

warning: enum `ComplianceError` is never used
   --> src\compliance\mod.rs:226:10
    |
226 | pub enum ComplianceError {
    |          ^^^^^^^^^^^^^^^

warning: struct `AzureComplianceEngine` is never constructed
   --> src\compliance\mod.rs:238:12
    |
238 | pub struct AzureComplianceEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\compliance\mod.rs:245:18
    |
244 | impl AzureComplianceEngine {
    | -------------------------- associated items in this implementation
245 |     pub async fn new(
    |                  ^^^
...
265 |     async fn load_frameworks(&mut self) -> Result<(), ComplianceError> {
    |              ^^^^^^^^^^^^^^^
...
282 |     fn create_iso27001_framework(&self) -> ComplianceFramework {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
345 |     fn create_pci_dss_framework(&self) -> ComplianceFramework {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
368 |     fn create_hipaa_framework(&self) -> ComplianceFramework {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
391 |     fn create_gdpr_framework(&self) -> ComplianceFramework {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
414 |     fn create_cis_framework(&self) -> ComplianceFramework {
    |        ^^^^^^^^^^^^^^^^^^^^
...
437 |     async fn execute_control_test(
    |              ^^^^^^^^^^^^^^^^^^^^
...
596 |     fn validate_configuration(
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
619 |     fn generate_evidence_ref(&self) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
623 |     async fn generate_artifact(
    |              ^^^^^^^^^^^^^^^^^
...
747 |     fn detect_configuration_drift(&self, resources: &[serde_json::Value]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
762 |     async fn save_artifact(
    |              ^^^^^^^^^^^^^
...
799 |     fn calculate_compliance_score(&self, results: &[ControlEvidenceResult]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
810 |     fn create_manifest(&self, artifacts: &[EvidenceArtifact]) -> EvidenceManifest {
    |        ^^^^^^^^^^^^^^^

warning: function `get_compliance_status` is never used
    --> src\compliance\mod.rs:1027:14
     |
1027 | pub async fn get_compliance_status(
     |              ^^^^^^^^^^^^^^^^^^^^^

warning: field `key_vault_uri` is never read
  --> src\config.rs:18:9
   |
12 | pub struct AppConfig {
   |            --------- field in this struct
...
18 |     pub key_vault_uri: Option<String>,
   |         ^^^^^^^^^^^^^
   |
   = note: `AppConfig` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis

warning: struct `CrossDomainEngine` is never constructed
  --> src\correlation\cross_domain_engine.rs:18:12
   |
18 | pub struct CrossDomainEngine {
   |            ^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\cross_domain_engine.rs:26:12
    |
25  | impl CrossDomainEngine {
    | ---------------------- associated items in this implementation
26  |     pub fn new() -> Self {
    |            ^^^
...
39  |     fn initialize_analyzers(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^^
...
47  |     fn initialize_rules(&mut self) {
    |        ^^^^^^^^^^^^^^^^
...
74  |     pub async fn analyze_correlations(&mut self, resources: Vec<AzureResource>) -> CorrelationAnalysis {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
101 |     fn build_dependency_graph(&mut self, resources: &[AzureResource]) {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
135 |     fn find_correlations(&self, resources: &[AzureResource]) -> Vec<Correlation> {
    |        ^^^^^^^^^^^^^^^^^
...
166 |     fn find_critical_paths(&self) -> Vec<CriticalPath> {
    |        ^^^^^^^^^^^^^^^^^^^
...
191 |     fn calculate_domain_impacts(&self, resources: &[AzureResource]) -> HashMap<String, DomainImpact> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
206 |     fn find_isolated_resources(&self) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
221 |     fn get_domain(&self, resource_type: &str) -> String {
    |        ^^^^^^^^^^
...
237 |     fn calculate_resource_risk(&self, resource: &AzureResource) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
258 |     fn resources_correlated(&self, source: &AzureResource, target: &AzureResource, rule: &CorrelationRule) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^
...
277 |     fn calculate_overall_risk(&self) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
290 |     fn generate_recommendations(&self) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: trait `DomainAnalyzer` is never used
   --> src\correlation\cross_domain_engine.rs:319:11
    |
319 | pub trait DomainAnalyzer: Send + Sync {
    |           ^^^^^^^^^^^^^^

warning: struct `ComputeDomainAnalyzer` is never constructed
   --> src\correlation\cross_domain_engine.rs:324:8
    |
324 | struct ComputeDomainAnalyzer;
    |        ^^^^^^^^^^^^^^^^^^^^^

warning: struct `StorageDomainAnalyzer` is never constructed
   --> src\correlation\cross_domain_engine.rs:342:8
    |
342 | struct StorageDomainAnalyzer;
    |        ^^^^^^^^^^^^^^^^^^^^^

warning: struct `NetworkDomainAnalyzer` is never constructed
   --> src\correlation\cross_domain_engine.rs:357:8
    |
357 | struct NetworkDomainAnalyzer;
    |        ^^^^^^^^^^^^^^^^^^^^^

warning: struct `IdentityDomainAnalyzer` is never constructed
   --> src\correlation\cross_domain_engine.rs:372:8
    |
372 | struct IdentityDomainAnalyzer;
    |        ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `DatabaseDomainAnalyzer` is never constructed
   --> src\correlation\cross_domain_engine.rs:387:8
    |
387 | struct DatabaseDomainAnalyzer;
    |        ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ImpactCalculator` is never constructed
   --> src\correlation\cross_domain_engine.rs:402:12
    |
402 | pub struct ImpactCalculator {
    |            ^^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_matrix`, and `calculate_cascade_impact` are never used
   --> src\correlation\cross_domain_engine.rs:407:12
    |
406 | impl ImpactCalculator {
    | --------------------- associated items in this implementation
407 |     pub fn new() -> Self {
    |            ^^^
...
415 |     fn initialize_matrix(&mut self) {
    |        ^^^^^^^^^^^^^^^^^
...
429 |     pub fn calculate_cascade_impact(&self, affected_domain: &str) -> HashMap<String, f64> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceNode` is never constructed
   --> src\correlation\cross_domain_engine.rs:439:12
    |
439 | pub struct ResourceNode {
    |            ^^^^^^^^^^^^

warning: struct `DependencyEdge` is never constructed
   --> src\correlation\cross_domain_engine.rs:448:12
    |
448 | pub struct DependencyEdge {
    |            ^^^^^^^^^^^^^^

warning: enum `DependencyType` is never used
   --> src\correlation\cross_domain_engine.rs:455:10
    |
455 | pub enum DependencyType {
    |          ^^^^^^^^^^^^^^

warning: struct `CorrelationRule` is never constructed
   --> src\correlation\cross_domain_engine.rs:503:12
    |
503 | pub struct CorrelationRule {
    |            ^^^^^^^^^^^^^^^

warning: struct `ResourceMapper` is never constructed
  --> src\correlation\resource_mapper.rs:19:12
   |
19 | pub struct ResourceMapper {
   |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\resource_mapper.rs:26:12
    |
25  | impl ResourceMapper {
    | ------------------- associated items in this implementation
26  |     pub fn new() -> Self {
    |            ^^^
...
35  |     pub fn build_map(&mut self, resources: Vec<ResourceInfo>) -> ResourceMap {
    |            ^^^^^^^^^
...
100 |     pub fn get_dependencies(&self, resource_id: &str) -> DependencyMap {
    |            ^^^^^^^^^^^^^^^^
...
144 |     fn clear(&mut self) {
    |        ^^^^^
...
150 |     fn infer_dependencies(&self, resource: &ResourceInfo, all_resources: &[ResourceInfo]) -> Vec<(String, DependencyType)> {
    |        ^^^^^^^^^^^^^^^^^^
...
185 |     fn find_dependency_chains(&self) -> Vec<DependencyChain> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
231 |     fn find_resource_clusters(&self) -> Vec<ResourceCluster> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
258 |     fn find_connected_component(&self, start: NodeIndex, visited: &mut HashSet<NodeIndex>) -> Vec<NodeIndex> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
280 |     fn identify_critical_resources(&self) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
298 |     fn calculate_max_depth(&self) -> usize {
    |        ^^^^^^^^^^^^^^^^^^^
...
317 |     fn bfs_depth(&self, start: NodeIndex) -> usize {
    |        ^^^^^^^^^
...
339 |     fn find_transitive_dependencies(&self, start: NodeIndex) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
367 |     fn calculate_impact_level(&self, dependency: &Dependency) -> ImpactLevel {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
379 |     fn calculate_total_impact(&self, resource_id: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
396 |     fn is_chain_critical(&self, path: &[NodeIndex]) -> bool {
    |        ^^^^^^^^^^^^^^^^^
...
410 |     fn calculate_cluster_density(&self, nodes: &[NodeIndex]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `Resource` is never constructed
   --> src\correlation\resource_mapper.rs:432:8
    |
432 | struct Resource {
    |        ^^^^^^^^

warning: struct `Dependency` is never constructed
   --> src\correlation\resource_mapper.rs:441:8
    |
441 | struct Dependency {
    |        ^^^^^^^^^^

warning: struct `ImpactAnalyzer` is never constructed
  --> src\correlation\impact_analyzer.rs:17:12
   |
17 | pub struct ImpactAnalyzer {
   |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\impact_analyzer.rs:24:12
    |
23  | impl ImpactAnalyzer {
    | ------------------- associated items in this implementation
24  |     pub fn new() -> Self {
    |            ^^^
...
36  |     fn initialize_models(&mut self) {
    |        ^^^^^^^^^^^^^^^^^
...
70  |     fn initialize_rules(&mut self) {
    |        ^^^^^^^^^^^^^^^^
...
94  |     pub fn analyze_impact(&self, event: ImpactEvent, resources: &[ResourceContext]) -> ImpactAssessment {
    |            ^^^^^^^^^^^^^^
...
160 |     fn calculate_initial_impact(&self, event: &ImpactEvent) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
173 |     fn find_dependent_resources<'a>(&self, resource_id: &str, resources: &'a [ResourceContext]) -> Vec<&'a ResourceContext> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
179 |     fn calculate_cascade_impact(&self, source_id: &str, dependent: &ResourceContext, source_impact: f64) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
195 |     fn determine_impact_type(&self, event: &ImpactEvent, resource: &ResourceContext) -> ImpactType {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
205 |     fn calculate_severity(&self, impact_score: f64) -> Severity {
    |        ^^^^^^^^^^^^^^^^^^
...
217 |     fn estimate_propagation_delay(&self, source: &str, target: &str) -> u32 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
228 |     fn calculate_business_impact(&self, affected: &HashMap<String, f64>, resources: &[ResourceContext]) -> BusinessImpact {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
257 |     fn assess_compliance_impact(&self, affected: &HashMap<String, f64>) -> ComplianceImpact {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
269 |     fn assess_reputation_impact(&self, affected_users: u32) -> ReputationImpact {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
279 |     fn generate_mitigation_strategies(&self, event: &ImpactEvent, cascades: &[CascadeEffect]) -> Vec<MitigationStrategy> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
325 |     fn calculate_total_impact(&self, affected: &HashMap<String, f64>) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
337 |     fn estimate_recovery_time(&self, event: &ImpactEvent, affected: &HashMap<String, f64>) -> u32 {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
352 |     fn determine_risk_level(&self, affected: &HashMap<String, f64>) -> RiskLevel {
    |        ^^^^^^^^^^^^^^^^^^^^
...
366 |     fn get_resource_domain(&self, resource_type: &str) -> String {
    |        ^^^^^^^^^^^^^^^^^^^

warning: struct `ImpactModel` is never constructed
   --> src\correlation\impact_analyzer.rs:383:8
    |
383 | struct ImpactModel {
    |        ^^^^^^^^^^^

warning: struct `PropagationRule` is never constructed
   --> src\correlation\impact_analyzer.rs:390:8
    |
390 | struct PropagationRule {
    |        ^^^^^^^^^^^^^^^

warning: struct `AdvancedCorrelationEngine` is never constructed
  --> src\correlation\advanced_correlation_engine.rs:17:12
   |
17 | pub struct AdvancedCorrelationEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\advanced_correlation_engine.rs:26:12
    |
25  | impl AdvancedCorrelationEngine {
    | ------------------------------ associated items in this implementation
26  |     pub fn new() -> Self {
    |            ^^^
...
37  |     pub async fn analyze_advanced_correlations(&mut self,
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
77  |     async fn predict_future_correlations(&self,
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
99  |     fn extrapolate_pattern(&self,
    |        ^^^^^^^^^^^^^^^^^^^
...
122 |     fn calculate_pattern_strength(&self, pattern: &TemporalPattern, events: &[ResourceEvent]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
144 |     fn check_precondition(&self, precondition: &PatternPrecondition, events: &[&ResourceEvent]) -> bool {
    |        ^^^^^^^^^^^^^^^^^^
...
163 |     fn calculate_correlation_confidence(&self, correlations: &[MLCorrelation]) -> HashMap<String, f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
187 |     fn calculate_data_quality_factor(&self, correlation: &MLCorrelation) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
208 |     fn calculate_consistency_factor(&self, correlation: &MLCorrelation) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
226 |     fn generate_correlation_insights(&self,
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
287 |     fn identify_temporal_clusters(&self, patterns: &[TemporalPattern]) -> Vec<TemporalCluster> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
315 |     fn identify_critical_correlation_paths(&self, correlations: &[MLCorrelation]) -> Vec<CriticalCorrelationPath> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
342 |     fn detect_correlation_anomalies(&self, events: &[ResourceEvent]) -> Vec<CorrelationAnomaly> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
378 |     fn pattern_exists_in_events(&self, pattern: &TemporalPattern, events: &[&ResourceEvent]) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `TemporalGraph` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:388:12
    |
388 | pub struct TemporalGraph {
    |            ^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\advanced_correlation_engine.rs:395:12
    |
394 | impl TemporalGraph {
    | ------------------ associated items in this implementation
395 |     pub fn new() -> Self {
    |            ^^^
...
403 |     pub fn update(&mut self, resources: &[AzureResource], events: &[ResourceEvent]) {
    |            ^^^^^^
...
417 |     fn take_snapshot(&mut self) {
    |        ^^^^^^^^^^^^^
...
427 |     fn update_nodes(&mut self, resources: &[AzureResource]) {
    |        ^^^^^^^^^^^^
...
443 |     fn update_edges(&mut self, events: &[ResourceEvent]) {
    |        ^^^^^^^^^^^^
...
461 |     fn get_domain(&self, resource_type: &str) -> String {
    |        ^^^^^^^^^^
...
473 |     fn calculate_avg_degree(&self) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
485 |     pub fn calculate_metrics(&self) -> GraphMetrics {
    |            ^^^^^^^^^^^^^^^^^
...
495 |     fn calculate_density(&self) -> f64 {
    |        ^^^^^^^^^^^^^^^^^
...
505 |     fn calculate_clustering_coefficient(&self) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PatternDetector` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:540:12
    |
540 | pub struct PatternDetector {
    |            ^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_patterns`, `detect_temporal_patterns`, and `match_pattern` are never used
   --> src\correlation\advanced_correlation_engine.rs:545:12
    |
544 | impl PatternDetector {
    | -------------------- associated items in this implementation
545 |     pub fn new() -> Self {
    |            ^^^
...
553 |     fn initialize_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^
...
579 |     pub fn detect_temporal_patterns(&self, events: &[ResourceEvent], window: Duration) -> Vec<TemporalPattern> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
...
596 |     fn match_pattern(&self, template: &PatternTemplate, events: &[&ResourceEvent]) -> Option<TemporalPattern> {
    |        ^^^^^^^^^^^^^

warning: struct `MLCorrelator` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:631:12
    |
631 | pub struct MLCorrelator {
    |            ^^^^^^^^^^^^

warning: associated items `new`, `find_correlations`, `predict_correlations`, `extract_features`, `calculate_similarity`, and `calculate_feature_variance` are never used
   --> src\correlation\advanced_correlation_engine.rs:636:12
    |
635 | impl MLCorrelator {
    | ----------------- associated items in this implementation
636 |     pub fn new() -> Self {
    |            ^^^
...
646 |     pub async fn find_correlations(&self,
    |                  ^^^^^^^^^^^^^^^^^
...
678 |     pub async fn predict_correlations(&self,
    |                  ^^^^^^^^^^^^^^^^^^^^
...
687 |     fn extract_features(&self, resources: &[AzureResource], events: &[ResourceEvent]) -> Vec<ResourceFeatures> {
    |        ^^^^^^^^^^^^^^^^
...
717 |     fn calculate_similarity(&self, features1: &ResourceFeatures, features2: &ResourceFeatures) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
736 |     fn calculate_feature_variance(&self, features: &HashMap<String, f64>) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `RealTimeAnalyzer` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:751:12
    |
751 | pub struct RealTimeAnalyzer {
    |            ^^^^^^^^^^^^^^^^

warning: associated items `new`, `analyze`, `find_real_time_correlations`, and `calculate_temporal_correlation` are never used
   --> src\correlation\advanced_correlation_engine.rs:757:12
    |
756 | impl RealTimeAnalyzer {
    | --------------------- associated items in this implementation
757 |     pub fn new() -> Self {
    |            ^^^
...
764 |     pub async fn analyze(&mut self, events: &[ResourceEvent]) -> Vec<RealTimeCorrelation> {
    |                  ^^^^^^^
...
784 |     fn find_real_time_correlations(&self) -> Vec<RealTimeCorrelation> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
813 |     fn calculate_temporal_correlation(&self, event1: &ResourceEvent, event2: &ResourceEvent) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `CorrelationMemory` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:836:12
    |
836 | pub struct CorrelationMemory {
    |            ^^^^^^^^^^^^^^^^^

warning: associated items `new`, `get_historical_patterns`, `get_historical_accuracy`, `find_similar_correlations`, `get_expected_patterns_for_time`, and `pattern_matches_time` are never used
   --> src\correlation\advanced_correlation_engine.rs:843:12
    |
842 | impl CorrelationMemory {
    | ---------------------- associated items in this implementation
843 |     pub fn new() -> Self {
    |            ^^^
...
851 |     pub fn get_historical_patterns(&self) -> &[TemporalPattern] {
    |            ^^^^^^^^^^^^^^^^^^^^^^^
...
855 |     pub fn get_historical_accuracy(&self, correlation_id: &str) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^^^^^
...
859 |     pub fn find_similar_correlations(&self, source_id: &str, target_id: &str) -> Vec<HistoricalCorrelation> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^
...
864 |     pub fn get_expected_patterns_for_time(&self, _time: DateTime<Utc>) -> Vec<TemporalPattern> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
872 |     fn pattern_matches_time(&self, _pattern: &TemporalPattern, _time: DateTime<Utc>) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^

warning: trait `FeatureExtractor` is never used
   --> src\correlation\advanced_correlation_engine.rs:879:11
    |
879 | pub trait FeatureExtractor: Send + Sync {
    |           ^^^^^^^^^^^^^^^^

warning: struct `ResourceFeatureExtractor` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:883:12
    |
883 | pub struct ResourceFeatureExtractor;
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `EventFeatureExtractor` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:904:12
    |
904 | pub struct EventFeatureExtractor;
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `TemporalFeatureExtractor` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:924:12
    |
924 | pub struct TemporalFeatureExtractor;
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceNode` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:961:12
    |
961 | pub struct ResourceNode {
    |            ^^^^^^^^^^^^

warning: struct `TemporalEdge` is never constructed
   --> src\correlation\advanced_correlation_engine.rs:970:12
    |
970 | pub struct TemporalEdge {
    |            ^^^^^^^^^^^^

warning: struct `TemporalCluster` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1136:12
     |
1136 | pub struct TemporalCluster {
     |            ^^^^^^^^^^^^^^^

warning: struct `CriticalCorrelationPath` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1144:12
     |
1144 | pub struct CriticalCorrelationPath {
     |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `GraphSnapshot` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1175:12
     |
1175 | pub struct GraphSnapshot {
     |            ^^^^^^^^^^^^^

warning: struct `PatternTemplate` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1192:12
     |
1192 | pub struct PatternTemplate {
     |            ^^^^^^^^^^^^^^^

warning: struct `ResourceFeatures` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1201:12
     |
1201 | pub struct ResourceFeatures {
     |            ^^^^^^^^^^^^^^^^

warning: struct `HistoricalCorrelation` is never constructed
    --> src\correlation\advanced_correlation_engine.rs:1209:12
     |
1209 | pub struct HistoricalCorrelation {
     |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `PredictiveImpactAnalyzer` is never constructed
  --> src\correlation\predictive_impact_analyzer.rs:37:12
   |
37 | pub struct PredictiveImpactAnalyzer {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\predictive_impact_analyzer.rs:46:12
    |
45  | impl PredictiveImpactAnalyzer {
    | ----------------------------- associated items in this implementation
46  |     pub fn new() -> Self {
    |            ^^^
...
59  |     fn initialize_advanced_models(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
101 |     pub async fn predict_impact(&self,
    |                  ^^^^^^^^^^^^^^
...
157 |     pub async fn what_if_analysis(&self,
    |                  ^^^^^^^^^^^^^^^^
...
197 |     pub async fn analyze_real_time_impact(&mut self,
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
231 |     fn predict_cascade_effects(&self,
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
278 |     fn apply_variation(&self, base_scenario: &ImpactScenario, variation: &ScenarioVariation) -> ImpactScenario {
    |        ^^^^^^^^^^^^^^^
...
308 |     fn compare_scenarios(&self, base: &PredictiveImpactResult, variation: &PredictiveImpactResult) -> ScenarioComparison {
    |        ^^^^^^^^^^^^^^^^^
...
319 |     fn generate_comparative_insights(&self,
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
377 |     fn identify_optimal_scenarios(&self, scenarios: &[WhatIfScenarioResult]) -> Vec<OptimalScenario> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
421 |     fn calculate_trade_offs(&self, comparison: &ScenarioComparison) -> Vec<TradeOff> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
444 |     fn create_compute_propagation_matrix(&self) -> HashMap<String, f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
452 |     fn create_compute_recovery_patterns(&self) -> Vec<RecoveryPattern> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
469 |     fn create_compute_cost_models(&self) -> HashMap<String, CostModel> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
479 |     fn create_storage_propagation_matrix(&self) -> HashMap<String, f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
487 |     fn create_storage_recovery_patterns(&self) -> Vec<RecoveryPattern> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
498 |     fn create_storage_cost_models(&self) -> HashMap<String, CostModel> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
508 |     fn create_network_propagation_matrix(&self) -> HashMap<String, f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
516 |     fn create_network_recovery_patterns(&self) -> Vec<RecoveryPattern> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
527 |     fn create_network_cost_models(&self) -> HashMap<String, CostModel> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
538 |     fn calculate_propagation_delay(&self, source: &str, target: &str) -> u32 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
549 |     fn calculate_cascade_confidence(&self, source: &str, target: &str, impact: f64) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
557 |     fn trace_propagation_path(&self, source: &str, target: &str, resources: &[ResourceContext]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
562 |     fn calculate_peak_impact_time(&self, effects: &[PredictedCascadeEffect]) -> DateTime<Utc> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
569 |     fn project_recovery_scenarios(&self, scenario: &ImpactScenario, resources: &[ResourceContext]) -> RecoveryProjections {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
579 |     fn identify_critical_dependencies(&self, resources: &[ResourceContext]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
586 |     fn generate_predictive_mitigations(&self, scenario: &ImpactScenario, cascade_effects: &[PredictedCascadeEffect]) -> Vec<Mi...
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
616 |     fn calculate_prediction_confidence(&self, scenario: &ImpactScenario, historical_data: &[HistoricalEvent]) -> ConfidenceMet...
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
629 |     fn calculate_historical_accuracy(&self, historical_data: &[HistoricalEvent]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
634 |     fn calculate_scenario_complexity(&self, scenario: &ImpactScenario) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
641 |     fn compare_risk_levels(&self, base: &RiskLevel, variation: &RiskLevel) -> i8 {
    |        ^^^^^^^^^^^^^^^^^^^
...
648 |     fn risk_level_to_score(&self, risk_level: &RiskLevel) -> u8 {
    |        ^^^^^^^^^^^^^^^^^^^
...
657 |     fn calculate_current_impact_state(&self, event: &OngoingEvent, state: &SystemState) -> CurrentImpactState {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
666 |     fn assess_current_severity(&self, event: &OngoingEvent) -> Severity {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
671 |     fn calculate_propagation_rate(&self, event: &OngoingEvent) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
676 |     fn predict_remaining_cascade(&self, event: &OngoingEvent, _state: &SystemState, resources: &[ResourceContext]) -> ImpactTi...
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
683 |     fn update_mitigation_recommendations(&self, event: &OngoingEvent, current_impact: &CurrentImpactState) -> Vec<DynamicMitig...
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
694 |     fn identify_intervention_windows(&self, timeline: &ImpactTimeline) -> Vec<InterventionWindow> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
698 |     fn calculate_real_time_accuracy(&self, event: &OngoingEvent) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
703 |     fn calculate_dynamic_risk_score(&self, current_impact: &CurrentImpactState, remaining_timeline: &ImpactTimeline) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
710 |     fn perform_sensitivity_analysis(&self, scenario: &ImpactScenario, resources: &[ResourceContext]) -> SensitivityAnalysis {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ScenarioCache` is never constructed
   --> src\correlation\predictive_impact_analyzer.rs:725:12
    |
725 | pub struct ScenarioCache {
    |            ^^^^^^^^^^^^^

warning: associated items `new`, `get`, `insert`, and `scenario_to_key` are never used
   --> src\correlation\predictive_impact_analyzer.rs:731:12
    |
730 | impl ScenarioCache {
    | ------------------ associated items in this implementation
731 |     pub fn new() -> Self {
    |            ^^^
...
738 |     pub fn get(&self, scenario: &ImpactScenario) -> Option<PredictiveImpactResult> {
    |            ^^^
...
743 |     pub fn insert(&mut self, scenario: ImpactScenario, result: PredictiveImpactResult) {
    |            ^^^^^^
...
755 |     fn scenario_to_key(&self, scenario: &ImpactScenario) -> String {
    |        ^^^^^^^^^^^^^^^

warning: struct `PredictiveEngine` is never constructed
   --> src\correlation\predictive_impact_analyzer.rs:764:12
    |
764 | pub struct PredictiveEngine {
    |            ^^^^^^^^^^^^^^^^

warning: associated items `new`, `predict_timeline`, and `update_with_real_time_data` are never used
   --> src\correlation\predictive_impact_analyzer.rs:769:12
    |
768 | impl PredictiveEngine {
    | --------------------- associated items in this implementation
769 |     pub fn new() -> Self {
    |            ^^^
...
775 |     pub fn predict_timeline(&self,
    |            ^^^^^^^^^^^^^^^^
...
806 |     pub fn update_with_real_time_data(&mut self, _event: &OngoingEvent) {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `WhatIfAnalyzer` is never constructed
   --> src\correlation\predictive_impact_analyzer.rs:811:12
    |
811 | pub struct WhatIfAnalyzer {
    |            ^^^^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\correlation\predictive_impact_analyzer.rs:816:12
    |
815 | impl WhatIfAnalyzer {
    | ------------------- associated function in this implementation
816 |     pub fn new() -> Self {
    |            ^^^

warning: struct `RiskQuantifier` is never constructed
   --> src\correlation\predictive_impact_analyzer.rs:823:12
    |
823 | pub struct RiskQuantifier {
    |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\correlation\predictive_impact_analyzer.rs:828:12
    |
827 | impl RiskQuantifier {
    | ------------------- associated items in this implementation
828 |     pub fn new() -> Self {
    |            ^^^
...
834 |     pub fn quantify_risks(&self,
    |            ^^^^^^^^^^^^^^
...
853 |     fn calculate_total_impact(&self, scenario: &ImpactScenario, cascade_effects: &[PredictedCascadeEffect]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
860 |     fn calculate_financial_impact(&self, cascade_effects: &[PredictedCascadeEffect], resources: &[ResourceContext]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
872 |     fn calculate_operational_impact(&self, cascade_effects: &[PredictedCascadeEffect], resources: &[ResourceContext]) -> Opera...
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
889 |     fn assess_compliance_risk(&self, cascade_effects: &[PredictedCascadeEffect]) -> ComplianceRisk {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
899 |     fn assess_reputation_risk(&self, cascade_effects: &[PredictedCascadeEffect], resources: &[ResourceContext]) -> ReputationR...
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
917 |     fn determine_overall_risk_level(&self, total_impact: f64) -> RiskLevel {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: method `description` is never used
    --> src\correlation\predictive_impact_analyzer.rs:1103:12
     |
1102 | impl ScenarioVariation {
     | ---------------------- method in this implementation
1103 |     pub fn description(&self) -> String {
     |            ^^^^^^^^^^^

warning: struct `SmartDependencyMapper` is never constructed
  --> src\correlation\smart_dependency_mapper.rs:21:12
   |
21 | pub struct SmartDependencyMapper {
   |            ^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
    --> src\correlation\smart_dependency_mapper.rs:32:12
     |
31   | impl SmartDependencyMapper {
     | -------------------------- associated items in this implementation
32   |     pub fn new() -> Self {
     |            ^^^
...
45   |     pub async fn build_smart_map(&mut self,
     |                  ^^^^^^^^^^^^^^^
...
95   |     pub fn get_smart_dependencies(&self, resource_id: &str) -> SmartDependencyInfo {
     |            ^^^^^^^^^^^^^^^^^^^^^^
...
114  |     pub async fn track_real_time_dependencies(&mut self,
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
145  |     pub async fn analyze_dependency_scenarios(&self,
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
173  |     fn build_basic_graph(&mut self, resources: &[SmartResourceInfo]) {
     |        ^^^^^^^^^^^^^^^^^
...
219  |     fn add_inferred_dependencies(&mut self, inferred_deps: Vec<InferredDependency>) {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
241  |     fn add_network_dependencies(&mut self, network_deps: Vec<NetworkDependency>) {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
263  |     fn add_runtime_dependencies(&mut self, runtime_deps: Vec<RuntimeDependency>) {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
285  |     fn discover_runtime_dependencies(&self, runtime_data: &[RuntimeMetric]) -> Vec<RuntimeDependency> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
315  |     fn calculate_metric_correlation(&self,
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
343  |     fn calculate_advanced_metrics(&self) -> DependencyMetrics {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
361  |     fn calculate_clustering_coefficient(&self) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
390  |     fn calculate_graph_diameter(&self) -> usize {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
403  |     fn calculate_centrality_measures(&self) -> CentralityMeasures {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
436  |     fn generate_dependency_insights(&self, _resources: &[SmartResourceInfo]) -> Vec<DependencyInsight> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
501  |     fn identify_critical_paths(&self) -> Vec<CriticalPath> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
565  |     fn calculate_path_criticality(&self, source: NodeIndex, target: NodeIndex) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
577  |     fn identify_bottlenecks_in_path(&self, source: NodeIndex, target: NodeIndex) -> Vec<String> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
582  |     fn assess_dependency_risks(&self) -> DependencyRiskAssessment {
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
624  |     fn calculate_overall_dependency_risk(&self,
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
641  |     fn generate_risk_mitigation_actions(&self, risk_score: f64) -> Vec<String> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
662  |     fn get_direct_dependencies(&self, node_idx: NodeIndex) -> Vec<DirectDependency> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
674  |     fn get_transitive_dependencies(&self, node_idx: NodeIndex) -> Vec<String> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
704  |     fn get_reverse_dependencies(&self, node_idx: NodeIndex) -> Vec<ReverseDependency> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
714  |     fn detect_circular_dependencies(&self, node_idx: NodeIndex) -> Vec<CircularDependency> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
738  |     fn calculate_dependency_strengths(&self, node_idx: NodeIndex) -> HashMap<String, f64> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
750  |     fn calculate_resource_criticality(&self, node_idx: NodeIndex) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
761  |     fn calculate_blast_radius(&self, node_idx: NodeIndex) -> BlastRadius {
     |        ^^^^^^^^^^^^^^^^^^^^^^
...
796  |     fn identify_recovery_dependencies(&self, node_idx: NodeIndex) -> Vec<RecoveryDependency> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
819  |     fn clear(&mut self) {
     |        ^^^^^
...
825  |     fn count_explicit_dependencies(&self) -> usize {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
831  |     fn count_inferred_dependencies(&self) -> usize {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
837  |     fn count_runtime_dependencies(&self) -> usize {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
843  |     fn find_smart_dependency_chains(&self) -> Vec<SmartDependencyChain> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
848  |     fn find_smart_clusters(&self) -> Vec<SmartResourceCluster> {
     |        ^^^^^^^^^^^^^^^^^^^
...
853  |     fn apply_dependency_change(&mut self, change: &DependencyChange) {
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
892  |     fn detect_dependency_anomalies(&self, events: &[ResourceEvent], metrics: &[RuntimeMetric]) -> Vec<DependencyAnomaly> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
913  |     fn calculate_graph_stability(&self) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
926  |     fn generate_real_time_recommendations(&self, changes: &[DependencyChange]) -> Vec<String> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
945  |     fn apply_scenario_to_graph(&self,
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
992  |     fn analyze_scenario_impact(&self,
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
1004 |     fn calculate_stability_change(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1020 |     fn find_new_critical_paths(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> Vec<String> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^
...
1025 |     fn calculate_risk_change(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> f64 {
     |        ^^^^^^^^^^^^^^^^^^^^^

warning: struct `DependencyInferenceEngine` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1033:12
     |
1033 | pub struct DependencyInferenceEngine {
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `infer_dependencies` are never used
    --> src\correlation\smart_dependency_mapper.rs:1038:12
     |
1037 | impl DependencyInferenceEngine {
     | ------------------------------ associated items in this implementation
1038 |     pub fn new() -> Self {
     |            ^^^
...
1044 |     pub async fn infer_dependencies(&self,
     |                  ^^^^^^^^^^^^^^^^^^

warning: struct `TopologyAnalyzer` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1053:12
     |
1053 | pub struct TopologyAnalyzer {
     |            ^^^^^^^^^^^^^^^^

warning: associated items `new` and `analyze_network_dependencies` are never used
    --> src\correlation\smart_dependency_mapper.rs:1058:12
     |
1057 | impl TopologyAnalyzer {
     | --------------------- associated items in this implementation
1058 |     pub fn new() -> Self {
     |            ^^^
...
1064 |     pub fn analyze_network_dependencies(&self,
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `DependencyPredictor` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1073:12
     |
1073 | pub struct DependencyPredictor {
     |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `predict_future_dependencies`, and `update_predictions` are never used
    --> src\correlation\smart_dependency_mapper.rs:1078:12
     |
1077 | impl DependencyPredictor {
     | ------------------------ associated items in this implementation
1078 |     pub fn new() -> Self {
     |            ^^^
...
1084 |     pub async fn predict_future_dependencies(&self,
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1092 |     pub async fn update_predictions(&self,
     |                  ^^^^^^^^^^^^^^^^^^

warning: struct `RealTimeDependencyTracker` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1101:12
     |
1101 | pub struct RealTimeDependencyTracker {
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `process_events` are never used
    --> src\correlation\smart_dependency_mapper.rs:1107:12
     |
1106 | impl RealTimeDependencyTracker {
     | ------------------------------ associated items in this implementation
1107 |     pub fn new() -> Self {
     |            ^^^
...
1114 |     pub fn process_events(&mut self,
     |            ^^^^^^^^^^^^^^

warning: trait `NetworkAnalyzer` is never used
    --> src\correlation\smart_dependency_mapper.rs:1124:11
     |
1124 | pub trait NetworkAnalyzer: Send + Sync {
     |           ^^^^^^^^^^^^^^^

warning: trait `PatternDetector` is never used
    --> src\correlation\smart_dependency_mapper.rs:1128:11
     |
1128 | pub trait PatternDetector: Send + Sync {
     |           ^^^^^^^^^^^^^^^

warning: struct `InferredDependency` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1571:12
     |
1571 | pub struct InferredDependency {
     |            ^^^^^^^^^^^^^^^^^^

warning: struct `NetworkDependency` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1583:12
     |
1583 | pub struct NetworkDependency {
     |            ^^^^^^^^^^^^^^^^^

warning: struct `RuntimeDependency` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1592:12
     |
1592 | pub struct RuntimeDependency {
     |            ^^^^^^^^^^^^^^^^^

warning: struct `MetricCorrelation` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1601:12
     |
1601 | pub struct MetricCorrelation {
     |            ^^^^^^^^^^^^^^^^^

warning: enum `CorrelationType` is never used
    --> src\correlation\smart_dependency_mapper.rs:1608:10
     |
1608 | pub enum CorrelationType {
     |          ^^^^^^^^^^^^^^^

warning: struct `InferenceModel` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1628:12
     |
1628 | pub struct InferenceModel {
     |            ^^^^^^^^^^^^^^

warning: struct `PredictionModel` is never constructed
    --> src\correlation\smart_dependency_mapper.rs:1635:12
     |
1635 | pub struct PredictionModel {
     |            ^^^^^^^^^^^^^^^

warning: trait `GraphDriver` is never used
   --> src\correlation\graph_driver.rs:105:11
    |
105 | pub trait GraphDriver: Send + Sync {
    |           ^^^^^^^^^^^

warning: struct `Neo4jDriver` is never constructed
   --> src\correlation\graph_driver.rs:179:12
    |
179 | pub struct Neo4jDriver {
    |            ^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\correlation\graph_driver.rs:185:12
    |
184 | impl Neo4jDriver {
    | ---------------- associated function in this implementation
185 |     pub fn new() -> Self {
    |            ^^^

warning: function `create_graph_driver` is never used
   --> src\correlation\graph_driver.rs:551:8
    |
551 | pub fn create_graph_driver(backend: GraphBackend) -> Box<dyn GraphDriver> {
    |        ^^^^^^^^^^^^^^^^^^^

warning: type alias `HmacSha256` is never used
  --> src\defender_streaming.rs:15:6
   |
15 | type HmacSha256 = Hmac<Sha256>;
   |      ^^^^^^^^^^

warning: struct `DefenderStreamingService` is never constructed
  --> src\defender_streaming.rs:69:12
   |
69 | pub struct DefenderStreamingService {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\defender_streaming.rs:79:18
    |
78  | impl DefenderStreamingService {
    | ----------------------------- associated items in this implementation
79  |     pub async fn new(
    |                  ^^^
...
98  |     pub async fn start_streaming(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    |                  ^^^^^^^^^^^^^^^
...
146 |     async fn fetch_and_process_alerts(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
210 |     fn parse_alert(&self, json: &serde_json::Value) -> Result<DefenderAlert, Box<dyn std::error::Error + Send + Sync>> {
    |        ^^^^^^^^^^^
...
240 |     fn parse_entities(&self, entities_json: &serde_json::Value) -> Vec<AlertEntity> {
    |        ^^^^^^^^^^^^^^
...
258 |     fn verify_signature(&self, alert: &DefenderAlert) -> bool {
    |        ^^^^^^^^^^^^^^^^
...
271 |     async fn send_to_correlation(&self, alert: &DefenderAlert) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    |              ^^^^^^^^^^^^^^^^^^^
...
286 |     pub fn get_alert_receiver(&self) -> mpsc::Receiver<DefenderAlert> {
    |            ^^^^^^^^^^^^^^^^^^

warning: function `handle_defender_webhook` is never used
   --> src\defender_streaming.rs:294:14
    |
294 | pub async fn handle_defender_webhook(
    |              ^^^^^^^^^^^^^^^^^^^^^^^

warning: methods `is_simulated`, `display_string`, and `css_class` are never used
  --> src\data_mode.rs:36:12
   |
21 | impl DataMode {
   | ------------- methods in this implementation
...
36 |     pub fn is_simulated(&self) -> bool {
   |            ^^^^^^^^^^^^
...
41 |     pub fn display_string(&self) -> &'static str {
   |            ^^^^^^^^^^^^^^
...
49 |     pub fn css_class(&self) -> &'static str {
   |            ^^^^^^^^^

warning: methods `ensure_real_data` and `get_mode` are never used
   --> src\data_mode.rs:133:12
    |
117 | impl DataModeGuard {
    | ------------------ methods in this implementation
...
133 |     pub fn ensure_real_data(&self) -> Result<(), DataModeError> {
    |            ^^^^^^^^^^^^^^^^
...
140 |     pub fn get_mode(&self) -> DataMode {
    |            ^^^^^^^^

warning: variants `RealDataRequired` and `ConnectionFailed` are never constructed
   --> src\data_mode.rs:151:5
    |
146 | pub enum DataModeError {
    |          ------------- variants in this enum
...
151 |     RealDataRequired,
    |     ^^^^^^^^^^^^^^^^
...
154 |     ConnectionFailed(String),
    |     ^^^^^^^^^^^^^^^^
    |
    = note: `DataModeError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis

warning: trait `EnforcementEngine` is never used
   --> src\enforcement.rs:156:11
    |
156 | pub trait EnforcementEngine: Send + Sync {
    |           ^^^^^^^^^^^^^^^^^

warning: struct `DefaultEnforcementEngine` is never constructed
   --> src\enforcement.rs:204:12
    |
204 | pub struct DefaultEnforcementEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `determine_enforcement_type`, and `requires_approval` are never used
   --> src\enforcement.rs:210:12
    |
209 | impl DefaultEnforcementEngine {
    | ----------------------------- associated items in this implementation
210 |     pub fn new(
    |            ^^^
...
221 |     fn determine_enforcement_type(&self, violations: &[Violation]) -> EnforcementType {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
235 |     fn requires_approval(&self, action: &EnforcementType, resource: &Resource) -> bool {
    |        ^^^^^^^^^^^^^^^^^

warning: method `determine_priority` is never used
   --> src\enforcement.rs:402:8
    |
401 | impl DefaultEnforcementEngine {
    | ----------------------------- method in this implementation
402 |     fn determine_priority(&self, violations: &[Violation]) -> Priority {
    |        ^^^^^^^^^^^^^^^^^^

warning: trait `Enforcer` is never used
   --> src\enforcement.rs:419:11
    |
419 | pub trait Enforcer: Send + Sync {
    |           ^^^^^^^^

warning: struct `EnforcerResult` is never constructed
   --> src\enforcement.rs:431:12
    |
431 | pub struct EnforcerResult {
    |            ^^^^^^^^^^^^^^

warning: struct `DefaultEnforcer` is never constructed
   --> src\enforcement.rs:439:12
    |
439 | pub struct DefaultEnforcer;
    |            ^^^^^^^^^^^^^^^

warning: trait `ApprovalService` is never used
   --> src\enforcement.rs:507:11
    |
507 | pub trait ApprovalService: Send + Sync {
    |           ^^^^^^^^^^^^^^^

warning: associated items `new`, `with_correlation_id`, and `with_metadata` are never used
   --> src\events\mod.rs:143:12
    |
142 | impl EventEnvelope {
    | ------------------ associated items in this implementation
143 |     pub fn new(event: GovernanceEvent, tenant_id: String, source: String) -> Self {
    |            ^^^
...
155 |     pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
    |            ^^^^^^^^^^^^^^^^^^^
...
160 |     pub fn with_metadata(mut self, key: String, value: String) -> Self {
    |            ^^^^^^^^^^^^^

warning: trait `EventBus` is never used
   --> src\events\mod.rs:168:11
    |
168 | pub trait EventBus: Send + Sync {
    |           ^^^^^^^^

warning: struct `InMemoryEventBus` is never constructed
   --> src\events\mod.rs:303:12
    |
303 | pub struct InMemoryEventBus {
    |            ^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\events\mod.rs:308:12
    |
307 | impl InMemoryEventBus {
    | --------------------- associated function in this implementation
308 |     pub fn new() -> Self {
    |            ^^^

warning: struct `EventSubscription` is never constructed
   --> src\events\mod.rs:336:12
    |
336 | pub struct EventSubscription {
    |            ^^^^^^^^^^^^^^^^^

warning: associated items `new`, `add_broadcast_receiver`, and `next` are never used
   --> src\events\mod.rs:343:8
    |
342 | impl EventSubscription {
    | ---------------------- associated items in this implementation
343 |     fn new() -> Self {
    |        ^^^
...
356 |     fn add_broadcast_receiver(&mut self, receiver: broadcast::Receiver<EventEnvelope>) {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
360 |     pub async fn next(&mut self) -> Option<EventEnvelope> {
    |                  ^^^^

warning: struct `EventProcessor` is never constructed
   --> src\events\mod.rs:385:12
    |
385 | pub struct EventProcessor {
    |            ^^^^^^^^^^^^^^

warning: associated items `new`, `register_handler`, and `process` are never used
   --> src\events\mod.rs:390:12
    |
389 | impl EventProcessor {
    | ------------------- associated items in this implementation
390 |     pub fn new() -> Self {
    |            ^^^
...
396 |     pub fn register_handler(&mut self, event_type: String, handler: Box<dyn EventHandler>) {
    |            ^^^^^^^^^^^^^^^^
...
400 |     pub async fn process(&self, event: EventEnvelope) -> Result<(), EventError> {
    |                  ^^^^^^^

warning: trait `EventHandler` is never used
   --> src\events\mod.rs:428:11
    |
428 | pub trait EventHandler: Send + Sync {
    |           ^^^^^^^^^^^^

warning: enum `EventError` is never used
   --> src\events\mod.rs:434:10
    |
434 | pub enum EventError {
    |          ^^^^^^^^^^

warning: struct `EventAggregator` is never constructed
   --> src\events\mod.rs:450:12
    |
450 | pub struct EventAggregator {
    |            ^^^^^^^^^^^^^^^

warning: associated items `new`, `add_event`, and `get_statistics` are never used
   --> src\events\mod.rs:456:12
    |
455 | impl EventAggregator {
    | -------------------- associated items in this implementation
456 |     pub fn new(window_size: std::time::Duration) -> Self {
    |            ^^^
...
463 |     pub async fn add_event(&self, event: EventEnvelope) {
    |                  ^^^^^^^^^
...
472 |     pub async fn get_statistics(&self) -> EventStatistics {
    |                  ^^^^^^^^^^^^^^

warning: struct `EventStatistics` is never constructed
   --> src\events\mod.rs:493:12
    |
493 | pub struct EventStatistics {
    |            ^^^^^^^^^^^^^^^

warning: struct `EvidencePipeline` is never constructed
  --> src\evidence_pipeline.rs:21:12
   |
21 | pub struct EvidencePipeline {
   |            ^^^^^^^^^^^^^^^^

warning: struct `SigningKeys` is never constructed
   --> src\evidence_pipeline.rs:103:8
    |
103 | struct SigningKeys {
    |        ^^^^^^^^^^^

warning: struct `SigningKey` is never constructed
   --> src\evidence_pipeline.rs:109:8
    |
109 | struct SigningKey {
    |        ^^^^^^^^^^

warning: enum `SigningAlgorithm` is never used
   --> src\evidence_pipeline.rs:120:6
    |
120 | enum SigningAlgorithm {
    |      ^^^^^^^^^^^^^^^^

warning: struct `VerificationResult` is never constructed
   --> src\evidence_pipeline.rs:127:8
    |
127 | struct VerificationResult {
    |        ^^^^^^^^^^^^^^^^^^

warning: enum `StorageBackend` is never used
   --> src\evidence_pipeline.rs:135:6
    |
135 | enum StorageBackend {
    |      ^^^^^^^^^^^^^^

warning: struct `AzureBlobConfig` is never constructed
   --> src\evidence_pipeline.rs:142:8
    |
142 | struct AzureBlobConfig {
    |        ^^^^^^^^^^^^^^^

warning: struct `S3Config` is never constructed
   --> src\evidence_pipeline.rs:149:8
    |
149 | struct S3Config {
    |        ^^^^^^^^

warning: multiple associated items are never used
   --> src\evidence_pipeline.rs:155:18
    |
154 | impl EvidencePipeline {
    | --------------------- associated items in this implementation
155 |     pub async fn new(
    |                  ^^^
...
171 |     pub async fn collect_evidence(
    |                  ^^^^^^^^^^^^^^^^
...
246 |     pub async fn verify_evidence(&self, evidence_id: Uuid) -> Result<VerificationStatus, String> {
    |                  ^^^^^^^^^^^^^^^
...
323 |     pub async fn add_custody_entry(
    |                  ^^^^^^^^^^^^^^^^^
...
353 |     pub async fn export_evidence(
    |                  ^^^^^^^^^^^^^^^
...
386 |     pub async fn archive_evidence(&self, older_than: DateTime<Utc>) -> Result<u32, String> {
    |                  ^^^^^^^^^^^^^^^^
...
410 |     pub async fn search_evidence(&self, criteria: SearchCriteria) -> Result<Vec<Evidence>, String> {
    |                  ^^^^^^^^^^^^^^^
...
428 |     fn calculate_hash(&self, data: &str) -> String {
    |        ^^^^^^^^^^^^^^
...
434 |     async fn sign_evidence(&self, hash: &str) -> Result<String, String> {
    |              ^^^^^^^^^^^^^
...
446 |     async fn verify_signature(
    |              ^^^^^^^^^^^^^^^^
...
464 |     async fn initialize_signing_keys() -> Result<SigningKeys, String> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
488 |     fn get_storage_location(&self, evidence_id: &Uuid) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^
...
503 |     async fn store_evidence(&self, evidence: Evidence) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^
...
509 |     async fn persist_to_storage(&self, evidence: &Evidence) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^^^
...
515 |     async fn persist_to_database(
    |              ^^^^^^^^^^^^^^^^^^^
...
551 |     async fn archive_to_storage(&self, evidence: &Evidence) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^^^
...
557 |     fn matches_criteria(&self, evidence: &Evidence, criteria: &SearchCriteria) -> bool {
    |        ^^^^^^^^^^^^^^^^

warning: struct `SearchCriteria` is never constructed
   --> src\evidence_pipeline.rs:614:12
    |
614 | pub struct SearchCriteria {
    |            ^^^^^^^^^^^^^^

warning: enum `ExportFormat` is never used
   --> src\evidence_pipeline.rs:625:10
    |
625 | pub enum ExportFormat {
    |          ^^^^^^^^^^^^

warning: function `matches_evidence_type` is never used
   --> src\evidence_pipeline.rs:631:4
    |
631 | fn matches_evidence_type(actual: &EvidenceType, expected: &EvidenceType) -> bool {
    |    ^^^^^^^^^^^^^^^^^^^^^

warning: trait `FinOpsEngine` is never used
   --> src\finops\mod.rs:115:11
    |
115 | pub trait FinOpsEngine: Send + Sync {
    |           ^^^^^^^^^^^^

warning: enum `FinOpsError` is never used
   --> src\finops\mod.rs:138:10
    |
138 | pub enum FinOpsError {
    |          ^^^^^^^^^^^

warning: struct `AzureFinOpsEngine` is never constructed
   --> src\finops\mod.rs:148:12
    |
148 | pub struct AzureFinOpsEngine {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `CachedMetrics` is never constructed
   --> src\finops\mod.rs:154:8
    |
154 | struct CachedMetrics {
    |        ^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\finops\mod.rs:160:12
    |
159 | impl AzureFinOpsEngine {
    | ---------------------- associated items in this implementation
160 |     pub fn new(azure_client: crate::azure_client_async::AsyncAzureClient) -> Self {
    |            ^^^
...
167 |     async fn generate_optimal_schedule(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
186 |     fn parse_optimization_id<'a>(
    |        ^^^^^^^^^^^^^^^^^^^^^
...
199 |     async fn get_recommended_sku(&self, _resource_id: &str) -> Result<String, FinOpsError> {
    |              ^^^^^^^^^^^^^^^^^^^
...
203 |     async fn analyze_cause_from_changes(&self, _changes: &[serde_json::Value]) -> String {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
207 |     fn calculate_idle_days(&self, _metrics: &serde_json::Value) -> u32 {
    |        ^^^^^^^^^^^^^^^^^^^
...
211 |     async fn generate_idle_recommendation(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
224 |     fn generate_idle_recommendation_json(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
239 |     fn assess_performance_impact(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
247 |     fn calculate_confidence(&self, _metrics: &serde_json::Value) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
251 |     fn extract_vm_family(&self, meter_name: &str) -> Option<String> {
    |        ^^^^^^^^^^^^^^^^^
...
261 |     fn calculate_real_savings(
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
293 |     fn calculate_breakeven(
    |        ^^^^^^^^^^^^^^^^^^^
...
308 |     fn calculate_commitment_confidence(&self, pattern: &UsagePattern) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
316 |     fn calculate_idle_score(&self, metrics: &serde_json::Value) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
351 |     fn calculate_rightsizing(&self, current: &VmSku, metrics: &ResourceMetrics) -> Option<VmSku> {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
356 |     fn calculate_rightsizing_from_json(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
393 |     async fn optimize_commitment_mix(&self, usage: &UsagePattern) -> CommitmentMix {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
406 |     async fn detect_cost_anomaly(&self, timeseries: &[CostDataPoint]) -> Vec<CostAnomaly> {
    |              ^^^^^^^^^^^^^^^^^^^
...
449 |     async fn infer_cause(&self, point: &CostDataPoint, history: &[CostDataPoint]) -> String {
    |              ^^^^^^^^^^^
...
462 |     fn find_smaller_sku(&self, current: &VmSku) -> Option<VmSku> {
    |        ^^^^^^^^^^^^^^^^

warning: struct `UsagePattern` is never constructed
   --> src\finops\mod.rs:524:8
    |
524 | struct UsagePattern {
    |        ^^^^^^^^^^^^

warning: struct `CommitmentMix` is never constructed
   --> src\finops\mod.rs:532:8
    |
532 | struct CommitmentMix {
    |        ^^^^^^^^^^^^^

warning: struct `SavingsEstimate` is never constructed
   --> src\finops\mod.rs:549:8
    |
549 | struct SavingsEstimate {
    |        ^^^^^^^^^^^^^^^

warning: function `get_finops_metrics` is never used
    --> src\finops\mod.rs:1028:14
     |
1028 | pub async fn get_finops_metrics(
     |              ^^^^^^^^^^^^^^^^^^

warning: struct `GovernanceCoordinator` is never constructed
  --> src\governance\mod.rs:31:12
   |
31 | pub struct GovernanceCoordinator {
   |            ^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `get_governance_health` are never used
   --> src\governance\mod.rs:71:18
    |
69  | impl GovernanceCoordinator {
    | -------------------------- associated items in this implementation
70  |     /// Create a new governance coordinator with all Azure integrations
71  |     pub async fn new(azure_client: Arc<AzureClient>) -> Result<Self, GovernanceError> {
    |                  ^^^
...
116 |     pub async fn get_governance_health(&self) -> Result<GovernanceHealthReport, GovernanceError> {
    |                  ^^^^^^^^^^^^^^^^^^^^^

warning: enum `GovernanceError` is never used
   --> src\governance\mod.rs:172:10
    |
172 | pub enum GovernanceError {
    |          ^^^^^^^^^^^^^^^

warning: enum `HealthStatus` is never used
   --> src\governance\mod.rs:218:10
    |
218 | pub enum HealthStatus {
    |          ^^^^^^^^^^^^

warning: struct `ComponentHealth` is never constructed
   --> src\governance\mod.rs:236:12
    |
236 | pub struct ComponentHealth {
    |            ^^^^^^^^^^^^^^^

warning: struct `GovernanceHealthReport` is never constructed
   --> src\governance\mod.rs:246:12
    |
246 | pub struct GovernanceHealthReport {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: function `calculate_overall_health` is never used
   --> src\governance\mod.rs:262:4
    |
262 | fn calculate_overall_health(statuses: &[HealthStatus]) -> HealthStatus {
    |    ^^^^^^^^^^^^^^^^^^^^^^^^

warning: type alias `GovernanceResult` is never used
   --> src\governance\mod.rs:273:10
    |
273 | pub type GovernanceResult<T> = Result<T, GovernanceError>;
    |          ^^^^^^^^^^^^^^^^

warning: struct `ResourceGraphClient` is never constructed
  --> src\governance\resource_graph.rs:22:12
   |
22 | pub struct ResourceGraphClient {
   |            ^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceGraphConfig` is never constructed
  --> src\governance\resource_graph.rs:38:12
   |
38 | pub struct ResourceGraphConfig {
   |            ^^^^^^^^^^^^^^^^^^^

warning: struct `CachedResourceData` is never constructed
  --> src\governance\resource_graph.rs:65:12
   |
65 | pub struct CachedResourceData {
   |            ^^^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\resource_graph.rs:77:12
   |
76 | impl CachedResourceData {
   | ----------------------- method in this implementation
77 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `QueryStatistics` is never constructed
  --> src\governance\resource_graph.rs:84:12
   |
84 | pub struct QueryStatistics {
   |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\resource_graph.rs:238:18
    |
236 | impl ResourceGraphClient {
    | ------------------------ associated items in this implementation
237 |     /// Create a new Resource Graph client
238 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
    |                  ^^^
...
250 |     pub async fn with_config(azure_client: Arc<AzureClient>, config: ResourceGraphConfig) -> GovernanceResult<Self> {
    |                  ^^^^^^^^^^^
...
260 |     pub async fn query_resources(&self, query: &str) -> GovernanceResult<ResourceQueryResult> {
    |                  ^^^^^^^^^^^^^^^
...
287 |     pub async fn get_compliance_state(&self, scope: &str) -> GovernanceResult<ComplianceState> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
336 |     pub async fn discover_resources_by_type(&self, resource_type: &str) -> GovernanceResult<Vec<AzureResource>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
361 |     pub async fn discover_resources(&self, filter: &ResourceFilter) -> GovernanceResult<Vec<AzureResource>> {
    |                  ^^^^^^^^^^^^^^^^^^
...
433 |     pub async fn get_resource_inventory(&self) -> GovernanceResult<ResourceInventory> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
473 |     async fn execute_query(&self, query: &str) -> GovernanceResult<ResourceQueryResult> {
    |              ^^^^^^^^^^^^^
...
515 |     async fn update_stats(&self, cache_hit: bool, query_time_ms: f64) {
    |              ^^^^^^^^^^^^
...
531 |     pub async fn get_statistics(&self) -> QueryStatistics {
    |                  ^^^^^^^^^^^^^^
...
536 |     pub fn clear_cache(&self) {
    |            ^^^^^^^^^^^
...
541 |     pub async fn health_check(&self) -> ComponentHealth {
    |                  ^^^^^^^^^^^^

warning: struct `PolicyEngine` is never constructed
  --> src\governance\policy_engine.rs:22:12
   |
22 | pub struct PolicyEngine {
   |            ^^^^^^^^^^^^

warning: struct `PolicyEngineConfig` is never constructed
  --> src\governance\policy_engine.rs:41:12
   |
41 | pub struct PolicyEngineConfig {
   |            ^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\policy_engine.rs:388:18
    |
386 | impl PolicyEngine {
    | ----------------- associated items in this implementation
387 |     /// Create a new Policy Engine
388 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
    |                  ^^^
...
401 |     pub async fn create_policy(&self, definition: PolicyDefinition) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^
...
449 |     pub async fn assign_policy(&self, assignment: PolicyAssignment) -> GovernanceResult<()> {
    |                  ^^^^^^^^^^^^^
...
485 |     pub async fn get_compliance_state(&self, scope: &str) -> GovernanceResult<ComplianceReport> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
512 |     pub async fn remediate_non_compliant(&self, resource_id: &str) -> GovernanceResult<RemediationTask> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
573 |     pub async fn create_exemption(&self, exemption: PolicyExemption) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^^^^
...
613 |     pub async fn evaluate_resource(&self, resource_id: &str) -> GovernanceResult<Vec<PolicyEvaluationResult>> {
    |                  ^^^^^^^^^^^^^^^^^
...
645 |     fn validate_policy_definition(&self, definition: &PolicyDefinition) -> GovernanceResult<()> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
672 |     async fn find_applicable_policies(&self, resource_id: &str) -> GovernanceResult<Vec<PolicyAssignment>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
693 |     fn parse_compliance_response(&self, scope: &str, data: serde_json::Value) -> GovernanceResult<ComplianceReport> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
734 |     fn parse_evaluation_results(&self, data: serde_json::Value) -> GovernanceResult<Vec<PolicyEvaluationResult>> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
749 |     fn parse_single_evaluation(&self, data: &serde_json::Value) -> GovernanceResult<PolicyEvaluationResult> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
777 |     pub async fn health_check(&self) -> ComponentHealth {
    |                  ^^^^^^^^^^^^

warning: struct `IdentityGovernanceClient` is never constructed
  --> src\governance\identity.rs:19:12
   |
19 | pub struct IdentityGovernanceClient {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `get_identity_governance_state`, `perform_access_review`, `manage_privileged_access`, and `health_check` are never used
  --> src\governance\identity.rs:46:18
   |
45 | impl IdentityGovernanceClient {
   | ----------------------------- associated items in this implementation
46 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
   |                  ^^^
...
50 |     pub async fn get_identity_governance_state(&self) -> GovernanceResult<IdentityState> {
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
60 |     pub async fn perform_access_review(&self, _scope: &str) -> GovernanceResult<AccessReviewResult> {
   |                  ^^^^^^^^^^^^^^^^^^^^^
...
69 |     pub async fn manage_privileged_access(&self, _request: PIMRequest) -> GovernanceResult<()> {
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
74 |     pub async fn health_check(&self) -> ComponentHealth {
   |                  ^^^^^^^^^^^^

warning: struct `GovernanceMonitor` is never constructed
  --> src\governance\monitoring.rs:19:12
   |
19 | pub struct GovernanceMonitor {
   |            ^^^^^^^^^^^^^^^^^

warning: associated items `new`, `create_governance_alerts`, `query_compliance_metrics`, `track_policy_violations`, and `health_check` are never used
  --> src\governance\monitoring.rs:52:18
   |
51 | impl GovernanceMonitor {
   | ---------------------- associated items in this implementation
52 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
   |                  ^^^
...
56 |     pub async fn create_governance_alerts(&self, _rules: Vec<AlertRule>) -> GovernanceResult<()> {
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
61 |     pub async fn query_compliance_metrics(&self, _kql: &str) -> GovernanceResult<MetricsResult> {
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
75 |     pub async fn track_policy_violations(&self) -> GovernanceResult<Vec<PolicyViolation>> {
   |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
80 |     pub async fn health_check(&self) -> ComponentHealth {
   |                  ^^^^^^^^^^^^

warning: struct `CostGovernanceEngine` is never constructed
  --> src\governance\cost_management.rs:20:12
   |
20 | pub struct CostGovernanceEngine {
   |            ^^^^^^^^^^^^^^^^^^^^

warning: struct `CachedCostData` is never constructed
  --> src\governance\cost_management.rs:30:12
   |
30 | pub struct CachedCostData {
   |            ^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\cost_management.rs:37:12
   |
36 | impl CachedCostData {
   | ------------------- method in this implementation
37 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `BudgetMonitor` is never constructed
   --> src\governance\cost_management.rs:321:12
    |
321 | pub struct BudgetMonitor {
    |            ^^^^^^^^^^^^^

warning: associated items `new` and `evaluate_budget_health` are never used
   --> src\governance\cost_management.rs:326:12
    |
325 | impl BudgetMonitor {
    | ------------------ associated items in this implementation
326 |     pub fn new() -> Self {
    |            ^^^
...
332 |     pub fn evaluate_budget_health(&self, budget: &BudgetStatus) -> BudgetHealthStatus {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ForecastEngine` is never constructed
   --> src\governance\cost_management.rs:343:12
    |
343 | pub struct ForecastEngine {
    |            ^^^^^^^^^^^^^^

warning: associated items `new` and `generate_forecast` are never used
   --> src\governance\cost_management.rs:348:12
    |
347 | impl ForecastEngine {
    | ------------------- associated items in this implementation
348 |     pub fn new() -> Self {
    |            ^^^
...
354 |     pub fn generate_forecast(&self, historical_costs: &[CostTrend], forecast_days: u32) -> SpendingForecast {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `OptimizationAnalyzer` is never constructed
   --> src\governance\cost_management.rs:399:12
    |
399 | pub struct OptimizationAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `analyze_optimization_potential` are never used
   --> src\governance\cost_management.rs:404:12
    |
403 | impl OptimizationAnalyzer {
    | ------------------------- associated items in this implementation
404 |     pub fn new() -> Self {
    |            ^^^
...
416 |     pub fn analyze_optimization_potential(&self, resource_costs: &[ServiceCost]) -> Vec<CostOptimization> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\cost_management.rs:471:18
    |
470 | impl CostGovernanceEngine {
    | ------------------------- associated items in this implementation
471 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
    |                  ^^^
...
482 |     pub async fn analyze_cost_trends(&self, scope: &str) -> GovernanceResult<CostTrendAnalysis> {
    |                  ^^^^^^^^^^^^^^^^^^^
...
537 |     pub async fn create_budget_alerts(&self, budget: BudgetDefinition) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
554 |     pub async fn forecast_spending(&self, scope: &str, forecast_days: u32) -> GovernanceResult<SpendingForecast> {
    |                  ^^^^^^^^^^^^^^^^^
...
565 |     pub async fn optimize_costs(&self, scope: &str) -> GovernanceResult<Vec<CostOptimization>> {
    |                  ^^^^^^^^^^^^^^
...
579 |     pub async fn get_finops_insights(&self, scope: &str) -> GovernanceResult<FinOpsInsights> {
    |                  ^^^^^^^^^^^^^^^^^^^
...
602 |     pub async fn monitor_budget_status(&self, budget_id: &str) -> GovernanceResult<BudgetStatus> {
    |                  ^^^^^^^^^^^^^^^^^^^^^
...
639 |     pub async fn health_check(&self) -> ComponentHealth {
    |                  ^^^^^^^^^^^^
...
655 |     async fn fetch_cost_data(&self, scope: &str, days: u32) -> GovernanceResult<CostData> {
    |              ^^^^^^^^^^^^^^^
...
686 |     fn build_cost_trends(&self, cost_data: &CostData) -> Vec<CostTrend> {
    |        ^^^^^^^^^^^^^^^^^
...
720 |     fn detect_cost_anomalies(&self, trends: &[CostTrend]) -> Vec<CostAnomaly> {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
756 |     fn calculate_trend_direction(&self, trends: &[CostTrend]) -> TrendDirection {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
772 |     fn calculate_variance(&self, trends: &[CostTrend]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^
...
782 |     fn build_budget_request(&self, budget: &BudgetDefinition) -> GovernanceResult<String> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
787 |     async fn setup_budget_monitoring(&self, budget_id: &str, budget: &BudgetDefinition) -> GovernanceResult<()> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
792 |     async fn get_historical_costs(&self, scope: &str, days: u32) -> GovernanceResult<Vec<CostTrend>> {
    |              ^^^^^^^^^^^^^^^^^^^^
...
797 |     async fn get_cost_breakdown_by_service(&self, scope: &str) -> GovernanceResult<Vec<ServiceCost>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
819 |     async fn enhance_optimizations_with_utilization(&self, optimizations: Vec<CostOptimization>) -> GovernanceResult<Vec<CostO...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
824 |     async fn calculate_cost_allocation_accuracy(&self, _scope: &str) -> GovernanceResult<f64> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
828 |     async fn calculate_showback_coverage(&self, _scope: &str) -> GovernanceResult<f64> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
832 |     async fn calculate_budget_variance(&self, _scope: &str) -> GovernanceResult<f64> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
836 |     async fn calculate_optimization_adoption_rate(&self, _scope: &str) -> GovernanceResult<f64> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
840 |     async fn get_business_unit_costs(&self, _scope: &str) -> GovernanceResult<Vec<BusinessUnitCost>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
857 |     async fn identify_top_cost_drivers(&self, _scope: &str) -> GovernanceResult<Vec<CostDriver>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
874 |     async fn get_budget_definition(&self, _budget_id: &str) -> GovernanceResult<BudgetDefinition> {
    |              ^^^^^^^^^^^^^^^^^^^^^
...
892 |     async fn get_current_spend(&self, _scope: &str) -> GovernanceResult<f64> {
    |              ^^^^^^^^^^^^^^^^^
...
896 |     async fn check_triggered_alerts(&self, _budget: &BudgetDefinition, _current_spend: f64, _forecasted_spend: f64) -> Governa...
    |              ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `SecurityPostureEngine` is never constructed
  --> src\governance\security_posture.rs:20:12
   |
20 | pub struct SecurityPostureEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `CachedSecurityData` is never constructed
  --> src\governance\security_posture.rs:30:12
   |
30 | pub struct CachedSecurityData {
   |            ^^^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\security_posture.rs:37:12
   |
36 | impl CachedSecurityData {
   | ----------------------- method in this implementation
37 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `ThreatDetector` is never constructed
   --> src\governance\security_posture.rs:312:12
    |
312 | pub struct ThreatDetector {
    |            ^^^^^^^^^^^^^^

warning: struct `ThreatRule` is never constructed
   --> src\governance\security_posture.rs:318:12
    |
318 | pub struct ThreatRule {
    |            ^^^^^^^^^^

warning: struct `ThreatModel` is never constructed
   --> src\governance\security_posture.rs:328:12
    |
328 | pub struct ThreatModel {
    |            ^^^^^^^^^^^

warning: struct `ComplianceMonitor` is never constructed
   --> src\governance\security_posture.rs:336:12
    |
336 | pub struct ComplianceMonitor {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `VulnerabilityScanner` is never constructed
   --> src\governance\security_posture.rs:342:12
    |
342 | pub struct VulnerabilityScanner {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: struct `VulnerabilityData` is never constructed
   --> src\governance\security_posture.rs:349:12
    |
349 | pub struct VulnerabilityData {
    |            ^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\security_posture.rs:358:18
    |
357 | impl SecurityPostureEngine {
    | -------------------------- associated items in this implementation
358 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
    |                  ^^^
...
369 |     pub async fn assess_security_posture(&self, scope: &str) -> GovernanceResult<SecurityData> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
393 |     pub async fn monitor_threats(&self, scope: &str) -> GovernanceResult<Vec<SecurityThreat>> {
    |                  ^^^^^^^^^^^^^^^
...
407 |     pub async fn scan_vulnerabilities(&self, scope: &str) -> GovernanceResult<Vec<Vulnerability>> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
412 |     pub async fn monitor_compliance(&self, framework: Option<&str>) -> GovernanceResult<ComplianceStatus> {
    |                  ^^^^^^^^^^^^^^^^^^
...
417 |     pub async fn generate_security_recommendations(&self, scope: &str) -> GovernanceResult<Vec<SecurityRecommendation>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
509 |     pub async fn auto_remediate_threats(&self, threat_id: &str) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
516 |     pub async fn get_security_metrics(&self, scope: &str) -> GovernanceResult<SecurityMetrics> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
537 |     pub async fn health_check(&self) -> ComponentHealth {
    |                  ^^^^^^^^^^^^
...
554 |     async fn fetch_security_data(&self, scope: &str) -> GovernanceResult<SecurityData> {
    |              ^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `detect_threats` are never used
   --> src\governance\security_posture.rs:741:12
    |
740 | impl ThreatDetector {
    | ------------------- associated items in this implementation
741 |     pub fn new() -> Self {
    |            ^^^
...
767 |     pub async fn detect_threats(&self, scope: &str) -> GovernanceResult<Vec<SecurityThreat>> {
    |                  ^^^^^^^^^^^^^^

warning: associated items `new` and `assess_compliance` are never used
   --> src\governance\security_posture.rs:802:12
    |
801 | impl ComplianceMonitor {
    | ---------------------- associated items in this implementation
802 |     pub fn new() -> Self {
    |            ^^^
...
814 |     pub async fn assess_compliance(&self, framework: Option<&str>) -> GovernanceResult<ComplianceStatus> {
    |                  ^^^^^^^^^^^^^^^^^

warning: associated items `new` and `scan_scope` are never used
   --> src\governance\security_posture.rs:836:12
    |
835 | impl VulnerabilityScanner {
    | ------------------------- associated items in this implementation
836 |     pub fn new() -> Self {
    |            ^^^
...
844 |     pub async fn scan_scope(&self, scope: &str) -> GovernanceResult<Vec<Vulnerability>> {
    |                  ^^^^^^^^^^

warning: struct `AccessGovernanceEngine` is never constructed
  --> src\governance\access_control.rs:20:12
   |
20 | pub struct AccessGovernanceEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `CachedAccessData` is never constructed
  --> src\governance\access_control.rs:30:12
   |
30 | pub struct CachedAccessData {
   |            ^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\access_control.rs:37:12
   |
36 | impl CachedAccessData {
   | --------------------- method in this implementation
37 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `RoleAnalyzer` is never constructed
   --> src\governance\access_control.rs:334:12
    |
334 | pub struct RoleAnalyzer {
    |            ^^^^^^^^^^^^

warning: struct `PermissionMonitor` is never constructed
   --> src\governance\access_control.rs:340:12
    |
340 | pub struct PermissionMonitor {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `PermissionUsage` is never constructed
   --> src\governance\access_control.rs:346:12
    |
346 | pub struct PermissionUsage {
    |            ^^^^^^^^^^^^^^^

warning: struct `IdentityTracker` is never constructed
   --> src\governance\access_control.rs:354:12
    |
354 | pub struct IdentityTracker {
    |            ^^^^^^^^^^^^^^^

warning: struct `IdentityInfo` is never constructed
   --> src\governance\access_control.rs:360:12
    |
360 | pub struct IdentityInfo {
    |            ^^^^^^^^^^^^

warning: struct `ActivityPattern` is never constructed
   --> src\governance\access_control.rs:370:12
    |
370 | pub struct ActivityPattern {
    |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\access_control.rs:378:18
    |
377 | impl AccessGovernanceEngine {
    | --------------------------- associated items in this implementation
378 |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
    |                  ^^^
...
389 |     pub async fn analyze_access_patterns(&self, scope: &str) -> GovernanceResult<AccessData> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
413 |     pub async fn detect_privilege_escalation(&self, scope: &str) -> GovernanceResult<Vec<AccessAnomaly>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
479 |     pub async fn enforce_least_privilege(&self, scope: &str) -> GovernanceResult<Vec<ComplianceRecommendation>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
558 |     pub async fn monitor_access_reviews(&self, scope: &str) -> GovernanceResult<Vec<AccessReview>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
590 |     pub async fn get_access_metrics(&self, scope: &str) -> GovernanceResult<AccessMetrics> {
    |                  ^^^^^^^^^^^^^^^^^^
...
609 |     pub async fn create_access_review(&self, review_definition: AccessReviewDefinition) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
618 |     pub async fn remediate_access_violations(&self, violation_ids: Vec<String>) -> GovernanceResult<Vec<RemediationResult>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
638 |     pub async fn health_check(&self) -> ComponentHealth {
    |                  ^^^^^^^^^^^^
...
655 |     async fn fetch_access_data(&self, scope: &str) -> GovernanceResult<AccessData> {
    |              ^^^^^^^^^^^^^^^^^
...
791 |     fn is_privilege_escalation(&self, assignment: &RoleAssignment) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
800 |     fn has_excessive_permissions(&self, assignment: &RoleAssignment) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
805 |     fn analyze_permission_patterns(&self, access_data: &AccessData) -> HashMap<String, u32> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
818 |     fn has_recent_access_review(&self, account: &PrivilegedAccount) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `get_privilege_level` are never used
   --> src\governance\access_control.rs:865:12
    |
864 | impl RoleAnalyzer {
    | ----------------- associated items in this implementation
865 |     pub fn new() -> Self {
    |            ^^^
...
881 |     pub fn get_privilege_level(&self, role_name: &str) -> PrivilegeLevel {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `track_permission_usage` are never used
   --> src\governance\access_control.rs:887:12
    |
886 | impl PermissionMonitor {
    | ---------------------- associated items in this implementation
887 |     pub fn new() -> Self {
    |            ^^^
...
894 |     pub fn track_permission_usage(&mut self, permission: &str, principal_id: &str) {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `track_identity_activity`, and `get_identity_risk_score` are never used
   --> src\governance\access_control.rs:911:12
    |
910 | impl IdentityTracker {
    | -------------------- associated items in this implementation
911 |     pub fn new() -> Self {
    |            ^^^
...
918 |     pub fn track_identity_activity(&mut self, principal_id: &str, activity: ActivityPattern) {
    |            ^^^^^^^^^^^^^^^^^^^^^^^
...
922 |     pub fn get_identity_risk_score(&self, principal_id: &str) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `NetworkGovernanceEngine` is never constructed
  --> src\governance\network.rs:21:12
   |
21 | pub struct NetworkGovernanceEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `CachedNetworkData` is never constructed
  --> src\governance\network.rs:31:12
   |
31 | pub struct CachedNetworkData {
   |            ^^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\network.rs:38:12
   |
37 | impl CachedNetworkData {
   | ---------------------- method in this implementation
38 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `NetworkSecurityAnalyzer` is never constructed
   --> src\governance\network.rs:516:12
    |
516 | pub struct NetworkSecurityAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `VulnerabilityPattern` is never constructed
   --> src\governance\network.rs:522:12
    |
522 | pub struct VulnerabilityPattern {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: struct `FirewallMonitor` is never constructed
   --> src\governance\network.rs:530:12
    |
530 | pub struct FirewallMonitor {
    |            ^^^^^^^^^^^^^^^

warning: struct `FirewallPolicy` is never constructed
   --> src\governance\network.rs:536:12
    |
536 | pub struct FirewallPolicy {
    |            ^^^^^^^^^^^^^^

warning: struct `TrafficPattern` is never constructed
   --> src\governance\network.rs:544:12
    |
544 | pub struct TrafficPattern {
    |            ^^^^^^^^^^^^^^

warning: struct `VNetManager` is never constructed
   --> src\governance\network.rs:554:12
    |
554 | pub struct VNetManager {
    |            ^^^^^^^^^^^

warning: multiple associated items are never used
    --> src\governance\network.rs:560:18
     |
559  | impl NetworkGovernanceEngine {
     | ---------------------------- associated items in this implementation
560  |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
     |                  ^^^
...
571  |     pub async fn analyze_network_security(&self, scope: &str) -> GovernanceResult<NetworkData> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
595  |     pub async fn detect_network_vulnerabilities(&self, scope: &str) -> GovernanceResult<Vec<NetworkVulnerability>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
623  |     pub async fn generate_network_recommendations(&self, scope: &str) -> GovernanceResult<Vec<NetworkRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
747  |     pub async fn monitor_firewall_health(&self, scope: &str) -> GovernanceResult<Vec<FirewallHealthMetric>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
780  |     pub async fn get_network_metrics(&self, scope: &str) -> GovernanceResult<NetworkMetrics> {
     |                  ^^^^^^^^^^^^^^^^^^^
...
811  |     pub async fn health_check(&self) -> ComponentHealth {
     |                  ^^^^^^^^^^^^
...
828  |     async fn fetch_network_data(&self, scope: &str) -> GovernanceResult<NetworkData> {
     |              ^^^^^^^^^^^^^^^^^^
...
1053 |     fn analyze_nsg_vulnerabilities(&self, nsg: &NetworkSecurityGroup) -> Vec<NetworkVulnerability> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1082 |     fn analyze_firewall_vulnerabilities(&self, firewall: &AzureFirewall) -> Vec<NetworkVulnerability> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1108 |     fn analyze_vnet_vulnerabilities(&self, vnet: &VirtualNetwork) -> Vec<NetworkVulnerability> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1134 |     fn analyze_public_ip_vulnerabilities(&self, public_ip: &PublicIpAddress) -> Vec<NetworkVulnerability> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
    --> src\governance\network.rs:1162:12
     |
1161 | impl NetworkSecurityAnalyzer {
     | ---------------------------- associated function in this implementation
1162 |     pub fn new() -> Self {
     |            ^^^

warning: associated function `new` is never used
    --> src\governance\network.rs:1192:12
     |
1191 | impl FirewallMonitor {
     | -------------------- associated function in this implementation
1192 |     pub fn new() -> Self {
     |            ^^^

warning: associated function `new` is never used
    --> src\governance\network.rs:1201:12
     |
1200 | impl VNetManager {
     | ---------------- associated function in this implementation
1201 |     pub fn new() -> Self {
     |            ^^^

warning: struct `OptimizationEngine` is never constructed
  --> src\governance\optimization.rs:21:12
   |
21 | pub struct OptimizationEngine {
   |            ^^^^^^^^^^^^^^^^^^

warning: struct `CachedRecommendationData` is never constructed
  --> src\governance\optimization.rs:33:12
   |
33 | pub struct CachedRecommendationData {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\optimization.rs:40:12
   |
39 | impl CachedRecommendationData {
   | ----------------------------- method in this implementation
40 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `CostOptimizationAnalyzer` is never constructed
   --> src\governance\optimization.rs:344:12
    |
344 | pub struct CostOptimizationAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PricingInfo` is never constructed
   --> src\governance\optimization.rs:351:12
    |
351 | pub struct PricingInfo {
    |            ^^^^^^^^^^^

warning: struct `UsagePattern` is never constructed
   --> src\governance\optimization.rs:361:12
    |
361 | pub struct UsagePattern {
    |            ^^^^^^^^^^^^

warning: struct `ReservationOpportunity` is never constructed
   --> src\governance\optimization.rs:371:12
    |
371 | pub struct ReservationOpportunity {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PerformanceAnalyzer` is never constructed
   --> src\governance\optimization.rs:387:12
    |
387 | pub struct PerformanceAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^

warning: struct `PerformanceBaseline` is never constructed
   --> src\governance\optimization.rs:393:12
    |
393 | pub struct PerformanceBaseline {
    |            ^^^^^^^^^^^^^^^^^^^

warning: struct `BottleneckDetector` is never constructed
   --> src\governance\optimization.rs:402:12
    |
402 | pub struct BottleneckDetector {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `SecurityOptimizer` is never constructed
   --> src\governance\optimization.rs:410:12
    |
410 | pub struct SecurityOptimizer {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `SecurityPolicy` is never constructed
   --> src\governance\optimization.rs:416:12
    |
416 | pub struct SecurityPolicy {
    |            ^^^^^^^^^^^^^^

warning: struct `VulnerabilityInfo` is never constructed
   --> src\governance\optimization.rs:425:12
    |
425 | pub struct VulnerabilityInfo {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `ReliabilityMonitor` is never constructed
   --> src\governance\optimization.rs:434:12
    |
434 | pub struct ReliabilityMonitor {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `SlaTarget` is never constructed
   --> src\governance\optimization.rs:440:12
    |
440 | pub struct SlaTarget {
    |            ^^^^^^^^^

warning: struct `DrPlan` is never constructed
   --> src\governance\optimization.rs:449:12
    |
449 | pub struct DrPlan {
    |            ^^^^^^

warning: struct `OperationalExcellenceAnalyzer` is never constructed
   --> src\governance\optimization.rs:458:12
    |
458 | pub struct OperationalExcellenceAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `AutomationOpportunity` is never constructed
   --> src\governance\optimization.rs:464:12
    |
464 | pub struct AutomationOpportunity {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `MonitoringGap` is never constructed
   --> src\governance\optimization.rs:473:12
    |
473 | pub struct MonitoringGap {
    |            ^^^^^^^^^^^^^

warning: multiple associated items are never used
    --> src\governance\optimization.rs:481:18
     |
480  | impl OptimizationEngine {
     | ----------------------- associated items in this implementation
481  |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
     |                  ^^^
...
494  |     pub async fn get_optimization_recommendations(&self, scope: &str) -> GovernanceResult<OptimizationData> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
518  |     pub async fn analyze_cost_optimization(&self, scope: &str) -> GovernanceResult<Vec<CostRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^
...
543  |     pub async fn analyze_performance_optimization(&self, scope: &str) -> GovernanceResult<Vec<PerformanceRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
569  |     pub async fn analyze_security_optimization(&self, scope: &str) -> GovernanceResult<Vec<SecurityRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
607  |     pub async fn analyze_reliability_optimization(&self, scope: &str) -> GovernanceResult<Vec<ReliabilityRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
633  |     pub async fn analyze_operational_excellence(&self, scope: &str) -> GovernanceResult<Vec<OperationalRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
660  |     pub async fn generate_optimization_roadmap(&self, scope: &str) -> GovernanceResult<OptimizationRoadmap> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
743  |     pub async fn get_optimization_metrics(&self, scope: &str) -> GovernanceResult<OptimizationMetrics> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
766  |     pub async fn health_check(&self) -> ComponentHealth {
     |                  ^^^^^^^^^^^^
...
784  |     async fn fetch_advisor_recommendations(&self, scope: &str) -> GovernanceResult<OptimizationData> {
     |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1030 |     fn count_high_impact_recommendations(&self, optimization_data: &OptimizationData) -> u32 {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1054 |     fn calculate_priority(&self, impact: &ImpactLevel, effort: &ImplementationEffort) -> RoadmapPriority {
     |        ^^^^^^^^^^^^^^^^^^
...
1066 |     fn estimate_timeline(&self, effort: &ImplementationEffort) -> String {
     |        ^^^^^^^^^^^^^^^^^
...
1075 |     fn create_implementation_phases(&self, _optimization_data: &OptimizationData) -> Vec<ImplementationPhase> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1105 |     pub async fn get_advisor_recommendations(&self) -> GovernanceResult<Vec<OptimizationRecommendation>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1141 |     pub async fn get_optimization_summary(&self) -> GovernanceResult<OptimizationSummary> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
1162 |     pub async fn apply_optimization(&self, _recommendation_id: &str) -> GovernanceResult<()> {
     |                  ^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `assess_risk_level` are never used
    --> src\governance\optimization.rs:1170:12
     |
1169 | impl CostOptimizationAnalyzer {
     | ----------------------------- associated items in this implementation
1170 |     pub fn new() -> Self {
     |            ^^^
...
1178 |     pub fn assess_risk_level(&self, recommendation: &CostRecommendation) -> RiskLevel {
     |            ^^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
    --> src\governance\optimization.rs:1189:12
     |
1188 | impl PerformanceAnalyzer {
     | ------------------------ associated function in this implementation
1189 |     pub fn new() -> Self {
     |            ^^^

warning: associated function `new` is never used
    --> src\governance\optimization.rs:1203:12
     |
1202 | impl SecurityOptimizer {
     | ---------------------- associated function in this implementation
1203 |     pub fn new() -> Self {
     |            ^^^

warning: associated function `new` is never used
    --> src\governance\optimization.rs:1212:12
     |
1211 | impl ReliabilityMonitor {
     | ----------------------- associated function in this implementation
1212 |     pub fn new() -> Self {
     |            ^^^

warning: associated function `new` is never used
    --> src\governance\optimization.rs:1221:12
     |
1220 | impl OperationalExcellenceAnalyzer {
     | ---------------------------------- associated function in this implementation
1221 |     pub fn new() -> Self {
     |            ^^^

warning: struct `GovernanceBlueprints` is never constructed
  --> src\governance\blueprints.rs:21:12
   |
21 | pub struct GovernanceBlueprints {
   |            ^^^^^^^^^^^^^^^^^^^^

warning: struct `CachedBlueprintData` is never constructed
  --> src\governance\blueprints.rs:32:12
   |
32 | pub struct CachedBlueprintData {
   |            ^^^^^^^^^^^^^^^^^^^

warning: method `is_expired` is never used
  --> src\governance\blueprints.rs:39:12
   |
38 | impl CachedBlueprintData {
   | ------------------------ method in this implementation
39 |     pub fn is_expired(&self) -> bool {
   |            ^^^^^^^^^^

warning: struct `BlueprintTemplateEngine` is never constructed
   --> src\governance\blueprints.rs:716:12
    |
716 | pub struct BlueprintTemplateEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `BlueprintTemplate` is never constructed
   --> src\governance\blueprints.rs:722:12
    |
722 | pub struct BlueprintTemplate {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `TemplateArtifact` is never constructed
   --> src\governance\blueprints.rs:732:12
    |
732 | pub struct TemplateArtifact {
    |            ^^^^^^^^^^^^^^^^

warning: struct `ParameterValidator` is never constructed
   --> src\governance\blueprints.rs:739:12
    |
739 | pub struct ParameterValidator {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `ValidationRule` is never constructed
   --> src\governance\blueprints.rs:745:12
    |
745 | pub struct ValidationRule {
    |            ^^^^^^^^^^^^^^

warning: struct `BlueprintComplianceMonitor` is never constructed
   --> src\governance\blueprints.rs:751:12
    |
751 | pub struct BlueprintComplianceMonitor {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `CompliancePolicy` is never constructed
   --> src\governance\blueprints.rs:757:12
    |
757 | pub struct CompliancePolicy {
    |            ^^^^^^^^^^^^^^^^

warning: struct `ComplianceControl` is never constructed
   --> src\governance\blueprints.rs:765:12
    |
765 | pub struct ComplianceControl {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `ComplianceEvaluationEngine` is never constructed
   --> src\governance\blueprints.rs:772:12
    |
772 | pub struct ComplianceEvaluationEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `EvaluationResult` is never constructed
   --> src\governance\blueprints.rs:778:12
    |
778 | pub struct EvaluationResult {
    |            ^^^^^^^^^^^^^^^^

warning: struct `TrendAnalyzer` is never constructed
   --> src\governance\blueprints.rs:785:12
    |
785 | pub struct TrendAnalyzer {
    |            ^^^^^^^^^^^^^

warning: struct `PredictionModel` is never constructed
   --> src\governance\blueprints.rs:791:12
    |
791 | pub struct PredictionModel {
    |            ^^^^^^^^^^^^^^^

warning: struct `DeploymentTracker` is never constructed
   --> src\governance\blueprints.rs:797:12
    |
797 | pub struct DeploymentTracker {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `PerformanceMetrics` is never constructed
   --> src\governance\blueprints.rs:803:12
    |
803 | pub struct PerformanceMetrics {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `GovernanceValidator` is never constructed
   --> src\governance\blueprints.rs:810:12
    |
810 | pub struct GovernanceValidator {
    |            ^^^^^^^^^^^^^^^^^^^

warning: struct `GovernanceRule` is never constructed
   --> src\governance\blueprints.rs:816:12
    |
816 | pub struct GovernanceRule {
    |            ^^^^^^^^^^^^^^

warning: enum `GovernanceRuleType` is never used
   --> src\governance\blueprints.rs:825:10
    |
825 | pub enum GovernanceRuleType {
    |          ^^^^^^^^^^^^^^^^^^

warning: struct `PolicyValidationEngine` is never constructed
   --> src\governance\blueprints.rs:833:12
    |
833 | pub struct PolicyValidationEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PolicyDefinition` is never constructed
   --> src\governance\blueprints.rs:839:12
    |
839 | pub struct PolicyDefinition {
    |            ^^^^^^^^^^^^^^^^

warning: struct `ValidationCacheEntry` is never constructed
   --> src\governance\blueprints.rs:847:12
    |
847 | pub struct ValidationCacheEntry {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
    --> src\governance\blueprints.rs:855:18
     |
854  | impl GovernanceBlueprints {
     | ------------------------- associated items in this implementation
855  |     pub async fn new(azure_client: Arc<AzureClient>) -> GovernanceResult<Self> {
     |                  ^^^
...
867  |     pub async fn get_blueprint_management_data(&self, scope: &str) -> GovernanceResult<BlueprintManagementData> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
891  |     pub async fn list_blueprint_definitions(&self) -> GovernanceResult<Vec<BlueprintDefinition>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
897  |     pub async fn create_blueprint_definition(&self, definition: CreateBlueprintRequest) -> GovernanceResult<BlueprintDefiniti...
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
927  |     pub async fn publish_blueprint(&self, blueprint_id: &str, version: &str) -> GovernanceResult<BlueprintVersion> {
     |                  ^^^^^^^^^^^^^^^^^
...
941  |     pub async fn create_blueprint_assignment(&self, assignment_request: CreateAssignmentRequest) -> GovernanceResult<Blueprin...
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
977  |     pub async fn assess_blueprint_compliance(&self, scope: &str) -> GovernanceResult<Vec<ComplianceAssessment>> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
990  |     pub async fn monitor_deployments(&self, scope: &str) -> GovernanceResult<Vec<DeploymentRecord>> {
     |                  ^^^^^^^^^^^^^^^^^^^
...
996  |     pub async fn get_governance_dashboard(&self, scope: &str) -> GovernanceResult<GovernanceDashboard> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
1019 |     pub async fn generate_blueprint_template(&self, requirements: BlueprintRequirements) -> GovernanceResult<BlueprintDefinit...
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1024 |     pub async fn validate_blueprint_deployment(&self, assignment_id: &str) -> GovernanceResult<ValidationResults> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1030 |     pub async fn rollback_deployment(&self, assignment_id: &str, rollback_strategy: RollbackStrategy) -> GovernanceResult<Rol...
     |                  ^^^^^^^^^^^^^^^^^^^
...
1045 |     pub async fn get_performance_analytics(&self, scope: &str) -> GovernanceResult<PerformanceAnalytics> {
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^
...
1061 |     pub async fn health_check(&self) -> ComponentHealth {
     |                  ^^^^^^^^^^^^
...
1079 |     async fn fetch_blueprint_data(&self, scope: &str) -> GovernanceResult<BlueprintManagementData> {
     |              ^^^^^^^^^^^^^^^^^^^^
...
1316 |     async fn evaluate_assignment_compliance(&self, assignment: &BlueprintAssignment) -> GovernanceResult<ComplianceAssessment> {
     |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1354 |     fn calculate_compliance_trend(&self, assessments: &[ComplianceAssessment]) -> Vec<ComplianceTrendPoint> {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1362 |     fn calculate_efficiency_improvements(&self, _blueprint_data: &BlueprintManagementData) -> EfficiencyMetrics {
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1372 |     fn generate_performance_recommendations(&self, _blueprint_data: &BlueprintManagementData) -> Vec<PerformanceRecommendatio...
     |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `generate_template` are never used
    --> src\governance\blueprints.rs:1396:12
     |
1395 | impl BlueprintTemplateEngine {
     | ---------------------------- associated items in this implementation
1396 |     pub fn new() -> Self {
     |            ^^^
...
1403 |     pub async fn generate_template(&self, requirements: BlueprintRequirements) -> GovernanceResult<BlueprintDefinition> {
     |                  ^^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
    --> src\governance\blueprints.rs:1439:12
     |
1438 | impl BlueprintComplianceMonitor {
     | ------------------------------- associated function in this implementation
1439 |     pub fn new() -> Self {
     |            ^^^

warning: associated items `new` and `calculate_trends` are never used
    --> src\governance\blueprints.rs:1454:12
     |
1453 | impl DeploymentTracker {
     | ---------------------- associated items in this implementation
1454 |     pub fn new() -> Self {
     |            ^^^
...
1461 |     pub fn calculate_trends(&self) -> TrendAnalysis {
     |            ^^^^^^^^^^^^^^^^

warning: associated items `new`, `validate_blueprint_definition`, `validate_assignment_parameters`, and `validate_deployment` are never used
    --> src\governance\blueprints.rs:1473:12
     |
1472 | impl GovernanceValidator {
     | ------------------------ associated items in this implementation
1473 |     pub fn new() -> Self {
     |            ^^^
...
1483 |     pub fn validate_blueprint_definition(&self, _definition: &CreateBlueprintRequest) -> GovernanceResult<()> {
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1488 |     pub fn validate_assignment_parameters(&self, _request: &CreateAssignmentRequest) -> GovernanceResult<()> {
     |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1493 |     pub async fn validate_deployment(&self, _assignment_id: &str) -> GovernanceResult<ValidationResults> {
     |                  ^^^^^^^^^^^^^^^^^^^

warning: struct `UnifiedGovernanceAPI` is never constructed
  --> src\governance\unified_api.rs:21:12
   |
21 | pub struct UnifiedGovernanceAPI {
   |            ^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\unified_api.rs:99:18
    |
98  | impl UnifiedGovernanceAPI {
    | ------------------------- associated items in this implementation
99  |     pub async fn new(
    |                  ^^^
...
116 |     pub async fn get_unified_dashboard(&self) -> GovernanceResult<GovernanceDashboard> {
    |                  ^^^^^^^^^^^^^^^^^^^^^
...
172 |     pub async fn analyze_cross_domain_correlations(&self, resource_id: &str) -> GovernanceResult<Vec<CrossDomainCorrelation>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
209 |     pub async fn generate_predictive_insights(&self, scope: &str) -> GovernanceResult<Vec<PredictiveInsight>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
255 |     pub async fn get_ai_recommendations(&self, domain: &str) -> GovernanceResult<Vec<String>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
279 |     pub async fn search_governance_data(&self, query: &str) -> GovernanceResult<Vec<HashMap<String, serde_json::Value>>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
307 |     pub async fn health_check(&self) -> GovernanceResult<HashMap<String, serde_json::Value>> {
    |                  ^^^^^^^^^^^^

warning: struct `AIGovernanceEngine` is never constructed
  --> src\governance\ai\mod.rs:27:12
   |
27 | pub struct AIGovernanceEngine {
   |            ^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\ai\mod.rs:113:18
    |
112 | impl AIGovernanceEngine {
    | ----------------------- associated items in this implementation
113 |     pub async fn new(
    |                  ^^^
...
148 |     pub async fn analyze_governance_correlations(&self, scope: &str) -> GovernanceResult<Vec<AIInsight>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
177 |     pub async fn process_natural_language_query(&self,
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
185 |     pub async fn generate_compliance_predictions(&self, time_horizon_days: u32) -> GovernanceResult<Vec<AIInsight>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
220 |     pub async fn generate_unified_insights(&self, scope: &str) -> GovernanceResult<Vec<AIInsight>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^
...
248 |     async fn generate_cost_optimization_insights(&self, scope: &str) -> GovernanceResult<Vec<AIInsight>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
275 |     async fn generate_security_insights(&self, scope: &str) -> GovernanceResult<Vec<AIInsight>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
302 |     pub async fn execute_automated_recommendations(&self, insight_id: &str) -> GovernanceResult<Vec<String>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
313 |     pub async fn health_check(&self) -> GovernanceResult<HashMap<String, String>> {
    |                  ^^^^^^^^^^^^

warning: struct `ConversationalGovernance` is never constructed
  --> src\governance\ai\conversation.rs:18:12
   |
18 | pub struct ConversationalGovernance {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `IntentClassifier` is never constructed
  --> src\governance\ai\conversation.rs:65:12
   |
65 | pub struct IntentClassifier {
   |            ^^^^^^^^^^^^^^^^

warning: struct `ContextManager` is never constructed
  --> src\governance\ai\conversation.rs:70:12
   |
70 | pub struct ContextManager {
   |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\ai\conversation.rs:76:18
    |
75  | impl ConversationalGovernance {
    | ----------------------------- associated items in this implementation
76  |     pub async fn new(
    |                  ^^^
...
95  |     pub async fn process_query(&self, query: &str, context: &ConversationContext) -> GovernanceResult<String> {
    |                  ^^^^^^^^^^^^^
...
116 |     async fn handle_resource_query(&self, query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Qu...
    |              ^^^^^^^^^^^^^^^^^^^^^
...
158 |     async fn handle_compliance_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult...
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
190 |     async fn handle_cost_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Query...
    |              ^^^^^^^^^^^^^^^^^
...
210 |     async fn handle_security_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Q...
    |              ^^^^^^^^^^^^^^^^^^^^^
...
230 |     async fn handle_policy_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Que...
    |              ^^^^^^^^^^^^^^^^^^^
...
252 |     async fn handle_incident_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Q...
    |              ^^^^^^^^^^^^^^^^^^^^^
...
271 |     async fn handle_report_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Que...
    |              ^^^^^^^^^^^^^^^^^^^
...
290 |     async fn handle_access_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Que...
    |              ^^^^^^^^^^^^^^^^^^^
...
309 |     async fn handle_optimization_query(&self, _query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResu...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
328 |     async fn handle_unknown_query(&self, query: &str, _context: &ConversationContext) -> GovernanceResult<QueryResponse> {
    |              ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `classify` are never used
   --> src\governance\ai\conversation.rs:357:8
    |
356 | impl IntentClassifier {
    | --------------------- associated items in this implementation
357 |     fn new() -> Self {
    |        ^^^
...
383 |     fn classify(&self, query: &str) -> GovernanceResult<Intent> {
    |        ^^^^^^^^

warning: associated items `new`, `get_context`, and `update_context` are never used
   --> src\governance\ai\conversation.rs:418:8
    |
417 | impl ContextManager {
    | ------------------- associated items in this implementation
418 |     fn new() -> Self {
    |        ^^^
...
424 |     pub fn get_context(&self, session_id: &str) -> Option<&ConversationContext> {
    |            ^^^^^^^^^^^
...
428 |     pub fn update_context(&mut self, session_id: String, context: ConversationContext) {
    |            ^^^^^^^^^^^^^^

warning: struct `CrossDomainCorrelationEngine` is never constructed
  --> src\governance\ai\correlation.rs:18:12
   |
18 | pub struct CrossDomainCorrelationEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PatternAnalyzer` is never constructed
  --> src\governance\ai\correlation.rs:93:12
   |
93 | pub struct PatternAnalyzer {
   |            ^^^^^^^^^^^^^^^

warning: struct `CorrelationRule` is never constructed
   --> src\governance\ai\correlation.rs:100:12
    |
100 | pub struct CorrelationRule {
    |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\ai\correlation.rs:110:18
    |
109 | impl CrossDomainCorrelationEngine {
    | --------------------------------- associated items in this implementation
110 |     pub async fn new(
    |                  ^^^
...
125 |     pub async fn analyze_cross_domain_patterns(&self, scope: &str) -> GovernanceResult<Vec<CorrelationPattern>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
164 |     async fn collect_cross_domain_events(&self, scope: &str) -> GovernanceResult<Vec<CrossDomainEvent>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
222 |     async fn analyze_security_cost_correlation(&self, events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPattern>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
267 |     async fn analyze_compliance_policy_correlation(&self, events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPatt...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
298 |     async fn analyze_identity_access_correlation(&self, _events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPatte...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
329 |     async fn analyze_cost_performance_correlation(&self, _events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPatt...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
359 |     async fn analyze_network_security_correlation(&self, _events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPatt...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
389 |     async fn analyze_resource_lifecycle_patterns(&self, events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPatter...
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
427 |     pub async fn monitor_correlations(&self, correlation_id: &str) -> GovernanceResult<CorrelationPattern> {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
433 |     pub async fn validate_correlation(&self, pattern: &CorrelationPattern) -> GovernanceResult<f64> {
    |                  ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `analyze_pattern` are never used
   --> src\governance\ai\correlation.rs:440:8
    |
439 | impl PatternAnalyzer {
    | -------------------- associated items in this implementation
440 |     fn new() -> Self {
    |        ^^^
...
463 |     pub fn analyze_pattern(&self, events: &[CrossDomainEvent]) -> Vec<CorrelationPattern> {
    |            ^^^^^^^^^^^^^^^

warning: struct `PredictiveComplianceEngine` is never constructed
  --> src\governance\ai\prediction.rs:18:12
   |
18 | pub struct PredictiveComplianceEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PredictionModel` is never constructed
  --> src\governance\ai\prediction.rs:89:12
   |
89 | pub struct PredictionModel {
   |            ^^^^^^^^^^^^^^^

warning: enum `ModelType` is never used
  --> src\governance\ai\prediction.rs:97:10
   |
97 | pub enum ModelType {
   |          ^^^^^^^^^

warning: struct `HistoricalDataStore` is never constructed
   --> src\governance\ai\prediction.rs:105:12
    |
105 | pub struct HistoricalDataStore {
    |            ^^^^^^^^^^^^^^^^^^^

warning: struct `TrendAnalyzer` is never constructed
   --> src\governance\ai\prediction.rs:139:12
    |
139 | pub struct TrendAnalyzer {
    |            ^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\governance\ai\prediction.rs:146:18
    |
145 | impl PredictiveComplianceEngine {
    | ------------------------------- associated items in this implementation
146 |     pub async fn new(
    |                  ^^^
...
196 |     pub async fn predict_compliance_drift(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
229 |     async fn predict_compliance_degradation(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
284 |     async fn predict_resource_violations(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
331 |     async fn predict_policy_drift(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |              ^^^^^^^^^^^^^^^^^^^^
...
371 |     async fn predict_cost_compliance_issues(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
411 |     async fn predict_security_compliance_gaps(&self, time_horizon_days: u32) -> GovernanceResult<Vec<CompliancePrediction>> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
456 |     pub async fn analyze_compliance_trends(&self, metric: &str, days: u32) -> GovernanceResult<ComplianceTrend> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^
...
494 |     pub async fn get_prediction_accuracy(&self) -> GovernanceResult<HashMap<String, f64>> {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^
...
505 |     pub async fn retrain_models(&mut self) -> GovernanceResult<()> {
    |                  ^^^^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\governance\ai\prediction.rs:517:8
    |
516 | impl HistoricalDataStore {
    | ------------------------ associated function in this implementation
517 |     fn new() -> Self {
    |        ^^^

warning: associated items `new` and `analyze_compliance_trend` are never used
   --> src\governance\ai\prediction.rs:543:8
    |
542 | impl TrendAnalyzer {
    | ------------------ associated items in this implementation
543 |     fn new() -> Self {
    |        ^^^
...
550 |     fn analyze_compliance_trend(&self, data: &[HistoricalCompliance]) -> ComplianceTrend {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: trait `PredictiveModel` is never used
   --> src\ml\mod.rs:103:11
    |
103 | pub trait PredictiveModel {
    |           ^^^^^^^^^^^^^^^

warning: struct `PredictiveComplianceEngine` is never constructed
  --> src\ml\predictive_compliance.rs:15:12
   |
15 | pub struct PredictiveComplianceEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\predictive_compliance.rs:42:12
    |
41  | impl PredictiveComplianceEngine {
    | ------------------------------- associated items in this implementation
42  |     pub fn new(azure_client: AzureClient) -> Self {
    |            ^^^
...
51  |     pub async fn predict_violations(&self, resource_id: &str, lookahead_hours: i64) -> Result<Vec<ViolationPrediction>, String> {
    |                  ^^^^^^^^^^^^^^^^^^
...
87  |     async fn get_resource_configuration(&self, resource_id: &str) -> Result<serde_json::Value, String> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
112 |     async fn get_applicable_policies(&self, resource: &serde_json::Value) -> Result<Vec<PolicyDefinition>, String> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
144 |     async fn analyze_drift(&self, resource: &serde_json::Value) -> Result<Vec<DriftIndicator>, String> {
    |              ^^^^^^^^^^^^^
...
174 |     async fn predict_policy_violation(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
222 |     fn calculate_business_impact(&self, resource: &serde_json::Value, policy: &PolicyDefinition) -> BusinessImpact {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
233 |     fn generate_remediation_suggestions(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
284 |     pub async fn train_models(&mut self, training_data: Vec<serde_json::Value>) -> Result<ModelMetrics, String> {
    |                  ^^^^^^^^^^^^

warning: method `prioritize_predictions` is never used
   --> src\ml\risk_scoring.rs:239:12
    |
68  | impl RiskScoringEngine {
    | ---------------------- method in this implementation
...
239 |     pub fn prioritize_predictions(&self, predictions: &mut Vec<ViolationPrediction>) {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PatternAnalyzer` is never constructed
  --> src\ml\pattern_analysis.rs:13:12
   |
13 | pub struct PatternAnalyzer {
   |            ^^^^^^^^^^^^^^^

warning: struct `AnomalyDetector` is never constructed
  --> src\ml\pattern_analysis.rs:56:8
   |
56 | struct AnomalyDetector {
   |        ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\pattern_analysis.rs:63:12
    |
62  | impl PatternAnalyzer {
    | -------------------- associated items in this implementation
63  |     pub fn new() -> Self {
    |            ^^^
...
75  |     fn initialize_pattern_library() -> HashMap<String, ViolationPattern> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
174 |     pub fn detect_patterns(&self, resource_id: &str, time_window: Duration) -> Vec<DetectedPattern> {
    |            ^^^^^^^^^^^^^^^
...
193 |     fn match_pattern(
    |        ^^^^^^^^^^^^^
...
226 |     fn extract_features(&self, time_series: &VecDeque<TimeSeriesPoint>) -> Vec<f64> {
    |        ^^^^^^^^^^^^^^^^
...
262 |     fn calculate_similarity(&self, features1: &[f64], features2: &[f64]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
279 |     fn check_temporal_sequence(
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
289 |     fn get_matched_events(
    |        ^^^^^^^^^^^^^^^^^^
...
297 |     fn generate_recommendation(&self, pattern: &ViolationPattern) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
320 |     pub fn add_data_point(&mut self, resource_id: String, value: f64, metadata: HashMap<String, String>) {
    |            ^^^^^^^^^^^^^^
...
340 |     pub fn detect_anomalies(&self, resource_id: &str) -> Vec<Anomaly> {
    |            ^^^^^^^^^^^^^^^^

warning: struct `DriftDetector` is never constructed
  --> src\ml\drift_detector.rs:13:12
   |
13 | pub struct DriftDetector {
   |            ^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\drift_detector.rs:73:12
    |
72  | impl DriftDetector {
    | ------------------ associated items in this implementation
73  |     pub fn new() -> Self {
    |            ^^^
...
81  |     pub fn establish_baseline(&mut self, resource_id: String, configuration: serde_json::Value) {
    |            ^^^^^^^^^^^^^^^^^^
...
95  |     fn identify_critical_properties(&self, configuration: &serde_json::Value) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
111 |     fn evaluate_compliance(&self, configuration: &serde_json::Value) -> ComplianceState {
    |        ^^^^^^^^^^^^^^^^^^^
...
143 |     pub fn detect_drift(&mut self, resource_id: &str, current_config: &serde_json::Value) -> DriftAnalysis {
    |            ^^^^^^^^^^^^
...
185 |     fn calculate_drift(
    |        ^^^^^^^^^^^^^^^
...
215 |     fn calculate_property_drift_score(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
242 |     fn determine_impact(&self, drift_score: f64) -> DriftImpact {
    |        ^^^^^^^^^^^^^^^^
...
254 |     fn calculate_total_drift_score(&self, drift_events: &[DriftEvent]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
272 |     fn calculate_drift_velocity(&self, resource_id: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
308 |     fn identify_critical_drifts(&self, drift_events: &[DriftEvent]) -> Vec<CriticalDrift> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
321 |     fn describe_impact(
    |        ^^^^^^^^^^^^^^^
...
341 |     fn predict_time_to_violation(&self, drift_score: f64, velocity: f64) -> Option<i64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
358 |     fn generate_recommendations(&self, drift_events: &[DriftEvent], total_drift_score: f64) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `NaturalLanguageEngine` is never constructed
  --> src\ml\natural_language.rs:17:12
   |
17 | pub struct NaturalLanguageEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `IntentClassifier` is never constructed
   --> src\ml\natural_language.rs:143:8
    |
143 | struct IntentClassifier {
    |        ^^^^^^^^^^^^^^^^

warning: struct `EntityExtractor` is never constructed
   --> src\ml\natural_language.rs:148:8
    |
148 | struct EntityExtractor {
    |        ^^^^^^^^^^^^^^^

warning: struct `PolicyTranslator` is never constructed
   --> src\ml\natural_language.rs:154:8
    |
154 | struct PolicyTranslator {
    |        ^^^^^^^^^^^^^^^^

warning: struct `GovernanceKnowledgeBase` is never constructed
   --> src\ml\natural_language.rs:176:8
    |
176 | struct GovernanceKnowledgeBase {
    |        ^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\natural_language.rs:210:12
    |
209 | impl NaturalLanguageEngine {
    | -------------------------- associated items in this implementation
210 |     pub fn new() -> Self {
    |            ^^^
...
220 |     pub async fn process_query(&mut self, query: ConversationQuery) -> ConversationResponse {
    |                  ^^^^^^^^^^^^^
...
252 |     async fn handle_compliance_status(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
292 |     async fn handle_policy_violations(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
327 |     async fn handle_explain_policy(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^^
...
366 |     async fn handle_remediation(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^
...
414 |     async fn handle_create_policy(&self, input: &str, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^
...
447 |     async fn handle_predict_violations(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
493 |     async fn handle_help_request(&self) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^
...
511 |     async fn handle_unknown_intent(&self) -> ConversationResponse {
    |              ^^^^^^^^^^^^^^^^^^^^^
...
523 |     pub fn translate_to_azure_policy(&self, natural_language: &str) -> Result<serde_json::Value, String> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `build_keyword_map`, and `classify` are never used
   --> src\ml\natural_language.rs:529:8
    |
528 | impl IntentClassifier {
    | --------------------- associated items in this implementation
529 |     fn new() -> Self {
    |        ^^^
...
573 |     fn build_keyword_map(patterns: &HashMap<IntentType, Vec<String>>) -> HashMap<String, Vec<IntentType>> {
    |        ^^^^^^^^^^^^^^^^^
...
589 |     fn classify(&self, input: &str) -> Intent {
    |        ^^^^^^^^

warning: associated items `new`, `init_patterns`, `init_azure_services`, and `extract` are never used
   --> src\ml\natural_language.rs:626:8
    |
625 | impl EntityExtractor {
    | -------------------- associated items in this implementation
626 |     fn new() -> Self {
    |        ^^^
...
640 |     fn init_patterns() -> HashMap<EntityType, Vec<String>> {
    |        ^^^^^^^^^^^^^
...
663 |     fn init_azure_services() -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^
...
674 |     fn extract(&self, input: &str) -> Vec<EntityInfo> {
    |        ^^^^^^^

warning: associated items `new`, `init_templates`, `translate_to_policy`, and `natural_language_to_policy` are never used
   --> src\ml\natural_language.rs:722:8
    |
721 | impl PolicyTranslator {
    | --------------------- associated items in this implementation
722 |     fn new() -> Self {
    |        ^^^
...
737 |     fn init_templates() -> HashMap<String, PolicyTemplate> {
    |        ^^^^^^^^^^^^^^
...
766 |     fn translate_to_policy(&self, input: &str, entities: &[EntityInfo]) -> serde_json::Value {
    |        ^^^^^^^^^^^^^^^^^^^
...
780 |     fn natural_language_to_policy(&self, natural_language: &str) -> Result<serde_json::Value, String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: methods `update` and `add_turn` are never used
   --> src\ml\natural_language.rs:796:8
    |
785 | impl ConversationContext {
    | ------------------------ methods in this implementation
...
796 |     fn update(&mut self, intent: &Intent, entities: &[EntityInfo]) {
    |        ^^^^^^
...
807 |     fn add_turn(&mut self, turn: ConversationTurn) {
    |        ^^^^^^^^

warning: associated functions `new`, `init_concepts`, `init_policies`, `init_best_practices`, and `init_compliance_mappings` are never used
   --> src\ml\natural_language.rs:818:8
    |
817 | impl GovernanceKnowledgeBase {
    | ---------------------------- associated functions in this implementation
818 |     fn new() -> Self {
    |        ^^^
...
827 |     fn init_concepts() -> HashMap<String, ConceptDefinition> {
    |        ^^^^^^^^^^^^^
...
840 |     fn init_policies() -> HashMap<String, PolicyInfo> {
    |        ^^^^^^^^^^^^^
...
844 |     fn init_best_practices() -> Vec<BestPractice> {
    |        ^^^^^^^^^^^^^^^^^^^
...
848 |     fn init_compliance_mappings() -> HashMap<String, Vec<String>> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `GraphNeuralNetwork` is never constructed
  --> src\ml\graph_neural_network.rs:18:12
   |
18 | pub struct GraphNeuralNetwork {
   |            ^^^^^^^^^^^^^^^^^^

warning: struct `GraphLayer` is never constructed
  --> src\ml\graph_neural_network.rs:75:8
   |
75 | struct GraphLayer {
   |        ^^^^^^^^^^

warning: enum `LayerType` is never used
  --> src\ml\graph_neural_network.rs:83:6
   |
83 | enum LayerType {
   |      ^^^^^^^^^

warning: enum `ActivationFunction` is never used
  --> src\ml\graph_neural_network.rs:91:6
   |
91 | enum ActivationFunction {
   |      ^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\graph_neural_network.rs:250:12
    |
249 | impl GraphNeuralNetwork {
    | ----------------------- associated items in this implementation
250 |     pub fn new() -> Self {
    |            ^^^
...
260 |     fn initialize_layers() -> Vec<GraphLayer> {
    |        ^^^^^^^^^^^^^^^^^
...
289 |     pub fn add_resource(&mut self, node: ResourceNode) {
    |            ^^^^^^^^^^^^
...
294 |     pub fn add_relationship(&mut self, edge: ResourceEdge) {
    |            ^^^^^^^^^^^^^^^^
...
298 |     pub fn analyze_cross_domain_impact(&self, source_resource: &str) -> CrossDomainImpact {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
319 |     fn propagate_impact(&self, source: &str) -> Vec<AffectedResource> {
    |        ^^^^^^^^^^^^^^^^
...
358 |     fn calculate_domain_impacts(&self, affected: &[AffectedResource]) -> HashMap<String, DomainImpact> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
400 |     fn identify_cascading_effects(&self, domain: &GovernanceDomain) -> Vec<CascadingEffect> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
440 |     fn predict_violations_from_impact(&self, source: &str) -> Vec<PredictedViolation> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
459 |     fn determine_risk_level(&self, score: f64) -> RiskLevel {
    |        ^^^^^^^^^^^^^^^^^^^^
...
466 |     fn calculate_priority(&self, score: f64) -> u32 {
    |        ^^^^^^^^^^^^^^^^^^
...
470 |     pub fn detect_correlation_patterns(&self) -> Vec<CorrelationPattern> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
519 |     fn group_resources_by_type(&self) -> HashMap<String, Vec<String>> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
529 |     pub fn perform_what_if_analysis(&self, scenario: WhatIfScenario) -> WhatIfAnalysis {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
...
553 |     fn capture_snapshot(&self) -> GraphSnapshot {
    |        ^^^^^^^^^^^^^^^^
...
569 |     fn count_domains(&self) -> HashMap<GovernanceDomain, usize> {
    |        ^^^^^^^^^^^^^
...
577 |     fn find_critical_paths(&self) -> Vec<Vec<String>> {
    |        ^^^^^^^^^^^^^^^^^^^
...
584 |     fn apply_change(&mut self, change: &ProposedChange) {
    |        ^^^^^^^^^^^^
...
590 |     fn analyze_scenario_impacts(&self, scenario: &WhatIfScenario) -> Vec<CrossDomainImpact> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
596 |     fn generate_recommendations(&self, impacts: &[CrossDomainImpact]) -> Vec<Recommendation> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
616 |     pub fn compute_embeddings(&mut self) {
    |            ^^^^^^^^^^^^^^^^^^
...
624 |     fn aggregate_neighbor_features(&self, node_id: &str) -> Vec<f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `CorrelationEngine` is never constructed
  --> src\ml\correlation_engine.rs:26:12
   |
26 | pub struct CorrelationEngine {
   |            ^^^^^^^^^^^^^^^^^

warning: struct `PatternDetector` is never constructed
   --> src\ml\correlation_engine.rs:100:8
    |
100 | struct PatternDetector {
    |        ^^^^^^^^^^^^^^^

warning: struct `PatternTemplate` is never constructed
   --> src\ml\correlation_engine.rs:106:8
    |
106 | struct PatternTemplate {
    |        ^^^^^^^^^^^^^^^

warning: struct `DetectionRule` is never constructed
   --> src\ml\correlation_engine.rs:113:8
    |
113 | struct DetectionRule {
    |        ^^^^^^^^^^^^^

warning: struct `DetectedPattern` is never constructed
   --> src\ml\correlation_engine.rs:119:8
    |
119 | struct DetectedPattern {
    |        ^^^^^^^^^^^^^^^

warning: struct `PatternMLModel` is never constructed
   --> src\ml\correlation_engine.rs:125:8
    |
125 | struct PatternMLModel {
    |        ^^^^^^^^^^^^^^

warning: struct `AnomalyDetector` is never constructed
   --> src\ml\correlation_engine.rs:131:8
    |
131 | struct AnomalyDetector {
    |        ^^^^^^^^^^^^^^^

warning: struct `BaselineMetric` is never constructed
   --> src\ml\correlation_engine.rs:137:8
    |
137 | struct BaselineMetric {
    |        ^^^^^^^^^^^^^^

warning: struct `AnomalyMLModel` is never constructed
   --> src\ml\correlation_engine.rs:145:8
    |
145 | struct AnomalyMLModel {
    |        ^^^^^^^^^^^^^^

warning: struct `IsolationForest` is never constructed
   --> src\ml\correlation_engine.rs:150:8
    |
150 | struct IsolationForest {
    |        ^^^^^^^^^^^^^^^

warning: struct `IsolationTree` is never constructed
   --> src\ml\correlation_engine.rs:155:8
    |
155 | struct IsolationTree {
    |        ^^^^^^^^^^^^^

warning: struct `TreeNode` is never constructed
   --> src\ml\correlation_engine.rs:160:8
    |
160 | struct TreeNode {
    |        ^^^^^^^^

warning: struct `Autoencoder` is never constructed
   --> src\ml\correlation_engine.rs:167:8
    |
167 | struct Autoencoder {
    |        ^^^^^^^^^^^

warning: struct `RealTimeProcessor` is never constructed
   --> src\ml\correlation_engine.rs:173:8
    |
173 | struct RealTimeProcessor {
    |        ^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\correlation_engine.rs:203:12
    |
202 | impl CorrelationEngine {
    | ---------------------- associated items in this implementation
203 |     pub fn new() -> Self {
    |            ^^^
...
213 |     pub async fn analyze_correlations(&self, resource_ids: Vec<String>) -> CorrelationResult {
    |                  ^^^^^^^^^^^^^^^^^^^^
...
252 |     async fn find_correlations(&self, resource_ids: &[String], gnn: &GraphNeuralNetwork) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^
...
273 |     async fn analyze_pair(&self, resource_a: &str, resource_b: &str, gnn: &GraphNeuralNetwork) -> Option<ResourceCorrelation> {
    |              ^^^^^^^^^^^^
...
303 |     fn map_relationship_to_correlation(&self, relationship: &RelationshipType) -> CorrelationType {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
317 |     fn get_shared_domains(&self, resource_a: &str, resource_b: &str, gnn: &GraphNeuralNetwork) -> Vec<GovernanceDomain> {
    |        ^^^^^^^^^^^^^^^^^^
...
330 |     async fn find_indirect_correlations(&self, resource_id: &str, gnn: &GraphNeuralNetwork) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
377 |     fn create_empty_impact(&self) -> CrossDomainImpact {
    |        ^^^^^^^^^^^^^^^^^^^
...
392 |     pub async fn perform_what_if_analysis(&self, scenario: WhatIfScenario) -> WhatIfAnalysis {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^
...
397 |     pub async fn process_event(&self, event: GovernanceEvent) {
    |                  ^^^^^^^^^^^^^
...
401 |     pub async fn get_real_time_insights(&self) -> RealTimeInsights {
    |                  ^^^^^^^^^^^^^^^^^^^^^^
...
415 |     async fn count_active_correlations(&self) -> usize {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
419 |     async fn identify_top_risks(&self, gnn: &GraphNeuralNetwork) -> Vec<RiskItem> {
    |              ^^^^^^^^^^^^^^^^^^
...
439 |     fn generate_recommendations(&self, patterns: &[CorrelationPattern]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_patterns`, and `detect_patterns` are never used
   --> src\ml\correlation_engine.rs:481:8
    |
480 | impl PatternDetector {
    | -------------------- associated items in this implementation
481 |     fn new() -> Self {
    |        ^^^
...
493 |     fn initialize_patterns() -> HashMap<String, PatternTemplate> {
    |        ^^^^^^^^^^^^^^^^^^^
...
512 |     fn detect_patterns(&self, correlations: &[ResourceCorrelation]) -> Vec<CorrelationPattern> {
    |        ^^^^^^^^^^^^^^^

warning: associated items `new`, `detect_anomalies`, and `get_recent_anomalies` are never used
   --> src\ml\correlation_engine.rs:543:8
    |
542 | impl AnomalyDetector {
    | -------------------- associated items in this implementation
543 |     fn new() -> Self {
    |        ^^^
...
561 |     async fn detect_anomalies(&self, resource_ids: &[String], gnn: &GraphNeuralNetwork) -> Vec<Anomaly> {
    |              ^^^^^^^^^^^^^^^^
...
597 |     fn get_recent_anomalies(&self) -> Vec<Anomaly> {
    |        ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `process_event`, and `process_batch` are never used
   --> src\ml\correlation_engine.rs:604:8
    |
603 | impl RealTimeProcessor {
    | ---------------------- associated items in this implementation
604 |     fn new() -> Self {
    |        ^^^
...
612 |     async fn process_event(&self, event: GovernanceEvent) {
    |              ^^^^^^^^^^^^^
...
621 |     async fn process_batch(&self) {
    |              ^^^^^^^^^^^^^

warning: struct `DataBuffer` is never constructed
  --> src\ml\continuous_training.rs:45:12
   |
45 | pub struct DataBuffer {
   |            ^^^^^^^^^^

warning: associated items `new`, `add_sample`, `get_batch`, `size`, and `clear` are never used
  --> src\ml\continuous_training.rs:51:12
   |
50 | impl DataBuffer {
   | --------------- associated items in this implementation
51 |     pub fn new(max_size: usize) -> Self {
   |            ^^^
...
58 |     pub async fn add_sample(&self, sample: TrainingSample) {
   |                  ^^^^^^^^^^
...
68 |     pub async fn get_batch(&self, size: usize) -> Vec<TrainingSample> {
   |                  ^^^^^^^^^
...
76 |     pub async fn size(&self) -> usize {
   |                  ^^^^
...
80 |     pub async fn clear(&self) {
   |                  ^^^^^

warning: struct `ContinuousTrainingPipeline` is never constructed
  --> src\ml\continuous_training.rs:97:12
   |
97 | pub struct ContinuousTrainingPipeline {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\continuous_training.rs:107:12
    |
106 | impl ContinuousTrainingPipeline {
    | ------------------------------- associated items in this implementation
107 |     pub fn new(config: TrainingConfig) -> Self {
    |            ^^^
...
119 |     pub async fn add_training_data(&self, sample: TrainingSample) -> Result<(), String> {
    |                  ^^^^^^^^^^^^^^^^^
...
132 |     async fn trigger_retraining(&self) -> Result<(), String> {
    |              ^^^^^^^^^^^^^^^^^^
...
188 |     async fn train_model(&self, training_data: &[TrainingSample]) -> Result<ModelVersion, String> {
    |              ^^^^^^^^^^^
...
206 |     async fn validate_model(&self, model: &ModelVersion, validation_data: &[TrainingSample]) -> Result<ValidationMetrics, Stri...
    |              ^^^^^^^^^^^^^^
...
220 |     async fn deploy_model(&self, model: ModelVersion) -> Result<(), String> {
    |              ^^^^^^^^^^^^
...
247 |     async fn record_training_metrics(&self, metrics: TrainingMetrics) {
    |              ^^^^^^^^^^^^^^^^^^^^^^^
...
259 |     pub async fn get_model_performance(&self) -> Result<ModelPerformance, String> {
    |                  ^^^^^^^^^^^^^^^^^^^^^
...
285 |     pub async fn rollback_model(&self, version_id: &str) -> Result<(), String> {
    |                  ^^^^^^^^^^^^^^

warning: struct `ModelMetadata` is never constructed
  --> src\ml\confidence_scoring.rs:19:12
   |
19 | pub struct ModelMetadata {
   |            ^^^^^^^^^^^^^

warning: struct `ConfidenceScorer` is never constructed
  --> src\ml\confidence_scoring.rs:28:12
   |
28 | pub struct ConfidenceScorer {
   |            ^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\confidence_scoring.rs:56:12
    |
55  | impl ConfidenceScorer {
    | --------------------- associated items in this implementation
56  |     pub fn new() -> Self {
    |            ^^^
...
66  |     pub fn calculate_confidence(&self, prediction: &PredictionOutput, features: &FeatureSet) -> ConfidenceScore {
    |            ^^^^^^^^^^^^^^^^^^^^
...
110 |     fn calculate_ensemble_confidence(&self, prediction: &PredictionOutput) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
125 |     fn calculate_historical_confidence(&self, prediction_type: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
132 |     fn calculate_temporal_confidence(&self, metadata: &ModelMetadata) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
147 |     fn generate_explanation(&self, score: f64, _weights: &ConfidenceWeights) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^
...
171 |     pub fn update_historical_accuracy(&mut self, prediction_type: String, accuracy: f64) {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
176 |     pub fn add_ensemble_model(&mut self, model: Box<dyn PredictionModel>) {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `FeatureQualityAnalyzer` is never constructed
   --> src\ml\confidence_scoring.rs:182:12
    |
182 | pub struct FeatureQualityAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `calculate_quality_score` are never used
   --> src\ml\confidence_scoring.rs:189:12
    |
188 | impl FeatureQualityAnalyzer {
    | --------------------------- associated items in this implementation
189 |     pub fn new() -> Self {
    |            ^^^
...
198 |     pub fn calculate_quality_score(&self, features: &FeatureSet) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ConfidenceWeights` is never constructed
   --> src\ml\confidence_scoring.rs:245:8
    |
245 | struct ConfidenceWeights {
    |        ^^^^^^^^^^^^^^^^^

warning: struct `EnsembleDisagreementCalculator` is never constructed
   --> src\ml\confidence_scoring.rs:253:12
    |
253 | pub struct EnsembleDisagreementCalculator {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `calculate_disagreement`, and `add_model` are never used
   --> src\ml\confidence_scoring.rs:258:12
    |
257 | impl EnsembleDisagreementCalculator {
    | ----------------------------------- associated items in this implementation
258 |     pub fn new() -> Self {
    |            ^^^
...
265 |     pub fn calculate_disagreement(&self, features: &FeatureSet) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^^^^
...
291 |     pub fn add_model(&mut self, model: Box<dyn PredictionModel>) {
    |            ^^^^^^^^^

warning: trait `PredictionModel` is never used
   --> src\ml\confidence_scoring.rs:297:11
    |
297 | pub trait PredictionModel: Send + Sync {
    |           ^^^^^^^^^^^^^^^

warning: struct `ModelPrediction` is never constructed
   --> src\ml\confidence_scoring.rs:304:12
    |
304 | pub struct ModelPrediction {
    |            ^^^^^^^^^^^^^^^

warning: struct `FeatureSet` is never constructed
   --> src\ml\confidence_scoring.rs:312:12
    |
312 | pub struct FeatureSet {
    |            ^^^^^^^^^^

warning: struct `PredictionOutput` is never constructed
   --> src\ml\confidence_scoring.rs:322:12
    |
322 | pub struct PredictionOutput {
    |            ^^^^^^^^^^^^^^^^

warning: struct `ConfidenceMonitor` is never constructed
   --> src\ml\confidence_scoring.rs:331:12
    |
331 | pub struct ConfidenceMonitor {
    |            ^^^^^^^^^^^^^^^^^

warning: associated items `new`, `check_threshold`, and `get_recent_alerts` are never used
   --> src\ml\confidence_scoring.rs:337:12
    |
336 | impl ConfidenceMonitor {
    | ---------------------- associated items in this implementation
337 |     pub fn new() -> Self {
    |            ^^^
...
350 |     pub fn check_threshold(&mut self, action_type: &str, confidence: f64) -> ThresholdCheck {
    |            ^^^^^^^^^^^^^^^
...
377 |     pub fn get_recent_alerts(&self, hours: i64) -> Vec<&ConfidenceAlert> {
    |            ^^^^^^^^^^^^^^^^^

warning: enum `ThresholdCheck` is never used
   --> src\ml\confidence_scoring.rs:387:10
    |
387 | pub enum ThresholdCheck {
    |          ^^^^^^^^^^^^^^

warning: struct `ConfidenceAlert` is never constructed
   --> src\ml\confidence_scoring.rs:394:12
    |
394 | pub struct ConfidenceAlert {
    |            ^^^^^^^^^^^^^^^

warning: struct `PredictionExplainer` is never constructed
  --> src\ml\explainability.rs:18:12
   |
18 | pub struct PredictionExplainer {
   |            ^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\explainability.rs:26:12
    |
25  | impl PredictionExplainer {
    | ------------------------ associated items in this implementation
26  |     pub fn new() -> Self {
    |            ^^^
...
42  |     pub fn explain_violation_prediction(
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
86  |     fn identify_top_factors(
    |        ^^^^^^^^^^^^^^^^^^^^
...
135 |     fn generate_narrative_explanation(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
174 |     fn generate_recommendations(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
258 |     fn generate_counterfactuals(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
301 |     fn calculate_feature_importance(&self, shap_values: &[f64]) -> HashMap<String, f64> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
316 |     fn assess_explanation_confidence(&self, shap_values: &[f64]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
332 |     fn generate_visualizations(&self, shap_values: &[f64]) -> Vec<VisualizationData> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
354 |     fn initialize_feature_mappings(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
374 |     fn initialize_templates() -> HashMap<String, String> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
385 |     fn format_feature_value(&self, feature_name: &str, value: f64) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^
...
395 |     fn get_threshold_value(&self, feature_name: &str) -> String {
    |        ^^^^^^^^^^^^^^^^^^^
...
404 |     fn generate_factor_explanation(&self, feature_name: &str, value: f64, impact: f64) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
417 |     fn calculate_counterfactual_value(&self, feature_name: &str, current: f64) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ShapValueCalculator` is never constructed
   --> src\ml\explainability.rs:428:12
    |
428 | pub struct ShapValueCalculator {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new` and `calculate_shap_values` are never used
   --> src\ml\explainability.rs:433:12
    |
432 | impl ShapValueCalculator {
    | ------------------------ associated items in this implementation
433 |     pub fn new() -> Self {
    |            ^^^
...
439 |     pub fn calculate_shap_values(&self, feature_values: &[f64]) -> Vec<f64> {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceData` is never constructed
   --> src\ml\explainability.rs:531:12
    |
531 | pub struct ResourceData {
    |            ^^^^^^^^^^^^

warning: struct `ViolationPatternLibrary` is never constructed
  --> src\ml\pattern_library.rs:17:12
   |
17 | pub struct ViolationPatternLibrary {
   |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\pattern_library.rs:23:12
    |
22  | impl ViolationPatternLibrary {
    | ---------------------------- associated items in this implementation
23  |     pub fn new() -> Self {
    |            ^^^
...
33  |     fn initialize_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^
...
179 |     pub fn match_patterns(&self, resource_config: &ResourceConfiguration) -> Vec<PatternMatch> {
    |            ^^^^^^^^^^^^^^
...
203 |     fn calculate_match_score(&self, config: &ResourceConfiguration, pattern: &ViolationPattern) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
224 |     fn evaluate_rule(&self, config: &ResourceConfiguration, condition: &str) -> bool {
    |        ^^^^^^^^^^^^^
...
244 |     fn get_matched_indicators(&self, config: &ResourceConfiguration, pattern: &ViolationPattern) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
259 |     pub fn get_pattern(&self, pattern_id: &str) -> Option<&ViolationPattern> {
    |            ^^^^^^^^^^^
...
264 |     pub fn get_patterns_by_category(&self, category: PatternCategory) -> Vec<&ViolationPattern> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^
...
271 |     pub fn get_high_confidence_patterns(&self) -> Vec<&ViolationPattern> {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
278 |     pub fn update_pattern_confidence(&mut self, pattern_id: &str, was_correct: bool) {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceConfiguration` is never constructed
   --> src\ml\pattern_library.rs:351:12
    |
351 | pub struct ResourceConfiguration {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: trait `PatternMatcher` is never used
   --> src\ml\pattern_library.rs:360:11
    |
360 | pub trait PatternMatcher: Send + Sync {
    |           ^^^^^^^^^^^^^^

warning: struct `CostPredictionModel` is never constructed
  --> src\ml\cost_prediction.rs:17:12
   |
17 | pub struct CostPredictionModel {
   |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `predict_monthly_cost`, `calculate_resource_cost`, `identify_optimization`, `calculate_confidence`, and `update_with_actual` are never used
   --> src\ml\cost_prediction.rs:25:12
    |
24  | impl CostPredictionModel {
    | ------------------------ associated items in this implementation
25  |     pub fn new() -> Self {
    |            ^^^
...
35  |     pub fn predict_monthly_cost(&self, resources: &[ResourceUsage]) -> CostPrediction {
    |            ^^^^^^^^^^^^^^^^^^^^
...
70  |     fn calculate_resource_cost(&self, resource: &ResourceUsage) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
83  |     fn identify_optimization(&self, resource: &ResourceUsage) -> Option<OptimizationOpportunity> {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
109 |     fn calculate_confidence(&self) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
124 |     pub fn update_with_actual(&mut self, actual: CostDataPoint) {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `TrendModel` is never constructed
   --> src\ml\cost_prediction.rs:141:12
    |
141 | pub struct TrendModel {
    |            ^^^^^^^^^^

warning: associated items `new`, `update`, and `get_trend_factor` are never used
   --> src\ml\cost_prediction.rs:148:12
    |
147 | impl TrendModel {
    | --------------- associated items in this implementation
148 |     pub fn new() -> Self {
    |            ^^^
...
156 |     pub fn update(&mut self, data: &[CostDataPoint]) {
    |            ^^^^^^
...
182 |     pub fn get_trend_factor(&self) -> f64 {
    |            ^^^^^^^^^^^^^^^^

warning: struct `SeasonalityModel` is never constructed
   --> src\ml\cost_prediction.rs:189:12
    |
189 | pub struct SeasonalityModel {
    |            ^^^^^^^^^^^^^^^^

warning: associated items `new`, `update`, and `get_seasonal_factor` are never used
   --> src\ml\cost_prediction.rs:195:12
    |
194 | impl SeasonalityModel {
    | --------------------- associated items in this implementation
195 |     pub fn new() -> Self {
    |            ^^^
...
202 |     pub fn update(&mut self, data: &[CostDataPoint]) {
    |            ^^^^^^
...
228 |     pub fn get_seasonal_factor(&self, date: DateTime<Utc>) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^

warning: struct `ResourceCostModel` is never constructed
   --> src\ml\cost_prediction.rs:235:12
    |
235 | pub struct ResourceCostModel {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `CostAnomalyDetector` is never constructed
   --> src\ml\cost_prediction.rs:301:12
    |
301 | pub struct CostAnomalyDetector {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `detect_anomaly`, and `update_baseline` are never used
   --> src\ml\cost_prediction.rs:307:12
    |
306 | impl CostAnomalyDetector {
    | ------------------------ associated items in this implementation
307 |     pub fn new() -> Self {
    |            ^^^
...
314 |     pub fn detect_anomaly(&self, current_cost: f64) -> Option<CostAnomaly> {
    |            ^^^^^^^^^^^^^^
...
334 |     pub fn update_baseline(&mut self, costs: &[f64]) {
    |            ^^^^^^^^^^^^^^^

warning: struct `AnomalyDetector` is never constructed
  --> src\ml\anomaly_detection.rs:17:12
   |
17 | pub struct AnomalyDetector {
   |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\anomaly_detection.rs:25:12
    |
24  | impl AnomalyDetector {
    | -------------------- associated items in this implementation
25  |     pub fn new() -> Self {
    |            ^^^
...
35  |     pub fn detect_anomalies(&self, metrics: &[ResourceMetrics]) -> Vec<AnomalyResult> {
    |            ^^^^^^^^^^^^^^^^
...
69  |     fn extract_features(&self, metrics: &ResourceMetrics) -> Vec<f64> {
    |        ^^^^^^^^^^^^^^^^
...
82  |     fn classify_anomaly(&self, metrics: &ResourceMetrics, score: f64) -> AnomalyType {
    |        ^^^^^^^^^^^^^^^^
...
96  |     fn calculate_confidence(&self, iso_score: f64, stat_score: f64, pattern_score: f64) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
110 |     fn generate_description(&self, metrics: &ResourceMetrics, score: f64) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^
...
122 |     fn calculate_severity(&self, score: f64) -> Severity {
    |        ^^^^^^^^^^^^^^^^^^
...
134 |     fn recommend_action(&self, metrics: &ResourceMetrics, score: f64) -> String {
    |        ^^^^^^^^^^^^^^^^

warning: struct `IsolationForest` is never constructed
   --> src\ml\anomaly_detection.rs:148:12
    |
148 | pub struct IsolationForest {
    |            ^^^^^^^^^^^^^^^

warning: associated items `new` and `anomaly_score` are never used
   --> src\ml\anomaly_detection.rs:155:12
    |
154 | impl IsolationForest {
    | -------------------- associated items in this implementation
155 |     pub fn new(num_trees: usize, sample_size: usize) -> Self {
    |            ^^^
...
163 |     pub fn anomaly_score(&self, features: &[f64]) -> f64 {
    |            ^^^^^^^^^^^^^

warning: struct `IsolationTree` is never constructed
   --> src\ml\anomaly_detection.rs:179:8
    |
179 | struct IsolationTree {
    |        ^^^^^^^^^^^^^

warning: struct `StatisticalDetector` is never constructed
   --> src\ml\anomaly_detection.rs:187:12
    |
187 | pub struct StatisticalDetector {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `detect`, and `update_statistics` are never used
   --> src\ml\anomaly_detection.rs:194:12
    |
193 | impl StatisticalDetector {
    | ------------------------ associated items in this implementation
194 |     pub fn new() -> Self {
    |            ^^^
...
202 |     pub fn detect(&self, features: &[f64]) -> f64 {
    |            ^^^^^^
...
215 |     pub fn update_statistics(&mut self, data: &[Vec<f64>]) {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `PatternBasedDetector` is never constructed
   --> src\ml\anomaly_detection.rs:249:12
    |
249 | pub struct PatternBasedDetector {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_patterns`, and `detect` are never used
   --> src\ml\anomaly_detection.rs:254:12
    |
253 | impl PatternBasedDetector {
    | ------------------------- associated items in this implementation
254 |     pub fn new() -> Self {
    |            ^^^
...
260 |     fn initialize_patterns() -> Vec<Pattern> {
    |        ^^^^^^^^^^^^^^^^^^^
...
287 |     pub fn detect(&self, metrics: &ResourceMetrics) -> f64 {
    |            ^^^^^^

warning: struct `Pattern` is never constructed
   --> src\ml\anomaly_detection.rs:300:8
    |
300 | struct Pattern {
    |        ^^^^^^^

warning: method `match_score` is never used
   --> src\ml\anomaly_detection.rs:306:8
    |
305 | impl Pattern {
    | ------------ method in this implementation
306 |     fn match_score(&self, metrics: &ResourceMetrics) -> f64 {
    |        ^^^^^^^^^^^

warning: enum `Indicator` is never used
   --> src\ml\anomaly_detection.rs:320:6
    |
320 | enum Indicator {
    |      ^^^^^^^^^

warning: method `matches` is never used
   --> src\ml\anomaly_detection.rs:331:8
    |
330 | impl Indicator {
    | -------------- method in this implementation
331 |     fn matches(&self, metrics: &ResourceMetrics) -> bool {
    |        ^^^^^^^

warning: constant `MAX_HISTORY_SIZE` is never used
  --> src\ml\conversation_memory.rs:20:7
   |
20 | const MAX_HISTORY_SIZE: usize = 50;
   |       ^^^^^^^^^^^^^^^^

warning: constant `SESSION_TIMEOUT_HOURS` is never used
  --> src\ml\conversation_memory.rs:22:7
   |
22 | const SESSION_TIMEOUT_HOURS: i64 = 24;
   |       ^^^^^^^^^^^^^^^^^^^^^

warning: struct `ConversationMemory` is never constructed
  --> src\ml\conversation_memory.rs:25:12
   |
25 | pub struct ConversationMemory {
   |            ^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\conversation_memory.rs:32:12
    |
31  | impl ConversationMemory {
    | ----------------------- associated items in this implementation
32  |     pub fn new() -> Self {
    |            ^^^
...
41  |     pub async fn get_or_create_session(&self, session_id: &str) -> ConversationSession {
    |                  ^^^^^^^^^^^^^^^^^^^^^
...
55  |     pub async fn update_session(
    |                  ^^^^^^^^^^^^^^
...
93  |     pub async fn get_context(&self, session_id: &str) -> ConversationContext {
    |                  ^^^^^^^^^^^
...
104 |     pub async fn get_history(&self, session_id: &str, limit: usize) -> Vec<ConversationExchange> {
    |                  ^^^^^^^^^^^
...
120 |     pub async fn clear_session(&self, session_id: &str) {
    |                  ^^^^^^^^^^^^^
...
128 |     pub async fn get_relevant_entities(&self, session_id: &str, query: &str) -> Vec<ExtractedEntity> {
    |                  ^^^^^^^^^^^^^^^^^^^^^

warning: struct `ConversationSession` is never constructed
   --> src\ml\conversation_memory.rs:137:12
    |
137 | pub struct ConversationSession {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `add_exchange`, `update_entity`, `update_topic_stack`, and `get_current_topic` are never used
   --> src\ml\conversation_memory.rs:149:12
    |
148 | impl ConversationSession {
    | ------------------------ associated items in this implementation
149 |     pub fn new(id: &str) -> Self {
    |            ^^^
...
162 |     pub fn add_exchange(&mut self, exchange: ConversationExchange) {
    |            ^^^^^^^^^^^^
...
176 |     pub fn update_entity(&mut self, entity: ExtractedEntity) {
    |            ^^^^^^^^^^^^^
...
180 |     fn update_topic_stack(&mut self, intent: &Intent) {
    |        ^^^^^^^^^^^^^^^^^^
...
195 |     pub fn get_current_topic(&self) -> Option<&ConversationTopic> {
    |            ^^^^^^^^^^^^^^^^^

warning: struct `EntityStore` is never constructed
   --> src\ml\conversation_memory.rs:300:12
    |
300 | pub struct EntityStore {
    |            ^^^^^^^^^^^

warning: associated items `new`, `add_entity`, `get_relevant_entities`, and `clear_session` are never used
   --> src\ml\conversation_memory.rs:306:12
    |
305 | impl EntityStore {
    | ---------------- associated items in this implementation
306 |     pub fn new() -> Self {
    |            ^^^
...
313 |     pub async fn add_entity(&mut self, session_id: &str, entity: ExtractedEntity) {
    |                  ^^^^^^^^^^
...
322 |     pub async fn get_relevant_entities(&self, session_id: &str, query: &str) -> Vec<ExtractedEntity> {
    |                  ^^^^^^^^^^^^^^^^^^^^^
...
334 |     pub async fn clear_session(&mut self, session_id: &str) {
    |                  ^^^^^^^^^^^^^

warning: struct `ContextAnalyzer` is never constructed
   --> src\ml\conversation_memory.rs:340:12
    |
340 | pub struct ContextAnalyzer {
    |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\conversation_memory.rs:347:12
    |
346 | impl ContextAnalyzer {
    | -------------------- associated items in this implementation
347 |     pub fn new() -> Self {
    |            ^^^
...
357 |     fn initialize_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^
...
390 |     pub async fn analyze(&self, session: &ConversationSession) -> ConversationContext {
    |                  ^^^^^^^
...
410 |     async fn analyze_entities(&self, session: &ConversationSession, context: &mut ConversationContext) {
    |              ^^^^^^^^^^^^^^^^
...
449 |     async fn analyze_conversation_flow(&self, session: &ConversationSession, context: &mut ConversationContext) {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
481 |     async fn analyze_user_intent(&self, session: &ConversationSession, context: &mut ConversationContext) {
    |              ^^^^^^^^^^^^^^^^^^^
...
516 |     async fn identify_clarifications_needed(&self, session: &ConversationSession, context: &mut ConversationContext) {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
559 |     fn contains_patterns(&self, text: &str, patterns: &[String]) -> bool {
    |        ^^^^^^^^^^^^^^^^^
...
563 |     fn calculate_intent_consistency(&self, intents: &[&IntentType]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
587 |     fn intents_are_related(&self, intent1: &IntentType, intent2: &IntentType) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^

warning: struct `ConversationMetrics` is never constructed
   --> src\ml\conversation_memory.rs:655:12
    |
655 | pub struct ConversationMetrics {
    |            ^^^^^^^^^^^^^^^^^^^

warning: method `get_metrics` is never used
   --> src\ml\conversation_memory.rs:666:18
    |
664 | impl ConversationMemory {
    | ----------------------- method in this implementation
665 |     /// Get conversation metrics
666 |     pub async fn get_metrics(&self, session_id: &str) -> ConversationMetrics {
    |                  ^^^^^^^^^^^

warning: struct `IntentRouter` is never constructed
  --> src\ml\intent_router.rs:18:12
   |
18 | pub struct IntentRouter {
   |            ^^^^^^^^^^^^

warning: associated items `new`, `register_handlers`, `route_query`, `aggregate_intents`, `get_handler_name`, and `execute_handler` are never used
   --> src\ml\intent_router.rs:25:12
    |
24  | impl IntentRouter {
    | ----------------- associated items in this implementation
25  |     pub fn new() -> Self {
    |            ^^^
...
40  |     fn register_handlers(&mut self) {
    |        ^^^^^^^^^^^^^^^^^
...
51  |     pub async fn route_query(
    |                  ^^^^^^^^^^^
...
83  |     fn aggregate_intents(&self, intents: Vec<Intent>) -> Vec<Intent> {
    |        ^^^^^^^^^^^^^^^^^
...
121 |     fn get_handler_name(&self, intent_type: &IntentType) -> String {
    |        ^^^^^^^^^^^^^^^^
...
136 |     pub async fn execute_handler(
    |                  ^^^^^^^^^^^^^^^

warning: struct `IntentRoute` is never constructed
   --> src\ml\intent_router.rs:157:12
    |
157 | pub struct IntentRoute {
    |            ^^^^^^^^^^^

warning: struct `HandlerResponse` is never constructed
   --> src\ml\intent_router.rs:166:12
    |
166 | pub struct HandlerResponse {
    |            ^^^^^^^^^^^^^^^

warning: trait `IntentClassifier` is never used
   --> src\ml\intent_router.rs:175:11
    |
175 | pub trait IntentClassifier: Send + Sync {
    |           ^^^^^^^^^^^^^^^^

warning: trait `IntentHandler` is never used
   --> src\ml\intent_router.rs:181:11
    |
181 | pub trait IntentHandler: Send + Sync {
    |           ^^^^^^^^^^^^^

warning: struct `RuleBasedClassifier` is never constructed
   --> src\ml\intent_router.rs:191:12
    |
191 | pub struct RuleBasedClassifier {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated functions `new` and `initialize_rules` are never used
   --> src\ml\intent_router.rs:196:12
    |
195 | impl RuleBasedClassifier {
    | ------------------------ associated functions in this implementation
196 |     pub fn new() -> Self {
    |            ^^^
...
202 |     fn initialize_rules() -> Vec<IntentRule> {
    |        ^^^^^^^^^^^^^^^^

warning: struct `KeywordClassifier` is never constructed
   --> src\ml\intent_router.rs:294:12
    |
294 | pub struct KeywordClassifier {
    |            ^^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\ml\intent_router.rs:299:12
    |
298 | impl KeywordClassifier {
    | ---------------------- associated function in this implementation
299 |     pub fn new() -> Self {
    |            ^^^

warning: struct `PatternClassifier` is never constructed
   --> src\ml\intent_router.rs:359:12
    |
359 | pub struct PatternClassifier;
    |            ^^^^^^^^^^^^^^^^^

warning: associated function `new` is never used
   --> src\ml\intent_router.rs:362:12
    |
361 | impl PatternClassifier {
    | ---------------------- associated function in this implementation
362 |     pub fn new() -> Self {
    |            ^^^

warning: struct `IntentRule` is never constructed
   --> src\ml\intent_router.rs:415:8
    |
415 | struct IntentRule {
    |        ^^^^^^^^^^

warning: struct `ViolationHandler` is never constructed
   --> src\ml\intent_router.rs:423:8
    |
423 | struct ViolationHandler;
    |        ^^^^^^^^^^^^^^^^

warning: struct `PredictionHandler` is never constructed
   --> src\ml\intent_router.rs:446:8
    |
446 | struct PredictionHandler;
    |        ^^^^^^^^^^^^^^^^^

warning: struct `RemediationHandler` is never constructed
   --> src\ml\intent_router.rs:474:8
    |
474 | struct RemediationHandler;
    |        ^^^^^^^^^^^^^^^^^^

warning: struct `PolicyHandler` is never constructed
   --> src\ml\intent_router.rs:499:8
    |
499 | struct PolicyHandler;
    |        ^^^^^^^^^^^^^

warning: struct `ReportHandler` is never constructed
   --> src\ml\intent_router.rs:524:8
    |
524 | struct ReportHandler;
    |        ^^^^^^^^^^^^^

warning: struct `CostHandler` is never constructed
   --> src\ml\intent_router.rs:549:8
    |
549 | struct CostHandler;
    |        ^^^^^^^^^^^

warning: struct `SecurityHandler` is never constructed
   --> src\ml\intent_router.rs:574:8
    |
574 | struct SecurityHandler;
    |        ^^^^^^^^^^^^^^^

warning: struct `PolicyGenerator` is never constructed
  --> src\ml\policy_generator.rs:18:12
   |
18 | pub struct PolicyGenerator {
   |            ^^^^^^^^^^^^^^^

warning: associated items `new`, `generate_from_nl`, `enhance_policy`, `validate_policy`, `generate_explanation`, and `generate_warnings` are never used
   --> src\ml\policy_generator.rs:25:12
    |
24  | impl PolicyGenerator {
    | -------------------- associated items in this implementation
25  |     pub fn new() -> Self {
    |            ^^^
...
34  |     pub async fn generate_from_nl(&self, description: &str) -> Result<GeneratedPolicy, String> {
    |                  ^^^^^^^^^^^^^^^^
...
61  |     fn enhance_policy(&self, mut policy: Value, requirements: &ParsedRequirements) -> Value {
    |        ^^^^^^^^^^^^^^
...
88  |     fn validate_policy(&self, policy: &Value) -> Result<(), String> {
    |        ^^^^^^^^^^^^^^^
...
105 |     fn generate_explanation(&self, requirements: &ParsedRequirements) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^
...
118 |     fn generate_warnings(&self, requirements: &ParsedRequirements) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^

warning: struct `ParsedRequirements` is never constructed
   --> src\ml\policy_generator.rs:139:12
    |
139 | pub struct ParsedRequirements {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `Condition` is never constructed
   --> src\ml\policy_generator.rs:151:12
    |
151 | pub struct Condition {
    |            ^^^^^^^^^

warning: struct `PolicyParameter` is never constructed
   --> src\ml\policy_generator.rs:160:12
    |
160 | pub struct PolicyParameter {
    |            ^^^^^^^^^^^^^^^

warning: struct `RequirementParser` is never constructed
   --> src\ml\policy_generator.rs:169:12
    |
169 | pub struct RequirementParser {
    |            ^^^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_patterns`, `parse`, `extract_parameters`, and `calculate_confidence` are never used
   --> src\ml\policy_generator.rs:176:12
    |
175 | impl RequirementParser {
    | ---------------------- associated items in this implementation
176 |     pub fn new() -> Self {
    |            ^^^
...
187 |     fn initialize_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^
...
236 |     pub fn parse(&self, description: &str) -> Result<ParsedRequirements, String> {
    |            ^^^^^
...
281 |     fn extract_parameters(&self, description: &str) -> Vec<PolicyParameter> {
    |        ^^^^^^^^^^^^^^^^^^
...
302 |     fn calculate_confidence(&self, resource_types: &[String], conditions: &[Condition]) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^

warning: struct `ConditionPattern` is never constructed
   --> src\ml\policy_generator.rs:322:8
    |
322 | struct ConditionPattern {
    |        ^^^^^^^^^^^^^^^^

warning: struct `PolicyTemplateLibrary` is never constructed
   --> src\ml\policy_generator.rs:328:12
    |
328 | pub struct PolicyTemplateLibrary {
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `initialize_templates`, and `find_matching_template` are never used
   --> src\ml\policy_generator.rs:333:12
    |
332 | impl PolicyTemplateLibrary {
    | -------------------------- associated items in this implementation
333 |     pub fn new() -> Self {
    |            ^^^
...
339 |     fn initialize_templates() -> Vec<PolicyTemplate> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
408 |     pub fn find_matching_template(&self, requirements: &ParsedRequirements) -> Option<&PolicyTemplate> {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: struct `PolicyTemplate` is never constructed
   --> src\ml\policy_generator.rs:421:12
    |
421 | pub struct PolicyTemplate {
    |            ^^^^^^^^^^^^^^

warning: method `to_policy` is never used
   --> src\ml\policy_generator.rs:428:12
    |
427 | impl PolicyTemplate {
    | ------------------- method in this implementation
428 |     pub fn to_policy(&self) -> Value {
    |            ^^^^^^^^^

warning: struct `PolicyBuilder` is never constructed
   --> src\ml\policy_generator.rs:434:12
    |
434 | pub struct PolicyBuilder {
    |            ^^^^^^^^^^^^^

warning: associated items `new`, `build_custom`, and `build_if_conditions` are never used
   --> src\ml\policy_generator.rs:439:12
    |
438 | impl PolicyBuilder {
    | ------------------ associated items in this implementation
439 |     pub fn new() -> Self {
    |            ^^^
...
443 |     pub fn build_custom(&self, requirements: &ParsedRequirements) -> Result<Value, String> {
    |            ^^^^^^^^^^^^
...
471 |     fn build_if_conditions(&self, requirements: &ParsedRequirements) -> Value {
    |        ^^^^^^^^^^^^^^^^^^^

warning: struct `GeneratedPolicy` is never constructed
   --> src\ml\policy_generator.rs:528:12
    |
528 | pub struct GeneratedPolicy {
    |            ^^^^^^^^^^^^^^^

warning: struct `PolicyValidation` is never constructed
   --> src\ml\policy_generator.rs:537:12
    |
537 | pub struct PolicyValidation {
    |            ^^^^^^^^^^^^^^^^

warning: method `validate_policy_definition` is never used
   --> src\ml\policy_generator.rs:546:18
    |
544 | impl PolicyGenerator {
    | -------------------- method in this implementation
545 |     /// Validate an existing policy definition
546 |     pub async fn validate_policy_definition(&self, policy: &Value) -> PolicyValidation {
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `EntityExtractor` is never constructed
  --> src\ml\entity_extractor.rs:47:12
   |
47 | pub struct EntityExtractor {
   |            ^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\entity_extractor.rs:55:12
    |
54  | impl EntityExtractor {
    | -------------------- associated items in this implementation
55  |     pub fn new() -> Self {
    |            ^^^
...
67  |     fn initialize_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^
...
178 |     pub fn extract_entities(&self, query: &str) -> ExtractionResult {
    |            ^^^^^^^^^^^^^^^^
...
222 |     fn extract_named_entities(&self, query: &str) -> Vec<Entity> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
259 |     fn calculate_confidence(&self, entity_type: &EntityType, value: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
282 |     fn deduplicate_entities(&self, entities: Vec<Entity>) -> Vec<Entity> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
307 |     fn normalize_query(&self, query: &str, entities: &[Entity]) -> String {
    |        ^^^^^^^^^^^^^^^
...
319 |     fn determine_domain(&self, entities: &[Entity]) -> String {
    |        ^^^^^^^^^^^^^^^^
...
346 |     fn generate_intent_hints(&self, entities: &[Entity]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
373 |     fn entity_type_to_string(&self, entity_type: &EntityType) -> String {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
390 |     fn initialize_azure_services() -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
412 |     fn initialize_resource_types() -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
427 |     fn initialize_policies() -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^

warning: struct `QueryUnderstandingEngine` is never constructed
   --> src\ml\query_understanding.rs:153:12
    |
153 | pub struct QueryUnderstandingEngine {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\ml\query_understanding.rs:160:12
    |
159 | impl QueryUnderstandingEngine {
    | ----------------------------- associated items in this implementation
160 |     pub fn new() -> Self {
    |            ^^^
...
172 |     pub fn understand_query(&self, query: &str) -> QueryUnderstanding {
    |            ^^^^^^^^^^^^^^^^
...
197 |     fn classify_intent(&self, query: &str, entities: &ExtractionResult) -> Intent {
    |        ^^^^^^^^^^^^^^^
...
258 |     fn parse_semantics(&self, query: &str, entities: &ExtractionResult) -> SemanticParse {
    |        ^^^^^^^^^^^^^^^
...
340 |     fn generate_execution_plan(&self, intent: &Intent, entities: &ExtractionResult, semantic_parse: &SemanticParse) -> Executi...
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
464 |     fn build_query_parameters(&self, entities: &ExtractionResult, semantic_parse: &SemanticParse) -> HashMap<String, serde_jso...
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
489 |     fn calculate_confidence(&self, intent: &Intent, entities: &ExtractionResult, _semantic_parse: &SemanticParse) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^
...
496 |     fn initialize_intent_patterns(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
534 |     fn initialize_domain_keywords(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: field `0` is never read
  --> src\observability.rs:21:26
   |
21 | pub struct CorrelationId(pub String);
   |            ------------- ^^^^^^^^^^
   |            |
   |            field in this struct
   |
   = help: consider removing this field
   = note: `CorrelationId` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis

warning: trait `PolicyEngine` is never used
   --> src\policy_engine.rs:146:11
    |
146 | pub trait PolicyEngine: Send + Sync {
    |           ^^^^^^^^^^^^

warning: struct `DefaultPolicyEngine` is never constructed
   --> src\policy_engine.rs:189:12
    |
189 | pub struct DefaultPolicyEngine {
    |            ^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `evaluate_condition`, `get_field_value`, and `navigate_json_path` are never used
   --> src\policy_engine.rs:195:12
    |
194 | impl DefaultPolicyEngine {
    | ------------------------ associated items in this implementation
195 |     pub fn new(action_executor: Box<dyn ActionExecutor>) -> Self {
    |            ^^^
...
203 |     fn evaluate_condition(&self, condition: &Condition, resource: &Resource) -> bool {
    |        ^^^^^^^^^^^^^^^^^^
...
274 |     fn get_field_value(&self, resource: &Resource, field: &str) -> Option<Value> {
    |        ^^^^^^^^^^^^^^^
...
305 |     fn navigate_json_path(&self, data: &HashMap<String, Value>, path: &[&str]) -> Option<Value> {
    |        ^^^^^^^^^^^^^^^^^^

warning: trait `ActionExecutor` is never used
   --> src\policy_engine.rs:451:11
    |
451 | pub trait ActionExecutor: Send + Sync {
    |           ^^^^^^^^^^^^^^

warning: struct `DefaultActionExecutor` is never constructed
   --> src\policy_engine.rs:461:12
    |
461 | pub struct DefaultActionExecutor;
    |            ^^^^^^^^^^^^^^^^^^^^^

warning: associated functions `free`, `pro`, and `enterprise` are never used
  --> src\quota_middleware.rs:28:12
   |
27 | impl TenantPlan {
   | --------------- associated functions in this implementation
28 |     pub fn free() -> Self {
   |            ^^^^
...
39 |     pub fn pro() -> Self {
   |            ^^^
...
50 |     pub fn enterprise() -> Self {
   |            ^^^^^^^^^^

warning: struct `QuotaManager` is never constructed
  --> src\quota_middleware.rs:63:12
   |
63 | pub struct QuotaManager {
   |            ^^^^^^^^^^^^

warning: associated items `new`, `check_rate_limit`, and `get_tenant_plan` are never used
   --> src\quota_middleware.rs:69:18
    |
68  | impl QuotaManager {
    | ----------------- associated items in this implementation
69  |     pub async fn new(redis_url: &str) -> Result<Self, redis::RedisError> {
    |                  ^^^
...
84  |     pub async fn check_rate_limit(
    |                  ^^^^^^^^^^^^^^^^
...
167 |     pub async fn get_tenant_plan(&self, tenant_id: &str) -> TenantPlan {
    |                  ^^^^^^^^^^^^^^^

warning: struct `RateLimitResult` is never constructed
   --> src\quota_middleware.rs:187:12
    |
187 | pub struct RateLimitResult {
    |            ^^^^^^^^^^^^^^^

warning: struct `QuotaExceededResponse` is never constructed
   --> src\quota_middleware.rs:196:8
    |
196 | struct QuotaExceededResponse {
    |        ^^^^^^^^^^^^^^^^^^^^^

warning: function `quota_middleware` is never used
   --> src\quota_middleware.rs:203:14
    |
203 | pub async fn quota_middleware(
    |              ^^^^^^^^^^^^^^^^

warning: function `extract_tenant_id` is never used
   --> src\quota_middleware.rs:287:4
    |
287 | fn extract_tenant_id(request: &Request) -> Option<String> {
    |    ^^^^^^^^^^^^^^^^^

warning: function `increment_request_counter` is never used
   --> src\quota_middleware.rs:318:10
    |
318 | async fn increment_request_counter(tenant_id: &str, plan_tier: &str, allowed: bool) {
    |          ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `UsageTracker` is never constructed
   --> src\quota_middleware.rs:338:12
    |
338 | pub struct UsageTracker {
    |            ^^^^^^^^^^^^

warning: associated items `new`, `track_usage`, and `flush_events` are never used
   --> src\quota_middleware.rs:344:18
    |
343 | impl UsageTracker {
    | ----------------- associated items in this implementation
344 |     pub async fn new(redis_url: &str, event_grid_endpoint: String) -> Result<Self, redis::RedisError> {
    |                  ^^^
...
354 |     pub async fn track_usage(&self, event: UsageEvent) -> Result<(), Box<dyn std::error::Error>> {
    |                  ^^^^^^^^^^^
...
371 |     async fn flush_events(&self) -> Result<(), Box<dyn std::error::Error>> {
    |              ^^^^^^^^^^^^

warning: fields `execution_id`, `started_at`, `pending_approvals`, and `checkpoints` are never read
  --> src\remediation\workflow_engine.rs:26:5
   |
24 | struct WorkflowExecution {
   |        ----------------- fields in this struct
25 |     workflow_id: Uuid,
26 |     execution_id: Uuid,
   |     ^^^^^^^^^^^^
...
29 |     started_at: DateTime<Utc>,
   |     ^^^^^^^^^^
30 |     completed_steps: Vec<CompletedStep>,
31 |     pending_approvals: Vec<String>,
   |     ^^^^^^^^^^^^^^^^^
32 |     checkpoints: Vec<Checkpoint>,
   |     ^^^^^^^^^^^
   |
   = note: `WorkflowExecution` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: associated function `internal_server` is never used
   --> src\remediation\quick_fixes.rs:100:12
    |
99  | impl ApiError {
    | ------------- associated function in this implementation
100 |     pub fn internal_server(message: String) -> Self {
    |            ^^^^^^^^^^^^^^^

warning: methods `get_definition`, `get_by_category`, and `get_all_definitions` are never used
   --> src\resources\categories.rs:568:12
    |
38  | impl ResourceCatalog {
    | -------------------- methods in this implementation
...
568 |     pub fn get_definition(&self, resource_type: &str) -> Option<&ResourceDefinition> {
    |            ^^^^^^^^^^^^^^
...
572 |     pub fn get_by_category(&self, category: ResourceCategory) -> Vec<&ResourceDefinition> {
    |            ^^^^^^^^^^^^^^^
...
579 |     pub fn get_all_definitions(&self) -> Vec<&ResourceDefinition> {
    |            ^^^^^^^^^^^^^^^^^^^

warning: trait `ResourceDiscovery` is never used
  --> src\resources\discovery.rs:16:11
   |
16 | pub trait ResourceDiscovery: Send + Sync {
   |           ^^^^^^^^^^^^^^^^^

warning: struct `IntelligentDiscovery` is never constructed
  --> src\resources\discovery.rs:21:12
   |
21 | pub struct IntelligentDiscovery {
   |            ^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\resources\discovery.rs:27:12
    |
26  | impl IntelligentDiscovery {
    | ------------------------- associated items in this implementation
27  |     pub fn new(azure_client: Arc<AzureClient>) -> Self {
    |            ^^^
...
37  |     fn register_discoverers(&mut self) {
    |        ^^^^^^^^^^^^^^^^^^^^
...
45  |     pub async fn discover_all(&self) -> Result<Vec<AzureResource>, Box<dyn std::error::Error + Send + Sync>> {
    |                  ^^^^^^^^^^^^
...
73  |     async fn apply_intelligence(&self, resources: &mut Vec<AzureResource>) {
    |              ^^^^^^^^^^^^^^^^^^
...
91  |     async fn predict_issues(&self, resource: &mut AzureResource) {
    |              ^^^^^^^^^^^^^^
...
105 |     async fn generate_recommendations(&self, resource: &mut AzureResource) {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^
...
126 |     async fn calculate_optimization_potential(&self, resource: &mut AzureResource) {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
136 |     async fn assess_compliance(&self, resource: &mut AzureResource) {
    |              ^^^^^^^^^^^^^^^^^

warning: struct `PolicyDiscoverer` is never constructed
   --> src\resources\discovery.rs:175:8
    |
175 | struct PolicyDiscoverer;
    |        ^^^^^^^^^^^^^^^^

warning: struct `ComputeDiscoverer` is never constructed
   --> src\resources\discovery.rs:193:8
    |
193 | struct ComputeDiscoverer;
    |        ^^^^^^^^^^^^^^^^^

warning: struct `StorageDiscoverer` is never constructed
   --> src\resources\discovery.rs:211:8
    |
211 | struct StorageDiscoverer;
    |        ^^^^^^^^^^^^^^^^^

warning: struct `NetworkDiscoverer` is never constructed
   --> src\resources\discovery.rs:228:8
    |
228 | struct NetworkDiscoverer;
    |        ^^^^^^^^^^^^^^^^^

warning: struct `SecurityDiscoverer` is never constructed
   --> src\resources\discovery.rs:246:8
    |
246 | struct SecurityDiscoverer;
    |        ^^^^^^^^^^^^^^^^^^

warning: struct `CrossDomainCorrelationEngine` is never constructed
  --> src\resources\correlations.rs:73:12
   |
73 | pub struct CrossDomainCorrelationEngine {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\resources\correlations.rs:79:12
    |
78  | impl CrossDomainCorrelationEngine {
    | --------------------------------- associated items in this implementation
79  |     pub fn new(model_registry: Arc<ModelRegistry>) -> Self {
    |            ^^^
...
86  |     pub async fn analyze_correlations(
    |                  ^^^^^^^^^^^^^^^^^^^^
...
111 |     async fn analyze_cost_correlations(&self, resources: &[AzureResource]) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
173 |     async fn analyze_security_correlations(&self, resources: &[AzureResource]) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
265 |     async fn analyze_performance_correlations(&self, resources: &[AzureResource]) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
325 |     async fn analyze_compliance_correlations(&self, resources: &[AzureResource]) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
389 |     async fn analyze_network_correlations(&self, resources: &[AzureResource]) -> Vec<ResourceCorrelation> {
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
439 |     async fn apply_ml_insights(
    |              ^^^^^^^^^^^^^^^^^
...
466 |     async fn detect_anomalies(
    |              ^^^^^^^^^^^^^^^^
...
493 |     fn calculate_cost_correlation(&self, cost_a: &CostData, cost_b: &CostData) -> f32 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^
...
511 |     fn has_security_dependency(&self, _key_vault: &AzureResource, resource: &AzureResource) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^
...
518 |     fn has_network_dependency(&self, network: &AzureResource, resource: &AzureResource) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
524 |     fn find_shared_violations(&self, violations_a: &[ComplianceViolation], violations_b: &[ComplianceViolation]) -> Vec<String> {
    |        ^^^^^^^^^^^^^^^^^^^^^^
...
537 |     fn calculate_performance_impact(&self, affected: &AzureResource, potential_cause: &AzureResource) -> f32 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: struct `SecretGuard` is never constructed
  --> src\secret_guard.rs:16:12
   |
16 | pub struct SecretGuard {
   |            ^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\secret_guard.rs:143:12
    |
142 | impl SecretGuard {
    | ---------------- associated items in this implementation
143 |     pub fn new() -> Self {
    |            ^^^
...
151 |     pub fn with_custom_patterns(mut self, patterns: Vec<SecretPattern>) -> Self {
    |            ^^^^^^^^^^^^^^^^^^^^
...
157 |     pub fn scan(&self, text: &str, context: &str) -> Vec<SecretViolation> {
    |            ^^^^
...
187 |     pub fn redact(&self, text: &str) -> String {
    |            ^^^^^^
...
206 |     pub fn redact_json(&self, json: &serde_json::Value) -> serde_json::Value {
    |            ^^^^^^^^^^^
...
231 |     fn is_secret_field(&self, field_name: &str) -> bool {
    |        ^^^^^^^^^^^^^^^
...
254 |     fn exceeds_entropy_threshold(&self, text: &str, threshold: f64) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
258 |     fn calculate_shannon_entropy(&self, text: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
...
275 |     fn get_context_snippet(&self, text: &str, start: usize, end: usize) -> String {
    |        ^^^^^^^^^^^^^^^^^^^

warning: struct `SecretRedactionMiddleware` is never constructed
   --> src\secret_guard.rs:290:12
    |
290 | pub struct SecretRedactionMiddleware {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `redact_headers`, and `redact_body` are never used
   --> src\secret_guard.rs:295:12
    |
294 | impl SecretRedactionMiddleware {
    | ------------------------------ associated items in this implementation
295 |     pub fn new() -> Self {
    |            ^^^
...
301 |     pub fn redact_headers(&self, headers: &mut axum::http::HeaderMap) {
    |            ^^^^^^^^^^^^^^
...
314 |     pub fn redact_body(&self, body: &str) -> String {
    |            ^^^^^^^^^^^

warning: struct `StaticSecretAnalyzer` is never constructed
   --> src\secret_guard.rs:320:12
    |
320 | pub struct StaticSecretAnalyzer {
    |            ^^^^^^^^^^^^^^^^^^^^

warning: associated items `new`, `scan_directory`, and `generate_report` are never used
   --> src\secret_guard.rs:326:12
    |
325 | impl StaticSecretAnalyzer {
    | ------------------------- associated items in this implementation
326 |     pub fn new() -> Self {
    |            ^^^
...
339 |     pub async fn scan_directory(&self, path: &std::path::Path) -> Vec<SecretViolation> {
    |                  ^^^^^^^^^^^^^^
...
370 |     pub fn generate_report(&self, violations: &[SecretViolation]) -> String {
    |            ^^^^^^^^^^^^^^^

warning: methods `set_secret`, `delete_secret`, `list_secrets`, `rotate_secret`, and `get_all_app_secrets` are never used
   --> src\secrets.rs:133:18
    |
46  | impl SecretsManager {
    | ------------------- methods in this implementation
...
133 |     pub async fn set_secret(
    |                  ^^^^^^^^^^
...
153 |     pub async fn delete_secret(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    |                  ^^^^^^^^^^^^^
...
169 |     pub async fn list_secrets(&self) -> Result<Vec<SecretMetadata>, Box<dyn std::error::Error>> {
    |                  ^^^^^^^^^^^^
...
198 |     pub async fn rotate_secret(
    |                  ^^^^^^^^^^^^^
...
213 |     pub async fn get_all_app_secrets(&self) -> HashMap<String, String> {
    |                  ^^^^^^^^^^^^^^^^^^^

warning: struct `SecretScanner` is never constructed
   --> src\secrets.rs:281:12
    |
281 | pub struct SecretScanner {
    |            ^^^^^^^^^^^^^

warning: struct `SecretPattern` is never constructed
   --> src\secrets.rs:286:8
    |
286 | struct SecretPattern {
    |        ^^^^^^^^^^^^^

warning: associated items `new`, `scan`, and `calculate_entropy` are never used
   --> src\secrets.rs:293:12
    |
292 | impl SecretScanner {
    | ------------------ associated items in this implementation
293 |     pub fn new() -> Self {
    |            ^^^
...
331 |     pub fn scan(&self, text: &str) -> Vec<SecretDetection> {
    |            ^^^^
...
357 |     fn calculate_entropy(s: &str) -> f64 {
    |        ^^^^^^^^^^^^^^^^^

warning: struct `SecretDetection` is never constructed
   --> src\secrets.rs:376:12
    |
376 | pub struct SecretDetection {
    |            ^^^^^^^^^^^^^^^

warning: enum `SecretSeverity` is never used
   --> src\secrets.rs:384:10
    |
384 | pub enum SecretSeverity {
    |          ^^^^^^^^^^^^^^

warning: struct `PathScoringWeights` is never constructed
   --> src\security_graph\mod.rs:122:12
    |
122 | pub struct PathScoringWeights {
    |            ^^^^^^^^^^^^^^^^^^

warning: struct `SecurityGraphEngine` is never constructed
   --> src\security_graph\mod.rs:141:12
    |
141 | pub struct SecurityGraphEngine {
    |            ^^^^^^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\security_graph\mod.rs:148:12
    |
147 | impl SecurityGraphEngine {
    | ------------------------ associated items in this implementation
148 |     pub fn new() -> Self {
    |            ^^^
...
157 |     pub async fn build_from_azure(
    |                  ^^^^^^^^^^^^^^^^
...
182 |     async fn add_identities(
    |              ^^^^^^^^^^^^^^
...
252 |     async fn add_roles_and_permissions(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
...
316 |     async fn add_resources(
    |              ^^^^^^^^^^^^^
...
402 |     async fn add_network_topology(
    |              ^^^^^^^^^^^^^^^^^^^^
...
510 |     async fn add_data_stores(
    |              ^^^^^^^^^^^^^^^
...
581 |     async fn build_relationships(
    |              ^^^^^^^^^^^^^^^^^^^
...
700 |     pub fn find_attack_paths(&self, target_sensitivity: &str) -> Vec<AttackPath> {
    |            ^^^^^^^^^^^^^^^^^
...
752 |     fn score_and_build_path(&self, node_indices: Vec<NodeIndex>) -> AttackPath {
    |        ^^^^^^^^^^^^^^^^^^^^
...
845 |     fn generate_mitigations(&self, _path: &[NodeIndex]) -> Vec<MitigationBundle> {
    |        ^^^^^^^^^^^^^^^^^^^^
...
934 |     pub async fn apply_mitigation(
    |                  ^^^^^^^^^^^^^^^^
...
961 |     async fn apply_control(
    |              ^^^^^^^^^^^^^
...
976 |     fn calculate_residual_risk(&self, bundle: &MitigationBundle) -> f64 {
    |        ^^^^^^^^^^^^^^^^^^^^^^^

warning: enum `SecurityGraphError` is never used
   --> src\security_graph\mod.rs:992:10
    |
992 | pub enum SecurityGraphError {
    |          ^^^^^^^^^^^^^^^^^^

warning: function `analyze_security_exposure` is never used
    --> src\security_graph\mod.rs:1002:14
     |
1002 | pub async fn analyze_security_exposure(
     |              ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: function `generate_prioritized_mitigations` is never used
    --> src\security_graph\mod.rs:1040:4
     |
1040 | fn generate_prioritized_mitigations(paths: &[AttackPath]) -> Vec<MitigationBundle> {
     |    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: associated functions `get_resources`, `get_rbac_data`, and `get_cost_data` are never used
   --> src\simulated_data.rs:103:12
    |
16  | impl SimulatedDataProvider {
    | -------------------------- associated functions in this implementation
...
103 |     pub fn get_resources() -> Vec<Resource> {
    |            ^^^^^^^^^^^^^
...
160 |     pub fn get_rbac_data() -> RbacData {
    |            ^^^^^^^^^^^^^
...
201 |     pub fn get_cost_data() -> CostData {
    |            ^^^^^^^^^^^^^

warning: associated items `new`, `update`, `is_exhausted`, and `consumption_percentage` are never used
   --> src\slo.rs:94:12
    |
93  | impl ErrorBudget {
    | ---------------- associated items in this implementation
94  |     pub fn new(slo_target: f64, window_days: u32) -> Self {
    |            ^^^
...
106 |     pub fn update(&mut self, error_rate: f64, time_elapsed: Duration) {
    |            ^^^^^^
...
127 |     pub fn is_exhausted(&self) -> bool {
    |            ^^^^^^^^^^^^
...
131 |     pub fn consumption_percentage(&self) -> f64 {
    |            ^^^^^^^^^^^^^^^^^^^^^^

warning: methods `success_rate` and `error_rate` are never used
   --> src\slo.rs:166:12
    |
165 | impl SLOMeasurement {
    | ------------------- methods in this implementation
166 |     pub fn success_rate(&self) -> f64 {
    |            ^^^^^^^^^^^^
...
174 |     pub fn error_rate(&self) -> f64 {
    |            ^^^^^^^^^^

warning: fields `slos`, `measurements`, and `status_cache` are never read
   --> src\slo.rs:205:5
    |
204 | pub struct SLOManager {
    |            ---------- fields in this struct
205 |     slos: Arc<RwLock<HashMap<String, SLO>>>,
    |     ^^^^
206 |     measurements: Arc<RwLock<HashMap<String, Vec<SLOMeasurement>>>>,
    |     ^^^^^^^^^^^^
207 |     status_cache: Arc<RwLock<HashMap<String, SLOStatus>>>,
    |     ^^^^^^^^^^^^

warning: multiple methods are never used
   --> src\slo.rs:219:18
    |
210 | impl SLOManager {
    | --------------- methods in this implementation
...
219 |     pub async fn create_slo(&self, slo: SLO) -> Result<String, String> {
    |                  ^^^^^^^^^^
...
231 |     pub async fn record_measurement(
    |                  ^^^^^^^^^^^^^^^^^^
...
255 |     async fn update_status(&self, slo_id: &str) -> Result<(), String> {
    |              ^^^^^^^^^^^^^
...
310 |     fn get_window_start(&self, window: &SLOWindow) -> DateTime<Utc> {
    |        ^^^^^^^^^^^^^^^^
...
381 |     fn calculate_sli(&self, sli: &SLI, measurements: &[SLOMeasurement]) -> f64 {
    |        ^^^^^^^^^^^^^
...
436 |     fn check_burn_rate(&self, budget: &ErrorBudget, window: &SLOWindow) -> Option<BurnRateAlert> {
    |        ^^^^^^^^^^^^^^^
...
483 |     async fn check_alerts(&self, slo_id: &str, status: &SLOStatus) {
    |              ^^^^^^^^^^^^
...
511 |     async fn send_alert(&self, slo: &SLO, alert: &SLOAlert, status: &SLOStatus) {
    |              ^^^^^^^^^^
...
528 |     pub async fn get_status(&self, slo_id: &str) -> Option<SLOStatus> {
    |                  ^^^^^^^^^^
...
533 |     pub async fn get_all_status(&self) -> Vec<SLOStatus> {
    |                  ^^^^^^^^^^^^^^
...
538 |     pub async fn should_block_release(&self, critical_slos: &[String]) -> bool {
    |                  ^^^^^^^^^^^^^^^^^^^^

warning: method `severity_string` is never used
   --> src\slo.rs:562:8
    |
561 | impl SLOAlert {
    | ------------- method in this implementation
562 |     fn severity_string(&self) -> &str {
    |        ^^^^^^^^^^^^^^^

warning: associated functions `from_token_claims` and `default` are never used
  --> src\tenant.rs:28:12
   |
27 | impl TenantContext {
   | ------------------ associated functions in this implementation
28 |     pub fn from_token_claims(claims: &crate::auth::Claims) -> Self {
   |            ^^^^^^^^^^^^^^^^^
...
40 |     pub fn default() -> Self {
   |            ^^^^^^^

warning: struct `TenantExtension` is never constructed
  --> src\tenant.rs:52:12
   |
52 | pub struct TenantExtension(pub TenantContext);
   |            ^^^^^^^^^^^^^^^

warning: function `tenant_middleware` is never used
  --> src\tenant.rs:55:14
   |
55 | pub async fn tenant_middleware(
   |              ^^^^^^^^^^^^^^^^^

warning: function `get_tenant` is never used
   --> src\tenant.rs:104:8
    |
104 | pub fn get_tenant(request: &Request) -> Option<TenantContext> {
    |        ^^^^^^^^^^

warning: struct `TenantAwareDb` is never constructed
   --> src\tenant.rs:112:12
    |
112 | pub struct TenantAwareDb {
    |            ^^^^^^^^^^^^^

warning: associated items `new`, `set_tenant_context`, `query_policies`, and `query_resources` are never used
   --> src\tenant.rs:118:12
    |
117 | impl TenantAwareDb {
    | ------------------ associated items in this implementation
118 |     pub fn new(pool: sqlx::PgPool, tenant_id: String) -> Self {
    |            ^^^
...
122 |     pub async fn set_tenant_context(&self) -> Result<(), sqlx::Error> {
    |                  ^^^^^^^^^^^^^^^^^^
...
130 |     pub async fn query_policies(&self) -> Result<Vec<Policy>, sqlx::Error> {
    |                  ^^^^^^^^^^^^^^
...
147 |     pub async fn query_resources(&self) -> Result<Vec<Resource>, sqlx::Error> {
    |                  ^^^^^^^^^^^^^^^

warning: associated items `from_claims`, `can_access_tenant`, and `can_access_resource` are never used
  --> src\tenant_isolation.rs:34:12
   |
32 | impl TenantContext {
   | ------------------ associated items in this implementation
33 |     /// Create a new tenant context from JWT claims
34 |     pub fn from_claims(claims: &serde_json::Value) -> Result<Self, String> {
   |            ^^^^^^^^^^^
...
69 |     pub fn can_access_tenant(&self, target_tenant_id: &Uuid) -> bool {
   |            ^^^^^^^^^^^^^^^^^
...
74 |     pub fn can_access_resource(&self, resource_tenant_id: &Uuid) -> bool {
   |            ^^^^^^^^^^^^^^^^^^^

warning: function `tenant_isolation_middleware` is never used
  --> src\tenant_isolation.rs:80:14
   |
80 | pub async fn tenant_isolation_middleware(
   |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

warning: trait `TenantFilter` is never used
  --> src\tenant_isolation.rs:96:11
   |
96 | pub trait TenantFilter {
   |           ^^^^^^^^^^^^

warning: associated items `new` and `belongs_to` are never used
   --> src\tenant_isolation.rs:117:12
    |
116 | impl<T> TenantResource<T> {
    | ------------------------- associated items in this implementation
117 |     pub fn new(tenant_id: Uuid, resource: T) -> Self {
    |            ^^^
...
125 |     pub fn belongs_to(&self, tenant: &TenantContext) -> bool {
    |            ^^^^^^^^^^

warning: struct `TenantDatabase` is never constructed
   --> src\tenant_isolation.rs:131:12
    |
131 | pub struct TenantDatabase {
    |            ^^^^^^^^^^^^^^

warning: multiple associated items are never used
   --> src\tenant_isolation.rs:136:12
    |
135 | impl TenantDatabase {
    | ------------------- associated items in this implementation
136 |     pub fn new(pool: Arc<PgPool>) -> Self {
    |            ^^^
...
141 |     pub async fn get_resources(
    |                  ^^^^^^^^^^^^^
...
177 |     pub async fn create_resource(
    |                  ^^^^^^^^^^^^^^^
...
206 |     pub async fn update_resource(
    |                  ^^^^^^^^^^^^^^^
...
247 |     pub async fn delete_resource(
    |                  ^^^^^^^^^^^^^^^
...
271 |     pub async fn get_policies(
    |                  ^^^^^^^^^^^^
...
302 |     pub async fn get_compliance(
    |                  ^^^^^^^^^^^^^^

warning: function `audit_log` is never used
   --> src\tenant_isolation.rs:352:14
    |
352 | pub async fn audit_log(
    |              ^^^^^^^^^

warning: associated function `for_critical_operations` is never used
  --> src\utils\retry.rs:34:12
   |
24 | impl RetryConfig {
   | ---------------- associated function in this implementation
...
34 |     pub fn for_critical_operations() -> Self {
   |            ^^^^^^^^^^^^^^^^^^^^^^^

warning: function `is_azure_throttled_error` is never used
   --> src\utils\retry.rs:101:8
    |
101 | pub fn is_azure_throttled_error(error_message: &str) -> bool {
    |        ^^^^^^^^^^^^^^^^^^^^^^^^

warning: `policycortex-core` (bin "policycortex-core") generated 946 warnings (234 duplicates)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.89s
warning: the following packages contain code that will be rejected by a future version of Rust: redis v0.24.0, sqlx-postgres v0.7.4
note: to see what the problems were, use the option `--future-incompat-report`, or run `cargo report future-incompatibilities --id 3`
     Running `target\debug\policycortex-core.exe`
2025-09-02T20:51:10.770739Z  INFO policycortex_core: Starting PolicyCortex Core Service
2025-09-02T20:51:10.770981Z  INFO policycortex_core: Patents: Unified AI Platform | Predictive Compliance | Conversational Intelligence | Cross-Domain Correlation
2025-09-02T20:51:31.585113Z  WARN policycortex_core: ⚠️ Failed to initialize async Azure client: No connection could be made because the target machine actively refused it. (os error 10061)
2025-09-02T20:51:31.585554Z  INFO policycortex_core::azure_client: Initializing Azure client for subscription: 205b477d-17e7-4b3b-92c1-32cf02626b78
2025-09-02T20:51:31.586250Z  INFO policycortex_core: ✅ Fallback Azure client initialized
2025-09-02T20:51:31.588268Z  INFO policycortex_core::secrets: ✅ Connected to Azure Key Vault: https://policycortex-kv.vault.azure.net/
2025-09-02T20:52:53.787233Z  WARN policycortex_core::secrets: Failed to get secret 'DATABASE_URL' from Key Vault: retry policy expired and the request will no longer be retried
2025-09-02T20:53:23.800478Z  WARN policycortex_core: DB pool connection failed: pool timed out while waiting for an open connection
2025-09-02T20:53:24.007277Z  INFO policycortex_core: PolicyCortex Core API listening on 0.0.0.0:8080
error: process didn't exit successfully: `target\debug\policycortex-core.exe` (exit code: 0xc000013a, STATUS_CONTROL_C_EXIT)
PS C:\Users\leona\Documents\policycortex\core> ^C
PS C:\Users\leona\Documents\policycortex\core> cargo fix --lib -p policycortex-core
error: the working directory of this package has uncommitted changes, and `cargo fix` can potentially perform destructive changes; if you'd like to suppress this error pass `--allow-dirty`, or commit the changes to these files:

  * core/Cargo.lock (dirty)
  * corerunissues.md (dirty)
  * nul (dirty)


PS C:\Users\leona\Documents\policycortex\core> cd ..
PS C:\Users\leona\Documents\policycortex> cargo fix --lib -p policycortex-core
error: the working directory of this package has uncommitted changes, and `cargo fix` can potentially perform destructive changes; if you'd like to suppress this error pass `--allow-dirty`, or commit the changes to these files:

  * core/Cargo.lock (dirty)
  * corerunissues.md (dirty)
  * nul (dirty)


PS C:\Users\leona\Documents\policycortex> cd .\core\
PS C:\Users\leona\Documents\policycortex\core> cargo fix --lib -p policycortex-core
error: the working directory of this package has uncommitted changes, and `cargo fix` can potentially perform destructive changes; if you'd like to suppress this error pass `--allow-dirty`, or commit the changes to these files:

  * core/Cargo.lock (dirty)
  * corerunissues.md (dirty)
  * nul (dirty)


PS C:\Users\leona\Documents\policycortex\core> cargo fix --lib -p policycortex-core --allow-dirty
    Checking policycortex-core v2.24.4 (C:\Users\leona\Documents\policycortex\core)
       Fixed src\azure\security.rs (3 fixes)
       Fixed src\api\quantum.rs (2 fixes)
       Fixed src\azure\client.rs (1 fix)
       Fixed src\azure\activity.rs (2 fixes)
       Fixed src\azure\governance.rs (2 fixes)
       Fixed src\azure\operations.rs (2 fixes)
       Fixed src\azure\resource_graph.rs (4 fixes)
       Fixed src\api\blockchain.rs (2 fixes)
       Fixed src\api\devsecops.rs (2 fixes)
       Fixed src\api\itsm.rs (1 fix)
       Fixed src\api\governance.rs (2 fixes)
       Fixed src\azure\mod.rs (1 fix)
       Fixed src\azure\devops.rs (4 fixes)
       Fixed src\api\executive.rs (1 fix)
       Fixed src\api\finops.rs (2 fixes)
       Fixed src\azure\cost.rs (2 fixes)
       Fixed src\api\edge.rs (1 fix)
       Fixed src\azure\monitor.rs (2 fixes)
       Fixed src\api\health.rs (2 fixes)
       Fixed src\azure\auth.rs (3 fixes)
       Fixed src\api\copilot.rs (1 fix)
warning: unused variable: `request`
   --> src\remediation\approval_manager.rs:355:45
    |
355 |     async fn determine_approval_gate(&self, request: &ApprovalRequest) -> Result<ApprovalGate, String> {
    |                                             ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`
    |
    = note: `#[warn(unused_variables)]` on by default

warning: unused variable: `policy`
   --> src\remediation\approval_manager.rs:357:13
    |
357 |         let policy = policies.get("standard").ok_or("No approval policy found")?;
    |             ^^^^^^ help: if this is intentional, prefix it with an underscore: `_policy`

warning: unused variable: `request`
   --> src\remediation\approval_manager.rs:435:53
    |
435 |     async fn auto_approve(&self, approval_id: &str, request: &ApprovalRequest) -> Result<(), String> {
    |                                                     ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `config`
   --> src\remediation\rollback_manager.rs:385:50
    |
385 |             RollbackAction::RestoreConfiguration(config) => {
    |                                                  ^^^^^^ help: if this is intentional, prefix it with an underscore: `_config`

warning: unused variable: `deployment_result`
   --> src\remediation\arm_executor.rs:187:13
    |
187 |         let deployment_result = DeploymentResult {
    |             ^^^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_deployment_result`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:593:42
    |
593 |     async fn validate_permissions(&self, request: &RemediationRequest, template: &RemediationTemplate) -> Result<ValidationChe...
    |                                          ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `template`
   --> src\remediation\validation_engine.rs:593:72
    |
593 | ...&RemediationRequest, template: &RemediationTemplate) -> Result<ValidationCheck, String> {
    |                         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_template`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:764:50
    |
764 | ...iation_success(&self, request: &RemediationRequest, result: &RemediationResult) -> Result<ValidationCheck, String> {
    |                          ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:805:50
    |
805 |     async fn validate_post_resource_state(&self, request: &RemediationRequest) -> Result<ValidationCheck, String> {
    |                                                  ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:843:46
    |
843 |     async fn validate_post_compliance(&self, request: &RemediationRequest) -> Result<ValidationCheck, String> {
    |                                              ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:881:49
    |
881 |     async fn validate_performance_impact(&self, request: &RemediationRequest) -> Result<ValidationCheck, String> {
    |                                                 ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:927:47
    |
927 |     async fn validate_security_posture(&self, request: &RemediationRequest) -> Result<ValidationCheck, String> {
    |                                               ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `rule`
   --> src\remediation\validation_engine.rs:965:42
    |
965 |     async fn evaluate_safety_rule(&self, rule: &SafetyRule, request: &RemediationRequest, template: &RemediationTemplate) -> R...
    |                                          ^^^^ help: if this is intentional, prefix it with an underscore: `_rule`

warning: unused variable: `request`
   --> src\remediation\validation_engine.rs:965:61
    |
965 | ...f, rule: &SafetyRule, request: &RemediationRequest, template: &RemediationTemplate) -> Result<bool, String> {
    |                          ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `template`
   --> src\remediation\validation_engine.rs:965:91
    |
965 | ...&RemediationRequest, template: &RemediationTemplate) -> Result<bool, String> {
    |                         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_template`

warning: unused variable: `template`
    --> src\remediation\validation_engine.rs:1089:9
     |
1089 |         template: &RemediationTemplate,
     |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_template`

warning: unused variable: `resource_id`
    --> src\remediation\validation_engine.rs:1134:42
     |
1134 |     pub async fn get_dependencies(&self, resource_id: &str) -> Result<Vec<ResourceInfo>, String> {
     |                                          ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resource_id`

warning: unused variable: `state`
  --> src\api\predictions.rs:83:11
   |
83 |     State(state): State<Arc<crate::api::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:736:11
    |
736 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:770:11
    |
770 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:803:11
    |
803 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:896:11
    |
896 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:911:11
    |
911 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:946:11
    |
946 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\itsm.rs:964:11
    |
964 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1021:11
     |
1021 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1057:11
     |
1057 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1079:11
     |
1079 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1111:11
     |
1111 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1126:11
     |
1126 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1164:11
     |
1164 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `id`
    --> src\api\itsm.rs:1165:10
     |
1165 |     Path(id): Path<String>,
     |          ^^ help: if this is intentional, prefix it with an underscore: `_id`

warning: unused variable: `state`
    --> src\api\itsm.rs:1193:11
     |
1193 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `id`
    --> src\api\itsm.rs:1194:10
     |
1194 |     Path(id): Path<String>,
     |          ^^ help: if this is intentional, prefix it with an underscore: `_id`

warning: unused variable: `state`
    --> src\api\itsm.rs:1209:11
     |
1209 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `id`
    --> src\api\itsm.rs:1210:10
     |
1210 |     Path(id): Path<String>,
     |          ^^ help: if this is intentional, prefix it with an underscore: `_id`

warning: unused variable: `state`
    --> src\api\itsm.rs:1234:11
     |
1234 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1260:11
     |
1260 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1286:11
     |
1286 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1314:11
     |
1314 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1342:11
     |
1342 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1367:11
     |
1367 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1392:11
     |
1392 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1430:11
     |
1430 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1458:11
     |
1458 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1486:11
     |
1486 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1500:11
     |
1500 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1538:11
     |
1538 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1560:11
     |
1560 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
    --> src\api\itsm.rs:1588:11
     |
1588 |     State(state): State<Arc<crate::api::AppState>>,
     |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `request`
   --> src\api\finops.rs:157:10
    |
157 |     Json(request): Json<OptimizationRequest>,
    |          ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_request`

warning: unused variable: `state`
   --> src\api\finops.rs:196:11
    |
196 |     State(state): State<Arc<crate::api::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
  --> src\api\executive.rs:85:11
   |
85 |     Query(params): Query<ExecutiveQuery>,
   |           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
  --> src\api\executive.rs:86:11
   |
86 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
   --> src\api\executive.rs:132:11
    |
132 |     Query(params): Query<ExecutiveQuery>,
    |           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
   --> src\api\executive.rs:133:11
    |
133 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\executive.rs:157:11
    |
157 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\executive.rs:197:11
    |
197 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\executive.rs:243:11
    |
243 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\executive.rs:288:11
    |
288 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\executive.rs:302:11
    |
302 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
  --> src\api\quantum.rs:80:11
   |
80 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:123:11
    |
123 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:166:11
    |
166 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:183:11
    |
183 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:225:11
    |
225 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:264:11
    |
264 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:279:11
    |
279 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\quantum.rs:297:11
    |
297 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
  --> src\api\edge.rs:98:11
   |
98 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:170:11
    |
170 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:259:11
    |
259 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:299:11
    |
299 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:329:11
    |
329 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:344:11
    |
344 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\edge.rs:363:11
    |
363 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
  --> src\api\blockchain.rs:75:11
   |
75 |     Query(params): Query<serde_json::Value>,
   |           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
  --> src\api\blockchain.rs:76:11
   |
76 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\blockchain.rs:130:11
    |
130 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\blockchain.rs:147:11
    |
147 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\blockchain.rs:165:11
    |
165 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\blockchain.rs:213:11
    |
213 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `entry`
   --> src\api\blockchain.rs:229:10
    |
229 |     Json(entry): Json<serde_json::Value>,
    |          ^^^^^ help: if this is intentional, prefix it with an underscore: `_entry`

warning: unused variable: `state`
   --> src\api\blockchain.rs:230:11
    |
230 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
   --> src\api\blockchain.rs:248:10
    |
248 |     Json(params): Json<serde_json::Value>,
    |          ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
   --> src\api\blockchain.rs:249:11
    |
249 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
   --> src\api\blockchain.rs:267:11
    |
267 |     Query(params): Query<serde_json::Value>,
    |           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
   --> src\api\blockchain.rs:268:11
    |
268 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
  --> src\api\copilot.rs:82:11
   |
82 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `params`
   --> src\api\copilot.rs:106:11
    |
106 |     Query(params): Query<serde_json::Value>,
    |           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_params`

warning: unused variable: `state`
   --> src\api\copilot.rs:107:11
    |
107 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\copilot.rs:181:11
    |
181 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `resource`
   --> src\api\copilot.rs:184:9
    |
184 |     let resource = request["resource"].as_str().unwrap_or("virtual_machine");
    |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resource`

warning: unused variable: `state`
   --> src\api\copilot.rs:237:11
    |
237 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\copilot.rs:279:11
    |
279 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `feedback`
   --> src\api\copilot.rs:314:10
    |
314 |     Json(feedback): Json<serde_json::Value>,
    |          ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_feedback`

warning: unused variable: `state`
   --> src\api\copilot.rs:315:11
    |
315 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
  --> src\api\devsecops.rs:63:11
   |
63 |     State(state): State<Arc<crate::AppState>>,
   |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\devsecops.rs:105:11
    |
105 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\devsecops.rs:142:11
    |
142 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\devsecops.rs:181:11
    |
181 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `state`
   --> src\api\devsecops.rs:195:11
    |
195 |     State(state): State<Arc<crate::AppState>>,
    |           ^^^^^ help: if this is intentional, prefix it with an underscore: `_state`

warning: unused variable: `query`
  --> src\ml\predictive_compliance.rs:89:13
   |
89 |         let query = format!(
   |             ^^^^^ help: if this is intentional, prefix it with an underscore: `_query`

warning: unused variable: `resource`
   --> src\ml\predictive_compliance.rs:112:45
    |
112 |     async fn get_applicable_policies(&self, resource: &serde_json::Value) -> Result<Vec<PolicyDefinition>, String> {
    |                                             ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resource`

warning: unused variable: `entities`
   --> src\ml\natural_language.rs:292:46
    |
292 |     async fn handle_policy_violations(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |                                              ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_entities`

warning: unused variable: `entities`
   --> src\ml\natural_language.rs:366:40
    |
366 |     async fn handle_remediation(&self, entities: &[EntityInfo]) -> ConversationResponse {
    |                                        ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_entities`

warning: unused variable: `validation_data`
   --> src\ml\continuous_training.rs:206:58
    |
206 | ...ModelVersion, validation_data: &[TrainingSample]) -> Result<ValidationMetrics, String> {
    |                  ^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_validation_data`

warning: unused variable: `task_data`
   --> src\governance\policy_engine.rs:553:13
    |
553 |         let task_data: serde_json::Value = serde_json::from_str(&response_text)
    |             ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_task_data`

warning: unused variable: `budget_request`
   --> src\governance\cost_management.rs:539:13
    |
539 |         let budget_request = self.build_budget_request(&budget)?;
    |             ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_budget_request`

warning: unused variable: `budget_id`
   --> src\governance\cost_management.rs:787:45
    |
787 |     async fn setup_budget_monitoring(&self, budget_id: &str, budget: &BudgetDefinition) -> GovernanceResult<()> {
    |                                             ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_budget_id`

warning: unused variable: `budget`
   --> src\governance\cost_management.rs:787:62
    |
787 |     async fn setup_budget_monitoring(&self, budget_id: &str, budget: &BudgetDefinition) -> GovernanceResult<()> {
    |                                                              ^^^^^^ help: if this is intentional, prefix it with an underscore: `_budget`

warning: unused variable: `cost_data`
   --> src\governance\cost_management.rs:798:13
    |
798 |         let cost_data = self.fetch_cost_data(scope, 30).await?;
    |             ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_cost_data`

warning: unused variable: `permission_set`
   --> src\governance\access_control.rs:509:14
    |
509 |         for (permission_set, usage_count) in frequent_permissions {
    |              ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_permission_set`

warning: unused variable: `review_definition`
   --> src\governance\access_control.rs:609:46
    |
609 |     pub async fn create_access_review(&self, review_definition: AccessReviewDefinition) -> GovernanceResult<String> {
    |                                              ^^^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_review_definition`

warning: unused variable: `blueprint_id`
   --> src\governance\blueprints.rs:927:43
    |
927 |     pub async fn publish_blueprint(&self, blueprint_id: &str, version: &str) -> GovernanceResult<BlueprintVersion> {
    |                                           ^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_blueprint_id`

warning: unused variable: `query`
   --> src\governance\ai\conversation.rs:116:43
    |
116 |     async fn handle_resource_query(&self, query: &str, intent: &Intent, _context: &ConversationContext) -> GovernanceResult<Qu...
    |                                           ^^^^^ help: if this is intentional, prefix it with an underscore: `_query`

warning: unused variable: `events`
   --> src\governance\ai\correlation.rs:267:59
    |
267 | ...icy_correlation(&self, events: &[CrossDomainEvent]) -> GovernanceResult<Vec<CorrelationPattern>> {
    |                           ^^^^^^ help: if this is intentional, prefix it with an underscore: `_events`

warning: unused variable: `correlation_id`
   --> src\governance\ai\correlation.rs:427:46
    |
427 |     pub async fn monitor_correlations(&self, correlation_id: &str) -> GovernanceResult<CorrelationPattern> {
    |                                              ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_correlation_id`

warning: unused variable: `header`
   --> src\auth.rs:213:23
    |
213 |             if let Ok(header) = jsonwebtoken::decode_header(token) {
    |                       ^^^^^^ help: if this is intentional, prefix it with an underscore: `_header`

warning: unused import: `super`
  --> src\ml\confidence_scoring.rs:12:5
   |
12 | use super::*;
   |     ^^^^^
   |
   = note: `#[warn(unused_imports)]` on by default

warning: unused variable: `lookahead_hours`
   --> src\api\predictions.rs:277:30
    |
277 | fn generate_demo_predictions(lookahead_hours: i64) -> Vec<ViolationPrediction> {
    |                              ^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_lookahead_hours`

warning: unused variable: `policy`
   --> src\ml\predictive_compliance.rs:222:71
    |
222 |     fn calculate_business_impact(&self, resource: &serde_json::Value, policy: &PolicyDefinition) -> BusinessImpact {
    |                                                                       ^^^^^^ help: if this is intentional, prefix it with an underscore: `_policy`

warning: unused variable: `resource`
   --> src\ml\predictive_compliance.rs:235:9
    |
235 |         resource: &serde_json::Value,
    |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resource`

warning: unused variable: `policy`
   --> src\ml\predictive_compliance.rs:236:9
    |
236 |         policy: &PolicyDefinition,
    |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_policy`

warning: value assigned to `risk` is never read
   --> src\ml\risk_scoring.rs:191:17
    |
191 |         let mut risk = 0.0;
    |                 ^^^^
    |
    = help: maybe it is overwritten before being read?
    = note: `#[warn(unused_assignments)]` on by default

warning: unused variable: `pattern_id`
   --> src\ml\pattern_analysis.rs:180:18
    |
180 |             for (pattern_id, pattern) in &self.pattern_library {
    |                  ^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_pattern_id`

warning: unused variable: `time_window`
   --> src\ml\pattern_analysis.rs:197:9
    |
197 |         time_window: Duration,
    |         ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_time_window`

warning: unused variable: `time_series`
   --> src\ml\pattern_analysis.rs:281:9
    |
281 |         time_series: &VecDeque<TimeSeriesPoint>,
    |         ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_time_series`

warning: unused variable: `sequence`
   --> src\ml\pattern_analysis.rs:282:9
    |
282 |         sequence: &[TemporalEvent],
    |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_sequence`

warning: unused variable: `time_series`
   --> src\ml\pattern_analysis.rs:291:9
    |
291 |         time_series: &VecDeque<TimeSeriesPoint>,
    |         ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_time_series`

warning: unused variable: `configuration`
  --> src\ml\drift_detector.rs:95:44
   |
95 |     fn identify_critical_properties(&self, configuration: &serde_json::Value) -> Vec<String> {
   |                                            ^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_configuration`

warning: unused variable: `entities`
   --> src\ml\natural_language.rs:766:48
    |
766 |     fn translate_to_policy(&self, input: &str, entities: &[EntityInfo]) -> serde_json::Value {
    |                                                ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_entities`

warning: unused variable: `p`
   --> src\governance\cost_management.rs:336:13
    |
336 |             p if budget.forecasted_spend > budget.budget.amount => BudgetHealthStatus::ExceededForecasted,
    |             ^ help: if this is intentional, prefix it with an underscore: `_p`

warning: unused variable: `i`
   --> src\governance\cost_management.rs:731:14
    |
731 |         for (i, trend) in trends.iter().enumerate() {
    |              ^ help: if this is intentional, prefix it with an underscore: `_i`

warning: unused variable: `events`
   --> src\governance\ai\correlation.rs:463:35
    |
463 |     pub fn analyze_pattern(&self, events: &[CrossDomainEvent]) -> Vec<CorrelationPattern> {
    |                                   ^^^^^^ help: if this is intentional, prefix it with an underscore: `_events`

warning: unused variable: `rule`
   --> src\correlation\cross_domain_engine.rs:258:84
    |
258 |     fn resources_correlated(&self, source: &AzureResource, target: &AzureResource, rule: &CorrelationRule) -> bool {
    |                                                                                    ^^^^ help: if this is intentional, prefix it with an underscore: `_rule`

warning: unused variable: `out_degree`
   --> src\correlation\resource_mapper.rs:285:17
    |
285 |             let out_degree = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
    |                 ^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_out_degree`

warning: unused variable: `resource`
   --> src\correlation\impact_analyzer.rs:195:58
    |
195 |     fn determine_impact_type(&self, event: &ImpactEvent, resource: &ResourceContext) -> ImpactType {
    |                                                          ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resource`

warning: unused variable: `resources`
   --> src\correlation\advanced_correlation_engine.rs:101:9
    |
101 |         resources: &[AzureResource],
    |         ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `timeline`
   --> src\correlation\predictive_impact_analyzer.rs:234:9
    |
234 |         timeline: &ImpactTimeline
    |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_timeline`

warning: unused variable: `base`
   --> src\correlation\predictive_impact_analyzer.rs:320:9
    |
320 |         base: &PredictiveImpactResult,
    |         ^^^^ help: if this is intentional, prefix it with an underscore: `_base`

warning: unused variable: `source`
   --> src\correlation\predictive_impact_analyzer.rs:549:44
    |
549 |     fn calculate_cascade_confidence(&self, source: &str, target: &str, impact: f64) -> f64 {
    |                                            ^^^^^^ help: if this is intentional, prefix it with an underscore: `_source`

warning: unused variable: `target`
   --> src\correlation\predictive_impact_analyzer.rs:549:58
    |
549 |     fn calculate_cascade_confidence(&self, source: &str, target: &str, impact: f64) -> f64 {
    |                                                          ^^^^^^ help: if this is intentional, prefix it with an underscore: `_target`

warning: unused variable: `resources`
   --> src\correlation\predictive_impact_analyzer.rs:557:66
    |
557 |     fn trace_propagation_path(&self, source: &str, target: &str, resources: &[ResourceContext]) -> Vec<String> {
    |                                                                  ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `historical_data`
   --> src\correlation\predictive_impact_analyzer.rs:629:45
    |
629 |     fn calculate_historical_accuracy(&self, historical_data: &[HistoricalEvent]) -> f64 {
    |                                             ^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_historical_data`

warning: unused variable: `event`
   --> src\correlation\predictive_impact_analyzer.rs:666:39
    |
666 |     fn assess_current_severity(&self, event: &OngoingEvent) -> Severity {
    |                                       ^^^^^ help: if this is intentional, prefix it with an underscore: `_event`

warning: unused variable: `event`
   --> src\correlation\predictive_impact_analyzer.rs:671:42
    |
671 |     fn calculate_propagation_rate(&self, event: &OngoingEvent) -> f64 {
    |                                          ^^^^^ help: if this is intentional, prefix it with an underscore: `_event`

warning: unused variable: `event`
   --> src\correlation\predictive_impact_analyzer.rs:676:41
    |
676 |     fn predict_remaining_cascade(&self, event: &OngoingEvent, _state: &SystemState, resources: &[ResourceContext]) -> ImpactTi...
    |                                         ^^^^^ help: if this is intentional, prefix it with an underscore: `_event`

warning: unused variable: `resources`
   --> src\correlation\predictive_impact_analyzer.rs:676:85
    |
676 | ...tate: &SystemState, resources: &[ResourceContext]) -> ImpactTimeline {
    |                        ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `event`
   --> src\correlation\predictive_impact_analyzer.rs:683:49
    |
683 |     fn update_mitigation_recommendations(&self, event: &OngoingEvent, current_impact: &CurrentImpactState) -> Vec<DynamicMitig...
    |                                                 ^^^^^ help: if this is intentional, prefix it with an underscore: `_event`

warning: unused variable: `current_impact`
   --> src\correlation\predictive_impact_analyzer.rs:683:71
    |
683 | ...&OngoingEvent, current_impact: &CurrentImpactState) -> Vec<DynamicMitigation> {
    |                   ^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_current_impact`

warning: unused variable: `timeline`
   --> src\correlation\predictive_impact_analyzer.rs:694:45
    |
694 |     fn identify_intervention_windows(&self, timeline: &ImpactTimeline) -> Vec<InterventionWindow> {
    |                                             ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_timeline`

warning: unused variable: `event`
   --> src\correlation\predictive_impact_analyzer.rs:698:44
    |
698 |     fn calculate_real_time_accuracy(&self, event: &OngoingEvent) -> f64 {
    |                                            ^^^^^ help: if this is intentional, prefix it with an underscore: `_event`

warning: unused variable: `scenario`
   --> src\correlation\predictive_impact_analyzer.rs:710:44
    |
710 |     fn perform_sensitivity_analysis(&self, scenario: &ImpactScenario, resources: &[ResourceContext]) -> SensitivityAnalysis {
    |                                            ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_scenario`

warning: unused variable: `resources`
   --> src\correlation\predictive_impact_analyzer.rs:710:71
    |
710 |     fn perform_sensitivity_analysis(&self, scenario: &ImpactScenario, resources: &[ResourceContext]) -> SensitivityAnalysis {
    |                                                                       ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `resources`
   --> src\correlation\predictive_impact_analyzer.rs:777:9
    |
777 |         resources: &[ResourceContext],
    |         ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `historical_data`
   --> src\correlation\predictive_impact_analyzer.rs:778:9
    |
778 |         historical_data: &[HistoricalEvent]
    |         ^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_historical_data`

warning: unused variable: `source`
   --> src\correlation\smart_dependency_mapper.rs:577:44
    |
577 |     fn identify_bottlenecks_in_path(&self, source: NodeIndex, target: NodeIndex) -> Vec<String> {
    |                                            ^^^^^^ help: if this is intentional, prefix it with an underscore: `_source`

warning: unused variable: `target`
   --> src\correlation\smart_dependency_mapper.rs:577:63
    |
577 |     fn identify_bottlenecks_in_path(&self, source: NodeIndex, target: NodeIndex) -> Vec<String> {
    |                                                               ^^^^^^ help: if this is intentional, prefix it with an underscore: `_target`

warning: value assigned to `circular_dependency_groups` is never read
   --> src\correlation\smart_dependency_mapper.rs:585:17
    |
585 |         let mut circular_dependency_groups = 0;
    |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
    = help: maybe it is overwritten before being read?

warning: unused variable: `metrics`
   --> src\correlation\smart_dependency_mapper.rs:892:69
    |
892 |     fn detect_dependency_anomalies(&self, events: &[ResourceEvent], metrics: &[RuntimeMetric]) -> Vec<DependencyAnomaly> {
    |                                                                     ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_metrics`

warning: unused variable: `test_graph`
   --> src\correlation\smart_dependency_mapper.rs:993:9
    |
993 |         test_graph: &DiGraph<SmartResource, SmartDependency>,
    |         ^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_test_graph`

warning: unused variable: `test_graph`
    --> src\correlation\smart_dependency_mapper.rs:1020:39
     |
1020 |     fn find_new_critical_paths(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> Vec<String> {
     |                                       ^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_test_graph`

warning: unused variable: `test_graph`
    --> src\correlation\smart_dependency_mapper.rs:1025:37
     |
1025 |     fn calculate_risk_change(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> f64 {
     |                                     ^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_test_graph`

warning: unused variable: `topology`
    --> src\correlation\smart_dependency_mapper.rs:1065:9
     |
1065 |         topology: &NetworkTopology,
     |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_topology`

warning: unused variable: `resources`
    --> src\correlation\smart_dependency_mapper.rs:1066:9
     |
1066 |         resources: &[SmartResourceInfo]
     |         ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_resources`

warning: unused variable: `events`
    --> src\correlation\smart_dependency_mapper.rs:1115:9
     |
1115 |         events: &[ResourceEvent],
     |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_events`

warning: unused variable: `metrics`
    --> src\correlation\smart_dependency_mapper.rs:1116:9
     |
1116 |         metrics: &[RuntimeMetric]
     |         ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_metrics`

warning: unused variable: `row`
   --> src\correlation\graph_driver.rs:305:21
    |
305 |         if let Some(row) = result.next().await.map_err(|e| e.to_string())? {
    |                     ^^^ help: if this is intentional, prefix it with an underscore: `_row`

warning: unused variable: `row`
   --> src\correlation\graph_driver.rs:422:24
    |
422 |         while let Some(row) = result.next().await.map_err(|e| e.to_string())? {
    |                        ^^^ help: if this is intentional, prefix it with an underscore: `_row`

warning: unused variable: `row`
   --> src\correlation\graph_driver.rs:451:21
    |
451 |         if let Some(row) = result.next().await.map_err(|e| e.to_string())? {
    |                     ^^^ help: if this is intentional, prefix it with an underscore: `_row`

warning: unused variable: `algorithm`
   --> src\correlation\graph_driver.rs:514:42
    |
514 |     async fn calculate_centrality(&self, algorithm: &str) -> Result<HashMap<String, f64>, String> {
    |                                          ^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_algorithm`

warning: unused variable: `model`
   --> src\ai\model_registry.rs:219:13
    |
219 |         let model = self
    |             ^^^^^ help: if this is intentional, prefix it with an underscore: `_model`

warning: unused variable: `pricing`
   --> src\finops\mod.rs:295:9
    |
295 |         pricing: &serde_json::Value,
    |         ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_pricing`

warning: unused variable: `current`
   --> src\finops\mod.rs:351:37
    |
351 |     fn calculate_rightsizing(&self, current: &VmSku, metrics: &ResourceMetrics) -> Option<VmSku> {
    |                                     ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_current`

warning: unused variable: `metrics`
   --> src\finops\mod.rs:351:54
    |
351 |     fn calculate_rightsizing(&self, current: &VmSku, metrics: &ResourceMetrics) -> Option<VmSku> {
    |                                                      ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_metrics`

warning: value assigned to `savings_achieved` is never read
   --> src\finops\mod.rs:962:17
    |
962 |         let mut savings_achieved = 0.0;
    |                 ^^^^^^^^^^^^^^^^
    |
    = help: maybe it is overwritten before being read?

warning: type `TemporalEvent` is more private than the item `pattern_analysis::ViolationPattern::temporal_sequence`
  --> src\ml\pattern_analysis.rs:24:5
   |
24 |     pub temporal_sequence: Vec<TemporalEvent>,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ field `pattern_analysis::ViolationPattern::temporal_sequence` is reachable at visibility `pub`
   |
note: but type `TemporalEvent` is only usable at visibility `pub(self)`
  --> src\ml\pattern_analysis.rs:49:1
   |
49 | struct TemporalEvent {
   | ^^^^^^^^^^^^^^^^^^^^
   = note: `#[warn(private_interfaces)]` on by default

warning: type `GraphLayer` is more private than the item `GraphNeuralNetwork::layers`
  --> src\ml\graph_neural_network.rs:21:5
   |
21 |     pub layers: Vec<GraphLayer>,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ field `GraphNeuralNetwork::layers` is reachable at visibility `pub`
   |
note: but type `GraphLayer` is only usable at visibility `pub(self)`
  --> src\ml\graph_neural_network.rs:75:1
   |
75 | struct GraphLayer {
   | ^^^^^^^^^^^^^^^^^

warning: type `GovernanceEvent` is more private than the item `CorrelationEngine::process_event`
   --> src\ml\correlation_engine.rs:397:5
    |
397 |     pub async fn process_event(&self, event: GovernanceEvent) {
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ method `CorrelationEngine::process_event` is reachable at visibility `pub`
    |
note: but type `GovernanceEvent` is only usable at visibility `pub(self)`
   --> src\ml\correlation_engine.rs:180:1
    |
180 | struct GovernanceEvent {
    | ^^^^^^^^^^^^^^^^^^^^^^

warning: fields `execution_id`, `started_at`, `pending_approvals`, and `checkpoints` are never read
  --> src\remediation\workflow_engine.rs:26:5
   |
24 | struct WorkflowExecution {
   |        ----------------- fields in this struct
25 |     workflow_id: Uuid,
26 |     execution_id: Uuid,
   |     ^^^^^^^^^^^^
...
29 |     started_at: DateTime<Utc>,
   |     ^^^^^^^^^^
30 |     completed_steps: Vec<CompletedStep>,
31 |     pending_approvals: Vec<String>,
   |     ^^^^^^^^^^^^^^^^^
32 |     checkpoints: Vec<Checkpoint>,
   |     ^^^^^^^^^^^
   |
   = note: `WorkflowExecution` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default

warning: variants `WaitingForApproval`, `Paused`, `RollingBack`, and `Cancelled` are never constructed
  --> src\remediation\workflow_engine.rs:47:5
   |
44 | enum WorkflowState {
   |      ------------- variants in this enum
...
47 |     WaitingForApproval,
   |     ^^^^^^^^^^^^^^^^^^
48 |     Paused,
   |     ^^^^^^
...
51 |     RollingBack,
   |     ^^^^^^^^^^^
52 |     Cancelled,
   |     ^^^^^^^^^
   |
   = note: `WorkflowState` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `step_id`, `started_at`, `completed_at`, `result`, and `changes` are never read
  --> src\remediation\workflow_engine.rs:57:5
   |
56 | struct CompletedStep {
   |        ------------- fields in this struct
57 |     step_id: String,
   |     ^^^^^^^
58 |     started_at: DateTime<Utc>,
   |     ^^^^^^^^^^
59 |     completed_at: DateTime<Utc>,
   |     ^^^^^^^^^^^^
60 |     result: StepResult,
   |     ^^^^^^
61 |     changes: Vec<AppliedChange>,
   |     ^^^^^^^
   |
   = note: `CompletedStep` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `success`, `output`, and `metrics` are never read
  --> src\remediation\workflow_engine.rs:66:5
   |
65 | struct StepResult {
   |        ---------- fields in this struct
66 |     success: bool,
   |     ^^^^^^^
67 |     output: serde_json::Value,
   |     ^^^^^^
68 |     metrics: StepMetrics,
   |     ^^^^^^^
   |
   = note: `StepResult` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `execution_time_ms`, `resources_modified`, `api_calls_made`, and `retry_count` are never read
  --> src\remediation\workflow_engine.rs:73:5
   |
72 | struct StepMetrics {
   |        ----------- fields in this struct
73 |     execution_time_ms: u64,
   |     ^^^^^^^^^^^^^^^^^
74 |     resources_modified: usize,
   |     ^^^^^^^^^^^^^^^^^^
75 |     api_calls_made: usize,
   |     ^^^^^^^^^^^^^^
76 |     retry_count: u32,
   |     ^^^^^^^^^^^
   |
   = note: `StepMetrics` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `checkpoint_id`, `step_id`, `timestamp`, `state_snapshot`, and `can_rollback` are never read
  --> src\remediation\workflow_engine.rs:81:5
   |
80 | struct Checkpoint {
   |        ---------- fields in this struct
81 |     checkpoint_id: Uuid,
   |     ^^^^^^^^^^^^^
82 |     step_id: String,
   |     ^^^^^^^
83 |     timestamp: DateTime<Utc>,
   |     ^^^^^^^^^
84 |     state_snapshot: serde_json::Value,
   |     ^^^^^^^^^^^^^^
85 |     can_rollback: bool,
   |     ^^^^^^^^^^^^
   |
   = note: `Checkpoint` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `variables`, `resource_states`, `policy_states`, and `azure_context` are never read
  --> src\remediation\workflow_engine.rs:90:5
   |
89 | struct ExecutionContext {
   |        ---------------- fields in this struct
90 |     variables: HashMap<String, serde_json::Value>,
   |     ^^^^^^^^^
91 |     resource_states: HashMap<String, serde_json::Value>,
   |     ^^^^^^^^^^^^^^^
92 |     policy_states: HashMap<String, PolicyState>,
   |     ^^^^^^^^^^^^^
93 |     azure_context: AzureContext,
   |     ^^^^^^^^^^^^^
   |
   = note: `ExecutionContext` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `policy_id`, `compliance_state`, `last_evaluated`, and `violations` are never read
   --> src\remediation\workflow_engine.rs:98:5
    |
97  | struct PolicyState {
    |        ----------- fields in this struct
98  |     policy_id: String,
    |     ^^^^^^^^^
99  |     compliance_state: String,
    |     ^^^^^^^^^^^^^^^^
100 |     last_evaluated: DateTime<Utc>,
    |     ^^^^^^^^^^^^^^
101 |     violations: Vec<String>,
    |     ^^^^^^^^^^
    |
    = note: `PolicyState` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `subscription_id`, `resource_group`, `tenant_id`, and `client_id` are never read
   --> src\remediation\workflow_engine.rs:106:5
    |
105 | struct AzureContext {
    |        ------------ fields in this struct
106 |     subscription_id: String,
    |     ^^^^^^^^^^^^^^^
107 |     resource_group: Option<String>,
    |     ^^^^^^^^^^^^^^
108 |     tenant_id: String,
    |     ^^^^^^^^^
109 |     client_id: String,
    |     ^^^^^^^^^
    |
    = note: `AzureContext` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `pending_approvals` and `approval_history` are never read
   --> src\remediation\workflow_engine.rs:113:5
    |
112 | struct ApprovalManager {
    |        --------------- fields in this struct
113 |     pending_approvals: Arc<RwLock<HashMap<String, PendingApproval>>>,
    |     ^^^^^^^^^^^^^^^^^
114 |     approval_history: Arc<RwLock<Vec<ApprovalRecord>>>,
    |     ^^^^^^^^^^^^^^^^

warning: fields `approval_id`, `gate`, `requested_at`, `expires_at`, and `approvers_responded` are never read
   --> src\remediation\workflow_engine.rs:118:5
    |
117 | struct PendingApproval {
    |        --------------- fields in this struct
118 |     approval_id: String,
    |     ^^^^^^^^^^^
119 |     gate: ApprovalGate,
    |     ^^^^
120 |     requested_at: DateTime<Utc>,
    |     ^^^^^^^^^^^^
121 |     expires_at: DateTime<Utc>,
    |     ^^^^^^^^^^
122 |     approvers_responded: HashMap<String, ApprovalResponse>,
    |     ^^^^^^^^^^^^^^^^^^^

warning: fields `approver`, `decision`, `responded_at`, and `comments` are never read
   --> src\remediation\workflow_engine.rs:127:5
    |
126 | struct ApprovalResponse {
    |        ---------------- fields in this struct
127 |     approver: String,
    |     ^^^^^^^^
128 |     decision: ApprovalDecision,
    |     ^^^^^^^^
129 |     responded_at: DateTime<Utc>,
    |     ^^^^^^^^^^^^
130 |     comments: Option<String>,
    |     ^^^^^^^^
    |
    = note: `ApprovalResponse` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: variants `Approved`, `Rejected`, and `Deferred` are never constructed
   --> src\remediation\workflow_engine.rs:135:5
    |
134 | enum ApprovalDecision {
    |      ---------------- variants in this enum
135 |     Approved,
    |     ^^^^^^^^
136 |     Rejected,
    |     ^^^^^^^^
137 |     Deferred,
    |     ^^^^^^^^
    |
    = note: `ApprovalDecision` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `approval_id`, `workflow_id`, `gate_id`, `outcome`, and `completed_at` are never read
   --> src\remediation\workflow_engine.rs:142:5
    |
141 | struct ApprovalRecord {
    |        -------------- fields in this struct
142 |     approval_id: String,
    |     ^^^^^^^^^^^
143 |     workflow_id: Uuid,
    |     ^^^^^^^^^^^
144 |     gate_id: String,
    |     ^^^^^^^
145 |     outcome: ApprovalOutcome,
    |     ^^^^^^^
146 |     completed_at: DateTime<Utc>,
    |     ^^^^^^^^^^^^
    |
    = note: `ApprovalRecord` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: variants `Approved`, `Rejected`, `TimedOut`, and `AutoApproved` are never constructed
   --> src\remediation\workflow_engine.rs:151:5
    |
150 | enum ApprovalOutcome {
    |      --------------- variants in this enum
151 |     Approved,
    |     ^^^^^^^^
152 |     Rejected,
    |     ^^^^^^^^
153 |     TimedOut,
    |     ^^^^^^^^
154 |     AutoApproved,
    |     ^^^^^^^^^^^^
    |
    = note: `ApprovalOutcome` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `rollback_points` and `rollback_history` are never read
   --> src\remediation\workflow_engine.rs:158:5
    |
157 | struct RollbackManager {
    |        --------------- fields in this struct
158 |     rollback_points: Arc<RwLock<HashMap<String, RollbackPoint>>>,
    |     ^^^^^^^^^^^^^^^
159 |     rollback_history: Arc<RwLock<Vec<RollbackRecord>>>,
    |     ^^^^^^^^^^^^^^^^

warning: fields `token`, `workflow_id`, `checkpoint`, `created_at`, `expires_at`, and `rollback_steps` are never read
   --> src\remediation\workflow_engine.rs:163:5
    |
162 | struct RollbackPoint {
    |        ------------- fields in this struct
163 |     token: String,
    |     ^^^^^
164 |     workflow_id: Uuid,
    |     ^^^^^^^^^^^
165 |     checkpoint: Checkpoint,
    |     ^^^^^^^^^^
166 |     created_at: DateTime<Utc>,
    |     ^^^^^^^^^^
167 |     expires_at: DateTime<Utc>,
    |     ^^^^^^^^^^
168 |     rollback_steps: Vec<RollbackStep>,
    |     ^^^^^^^^^^^^^^

warning: fields `step_id`, `action`, `resource_id`, and `original_state` are never read
   --> src\remediation\workflow_engine.rs:172:5
    |
171 | struct RollbackStep {
    |        ------------ fields in this struct
172 |     step_id: String,
    |     ^^^^^^^
173 |     action: RollbackAction,
    |     ^^^^^^
174 |     resource_id: String,
    |     ^^^^^^^^^^^
175 |     original_state: serde_json::Value,
    |     ^^^^^^^^^^^^^^

warning: variants `RestoreConfiguration`, `DeleteResource`, `RevertPolicyAssignment`, `RestoreAccessControl`, and `Custom` are never constructed
   --> src\remediation\workflow_engine.rs:179:5
    |
178 | enum RollbackAction {
    |      -------------- variants in this enum
179 |     RestoreConfiguration,
    |     ^^^^^^^^^^^^^^^^^^^^
180 |     DeleteResource,
    |     ^^^^^^^^^^^^^^
181 |     RevertPolicyAssignment,
    |     ^^^^^^^^^^^^^^^^^^^^^^
182 |     RestoreAccessControl,
    |     ^^^^^^^^^^^^^^^^^^^^
183 |     Custom(String),
    |     ^^^^^^

warning: multiple fields are never read
   --> src\remediation\workflow_engine.rs:187:5
    |
186 | struct RollbackRecord {
    |        -------------- fields in this struct
187 |     rollback_id: Uuid,
    |     ^^^^^^^^^^^
188 |     workflow_id: Uuid,
    |     ^^^^^^^^^^^
189 |     initiated_at: DateTime<Utc>,
    |     ^^^^^^^^^^^^
190 |     completed_at: Option<DateTime<Utc>>,
    |     ^^^^^^^^^^^^
191 |     success: bool,
    |     ^^^^^^^
192 |     steps_rolled_back: usize,
    |     ^^^^^^^^^^^^^^^^^
193 |     error: Option<String>,
    |     ^^^^^

warning: fields `email_client`, `teams_client`, and `slack_client` are never read
   --> src\remediation\approval_manager.rs:223:5
    |
222 | pub struct NotificationService {
    |            ------------------- fields in this struct
223 |     email_client: Option<EmailClient>,
    |     ^^^^^^^^^^^^
224 |     teams_client: Option<TeamsClient>,
    |     ^^^^^^^^^^^^
225 |     slack_client: Option<SlackClient>,
    |     ^^^^^^^^^^^^

warning: method `get_snapshots` is never used
   --> src\remediation\rollback_manager.rs:446:14
    |
434 | impl SnapshotStore {
    | ------------------ method in this implementation
...
446 |     async fn get_snapshots(&self, token: &str) -> Result<Vec<ResourceSnapshot>, String> {
    |              ^^^^^^^^^^^^^

warning: field `patterns` is never read
   --> src\remediation\bulk_remediation.rs:382:5
    |
381 | pub struct PatternAnalyzer {
    |            --------------- field in this struct
382 |     patterns: Arc<RwLock<HashMap<String, String>>>,
    |     ^^^^^^^^

warning: field `azure_client` is never read
  --> src\remediation\validation_engine.rs:23:5
   |
22 | pub struct ValidationEngine {
   |            ---------------- field in this struct
23 |     azure_client: Arc<AsyncAzureClient>,
   |     ^^^^^^^^^^^^
   |
   = note: `ValidationEngine` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: fields `dependency_graph` and `resource_registry` are never read
   --> src\remediation\validation_engine.rs:299:5
    |
298 | pub struct DependencyChecker {
    |            ----------------- fields in this struct
299 |     dependency_graph: petgraph::Graph<String, DependencyRelation>,
    |     ^^^^^^^^^^^^^^^^
300 |     resource_registry: HashMap<String, ResourceInfo>,
    |     ^^^^^^^^^^^^^^^^^
    |
    = note: `DependencyChecker` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: function `mock_azure_result` is never used
  --> src\api\dashboard.rs:18:10
   |
18 | async fn mock_azure_result<T>() -> Result<T> where T: Default {
   |          ^^^^^^^^^^^^^^^^^

warning: function `mock_azure_result` is never used
  --> src\api\governance.rs:16:10
   |
16 | async fn mock_azure_result<T>() -> anyhow::Result<T> where T: Default {
   |          ^^^^^^^^^^^^^^^^^

warning: fields `azure_client`, `models`, `violation_history`, and `pattern_cache` are never read
  --> src\ml\predictive_compliance.rs:16:5
   |
15 | pub struct PredictiveComplianceEngine {
   |            -------------------------- fields in this struct
16 |     azure_client: AzureClient,
   |     ^^^^^^^^^^^^
17 |     models: HashMap<String, Box<dyn PredictiveModel + Send + Sync>>,
   |     ^^^^^^
18 |     violation_history: Vec<ViolationHistory>,
   |     ^^^^^^^^^^^^^^^^^
19 |     pattern_cache: HashMap<String, PatternSignature>,
   |     ^^^^^^^^^^^^^

warning: field `historical_data` is never read
  --> src\ml\risk_scoring.rs:16:5
   |
13 | pub struct RiskScoringEngine {
   |            ----------------- field in this struct
...
16 |     historical_data: HashMap<String, Vec<RiskEvent>>,
   |     ^^^^^^^^^^^^^^^

warning: field `time_decay_factor` is never read
  --> src\ml\risk_scoring.rs:25:5
   |
20 | struct RiskWeights {
   |        ----------- field in this struct
...
25 |     time_decay_factor: f64,
   |     ^^^^^^^^^^^^^^^^^
   |
   = note: `RiskWeights` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: field `low` is never read
  --> src\ml\risk_scoring.rs:33:5
   |
29 | struct RiskThresholds {
   |        -------------- field in this struct
...
33 |     low: f64,
   |     ^^^
   |
   = note: `RiskThresholds` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: field `sensitivity` is never read
  --> src\ml\pattern_analysis.rs:57:5
   |
56 | struct AnomalyDetector {
   |        --------------- field in this struct
57 |     sensitivity: f64,
   |     ^^^^^^^^^^^

warning: field `knowledge_base` is never read
  --> src\ml\natural_language.rs:22:5
   |
17 | pub struct NaturalLanguageEngine {
   |            --------------------- field in this struct
...
22 |     knowledge_base: GovernanceKnowledgeBase,
   |     ^^^^^^^^^^^^^^

warning: field `patterns` is never read
   --> src\ml\natural_language.rs:149:5
    |
148 | struct EntityExtractor {
    |        --------------- field in this struct
149 |     patterns: HashMap<EntityType, Vec<String>>,
    |     ^^^^^^^^

warning: fields `concepts`, `policies`, `best_practices`, and `compliance_mappings` are never read
   --> src\ml\natural_language.rs:177:5
    |
176 | struct GovernanceKnowledgeBase {
    |        ----------------------- fields in this struct
177 |     concepts: HashMap<String, ConceptDefinition>,
    |     ^^^^^^^^
178 |     policies: HashMap<String, PolicyInfo>,
    |     ^^^^^^^^
179 |     best_practices: Vec<BestPractice>,
    |     ^^^^^^^^^^^^^^
180 |     compliance_mappings: HashMap<String, Vec<String>>,
    |     ^^^^^^^^^^^^^^^^^^^

warning: fields `layer_type`, `weights`, `biases`, and `activation` are never read
  --> src\ml\graph_neural_network.rs:76:5
   |
75 | struct GraphLayer {
   |        ---------- fields in this struct
76 |     layer_type: LayerType,
   |     ^^^^^^^^^^
77 |     weights: Vec<Vec<f64>>,
   |     ^^^^^^^
78 |     biases: Vec<f64>,
   |     ^^^^^^
79 |     activation: ActivationFunction,
   |     ^^^^^^^^^^
   |
   = note: `GraphLayer` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: variants `Tanh` and `Softmax` are never constructed
  --> src\ml\graph_neural_network.rs:94:5
   |
91 | enum ActivationFunction {
   |      ------------------ variants in this enum
...
94 |     Tanh,
   |     ^^^^
95 |     Softmax,
   |     ^^^^^^^
   |
   = note: `ActivationFunction` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `known_patterns`, `pattern_history`, and `ml_model` are never read
   --> src\ml\correlation_engine.rs:101:5
    |
100 | struct PatternDetector {
    |        --------------- fields in this struct
101 |     known_patterns: HashMap<String, PatternTemplate>,
    |     ^^^^^^^^^^^^^^
102 |     pattern_history: Vec<DetectedPattern>,
    |     ^^^^^^^^^^^^^^^
103 |     ml_model: PatternMLModel,
    |     ^^^^^^^^

warning: fields `pattern_id`, `pattern_type`, `detection_rules`, and `min_confidence` are never read
   --> src\ml\correlation_engine.rs:107:5
    |
106 | struct PatternTemplate {
    |        --------------- fields in this struct
107 |     pattern_id: String,
    |     ^^^^^^^^^^
108 |     pattern_type: PatternType,
    |     ^^^^^^^^^^^^
109 |     detection_rules: Vec<DetectionRule>,
    |     ^^^^^^^^^^^^^^^
110 |     min_confidence: f64,
    |     ^^^^^^^^^^^^^^

warning: fields `rule_type`, `condition`, and `weight` are never read
   --> src\ml\correlation_engine.rs:114:5
    |
113 | struct DetectionRule {
    |        ------------- fields in this struct
114 |     rule_type: String,
    |     ^^^^^^^^^
115 |     condition: String,
    |     ^^^^^^^^^
116 |     weight: f64,
    |     ^^^^^^

warning: fields `pattern`, `detection_time`, and `matched_template` are never read
   --> src\ml\correlation_engine.rs:120:5
    |
119 | struct DetectedPattern {
    |        --------------- fields in this struct
120 |     pattern: CorrelationPattern,
    |     ^^^^^^^
121 |     detection_time: DateTime<Utc>,
    |     ^^^^^^^^^^^^^^
122 |     matched_template: String,
    |     ^^^^^^^^^^^^^^^^

warning: fields `weights`, `biases`, and `threshold` are never read
   --> src\ml\correlation_engine.rs:126:5
    |
125 | struct PatternMLModel {
    |        -------------- fields in this struct
126 |     weights: Vec<Vec<f64>>,
    |     ^^^^^^^
127 |     biases: Vec<f64>,
    |     ^^^^^^
128 |     threshold: f64,
    |     ^^^^^^^^^

warning: fields `baseline_metrics`, `anomaly_threshold`, and `ml_detector` are never read
   --> src\ml\correlation_engine.rs:132:5
    |
131 | struct AnomalyDetector {
    |        --------------- fields in this struct
132 |     baseline_metrics: HashMap<String, BaselineMetric>,
    |     ^^^^^^^^^^^^^^^^
133 |     anomaly_threshold: f64,
    |     ^^^^^^^^^^^^^^^^^
134 |     ml_detector: AnomalyMLModel,
    |     ^^^^^^^^^^^

warning: fields `resource_id`, `metric_name`, `mean`, `std_dev`, and `last_updated` are never read
   --> src\ml\correlation_engine.rs:138:5
    |
137 | struct BaselineMetric {
    |        -------------- fields in this struct
138 |     resource_id: String,
    |     ^^^^^^^^^^^
139 |     metric_name: String,
    |     ^^^^^^^^^^^
140 |     mean: f64,
    |     ^^^^
141 |     std_dev: f64,
    |     ^^^^^^^
142 |     last_updated: DateTime<Utc>,
    |     ^^^^^^^^^^^^

warning: fields `isolation_forest` and `autoencoder` are never read
   --> src\ml\correlation_engine.rs:146:5
    |
145 | struct AnomalyMLModel {
    |        -------------- fields in this struct
146 |     isolation_forest: IsolationForest,
    |     ^^^^^^^^^^^^^^^^
147 |     autoencoder: Autoencoder,
    |     ^^^^^^^^^^^

warning: fields `trees` and `sample_size` are never read
   --> src\ml\correlation_engine.rs:151:5
    |
150 | struct IsolationForest {
    |        --------------- fields in this struct
151 |     trees: Vec<IsolationTree>,
    |     ^^^^^
152 |     sample_size: usize,
    |     ^^^^^^^^^^^

warning: fields `root` and `max_depth` are never read
   --> src\ml\correlation_engine.rs:156:5
    |
155 | struct IsolationTree {
    |        ------------- fields in this struct
156 |     root: TreeNode,
    |     ^^^^
157 |     max_depth: usize,
    |     ^^^^^^^^^

warning: fields `split_feature`, `split_value`, `left`, and `right` are never read
   --> src\ml\correlation_engine.rs:161:5
    |
160 | struct TreeNode {
    |        -------- fields in this struct
161 |     split_feature: usize,
    |     ^^^^^^^^^^^^^
162 |     split_value: f64,
    |     ^^^^^^^^^^^
163 |     left: Option<Box<TreeNode>>,
    |     ^^^^
164 |     right: Option<Box<TreeNode>>,
    |     ^^^^^

warning: fields `encoder_weights`, `decoder_weights`, and `latent_dim` are never read
   --> src\ml\correlation_engine.rs:168:5
    |
167 | struct Autoencoder {
    |        ----------- fields in this struct
168 |     encoder_weights: Vec<Vec<f64>>,
    |     ^^^^^^^^^^^^^^^
169 |     decoder_weights: Vec<Vec<f64>>,
    |     ^^^^^^^^^^^^^^^
170 |     latent_dim: usize,
    |     ^^^^^^^^^^

warning: field `processing_interval` is never read
   --> src\ml\correlation_engine.rs:175:5
    |
173 | struct RealTimeProcessor {
    |        ----------------- field in this struct
174 |     event_buffer: Arc<RwLock<Vec<GovernanceEvent>>>,
175 |     processing_interval: Duration,
    |     ^^^^^^^^^^^^^^^^^^^

warning: field `explanation_templates` is never read
  --> src\ml\explainability.rs:21:5
   |
18 | pub struct PredictionExplainer {
   |            ------------------- field in this struct
...
21 |     explanation_templates: HashMap<String, String>,
   |     ^^^^^^^^^^^^^^^^^^^^^

warning: field `pattern_matchers` is never read
  --> src\ml\pattern_library.rs:19:5
   |
17 | pub struct ViolationPatternLibrary {
   |            ----------------------- field in this struct
18 |     patterns: HashMap<String, ViolationPattern>,
19 |     pattern_matchers: Vec<Box<dyn PatternMatcher>>,
   |     ^^^^^^^^^^^^^^^^

warning: field `resource_models` is never read
  --> src\ml\cost_prediction.rs:21:5
   |
17 | pub struct CostPredictionModel {
   |            ------------------- field in this struct
...
21 |     resource_models: HashMap<String, ResourceCostModel>,
   |     ^^^^^^^^^^^^^^^

warning: fields `resource_type`, `pricing_model`, and `usage_patterns` are never read
   --> src\ml\cost_prediction.rs:236:5
    |
235 | pub struct ResourceCostModel {
    |            ----------------- fields in this struct
236 |     resource_type: String,
    |     ^^^^^^^^^^^^^
237 |     pricing_model: PricingModel,
    |     ^^^^^^^^^^^^^
238 |     usage_patterns: UsagePattern,
    |     ^^^^^^^^^^^^^^

warning: fields `num_trees`, `sample_size`, and `trees` are never read
   --> src\ml\anomaly_detection.rs:149:5
    |
148 | pub struct IsolationForest {
    |            --------------- fields in this struct
149 |     num_trees: usize,
    |     ^^^^^^^^^
150 |     sample_size: usize,
    |     ^^^^^^^^^^^
151 |     trees: Vec<IsolationTree>,
    |     ^^^^^

warning: fields `split_feature`, `split_value`, `left`, and `right` are never read
   --> src\ml\anomaly_detection.rs:180:5
    |
179 | struct IsolationTree {
    |        ------------- fields in this struct
180 |     split_feature: usize,
    |     ^^^^^^^^^^^^^
181 |     split_value: f64,
    |     ^^^^^^^^^^^
182 |     left: Option<Box<IsolationTree>>,
    |     ^^^^
183 |     right: Option<Box<IsolationTree>>,
    |     ^^^^^

warning: field `name` is never read
   --> src\ml\anomaly_detection.rs:301:5
    |
300 | struct Pattern {
    |        ------- field in this struct
301 |     name: String,
    |     ^^^^

warning: field `policies` is never read
  --> src\ml\entity_extractor.rs:51:5
   |
47 | pub struct EntityExtractor {
   |            --------------- field in this struct
...
51 |     policies: Vec<String>,
   |     ^^^^^^^^

warning: struct `HttpResponse` is never constructed
   --> src\governance\resource_graph.rs:591:8
    |
591 | struct HttpResponse {
    |        ^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\identity.rs:20:5
   |
19 | pub struct IdentityGovernanceClient {
   |            ------------------------ field in this struct
20 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\monitoring.rs:20:5
   |
19 | pub struct GovernanceMonitor {
   |            ----------------- field in this struct
20 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\cost_management.rs:21:5
   |
20 | pub struct CostGovernanceEngine {
   |            -------------------- field in this struct
21 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `alert_thresholds` is never read
   --> src\governance\cost_management.rs:322:5
    |
321 | pub struct BudgetMonitor {
    |            ------------- field in this struct
322 |     alert_thresholds: Vec<f64>,
    |     ^^^^^^^^^^^^^^^^

warning: field `historical_periods` is never read
   --> src\governance\cost_management.rs:344:5
    |
343 | pub struct ForecastEngine {
    |            -------------- field in this struct
344 |     historical_periods: u32,
    |     ^^^^^^^^^^^^^^^^^^

warning: field `savings_thresholds` is never read
   --> src\governance\cost_management.rs:400:5
    |
399 | pub struct OptimizationAnalyzer {
    |            -------------------- field in this struct
400 |     savings_thresholds: HashMap<OptimizationType, f64>,
    |     ^^^^^^^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\security_posture.rs:21:5
   |
20 | pub struct SecurityPostureEngine {
   |            --------------------- field in this struct
21 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `ml_models` is never read
   --> src\governance\security_posture.rs:314:5
    |
312 | pub struct ThreatDetector {
    |            -------------- field in this struct
313 |     alert_rules: Vec<ThreatRule>,
314 |     ml_models: HashMap<String, ThreatModel>,
    |     ^^^^^^^^^

warning: field `assessment_cache` is never read
   --> src\governance\security_posture.rs:338:5
    |
336 | pub struct ComplianceMonitor {
    |            ----------------- field in this struct
337 |     frameworks: Vec<String>,
338 |     assessment_cache: HashMap<String, DateTime<Utc>>,
    |     ^^^^^^^^^^^^^^^^

warning: fields `scan_frequency`, `last_scan`, and `vulnerability_database` are never read
   --> src\governance\security_posture.rs:343:5
    |
342 | pub struct VulnerabilityScanner {
    |            -------------------- fields in this struct
343 |     scan_frequency: Duration,
    |     ^^^^^^^^^^^^^^
344 |     last_scan: HashMap<String, DateTime<Utc>>,
    |     ^^^^^^^^^
345 |     vulnerability_database: HashMap<String, VulnerabilityData>,
    |     ^^^^^^^^^^^^^^^^^^^^^^

warning: fields `azure_client` and `identity_tracker` are never read
  --> src\governance\access_control.rs:21:5
   |
20 | pub struct AccessGovernanceEngine {
   |            ---------------------- fields in this struct
21 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^
...
25 |     identity_tracker: IdentityTracker,
   |     ^^^^^^^^^^^^^^^^

warning: field `role_relationships` is never read
   --> src\governance\access_control.rs:336:5
    |
334 | pub struct RoleAnalyzer {
    |            ------------ field in this struct
335 |     privilege_matrix: HashMap<String, PrivilegeLevel>,
336 |     role_relationships: HashMap<String, Vec<String>>,
    |     ^^^^^^^^^^^^^^^^^^

warning: field `baseline_permissions` is never read
   --> src\governance\access_control.rs:342:5
    |
340 | pub struct PermissionMonitor {
    |            ----------------- field in this struct
341 |     permission_usage_tracking: HashMap<String, PermissionUsage>,
342 |     baseline_permissions: HashMap<String, Vec<String>>,
    |     ^^^^^^^^^^^^^^^^^^^^

warning: field `identity_cache` is never read
   --> src\governance\access_control.rs:355:5
    |
354 | pub struct IdentityTracker {
    |            --------------- field in this struct
355 |     identity_cache: HashMap<String, IdentityInfo>,
    |     ^^^^^^^^^^^^^^

warning: fields `azure_client` and `vnet_manager` are never read
  --> src\governance\network.rs:22:5
   |
21 | pub struct NetworkGovernanceEngine {
   |            ----------------------- fields in this struct
22 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^
...
26 |     vnet_manager: VNetManager,
   |     ^^^^^^^^^^^^

warning: field `rule_patterns` is never read
   --> src\governance\network.rs:517:5
    |
516 | pub struct NetworkSecurityAnalyzer {
    |            ----------------------- field in this struct
517 |     rule_patterns: HashMap<String, SecurityRiskLevel>,
    |     ^^^^^^^^^^^^^

warning: field `traffic_patterns` is never read
   --> src\governance\network.rs:532:5
    |
530 | pub struct FirewallMonitor {
    |            --------------- field in this struct
531 |     policy_cache: HashMap<String, FirewallPolicy>,
532 |     traffic_patterns: HashMap<String, TrafficPattern>,
    |     ^^^^^^^^^^^^^^^^

warning: fields `peering_relationships` and `address_space_allocations` are never read
   --> src\governance\network.rs:555:5
    |
554 | pub struct VNetManager {
    |            ----------- fields in this struct
555 |     peering_relationships: HashMap<String, Vec<String>>,
    |     ^^^^^^^^^^^^^^^^^^^^^
556 |     address_space_allocations: HashMap<String, Vec<String>>,
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\optimization.rs:22:5
   |
21 | pub struct OptimizationEngine {
   |            ------------------ field in this struct
22 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `reservation_opportunities` is never read
   --> src\governance\optimization.rs:347:5
    |
344 | pub struct CostOptimizationAnalyzer {
    |            ------------------------ field in this struct
...
347 |     reservation_opportunities: HashMap<String, ReservationOpportunity>,
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^

warning: field `bottleneck_detector` is never read
   --> src\governance\optimization.rs:389:5
    |
387 | pub struct PerformanceAnalyzer {
    |            ------------------- field in this struct
388 |     performance_baselines: HashMap<String, PerformanceBaseline>,
389 |     bottleneck_detector: BottleneckDetector,
    |     ^^^^^^^^^^^^^^^^^^^

warning: fields `cpu_threshold`, `memory_threshold`, `storage_threshold`, and `network_threshold` are never read
   --> src\governance\optimization.rs:403:5
    |
402 | pub struct BottleneckDetector {
    |            ------------------ fields in this struct
403 |     cpu_threshold: f64,
    |     ^^^^^^^^^^^^^
404 |     memory_threshold: f64,
    |     ^^^^^^^^^^^^^^^^
405 |     storage_threshold: f64,
    |     ^^^^^^^^^^^^^^^^^
406 |     network_threshold: f64,
    |     ^^^^^^^^^^^^^^^^^

warning: field `vulnerability_database` is never read
   --> src\governance\optimization.rs:412:5
    |
410 | pub struct SecurityOptimizer {
    |            ----------------- field in this struct
411 |     security_policies: HashMap<String, SecurityPolicy>,
412 |     vulnerability_database: HashMap<String, VulnerabilityInfo>,
    |     ^^^^^^^^^^^^^^^^^^^^^^

warning: field `azure_client` is never read
  --> src\governance\blueprints.rs:22:5
   |
21 | pub struct GovernanceBlueprints {
   |            -------------------- field in this struct
22 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^

warning: field `parameter_validators` is never read
   --> src\governance\blueprints.rs:718:5
    |
716 | pub struct BlueprintTemplateEngine {
    |            ----------------------- field in this struct
717 |     template_library: HashMap<String, BlueprintTemplate>,
718 |     parameter_validators: HashMap<String, ParameterValidator>,
    |     ^^^^^^^^^^^^^^^^^^^^

warning: field `evaluation_engine` is never read
   --> src\governance\blueprints.rs:753:5
    |
751 | pub struct BlueprintComplianceMonitor {
    |            -------------------------- field in this struct
752 |     compliance_policies: HashMap<String, CompliancePolicy>,
753 |     evaluation_engine: ComplianceEvaluationEngine,
    |     ^^^^^^^^^^^^^^^^^

warning: fields `deployment_history` and `performance_metrics` are never read
   --> src\governance\blueprints.rs:798:5
    |
797 | pub struct DeploymentTracker {
    |            ----------------- fields in this struct
798 |     deployment_history: HashMap<String, Vec<DeploymentRecord>>,
    |     ^^^^^^^^^^^^^^^^^^
799 |     performance_metrics: HashMap<String, PerformanceMetrics>,
    |     ^^^^^^^^^^^^^^^^^^^

warning: field `policy_engine` is never read
   --> src\governance\blueprints.rs:812:5
    |
810 | pub struct GovernanceValidator {
    |            ------------------- field in this struct
811 |     validation_rules: HashMap<String, GovernanceRule>,
812 |     policy_engine: PolicyValidationEngine,
    |     ^^^^^^^^^^^^^

warning: fields `policy_definitions` and `validation_cache` are never read
   --> src\governance\blueprints.rs:834:5
    |
833 | pub struct PolicyValidationEngine {
    |            ---------------------- fields in this struct
834 |     policy_definitions: HashMap<String, PolicyDefinition>,
    |     ^^^^^^^^^^^^^^^^^^
835 |     validation_cache: HashMap<String, ValidationCacheEntry>,
    |     ^^^^^^^^^^^^^^^^

warning: fields `policy_engine`, `monitoring`, and `ai_engine` are never read
  --> src\governance\unified_api.rs:23:5
   |
21 | pub struct UnifiedGovernanceAPI {
   |            -------------------- fields in this struct
22 |     resource_graph: Arc<crate::governance::resource_graph::ResourceGraphClient>,
23 |     policy_engine: Arc<crate::governance::policy_engine::PolicyEngine>,
   |     ^^^^^^^^^^^^^
24 |     identity: Arc<crate::governance::identity::IdentityGovernanceClient>,
25 |     monitoring: Arc<crate::governance::monitoring::GovernanceMonitor>,
   |     ^^^^^^^^^^
26 |     ai_engine: Arc<crate::governance::ai::AIGovernanceEngine>,
   |     ^^^^^^^^^

warning: fields `resource_graph`, `policy_engine`, `identity`, and `monitoring` are never read
  --> src\governance\ai\mod.rs:28:5
   |
27 | pub struct AIGovernanceEngine {
   |            ------------------ fields in this struct
28 |     resource_graph: Arc<crate::governance::resource_graph::ResourceGraphClient>,
   |     ^^^^^^^^^^^^^^
29 |     policy_engine: Arc<crate::governance::policy_engine::PolicyEngine>,
   |     ^^^^^^^^^^^^^
30 |     identity: Arc<crate::governance::identity::IdentityGovernanceClient>,
   |     ^^^^^^^^
31 |     monitoring: Arc<crate::governance::monitoring::GovernanceMonitor>,
   |     ^^^^^^^^^^

warning: fields `identity`, `monitoring`, and `context_manager` are never read
  --> src\governance\ai\conversation.rs:21:5
   |
18 | pub struct ConversationalGovernance {
   |            ------------------------ fields in this struct
...
21 |     identity: Arc<crate::governance::identity::IdentityGovernanceClient>,
   |     ^^^^^^^^
22 |     monitoring: Arc<crate::governance::monitoring::GovernanceMonitor>,
   |     ^^^^^^^^^^
23 |     intent_classifier: IntentClassifier,
24 |     context_manager: ContextManager,
   |     ^^^^^^^^^^^^^^^

warning: fields `policy_engine`, `correlation_cache`, and `pattern_analyzer` are never read
  --> src\governance\ai\correlation.rs:20:5
   |
18 | pub struct CrossDomainCorrelationEngine {
   |            ---------------------------- fields in this struct
19 |     resource_graph: Arc<crate::governance::resource_graph::ResourceGraphClient>,
20 |     policy_engine: Arc<crate::governance::policy_engine::PolicyEngine>,
   |     ^^^^^^^^^^^^^
21 |     correlation_cache: HashMap<String, Vec<CorrelationPattern>>,
   |     ^^^^^^^^^^^^^^^^^
22 |     pattern_analyzer: PatternAnalyzer,
   |     ^^^^^^^^^^^^^^^^

warning: field `correlation_rules` is never read
  --> src\governance\ai\correlation.rs:96:5
   |
93 | pub struct PatternAnalyzer {
   |            --------------- field in this struct
...
96 |     correlation_rules: Vec<CorrelationRule>,
   |     ^^^^^^^^^^^^^^^^^

warning: field `policy_engine` is never read
  --> src\governance\ai\prediction.rs:20:5
   |
18 | pub struct PredictiveComplianceEngine {
   |            -------------------------- field in this struct
19 |     resource_graph: Arc<crate::governance::resource_graph::ResourceGraphClient>,
20 |     policy_engine: Arc<crate::governance::policy_engine::PolicyEngine>,
   |     ^^^^^^^^^^^^^

warning: fields `model_type` and `features` are never read
  --> src\governance\ai\prediction.rs:90:5
   |
89 | pub struct PredictionModel {
   |            --------------- fields in this struct
90 |     model_type: ModelType,
   |     ^^^^^^^^^^
...
93 |     features: Vec<String>,
   |     ^^^^^^^^

warning: fields `violation_history` and `resource_history` are never read
   --> src\governance\ai\prediction.rs:107:5
    |
105 | pub struct HistoricalDataStore {
    |            ------------------- fields in this struct
106 |     compliance_history: Vec<HistoricalCompliance>,
107 |     violation_history: Vec<HistoricalViolation>,
    |     ^^^^^^^^^^^^^^^^^
108 |     resource_history: Vec<HistoricalResource>,
    |     ^^^^^^^^^^^^^^^^

warning: field `confidence_threshold` is never read
   --> src\governance\ai\prediction.rs:142:5
    |
139 | pub struct TrendAnalyzer {
    |            ------------- field in this struct
...
142 |     confidence_threshold: f64,
    |     ^^^^^^^^^^^^^^^^^^^^

warning: field `impact_calculator` is never read
  --> src\correlation\cross_domain_engine.rs:22:5
   |
18 | pub struct CrossDomainEngine {
   |            ----------------- field in this struct
...
22 |     impact_calculator: ImpactCalculator,
   |     ^^^^^^^^^^^^^^^^^

warning: fields `id`, `name`, `resource_type`, `location`, and `metadata` are never read
   --> src\correlation\resource_mapper.rs:433:5
    |
432 | struct Resource {
    |        -------- fields in this struct
433 |     id: String,
    |     ^^
434 |     name: String,
    |     ^^^^
435 |     resource_type: String,
    |     ^^^^^^^^^^^^^
436 |     location: String,
    |     ^^^^^^^^
437 |     metadata: HashMap<String, String>,
    |     ^^^^^^^^
    |
    = note: `Resource` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis

warning: fields `domain`, `base_impact`, and `recovery_time_hours` are never read
   --> src\correlation\impact_analyzer.rs:384:5
    |
383 | struct ImpactModel {
    |        ----------- fields in this struct
384 |     domain: String,
    |     ^^^^^^
385 |     base_impact: f64,
    |     ^^^^^^^^^^^
386 |     propagation_factor: f64,
387 |     recovery_time_hours: u32,
    |     ^^^^^^^^^^^^^^^^^^^

warning: fields `name`, `source_type`, `affected_types`, and `impact_multiplier` are never read
   --> src\correlation\impact_analyzer.rs:391:5
    |
390 | struct PropagationRule {
    |        --------------- fields in this struct
391 |     name: String,
    |     ^^^^
392 |     source_type: String,
    |     ^^^^^^^^^^^
393 |     affected_types: Vec<String>,
    |     ^^^^^^^^^^^^^^
394 |     impact_multiplier: f64,
    |     ^^^^^^^^^^^^^^^^^

warning: field `what_if_analyzer` is never read
  --> src\correlation\predictive_impact_analyzer.rs:41:5
   |
37 | pub struct PredictiveImpactAnalyzer {
   |            ------------------------ field in this struct
...
41 |     what_if_analyzer: WhatIfAnalyzer,
   |     ^^^^^^^^^^^^^^^^

warning: field `historical_patterns` is never read
   --> src\correlation\predictive_impact_analyzer.rs:765:5
    |
764 | pub struct PredictiveEngine {
    |            ---------------- field in this struct
765 |     historical_patterns: Vec<ImpactPattern>,
    |     ^^^^^^^^^^^^^^^^^^^

warning: field `scenario_templates` is never read
   --> src\correlation\predictive_impact_analyzer.rs:812:5
    |
811 | pub struct WhatIfAnalyzer {
    |            -------------- field in this struct
812 |     scenario_templates: Vec<ScenarioTemplate>,
    |     ^^^^^^^^^^^^^^^^^^

warning: field `quantification_models` is never read
   --> src\correlation\predictive_impact_analyzer.rs:824:5
    |
823 | pub struct RiskQuantifier {
    |            -------------- field in this struct
824 |     quantification_models: HashMap<String, QuantificationModel>,
    |     ^^^^^^^^^^^^^^^^^^^^^

warning: field `inference_models` is never read
    --> src\correlation\smart_dependency_mapper.rs:1034:5
     |
1033 | pub struct DependencyInferenceEngine {
     |            ------------------------- field in this struct
1034 |     inference_models: HashMap<String, InferenceModel>,
     |     ^^^^^^^^^^^^^^^^

warning: field `network_analyzers` is never read
    --> src\correlation\smart_dependency_mapper.rs:1054:5
     |
1053 | pub struct TopologyAnalyzer {
     |            ---------------- field in this struct
1054 |     network_analyzers: Vec<Box<dyn NetworkAnalyzer>>,
     |     ^^^^^^^^^^^^^^^^^

warning: field `prediction_models` is never read
    --> src\correlation\smart_dependency_mapper.rs:1074:5
     |
1073 | pub struct DependencyPredictor {
     |            ------------------- field in this struct
1074 |     prediction_models: HashMap<String, PredictionModel>,
     |     ^^^^^^^^^^^^^^^^^

warning: fields `event_buffer` and `pattern_detectors` are never read
    --> src\correlation\smart_dependency_mapper.rs:1102:5
     |
1101 | pub struct RealTimeDependencyTracker {
     |            ------------------------- fields in this struct
1102 |     event_buffer: VecDeque<ResourceEvent>,
     |     ^^^^^^^^^^^^
1103 |     pattern_detectors: Vec<Box<dyn PatternDetector>>,
     |     ^^^^^^^^^^^^^^^^^

warning: method `find_key` is never used
   --> src\auth.rs:159:8
    |
106 | impl TokenValidator {
    | ------------------- method in this implementation
...
159 |     fn find_key<'a>(&self, kid: &str, jwks: &'a JwksResponse) -> Option<&'a Jwk> {
    |        ^^^^^^^^

warning: field `config` is never read
  --> src\azure\auth.rs:20:5
   |
17 | pub struct AzureAuthProvider {
   |            ----------------- field in this struct
...
20 |     config: super::AzureConfig,
   |     ^^^^^^
   |
   = note: `AzureAuthProvider` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: fields `id`, `name`, `resource_type`, `location`, and `tags` are never read
  --> src\azure_client.rs:32:5
   |
31 | struct AzureResource {
   |        ------------- fields in this struct
32 |     id: String,
   |     ^^
33 |     name: String,
   |     ^^^^
34 |     #[serde(rename = "type")]
35 |     resource_type: String,
   |     ^^^^^^^^^^^^^
36 |     location: String,
   |     ^^^^^^^^
37 |     tags: Option<serde_json::Value>,
   |     ^^^^

warning: fields `id`, `name`, `display_name`, `policy_definition_id`, and `enforcement_mode` are never read
  --> src\azure_client.rs:42:5
   |
41 | struct AzurePolicyAssignment {
   |        --------------------- fields in this struct
42 |     id: String,
   |     ^^
43 |     name: String,
   |     ^^^^
44 |     #[serde(rename = "displayName")]
45 |     display_name: String,
   |     ^^^^^^^^^^^^
46 |     #[serde(rename = "policyDefinitionId")]
47 |     policy_definition_id: String,
   |     ^^^^^^^^^^^^^^^^^^^^
48 |     enforcement_mode: Option<String>,
   |     ^^^^^^^^^^^^^^^^

warning: fields `id`, `principal_id`, `role_definition_id`, and `scope` are never read
  --> src\azure_client.rs:53:5
   |
52 | struct AzureRoleAssignment {
   |        ------------------- fields in this struct
53 |     id: String,
   |     ^^
54 |     #[serde(rename = "principalId")]
55 |     principal_id: String,
   |     ^^^^^^^^^^^^
56 |     #[serde(rename = "roleDefinitionId")]
57 |     role_definition_id: String,
   |     ^^^^^^^^^^^^^^^^^^
58 |     scope: String,
   |     ^^^^^

warning: fields `billing_period`, `usage_start`, `usage_end`, `pretax_cost`, and `currency` are never read
  --> src\azure_client.rs:64:5
   |
62 | struct CostManagementUsage {
   |        ------------------- fields in this struct
63 |     #[serde(rename = "billingPeriod")]
64 |     billing_period: String,
   |     ^^^^^^^^^^^^^^
65 |     #[serde(rename = "usageStart")]
66 |     usage_start: DateTime<Utc>,
   |     ^^^^^^^^^^^
67 |     #[serde(rename = "usageEnd")]
68 |     usage_end: DateTime<Utc>,
   |     ^^^^^^^^^
69 |     #[serde(rename = "pretaxCost")]
70 |     pretax_cost: f64,
   |     ^^^^^^^^^^^
71 |     currency: String,
   |     ^^^^^^^^

warning: fields `azure_client` and `catalog` are never read
  --> src\resources\manager.rs:18:5
   |
17 | pub struct ResourceManager {
   |            --------------- fields in this struct
18 |     azure_client: Arc<AzureClient>,
   |     ^^^^^^^^^^^^
19 |     resources: Arc<RwLock<Vec<AzureResource>>>,
20 |     catalog: Arc<ResourceCatalog>,
   |     ^^^^^^^

warning: field `model_registry` is never read
  --> src\resources\correlations.rs:74:5
   |
73 | pub struct CrossDomainCorrelationEngine {
   |            ---------------------------- field in this struct
74 |     model_registry: Arc<ModelRegistry>,
   |     ^^^^^^^^^^^^^^

warning: fields `vault_url` and `rotation_check_interval` are never read
  --> src\secrets.rs:24:5
   |
21 | pub struct SecretsManager {
   |            -------------- fields in this struct
...
24 |     vault_url: String,
   |     ^^^^^^^^^
25 |     cache_ttl: Duration,
26 |     rotation_check_interval: Duration,
   |     ^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `SecretsManager` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: field `version` is never read
  --> src\secrets.rs:33:5
   |
30 | struct CachedSecret {
   |        ------------ field in this struct
...
33 |     version: Option<String>,
   |     ^^^^^^^
   |
   = note: `CachedSecret` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis

warning: fields `redis_url` and `default_ttl` are never read
  --> src\cache.rs:19:5
   |
18 | pub struct CacheManager {
   |            ------------ fields in this struct
19 |     redis_url: String,
   |     ^^^^^^^^^
20 |     default_ttl: Duration,
   |     ^^^^^^^^^^^
   |
   = note: `CacheManager` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: field `cache` is never read
   --> src\finops\mod.rs:150:5
    |
148 | pub struct AzureFinOpsEngine {
    |            ----------------- field in this struct
149 |     azure_client: crate::azure_client_async::AsyncAzureClient,
150 |     cache: std::sync::Arc<tokio::sync::RwLock<HashMap<String, CachedMetrics>>>,
    |     ^^^^^

warning: fields `data` and `timestamp` are never read
   --> src\finops\mod.rs:155:5
    |
154 | struct CachedMetrics {
    |        ------------- fields in this struct
155 |     data: Vec<u8>,
    |     ^^^^
156 |     timestamp: DateTime<Utc>,
    |     ^^^^^^^^^
    |
    = note: `CachedMetrics` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis

warning: methods `generate_idle_recommendation`, `calculate_rightsizing`, and `find_smaller_sku` are never used
   --> src\finops\mod.rs:211:14
    |
159 | impl AzureFinOpsEngine {
    | ---------------------- methods in this implementation
...
211 |     async fn generate_idle_recommendation(
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
351 |     fn calculate_rightsizing(&self, current: &VmSku, metrics: &ResourceMetrics) -> Option<VmSku> {
    |        ^^^^^^^^^^^^^^^^^^^^^
...
462 |     fn find_smaller_sku(&self, current: &VmSku) -> Option<VmSku> {
    |        ^^^^^^^^^^^^^^^^

warning: struct `ResourceMetrics` is never constructed
   --> src\finops\mod.rs:505:8
    |
505 | struct ResourceMetrics {
    |        ^^^^^^^^^^^^^^^

warning: struct `VmSku` is never constructed
   --> src\finops\mod.rs:516:8
    |
516 | struct VmSku {
    |        ^^^^^

warning: use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified
   --> src\remediation\mod.rs:359:5
    |
359 |     async fn execute(&self, request: RemediationRequest) -> Result<RemediationResult, String>;
    |     ^^^^^
    |
    = note: you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
    = note: `#[warn(async_fn_in_trait)]` on by default
help: you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change
    |
359 -     async fn execute(&self, request: RemediationRequest) -> Result<RemediationResult, String>;
359 +     fn execute(&self, request: RemediationRequest) -> impl std::future::Future<Output = Result<RemediationResult, String>> + Send;
    |

warning: use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified
   --> src\remediation\mod.rs:360:5
    |
360 |     async fn validate(&self, request: &RemediationRequest) -> Result<bool, String>;
    |     ^^^^^
    |
    = note: you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
help: you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change
    |
360 -     async fn validate(&self, request: &RemediationRequest) -> Result<bool, String>;
360 +     fn validate(&self, request: &RemediationRequest) -> impl std::future::Future<Output = Result<bool, String>> + Send;
    |

warning: use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified
   --> src\remediation\mod.rs:361:5
    |
361 |     async fn rollback(&self, rollback_token: String) -> Result<RemediationResult, String>;
    |     ^^^^^
    |
    = note: you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
help: you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change
    |
361 -     async fn rollback(&self, rollback_token: String) -> Result<RemediationResult, String>;
361 +     fn rollback(&self, rollback_token: String) -> impl std::future::Future<Output = Result<RemediationResult, String>> + Send;
    |

warning: use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified
   --> src\remediation\mod.rs:362:5
    |
362 |     async fn get_status(&self, request_id: Uuid) -> Result<RemediationStatus, String>;
    |     ^^^^^
    |
    = note: you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
help: you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change
    |
362 -     async fn get_status(&self, request_id: Uuid) -> Result<RemediationStatus, String>;
362 +     fn get_status(&self, request_id: Uuid) -> impl std::future::Future<Output = Result<RemediationStatus, String>> + Send;
    |

warning: `policycortex-core` (lib) generated 309 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 06s
warning: the following packages contain code that will be rejected by a future version of Rust: redis v0.24.0, sqlx-postgres v0.7.4
note: to see what the problems were, use the option `--future-incompat-report`, or run `cargo report future-incompatibilities --id 