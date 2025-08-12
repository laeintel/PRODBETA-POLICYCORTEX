use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Idempotent Action Orchestrator with retry logic and state management
/// Ensures actions are executed exactly once with proper error handling
pub struct ActionOrchestrator {
    pending_actions: Arc<RwLock<HashMap<String, ActionState>>>,
    completed_actions: Arc<RwLock<HashMap<String, CompletedAction>>>,
    idempotency_cache: Arc<RwLock<HashMap<String, IdempotencyRecord>>>,
    retry_policy: RetryPolicy,
    db_pool: Option<sqlx::PgPool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: Uuid,
    pub action_type: ActionType,
    pub target_resource: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub idempotency_key: String,
    pub tenant_id: String,
    pub user_id: String,
    pub correlation_id: Option<Uuid>,
    pub timeout: Duration,
    pub requires_approval: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    CreateResource,
    UpdateResource,
    DeleteResource,
    ModifyPolicy,
    GrantAccess,
    RevokeAccess,
    RestartService,
    ScaleResource,
    ApplyConfiguration,
    ExecuteScript,
    Remediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionState {
    pub action: Action,
    pub status: ActionStatus,
    pub attempts: u32,
    pub last_error: Option<String>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub checkpoint: Option<ActionCheckpoint>,
    pub rollback_actions: Vec<Action>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    RollingBack,
    RolledBack,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCheckpoint {
    pub step: String,
    pub progress: f32,
    pub state_data: serde_json::Value,
    pub can_resume: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedAction {
    pub action: Action,
    pub result: ActionResult,
    pub execution_time: Duration,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionResult {
    Success(serde_json::Value),
    Failure(String),
    PartialSuccess(serde_json::Value, String),
    Cancelled,
}

#[derive(Debug, Clone)]
struct IdempotencyRecord {
    key: String,
    action_id: Uuid,
    result: Option<ActionResult>,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f32,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::seconds(1),
            max_delay: Duration::seconds(60),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

impl ActionOrchestrator {
    pub async fn new(db_pool: Option<sqlx::PgPool>) -> Self {
        Self {
            pending_actions: Arc::new(RwLock::new(HashMap::new())),
            completed_actions: Arc::new(RwLock::new(HashMap::new())),
            idempotency_cache: Arc::new(RwLock::new(HashMap::new())),
            retry_policy: RetryPolicy::default(),
            db_pool,
        }
    }

    /// Submit an action for execution with idempotency guarantee
    pub async fn submit_action(&self, mut action: Action) -> Result<Uuid, String> {
        // Generate idempotency key if not provided
        if action.idempotency_key.is_empty() {
            action.idempotency_key = self.generate_idempotency_key(&action);
        }

        // Check idempotency cache
        if let Some(existing) = self.check_idempotency(&action.idempotency_key).await? {
            info!(
                "Action with idempotency key {} already exists",
                action.idempotency_key
            );
            return Ok(existing.action_id);
        }

        // Validate action
        self.validate_action(&action)?;

        // Check if approval is required
        if action.requires_approval && !self.has_approval(&action).await? {
            return Err("Action requires approval".to_string());
        }

        // Store idempotency record
        self.store_idempotency_record(&action).await?;

        // Create action state
        let action_state = ActionState {
            action: action.clone(),
            status: ActionStatus::Pending,
            attempts: 0,
            last_error: None,
            started_at: Utc::now(),
            updated_at: Utc::now(),
            checkpoint: None,
            rollback_actions: self.generate_rollback_actions(&action),
        };

        // Store in pending actions
        let mut pending = self.pending_actions.write().await;
        pending.insert(action.idempotency_key.clone(), action_state);

        // Persist to database if available
        if let Some(ref pool) = self.db_pool {
            self.persist_action(pool, &action).await?;
        }

        // Execute action asynchronously
        let orchestrator = self.clone();
        let action_clone = action.clone();
        tokio::spawn(async move {
            orchestrator.execute_action(action_clone).await;
        });

        Ok(action.id)
    }

    /// Execute an action with retry logic and state management
    async fn execute_action(&self, action: Action) {
        let mut attempts = 0;
        let mut delay = self.retry_policy.initial_delay;

        loop {
            attempts += 1;
            info!(
                "Executing action {} (attempt {}/{})",
                action.id, attempts, self.retry_policy.max_attempts
            );

            // Update action state
            self.update_action_status(&action.idempotency_key, ActionStatus::Running)
                .await;

            // Execute based on action type
            let result = if action.dry_run {
                self.simulate_action(&action).await
            } else {
                self.execute_action_internal(&action).await
            };

            match result {
                Ok(action_result) => {
                    info!("Action {} completed successfully", action.id);
                    self.complete_action(action.clone(), action_result).await;
                    break;
                }
                Err(e) => {
                    error!("Action {} failed: {}", action.id, e);

                    // Update error state
                    self.update_action_error(&action.idempotency_key, e.clone())
                        .await;

                    // Check if we should retry
                    if attempts >= self.retry_policy.max_attempts {
                        error!("Action {} failed after {} attempts", action.id, attempts);
                        self.fail_action(action.clone(), e).await;
                        break;
                    }

                    // Check if error is retryable
                    if !self.is_retryable_error(&e) {
                        error!("Action {} failed with non-retryable error", action.id);
                        self.fail_action(action.clone(), e).await;
                        break;
                    }

                    // Wait before retry with exponential backoff
                    info!("Retrying action {} after {:?}", action.id, delay);
                    tokio::time::sleep(delay.to_std().unwrap()).await;

                    // Calculate next delay
                    delay = self.calculate_next_delay(delay);
                }
            }
        }
    }

    /// Execute the actual action based on its type
    async fn execute_action_internal(&self, action: &Action) -> Result<ActionResult, String> {
        match action.action_type {
            ActionType::CreateResource => self.create_resource(action).await,
            ActionType::UpdateResource => self.update_resource(action).await,
            ActionType::DeleteResource => self.delete_resource(action).await,
            ActionType::ModifyPolicy => self.modify_policy(action).await,
            ActionType::GrantAccess => self.grant_access(action).await,
            ActionType::RevokeAccess => self.revoke_access(action).await,
            ActionType::RestartService => self.restart_service(action).await,
            ActionType::ScaleResource => self.scale_resource(action).await,
            ActionType::ApplyConfiguration => self.apply_configuration(action).await,
            ActionType::ExecuteScript => self.execute_script(action).await,
            ActionType::Remediate => self.remediate(action).await,
        }
    }

    /// Simulate action execution for dry-run mode
    async fn simulate_action(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Simulating action {} in dry-run mode", action.id);

        // Simulate processing time
        tokio::time::sleep(Duration::seconds(1).to_std().unwrap()).await;

        // Return simulated success result
        Ok(ActionResult::Success(serde_json::json!({
            "simulated": true,
            "action_type": format!("{:?}", action.action_type),
            "target": action.target_resource,
            "message": "Action would be executed successfully"
        })))
    }

    /// Create a new resource
    async fn create_resource(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Creating resource: {}", action.target_resource);

        // TODO: Implement actual resource creation logic
        // This would integrate with Azure Resource Manager or other cloud providers

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": Uuid::new_v4().to_string(),
            "status": "created"
        })))
    }

    /// Update an existing resource
    async fn update_resource(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Updating resource: {}", action.target_resource);

        // TODO: Implement actual resource update logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "status": "updated"
        })))
    }

    /// Delete a resource
    async fn delete_resource(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Deleting resource: {}", action.target_resource);

        // TODO: Implement actual resource deletion logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "status": "deleted"
        })))
    }

    /// Modify policy settings
    async fn modify_policy(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Modifying policy for: {}", action.target_resource);

        // TODO: Implement actual policy modification logic

        Ok(ActionResult::Success(serde_json::json!({
            "policy_id": action.target_resource,
            "status": "modified"
        })))
    }

    /// Grant access to a resource
    async fn grant_access(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Granting access to: {}", action.target_resource);

        // TODO: Implement actual access grant logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "access_granted": true
        })))
    }

    /// Revoke access from a resource
    async fn revoke_access(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Revoking access from: {}", action.target_resource);

        // TODO: Implement actual access revocation logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "access_revoked": true
        })))
    }

    /// Restart a service
    async fn restart_service(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Restarting service: {}", action.target_resource);

        // TODO: Implement actual service restart logic

        Ok(ActionResult::Success(serde_json::json!({
            "service_id": action.target_resource,
            "status": "restarted"
        })))
    }

    /// Scale a resource
    async fn scale_resource(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Scaling resource: {}", action.target_resource);

        let scale_to = action
            .parameters
            .get("scale_to")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);

        // TODO: Implement actual scaling logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "scaled_to": scale_to
        })))
    }

    /// Apply configuration changes
    async fn apply_configuration(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Applying configuration to: {}", action.target_resource);

        // TODO: Implement actual configuration application logic

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "configuration_applied": true
        })))
    }

    /// Execute a script
    async fn execute_script(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Executing script on: {}", action.target_resource);

        // TODO: Implement actual script execution logic
        // This should include proper sandboxing and security measures

        Ok(ActionResult::Success(serde_json::json!({
            "target": action.target_resource,
            "script_executed": true
        })))
    }

    /// Remediate an issue
    async fn remediate(&self, action: &Action) -> Result<ActionResult, String> {
        info!("Remediating issue on: {}", action.target_resource);

        // TODO: Implement actual remediation logic based on issue type

        Ok(ActionResult::Success(serde_json::json!({
            "resource_id": action.target_resource,
            "remediated": true
        })))
    }

    /// Generate idempotency key from action parameters
    fn generate_idempotency_key(&self, action: &Action) -> String {
        let mut hasher = Sha256::new();
        hasher.update(action.action_type.to_string().as_bytes());
        hasher.update(action.target_resource.as_bytes());
        hasher.update(action.tenant_id.as_bytes());
        hasher.update(
            serde_json::to_string(&action.parameters)
                .unwrap_or_default()
                .as_bytes(),
        );
        format!("{:x}", hasher.finalize())
    }

    /// Check if an action with the given idempotency key exists
    async fn check_idempotency(&self, key: &str) -> Result<Option<IdempotencyRecord>, String> {
        let cache = self.idempotency_cache.read().await;

        if let Some(record) = cache.get(key) {
            if record.expires_at > Utc::now() {
                return Ok(Some(record.clone()));
            }
        }

        // Check database if available
        if let Some(ref pool) = self.db_pool {
            // TODO: Query database for idempotency record
        }

        Ok(None)
    }

    /// Store idempotency record
    async fn store_idempotency_record(&self, action: &Action) -> Result<(), String> {
        let record = IdempotencyRecord {
            key: action.idempotency_key.clone(),
            action_id: action.id,
            result: None,
            created_at: Utc::now(),
            expires_at: Utc::now() + Duration::hours(24),
        };

        let mut cache = self.idempotency_cache.write().await;
        cache.insert(action.idempotency_key.clone(), record);

        // Persist to database if available
        if let Some(ref pool) = self.db_pool {
            // TODO: Insert idempotency record into database
        }

        Ok(())
    }

    /// Validate action parameters
    fn validate_action(&self, action: &Action) -> Result<(), String> {
        if action.target_resource.is_empty() {
            return Err("Target resource cannot be empty".to_string());
        }

        if action.timeout < Duration::seconds(1) {
            return Err("Timeout must be at least 1 second".to_string());
        }

        // TODO: Add more validation based on action type

        Ok(())
    }

    /// Check if action has required approvals
    async fn has_approval(&self, action: &Action) -> Result<bool, String> {
        // TODO: Check approval system for this action
        // For now, return true for dry-run mode
        Ok(action.dry_run)
    }

    /// Generate rollback actions for an action
    fn generate_rollback_actions(&self, action: &Action) -> Vec<Action> {
        let mut rollback_actions = Vec::new();

        match action.action_type {
            ActionType::CreateResource => {
                // Rollback: Delete the created resource
                rollback_actions.push(Action {
                    id: Uuid::new_v4(),
                    action_type: ActionType::DeleteResource,
                    target_resource: action.target_resource.clone(),
                    parameters: HashMap::new(),
                    idempotency_key: format!("rollback-{}", action.idempotency_key),
                    tenant_id: action.tenant_id.clone(),
                    user_id: action.user_id.clone(),
                    correlation_id: Some(action.id),
                    timeout: action.timeout,
                    requires_approval: false,
                    dry_run: false,
                });
            }
            ActionType::DeleteResource => {
                // Rollback: Restore from backup if available
                // This would require storing resource state before deletion
            }
            ActionType::UpdateResource => {
                // Rollback: Restore previous configuration
                // This would require storing previous state
            }
            _ => {
                // Other action types may have specific rollback logic
            }
        }

        rollback_actions
    }

    /// Update action status
    async fn update_action_status(&self, key: &str, status: ActionStatus) {
        let mut pending = self.pending_actions.write().await;
        if let Some(action_state) = pending.get_mut(key) {
            action_state.status = status;
            action_state.updated_at = Utc::now();
        }
    }

    /// Update action error
    async fn update_action_error(&self, key: &str, error: String) {
        let mut pending = self.pending_actions.write().await;
        if let Some(action_state) = pending.get_mut(key) {
            action_state.last_error = Some(error);
            action_state.attempts += 1;
            action_state.updated_at = Utc::now();
        }
    }

    /// Complete an action successfully
    async fn complete_action(&self, action: Action, result: ActionResult) {
        // Remove from pending
        let mut pending = self.pending_actions.write().await;
        pending.remove(&action.idempotency_key);

        // Add to completed
        let completed = CompletedAction {
            action: action.clone(),
            result: result.clone(),
            execution_time: chrono::Duration::seconds(5), // Placeholder - actual timing would need proper implementation
            completed_at: Utc::now(),
        };

        let mut completed_actions = self.completed_actions.write().await;
        completed_actions.insert(action.idempotency_key.clone(), completed);

        // Update idempotency cache
        let mut cache = self.idempotency_cache.write().await;
        if let Some(record) = cache.get_mut(&action.idempotency_key) {
            record.result = Some(result);
        }
    }

    /// Mark an action as failed
    async fn fail_action(&self, action: Action, error: String) {
        self.complete_action(action, ActionResult::Failure(error))
            .await;
    }

    /// Check if an error is retryable
    fn is_retryable_error(&self, error: &str) -> bool {
        // Network errors, timeouts, and rate limits are retryable
        error.contains("timeout")
            || error.contains("network")
            || error.contains("rate limit")
            || error.contains("throttled")
            || error.contains("temporary")
    }

    /// Calculate next retry delay with exponential backoff
    fn calculate_next_delay(&self, current_delay: Duration) -> Duration {
        let next = current_delay.num_milliseconds() as f32 * self.retry_policy.exponential_base;
        let next_ms = next.min(self.retry_policy.max_delay.num_milliseconds() as f32) as i64;

        let mut delay = Duration::milliseconds(next_ms);

        // Add jitter if enabled
        if self.retry_policy.jitter {
            let jitter = rand::random::<f32>() * 0.3; // Up to 30% jitter
            delay =
                delay + Duration::milliseconds((delay.num_milliseconds() as f32 * jitter) as i64);
        }

        delay
    }

    /// Persist action to database
    async fn persist_action(&self, pool: &sqlx::PgPool, action: &Action) -> Result<(), String> {
        // TODO: Implement database persistence
        Ok(())
    }

    /// Get action status by ID
    pub async fn get_action_status(&self, action_id: Uuid) -> Option<ActionStatus> {
        // Check pending actions
        let pending = self.pending_actions.read().await;
        for (_, state) in pending.iter() {
            if state.action.id == action_id {
                return Some(state.status.clone());
            }
        }

        // Check completed actions
        let completed = self.completed_actions.read().await;
        for (_, completed_action) in completed.iter() {
            if completed_action.action.id == action_id {
                return Some(ActionStatus::Completed);
            }
        }

        None
    }

    /// Rollback an action
    pub async fn rollback_action(&self, action_id: Uuid) -> Result<(), String> {
        // Find the action
        let pending = self.pending_actions.read().await;
        let action_state = pending
            .values()
            .find(|s| s.action.id == action_id)
            .ok_or("Action not found")?
            .clone();

        drop(pending);

        // Execute rollback actions
        for rollback_action in action_state.rollback_actions {
            self.submit_action(rollback_action).await?;
        }

        Ok(())
    }
}

impl Clone for ActionOrchestrator {
    fn clone(&self) -> Self {
        Self {
            pending_actions: self.pending_actions.clone(),
            completed_actions: self.completed_actions.clone(),
            idempotency_cache: self.idempotency_cache.clone(),
            retry_policy: self.retry_policy.clone(),
            db_pool: self.db_pool.clone(),
        }
    }
}

impl ActionType {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}
