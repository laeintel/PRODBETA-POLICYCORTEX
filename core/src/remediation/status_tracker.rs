// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Remediation Status Tracker
// Tracks real-time status and progress of remediation operations

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug)]
pub struct RemediationTracker {
    active_remediations: Arc<RwLock<HashMap<Uuid, RemediationStatusDetail>>>,
    status_history: Arc<RwLock<Vec<StatusHistoryEntry>>>,
    progress_broadcasters: Arc<RwLock<HashMap<Uuid, broadcast::Sender<StatusUpdate>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub id: Uuid,
    pub status: RemediationStatus,
    pub current_step: String,
    pub total_steps: u32,
    pub percentage: f64,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationStatusDetail {
    pub remediation_id: Uuid,
    pub status: RemediationStatus,
    pub current_step: String,
    pub total_steps: u32,
    pub completed_steps: u32,
    pub percentage: f64,
    pub started_at: DateTime<Utc>,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub last_updated: DateTime<Utc>,
    pub steps: Vec<StepStatus>,
    pub logs: Vec<LogEntry>,
    pub metrics: RemediationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStatus {
    pub step_id: String,
    pub step_name: String,
    pub status: StepExecutionStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub error: Option<String>,
    pub output: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    WaitingForApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub step_id: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationMetrics {
    pub execution_time_ms: u64,
    pub api_calls_made: u32,
    pub resources_modified: u32,
    pub rollbacks_available: u32,
    pub approval_wait_time_ms: u64,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusHistoryEntry {
    pub remediation_id: Uuid,
    pub old_status: RemediationStatus,
    pub new_status: RemediationStatus,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub step_id: Option<String>,
}

impl RemediationTracker {
    pub fn new() -> Self {
        Self {
            active_remediations: Arc::new(RwLock::new(HashMap::new())),
            status_history: Arc::new(RwLock::new(Vec::new())),
            progress_broadcasters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_tracking(&self, remediation_id: Uuid, total_steps: u32) -> Result<(), String> {
        let status = RemediationStatusDetail {
            remediation_id,
            status: RemediationStatus::Pending,
            current_step: "Initializing".to_string(),
            total_steps,
            completed_steps: 0,
            percentage: 0.0,
            started_at: Utc::now(),
            estimated_completion: None,
            last_updated: Utc::now(),
            steps: Vec::new(),
            logs: Vec::new(),
            metrics: RemediationMetrics {
                execution_time_ms: 0,
                api_calls_made: 0,
                resources_modified: 0,
                rollbacks_available: 0,
                approval_wait_time_ms: 0,
                retry_count: 0,
            },
        };

        // Create broadcast channel for this remediation
        let (tx, _) = broadcast::channel::<StatusUpdate>(100);
        self.progress_broadcasters.write().await.insert(remediation_id, tx.clone());

        // Store initial status
        self.active_remediations.write().await.insert(remediation_id, status);

        // Send initial status update
        let initial_update = StatusUpdate {
            id: remediation_id,
            status: RemediationStatus::Pending,
            current_step: "Initializing".to_string(),
            total_steps,
            percentage: 0.0,
            estimated_completion: None,
            message: "Remediation started".to_string(),
            timestamp: Utc::now(),
        };

        let _ = tx.send(initial_update);

        tracing::info!("Started tracking remediation {}", remediation_id);
        Ok(())
    }

    pub async fn update_status(
        &self,
        remediation_id: Uuid,
        new_status: RemediationStatus,
        step_name: &str,
        completed_steps: u32,
        message: &str,
    ) -> Result<(), String> {
        let mut remediations = self.active_remediations.write().await;
        
        let status_detail = remediations.get_mut(&remediation_id)
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))?;

        let old_status = status_detail.status.clone();
        
        // Update status
        status_detail.status = new_status.clone();
        status_detail.current_step = step_name.to_string();
        status_detail.completed_steps = completed_steps;
        status_detail.percentage = (completed_steps as f64 / status_detail.total_steps as f64) * 100.0;
        status_detail.last_updated = Utc::now();

        // Calculate estimated completion
        if completed_steps > 0 && status_detail.status == RemediationStatus::InProgress {
            let elapsed = Utc::now().signed_duration_since(status_detail.started_at);
            let estimated_total = elapsed.num_milliseconds() as f64 / (completed_steps as f64 / status_detail.total_steps as f64);
            status_detail.estimated_completion = Some(
                status_detail.started_at + chrono::Duration::milliseconds(estimated_total as i64)
            );
        }

        // Add log entry
        status_detail.logs.push(LogEntry {
            timestamp: Utc::now(),
            level: if matches!(new_status, RemediationStatus::Failed) { LogLevel::Error } else { LogLevel::Info },
            message: message.to_string(),
            step_id: Some(step_name.to_string()),
            metadata: None,
        });

        // Record status change in history
        if old_status != new_status {
            self.status_history.write().await.push(StatusHistoryEntry {
                remediation_id,
                old_status,
                new_status: new_status.clone(),
                timestamp: Utc::now(),
                reason: message.to_string(),
                step_id: Some(step_name.to_string()),
            });
        }

        // Broadcast status update
        if let Some(tx) = self.progress_broadcasters.read().await.get(&remediation_id) {
            let update = StatusUpdate {
                id: remediation_id,
                status: new_status,
                current_step: step_name.to_string(),
                total_steps: status_detail.total_steps,
                percentage: status_detail.percentage,
                estimated_completion: status_detail.estimated_completion,
                message: message.to_string(),
                timestamp: Utc::now(),
            };

            let _ = tx.send(update);
        }

        Ok(())
    }

    pub async fn update_step_status(
        &self,
        remediation_id: Uuid,
        step_id: &str,
        step_name: &str,
        status: StepExecutionStatus,
        error: Option<String>,
        output: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let mut remediations = self.active_remediations.write().await;
        
        let status_detail = remediations.get_mut(&remediation_id)
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))?;

        // Find existing step or create new one
        let step_index = status_detail.steps.iter().position(|s| s.step_id == step_id);
        
        let now = Utc::now();
        
        if let Some(index) = step_index {
            let step = &mut status_detail.steps[index];
            let old_status = step.status.clone();
            
            step.status = status.clone();
            
            if matches!(old_status, StepExecutionStatus::Pending) && 
               matches!(status, StepExecutionStatus::Running) {
                step.started_at = Some(now);
            }
            
            if matches!(status, StepExecutionStatus::Completed | StepExecutionStatus::Failed) {
                step.completed_at = Some(now);
                if let Some(started) = step.started_at {
                    step.duration_ms = Some(now.signed_duration_since(started).num_milliseconds() as u64);
                }
            }
            
            step.error = error;
            step.output = output;
        } else {
            // Create new step
            let step = StepStatus {
                step_id: step_id.to_string(),
                step_name: step_name.to_string(),
                status: status.clone(),
                started_at: if matches!(status, StepExecutionStatus::Running) { Some(now) } else { None },
                completed_at: if matches!(status, StepExecutionStatus::Completed | StepExecutionStatus::Failed) { Some(now) } else { None },
                duration_ms: None,
                error,
                output,
            };
            
            status_detail.steps.push(step);
        }

        Ok(())
    }

    pub async fn track_progress(&self, remediation_id: Uuid) -> Result<StatusUpdate, String> {
        let remediations = self.active_remediations.read().await;
        
        let status = remediations.get(&remediation_id)
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))?;

        Ok(StatusUpdate {
            id: remediation_id,
            status: status.status.clone(),
            current_step: status.current_step.clone(),
            total_steps: status.total_steps,
            percentage: status.percentage,
            estimated_completion: status.estimated_completion,
            message: format!("Step {} of {}", status.completed_steps, status.total_steps),
            timestamp: status.last_updated,
        })
    }

    pub async fn get_detailed_status(&self, remediation_id: Uuid) -> Result<RemediationStatusDetail, String> {
        let remediations = self.active_remediations.read().await;
        
        remediations.get(&remediation_id)
            .cloned()
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))
    }

    pub async fn subscribe_to_updates(&self, remediation_id: Uuid) -> Result<broadcast::Receiver<StatusUpdate>, String> {
        let broadcasters = self.progress_broadcasters.read().await;
        
        broadcasters.get(&remediation_id)
            .map(|tx| tx.subscribe())
            .ok_or_else(|| format!("No tracker found for remediation {}", remediation_id))
    }

    pub async fn complete_tracking(&self, remediation_id: Uuid, final_status: RemediationStatus) -> Result<RemediationStatusDetail, String> {
        let mut remediations = self.active_remediations.write().await;
        
        if let Some(mut status) = remediations.remove(&remediation_id) {
            status.status = final_status.clone();
            status.last_updated = Utc::now();
            status.percentage = if matches!(final_status, RemediationStatus::Completed) { 100.0 } else { status.percentage };
            
            // Calculate final metrics
            status.metrics.execution_time_ms = Utc::now()
                .signed_duration_since(status.started_at)
                .num_milliseconds() as u64;

            // Add final log entry
            status.logs.push(LogEntry {
                timestamp: Utc::now(),
                level: match final_status {
                    RemediationStatus::Completed => LogLevel::Info,
                    RemediationStatus::Failed => LogLevel::Error,
                    _ => LogLevel::Warning,
                },
                message: format!("Remediation completed with status: {:?}", final_status),
                step_id: None,
                metadata: None,
            });

            // Send final status update
            if let Some(tx) = self.progress_broadcasters.write().await.remove(&remediation_id) {
                let final_update = StatusUpdate {
                    id: remediation_id,
                    status: final_status.clone(),
                    current_step: "Completed".to_string(),
                    total_steps: status.total_steps,
                    percentage: status.percentage,
                    estimated_completion: None,
                    message: "Remediation finished".to_string(),
                    timestamp: Utc::now(),
                };

                let _ = tx.send(final_update);
            }

            tracing::info!("Completed tracking for remediation {} with status {:?}", remediation_id, final_status);
            Ok(status)
        } else {
            Err(format!("Remediation {} not found", remediation_id))
        }
    }

    pub async fn list_active_remediations(&self) -> Vec<(Uuid, RemediationStatus)> {
        self.active_remediations.read().await
            .iter()
            .map(|(id, status)| (*id, status.status.clone()))
            .collect()
    }

    pub async fn get_status_history(&self, remediation_id: Option<Uuid>) -> Vec<StatusHistoryEntry> {
        let history = self.status_history.read().await;
        
        if let Some(id) = remediation_id {
            history.iter()
                .filter(|entry| entry.remediation_id == id)
                .cloned()
                .collect()
        } else {
            history.clone()
        }
    }

    pub async fn cleanup_completed(&self, older_than_hours: i64) -> usize {
        let mut remediations = self.active_remediations.write().await;
        let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours);
        
        let before_count = remediations.len();
        
        remediations.retain(|_, status| {
            !matches!(
                status.status,
                RemediationStatus::Completed | RemediationStatus::Failed | RemediationStatus::Cancelled
            ) || status.last_updated > cutoff
        });
        
        before_count - remediations.len()
    }

    pub async fn add_log_entry(
        &self,
        remediation_id: Uuid,
        level: LogLevel,
        message: &str,
        step_id: Option<&str>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<(), String> {
        let mut remediations = self.active_remediations.write().await;
        
        let status = remediations.get_mut(&remediation_id)
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))?;

        status.logs.push(LogEntry {
            timestamp: Utc::now(),
            level,
            message: message.to_string(),
            step_id: step_id.map(|s| s.to_string()),
            metadata,
        });

        Ok(())
    }

    pub async fn update_metrics(
        &self,
        remediation_id: Uuid,
        api_calls: Option<u32>,
        resources_modified: Option<u32>,
        retry_count: Option<u32>,
    ) -> Result<(), String> {
        let mut remediations = self.active_remediations.write().await;
        
        let status = remediations.get_mut(&remediation_id)
            .ok_or_else(|| format!("Remediation {} not found", remediation_id))?;

        if let Some(calls) = api_calls {
            status.metrics.api_calls_made += calls;
        }
        
        if let Some(resources) = resources_modified {
            status.metrics.resources_modified += resources;
        }
        
        if let Some(retries) = retry_count {
            status.metrics.retry_count += retries;
        }

        Ok(())
    }

    pub async fn get_summary_statistics(&self) -> RemediationStatistics {
        let remediations = self.active_remediations.read().await;
        let history = self.status_history.read().await;
        
        let mut stats = RemediationStatistics {
            total_active: remediations.len(),
            completed_today: 0,
            failed_today: 0,
            average_execution_time_ms: 0,
            success_rate_percentage: 0.0,
            status_breakdown: HashMap::new(),
        };

        let today = Utc::now().date_naive();
        
        // Count completed/failed today from history
        for entry in history.iter() {
            if entry.timestamp.date_naive() == today {
                match entry.new_status {
                    RemediationStatus::Completed => stats.completed_today += 1,
                    RemediationStatus::Failed => stats.failed_today += 1,
                    _ => {}
                }
            }
        }

        // Status breakdown
        for status in remediations.values() {
            *stats.status_breakdown.entry(format!("{:?}", status.status)).or_insert(0) += 1;
        }

        // Calculate average execution time for completed remediations
        let completed: Vec<_> = remediations.values()
            .filter(|s| matches!(s.status, RemediationStatus::Completed))
            .collect();
        
        if !completed.is_empty() {
            stats.average_execution_time_ms = completed.iter()
                .map(|s| s.metrics.execution_time_ms)
                .sum::<u64>() / completed.len() as u64;
        }

        // Calculate success rate
        let total_finished = stats.completed_today + stats.failed_today;
        if total_finished > 0 {
            stats.success_rate_percentage = (stats.completed_today as f64 / total_finished as f64) * 100.0;
        }

        stats
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationStatistics {
    pub total_active: usize,
    pub completed_today: u32,
    pub failed_today: u32,
    pub average_execution_time_ms: u64,
    pub success_rate_percentage: f64,
    pub status_breakdown: HashMap<String, u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_remediation_tracking() {
        let tracker = RemediationTracker::new();
        let remediation_id = Uuid::new_v4();
        
        // Start tracking
        tracker.start_tracking(remediation_id, 3).await.unwrap();
        
        // Update status
        tracker.update_status(
            remediation_id,
            RemediationStatus::InProgress,
            "Step 1",
            1,
            "Processing first step"
        ).await.unwrap();
        
        // Get status
        let status = tracker.track_progress(remediation_id).await.unwrap();
        assert_eq!(status.current_step, "Step 1");
        assert!(status.percentage > 0.0);
        
        // Complete tracking
        let final_status = tracker.complete_tracking(remediation_id, RemediationStatus::Completed).await.unwrap();
        assert_eq!(final_status.status, RemediationStatus::Completed);
    }

    #[tokio::test]
    async fn test_step_status_tracking() {
        let tracker = RemediationTracker::new();
        let remediation_id = Uuid::new_v4();
        
        tracker.start_tracking(remediation_id, 2).await.unwrap();
        
        // Update step status
        tracker.update_step_status(
            remediation_id,
            "step1",
            "Validation",
            StepExecutionStatus::Running,
            None,
            None
        ).await.unwrap();
        
        tracker.update_step_status(
            remediation_id,
            "step1",
            "Validation",
            StepExecutionStatus::Completed,
            None,
            Some(serde_json::json!({"result": "success"}))
        ).await.unwrap();
        
        let status = tracker.get_detailed_status(remediation_id).await.unwrap();
        assert_eq!(status.steps.len(), 1);
        assert_eq!(status.steps[0].status, StepExecutionStatus::Completed);
    }

    #[tokio::test]
    async fn test_status_updates_subscription() {
        let tracker = RemediationTracker::new();
        let remediation_id = Uuid::new_v4();
        
        tracker.start_tracking(remediation_id, 1).await.unwrap();
        
        let mut receiver = tracker.subscribe_to_updates(remediation_id).await.unwrap();
        
        // Update status (should trigger broadcast)
        tracker.update_status(
            remediation_id,
            RemediationStatus::InProgress,
            "Test Step",
            1,
            "Test message"
        ).await.unwrap();
        
        // Should receive the update
        let update = receiver.recv().await.unwrap();
        assert_eq!(update.current_step, "Test Step");
        assert_eq!(update.status, RemediationStatus::InProgress);
    }
}