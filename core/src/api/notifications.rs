// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Notification API Endpoints
// Manages notification channels, templates, and delivery for approval workflows

use crate::api::{AppState, ApiError};
use crate::auth::{AuthUser, TokenValidator};
use crate::remediation::notification_system::*;
use axum::{
    extract::{Path, Query, State},
    response::{IntoResponse, Json},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct SendNotificationRequest {
    pub event_type: NotificationEventType,
    pub priority: NotificationPriority,
    pub recipients: Vec<NotificationRecipient>,
    pub subject: String,
    pub message: String,
    pub html_message: Option<String>,
    pub data: HashMap<String, serde_json::Value>,
    pub channels: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SendNotificationResponse {
    pub notification_id: Uuid,
    pub status: String,
    pub total_recipients: u32,
    pub successful_deliveries: u32,
    pub failed_deliveries: u32,
    pub channel_results: Vec<ChannelResult>,
}

#[derive(Debug, Deserialize)]
pub struct CreateChannelRequest {
    pub channel_type: NotificationChannelType,
    pub config: ChannelConfig,
    pub enabled: bool,
    pub priority: u8,
    pub rate_limit: RateLimit,
}

#[derive(Debug, Serialize)]
pub struct CreateChannelResponse {
    pub channel_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct NotificationHistoryQuery {
    pub limit: Option<usize>,
    pub event_type: Option<NotificationEventType>,
    pub priority: Option<NotificationPriority>,
    pub status: Option<NotificationStatus>,
}

/// Send a notification
pub async fn send_notification(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(request): Json<SendNotificationRequest>,
) -> impl IntoResponse {
    // Verify user has permission to send notifications
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Notify"]) {
        return ApiError::Forbidden("Insufficient permissions to send notifications".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    let notification_request = NotificationRequest {
        notification_id: Uuid::new_v4(),
        event_type: request.event_type,
        priority: request.priority,
        recipients: request.recipients,
        subject: request.subject,
        message: request.message,
        html_message: request.html_message,
        data: request.data,
        channels: request.channels,
        scheduled_at: None,
        expires_at: None,
    };

    match notification_system.send_notification(notification_request).await {
        Ok(result) => {
            Json(SendNotificationResponse {
                notification_id: result.notification_id,
                status: format!("{:?}", result.status),
                total_recipients: result.total_recipients,
                successful_deliveries: result.successful_deliveries,
                failed_deliveries: result.failed_deliveries,
                channel_results: result.channel_results,
            }).into_response()
        }
        Err(e) => {
            ApiError::BadRequest(format!("Failed to send notification: {}", e))
                .into_response()
        }
    }
}

/// Create a new notification channel
pub async fn create_channel(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateChannelRequest>,
) -> impl IntoResponse {
    // Verify user has admin permissions
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Admin"]) {
        return ApiError::Forbidden("Insufficient permissions to manage notification channels".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    let channel_id = format!("{}-{}", 
        format!("{:?}", request.channel_type).to_lowercase(),
        Uuid::new_v4().to_string()[..8].to_string()
    );

    let channel = NotificationChannel {
        channel_id: channel_id.clone(),
        channel_type: request.channel_type,
        config: request.config,
        enabled: request.enabled,
        priority: request.priority,
        rate_limit: request.rate_limit,
        retry_policy: RetryPolicy::default(),
    };

    match notification_system.add_channel(channel).await {
        Ok(_) => {
            Json(CreateChannelResponse {
                channel_id,
                status: "created".to_string(),
                message: "Notification channel created successfully".to_string(),
            }).into_response()
        }
        Err(e) => {
            ApiError::BadRequest(format!("Failed to create channel: {}", e))
                .into_response()
        }
    }
}

/// List all notification channels
pub async fn list_channels(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Verify user has permission to view channels
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.View"]) {
        return ApiError::Forbidden("Insufficient permissions to view notification channels".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    let channels = notification_system.list_channels().await;
    Json(channels).into_response()
}

/// Delete a notification channel
pub async fn delete_channel(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Path(channel_id): Path<String>,
) -> impl IntoResponse {
    // Verify user has admin permissions
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Admin"]) {
        return ApiError::Forbidden("Insufficient permissions to delete notification channels".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    match notification_system.remove_channel(&channel_id).await {
        Ok(_) => {
            Json(serde_json::json!({
                "status": "deleted",
                "message": "Notification channel deleted successfully"
            })).into_response()
        }
        Err(e) => {
            ApiError::BadRequest(format!("Failed to delete channel: {}", e))
                .into_response()
        }
    }
}

/// Get notification history
pub async fn get_notification_history(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Query(query): Query<NotificationHistoryQuery>,
) -> impl IntoResponse {
    // Verify user has permission to view notification history
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.View"]) {
        return ApiError::Forbidden("Insufficient permissions to view notification history".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    let mut history = notification_system.get_notification_history(query.limit).await;

    // Apply filters
    if let Some(event_type) = query.event_type {
        history.retain(|entry| entry.event_type == event_type);
    }

    if let Some(priority) = query.priority {
        history.retain(|entry| entry.priority == priority);
    }

    if let Some(status) = query.status {
        history.retain(|entry| entry.status == status);
    }

    Json(history).into_response()
}

/// Test notification channel
pub async fn test_channel(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Path(channel_id): Path<String>,
) -> impl IntoResponse {
    // Verify user has admin permissions
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.Admin"]) {
        return ApiError::Forbidden("Insufficient permissions to test notification channels".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    // Create a test notification
    let test_request = NotificationRequest {
        notification_id: Uuid::new_v4(),
        event_type: NotificationEventType::SystemAlert,
        priority: NotificationPriority::Low,
        recipients: vec![NotificationRecipient {
            recipient_type: RecipientType::Email,
            identifier: "test@example.com".to_string(),
            name: Some("Test User".to_string()),
            preferences: None,
        }],
        subject: "Test Notification".to_string(),
        message: "This is a test notification from PolicyCortex".to_string(),
        html_message: None,
        data: HashMap::from([
            ("test".to_string(), serde_json::Value::Bool(true)),
            ("timestamp".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339())),
        ]),
        channels: vec![channel_id.clone()],
        scheduled_at: None,
        expires_at: None,
    };

    match notification_system.send_notification(test_request).await {
        Ok(result) => {
            Json(serde_json::json!({
                "status": "test_sent",
                "channel_id": channel_id,
                "notification_id": result.notification_id,
                "successful_deliveries": result.successful_deliveries,
                "failed_deliveries": result.failed_deliveries,
                "channel_results": result.channel_results
            })).into_response()
        }
        Err(e) => {
            ApiError::BadRequest(format!("Failed to send test notification: {}", e))
                .into_response()
        }
    }
}

/// Get notification statistics
pub async fn get_notification_stats(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Verify user has permission to view stats
    if !TokenValidator::new().check_permissions(&auth_user.claims, &["PolicyCortex.View"]) {
        return ApiError::Forbidden("Insufficient permissions to view notification statistics".to_string())
            .into_response();
    }

    let notification_system = match &state.notification_system {
        Some(system) => system,
        None => {
            return ApiError::ServiceUnavailable("Notification system not initialized".to_string())
                .into_response();
        }
    };

    let history = notification_system.get_notification_history(Some(1000)).await;
    let total_notifications = history.len();
    let successful_notifications = history.iter().filter(|h| h.status == NotificationStatus::Sent).count();
    let failed_notifications = history.iter().filter(|h| h.status == NotificationStatus::Failed).count();
    
    let channels = notification_system.list_channels().await;
    let active_channels = channels.iter().filter(|c| c.enabled).count();
    let total_channels = channels.len();

    // Calculate stats by event type
    let mut event_type_stats = HashMap::new();
    for entry in &history {
        *event_type_stats.entry(format!("{:?}", entry.event_type)).or_insert(0) += 1;
    }

    // Calculate stats by priority
    let mut priority_stats = HashMap::new();
    for entry in &history {
        *priority_stats.entry(format!("{:?}", entry.priority)).or_insert(0) += 1;
    }

    Json(serde_json::json!({
        "total_notifications": total_notifications,
        "successful_notifications": successful_notifications,
        "failed_notifications": failed_notifications,
        "success_rate": if total_notifications > 0 { 
            (successful_notifications as f64 / total_notifications as f64) * 100.0 
        } else { 
            0.0 
        },
        "active_channels": active_channels,
        "total_channels": total_channels,
        "event_type_breakdown": event_type_stats,
        "priority_breakdown": priority_stats,
        "recent_notifications": history.into_iter().take(10).collect::<Vec<_>>()
    })).into_response()
}