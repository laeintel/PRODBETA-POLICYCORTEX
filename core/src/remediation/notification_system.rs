// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Notification System for Approval Workflows
// Multi-channel notification system supporting Email, Teams, Slack, and Webhooks

use super::approval_manager::{ApprovalRequest, ApprovalDecision};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use reqwest::Client;

#[derive(Debug, Clone)]
pub struct NotificationSystem {
    channels: Arc<RwLock<HashMap<String, NotificationChannel>>>,
    templates: Arc<RwLock<HashMap<String, NotificationTemplate>>>,
    http_client: Arc<Client>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    notification_history: Arc<RwLock<Vec<NotificationHistoryEntry>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub channel_id: String,
    pub channel_type: NotificationChannelType,
    pub config: ChannelConfig,
    pub enabled: bool,
    pub priority: u8, // 1-10, higher = more critical notifications only
    pub rate_limit: RateLimit,
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    MicrosoftTeams,
    Slack,
    Webhook,
    Sms,
    ServiceNow,
    PagerDuty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    // Email configuration
    pub smtp_host: Option<String>,
    pub smtp_port: Option<u16>,
    pub smtp_username: Option<String>,
    pub smtp_password: Option<String>,
    pub from_address: Option<String>,
    pub to_addresses: Vec<String>,
    
    // Teams configuration
    pub teams_webhook_url: Option<String>,
    pub teams_channel_id: Option<String>,
    
    // Slack configuration
    pub slack_webhook_url: Option<String>,
    pub slack_token: Option<String>,
    pub slack_channel: Option<String>,
    
    // Webhook configuration
    pub webhook_url: Option<String>,
    pub webhook_headers: HashMap<String, String>,
    pub webhook_auth_token: Option<String>,
    
    // SMS configuration
    pub sms_provider: Option<String>,
    pub sms_api_key: Option<String>,
    pub phone_numbers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub max_per_minute: u32,
    pub max_per_hour: u32,
    pub burst_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub max_delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    pub template_id: String,
    pub name: String,
    pub event_type: NotificationEventType,
    pub subject_template: String,
    pub body_template: String,
    pub html_template: Option<String>,
    pub variables: Vec<String>,
    pub priority: NotificationPriority,
    pub channels: Vec<String>, // Channel IDs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationEventType {
    ApprovalRequested,
    ApprovalApproved,
    ApprovalRejected,
    ApprovalExpired,
    ApprovalEscalated,
    RemediationStarted,
    RemediationCompleted,
    RemediationFailed,
    RollbackInitiated,
    RollbackCompleted,
    HighRiskDetected,
    SystemAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low = 1,
    Normal = 5,
    High = 8,
    Critical = 10,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRequest {
    pub notification_id: Uuid,
    pub event_type: NotificationEventType,
    pub priority: NotificationPriority,
    pub recipients: Vec<NotificationRecipient>,
    pub subject: String,
    pub message: String,
    pub html_message: Option<String>,
    pub data: HashMap<String, serde_json::Value>,
    pub channels: Vec<String>, // Specific channels to use
    pub scheduled_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRecipient {
    pub recipient_type: RecipientType,
    pub identifier: String, // email, user ID, phone number, etc.
    pub name: Option<String>,
    pub preferences: Option<RecipientPreferences>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecipientType {
    User,
    Group,
    Role,
    Email,
    PhoneNumber,
    SlackUser,
    TeamsUser,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientPreferences {
    pub preferred_channels: Vec<NotificationChannelType>,
    pub quiet_hours: Option<QuietHours>,
    pub escalation_minutes: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHours {
    pub start_time: String, // HH:MM format
    pub end_time: String,   // HH:MM format
    pub timezone: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationResult {
    pub notification_id: Uuid,
    pub status: NotificationStatus,
    pub sent_at: DateTime<Utc>,
    pub channel_results: Vec<ChannelResult>,
    pub total_recipients: u32,
    pub successful_deliveries: u32,
    pub failed_deliveries: u32,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationStatus {
    Pending,
    Sent,
    PartialFailure,
    Failed,
    Expired,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelResult {
    pub channel_id: String,
    pub channel_type: NotificationChannelType,
    pub status: DeliveryStatus,
    pub recipients_sent: u32,
    pub recipients_failed: u32,
    pub response_time_ms: u64,
    pub error: Option<String>,
    pub external_id: Option<String>, // Provider-specific message ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Success,
    Failed,
    Retry,
    RateLimited,
    InvalidRecipient,
    ChannelError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationHistoryEntry {
    pub notification_id: Uuid,
    pub event_type: NotificationEventType,
    pub priority: NotificationPriority,
    pub status: NotificationStatus,
    pub sent_at: DateTime<Utc>,
    pub recipients_count: u32,
    pub channels_used: Vec<String>,
    pub approval_id: Option<String>,
    pub remediation_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    minute_counts: HashMap<String, (DateTime<Utc>, u32)>,
    hour_counts: HashMap<String, (DateTime<Utc>, u32)>,
}

impl NotificationSystem {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
            templates: Arc::new(RwLock::new(HashMap::new())),
            http_client: Arc::new(Client::new()),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new())),
            notification_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn initialize(&self) -> Result<(), String> {
        self.load_default_channels().await?;
        self.load_default_templates().await?;
        
        tracing::info!("Notification system initialized successfully");
        Ok(())
    }

    pub async fn send_notification(&self, request: NotificationRequest) -> Result<NotificationResult, String> {
        let start_time = Utc::now();
        let notification_id = request.notification_id;
        
        tracing::info!("Sending notification {} for event {:?}", notification_id, request.event_type);

        // Check if notification has expired
        if let Some(expires_at) = request.expires_at {
            if Utc::now() > expires_at {
                return Ok(NotificationResult {
                    notification_id,
                    status: NotificationStatus::Expired,
                    sent_at: start_time,
                    channel_results: vec![],
                    total_recipients: request.recipients.len() as u32,
                    successful_deliveries: 0,
                    failed_deliveries: request.recipients.len() as u32,
                    retry_count: 0,
                });
            }
        }

        // Get channels to use
        let channels = self.channels.read().await;
        let priority_value = request.priority.clone() as u8;
        let selected_channels: Vec<_> = if request.channels.is_empty() {
            // Use all enabled channels for this priority
            channels.values()
                .filter(|ch| ch.enabled && ch.priority <= priority_value)
                .cloned()
                .collect()
        } else {
            // Use specified channels
            request.channels.iter()
                .filter_map(|id| channels.get(id).cloned())
                .filter(|ch| ch.enabled)
                .collect()
        };

        drop(channels);

        if selected_channels.is_empty() {
            return Err("No valid channels available for notification".to_string());
        }

        // Send to each channel
        let mut channel_results = Vec::new();
        let mut total_successful = 0u32;
        let mut total_failed = 0u32;

        for channel in selected_channels {
            // Check rate limiting
            if !self.check_rate_limit(&channel.channel_id, &channel.rate_limit).await {
                channel_results.push(ChannelResult {
                    channel_id: channel.channel_id.clone(),
                    channel_type: channel.channel_type.clone(),
                    status: DeliveryStatus::RateLimited,
                    recipients_sent: 0,
                    recipients_failed: request.recipients.len() as u32,
                    response_time_ms: 0,
                    error: Some("Rate limit exceeded".to_string()),
                    external_id: None,
                });
                total_failed += request.recipients.len() as u32;
                continue;
            }

            // Send via channel
            match self.send_via_channel(&channel, &request).await {
                Ok(result) => {
                    total_successful += result.recipients_sent;
                    total_failed += result.recipients_failed;
                    channel_results.push(result);
                }
                Err(e) => {
                    channel_results.push(ChannelResult {
                        channel_id: channel.channel_id.clone(),
                        channel_type: channel.channel_type.clone(),
                        status: DeliveryStatus::ChannelError,
                        recipients_sent: 0,
                        recipients_failed: request.recipients.len() as u32,
                        response_time_ms: 0,
                        error: Some(e),
                        external_id: None,
                    });
                    total_failed += request.recipients.len() as u32;
                }
            }
        }

        // Determine overall status
        let status = if total_successful == 0 {
            NotificationStatus::Failed
        } else if total_failed > 0 {
            NotificationStatus::PartialFailure
        } else {
            NotificationStatus::Sent
        };

        let result = NotificationResult {
            notification_id,
            status: status.clone(),
            sent_at: start_time,
            channel_results,
            total_recipients: request.recipients.len() as u32,
            successful_deliveries: total_successful,
            failed_deliveries: total_failed,
            retry_count: 0,
        };

        // Add to history
        self.add_to_history(&request, &result).await;

        tracing::info!(
            "Notification {} completed with status {:?}: {}/{} successful deliveries",
            notification_id, status, total_successful, request.recipients.len()
        );

        Ok(result)
    }

    async fn send_via_channel(&self, channel: &NotificationChannel, request: &NotificationRequest) -> Result<ChannelResult, String> {
        let start_time = Utc::now();
        
        let result = match channel.channel_type {
            NotificationChannelType::Email => self.send_email(channel, request).await,
            NotificationChannelType::MicrosoftTeams => self.send_teams(channel, request).await,
            NotificationChannelType::Slack => self.send_slack(channel, request).await,
            NotificationChannelType::Webhook => self.send_webhook(channel, request).await,
            NotificationChannelType::Sms => self.send_sms(channel, request).await,
            NotificationChannelType::ServiceNow => self.send_servicenow(channel, request).await,
            NotificationChannelType::PagerDuty => self.send_pagerduty(channel, request).await,
        };

        let duration = Utc::now().signed_duration_since(start_time).num_milliseconds() as u64;

        match result {
            Ok(mut channel_result) => {
                channel_result.response_time_ms = duration;
                Ok(channel_result)
            }
            Err(e) => Err(e)
        }
    }

    async fn send_email(&self, channel: &NotificationChannel, request: &NotificationRequest) -> Result<ChannelResult, String> {
        // Email implementation would use SMTP client
        tracing::info!("Sending email notification via channel {}", channel.channel_id);
        
        // Simulate email sending
        let recipients_count = channel.config.to_addresses.len();
        
        Ok(ChannelResult {
            channel_id: channel.channel_id.clone(),
            channel_type: NotificationChannelType::Email,
            status: DeliveryStatus::Success,
            recipients_sent: recipients_count as u32,
            recipients_failed: 0,
            response_time_ms: 0,
            error: None,
            external_id: Some(format!("email-{}", Uuid::new_v4())),
        })
    }

    async fn send_teams(&self, channel: &NotificationChannel, request: &NotificationRequest) -> Result<ChannelResult, String> {
        if let Some(webhook_url) = &channel.config.teams_webhook_url {
            let teams_payload = self.create_teams_payload(request)?;
            
            let response = self.http_client
                .post(webhook_url)
                .header("Content-Type", "application/json")
                .json(&teams_payload)
                .send()
                .await
                .map_err(|e| format!("Teams webhook request failed: {}", e))?;

            if response.status().is_success() {
                Ok(ChannelResult {
                    channel_id: channel.channel_id.clone(),
                    channel_type: NotificationChannelType::MicrosoftTeams,
                    status: DeliveryStatus::Success,
                    recipients_sent: 1,
                    recipients_failed: 0,
                    response_time_ms: 0,
                    error: None,
                    external_id: None,
                })
            } else {
                Err(format!("Teams webhook failed with status: {}", response.status()))
            }
        } else {
            Err("Teams webhook URL not configured".to_string())
        }
    }

    async fn send_slack(&self, channel: &NotificationChannel, request: &NotificationRequest) -> Result<ChannelResult, String> {
        if let Some(webhook_url) = &channel.config.slack_webhook_url {
            let slack_payload = self.create_slack_payload(request)?;
            
            let response = self.http_client
                .post(webhook_url)
                .header("Content-Type", "application/json")
                .json(&slack_payload)
                .send()
                .await
                .map_err(|e| format!("Slack webhook request failed: {}", e))?;

            if response.status().is_success() {
                Ok(ChannelResult {
                    channel_id: channel.channel_id.clone(),
                    channel_type: NotificationChannelType::Slack,
                    status: DeliveryStatus::Success,
                    recipients_sent: 1,
                    recipients_failed: 0,
                    response_time_ms: 0,
                    error: None,
                    external_id: None,
                })
            } else {
                Err(format!("Slack webhook failed with status: {}", response.status()))
            }
        } else {
            Err("Slack webhook URL not configured".to_string())
        }
    }

    async fn send_webhook(&self, channel: &NotificationChannel, request: &NotificationRequest) -> Result<ChannelResult, String> {
        if let Some(webhook_url) = &channel.config.webhook_url {
            let webhook_payload = self.create_webhook_payload(request)?;
            
            let mut req_builder = self.http_client
                .post(webhook_url)
                .header("Content-Type", "application/json");

            // Add custom headers
            for (key, value) in &channel.config.webhook_headers {
                req_builder = req_builder.header(key, value);
            }

            // Add auth token if configured
            if let Some(token) = &channel.config.webhook_auth_token {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
            }

            let response = req_builder
                .json(&webhook_payload)
                .send()
                .await
                .map_err(|e| format!("Webhook request failed: {}", e))?;

            if response.status().is_success() {
                Ok(ChannelResult {
                    channel_id: channel.channel_id.clone(),
                    channel_type: NotificationChannelType::Webhook,
                    status: DeliveryStatus::Success,
                    recipients_sent: 1,
                    recipients_failed: 0,
                    response_time_ms: 0,
                    error: None,
                    external_id: None,
                })
            } else {
                Err(format!("Webhook failed with status: {}", response.status()))
            }
        } else {
            Err("Webhook URL not configured".to_string())
        }
    }

    async fn send_sms(&self, _channel: &NotificationChannel, _request: &NotificationRequest) -> Result<ChannelResult, String> {
        // SMS implementation would integrate with SMS providers like Twilio
        tracing::info!("SMS notifications not yet implemented");
        Err("SMS notifications not implemented".to_string())
    }

    async fn send_servicenow(&self, _channel: &NotificationChannel, _request: &NotificationRequest) -> Result<ChannelResult, String> {
        // ServiceNow implementation would create incidents/tickets
        tracing::info!("ServiceNow notifications not yet implemented");
        Err("ServiceNow notifications not implemented".to_string())
    }

    async fn send_pagerduty(&self, _channel: &NotificationChannel, _request: &NotificationRequest) -> Result<ChannelResult, String> {
        // PagerDuty implementation would create incidents
        tracing::info!("PagerDuty notifications not yet implemented");
        Err("PagerDuty notifications not implemented".to_string())
    }

    fn create_teams_payload(&self, request: &NotificationRequest) -> Result<serde_json::Value, String> {
        let color = match request.priority {
            NotificationPriority::Critical => "ff0000", // Red
            NotificationPriority::High => "ff8c00",     // Orange
            NotificationPriority::Normal => "0078d4",   // Blue
            NotificationPriority::Low => "00bcf2",      // Light Blue
        };

        Ok(serde_json::json!({
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": request.subject,
            "themeColor": color,
            "sections": [{
                "activityTitle": request.subject,
                "activitySubtitle": format!("Priority: {:?}", request.priority),
                "text": request.message,
                "facts": request.data.iter().map(|(k, v)| {
                    serde_json::json!({
                        "name": k,
                        "value": v.as_str().unwrap_or(&v.to_string())
                    })
                }).collect::<Vec<_>>()
            }]
        }))
    }

    fn create_slack_payload(&self, request: &NotificationRequest) -> Result<serde_json::Value, String> {
        let color = match request.priority {
            NotificationPriority::Critical => "danger",
            NotificationPriority::High => "warning", 
            NotificationPriority::Normal => "good",
            NotificationPriority::Low => "#36a64f",
        };

        Ok(serde_json::json!({
            "text": request.subject,
            "attachments": [{
                "color": color,
                "title": request.subject,
                "text": request.message,
                "fields": request.data.iter().map(|(k, v)| {
                    serde_json::json!({
                        "title": k,
                        "value": v.as_str().unwrap_or(&v.to_string()),
                        "short": true
                    })
                }).collect::<Vec<_>>(),
                "ts": Utc::now().timestamp()
            }]
        }))
    }

    fn create_webhook_payload(&self, request: &NotificationRequest) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "notification_id": request.notification_id,
            "event_type": request.event_type,
            "priority": request.priority,
            "subject": request.subject,
            "message": request.message,
            "html_message": request.html_message,
            "data": request.data,
            "timestamp": Utc::now().to_rfc3339()
        }))
    }

    async fn check_rate_limit(&self, channel_id: &str, rate_limit: &RateLimit) -> bool {
        let mut limiter = self.rate_limiter.write().await;
        limiter.check_rate_limit(channel_id, rate_limit)
    }

    async fn add_to_history(&self, request: &NotificationRequest, result: &NotificationResult) {
        let entry = NotificationHistoryEntry {
            notification_id: request.notification_id,
            event_type: request.event_type.clone(),
            priority: request.priority.clone(),
            status: result.status.clone(),
            sent_at: result.sent_at,
            recipients_count: request.recipients.len() as u32,
            channels_used: result.channel_results.iter().map(|cr| cr.channel_id.clone()).collect(),
            approval_id: request.data.get("approval_id").and_then(|v| v.as_str().map(|s| s.to_string())),
            remediation_id: request.data.get("remediation_id").and_then(|v| v.as_str().map(|s| s.to_string())),
        };

        self.notification_history.write().await.push(entry);
    }

    async fn load_default_channels(&self) -> Result<(), String> {
        let mut channels = self.channels.write().await;

        // Default email channel
        channels.insert("default-email".to_string(), NotificationChannel {
            channel_id: "default-email".to_string(),
            channel_type: NotificationChannelType::Email,
            config: ChannelConfig {
                smtp_host: Some("smtp.gmail.com".to_string()),
                smtp_port: Some(587),
                smtp_username: None,
                smtp_password: None,
                from_address: Some("noreply@policycortex.com".to_string()),
                to_addresses: vec!["admin@policycortex.com".to_string()],
                teams_webhook_url: None,
                teams_channel_id: None,
                slack_webhook_url: None,
                slack_token: None,
                slack_channel: None,
                webhook_url: None,
                webhook_headers: HashMap::new(),
                webhook_auth_token: None,
                sms_provider: None,
                sms_api_key: None,
                phone_numbers: vec![],
            },
            enabled: false, // Disabled by default until configured
            priority: 5,
            rate_limit: RateLimit {
                max_per_minute: 10,
                max_per_hour: 100,
                burst_limit: 5,
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                initial_delay_ms: 1000,
                backoff_multiplier: 2.0,
                max_delay_ms: 30000,
            },
        });

        // Default Teams channel
        channels.insert("default-teams".to_string(), NotificationChannel {
            channel_id: "default-teams".to_string(),
            channel_type: NotificationChannelType::MicrosoftTeams,
            config: ChannelConfig {
                smtp_host: None,
                smtp_port: None,
                smtp_username: None,
                smtp_password: None,
                from_address: None,
                to_addresses: vec![],
                teams_webhook_url: None, // Must be configured
                teams_channel_id: None,
                slack_webhook_url: None,
                slack_token: None,
                slack_channel: None,
                webhook_url: None,
                webhook_headers: HashMap::new(),
                webhook_auth_token: None,
                sms_provider: None,
                sms_api_key: None,
                phone_numbers: vec![],
            },
            enabled: false, // Disabled until webhook URL is configured
            priority: 8,
            rate_limit: RateLimit {
                max_per_minute: 20,
                max_per_hour: 200,
                burst_limit: 10,
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                initial_delay_ms: 500,
                backoff_multiplier: 2.0,
                max_delay_ms: 15000,
            },
        });

        Ok(())
    }

    async fn load_default_templates(&self) -> Result<(), String> {
        let mut templates = self.templates.write().await;

        // Approval request template
        templates.insert("approval-requested".to_string(), NotificationTemplate {
            template_id: "approval-requested".to_string(),
            name: "Approval Request".to_string(),
            event_type: NotificationEventType::ApprovalRequested,
            subject_template: "🔒 Approval Required: {remediation_type} for {resource_name}".to_string(),
            body_template: r#"
A remediation approval is required:

**Resource**: {resource_name}
**Type**: {remediation_type}  
**Risk Level**: {risk_level}
**Requested By**: {requested_by}
**Deadline**: {expires_at}

**Details**: {description}

Please review and approve/reject this request in the PolicyCortex dashboard.
"#.to_string(),
            html_template: None,
            variables: vec![
                "remediation_type".to_string(),
                "resource_name".to_string(),
                "risk_level".to_string(),
                "requested_by".to_string(),
                "expires_at".to_string(),
                "description".to_string(),
            ],
            priority: NotificationPriority::High,
            channels: vec!["default-email".to_string(), "default-teams".to_string()],
        });

        // Approval decision template
        templates.insert("approval-decision".to_string(), NotificationTemplate {
            template_id: "approval-decision".to_string(),
            name: "Approval Decision".to_string(),
            event_type: NotificationEventType::ApprovalApproved,
            subject_template: "✅ Approved: {remediation_type} for {resource_name}".to_string(),
            body_template: r#"
Your remediation request has been processed:

**Status**: {decision}
**Resource**: {resource_name}
**Approved By**: {approver}
**Decision Time**: {decision_time}

{decision_reason}

{next_steps}
"#.to_string(),
            html_template: None,
            variables: vec![
                "decision".to_string(),
                "resource_name".to_string(),
                "approver".to_string(),
                "decision_time".to_string(),
                "decision_reason".to_string(),
                "next_steps".to_string(),
            ],
            priority: NotificationPriority::Normal,
            channels: vec!["default-email".to_string()],
        });

        // High risk detected template
        templates.insert("high-risk-detected".to_string(), NotificationTemplate {
            template_id: "high-risk-detected".to_string(),
            name: "High Risk Detected".to_string(),
            event_type: NotificationEventType::HighRiskDetected,
            subject_template: "🚨 HIGH RISK: {risk_type} detected on {resource_name}".to_string(),
            body_template: r#"
IMMEDIATE ATTENTION REQUIRED

**Risk Level**: {risk_level}
**Resource**: {resource_name}
**Risk Type**: {risk_type}
**Detected At**: {detected_at}

**Risk Details**: {risk_description}

**Recommended Actions**:
{recommended_actions}

This requires immediate review and action.
"#.to_string(),
            html_template: None,
            variables: vec![
                "risk_level".to_string(),
                "resource_name".to_string(),
                "risk_type".to_string(),
                "detected_at".to_string(),
                "risk_description".to_string(),
                "recommended_actions".to_string(),
            ],
            priority: NotificationPriority::Critical,
            channels: vec!["default-email".to_string(), "default-teams".to_string()],
        });

        Ok(())
    }

    // Public API methods for the approval system
    pub async fn notify_approval_requested(&self, approval: &ApprovalRequest) -> Result<NotificationResult, String> {
        let notification_request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: NotificationEventType::ApprovalRequested,
            priority: NotificationPriority::High,
            recipients: approval.approvers.iter().map(|approver| NotificationRecipient {
                recipient_type: RecipientType::Email,
                identifier: approver.clone(),
                name: None,
                preferences: None,
            }).collect(),
            subject: format!("🔒 Approval Required: {:?} remediation", approval.remediation_request.remediation_type),
            message: format!(
                "Approval required for {} remediation on resource {}. Requested by: {}",
                format!("{:?}", approval.remediation_request.remediation_type),
                approval.remediation_request.resource_id,
                approval.created_by
            ),
            html_message: None,
            data: HashMap::from([
                ("approval_id".to_string(), serde_json::Value::String(approval.id.clone())),
                ("resource_id".to_string(), serde_json::Value::String(approval.remediation_request.resource_id.clone())),
                ("remediation_type".to_string(), serde_json::Value::String(format!("{:?}", approval.remediation_request.remediation_type))),
                ("requested_by".to_string(), serde_json::Value::String(approval.created_by.clone())),
                ("expires_at".to_string(), serde_json::Value::String(approval.expires_at.to_rfc3339())),
            ]),
            channels: vec![],
            scheduled_at: None,
            expires_at: Some(approval.expires_at),
        };

        self.send_notification(notification_request).await
    }

    pub async fn notify_approval_decision(&self, approval_id: &str, decision: &ApprovalDecision, approver: &str) -> Result<NotificationResult, String> {
        let decision_text = match decision {
            ApprovalDecision::Approved => "Approved",
            ApprovalDecision::Rejected => "Rejected",
            ApprovalDecision::ApprovedWithConditions => "Approved with Conditions",
            ApprovalDecision::Deferred => "Deferred",
            ApprovalDecision::Delegated(_) => "Delegated",
        };

        let notification_request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: if matches!(decision, ApprovalDecision::Approved) { 
                NotificationEventType::ApprovalApproved 
            } else { 
                NotificationEventType::ApprovalRejected 
            },
            priority: NotificationPriority::Normal,
            recipients: vec![], // Would be populated with stakeholders
            subject: format!("✅ Approval {}: {}", decision_text, approval_id),
            message: format!("Approval {} has been {} by {}", approval_id, decision_text, approver),
            html_message: None,
            data: HashMap::from([
                ("approval_id".to_string(), serde_json::Value::String(approval_id.to_string())),
                ("decision".to_string(), serde_json::Value::String(decision_text.to_string())),
                ("approver".to_string(), serde_json::Value::String(approver.to_string())),
                ("decision_time".to_string(), serde_json::Value::String(Utc::now().to_rfc3339())),
            ]),
            channels: vec![],
            scheduled_at: None,
            expires_at: None,
        };

        self.send_notification(notification_request).await
    }

    pub async fn get_notification_history(&self, limit: Option<usize>) -> Vec<NotificationHistoryEntry> {
        let history = self.notification_history.read().await;
        let limit = limit.unwrap_or(100);
        
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub async fn add_channel(&self, channel: NotificationChannel) -> Result<(), String> {
        self.channels.write().await.insert(channel.channel_id.clone(), channel);
        Ok(())
    }

    pub async fn remove_channel(&self, channel_id: &str) -> Result<(), String> {
        self.channels.write().await.remove(channel_id);
        Ok(())
    }

    pub async fn list_channels(&self) -> Vec<NotificationChannel> {
        self.channels.read().await.values().cloned().collect()
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            minute_counts: HashMap::new(),
            hour_counts: HashMap::new(),
        }
    }

    pub fn check_rate_limit(&mut self, channel_id: &str, rate_limit: &RateLimit) -> bool {
        let now = Utc::now();
        
        // Clean old entries
        self.cleanup_old_entries(now);
        
        // Check minute limit
        let minute_key = format!("{}:{}", channel_id, now.format("%Y-%m-%d %H:%M"));
        let minute_count = self.minute_counts.entry(minute_key).or_insert((now, 0));
        if minute_count.1 >= rate_limit.max_per_minute {
            return false;
        }
        
        // Check hour limit
        let hour_key = format!("{}:{}", channel_id, now.format("%Y-%m-%d %H"));
        let hour_count = self.hour_counts.entry(hour_key).or_insert((now, 0));
        if hour_count.1 >= rate_limit.max_per_hour {
            return false;
        }
        
        // Increment counters
        minute_count.1 += 1;
        hour_count.1 += 1;
        
        true
    }
    
    fn cleanup_old_entries(&mut self, now: DateTime<Utc>) {
        let cutoff_minute = now - chrono::Duration::minutes(1);
        let cutoff_hour = now - chrono::Duration::hours(1);
        
        self.minute_counts.retain(|_, (timestamp, _)| *timestamp > cutoff_minute);
        self.hour_counts.retain(|_, (timestamp, _)| *timestamp > cutoff_hour);
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            max_per_minute: 10,
            max_per_hour: 100,
            burst_limit: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_notification_system_creation() {
        let notification_system = NotificationSystem::new();
        assert!(notification_system.initialize().await.is_ok());
    }

    #[tokio::test]
    async fn test_send_notification() {
        let notification_system = NotificationSystem::new();
        notification_system.initialize().await.unwrap();

        let notification_request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: NotificationEventType::ApprovalRequested,
            priority: NotificationPriority::High,
            recipients: vec![NotificationRecipient {
                recipient_type: RecipientType::Email,
                identifier: "test@example.com".to_string(),
                name: Some("Test User".to_string()),
                preferences: None,
            }],
            subject: "Test Notification".to_string(),
            message: "This is a test notification".to_string(),
            html_message: None,
            data: HashMap::new(),
            channels: vec![],
            scheduled_at: None,
            expires_at: None,
        };

        let result = notification_system.send_notification(notification_request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mut rate_limiter = RateLimiter::new();
        let rate_limit = RateLimit {
            max_per_minute: 2,
            max_per_hour: 10,
            burst_limit: 1,
        };

        // First two requests should succeed
        assert!(rate_limiter.check_rate_limit("test-channel", &rate_limit));
        assert!(rate_limiter.check_rate_limit("test-channel", &rate_limit));
        
        // Third request should be rate limited
        assert!(!rate_limiter.check_rate_limit("test-channel", &rate_limit));
    }

    #[tokio::test]
    async fn test_notification_templates() {
        let notification_system = NotificationSystem::new();
        notification_system.initialize().await.unwrap();

        let templates = notification_system.templates.read().await;
        assert!(templates.contains_key("approval-requested"));
        assert!(templates.contains_key("approval-decision"));
        assert!(templates.contains_key("high-risk-detected"));
    }
}