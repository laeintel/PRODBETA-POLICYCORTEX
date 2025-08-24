// Notification System Integration Tests
// Tests multi-channel notification delivery, rate limiting, and template processing

use super::*;
use policycortex_core::remediation::notification_system::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_notification_system_integration() {
        let mut test_ctx = TestContext::new();
        let mut results = TestResults::new();

        println!("🔔 Testing Notification System Integration");

        // Initialize notification system
        let notification_system = Arc::new(NotificationSystem::new());

        // Test Case 1: Multi-channel notification delivery
        match test_multi_channel_notification(&notification_system).await {
            Ok(_) => {
                println!("  ✅ Multi-channel notification delivery - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Multi-channel notification delivery - FAILED: {}", e);
                results.record_failure(format!("Multi-channel notification: {}", e));
            }
        }

        // Test Case 2: Template processing and variable substitution
        match test_template_processing(&notification_system).await {
            Ok(_) => {
                println!("  ✅ Template processing - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Template processing - FAILED: {}", e);
                results.record_failure(format!("Template processing: {}", e));
            }
        }

        // Test Case 3: Rate limiting and throttling
        match test_rate_limiting(&notification_system).await {
            Ok(_) => {
                println!("  ✅ Rate limiting - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Rate limiting - FAILED: {}", e);
                results.record_failure(format!("Rate limiting: {}", e));
            }
        }

        // Test Case 4: Retry mechanism and failure handling
        match test_retry_mechanism(&notification_system).await {
            Ok(_) => {
                println!("  ✅ Retry mechanism - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Retry mechanism - FAILED: {}", e);
                results.record_failure(format!("Retry mechanism: {}", e));
            }
        }

        // Test Case 5: Notification history and statistics
        match test_notification_history(&notification_system).await {
            Ok(_) => {
                println!("  ✅ Notification history - PASSED");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Notification history - FAILED: {}", e);
                results.record_failure(format!("Notification history: {}", e));
            }
        }

        test_ctx.cleanup().await;

        println!("📊 Notification System Test Results:");
        println!("   Passed: {}", results.passed);
        println!("   Failed: {}", results.failed);
        println!("   Success Rate: {:.1}%", results.success_rate());

        if results.failed > 0 {
            println!("❌ Failures:");
            for failure in &results.failures {
                println!("   - {}", failure);
            }
            panic!("Notification integration tests failed");
        }

        assert!(
            results.success_rate() >= 100.0,
            "All notification tests must pass"
        );
    }

    async fn test_multi_channel_notification(
        notification_system: &Arc<NotificationSystem>,
    ) -> Result<(), String> {
        // Setup test channels
        notification_system.initialize_default_channels().await?;

        // Create notification request for multiple channels
        let notification_request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: NotificationEventType::ApprovalRequested,
            priority: NotificationPriority::High,
            recipients: vec![
                NotificationRecipient {
                    recipient_type: RecipientType::Email,
                    identifier: "approver@company.com".to_string(),
                    name: Some("Security Approver".to_string()),
                    preferences: None,
                },
                NotificationRecipient {
                    recipient_type: RecipientType::Teams,
                    identifier: "security-team".to_string(),
                    name: Some("Security Team".to_string()),
                    preferences: None,
                },
            ],
            subject: "High Priority Approval Required".to_string(),
            message: "A high-risk remediation requires your approval".to_string(),
            html_message: Some(
                "<p>A <strong>high-risk</strong> remediation requires your approval</p>"
                    .to_string(),
            ),
            data: std::collections::HashMap::from([
                (
                    "approval_id".to_string(),
                    serde_json::Value::String("test-approval-123".to_string()),
                ),
                (
                    "resource_type".to_string(),
                    serde_json::Value::String("Storage Account".to_string()),
                ),
            ]),
            channels: vec![], // Use all available channels
            scheduled_at: None,
            expires_at: None,
        };

        // Send notification
        let result = notification_system
            .send_notification(notification_request)
            .await?;

        // Verify delivery to multiple channels
        if result.successful_deliveries < 2 {
            return Err(format!(
                "Expected at least 2 successful deliveries, got {}",
                result.successful_deliveries
            ));
        }

        if result.status != NotificationStatus::Sent {
            return Err(format!("Expected status Sent, got {:?}", result.status));
        }

        Ok(())
    }

    async fn test_template_processing(
        notification_system: &Arc<NotificationSystem>,
    ) -> Result<(), String> {
        // Create template with variables
        let template = NotificationTemplate {
            template_id: "approval-request-template".to_string(),
            name: "Approval Request Template".to_string(),
            event_type: NotificationEventType::ApprovalRequested,
            subject_template: "Approval Required: {{resource_type}} {{resource_name}}".to_string(),
            body_template: "Hello {{approver_name}},\n\nA {{risk_level}} risk operation on {{resource_type}} requires your approval.\n\nResource: {{resource_name}}\nRequested by: {{requested_by}}\n\nPlease review and approve/reject.".to_string(),
            html_template: Some("<p>Hello {{approver_name}},</p><p>A <strong>{{risk_level}}</strong> risk operation on {{resource_type}} requires your approval.</p>".to_string()),
            variables: vec!["approver_name".to_string(), "resource_type".to_string(), "resource_name".to_string(), "risk_level".to_string(), "requested_by".to_string()],
            priority: NotificationPriority::High,
            channels: vec!["email".to_string(), "teams".to_string()],
        };

        // Add template to system
        notification_system.add_template(template).await?;

        // Create notification using template
        let notification_request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: NotificationEventType::ApprovalRequested,
            priority: NotificationPriority::High,
            recipients: vec![NotificationRecipient {
                recipient_type: RecipientType::Email,
                identifier: "approver@company.com".to_string(),
                name: Some("John Approver".to_string()),
                preferences: None,
            }],
            subject: "".to_string(), // Will be filled by template
            message: "".to_string(), // Will be filled by template
            html_message: None,      // Will be filled by template
            data: std::collections::HashMap::from([
                (
                    "approver_name".to_string(),
                    serde_json::Value::String("John Approver".to_string()),
                ),
                (
                    "resource_type".to_string(),
                    serde_json::Value::String("Storage Account".to_string()),
                ),
                (
                    "resource_name".to_string(),
                    serde_json::Value::String("productionstg123".to_string()),
                ),
                (
                    "risk_level".to_string(),
                    serde_json::Value::String("high".to_string()),
                ),
                (
                    "requested_by".to_string(),
                    serde_json::Value::String("alice@company.com".to_string()),
                ),
            ]),
            channels: vec!["email".to_string()],
            scheduled_at: None,
            expires_at: None,
        };

        // Send notification (template should be applied automatically)
        let result = notification_system
            .send_notification(notification_request)
            .await?;

        if result.successful_deliveries != 1 {
            return Err(format!(
                "Template notification failed: {} deliveries",
                result.successful_deliveries
            ));
        }

        // Verify template variables were substituted (check history)
        let history = notification_system.get_notification_history(Some(1)).await;
        if history.is_empty() {
            return Err("No notification history found".to_string());
        }

        let last_notification = &history[0];
        if !last_notification
            .processed_subject
            .contains("Storage Account productionstg123")
        {
            return Err("Template variables not properly substituted in subject".to_string());
        }

        Ok(())
    }

    async fn test_rate_limiting(
        notification_system: &Arc<NotificationSystem>,
    ) -> Result<(), String> {
        // Configure a channel with strict rate limits for testing
        let rate_limited_channel = NotificationChannel {
            channel_id: "test-rate-limited".to_string(),
            channel_type: NotificationChannelType::Email,
            config: ChannelConfig {
                smtp_server: Some("test.smtp.com".to_string()),
                smtp_port: Some(587),
                smtp_username: Some("test".to_string()),
                smtp_password: Some("test".to_string()),
                from_address: Some("test@company.com".to_string()),
                to_addresses: vec!["recipient@company.com".to_string()],
                ..Default::default()
            },
            enabled: true,
            priority: 5,
            rate_limit: RateLimit {
                max_per_minute: 2, // Very strict limit for testing
                max_per_hour: 10,
                burst_limit: 1,
            },
            retry_policy: RetryPolicy::default(),
        };

        notification_system
            .add_channel(rate_limited_channel)
            .await?;

        // Send multiple notifications rapidly to trigger rate limiting
        let mut sent_count = 0;
        let mut rate_limited_count = 0;

        for i in 0..5 {
            let request = NotificationRequest {
                notification_id: Uuid::new_v4(),
                event_type: NotificationEventType::SystemAlert,
                priority: NotificationPriority::Normal,
                recipients: vec![NotificationRecipient {
                    recipient_type: RecipientType::Email,
                    identifier: "test@company.com".to_string(),
                    name: Some("Test User".to_string()),
                    preferences: None,
                }],
                subject: format!("Test notification {}", i),
                message: format!("This is test notification number {}", i),
                html_message: None,
                data: std::collections::HashMap::new(),
                channels: vec!["test-rate-limited".to_string()],
                scheduled_at: None,
                expires_at: None,
            };

            let result = notification_system.send_notification(request).await?;

            if result.successful_deliveries > 0 {
                sent_count += 1;
            } else {
                rate_limited_count += 1;
            }

            // Small delay between requests
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Should have rate limited some notifications
        if rate_limited_count == 0 {
            return Err("Rate limiting did not activate when expected".to_string());
        }

        if sent_count == 0 {
            return Err("No notifications were sent at all".to_string());
        }

        println!(
            "    📊 Rate limiting test: {} sent, {} rate limited",
            sent_count, rate_limited_count
        );

        Ok(())
    }

    async fn test_retry_mechanism(
        notification_system: &Arc<NotificationSystem>,
    ) -> Result<(), String> {
        // Create a channel that will fail initially but succeed on retry
        let unreliable_channel = NotificationChannel {
            channel_id: "test-unreliable".to_string(),
            channel_type: NotificationChannelType::Webhook,
            config: ChannelConfig {
                webhook_url: Some("http://localhost:99999/webhook".to_string()), // Will fail
                webhook_headers: std::collections::HashMap::new(),
                webhook_auth_token: None,
                ..Default::default()
            },
            enabled: true,
            priority: 5,
            rate_limit: RateLimit::default(),
            retry_policy: RetryPolicy {
                max_retries: 3,
                initial_delay_ms: 100,
                backoff_multiplier: 2.0,
                max_delay_ms: 1000,
            },
        };

        notification_system.add_channel(unreliable_channel).await?;

        let request = NotificationRequest {
            notification_id: Uuid::new_v4(),
            event_type: NotificationEventType::SystemAlert,
            priority: NotificationPriority::Normal,
            recipients: vec![NotificationRecipient {
                recipient_type: RecipientType::Webhook,
                identifier: "test-webhook".to_string(),
                name: Some("Test Webhook".to_string()),
                preferences: None,
            }],
            subject: "Test retry notification".to_string(),
            message: "Testing retry mechanism".to_string(),
            html_message: None,
            data: std::collections::HashMap::new(),
            channels: vec!["test-unreliable".to_string()],
            scheduled_at: None,
            expires_at: None,
        };

        // This should fail but attempt retries
        let result = notification_system.send_notification(request).await?;

        // Check that retries were attempted
        let history = notification_system.get_notification_history(Some(1)).await;
        if history.is_empty() {
            return Err("No notification history found".to_string());
        }

        let notification_entry = &history[0];
        if notification_entry.retry_count == 0 {
            return Err("No retry attempts were made".to_string());
        }

        println!(
            "    🔄 Retry test: {} retry attempts made",
            notification_entry.retry_count
        );

        Ok(())
    }

    async fn test_notification_history(
        notification_system: &Arc<NotificationSystem>,
    ) -> Result<(), String> {
        // Send several notifications to build history
        for i in 0..3 {
            let request = NotificationRequest {
                notification_id: Uuid::new_v4(),
                event_type: NotificationEventType::RemediationCompleted,
                priority: NotificationPriority::Normal,
                recipients: vec![NotificationRecipient {
                    recipient_type: RecipientType::Email,
                    identifier: format!("user{}@company.com", i),
                    name: Some(format!("User {}", i)),
                    preferences: None,
                }],
                subject: format!("Remediation {} completed", i),
                message: format!("The remediation operation {} has completed successfully", i),
                html_message: None,
                data: std::collections::HashMap::from([(
                    "remediation_id".to_string(),
                    serde_json::Value::String(format!("rem-{}", i)),
                )]),
                channels: vec![],
                scheduled_at: None,
                expires_at: None,
            };

            notification_system.send_notification(request).await?;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Test history retrieval
        let history = notification_system.get_notification_history(Some(5)).await;
        if history.len() < 3 {
            return Err(format!(
                "Expected at least 3 history entries, got {}",
                history.len()
            ));
        }

        // Test filtering by event type
        let filtered_history: Vec<_> = history
            .into_iter()
            .filter(|entry| entry.event_type == NotificationEventType::RemediationCompleted)
            .collect();

        if filtered_history.len() < 3 {
            return Err(format!(
                "Expected 3 RemediationCompleted entries, got {}",
                filtered_history.len()
            ));
        }

        // Test statistics
        let stats = notification_system.get_notification_statistics().await?;
        if stats.total_notifications == 0 {
            return Err("No notifications recorded in statistics".to_string());
        }

        println!(
            "    📈 Statistics: {} total notifications, {:.1}% success rate",
            stats.total_notifications, stats.success_rate
        );

        Ok(())
    }
}

// Mock implementations for testing
impl NotificationSystem {
    pub async fn initialize_default_channels(&self) -> Result<(), String> {
        // Add mock email channel
        let email_channel = NotificationChannel {
            channel_id: "default-email".to_string(),
            channel_type: NotificationChannelType::Email,
            config: ChannelConfig {
                smtp_server: Some("smtp.company.com".to_string()),
                smtp_port: Some(587),
                smtp_username: Some("noreply@company.com".to_string()),
                smtp_password: Some("password".to_string()),
                from_address: Some("noreply@company.com".to_string()),
                to_addresses: vec![],
                ..Default::default()
            },
            enabled: true,
            priority: 5,
            rate_limit: RateLimit::default(),
            retry_policy: RetryPolicy::default(),
        };

        // Add mock Teams channel
        let teams_channel = NotificationChannel {
            channel_id: "default-teams".to_string(),
            channel_type: NotificationChannelType::Teams,
            config: ChannelConfig {
                teams_webhook_url: Some(
                    "https://company.webhook.office.com/webhookb2/test".to_string(),
                ),
                teams_channel_id: Some("security-team".to_string()),
                ..Default::default()
            },
            enabled: true,
            priority: 5,
            rate_limit: RateLimit::default(),
            retry_policy: RetryPolicy::default(),
        };

        self.add_channel(email_channel).await?;
        self.add_channel(teams_channel).await?;

        Ok(())
    }

    pub async fn get_notification_statistics(&self) -> Result<NotificationStatistics, String> {
        let history = self.get_notification_history(None).await;
        let total = history.len();
        let successful = history
            .iter()
            .filter(|h| h.status == NotificationStatus::Sent)
            .count();

        Ok(NotificationStatistics {
            total_notifications: total,
            successful_notifications: successful,
            failed_notifications: total - successful,
            success_rate: if total > 0 {
                (successful as f64 / total as f64) * 100.0
            } else {
                0.0
            },
            average_delivery_time_ms: 150.0, // Mock value
            channels_active: 2,
            last_24h_count: total,
        })
    }
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            smtp_server: None,
            smtp_port: None,
            smtp_username: None,
            smtp_password: None,
            from_address: None,
            to_addresses: vec![],
            teams_webhook_url: None,
            teams_channel_id: None,
            slack_webhook_url: None,
            slack_token: None,
            slack_channel: None,
            webhook_url: None,
            webhook_headers: std::collections::HashMap::new(),
            webhook_auth_token: None,
            sms_provider: None,
            sms_api_key: None,
            phone_numbers: vec![],
        }
    }
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            max_per_minute: 60,
            max_per_hour: 1000,
            burst_limit: 10,
        }
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

#[derive(Debug)]
pub struct NotificationStatistics {
    pub total_notifications: usize,
    pub successful_notifications: usize,
    pub failed_notifications: usize,
    pub success_rate: f64,
    pub average_delivery_time_ms: f64,
    pub channels_active: usize,
    pub last_24h_count: usize,
}
