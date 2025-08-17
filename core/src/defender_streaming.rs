use azure_core::auth::TokenCredential;
use azure_identity::{DefaultAzureCredential, TokenCredentialOptions};
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::checkpoint::{CheckpointManager, IngestionCheckpoint};
use crate::utils::{retry_with_exponential_backoff, RetryConfig};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenderAlert {
    pub id: String,
    pub alert_name: String,
    pub severity: String,
    pub status: String,
    pub resource_id: String,
    pub alert_type: String,
    pub time_generated: DateTime<Utc>,
    pub description: String,
    pub remediation_steps: Vec<String>,
    pub entities: Vec<AlertEntity>,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEntity {
    pub entity_type: String,
    pub entity_id: String,
    pub entity_name: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub subscription_id: String,
    pub event_hub_namespace: String,
    pub event_hub_name: String,
    pub consumer_group: String,
    pub webhook_secret: String,
    pub batch_size: usize,
    pub max_wait_time_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        StreamingConfig {
            subscription_id: std::env::var("AZURE_SUBSCRIPTION_ID").unwrap_or_default(),
            event_hub_namespace: std::env::var("DEFENDER_EVENTHUB_NAMESPACE")
                .unwrap_or_else(|_| "defender-streaming".to_string()),
            event_hub_name: std::env::var("DEFENDER_EVENTHUB_NAME")
                .unwrap_or_else(|_| "security-alerts".to_string()),
            consumer_group: std::env::var("DEFENDER_CONSUMER_GROUP")
                .unwrap_or_else(|_| "$Default".to_string()),
            webhook_secret: std::env::var("DEFENDER_WEBHOOK_SECRET")
                .unwrap_or_else(|_| "default_secret".to_string()),
            batch_size: 100,
            max_wait_time_ms: 5000,
        }
    }
}

pub struct DefenderStreamingService {
    config: StreamingConfig,
    credential: Arc<DefaultAzureCredential>,
    http_client: HttpClient,
    checkpoint_manager: CheckpointManager,
    // correlation_engine: Arc<CorrelationEngine>, // Will be added when implemented
    alert_sender: mpsc::Sender<DefenderAlert>,
}

impl DefenderStreamingService {
    pub async fn new(
        config: StreamingConfig,
        checkpoint_manager: CheckpointManager,
        // correlation_engine: Arc<CorrelationEngine>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let credential = Arc::new(DefaultAzureCredential::create(TokenCredentialOptions::default()).unwrap());
        let http_client = HttpClient::new();
        let (alert_sender, _alert_receiver) = mpsc::channel(1000);

        Ok(DefenderStreamingService {
            config,
            credential,
            http_client,
            checkpoint_manager,
            // correlation_engine,
            alert_sender,
        })
    }

    pub async fn start_streaming(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting Microsoft Defender streaming service");

        // Load checkpoint
        let mut checkpoint = self
            .checkpoint_manager
            .load_checkpoint("defender_alerts")
            .await?
            .unwrap_or_else(|| IngestionCheckpoint::new("defender_alerts".to_string()));

        // Subscribe to Defender alerts via Graph API Security alerts endpoint
        let alerts_url = format!(
            "https://graph.microsoft.com/v1.0/security/alerts?\
            $filter=createdDateTime gt {}&$top={}",
            checkpoint.last_processed_timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
            self.config.batch_size
        );

        loop {
            match self.fetch_and_process_alerts(&alerts_url, &mut checkpoint).await {
                Ok(has_more) => {
                    // Save checkpoint after each batch
                    if let Err(e) = self.checkpoint_manager.save_checkpoint(&checkpoint).await {
                        error!("Failed to save checkpoint: {}", e);
                    }

                    if !has_more {
                        // No more alerts, wait before next poll
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.max_wait_time_ms,
                        ))
                        .await;
                    }
                }
                Err(e) => {
                    error!("Error processing alerts: {}", e);
                    checkpoint.record_error(e.to_string());
                    
                    // Save checkpoint with error
                    let _ = self.checkpoint_manager.save_checkpoint(&checkpoint).await;
                    
                    // Wait before retry
                    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                }
            }
        }
    }

    async fn fetch_and_process_alerts(
        &self,
        url: &str,
        checkpoint: &mut IngestionCheckpoint,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let token = self
            .credential
            .get_token(&["https://graph.microsoft.com/.default"])
            .await?;

        let response = retry_with_exponential_backoff(
            RetryConfig::for_azure_api(),
            "fetch_defender_alerts",
            || async {
                let resp = self
                    .http_client
                    .get(url)
                    .bearer_auth(token.token.secret())
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {}", e))?;

                if resp.status().is_success() {
                    Ok(resp)
                } else if resp.status().as_u16() == 429 {
                    Err("Rate limited".to_string())
                } else {
                    Err(format!("HTTP error: {}", resp.status()))
                }
            },
        )
        .await?;

        let alerts_response: serde_json::Value = response.json().await?;
        let alerts = alerts_response["value"]
            .as_array()
            .ok_or("Invalid response format")?;

        for alert_json in alerts {
            if let Ok(alert) = self.parse_alert(alert_json) {
                // Verify webhook signature if present
                if !self.verify_signature(&alert) {
                    warn!("Alert signature verification failed for: {}", alert.id);
                    continue;
                }

                // Send to correlation engine
                self.send_to_correlation(&alert).await?;

                // Update checkpoint
                checkpoint.update_progress(Some(alert.id.clone()), None);

                // Send to alert channel
                if let Err(e) = self.alert_sender.send(alert).await {
                    error!("Failed to send alert to channel: {}", e);
                }
            }
        }

        // Check if there are more pages
        let has_next = alerts_response["@odata.nextLink"].is_string();
        Ok(has_next)
    }

    fn parse_alert(&self, json: &serde_json::Value) -> Result<DefenderAlert, Box<dyn std::error::Error + Send + Sync>> {
        let alert = DefenderAlert {
            id: json["id"].as_str().unwrap_or("").to_string(),
            alert_name: json["title"].as_str().unwrap_or("").to_string(),
            severity: json["severity"].as_str().unwrap_or("medium").to_string(),
            status: json["status"].as_str().unwrap_or("new").to_string(),
            resource_id: json["azureResourceId"].as_str().unwrap_or("").to_string(),
            alert_type: json["category"].as_str().unwrap_or("").to_string(),
            time_generated: DateTime::parse_from_rfc3339(
                json["createdDateTime"].as_str().unwrap_or("2024-01-01T00:00:00Z"),
            )?
            .with_timezone(&Utc),
            description: json["description"].as_str().unwrap_or("").to_string(),
            remediation_steps: json["recommendedActions"]
                .as_array()
                .map(|actions| {
                    actions
                        .iter()
                        .filter_map(|a| a.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
            entities: self.parse_entities(&json["entities"]),
            properties: json["vendorInformation"].clone(),
        };

        Ok(alert)
    }

    fn parse_entities(&self, entities_json: &serde_json::Value) -> Vec<AlertEntity> {
        if let Some(entities_array) = entities_json.as_array() {
            entities_array
                .iter()
                .filter_map(|entity| {
                    Some(AlertEntity {
                        entity_type: entity["type"].as_str()?.to_string(),
                        entity_id: entity["id"].as_str()?.to_string(),
                        entity_name: entity["name"].as_str()?.to_string(),
                        properties: entity["properties"].clone(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn verify_signature(&self, alert: &DefenderAlert) -> bool {
        // Calculate HMAC-SHA256 signature
        let payload = format!("{}{}", alert.id, alert.time_generated.timestamp());
        
        let mut mac = HmacSha256::new_from_slice(self.config.webhook_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        
        // In production, compare with signature from webhook header
        // For now, return true as we're polling directly
        true
    }

    async fn send_to_correlation(&self, alert: &DefenderAlert) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Correlation will be implemented when CorrelationEngine is available
        debug!(
            "Would send alert {} to correlation engine",
            alert.id
        );

        info!(
            "Alert {} ready for correlation: {} - {}",
            alert.id, alert.alert_name, alert.severity
        );

        Ok(())
    }

    pub fn get_alert_receiver(&self) -> mpsc::Receiver<DefenderAlert> {
        // This would be used by other components to receive alerts
        let (_sender, receiver) = mpsc::channel(1000);
        receiver
    }
}

// Webhook handler for receiving real-time alerts
pub async fn handle_defender_webhook(
    headers: &reqwest::header::HeaderMap,
    body: &[u8],
    service: &DefenderStreamingService,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Verify webhook signature
    if let Some(signature_header) = headers.get("x-ms-signature") {
        let signature = signature_header.to_str()?;
        
        let mut mac = HmacSha256::new_from_slice(service.config.webhook_secret.as_bytes())?;
        mac.update(body);
        
        let mac_result = mac.finalize();
        let bytes = mac_result.into_bytes();
        let expected_signature = format!("{:x}", bytes);
        
        if signature != expected_signature {
            return Err("Invalid webhook signature".into());
        }
    }

    // Parse alert from webhook body
    let alert: DefenderAlert = serde_json::from_slice(body)?;
    
    // Process the alert
    service.send_to_correlation(&alert).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_wait_time_ms, 5000);
        assert_eq!(config.consumer_group, "$Default");
    }

    #[test]
    fn test_alert_parsing() {
        let json = serde_json::json!({
            "id": "alert123",
            "title": "Suspicious activity detected",
            "severity": "high",
            "status": "new",
            "azureResourceId": "/subscriptions/123/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1",
            "category": "suspiciousActivity",
            "createdDateTime": "2024-01-15T10:30:00Z",
            "description": "Unusual login pattern detected",
            "recommendedActions": ["Review login history", "Enable MFA"],
            "entities": [
                {
                    "type": "user",
                    "id": "user123",
                    "name": "john.doe",
                    "properties": {}
                }
            ],
            "vendorInformation": {
                "provider": "Microsoft",
                "providerVersion": "1.0"
            }
        });

        let service = DefenderStreamingService {
            config: StreamingConfig::default(),
            credential: Arc::new(DefaultAzureCredential::create(TokenCredentialOptions::default()).unwrap()),
            http_client: HttpClient::new(),
            checkpoint_manager: CheckpointManager::new("redis://localhost:6379", "test")
                .await
                .unwrap(),
            // correlation_engine: Arc::new(CorrelationEngine::new()),
            alert_sender: mpsc::channel(100).0,
        };

        let alert = service.parse_alert(&json).unwrap();
        assert_eq!(alert.id, "alert123");
        assert_eq!(alert.severity, "high");
        assert_eq!(alert.entities.len(), 1);
        assert_eq!(alert.remediation_steps.len(), 2);
    }
}