use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 30000,
            exponential_base: 2.0,
        }
    }
}

impl RetryConfig {
    pub fn for_azure_api() -> Self {
        RetryConfig {
            max_retries: 5,
            initial_delay_ms: 500,
            max_delay_ms: 60000,
            exponential_base: 2.0,
        }
    }

    pub fn for_critical_operations() -> Self {
        RetryConfig {
            max_retries: 10,
            initial_delay_ms: 1000,
            max_delay_ms: 120000,
            exponential_base: 1.5,
        }
    }
}

pub async fn retry_with_exponential_backoff<F, Fut, T, E>(
    config: RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut delay = config.initial_delay_ms;

    loop {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    debug!(
                        "Operation '{}' succeeded after {} retries",
                        operation_name, attempt
                    );
                }
                return Ok(result);
            }
            Err(err) if attempt >= config.max_retries => {
                warn!(
                    "Operation '{}' failed after {} attempts: {}",
                    operation_name, config.max_retries, err
                );
                return Err(err);
            }
            Err(err) => {
                attempt += 1;
                warn!(
                    "Operation '{}' failed (attempt {}/{}): {}. Retrying in {}ms...",
                    operation_name, attempt, config.max_retries, err, delay
                );
                
                sleep(Duration::from_millis(delay)).await;
                
                // Calculate next delay with exponential backoff
                delay = ((delay as f64) * config.exponential_base) as u64;
                delay = delay.min(config.max_delay_ms);
            }
        }
    }
}

// Retry specifically for HTTP status codes
pub fn should_retry_http_status(status: u16) -> bool {
    matches!(
        status,
        408 | 429 | 500 | 502 | 503 | 504 // Timeout, Too Many Requests, Server Errors
    )
}

// Azure-specific throttling detection
pub fn is_azure_throttled_error(error_message: &str) -> bool {
    error_message.contains("429") 
        || error_message.contains("Too Many Requests")
        || error_message.contains("throttled")
        || error_message.contains("rate limit")
        || error_message.contains("quota exceeded")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_on_third_attempt() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_retries: 5,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            exponential_base: 2.0,
        };

        let result = retry_with_exponential_backoff(
            config,
            "test_operation",
            || {
                let count = attempts_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err("Simulated failure")
                    } else {
                        Ok("Success")
                    }
                }
            },
        )
        .await;

        assert_eq!(result, Ok("Success"));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_max_attempts_exceeded() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            exponential_base: 2.0,
        };

        let result = retry_with_exponential_backoff(
            config,
            "test_operation",
            || {
                attempts_clone.fetch_add(1, Ordering::SeqCst);
                async move { Err::<(), _>("Always fails") }
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 4); // initial + 3 retries
    }

    #[test]
    fn test_should_retry_http_status() {
        assert!(should_retry_http_status(429));
        assert!(should_retry_http_status(500));
        assert!(should_retry_http_status(503));
        assert!(!should_retry_http_status(200));
        assert!(!should_retry_http_status(404));
        assert!(!should_retry_http_status(401));
    }

    #[test]
    fn test_is_azure_throttled_error() {
        assert!(is_azure_throttled_error("Error 429: Too Many Requests"));
        assert!(is_azure_throttled_error("Request was throttled"));
        assert!(is_azure_throttled_error("API rate limit exceeded"));
        assert!(is_azure_throttled_error("Subscription quota exceeded"));
        assert!(!is_azure_throttled_error("Internal server error"));
    }
}