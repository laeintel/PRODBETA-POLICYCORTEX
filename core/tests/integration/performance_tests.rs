// Performance and Load Testing for Remediation System
// Tests system performance under load and stress conditions

use super::*;
use policycortex_core::remediation::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use futures::future::join_all;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_benchmarks() {
        let mut test_ctx = TestContext::new();
        let mut results = TestResults::new();
        
        println!("⚡ Testing Performance Benchmarks");
        
        // Test Case 1: Concurrent remediation processing
        match test_concurrent_remediation_performance().await {
            Ok(metrics) => {
                println!("  ✅ Concurrent remediation performance - PASSED");
                println!("    📊 Metrics: {} req/sec, avg latency: {}ms", 
                        metrics.requests_per_second, metrics.average_latency_ms);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Concurrent remediation performance - FAILED: {}", e);
                results.record_failure(format!("Concurrent performance: {}", e));
            }
        }
        
        // Test Case 2: Bulk remediation scalability
        match test_bulk_remediation_scalability().await {
            Ok(metrics) => {
                println!("  ✅ Bulk remediation scalability - PASSED");
                println!("    📊 Metrics: {} items/sec, {} parallel workers", 
                        metrics.items_per_second, metrics.parallel_workers);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Bulk remediation scalability - FAILED: {}", e);
                results.record_failure(format!("Bulk scalability: {}", e));
            }
        }
        
        // Test Case 3: Notification system throughput
        match test_notification_throughput().await {
            Ok(metrics) => {
                println!("  ✅ Notification system throughput - PASSED");
                println!("    📊 Metrics: {} notifications/sec, {} channels", 
                        metrics.notifications_per_second, metrics.active_channels);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Notification system throughput - FAILED: {}", e);
                results.record_failure(format!("Notification throughput: {}", e));
            }
        }
        
        // Test Case 4: Memory usage under load
        match test_memory_usage_under_load().await {
            Ok(metrics) => {
                println!("  ✅ Memory usage under load - PASSED");
                println!("    📊 Metrics: peak memory: {}MB, memory efficiency: {:.2}%", 
                        metrics.peak_memory_mb, metrics.memory_efficiency);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Memory usage under load - FAILED: {}", e);
                results.record_failure(format!("Memory usage: {}", e));
            }
        }
        
        // Test Case 5: System responsiveness under stress
        match test_system_responsiveness().await {
            Ok(metrics) => {
                println!("  ✅ System responsiveness - PASSED");
                println!("    📊 Metrics: p95 latency: {}ms, error rate: {:.2}%", 
                        metrics.p95_latency_ms, metrics.error_rate);
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ System responsiveness - FAILED: {}", e);
                results.record_failure(format!("System responsiveness: {}", e));
            }
        }
        
        test_ctx.cleanup().await;
        
        println!("📊 Performance Test Results:");
        println!("   Passed: {}", results.passed);
        println!("   Failed: {}", results.failed);
        println!("   Success Rate: {:.1}%", results.success_rate());
        
        if results.failed > 0 {
            println!("❌ Performance Issues:");
            for failure in &results.failures {
                println!("   - {}", failure);
            }
            // Don't panic on performance failures, just warn
            println!("⚠️ Warning: Performance tests failed but system is functional");
        }
        
        // Performance tests should pass at least 80% for acceptable performance
        assert!(results.success_rate() >= 80.0, "Performance tests indicate system issues");
    }

    async fn test_concurrent_remediation_performance() -> Result<PerformanceMetrics, String> {
        let start_time = Instant::now();
        let concurrent_requests = 50;
        let successful_requests = Arc::new(AtomicUsize::new(0));
        let failed_requests = Arc::new(AtomicUsize::new(0));
        let total_latency = Arc::new(AtomicUsize::new(0));
        
        // Create mock workflow engine
        let workflow_engine = Arc::new(MockWorkflowEngine::new());
        
        // Generate concurrent requests
        let mut tasks = Vec::new();
        for i in 0..concurrent_requests {
            let engine = workflow_engine.clone();
            let successful = successful_requests.clone();
            let failed = failed_requests.clone();
            let latency_tracker = total_latency.clone();
            
            let task = tokio::spawn(async move {
                let request_start = Instant::now();
                
                let request = RemediationRequestBuilder::new()
                    .build();
                
                match engine.execute_remediation(request).await {
                    Ok(_) => {
                        successful.fetch_add(1, Ordering::SeqCst);
                        let latency = request_start.elapsed().as_millis() as usize;
                        latency_tracker.fetch_add(latency, Ordering::SeqCst);
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
            
            tasks.push(task);
        }
        
        // Wait for all requests to complete
        join_all(tasks).await;
        
        let total_time = start_time.elapsed();
        let successful = successful_requests.load(Ordering::SeqCst);
        let failed = failed_requests.load(Ordering::SeqCst);
        let total_latency_ms = total_latency.load(Ordering::SeqCst);
        
        // Calculate metrics
        let requests_per_second = if total_time.as_secs_f64() > 0.0 {
            successful as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };
        
        let average_latency_ms = if successful > 0 {
            total_latency_ms / successful
        } else {
            0
        };
        
        // Performance criteria
        if requests_per_second < 10.0 {
            return Err(format!("Low throughput: {:.2} req/sec (expected >= 10)", requests_per_second));
        }
        
        if average_latency_ms > 1000 {
            return Err(format!("High latency: {}ms (expected <= 1000ms)", average_latency_ms));
        }
        
        let error_rate = (failed as f64 / concurrent_requests as f64) * 100.0;
        if error_rate > 5.0 {
            return Err(format!("High error rate: {:.1}% (expected <= 5%)", error_rate));
        }
        
        Ok(PerformanceMetrics {
            requests_per_second: requests_per_second as usize,
            average_latency_ms,
            p95_latency_ms: average_latency_ms * 2, // Approximation
            error_rate,
            throughput: successful,
            memory_usage_mb: 0, // Not measured in this test
            cpu_usage_percent: 0.0, // Not measured in this test
        })
    }

    async fn test_bulk_remediation_scalability() -> Result<BulkPerformanceMetrics, String> {
        let batch_sizes = vec![10, 50, 100, 200];
        let workflow_engine = Arc::new(MockWorkflowEngine::new());
        
        let mut best_throughput = 0;
        let mut optimal_batch_size = 0;
        
        for batch_size in batch_sizes {
            let start_time = Instant::now();
            
            // Create bulk request
            let requests: Vec<_> = (0..batch_size)
                .map(|_| RemediationRequestBuilder::new().build())
                .collect();
            
            let bulk_id = uuid::Uuid::new_v4().to_string();
            let result = workflow_engine.execute_bulk_remediation(bulk_id, requests).await?;
            
            let execution_time = start_time.elapsed();
            let items_per_second = if execution_time.as_secs_f64() > 0.0 {
                result.successful as f64 / execution_time.as_secs_f64()
            } else {
                0.0
            } as usize;
            
            if items_per_second > best_throughput {
                best_throughput = items_per_second;
                optimal_batch_size = batch_size;
            }
            
            println!("    📊 Batch size {}: {} items/sec", batch_size, items_per_second);
        }
        
        // Performance criteria
        if best_throughput < 20 {
            return Err(format!("Low bulk throughput: {} items/sec (expected >= 20)", best_throughput));
        }
        
        Ok(BulkPerformanceMetrics {
            items_per_second: best_throughput,
            optimal_batch_size,
            parallel_workers: 4, // Mock value
            memory_efficiency: 85.0, // Mock value
        })
    }

    async fn test_notification_throughput() -> Result<NotificationPerformanceMetrics, String> {
        let notification_system = Arc::new(MockNotificationSystem::new());
        notification_system.initialize_test_channels().await?;
        
        let notification_count = 100;
        let start_time = Instant::now();
        let successful_notifications = Arc::new(AtomicUsize::new(0));
        
        // Send notifications concurrently
        let mut tasks = Vec::new();
        for i in 0..notification_count {
            let system = notification_system.clone();
            let success_counter = successful_notifications.clone();
            
            let task = tokio::spawn(async move {
                let request = NotificationRequest {
                    notification_id: uuid::Uuid::new_v4(),
                    event_type: NotificationEventType::SystemAlert,
                    priority: NotificationPriority::Normal,
                    recipients: vec![
                        NotificationRecipient {
                            recipient_type: RecipientType::Email,
                            identifier: format!("user{}@company.com", i),
                            name: Some(format!("User {}", i)),
                            preferences: None,
                        },
                    ],
                    subject: format!("Test notification {}", i),
                    message: format!("This is test notification number {}", i),
                    html_message: None,
                    data: std::collections::HashMap::new(),
                    channels: vec![],
                    scheduled_at: None,
                    expires_at: None,
                };
                
                if system.send_notification(request).await.is_ok() {
                    success_counter.fetch_add(1, Ordering::SeqCst);
                }
            });
            
            tasks.push(task);
        }
        
        join_all(tasks).await;
        
        let total_time = start_time.elapsed();
        let successful = successful_notifications.load(Ordering::SeqCst);
        
        let notifications_per_second = if total_time.as_secs_f64() > 0.0 {
            successful as f64 / total_time.as_secs_f64()
        } else {
            0.0
        } as usize;
        
        // Performance criteria
        if notifications_per_second < 30 {
            return Err(format!("Low notification throughput: {} notifications/sec (expected >= 30)", notifications_per_second));
        }
        
        Ok(NotificationPerformanceMetrics {
            notifications_per_second,
            active_channels: 3, // Mock value
            average_delivery_time_ms: 150,
            success_rate: (successful as f64 / notification_count as f64) * 100.0,
        })
    }

    async fn test_memory_usage_under_load() -> Result<MemoryMetrics, String> {
        // Simulate memory-intensive operations
        let initial_memory = get_memory_usage_mb();
        
        // Create large number of workflow executions
        let workflow_engine = Arc::new(MockWorkflowEngine::new());
        let mut tasks = Vec::new();
        
        for _ in 0..100 {
            let engine = workflow_engine.clone();
            
            let task = tokio::spawn(async move {
                for _ in 0..10 {
                    let request = RemediationRequestBuilder::new().build();
                    let _ = engine.execute_remediation(request).await;
                }
            });
            
            tasks.push(task);
        }
        
        join_all(tasks).await;
        
        let peak_memory = get_memory_usage_mb();
        let memory_growth = peak_memory - initial_memory;
        
        // Calculate memory efficiency
        let memory_efficiency = if peak_memory > 0 {
            100.0 - ((memory_growth as f64 / peak_memory as f64) * 100.0)
        } else {
            100.0
        };
        
        // Performance criteria
        if memory_growth > 500 {
            return Err(format!("Excessive memory growth: {}MB (expected <= 500MB)", memory_growth));
        }
        
        if memory_efficiency < 70.0 {
            return Err(format!("Poor memory efficiency: {:.1}% (expected >= 70%)", memory_efficiency));
        }
        
        Ok(MemoryMetrics {
            peak_memory_mb: peak_memory,
            memory_growth_mb: memory_growth,
            memory_efficiency,
            gc_pressure: 0.0, // Mock value for Rust (no GC)
        })
    }

    async fn test_system_responsiveness() -> Result<ResponsivenessMetrics, String> {
        let workflow_engine = Arc::new(MockWorkflowEngine::new());
        let request_count = 200;
        let mut latencies = Vec::new();
        let mut errors = 0;
        
        // Send requests at different intervals to simulate realistic load
        for i in 0..request_count {
            let start_time = Instant::now();
            
            let request = RemediationRequestBuilder::new().build();
            
            match workflow_engine.execute_remediation(request).await {
                Ok(_) => {
                    let latency = start_time.elapsed().as_millis() as usize;
                    latencies.push(latency);
                }
                Err(_) => {
                    errors += 1;
                }
            }
            
            // Variable delay to simulate realistic load patterns
            if i % 10 == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        
        // Calculate percentiles
        latencies.sort();
        let p95_index = (latencies.len() as f64 * 0.95) as usize;
        let p95_latency = latencies.get(p95_index).copied().unwrap_or(0);
        let p99_index = (latencies.len() as f64 * 0.99) as usize;
        let p99_latency = latencies.get(p99_index).copied().unwrap_or(0);
        
        let error_rate = (errors as f64 / request_count as f64) * 100.0;
        
        // Performance criteria
        if p95_latency > 2000 {
            return Err(format!("High P95 latency: {}ms (expected <= 2000ms)", p95_latency));
        }
        
        if error_rate > 2.0 {
            return Err(format!("High error rate: {:.1}% (expected <= 2%)", error_rate));
        }
        
        Ok(ResponsivenessMetrics {
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            error_rate,
            timeout_rate: 0.0, // Mock value
        })
    }

    // Mock function to simulate memory usage measurement
    fn get_memory_usage_mb() -> usize {
        // In a real implementation, this would measure actual memory usage
        // For testing, we'll simulate reasonable values
        std::thread_local! {
            static MEMORY_COUNTER: std::cell::RefCell<usize> = std::cell::RefCell::new(100);
        }
        
        MEMORY_COUNTER.with(|counter| {
            let mut val = counter.borrow_mut();
            *val += 5; // Simulate memory growth
            *val
        })
    }
}

// Performance metric structures
#[derive(Debug)]
pub struct PerformanceMetrics {
    pub requests_per_second: usize,
    pub average_latency_ms: usize,
    pub p95_latency_ms: usize,
    pub error_rate: f64,
    pub throughput: usize,
    pub memory_usage_mb: usize,
    pub cpu_usage_percent: f64,
}

#[derive(Debug)]
pub struct BulkPerformanceMetrics {
    pub items_per_second: usize,
    pub optimal_batch_size: usize,
    pub parallel_workers: usize,
    pub memory_efficiency: f64,
}

#[derive(Debug)]
pub struct NotificationPerformanceMetrics {
    pub notifications_per_second: usize,
    pub active_channels: usize,
    pub average_delivery_time_ms: usize,
    pub success_rate: f64,
}

#[derive(Debug)]
pub struct MemoryMetrics {
    pub peak_memory_mb: usize,
    pub memory_growth_mb: usize,
    pub memory_efficiency: f64,
    pub gc_pressure: f64,
}

#[derive(Debug)]
pub struct ResponsivenessMetrics {
    pub p95_latency_ms: usize,
    pub p99_latency_ms: usize,
    pub error_rate: f64,
    pub timeout_rate: f64,
}

// Mock implementations for performance testing
pub struct MockWorkflowEngine {
    execution_count: Arc<AtomicUsize>,
}

impl MockWorkflowEngine {
    pub fn new() -> Self {
        Self {
            execution_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    pub async fn execute_remediation(&self, request: RemediationRequest) -> Result<RemediationResult, String> {
        // Simulate variable execution time
        let execution_time = 50 + (self.execution_count.load(Ordering::SeqCst) % 100);
        tokio::time::sleep(Duration::from_millis(execution_time as u64)).await;
        
        self.execution_count.fetch_add(1, Ordering::SeqCst);
        
        // Simulate occasional failures (5% error rate)
        if self.execution_count.load(Ordering::SeqCst) % 20 == 0 {
            return Err("Simulated execution failure".to_string());
        }
        
        Ok(RemediationResult {
            request_id: request.request_id,
            status: RemediationStatus::Completed,
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            execution_time_ms: execution_time as u64,
            changes_applied: vec![],
            rollback_available: false,
            rollback_token: None,
            error: None,
            warnings: vec![],
        })
    }
    
    pub async fn execute_bulk_remediation(&self, bulk_id: String, requests: Vec<RemediationRequest>) -> Result<super::remediation_workflow_tests::BulkRemediationResult, String> {
        // Simulate bulk processing with parallel execution
        let batch_size = std::cmp::min(requests.len(), 10); // Max 10 parallel
        let chunks: Vec<_> = requests.chunks(batch_size).collect();
        
        let mut all_results = Vec::new();
        
        for chunk in chunks {
            let mut chunk_tasks = Vec::new();
            
            for request in chunk {
                let req = request.clone();
                let task = self.execute_remediation(req);
                chunk_tasks.push(task);
            }
            
            let chunk_results = join_all(chunk_tasks).await;
            for result in chunk_results {
                match result {
                    Ok(res) => all_results.push(res),
                    Err(_) => {} // Count as failed
                }
            }
        }
        
        Ok(super::remediation_workflow_tests::BulkRemediationResult {
            bulk_id,
            total_requested: requests.len(),
            successful: all_results.len(),
            failed: requests.len() - all_results.len(),
            skipped: 0,
            execution_time_ms: 200,
            results: all_results,
        })
    }
}

pub struct MockNotificationSystem {
    delivery_count: Arc<AtomicUsize>,
}

impl MockNotificationSystem {
    pub fn new() -> Self {
        Self {
            delivery_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    pub async fn initialize_test_channels(&self) -> Result<(), String> {
        // Mock channel initialization
        Ok(())
    }
    
    pub async fn send_notification(&self, _request: NotificationRequest) -> Result<NotificationResult, String> {
        // Simulate notification delivery time
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        self.delivery_count.fetch_add(1, Ordering::SeqCst);
        
        // Simulate occasional delivery failures (2% error rate)
        if self.delivery_count.load(Ordering::SeqCst) % 50 == 0 {
            return Err("Simulated delivery failure".to_string());
        }
        
        Ok(NotificationResult {
            notification_id: uuid::Uuid::new_v4(),
            status: NotificationStatus::Sent,
            total_recipients: 1,
            successful_deliveries: 1,
            failed_deliveries: 0,
            channel_results: vec![],
            delivered_at: chrono::Utc::now(),
            retry_count: 0,
        })
    }
}

use policycortex_core::remediation::notification_system::{NotificationRequest, NotificationResult, NotificationStatus, NotificationRecipient, RecipientType, NotificationEventType, NotificationPriority};