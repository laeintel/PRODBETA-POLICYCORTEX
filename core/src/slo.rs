use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Service Level Objective (SLO) management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLO {
    pub id: String,
    pub name: String,
    pub description: String,
    pub service: String,
    pub endpoint: Option<String>,
    pub target_percentage: f64, // e.g., 99.9
    pub window: SLOWindow,
    pub sli: SLI,
    pub error_budget: ErrorBudget,
    pub alerts: Vec<SLOAlert>,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SLOWindow {
    Rolling { days: u32 },
    Calendar { period: CalendarPeriod },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalendarPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

/// Service Level Indicator (SLI) definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLI {
    pub metric_type: SLIType,
    pub threshold: Option<f64>,
    pub aggregation: Aggregation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SLIType {
    Availability,           // Success rate
    Latency(LatencyTarget), // Response time
    Quality,                // Data quality
    Throughput,             // Requests per second
    ErrorRate,              // Error percentage
    Custom(String),         // Custom metric
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyTarget {
    pub percentile: f64,   // e.g., 95.0 for p95
    pub threshold_ms: u64, // e.g., 200ms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Aggregation {
    Average,
    Sum,
    Min,
    Max,
    Count,
    Percentile(f64),
}

/// Error budget tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBudget {
    pub total_budget: f64,                // Total allowed errors (1 - SLO target)
    pub consumed: f64,                    // Budget consumed so far
    pub remaining: f64,                   // Remaining budget
    pub burn_rate: f64,                   // Current burn rate
    pub time_remaining: Option<Duration>, // Estimated time until budget exhausted
    pub reset_at: DateTime<Utc>,
}

impl ErrorBudget {
    pub fn new(slo_target: f64, window_days: u32) -> Self {
        let total_budget = (100.0 - slo_target) / 100.0;
        ErrorBudget {
            total_budget,
            consumed: 0.0,
            remaining: total_budget,
            burn_rate: 0.0,
            time_remaining: None,
            reset_at: Utc::now() + Duration::days(window_days as i64),
        }
    }

    pub fn update(&mut self, error_rate: f64, time_elapsed: Duration) {
        self.consumed += error_rate * (time_elapsed.num_seconds() as f64 / 3600.0);
        self.remaining = (self.total_budget - self.consumed).max(0.0);

        // Calculate burn rate (errors per hour)
        self.burn_rate = if time_elapsed.num_seconds() > 0 {
            self.consumed / (time_elapsed.num_seconds() as f64 / 3600.0)
        } else {
            0.0
        };

        // Estimate time remaining
        self.time_remaining = if self.burn_rate > 0.0 && self.remaining > 0.0 {
            Some(Duration::seconds(
                (self.remaining / self.burn_rate * 3600.0) as i64,
            ))
        } else {
            None
        };
    }

    pub fn is_exhausted(&self) -> bool {
        self.remaining <= 0.0
    }

    pub fn consumption_percentage(&self) -> f64 {
        (self.consumed / self.total_budget * 100.0).min(100.0)
    }
}

/// SLO alerts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLOAlert {
    pub threshold: f64, // Budget consumption percentage
    pub severity: AlertSeverity,
    pub channels: Vec<String>, // Notification channels
    pub cooldown: Duration,    // Time between alerts
    pub last_fired: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Page, // Wake someone up
}

/// SLO measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLOMeasurement {
    pub timestamp: DateTime<Utc>,
    pub success_count: u64,
    pub total_count: u64,
    pub error_count: u64,
    pub latency_ms: Option<f64>,
    pub metadata: HashMap<String, String>,
}

impl SLOMeasurement {
    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            100.0
        } else {
            (self.success_count as f64 / self.total_count as f64) * 100.0
        }
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.error_count as f64 / self.total_count as f64) * 100.0
        }
    }
}

/// SLO status and compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLOStatus {
    pub slo_id: String,
    pub current_value: f64,
    pub target_value: f64,
    pub is_meeting: bool,
    pub error_budget: ErrorBudget,
    pub measurements: Vec<SLOMeasurement>,
    pub burn_rate_alert: Option<BurnRateAlert>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnRateAlert {
    pub severity: AlertSeverity,
    pub message: String,
    pub burn_rate_multiplier: f64, // How many times faster than sustainable
}

/// SLO Manager for tracking and alerting
pub struct SLOManager {
    slos: Arc<RwLock<HashMap<String, SLO>>>,
    measurements: Arc<RwLock<HashMap<String, Vec<SLOMeasurement>>>>,
    status_cache: Arc<RwLock<HashMap<String, SLOStatus>>>,
}

impl SLOManager {
    pub fn new() -> Self {
        SLOManager {
            slos: Arc::new(RwLock::new(HashMap::new())),
            measurements: Arc::new(RwLock::new(HashMap::new())),
            status_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_slo(&self, slo: SLO) -> Result<String, String> {
        let mut slos = self.slos.write().await;
        let id = slo.id.clone();

        if slos.contains_key(&id) {
            return Err(format!("SLO with id {} already exists", id));
        }

        slos.insert(id.clone(), slo);
        Ok(id)
    }

    pub async fn record_measurement(
        &self,
        slo_id: &str,
        measurement: SLOMeasurement,
    ) -> Result<(), String> {
        let slos = self.slos.read().await;

        if !slos.contains_key(slo_id) {
            return Err(format!("SLO {} not found", slo_id));
        }

        let mut measurements = self.measurements.write().await;
        measurements
            .entry(slo_id.to_string())
            .or_insert_with(Vec::new)
            .push(measurement.clone());

        // Update status
        drop(measurements);
        self.update_status(slo_id).await?;

        Ok(())
    }

    async fn update_status(&self, slo_id: &str) -> Result<(), String> {
        let slos = self.slos.read().await;
        let slo = slos
            .get(slo_id)
            .ok_or_else(|| format!("SLO {} not found", slo_id))?;

        let measurements = self.measurements.read().await;
        let slo_measurements = measurements.get(slo_id).cloned().unwrap_or_default();

        // Calculate current SLI value based on window
        let window_start = self.get_window_start(&slo.window);
        let window_measurements: Vec<_> = slo_measurements
            .iter()
            .filter(|m| m.timestamp >= window_start)
            .cloned()
            .collect();

        if window_measurements.is_empty() {
            return Ok(());
        }

        let current_value = self.calculate_sli(&slo.sli, &window_measurements);
        let is_meeting = current_value >= slo.target_percentage;

        // Update error budget
        let mut error_budget = slo.error_budget.clone();
        let time_elapsed = Utc::now() - window_start;
        let error_rate = 100.0 - current_value;
        error_budget.update(error_rate / 100.0, time_elapsed);

        // Check for burn rate alerts
        let burn_rate_alert = self.check_burn_rate(&error_budget, &slo.window);

        let status = SLOStatus {
            slo_id: slo_id.to_string(),
            current_value,
            target_value: slo.target_percentage,
            is_meeting,
            error_budget,
            measurements: window_measurements,
            burn_rate_alert,
            last_updated: Utc::now(),
        };

        let mut status_cache = self.status_cache.write().await;
        status_cache.insert(slo_id.to_string(), status.clone());

        // Check and fire alerts
        drop(status_cache);
        drop(slos);
        self.check_alerts(slo_id, &status).await;

        Ok(())
    }

    fn get_window_start(&self, window: &SLOWindow) -> DateTime<Utc> {
        match window {
            SLOWindow::Rolling { days } => Utc::now() - Duration::days(*days as i64),
            SLOWindow::Calendar { period } => {
                let now = Utc::now();
                match period {
                    CalendarPeriod::Daily => {
                        now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()
                    }
                    CalendarPeriod::Weekly => {
                        let days_since_monday = now.weekday().num_days_from_monday();
                        (now - Duration::days(days_since_monday as i64))
                            .date_naive()
                            .and_hms_opt(0, 0, 0)
                            .unwrap()
                            .and_utc()
                    }
                    CalendarPeriod::Monthly => now
                        .date_naive()
                        .with_day(1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc(),
                    CalendarPeriod::Quarterly => {
                        let quarter_start_month = ((now.month() - 1) / 3) * 3 + 1;
                        now.date_naive()
                            .with_month(quarter_start_month)
                            .unwrap()
                            .with_day(1)
                            .unwrap()
                            .and_hms_opt(0, 0, 0)
                            .unwrap()
                            .and_utc()
                    }
                    CalendarPeriod::Yearly => now
                        .date_naive()
                        .with_month(1)
                        .unwrap()
                        .with_day(1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc(),
                }
            }
        }
    }

    fn calculate_sli(&self, sli: &SLI, measurements: &[SLOMeasurement]) -> f64 {
        if measurements.is_empty() {
            return 100.0;
        }

        match &sli.metric_type {
            SLIType::Availability => {
                let total: u64 = measurements.iter().map(|m| m.total_count).sum();
                let success: u64 = measurements.iter().map(|m| m.success_count).sum();
                if total == 0 {
                    100.0
                } else {
                    (success as f64 / total as f64) * 100.0
                }
            }
            SLIType::Latency(target) => {
                let mut latencies: Vec<f64> =
                    measurements.iter().filter_map(|m| m.latency_ms).collect();
                use std::cmp::Ordering;
                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

                if latencies.is_empty() {
                    100.0
                } else {
                    let index = ((target.percentile / 100.0) * latencies.len() as f64) as usize;
                    let percentile_value = latencies[index.min(latencies.len() - 1)];
                    if percentile_value <= target.threshold_ms as f64 {
                        100.0
                    } else {
                        (target.threshold_ms as f64 / percentile_value) * 100.0
                    }
                }
            }
            SLIType::ErrorRate => {
                let total: u64 = measurements.iter().map(|m| m.total_count).sum();
                let errors: u64 = measurements.iter().map(|m| m.error_count).sum();
                if total == 0 {
                    100.0
                } else {
                    100.0 - ((errors as f64 / total as f64) * 100.0)
                }
            }
            _ => {
                // Default to availability calculation
                let total: u64 = measurements.iter().map(|m| m.total_count).sum();
                let success: u64 = measurements.iter().map(|m| m.success_count).sum();
                if total == 0 {
                    100.0
                } else {
                    (success as f64 / total as f64) * 100.0
                }
            }
        }
    }

    fn check_burn_rate(&self, budget: &ErrorBudget, window: &SLOWindow) -> Option<BurnRateAlert> {
        let window_hours = match window {
            SLOWindow::Rolling { days } => *days as f64 * 24.0,
            SLOWindow::Calendar { period } => match period {
                CalendarPeriod::Daily => 24.0,
                CalendarPeriod::Weekly => 168.0,
                CalendarPeriod::Monthly => 720.0,
                CalendarPeriod::Quarterly => 2160.0,
                CalendarPeriod::Yearly => 8760.0,
            },
        };

        let sustainable_burn_rate = budget.total_budget / window_hours;
        let burn_rate_multiplier = budget.burn_rate / sustainable_burn_rate;

        if burn_rate_multiplier > 10.0 {
            Some(BurnRateAlert {
                severity: AlertSeverity::Page,
                message: format!(
                    "Error budget burn rate is {}x sustainable rate - immediate action required!",
                    burn_rate_multiplier as u32
                ),
                burn_rate_multiplier,
            })
        } else if burn_rate_multiplier > 5.0 {
            Some(BurnRateAlert {
                severity: AlertSeverity::Critical,
                message: format!(
                    "Error budget burn rate is {}x sustainable rate",
                    burn_rate_multiplier as u32
                ),
                burn_rate_multiplier,
            })
        } else if burn_rate_multiplier > 2.0 {
            Some(BurnRateAlert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "Error budget burn rate is {:.1}x sustainable rate",
                    burn_rate_multiplier
                ),
                burn_rate_multiplier,
            })
        } else {
            None
        }
    }

    async fn check_alerts(&self, slo_id: &str, status: &SLOStatus) {
        let mut slos = self.slos.write().await;
        let slo = match slos.get_mut(slo_id) {
            Some(s) => s,
            None => return,
        };

        let consumption = status.error_budget.consumption_percentage();

        for i in 0..slo.alerts.len() {
            let should_fire = consumption >= slo.alerts[i].threshold;

            if should_fire {
                // Check cooldown
                if let Some(last_fired) = slo.alerts[i].last_fired {
                    if Utc::now() - last_fired < slo.alerts[i].cooldown {
                        continue;
                    }
                }

                // Fire alert
                slo.alerts[i].last_fired = Some(Utc::now());
                let alert = &slo.alerts[i];
                self.send_alert(slo, alert, status).await;
            }
        }
    }

    async fn send_alert(&self, slo: &SLO, alert: &SLOAlert, status: &SLOStatus) {
        // In production, this would integrate with notification systems
        let message = format!(
            "SLO Alert: {} - Error budget {}% consumed (threshold: {}%)",
            slo.name,
            status.error_budget.consumption_percentage() as u32,
            alert.threshold as u32
        );

        eprintln!("[{}] {}", alert.severity_string(), message);

        // Here you would send to:
        // - PagerDuty for Page severity
        // - Slack/Teams for Warning/Critical
        // - Email for Info
    }

    pub async fn get_status(&self, slo_id: &str) -> Option<SLOStatus> {
        let status_cache = self.status_cache.read().await;
        status_cache.get(slo_id).cloned()
    }

    pub async fn get_all_status(&self) -> Vec<SLOStatus> {
        let status_cache = self.status_cache.read().await;
        status_cache.values().cloned().collect()
    }

    pub async fn should_block_release(&self, critical_slos: &[String]) -> bool {
        let status_cache = self.status_cache.read().await;

        for slo_id in critical_slos {
            if let Some(status) = status_cache.get(slo_id) {
                // Block if error budget is exhausted or nearly exhausted (>95% consumed)
                if status.error_budget.consumption_percentage() > 95.0 {
                    return true;
                }

                // Block if there's a page-level burn rate alert
                if let Some(ref alert) = status.burn_rate_alert {
                    if alert.severity == AlertSeverity::Page {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl SLOAlert {
    fn severity_string(&self) -> &str {
        match self.severity {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARN",
            AlertSeverity::Critical => "CRIT",
            AlertSeverity::Page => "PAGE",
        }
    }
}

/// Dashboard data for SLO visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLODashboard {
    pub summary: DashboardSummary,
    pub slos: Vec<SLODashboardItem>,
    pub alerts: Vec<ActiveAlert>,
    pub trends: Vec<TrendData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    pub total_slos: usize,
    pub meeting_target: usize,
    pub at_risk: usize,
    pub breaching: usize,
    pub average_budget_remaining: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLODashboardItem {
    pub slo: SLO,
    pub status: SLOStatus,
    pub trend: Trend,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Stable,
    Degrading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,      // >20% budget remaining
    Medium,   // 10-20% budget remaining
    High,     // 5-10% budget remaining
    Critical, // <5% budget remaining
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub slo_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    pub timestamp: DateTime<Utc>,
    pub slo_id: String,
    pub value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_budget_calculation() {
        let mut budget = ErrorBudget::new(99.9, 30);
        assert_eq!(budget.total_budget, 0.001);

        // Simulate 0.05% error rate for 1 hour
        budget.update(0.0005, Duration::hours(1));
        assert!(budget.remaining > 0.0);
        assert!(budget.consumption_percentage() < 100.0);
    }

    #[tokio::test]
    async fn test_slo_manager() {
        let manager = SLOManager::new();

        let slo = SLO {
            id: "api-availability".to_string(),
            name: "API Availability".to_string(),
            description: "Main API availability SLO".to_string(),
            service: "api-gateway".to_string(),
            endpoint: Some("/api/v1".to_string()),
            target_percentage: 99.9,
            window: SLOWindow::Rolling { days: 30 },
            sli: SLI {
                metric_type: SLIType::Availability,
                threshold: None,
                aggregation: Aggregation::Average,
            },
            error_budget: ErrorBudget::new(99.9, 30),
            alerts: vec![
                SLOAlert {
                    threshold: 50.0,
                    severity: AlertSeverity::Warning,
                    channels: vec!["slack".to_string()],
                    cooldown: Duration::hours(1),
                    last_fired: None,
                },
                SLOAlert {
                    threshold: 80.0,
                    severity: AlertSeverity::Critical,
                    channels: vec!["pagerduty".to_string()],
                    cooldown: Duration::minutes(30),
                    last_fired: None,
                },
            ],
            tags: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let id = manager.create_slo(slo).await.unwrap();

        // Record successful measurements
        for _ in 0..100 {
            let measurement = SLOMeasurement {
                timestamp: Utc::now(),
                success_count: 999,
                total_count: 1000,
                error_count: 1,
                latency_ms: Some(50.0),
                metadata: HashMap::new(),
            };
            manager.record_measurement(&id, measurement).await.unwrap();
        }

        let status = manager.get_status(&id).await.unwrap();
        assert!(status.is_meeting);
        assert!(status.current_value >= 99.9);
    }
}
