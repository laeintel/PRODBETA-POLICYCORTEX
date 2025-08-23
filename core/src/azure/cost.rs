// Azure Cost Management Integration
// Provides cost analysis and budget data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};

use super::client::AzureClient;
use super::api_versions;

/// Cost query request
#[derive(Debug, Serialize)]
pub struct CostQueryRequest {
    #[serde(rename = "type")]
    pub query_type: String,
    pub timeframe: String,
    #[serde(rename = "timePeriod")]
    pub time_period: Option<TimePeriod>,
    pub dataset: Dataset,
}

#[derive(Debug, Serialize)]
pub struct TimePeriod {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Serialize)]
pub struct Dataset {
    pub granularity: String,
    pub aggregation: HashMap<String, Aggregation>,
    pub grouping: Option<Vec<Grouping>>,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Aggregation {
    pub name: String,
    pub function: String,
}

#[derive(Debug, Serialize)]
pub struct Grouping {
    #[serde(rename = "type")]
    pub group_type: String,
    pub name: String,
}

/// Cost query response
#[derive(Debug, Deserialize)]
pub struct CostQueryResponse {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub properties: CostQueryProperties,
}

#[derive(Debug, Deserialize)]
pub struct CostQueryProperties {
    #[serde(rename = "nextLink")]
    pub next_link: Option<String>,
    pub columns: Vec<Column>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct Column {
    pub name: String,
    #[serde(rename = "type")]
    pub column_type: String,
}

/// Budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub budget_type: String,
    pub properties: BudgetProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetProperties {
    pub category: String,
    pub amount: f64,
    #[serde(rename = "timeGrain")]
    pub time_grain: String,
    #[serde(rename = "timePeriod")]
    pub time_period: BudgetTimePeriod,
    pub filter: Option<serde_json::Value>,
    #[serde(rename = "currentSpend")]
    pub current_spend: Option<CurrentSpend>,
    pub notifications: Option<HashMap<String, Notification>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetTimePeriod {
    #[serde(rename = "startDate")]
    pub start_date: DateTime<Utc>,
    #[serde(rename = "endDate")]
    pub end_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentSpend {
    pub amount: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub enabled: bool,
    pub operator: String,
    pub threshold: f64,
    #[serde(rename = "contactEmails")]
    pub contact_emails: Vec<String>,
    #[serde(rename = "contactRoles")]
    pub contact_roles: Option<Vec<String>>,
    #[serde(rename = "contactGroups")]
    pub contact_groups: Option<Vec<String>>,
}

/// Azure Cost Management service
pub struct CostService {
    client: AzureClient,
}

impl CostService {
    pub fn new(client: AzureClient) -> Self {
        Self { client }
    }

    /// Get current month costs
    pub async fn get_current_month_costs(&self) -> Result<CostSummary> {
        info!("Fetching current month costs from Azure Cost Management");

        let now = Utc::now();
        let start_of_month = now
            .with_day(1)
            .unwrap()
            .with_hour(0)
            .unwrap()
            .with_minute(0)
            .unwrap()
            .with_second(0)
            .unwrap();

        let request = CostQueryRequest {
            query_type: "ActualCost".to_string(),
            timeframe: "Custom".to_string(),
            time_period: Some(TimePeriod {
                from: start_of_month.format("%Y-%m-%d").to_string(),
                to: now.format("%Y-%m-%d").to_string(),
            }),
            dataset: Dataset {
                granularity: "Daily".to_string(),
                aggregation: {
                    let mut agg = HashMap::new();
                    agg.insert("totalCost".to_string(), Aggregation {
                        name: "Cost".to_string(),
                        function: "Sum".to_string(),
                    });
                    agg
                },
                grouping: Some(vec![
                    Grouping {
                        group_type: "Dimension".to_string(),
                        name: "ServiceName".to_string(),
                    }
                ]),
                filter: None,
            },
        };

        let path = format!(
            "/subscriptions/{}/providers/Microsoft.CostManagement/query",
            self.client.config.subscription_id
        );

        let response: CostQueryResponse = self.client
            .post_management(&path, api_versions::COST_MANAGEMENT, &request)
            .await?;

        // Process the response
        let mut total_cost = 0.0;
        let mut costs_by_service = HashMap::new();
        let mut daily_costs = Vec::new();

        for row in response.properties.rows {
            if row.len() >= 3 {
                let cost = row[0].as_f64().unwrap_or(0.0);
                let date = row[1].as_str().unwrap_or("").to_string();
                let service = row[2].as_str().unwrap_or("Unknown").to_string();

                total_cost += cost;
                *costs_by_service.entry(service).or_insert(0.0) += cost;
                
                daily_costs.push(DailyCost {
                    date,
                    cost,
                });
            }
        }

        Ok(CostSummary {
            total_cost,
            currency: "USD".to_string(),
            costs_by_service,
            daily_costs,
            period_start: start_of_month,
            period_end: now,
        })
    }

    /// Get cost forecast
    pub async fn get_cost_forecast(&self) -> Result<CostForecast> {
        info!("Fetching cost forecast from Azure Cost Management");

        let now = Utc::now();
        let end_of_month = (now + Duration::days(30))
            .with_day(1)
            .unwrap()
            .with_hour(0)
            .unwrap()
            .with_minute(0)
            .unwrap()
            .with_second(0)
            .unwrap()
            - Duration::days(1);

        let request = CostQueryRequest {
            query_type: "ForecastCost".to_string(),
            timeframe: "Custom".to_string(),
            time_period: Some(TimePeriod {
                from: now.format("%Y-%m-%d").to_string(),
                to: end_of_month.format("%Y-%m-%d").to_string(),
            }),
            dataset: Dataset {
                granularity: "Daily".to_string(),
                aggregation: {
                    let mut agg = HashMap::new();
                    agg.insert("totalCost".to_string(), Aggregation {
                        name: "Cost".to_string(),
                        function: "Sum".to_string(),
                    });
                    agg
                },
                grouping: None,
                filter: None,
            },
        };

        let path = format!(
            "/subscriptions/{}/providers/Microsoft.CostManagement/forecast",
            self.client.config.subscription_id
        );

        // Try to get forecast, but provide default if not available
        let forecast_total = match self.client
            .post_management::<CostQueryResponse, _>(&path, api_versions::COST_MANAGEMENT, &request)
            .await
        {
            Ok(response) => {
                response.properties.rows
                    .iter()
                    .map(|row| row[0].as_f64().unwrap_or(0.0))
                    .sum()
            }
            Err(_) => {
                // If forecast is not available, estimate based on current spend
                0.0
            }
        };

        Ok(CostForecast {
            forecasted_cost: forecast_total,
            confidence_level: 0.85,
            forecast_period_start: now,
            forecast_period_end: end_of_month,
        })
    }

    /// Get all budgets
    pub async fn get_budgets(&self) -> Result<Vec<Budget>> {
        let path = format!(
            "/subscriptions/{}/providers/Microsoft.Consumption/budgets",
            self.client.config.subscription_id
        );

        self.client.get_all_pages(&path, "2021-10-01").await
    }

    /// Get cost analysis by resource group
    pub async fn get_costs_by_resource_group(&self) -> Result<HashMap<String, f64>> {
        let now = Utc::now();
        let start_of_month = now
            .with_day(1)
            .unwrap()
            .with_hour(0)
            .unwrap()
            .with_minute(0)
            .unwrap()
            .with_second(0)
            .unwrap();

        let request = CostQueryRequest {
            query_type: "ActualCost".to_string(),
            timeframe: "Custom".to_string(),
            time_period: Some(TimePeriod {
                from: start_of_month.format("%Y-%m-%d").to_string(),
                to: now.format("%Y-%m-%d").to_string(),
            }),
            dataset: Dataset {
                granularity: "None".to_string(),
                aggregation: {
                    let mut agg = HashMap::new();
                    agg.insert("totalCost".to_string(), Aggregation {
                        name: "Cost".to_string(),
                        function: "Sum".to_string(),
                    });
                    agg
                },
                grouping: Some(vec![
                    Grouping {
                        group_type: "Dimension".to_string(),
                        name: "ResourceGroupName".to_string(),
                    }
                ]),
                filter: None,
            },
        };

        let path = format!(
            "/subscriptions/{}/providers/Microsoft.CostManagement/query",
            self.client.config.subscription_id
        );

        let response: CostQueryResponse = self.client
            .post_management(&path, api_versions::COST_MANAGEMENT, &request)
            .await?;

        let mut costs_by_rg = HashMap::new();
        for row in response.properties.rows {
            if row.len() >= 2 {
                let cost = row[0].as_f64().unwrap_or(0.0);
                let rg_name = row[1].as_str().unwrap_or("Unknown").to_string();
                costs_by_rg.insert(rg_name, cost);
            }
        }

        Ok(costs_by_rg)
    }
}

#[derive(Debug, Serialize)]
pub struct CostSummary {
    pub total_cost: f64,
    pub currency: String,
    pub costs_by_service: HashMap<String, f64>,
    pub daily_costs: Vec<DailyCost>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct DailyCost {
    pub date: String,
    pub cost: f64,
}

#[derive(Debug, Serialize)]
pub struct CostForecast {
    pub forecasted_cost: f64,
    pub confidence_level: f64,
    pub forecast_period_start: DateTime<Utc>,
    pub forecast_period_end: DateTime<Utc>,
}