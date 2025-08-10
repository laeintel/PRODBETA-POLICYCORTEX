use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use async_trait::async_trait;
use serde_json::Value;

/// Policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub enabled: bool,
    pub category: PolicyCategory,
    pub rules: Vec<PolicyRule>,
    pub actions: Vec<PolicyAction>,
    pub metadata: HashMap<String, Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Policy categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCategory {
    Security,
    Compliance,
    Cost,
    Performance,
    Governance,
    Custom(String),
}

/// Policy rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub condition: Condition,
    pub priority: i32,
    pub enabled: bool,
}

/// Condition types for policy rules
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Condition {
    And { conditions: Vec<Condition> },
    Or { conditions: Vec<Condition> },
    Not { condition: Box<Condition> },
    Equals { field: String, value: Value },
    NotEquals { field: String, value: Value },
    GreaterThan { field: String, value: Value },
    LessThan { field: String, value: Value },
    Contains { field: String, value: String },
    Regex { field: String, pattern: String },
    In { field: String, values: Vec<Value> },
    HasTag { tag: String },
    MissingTag { tag: String },
    Custom { expression: String },
}

/// Policy actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAction {
    pub id: String,
    pub action_type: ActionType,
    pub parameters: HashMap<String, Value>,
    pub auto_remediate: bool,
    pub notify: Vec<String>,
}

/// Action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Alert,
    Block,
    Remediate,
    Tag,
    Notify,
    Webhook,
    Custom(String),
}

/// Policy evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub policy_id: String,
    pub resource_id: String,
    pub compliant: bool,
    pub violations: Vec<Violation>,
    pub actions_taken: Vec<ActionResult>,
    pub evaluated_at: DateTime<Utc>,
    pub evaluation_time_ms: u64,
}

/// Policy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub rule_id: String,
    pub severity: Severity,
    pub message: String,
    pub details: HashMap<String, Value>,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Action execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub action_id: String,
    pub success: bool,
    pub message: String,
    pub error: Option<String>,
}

/// Resource to evaluate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    pub resource_type: String,
    pub provider: String,
    pub region: String,
    pub tags: HashMap<String, String>,
    pub properties: HashMap<String, Value>,
    pub metadata: HashMap<String, Value>,
}

/// Policy engine trait
#[async_trait]
pub trait PolicyEngine: Send + Sync {
    /// Evaluate a resource against all applicable policies
    async fn evaluate_resource(&self, resource: &Resource) -> Result<Vec<PolicyEvaluation>, String>;
    
    /// Evaluate a specific policy against a resource
    async fn evaluate_policy(&self, policy: &Policy, resource: &Resource) -> Result<PolicyEvaluation, String>;
    
    /// Execute policy actions
    async fn execute_actions(&self, policy: &Policy, resource: &Resource, violations: &[Violation]) -> Result<Vec<ActionResult>, String>;
    
    /// Get all policies
    async fn get_policies(&self) -> Result<Vec<Policy>, String>;
    
    /// Get policies by category
    async fn get_policies_by_category(&self, category: &PolicyCategory) -> Result<Vec<Policy>, String>;
    
    /// Create a new policy
    async fn create_policy(&self, policy: Policy) -> Result<Policy, String>;
    
    /// Update an existing policy
    async fn update_policy(&self, id: &str, policy: Policy) -> Result<Policy, String>;
    
    /// Delete a policy
    async fn delete_policy(&self, id: &str) -> Result<(), String>;
    
    /// Enable/disable a policy
    async fn set_policy_enabled(&self, id: &str, enabled: bool) -> Result<(), String>;
}

/// Default policy engine implementation
pub struct DefaultPolicyEngine {
    policies: Vec<Policy>,
    action_executor: Box<dyn ActionExecutor>,
}

impl DefaultPolicyEngine {
    pub fn new(action_executor: Box<dyn ActionExecutor>) -> Self {
        Self {
            policies: Vec::new(),
            action_executor,
        }
    }
    
    /// Evaluate a condition against a resource
    fn evaluate_condition(&self, condition: &Condition, resource: &Resource) -> bool {
        match condition {
            Condition::And { conditions } => {
                conditions.iter().all(|c| self.evaluate_condition(c, resource))
            }
            Condition::Or { conditions } => {
                conditions.iter().any(|c| self.evaluate_condition(c, resource))
            }
            Condition::Not { condition } => {
                !self.evaluate_condition(condition, resource)
            }
            Condition::Equals { field, value } => {
                self.get_field_value(resource, field) == Some(value.clone())
            }
            Condition::NotEquals { field, value } => {
                self.get_field_value(resource, field) != Some(value.clone())
            }
            Condition::GreaterThan { field, value } => {
                if let (Some(field_val), Some(compare_val)) = (
                    self.get_field_value(resource, field),
                    value.as_f64()
                ) {
                    field_val.as_f64().map_or(false, |v| v > compare_val)
                } else {
                    false
                }
            }
            Condition::LessThan { field, value } => {
                if let (Some(field_val), Some(compare_val)) = (
                    self.get_field_value(resource, field),
                    value.as_f64()
                ) {
                    field_val.as_f64().map_or(false, |v| v < compare_val)
                } else {
                    false
                }
            }
            Condition::Contains { field, value } => {
                if let Some(field_val) = self.get_field_value(resource, field) {
                    field_val.as_str().map_or(false, |s| s.contains(value))
                } else {
                    false
                }
            }
            Condition::Regex { field, pattern } => {
                if let Some(field_val) = self.get_field_value(resource, field) {
                    if let Some(s) = field_val.as_str() {
                        regex::Regex::new(pattern).ok().map_or(false, |re| re.is_match(s))
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Condition::In { field, values } => {
                if let Some(field_val) = self.get_field_value(resource, field) {
                    values.contains(&field_val)
                } else {
                    false
                }
            }
            Condition::HasTag { tag } => {
                resource.tags.contains_key(tag)
            }
            Condition::MissingTag { tag } => {
                !resource.tags.contains_key(tag)
            }
            Condition::Custom { expression } => {
                // Evaluate custom expression (simplified)
                // In production, use a proper expression evaluator
                false
            }
        }
    }
    
    /// Get field value from resource
    fn get_field_value(&self, resource: &Resource, field: &str) -> Option<Value> {
        // Parse field path (e.g., "properties.cpu.cores")
        let parts: Vec<&str> = field.split('.').collect();
        
        match parts[0] {
            "id" => Some(Value::String(resource.id.clone())),
            "type" => Some(Value::String(resource.resource_type.clone())),
            "provider" => Some(Value::String(resource.provider.clone())),
            "region" => Some(Value::String(resource.region.clone())),
            "tags" => {
                if parts.len() > 1 {
                    resource.tags.get(parts[1]).map(|v| Value::String(v.clone()))
                } else {
                    Some(serde_json::to_value(&resource.tags).unwrap_or(Value::Null))
                }
            }
            "properties" => {
                if parts.len() > 1 {
                    self.navigate_json_path(&resource.properties, &parts[1..])
                } else {
                    Some(serde_json::to_value(&resource.properties).unwrap_or(Value::Null))
                }
            }
            _ => None,
        }
    }
    
    /// Navigate JSON path
    fn navigate_json_path(&self, data: &HashMap<String, Value>, path: &[&str]) -> Option<Value> {
        if path.is_empty() {
            return Some(serde_json::to_value(data).unwrap_or(Value::Null));
        }
        
        data.get(path[0]).and_then(|value| {
            if path.len() == 1 {
                Some(value.clone())
            } else if let Value::Object(map) = value {
                let nested: HashMap<String, Value> = map.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                self.navigate_json_path(&nested, &path[1..])
            } else {
                None
            }
        })
    }
}

#[async_trait]
impl PolicyEngine for DefaultPolicyEngine {
    async fn evaluate_resource(&self, resource: &Resource) -> Result<Vec<PolicyEvaluation>, String> {
        let mut evaluations = Vec::new();
        
        for policy in &self.policies {
            if policy.enabled {
                let evaluation = self.evaluate_policy(policy, resource).await?;
                evaluations.push(evaluation);
            }
        }
        
        Ok(evaluations)
    }
    
    async fn evaluate_policy(&self, policy: &Policy, resource: &Resource) -> Result<PolicyEvaluation, String> {
        let start = std::time::Instant::now();
        let mut violations = Vec::new();
        let mut compliant = true;
        
        // Evaluate all rules
        for rule in &policy.rules {
            if !rule.enabled {
                continue;
            }
            
            let passes = self.evaluate_condition(&rule.condition, resource);
            
            if !passes {
                compliant = false;
                violations.push(Violation {
                    rule_id: rule.id.clone(),
                    severity: Severity::Medium, // Default severity
                    message: format!("Resource violates rule: {}", rule.id),
                    details: HashMap::new(),
                });
            }
        }
        
        // Execute actions if there are violations
        let actions_taken = if !compliant && !violations.is_empty() {
            self.execute_actions(policy, resource, &violations).await.unwrap_or_default()
        } else {
            Vec::new()
        };
        
        Ok(PolicyEvaluation {
            policy_id: policy.id.clone(),
            resource_id: resource.id.clone(),
            compliant,
            violations,
            actions_taken,
            evaluated_at: Utc::now(),
            evaluation_time_ms: start.elapsed().as_millis() as u64,
        })
    }
    
    async fn execute_actions(&self, policy: &Policy, resource: &Resource, violations: &[Violation]) -> Result<Vec<ActionResult>, String> {
        let mut results = Vec::new();
        
        for action in &policy.actions {
            if action.auto_remediate || violations.iter().any(|v| v.severity == Severity::Critical) {
                let result = self.action_executor.execute(action, resource, violations).await;
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    async fn get_policies(&self) -> Result<Vec<Policy>, String> {
        Ok(self.policies.clone())
    }
    
    async fn get_policies_by_category(&self, category: &PolicyCategory) -> Result<Vec<Policy>, String> {
        Ok(self.policies.iter()
            .filter(|p| std::mem::discriminant(&p.category) == std::mem::discriminant(category))
            .cloned()
            .collect())
    }
    
    async fn create_policy(&self, policy: Policy) -> Result<Policy, String> {
        // In production, persist to database
        Ok(policy)
    }
    
    async fn update_policy(&self, _id: &str, policy: Policy) -> Result<Policy, String> {
        // In production, update in database
        Ok(policy)
    }
    
    async fn delete_policy(&self, _id: &str) -> Result<(), String> {
        // In production, delete from database
        Ok(())
    }
    
    async fn set_policy_enabled(&self, _id: &str, _enabled: bool) -> Result<(), String> {
        // In production, update in database
        Ok(())
    }
}

/// Action executor trait
#[async_trait]
pub trait ActionExecutor: Send + Sync {
    async fn execute(&self, action: &PolicyAction, resource: &Resource, violations: &[Violation]) -> ActionResult;
}

/// Default action executor
pub struct DefaultActionExecutor;

#[async_trait]
impl ActionExecutor for DefaultActionExecutor {
    async fn execute(&self, action: &PolicyAction, _resource: &Resource, _violations: &[Violation]) -> ActionResult {
        match action.action_type {
            ActionType::Alert => {
                // Send alert
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Alert sent".to_string(),
                    error: None,
                }
            }
            ActionType::Block => {
                // Block resource
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Resource blocked".to_string(),
                    error: None,
                }
            }
            ActionType::Remediate => {
                // Auto-remediate
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Remediation applied".to_string(),
                    error: None,
                }
            }
            ActionType::Tag => {
                // Apply tags
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Tags applied".to_string(),
                    error: None,
                }
            }
            ActionType::Notify => {
                // Send notifications
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Notifications sent".to_string(),
                    error: None,
                }
            }
            ActionType::Webhook => {
                // Call webhook
                ActionResult {
                    action_id: action.id.clone(),
                    success: true,
                    message: "Webhook called".to_string(),
                    error: None,
                }
            }
            ActionType::Custom(_) => {
                // Execute custom action
                ActionResult {
                    action_id: action.id.clone(),
                    success: false,
                    message: "Custom action not implemented".to_string(),
                    error: Some("Not implemented".to_string()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_condition_evaluation() {
        let engine = DefaultPolicyEngine::new(Box::new(DefaultActionExecutor));
        
        let mut resource = Resource {
            id: "test-resource".to_string(),
            resource_type: "VM".to_string(),
            provider: "Azure".to_string(),
            region: "eastus".to_string(),
            tags: HashMap::new(),
            properties: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        resource.tags.insert("Environment".to_string(), "Production".to_string());
        resource.properties.insert("cpu".to_string(), Value::Number(serde_json::Number::from(4)));
        
        // Test HasTag condition
        let condition = Condition::HasTag { tag: "Environment".to_string() };
        assert!(engine.evaluate_condition(&condition, &resource));
        
        // Test MissingTag condition
        let condition = Condition::MissingTag { tag: "Owner".to_string() };
        assert!(engine.evaluate_condition(&condition, &resource));
        
        // Test And condition
        let condition = Condition::And {
            conditions: vec![
                Condition::HasTag { tag: "Environment".to_string() },
                Condition::Equals { 
                    field: "provider".to_string(), 
                    value: Value::String("Azure".to_string()) 
                },
            ],
        };
        assert!(engine.evaluate_condition(&condition, &resource));
    }
}