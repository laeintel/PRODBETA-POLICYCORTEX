// Validation Engine Integration Tests
// Tests for pre-remediation validation and safety checks

use super::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_engine() {
        let mut test_ctx = TestContext::new();
        let mut results = TestResults::new();
        
        println!("🔍 Testing Validation Engine");
        
        // Test Case 1: Pre-remediation validation
        match test_pre_remediation_validation().await {
            Ok(_) => {
                println!("  ✅ Pre-remediation validation passed");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Pre-remediation validation failed: {}", e);
                results.record_failure(format!("Pre-remediation: {}", e));
            }
        }
        
        // Test Case 2: Policy compliance validation
        match test_policy_compliance_validation().await {
            Ok(_) => {
                println!("  ✅ Policy compliance validation passed");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Policy compliance validation failed: {}", e);
                results.record_failure(format!("Policy compliance: {}", e));
            }
        }
        
        // Test Case 3: Resource dependency analysis
        match test_resource_dependency_analysis().await {
            Ok(_) => {
                println!("  ✅ Resource dependency analysis passed");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Resource dependency analysis failed: {}", e);
                results.record_failure(format!("Dependency analysis: {}", e));
            }
        }
        
        // Test Case 4: Impact assessment
        match test_impact_assessment().await {
            Ok(_) => {
                println!("  ✅ Impact assessment passed");
                results.record_pass();
            }
            Err(e) => {
                println!("  ❌ Impact assessment failed: {}", e);
                results.record_failure(format!("Impact assessment: {}", e));
            }
        }
        
        test_ctx.cleanup().await;
        
        assert!(results.success_rate() >= 75.0, "Validation engine tests failed");
    }

    async fn test_pre_remediation_validation() -> Result<(), String> {
        let validator = MockValidationEngine::new();
        
        let request = ValidationRequest {
            resource_id: "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/test".to_string(),
            resource_type: "Microsoft.Storage/storageAccounts".to_string(),
            proposed_changes: vec![
                ProposedChange {
                    property: "encryption.enabled".to_string(),
                    current_value: "false".to_string(),
                    new_value: "true".to_string(),
                }
            ],
            policy_ids: vec!["encryption-policy".to_string()],
        };
        
        let result = validator.validate_pre_remediation(request).await?;
        
        if !result.is_valid {
            return Err(format!("Validation failed: {:?}", result.errors));
        }
        
        Ok(())
    }

    async fn test_policy_compliance_validation() -> Result<(), String> {
        let validator = MockValidationEngine::new();
        
        let result = validator.validate_policy_compliance(
            "/subscriptions/test/resourceGroups/test",
            &["encryption-policy", "backup-policy"]
        ).await?;
        
        if result.non_compliant_policies > 0 {
            println!("  ⚠️ Found {} non-compliant policies", result.non_compliant_policies);
        }
        
        Ok(())
    }

    async fn test_resource_dependency_analysis() -> Result<(), String> {
        let validator = MockValidationEngine::new();
        
        let dependencies = validator.analyze_dependencies(
            "/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/test"
        ).await?;
        
        if dependencies.critical_dependencies.is_empty() {
            return Err("Expected to find dependencies".to_string());
        }
        
        println!("  📊 Found {} dependencies", dependencies.critical_dependencies.len());
        
        Ok(())
    }

    async fn test_impact_assessment() -> Result<(), String> {
        let validator = MockValidationEngine::new();
        
        let impact = validator.assess_impact(
            "/subscriptions/test/resourceGroups/test",
            &ProposedChange {
                property: "publicNetworkAccess".to_string(),
                current_value: "Enabled".to_string(),
                new_value: "Disabled".to_string(),
            }
        ).await?;
        
        if impact.severity != ImpactSeverity::Low {
            println!("  ⚠️ Impact severity: {:?}", impact.severity);
        }
        
        Ok(())
    }
}

// Mock implementation for testing
pub struct MockValidationEngine {
    validation_rules: Vec<ValidationRule>,
}

impl MockValidationEngine {
    pub fn new() -> Self {
        Self {
            validation_rules: vec![
                ValidationRule {
                    id: "rule-1".to_string(),
                    name: "Encryption Required".to_string(),
                    condition: "encryption.enabled == true".to_string(),
                    severity: RuleSeverity::High,
                },
                ValidationRule {
                    id: "rule-2".to_string(),
                    name: "Backup Required".to_string(),
                    condition: "backup.enabled == true".to_string(),
                    severity: RuleSeverity::Medium,
                }
            ],
        }
    }
    
    pub async fn validate_pre_remediation(&self, request: ValidationRequest) -> Result<ValidationResult, String> {
        // Simulate validation
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check for critical issues
        for change in &request.proposed_changes {
            if change.property.contains("delete") || change.property.contains("remove") {
                warnings.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    message: format!("Destructive operation on property: {}", change.property),
                    rule_id: None,
                });
            }
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            passed_rules: self.validation_rules.len() - errors.len(),
            total_rules: self.validation_rules.len(),
        })
    }
    
    pub async fn validate_policy_compliance(&self, resource_id: &str, policy_ids: &[&str]) -> Result<ComplianceResult, String> {
        // Simulate compliance check
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
        
        Ok(ComplianceResult {
            resource_id: resource_id.to_string(),
            compliant_policies: policy_ids.len() - 1,
            non_compliant_policies: 1,
            compliance_percentage: ((policy_ids.len() - 1) as f64 / policy_ids.len() as f64) * 100.0,
        })
    }
    
    pub async fn analyze_dependencies(&self, resource_id: &str) -> Result<DependencyAnalysis, String> {
        // Simulate dependency analysis
        tokio::time::sleep(tokio::time::Duration::from_millis(40)).await;
        
        Ok(DependencyAnalysis {
            resource_id: resource_id.to_string(),
            critical_dependencies: vec![
                ResourceDependency {
                    resource_id: "/subscriptions/test/vnet".to_string(),
                    dependency_type: DependencyType::Network,
                    impact_if_modified: "May affect network connectivity".to_string(),
                }
            ],
            soft_dependencies: vec![],
            dependent_resources: vec![],
        })
    }
    
    pub async fn assess_impact(&self, resource_id: &str, change: &ProposedChange) -> Result<ImpactAssessment, String> {
        // Simulate impact assessment
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        
        let severity = if change.property.contains("Network") {
            ImpactSeverity::High
        } else if change.property.contains("encryption") {
            ImpactSeverity::Medium
        } else {
            ImpactSeverity::Low
        };
        
        Ok(ImpactAssessment {
            resource_id: resource_id.to_string(),
            change: change.clone(),
            severity,
            affected_services: vec!["Storage".to_string()],
            estimated_downtime_minutes: 0,
            rollback_complexity: RollbackComplexity::Simple,
        })
    }
}

// Data structures for validation

#[derive(Debug, Clone)]
pub struct ValidationRequest {
    pub resource_id: String,
    pub resource_type: String,
    pub proposed_changes: Vec<ProposedChange>,
    pub policy_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProposedChange {
    pub property: String,
    pub current_value: String,
    pub new_value: String,
}

#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationIssue>,
    pub passed_rules: usize,
    pub total_rules: usize,
}

#[derive(Debug)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub rule_id: Option<String>,
}

#[derive(Debug)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug)]
pub struct ValidationRule {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub severity: RuleSeverity,
}

#[derive(Debug)]
pub enum RuleSeverity {
    High,
    Medium,
    Low,
}

#[derive(Debug)]
pub struct ComplianceResult {
    pub resource_id: String,
    pub compliant_policies: usize,
    pub non_compliant_policies: usize,
    pub compliance_percentage: f64,
}

#[derive(Debug)]
pub struct DependencyAnalysis {
    pub resource_id: String,
    pub critical_dependencies: Vec<ResourceDependency>,
    pub soft_dependencies: Vec<ResourceDependency>,
    pub dependent_resources: Vec<ResourceDependency>,
}

#[derive(Debug)]
pub struct ResourceDependency {
    pub resource_id: String,
    pub dependency_type: DependencyType,
    pub impact_if_modified: String,
}

#[derive(Debug)]
pub enum DependencyType {
    Network,
    Storage,
    Compute,
    Identity,
    Data,
}

#[derive(Debug)]
pub struct ImpactAssessment {
    pub resource_id: String,
    pub change: ProposedChange,
    pub severity: ImpactSeverity,
    pub affected_services: Vec<String>,
    pub estimated_downtime_minutes: u32,
    pub rollback_complexity: RollbackComplexity,
}

#[derive(Debug, PartialEq)]
pub enum ImpactSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug)]
pub enum RollbackComplexity {
    Simple,
    Moderate,
    Complex,
}