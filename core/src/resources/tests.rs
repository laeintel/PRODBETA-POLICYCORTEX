// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::azure_client::AzureClient;
    use std::sync::Arc;
    use tokio;

    #[tokio::test]
    async fn test_resource_categories() {
        let catalog = categories::ResourceCatalog::new();
        
        // Test Policy resources
        let policy_resources = catalog.get_by_category(ResourceCategory::Policy);
        assert_eq!(policy_resources.len(), 5, "Should have 5 policy resources");
        
        // Test Cost Management resources
        let cost_resources = catalog.get_by_category(ResourceCategory::CostManagement);
        assert!(cost_resources.len() >= 3, "Should have at least 3 cost management resources");
        
        // Test Security resources
        let security_resources = catalog.get_by_category(ResourceCategory::SecurityControls);
        assert!(security_resources.len() >= 4, "Should have at least 4 security resources");
        
        // Test Compute/Storage resources
        let compute_resources = catalog.get_by_category(ResourceCategory::ComputeStorage);
        assert!(compute_resources.len() >= 4, "Should have at least 4 compute/storage resources");
        
        // Test Network resources
        let network_resources = catalog.get_by_category(ResourceCategory::NetworksFirewalls);
        assert!(network_resources.len() >= 4, "Should have at least 4 network resources");
    }

    #[tokio::test]
    async fn test_resource_definition_lookup() {
        let catalog = categories::ResourceCatalog::new();
        
        // Test specific resource lookups
        let vm_def = catalog.get_definition("Microsoft.Compute/virtualMachines");
        assert!(vm_def.is_some(), "Should find VM definition");
        assert_eq!(vm_def.unwrap().display_name, "Virtual Machines");
        
        let policy_def = catalog.get_definition("Microsoft.Authorization/policyDefinitions");
        assert!(policy_def.is_some(), "Should find Policy definition");
        assert_eq!(policy_def.unwrap().category, ResourceCategory::Policy);
        
        let keyvault_def = catalog.get_definition("Microsoft.KeyVault/vaults");
        assert!(keyvault_def.is_some(), "Should find Key Vault definition");
        assert_eq!(keyvault_def.unwrap().category, ResourceCategory::SecurityControls);
    }

    #[tokio::test]
    async fn test_resource_health_status() {
        let resource = create_test_resource("test-vm", ResourceCategory::ComputeStorage);
        
        assert_eq!(resource.health.status, HealthStatus::Healthy);
        assert!(resource.health.issues.is_empty());
        assert!(!resource.health.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_resource_cost_data() {
        let mut resource = create_test_resource("test-vm", ResourceCategory::ComputeStorage);
        resource.cost_data = Some(CostData {
            daily_cost: 100.0,
            monthly_cost: 3000.0,
            yearly_projection: 36000.0,
            cost_trend: CostTrend::Increasing(5.0),
            optimization_potential: 500.0,
            currency: "USD".to_string(),
        });
        
        let cost = resource.cost_data.unwrap();
        assert_eq!(cost.daily_cost, 100.0);
        assert_eq!(cost.monthly_cost, 3000.0);
        assert!(matches!(cost.cost_trend, CostTrend::Increasing(_)));
        assert_eq!(cost.optimization_potential, 500.0);
    }

    #[tokio::test]
    async fn test_compliance_scoring() {
        let mut resource = create_test_resource("test-resource", ResourceCategory::Policy);
        
        // Initially compliant
        assert!(resource.compliance_status.is_compliant);
        assert_eq!(resource.compliance_status.compliance_score, 100.0);
        
        // Add violations
        resource.compliance_status.violations.push(ComplianceViolation {
            policy_id: "policy-001".to_string(),
            policy_name: "Test Policy".to_string(),
            severity: IssueSeverity::High,
            description: "Test violation".to_string(),
            remediation: Some("Fix this".to_string()),
        });
        
        // Recalculate compliance
        resource.compliance_status.is_compliant = false;
        resource.compliance_status.compliance_score = 85.0;
        
        assert!(!resource.compliance_status.is_compliant);
        assert_eq!(resource.compliance_status.violations.len(), 1);
    }

    #[tokio::test]
    async fn test_resource_insights() {
        let mut resource = create_test_resource("test-vm", ResourceCategory::ComputeStorage);
        
        resource.insights.push(ResourceInsight {
            insight_type: InsightType::CostOptimization,
            title: "Underutilized VM".to_string(),
            description: "VM running at 10% capacity".to_string(),
            impact: "Wasting $500/month".to_string(),
            recommendation: Some("Downsize to smaller instance".to_string()),
            confidence: 0.95,
        });
        
        assert_eq!(resource.insights.len(), 1);
        assert_eq!(resource.insights[0].confidence, 0.95);
        assert!(matches!(resource.insights[0].insight_type, InsightType::CostOptimization));
    }

    #[tokio::test]
    async fn test_quick_actions() {
        let resource = create_test_resource("test-vm", ResourceCategory::ComputeStorage);
        
        assert!(!resource.quick_actions.is_empty());
        
        let start_action = resource.quick_actions.iter()
            .find(|a| matches!(a.action_type, ActionType::Start));
        assert!(start_action.is_some());
        
        let stop_action = resource.quick_actions.iter()
            .find(|a| matches!(a.action_type, ActionType::Stop));
        assert!(stop_action.is_some());
    }

    #[tokio::test]
    async fn test_resource_filter() {
        let resources = vec![
            create_test_resource("vm1", ResourceCategory::ComputeStorage),
            create_test_resource("policy1", ResourceCategory::Policy),
            create_test_resource("vnet1", ResourceCategory::NetworksFirewalls),
        ];
        
        // Test category filter
        let filter = ResourceFilter {
            categories: Some(vec![ResourceCategory::ComputeStorage]),
            resource_types: None,
            locations: None,
            tags: None,
            health_status: None,
            compliance_filter: None,
            cost_range: None,
        };
        
        let filtered: Vec<_> = resources.iter()
            .filter(|r| match &filter.categories {
                Some(cats) => cats.contains(&r.category),
                None => true,
            })
            .collect();
        
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "vm1");
    }

    #[tokio::test]
    async fn test_resource_summary() {
        let summary = ResourceSummary {
            total_resources: 50,
            by_category: {
                let mut map = HashMap::new();
                map.insert(ResourceCategory::Policy, 10);
                map.insert(ResourceCategory::CostManagement, 10);
                map.insert(ResourceCategory::SecurityControls, 15);
                map.insert(ResourceCategory::ComputeStorage, 15);
                map.insert(ResourceCategory::NetworksFirewalls, 10);
                map
            },
            by_health: {
                let mut map = HashMap::new();
                map.insert(HealthStatus::Healthy, 35);
                map.insert(HealthStatus::Degraded, 10);
                map.insert(HealthStatus::Unhealthy, 3);
                map.insert(HealthStatus::Unknown, 2);
                map
            },
            total_daily_cost: 12500.0,
            compliance_score: 92.5,
            critical_issues: 3,
            optimization_opportunities: 15,
        };
        
        assert_eq!(summary.total_resources, 50);
        assert_eq!(summary.by_category.get(&ResourceCategory::Policy), Some(&10));
        assert_eq!(summary.total_daily_cost, 12500.0);
        assert_eq!(summary.critical_issues, 3);
    }

    // Helper function to create test resources
    fn create_test_resource(name: &str, category: ResourceCategory) -> AzureResource {
        AzureResource {
            id: format!("{}-id", name),
            name: name.to_string(),
            display_name: format!("{} Display", name),
            resource_type: match category {
                ResourceCategory::ComputeStorage => "Microsoft.Compute/virtualMachines".to_string(),
                ResourceCategory::Policy => "Microsoft.Authorization/policyDefinitions".to_string(),
                ResourceCategory::NetworksFirewalls => "Microsoft.Network/virtualNetworks".to_string(),
                _ => "Microsoft.Resources/generic".to_string(),
            },
            category,
            location: Some("East US".to_string()),
            tags: HashMap::new(),
            status: ResourceStatus {
                state: "Running".to_string(),
                provisioning_state: Some("Succeeded".to_string()),
                availability: 99.9,
                performance_score: 85.0,
            },
            health: ResourceHealth {
                status: HealthStatus::Healthy,
                issues: vec![],
                recommendations: vec!["Enable monitoring".to_string()],
            },
            cost_data: Some(CostData {
                daily_cost: 50.0,
                monthly_cost: 1500.0,
                yearly_projection: 18000.0,
                cost_trend: CostTrend::Stable,
                optimization_potential: 200.0,
                currency: "USD".to_string(),
            }),
            compliance_status: ComplianceStatus {
                is_compliant: true,
                compliance_score: 100.0,
                violations: vec![],
                last_assessment: chrono::Utc::now(),
            },
            quick_actions: vec![
                QuickAction {
                    id: "start".to_string(),
                    label: "Start".to_string(),
                    icon: "play".to_string(),
                    action_type: ActionType::Start,
                    confirmation_required: false,
                    estimated_impact: None,
                },
                QuickAction {
                    id: "stop".to_string(),
                    label: "Stop".to_string(),
                    icon: "square".to_string(),
                    action_type: ActionType::Stop,
                    confirmation_required: true,
                    estimated_impact: Some("Service will be unavailable".to_string()),
                },
            ],
            insights: vec![],
            last_updated: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod correlation_tests {
    use super::super::*;
    use crate::ai::model_registry::ModelRegistry;
    use crate::resources::correlations::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_cost_correlation_detection() {
        let engine = CrossDomainCorrelationEngine::new(Arc::new(ModelRegistry::new()));
        
        let resources = vec![
            create_correlated_resource("vm1", 100.0, CostTrend::Increasing(5.0)),
            create_correlated_resource("vm2", 105.0, CostTrend::Increasing(5.0)),
            create_correlated_resource("vm3", 50.0, CostTrend::Decreasing(3.0)),
        ];
        
        let correlations = engine.analyze_correlations(&resources).await;
        
        // Should find correlation between vm1 and vm2
        let cost_correlations: Vec<_> = correlations.iter()
            .filter(|c| matches!(c.correlation_type, CorrelationType::CostDependency))
            .collect();
        
        assert!(!cost_correlations.is_empty(), "Should find cost correlations");
        assert!(cost_correlations[0].strength > 0.7, "Should have strong correlation");
    }

    #[tokio::test]
    async fn test_security_correlation_detection() {
        let engine = CrossDomainCorrelationEngine::new(Arc::new(ModelRegistry::new()));
        
        let mut resources = vec![
            create_resource_with_type("keyvault1", "Microsoft.KeyVault/vaults"),
            create_resource_with_type("webapp1", "Microsoft.Web/sites"),
            create_resource_with_type("vm1", "Microsoft.Compute/virtualMachines"),
        ];
        
        // Add compliance violations
        for resource in &mut resources {
            if resource.name != "keyvault1" {
                resource.compliance_status.is_compliant = false;
                resource.compliance_status.violations.push(ComplianceViolation {
                    policy_id: "mfa-policy".to_string(),
                    policy_name: "MFA Required".to_string(),
                    severity: IssueSeverity::High,
                    description: "MFA not enabled".to_string(),
                    remediation: Some("Enable MFA".to_string()),
                });
            }
        }
        
        let correlations = engine.analyze_correlations(&resources).await;
        
        // Should find security dependencies
        let security_correlations: Vec<_> = correlations.iter()
            .filter(|c| matches!(c.correlation_type, CorrelationType::SecurityRelationship | CorrelationType::ComplianceLink))
            .collect();
        
        assert!(!security_correlations.is_empty(), "Should find security correlations");
    }

    #[tokio::test]
    async fn test_performance_correlation_detection() {
        let engine = CrossDomainCorrelationEngine::new(Arc::new(ModelRegistry::new()));
        
        let mut resources = vec![
            create_resource_with_performance("db1", 95.0, 100.0),
            create_resource_with_performance("app1", 60.0, 99.5),
            create_resource_with_performance("cache1", 50.0, 98.0),
        ];
        
        let correlations = engine.analyze_correlations(&resources).await;
        
        // Should find performance bottlenecks
        let perf_correlations: Vec<_> = correlations.iter()
            .filter(|c| matches!(c.correlation_type, CorrelationType::PerformanceImpact))
            .collect();
        
        assert!(!perf_correlations.is_empty(), "Should find performance correlations");
    }

    #[tokio::test]
    async fn test_correlation_strength_calculation() {
        let engine = CrossDomainCorrelationEngine::new(Arc::new(ModelRegistry::new()));
        
        let cost_a = CostData {
            daily_cost: 100.0,
            monthly_cost: 3000.0,
            yearly_projection: 36000.0,
            cost_trend: CostTrend::Increasing(5.0),
            optimization_potential: 0.0,
            currency: "USD".to_string(),
        };
        
        let cost_b = CostData {
            daily_cost: 105.0,
            monthly_cost: 3150.0,
            yearly_projection: 37800.0,
            cost_trend: CostTrend::Increasing(5.0),
            optimization_potential: 0.0,
            currency: "USD".to_string(),
        };
        
        // Test private method through correlation analysis
        let resources = vec![
            create_resource_with_cost("r1", cost_a),
            create_resource_with_cost("r2", cost_b),
        ];
        
        let correlations = engine.analyze_correlations(&resources).await;
        let cost_corr = correlations.iter()
            .find(|c| matches!(c.correlation_type, CorrelationType::CostDependency));
        
        assert!(cost_corr.is_some());
        assert!(cost_corr.unwrap().strength > 0.8, "Similar costs with same trend should have high correlation");
    }

    // Helper functions for correlation tests
    fn create_correlated_resource(name: &str, daily_cost: f64, trend: CostTrend) -> AzureResource {
        let mut resource = super::tests::create_test_resource(name, ResourceCategory::ComputeStorage);
        resource.cost_data = Some(CostData {
            daily_cost,
            monthly_cost: daily_cost * 30.0,
            yearly_projection: daily_cost * 365.0,
            cost_trend: trend,
            optimization_potential: daily_cost * 0.1,
            currency: "USD".to_string(),
        });
        resource
    }

    fn create_resource_with_type(name: &str, resource_type: &str) -> AzureResource {
        let mut resource = super::tests::create_test_resource(name, ResourceCategory::ComputeStorage);
        resource.resource_type = resource_type.to_string();
        resource
    }

    fn create_resource_with_performance(name: &str, perf_score: f32, availability: f32) -> AzureResource {
        let mut resource = super::tests::create_test_resource(name, ResourceCategory::ComputeStorage);
        resource.status.performance_score = perf_score;
        resource.status.availability = availability;
        resource
    }

    fn create_resource_with_cost(name: &str, cost_data: CostData) -> AzureResource {
        let mut resource = super::tests::create_test_resource(name, ResourceCategory::ComputeStorage);
        resource.cost_data = Some(cost_data);
        resource
    }
}

use std::collections::HashMap;
use chrono;