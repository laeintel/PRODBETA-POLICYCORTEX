// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Impact Analysis System
// Analyzes cascading impacts of changes and failures across resources

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use chrono::{DateTime, Utc};

/// Impact analyzer for assessing cascading effects
pub struct ImpactAnalyzer {
    impact_models: HashMap<String, ImpactModel>,
    propagation_rules: Vec<PropagationRule>,
    threshold: f64,
}

impl ImpactAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self {
            impact_models: HashMap::new(),
            propagation_rules: Vec::new(),
            threshold: 0.3,
        };
        
        analyzer.initialize_models();
        analyzer.initialize_rules();
        analyzer
    }
    
    fn initialize_models(&mut self) {
        // Compute impact model
        self.impact_models.insert("compute".to_string(), ImpactModel {
            domain: "compute".to_string(),
            base_impact: 0.8,
            propagation_factor: 0.7,
            recovery_time_hours: 2,
        });
        
        // Storage impact model
        self.impact_models.insert("storage".to_string(), ImpactModel {
            domain: "storage".to_string(),
            base_impact: 0.9,
            propagation_factor: 0.8,
            recovery_time_hours: 4,
        });
        
        // Network impact model
        self.impact_models.insert("network".to_string(), ImpactModel {
            domain: "network".to_string(),
            base_impact: 0.95,
            propagation_factor: 0.9,
            recovery_time_hours: 1,
        });
        
        // Database impact model
        self.impact_models.insert("database".to_string(), ImpactModel {
            domain: "database".to_string(),
            base_impact: 0.85,
            propagation_factor: 0.75,
            recovery_time_hours: 6,
        });
    }
    
    fn initialize_rules(&mut self) {
        self.propagation_rules = vec![
            PropagationRule {
                name: "VM Failure Impact".to_string(),
                source_type: "VirtualMachine".to_string(),
                affected_types: vec!["WebApp".to_string(), "FunctionApp".to_string()],
                impact_multiplier: 0.9,
            },
            PropagationRule {
                name: "Storage Failure Impact".to_string(),
                source_type: "StorageAccount".to_string(),
                affected_types: vec!["VirtualMachine".to_string(), "Database".to_string()],
                impact_multiplier: 0.8,
            },
            PropagationRule {
                name: "Network Failure Impact".to_string(),
                source_type: "VirtualNetwork".to_string(),
                affected_types: vec!["VirtualMachine".to_string(), "LoadBalancer".to_string()],
                impact_multiplier: 1.0,
            },
        ];
    }
    
    /// Analyze impact of a change or failure
    pub fn analyze_impact(&self, event: ImpactEvent, resources: &[ResourceContext]) -> ImpactAssessment {
        let mut affected_resources = HashMap::new();
        let mut cascade_effects = Vec::new();
        let mut visited = HashSet::new();
        
        // Initialize with directly affected resource
        let initial_impact = self.calculate_initial_impact(&event);
        affected_resources.insert(event.resource_id.clone(), initial_impact);
        visited.insert(event.resource_id.clone());
        
        // BFS to find cascading impacts
        let mut queue = VecDeque::new();
        queue.push_back((event.resource_id.clone(), initial_impact));
        
        while let Some((resource_id, impact_score)) = queue.pop_front() {
            // Find resources that depend on this one
            let dependent_resources = self.find_dependent_resources(&resource_id, resources);
            
            for dependent in dependent_resources {
                if !visited.contains(&dependent.id) && impact_score > self.threshold {
                    visited.insert(dependent.id.clone());
                    
                    // Calculate cascading impact
                    let cascade_impact = self.calculate_cascade_impact(
                        &resource_id,
                        &dependent,
                        impact_score,
                    );
                    
                    affected_resources.insert(dependent.id.clone(), cascade_impact);
                    
                    // Create cascade effect record
                    cascade_effects.push(CascadeEffect {
                        source_resource: resource_id.clone(),
                        affected_resource: dependent.id.clone(),
                        impact_type: self.determine_impact_type(&event, &dependent),
                        severity: self.calculate_severity(cascade_impact),
                        propagation_delay_minutes: self.estimate_propagation_delay(&resource_id, &dependent.id),
                    });
                    
                    // Continue propagation if impact is significant
                    if cascade_impact > self.threshold {
                        queue.push_back((dependent.id.clone(), cascade_impact));
                    }
                }
            }
        }
        
        // Calculate business impact
        let business_impact = self.calculate_business_impact(&affected_resources, resources);
        
        // Generate mitigation strategies
        let mitigation_strategies = self.generate_mitigation_strategies(&event, &cascade_effects);
        
        ImpactAssessment {
            event: event.clone(),
            affected_resources,
            cascade_effects,
            total_impact_score: self.calculate_total_impact(&affected_resources),
            estimated_recovery_time: self.estimate_recovery_time(&event, &affected_resources),
            business_impact,
            mitigation_strategies,
            risk_level: self.determine_risk_level(&affected_resources),
        }
    }
    
    fn calculate_initial_impact(&self, event: &ImpactEvent) -> f64 {
        let base_impact = match event.event_type {
            EventType::Failure => 1.0,
            EventType::Degradation => 0.7,
            EventType::ConfigChange => 0.5,
            EventType::SecurityBreach => 0.9,
            EventType::Maintenance => 0.3,
        };
        
        // Adjust based on resource criticality
        base_impact * event.criticality_factor
    }
    
    fn find_dependent_resources<'a>(&self, resource_id: &str, resources: &'a [ResourceContext]) -> Vec<&'a ResourceContext> {
        resources.iter()
            .filter(|r| r.dependencies.contains(&resource_id.to_string()))
            .collect()
    }
    
    fn calculate_cascade_impact(&self, source_id: &str, dependent: &ResourceContext, source_impact: f64) -> f64 {
        // Get domain model for dependent resource
        let domain = self.get_resource_domain(&dependent.resource_type);
        let model = self.impact_models.get(&domain);
        
        let base_cascade = source_impact * 0.7; // Base propagation
        
        // Apply domain-specific propagation factor
        let domain_factor = model.map(|m| m.propagation_factor).unwrap_or(0.5);
        
        // Apply dependency strength
        let dependency_strength = dependent.dependency_strength.get(source_id).unwrap_or(&0.5);
        
        (base_cascade * domain_factor * dependency_strength).min(1.0)
    }
    
    fn determine_impact_type(&self, event: &ImpactEvent, resource: &ResourceContext) -> ImpactType {
        match event.event_type {
            EventType::Failure => ImpactType::ServiceUnavailable,
            EventType::Degradation => ImpactType::PerformanceDegradation,
            EventType::ConfigChange => ImpactType::ConfigurationDrift,
            EventType::SecurityBreach => ImpactType::SecurityCompromise,
            EventType::Maintenance => ImpactType::PlannedDowntime,
        }
    }
    
    fn calculate_severity(&self, impact_score: f64) -> Severity {
        if impact_score > 0.8 {
            Severity::Critical
        } else if impact_score > 0.6 {
            Severity::High
        } else if impact_score > 0.4 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }
    
    fn estimate_propagation_delay(&self, source: &str, target: &str) -> u32 {
        // Simplified estimation - in production would use actual network topology
        if source.contains("Network") || target.contains("Network") {
            1 // Network issues propagate quickly
        } else if source.contains("Database") {
            5 // Database issues take time to manifest
        } else {
            3 // Default propagation delay
        }
    }
    
    fn calculate_business_impact(&self, affected: &HashMap<String, f64>, resources: &[ResourceContext]) -> BusinessImpact {
        let mut cost_impact = 0.0;
        let mut affected_users = 0;
        let mut affected_services = HashSet::new();
        
        for (resource_id, impact_score) in affected {
            if let Some(resource) = resources.iter().find(|r| &r.id == resource_id) {
                // Calculate cost impact
                cost_impact += resource.hourly_cost * impact_score * 24.0; // Daily impact
                
                // Calculate user impact
                affected_users += (resource.user_count as f64 * impact_score) as u32;
                
                // Track affected services
                if let Some(service) = &resource.service_name {
                    affected_services.insert(service.clone());
                }
            }
        }
        
        BusinessImpact {
            estimated_cost: cost_impact,
            affected_users,
            affected_services: affected_services.into_iter().collect(),
            compliance_impact: self.assess_compliance_impact(affected),
            reputation_impact: self.assess_reputation_impact(affected_users),
        }
    }
    
    fn assess_compliance_impact(&self, affected: &HashMap<String, f64>) -> ComplianceImpact {
        let high_impact_count = affected.values().filter(|&&v| v > 0.7).count();
        
        if high_impact_count > 5 {
            ComplianceImpact::High
        } else if high_impact_count > 2 {
            ComplianceImpact::Medium
        } else {
            ComplianceImpact::Low
        }
    }
    
    fn assess_reputation_impact(&self, affected_users: u32) -> ReputationImpact {
        if affected_users > 10000 {
            ReputationImpact::Severe
        } else if affected_users > 1000 {
            ReputationImpact::Moderate
        } else {
            ReputationImpact::Minimal
        }
    }
    
    fn generate_mitigation_strategies(&self, event: &ImpactEvent, cascades: &[CascadeEffect]) -> Vec<MitigationStrategy> {
        let mut strategies = Vec::new();
        
        // Immediate actions
        strategies.push(MitigationStrategy {
            priority: 1,
            action: "Isolate affected resource to prevent further cascade".to_string(),
            estimated_time_minutes: 5,
            requires_approval: false,
        });
        
        // Event-specific strategies
        match event.event_type {
            EventType::Failure => {
                strategies.push(MitigationStrategy {
                    priority: 2,
                    action: "Activate failover to backup resource".to_string(),
                    estimated_time_minutes: 15,
                    requires_approval: true,
                });
            }
            EventType::SecurityBreach => {
                strategies.push(MitigationStrategy {
                    priority: 1,
                    action: "Revoke all access tokens and reset credentials".to_string(),
                    estimated_time_minutes: 10,
                    requires_approval: false,
                });
            }
            _ => {}
        }
        
        // Cascade-specific strategies
        if cascades.len() > 5 {
            strategies.push(MitigationStrategy {
                priority: 2,
                action: "Implement circuit breaker pattern to limit cascade".to_string(),
                estimated_time_minutes: 20,
                requires_approval: true,
            });
        }
        
        strategies.sort_by_key(|s| s.priority);
        strategies
    }
    
    fn calculate_total_impact(&self, affected: &HashMap<String, f64>) -> f64 {
        if affected.is_empty() {
            return 0.0;
        }
        
        let sum: f64 = affected.values().sum();
        let max: f64 = affected.values().cloned().fold(0.0, f64::max);
        
        // Weighted combination of average and max impact
        (sum / affected.len() as f64 * 0.6 + max * 0.4).min(1.0)
    }
    
    fn estimate_recovery_time(&self, event: &ImpactEvent, affected: &HashMap<String, f64>) -> u32 {
        let base_recovery = match event.event_type {
            EventType::Failure => 120,
            EventType::Degradation => 60,
            EventType::ConfigChange => 30,
            EventType::SecurityBreach => 240,
            EventType::Maintenance => 15,
        };
        
        // Adjust based on cascade complexity
        let complexity_factor = (affected.len() as f64 / 10.0).min(2.0).max(1.0);
        
        (base_recovery as f64 * complexity_factor) as u32
    }
    
    fn determine_risk_level(&self, affected: &HashMap<String, f64>) -> RiskLevel {
        let total_impact = self.calculate_total_impact(affected);
        
        if total_impact > 0.8 || affected.len() > 20 {
            RiskLevel::Critical
        } else if total_impact > 0.6 || affected.len() > 10 {
            RiskLevel::High
        } else if total_impact > 0.4 || affected.len() > 5 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }
    
    fn get_resource_domain(&self, resource_type: &str) -> String {
        if resource_type.contains("Compute") || resource_type.contains("VirtualMachine") {
            "compute".to_string()
        } else if resource_type.contains("Storage") {
            "storage".to_string()
        } else if resource_type.contains("Network") {
            "network".to_string()
        } else if resource_type.contains("Database") || resource_type.contains("Sql") {
            "database".to_string()
        } else {
            "other".to_string()
        }
    }
}

// Data structures

struct ImpactModel {
    domain: String,
    base_impact: f64,
    propagation_factor: f64,
    recovery_time_hours: u32,
}

struct PropagationRule {
    name: String,
    source_type: String,
    affected_types: Vec<String>,
    impact_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEvent {
    pub resource_id: String,
    pub resource_type: String,
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub criticality_factor: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Failure,
    Degradation,
    ConfigChange,
    SecurityBreach,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContext {
    pub id: String,
    pub resource_type: String,
    pub dependencies: Vec<String>,
    pub dependency_strength: HashMap<String, f64>,
    pub criticality: f64,
    pub hourly_cost: f64,
    pub user_count: u32,
    pub service_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub event: ImpactEvent,
    pub affected_resources: HashMap<String, f64>,
    pub cascade_effects: Vec<CascadeEffect>,
    pub total_impact_score: f64,
    pub estimated_recovery_time: u32,
    pub business_impact: BusinessImpact,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEffect {
    pub source_resource: String,
    pub affected_resource: String,
    pub impact_type: ImpactType,
    pub severity: Severity,
    pub propagation_delay_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactType {
    ServiceUnavailable,
    PerformanceDegradation,
    ConfigurationDrift,
    SecurityCompromise,
    PlannedDowntime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub estimated_cost: f64,
    pub affected_users: u32,
    pub affected_services: Vec<String>,
    pub compliance_impact: ComplianceImpact,
    pub reputation_impact: ReputationImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceImpact {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReputationImpact {
    Severe,
    Moderate,
    Minimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub priority: u32,
    pub action: String,
    pub estimated_time_minutes: u32,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}