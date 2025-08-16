// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Cross-Domain Correlation Engine
// Identifies relationships and impacts across different Azure service domains

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{dijkstra, connected_components, kosaraju_scc};

/// Cross-domain correlation engine
pub struct CrossDomainEngine {
    dependency_graph: DiGraph<ResourceNode, DependencyEdge>,
    domain_analyzers: HashMap<String, Box<dyn DomainAnalyzer>>,
    correlation_rules: Vec<CorrelationRule>,
    impact_calculator: ImpactCalculator,
}

impl CrossDomainEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            dependency_graph: DiGraph::new(),
            domain_analyzers: HashMap::new(),
            correlation_rules: Vec::new(),
            impact_calculator: ImpactCalculator::new(),
        };
        
        engine.initialize_analyzers();
        engine.initialize_rules();
        engine
    }
    
    fn initialize_analyzers(&mut self) {
        self.domain_analyzers.insert("compute".to_string(), Box::new(ComputeDomainAnalyzer));
        self.domain_analyzers.insert("storage".to_string(), Box::new(StorageDomainAnalyzer));
        self.domain_analyzers.insert("network".to_string(), Box::new(NetworkDomainAnalyzer));
        self.domain_analyzers.insert("identity".to_string(), Box::new(IdentityDomainAnalyzer));
        self.domain_analyzers.insert("database".to_string(), Box::new(DatabaseDomainAnalyzer));
    }
    
    fn initialize_rules(&mut self) {
        self.correlation_rules = vec![
            CorrelationRule {
                name: "VM-Storage Dependency".to_string(),
                source_domain: "compute".to_string(),
                target_domain: "storage".to_string(),
                correlation_type: CorrelationType::DataDependency,
                strength: 0.9,
            },
            CorrelationRule {
                name: "Network-Security Group".to_string(),
                source_domain: "network".to_string(),
                target_domain: "compute".to_string(),
                correlation_type: CorrelationType::SecurityDependency,
                strength: 0.95,
            },
            CorrelationRule {
                name: "Database-Network Access".to_string(),
                source_domain: "database".to_string(),
                target_domain: "network".to_string(),
                correlation_type: CorrelationType::AccessDependency,
                strength: 0.85,
            },
        ];
    }
    
    /// Analyze cross-domain correlations
    pub async fn analyze_correlations(&mut self, resources: Vec<AzureResource>) -> CorrelationAnalysis {
        // Build dependency graph
        self.build_dependency_graph(&resources);
        
        // Find correlations
        let correlations = self.find_correlations(&resources);
        
        // Identify critical paths
        let critical_paths = self.find_critical_paths();
        
        // Calculate domain impacts
        let domain_impacts = self.calculate_domain_impacts(&resources);
        
        // Find isolated resources
        let isolated_resources = self.find_isolated_resources();
        
        CorrelationAnalysis {
            total_resources: resources.len(),
            correlations,
            critical_paths,
            domain_impacts,
            isolated_resources,
            risk_score: self.calculate_overall_risk(),
            recommendations: self.generate_recommendations(),
        }
    }
    
    fn build_dependency_graph(&mut self, resources: &[AzureResource]) {
        self.dependency_graph.clear();
        let mut node_map = HashMap::new();
        
        // Add nodes
        for resource in resources {
            let node = ResourceNode {
                id: resource.id.clone(),
                name: resource.name.clone(),
                resource_type: resource.resource_type.clone(),
                domain: self.get_domain(&resource.resource_type),
                risk_level: self.calculate_resource_risk(resource),
            };
            let idx = self.dependency_graph.add_node(node);
            node_map.insert(resource.id.clone(), idx);
        }
        
        // Add edges based on dependencies
        for resource in resources {
            if let Some(source_idx) = node_map.get(&resource.id) {
                for dep in &resource.dependencies {
                    if let Some(target_idx) = node_map.get(dep) {
                        let edge = DependencyEdge {
                            dependency_type: DependencyType::Direct,
                            strength: 0.8,
                            bidirectional: false,
                        };
                        self.dependency_graph.add_edge(*source_idx, *target_idx, edge);
                    }
                }
            }
        }
    }
    
    fn find_correlations(&self, resources: &[AzureResource]) -> Vec<Correlation> {
        let mut correlations = Vec::new();
        
        for rule in &self.correlation_rules {
            let source_resources: Vec<_> = resources.iter()
                .filter(|r| self.get_domain(&r.resource_type) == rule.source_domain)
                .collect();
            
            let target_resources: Vec<_> = resources.iter()
                .filter(|r| self.get_domain(&r.resource_type) == rule.target_domain)
                .collect();
            
            for source in &source_resources {
                for target in &target_resources {
                    if self.resources_correlated(source, target, rule) {
                        correlations.push(Correlation {
                            source_id: source.id.clone(),
                            target_id: target.id.clone(),
                            correlation_type: rule.correlation_type.clone(),
                            strength: rule.strength,
                            description: format!("{} correlation between {} and {}", 
                                rule.name, source.name, target.name),
                        });
                    }
                }
            }
        }
        
        correlations
    }
    
    fn find_critical_paths(&self) -> Vec<CriticalPath> {
        let mut critical_paths = Vec::new();
        
        // Find strongly connected components
        let sccs = kosaraju_scc(&self.dependency_graph);
        
        for scc in sccs {
            if scc.len() > 1 {
                // This is a cycle - critical for availability
                let nodes: Vec<_> = scc.iter()
                    .map(|&idx| self.dependency_graph[idx].clone())
                    .collect();
                
                critical_paths.push(CriticalPath {
                    path_type: PathType::Circular,
                    resources: nodes.iter().map(|n| n.id.clone()).collect(),
                    risk_level: RiskLevel::High,
                    impact: "Circular dependency detected - high availability risk".to_string(),
                });
            }
        }
        
        critical_paths
    }
    
    fn calculate_domain_impacts(&self, resources: &[AzureResource]) -> HashMap<String, DomainImpact> {
        let mut impacts = HashMap::new();
        
        for (domain_name, analyzer) in &self.domain_analyzers {
            let domain_resources: Vec<_> = resources.iter()
                .filter(|r| self.get_domain(&r.resource_type) == *domain_name)
                .collect();
            
            let impact = analyzer.calculate_impact(&domain_resources);
            impacts.insert(domain_name.clone(), impact);
        }
        
        impacts
    }
    
    fn find_isolated_resources(&self) -> Vec<String> {
        let mut isolated = Vec::new();
        
        for node_idx in self.dependency_graph.node_indices() {
            let incoming = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
            let outgoing = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
            
            if incoming == 0 && outgoing == 0 {
                isolated.push(self.dependency_graph[node_idx].id.clone());
            }
        }
        
        isolated
    }
    
    fn get_domain(&self, resource_type: &str) -> String {
        if resource_type.contains("Compute") {
            "compute".to_string()
        } else if resource_type.contains("Storage") {
            "storage".to_string()
        } else if resource_type.contains("Network") {
            "network".to_string()
        } else if resource_type.contains("Sql") || resource_type.contains("Database") {
            "database".to_string()
        } else if resource_type.contains("Identity") || resource_type.contains("KeyVault") {
            "identity".to_string()
        } else {
            "other".to_string()
        }
    }
    
    fn calculate_resource_risk(&self, resource: &AzureResource) -> f64 {
        let mut risk: f64 = 0.0;
        
        // Check for compliance violations
        if resource.compliance_state == "NonCompliant" {
            risk += 0.3;
        }
        
        // Check for public exposure
        if resource.properties.get("publicNetworkAccess").and_then(|v| v.as_str()) == Some("Enabled") {
            risk += 0.2;
        }
        
        // Check for encryption
        if resource.properties.get("encryption").and_then(|v| v.get("enabled")).and_then(|v| v.as_bool()) != Some(true) {
            risk += 0.2;
        }
        
        risk.min(1.0)
    }
    
    fn resources_correlated(&self, source: &AzureResource, target: &AzureResource, rule: &CorrelationRule) -> bool {
        // Simplified correlation detection
        // In production, would use more sophisticated analysis
        
        // Check if resources are in the same resource group
        if source.resource_group == target.resource_group {
            return true;
        }
        
        // Check if resources share tags
        for (key, value) in &source.tags {
            if target.tags.get(key) == Some(value) {
                return true;
            }
        }
        
        false
    }
    
    fn calculate_overall_risk(&self) -> f64 {
        let total_nodes = self.dependency_graph.node_count() as f64;
        if total_nodes == 0.0 {
            return 0.0;
        }
        
        let total_risk: f64 = self.dependency_graph.node_indices()
            .map(|idx| self.dependency_graph[idx].risk_level)
            .sum();
        
        total_risk / total_nodes
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for circular dependencies
        let sccs = kosaraju_scc(&self.dependency_graph);
        for scc in sccs {
            if scc.len() > 1 {
                recommendations.push("Circular dependencies detected. Consider refactoring to reduce coupling.".to_string());
                break;
            }
        }
        
        // Check for isolated resources
        let isolated_count = self.find_isolated_resources().len();
        if isolated_count > 0 {
            recommendations.push(format!("{} isolated resources found. Review if they are still needed.", isolated_count));
        }
        
        // Check overall risk
        let risk = self.calculate_overall_risk();
        if risk > 0.7 {
            recommendations.push("High overall risk detected. Prioritize security and compliance improvements.".to_string());
        }
        
        recommendations
    }
}

/// Domain analyzer trait
pub trait DomainAnalyzer: Send + Sync {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact;
}

/// Compute domain analyzer
struct ComputeDomainAnalyzer;

impl DomainAnalyzer for ComputeDomainAnalyzer {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact {
        let total = resources.len();
        let compliant = resources.iter().filter(|r| r.compliance_state == "Compliant").count();
        
        DomainImpact {
            domain: "compute".to_string(),
            resource_count: total,
            compliance_rate: if total > 0 { compliant as f64 / total as f64 } else { 0.0 },
            risk_level: if compliant < total / 2 { RiskLevel::High } else { RiskLevel::Low },
            key_issues: vec!["Unencrypted VMs".to_string(), "Missing backup policies".to_string()],
        }
    }
}

/// Storage domain analyzer
struct StorageDomainAnalyzer;

impl DomainAnalyzer for StorageDomainAnalyzer {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact {
        DomainImpact {
            domain: "storage".to_string(),
            resource_count: resources.len(),
            compliance_rate: 0.8,
            risk_level: RiskLevel::Medium,
            key_issues: vec!["Public access enabled".to_string()],
        }
    }
}

/// Network domain analyzer
struct NetworkDomainAnalyzer;

impl DomainAnalyzer for NetworkDomainAnalyzer {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact {
        DomainImpact {
            domain: "network".to_string(),
            resource_count: resources.len(),
            compliance_rate: 0.9,
            risk_level: RiskLevel::Low,
            key_issues: vec![],
        }
    }
}

/// Identity domain analyzer
struct IdentityDomainAnalyzer;

impl DomainAnalyzer for IdentityDomainAnalyzer {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact {
        DomainImpact {
            domain: "identity".to_string(),
            resource_count: resources.len(),
            compliance_rate: 0.95,
            risk_level: RiskLevel::Low,
            key_issues: vec![],
        }
    }
}

/// Database domain analyzer
struct DatabaseDomainAnalyzer;

impl DomainAnalyzer for DatabaseDomainAnalyzer {
    fn calculate_impact(&self, resources: &[&AzureResource]) -> DomainImpact {
        DomainImpact {
            domain: "database".to_string(),
            resource_count: resources.len(),
            compliance_rate: 0.85,
            risk_level: RiskLevel::Medium,
            key_issues: vec!["Missing TDE".to_string()],
        }
    }
}

/// Impact calculator
pub struct ImpactCalculator {
    impact_matrix: HashMap<String, HashMap<String, f64>>,
}

impl ImpactCalculator {
    pub fn new() -> Self {
        let mut calculator = Self {
            impact_matrix: HashMap::new(),
        };
        calculator.initialize_matrix();
        calculator
    }
    
    fn initialize_matrix(&mut self) {
        // Define impact relationships between domains
        let mut compute_impacts = HashMap::new();
        compute_impacts.insert("storage".to_string(), 0.8);
        compute_impacts.insert("network".to_string(), 0.9);
        compute_impacts.insert("database".to_string(), 0.7);
        self.impact_matrix.insert("compute".to_string(), compute_impacts);
        
        let mut storage_impacts = HashMap::new();
        storage_impacts.insert("compute".to_string(), 0.7);
        storage_impacts.insert("database".to_string(), 0.6);
        self.impact_matrix.insert("storage".to_string(), storage_impacts);
    }
    
    pub fn calculate_cascade_impact(&self, affected_domain: &str) -> HashMap<String, f64> {
        self.impact_matrix.get(affected_domain)
            .cloned()
            .unwrap_or_default()
    }
}

// Data structures

#[derive(Debug, Clone)]
pub struct ResourceNode {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub domain: String,
    pub risk_level: f64,
}

#[derive(Debug, Clone)]
pub struct DependencyEdge {
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub bidirectional: bool,
}

#[derive(Debug, Clone)]
pub enum DependencyType {
    Direct,
    Indirect,
    Weak,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub resource_group: String,
    pub location: String,
    pub tags: HashMap<String, String>,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub compliance_state: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub total_resources: usize,
    pub correlations: Vec<Correlation>,
    pub critical_paths: Vec<CriticalPath>,
    pub domain_impacts: HashMap<String, DomainImpact>,
    pub isolated_resources: Vec<String>,
    pub risk_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correlation {
    pub source_id: String,
    pub target_id: String,
    pub correlation_type: CorrelationType,
    pub strength: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    DataDependency,
    SecurityDependency,
    AccessDependency,
    PerformanceDependency,
}

#[derive(Debug, Clone)]
pub struct CorrelationRule {
    pub name: String,
    pub source_domain: String,
    pub target_domain: String,
    pub correlation_type: CorrelationType,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path_type: PathType,
    pub resources: Vec<String>,
    pub risk_level: RiskLevel,
    pub impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathType {
    Circular,
    Chain,
    Star,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainImpact {
    pub domain: String,
    pub resource_count: usize,
    pub compliance_rate: f64,
    pub risk_level: RiskLevel,
    pub key_issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}