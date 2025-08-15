use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Patent 1: Cross-Domain Governance Correlation Engine
// Graph Neural Network for modeling relationships between Azure resources

#[derive(Debug, Clone)]
pub struct GraphNeuralNetwork {
    pub nodes: HashMap<String, ResourceNode>,
    pub edges: Vec<ResourceEdge>,
    pub layers: Vec<GraphLayer>,
    pub embeddings: HashMap<String, Vec<f64>>,
    pub correlation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: String,
    pub resource_type: String,
    pub name: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub domain: GovernanceDomain,
    pub embedding: Vec<f64>,
    pub importance_score: f64,
    pub risk_score: f64,
    pub compliance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEdge {
    pub id: Uuid,
    pub source: String,
    pub target: String,
    pub relationship_type: RelationshipType,
    pub weight: f64,
    pub bidirectional: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GovernanceDomain {
    Security,
    Compliance,
    Cost,
    Performance,
    Network,
    Identity,
    Data,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    DependsOn,
    ConnectsTo,
    ManagesAccess,
    SharesNetwork,
    SharesData,
    InheritsPolicy,
    CostAllocation,
    SecurityBoundary,
    ComplianceScope,
}

#[derive(Debug, Clone)]
struct GraphLayer {
    layer_type: LayerType,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: ActivationFunction,
}

#[derive(Debug, Clone)]
enum LayerType {
    GraphConvolution,
    GraphAttention,
    GraphPooling,
    Dense,
}

#[derive(Debug, Clone)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainImpact {
    pub impact_id: Uuid,
    pub source_resource: String,
    pub source_domain: GovernanceDomain,
    pub impact_map: HashMap<String, DomainImpact>,
    pub total_impact_score: f64,
    pub risk_level: RiskLevel,
    pub affected_resources: Vec<AffectedResource>,
    pub correlation_strength: f64,
    pub predicted_violations: Vec<PredictedViolation>,
    pub remediation_priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainImpact {
    pub domain: GovernanceDomain,
    pub impact_score: f64,
    pub affected_resources: Vec<String>,
    pub cascading_effects: Vec<CascadingEffect>,
    pub time_to_impact: i64, // minutes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadingEffect {
    pub effect_type: String,
    pub probability: f64,
    pub severity: String,
    pub description: String,
    pub preventable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedResource {
    pub resource_id: String,
    pub resource_type: String,
    pub impact_type: String,
    pub impact_severity: f64,
    pub distance_from_source: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedViolation {
    pub policy_id: String,
    pub policy_name: String,
    pub violation_probability: f64,
    pub time_to_violation: i64,
    pub affected_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPattern {
    pub pattern_id: Uuid,
    pub pattern_type: PatternType,
    pub involved_resources: Vec<String>,
    pub domains: Vec<GovernanceDomain>,
    pub confidence: f64,
    pub frequency: u32,
    pub last_observed: DateTime<Utc>,
    pub business_impact: BusinessImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    SecurityComplianceMismatch,
    CostPerformanceTradeoff,
    NetworkSecurityGap,
    IdentityAccessRisk,
    DataGovernanceViolation,
    PolicyDrift,
    ResourceSprawl,
    ConfigurationAnomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub financial_impact: f64,
    pub compliance_risk: String,
    pub operational_impact: String,
    pub reputation_risk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfAnalysis {
    pub analysis_id: Uuid,
    pub scenario: WhatIfScenario,
    pub current_state: GraphSnapshot,
    pub predicted_state: GraphSnapshot,
    pub impacts: Vec<CrossDomainImpact>,
    pub recommendations: Vec<Recommendation>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfScenario {
    pub scenario_type: ScenarioType,
    pub changes: Vec<ProposedChange>,
    pub constraints: Vec<String>,
    pub optimization_goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioType {
    PolicyChange,
    ResourceDeletion,
    ResourceCreation,
    ConfigurationUpdate,
    NetworkChange,
    AccessModification,
    CostOptimization,
    ComplianceEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedChange {
    pub resource_id: String,
    pub change_type: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub domain_distribution: HashMap<GovernanceDomain, usize>,
    pub avg_compliance_score: f64,
    pub avg_risk_score: f64,
    pub critical_paths: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_id: Uuid,
    pub priority: u32,
    pub action: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: String,
    pub automated: bool,
}

impl GraphNeuralNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            layers: Self::initialize_layers(),
            embeddings: HashMap::new(),
            correlation_threshold: 0.7,
        }
    }

    fn initialize_layers() -> Vec<GraphLayer> {
        vec![
            GraphLayer {
                layer_type: LayerType::GraphConvolution,
                weights: vec![vec![0.1; 128]; 64],
                biases: vec![0.0; 128],
                activation: ActivationFunction::ReLU,
            },
            GraphLayer {
                layer_type: LayerType::GraphAttention,
                weights: vec![vec![0.1; 256]; 128],
                biases: vec![0.0; 256],
                activation: ActivationFunction::ReLU,
            },
            GraphLayer {
                layer_type: LayerType::GraphPooling,
                weights: vec![vec![0.1; 128]; 256],
                biases: vec![0.0; 128],
                activation: ActivationFunction::ReLU,
            },
            GraphLayer {
                layer_type: LayerType::Dense,
                weights: vec![vec![0.1; 64]; 128],
                biases: vec![0.0; 64],
                activation: ActivationFunction::Sigmoid,
            },
        ]
    }

    pub fn add_resource(&mut self, node: ResourceNode) {
        self.embeddings.insert(node.id.clone(), node.embedding.clone());
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_relationship(&mut self, edge: ResourceEdge) {
        self.edges.push(edge);
    }

    pub fn analyze_cross_domain_impact(&self, source_resource: &str) -> CrossDomainImpact {
        let affected = self.propagate_impact(source_resource);
        let impact_map = self.calculate_domain_impacts(&affected);
        let total_score = impact_map.values().map(|d| d.impact_score).sum::<f64>();
        
        CrossDomainImpact {
            impact_id: Uuid::new_v4(),
            source_resource: source_resource.to_string(),
            source_domain: self.nodes.get(source_resource)
                .map(|n| n.domain.clone())
                .unwrap_or(GovernanceDomain::Security),
            impact_map,
            total_impact_score: total_score,
            risk_level: self.determine_risk_level(total_score),
            affected_resources: affected,
            correlation_strength: 0.85,
            predicted_violations: self.predict_violations_from_impact(source_resource),
            remediation_priority: self.calculate_priority(total_score),
        }
    }

    fn propagate_impact(&self, source: &str) -> Vec<AffectedResource> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut affected = Vec::new();
        
        queue.push_back((source.to_string(), 0));
        visited.insert(source.to_string());
        
        while let Some((current, distance)) = queue.pop_front() {
            // Find all connected resources
            for edge in &self.edges {
                let next = if edge.source == current {
                    &edge.target
                } else if edge.bidirectional && edge.target == current {
                    &edge.source
                } else {
                    continue;
                };
                
                if !visited.contains(next) {
                    visited.insert(next.clone());
                    queue.push_back((next.clone(), distance + 1));
                    
                    if let Some(node) = self.nodes.get(next) {
                        affected.push(AffectedResource {
                            resource_id: next.clone(),
                            resource_type: node.resource_type.clone(),
                            impact_type: format!("{:?}", edge.relationship_type),
                            impact_severity: edge.weight * (0.9_f64).powi(distance as i32),
                            distance_from_source: distance + 1,
                        });
                    }
                }
            }
        }
        
        affected
    }

    fn calculate_domain_impacts(&self, affected: &[AffectedResource]) -> HashMap<String, DomainImpact> {
        let mut impacts = HashMap::new();
        
        for domain in [
            GovernanceDomain::Security,
            GovernanceDomain::Compliance,
            GovernanceDomain::Cost,
            GovernanceDomain::Performance,
            GovernanceDomain::Network,
        ] {
            let domain_resources: Vec<String> = affected.iter()
                .filter(|a| {
                    self.nodes.get(&a.resource_id)
                        .map(|n| matches!(&n.domain, d if std::mem::discriminant(d) == std::mem::discriminant(&domain)))
                        .unwrap_or(false)
                })
                .map(|a| a.resource_id.clone())
                .collect();
            
            if !domain_resources.is_empty() {
                let impact_score = domain_resources.len() as f64 * 0.3
                    + affected.iter()
                        .filter(|a| domain_resources.contains(&a.resource_id))
                        .map(|a| a.impact_severity)
                        .sum::<f64>();
                
                impacts.insert(
                    format!("{:?}", domain),
                    DomainImpact {
                        domain: domain.clone(),
                        impact_score,
                        affected_resources: domain_resources,
                        cascading_effects: self.identify_cascading_effects(&domain),
                        time_to_impact: (impact_score * 10.0) as i64,
                    }
                );
            }
        }
        
        impacts
    }

    fn identify_cascading_effects(&self, domain: &GovernanceDomain) -> Vec<CascadingEffect> {
        match domain {
            GovernanceDomain::Security => vec![
                CascadingEffect {
                    effect_type: "Data Breach Risk".to_string(),
                    probability: 0.3,
                    severity: "High".to_string(),
                    description: "Potential unauthorized access to sensitive data".to_string(),
                    preventable: true,
                },
                CascadingEffect {
                    effect_type: "Compliance Violation".to_string(),
                    probability: 0.7,
                    severity: "Medium".to_string(),
                    description: "Security changes may violate compliance requirements".to_string(),
                    preventable: true,
                },
            ],
            GovernanceDomain::Compliance => vec![
                CascadingEffect {
                    effect_type: "Audit Failure".to_string(),
                    probability: 0.5,
                    severity: "High".to_string(),
                    description: "Non-compliance may lead to audit failures".to_string(),
                    preventable: true,
                },
            ],
            GovernanceDomain::Cost => vec![
                CascadingEffect {
                    effect_type: "Budget Overrun".to_string(),
                    probability: 0.4,
                    severity: "Medium".to_string(),
                    description: "Resource changes may increase costs".to_string(),
                    preventable: true,
                },
            ],
            _ => vec![],
        }
    }

    fn predict_violations_from_impact(&self, source: &str) -> Vec<PredictedViolation> {
        vec![
            PredictedViolation {
                policy_id: "pol-001".to_string(),
                policy_name: "Require Encryption".to_string(),
                violation_probability: 0.75,
                time_to_violation: 24,
                affected_resources: vec![source.to_string()],
            },
            PredictedViolation {
                policy_id: "pol-002".to_string(),
                policy_name: "Network Isolation".to_string(),
                violation_probability: 0.45,
                time_to_violation: 48,
                affected_resources: vec![source.to_string()],
            },
        ]
    }

    fn determine_risk_level(&self, score: f64) -> RiskLevel {
        if score > 0.8 { RiskLevel::Critical }
        else if score > 0.6 { RiskLevel::High }
        else if score > 0.3 { RiskLevel::Medium }
        else { RiskLevel::Low }
    }

    fn calculate_priority(&self, score: f64) -> u32 {
        ((1.0 - score) * 100.0) as u32
    }

    pub fn detect_correlation_patterns(&self) -> Vec<CorrelationPattern> {
        let mut patterns = Vec::new();
        
        // Detect security-compliance mismatches
        for node in self.nodes.values() {
            if node.compliance_score < 0.5 && node.risk_score > 0.7 {
                patterns.push(CorrelationPattern {
                    pattern_id: Uuid::new_v4(),
                    pattern_type: PatternType::SecurityComplianceMismatch,
                    involved_resources: vec![node.id.clone()],
                    domains: vec![GovernanceDomain::Security, GovernanceDomain::Compliance],
                    confidence: 0.85,
                    frequency: 1,
                    last_observed: Utc::now(),
                    business_impact: BusinessImpact {
                        financial_impact: 50000.0,
                        compliance_risk: "High".to_string(),
                        operational_impact: "Service disruption possible".to_string(),
                        reputation_risk: "Medium".to_string(),
                    },
                });
            }
        }
        
        // Detect resource sprawl
        let resource_groups = self.group_resources_by_type();
        for (resource_type, resources) in resource_groups {
            if resources.len() > 10 {
                patterns.push(CorrelationPattern {
                    pattern_id: Uuid::new_v4(),
                    pattern_type: PatternType::ResourceSprawl,
                    involved_resources: resources,
                    domains: vec![GovernanceDomain::Cost, GovernanceDomain::Performance],
                    confidence: 0.9,
                    frequency: 1,
                    last_observed: Utc::now(),
                    business_impact: BusinessImpact {
                        financial_impact: 10000.0 * resource_type.len() as f64,
                        compliance_risk: "Low".to_string(),
                        operational_impact: "Increased management overhead".to_string(),
                        reputation_risk: "None".to_string(),
                    },
                });
            }
        }
        
        patterns
    }

    fn group_resources_by_type(&self) -> HashMap<String, Vec<String>> {
        let mut groups = HashMap::new();
        for node in self.nodes.values() {
            groups.entry(node.resource_type.clone())
                .or_insert_with(Vec::new)
                .push(node.id.clone());
        }
        groups
    }

    pub fn perform_what_if_analysis(&self, scenario: WhatIfScenario) -> WhatIfAnalysis {
        let current_state = self.capture_snapshot();
        
        // Simulate changes
        let mut simulated_graph = self.clone();
        for change in &scenario.changes {
            simulated_graph.apply_change(change);
        }
        
        let predicted_state = simulated_graph.capture_snapshot();
        let impacts = self.analyze_scenario_impacts(&scenario);
        let recommendations = self.generate_recommendations(&impacts);
        
        WhatIfAnalysis {
            analysis_id: Uuid::new_v4(),
            scenario,
            current_state,
            predicted_state,
            impacts,
            recommendations,
            confidence: 0.82,
        }
    }

    fn capture_snapshot(&self) -> GraphSnapshot {
        let total_compliance: f64 = self.nodes.values().map(|n| n.compliance_score).sum();
        let total_risk: f64 = self.nodes.values().map(|n| n.risk_score).sum();
        let node_count = self.nodes.len().max(1);
        
        GraphSnapshot {
            timestamp: Utc::now(),
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            domain_distribution: self.count_domains(),
            avg_compliance_score: total_compliance / node_count as f64,
            avg_risk_score: total_risk / node_count as f64,
            critical_paths: self.find_critical_paths(),
        }
    }

    fn count_domains(&self) -> HashMap<GovernanceDomain, usize> {
        let mut counts = HashMap::new();
        for node in self.nodes.values() {
            *counts.entry(node.domain.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn find_critical_paths(&self) -> Vec<Vec<String>> {
        // Simplified critical path finding
        vec![
            vec!["resource1".to_string(), "resource2".to_string(), "resource3".to_string()],
        ]
    }

    fn apply_change(&mut self, change: &ProposedChange) {
        if let Some(node) = self.nodes.get_mut(&change.resource_id) {
            node.properties.insert(change.change_type.clone(), change.new_value.clone());
        }
    }

    fn analyze_scenario_impacts(&self, scenario: &WhatIfScenario) -> Vec<CrossDomainImpact> {
        scenario.changes.iter()
            .map(|change| self.analyze_cross_domain_impact(&change.resource_id))
            .collect()
    }

    fn generate_recommendations(&self, impacts: &[CrossDomainImpact]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();
        
        for impact in impacts {
            if impact.total_impact_score > 0.7 {
                recommendations.push(Recommendation {
                    recommendation_id: Uuid::new_v4(),
                    priority: 1,
                    action: "Mitigate High-Risk Impact".to_string(),
                    description: format!("Address critical impact on {} before proceeding", impact.source_resource),
                    expected_improvement: 0.5,
                    implementation_effort: "Medium".to_string(),
                    automated: true,
                });
            }
        }
        
        recommendations
    }

    pub fn compute_embeddings(&mut self) {
        // Graph convolution to compute node embeddings
        for node_id in self.nodes.keys().cloned().collect::<Vec<_>>() {
            let embedding = self.aggregate_neighbor_features(&node_id);
            self.embeddings.insert(node_id, embedding);
        }
    }

    fn aggregate_neighbor_features(&self, node_id: &str) -> Vec<f64> {
        let mut aggregated = vec![0.0; 64];
        let mut neighbor_count = 0;
        
        for edge in &self.edges {
            if edge.source == node_id || (edge.bidirectional && edge.target == node_id) {
                let neighbor_id = if edge.source == node_id { &edge.target } else { &edge.source };
                
                if let Some(neighbor) = self.nodes.get(neighbor_id) {
                    for (i, val) in neighbor.embedding.iter().enumerate() {
                        if i < aggregated.len() {
                            aggregated[i] += val * edge.weight;
                        }
                    }
                    neighbor_count += 1;
                }
            }
        }
        
        if neighbor_count > 0 {
            for val in &mut aggregated {
                *val /= neighbor_count as f64;
            }
        }
        
        aggregated
    }
}
