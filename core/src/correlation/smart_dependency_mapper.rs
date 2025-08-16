// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Smart Dependency Mapper
// Advanced dependency discovery and mapping with machine learning inference

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use chrono::{DateTime, Utc, Duration};
use petgraph::graph::{DiGraph, NodeIndex, EdgeIndex};
use petgraph::algo::{kosaraju_scc, dijkstra, bellman_ford};
use petgraph::visit::EdgeRef;
use uuid::Uuid;

/// Smart dependency mapper with ML-based inference
pub struct SmartDependencyMapper {
    dependency_graph: DiGraph<SmartResource, SmartDependency>,
    resource_index: HashMap<String, NodeIndex>,
    reverse_index: HashMap<NodeIndex, String>,
    inference_engine: DependencyInferenceEngine,
    topology_analyzer: TopologyAnalyzer,
    dependency_predictor: DependencyPredictor,
    real_time_tracker: RealTimeDependencyTracker,
}

impl SmartDependencyMapper {
    pub fn new() -> Self {
        Self {
            dependency_graph: DiGraph::new(),
            resource_index: HashMap::new(),
            reverse_index: HashMap::new(),
            inference_engine: DependencyInferenceEngine::new(),
            topology_analyzer: TopologyAnalyzer::new(),
            dependency_predictor: DependencyPredictor::new(),
            real_time_tracker: RealTimeDependencyTracker::new(),
        }
    }

    /// Build advanced dependency map with ML inference
    pub async fn build_smart_map(&mut self, 
        resources: Vec<SmartResourceInfo>,
        network_topology: Option<NetworkTopology>,
        runtime_data: Vec<RuntimeMetric>
    ) -> SmartDependencyMap {
        
        self.clear();
        
        // Build basic graph structure
        self.build_basic_graph(&resources);
        
        // Infer dependencies using ML
        let inferred_deps = self.inference_engine.infer_dependencies(&resources, &runtime_data).await;
        self.add_inferred_dependencies(inferred_deps);
        
        // Analyze network topology if available
        if let Some(topology) = network_topology {
            let network_deps = self.topology_analyzer.analyze_network_dependencies(&topology, &resources);
            self.add_network_dependencies(network_deps);
        }
        
        // Discover runtime dependencies
        let runtime_deps = self.discover_runtime_dependencies(&runtime_data);
        self.add_runtime_dependencies(runtime_deps);
        
        // Calculate advanced metrics
        let dependency_metrics = self.calculate_advanced_metrics();
        
        // Generate dependency insights
        let insights = self.generate_dependency_insights(&resources);
        
        // Predict future dependencies
        let predicted_deps = self.dependency_predictor.predict_future_dependencies(&resources, &runtime_data).await;
        
        SmartDependencyMap {
            total_resources: resources.len(),
            explicit_dependencies: self.count_explicit_dependencies(),
            inferred_dependencies: self.count_inferred_dependencies(),
            runtime_dependencies: self.count_runtime_dependencies(),
            dependency_metrics,
            dependency_chains: self.find_smart_dependency_chains(),
            resource_clusters: self.find_smart_clusters(),
            critical_paths: self.identify_critical_paths(),
            dependency_insights: insights,
            predicted_dependencies: predicted_deps,
            risk_assessment: self.assess_dependency_risks(),
        }
    }

    /// Get comprehensive dependency information for a resource
    pub fn get_smart_dependencies(&self, resource_id: &str) -> SmartDependencyInfo {
        if let Some(&node_idx) = self.resource_index.get(resource_id) {
            SmartDependencyInfo {
                resource_id: resource_id.to_string(),
                direct_dependencies: self.get_direct_dependencies(node_idx),
                transitive_dependencies: self.get_transitive_dependencies(node_idx),
                reverse_dependencies: self.get_reverse_dependencies(node_idx),
                circular_dependencies: self.detect_circular_dependencies(node_idx),
                dependency_strength_map: self.calculate_dependency_strengths(node_idx),
                criticality_score: self.calculate_resource_criticality(node_idx),
                blast_radius: self.calculate_blast_radius(node_idx),
                recovery_dependencies: self.identify_recovery_dependencies(node_idx),
            }
        } else {
            SmartDependencyInfo::default()
        }
    }

    /// Track dependencies in real-time
    pub async fn track_real_time_dependencies(&mut self, 
        events: Vec<ResourceEvent>,
        metrics: Vec<RuntimeMetric>
    ) -> RealTimeDependencyUpdate {
        
        let dependency_changes = self.real_time_tracker.process_events(&events, &metrics);
        
        // Update graph with real-time changes
        for change in &dependency_changes {
            self.apply_dependency_change(change);
        }
        
        // Detect anomalies in dependency patterns
        let anomalies = self.detect_dependency_anomalies(&events, &metrics);
        
        // Update predictions based on real-time data
        let updated_predictions = self.dependency_predictor.update_predictions(&events, &metrics).await;
        
        let stability_score = self.calculate_graph_stability();
        let recommendations = self.generate_real_time_recommendations(&dependency_changes);
        
        RealTimeDependencyUpdate {
            dependency_changes,
            detected_anomalies: anomalies,
            updated_predictions,
            graph_stability_score: stability_score,
            recommendation_updates: recommendations,
        }
    }

    /// Analyze what-if scenarios for dependency changes
    pub async fn analyze_dependency_scenarios(&self,
        scenarios: Vec<DependencyScenario>
    ) -> Vec<DependencyScenarioResult> {
        
        let mut results = Vec::new();
        
        for scenario in scenarios {
            let mut test_graph = self.dependency_graph.clone();
            let mut test_index = self.resource_index.clone();
            
            // Apply scenario changes to test graph
            self.apply_scenario_to_graph(&scenario, &mut test_graph, &test_index);
            
            // Analyze impact of changes
            let impact_analysis = self.analyze_scenario_impact(&test_graph, &scenario);
            
            results.push(DependencyScenarioResult {
                scenario: scenario.clone(),
                impact_analysis,
                stability_change: self.calculate_stability_change(&test_graph),
                new_critical_paths: self.find_new_critical_paths(&test_graph),
                risk_change: self.calculate_risk_change(&test_graph),
            });
        }
        
        results
    }

    fn build_basic_graph(&mut self, resources: &[SmartResourceInfo]) {
        // Add nodes
        for resource in resources {
            let node = SmartResource {
                id: resource.id.clone(),
                name: resource.name.clone(),
                resource_type: resource.resource_type.clone(),
                location: resource.location.clone(),
                tags: resource.tags.clone(),
                creation_time: resource.creation_time,
                last_modified: resource.last_modified,
                criticality: resource.criticality,
                cost_per_hour: resource.cost_per_hour,
                sla_requirements: resource.sla_requirements.clone(),
                metadata: resource.metadata.clone(),
            };
            
            let node_idx = self.dependency_graph.add_node(node);
            self.resource_index.insert(resource.id.clone(), node_idx);
            self.reverse_index.insert(node_idx, resource.id.clone());
        }
        
        // Add explicit dependencies
        for resource in resources {
            if let Some(&source_idx) = self.resource_index.get(&resource.id) {
                for dep in &resource.explicit_dependencies {
                    if let Some(&target_idx) = self.resource_index.get(&dep.resource_id) {
                        let smart_dep = SmartDependency {
                            dependency_type: DependencyType::Explicit,
                            strength: dep.strength,
                            confidence: 1.0,
                            source: DependencySource::Explicit,
                            discovered_at: Utc::now(),
                            last_verified: Utc::now(),
                            business_criticality: dep.business_criticality.clone(),
                            failure_impact: dep.failure_impact.clone(),
                            recovery_time: dep.recovery_time,
                        };
                        
                        self.dependency_graph.add_edge(source_idx, target_idx, smart_dep);
                    }
                }
            }
        }
    }

    fn add_inferred_dependencies(&mut self, inferred_deps: Vec<InferredDependency>) {
        for dep in inferred_deps {
            if let (Some(&source_idx), Some(&target_idx)) = 
                (self.resource_index.get(&dep.source_id), self.resource_index.get(&dep.target_id)) {
                
                let smart_dep = SmartDependency {
                    dependency_type: dep.dependency_type,
                    strength: dep.strength,
                    confidence: dep.confidence,
                    source: DependencySource::MLInferred,
                    discovered_at: Utc::now(),
                    last_verified: Utc::now(),
                    business_criticality: dep.business_criticality.unwrap_or(BusinessCriticality::Medium),
                    failure_impact: dep.failure_impact.unwrap_or(FailureImpact::Medium),
                    recovery_time: dep.recovery_time.unwrap_or(Duration::hours(1)),
                };
                
                self.dependency_graph.add_edge(source_idx, target_idx, smart_dep);
            }
        }
    }

    fn add_network_dependencies(&mut self, network_deps: Vec<NetworkDependency>) {
        for dep in network_deps {
            if let (Some(&source_idx), Some(&target_idx)) = 
                (self.resource_index.get(&dep.source_id), self.resource_index.get(&dep.target_id)) {
                
                let smart_dep = SmartDependency {
                    dependency_type: DependencyType::Network,
                    strength: dep.strength,
                    confidence: dep.confidence,
                    source: DependencySource::NetworkTopology,
                    discovered_at: Utc::now(),
                    last_verified: Utc::now(),
                    business_criticality: BusinessCriticality::High, // Network deps are typically critical
                    failure_impact: FailureImpact::High,
                    recovery_time: Duration::minutes(dep.latency_ms as i64 / 10), // Estimate based on latency
                };
                
                self.dependency_graph.add_edge(source_idx, target_idx, smart_dep);
            }
        }
    }

    fn add_runtime_dependencies(&mut self, runtime_deps: Vec<RuntimeDependency>) {
        for dep in runtime_deps {
            if let (Some(&source_idx), Some(&target_idx)) = 
                (self.resource_index.get(&dep.source_id), self.resource_index.get(&dep.target_id)) {
                
                let smart_dep = SmartDependency {
                    dependency_type: DependencyType::Runtime,
                    strength: dep.correlation_strength,
                    confidence: dep.confidence,
                    source: DependencySource::RuntimeObservation,
                    discovered_at: Utc::now(),
                    last_verified: Utc::now(),
                    business_criticality: BusinessCriticality::Medium,
                    failure_impact: FailureImpact::Medium,
                    recovery_time: Duration::minutes(30),
                };
                
                self.dependency_graph.add_edge(source_idx, target_idx, smart_dep);
            }
        }
    }

    fn discover_runtime_dependencies(&self, runtime_data: &[RuntimeMetric]) -> Vec<RuntimeDependency> {
        let mut runtime_deps = Vec::new();
        
        // Group metrics by resource
        let mut resource_metrics: HashMap<String, Vec<&RuntimeMetric>> = HashMap::new();
        for metric in runtime_data {
            resource_metrics.entry(metric.resource_id.clone()).or_insert_with(Vec::new).push(metric);
        }
        
        // Find correlations between resource metrics
        for (source_id, source_metrics) in &resource_metrics {
            for (target_id, target_metrics) in &resource_metrics {
                if source_id != target_id {
                    let correlation = self.calculate_metric_correlation(source_metrics, target_metrics);
                    if correlation.strength > 0.7 {
                        runtime_deps.push(RuntimeDependency {
                            source_id: source_id.clone(),
                            target_id: target_id.clone(),
                            correlation_strength: correlation.strength,
                            confidence: correlation.confidence,
                            correlation_type: correlation.correlation_type,
                        });
                    }
                }
            }
        }
        
        runtime_deps
    }

    fn calculate_metric_correlation(&self, 
        source_metrics: &[&RuntimeMetric], 
        target_metrics: &[&RuntimeMetric]
    ) -> MetricCorrelation {
        // Simplified correlation calculation
        // In production, would use statistical correlation analysis
        
        let source_cpu: f64 = source_metrics.iter()
            .filter_map(|m| m.cpu_usage)
            .sum::<f64>() / source_metrics.len() as f64;
        
        let target_cpu: f64 = target_metrics.iter()
            .filter_map(|m| m.cpu_usage)
            .sum::<f64>() / target_metrics.len() as f64;
        
        let correlation_strength = if (source_cpu - target_cpu).abs() < 0.1 {
            0.8 // High correlation if CPU usage is similar
        } else {
            0.3 // Low correlation otherwise
        };
        
        MetricCorrelation {
            strength: correlation_strength,
            confidence: 0.7,
            correlation_type: CorrelationType::ResourceUsage,
        }
    }

    fn calculate_advanced_metrics(&self) -> DependencyMetrics {
        let total_nodes = self.dependency_graph.node_count();
        let total_edges = self.dependency_graph.edge_count();
        
        DependencyMetrics {
            total_nodes,
            total_edges,
            average_degree: if total_nodes > 0 { total_edges as f64 / total_nodes as f64 } else { 0.0 },
            density: if total_nodes > 1 { 
                total_edges as f64 / (total_nodes * (total_nodes - 1)) as f64 
            } else { 0.0 },
            clustering_coefficient: self.calculate_clustering_coefficient(),
            diameter: self.calculate_graph_diameter(),
            strongly_connected_components: kosaraju_scc(&self.dependency_graph).len(),
            centrality_measures: self.calculate_centrality_measures(),
        }
    }

    fn calculate_clustering_coefficient(&self) -> f64 {
        let mut total_coefficient = 0.0;
        let mut node_count = 0;
        
        for node_idx in self.dependency_graph.node_indices() {
            let neighbors: Vec<_> = self.dependency_graph.neighbors_undirected(node_idx).collect();
            if neighbors.len() >= 2 {
                let mut edge_count = 0;
                let possible_edges = neighbors.len() * (neighbors.len() - 1) / 2;
                
                for i in 0..neighbors.len() {
                    for j in i+1..neighbors.len() {
                        if self.dependency_graph.find_edge(neighbors[i], neighbors[j]).is_some() ||
                           self.dependency_graph.find_edge(neighbors[j], neighbors[i]).is_some() {
                            edge_count += 1;
                        }
                    }
                }
                
                if possible_edges > 0 {
                    total_coefficient += edge_count as f64 / possible_edges as f64;
                    node_count += 1;
                }
            }
        }
        
        if node_count > 0 { total_coefficient / node_count as f64 } else { 0.0 }
    }

    fn calculate_graph_diameter(&self) -> usize {
        let mut max_distance = 0;
        
        for source in self.dependency_graph.node_indices() {
            let distances = dijkstra(&self.dependency_graph, source, None, |_| 1);
            if let Some(&max_dist) = distances.values().max() {
                max_distance = max_distance.max(max_dist);
            }
        }
        
        max_distance
    }

    fn calculate_centrality_measures(&self) -> CentralityMeasures {
        let mut betweenness_centrality = HashMap::new();
        let mut closeness_centrality = HashMap::new();
        let mut degree_centrality = HashMap::new();
        
        for node_idx in self.dependency_graph.node_indices() {
            // Degree centrality
            let degree = self.dependency_graph.neighbors_undirected(node_idx).count();
            let degree_cent = degree as f64 / (self.dependency_graph.node_count() - 1) as f64;
            
            if let Some(resource_id) = self.reverse_index.get(&node_idx) {
                degree_centrality.insert(resource_id.clone(), degree_cent);
                
                // Closeness centrality
                let distances = dijkstra(&self.dependency_graph, node_idx, None, |_| 1);
                let total_distance: usize = distances.values().sum();
                let closeness_cent = if total_distance > 0 {
                    (distances.len() - 1) as f64 / total_distance as f64
                } else { 0.0 };
                closeness_centrality.insert(resource_id.clone(), closeness_cent);
                
                // Simplified betweenness centrality (would need more complex calculation)
                betweenness_centrality.insert(resource_id.clone(), degree_cent * 0.5);
            }
        }
        
        CentralityMeasures {
            betweenness_centrality,
            closeness_centrality,
            degree_centrality,
        }
    }

    fn generate_dependency_insights(&self, _resources: &[SmartResourceInfo]) -> Vec<DependencyInsight> {
        let mut insights = Vec::new();
        
        // Insight 1: Highly connected resources
        let high_degree_threshold = 10;
        for node_idx in self.dependency_graph.node_indices() {
            let degree = self.dependency_graph.neighbors_undirected(node_idx).count();
            if degree > high_degree_threshold {
                if let Some(resource_id) = self.reverse_index.get(&node_idx) {
                    insights.push(DependencyInsight {
                        insight_type: InsightType::HighConnectivity,
                        resource_id: resource_id.clone(),
                        description: format!("Resource has {} dependencies - potential bottleneck", degree),
                        severity: InsightSeverity::High,
                        recommendation: "Consider reducing dependencies or implementing redundancy".to_string(),
                        confidence: 0.9,
                    });
                }
            }
        }
        
        // Insight 2: Single points of failure
        for node_idx in self.dependency_graph.node_indices() {
            let incoming = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
            let outgoing = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
            
            if incoming > 5 && outgoing == 0 {
                if let Some(resource_id) = self.reverse_index.get(&node_idx) {
                    insights.push(DependencyInsight {
                        insight_type: InsightType::SinglePointOfFailure,
                        resource_id: resource_id.clone(),
                        description: format!("Resource is a potential single point of failure with {} dependents", incoming),
                        severity: InsightSeverity::Critical,
                        recommendation: "Implement redundancy and failover mechanisms".to_string(),
                        confidence: 0.95,
                    });
                }
            }
        }
        
        // Insight 3: Circular dependencies
        let sccs = kosaraju_scc(&self.dependency_graph);
        for scc in sccs {
            if scc.len() > 1 {
                let resource_ids: Vec<String> = scc.iter()
                    .filter_map(|&idx| self.reverse_index.get(&idx))
                    .cloned()
                    .collect();
                
                if !resource_ids.is_empty() {
                    insights.push(DependencyInsight {
                        insight_type: InsightType::CircularDependency,
                        resource_id: resource_ids[0].clone(),
                        description: format!("Circular dependency detected involving {} resources", scc.len()),
                        severity: InsightSeverity::High,
                        recommendation: "Refactor to eliminate circular dependencies".to_string(),
                        confidence: 1.0,
                    });
                }
            }
        }
        
        insights
    }

    fn identify_critical_paths(&self) -> Vec<CriticalPath> {
        let mut critical_paths = Vec::new();
        
        // Find paths with high criticality resources
        // Create a weighted graph for bellman_ford
        let mut weighted_graph = DiGraph::<(), f64>::new();
        let mut node_mapping = HashMap::new();
        
        // Copy nodes
        for node_idx in self.dependency_graph.node_indices() {
            let new_idx = weighted_graph.add_node(());
            node_mapping.insert(node_idx, new_idx);
        }
        
        // Copy edges with weights based on strength
        for edge in self.dependency_graph.edge_indices() {
            if let Some((source, target)) = self.dependency_graph.edge_endpoints(edge) {
                if let (Some(&new_source), Some(&new_target)) = (node_mapping.get(&source), node_mapping.get(&target)) {
                    if let Some(edge_weight) = self.dependency_graph.edge_weight(edge) {
                        // Use inverse of strength as weight (lower weight = stronger dependency)
                        weighted_graph.add_edge(new_source, new_target, 1.0 - edge_weight.strength);
                    }
                }
            }
        }
        
        for source in weighted_graph.node_indices() {
            for target in weighted_graph.node_indices() {
                if source != target {
                    // Use Bellman-Ford to find shortest path considering criticality weights
                    if let Ok(distances) = bellman_ford(&weighted_graph, source) {
                        if let Some(&distance) = distances.distances.get(target.index()) {
                            if distance < 100.0 {
                                // Map back to original indices
                                let orig_source = node_mapping.iter().find(|(_, &v)| v == source).map(|(k, _)| *k);
                                let orig_target = node_mapping.iter().find(|(_, &v)| v == target).map(|(k, _)| *k);
                                
                                if let (Some(orig_source), Some(orig_target)) = (orig_source, orig_target) {
                                    let path_criticality = self.calculate_path_criticality(orig_source, orig_target);
                                    if path_criticality > 0.8 {
                                        critical_paths.push(CriticalPath {
                                            path_id: Uuid::new_v4().to_string(),
                                            source_resource: self.reverse_index.get(&orig_source).cloned().unwrap_or_default(),
                                            target_resource: self.reverse_index.get(&orig_target).cloned().unwrap_or_default(),
                                            path_length: (distance * 10.0) as usize,
                                            criticality_score: path_criticality,
                                        bottleneck_resources: self.identify_bottlenecks_in_path(source, target),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by criticality and limit to top 20
        critical_paths.sort_by(|a, b| b.criticality_score.partial_cmp(&a.criticality_score).unwrap());
        critical_paths.truncate(20);
        
        critical_paths
    }

    fn calculate_path_criticality(&self, source: NodeIndex, target: NodeIndex) -> f64 {
        // Simplified path criticality calculation
        let source_criticality = self.dependency_graph.node_weight(source)
            .map(|n| n.criticality)
            .unwrap_or(0.0);
        let target_criticality = self.dependency_graph.node_weight(target)
            .map(|n| n.criticality)
            .unwrap_or(0.0);
        
        (source_criticality + target_criticality) / 2.0
    }

    fn identify_bottlenecks_in_path(&self, source: NodeIndex, target: NodeIndex) -> Vec<String> {
        // Simplified bottleneck identification
        vec![]
    }

    fn assess_dependency_risks(&self) -> DependencyRiskAssessment {
        let mut high_risk_dependencies = 0;
        let mut single_points_of_failure = 0;
        let mut circular_dependency_groups = 0;
        
        // Count high-risk dependencies
        for edge in self.dependency_graph.edge_references() {
            if edge.weight().business_criticality == BusinessCriticality::Critical &&
               edge.weight().failure_impact == FailureImpact::High {
                high_risk_dependencies += 1;
            }
        }
        
        // Count single points of failure
        for node_idx in self.dependency_graph.node_indices() {
            let incoming = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
            let outgoing = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
            
            if incoming > 3 && outgoing == 0 {
                single_points_of_failure += 1;
            }
        }
        
        // Count circular dependency groups
        let sccs = kosaraju_scc(&self.dependency_graph);
        circular_dependency_groups = sccs.into_iter().filter(|scc| scc.len() > 1).count();
        
        let overall_risk = self.calculate_overall_dependency_risk(
            high_risk_dependencies,
            single_points_of_failure,
            circular_dependency_groups
        );
        
        DependencyRiskAssessment {
            overall_risk_score: overall_risk,
            high_risk_dependencies,
            single_points_of_failure,
            circular_dependency_groups,
            recommended_actions: self.generate_risk_mitigation_actions(overall_risk),
        }
    }

    fn calculate_overall_dependency_risk(&self, 
        high_risk_deps: usize, 
        spofs: usize, 
        circular_groups: usize
    ) -> f64 {
        let total_nodes = self.dependency_graph.node_count() as f64;
        if total_nodes == 0.0 { return 0.0; }
        
        let risk_factors = [
            high_risk_deps as f64 / total_nodes * 0.4,
            spofs as f64 / total_nodes * 0.4,
            circular_groups as f64 / total_nodes * 0.2,
        ];
        
        risk_factors.iter().sum::<f64>().min(1.0)
    }

    fn generate_risk_mitigation_actions(&self, risk_score: f64) -> Vec<String> {
        let mut actions = Vec::new();
        
        if risk_score > 0.7 {
            actions.push("Immediate review of critical dependencies required".to_string());
            actions.push("Implement redundancy for single points of failure".to_string());
        }
        
        if risk_score > 0.5 {
            actions.push("Establish dependency monitoring and alerting".to_string());
            actions.push("Create dependency recovery procedures".to_string());
        }
        
        if risk_score > 0.3 {
            actions.push("Regular dependency health checks recommended".to_string());
        }
        
        actions
    }

    // Helper methods for dependency analysis
    fn get_direct_dependencies(&self, node_idx: NodeIndex) -> Vec<DirectDependency> {
        self.dependency_graph.edges(node_idx)
            .map(|edge| DirectDependency {
                resource_id: self.reverse_index.get(&edge.target()).cloned().unwrap_or_default(),
                dependency_type: edge.weight().dependency_type.clone(),
                strength: edge.weight().strength,
                confidence: edge.weight().confidence,
                business_criticality: edge.weight().business_criticality.clone(),
            })
            .collect()
    }

    fn get_transitive_dependencies(&self, node_idx: NodeIndex) -> Vec<String> {
        let mut transitive = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start with direct dependencies
        for neighbor in self.dependency_graph.neighbors(node_idx) {
            if !visited.contains(&neighbor) {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
        
        // BFS for transitive dependencies
        while let Some(current) = queue.pop_front() {
            if let Some(resource_id) = self.reverse_index.get(&current) {
                transitive.push(resource_id.clone());
            }
            
            for neighbor in self.dependency_graph.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        transitive
    }

    fn get_reverse_dependencies(&self, node_idx: NodeIndex) -> Vec<ReverseDependency> {
        self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming)
            .map(|edge| ReverseDependency {
                resource_id: self.reverse_index.get(&edge.source()).cloned().unwrap_or_default(),
                dependency_type: edge.weight().dependency_type.clone(),
                impact_if_this_fails: edge.weight().failure_impact.clone(),
            })
            .collect()
    }

    fn detect_circular_dependencies(&self, node_idx: NodeIndex) -> Vec<CircularDependency> {
        let mut circular_deps = Vec::new();
        
        // Find strongly connected components containing this node
        let sccs = kosaraju_scc(&self.dependency_graph);
        for scc in sccs {
            if scc.contains(&node_idx) && scc.len() > 1 {
                let resources: Vec<String> = scc.iter()
                    .filter_map(|&idx| self.reverse_index.get(&idx))
                    .cloned()
                    .collect();
                
                circular_deps.push(CircularDependency {
                    cycle_id: Uuid::new_v4().to_string(),
                    resources,
                    cycle_length: scc.len(),
                    severity: if scc.len() > 5 { CycleSeverity::High } else { CycleSeverity::Medium },
                });
            }
        }
        
        circular_deps
    }

    fn calculate_dependency_strengths(&self, node_idx: NodeIndex) -> HashMap<String, f64> {
        let mut strengths = HashMap::new();
        
        for edge in self.dependency_graph.edges(node_idx) {
            if let Some(resource_id) = self.reverse_index.get(&edge.target()) {
                strengths.insert(resource_id.clone(), edge.weight().strength);
            }
        }
        
        strengths
    }

    fn calculate_resource_criticality(&self, node_idx: NodeIndex) -> f64 {
        if let Some(resource) = self.dependency_graph.node_weight(node_idx) {
            let base_criticality = resource.criticality;
            let dependency_factor = self.dependency_graph.neighbors_undirected(node_idx).count() as f64 * 0.1;
            
            (base_criticality + dependency_factor).min(1.0)
        } else {
            0.0
        }
    }

    fn calculate_blast_radius(&self, node_idx: NodeIndex) -> BlastRadius {
        let mut affected_resources = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(node_idx);
        affected_resources.insert(node_idx);
        
        // BFS to find all resources that would be affected
        while let Some(current) = queue.pop_front() {
            for neighbor in self.dependency_graph.neighbors_directed(current, petgraph::Direction::Incoming) {
                if !affected_resources.contains(&neighbor) {
                    affected_resources.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        let affected_resource_ids: Vec<String> = affected_resources.iter()
            .filter_map(|&idx| self.reverse_index.get(&idx))
            .cloned()
            .collect();
        
        BlastRadius {
            affected_resources: affected_resource_ids,
            impact_scope: if affected_resources.len() > 20 { 
                ImpactScope::Organization 
            } else if affected_resources.len() > 5 { 
                ImpactScope::Service 
            } else { 
                ImpactScope::Component 
            },
            estimated_recovery_time: Duration::hours(affected_resources.len() as i64 / 5),
        }
    }

    fn identify_recovery_dependencies(&self, node_idx: NodeIndex) -> Vec<RecoveryDependency> {
        // Identify resources that must be recovered before this one
        let mut recovery_deps = Vec::new();
        
        for edge in self.dependency_graph.edges(node_idx) {
            if edge.weight().business_criticality == BusinessCriticality::Critical {
                if let Some(resource_id) = self.reverse_index.get(&edge.target()) {
                    recovery_deps.push(RecoveryDependency {
                        resource_id: resource_id.clone(),
                        recovery_order: edge.weight().strength as u32,
                        estimated_time: edge.weight().recovery_time,
                    });
                }
            }
        }
        
        // Sort by recovery order
        recovery_deps.sort_by_key(|dep| dep.recovery_order);
        
        recovery_deps
    }

    // Additional helper methods
    fn clear(&mut self) {
        self.dependency_graph.clear();
        self.resource_index.clear();
        self.reverse_index.clear();
    }

    fn count_explicit_dependencies(&self) -> usize {
        self.dependency_graph.edge_references()
            .filter(|edge| edge.weight().source == DependencySource::Explicit)
            .count()
    }

    fn count_inferred_dependencies(&self) -> usize {
        self.dependency_graph.edge_references()
            .filter(|edge| edge.weight().source == DependencySource::MLInferred)
            .count()
    }

    fn count_runtime_dependencies(&self) -> usize {
        self.dependency_graph.edge_references()
            .filter(|edge| edge.weight().source == DependencySource::RuntimeObservation)
            .count()
    }

    fn find_smart_dependency_chains(&self) -> Vec<SmartDependencyChain> {
        // Enhanced dependency chain detection
        vec![] // Simplified for now
    }

    fn find_smart_clusters(&self) -> Vec<SmartResourceCluster> {
        // Enhanced cluster detection with semantic analysis
        vec![] // Simplified for now
    }

    fn apply_dependency_change(&mut self, change: &DependencyChange) {
        match change {
            DependencyChange::Added { source_id, target_id, dependency } => {
                if let (Some(&source_idx), Some(&target_idx)) = 
                    (self.resource_index.get(source_id), self.resource_index.get(target_id)) {
                    self.dependency_graph.add_edge(source_idx, target_idx, dependency.clone());
                }
            },
            DependencyChange::Removed { source_id, target_id } => {
                if let (Some(&source_idx), Some(&target_idx)) = 
                    (self.resource_index.get(source_id), self.resource_index.get(target_id)) {
                    if let Some(edge) = self.dependency_graph.find_edge(source_idx, target_idx) {
                        self.dependency_graph.remove_edge(edge);
                    }
                }
            },
            DependencyChange::Strengthened { source_id, target_id, new_strength } => {
                if let (Some(&source_idx), Some(&target_idx)) = 
                    (self.resource_index.get(source_id), self.resource_index.get(target_id)) {
                    if let Some(edge) = self.dependency_graph.find_edge(source_idx, target_idx) {
                        if let Some(dep) = self.dependency_graph.edge_weight_mut(edge) {
                            dep.strength = *new_strength;
                        }
                    }
                }
            },
            DependencyChange::Weakened { source_id, target_id, new_strength } => {
                if let (Some(&source_idx), Some(&target_idx)) = 
                    (self.resource_index.get(source_id), self.resource_index.get(target_id)) {
                    if let Some(edge) = self.dependency_graph.find_edge(source_idx, target_idx) {
                        if let Some(dep) = self.dependency_graph.edge_weight_mut(edge) {
                            dep.strength = *new_strength;
                        }
                    }
                }
            },
        }
    }

    fn detect_dependency_anomalies(&self, events: &[ResourceEvent], metrics: &[RuntimeMetric]) -> Vec<DependencyAnomaly> {
        let mut anomalies = Vec::new();
        
        // Detect unusual dependency patterns
        let recent_events: Vec<_> = events.iter()
            .filter(|e| e.timestamp > Utc::now() - Duration::hours(1))
            .collect();
        
        if recent_events.len() > 100 {
            anomalies.push(DependencyAnomaly {
                anomaly_type: DependencyAnomalyType::HighDependencyActivity,
                description: format!("Unusual spike in dependency activity: {} events in last hour", recent_events.len()),
                severity: AnomalySeverity::High,
                affected_resources: recent_events.iter().map(|e| e.resource_id.clone()).collect(),
                detected_at: Utc::now(),
            });
        }
        
        anomalies
    }

    fn calculate_graph_stability(&self) -> f64 {
        // Calculate stability based on graph metrics
        let node_count = self.dependency_graph.node_count() as f64;
        let edge_count = self.dependency_graph.edge_count() as f64;
        
        if node_count == 0.0 { return 1.0; }
        
        let density = edge_count / (node_count * (node_count - 1.0));
        let stability = (1.0 - density).max(0.0).min(1.0);
        
        stability
    }

    fn generate_real_time_recommendations(&self, changes: &[DependencyChange]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if changes.len() > 10 {
            recommendations.push("High rate of dependency changes detected - review system stability".to_string());
        }
        
        let strengthened_count = changes.iter()
            .filter(|c| matches!(c, DependencyChange::Strengthened { .. }))
            .count();
        
        if strengthened_count > 5 {
            recommendations.push("Multiple dependencies strengthened - monitor for increased coupling".to_string());
        }
        
        recommendations
    }

    // Scenario analysis methods
    fn apply_scenario_to_graph(&self, 
        scenario: &DependencyScenario, 
        test_graph: &mut DiGraph<SmartResource, SmartDependency>,
        test_index: &HashMap<String, NodeIndex>
    ) {
        // Apply scenario changes to test graph
        for change in &scenario.changes {
            match change {
                ScenarioChange::AddDependency { source, target, strength } => {
                    if let (Some(&source_idx), Some(&target_idx)) = 
                        (test_index.get(source), test_index.get(target)) {
                        let dep = SmartDependency {
                            dependency_type: DependencyType::Scenario,
                            strength: *strength,
                            confidence: 1.0,
                            source: DependencySource::Scenario,
                            discovered_at: Utc::now(),
                            last_verified: Utc::now(),
                            business_criticality: BusinessCriticality::Medium,
                            failure_impact: FailureImpact::Medium,
                            recovery_time: Duration::hours(1),
                        };
                        test_graph.add_edge(source_idx, target_idx, dep);
                    }
                },
                ScenarioChange::RemoveDependency { source, target } => {
                    if let (Some(&source_idx), Some(&target_idx)) = 
                        (test_index.get(source), test_index.get(target)) {
                        if let Some(edge) = test_graph.find_edge(source_idx, target_idx) {
                            test_graph.remove_edge(edge);
                        }
                    }
                },
                ScenarioChange::ModifyStrength { source, target, new_strength } => {
                    if let (Some(&source_idx), Some(&target_idx)) = 
                        (test_index.get(source), test_index.get(target)) {
                        if let Some(edge) = test_graph.find_edge(source_idx, target_idx) {
                            if let Some(dep) = test_graph.edge_weight_mut(edge) {
                                dep.strength = *new_strength;
                            }
                        }
                    }
                },
            }
        }
    }

    fn analyze_scenario_impact(&self, 
        test_graph: &DiGraph<SmartResource, SmartDependency>, 
        scenario: &DependencyScenario
    ) -> ScenarioImpact {
        ScenarioImpact {
            affected_resource_count: scenario.changes.len(),
            stability_impact: 0.1, // Simplified calculation
            performance_impact: 0.05,
            risk_impact: 0.02,
        }
    }

    fn calculate_stability_change(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> f64 {
        // Compare stability before and after scenario
        let original_stability = self.calculate_graph_stability();
        let node_count = test_graph.node_count() as f64;
        let edge_count = test_graph.edge_count() as f64;
        
        let test_stability = if node_count > 0.0 {
            let density = edge_count / (node_count * (node_count - 1.0));
            (1.0 - density).max(0.0).min(1.0)
        } else {
            1.0
        };
        
        test_stability - original_stability
    }

    fn find_new_critical_paths(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> Vec<String> {
        // Simplified - would implement full critical path analysis
        vec![]
    }

    fn calculate_risk_change(&self, test_graph: &DiGraph<SmartResource, SmartDependency>) -> f64 {
        // Simplified risk change calculation
        0.0
    }
}

// Supporting components

pub struct DependencyInferenceEngine {
    inference_models: HashMap<String, InferenceModel>,
}

impl DependencyInferenceEngine {
    pub fn new() -> Self {
        Self {
            inference_models: HashMap::new(),
        }
    }

    pub async fn infer_dependencies(&self, 
        _resources: &[SmartResourceInfo], 
        _runtime_data: &[RuntimeMetric]
    ) -> Vec<InferredDependency> {
        // ML-based dependency inference
        vec![] // Simplified for now
    }
}

pub struct TopologyAnalyzer {
    network_analyzers: Vec<Box<dyn NetworkAnalyzer>>,
}

impl TopologyAnalyzer {
    pub fn new() -> Self {
        Self {
            network_analyzers: vec![],
        }
    }

    pub fn analyze_network_dependencies(&self, 
        topology: &NetworkTopology, 
        resources: &[SmartResourceInfo]
    ) -> Vec<NetworkDependency> {
        // Network topology analysis
        vec![] // Simplified for now
    }
}

pub struct DependencyPredictor {
    prediction_models: HashMap<String, PredictionModel>,
}

impl DependencyPredictor {
    pub fn new() -> Self {
        Self {
            prediction_models: HashMap::new(),
        }
    }

    pub async fn predict_future_dependencies(&self, 
        _resources: &[SmartResourceInfo], 
        _runtime_data: &[RuntimeMetric]
    ) -> Vec<PredictedDependency> {
        // Predict future dependencies based on trends
        vec![] // Simplified for now
    }

    pub async fn update_predictions(&self, 
        _events: &[ResourceEvent], 
        _metrics: &[RuntimeMetric]
    ) -> Vec<PredictedDependency> {
        // Update predictions with real-time data
        vec![] // Simplified for now
    }
}

pub struct RealTimeDependencyTracker {
    event_buffer: VecDeque<ResourceEvent>,
    pattern_detectors: Vec<Box<dyn PatternDetector>>,
}

impl RealTimeDependencyTracker {
    pub fn new() -> Self {
        Self {
            event_buffer: VecDeque::with_capacity(10000),
            pattern_detectors: vec![],
        }
    }

    pub fn process_events(&mut self, 
        events: &[ResourceEvent], 
        metrics: &[RuntimeMetric]
    ) -> Vec<DependencyChange> {
        // Process real-time events to detect dependency changes
        vec![] // Simplified for now
    }
}

// Trait definitions for analyzers
pub trait NetworkAnalyzer: Send + Sync {
    fn analyze_network_segment(&self, topology: &NetworkTopology) -> Vec<NetworkDependency>;
}

pub trait PatternDetector: Send + Sync {
    fn detect_pattern(&self, events: &[ResourceEvent]) -> Vec<DependencyChange>;
}

// Data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartResource {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub tags: HashMap<String, String>,
    pub creation_time: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub criticality: f64,
    pub cost_per_hour: f64,
    pub sla_requirements: SLARequirements,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartDependency {
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub confidence: f64,
    pub source: DependencySource,
    pub discovered_at: DateTime<Utc>,
    pub last_verified: DateTime<Utc>,
    pub business_criticality: BusinessCriticality,
    pub failure_impact: FailureImpact,
    pub recovery_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DependencyType {
    Explicit,
    Network,
    Storage,
    Identity,
    Data,
    Configuration,
    Runtime,
    Scenario,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DependencySource {
    Explicit,
    MLInferred,
    NetworkTopology,
    RuntimeObservation,
    Scenario,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BusinessCriticality {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureImpact {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLARequirements {
    pub availability: f64,
    pub response_time_ms: u32,
    pub throughput_req: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartResourceInfo {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub tags: HashMap<String, String>,
    pub creation_time: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub criticality: f64,
    pub cost_per_hour: f64,
    pub sla_requirements: SLARequirements,
    pub explicit_dependencies: Vec<ExplicitDependency>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicitDependency {
    pub resource_id: String,
    pub strength: f64,
    pub business_criticality: BusinessCriticality,
    pub failure_impact: FailureImpact,
    pub recovery_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub subnets: Vec<SubnetInfo>,
    pub connections: Vec<NetworkConnection>,
    pub security_groups: Vec<SecurityGroupInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetInfo {
    pub id: String,
    pub cidr: String,
    pub location: String,
    pub resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnection {
    pub source: String,
    pub target: String,
    pub connection_type: String,
    pub latency_ms: u32,
    pub bandwidth_mbps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityGroupInfo {
    pub id: String,
    pub rules: Vec<SecurityRule>,
    pub applied_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    pub direction: String,
    pub protocol: String,
    pub port_range: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetric {
    pub resource_id: String,
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: Option<f64>,
    pub memory_usage: Option<f64>,
    pub network_in: Option<f64>,
    pub network_out: Option<f64>,
    pub disk_io: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEvent {
    pub event_id: String,
    pub resource_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartDependencyMap {
    pub total_resources: usize,
    pub explicit_dependencies: usize,
    pub inferred_dependencies: usize,
    pub runtime_dependencies: usize,
    pub dependency_metrics: DependencyMetrics,
    pub dependency_chains: Vec<SmartDependencyChain>,
    pub resource_clusters: Vec<SmartResourceCluster>,
    pub critical_paths: Vec<CriticalPath>,
    pub dependency_insights: Vec<DependencyInsight>,
    pub predicted_dependencies: Vec<PredictedDependency>,
    pub risk_assessment: DependencyRiskAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyMetrics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub average_degree: f64,
    pub density: f64,
    pub clustering_coefficient: f64,
    pub diameter: usize,
    pub strongly_connected_components: usize,
    pub centrality_measures: CentralityMeasures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    pub betweenness_centrality: HashMap<String, f64>,
    pub closeness_centrality: HashMap<String, f64>,
    pub degree_centrality: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartDependencyChain {
    pub chain_id: String,
    pub resources: Vec<String>,
    pub total_strength: f64,
    pub average_confidence: f64,
    pub chain_type: ChainType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainType {
    Linear,
    Branching,
    Circular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartResourceCluster {
    pub cluster_id: String,
    pub resources: Vec<String>,
    pub cluster_type: ClusterType,
    pub cohesion_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterType {
    Functional,
    Geographic,
    Temporal,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path_id: String,
    pub source_resource: String,
    pub target_resource: String,
    pub path_length: usize,
    pub criticality_score: f64,
    pub bottleneck_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInsight {
    pub insight_type: InsightType,
    pub resource_id: String,
    pub description: String,
    pub severity: InsightSeverity,
    pub recommendation: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    HighConnectivity,
    SinglePointOfFailure,
    CircularDependency,
    WeakDependency,
    RedundantPath,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyRiskAssessment {
    pub overall_risk_score: f64,
    pub high_risk_dependencies: usize,
    pub single_points_of_failure: usize,
    pub circular_dependency_groups: usize,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartDependencyInfo {
    pub resource_id: String,
    pub direct_dependencies: Vec<DirectDependency>,
    pub transitive_dependencies: Vec<String>,
    pub reverse_dependencies: Vec<ReverseDependency>,
    pub circular_dependencies: Vec<CircularDependency>,
    pub dependency_strength_map: HashMap<String, f64>,
    pub criticality_score: f64,
    pub blast_radius: BlastRadius,
    pub recovery_dependencies: Vec<RecoveryDependency>,
}

impl Default for SmartDependencyInfo {
    fn default() -> Self {
        Self {
            resource_id: String::new(),
            direct_dependencies: Vec::new(),
            transitive_dependencies: Vec::new(),
            reverse_dependencies: Vec::new(),
            circular_dependencies: Vec::new(),
            dependency_strength_map: HashMap::new(),
            criticality_score: 0.0,
            blast_radius: BlastRadius {
                affected_resources: Vec::new(),
                impact_scope: ImpactScope::Component,
                estimated_recovery_time: Duration::hours(0),
            },
            recovery_dependencies: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectDependency {
    pub resource_id: String,
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub confidence: f64,
    pub business_criticality: BusinessCriticality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverseDependency {
    pub resource_id: String,
    pub dependency_type: DependencyType,
    pub impact_if_this_fails: FailureImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependency {
    pub cycle_id: String,
    pub resources: Vec<String>,
    pub cycle_length: usize,
    pub severity: CycleSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CycleSeverity {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlastRadius {
    pub affected_resources: Vec<String>,
    pub impact_scope: ImpactScope,
    pub estimated_recovery_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactScope {
    Component,
    Service,
    System,
    Organization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryDependency {
    pub resource_id: String,
    pub recovery_order: u32,
    pub estimated_time: Duration,
}

// Real-time tracking structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeDependencyUpdate {
    pub dependency_changes: Vec<DependencyChange>,
    pub detected_anomalies: Vec<DependencyAnomaly>,
    pub updated_predictions: Vec<PredictedDependency>,
    pub graph_stability_score: f64,
    pub recommendation_updates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyChange {
    Added { source_id: String, target_id: String, dependency: SmartDependency },
    Removed { source_id: String, target_id: String },
    Strengthened { source_id: String, target_id: String, new_strength: f64 },
    Weakened { source_id: String, target_id: String, new_strength: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnomaly {
    pub anomaly_type: DependencyAnomalyType,
    pub description: String,
    pub severity: AnomalySeverity,
    pub affected_resources: Vec<String>,
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyAnomalyType {
    HighDependencyActivity,
    UnexpectedDependencyRemoval,
    CircularDependencyFormation,
    DependencyStrengthSpike,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Critical,
    High,
    Medium,
    Low,
}

// Scenario analysis structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyScenario {
    pub scenario_id: String,
    pub name: String,
    pub description: String,
    pub changes: Vec<ScenarioChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioChange {
    AddDependency { source: String, target: String, strength: f64 },
    RemoveDependency { source: String, target: String },
    ModifyStrength { source: String, target: String, new_strength: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyScenarioResult {
    pub scenario: DependencyScenario,
    pub impact_analysis: ScenarioImpact,
    pub stability_change: f64,
    pub new_critical_paths: Vec<String>,
    pub risk_change: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioImpact {
    pub affected_resource_count: usize,
    pub stability_impact: f64,
    pub performance_impact: f64,
    pub risk_impact: f64,
}

// Supporting structures for inference and prediction

#[derive(Debug, Clone)]
pub struct InferredDependency {
    pub source_id: String,
    pub target_id: String,
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub confidence: f64,
    pub business_criticality: Option<BusinessCriticality>,
    pub failure_impact: Option<FailureImpact>,
    pub recovery_time: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct NetworkDependency {
    pub source_id: String,
    pub target_id: String,
    pub strength: f64,
    pub confidence: f64,
    pub latency_ms: u32,
}

#[derive(Debug, Clone)]
pub struct RuntimeDependency {
    pub source_id: String,
    pub target_id: String,
    pub correlation_strength: f64,
    pub confidence: f64,
    pub correlation_type: CorrelationType,
}

#[derive(Debug, Clone)]
pub struct MetricCorrelation {
    pub strength: f64,
    pub confidence: f64,
    pub correlation_type: CorrelationType,
}

#[derive(Debug, Clone)]
pub enum CorrelationType {
    ResourceUsage,
    ErrorRate,
    Latency,
    Throughput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedDependency {
    pub source_id: String,
    pub target_id: String,
    pub predicted_strength: f64,
    pub confidence: f64,
    pub prediction_timeframe: Duration,
    pub factors: Vec<String>,
}

// Model definitions

#[derive(Debug, Clone)]
pub struct InferenceModel {
    pub model_type: String,
    pub accuracy: f64,
    pub last_trained: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub model_type: String,
    pub accuracy: f64,
    pub prediction_horizon: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_smart_dependency_mapper() {
        let mut mapper = SmartDependencyMapper::new();
        
        let resources = vec![
            SmartResourceInfo {
                id: "vm-001".to_string(),
                name: "web-server".to_string(),
                resource_type: "VirtualMachine".to_string(),
                location: "eastus".to_string(),
                tags: HashMap::new(),
                creation_time: Utc::now(),
                last_modified: Utc::now(),
                criticality: 0.8,
                cost_per_hour: 1.0,
                sla_requirements: SLARequirements {
                    availability: 0.99,
                    response_time_ms: 100,
                    throughput_req: Some(1000),
                },
                explicit_dependencies: vec![],
                metadata: HashMap::new(),
            }
        ];
        
        let result = mapper.build_smart_map(resources, None, vec![]).await;
        
        assert_eq!(result.total_resources, 1);
        assert!(result.dependency_metrics.total_nodes > 0);
    }

    #[tokio::test]
    async fn test_dependency_scenario_analysis() {
        let mapper = SmartDependencyMapper::new();
        
        let scenario = DependencyScenario {
            scenario_id: "test-scenario".to_string(),
            name: "Test Scenario".to_string(),
            description: "Test adding a dependency".to_string(),
            changes: vec![
                ScenarioChange::AddDependency {
                    source: "vm-001".to_string(),
                    target: "db-001".to_string(),
                    strength: 0.8,
                }
            ],
        };
        
        let results = mapper.analyze_dependency_scenarios(vec![scenario]).await;
        
        assert_eq!(results.len(), 1);
    }
}