// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Resource Dependency Mapper
// Maps and tracks dependencies between Azure resources

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{all_simple_paths, has_path_connecting};
use petgraph::visit::EdgeRef;

/// Resource mapper for dependency tracking
pub struct ResourceMapper {
    dependency_graph: DiGraph<Resource, Dependency>,
    resource_index: HashMap<String, NodeIndex>,
    reverse_index: HashMap<NodeIndex, String>,
}

impl ResourceMapper {
    pub fn new() -> Self {
        Self {
            dependency_graph: DiGraph::new(),
            resource_index: HashMap::new(),
            reverse_index: HashMap::new(),
        }
    }
    
    /// Build resource map from Azure resources
    pub fn build_map(&mut self, resources: Vec<ResourceInfo>) -> ResourceMap {
        self.clear();
        
        // Add all resources as nodes
        for resource in &resources {
            let node_idx = self.dependency_graph.add_node(Resource {
                id: resource.id.clone(),
                name: resource.name.clone(),
                resource_type: resource.resource_type.clone(),
                location: resource.location.clone(),
                metadata: resource.metadata.clone(),
            });
            
            self.resource_index.insert(resource.id.clone(), node_idx);
            self.reverse_index.insert(node_idx, resource.id.clone());
        }
        
        // Add dependencies as edges
        for resource in &resources {
            if let Some(source_idx) = self.resource_index.get(&resource.id) {
                // Add explicit dependencies
                for dep in &resource.explicit_dependencies {
                    if let Some(target_idx) = self.resource_index.get(dep) {
                        self.dependency_graph.add_edge(
                            *source_idx,
                            *target_idx,
                            Dependency {
                                dependency_type: DependencyType::Explicit,
                                strength: 1.0,
                                critical: true,
                            },
                        );
                    }
                }
                
                // Infer implicit dependencies
                let implicit_deps = self.infer_dependencies(resource, &resources);
                for (dep_id, dep_type) in implicit_deps {
                    if let Some(target_idx) = self.resource_index.get(&dep_id) {
                        self.dependency_graph.add_edge(
                            *source_idx,
                            *target_idx,
                            Dependency {
                                dependency_type: dep_type,
                                strength: 0.7,
                                critical: false,
                            },
                        );
                    }
                }
            }
        }
        
        // Generate resource map
        ResourceMap {
            total_resources: resources.len(),
            total_dependencies: self.dependency_graph.edge_count(),
            dependency_chains: self.find_dependency_chains(),
            resource_clusters: self.find_resource_clusters(),
            critical_resources: self.identify_critical_resources(),
            dependency_depth: self.calculate_max_depth(),
        }
    }
    
    /// Find all dependencies for a resource
    pub fn get_dependencies(&self, resource_id: &str) -> DependencyMap {
        let mut direct_deps = Vec::new();
        let mut transitive_deps = Vec::new();
        let mut dependents = Vec::new();
        
        if let Some(&node_idx) = self.resource_index.get(resource_id) {
            // Direct dependencies (outgoing edges)
            for edge in self.dependency_graph.edges(node_idx) {
                let target_idx = edge.target();
                if let Some(target_id) = self.reverse_index.get(&target_idx) {
                    direct_deps.push(DependencyInfo {
                        resource_id: target_id.clone(),
                        dependency_type: edge.weight().dependency_type.clone(),
                        strength: edge.weight().strength,
                        critical: edge.weight().critical,
                    });
                }
            }
            
            // Transitive dependencies (BFS)
            transitive_deps = self.find_transitive_dependencies(node_idx);
            
            // Resources that depend on this one (incoming edges)
            for edge in self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                let source_idx = edge.source();
                if let Some(source_id) = self.reverse_index.get(&source_idx) {
                    dependents.push(DependentInfo {
                        resource_id: source_id.clone(),
                        dependency_type: edge.weight().dependency_type.clone(),
                        impact_level: self.calculate_impact_level(edge.weight()),
                    });
                }
            }
        }
        
        DependencyMap {
            resource_id: resource_id.to_string(),
            direct_dependencies: direct_deps,
            transitive_dependencies: transitive_deps,
            dependents,
            total_impact_score: self.calculate_total_impact(resource_id),
        }
    }
    
    fn clear(&mut self) {
        self.dependency_graph.clear();
        self.resource_index.clear();
        self.reverse_index.clear();
    }
    
    fn infer_dependencies(&self, resource: &ResourceInfo, all_resources: &[ResourceInfo]) -> Vec<(String, DependencyType)> {
        let mut inferred = Vec::new();
        
        // Infer network dependencies
        if resource.resource_type.contains("VirtualMachine") {
            for other in all_resources {
                if other.resource_type.contains("VirtualNetwork") && 
                   other.location == resource.location {
                    inferred.push((other.id.clone(), DependencyType::Network));
                }
            }
        }
        
        // Infer storage dependencies
        if resource.resource_type.contains("WebApp") || resource.resource_type.contains("FunctionApp") {
            for other in all_resources {
                if other.resource_type.contains("StorageAccount") &&
                   other.metadata.get("tier") == Some(&"Standard".to_string()) {
                    inferred.push((other.id.clone(), DependencyType::Storage));
                }
            }
        }
        
        // Infer identity dependencies
        if resource.metadata.contains_key("managedIdentity") {
            for other in all_resources {
                if other.resource_type.contains("KeyVault") {
                    inferred.push((other.id.clone(), DependencyType::Identity));
                }
            }
        }
        
        inferred
    }
    
    fn find_dependency_chains(&self) -> Vec<DependencyChain> {
        let mut chains = Vec::new();
        
        // Find all nodes with no incoming edges (roots)
        let roots: Vec<_> = self.dependency_graph.node_indices()
            .filter(|&idx| {
                self.dependency_graph.edges_directed(idx, petgraph::Direction::Incoming).count() == 0
            })
            .collect();
        
        // Find all nodes with no outgoing edges (leaves)
        let leaves: Vec<_> = self.dependency_graph.node_indices()
            .filter(|&idx| {
                self.dependency_graph.edges_directed(idx, petgraph::Direction::Outgoing).count() == 0
            })
            .collect();
        
        // Find paths from roots to leaves
        for &root in &roots {
            for &leaf in &leaves {
                if has_path_connecting(&self.dependency_graph, root, leaf, None) {
                    let paths: Vec<Vec<NodeIndex>> = all_simple_paths(&self.dependency_graph, root, leaf, 0, Some(10))
                        .collect();
                    
                    for path in paths {
                        let resource_path: Vec<String> = path.iter()
                            .filter_map(|idx| self.reverse_index.get(idx))
                            .cloned()
                            .collect();
                        
                        if resource_path.len() > 2 {  // Only include chains with 3+ resources
                            chains.push(DependencyChain {
                                chain_id: format!("chain_{}", chains.len()),
                                resources: resource_path.clone(),
                                length: resource_path.len(),
                                critical: self.is_chain_critical(&path),
                            });
                        }
                    }
                }
            }
        }
        
        chains
    }
    
    fn find_resource_clusters(&self) -> Vec<ResourceCluster> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();
        
        for node_idx in self.dependency_graph.node_indices() {
            if !visited.contains(&node_idx) {
                let cluster_nodes = self.find_connected_component(node_idx, &mut visited);
                
                if cluster_nodes.len() > 1 {
                    let resources: Vec<String> = cluster_nodes.iter()
                        .filter_map(|idx| self.reverse_index.get(idx))
                        .cloned()
                        .collect();
                    
                    clusters.push(ResourceCluster {
                        cluster_id: format!("cluster_{}", clusters.len()),
                        resources: resources.clone(),
                        size: resources.len(),
                        density: self.calculate_cluster_density(&cluster_nodes),
                    });
                }
            }
        }
        
        clusters
    }
    
    fn find_connected_component(&self, start: NodeIndex, visited: &mut HashSet<NodeIndex>) -> Vec<NodeIndex> {
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(node) = queue.pop_front() {
            component.push(node);
            
            // Add neighbors (both directions)
            for neighbor in self.dependency_graph.neighbors_undirected(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        component
    }
    
    fn identify_critical_resources(&self) -> Vec<String> {
        let mut critical = Vec::new();
        
        for node_idx in self.dependency_graph.node_indices() {
            let in_degree = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Incoming).count();
            let out_degree = self.dependency_graph.edges_directed(node_idx, petgraph::Direction::Outgoing).count();
            
            // Resource is critical if many resources depend on it
            if in_degree > 3 {
                if let Some(resource_id) = self.reverse_index.get(&node_idx) {
                    critical.push(resource_id.clone());
                }
            }
        }
        
        critical
    }
    
    fn calculate_max_depth(&self) -> usize {
        let mut max_depth = 0;
        
        // Find all root nodes
        let roots: Vec<_> = self.dependency_graph.node_indices()
            .filter(|&idx| {
                self.dependency_graph.edges_directed(idx, petgraph::Direction::Incoming).count() == 0
            })
            .collect();
        
        // BFS from each root to find max depth
        for root in roots {
            let depth = self.bfs_depth(root);
            max_depth = max_depth.max(depth);
        }
        
        max_depth
    }
    
    fn bfs_depth(&self, start: NodeIndex) -> usize {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut max_depth = 0;
        
        queue.push_back((start, 0));
        visited.insert(start);
        
        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            
            for neighbor in self.dependency_graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        
        max_depth
    }
    
    fn find_transitive_dependencies(&self, start: NodeIndex) -> Vec<String> {
        let mut transitive = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start with direct dependencies
        for neighbor in self.dependency_graph.neighbors(start) {
            queue.push_back(neighbor);
            visited.insert(neighbor);
        }
        
        // BFS to find all transitive dependencies
        while let Some(node) = queue.pop_front() {
            if let Some(resource_id) = self.reverse_index.get(&node) {
                transitive.push(resource_id.clone());
            }
            
            for neighbor in self.dependency_graph.neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        transitive
    }
    
    fn calculate_impact_level(&self, dependency: &Dependency) -> ImpactLevel {
        if dependency.critical {
            ImpactLevel::Critical
        } else if dependency.strength > 0.8 {
            ImpactLevel::High
        } else if dependency.strength > 0.5 {
            ImpactLevel::Medium
        } else {
            ImpactLevel::Low
        }
    }
    
    fn calculate_total_impact(&self, resource_id: &str) -> f64 {
        if let Some(&node_idx) = self.resource_index.get(resource_id) {
            let dependent_count = self.dependency_graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .count() as f64;
            
            let max_strength: f64 = self.dependency_graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .map(|e| e.weight().strength)
                .fold(0.0, f64::max);
            
            (dependent_count * 0.3 + max_strength * 0.7).min(1.0)
        } else {
            0.0
        }
    }
    
    fn is_chain_critical(&self, path: &[NodeIndex]) -> bool {
        // Check if any edge in the path is critical
        for window in path.windows(2) {
            if let (Some(&from), Some(&to)) = (window.first(), window.get(1)) {
                if let Some(edge) = self.dependency_graph.find_edge(from, to) {
                    if self.dependency_graph[edge].critical {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    fn calculate_cluster_density(&self, nodes: &[NodeIndex]) -> f64 {
        if nodes.len() < 2 {
            return 0.0;
        }
        
        let mut edge_count = 0;
        for &node in nodes {
            for &other in nodes {
                if node != other && self.dependency_graph.find_edge(node, other).is_some() {
                    edge_count += 1;
                }
            }
        }
        
        let max_edges = nodes.len() * (nodes.len() - 1);
        edge_count as f64 / max_edges as f64
    }
}

// Data structures

#[derive(Debug, Clone)]
struct Resource {
    id: String,
    name: String,
    resource_type: String,
    location: String,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct Dependency {
    dependency_type: DependencyType,
    strength: f64,
    critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Explicit,
    Network,
    Storage,
    Identity,
    Data,
    Configuration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub explicit_dependencies: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMap {
    pub total_resources: usize,
    pub total_dependencies: usize,
    pub dependency_chains: Vec<DependencyChain>,
    pub resource_clusters: Vec<ResourceCluster>,
    pub critical_resources: Vec<String>,
    pub dependency_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyChain {
    pub chain_id: String,
    pub resources: Vec<String>,
    pub length: usize,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCluster {
    pub cluster_id: String,
    pub resources: Vec<String>,
    pub size: usize,
    pub density: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyMap {
    pub resource_id: String,
    pub direct_dependencies: Vec<DependencyInfo>,
    pub transitive_dependencies: Vec<String>,
    pub dependents: Vec<DependentInfo>,
    pub total_impact_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub resource_id: String,
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependentInfo {
    pub resource_id: String,
    pub dependency_type: DependencyType,
    pub impact_level: ImpactLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
}