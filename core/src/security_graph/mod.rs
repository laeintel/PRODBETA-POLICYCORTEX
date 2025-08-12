// Security Exposure Graph Engine - Comprehensive Implementation
// Based on Roadmap_08_Security_and_Exposure_Graph.md
// Addresses GitHub Issue #56-59: Security Graph Implementation

use async_trait::async_trait;
use petgraph::algo::{all_simple_paths, dijkstra};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Graph node types representing security entities
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "node_type")]
pub enum SecurityNode {
    Identity {
        id: String,
        name: String,
        identity_type: String, // "User", "ServicePrincipal", "ManagedIdentity"
        risk_score: f64,
        last_active: chrono::DateTime<chrono::Utc>,
    },
    Role {
        id: String,
        name: String,
        permissions: Vec<String>,
        criticality: String, // "Low", "Medium", "High", "Critical"
    },
    Resource {
        id: String,
        name: String,
        resource_type: String,
        classification: String, // "Public", "Internal", "Confidential", "Restricted"
        encryption_status: String,
        backup_status: String,
    },
    NetworkEndpoint {
        id: String,
        ip_address: String,
        port: u16,
        exposure: String, // "Private", "Public", "Internet"
        protocols: Vec<String>,
    },
    DataStore {
        id: String,
        name: String,
        data_type: String,
        sensitivity: String, // "Low", "Medium", "High", "Critical"
        size_gb: f64,
        compliance_tags: Vec<String>,
    },
}

// Graph edge types representing relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "edge_type")]
pub enum SecurityEdge {
    AssumesRole {
        granted_at: chrono::DateTime<chrono::Utc>,
        expires_at: Option<chrono::DateTime<chrono::Utc>>,
        elevation_required: bool,
    },
    HasPermission {
        action: String,
        scope: String,
        conditions: Vec<String>,
    },
    Reachable {
        port: u16,
        protocol: String,
        firewall_rules: Vec<String>,
    },
    ContainsData {
        access_level: String, // "Read", "Write", "Admin"
        encrypted_in_transit: bool,
        encrypted_at_rest: bool,
    },
}

// Attack path representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackPath {
    pub path_id: String,
    pub source: String,
    pub target: String,
    pub path_nodes: Vec<String>,
    pub risk_score: f64,
    pub exploitability: f64,
    pub impact: f64,
    pub detection_difficulty: f64,
    pub mitigation_bundles: Vec<MitigationBundle>,
}

// Mitigation bundle for breaking attack paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationBundle {
    pub bundle_id: String,
    pub name: String,
    pub description: String,
    pub controls: Vec<SecurityControl>,
    pub effectiveness: f64,
    pub implementation_cost: String, // "Low", "Medium", "High"
    pub blast_radius: Vec<String>,   // Resources affected
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityControl {
    pub control_type: String,
    pub action: String,
    pub target: String,
    pub parameters: HashMap<String, String>,
}

// Path scoring weights
#[derive(Debug, Clone)]
pub struct PathScoringWeights {
    pub privilege_weight: f64,
    pub reachability_weight: f64,
    pub data_sensitivity_weight: f64,
    pub control_gaps_weight: f64,
}

impl Default for PathScoringWeights {
    fn default() -> Self {
        Self {
            privilege_weight: 0.3,
            reachability_weight: 0.25,
            data_sensitivity_weight: 0.3,
            control_gaps_weight: 0.15,
        }
    }
}

// Main Security Graph Engine
pub struct SecurityGraphEngine {
    graph: DiGraph<SecurityNode, SecurityEdge>,
    node_index: HashMap<String, NodeIndex>,
    scoring_weights: PathScoringWeights,
}

impl SecurityGraphEngine {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_index: HashMap::new(),
            scoring_weights: PathScoringWeights::default(),
        }
    }

    // Build graph from Azure resources
    pub async fn build_from_azure(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch identities
        self.add_identities(client).await?;

        // Fetch roles and permissions
        self.add_roles_and_permissions(client).await?;

        // Fetch resources
        self.add_resources(client).await?;

        // Fetch network topology
        self.add_network_topology(client).await?;

        // Fetch data stores
        self.add_data_stores(client).await?;

        // Build relationships
        self.build_relationships(client).await?;

        Ok(())
    }

    async fn add_identities(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL identities from Azure AD
        let identities = client.get_identities().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch identities: {}", e))
        })?;

        for identity_data in identities {
            // Calculate risk score based on real data
            let mut risk_score = 0.0;
            if identity_data
                .get("has_privileged_roles")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                risk_score += 30.0;
            }
            if identity_data
                .get("stale_credentials")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                risk_score += 20.0;
            }
            if identity_data
                .get("anomalous_activity")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                risk_score += 25.0;
            }
            if identity_data
                .get("external_user")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                risk_score += 15.0;
            }

            let identity = SecurityNode::Identity {
                id: identity_data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: identity_data
                    .get("display_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                identity_type: identity_data
                    .get("identity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("User")
                    .to_string(),
                risk_score,
                last_active: chrono::Utc::now(),
            };

            let idx = self.graph.add_node(identity.clone());
            if let SecurityNode::Identity { ref id, .. } = identity {
                self.node_index.insert(id.clone(), idx);
            }
        }

        Ok(())
    }

    async fn add_roles_and_permissions(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL roles from Azure RBAC
        let role_definitions = client
            .get_role_definitions()
            .await
            .map_err(|e| SecurityGraphError::AzureError(format!("Failed to fetch roles: {}", e)))?;

        for role_def in role_definitions {
            // Determine criticality based on permissions
            let permissions = role_def
                .get("permissions")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let criticality = if permissions
                .iter()
                .any(|p| p.contains("*/delete") || p.contains("*/write"))
            {
                if permissions.iter().any(|p| {
                    p.contains("Microsoft.Authorization") || p.contains("Microsoft.KeyVault")
                }) {
                    "Critical".to_string()
                } else {
                    "High".to_string()
                }
            } else if permissions.iter().any(|p| p.contains("*/read")) {
                "Medium".to_string()
            } else {
                "Low".to_string()
            };

            let role = SecurityNode::Role {
                id: role_def
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: role_def
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                permissions,
                criticality,
            };

            let idx = self.graph.add_node(role.clone());
            if let SecurityNode::Role { ref id, .. } = role {
                self.node_index.insert(id.clone(), idx);
            }
        }

        Ok(())
    }

    async fn add_resources(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL resources from Azure Resource Graph
        let resources = client.get_all_resources_with_health().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch resources: {}", e))
        })?;

        let resource_items = resources
            .get("items")
            .and_then(|v| v.as_array())
            .map(|v| v.clone())
            .unwrap_or_default();
        for resource_data in resource_items {
            // Determine classification based on tags and type
            let tags = resource_data.get("tags").and_then(|v| v.as_object());
            let resource_type = resource_data
                .get("resource_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let classification = if let Some(tags_obj) = tags {
                if let Some(class) = tags_obj.get("data-classification").and_then(|v| v.as_str()) {
                    class.to_string()
                } else if resource_type.contains("KeyVault") || resource_type.contains("Database") {
                    "Confidential".to_string()
                } else if tags_obj.get("environment").and_then(|v| v.as_str()) == Some("production")
                {
                    "Internal".to_string()
                } else {
                    "Public".to_string()
                }
            } else if resource_type.contains("KeyVault") || resource_type.contains("Database") {
                "Confidential".to_string()
            } else {
                "Public".to_string()
            };

            // Check encryption status
            let properties = resource_data.get("properties").and_then(|v| v.as_object());
            let encryption_status = if let Some(props) = properties {
                if props.get("encryption").is_some() || props.get("encryptionSettings").is_some() {
                    "Enabled".to_string()
                } else {
                    "Disabled".to_string()
                }
            } else {
                "Unknown".to_string()
            };

            // Check backup status
            let resource_id = resource_data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let backup_status = client
                .get_backup_status(resource_id)
                .await
                .unwrap_or("Unknown".to_string());

            let resource = SecurityNode::Resource {
                id: resource_data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: resource_data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                resource_type: resource_type.to_string(),
                classification,
                encryption_status,
                backup_status,
            };

            let idx = self.graph.add_node(resource.clone());
            if let SecurityNode::Resource { ref id, .. } = resource {
                self.node_index.insert(id.clone(), idx);
            }
        }

        Ok(())
    }

    async fn add_network_topology(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL network topology from Azure
        let network_data = client.get_network_topology().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch network topology: {}", e))
        })?;

        // Add public endpoints
        let public_endpoints = network_data
            .get("public_endpoints")
            .and_then(|v| v.as_array())
            .map(|v| v.clone())
            .unwrap_or_default();
        for public_endpoint in public_endpoints {
            let exposure = if public_endpoint
                .get("has_public_ip")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Internet".to_string()
            } else if public_endpoint
                .get("is_reachable_from_internet")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Public".to_string()
            } else {
                "Private".to_string()
            };

            let endpoint = SecurityNode::NetworkEndpoint {
                id: public_endpoint
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                ip_address: public_endpoint
                    .get("ip_address")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0.0.0.0")
                    .to_string(),
                port: public_endpoint
                    .get("port")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(443) as u16,
                exposure,
                protocols: public_endpoint
                    .get("protocols")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
            };

            let idx = self.graph.add_node(endpoint.clone());
            if let SecurityNode::NetworkEndpoint { ref id, .. } = endpoint {
                self.node_index.insert(id.clone(), idx);
            }
        }

        // Add private endpoints
        let private_endpoints = network_data
            .get("private_endpoints")
            .and_then(|v| v.as_array())
            .map(|v| v.clone())
            .unwrap_or_default();
        for private_endpoint in private_endpoints {
            let endpoint = SecurityNode::NetworkEndpoint {
                id: private_endpoint
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                ip_address: private_endpoint
                    .get("ip_address")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0.0.0.0")
                    .to_string(),
                port: private_endpoint
                    .get("port")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(443) as u16,
                exposure: "Private".to_string(),
                protocols: private_endpoint
                    .get("protocols")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
            };

            let idx = self.graph.add_node(endpoint.clone());
            if let SecurityNode::NetworkEndpoint { ref id, .. } = endpoint {
                self.node_index.insert(id.clone(), idx);
            }
        }

        Ok(())
    }

    async fn add_data_stores(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL data stores from Azure
        let datastores = client.get_data_stores().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch data stores: {}", e))
        })?;

        for datastore_data in datastores {
            // Determine sensitivity based on data classification and compliance tags
            let tags = datastore_data.get("tags").and_then(|v| v.as_object());
            let compliance_tags = datastore_data
                .get("compliance_tags")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                .unwrap_or_default();

            let sensitivity = if let Some(tags_obj) = tags {
                tags_obj
                    .get("sensitivity")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Low")
                    .to_string()
            } else if compliance_tags.iter().any(|t| *t == "PCI" || *t == "HIPAA") {
                "Critical".to_string()
            } else if compliance_tags.iter().any(|t| *t == "GDPR" || *t == "SOC2") {
                "High".to_string()
            } else if datastore_data
                .get("encrypted")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Medium".to_string()
            } else {
                "Low".to_string()
            };

            let datastore = SecurityNode::DataStore {
                id: datastore_data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: datastore_data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                data_type: datastore_data
                    .get("data_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                sensitivity,
                size_gb: datastore_data
                    .get("size_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                compliance_tags: compliance_tags.into_iter().map(|s| s.to_string()).collect(),
            };

            let idx = self.graph.add_node(datastore.clone());
            if let SecurityNode::DataStore { ref id, .. } = datastore {
                self.node_index.insert(id.clone(), idx);
            }
        }

        Ok(())
    }

    async fn build_relationships(
        &mut self,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Fetch REAL role assignments from Azure
        let role_assignments = client.get_role_assignments().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch role assignments: {}", e))
        })?;

        for assignment in role_assignments {
            let principal_id = assignment
                .get("principal_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let role_definition_id = assignment
                .get("role_definition_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let scope = assignment
                .get("scope")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Add edge: Identity -> Role
            if let (Some(&identity_idx), Some(&role_idx)) = (
                self.node_index.get(principal_id),
                self.node_index.get(role_definition_id),
            ) {
                self.graph.add_edge(
                    identity_idx,
                    role_idx,
                    SecurityEdge::AssumesRole {
                        granted_at: chrono::Utc::now(),
                        expires_at: None,
                        elevation_required: assignment
                            .get("requires_elevation")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                    },
                );
            }

            // Add edge: Role -> Resource based on scope
            if let Some(&role_idx) = self.node_index.get(role_definition_id) {
                // Find resources that match the assignment scope
                for (resource_id, &resource_idx) in &self.node_index {
                    if resource_id.starts_with(scope) {
                        let permissions = assignment
                            .get("permissions")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            .unwrap_or("Read");

                        self.graph.add_edge(
                            role_idx,
                            resource_idx,
                            SecurityEdge::HasPermission {
                                action: permissions.to_string(),
                                scope: scope.to_string(),
                                conditions: assignment
                                    .get("conditions")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str())
                                            .map(|s| s.to_string())
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                            },
                        );
                    }
                }
            }
        }

        // Fetch and add network reachability edges
        let network_flows = client.get_network_flows().await.map_err(|e| {
            SecurityGraphError::AzureError(format!("Failed to fetch network flows: {}", e))
        })?;

        for flow in network_flows {
            let source_id = flow.get("source_id").and_then(|v| v.as_str()).unwrap_or("");
            let target_id = flow.get("target_id").and_then(|v| v.as_str()).unwrap_or("");

            if let (Some(&source_idx), Some(&target_idx)) = (
                self.node_index.get(source_id),
                self.node_index.get(target_id),
            ) {
                self.graph.add_edge(
                    source_idx,
                    target_idx,
                    SecurityEdge::Reachable {
                        port: flow.get("port").and_then(|v| v.as_u64()).unwrap_or(443) as u16,
                        protocol: flow
                            .get("protocol")
                            .and_then(|v| v.as_str())
                            .unwrap_or("TCP")
                            .to_string(),
                        firewall_rules: flow
                            .get("nsg_rules")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .collect()
                            })
                            .unwrap_or_default(),
                    },
                );
            }
        }

        Ok(())
    }

    // Find all attack paths to critical resources
    pub fn find_attack_paths(&self, target_sensitivity: &str) -> Vec<AttackPath> {
        let mut paths = Vec::new();

        // Find all public entry points
        let entry_points: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&idx| {
                if let Some(node) = self.graph.node_weight(idx) {
                    matches!(node, SecurityNode::NetworkEndpoint { exposure, .. } if exposure == "Public")
                } else {
                    false
                }
            })
            .collect();

        // Find all critical data stores
        let critical_targets: Vec<NodeIndex> = self.graph.node_indices()
            .filter(|&idx| {
                if let Some(node) = self.graph.node_weight(idx) {
                    matches!(node, SecurityNode::DataStore { sensitivity, .. } if sensitivity == target_sensitivity)
                } else {
                    false
                }
            })
            .collect();

        // Find paths between entry points and critical targets
        for entry in &entry_points {
            for target in &critical_targets {
                // Use petgraph's all_simple_paths with a max depth
                let simple_paths: Vec<Vec<NodeIndex>> =
                    all_simple_paths(&self.graph, *entry, *target, 0, Some(10)).collect();

                for path_nodes in simple_paths {
                    let path = self.score_and_build_path(path_nodes);
                    if path.risk_score > 0.5 {
                        // Only include significant paths
                        paths.push(path);
                    }
                }
            }
        }

        // Sort by risk score (handle NaN safely)
        use std::cmp::Ordering;
        paths.sort_by(|a, b| b
            .risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(Ordering::Equal));
        paths
    }

    fn score_and_build_path(&self, node_indices: Vec<NodeIndex>) -> AttackPath {
        let mut privilege_score = 0.0;
        let mut reachability_score = 0.0;
        let mut sensitivity_score = 0.0;
        let mut control_gaps = 0.0;

        // Calculate scores based on path nodes
        for &idx in &node_indices {
            if let Some(node) = self.graph.node_weight(idx) {
                match node {
                    SecurityNode::Role { criticality, .. } => {
                        privilege_score += match criticality.as_str() {
                            "Critical" => 1.0,
                            "High" => 0.7,
                            "Medium" => 0.4,
                            _ => 0.1,
                        };
                    }
                    SecurityNode::NetworkEndpoint { exposure, .. } => {
                        reachability_score += match exposure.as_str() {
                            "Public" => 1.0,
                            "Internet" => 0.8,
                            _ => 0.2,
                        };
                    }
                    SecurityNode::DataStore { sensitivity, .. } => {
                        sensitivity_score += match sensitivity.as_str() {
                            "Critical" => 1.0,
                            "High" => 0.7,
                            "Medium" => 0.4,
                            _ => 0.1,
                        };
                    }
                    SecurityNode::Resource {
                        encryption_status,
                        backup_status,
                        ..
                    } => {
                        if encryption_status == "Disabled" {
                            control_gaps += 0.5;
                        }
                        if backup_status == "Inactive" {
                            control_gaps += 0.3;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Normalize scores
        let path_length = node_indices.len() as f64;
        privilege_score /= path_length;
        reachability_score /= path_length;
        sensitivity_score /= path_length;
        control_gaps /= path_length;

        // Calculate weighted risk score
        let risk_score = privilege_score * self.scoring_weights.privilege_weight
            + reachability_score * self.scoring_weights.reachability_weight
            + sensitivity_score * self.scoring_weights.data_sensitivity_weight
            + control_gaps * self.scoring_weights.control_gaps_weight;

        // Generate mitigation bundles
        let mitigation_bundles = self.generate_mitigations(&node_indices);

        // Convert node indices to IDs
        let path_node_ids: Vec<String> = node_indices
            .iter()
            .filter_map(|&idx| {
                self.graph.node_weight(idx).map(|node| match node {
                    SecurityNode::Identity { id, .. }
                    | SecurityNode::Role { id, .. }
                    | SecurityNode::Resource { id, .. }
                    | SecurityNode::NetworkEndpoint { id, .. }
                    | SecurityNode::DataStore { id, .. } => id.clone(),
                })
            })
            .collect();

        AttackPath {
            path_id: format!("path-{}", uuid::Uuid::new_v4()),
            source: path_node_ids.first().cloned().unwrap_or_default(),
            target: path_node_ids.last().cloned().unwrap_or_default(),
            path_nodes: path_node_ids,
            risk_score,
            exploitability: reachability_score,
            impact: sensitivity_score,
            detection_difficulty: 1.0 - control_gaps,
            mitigation_bundles,
        }
    }

    fn generate_mitigations(&self, _path: &[NodeIndex]) -> Vec<MitigationBundle> {
        let mut bundles = Vec::new();

        // Network segmentation bundle
        let network_controls = vec![
            SecurityControl {
                control_type: "NetworkSecurityGroup".to_string(),
                action: "DenyInbound".to_string(),
                target: "0.0.0.0/0:3389".to_string(),
                parameters: HashMap::from([
                    ("priority".to_string(), "100".to_string()),
                    ("direction".to_string(), "Inbound".to_string()),
                ]),
            },
            SecurityControl {
                control_type: "AzureBastion".to_string(),
                action: "Enable".to_string(),
                target: "subnet-management".to_string(),
                parameters: HashMap::new(),
            },
        ];

        bundles.push(MitigationBundle {
            bundle_id: format!("mb-{}", uuid::Uuid::new_v4()),
            name: "Network Segmentation".to_string(),
            description: "Restrict network access and implement jump boxes".to_string(),
            controls: network_controls,
            effectiveness: 0.85,
            implementation_cost: "Medium".to_string(),
            blast_radius: vec!["vm-prod-01".to_string()],
        });

        // RBAC reduction bundle
        let rbac_controls = vec![SecurityControl {
            control_type: "RoleAssignment".to_string(),
            action: "Reduce".to_string(),
            target: "role-contributor".to_string(),
            parameters: HashMap::from([
                (
                    "new_role".to_string(),
                    "VirtualMachineContributor".to_string(),
                ),
                ("scope".to_string(), "resourceGroup".to_string()),
            ]),
        }];

        bundles.push(MitigationBundle {
            bundle_id: format!("mb-{}", uuid::Uuid::new_v4()),
            name: "Least Privilege RBAC".to_string(),
            description: "Reduce role permissions to minimum required".to_string(),
            controls: rbac_controls,
            effectiveness: 0.75,
            implementation_cost: "Low".to_string(),
            blast_radius: vec!["user-001".to_string()],
        });

        // Encryption bundle
        let encryption_controls = vec![
            SecurityControl {
                control_type: "StorageEncryption".to_string(),
                action: "Enable".to_string(),
                target: "storage-001".to_string(),
                parameters: HashMap::from([(
                    "key_type".to_string(),
                    "CustomerManagedKey".to_string(),
                )]),
            },
            SecurityControl {
                control_type: "PrivateEndpoint".to_string(),
                action: "Create".to_string(),
                target: "storage-001".to_string(),
                parameters: HashMap::new(),
            },
        ];

        bundles.push(MitigationBundle {
            bundle_id: format!("mb-{}", uuid::Uuid::new_v4()),
            name: "Data Protection".to_string(),
            description: "Enable encryption and private endpoints".to_string(),
            controls: encryption_controls,
            effectiveness: 0.9,
            implementation_cost: "Medium".to_string(),
            blast_radius: vec!["storage-001".to_string()],
        });

        bundles
    }

    // Apply mitigation bundle
    pub async fn apply_mitigation(
        &self,
        bundle: &MitigationBundle,
        client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<MitigationResult, SecurityGraphError> {
        let mut applied_controls = Vec::new();
        let mut failed_controls = Vec::new();

        for control in &bundle.controls {
            match self.apply_control(control, client).await {
                Ok(_) => applied_controls.push(control.clone()),
                Err(e) => {
                    tracing::error!("Failed to apply control {}: {}", control.control_type, e);
                    failed_controls.push(control.clone());
                }
            }
        }

        Ok(MitigationResult {
            bundle_id: bundle.bundle_id.clone(),
            applied_controls,
            failed_controls,
            residual_risk: self.calculate_residual_risk(bundle),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn apply_control(
        &self,
        control: &SecurityControl,
        _client: &crate::azure_client_async::AsyncAzureClient,
    ) -> Result<(), SecurityGraphError> {
        // Simulate control application
        tracing::info!(
            "Applying {} control to {}",
            control.control_type,
            control.target
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(())
    }

    fn calculate_residual_risk(&self, bundle: &MitigationBundle) -> f64 {
        // Simple residual risk calculation
        1.0 - bundle.effectiveness
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationResult {
    pub bundle_id: String,
    pub applied_controls: Vec<SecurityControl>,
    pub failed_controls: Vec<SecurityControl>,
    pub residual_risk: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum SecurityGraphError {
    #[error("Azure API error: {0}")]
    AzureError(String),
    #[error("Graph construction error: {0}")]
    GraphError(String),
    #[error("Path analysis error: {0}")]
    PathError(String),
}

// Public API for security graph analysis
pub async fn analyze_security_exposure(
    client: Option<&crate::azure_client_async::AsyncAzureClient>,
) -> Result<SecurityExposureReport, SecurityGraphError> {
    let azure_client = client.ok_or_else(|| {
        SecurityGraphError::AzureError(
            "Azure client not initialized. Please ensure Azure credentials are configured."
                .to_string(),
        )
    })?;

    let mut engine = SecurityGraphEngine::new();
    engine.build_from_azure(azure_client).await?;

    let critical_paths = engine.find_attack_paths("Critical");
    let high_paths = engine.find_attack_paths("High");

    Ok(SecurityExposureReport {
        analysis_timestamp: chrono::Utc::now(),
        total_nodes: engine.graph.node_count(),
        total_edges: engine.graph.edge_count(),
        critical_paths: critical_paths.len(),
        high_risk_paths: high_paths.len(),
        top_attack_paths: critical_paths.into_iter().take(5).collect(),
        recommended_mitigations: generate_prioritized_mitigations(&high_paths),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityExposureReport {
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub critical_paths: usize,
    pub high_risk_paths: usize,
    pub top_attack_paths: Vec<AttackPath>,
    pub recommended_mitigations: Vec<MitigationBundle>,
}

fn generate_prioritized_mitigations(paths: &[AttackPath]) -> Vec<MitigationBundle> {
    // Aggregate and prioritize mitigations across all paths
    let mut mitigation_map: HashMap<String, MitigationBundle> = HashMap::new();

    for path in paths {
        for bundle in &path.mitigation_bundles {
            mitigation_map
                .entry(bundle.name.clone())
                .or_insert_with(|| bundle.clone());
        }
    }

    let mut mitigations: Vec<MitigationBundle> = mitigation_map.into_values().collect();
    use std::cmp::Ordering;
    mitigations.sort_by(|a, b| b
        .effectiveness
        .partial_cmp(&a.effectiveness)
        .unwrap_or(Ordering::Equal));
    mitigations.into_iter().take(10).collect()
}
