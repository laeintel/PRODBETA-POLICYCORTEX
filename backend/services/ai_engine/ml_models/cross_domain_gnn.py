"""
Cross-Domain Graph Neural Network for Governance Correlation
Patent 1: Cross-Domain Governance Correlation Engine

This module implements the patented GNN system for identifying complex relationships
and dependencies across different governance domains in PolicyCortex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class CorrelationConfig:
    """Configuration for the Cross-Domain GNN system"""

    # Graph neural network configuration
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_gnn_layers: int = 4
    dropout_rate: float = 0.1

    # Correlation detection parameters
    correlation_threshold: float = 0.7
    temporal_window: int = 24  # hours
    max_correlation_depth: int = 3

    # Performance optimization
    batch_size: int = 32
    max_nodes_per_batch: int = 10000
    use_gpu: bool = True

    # Learning parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Model persistence
    model_save_path: str = "/app/models/cross_domain_gnn"
    checkpoint_frequency: int = 100

class GovernanceGraphBuilder:
    """Build heterogeneous governance graphs from cloud resources and policies"""

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.node_type_mapping = {
            'resource': 0, 'policy': 1, 'domain': 2, 'event': 3, 'user': 4
        }
        self.edge_type_mapping = {
            'dependency': 0, 'policy_application': 1, 'correlation': 2,
            'impact': 3, 'temporal': 4
        }
        logger.info("Initialized GovernanceGraphBuilder")

    def build_graph(self, governance_data: Dict[str, Any]) -> HeteroData:
        """Build heterogeneous graph from governance data"""

        graph = HeteroData()

        try:
            # Add nodes for each type
            self._add_resource_nodes(graph, governance_data.get('resources', []))
            self._add_policy_nodes(graph, governance_data.get('policies', []))
            self._add_domain_nodes(graph, governance_data.get('domains', []))
            self._add_event_nodes(graph, governance_data.get('events', []))
            self._add_user_nodes(graph, governance_data.get('users', []))

            # Add edges for each relationship type
            self._add_dependency_edges(graph, governance_data.get('dependencies', []))
            self._add_policy_edges(graph, governance_data.get('policy_applications', []))
            self._add_correlation_edges(graph, governance_data.get('correlations', []))
            self._add_impact_edges(graph, governance_data.get('impacts', []))
            self._add_temporal_edges(graph, governance_data.get('temporal_relationships', []))

            logger.info(f"Built governance graph with {len(graph.node_types)} node types and
                {len(graph.edge_types)} edge types")
            return graph

        except Exception as e:
            logger.error(f"Error building governance graph: {e}")
            raise

    def _add_resource_nodes(self, graph: HeteroData, resources: List[Dict]):
        """Add resource nodes with features"""

        if not resources:
            logger.debug("No resources to add to graph")
            return

        # Extract features for each resource
        node_features = []
        node_ids = []

        for resource in resources:
            try:
                features = self._extract_resource_features(resource)
                node_features.append(features)
                node_ids.append(resource.get('id', f"resource_{len(node_ids)}"))
            except Exception as e:
                logger.warning(
                    f"Error extracting features for resource {resource.get('id',
                    'unknown')}: {e}"
                )
                continue

        if node_features:
            graph['resource'].x = torch.FloatTensor(node_features)
            graph['resource'].node_id = node_ids
            logger.debug(f"Added {len(node_features)} resource nodes")

    def _extract_resource_features(self, resource: Dict) -> List[float]:
        """Extract numerical features from resource data"""

        features = []

        # Resource type encoding (one-hot or embedding)
        resource_type = resource.get('type', 'unknown')
        type_encoding = self._encode_resource_type(resource_type)
        features.extend(type_encoding)

        # Configuration features (normalized)
        features.append(min(resource.get('cpu_cores', 0) / 100.0, 1.0))
        features.append(min(resource.get('memory_gb', 0) / 1000.0, 1.0))
        features.append(min(resource.get('storage_gb', 0) / 10000.0, 1.0))

        # Cost features
        features.append(min(resource.get('monthly_cost', 0) / 10000.0, 1.0))
        features.append(max(-1.0, min(resource.get('cost_trend', 0), 1.0)))

        # Security features
        features.append(max(0.0, min(resource.get('security_score', 0.5), 1.0)))
        features.append(max(0.0, min(resource.get('compliance_score', 0.5), 1.0)))
        features.append(min(resource.get('vulnerability_count', 0) / 100.0, 1.0))

        # Performance features
        features.append(max(0.0, min(resource.get('cpu_utilization', 0.5), 1.0)))
        features.append(max(0.0, min(resource.get('memory_utilization', 0.5), 1.0)))
        features.append(max(0.0, min(resource.get('network_utilization', 0.5), 1.0)))

        # Temporal features
        features.append(min(resource.get('age_days', 0) / 365.0, 10.0))  # Cap at 10 years
        features.append(min(resource.get('last_modified_hours', 0) / 168.0, 1.0))  # Cap at 1 week

        return features

    def _encode_resource_type(self, resource_type: str) -> List[float]:
        """Encode resource type as features"""

        # Common Azure resource types
        type_mapping = {
            'virtual_machine': [1, 0, 0, 0, 0, 0, 0, 0],
            'storage_account': [0, 1, 0, 0, 0, 0, 0, 0],
            'sql_database': [0, 0, 1, 0, 0, 0, 0, 0],
            'app_service': [0, 0, 0, 1, 0, 0, 0, 0],
            'function_app': [0, 0, 0, 0, 1, 0, 0, 0],
            'container_instance': [0, 0, 0, 0, 0, 1, 0, 0],
            'kubernetes_service': [0, 0, 0, 0, 0, 0, 1, 0],
            'unknown': [0, 0, 0, 0, 0, 0, 0, 1]
        }

        return type_mapping.get(resource_type.lower(), type_mapping['unknown'])

    def _add_policy_nodes(self, graph: HeteroData, policies: List[Dict]):
        """Add policy nodes with features"""

        if not policies:
            return

        node_features = []
        node_ids = []

        for policy in policies:
            features = self._extract_policy_features(policy)
            node_features.append(features)
            node_ids.append(policy.get('id', f"policy_{len(node_ids)}"))

        if node_features:
            graph['policy'].x = torch.FloatTensor(node_features)
            graph['policy'].node_id = node_ids
            logger.debug(f"Added {len(node_features)} policy nodes")

    def _extract_policy_features(self, policy: Dict) -> List[float]:
        """Extract features from policy data"""

        features = []

        # Policy type encoding
        policy_type = policy.get('type', 'unknown')
        type_encoding = self._encode_policy_type(policy_type)
        features.extend(type_encoding)

        # Policy attributes
        features.append(1.0 if policy.get('active', False) else 0.0)
        features.append(policy.get('priority', 0.5))  # 0-1 range
        features.append(policy.get('severity', 0.5))  # 0-1 range
        features.append(len(policy.get('rules', [])) / 10.0)  # Normalized rule count

        # Domain coverage
        domains = policy.get('domains', [])
        domain_coverage = [
            1.0 if 'security' in domains else 0.0,
            1.0 if 'compliance' in domains else 0.0,
            1.0 if 'cost' in domains else 0.0,
            1.0 if 'performance' in domains else 0.0
        ]
        features.extend(domain_coverage)

        return features

    def _encode_policy_type(self, policy_type: str) -> List[float]:
        """Encode policy type as features"""

        type_mapping = {
            'security': [1, 0, 0, 0],
            'compliance': [0, 1, 0, 0],
            'cost': [0, 0, 1, 0],
            'performance': [0, 0, 0, 1],
            'unknown': [0, 0, 0, 0]
        }

        return type_mapping.get(policy_type.lower(), type_mapping['unknown'])

    def _add_domain_nodes(self, graph: HeteroData, domains: List[Dict]):
        """Add governance domain nodes"""

        if not domains:
            # Create default domain nodes
            domains = [
                {'id': 'security', 'name': 'Security', 'weight': 1.0},
                {'id': 'compliance', 'name': 'Compliance', 'weight': 1.0},
                {'id': 'cost', 'name': 'Cost Management', 'weight': 1.0},
                {'id': 'performance', 'name': 'Performance', 'weight': 1.0}
            ]

        node_features = []
        node_ids = []

        for domain in domains:
            features = [
                domain.get('weight', 1.0),
                len(domain.get('policies', [])) / 100.0,  # Normalized policy count
                domain.get('compliance_score', 0.5),
                domain.get('effectiveness_score', 0.5)
            ]
            node_features.append(features)
            node_ids.append(domain.get('id', f"domain_{len(node_ids)}"))

        if node_features:
            graph['domain'].x = torch.FloatTensor(node_features)
            graph['domain'].node_id = node_ids
            logger.debug(f"Added {len(node_features)} domain nodes")

    def _add_event_nodes(self, graph: HeteroData, events: List[Dict]):
        """Add governance event nodes"""

        if not events:
            return

        node_features = []
        node_ids = []

        for event in events:
            features = self._extract_event_features(event)
            node_features.append(features)
            node_ids.append(event.get('id', f"event_{len(node_ids)}"))

        if node_features:
            graph['event'].x = torch.FloatTensor(node_features)
            graph['event'].node_id = node_ids
            logger.debug(f"Added {len(node_features)} event nodes")

    def _extract_event_features(self, event: Dict) -> List[float]:
        """Extract features from event data"""

        features = []

        # Event type encoding
        event_type = event.get('type', 'unknown')
        type_encoding = self._encode_event_type(event_type)
        features.extend(type_encoding)

        # Event attributes
        features.append(event.get('severity', 0.5))
        features.append(1.0 if event.get('resolved', False) else 0.0)
        features.append(event.get('impact_score', 0.5))

        # Temporal features
        timestamp = event.get('timestamp', datetime.now().timestamp())
        age_hours = (datetime.now().timestamp() - timestamp) / 3600.0
        features.append(min(age_hours / 168.0, 1.0))  # Normalized to weeks

        return features

    def _encode_event_type(self, event_type: str) -> List[float]:
        """Encode event type as features"""

        type_mapping = {
            'violation': [1, 0, 0, 0, 0],
            'alert': [0, 1, 0, 0, 0],
            'change': [0, 0, 1, 0, 0],
            'audit': [0, 0, 0, 1, 0],
            'unknown': [0, 0, 0, 0, 1]
        }

        return type_mapping.get(event_type.lower(), type_mapping['unknown'])

    def _add_user_nodes(self, graph: HeteroData, users: List[Dict]):
        """Add user nodes with features"""

        if not users:
            return

        node_features = []
        node_ids = []

        for user in users:
            features = self._extract_user_features(user)
            node_features.append(features)
            node_ids.append(user.get('id', f"user_{len(node_ids)}"))

        if node_features:
            graph['user'].x = torch.FloatTensor(node_features)
            graph['user'].node_id = node_ids
            logger.debug(f"Added {len(node_features)} user nodes")

    def _extract_user_features(self, user: Dict) -> List[float]:
        """Extract features from user data"""

        features = []

        # Role encoding
        role = user.get('role', 'user')
        role_encoding = self._encode_user_role(role)
        features.extend(role_encoding)

        # User attributes
        features.append(len(user.get('permissions', [])) / 50.0)  # Normalized permission count
        features.append(user.get('risk_score', 0.5))
        features.append(user.get('activity_score', 0.5))
        features.append(1.0 if user.get('active', True) else 0.0)

        return features

    def _encode_user_role(self, role: str) -> List[float]:
        """Encode user role as features"""

        role_mapping = {
            'admin': [1, 0, 0, 0],
            'manager': [0, 1, 0, 0],
            'developer': [0, 0, 1, 0],
            'user': [0, 0, 0, 1]
        }

        return role_mapping.get(role.lower(), role_mapping['user'])

    def _add_dependency_edges(self, graph: HeteroData, dependencies: List[Dict]):
        """Add dependency edges between resources"""

        if not dependencies:
            return

        source_indices = []
        target_indices = []
        edge_attributes = []

        for dep in dependencies:
            source_id = dep.get('source_id')
            target_id = dep.get('target_id')

            # Find node indices (simplified - would need proper ID mapping)
            source_idx = hash(source_id) % 1000  # Mock index
            target_idx = hash(target_id) % 1000  # Mock index

            source_indices.append(source_idx)
            target_indices.append(target_idx)

            # Edge attributes
            attributes = [
                dep.get('strength', 0.5),
                dep.get('criticality', 0.5),
                1.0 if dep.get('bidirectional', False) else 0.0
            ]
            edge_attributes.append(attributes)

        if source_indices:
            edge_index = torch.LongTensor([source_indices, target_indices])
            graph['resource', 'dependency', 'resource'].edge_index = edge_index
            graph['resource', 'dependency', 'resource'].edge_attr = torch.FloatTensor(edge_attributes)
            logger.debug(f"Added {len(source_indices)} dependency edges")

    def _add_policy_edges(self, graph: HeteroData, policy_applications: List[Dict]):
        """Add policy application edges"""

        if not policy_applications:
            return

        # Implementation for policy application edges
        # Similar structure to dependency edges
        pass

    def _add_correlation_edges(self, graph: HeteroData, correlations: List[Dict]):
        """Add discovered correlation edges"""

        if not correlations:
            return

        # Implementation for correlation edges
        pass

    def _add_impact_edges(self, graph: HeteroData, impacts: List[Dict]):
        """Add impact relationship edges"""

        if not impacts:
            return

        # Implementation for impact edges
        pass

    def _add_temporal_edges(self, graph: HeteroData, temporal_relationships: List[Dict]):
        """Add temporal relationship edges"""

        if not temporal_relationships:
            return

        # Implementation for temporal edges
        pass

class CrossDomainGNN(nn.Module):
    """Graph Neural Network for cross-domain governance correlation"""

    def __init__(self, config: CorrelationConfig, node_feature_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.node_feature_dims = node_feature_dims

        logger.info(f"Initializing CrossDomainGNN with feature dims: {node_feature_dims}")

        # Node embedding layers for each type
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Linear(feature_dim, config.hidden_dim)
            for node_type, feature_dim in node_feature_dims.items()
        })

        # Heterogeneous graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(config.num_gnn_layers):
            conv_dict = {}

            # Define convolutions for each edge type
            edge_types = [
                ('resource', 'dependency', 'resource'),
                ('resource', 'policy_application', 'policy'),
                ('policy', 'policy_application', 'resource'),
                ('resource', 'correlation', 'domain'),
                ('domain', 'correlation', 'resource'),
                ('resource', 'impact', 'resource'),
                ('event', 'temporal', 'event')
            ]

            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    config.hidden_dim,
                    config.hidden_dim // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    dropout=config.dropout_rate,
                    concat=True if i < config.num_gnn_layers - 1 else False
                )

            hetero_conv = HeteroConv(conv_dict, aggr='mean')
            self.conv_layers.append(hetero_conv)

        # Correlation detection heads
        self.correlation_detector = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Impact prediction heads
        self.impact_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 5),  # 5 impact categories
            nn.Softmax(dim=-1)
        )

        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 4),  # 4 governance domains
            nn.Softmax(dim=-1)
        )

        # Initialize weights
        self.apply(self._init_weights)
        logger.info("CrossDomainGNN initialized successfully")

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, graph: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN"""

        try:
            # Embed node features
            x_dict = {}
            for node_type, embedding_layer in self.node_embeddings.items():
                if node_type in graph.x_dict and graph.x_dict[node_type] is not None:
                    x_dict[node_type] = embedding_layer(graph.x_dict[node_type])

            if not x_dict:
                logger.warning("No node features found in graph")
                return {'embeddings': {}}

            # Apply graph convolution layers
            for i, conv_layer in enumerate(self.conv_layers):
                try:
                    x_dict = conv_layer(x_dict, graph.edge_index_dict)

                    # Apply activation and dropout
                    for node_type in x_dict:
                        x_dict[node_type] = F.relu(x_dict[node_type])
                        x_dict[node_type] = F.dropout(
                            x_dict[node_type],
                            p=self.config.dropout_rate,
                            training=self.training
                        )
                except Exception as e:
                    logger.warning(f"Error in GNN layer {i}: {e}")
                    break

            # Generate predictions
            outputs = {}

            # Correlation detection for resource pairs
            if 'resource' in x_dict and x_dict['resource'].size(0) > 1:
                try:
                    correlations = self._detect_correlations(x_dict['resource'])
                    outputs['correlations'] = correlations
                except Exception as e:
                    logger.warning(f"Error in correlation detection: {e}")

            # Impact prediction
            if 'resource' in x_dict:
                try:
                    impacts = self._predict_impacts(x_dict)
                    outputs['impacts'] = impacts
                except Exception as e:
                    logger.warning(f"Error in impact prediction: {e}")

            # Domain classification
            if 'resource' in x_dict:
                try:
                    domain_predictions = self.domain_classifier(x_dict['resource'])
                    outputs['domain_predictions'] = domain_predictions
                except Exception as e:
                    logger.warning(f"Error in domain classification: {e}")

            # Return embeddings for downstream tasks
            outputs['embeddings'] = x_dict

            return outputs

        except Exception as e:
            logger.error(f"Error in GNN forward pass: {e}")
            return {'embeddings': {}}

    def _detect_correlations(self, resource_embeddings: torch.Tensor) -> torch.Tensor:
        """Detect correlations between resource pairs"""

        num_resources = resource_embeddings.size(0)
        if num_resources < 2:
            return torch.empty(0, 1)

        # Limit pairs to prevent memory issues
        max_pairs = min(100, num_resources * (num_resources - 1) // 2)
        correlations = []

        pairs_processed = 0
        for i in range(num_resources):
            if pairs_processed >= max_pairs:
                break
            for j in range(i + 1, num_resources):
                if pairs_processed >= max_pairs:
                    break

                pair_embedding = torch.cat([
                    resource_embeddings[i],
                    resource_embeddings[j]
                ], dim=0)

                correlation_score = self.correlation_detector(pair_embedding.unsqueeze(0))
                correlations.append(correlation_score)
                pairs_processed += 1

        return torch.cat(correlations) if correlations else torch.empty(0, 1)

    def _predict_impacts(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict cross-domain impacts"""

        if 'resource' not in x_dict:
            return torch.empty(0, 5)

        resource_embeddings = x_dict['resource']
        num_resources = resource_embeddings.size(0)

        if num_resources < 2:
            return torch.empty(0, 5)

        # Limit impact predictions to prevent memory issues
        max_predictions = min(50, num_resources * (num_resources - 1))
        impacts = []

        predictions_made = 0
        for i in range(num_resources):
            if predictions_made >= max_predictions:
                break
            for j in range(num_resources):
                if i != j and predictions_made < max_predictions:
                    impact_input = torch.cat([
                        resource_embeddings[i],
                        resource_embeddings[j]
                    ], dim=0)

                    impact_prediction = self.impact_predictor(impact_input.unsqueeze(0))
                    impacts.append(impact_prediction)
                    predictions_made += 1

        return torch.cat(impacts) if impacts else torch.empty(0, 5)

class CorrelationEngine:
    """Main engine for cross-domain governance correlation"""

    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.graph_builder = GovernanceGraphBuilder(config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if config.use_gpu and
            torch.cuda.is_available() else 'cpu')

        logger.info(f"Initialized CorrelationEngine on device: {self.device}")

    def initialize_model(self, node_feature_dims: Dict[str, int]):
        """Initialize the GNN model"""

        try:
            self.model = CrossDomainGNN(self.config, node_feature_dims)
            self.model.to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,
                eta_min=self.config.learning_rate * 0.01
            )

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def detect_correlations(self, governance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect cross-domain correlations in governance data"""

        try:
            # Build graph from governance data
            graph = self.graph_builder.build_graph(governance_data)
            graph = graph.to(self.device)

            if self.model is None:
                # Initialize model with default feature dimensions
                node_feature_dims = self._infer_feature_dimensions(graph)
                self.initialize_model(node_feature_dims)

            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(graph)

            # Process and return results
            results = {
                'correlations': self._process_correlations(outputs.get('correlations')),
                'impacts': self._process_impacts(outputs.get('impacts')),
                'domain_predictions': self._process_domain_predictions(outputs.get('domain_predictions')),
                'graph_stats': self._get_graph_statistics(graph),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Correlation detection completed. Found {len(results['correlations'])} correlations")
            return results

        except Exception as e:
            logger.error(f"Error in correlation detection: {e}")
            return {
                'error': str(e),
                'correlations': [],
                'impacts': [],
                'domain_predictions': [],
                'timestamp': datetime.now().isoformat()
            }

    def _infer_feature_dimensions(self, graph: HeteroData) -> Dict[str, int]:
        """Infer feature dimensions from graph"""

        feature_dims = {}
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                feature_dims[node_type] = graph[node_type].x.size(1)
            else:
                # Default feature dimensions
                default_dims = {
                    'resource': 20,
                    'policy': 12,
                    'domain': 4,
                    'event': 9,
                    'user': 8
                }
                feature_dims[node_type] = default_dims.get(node_type, 10)

        logger.debug(f"Inferred feature dimensions: {feature_dims}")
        return feature_dims

    def _get_graph_statistics(self, graph: HeteroData) -> Dict[str, Any]:
        """Get statistics about the graph"""

        stats = {
            'node_types': len(graph.node_types),
            'edge_types': len(graph.edge_types),
            'total_nodes': sum(graph[node_type].x.size(0) if hasattr(graph[node_type], 'x') and
                graph[node_type].x is not None else 0 for node_type in graph.node_types),
            'total_edges': sum(
                graph[edge_type].edge_index.size(1) if hasattr(graph[edge_type],
                'edge_index') else 0 for edge_type in graph.edge_types
            )
        }

        return stats

    def _process_correlations(self, correlations: Optional[torch.Tensor]) -> List[Dict]:
        """Process correlation predictions into structured results"""

        if correlations is None or correlations.numel() == 0:
            return []

        correlation_results = []
        correlations_np = correlations.cpu().numpy()

        for i, correlation_score in enumerate(correlations_np.flatten()):
            if correlation_score > self.config.correlation_threshold:
                correlation_results.append({
                    'pair_index': i,
                    'correlation_score': float(correlation_score),
                    'confidence': 'high' if correlation_score > 0.8 else 'medium',
                    'correlation_type': 'resource_correlation'
                })

        return correlation_results

    def _process_impacts(self, impacts: Optional[torch.Tensor]) -> List[Dict]:
        """Process impact predictions into structured results"""

        if impacts is None or impacts.numel() == 0:
            return []

        impact_results = []
        impacts_np = impacts.cpu().numpy()

        impact_categories = ['security', 'compliance', 'cost', 'performance', 'operations']

        for i, impact_probs in enumerate(impacts_np):
            max_impact_idx = np.argmax(impact_probs)
            max_impact_prob = impact_probs[max_impact_idx]

            if max_impact_prob > 0.5:  # Threshold for significant impact
                impact_results.append({
                    'pair_index': i,
                    'impact_category': impact_categories[max_impact_idx],
                    'impact_probability': float(max_impact_prob),
                    'all_probabilities': {
                        category: float(prob)
                        for category, prob in zip(impact_categories, impact_probs)
                    }
                })

        return impact_results

    def _process_domain_predictions(self, domain_predictions: Optional[torch.Tensor]) -> List[Dict]:
        """Process domain classification predictions"""

        if domain_predictions is None or domain_predictions.numel() == 0:
            return []

        domain_results = []
        predictions_np = domain_predictions.cpu().numpy()

        domain_names = ['security', 'compliance', 'cost', 'performance']

        for i, domain_probs in enumerate(predictions_np):
            primary_domain_idx = np.argmax(domain_probs)
            primary_domain_prob = domain_probs[primary_domain_idx]

            domain_results.append({
                'resource_index': i,
                'primary_domain': domain_names[primary_domain_idx],
                'confidence': float(primary_domain_prob),
                'all_domains': {
                    domain: float(prob)
                    for domain, prob in zip(domain_names, domain_probs)
                }
            })

        return domain_results

    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""

        if self.model is None:
            logger.warning("No model to save")
            return

        save_path = path or self.config.model_save_path

        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(checkpoint, f"{save_path}.pth")
            logger.info(f"Model saved to {save_path}.pth")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: Optional[str] = None):
        """Load a trained model"""

        load_path = path or f"{self.config.model_save_path}.pth"

        try:
            checkpoint = torch.load(load_path, map_location=self.device)

            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])

                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                logger.info(f"Model loaded from {load_path}")
            else:
                logger.warning("No model initialized to load state into")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
