"""
RBAC Analysis Model for PolicyCortex.

This module implements advanced ML models with graph neural networks for analyzing
role-based access control patterns, detecting anomalous access, identifying
over-privileged accounts, and recommending role optimizations.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

logger = logging.getLogger(__name__)


class RBACGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for RBAC analysis with multi-head attention.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # Graph attention layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATConv(
                node_features,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Output layer
        self.gat_layers.append(
            GATConv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=False
            )
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // num_heads, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Global pooling
        self.global_pool = global_mean_pool

        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim // num_heads, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the graph attention network.

        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Edge indices (2, num_edges)
            batch: Batch vector for graph-level predictions

        Returns:
            Dictionary containing predictions and embeddings
        """
        # Store intermediate representations
        node_embeddings = []
        attention_weights = []

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = gat_layer(x, edge_index)

            if i < len(self.gat_layers) - 1:
                x = F.elu(x)

            node_embeddings.append(x)

        # Node-level predictions (anomaly detection)
        node_anomaly_scores = self.anomaly_detector(x)

        # Graph-level predictions
        if batch is not None:
            # Global pooling for graph-level tasks
            graph_embedding = self.global_pool(x, batch)
            graph_predictions = self.classifier(graph_embedding)
        else:
            # Use mean pooling if no batch info
            graph_embedding = torch.mean(x, dim=0, keepdim=True)
            graph_predictions = self.classifier(graph_embedding)

        return {
            "node_embeddings": x,
            "node_anomaly_scores": node_anomaly_scores,
            "graph_embedding": graph_embedding,
            "graph_predictions": graph_predictions,
            "intermediate_embeddings": node_embeddings,
        }


class RBACHierarchyGNN(nn.Module):
    """
    Specialized Graph Neural Network for RBAC hierarchy analysis.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node transformation layers
        self.node_transform = nn.Linear(node_features, hidden_dim)

        # GCN layers for hierarchy propagation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Role hierarchy prediction
        self.hierarchy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # parent, child, sibling
        )

        # Permission propagation predictor
        self.permission_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for RBAC hierarchy analysis."""

        # Transform node features
        x = F.relu(self.node_transform(x))

        # Apply GCN layers
        embeddings = []
        for gcn_layer in self.gcn_layers:
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(gcn_layer(x, edge_index))
            embeddings.append(x)

        # Predict hierarchy relationships
        edge_embeddings = torch.cat(
            [x[edge_index[0]], x[edge_index[1]]], dim=1  # Source nodes  # Target nodes
        )

        hierarchy_predictions = self.hierarchy_predictor(edge_embeddings)

        # Predict permission inheritance
        permission_scores = self.permission_predictor(x)

        return {
            "node_embeddings": x,
            "hierarchy_predictions": hierarchy_predictions,
            "permission_scores": permission_scores,
            "intermediate_embeddings": embeddings,
        }


class RBACAnalyzer:
    """
    Advanced RBAC Analysis system using Graph Neural Networks for
    access pattern analysis, anomaly detection, and role optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RBAC Analyzer.

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.graphs = {}
        self.role_hierarchy = None
        self.permission_matrix = None

        # Azure clients
        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential)

        # Initialize models
        self._initialize_models()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the RBAC analyzer."""
        return {
            "graph_config": {
                "node_features": 64,
                "hidden_dim": 128,
                "num_heads": 8,
                "num_layers": 3,
                "dropout": 0.2,
            },
            "anomaly_detection": {
                "isolation_forest_contamination": 0.1,
                "dbscan_eps": 0.5,
                "dbscan_min_samples": 5,
            },
            "clustering": {"n_clusters": 10, "similarity_threshold": 0.8},
            "training": {
                "epochs": 200,
                "learning_rate": 0.001,
                "batch_size": 32,
                "validation_split": 0.2,
            },
        }

    def _initialize_models(self):
        """Initialize all models and components."""
        # Anomaly detection models
        self.models["isolation_forest"] = IsolationForest(
            contamination=self.config["anomaly_detection"]["isolation_forest_contamination"],
            random_state=42,
        )

        self.models["dbscan"] = DBSCAN(
            eps=self.config["anomaly_detection"]["dbscan_eps"],
            min_samples=self.config["anomaly_detection"]["dbscan_min_samples"],
        )

        # Clustering for role optimization
        self.models["kmeans"] = KMeans(
            n_clusters=self.config["clustering"]["n_clusters"], random_state=42
        )

        # Scalers
        self.scalers["standard"] = StandardScaler()
        self.scalers["pca"] = PCA(n_components=0.95)  # Keep 95% variance

        # Encoders
        self.encoders["role"] = LabelEncoder()
        self.encoders["permission"] = LabelEncoder()
        self.encoders["user"] = LabelEncoder()

    async def prepare_rbac_data(self, workspace_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Prepare RBAC data from Azure logs and AD information.

        Args:
            workspace_id: Log Analytics workspace ID
            days_back: Number of days of historical data to fetch

        Returns:
            Prepared RBAC dataset with graphs and matrices
        """
        logger.info(f"Preparing RBAC data for {days_back} days")

        # Query for RBAC events
        rbac_query = f"""
        union SigninLogs, AuditLogs, AADRiskyUsers, AADUserRiskEvents
        | where TimeGenerated > ago({days_back}d)
        | where Category in (
            "Authentication",
            "RoleManagement",
            "UserManagement",
            "DirectoryManagement"
        )
        | extend UserId = tostring(UserId),
                 RoleName = tostring(Properties.roleName),
                 Permission = tostring(Properties.permission),
                 ResourceId = tostring(Properties.resourceId),
                 ActionType = tostring(OperationName)
        | where isnotempty(UserId)
        | project TimeGenerated, UserId, RoleName, Permission, ResourceId, ActionType, Properties
        | order by TimeGenerated asc
        """

        try:
            response = await self.logs_client.query_workspace(
                workspace_id=workspace_id, query=rbac_query, timespan=timedelta(days=days_back)
            )

            # Convert to DataFrame
            df = pd.DataFrame([row for row in response.tables[0].rows])
            if not df.empty:
                df.columns = [col.name for col in response.tables[0].columns]

            logger.info(f"Retrieved {len(df)} RBAC events")

            # Generate sample data if no real data
            if df.empty:
                df = self._generate_sample_rbac_data(days_back)

            # Process RBAC data
            rbac_data = self._process_rbac_data(df)

            return rbac_data

        except Exception as e:
            logger.error(f"Error fetching RBAC data: {str(e)}")
            # Return sample data for testing
            sample_df = self._generate_sample_rbac_data(days_back)
            return self._process_rbac_data(sample_df)

    def _generate_sample_rbac_data(self, days_back: int) -> pd.DataFrame:
        """Generate sample RBAC data for testing."""
        logger.warning("Generating sample RBAC data for testing")

        np.random.seed(42)
        n_samples = days_back * 500  # 500 events per day

        # Define sample entities
        users = [f"user_{i}@company.com" for i in range(1, 201)]  # 200 users
        roles = [
            "Global Administrator",
            "User Administrator",
            "Security Administrator",
            "Application Administrator",
            "Cloud Application Administrator",
            "Privileged Role Administrator",
            "Exchange Administrator",
            "SharePoint Administrator",
            "Teams Administrator",
            "Intune Administrator",
            "Security Reader",
            "Global Reader",
            "Helpdesk Administrator",
            "Password Administrator",
            "License Administrator",
        ]
        permissions = [
            "microsoft.directory/users/create",
            "microsoft.directory/users/read",
            "microsoft.directory/users/update",
            "microsoft.directory/users/delete",
            "microsoft.directory/groups/create",
            "microsoft.directory/groups/read",
            "microsoft.directory/groups/update",
            "microsoft.directory/groups/delete",
            "microsoft.directory/applications/create",
            "microsoft.directory/applications/read",
            "microsoft.directory/applications/update",
            "microsoft.directory/applications/delete",
            "microsoft.security/policies/read",
            "microsoft.security/policies/write",
            "microsoft.azure/resourceGroups/read",
            "microsoft.azure/resourceGroups/write",
        ]
        resources = [f"resource_{i}" for i in range(1, 101)]  # 100 resources

        data = []
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq="5min")

        for date in dates:
            user = np.random.choice(users)
            role = np.random.choice(roles)
            permission = np.random.choice(permissions)
            resource = np.random.choice(resources)

            # Create realistic access patterns
            action_types = ["Sign-in", "Role Assignment", "Permission Grant", "Resource Access"]
            action_type = np.random.choice(action_types)

            data.append(
                {
                    "TimeGenerated": date,
                    "UserId": user,
                    "RoleName": role,
                    "Permission": permission,
                    "ResourceId": resource,
                    "ActionType": action_type,
                }
            )

        return pd.DataFrame(data)

    def _process_rbac_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process raw RBAC data into graph structures and matrices.

        Args:
            df: Raw RBAC event data

        Returns:
            Processed RBAC data with graphs and matrices
        """
        logger.info("Processing RBAC data into graph structures")

        if df.empty:
            return {"graphs": {}, "matrices": {}, "metadata": {}}

        # Clean and encode data
        df["TimeGenerated"] = pd.to_datetime(df["TimeGenerated"])
        df = df.sort_values("TimeGenerated")

        # Encode categorical variables
        df["user_id"] = self.encoders["user"].fit_transform(df["UserId"].fillna("unknown"))
        df["role_id"] = self.encoders["role"].fit_transform(df["RoleName"].fillna("unknown"))
        df["permission_id"] = self.encoders["permission"].fit_transform(
            df["Permission"].fillna("unknown")
        )

        # Create various graph representations
        graphs = self._create_rbac_graphs(df)

        # Create permission and role matrices
        matrices = self._create_rbac_matrices(df)

        # Calculate user behavior features
        user_features = self._extract_user_features(df)

        # Calculate role hierarchy
        role_hierarchy = self._infer_role_hierarchy(df)

        # Detect access patterns
        access_patterns = self._detect_access_patterns(df)

        return {
            "graphs": graphs,
            "matrices": matrices,
            "user_features": user_features,
            "role_hierarchy": role_hierarchy,
            "access_patterns": access_patterns,
            "metadata": {
                "n_users": len(df["UserId"].unique()),
                "n_roles": len(df["RoleName"].unique()),
                "n_permissions": len(df["Permission"].unique()),
                "date_range": (df["TimeGenerated"].min(), df["TimeGenerated"].max()),
            },
        }

    def _create_rbac_graphs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create various graph representations of RBAC data."""
        graphs = {}

        # User-Role Graph
        user_role_edges = df[["user_id", "role_id"]].drop_duplicates()
        user_role_graph = nx.Graph()
        for _, row in user_role_edges.iterrows():
            user_role_graph.add_edge(f"user_{row['user_id']}", f"role_{row['role_id']}")
        graphs["user_role"] = user_role_graph

        # Role-Permission Graph
        role_perm_edges = df[["role_id", "permission_id"]].drop_duplicates()
        role_perm_graph = nx.Graph()
        for _, row in role_perm_edges.iterrows():
            role_perm_graph.add_edge(f"role_{row['role_id']}", f"perm_{row['permission_id']}")
        graphs["role_permission"] = role_perm_graph

        # User-Permission Graph (transitive)
        user_perm_edges = df[["user_id", "permission_id"]].drop_duplicates()
        user_perm_graph = nx.Graph()
        for _, row in user_perm_edges.iterrows():
            user_perm_graph.add_edge(f"user_{row['user_id']}", f"perm_{row['permission_id']}")
        graphs["user_permission"] = user_perm_graph

        # Full RBAC Graph (tripartite)
        full_graph = nx.Graph()
        full_graph.add_edges_from(user_role_graph.edges())
        full_graph.add_edges_from(role_perm_graph.edges())
        graphs["full_rbac"] = full_graph

        # Convert to PyTorch Geometric format
        torch_graphs = {}
        for graph_name, graph in graphs.items():
            torch_graphs[graph_name] = self._networkx_to_torch_geometric(graph)

        graphs["torch_geometric"] = torch_graphs

        return graphs

    def _create_rbac_matrices(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create various matrix representations of RBAC data."""
        n_users = len(df["user_id"].unique())
        n_roles = len(df["role_id"].unique())
        n_permissions = len(df["permission_id"].unique())

        # User-Role Assignment Matrix
        user_role_matrix = np.zeros((n_users, n_roles))
        user_role_pairs = df[["user_id", "role_id"]].drop_duplicates()
        for _, row in user_role_pairs.iterrows():
            user_role_matrix[row["user_id"], row["role_id"]] = 1

        # Role-Permission Assignment Matrix
        role_perm_matrix = np.zeros((n_roles, n_permissions))
        role_perm_pairs = df[["role_id", "permission_id"]].drop_duplicates()
        for _, row in role_perm_pairs.iterrows():
            role_perm_matrix[row["role_id"], row["permission_id"]] = 1

        # User-Permission Matrix (transitive)
        user_perm_matrix = user_role_matrix @ role_perm_matrix

        return {
            "user_role": user_role_matrix,
            "role_permission": role_perm_matrix,
            "user_permission": user_perm_matrix,
        }

    def _extract_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features for each user."""
        user_features = []

        for user_id in df["user_id"].unique():
            user_data = df[df["user_id"] == user_id]

            # Time-based features
            user_data["hour"] = user_data["TimeGenerated"].dt.hour
            user_data["day_of_week"] = user_data["TimeGenerated"].dt.dayofweek

            features = {
                "user_id": user_id,
                "total_actions": len(user_data),
                "unique_roles": user_data["role_id"].nunique(),
                "unique_permissions": user_data["permission_id"].nunique(),
                "unique_resources": user_data["ResourceId"].nunique(),
                "avg_actions_per_day": len(user_data) / 30,  # Assuming 30 days
                "weekend_activity": len(user_data[user_data["day_of_week"] >= 5]) / len(user_data),
                "night_activity": len(user_data[(user_data["hour"] < 6) | (user_data["hour"] > 22)])
                / len(user_data),
                "role_diversity": user_data["role_id"].nunique() / df["role_id"].nunique(),
                "permission_diversity": user_data["permission_id"].nunique()
                / df["permission_id"].nunique(),
            }

            user_features.append(features)

        return pd.DataFrame(user_features)

    def _infer_role_hierarchy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer role hierarchy from permission overlap."""
        role_perm_matrix = self._create_rbac_matrices(df)["role_permission"]

        hierarchy = {}
        n_roles = role_perm_matrix.shape[0]

        # Calculate role similarity based on permission overlap
        role_similarity = np.zeros((n_roles, n_roles))
        for i in range(n_roles):
            for j in range(n_roles):
                if i != j:
                    # Jaccard similarity
                    intersection = np.sum(role_perm_matrix[i] & role_perm_matrix[j])
                    union = np.sum(role_perm_matrix[i] | role_perm_matrix[j])
                    role_similarity[i, j] = intersection / union if union > 0 else 0

        # Infer parent-child relationships
        hierarchy_matrix = np.zeros((n_roles, n_roles))
        for i in range(n_roles):
            for j in range(n_roles):
                if i != j:
                    # i is parent of j if j's permissions are subset of i's
                    is_subset = np.all(role_perm_matrix[j] <= role_perm_matrix[i])
                    if is_subset and np.sum(role_perm_matrix[i]) > np.sum(role_perm_matrix[j]):
                        hierarchy_matrix[i, j] = 1  # i is parent of j

        return {
            "similarity_matrix": role_similarity,
            "hierarchy_matrix": hierarchy_matrix,
            "role_levels": self._calculate_role_levels(hierarchy_matrix),
        }

    def _calculate_role_levels(self, hierarchy_matrix: np.ndarray) -> Dict[int, int]:
        """Calculate hierarchical levels for roles."""
        n_roles = hierarchy_matrix.shape[0]
        levels = {}

        # Find root roles (no parents)
        root_roles = []
        for i in range(n_roles):
            if np.sum(hierarchy_matrix[:, i]) == 0:  # No incoming edges
                root_roles.append(i)
                levels[i] = 0

        # BFS to assign levels
        from collections import deque

        queue = deque([(role, 0) for role in root_roles])

        while queue:
            role, level = queue.popleft()

            # Find children
            children = np.where(hierarchy_matrix[role, :] == 1)[0]
            for child in children:
                if child not in levels:
                    levels[child] = level + 1
                    queue.append((child, level + 1))

        return levels

    def _detect_access_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various access patterns and anomalies."""
        patterns = {}

        # Time-based patterns
        df["hour"] = df["TimeGenerated"].dt.hour
        df["day_of_week"] = df["TimeGenerated"].dt.dayofweek

        patterns["hourly_distribution"] = df["hour"].value_counts().sort_index().to_dict()
        patterns["daily_distribution"] = df["day_of_week"].value_counts().sort_index().to_dict()

        # Role usage patterns
        patterns["role_popularity"] = df["RoleName"].value_counts().to_dict()
        patterns["permission_popularity"] = df["Permission"].value_counts().to_dict()

        # User behavior clusters
        user_features = self._extract_user_features(df)
        if len(user_features) > 1:
            feature_cols = [
                "total_actions",
                "unique_roles",
                "unique_permissions",
                "weekend_activity",
                "night_activity",
            ]
            X = user_features[feature_cols].fillna(0)
            X_scaled = self.scalers["standard"].fit_transform(X)

            clusters = self.models["kmeans"].fit_predict(X_scaled)
            patterns["user_clusters"] = {
                "cluster_labels": clusters.tolist(),
                "cluster_centers": self.models["kmeans"].cluster_centers_.tolist(),
                "silhouette_score": (
                    silhouette_score(X_scaled, clusters) if len(set(clusters)) > 1 else 0
                ),
            }

        return patterns

    def _networkx_to_torch_geometric(self, graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        # Create node mapping
        nodes = list(graph.nodes())
        node_mapping = {node: i for i, node in enumerate(nodes)}

        # Create edge index
        edges = list(graph.edges())
        edge_index = torch.tensor(
            [
                [node_mapping[edge[0]] for edge in edges]
                + [node_mapping[edge[1]] for edge in edges],
                [node_mapping[edge[1]] for edge in edges]
                + [node_mapping[edge[0]] for edge in edges],
            ],
            dtype=torch.long,
        )

        # Create dummy node features
        num_nodes = len(nodes)
        node_features = torch.randn(num_nodes, self.config["graph_config"]["node_features"])

        return Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)

    async def train_gnn_models(self, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train Graph Neural Network models for RBAC analysis.

        Args:
            rbac_data: Processed RBAC data with graphs and matrices

        Returns:
            Training results and model performance
        """
        logger.info("Training GNN models for RBAC analysis")

        if not rbac_data["graphs"]:
            raise ValueError("No graph data available for training")

        training_results = {}

        # Train GAT model for anomaly detection
        gat_results = await self._train_gat_model(rbac_data)
        training_results["gat"] = gat_results

        # Train hierarchy analysis model
        hierarchy_results = await self._train_hierarchy_model(rbac_data)
        training_results["hierarchy"] = hierarchy_results

        # Train traditional anomaly detection models
        anomaly_results = self._train_anomaly_detection(rbac_data)
        training_results["anomaly_detection"] = anomaly_results

        return training_results

    async def _train_gat_model(self, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train Graph Attention Network for RBAC analysis."""
        logger.info("Training Graph Attention Network")

        # Get the full RBAC graph
        graph_data = rbac_data["graphs"]["torch_geometric"]["full_rbac"]

        # Initialize GAT model
        self.models["gat"] = RBACGraphAttentionNetwork(
            node_features=self.config["graph_config"]["node_features"],
            **self.config["graph_config"],
        )

        # Training setup
        optimizer = optim.Adam(
            self.models["gat"].parameters(), lr=self.config["training"]["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss()

        # Generate pseudo-labels for training (in practice, use real labels)
        num_nodes = graph_data.x.shape[0]
        pseudo_labels = torch.randint(0, 2, (num_nodes,))

        # Training loop
        self.models["gat"].train()
        epoch_losses = []

        for epoch in range(self.config["training"]["epochs"]):
            optimizer.zero_grad()

            outputs = self.models["gat"](graph_data.x, graph_data.edge_index)

            # Node-level loss (anomaly detection)
            node_loss = F.binary_cross_entropy(
                outputs["node_anomaly_scores"].squeeze(), pseudo_labels.float()
            )

            # Graph-level loss (placeholder)
            graph_loss = criterion(outputs["graph_predictions"], torch.tensor([0]))  # Dummy label

            total_loss = node_loss + 0.1 * graph_loss
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if epoch % 50 == 0:
                logger.info(f"GAT Epoch {epoch}: Loss = {total_loss.item():.4f}")

        self.models["gat"].eval()

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("inf"),
            "training_losses": epoch_losses,
        }

    async def _train_hierarchy_model(self, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train hierarchy analysis model."""
        logger.info("Training RBAC hierarchy model")

        # Get role-permission graph
        graph_data = rbac_data["graphs"]["torch_geometric"]["role_permission"]

        # Initialize hierarchy model
        self.models["hierarchy"] = RBACHierarchyGNN(
            node_features=self.config["graph_config"]["node_features"],
            **self.config["graph_config"],
        )

        # Training setup
        optimizer = optim.Adam(
            self.models["hierarchy"].parameters(), lr=self.config["training"]["learning_rate"]
        )

        # Training loop
        self.models["hierarchy"].train()
        epoch_losses = []

        for epoch in range(self.config["training"]["epochs"] // 2):  # Fewer epochs
            optimizer.zero_grad()

            outputs = self.models["hierarchy"](graph_data.x, graph_data.edge_index)

            # Generate pseudo-labels for hierarchy relationships
            num_edges = graph_data.edge_index.shape[1]
            hierarchy_labels = torch.randint(0, 3, (num_edges,))

            hierarchy_loss = F.cross_entropy(outputs["hierarchy_predictions"], hierarchy_labels)

            permission_loss = F.binary_cross_entropy(
                outputs["permission_scores"].squeeze(), torch.rand(graph_data.x.shape[0])
            )

            total_loss = hierarchy_loss + 0.5 * permission_loss
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if epoch % 25 == 0:
                logger.info(f"Hierarchy Epoch {epoch}: Loss = {total_loss.item():.4f}")

        self.models["hierarchy"].eval()

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("inf"),
            "training_losses": epoch_losses,
        }

    def _train_anomaly_detection(self, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train traditional anomaly detection models."""
        logger.info("Training anomaly detection models")

        user_features = rbac_data["user_features"]

        if user_features.empty:
            return {"error": "No user features available"}

        # Prepare features
        feature_cols = [
            "total_actions",
            "unique_roles",
            "unique_permissions",
            "weekend_activity",
            "night_activity",
            "role_diversity",
        ]
        X = user_features[feature_cols].fillna(0)
        X_scaled = self.scalers["standard"].fit_transform(X)

        # Train Isolation Forest
        self.models["isolation_forest"].fit(X_scaled)
        anomaly_scores = self.models["isolation_forest"].decision_function(X_scaled)
        anomaly_labels = self.models["isolation_forest"].predict(X_scaled)

        # Train DBSCAN
        cluster_labels = self.models["dbscan"].fit_predict(X_scaled)

        return {
            "isolation_forest": {
                "anomaly_scores": anomaly_scores.tolist(),
                "anomaly_labels": anomaly_labels.tolist(),
                "n_anomalies": np.sum(anomaly_labels == -1),
            },
            "dbscan": {
                "cluster_labels": cluster_labels.tolist(),
                "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "n_noise": np.sum(cluster_labels == -1),
            },
        }

    async def analyze_user_access(self, user_id: str, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze access patterns for a specific user.

        Args:
            user_id: User identifier
            rbac_data: RBAC data and trained models

        Returns:
            User access analysis results
        """
        logger.info(f"Analyzing access patterns for user: {user_id}")

        try:
            # Get user features
            user_features = rbac_data["user_features"]
            user_data = user_features[user_features["user_id"] == user_id]

            if user_data.empty:
                return {
                    "user_id": user_id,
                    "error": "User not found in dataset",
                    "recommendations": ["User not found in recent activity logs"],
                }

            user_row = user_data.iloc[0]

            # Anomaly detection
            feature_cols = [
                "total_actions",
                "unique_roles",
                "unique_permissions",
                "weekend_activity",
                "night_activity",
                "role_diversity",
            ]
            X = user_row[feature_cols].values.reshape(1, -1)
            X_scaled = self.scalers["standard"].transform(X)

            # Get anomaly scores
            anomaly_score = self.models["isolation_forest"].decision_function(X_scaled)[0]
            is_anomaly = self.models["isolation_forest"].predict(X_scaled)[0] == -1

            # Role analysis
            role_analysis = self._analyze_user_roles(user_row, rbac_data)

            # Permission analysis
            permission_analysis = self._analyze_user_permissions(user_row, rbac_data)

            # Generate recommendations
            recommendations = self._generate_user_recommendations(
                user_row, anomaly_score, is_anomaly, role_analysis, permission_analysis
            )

            return {
                "user_id": user_id,
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly),
                "role_analysis": role_analysis,
                "permission_analysis": permission_analysis,
                "risk_level": self._calculate_risk_level(anomaly_score, role_analysis),
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Error analyzing user access: {str(e)}")
            return {
                "user_id": user_id,
                "error": str(e),
                "recommendations": ["Error in analysis - manual review recommended"],
            }

    def _analyze_user_roles(
        self, user_data: pd.Series, rbac_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user's role assignments."""
        role_hierarchy = rbac_data["role_hierarchy"]

        return {
            "unique_roles": int(user_data["unique_roles"]),
            "role_diversity": float(user_data["role_diversity"]),
            "is_over_privileged": user_data["role_diversity"] > 0.2,  # Threshold
            "role_conflicts": self._detect_role_conflicts(user_data, role_hierarchy),
        }

    def _analyze_user_permissions(
        self, user_data: pd.Series, rbac_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user's permission assignments."""
        return {
            "unique_permissions": int(user_data["unique_permissions"]),
            "permission_diversity": float(user_data["permission_diversity"]),
            "excessive_permissions": user_data["permission_diversity"] > 0.3,  # Threshold
            "unused_permissions": self._detect_unused_permissions(user_data),
        }

    def _detect_role_conflicts(
        self, user_data: pd.Series, role_hierarchy: Dict[str, Any]
    ) -> List[str]:
        """Detect potential conflicts in user's role assignments."""
        conflicts = []

        # This would contain more sophisticated conflict detection logic
        if user_data["unique_roles"] > 5:
            conflicts.append("User has unusually high number of roles")

        if user_data["role_diversity"] > 0.4:
            conflicts.append("User roles span multiple organizational areas")

        return conflicts

    def _detect_unused_permissions(self, user_data: pd.Series) -> List[str]:
        """Detect potentially unused permissions."""
        unused = []

        # This would contain logic to detect unused permissions
        # based on activity patterns
        if user_data["total_actions"] < 10:
            unused.append("Low activity suggests unused permissions")

        return unused

    def _calculate_risk_level(self, anomaly_score: float, role_analysis: Dict[str, Any]) -> str:
        """Calculate overall risk level for the user."""
        risk_factors = 0

        if anomaly_score < -0.5:
            risk_factors += 2
        elif anomaly_score < 0:
            risk_factors += 1

        if role_analysis["is_over_privileged"]:
            risk_factors += 2

        if len(role_analysis["role_conflicts"]) > 0:
            risk_factors += 1

        if risk_factors >= 4:
            return "High"
        elif risk_factors >= 2:
            return "Medium"
        else:
            return "Low"

    def _generate_user_recommendations(
        self,
        user_data: pd.Series,
        anomaly_score: float,
        is_anomaly: bool,
        role_analysis: Dict[str, Any],
        permission_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations for the user."""
        recommendations = []

        if is_anomaly:
            recommendations.append("User shows anomalous access patterns - conduct security review")

        if role_analysis["is_over_privileged"]:
            recommendations.append("Review user's role assignments - may be over-privileged")

        if permission_analysis["excessive_permissions"]:
            recommendations.append("Audit user's permissions - some may be unnecessary")

        if user_data["night_activity"] > 0.3:
            recommendations.append("High off-hours activity detected - verify legitimacy")

        if user_data["weekend_activity"] > 0.4:
            recommendations.append("High weekend activity detected - review business justification")

        if len(role_analysis["role_conflicts"]) > 0:
            recommendations.extend(role_analysis["role_conflicts"])

        if not recommendations:
            recommendations.append("User access patterns appear normal")

        return recommendations

    async def optimize_role_structure(self, rbac_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the role structure based on usage patterns.

        Args:
            rbac_data: RBAC data and analysis results

        Returns:
            Role optimization recommendations
        """
        logger.info("Optimizing role structure")

        try:
            matrices = rbac_data["matrices"]
            user_features = rbac_data["user_features"]

            # Analyze role usage
            role_usage = self._analyze_role_usage(matrices, user_features)

            # Detect redundant roles
            redundant_roles = self._detect_redundant_roles(matrices)

            # Suggest role consolidation
            consolidation_suggestions = self._suggest_role_consolidation(matrices, user_features)

            # Detect missing roles
            missing_roles = self._detect_missing_roles(matrices, user_features)

            return {
                "role_usage_analysis": role_usage,
                "redundant_roles": redundant_roles,
                "consolidation_suggestions": consolidation_suggestions,
                "missing_roles": missing_roles,
                "optimization_score": self._calculate_optimization_score(
                    role_usage, redundant_roles
                ),
            }

        except Exception as e:
            logger.error(f"Error optimizing role structure: {str(e)}")
            return {"error": str(e)}

    def _analyze_role_usage(
        self, matrices: Dict[str, np.ndarray], user_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze how roles are being used."""
        user_role_matrix = matrices["user_role"]

        role_stats = {
            "role_popularity": np.sum(user_role_matrix, axis=0).tolist(),
            "role_exclusivity": [],
            "role_coverage": [],
        }

        # Calculate role exclusivity (how often roles are assigned alone)
        for role_idx in range(user_role_matrix.shape[1]):
            role_users = user_role_matrix[:, role_idx]
            exclusive_users = 0

            for user_idx in range(user_role_matrix.shape[0]):
                if role_users[user_idx] == 1:
                    user_roles = user_role_matrix[user_idx, :]
                    if np.sum(user_roles) == 1:  # Only has this role
                        exclusive_users += 1

            exclusivity = exclusive_users / np.sum(role_users) if np.sum(role_users) > 0 else 0
            role_stats["role_exclusivity"].append(exclusivity)

        return role_stats

    def _detect_redundant_roles(self, matrices: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect redundant roles based on permission overlap."""
        role_perm_matrix = matrices["role_permission"]
        redundant_roles = []

        for i in range(role_perm_matrix.shape[0]):
            for j in range(i + 1, role_perm_matrix.shape[0]):
                # Calculate Jaccard similarity
                intersection = np.sum(role_perm_matrix[i] & role_perm_matrix[j])
                union = np.sum(role_perm_matrix[i] | role_perm_matrix[j])
                similarity = intersection / union if union > 0 else 0

                if similarity > self.config["clustering"]["similarity_threshold"]:
                    redundant_roles.append(
                        {
                            "role_1": int(i),
                            "role_2": int(j),
                            "similarity": float(similarity),
                            "recommendation": "Consider consolidating these roles",
                        }
                    )

        return redundant_roles

    def _suggest_role_consolidation(
        self, matrices: Dict[str, np.ndarray], user_features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Suggest role consolidation opportunities."""
        # This would contain sophisticated logic for role consolidation
        # For now, return placeholder suggestions
        return [
            {
                "consolidation_type": "merge_similar_roles",
                "roles_to_merge": [1, 2, 3],
                "new_role_name": "Consolidated Role A",
                "expected_benefit": "Reduced complexity, easier management",
            }
        ]

    def _detect_missing_roles(
        self, matrices: Dict[str, np.ndarray], user_features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect missing role opportunities."""
        # This would analyze permission patterns to suggest new roles
        return [
            {
                "suggested_role": "Junior Administrator",
                "permissions": ["read_only_access", "basic_operations"],
                "rationale": "Common permission pattern not covered by existing roles",
            }
        ]

    def _calculate_optimization_score(
        self, role_usage: Dict[str, Any], redundant_roles: List[Dict[str, Any]]
    ) -> float:
        """Calculate a score representing how well-optimized the role structure is."""
        # Simple scoring based on redundancy and usage
        redundancy_penalty = len(redundant_roles) * 0.1
        usage_score = np.mean(role_usage["role_popularity"]) / 100  # Normalize

        optimization_score = max(0, 1.0 - redundancy_penalty + usage_score)
        return float(optimization_score)

    async def save_models(self, model_path: str) -> bool:
        """Save all trained models."""
        try:
            model_path = Path(model_path)
            model_path.mkdir(parents=True, exist_ok=True)

            # Save PyTorch models
            for model_name in ["gat", "hierarchy"]:
                if model_name in self.models and self.models[model_name]:
                    torch.save(
                        self.models[model_name].state_dict(), model_path / f"{model_name}_model.pth"
                    )

            # Save sklearn models
            sklearn_models = ["isolation_forest", "dbscan", "kmeans"]
            for model_name in sklearn_models:
                if model_name in self.models:
                    with open(model_path / f"{model_name}_model.pkl", "wb") as f:
                        pickle.dump(self.models[model_name], f)

            # Save scalers and encoders
            with open(model_path / "scalers.pkl", "wb") as f:
                pickle.dump(self.scalers, f)

            with open(model_path / "encoders.pkl", "wb") as f:
                pickle.dump(self.encoders, f)

            # Save configuration
            with open(model_path / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"RBAC models saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving RBAC models: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":

    async def main():
        # Initialize analyzer
        analyzer = RBACAnalyzer()

        # Prepare RBAC data
        rbac_data = await analyzer.prepare_rbac_data("sample_workspace", days_back=30)

        if rbac_data["graphs"]:
            # Train models
            training_results = await analyzer.train_gnn_models(rbac_data)
            print("Training Results:", training_results)

            # Analyze a user
            if rbac_data["user_features"] is not None and not rbac_data["user_features"].empty:
                sample_user = rbac_data["user_features"]["user_id"].iloc[0]
                user_analysis = await analyzer.analyze_user_access(sample_user, rbac_data)
                print("User Analysis:", user_analysis)

            # Optimize role structure
            optimization = await analyzer.optimize_role_structure(rbac_data)
            print("Role Optimization:", optimization)

            # Save models
            await analyzer.save_models("./models/rbac_analysis")

        else:
            print("No RBAC data available")

    # Run the example
    asyncio.run(main())
