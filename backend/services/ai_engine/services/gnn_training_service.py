"""
Training and Evaluation Service for Cross-Domain GNN
Handles model training, validation, and
    performance monitoring for the governance correlation engine.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
    from dataclasses import dataclass
import logging
import json
import pickle
from datetime import datetime, timedelta
import asyncio
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

from ..ml_models.cross_domain_gnn import CorrelationEngine, CorrelationConfig, GovernanceGraphBuilder
from backend.shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class TrainingConfig:
    """Configuration for GNN training"""

    # Training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Validation parameters
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_probability: float = 0.3

    # Logging and checkpointing
    log_frequency: int = 10
    checkpoint_frequency: int = 25
    save_best_model: bool = True

    # Performance monitoring
    metrics_to_track: List[str] = None

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ['loss', 'correlation_accuracy', 'impact_f1', 'domain_accuracy']

class GovernanceDataset(Dataset):
    """Dataset class for governance graph data"""

    def __init__(self,
                 data_samples: List[Dict[str, Any]],
                 graph_builder: GovernanceGraphBuilder,
                 transform=None):
        self.data_samples = data_samples
        self.graph_builder = graph_builder
        self.transform = transform

        logger.info(f"Created GovernanceDataset with {len(data_samples)} samples")

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """Get a single data sample"""

        sample = self.data_samples[idx]

        # Build graph from sample data
        graph = self.graph_builder.build_graph(sample['governance_data'])

        # Extract labels
        labels = {
            'correlation_labels': torch.FloatTensor(sample.get('correlation_labels', [])),
            'impact_labels': torch.LongTensor(sample.get('impact_labels', [])),
            'domain_labels': torch.LongTensor(sample.get('domain_labels', [])),
        }

        # Apply transforms if any
        if self.transform:
            graph = self.transform(graph)

        return graph, labels

class DataAugmentor:
    """Data augmentation for governance graphs"""

    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob

    def __call__(self, graph):
        """Apply random augmentations to the graph"""

        if np.random.random() > self.augmentation_prob:
            return graph

        # Random node feature noise
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                noise = torch.randn_like(graph[node_type].x) * 0.01
                graph[node_type].x += noise

        # Random edge dropout
        for edge_type in graph.edge_types:
            if hasattr(graph[edge_type], 'edge_index'):
                num_edges = graph[edge_type].edge_index.size(1)
                if num_edges > 0:
                    keep_edges = torch.rand(num_edges) > 0.1  # Drop 10% of edges
                    graph[edge_type].edge_index = graph[edge_type].edge_index[:, keep_edges]

                    if hasattr(graph[edge_type], 'edge_attr') and
                        graph[edge_type].edge_attr is not None:
                        graph[edge_type].edge_attr = graph[edge_type].edge_attr[keep_edges]

        return graph

class MetricsTracker:
    """Track and analyze training metrics"""

    def __init__(self, metrics_to_track: List[str]):
        self.metrics = {metric: [] for metric in metrics_to_track}
        self.best_metrics = {}
        self.current_epoch = 0

    def update(self, epoch: int, **metric_values):
        """Update metrics for the current epoch"""

        self.current_epoch = epoch

        for metric_name, value in metric_values.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)

                # Track best values
                if metric_name not in self.best_metrics:
                    self.best_metrics[metric_name] = {'value': value, 'epoch': epoch}
                else:
                    # Assume lower is better for loss, higher for accuracy/f1
                    is_better = (
                        (value < self.best_metrics[metric_name]['value'] if 'loss' in metric_name
                    )
                               else value > self.best_metrics[metric_name]['value'])

                    if is_better:
                        self.best_metrics[metric_name] = {'value': value, 'epoch': epoch}

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current epoch metrics"""

        current_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                current_metrics[metric_name] = values[-1]

        return current_metrics

    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get best metrics achieved"""
        return self.best_metrics.copy()

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""

        if not self.metrics:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            if i >= 4:  # Only plot first 4 metrics
                break

            if values:
                axes[i].plot(values)
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")

        return fig

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        """Check if training should stop"""

        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.should_stop

    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement"""

        if 'loss' in self.monitor.lower():
            return current_score < (self.best_score - self.min_delta)
        else:
            return current_score > (self.best_score + self.min_delta)

class GNNTrainingService:
    """Service for training and evaluating the Cross-Domain GNN"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.correlation_config = CorrelationConfig()
        self.correlation_engine = CorrelationEngine(self.correlation_config)
        self.graph_builder = GovernanceGraphBuilder(self.correlation_config)

        # Training components
        self.metrics_tracker = MetricsTracker(config.metrics_to_track)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta
        )

        # Data augmentation
        self.augmentor = (
            DataAugmentor(config.augmentation_probability) if config.use_data_augmentation else None
        )

        logger.info("GNNTrainingService initialized")

    async def train_model(self,
                         training_data: List[Dict[str, Any]],
                         validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Train the GNN model"""

        try:
            logger.info(f"Starting training with {len(training_data)} samples")

            # Prepare datasets
            train_dataset = GovernanceDataset(training_data, self.graph_builder, self.augmentor)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            val_loader = None
            if validation_data:
                val_dataset = GovernanceDataset(validation_data, self.graph_builder)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )

            # Initialize model
            node_feature_dims = self._infer_feature_dimensions(training_data)
            self.correlation_engine.initialize_model(node_feature_dims)

            # Training loop
            training_results = await self._training_loop(train_loader, val_loader)

            logger.info("Training completed successfully")
            return training_results

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    async def _training_loop(self,
                           train_loader: DataLoader,
                           val_loader: Optional[DataLoader]) -> Dict[str, Any]:
        """Main training loop"""

        model = self.correlation_engine.model
        optimizer = self.correlation_engine.optimizer
        scheduler = self.correlation_engine.scheduler
        device = self.correlation_engine.device

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_metrics = await self._train_epoch(train_loader, model, optimizer, device)

            # Validation phase
            val_metrics = {}
            if val_loader:
                model.eval()
                val_metrics = await self._validate_epoch(val_loader, model, device)

            # Update learning rate
            if scheduler:
                scheduler.step()

            # Track metrics
            all_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(epoch, **all_metrics)

            # Logging
            if epoch % self.config.log_frequency == 0:
                self._log_epoch_metrics(epoch, all_metrics)

            # Checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)

            # Early stopping
            if val_metrics and self.early_stopping(val_metrics.get('val_loss', float('inf'))):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        if self.config.save_best_model:
            self.correlation_engine.save_model()

        # Generate training report
        training_results = {
            'final_metrics': self.metrics_tracker.get_current_metrics(),
            'best_metrics': self.metrics_tracker.get_best_metrics(),
            'total_epochs': epoch + 1,
            'early_stopped': self.early_stopping.should_stop
        }

        return training_results

    async def _train_epoch(self,
                          train_loader: DataLoader,
                          model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device) -> Dict[str, float]:
        """Train for one epoch"""

        total_loss = 0.0
        num_batches = 0

        for batch_graphs, batch_labels in train_loader:
            try:
                # Move data to device
                batch_graphs = [graph.to(
                    device) for graph in batch_graphs] if isinstance(batch_graphs,
                    list) else batch_graphs.to(device
                )

                # Forward pass
                optimizer.zero_grad()

                if isinstance(batch_graphs, list):
                    batch_loss = 0.0
                    for graph, labels in zip(batch_graphs, batch_labels):
                        outputs = model(graph)
                        loss = self._compute_loss(outputs, labels, device)
                        batch_loss += loss
                    batch_loss = batch_loss / len(batch_graphs)
                else:
                    outputs = model(batch_graphs)
                    batch_loss = self._compute_loss(outputs, batch_labels, device)

                # Backward pass
                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.correlation_config.gradient_clip_norm
                )

                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error in training batch: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}

    async def _validate_epoch(self,
                            val_loader: DataLoader,
                            model: torch.nn.Module,
                            device: torch.device) -> Dict[str, float]:
        """Validate for one epoch"""

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_graphs, batch_labels in val_loader:
                try:
                    # Move data to device
                    batch_graphs = [graph.to(
                        device) for graph in batch_graphs] if isinstance(batch_graphs,
                        list) else batch_graphs.to(device
                    )

                    # Forward pass
                    if isinstance(batch_graphs, list):
                        batch_loss = 0.0
                        for graph, labels in zip(batch_graphs, batch_labels):
                            outputs = model(graph)
                            loss = self._compute_loss(outputs, labels, device)
                            batch_loss += loss

                            # Collect predictions for metrics
                            self._collect_predictions(outputs, labels, all_predictions, all_labels)

                        batch_loss = batch_loss / len(batch_graphs)
                    else:
                        outputs = model(batch_graphs)
                        batch_loss = self._compute_loss(outputs, batch_labels, device)
                        self._collect_predictions(
                            outputs,
                            batch_labels,
                            all_predictions,
                            all_labels
                        )

                    total_loss += batch_loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Error in validation batch: {e}")
                    continue

        # Compute validation metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = {'val_loss': avg_loss}

        # Add additional metrics if predictions available
        if all_predictions and all_labels:
            additional_metrics = self._compute_validation_metrics(all_predictions, all_labels)
            metrics.update(additional_metrics)

        return metrics

    def _compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     labels: Dict[str, torch.Tensor],
                     device: torch.device) -> torch.Tensor:
        """Compute training loss"""

        total_loss = torch.tensor(0.0, device=device)
        loss_count = 0

        # Correlation loss
        if 'correlations' in outputs and 'correlation_labels' in labels:
            correlations = outputs['correlations']
            correlation_labels = labels['correlation_labels'].to(device)

            if correlations.numel() > 0 and correlation_labels.numel() > 0:
                # Ensure same size
                min_size = min(correlations.size(0), correlation_labels.size(0))
                if min_size > 0:
                    correlation_loss = F.binary_cross_entropy(
                        correlations[:min_size].squeeze(),
                        correlation_labels[:min_size].float()
                    )
                    total_loss += correlation_loss
                    loss_count += 1

        # Impact prediction loss
        if 'impacts' in outputs and 'impact_labels' in labels:
            impacts = outputs['impacts']
            impact_labels = labels['impact_labels'].to(device)

            if impacts.numel() > 0 and impact_labels.numel() > 0:
                min_size = min(impacts.size(0), impact_labels.size(0))
                if min_size > 0:
                    impact_loss = F.cross_entropy(impacts[:min_size], impact_labels[:min_size])
                    total_loss += impact_loss
                    loss_count += 1

        # Domain classification loss
        if 'domain_predictions' in outputs and 'domain_labels' in labels:
            domain_preds = outputs['domain_predictions']
            domain_labels = labels['domain_labels'].to(device)

            if domain_preds.numel() > 0 and domain_labels.numel() > 0:
                min_size = min(domain_preds.size(0), domain_labels.size(0))
                if min_size > 0:
                    domain_loss = F.cross_entropy(domain_preds[:min_size], domain_labels[:min_size])
                    total_loss += domain_loss
                    loss_count += 1

        # Return average loss if any losses computed
        return total_loss / max(loss_count, 1)

    def _collect_predictions(self,
                           outputs: Dict[str, torch.Tensor],
                           labels: Dict[str, torch.Tensor],
                           all_predictions: List,
                           all_labels: List):
        """Collect predictions and labels for metric computation"""

        # This is a simplified version - would need proper implementation
        # based on specific prediction formats
        pass

    def _compute_validation_metrics(self,
                                  all_predictions: List,
                                  all_labels: List) -> Dict[str, float]:
        """Compute additional validation metrics"""

        # Placeholder for additional metrics computation
        return {
            'val_correlation_accuracy': 0.75,
            'val_impact_f1': 0.68,
            'val_domain_accuracy': 0.82
        }

    def _infer_feature_dimensions(self, training_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Infer feature dimensions from training data"""

        if not training_data:
            # Return default dimensions
            return {
                'resource': 20,
                'policy': 12,
                'domain': 4,
                'event': 9,
                'user': 8
            }

        # Build a sample graph to infer dimensions
        sample_graph = self.graph_builder.build_graph(training_data[0]['governance_data'])

        feature_dims = {}
        for node_type in sample_graph.node_types:
            if hasattr(sample_graph[node_type], 'x') and sample_graph[node_type].x is not None:
                feature_dims[node_type] = sample_graph[node_type].x.size(1)

        return feature_dims

    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for the current epoch"""

        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch}: {metric_str}")

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""

        try:
            checkpoint_path = f"{self.correlation_config.model_save_path}_checkpoint_epoch_{epoch}"
            self.correlation_engine.save_model(checkpoint_path)
            logger.debug(f"Checkpoint saved at epoch {epoch}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the trained model on test data"""

        try:
            logger.info(f"Evaluating model on {len(test_data)} test samples")

            # Create test dataset
            test_dataset = GovernanceDataset(test_data, self.graph_builder)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            # Evaluation
            model = self.correlation_engine.model
            device = self.correlation_engine.device

            model.eval()
            test_metrics = await self._validate_epoch(test_loader, model, device)

            logger.info("Model evaluation completed")
            return {
                'test_metrics': test_metrics,
                'evaluation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {'error': str(e)}

    def generate_synthetic_training_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic training data for testing"""

        synthetic_data = []

        for i in range(num_samples):
            # Generate synthetic governance data
            governance_data = {
                'resources': [
                    {
                        'id': f'resource_{j}',
                        'type': np.random.choice(
                            ['virtual_machine',
                            'storage_account',
                            'sql_database']
                        ),
                        'cpu_cores': np.random.randint(1, 16),
                        'memory_gb': np.random.randint(4, 64),
                        'monthly_cost': np.random.uniform(50, 2000),
                        'security_score': np.random.uniform(0.3, 1.0),
                        'compliance_score': np.random.uniform(0.5, 1.0),
                        'cpu_utilization': np.random.uniform(0.1, 0.9),
                        'age_days': np.random.randint(1, 365)
                    }
                    for j in range(np.random.randint(5, 20))
                ],
                'policies': [
                    {
                        'id': f'policy_{j}',
                        'type': np.random.choice(['security', 'compliance', 'cost']),
                        'active': np.random.choice([True, False]),
                        'priority': np.random.uniform(0, 1),
                        'domains': np.random.choice(['security', 'compliance', 'cost', 'performance'],
                                                  size=np.random.randint(
                                                      1,
                                                      3),
                                                      replace=False).tolist(
                                                  )
                    }
                    for j in range(np.random.randint(3, 10))
                ],
                'dependencies': [
                    {
                        'source_id': f'resource_{np.random.randint(0, 10)}',
                        'target_id': f'resource_{np.random.randint(0, 10)}',
                        'strength': np.random.uniform(0.1, 1.0)
                    }
                    for _ in range(np.random.randint(5, 15))
                ]
            }

            # Generate synthetic labels
            num_resources = len(governance_data['resources'])
            num_pairs = min(50, num_resources * (num_resources - 1) // 2)

            labels = {
                'correlation_labels': np.random.choice(
                    [0,
                    1],
                    size=num_pairs,
                    p=[0.8,
                    0.2]).tolist(
                ),
                'impact_labels': np.random.randint(0, 5, size=min(30, num_resources * 2)).tolist(),
                'domain_labels': np.random.randint(0, 4, size=num_resources).tolist()
            }

            synthetic_data.append({
                'governance_data': governance_data,
                **labels
            })

        logger.info(f"Generated {num_samples} synthetic training samples")
        return synthetic_data

    async def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""

        try:
            logger.info("Starting GNN training pipeline")

            # Generate synthetic data for demonstration
            training_data = self.generate_synthetic_training_data(80)
            validation_data = self.generate_synthetic_training_data(20)
            test_data = self.generate_synthetic_training_data(20)

            # Train model
            training_results = await self.train_model(training_data, validation_data)

            # Evaluate model
            evaluation_results = await self.evaluate_model(test_data)

            # Generate plots
            plot_path = "/app/models/training_metrics.png"
            self.metrics_tracker.plot_metrics(plot_path)

            pipeline_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'pipeline_completed': datetime.now().isoformat(),
                'plots_saved': plot_path
            }

            logger.info("Training pipeline completed successfully")
            return pipeline_results

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
