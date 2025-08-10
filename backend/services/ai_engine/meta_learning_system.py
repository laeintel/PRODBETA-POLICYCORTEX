"""
Meta-Learning System with MAML and Few-Shot Learning
Enables rapid adaptation to new cloud services, compliance frameworks, and organizational needs
Implements Model-Agnostic Meta-Learning (MAML) and Prototypical Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import copy
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a meta-learning task"""
    name: str
    domain: str  # 'azure', 'aws', 'gcp', 'compliance', 'security'
    support_set: List[Tuple[torch.Tensor, torch.Tensor]]  # (input, target) pairs
    query_set: List[Tuple[torch.Tensor, torch.Tensor]]
    metadata: Dict[str, Any]
    difficulty: float = 1.0  # Task difficulty for curriculum learning


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML)
    Learns initialization that can quickly adapt to new tasks with few examples
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01, 
                 outer_lr: float = 0.001, num_inner_steps: int = 5):
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=outer_lr)
        
        # Track meta-learning progress
        self.meta_stats = {
            'tasks_seen': 0,
            'average_adaptation_loss': 0.0,
            'average_query_accuracy': 0.0,
            'domain_performance': defaultdict(float)
        }
        
    def inner_loop(self, task: Task, train: bool = True) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Inner loop: Adapt to specific task using support set
        
        Returns adapted model and metrics
        """
        # Clone model for task-specific adaptation
        adapted_model = self._clone_model()
        
        # Task-specific optimizer
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        metrics = {'support_losses': [], 'query_losses': []}
        
        # Adapt on support set
        for step in range(self.num_inner_steps):
            support_loss = 0
            
            for x_support, y_support in task.support_set:
                # Forward pass
                y_pred = adapted_model(x_support)
                
                # Task-specific loss (can be customized per domain)
                if task.domain == 'compliance':
                    loss = F.binary_cross_entropy_with_logits(y_pred, y_support)
                else:
                    loss = F.mse_loss(y_pred, y_support)
                
                support_loss += loss
                
            support_loss /= len(task.support_set)
            metrics['support_losses'].append(support_loss.item())
            
            if train:
                # Update adapted model
                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()
        
        # Evaluate on query set
        query_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad() if not train else torch.enable_grad():
            for x_query, y_query in task.query_set:
                y_pred = adapted_model(x_query)
                
                if task.domain == 'compliance':
                    loss = F.binary_cross_entropy_with_logits(y_pred, y_query)
                    predictions = (torch.sigmoid(y_pred) > 0.5).float()
                    correct += (predictions == y_query).sum().item()
                    total += y_query.numel()
                else:
                    loss = F.mse_loss(y_pred, y_query)
                    
                query_loss += loss
                
        query_loss /= len(task.query_set)
        metrics['query_losses'].append(query_loss.item())
        
        if total > 0:
            metrics['accuracy'] = correct / total
        
        return adapted_model, metrics
    
    def outer_loop(self, tasks: List[Task], num_iterations: int = 100):
        """
        Outer loop: Update meta-parameters based on performance across tasks
        """
        logger.info(f"Starting meta-training on {len(tasks)} tasks")
        
        for iteration in range(num_iterations):
            meta_loss = 0
            iteration_metrics = defaultdict(list)
            
            # Sample batch of tasks
            task_batch = np.random.choice(tasks, min(4, len(tasks)), replace=False)
            
            for task in task_batch:
                # Inner loop adaptation
                adapted_model, metrics = self.inner_loop(task, train=True)
                
                # Compute meta-loss on query set
                query_loss = 0
                for x_query, y_query in task.query_set:
                    y_pred = adapted_model(x_query)
                    
                    if task.domain == 'compliance':
                        loss = F.binary_cross_entropy_with_logits(y_pred, y_query)
                    else:
                        loss = F.mse_loss(y_pred, y_query)
                        
                    query_loss += loss
                    
                query_loss /= len(task.query_set)
                meta_loss += query_loss
                
                # Track metrics
                iteration_metrics[task.domain].append(metrics)
                self.meta_stats['tasks_seen'] += 1
                
            # Meta-update
            meta_loss /= len(task_batch)
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
            self.meta_optimizer.step()
            
            # Update statistics
            self._update_stats(iteration_metrics)
            
            if iteration % 10 == 0:
                logger.info(f"Meta-iteration {iteration}: Loss={meta_loss.item():.4f}, "
                          f"Tasks seen={self.meta_stats['tasks_seen']}")
                
    def adapt_to_new_task(self, task: Task) -> nn.Module:
        """
        Adapt to a completely new task using learned meta-parameters
        """
        logger.info(f"Adapting to new task: {task.name}")
        
        adapted_model, metrics = self.inner_loop(task, train=True)
        
        logger.info(f"Adaptation complete: Query loss={metrics['query_losses'][-1]:.4f}")
        
        return adapted_model
    
    def _clone_model(self) -> nn.Module:
        """Create a functional clone of the base model"""
        cloned_model = copy.deepcopy(self.base_model)
        cloned_model.load_state_dict(self.base_model.state_dict())
        return cloned_model
    
    def _update_stats(self, metrics: Dict[str, List]):
        """Update meta-learning statistics"""
        for domain, domain_metrics in metrics.items():
            if domain_metrics:
                avg_query_loss = np.mean([m['query_losses'][-1] for m in domain_metrics])
                self.meta_stats['domain_performance'][domain] = avg_query_loss
                
                if 'accuracy' in domain_metrics[0]:
                    avg_accuracy = np.mean([m['accuracy'] for m in domain_metrics])
                    self.meta_stats['average_query_accuracy'] = avg_accuracy


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning
    Learns to classify based on distance to class prototypes
    Perfect for policy classification and compliance checking
    """
    
    def __init__(self, encoder: nn.Module, embedding_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        
        # Learned distance metric
        self.distance_metric = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # Prototype memory
        self.prototypes = {}
        self.prototype_counts = {}
        
    def compute_prototypes(self, support_set: List[Tuple[torch.Tensor, int]]) -> Dict[int, torch.Tensor]:
        """
        Compute class prototypes from support set
        
        Args:
            support_set: List of (input, label) pairs
        """
        prototypes = defaultdict(list)
        
        for x, y in support_set:
            embedding = self.encoder(x)
            prototypes[y.item() if hasattr(y, 'item') else y].append(embedding)
            
        # Average embeddings to get prototypes
        class_prototypes = {}
        for class_id, embeddings in prototypes.items():
            class_prototypes[class_id] = torch.stack(embeddings).mean(dim=0)
            
        return class_prototypes
    
    def forward(self, query: torch.Tensor, prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Classify query based on distance to prototypes
        """
        query_embedding = self.encoder(query)
        
        distances = []
        class_ids = []
        
        for class_id, prototype in prototypes.items():
            # Compute distance using learned metric
            combined = torch.cat([query_embedding, prototype], dim=-1)
            distance = self.distance_metric(combined)
            distances.append(distance)
            class_ids.append(class_id)
            
        # Convert distances to probabilities
        distances = torch.cat(distances)
        probabilities = F.softmax(-distances, dim=0)  # Negative because closer = better
        
        return probabilities, class_ids
    
    def update_prototypes(self, class_id: Any, new_examples: List[torch.Tensor]):
        """
        Update prototype with new examples (continual learning)
        """
        new_embeddings = [self.encoder(x) for x in new_examples]
        new_prototype = torch.stack(new_embeddings).mean(dim=0)
        
        if class_id in self.prototypes:
            # Weighted average with existing prototype
            count = self.prototype_counts[class_id]
            self.prototypes[class_id] = (
                self.prototypes[class_id] * count + new_prototype * len(new_examples)
            ) / (count + len(new_examples))
            self.prototype_counts[class_id] += len(new_examples)
        else:
            self.prototypes[class_id] = new_prototype
            self.prototype_counts[class_id] = len(new_examples)


class ReptileMetaLearner:
    """
    Reptile: Simplified meta-learning algorithm
    More scalable than MAML for large models
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, num_inner_steps: int = 10):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Store initial parameters
        self.meta_params = OrderedDict(model.named_parameters())
        
    def train_on_task(self, task: Task) -> Dict[str, float]:
        """
        Train on a single task and update meta-parameters
        """
        # Save current parameters
        old_params = OrderedDict((name, param.clone()) 
                                 for name, param in self.model.named_parameters())
        
        # Create task-specific optimizer
        task_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Train on task
        task_losses = []
        for _ in range(self.num_inner_steps):
            loss = 0
            for x, y in task.support_set:
                pred = self.model(x)
                loss += F.mse_loss(pred, y)
            loss /= len(task.support_set)
            
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
            
            task_losses.append(loss.item())
            
        # Reptile update: Move towards task-specific parameters
        for name, param in self.model.named_parameters():
            param.data = old_params[name] + self.outer_lr * (param.data - old_params[name])
            
        return {'task_losses': task_losses, 'final_loss': task_losses[-1]}


class TaskGenerator:
    """
    Generates diverse meta-learning tasks for cloud governance
    """
    
    def __init__(self, feature_dim: int = 768):
        self.feature_dim = feature_dim
        self.task_templates = {
            'policy_classification': self._generate_policy_task,
            'compliance_checking': self._generate_compliance_task,
            'cost_optimization': self._generate_cost_task,
            'security_assessment': self._generate_security_task,
            'service_adaptation': self._generate_service_task
        }
        
    def generate_task_batch(self, batch_size: int = 10, 
                           k_shot: int = 5,
                           num_query: int = 10) -> List[Task]:
        """Generate batch of diverse tasks"""
        tasks = []
        
        for _ in range(batch_size):
            task_type = np.random.choice(list(self.task_templates.keys()))
            task = self.task_templates[task_type](k_shot, num_query)
            tasks.append(task)
            
        return tasks
    
    def _generate_policy_task(self, k_shot: int, num_query: int) -> Task:
        """Generate policy classification task"""
        # Simulate different policy types
        policy_types = ['security', 'compliance', 'cost', 'performance', 'reliability']
        selected_types = np.random.choice(policy_types, 3, replace=False)
        
        support_set = []
        query_set = []
        
        # Generate support examples
        for _ in range(k_shot):
            for i, policy_type in enumerate(selected_types):
                x = torch.randn(1, self.feature_dim)
                y = torch.tensor([float(i)])
                support_set.append((x, y))
                
        # Generate query examples
        for _ in range(num_query):
            i = np.random.randint(len(selected_types))
            x = torch.randn(1, self.feature_dim)
            y = torch.tensor([float(i)])
            query_set.append((x, y))
            
        return Task(
            name=f"policy_classification_{'-'.join(selected_types)}",
            domain='policy',
            support_set=support_set,
            query_set=query_set,
            metadata={'policy_types': selected_types}
        )
    
    def _generate_compliance_task(self, k_shot: int, num_query: int) -> Task:
        """Generate compliance checking task"""
        frameworks = ['hipaa', 'gdpr', 'sox', 'pci-dss', 'iso27001']
        framework = np.random.choice(frameworks)
        
        support_set = []
        query_set = []
        
        # Binary classification: compliant or not
        for _ in range(k_shot):
            # Compliant example
            x = torch.randn(1, self.feature_dim)
            y = torch.tensor([1.0])
            support_set.append((x, y))
            
            # Non-compliant example
            x = torch.randn(1, self.feature_dim)
            y = torch.tensor([0.0])
            support_set.append((x, y))
            
        for _ in range(num_query):
            x = torch.randn(1, self.feature_dim)
            y = torch.tensor([float(np.random.randint(2))])
            query_set.append((x, y))
            
        return Task(
            name=f"compliance_{framework}",
            domain='compliance',
            support_set=support_set,
            query_set=query_set,
            metadata={'framework': framework}
        )
    
    def _generate_cost_task(self, k_shot: int, num_query: int) -> Task:
        """Generate cost optimization task"""
        # Regression task: predict cost savings
        support_set = []
        query_set = []
        
        for _ in range(k_shot):
            x = torch.randn(1, self.feature_dim)
            # Simulate cost savings percentage (0-50%)
            y = torch.tensor([np.random.uniform(0, 0.5)])
            support_set.append((x, y))
            
        for _ in range(num_query):
            x = torch.randn(1, self.feature_dim)
            y = torch.tensor([np.random.uniform(0, 0.5)])
            query_set.append((x, y))
            
        return Task(
            name="cost_optimization",
            domain='cost',
            support_set=support_set,
            query_set=query_set,
            metadata={'optimization_type': 'percentage_savings'}
        )
    
    def _generate_security_task(self, k_shot: int, num_query: int) -> Task:
        """Generate security assessment task"""
        threat_levels = ['low', 'medium', 'high', 'critical']
        
        support_set = []
        query_set = []
        
        for _ in range(k_shot):
            for i, level in enumerate(threat_levels):
                x = torch.randn(1, self.feature_dim)
                y = torch.tensor([float(i) / len(threat_levels)])
                support_set.append((x, y))
                
        for _ in range(num_query):
            x = torch.randn(1, self.feature_dim)
            level = np.random.randint(len(threat_levels))
            y = torch.tensor([float(level) / len(threat_levels)])
            query_set.append((x, y))
            
        return Task(
            name="security_assessment",
            domain='security',
            support_set=support_set,
            query_set=query_set,
            metadata={'threat_levels': threat_levels}
        )
    
    def _generate_service_task(self, k_shot: int, num_query: int) -> Task:
        """Generate cloud service adaptation task"""
        services = ['compute', 'storage', 'network', 'database', 'ai']
        service = np.random.choice(services)
        
        support_set = []
        query_set = []
        
        # Task: predict optimal configuration
        for _ in range(k_shot):
            x = torch.randn(1, self.feature_dim)
            # Configuration vector (simplified)
            y = torch.randn(1, 10)  # 10-dim config space
            support_set.append((x, y))
            
        for _ in range(num_query):
            x = torch.randn(1, self.feature_dim)
            y = torch.randn(1, 10)
            query_set.append((x, y))
            
        return Task(
            name=f"service_adaptation_{service}",
            domain='azure',
            support_set=support_set,
            query_set=query_set,
            metadata={'service': service}
        )


class MetaLearningOrchestrator:
    """
    Orchestrates different meta-learning algorithms
    Selects best approach based on task characteristics
    """
    
    def __init__(self, base_model: nn.Module, feature_dim: int = 768):
        self.feature_dim = feature_dim
        
        # Initialize different meta-learners
        self.maml = MAML(base_model)
        self.reptile = ReptileMetaLearner(copy.deepcopy(base_model))
        self.prototypical = PrototypicalNetwork(
            encoder=copy.deepcopy(base_model),
            embedding_dim=feature_dim
        )
        
        # Task generator
        self.task_generator = TaskGenerator(feature_dim)
        
        # Performance tracking
        self.algorithm_performance = {
            'maml': [],
            'reptile': [],
            'prototypical': []
        }
        
    def train_all(self, num_iterations: int = 100):
        """Train all meta-learning algorithms"""
        logger.info("Training all meta-learning algorithms")
        
        # Generate training tasks
        tasks = self.task_generator.generate_task_batch(
            batch_size=num_iterations,
            k_shot=5,
            num_query=10
        )
        
        # Train MAML
        self.maml.outer_loop(tasks[:num_iterations//3], num_iterations//3)
        
        # Train Reptile
        for task in tasks[num_iterations//3:2*num_iterations//3]:
            metrics = self.reptile.train_on_task(task)
            self.algorithm_performance['reptile'].append(metrics['final_loss'])
            
        # Train Prototypical Network
        for task in tasks[2*num_iterations//3:]:
            prototypes = self.prototypical.compute_prototypes(task.support_set)
            
            correct = 0
            total = 0
            for x, y in task.query_set:
                probs, class_ids = self.prototypical(x, prototypes)
                pred_class = class_ids[torch.argmax(probs)]
                if pred_class == y:
                    correct += 1
                total += 1
                    
            accuracy = correct / total if total > 0 else 0
            self.algorithm_performance['prototypical'].append(1 - accuracy)
            
        logger.info("Meta-learning training complete")
        
    def select_best_algorithm(self, task: Task) -> str:
        """Select best meta-learning algorithm for given task"""
        # Simple heuristic based on task characteristics
        if task.domain == 'compliance':
            return 'prototypical'  # Good for classification
        elif len(task.support_set) < 5:
            return 'maml'  # Good for very few examples
        else:
            return 'reptile'  # More scalable
            
    def adapt_to_task(self, task: Task) -> nn.Module:
        """Adapt to new task using best algorithm"""
        algorithm = self.select_best_algorithm(task)
        logger.info(f"Using {algorithm} for task {task.name}")
        
        if algorithm == 'maml':
            return self.maml.adapt_to_new_task(task)
        elif algorithm == 'reptile':
            self.reptile.train_on_task(task)
            return self.reptile.model
        else:
            prototypes = self.prototypical.compute_prototypes(task.support_set)
            # Return a wrapper that uses prototypical network
            class PrototypicalWrapper(nn.Module):
                def __init__(self, proto_net, prototypes):
                    super().__init__()
                    self.proto_net = proto_net
                    self.prototypes = prototypes
                    
                def forward(self, x):
                    probs, _ = self.proto_net(x, self.prototypes)
                    return probs
                    
            return PrototypicalWrapper(self.prototypical, prototypes)


# Global meta-learning system
meta_learning_system = None

def initialize_meta_learning(base_model: nn.Module, feature_dim: int = 768):
    """Initialize the meta-learning system"""
    global meta_learning_system
    meta_learning_system = MetaLearningOrchestrator(base_model, feature_dim)
    logger.info("Meta-learning system initialized")
    return meta_learning_system


# Export main components
__all__ = [
    'MAML',
    'PrototypicalNetwork',
    'ReptileMetaLearner',
    'TaskGenerator',
    'MetaLearningOrchestrator',
    'Task',
    'initialize_meta_learning',
    'meta_learning_system'
]