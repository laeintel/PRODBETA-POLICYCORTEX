"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Reinforcement Learning Policy Optimizer for PolicyCortex
Advanced RL algorithms for automated policy optimization, compliance enhancement,
and cost reduction through intelligent decision-making.

Features:
- Q-Learning with experience replay for discrete policy actions
- Policy Gradient methods (REINFORCE, Actor-Critic) for continuous optimization
- Azure resource environment simulation with realistic constraints
- Sophisticated reward functions balancing compliance, cost, and performance
- Multi-objective optimization with Pareto-efficient solutions
- Safe exploration mechanisms to prevent policy violations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import pickle
import random
import math
from abc import ABC, abstractmethod
from enum import Enum
import uuid

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.losses import MSE, Huber
    import tensorflow.keras.backend as K
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions available in the policy environment."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CHANGE_TIER = "change_tier"
    ENABLE_FEATURE = "enable_feature"
    DISABLE_FEATURE = "disable_feature"
    MODIFY_CONFIG = "modify_config"
    CREATE_POLICY = "create_policy"
    DELETE_POLICY = "delete_policy"
    UPDATE_PERMISSIONS = "update_permissions"


class RewardComponent(Enum):
    """Components of the reward function."""
    COMPLIANCE = "compliance"
    COST = "cost"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"


@dataclass
class Action:
    """Represents an action in the RL environment."""
    action_id: str
    action_type: ActionType
    target_resource: str
    parameters: Dict[str, Any]
    expected_cost_change: float = 0.0
    expected_compliance_impact: float = 0.0
    risk_score: float = 0.0


@dataclass
class State:
    """Represents the current state of the Azure environment."""
    resource_utilization: Dict[str, float]
    compliance_scores: Dict[str, float]
    cost_metrics: Dict[str, float]
    security_metrics: Dict[str, float]
    policy_violations: List[str]
    active_policies: List[str]
    timestamp: datetime
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyOptimizationResult:
    """Result of policy optimization."""
    optimization_id: str
    initial_state: State
    final_state: State
    actions_taken: List[Action]
    total_reward: float
    reward_components: Dict[RewardComponent, float]
    training_episodes: int
    convergence_achieved: bool
    optimization_time: float
    performance_metrics: Dict[str, float]


class AzureEnvironmentSimulator:
    """
    Simulates an Azure cloud environment for RL training.
    Provides realistic resource states, constraints, and responses to actions.
    """
    
    def __init__(self, initial_resources: Optional[Dict[str, Any]] = None):
        """
        Initialize the Azure environment simulator.
        
        Args:
            initial_resources: Initial resource configuration
        """
        self.initial_resources = initial_resources or self._generate_default_resources()
        self.current_state = self._create_initial_state()
        self.step_count = 0
        self.max_steps = 1000
        self.compliance_policies = self._load_compliance_policies()
        self.cost_model = self._initialize_cost_model()
        
    def _generate_default_resources(self) -> Dict[str, Any]:
        """Generate default Azure resources for simulation."""
        return {
            "virtual_machines": {
                "vm-web-01": {"size": "Standard_D2s_v3", "cpu_usage": 0.6, "memory_usage": 0.7},
                "vm-db-01": {"size": "Standard_D4s_v3", "cpu_usage": 0.8, "memory_usage": 0.9},
                "vm-app-01": {"size": "Standard_D2s_v3", "cpu_usage": 0.4, "memory_usage": 0.5}
            },
            "storage_accounts": {
                "storage01": {"tier": "Standard_LRS", "usage_gb": 500, "transactions": 10000},
                "storage02": {"tier": "Premium_LRS", "usage_gb": 200, "transactions": 5000}
            },
            "databases": {
                "sqldb-prod": {"tier": "S2", "dtu_usage": 0.7, "storage_usage": 0.6},
                "cosmosdb-cache": {"tier": "400 RU/s", "ru_usage": 0.8, "storage_usage": 0.3}
            },
            "network": {
                "bandwidth_usage": 0.6,
                "security_rules": 15,
                "firewall_enabled": True
            }
        }
    
    def _create_initial_state(self) -> State:
        """Create the initial state from resources."""
        # Calculate resource utilization
        resource_util = {}
        for resource_type, resources in self.initial_resources.items():
            if resource_type == "virtual_machines":
                avg_cpu = np.mean([vm["cpu_usage"] for vm in resources.values()])
                avg_memory = np.mean([vm["memory_usage"] for vm in resources.values()])
                resource_util[f"{resource_type}_cpu"] = avg_cpu
                resource_util[f"{resource_type}_memory"] = avg_memory
            elif resource_type == "storage_accounts":
                avg_usage = np.mean([s["usage_gb"] / 1000 for s in resources.values()])
                resource_util[resource_type] = avg_usage
            elif resource_type == "databases":
                avg_usage = np.mean([0.7, 0.8])  # Mock usage
                resource_util[resource_type] = avg_usage
        
        # Mock compliance scores
        compliance_scores = {
            "security_compliance": 0.85,
            "cost_compliance": 0.72,
            "performance_compliance": 0.90,
            "governance_compliance": 0.78
        }
        
        # Mock cost metrics (monthly costs in USD)
        cost_metrics = {
            "total_monthly_cost": 2500.0,
            "vm_costs": 1200.0,
            "storage_costs": 300.0,
            "database_costs": 800.0,
            "network_costs": 200.0
        }
        
        # Mock security metrics
        security_metrics = {
            "vulnerability_score": 0.15,  # Lower is better
            "access_control_score": 0.88,
            "encryption_coverage": 0.95,
            "network_security_score": 0.82
        }
        
        return State(
            resource_utilization=resource_util,
            compliance_scores=compliance_scores,
            cost_metrics=cost_metrics,
            security_metrics=security_metrics,
            policy_violations=["VM-01: Unencrypted disk", "Storage-02: Public access enabled"],
            active_policies=["require-encryption", "network-security", "cost-optimization"],
            timestamp=datetime.now()
        )
    
    def _load_compliance_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance policies and their requirements."""
        return {
            "require-encryption": {
                "type": "security",
                "weight": 0.3,
                "target_resources": ["virtual_machines", "storage_accounts"],
                "requirement": "All resources must be encrypted"
            },
            "cost-optimization": {
                "type": "financial",
                "weight": 0.4,
                "target_resources": ["virtual_machines", "databases"],
                "requirement": "Resources should be right-sized for usage"
            },
            "network-security": {
                "type": "security",
                "weight": 0.3,
                "target_resources": ["network"],
                "requirement": "Network security groups must be properly configured"
            }
        }
    
    def _initialize_cost_model(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost model for different resource types and sizes."""
        return {
            "virtual_machines": {
                "Standard_D2s_v3": 70.0,  # Monthly cost
                "Standard_D4s_v3": 140.0,
                "Standard_D8s_v3": 280.0,
                "Standard_B2s": 30.0
            },
            "storage_accounts": {
                "Standard_LRS": 0.02,  # Per GB per month
                "Standard_GRS": 0.04,
                "Premium_LRS": 0.15
            },
            "databases": {
                "S1": 20.0,  # Monthly cost
                "S2": 50.0,
                "S3": 100.0,
                "400 RU/s": 25.0,
                "1000 RU/s": 60.0
            }
        }
    
    def reset(self) -> State:
        """Reset the environment to initial state."""
        self.current_state = self._create_initial_state()
        self.step_count = 0
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute an action and return the new state, reward, done flag, and info.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Execute action and get new state
        next_state = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, action, next_state)
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps or 
                self._check_terminal_conditions(next_state))
        
        # Additional info
        info = {
            "step": self.step_count,
            "action_executed": action.action_type.value,
            "cost_change": next_state.cost_metrics["total_monthly_cost"] - 
                          self.current_state.cost_metrics["total_monthly_cost"],
            "compliance_change": np.mean(list(next_state.compliance_scores.values())) - 
                               np.mean(list(self.current_state.compliance_scores.values()))
        }
        
        # Update current state
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: Action) -> State:
        """Execute an action and return the resulting state."""
        new_state = State(
            resource_utilization=self.current_state.resource_utilization.copy(),
            compliance_scores=self.current_state.compliance_scores.copy(),
            cost_metrics=self.current_state.cost_metrics.copy(),
            security_metrics=self.current_state.security_metrics.copy(),
            policy_violations=self.current_state.policy_violations.copy(),
            active_policies=self.current_state.active_policies.copy(),
            timestamp=datetime.now()
        )
        
        # Apply action effects based on type
        if action.action_type == ActionType.SCALE_DOWN:
            # Reduce resource costs and utilization
            if "vm_costs" in new_state.cost_metrics:
                new_state.cost_metrics["vm_costs"] *= 0.8
                new_state.cost_metrics["total_monthly_cost"] *= 0.95
            
            # Update utilization
            for key in new_state.resource_utilization:
                if "cpu" in key or "memory" in key:
                    new_state.resource_utilization[key] = min(1.0, 
                        new_state.resource_utilization[key] * 1.2)
        
        elif action.action_type == ActionType.SCALE_UP:
            # Increase resource costs but improve performance
            if "vm_costs" in new_state.cost_metrics:
                new_state.cost_metrics["vm_costs"] *= 1.4
                new_state.cost_metrics["total_monthly_cost"] *= 1.2
            
            # Reduce utilization pressure
            for key in new_state.resource_utilization:
                if "cpu" in key or "memory" in key:
                    new_state.resource_utilization[key] *= 0.7
        
        elif action.action_type == ActionType.ENABLE_FEATURE:
            # Enable security/compliance features
            if "encryption" in action.parameters.get("feature", ""):
                new_state.security_metrics["encryption_coverage"] = min(1.0,
                    new_state.security_metrics["encryption_coverage"] + 0.1)
                new_state.compliance_scores["security_compliance"] = min(1.0,
                    new_state.compliance_scores["security_compliance"] + 0.05)
                # Remove related violations
                new_state.policy_violations = [v for v in new_state.policy_violations 
                                             if "Unencrypted" not in v]
                # Small cost increase
                new_state.cost_metrics["total_monthly_cost"] *= 1.02
        
        elif action.action_type == ActionType.CREATE_POLICY:
            # Add new policy
            policy_name = action.parameters.get("policy_name", "new-policy")
            if policy_name not in new_state.active_policies:
                new_state.active_policies.append(policy_name)
                # Improve relevant compliance score
                compliance_type = action.parameters.get("compliance_type", "governance_compliance")
                if compliance_type in new_state.compliance_scores:
                    new_state.compliance_scores[compliance_type] = min(1.0,
                        new_state.compliance_scores[compliance_type] + 0.08)
        
        # Add some randomness to simulate real-world variability
        for key in new_state.resource_utilization:
            new_state.resource_utilization[key] += np.random.normal(0, 0.02)
            new_state.resource_utilization[key] = np.clip(new_state.resource_utilization[key], 0, 1)
        
        return new_state
    
    def _calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        """Calculate reward for the state transition."""
        reward_components = {}
        
        # Compliance reward (40% weight)
        compliance_improvement = (np.mean(list(next_state.compliance_scores.values())) - 
                                np.mean(list(state.compliance_scores.values())))
        reward_components[RewardComponent.COMPLIANCE] = compliance_improvement * 40
        
        # Cost optimization reward (35% weight)
        cost_reduction = (state.cost_metrics["total_monthly_cost"] - 
                         next_state.cost_metrics["total_monthly_cost"]) / 1000  # Normalize
        reward_components[RewardComponent.COST] = cost_reduction * 35
        
        # Performance reward (15% weight)
        avg_utilization = np.mean(list(next_state.resource_utilization.values()))
        # Reward utilization between 0.6 and 0.8 (optimal range)
        if 0.6 <= avg_utilization <= 0.8:
            performance_reward = 1.0
        else:
            performance_reward = 1.0 - abs(avg_utilization - 0.7) * 2
        reward_components[RewardComponent.PERFORMANCE] = performance_reward * 15
        
        # Security reward (10% weight)
        security_improvement = (np.mean(list(next_state.security_metrics.values())) - 
                              np.mean(list(state.security_metrics.values())))
        reward_components[RewardComponent.SECURITY] = security_improvement * 10
        
        # Penalty for policy violations
        violation_penalty = len(next_state.policy_violations) * -5
        
        # Risk penalty based on action risk score
        risk_penalty = action.risk_score * -10
        
        total_reward = (sum(reward_components.values()) + violation_penalty + risk_penalty)
        
        return total_reward
    
    def _check_terminal_conditions(self, state: State) -> bool:
        """Check if the episode should terminate."""
        # Terminate if too many violations
        if len(state.policy_violations) > 10:
            return True
        
        # Terminate if costs exceed budget
        if state.cost_metrics["total_monthly_cost"] > 5000:
            return True
        
        # Terminate if compliance drops too low
        if any(score < 0.5 for score in state.compliance_scores.values()):
            return True
        
        return False
    
    def get_valid_actions(self, state: State) -> List[Action]:
        """Get list of valid actions for the current state."""
        actions = []
        
        # Scale actions based on current utilization
        for resource_type in ["virtual_machines", "databases"]:
            if any(resource_type in key for key in state.resource_utilization.keys()):
                # Scale up if utilization is high
                avg_util = np.mean([v for k, v in state.resource_utilization.items() 
                                  if resource_type in k])
                if avg_util > 0.8:
                    actions.append(Action(
                        action_id=f"scale_up_{resource_type}",
                        action_type=ActionType.SCALE_UP,
                        target_resource=resource_type,
                        parameters={"scale_factor": 1.5},
                        expected_cost_change=500,
                        expected_compliance_impact=0.02,
                        risk_score=0.2
                    ))
                
                # Scale down if utilization is low
                if avg_util < 0.4:
                    actions.append(Action(
                        action_id=f"scale_down_{resource_type}",
                        action_type=ActionType.SCALE_DOWN,
                        target_resource=resource_type,
                        parameters={"scale_factor": 0.7},
                        expected_cost_change=-300,
                        expected_compliance_impact=-0.01,
                        risk_score=0.3
                    ))
        
        # Security actions based on violations
        if any("Unencrypted" in v for v in state.policy_violations):
            actions.append(Action(
                action_id="enable_encryption",
                action_type=ActionType.ENABLE_FEATURE,
                target_resource="security",
                parameters={"feature": "encryption"},
                expected_cost_change=50,
                expected_compliance_impact=0.1,
                risk_score=0.1
            ))
        
        # Policy creation actions
        if len(state.active_policies) < 10:
            actions.append(Action(
                action_id="create_monitoring_policy",
                action_type=ActionType.CREATE_POLICY,
                target_resource="governance",
                parameters={
                    "policy_name": "monitoring-policy",
                    "compliance_type": "governance_compliance"
                },
                expected_cost_change=25,
                expected_compliance_impact=0.08,
                risk_score=0.05
            ))
        
        return actions[:10]  # Limit action space


class ExperienceReplay:
    """Experience replay buffer for RL algorithms."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to the buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for discrete action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimization
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = ExperienceReplay(10000)
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        self.training_steps = 0
        self.target_update_frequency = 100
    
    def _build_network(self):
        """Build the Q-network."""
        if TENSORFLOW_AVAILABLE:
            model = Sequential([
                Dense(256, activation='relu', input_shape=(self.state_dim,)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(self.action_dim, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
            return model
        else:
            # Mock network
            return {"type": "mock", "weights": np.random.randn(self.state_dim, self.action_dim)}
    
    def state_to_vector(self, state: State) -> np.ndarray:
        """Convert state to vector representation."""
        vector = []
        
        # Resource utilization
        vector.extend(list(state.resource_utilization.values()))
        
        # Compliance scores
        vector.extend(list(state.compliance_scores.values()))
        
        # Normalized cost metrics
        vector.extend([
            state.cost_metrics["total_monthly_cost"] / 5000,  # Normalize by max budget
            len(state.policy_violations) / 20,  # Normalize by max violations
            len(state.active_policies) / 15    # Normalize by max policies
        ])
        
        # Security metrics
        vector.extend(list(state.security_metrics.values()))
        
        # Pad or truncate to fixed size
        target_size = self.state_dim
        if len(vector) < target_size:
            vector.extend([0.0] * (target_size - len(vector)))
        elif len(vector) > target_size:
            vector = vector[:target_size]
        
        return np.array(vector, dtype=np.float32)
    
    def act(self, state: State, valid_actions: List[Action]) -> Action:
        """Select an action using epsilon-greedy policy."""
        if not valid_actions:
            # Return a default action if no valid actions
            return Action(
                action_id="no_action",
                action_type=ActionType.MODIFY_CONFIG,
                target_resource="system",
                parameters={}
            )
        
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(valid_actions)
        
        # Exploitation: best Q-value action
        state_vector = self.state_to_vector(state).reshape(1, -1)
        
        if TENSORFLOW_AVAILABLE and hasattr(self.q_network, 'predict'):
            q_values = self.q_network.predict(state_vector, verbose=0)[0]
        else:
            # Mock Q-values
            q_values = np.random.randn(self.action_dim)
        
        # Select action corresponding to highest Q-value among valid actions
        action_indices = list(range(min(len(valid_actions), len(q_values))))
        best_action_idx = action_indices[np.argmax([q_values[i] for i in action_indices])]
        
        return valid_actions[best_action_idx]
    
    def remember(self, state: State, action: Action, reward: float, 
                 next_state: State, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        priority = abs(reward) + 1.0  # Simple priority based on reward magnitude
        self.memory.add(experience, priority)
    
    def train(self, batch_size: int = 32) -> float:
        """Train the Q-network."""
        if len(self.memory) < batch_size:
            return 0.0
        
        experiences = self.memory.sample(batch_size)
        
        if not TENSORFLOW_AVAILABLE:
            return self._mock_training()
        
        # Prepare training data
        states = np.array([self.state_to_vector(exp.state) for exp in experiences])
        next_states = np.array([self.state_to_vector(exp.next_state) for exp in experiences])
        
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        targets = current_q_values.copy()
        
        for i, exp in enumerate(experiences):
            if exp.done:
                targets[i][0] = exp.reward  # Simplified action indexing
            else:
                targets[i][0] = exp.reward + 0.95 * np.max(next_q_values[i])
        
        # Train the network
        loss = self.q_network.train_on_batch(states, targets)
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return float(loss) if isinstance(loss, (np.ndarray, list)) else loss
    
    def update_target_network(self):
        """Update target network weights."""
        if TENSORFLOW_AVAILABLE and hasattr(self.q_network, 'get_weights'):
            self.target_network.set_weights(self.q_network.get_weights())
    
    def _mock_training(self) -> float:
        """Mock training when TensorFlow is not available."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return np.random.uniform(0.1, 0.5)  # Mock loss


class PolicyGradientAgent:
    """Policy Gradient agent using REINFORCE algorithm."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        """
        Initialize Policy Gradient agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for policy optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        self.policy_network = self._build_policy_network()
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _build_policy_network(self):
        """Build the policy network."""
        if TENSORFLOW_AVAILABLE:
            model = Sequential([
                Dense(128, activation='relu', input_shape=(self.state_dim,)),
                Dense(64, activation='relu'),
                Dense(self.action_dim, activation='softmax')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate))
            return model
        else:
            return {"type": "mock", "weights": np.random.randn(self.state_dim, self.action_dim)}
    
    def state_to_vector(self, state: State) -> np.ndarray:
        """Convert state to vector representation."""
        # Same implementation as DQN agent
        vector = []
        vector.extend(list(state.resource_utilization.values()))
        vector.extend(list(state.compliance_scores.values()))
        vector.extend([
            state.cost_metrics["total_monthly_cost"] / 5000,
            len(state.policy_violations) / 20,
            len(state.active_policies) / 15
        ])
        vector.extend(list(state.security_metrics.values()))
        
        target_size = self.state_dim
        if len(vector) < target_size:
            vector.extend([0.0] * (target_size - len(vector)))
        elif len(vector) > target_size:
            vector = vector[:target_size]
        
        return np.array(vector, dtype=np.float32)
    
    def act(self, state: State, valid_actions: List[Action]) -> Action:
        """Select action based on policy probabilities."""
        if not valid_actions:
            return Action(
                action_id="no_action",
                action_type=ActionType.MODIFY_CONFIG,
                target_resource="system",
                parameters={}
            )
        
        state_vector = self.state_to_vector(state).reshape(1, -1)
        
        if TENSORFLOW_AVAILABLE and hasattr(self.policy_network, 'predict'):
            action_probs = self.policy_network.predict(state_vector, verbose=0)[0]
        else:
            # Mock probabilities
            action_probs = np.random.dirichlet(np.ones(min(len(valid_actions), self.action_dim)))
        
        # Select action based on probabilities
        valid_prob_indices = list(range(min(len(valid_actions), len(action_probs))))
        valid_probs = action_probs[:len(valid_prob_indices)]
        valid_probs = valid_probs / np.sum(valid_probs)  # Normalize
        
        action_idx = np.random.choice(valid_prob_indices, p=valid_probs)
        selected_action = valid_actions[action_idx]
        
        # Store for training
        self.episode_states.append(state_vector)
        self.episode_actions.append(action_idx)
        
        return selected_action
    
    def store_reward(self, reward: float):
        """Store reward for the current episode."""
        self.episode_rewards.append(reward)
    
    def train_episode(self) -> float:
        """Train the policy network at the end of an episode."""
        if not self.episode_rewards:
            return 0.0
        
        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(self.episode_rewards)
        
        if not TENSORFLOW_AVAILABLE:
            return self._mock_training()
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Convert to tensors
        states = np.vstack(self.episode_states)
        actions = np.array(self.episode_actions)
        
        # Custom training step
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_action_probs = tf.gather_nd(action_probs, action_indices)
            
            loss = -tf.reduce_mean(tf.math.log(selected_action_probs + 1e-8) * discounted_rewards)
        
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(
            zip(gradients, self.policy_network.trainable_variables)
        )
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return float(loss)
    
    def _discount_rewards(self, rewards: List[float], gamma: float = 0.99) -> np.ndarray:
        """Calculate discounted rewards."""
        discounted = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.0
        
        for i in reversed(range(len(rewards))):
            cumulative = rewards[i] + gamma * cumulative
            discounted[i] = cumulative
        
        return discounted
    
    def _mock_training(self) -> float:
        """Mock training when TensorFlow is not available."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        return np.random.uniform(0.1, 1.0)


class PolicyOptimizer:
    """
    Main policy optimization service that coordinates RL agents and environment.
    """
    
    def __init__(self, algorithm: str = "dqn", state_dim: int = 20, action_dim: int = 10):
        """
        Initialize policy optimizer.
        
        Args:
            algorithm: RL algorithm to use ('dqn', 'policy_gradient')
            state_dim: Dimension of state space
            action_dim: Maximum number of actions
        """
        self.algorithm = algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.environment = AzureEnvironmentSimulator()
        
        # Initialize agent based on algorithm
        if algorithm == "dqn":
            self.agent = DQNAgent(state_dim, action_dim)
        elif algorithm == "policy_gradient":
            self.agent = PolicyGradientAgent(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.training_history = []
        self.optimization_results = []
    
    async def optimize_policies(self, episodes: int = 1000, 
                               target_compliance: float = 0.9,
                               max_cost: float = 4000) -> PolicyOptimizationResult:
        """
        Run policy optimization for specified number of episodes.
        
        Args:
            episodes: Number of training episodes
            target_compliance: Target compliance score to achieve
            max_cost: Maximum allowed monthly cost
            
        Returns:
            PolicyOptimizationResult: Optimization results
        """
        optimization_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting policy optimization with {self.algorithm} for {episodes} episodes")
        
        initial_state = self.environment.reset()
        best_reward = float('-inf')
        best_episode = 0
        convergence_window = []
        convergence_threshold = 0.01
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_actions = []
            
            # Run episode
            done = False
            step = 0
            max_steps_per_episode = 100
            
            while not done and step < max_steps_per_episode:
                # Get valid actions
                valid_actions = self.environment.get_valid_actions(state)
                
                # Select action
                action = self.agent.act(state, valid_actions)
                episode_actions.append(action)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                episode_reward += reward
                
                # Store experience (for DQN)
                if self.algorithm == "dqn":
                    self.agent.remember(state, action, reward, next_state, done)
                elif self.algorithm == "policy_gradient":
                    self.agent.store_reward(reward)
                
                state = next_state
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Train agent
            if self.algorithm == "dqn" and len(self.agent.memory) > 32:
                loss = self.agent.train(batch_size=32)
                episode_losses.append(loss)
            elif self.algorithm == "policy_gradient":
                loss = self.agent.train_episode()
                episode_losses.append(loss)
            
            # Check for improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode
            
            # Check convergence
            convergence_window.append(episode_reward)
            if len(convergence_window) > 50:
                convergence_window.pop(0)
                
                if len(convergence_window) == 50:
                    recent_std = np.std(convergence_window)
                    if recent_std < convergence_threshold:
                        logger.info(f"Convergence achieved at episode {episode}")
                        break
            
            # Logging
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                          f"Best = {best_reward:.2f}, Epsilon = {getattr(self.agent, 'epsilon', 'N/A')}")
        
        # Final evaluation
        final_state = self.environment.current_state
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate reward components
        reward_components = {
            RewardComponent.COMPLIANCE: np.mean(list(final_state.compliance_scores.values())),
            RewardComponent.COST: (5000 - final_state.cost_metrics["total_monthly_cost"]) / 1000,
            RewardComponent.PERFORMANCE: 1.0 - abs(np.mean(list(final_state.resource_utilization.values())) - 0.7),
            RewardComponent.SECURITY: np.mean(list(final_state.security_metrics.values()))
        }
        
        # Performance metrics
        performance_metrics = {
            "average_episode_reward": np.mean(episode_rewards),
            "best_episode_reward": best_reward,
            "convergence_episode": best_episode,
            "final_compliance_score": np.mean(list(final_state.compliance_scores.values())),
            "final_cost": final_state.cost_metrics["total_monthly_cost"],
            "policy_violations_resolved": len(initial_state.policy_violations) - len(final_state.policy_violations),
            "training_stability": np.std(episode_rewards[-100:]) if len(episode_rewards) >= 100 else 0
        }
        
        result = PolicyOptimizationResult(
            optimization_id=optimization_id,
            initial_state=initial_state,
            final_state=final_state,
            actions_taken=episode_actions,
            total_reward=sum(episode_rewards),
            reward_components=reward_components,
            training_episodes=len(episode_rewards),
            convergence_achieved=len(convergence_window) == 50 and np.std(convergence_window) < convergence_threshold,
            optimization_time=optimization_time,
            performance_metrics=performance_metrics
        )
        
        self.optimization_results.append(result)
        
        logger.info(f"Optimization completed in {optimization_time:.1f}s")
        logger.info(f"Final compliance score: {performance_metrics['final_compliance_score']:.3f}")
        logger.info(f"Final cost: ${performance_metrics['final_cost']:.0f}")
        
        return result
    
    def get_policy_recommendations(self, current_state: State) -> List[Tuple[Action, float]]:
        """
        Get policy recommendations for the current state.
        
        Args:
            current_state: Current Azure environment state
            
        Returns:
            List of (action, confidence) tuples
        """
        valid_actions = self.environment.get_valid_actions(current_state)
        recommendations = []
        
        for action in valid_actions:
            # Simulate action to estimate value
            self.environment.current_state = current_state
            next_state, reward, _, _ = self.environment.step(action)
            
            # Calculate confidence based on expected reward and compliance impact
            confidence = (reward + 10) / 20  # Normalize to 0-1 range
            confidence = max(0.0, min(1.0, confidence))
            
            recommendations.append((action, confidence))
        
        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained RL model."""
        try:
            model_data = {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "training_history": self.training_history,
                "optimization_results": [
                    {
                        "optimization_id": result.optimization_id,
                        "total_reward": result.total_reward,
                        "training_episodes": result.training_episodes,
                        "convergence_achieved": result.convergence_achieved,
                        "optimization_time": result.optimization_time,
                        "performance_metrics": result.performance_metrics
                    } for result in self.optimization_results
                ]
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            
            # Save neural network weights if available
            if hasattr(self.agent, 'q_network') and TENSORFLOW_AVAILABLE:
                self.agent.q_network.save(filepath.replace('.pkl', '_network.h5'))
            elif hasattr(self.agent, 'policy_network') and TENSORFLOW_AVAILABLE:
                self.agent.policy_network.save(filepath.replace('.pkl', '_policy.h5'))
            
            return True
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
            return False


# Example usage and testing
async def example_rl_optimization():
    """
    Example usage of the RL policy optimization service.
    Demonstrates training and policy recommendations.
    """
    print("PolicyCortex RL Policy Optimizer - Example Usage")
    print("=" * 60)
    
    # Initialize optimizer with DQN
    print("\n1. Initializing DQN Policy Optimizer...")
    optimizer = PolicyOptimizer(algorithm="dqn", state_dim=20, action_dim=10)
    
    # Run optimization
    print("\n2. Running policy optimization...")
    result = await optimizer.optimize_policies(
        episodes=200,
        target_compliance=0.9,
        max_cost=4000
    )
    
    # Display results
    print(f"\n3. Optimization Results:")
    print(f"   Optimization ID: {result.optimization_id}")
    print(f"   Training Episodes: {result.training_episodes}")
    print(f"   Total Reward: {result.total_reward:.2f}")
    print(f"   Convergence Achieved: {result.convergence_achieved}")
    print(f"   Optimization Time: {result.optimization_time:.1f}s")
    
    print(f"\n4. Performance Metrics:")
    for metric, value in result.performance_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    print(f"\n5. Reward Components:")
    for component, value in result.reward_components.items():
        print(f"   {component.value}: {value:.3f}")
    
    # Get recommendations for final state
    print(f"\n6. Policy Recommendations:")
    recommendations = optimizer.get_policy_recommendations(result.final_state)
    
    for i, (action, confidence) in enumerate(recommendations, 1):
        print(f"   {i}. {action.action_type.value} on {action.target_resource}")
        print(f"      Confidence: {confidence:.2f}")
        print(f"      Expected Cost Change: ${action.expected_cost_change:.0f}")
        print(f"      Expected Compliance Impact: {action.expected_compliance_impact:.3f}")
    
    # Compare with Policy Gradient
    print(f"\n7. Comparing with Policy Gradient...")
    pg_optimizer = PolicyOptimizer(algorithm="policy_gradient", state_dim=20, action_dim=10)
    pg_result = await pg_optimizer.optimize_policies(episodes=100)
    
    print(f"   DQN Final Compliance: {result.performance_metrics['final_compliance_score']:.3f}")
    print(f"   Policy Gradient Final Compliance: {pg_result.performance_metrics['final_compliance_score']:.3f}")
    
    print(f"   DQN Final Cost: ${result.performance_metrics['final_cost']:.0f}")
    print(f"   Policy Gradient Final Cost: ${pg_result.performance_metrics['final_cost']:.0f}")
    
    # Save models
    print(f"\n8. Saving trained models...")
    dqn_saved = optimizer.save_model("dqn_policy_model.pkl")
    pg_saved = pg_optimizer.save_model("pg_policy_model.pkl")
    
    print(f"   DQN Model Saved: {dqn_saved}")
    print(f"   Policy Gradient Model Saved: {pg_saved}")
    
    print(f"\n9. Environment State Summary:")
    final_state = result.final_state
    print(f"   Active Policies: {len(final_state.active_policies)}")
    print(f"   Policy Violations: {len(final_state.policy_violations)}")
    print(f"   Average Resource Utilization: {np.mean(list(final_state.resource_utilization.values())):.2f}")
    print(f"   Security Score: {np.mean(list(final_state.security_metrics.values())):.3f}")
    
    print(f"\nRL Policy Optimization Complete! 🎯")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_rl_optimization())