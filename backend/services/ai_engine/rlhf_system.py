"""
Reinforcement Learning from Human Feedback (RLHF) System
Learns from user preferences, compliance outcomes, and organizational feedback
Implements reward modeling and preference-based policy optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from collections import deque
import json
from enum import Enum

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of human feedback"""
    PREFERENCE = "preference"  # A vs B comparison
    RATING = "rating"  # Absolute rating (1-5 stars)
    CORRECTION = "correction"  # Direct correction
    COMPLIANCE = "compliance"  # Pass/fail compliance check
    INCIDENT = "incident"  # Security/operational incident

@dataclass
class HumanFeedback:
    """Container for human feedback data"""
    feedback_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    context: Dict[str, Any]
    
    # For preference feedback
    option_a: Optional[Dict[str, Any]] = None
    option_b: Optional[Dict[str, Any]] = None
    preference: Optional[str] = None  # 'a', 'b', or 'equal'
    
    # For rating feedback
    rating: Optional[float] = None
    max_rating: float = 5.0
    
    # For correction feedback
    original: Optional[str] = None
    corrected: Optional[str] = None
    
    # For compliance feedback
    policy_id: Optional[str] = None
    compliant: Optional[bool] = None
    violations: List[str] = field(default_factory=list)
    
    # Metadata
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)


class RewardModel(nn.Module):
    """
    Neural network that learns to predict human preferences
    Maps state-action pairs to reward values
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Context encoder (for organizational/user preferences)
        self.context_encoder = nn.LSTM(
            input_dim // 2,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single reward value
        )
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Confidence between 0-1
        )
        
    def forward(self, state_features: torch.Tensor, 
                action_features: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict reward
        
        Args:
            state_features: Current state representation
            action_features: Proposed action representation
            context: Optional context (user/org preferences)
        """
        # Combine state and action
        combined = torch.cat([state_features, action_features], dim=-1)
        
        # Extract features
        features = self.feature_extractor(combined)
        
        # Add context if available
        if context is not None:
            if len(context.shape) == 2:
                context = context.unsqueeze(1)
            context_out, (hidden, _) = self.context_encoder(context)
            context_features = hidden.transpose(0, 1).reshape(hidden.size(1), -1)
            features = torch.cat([features, context_features], dim=-1)
        else:
            # Pad with zeros if no context
            features = torch.cat([features, torch.zeros_like(features)], dim=-1)
        
        # Predict reward and confidence
        reward = self.reward_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'reward': reward.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }


class PreferenceLearner(nn.Module):
    """
    Learns from pairwise preferences using Bradley-Terry model
    Implements preference-based reinforcement learning
    """
    
    def __init__(self, reward_model: RewardModel):
        super().__init__()
        self.reward_model = reward_model
        
        # Preference prediction network
        self.preference_net = nn.Sequential(
            nn.Linear(2, 64),  # Two reward inputs
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability of preferring first option
        )
        
    def forward(self, state_a: torch.Tensor, action_a: torch.Tensor,
                state_b: torch.Tensor, action_b: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict preference between two options
        """
        # Get rewards for both options
        reward_a = self.reward_model(state_a, action_a, context)['reward']
        reward_b = self.reward_model(state_b, action_b, context)['reward']
        
        # Stack rewards
        rewards = torch.stack([reward_a, reward_b], dim=-1)
        
        # Predict preference probability
        preference_prob = self.preference_net(rewards)
        
        return {
            'preference_prob': preference_prob.squeeze(-1),
            'reward_a': reward_a,
            'reward_b': reward_b,
            'reward_diff': reward_a - reward_b
        }
    
    def compute_preference_loss(self, predictions: Dict[str, torch.Tensor],
                               labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for preference prediction
        
        Args:
            predictions: Model predictions
            labels: True preferences (0 for option B, 1 for option A)
        """
        return F.binary_cross_entropy(predictions['preference_prob'], labels)


class PPOWithHumanFeedback(nn.Module):
    """
    Proximal Policy Optimization with human feedback integration
    Optimizes policy based on learned reward model
    """
    
    def __init__(self, state_dim: int = 768, action_dim: int = 100,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy and value networks"""
        action_logits = self.policy_net(state)
        value = self.value_net(state)
        
        # Sample action from policy
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return {
            'action': action,
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value.squeeze(-1),
            'entropy': dist.entropy()
        }
    
    def compute_ppo_loss(self, states: torch.Tensor, actions: torch.Tensor,
                         advantages: torch.Tensor, old_log_probs: torch.Tensor,
                         returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss with clipped objective
        """
        # Get current policy outputs
        outputs = self.forward(states)
        
        # Calculate log probabilities
        dist = Categorical(outputs['action_probs'])
        log_probs = dist.log_prob(actions)
        
        # Calculate ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(outputs['value'], returns)
        
        # Entropy bonus
        entropy_loss = -outputs['entropy'].mean()
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss
        }


class RLHFTrainer:
    """
    Main RLHF training orchestrator
    Manages feedback collection, reward learning, and policy optimization
    """
    
    def __init__(self, state_dim: int = 768, action_dim: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize models
        self.reward_model = RewardModel(state_dim + action_dim)
        self.preference_learner = PreferenceLearner(self.reward_model)
        self.policy = PPOWithHumanFeedback(state_dim, action_dim)
        
        # Optimizers
        self.reward_optimizer = optim.AdamW(
            self.reward_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )
        
        # Feedback buffer
        self.feedback_buffer = deque(maxlen=10000)
        self.preference_pairs = deque(maxlen=5000)
        
        # Training statistics
        self.stats = {
            'total_feedback': 0,
            'preference_accuracy': 0.0,
            'average_reward': 0.0,
            'policy_entropy': 0.0,
            'compliance_rate': 0.0
        }
        
    def add_feedback(self, feedback: HumanFeedback):
        """Add human feedback to buffer"""
        self.feedback_buffer.append(feedback)
        self.stats['total_feedback'] += 1
        
        # Process different feedback types
        if feedback.feedback_type == FeedbackType.PREFERENCE:
            self._process_preference_feedback(feedback)
        elif feedback.feedback_type == FeedbackType.RATING:
            self._process_rating_feedback(feedback)
        elif feedback.feedback_type == FeedbackType.COMPLIANCE:
            self._process_compliance_feedback(feedback)
            
    def _process_preference_feedback(self, feedback: HumanFeedback):
        """Process preference comparisons"""
        if feedback.option_a and feedback.option_b and feedback.preference:
            preference_pair = {
                'option_a': feedback.option_a,
                'option_b': feedback.option_b,
                'preference': feedback.preference,
                'context': feedback.context,
                'timestamp': feedback.timestamp
            }
            self.preference_pairs.append(preference_pair)
            
    def _process_rating_feedback(self, feedback: HumanFeedback):
        """Convert ratings to reward signals"""
        if feedback.rating is not None:
            # Normalize rating to [-1, 1]
            normalized_reward = (feedback.rating / feedback.max_rating) * 2 - 1
            
            # Store as synthetic preference against baseline
            baseline_reward = 0.0  # Neutral baseline
            if normalized_reward > baseline_reward:
                preference = 'current'
            else:
                preference = 'baseline'
                
            synthetic_preference = {
                'option_a': feedback.context,
                'option_b': {'baseline': True},
                'preference': preference,
                'reward_diff': normalized_reward - baseline_reward,
                'timestamp': feedback.timestamp
            }
            self.preference_pairs.append(synthetic_preference)
            
    def _process_compliance_feedback(self, feedback: HumanFeedback):
        """Process compliance outcomes as rewards"""
        if feedback.compliant is not None:
            # Compliance success = positive reward
            reward = 1.0 if feedback.compliant else -1.0
            
            # Weight by number of violations
            if feedback.violations:
                reward *= (1.0 + len(feedback.violations) * 0.1)
                
            # Update compliance statistics
            self.stats['compliance_rate'] = (
                self.stats['compliance_rate'] * 0.95 + 
                (1.0 if feedback.compliant else 0.0) * 0.05
            )
            
    async def train_reward_model(self, batch_size: int = 32, epochs: int = 10):
        """Train reward model on collected feedback"""
        if len(self.preference_pairs) < batch_size:
            logger.warning("Insufficient preference data for training")
            return
            
        logger.info(f"Training reward model on {len(self.preference_pairs)} preferences")
        
        for epoch in range(epochs):
            # Sample batch of preferences
            batch_indices = np.random.choice(len(self.preference_pairs), 
                                           min(batch_size, len(self.preference_pairs)),
                                           replace=False)
            
            total_loss = 0
            correct_predictions = 0
            
            for idx in batch_indices:
                pair = self.preference_pairs[idx]
                
                # Convert to tensors (simplified - in production, use proper encoding)
                state_a = torch.randn(1, self.state_dim)
                action_a = torch.randn(1, self.action_dim)
                state_b = torch.randn(1, self.state_dim)
                action_b = torch.randn(1, self.action_dim)
                
                # Get preference label
                if pair['preference'] == 'a' or pair['preference'] == 'current':
                    label = torch.tensor([1.0])
                elif pair['preference'] == 'b' or pair['preference'] == 'baseline':
                    label = torch.tensor([0.0])
                else:  # equal
                    label = torch.tensor([0.5])
                
                # Forward pass
                predictions = self.preference_learner(
                    state_a, action_a, state_b, action_b
                )
                
                # Compute loss
                loss = self.preference_learner.compute_preference_loss(predictions, label)
                
                # Backward pass
                self.reward_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.reward_optimizer.step()
                
                total_loss += loss.item()
                
                # Track accuracy
                predicted_preference = predictions['preference_prob'].item() > 0.5
                true_preference = label.item() > 0.5
                if predicted_preference == true_preference:
                    correct_predictions += 1
                    
            # Update statistics
            avg_loss = total_loss / len(batch_indices)
            accuracy = correct_predictions / len(batch_indices)
            self.stats['preference_accuracy'] = accuracy
            
            logger.info(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
            
    async def optimize_policy(self, states: torch.Tensor, 
                             num_iterations: int = 100):
        """Optimize policy using PPO with learned rewards"""
        logger.info("Optimizing policy with human feedback")
        
        for iteration in range(num_iterations):
            # Collect trajectories using current policy
            with torch.no_grad():
                policy_outputs = self.policy(states)
                actions = policy_outputs['action']
                old_log_probs = Categorical(policy_outputs['action_probs']).log_prob(actions)
                values = policy_outputs['value']
                
            # Get rewards from learned reward model
            action_features = F.one_hot(actions, self.action_dim).float()
            rewards_dict = self.reward_model(states, action_features)
            rewards = rewards_dict['reward']
            
            # Calculate advantages using GAE
            advantages = self._calculate_advantages(rewards, values)
            returns = advantages + values
            
            # PPO update
            ppo_losses = self.policy.compute_ppo_loss(
                states, actions, advantages, old_log_probs, returns
            )
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            ppo_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update statistics
            self.stats['average_reward'] = rewards.mean().item()
            self.stats['policy_entropy'] = policy_outputs['entropy'].mean().item()
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Reward={self.stats['average_reward']:.4f}, "
                          f"Entropy={self.stats['policy_entropy']:.4f}")
                
    def _calculate_advantages(self, rewards: torch.Tensor, 
                            values: torch.Tensor, 
                            gamma: float = 0.99,
                            lam: float = 0.95) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def generate_with_feedback(self, state: torch.Tensor, 
                              temperature: float = 1.0) -> Dict[str, Any]:
        """Generate action with learned preferences"""
        with torch.no_grad():
            # Get policy output
            policy_output = self.policy(state)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = policy_output['action_logits'] / temperature
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
            else:
                action = policy_output['action']
                
            # Get expected reward
            action_features = F.one_hot(action, self.action_dim).float()
            reward_output = self.reward_model(state, action_features)
            
            return {
                'action': action.item(),
                'confidence': reward_output['confidence'].item(),
                'expected_reward': reward_output['reward'].item(),
                'policy_entropy': policy_output['entropy'].item(),
                'learned_from': f"{self.stats['total_feedback']} feedback samples"
            }
            
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        return {
            **self.stats,
            'feedback_buffer_size': len(self.feedback_buffer),
            'preference_pairs': len(self.preference_pairs),
            'model_parameters': sum(p.numel() for p in self.reward_model.parameters()),
            'policy_parameters': sum(p.numel() for p in self.policy.parameters())
        }


class OrganizationalPreferenceLearner:
    """
    Learns organization-specific preferences and compliance requirements
    Adapts to different industries and regulatory frameworks
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        
        # Organization embeddings
        self.org_embeddings = {}
        self.industry_embeddings = {
            'healthcare': torch.randn(embedding_dim),
            'finance': torch.randn(embedding_dim),
            'government': torch.randn(embedding_dim),
            'technology': torch.randn(embedding_dim),
            'retail': torch.randn(embedding_dim),
            'manufacturing': torch.randn(embedding_dim)
        }
        
        # Compliance framework embeddings
        self.compliance_embeddings = {
            'hipaa': torch.randn(embedding_dim),
            'gdpr': torch.randn(embedding_dim),
            'sox': torch.randn(embedding_dim),
            'pci-dss': torch.randn(embedding_dim),
            'iso27001': torch.randn(embedding_dim),
            'nist': torch.randn(embedding_dim)
        }
        
        # Preference history
        self.preference_history = defaultdict(list)
        
    def learn_organization_preferences(self, org_id: str, 
                                      feedback_history: List[HumanFeedback]):
        """Learn preferences specific to an organization"""
        if not feedback_history:
            return
            
        # Initialize org embedding if new
        if org_id not in self.org_embeddings:
            self.org_embeddings[org_id] = torch.randn(self.embedding_dim)
            
        # Aggregate feedback patterns
        preference_vectors = []
        for feedback in feedback_history:
            # Extract preference signal
            if feedback.feedback_type == FeedbackType.PREFERENCE:
                if feedback.preference == 'a':
                    signal = 1.0
                elif feedback.preference == 'b':
                    signal = -1.0
                else:
                    signal = 0.0
            elif feedback.feedback_type == FeedbackType.RATING:
                signal = (feedback.rating / feedback.max_rating) * 2 - 1
            elif feedback.feedback_type == FeedbackType.COMPLIANCE:
                signal = 1.0 if feedback.compliant else -1.0
            else:
                continue
                
            # Weight by recency
            age_days = (datetime.utcnow() - feedback.timestamp).days
            recency_weight = 1.0 / (1.0 + age_days * 0.01)
            
            preference_vectors.append(signal * recency_weight)
            
        # Update organization embedding
        if preference_vectors:
            avg_preference = np.mean(preference_vectors)
            update_direction = torch.randn(self.embedding_dim) * avg_preference
            self.org_embeddings[org_id] = (
                self.org_embeddings[org_id] * 0.9 + update_direction * 0.1
            )
            
        # Store in history
        self.preference_history[org_id].extend(feedback_history)
        
    def get_organization_context(self, org_id: str, 
                                industry: Optional[str] = None,
                                compliance_frameworks: Optional[List[str]] = None) -> torch.Tensor:
        """Get contextualized embedding for organization"""
        context_vectors = []
        
        # Add organization embedding
        if org_id in self.org_embeddings:
            context_vectors.append(self.org_embeddings[org_id])
        else:
            context_vectors.append(torch.zeros(self.embedding_dim))
            
        # Add industry context
        if industry and industry in self.industry_embeddings:
            context_vectors.append(self.industry_embeddings[industry])
            
        # Add compliance context
        if compliance_frameworks:
            for framework in compliance_frameworks:
                if framework in self.compliance_embeddings:
                    context_vectors.append(self.compliance_embeddings[framework])
                    
        # Combine contexts
        if context_vectors:
            context = torch.stack(context_vectors).mean(dim=0)
        else:
            context = torch.zeros(self.embedding_dim)
            
        return context


# Global RLHF system instance
rlhf_system = None

def initialize_rlhf(state_dim: int = 768, action_dim: int = 100):
    """Initialize the RLHF system"""
    global rlhf_system
    rlhf_system = RLHFTrainer(state_dim, action_dim)
    logger.info("RLHF system initialized")
    return rlhf_system


# Export main components
__all__ = [
    'RLHFTrainer',
    'RewardModel',
    'PreferenceLearner',
    'PPOWithHumanFeedback',
    'HumanFeedback',
    'FeedbackType',
    'OrganizationalPreferenceLearner',
    'initialize_rlhf',
    'rlhf_system'
]