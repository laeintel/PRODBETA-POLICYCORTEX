"""
Patent #4: Predictive Policy Compliance Engine
SHAP-based Explainability Engine

This module implements the SHAP explainability subsystem as specified in Patent #4,
providing local and global explanations, feature importance, and decision attribution
for regulatory compliance and audit requirements.
"""

import shap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityConfig:
    """Configuration for SHAP explainability engine."""
    background_samples: int = 100
    max_display_features: int = 20
    interaction_analysis: bool = True
    waterfall_plots: bool = True
    force_plots: bool = True
    decision_plots: bool = True
    summary_plots: bool = True
    dependence_plots: bool = True
    cache_explanations: bool = True
    explanation_sample_size: int = 1000
    feature_names: Optional[List[str]] = None
    

class SHAPExplainer:
    """
    SHAP-based explainability engine for model interpretability.
    Patent Specification: Local and global explanations with feature importance.
    """
    
    def __init__(self, model: Any, model_type: str, config: Optional[ExplainabilityConfig] = None):
        """
        Initialize SHAP explainer for different model types.
        
        Args:
            model: The model to explain
            model_type: Type of model ('neural', 'tree', 'ensemble', 'linear')
            config: Explainability configuration
        """
        self.model = model
        self.model_type = model_type
        self.config = config or ExplainabilityConfig()
        
        # Initialize appropriate explainer
        self.explainer = None
        self.background_data = None
        self.shap_values_cache = {}
        
        logger.info(f"Initialized SHAP explainer for {model_type} model")
        
    def initialize_explainer(self, background_data: Union[np.ndarray, torch.Tensor]):
        """
        Initialize the appropriate SHAP explainer based on model type.
        
        Args:
            background_data: Background dataset for SHAP explanations
        """
        # Store background data
        if isinstance(background_data, torch.Tensor):
            self.background_data = background_data.detach().cpu().numpy()
        else:
            self.background_data = background_data
            
        # Sample background if too large
        if len(self.background_data) > self.config.background_samples:
            indices = np.random.choice(len(self.background_data), 
                                     self.config.background_samples, 
                                     replace=False)
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data
        
        # Create appropriate explainer
        if self.model_type == 'neural':
            # For neural networks, use DeepExplainer or GradientExplainer
            if isinstance(self.model, nn.Module):
                self.explainer = self._create_deep_explainer(background_sample)
            else:
                raise ValueError("Neural model must be a torch.nn.Module")
                
        elif self.model_type == 'tree':
            # For tree-based models, use TreeExplainer
            self.explainer = shap.TreeExplainer(self.model, background_sample)
            
        elif self.model_type == 'ensemble':
            # For ensemble models, use KernelExplainer
            self.explainer = shap.KernelExplainer(
                self._model_predict_function, 
                background_sample
            )
            
        elif self.model_type == 'linear':
            # For linear models, use LinearExplainer
            self.explainer = shap.LinearExplainer(self.model, background_sample)
            
        else:
            # Default to KernelExplainer for unknown types
            self.explainer = shap.KernelExplainer(
                self._model_predict_function,
                background_sample
            )
            
        logger.info(f"Explainer initialized with {len(background_sample)} background samples")
        
    def _create_deep_explainer(self, background: np.ndarray) -> shap.DeepExplainer:
        """
        Create DeepExplainer for neural networks.
        
        Args:
            background: Background data
            
        Returns:
            DeepExplainer instance
        """
        # Convert to tensor
        background_tensor = torch.FloatTensor(background)
        
        # Create wrapper for model prediction
        def model_wrapper(x):
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, dict):
                    # Handle dictionary output
                    return output['predictions'].cpu().numpy()
                return output.cpu().numpy()
        
        # Use GradientExplainer for better stability with deep networks
        return shap.GradientExplainer(model_wrapper, background_tensor)
        
    def _model_predict_function(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper function for model predictions.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        if isinstance(self.model, nn.Module):
            x_tensor = torch.FloatTensor(x)
            with torch.no_grad():
                output = self.model(x_tensor)
                if isinstance(output, dict):
                    return output['predictions'].cpu().numpy()
                return output.cpu().numpy()
        else:
            return self.model.predict(x)
            
    def explain_local(self, instance: Union[np.ndarray, torch.Tensor], 
                     prediction: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            instance: Single instance to explain
            prediction: Optional pre-computed prediction
            
        Returns:
            Dictionary with local explanation data
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
            
        # Convert instance to numpy
        if isinstance(instance, torch.Tensor):
            instance_np = instance.detach().cpu().numpy()
        else:
            instance_np = instance
            
        # Ensure 2D shape
        if len(instance_np.shape) == 1:
            instance_np = instance_np.reshape(1, -1)
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance_np)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
        # Get prediction if not provided
        if prediction is None:
            prediction = self._model_predict_function(instance_np)[0]
            
        # Calculate feature importance
        feature_importance = np.abs(shap_values[0])
        top_features_idx = np.argsort(feature_importance)[-self.config.max_display_features:][::-1]
        
        # Create feature names if not provided
        if self.config.feature_names:
            feature_names = self.config.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(instance_np.shape[1])]
            
        # Build explanation dictionary
        explanation = {
            'shap_values': shap_values[0].tolist(),
            'feature_values': instance_np[0].tolist(),
            'prediction': float(prediction) if isinstance(prediction, (np.ndarray, torch.Tensor)) else prediction,
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0,
            'top_features': {
                'names': [feature_names[i] for i in top_features_idx],
                'values': [float(instance_np[0, i]) for i in top_features_idx],
                'impacts': [float(shap_values[0, i]) for i in top_features_idx]
            }
        }
        
        # Generate visualizations if configured
        if self.config.waterfall_plots:
            explanation['waterfall_data'] = self._generate_waterfall_data(
                shap_values[0], instance_np[0], feature_names
            )
            
        if self.config.force_plots:
            explanation['force_plot_data'] = self._generate_force_plot_data(
                shap_values[0], instance_np[0], feature_names
            )
            
        return explanation
        
    def explain_global(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Generate global explanation across multiple instances.
        
        Args:
            data: Dataset to explain
            
        Returns:
            Dictionary with global explanation data
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
            
        # Convert to numpy
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        # Sample if too large
        if len(data_np) > self.config.explanation_sample_size:
            indices = np.random.choice(len(data_np), 
                                     self.config.explanation_sample_size, 
                                     replace=False)
            data_sample = data_np[indices]
        else:
            data_sample = data_np
            
        # Calculate SHAP values for all samples
        logger.info(f"Calculating SHAP values for {len(data_sample)} samples...")
        shap_values = self.explainer.shap_values(data_sample)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
        # Calculate global feature importance
        global_importance = np.abs(shap_values).mean(axis=0)
        importance_std = np.abs(shap_values).std(axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(global_importance)[::-1]
        
        # Create feature names
        if self.config.feature_names:
            feature_names = self.config.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(data_sample.shape[1])]
            
        # Build global explanation
        global_explanation = {
            'feature_importance': {
                'names': [feature_names[i] for i in sorted_idx[:self.config.max_display_features]],
                'values': [float(global_importance[i]) for i in sorted_idx[:self.config.max_display_features]],
                'std': [float(importance_std[i]) for i in sorted_idx[:self.config.max_display_features]]
            },
            'summary_stats': {
                'mean_abs_shap': float(np.abs(shap_values).mean()),
                'max_abs_shap': float(np.abs(shap_values).max()),
                'min_abs_shap': float(np.abs(shap_values).min()),
                'total_features': len(feature_names),
                'samples_explained': len(data_sample)
            }
        }
        
        # Add interaction effects if configured
        if self.config.interaction_analysis:
            interaction_values = self._calculate_interaction_effects(shap_values, data_sample)
            global_explanation['interaction_effects'] = interaction_values
            
        # Add distribution analysis
        global_explanation['shap_distributions'] = self._analyze_shap_distributions(
            shap_values, feature_names, sorted_idx[:10]
        )
        
        return global_explanation
        
    def _generate_waterfall_data(self, shap_values: np.ndarray, 
                                feature_values: np.ndarray, 
                                feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate data for waterfall plot visualization.
        
        Args:
            shap_values: SHAP values for instance
            feature_values: Feature values for instance
            feature_names: Names of features
            
        Returns:
            Waterfall plot data
        """
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:self.config.max_display_features]
        
        waterfall_data = {
            'features': [feature_names[i] for i in sorted_idx],
            'values': [float(shap_values[i]) for i in sorted_idx],
            'feature_values': [float(feature_values[i]) for i in sorted_idx],
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0
        }
        
        return waterfall_data
        
    def _generate_force_plot_data(self, shap_values: np.ndarray,
                                 feature_values: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate data for force plot visualization.
        
        Args:
            shap_values: SHAP values for instance
            feature_values: Feature values for instance
            feature_names: Names of features
            
        Returns:
            Force plot data
        """
        # Separate positive and negative contributions
        positive_idx = shap_values > 0
        negative_idx = shap_values < 0
        
        force_plot_data = {
            'positive_forces': {
                'features': [feature_names[i] for i, p in enumerate(positive_idx) if p],
                'values': [float(shap_values[i]) for i, p in enumerate(positive_idx) if p],
                'feature_values': [float(feature_values[i]) for i, p in enumerate(positive_idx) if p]
            },
            'negative_forces': {
                'features': [feature_names[i] for i, n in enumerate(negative_idx) if n],
                'values': [float(shap_values[i]) for i, n in enumerate(negative_idx) if n],
                'feature_values': [float(feature_values[i]) for i, n in enumerate(negative_idx) if n]
            },
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0
        }
        
        return force_plot_data
        
    def _calculate_interaction_effects(self, shap_values: np.ndarray,
                                      data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate SHAP interaction values between features.
        
        Args:
            shap_values: SHAP values matrix
            data: Original data
            
        Returns:
            Interaction effects analysis
        """
        n_features = shap_values.shape[1]
        
        # Calculate correlation between SHAP values
        shap_df = pd.DataFrame(shap_values)
        shap_corr = shap_df.corr().values
        
        # Find top interactions
        interaction_scores = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                score = abs(shap_corr[i, j])
                if score > 0.3:  # Threshold for significant interaction
                    interaction_scores.append({
                        'feature_1': i,
                        'feature_2': j,
                        'interaction_strength': float(score)
                    })
                    
        # Sort by interaction strength
        interaction_scores.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        # Create feature names
        if self.config.feature_names:
            feature_names = self.config.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        # Format results
        top_interactions = []
        for interaction in interaction_scores[:10]:  # Top 10 interactions
            top_interactions.append({
                'features': f"{feature_names[interaction['feature_1']]} × {feature_names[interaction['feature_2']]}",
                'strength': interaction['interaction_strength']
            })
            
        return {
            'top_interactions': top_interactions,
            'interaction_matrix': shap_corr.tolist() if n_features <= 20 else None
        }
        
    def _analyze_shap_distributions(self, shap_values: np.ndarray,
                                   feature_names: List[str],
                                   feature_indices: List[int]) -> Dict[str, Any]:
        """
        Analyze SHAP value distributions for top features.
        
        Args:
            shap_values: SHAP values matrix
            feature_names: Names of features
            feature_indices: Indices of features to analyze
            
        Returns:
            Distribution analysis
        """
        distributions = {}
        
        for idx in feature_indices:
            feature_shap = shap_values[:, idx]
            distributions[feature_names[idx]] = {
                'mean': float(np.mean(feature_shap)),
                'std': float(np.std(feature_shap)),
                'min': float(np.min(feature_shap)),
                'max': float(np.max(feature_shap)),
                'q25': float(np.percentile(feature_shap, 25)),
                'q50': float(np.percentile(feature_shap, 50)),
                'q75': float(np.percentile(feature_shap, 75)),
                'positive_ratio': float((feature_shap > 0).mean()),
                'zero_ratio': float((np.abs(feature_shap) < 1e-6).mean())
            }
            
        return distributions


class AttentionVisualizer:
    """
    Visualizer for attention mechanisms in neural networks.
    Extracts and visualizes attention weights for interpretability.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Neural network model with attention mechanisms
        """
        self.model = model
        self.attention_weights = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture attention weights during forward pass."""
        def hook_fn(module, input, output, name):
            if isinstance(output, tuple) and len(output) > 1:
                # Assume second output is attention weights
                self.attention_weights[name] = output[1].detach().cpu()
            elif hasattr(output, 'attention_weights'):
                self.attention_weights[name] = output.attention_weights.detach().cpu()
                
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'multihead' in name.lower():
                module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                
    def extract_attention_weights(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for given input.
        
        Args:
            input_data: Input to the model
            
        Returns:
            Dictionary of attention weights by layer
        """
        self.attention_weights = {}
        
        # Forward pass to capture attention
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_data)
            
        return self.attention_weights.copy()
        
    def visualize_attention_pattern(self, attention_weights: torch.Tensor,
                                   sequence_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate visualization data for attention patterns.
        
        Args:
            attention_weights: Attention weight tensor [batch, heads, seq_len, seq_len]
            sequence_labels: Optional labels for sequence positions
            
        Returns:
            Visualization data
        """
        # Average over batch and heads
        if len(attention_weights.shape) == 4:
            avg_attention = attention_weights.mean(dim=0).mean(dim=0)
        elif len(attention_weights.shape) == 3:
            avg_attention = attention_weights.mean(dim=0)
        else:
            avg_attention = attention_weights
            
        avg_attention = avg_attention.numpy()
        
        # Create labels if not provided
        if sequence_labels is None:
            sequence_labels = [f"Pos_{i}" for i in range(avg_attention.shape[0])]
            
        # Find most attended positions
        attention_sum = avg_attention.sum(axis=0)
        top_positions = np.argsort(attention_sum)[-10:][::-1]
        
        visualization_data = {
            'attention_matrix': avg_attention.tolist(),
            'sequence_labels': sequence_labels,
            'top_attended_positions': {
                'positions': [int(p) for p in top_positions],
                'labels': [sequence_labels[p] for p in top_positions],
                'scores': [float(attention_sum[p]) for p in top_positions]
            },
            'attention_statistics': {
                'max_attention': float(avg_attention.max()),
                'min_attention': float(avg_attention.min()),
                'mean_attention': float(avg_attention.mean()),
                'attention_entropy': float(-np.sum(avg_attention * np.log(avg_attention + 1e-8)))
            }
        }
        
        return visualization_data
        
    def analyze_head_specialization(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze specialization patterns across attention heads.
        
        Args:
            attention_weights: Multi-head attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Head specialization analysis
        """
        if len(attention_weights.shape) != 4:
            return {'error': 'Expected 4D tensor for multi-head attention'}
            
        n_heads = attention_weights.shape[1]
        
        # Analyze each head
        head_analysis = []
        for head_idx in range(n_heads):
            head_weights = attention_weights[:, head_idx, :, :].mean(dim=0).numpy()
            
            # Calculate entropy (measure of focus)
            entropy = -np.sum(head_weights * np.log(head_weights + 1e-8))
            
            # Calculate diagonal attention (self-attention strength)
            diagonal_strength = np.mean(np.diagonal(head_weights))
            
            # Calculate distance-based attention pattern
            seq_len = head_weights.shape[0]
            distances = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if head_weights[i, j] > 0.1:  # Threshold for significant attention
                        distances.append(abs(i - j))
                        
            avg_distance = np.mean(distances) if distances else 0
            
            head_analysis.append({
                'head_index': head_idx,
                'entropy': float(entropy),
                'diagonal_strength': float(diagonal_strength),
                'average_attention_distance': float(avg_distance),
                'max_attention': float(head_weights.max()),
                'pattern_type': self._classify_attention_pattern(head_weights)
            })
            
        return {
            'n_heads': n_heads,
            'head_analysis': head_analysis,
            'head_diversity': float(np.std([h['entropy'] for h in head_analysis]))
        }
        
    def _classify_attention_pattern(self, attention_matrix: np.ndarray) -> str:
        """
        Classify the type of attention pattern.
        
        Args:
            attention_matrix: Attention weight matrix
            
        Returns:
            Pattern type string
        """
        # Check for diagonal pattern (local attention)
        diagonal_sum = np.trace(attention_matrix)
        total_sum = attention_matrix.sum()
        
        if diagonal_sum / total_sum > 0.5:
            return 'local/diagonal'
            
        # Check for global pattern (uniform attention)
        std_dev = attention_matrix.std()
        if std_dev < 0.1:
            return 'global/uniform'
            
        # Check for block pattern
        seq_len = attention_matrix.shape[0]
        block_size = seq_len // 4
        block_sums = []
        for i in range(0, seq_len, block_size):
            for j in range(0, seq_len, block_size):
                block = attention_matrix[i:i+block_size, j:j+block_size]
                block_sums.append(block.sum())
                
        if max(block_sums) / sum(block_sums) > 0.5:
            return 'block/structured'
            
        return 'mixed/complex'


class DecisionTreeExtractor:
    """
    Extract and visualize decision paths from tree-based models.
    Provides rule-based explanations for interpretability.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize decision tree extractor.
        
        Args:
            model: Tree-based model (RandomForest, GradientBoosting, etc.)
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names
        
    def extract_decision_path(self, instance: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract decision path for a single instance.
        
        Args:
            instance: Input instance
            
        Returns:
            List of decision rules along the path
        """
        if not hasattr(self.model, 'tree_') and not hasattr(self.model, 'estimators_'):
            raise ValueError("Model must be a tree-based model")
            
        # Handle single tree
        if hasattr(self.model, 'tree_'):
            return self._extract_path_from_tree(self.model, instance)
            
        # Handle ensemble of trees
        if hasattr(self.model, 'estimators_'):
            paths = []
            for estimator in self.model.estimators_[:5]:  # Limit to first 5 trees
                if hasattr(estimator, 'tree_'):
                    path = self._extract_path_from_tree(estimator, instance)
                    paths.extend(path)
            return paths
            
        return []
        
    def _extract_path_from_tree(self, tree_model: Any, instance: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract decision path from a single tree.
        
        Args:
            tree_model: Single decision tree
            instance: Input instance
            
        Returns:
            List of decision rules
        """
        tree = tree_model.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        # Get decision path
        node_indicator = tree_model.decision_path(instance.reshape(1, -1))
        leaf_id = tree_model.apply(instance.reshape(1, -1))[0]
        
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        rules = []
        for node_id in node_index:
            if feature[node_id] != -2:  # Not a leaf node
                feature_name = self.feature_names[feature[node_id]] if self.feature_names else f"Feature_{feature[node_id]}"
                
                if instance[feature[node_id]] <= threshold[node_id]:
                    rule = f"{feature_name} <= {threshold[node_id]:.3f}"
                else:
                    rule = f"{feature_name} > {threshold[node_id]:.3f}"
                    
                rules.append({
                    'feature': feature_name,
                    'threshold': float(threshold[node_id]),
                    'value': float(instance[feature[node_id]]),
                    'comparison': '<=' if instance[feature[node_id]] <= threshold[node_id] else '>',
                    'node_id': int(node_id)
                })
                
        return rules
        
    def extract_important_rules(self, data: np.ndarray, 
                               max_rules: int = 20) -> List[Dict[str, Any]]:
        """
        Extract most important decision rules from the model.
        
        Args:
            data: Dataset to analyze
            max_rules: Maximum number of rules to extract
            
        Returns:
            List of important rules with coverage and accuracy
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
            
        # Get feature importance
        importance = self.model.feature_importances_
        top_features = np.argsort(importance)[-10:][::-1]
        
        # Fit a simplified tree to extract rules
        simplified_tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50)
        simplified_tree.fit(data, self.model.predict(data))
        
        # Extract rules from simplified tree
        rules = self._extract_all_rules(simplified_tree, data)
        
        # Sort by importance (coverage * accuracy)
        rules.sort(key=lambda x: x['importance'], reverse=True)
        
        return rules[:max_rules]
        
    def _extract_all_rules(self, tree_model: Any, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract all rules from a decision tree.
        
        Args:
            tree_model: Decision tree model
            data: Dataset for coverage calculation
            
        Returns:
            List of all rules with metrics
        """
        tree = tree_model.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        
        rules = []
        
        def recurse(node, path, data_indices):
            if feature[node] != -2:  # Not a leaf
                feature_name = self.feature_names[feature[node]] if self.feature_names else f"Feature_{feature[node]}"
                
                # Left child (<=)
                left_indices = data_indices[data[data_indices, feature[node]] <= threshold[node]]
                if len(left_indices) > 0:
                    left_path = path + [f"{feature_name} <= {threshold[node]:.3f}"]
                    recurse(tree.children_left[node], left_path, left_indices)
                    
                # Right child (>)
                right_indices = data_indices[data[data_indices, feature[node]] > threshold[node]]
                if len(right_indices) > 0:
                    right_path = path + [f"{feature_name} > {threshold[node]:.3f}"]
                    recurse(tree.children_right[node], right_path, right_indices)
                    
            else:  # Leaf node
                if len(path) > 0:
                    coverage = len(data_indices) / len(data)
                    prediction = value[node][0][0] if len(value[node][0]) > 0 else 0
                    
                    rules.append({
                        'rule': ' AND '.join(path),
                        'coverage': float(coverage),
                        'samples': len(data_indices),
                        'prediction': float(prediction),
                        'importance': float(coverage * abs(prediction))
                    })
                    
        # Start recursion from root
        recurse(0, [], np.arange(len(data)))
        
        return rules


def create_integrated_explainer(model: Any, model_type: str, 
                               config: Optional[ExplainabilityConfig] = None) -> Dict[str, Any]:
    """
    Create an integrated explainability system combining SHAP, attention, and tree extraction.
    
    Args:
        model: Model to explain
        model_type: Type of model
        config: Explainability configuration
        
    Returns:
        Dictionary of explainability components
    """
    components = {}
    
    # Create SHAP explainer
    components['shap'] = SHAPExplainer(model, model_type, config)
    
    # Add attention visualizer for neural models
    if model_type == 'neural' and isinstance(model, nn.Module):
        components['attention'] = AttentionVisualizer(model)
        
    # Add tree extractor for tree-based models
    if model_type == 'tree' or (hasattr(model, 'estimators_') and hasattr(model, 'feature_importances_')):
        feature_names = config.feature_names if config else None
        components['tree'] = DecisionTreeExtractor(model, feature_names)
        
    logger.info(f"Created integrated explainer with components: {list(components.keys())}")
    
    return components


if __name__ == "__main__":
    # Test the explainability system
    import sys
    sys.path.append('..')
    from policy_compliance_predictor import create_policy_compliance_predictor
    
    # Create a simple model for testing
    config = {'input_size': 256, 'num_classes': 2}
    model = create_policy_compliance_predictor(config)
    
    # Create test data
    n_samples = 100
    seq_len = 100
    input_size = 256
    
    resource_features = torch.randn(n_samples, seq_len, input_size)
    policy_features = torch.randn(n_samples, input_size)
    
    # Create explainability configuration
    explain_config = ExplainabilityConfig(
        background_samples=50,
        max_display_features=10,
        feature_names=[f"Config_{i}" for i in range(input_size)]
    )
    
    # Create integrated explainer
    explainers = create_integrated_explainer(model, 'neural', explain_config)
    
    # Initialize SHAP explainer with background data
    print("Initializing SHAP explainer...")
    # For neural networks with complex inputs, we need a wrapper
    def model_wrapper(x):
        # Assume x is flattened [batch, seq_len * input_size]
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, seq_len, input_size)
        x_tensor = torch.FloatTensor(x_reshaped)
        policy_tensor = torch.randn(batch_size, input_size)  # Dummy policy features
        
        with torch.no_grad():
            output = model(x_tensor, policy_tensor)
            return output['violation_probability'].numpy()
    
    # Flatten data for SHAP
    background_flat = resource_features[:10].reshape(10, -1).numpy()
    explainers['shap'].model = model_wrapper
    explainers['shap'].model_type = 'ensemble'  # Use KernelExplainer for complex model
    explainers['shap'].initialize_explainer(background_flat)
    
    # Test local explanation
    print("\nGenerating local explanation...")
    test_instance = resource_features[0].reshape(1, -1).numpy()
    local_explanation = explainers['shap'].explain_local(test_instance)
    
    print(f"Local Explanation Results:")
    print(f"  Prediction: {local_explanation['prediction']:.4f}")
    print(f"  Base value: {local_explanation['base_value']:.4f}")
    print(f"  Top features: {len(local_explanation['top_features']['names'])}")
    
    # Test attention visualization
    if 'attention' in explainers:
        print("\nExtracting attention weights...")
        attention_weights = explainers['attention'].extract_attention_weights(
            resource_features[:1]
        )
        print(f"  Captured attention from {len(attention_weights)} layers")
        
        if attention_weights:
            # Visualize first attention layer
            first_layer = list(attention_weights.keys())[0]
            viz_data = explainers['attention'].visualize_attention_pattern(
                attention_weights[first_layer]
            )
            print(f"  Top attended positions: {viz_data['top_attended_positions']['positions'][:5]}")
            print(f"  Attention entropy: {viz_data['attention_statistics']['attention_entropy']:.4f}")
    
    print("\nExplainability system test completed successfully!")