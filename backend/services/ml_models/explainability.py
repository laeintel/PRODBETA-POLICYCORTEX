"""
Patent #4: SHAP-Based Explainability Engine
Feature importance and decision attribution for regulatory compliance
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- Local explanations for individual predictions
- Global feature importance analysis
- Interaction effect analysis
- Waterfall plot generation
- Attention mechanism visualization
"""

import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Container for explanation results"""
    feature_importance: Dict[str, float]
    shap_values: np.ndarray
    base_value: float
    prediction: float
    confidence: float
    interaction_effects: Optional[Dict[str, Dict[str, float]]]
    visualization_data: Dict[str, Any]


class SHAPExplainer:
    """
    SHAP-based explainability for all model types
    Implements TreeExplainer, DeepExplainer, and KernelExplainer
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.explainers = {}
        self.feature_names = []
        self.background_data = None
        
    def initialize(self, models: Dict[str, Any], background_data: np.ndarray, 
                  feature_names: List[str]):
        """Initialize SHAP explainers for different model types"""
        self.feature_names = feature_names
        self.background_data = background_data
        
        for model_name, model in models.items():
            if 'gradient_boost' in model_name.lower():
                # TreeExplainer for gradient boosting models
                self.explainers[model_name] = shap.TreeExplainer(model)
                logger.info(f"Initialized TreeExplainer for {model_name}")
                
            elif 'neural' in model_name.lower() or 'lstm' in model_name.lower():
                # DeepExplainer for neural networks
                if isinstance(model, nn.Module):
                    # Convert background data to tensor
                    background_tensor = torch.FloatTensor(background_data[:100])  # Use subset for efficiency
                    self.explainers[model_name] = shap.DeepExplainer(model, background_tensor)
                    logger.info(f"Initialized DeepExplainer for {model_name}")
                    
            else:
                # KernelExplainer for black-box models
                if background_data is not None:
                    # Use subset of background data for efficiency
                    background_subset = shap.sample(background_data, min(100, len(background_data)))
                    self.explainers[model_name] = shap.KernelExplainer(
                        lambda x: self._model_predict(model, x),
                        background_subset
                    )
                    logger.info(f"Initialized KernelExplainer for {model_name}")
    
    def _model_predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Wrapper for model predictions"""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            return model.predict(X)
        else:
            raise ValueError(f"Model {type(model)} does not have predict method")
    
    def explain_instance(self, X: np.ndarray, model_name: str) -> ExplanationResult:
        """
        Generate local explanation for individual prediction
        Patent Requirement: Local explanations with feature attribution
        """
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model {model_name}")
        
        explainer = self.explainers[model_name]
        
        # Calculate SHAP values
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
        else:
            shap_values = explainer.shap_values(X)
        
        # Get base value and prediction
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = 0.0
        
        # Calculate prediction
        prediction = base_value + np.sum(shap_values)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            if i < len(shap_values):
                feature_importance[feature] = float(shap_values[i])
        
        # Sort by absolute importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        # Create visualization data
        viz_data = self._create_visualization_data(shap_values, X, base_value)
        
        return ExplanationResult(
            feature_importance=feature_importance,
            shap_values=shap_values,
            base_value=float(base_value),
            prediction=float(prediction),
            confidence=self._calculate_confidence(shap_values),
            interaction_effects=None,  # Will be calculated separately if needed
            visualization_data=viz_data
        )
    
    def explain_global(self, X: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Generate global feature importance analysis
        Patent Requirement: Global feature importance across all predictions
        """
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model {model_name}")
        
        explainer = self.explainers[model_name]
        
        # Calculate SHAP values for all samples
        logger.info(f"Calculating global SHAP values for {len(X)} samples...")
        
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For kernel explainer, use sampling for efficiency
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create global importance dictionary
        global_importance = {}
        for i, feature in enumerate(self.feature_names):
            if i < len(mean_abs_shap):
                global_importance[feature] = float(mean_abs_shap[i])
        
        # Sort by importance
        global_importance = dict(sorted(
            global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Calculate additional statistics
        shap_stats = {
            'mean': np.mean(shap_values, axis=0).tolist(),
            'std': np.std(shap_values, axis=0).tolist(),
            'min': np.min(shap_values, axis=0).tolist(),
            'max': np.max(shap_values, axis=0).tolist(),
            'median': np.median(shap_values, axis=0).tolist()
        }
        
        return {
            'global_importance': global_importance,
            'top_features': list(global_importance.keys())[:10],
            'shap_statistics': shap_stats,
            'sample_size': len(shap_values)
        }
    
    def calculate_interaction_effects(self, X: np.ndarray, model_name: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate SHAP interaction values between features
        Patent Requirement: Interaction effect analysis
        """
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model {model_name}")
        
        explainer = self.explainers[model_name]
        
        # Only TreeExplainer supports interaction values efficiently
        if not isinstance(explainer, shap.TreeExplainer):
            logger.warning(f"Interaction values not efficiently supported for {type(explainer)}")
            return {}
        
        # Calculate interaction values
        interaction_values = explainer.shap_interaction_values(X)
        
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]  # For binary classification
        
        # Process interaction matrix
        interactions = {}
        n_features = min(len(self.feature_names), interaction_values.shape[1])
        
        for i in range(n_features):
            feature_i = self.feature_names[i]
            interactions[feature_i] = {}
            
            for j in range(n_features):
                if i != j:  # Skip self-interactions
                    feature_j = self.feature_names[j]
                    # Average interaction across samples
                    avg_interaction = np.mean(interaction_values[:, i, j])
                    interactions[feature_i][feature_j] = float(avg_interaction)
        
        # Sort interactions by absolute value
        for feature in interactions:
            interactions[feature] = dict(sorted(
                interactions[feature].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
        
        return interactions
    
    def _create_visualization_data(self, shap_values: np.ndarray, X: np.ndarray, 
                                  base_value: float) -> Dict[str, Any]:
        """Create data for visualization plots"""
        # Waterfall plot data
        waterfall_data = {
            'features': self.feature_names[:len(shap_values)],
            'values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': base_value,
            'feature_values': X.flatten().tolist() if isinstance(X, np.ndarray) else X
        }
        
        # Force plot data
        force_plot_data = {
            'positive_features': [],
            'negative_features': [],
            'base_value': base_value
        }
        
        for i, (feature, value) in enumerate(zip(self.feature_names, shap_values)):
            if i >= len(shap_values):
                break
            
            feature_data = {
                'name': feature,
                'value': float(value),
                'feature_value': float(X.flatten()[i]) if i < len(X.flatten()) else 0
            }
            
            if value > 0:
                force_plot_data['positive_features'].append(feature_data)
            else:
                force_plot_data['negative_features'].append(feature_data)
        
        # Sort by absolute value
        force_plot_data['positive_features'].sort(key=lambda x: x['value'], reverse=True)
        force_plot_data['negative_features'].sort(key=lambda x: abs(x['value']), reverse=True)
        
        return {
            'waterfall': waterfall_data,
            'force_plot': force_plot_data
        }
    
    def _calculate_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate confidence based on SHAP value distribution"""
        # Higher confidence when few features dominate
        abs_values = np.abs(shap_values)
        if np.sum(abs_values) == 0:
            return 0.5
        
        # Calculate entropy of normalized absolute SHAP values
        normalized = abs_values / (np.sum(abs_values) + 1e-8)
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        
        # Lower entropy means higher confidence
        max_entropy = np.log(len(shap_values))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return float(confidence)


class AttentionVisualizer:
    """
    Visualize attention weights from LSTM networks
    Patent Requirement: Attention mechanism visualization
    """
    
    def __init__(self):
        self.attention_cache = {}
        
    def extract_attention_weights(self, model: nn.Module, X: torch.Tensor) -> np.ndarray:
        """Extract attention weights from model forward pass"""
        model.eval()
        
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Assuming second output is attention weights
                attention_weights.append(output[1].detach().cpu().numpy())
        
        # Register hook on attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(X)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if attention_weights:
            return np.concatenate(attention_weights, axis=0)
        else:
            logger.warning("No attention weights captured")
            return np.array([])
    
    def visualize_temporal_attention(self, attention_weights: np.ndarray, 
                                    sequence_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Visualize temporal attention patterns
        Returns visualization data for rendering
        """
        if len(attention_weights.shape) == 4:
            # Multi-head attention: [batch, heads, seq, seq]
            # Average across heads
            attention_weights = np.mean(attention_weights, axis=1)
        
        if len(attention_weights.shape) == 3:
            # [batch, seq, seq]
            # Average across batch
            attention_weights = np.mean(attention_weights, axis=0)
        
        # Create labels if not provided
        if sequence_labels is None:
            sequence_labels = [f"T-{i}" for i in range(attention_weights.shape[0])]
        
        # Find most attended positions
        attention_sum = np.sum(attention_weights, axis=0)
        top_positions = np.argsort(attention_sum)[-5:][::-1]
        
        # Create visualization data
        viz_data = {
            'heatmap': attention_weights.tolist(),
            'labels': sequence_labels,
            'top_attended_positions': top_positions.tolist(),
            'attention_distribution': attention_sum.tolist(),
            'max_attention': float(np.max(attention_weights)),
            'mean_attention': float(np.mean(attention_weights))
        }
        
        return viz_data
    
    def analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Analyze attention patterns for insights"""
        analysis = {}
        
        # Temporal focus analysis
        if len(attention_weights.shape) >= 2:
            temporal_focus = np.mean(attention_weights, axis=0)
            
            # Find peaks in attention
            peaks = []
            for i in range(1, len(temporal_focus) - 1):
                if temporal_focus[i] > temporal_focus[i-1] and temporal_focus[i] > temporal_focus[i+1]:
                    peaks.append(i)
            
            analysis['attention_peaks'] = peaks
            analysis['peak_count'] = len(peaks)
            
            # Calculate attention entropy
            normalized = temporal_focus / (np.sum(temporal_focus) + 1e-8)
            entropy = -np.sum(normalized * np.log(normalized + 1e-8))
            analysis['attention_entropy'] = float(entropy)
            
            # Attention concentration (Gini coefficient)
            sorted_attention = np.sort(temporal_focus)
            n = len(sorted_attention)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_attention)) / (n * np.sum(sorted_attention)) - (n + 1) / n
            analysis['attention_concentration'] = float(gini)
        
        return analysis


class DecisionTreeExtractor:
    """
    Extract and explain decision paths from tree-based models
    Patent Requirement: Decision tree path extraction
    """
    
    def __init__(self):
        self.tree_rules = []
        
    def extract_decision_path(self, model: Any, X: np.ndarray) -> List[Dict[str, Any]]:
        """Extract decision path for a sample through tree-based model"""
        if not hasattr(model, 'tree_') and not hasattr(model, 'estimators_'):
            logger.warning("Model does not appear to be tree-based")
            return []
        
        paths = []
        
        if hasattr(model, 'tree_'):
            # Single tree
            path = self._extract_single_tree_path(model.tree_, X[0])
            paths.append(path)
        elif hasattr(model, 'estimators_'):
            # Ensemble of trees
            for estimator in model.estimators_[:5]:  # Limit to first 5 trees
                if hasattr(estimator, 'tree_'):
                    path = self._extract_single_tree_path(estimator.tree_, X[0])
                    paths.append(path)
        
        return paths
    
    def _extract_single_tree_path(self, tree, sample: np.ndarray) -> Dict[str, Any]:
        """Extract path through a single decision tree"""
        path = []
        node = 0  # Start at root
        
        while tree.feature[node] != -2:  # -2 indicates leaf node
            feature_idx = tree.feature[node]
            threshold = tree.threshold[node]
            feature_value = sample[feature_idx]
            
            decision = {
                'node_id': int(node),
                'feature_index': int(feature_idx),
                'threshold': float(threshold),
                'feature_value': float(feature_value),
                'direction': 'left' if feature_value <= threshold else 'right'
            }
            path.append(decision)
            
            # Move to next node
            if feature_value <= threshold:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        
        # Add leaf information
        leaf_info = {
            'node_id': int(node),
            'prediction': float(tree.value[node][0][0]) if len(tree.value[node][0]) > 0 else 0,
            'samples': int(tree.n_node_samples[node]),
            'impurity': float(tree.impurity[node])
        }
        
        return {
            'path': path,
            'leaf': leaf_info,
            'depth': len(path)
        }
    
    def generate_rule_set(self, paths: List[Dict[str, Any]], 
                         feature_names: List[str]) -> List[str]:
        """Generate human-readable rules from decision paths"""
        rules = []
        
        for path_data in paths:
            rule_parts = []
            for decision in path_data['path']:
                feature_idx = decision['feature_index']
                if feature_idx < len(feature_names):
                    feature = feature_names[feature_idx]
                else:
                    feature = f"feature_{feature_idx}"
                
                threshold = decision['threshold']
                direction = decision['direction']
                
                if direction == 'left':
                    rule_parts.append(f"{feature} <= {threshold:.3f}")
                else:
                    rule_parts.append(f"{feature} > {threshold:.3f}")
            
            if rule_parts:
                prediction = path_data['leaf']['prediction']
                confidence = path_data['leaf']['samples'] / 100.0  # Normalized confidence
                rule = f"IF {' AND '.join(rule_parts)} THEN prediction={prediction:.3f} (confidence={confidence:.2f})"
                rules.append(rule)
        
        return rules


class ExplainabilityEngine:
    """
    Main explainability engine orchestrating all explanation components
    Provides unified interface for model interpretability
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.shap_explainer = SHAPExplainer()
        self.attention_visualizer = AttentionVisualizer()
        self.tree_extractor = DecisionTreeExtractor()
        self.explanation_cache = {}
        
    def initialize(self, models: Dict[str, Any], background_data: np.ndarray):
        """Initialize all explainability components"""
        logger.info("Initializing Explainability Engine...")
        self.shap_explainer.initialize(models, background_data, self.feature_names)
        logger.info("Explainability engine ready")
    
    def explain_prediction(self, X: np.ndarray, model_name: str, 
                          model: Optional[Any] = None,
                          include_interactions: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction
        Patent Requirements: SHAP + Attention + Decision paths
        """
        result = {
            'model': model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # SHAP explanation
        try:
            shap_result = self.shap_explainer.explain_instance(X, model_name)
            result['shap'] = {
                'feature_importance': shap_result.feature_importance,
                'base_value': shap_result.base_value,
                'prediction': shap_result.prediction,
                'confidence': shap_result.confidence,
                'visualization': shap_result.visualization_data
            }
            
            # Top contributing features
            top_features = list(shap_result.feature_importance.keys())[:5]
            result['top_features'] = [
                {
                    'name': f,
                    'contribution': shap_result.feature_importance[f],
                    'direction': 'positive' if shap_result.feature_importance[f] > 0 else 'negative'
                }
                for f in top_features
            ]
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            result['shap'] = {'error': str(e)}
        
        # Interaction effects if requested
        if include_interactions:
            try:
                interactions = self.shap_explainer.calculate_interaction_effects(X, model_name)
                result['interactions'] = interactions
            except Exception as e:
                logger.error(f"Interaction calculation failed: {e}")
                result['interactions'] = {'error': str(e)}
        
        # Attention visualization if model is neural
        if model and isinstance(model, nn.Module):
            try:
                X_tensor = torch.FloatTensor(X)
                attention_weights = self.attention_visualizer.extract_attention_weights(model, X_tensor)
                
                if len(attention_weights) > 0:
                    viz_data = self.attention_visualizer.visualize_temporal_attention(attention_weights)
                    analysis = self.attention_visualizer.analyze_attention_patterns(attention_weights)
                    
                    result['attention'] = {
                        'visualization': viz_data,
                        'analysis': analysis
                    }
            except Exception as e:
                logger.error(f"Attention extraction failed: {e}")
                result['attention'] = {'error': str(e)}
        
        # Decision tree paths if applicable
        if model and hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            try:
                paths = self.tree_extractor.extract_decision_path(model, X)
                rules = self.tree_extractor.generate_rule_set(paths, self.feature_names)
                
                result['decision_paths'] = {
                    'paths': paths[:3],  # Limit to top 3 paths
                    'rules': rules[:3]   # Limit to top 3 rules
                }
            except Exception as e:
                logger.error(f"Decision path extraction failed: {e}")
                result['decision_paths'] = {'error': str(e)}
        
        # Natural language explanation
        result['explanation_text'] = self._generate_natural_language_explanation(result)
        
        # Cache explanation
        cache_key = f"{model_name}_{hash(X.tobytes())}"
        self.explanation_cache[cache_key] = result
        
        return result
    
    def _generate_natural_language_explanation(self, explanation_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation text"""
        parts = []
        
        # Start with prediction
        if 'shap' in explanation_data and 'prediction' in explanation_data['shap']:
            pred = explanation_data['shap']['prediction']
            conf = explanation_data['shap'].get('confidence', 0.5)
            parts.append(f"The model predicts {'compliance' if pred > 0.5 else 'non-compliance'} "
                        f"with {conf*100:.1f}% confidence.")
        
        # Add top features
        if 'top_features' in explanation_data:
            top_3 = explanation_data['top_features'][:3]
            if top_3:
                feature_text = []
                for f in top_3:
                    direction = "increases" if f['direction'] == 'positive' else "decreases"
                    feature_text.append(f"{f['name']} {direction} compliance likelihood")
                
                parts.append(f"Key factors: {', '.join(feature_text)}.")
        
        # Add attention insights if available
        if 'attention' in explanation_data and 'analysis' in explanation_data['attention']:
            analysis = explanation_data['attention']['analysis']
            if 'attention_peaks' in analysis and analysis['attention_peaks']:
                parts.append(f"The model focused on {len(analysis['attention_peaks'])} "
                           f"critical time points in the configuration history.")
        
        # Add decision rule if available
        if 'decision_paths' in explanation_data and 'rules' in explanation_data['decision_paths']:
            rules = explanation_data['decision_paths']['rules']
            if rules:
                parts.append(f"Primary decision rule: {rules[0]}")
        
        return " ".join(parts) if parts else "Explanation generation failed."
    
    def generate_report(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive explainability report for regulatory compliance"""
        report = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_predictions': len(explanations),
            'summary': {},
            'feature_analysis': {},
            'compliance_factors': []
        }
        
        # Aggregate feature importance
        all_features = {}
        for exp in explanations:
            if 'shap' in exp and 'feature_importance' in exp['shap']:
                for feature, value in exp['shap']['feature_importance'].items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(value)
        
        # Calculate aggregate statistics
        for feature, values in all_features.items():
            report['feature_analysis'][feature] = {
                'mean_importance': float(np.mean(np.abs(values))),
                'std_importance': float(np.std(values)),
                'positive_impact': sum(1 for v in values if v > 0) / len(values),
                'negative_impact': sum(1 for v in values if v < 0) / len(values)
            }
        
        # Sort by mean importance
        report['feature_analysis'] = dict(sorted(
            report['feature_analysis'].items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        ))
        
        # Identify key compliance factors
        top_features = list(report['feature_analysis'].keys())[:10]
        for feature in top_features:
            stats = report['feature_analysis'][feature]
            report['compliance_factors'].append({
                'feature': feature,
                'importance': stats['mean_importance'],
                'consistency': 1 - stats['std_importance'] / (stats['mean_importance'] + 1e-8),
                'direction': 'positive' if stats['positive_impact'] > stats['negative_impact'] else 'negative'
            })
        
        # Summary statistics
        report['summary'] = {
            'top_compliance_factor': top_features[0] if top_features else 'unknown',
            'average_confidence': float(np.mean([
                exp['shap'].get('confidence', 0.5) 
                for exp in explanations 
                if 'shap' in exp and 'confidence' in exp['shap']
            ])),
            'feature_consistency': float(np.mean([
                1 - f['std_importance'] / (f['mean_importance'] + 1e-8)
                for f in report['feature_analysis'].values()
            ]))
        }
        
        return report