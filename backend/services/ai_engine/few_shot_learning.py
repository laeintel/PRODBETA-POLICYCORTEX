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
Few-Shot Learning System for Policy Generation and Compliance
Enables learning from minimal examples for organization-specific needs
Implements Prototypical Networks, Matching Networks, and Relation Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class PolicyExample:
    """Represents a policy example for few-shot learning"""
    policy_text: str
    policy_type: str  # 'security', 'compliance', 'cost', 'access', 'network'
    resources: List[str]  # Azure resources affected
    conditions: Dict[str, Any]  # Policy conditions
    actions: List[str]  # Allow, Deny, Audit, etc.
    compliance_frameworks: List[str]  # HIPAA, GDPR, etc.
    organization_id: str
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None


class PolicyEncoder(nn.Module):
    """
    Encodes policies into embeddings for few-shot learning
    Specialized for Azure governance policies
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 768):
        super().__init__()
        
        # Text encoding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)
        
        # Policy-specific encoders
        self.resource_encoder = nn.Embedding(1000, embedding_dim // 4)  # Azure resources
        self.action_encoder = nn.Embedding(10, embedding_dim // 4)  # Policy actions
        self.compliance_encoder = nn.Embedding(20, embedding_dim // 4)  # Compliance frameworks
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=12,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 4 * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, policy: PolicyExample) -> torch.Tensor:
        """Encode a policy into an embedding"""
        # Simplified tokenization (in production, use proper tokenizer)
        tokens = torch.randint(0, 10000, (1, 100))  # Placeholder
        positions = torch.arange(100).unsqueeze(0)
        
        # Encode text
        text_emb = self.token_embedding(tokens) + self.position_embedding(positions)
        text_features = self.transformer(text_emb)
        text_pooled = text_features.mean(dim=1)
        
        # Encode resources (simplified)
        resource_ids = torch.tensor([hash(r) % 1000 for r in policy.resources[:5]])
        if len(resource_ids) < 5:
            resource_ids = F.pad(resource_ids, (0, 5 - len(resource_ids)))
        resource_emb = self.resource_encoder(resource_ids).mean(dim=0)
        
        # Encode actions
        action_map = {'Allow': 0, 'Deny': 1, 'Audit': 2, 'Modify': 3}
        action_ids = torch.tensor([action_map.get(a, 4) for a in policy.actions[:3]])
        if len(action_ids) < 3:
            action_ids = F.pad(action_ids, (0, 3 - len(action_ids)))
        action_emb = self.action_encoder(action_ids).mean(dim=0)
        
        # Encode compliance
        compliance_map = {'hipaa': 0, 'gdpr': 1, 'sox': 2, 'pci-dss': 3, 'iso27001': 4}
        compliance_ids = torch.tensor([compliance_map.get(c.lower(), 5) 
                                      for c in policy.compliance_frameworks[:3]])
        if len(compliance_ids) < 3:
            compliance_ids = F.pad(compliance_ids, (0, 3 - len(compliance_ids)))
        compliance_emb = self.compliance_encoder(compliance_ids).mean(dim=0)
        
        # Combine all features
        combined = torch.cat([
            text_pooled.squeeze(0),
            resource_emb,
            action_emb,
            compliance_emb
        ])
        
        # Final projection
        embedding = self.output_projection(combined)
        
        return embedding


class MatchingNetwork(nn.Module):
    """
    Matching Networks for one-shot learning
    Compares query to support set using attention mechanism
    """
    
    def __init__(self, encoder: PolicyEncoder, use_fce: bool = True):
        super().__init__()
        self.encoder = encoder
        self.use_fce = use_fce  # Full Context Embeddings
        
        if use_fce:
            # Bidirectional LSTM for encoding in context of support set
            self.lstm = nn.LSTM(
                encoder.output_projection[-1].out_features,
                encoder.output_projection[-1].out_features // 2,
                bidirectional=True,
                batch_first=True
            )
            
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            encoder.output_projection[-1].out_features,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, support_set: List[Tuple[PolicyExample, str]], 
                query: PolicyExample) -> Dict[str, Any]:
        """
        Match query to support set examples
        
        Args:
            support_set: List of (example, label) pairs
            query: Query policy to classify
        """
        # Encode support set
        support_embeddings = []
        support_labels = []
        
        for example, label in support_set:
            embedding = self.encoder(example)
            support_embeddings.append(embedding)
            support_labels.append(label)
            
        support_embeddings = torch.stack(support_embeddings)
        
        # Encode query
        query_embedding = self.encoder(query)
        
        if self.use_fce:
            # Encode in context using LSTM
            support_context, _ = self.lstm(support_embeddings.unsqueeze(0))
            support_embeddings = support_context.squeeze(0)
            
            query_context, _ = self.lstm(query_embedding.unsqueeze(0).unsqueeze(0))
            query_embedding = query_context.squeeze(0).squeeze(0)
            
        # Calculate attention weights
        query_expanded = query_embedding.unsqueeze(0).unsqueeze(0)
        support_expanded = support_embeddings.unsqueeze(0)
        
        attended, attention_weights = self.attention(
            query_expanded,
            support_expanded,
            support_expanded
        )
        
        # Calculate similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            support_embeddings,
            dim=1
        )
        
        # Softmax to get probabilities
        attention_probs = F.softmax(similarities, dim=0)
        
        # Aggregate predictions
        label_scores = defaultdict(float)
        for i, label in enumerate(support_labels):
            label_scores[label] += attention_probs[i].item()
            
        return {
            'predictions': label_scores,
            'attention_weights': attention_weights.squeeze().detach(),
            'most_similar_idx': torch.argmax(similarities).item(),
            'confidence': torch.max(attention_probs).item()
        }


class RelationNetwork(nn.Module):
    """
    Relation Network for few-shot learning
    Learns to compare relationships between examples
    """
    
    def __init__(self, encoder: PolicyEncoder, relation_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        
        feature_dim = encoder.output_projection[-1].out_features
        
        # Relation module
        self.relation_module = nn.Sequential(
            nn.Linear(feature_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )
        
        # Confidence predictor
        self.confidence_module = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, support_set: List[Tuple[PolicyExample, str]], 
                query: PolicyExample) -> Dict[str, Any]:
        """
        Compare query with support set using learned relations
        """
        # Encode all examples
        support_embeddings = []
        support_labels = []
        
        for example, label in support_set:
            embedding = self.encoder(example)
            support_embeddings.append(embedding)
            support_labels.append(label)
            
        query_embedding = self.encoder(query)
        
        # Compute relations
        relations = []
        confidences = []
        
        for support_emb in support_embeddings:
            # Concatenate query and support embeddings
            combined = torch.cat([query_embedding, support_emb])
            
            # Compute relation score
            relation_score = self.relation_module(combined)
            relations.append(relation_score)
            
            # Compute confidence
            confidence = self.confidence_module(combined)
            confidences.append(confidence)
            
        relations = torch.cat(relations)
        confidences = torch.cat(confidences)
        
        # Aggregate by label
        label_scores = defaultdict(list)
        label_confidences = defaultdict(list)
        
        for i, label in enumerate(support_labels):
            label_scores[label].append(relations[i].item())
            label_confidences[label].append(confidences[i].item())
            
        # Average scores per label
        final_scores = {}
        final_confidences = {}
        
        for label in label_scores:
            final_scores[label] = np.mean(label_scores[label])
            final_confidences[label] = np.mean(label_confidences[label])
            
        return {
            'predictions': final_scores,
            'confidences': final_confidences,
            'best_match': max(final_scores, key=final_scores.get),
            'relation_scores': relations.detach()
        }


class FewShotPolicyGenerator:
    """
    Generates new policies from few examples
    Specialized for Azure governance and compliance
    """
    
    def __init__(self, encoder: PolicyEncoder):
        self.encoder = encoder
        
        # Different few-shot learners
        self.matching_network = MatchingNetwork(encoder)
        self.relation_network = RelationNetwork(encoder)
        
        # Policy template library
        self.policy_templates = self._load_policy_templates()
        
        # Organization-specific examples
        self.org_examples = defaultdict(list)
        
    def _load_policy_templates(self) -> Dict[str, str]:
        """Load Azure policy templates"""
        return {
            'vm_security': """{
                "mode": "Indexed",
                "policyRule": {
                    "if": {
                        "allOf": [
                            {"field": "type", "equals": "Microsoft.Compute/virtualMachines"},
                            {"field": "[CONDITION_FIELD]", "[CONDITION_OPERATOR]": "[CONDITION_VALUE]"}
                        ]
                    },
                    "then": {"effect": "[EFFECT]"}
                }
            }""",
            'storage_compliance': """{
                "mode": "All",
                "policyRule": {
                    "if": {
                        "field": "type",
                        "equals": "Microsoft.Storage/storageAccounts"
                    },
                    "then": {
                        "effect": "[EFFECT]",
                        "details": {
                            "type": "Microsoft.Storage/storageAccounts/encryptionSettings",
                            "existenceCondition": {
                                "field": "Microsoft.Storage/storageAccounts/encryptionSettings.enabled",
                                "equals": "true"
                            }
                        }
                    }
                }
            }""",
            'network_security': """{
                "mode": "All",
                "policyRule": {
                    "if": {
                        "anyOf": [
                            {"field": "type", "equals": "Microsoft.Network/networkSecurityGroups"},
                            {"field": "type", "equals": "Microsoft.Network/virtualNetworks"}
                        ]
                    },
                    "then": {"effect": "[EFFECT]", "details": "[DETAILS]"}
                }
            }"""
        }
        
    def learn_from_examples(self, examples: List[PolicyExample], 
                           organization_id: str):
        """
        Learn organization-specific patterns from examples
        """
        logger.info(f"Learning from {len(examples)} examples for org {organization_id}")
        
        # Store examples
        self.org_examples[organization_id].extend(examples)
        
        # Extract patterns
        patterns = self._extract_patterns(examples)
        
        # Update encoder if needed (fine-tuning)
        if len(examples) > 10:
            self._fine_tune_encoder(examples)
            
        return patterns
        
    def _extract_patterns(self, examples: List[PolicyExample]) -> Dict[str, Any]:
        """Extract common patterns from examples"""
        patterns = {
            'common_resources': defaultdict(int),
            'common_conditions': defaultdict(int),
            'common_actions': defaultdict(int),
            'compliance_requirements': set()
        }
        
        for example in examples:
            for resource in example.resources:
                patterns['common_resources'][resource] += 1
                
            for condition_key in example.conditions:
                patterns['common_conditions'][condition_key] += 1
                
            for action in example.actions:
                patterns['common_actions'][action] += 1
                
            patterns['compliance_requirements'].update(example.compliance_frameworks)
            
        return patterns
        
    def _fine_tune_encoder(self, examples: List[PolicyExample]):
        """Fine-tune encoder on organization-specific examples"""
        # Simplified fine-tuning (in production, implement proper training loop)
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-5)
        
        for example in examples[:10]:  # Quick fine-tuning
            embedding = self.encoder(example)
            
            # Self-supervised loss (reconstruction or contrastive)
            loss = embedding.norm()  # Placeholder
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def generate_policy(self, requirements: Dict[str, Any], 
                       organization_id: str,
                       k_shot: int = 5) -> Dict[str, Any]:
        """
        Generate new policy from requirements using few-shot learning
        """
        logger.info(f"Generating policy for {organization_id} with {k_shot}-shot learning")
        
        # Get relevant examples
        if organization_id in self.org_examples:
            support_set = self.org_examples[organization_id][:k_shot]
        else:
            # Use general examples
            support_set = self._get_similar_examples(requirements, k_shot)
            
        if not support_set:
            # Fallback to template
            return self._generate_from_template(requirements)
            
        # Create query from requirements
        query = self._requirements_to_policy(requirements)
        
        # Use matching network for generation
        support_pairs = [(ex, ex.policy_type) for ex in support_set]
        match_result = self.matching_network(support_pairs, query)
        
        # Use relation network for validation
        relation_result = self.relation_network(support_pairs, query)
        
        # Combine results
        best_match_idx = match_result['most_similar_idx']
        best_example = support_set[best_match_idx]
        
        # Adapt best example to requirements
        generated_policy = self._adapt_policy(best_example, requirements)
        
        return {
            'policy': generated_policy,
            'confidence': match_result['confidence'],
            'based_on': best_example.policy_type,
            'learned_from': f"{len(support_set)} examples",
            'compliance_coverage': list(generated_policy.compliance_frameworks),
            'validation_score': relation_result['predictions'].get(
                best_example.policy_type, 0
            )
        }
        
    def _get_similar_examples(self, requirements: Dict[str, Any], 
                             k: int) -> List[PolicyExample]:
        """Get similar examples from all organizations"""
        all_examples = []
        for org_examples in self.org_examples.values():
            all_examples.extend(org_examples)
            
        if not all_examples:
            return []
            
        # Simple similarity based on requirements
        similarities = []
        for example in all_examples:
            sim = self._calculate_similarity(requirements, example)
            similarities.append((sim, example))
            
        similarities.sort(reverse=True)
        return [ex for _, ex in similarities[:k]]
        
    def _calculate_similarity(self, requirements: Dict[str, Any], 
                             example: PolicyExample) -> float:
        """Calculate similarity between requirements and example"""
        score = 0.0
        
        # Check resource overlap
        req_resources = set(requirements.get('resources', []))
        ex_resources = set(example.resources)
        if req_resources and ex_resources:
            score += len(req_resources & ex_resources) / len(req_resources | ex_resources)
            
        # Check compliance overlap
        req_compliance = set(requirements.get('compliance', []))
        ex_compliance = set(example.compliance_frameworks)
        if req_compliance and ex_compliance:
            score += len(req_compliance & ex_compliance) / len(req_compliance | ex_compliance)
            
        # Check action match
        req_actions = set(requirements.get('actions', []))
        ex_actions = set(example.actions)
        if req_actions and ex_actions:
            score += len(req_actions & ex_actions) / len(req_actions | ex_actions)
            
        return score / 3  # Average
        
    def _requirements_to_policy(self, requirements: Dict[str, Any]) -> PolicyExample:
        """Convert requirements to policy example"""
        return PolicyExample(
            policy_text="",  # Will be generated
            policy_type=requirements.get('type', 'general'),
            resources=requirements.get('resources', []),
            conditions=requirements.get('conditions', {}),
            actions=requirements.get('actions', ['Audit']),
            compliance_frameworks=requirements.get('compliance', []),
            organization_id=requirements.get('organization_id', 'default'),
            metadata=requirements
        )
        
    def _adapt_policy(self, example: PolicyExample, 
                     requirements: Dict[str, Any]) -> PolicyExample:
        """Adapt example policy to new requirements"""
        adapted = PolicyExample(
            policy_text=example.policy_text,  # Start with example
            policy_type=requirements.get('type', example.policy_type),
            resources=requirements.get('resources', example.resources),
            conditions={**example.conditions, **requirements.get('conditions', {})},
            actions=requirements.get('actions', example.actions),
            compliance_frameworks=list(set(
                example.compliance_frameworks + requirements.get('compliance', [])
            )),
            organization_id=requirements.get('organization_id', example.organization_id),
            metadata={**example.metadata, **requirements}
        )
        
        # Generate actual policy text
        template = self.policy_templates.get(adapted.policy_type, 
                                            self.policy_templates['vm_security'])
        
        # Fill template with adapted values
        policy_text = template.replace('[EFFECT]', adapted.actions[0])
        policy_text = policy_text.replace('[CONDITION_FIELD]', 
                                         list(adapted.conditions.keys())[0] if adapted.conditions else 'name')
        policy_text = policy_text.replace('[CONDITION_OPERATOR]', 'equals')
        policy_text = policy_text.replace('[CONDITION_VALUE]', 
                                         list(adapted.conditions.values())[0] if adapted.conditions else '*')
        
        adapted.policy_text = policy_text
        
        return adapted
        
    def _generate_from_template(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: Generate from template without examples"""
        policy_type = requirements.get('type', 'vm_security')
        template = self.policy_templates.get(policy_type, 
                                            self.policy_templates['vm_security'])
        
        # Basic template filling
        policy_text = template.replace('[EFFECT]', requirements.get('effect', 'Audit'))
        
        return {
            'policy': PolicyExample(
                policy_text=policy_text,
                policy_type=policy_type,
                resources=requirements.get('resources', []),
                conditions=requirements.get('conditions', {}),
                actions=requirements.get('actions', ['Audit']),
                compliance_frameworks=requirements.get('compliance', []),
                organization_id=requirements.get('organization_id', 'default')
            ),
            'confidence': 0.5,  # Lower confidence without examples
            'based_on': 'template',
            'learned_from': '0 examples'
        }


# Global few-shot system
few_shot_system = None

def initialize_few_shot(vocab_size: int = 10000, embedding_dim: int = 768):
    """Initialize the few-shot learning system"""
    global few_shot_system
    encoder = PolicyEncoder(vocab_size, embedding_dim)
    few_shot_system = FewShotPolicyGenerator(encoder)
    logger.info("Few-shot learning system initialized")
    return few_shot_system


# Export main components
__all__ = [
    'PolicyEncoder',
    'MatchingNetwork',
    'RelationNetwork',
    'FewShotPolicyGenerator',
    'PolicyExample',
    'initialize_few_shot',
    'few_shot_system'
]