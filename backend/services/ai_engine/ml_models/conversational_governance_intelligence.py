"""
Conversational Governance Intelligence System Implementation
Patent 3: System and Method for Conversational Governance Intelligence with 
Multi-Domain Natural Language Understanding and Policy Synthesis

This module implements the patented conversational AI system that enables:
- Domain-specific Natural Language Understanding (NLU)
- Intent classification for governance queries
- Policy synthesis from natural language descriptions
- Multi-turn conversation management with context awareness
- Query translation to governance API calls
- Response generation with governance expertise

Reference: docs/builddetails/PolicyCortex Detailed Technical Specifications.md (Lines 710-1102)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, BertTokenizer, BertForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog
import asyncio
import json
import re
from abc import ABC, abstractmethod
from enum import Enum
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.pytorch
from prometheus_client import Counter, Histogram, Gauge
import warnings
from pathlib import Path

logger = structlog.get_logger(__name__)

# Metrics for monitoring conversational AI
CONVERSATION_REQUESTS = Counter(
    'conversation_requests_total',
    'Total conversation requests',
    ['intent', 'domain', 'status']
)
INTENT_CLASSIFICATION_DURATION = Histogram(
    'intent_classification_duration_seconds',
    'Intent classification duration'
)
POLICY_SYNTHESIS_DURATION = Histogram(
    'policy_synthesis_duration_seconds',
    'Policy synthesis duration'
)
CONVERSATION_ACCURACY = Gauge(
    'conversation_accuracy',
    'Conversation understanding accuracy',
    ['domain']
)

class GovernanceIntent(Enum):
    """Governance-specific intent categories"""
    POLICY_QUERY = "policy_query"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_ANALYSIS = "security_analysis"
    COST_OPTIMIZATION = "cost_optimization"
    RESOURCE_MANAGEMENT = "resource_management"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    AUDIT_REQUEST = "audit_request"
    REMEDIATION_REQUEST = "remediation_request"
    REPORTING_REQUEST = "reporting_request"
    CONFIGURATION_CHANGE = "configuration_change"
    ALERT_INVESTIGATION = "alert_investigation"
    POLICY_CREATION = "policy_creation"
    UNKNOWN = "unknown"

class ConversationState(Enum):
    """Conversation state management"""
    INITIAL = "initial"
    CONTEXT_GATHERING = "context_gathering"
    PROCESSING = "processing"
    CLARIFICATION_NEEDED = "clarification_needed"
    READY_TO_EXECUTE = "ready_to_execute"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationConfig:
    """Configuration for Conversational Governance Intelligence"""
    
    # Model configurations
    intent_model_name: str = "microsoft/DialoGPT-medium"
    nlu_model_name: str = "bert-base-uncased"
    policy_synthesis_model: str = "t5-base"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # NLU parameters
    max_sequence_length: int = 512
    intent_confidence_threshold: float = 0.8
    entity_confidence_threshold: float = 0.7
    
    # Conversation parameters
    max_conversation_turns: int = 10
    context_window_size: int = 5
    session_timeout_minutes: int = 30
    
    # Policy synthesis parameters
    max_policy_length: int = 2048
    policy_temperature: float = 0.7
    policy_top_p: float = 0.9
    
    # Domain-specific vocabulary
    governance_domains: List[str] = field(default_factory=lambda: [
        "security", "compliance", "cost", "performance", "operations"
    ])
    
    # Intent examples for few-shot learning
    intent_examples: Dict[str, List[str]] = field(default_factory=lambda: {
        "policy_query": [
            "What are the current security policies for virtual machines?",
            "Show me the compliance requirements for data storage",
            "List all policies related to network security groups"
        ],
        "compliance_check": [
            "Are we compliant with SOX requirements?",
            "Check GDPR compliance for our data processing",
            "Verify PCI DSS compliance across all systems"
        ],
        "security_analysis": [
            "Analyze security posture for production environment",
            "What are the current security vulnerabilities?",
            "Show me security risk assessment for last month"
        ],
        "cost_optimization": [
            "How can we reduce our Azure spending?",
            "Identify underutilized resources",
            "Optimize costs for compute resources"
        ]
    })

@dataclass
class ConversationContext:
    """Context information for ongoing conversations"""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_intent: Optional[GovernanceIntent] = None
    current_domain: Optional[str] = None
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    state: ConversationState = ConversationState.INITIAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_interaction: datetime = field(default_factory=datetime.utcnow)
    clarification_needed: List[str] = field(default_factory=list)

class GovernanceEntityExtractor:
    """Extract governance-specific entities from text"""
    
    def __init__(self):
        # Load spaCy model for general NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found, using basic extraction")
            self.nlp = None
        
        # Governance-specific patterns
        self.patterns = {
            'resource_type': [
                r'\b(virtual machine|vm|storage account|network|database|key vault|app service)\b',
                r'\b(resource group|subscription|management group)\b'
            ],
            'azure_service': [
                r'\b(azure\s+\w+|cosmosdb|sql\s+database|blob\s+storage)\b',
                r'\b(application\s+gateway|load\s+balancer|vpn\s+gateway)\b'
            ],
            'compliance_standard': [
                r'\b(SOX|GDPR|HIPAA|PCI\s+DSS|ISO\s+27001|FedRAMP)\b',
                r'\b(SOC\s+2|NIST|CIS|Azure\s+Security\s+Benchmark)\b'
            ],
            'time_period': [
                r'\b(last\s+\d+\s+days?|past\s+week|this\s+month|last\s+quarter)\b',
                r'\b(yesterday|today|last\s+\d+\s+hours?)\b'
            ],
            'cost_metrics': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d+\s*(dollars?|USD|cents?)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract governance entities from text"""
        entities = {}
        
        # Extract using patterns
        for entity_type, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, text, re.IGNORECASE))
            if matches:
                entities[entity_type] = list(set(matches))
        
        # Extract using spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                ent_type = f"spacy_{ent.label_.lower()}"
                if ent_type not in entities:
                    entities[ent_type] = []
                entities[ent_type].append(ent.text)
        
        return entities

class IntentClassifier(nn.Module):
    """Neural network for governance intent classification"""
    
    def __init__(self, model_name: str, num_intents: int, config: ConversationConfig):
        super(IntentClassifier, self).__init__()
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Intent classification head
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_intents)
        )
        
        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(config.governance_domains))
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for intent and domain classification"""
        
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Classify intent and domain
        intent_logits = self.intent_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output)
        
        return {
            'intent_logits': intent_logits,
            'domain_logits': domain_logits,
            'embeddings': pooled_output
        }

class PolicySynthesizer:
    """Generate governance policies from natural language descriptions"""
    
    def __init__(self, model_name: str = "t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Policy templates for different domains
        self.policy_templates = {
            "security": {
                "network": """
                Create a network security policy that:
                - {requirements}
                - Includes proper ingress and egress rules
                - Implements least privilege access
                - Enables security monitoring and logging
                """,
                "identity": """
                Create an identity and access policy that:
                - {requirements}
                - Implements role-based access control
                - Includes multi-factor authentication
                - Follows zero-trust principles
                """
            },
            "compliance": {
                "data_protection": """
                Create a data protection policy that:
                - {requirements}
                - Ensures data encryption at rest and in transit
                - Implements data retention policies
                - Includes audit trail requirements
                """,
                "access_control": """
                Create an access control policy that:
                - {requirements}
                - Defines user access levels
                - Includes regular access reviews
                - Implements segregation of duties
                """
            }
        }
    
    def synthesize_policy(self, description: str, domain: str, 
                         policy_type: str = "general") -> Dict[str, Any]:
        """Synthesize a governance policy from natural language description"""
        
        try:
            # Prepare input for T5
            if domain in self.policy_templates and policy_type in self.policy_templates[domain]:
                template = self.policy_templates[domain][policy_type]
                input_text = f"generate policy: {template.format(requirements=description)}"
            else:
                input_text = f"generate {domain} policy: {description}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Generate policy
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=2048,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True
                )
            
            # Decode generated policy
            generated_policy = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process and structure the policy
            structured_policy = self._structure_policy(generated_policy, domain, policy_type)
            
            return {
                'policy_text': generated_policy,
                'structured_policy': structured_policy,
                'domain': domain,
                'policy_type': policy_type,
                'generated_at': datetime.utcnow().isoformat(),
                'confidence_score': self._calculate_policy_confidence(generated_policy)
            }
            
        except Exception as e:
            logger.error("Policy synthesis failed", error=str(e))
            return {
                'error': str(e),
                'success': False
            }
    
    def _structure_policy(self, policy_text: str, domain: str, policy_type: str) -> Dict[str, Any]:
        """Structure the generated policy into components"""
        
        structured = {
            'name': f"{domain}_{policy_type}_policy",
            'domain': domain,
            'type': policy_type,
            'description': policy_text,
            'rules': [],
            'conditions': [],
            'actions': []
        }
        
        # Extract rules, conditions, and actions using patterns
        rule_patterns = [
            r'- ([^-\n]+(?:must|shall|should|will)[^-\n]+)',
            r'\d+\.\s+([^0-9\n]+(?:must|shall|should|will)[^0-9\n]+)'
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            structured['rules'].extend([match.strip() for match in matches])
        
        # Extract conditions
        condition_patterns = [
            r'(?:if|when|where)\s+([^,\.\n]+)',
            r'under\s+(?:the\s+)?condition(?:s)?\s+that\s+([^,\.\n]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            structured['conditions'].extend([match.strip() for match in matches])
        
        # Extract actions
        action_patterns = [
            r'(?:then|shall|must|will)\s+((?:allow|deny|block|permit|restrict|enable|disable)[^,\.\n]*)',
            r'action:\s*([^,\.\n]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            structured['actions'].extend([match.strip() for match in matches])
        
        return structured
    
    def _calculate_policy_confidence(self, policy_text: str) -> float:
        """Calculate confidence score for generated policy"""
        
        # Simple heuristic based on policy completeness
        score = 0.5  # Base score
        
        # Check for key governance elements
        governance_keywords = [
            'must', 'shall', 'should', 'compliance', 'security',
            'access', 'authorization', 'audit', 'monitoring',
            'encryption', 'logging', 'review', 'approval'
        ]
        
        found_keywords = sum(1 for keyword in governance_keywords 
                           if keyword.lower() in policy_text.lower())
        
        score += min(found_keywords * 0.05, 0.3)  # Up to 0.3 boost
        
        # Check for structure
        if any(pattern in policy_text.lower() for pattern in ['if', 'then', 'when']):
            score += 0.1
        
        if any(pattern in policy_text for pattern in ['-', '1.', '2.', '3.']):
            score += 0.1
        
        return min(score, 1.0)

class QueryTranslator:
    """Translate natural language queries to API calls"""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        
        # API endpoint mappings
        self.api_mappings = {
            GovernanceIntent.POLICY_QUERY: {
                'endpoint': '/api/v1/azure/policies',
                'method': 'GET',
                'parameters': ['resource_type', 'domain', 'policy_name']
            },
            GovernanceIntent.COMPLIANCE_CHECK: {
                'endpoint': '/api/v1/azure/compliance/check',
                'method': 'POST',
                'parameters': ['standard', 'scope', 'resource_ids']
            },
            GovernanceIntent.SECURITY_ANALYSIS: {
                'endpoint': '/api/v1/ai/security/analyze',
                'method': 'POST',
                'parameters': ['scope', 'analysis_type', 'time_range']
            },
            GovernanceIntent.COST_OPTIMIZATION: {
                'endpoint': '/api/v1/ai/cost/optimize',
                'method': 'POST',
                'parameters': ['scope', 'optimization_type', 'constraints']
            },
            GovernanceIntent.RESOURCE_MANAGEMENT: {
                'endpoint': '/api/v1/azure/resources',
                'method': 'GET',
                'parameters': ['resource_type', 'resource_group', 'tags']
            }
        }
    
    def translate_query(self, intent: GovernanceIntent, entities: Dict[str, Any],
                       query_text: str) -> Dict[str, Any]:
        """Translate natural language query to API call specification"""
        
        if intent not in self.api_mappings:
            return {
                'error': f"No API mapping for intent: {intent}",
                'success': False
            }
        
        mapping = self.api_mappings[intent]
        
        # Build API call specification
        api_call = {
            'endpoint': mapping['endpoint'],
            'method': mapping['method'],
            'parameters': {},
            'headers': {
                'Content-Type': 'application/json',
                'X-Query-Source': 'conversational-ai'
            }
        }
        
        # Map entities to API parameters
        parameter_mapping = {
            'resource_type': entities.get('resource_type', []),
            'compliance_standard': entities.get('compliance_standard', []),
            'time_period': entities.get('time_period', []),
            'azure_service': entities.get('azure_service', [])
        }
        
        # Populate parameters based on intent
        if intent == GovernanceIntent.POLICY_QUERY:
            if parameter_mapping['resource_type']:
                api_call['parameters']['resource_type'] = parameter_mapping['resource_type'][0]
        
        elif intent == GovernanceIntent.COMPLIANCE_CHECK:
            if parameter_mapping['compliance_standard']:
                api_call['parameters']['standard'] = parameter_mapping['compliance_standard'][0]
        
        elif intent == GovernanceIntent.SECURITY_ANALYSIS:
            api_call['parameters']['analysis_type'] = 'comprehensive'
            if parameter_mapping['time_period']:
                api_call['parameters']['time_range'] = parameter_mapping['time_period'][0]
        
        elif intent == GovernanceIntent.COST_OPTIMIZATION:
            api_call['parameters']['optimization_type'] = 'automated'
            if parameter_mapping['azure_service']:
                api_call['parameters']['scope'] = parameter_mapping['azure_service']
        
        return {
            'api_call': api_call,
            'confidence': self._calculate_translation_confidence(entities, query_text),
            'success': True
        }
    
    def _calculate_translation_confidence(self, entities: Dict[str, Any], 
                                        query_text: str) -> float:
        """Calculate confidence in query translation"""
        
        base_confidence = 0.7
        
        # Boost confidence based on extracted entities
        entity_boost = min(len(entities) * 0.05, 0.2)
        
        # Reduce confidence for ambiguous queries
        ambiguous_phrases = ['maybe', 'possibly', 'not sure', 'think', 'probably']
        ambiguity_penalty = sum(0.05 for phrase in ambiguous_phrases 
                              if phrase in query_text.lower())
        
        confidence = base_confidence + entity_boost - ambiguity_penalty
        
        return max(0.1, min(confidence, 1.0))

class ResponseGenerator:
    """Generate contextual responses for governance queries"""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        
        # Response templates for different scenarios
        self.response_templates = {
            'successful_query': [
                "I found {count} {resource_type} that match your criteria. {details}",
                "Here are the {resource_type} results: {details}",
                "Based on your query, I retrieved {count} items: {details}"
            ],
            'policy_found': [
                "I found {count} policies related to {domain}: {policy_list}",
                "Here are the relevant {domain} policies: {policy_list}",
                "The following policies apply to your query: {policy_list}"
            ],
            'compliance_status': [
                "Compliance status for {standard}: {status}. {details}",
                "Your {standard} compliance is {status}. {details}",
                "{standard} assessment shows {status} compliance. {details}"
            ],
            'clarification_needed': [
                "I need more information to help you better. Could you please specify {missing_info}?",
                "To provide accurate results, please clarify {missing_info}.",
                "I'd like to help, but I need you to specify {missing_info}."
            ],
            'error': [
                "I encountered an issue processing your request: {error_message}",
                "Sorry, I couldn't complete your request due to: {error_message}",
                "There was a problem: {error_message}. Please try rephrasing your query."
            ]
        }
    
    def generate_response(self, query_result: Dict[str, Any], 
                         context: ConversationContext) -> Dict[str, Any]:
        """Generate contextual response based on query results"""
        
        try:
            if not query_result.get('success', False):
                return self._generate_error_response(query_result, context)
            
            # Determine response type based on intent
            if context.current_intent == GovernanceIntent.POLICY_QUERY:
                return self._generate_policy_response(query_result, context)
            
            elif context.current_intent == GovernanceIntent.COMPLIANCE_CHECK:
                return self._generate_compliance_response(query_result, context)
            
            elif context.current_intent in [GovernanceIntent.SECURITY_ANALYSIS, 
                                          GovernanceIntent.COST_OPTIMIZATION]:
                return self._generate_analysis_response(query_result, context)
            
            else:
                return self._generate_generic_response(query_result, context)
        
        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return {
                'response': "I apologize, but I encountered an error generating a response. Please try again.",
                'response_type': 'error',
                'success': False
            }
    
    def _generate_policy_response(self, query_result: Dict[str, Any],
                                context: ConversationContext) -> Dict[str, Any]:
        """Generate response for policy queries"""
        
        policies = query_result.get('data', {}).get('policies', [])
        count = len(policies)
        
        if count == 0:
            response = "I didn't find any policies matching your criteria. You might want to check your search parameters or create a new policy."
        else:
            policy_list = ", ".join([p.get('name', 'Unnamed') for p in policies[:5]])
            if count > 5:
                policy_list += f" and {count - 5} more"
            
            template = np.random.choice(self.response_templates['policy_found'])
            response = template.format(
                count=count,
                domain=context.current_domain or 'governance',
                policy_list=policy_list
            )
        
        return {
            'response': response,
            'response_type': 'policy_query',
            'data': {'policies': policies, 'count': count},
            'success': True
        }
    
    def _generate_compliance_response(self, query_result: Dict[str, Any],
                                    context: ConversationContext) -> Dict[str, Any]:
        """Generate response for compliance queries"""
        
        compliance_data = query_result.get('data', {})
        standard = compliance_data.get('standard', 'Unknown')
        status = compliance_data.get('overall_status', 'Unknown')
        
        # Generate detailed compliance information
        details = []
        if 'compliant_controls' in compliance_data:
            details.append(f"{compliance_data['compliant_controls']} controls are compliant")
        if 'non_compliant_controls' in compliance_data:
            details.append(f"{compliance_data['non_compliant_controls']} controls need attention")
        
        details_text = ". ".join(details) if details else "See detailed report for more information"
        
        template = np.random.choice(self.response_templates['compliance_status'])
        response = template.format(
            standard=standard,
            status=status,
            details=details_text
        )
        
        return {
            'response': response,
            'response_type': 'compliance_check',
            'data': compliance_data,
            'success': True
        }
    
    def _generate_analysis_response(self, query_result: Dict[str, Any],
                                  context: ConversationContext) -> Dict[str, Any]:
        """Generate response for analysis queries"""
        
        analysis_data = query_result.get('data', {})
        analysis_type = context.current_intent.value.replace('_', ' ')
        
        # Extract key insights
        insights = analysis_data.get('insights', [])
        recommendations = analysis_data.get('recommendations', [])
        
        response_parts = [f"I've completed the {analysis_type}."]
        
        if insights:
            response_parts.append(f"Key findings: {'; '.join(insights[:3])}")
        
        if recommendations:
            response_parts.append(f"Recommendations: {'; '.join(recommendations[:2])}")
        
        response = " ".join(response_parts)
        
        return {
            'response': response,
            'response_type': 'analysis',
            'data': analysis_data,
            'success': True
        }
    
    def _generate_generic_response(self, query_result: Dict[str, Any],
                                 context: ConversationContext) -> Dict[str, Any]:
        """Generate generic response"""
        
        data = query_result.get('data', {})
        count = len(data.get('items', [])) if 'items' in data else 0
        
        if count == 0:
            response = "I didn't find any results matching your query. Please try refining your search criteria."
        else:
            template = np.random.choice(self.response_templates['successful_query'])
            response = template.format(
                count=count,
                resource_type="items",
                details=f"The results include various governance-related items"
            )
        
        return {
            'response': response,
            'response_type': 'generic',
            'data': data,
            'success': True
        }
    
    def _generate_error_response(self, query_result: Dict[str, Any],
                               context: ConversationContext) -> Dict[str, Any]:
        """Generate error response"""
        
        error_message = query_result.get('error', 'Unknown error occurred')
        template = np.random.choice(self.response_templates['error'])
        response = template.format(error_message=error_message)
        
        return {
            'response': response,
            'response_type': 'error',
            'success': False
        }

class ConversationalGovernanceIntelligence:
    """
    Main Conversational Governance Intelligence System
    Implements Patent 3: Conversational Governance Intelligence System
    """
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.entity_extractor = GovernanceEntityExtractor()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(
            model_name=config.nlu_model_name,
            num_intents=len(GovernanceIntent),
            config=config
        ).to(self.device)
        
        # Initialize policy synthesizer
        self.policy_synthesizer = PolicySynthesizer(config.policy_synthesis_model)
        
        # Initialize query translator and response generator
        self.query_translator = QueryTranslator(config)
        self.response_generator = ResponseGenerator(config)
        
        # Conversation session storage (in production, use Redis/database)
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        # Intent to index mapping
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(GovernanceIntent)}
        self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        
        logger.info("Conversational Governance Intelligence initialized",
                   device=str(self.device),
                   num_intents=len(GovernanceIntent))
    
    async def process_conversation(self, user_input: str, session_id: str,
                                 user_id: str) -> Dict[str, Any]:
        """
        Process conversational input and generate intelligent response
        
        Args:
            user_input: Natural language input from user
            session_id: Unique session identifier
            user_id: User identifier
            
        Returns:
            Conversation response with intent, entities, and actions
        """
        
        try:
            # Get or create conversation context
            context = self._get_or_create_context(session_id, user_id)
            
            # Update conversation history
            context.conversation_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_input': user_input,
                'turn': len(context.conversation_history) + 1
            })
            
            # Extract entities
            with INTENT_CLASSIFICATION_DURATION.time():
                entities = self.entity_extractor.extract_entities(user_input)
                intent_result = await self._classify_intent(user_input, context)
            
            # Update context with new information
            context.extracted_entities.update(entities)
            context.current_intent = intent_result['intent']
            context.current_domain = intent_result.get('domain')
            context.last_interaction = datetime.utcnow()
            
            # Check if we need clarification
            clarification = self._check_clarification_needed(context, entities)
            if clarification:
                context.state = ConversationState.CLARIFICATION_NEEDED
                return await self._handle_clarification(clarification, context)
            
            # Process the request based on intent
            context.state = ConversationState.PROCESSING
            
            if context.current_intent == GovernanceIntent.POLICY_CREATION:
                return await self._handle_policy_creation(user_input, context)
            else:
                return await self._handle_governance_query(user_input, context)
        
        except Exception as e:
            logger.error("Conversation processing failed", 
                        session_id=session_id, error=str(e))
            
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try again.",
                'intent': GovernanceIntent.UNKNOWN.value,
                'entities': {},
                'success': False,
                'error': str(e)
            }
    
    async def _classify_intent(self, user_input: str, 
                             context: ConversationContext) -> Dict[str, Any]:
        """Classify user intent and domain"""
        
        # Prepare input for the model
        inputs = self.intent_classifier.tokenizer(
            user_input,
            return_tensors="pt",
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get predictions
        self.intent_classifier.eval()
        with torch.no_grad():
            outputs = self.intent_classifier(**inputs)
        
        # Process intent prediction
        intent_probs = F.softmax(outputs['intent_logits'], dim=-1)
        intent_idx = torch.argmax(intent_probs, dim=-1).item()
        intent_confidence = intent_probs[0, intent_idx].item()
        
        # Process domain prediction
        domain_probs = F.softmax(outputs['domain_logits'], dim=-1)
        domain_idx = torch.argmax(domain_probs, dim=-1).item()
        domain_confidence = domain_probs[0, domain_idx].item()
        
        # Map indices to labels
        predicted_intent = self.idx_to_intent.get(intent_idx, GovernanceIntent.UNKNOWN)
        predicted_domain = self.config.governance_domains[domain_idx] if domain_idx < len(self.config.governance_domains) else None
        
        # Log metrics
        CONVERSATION_REQUESTS.labels(
            intent=predicted_intent.value,
            domain=predicted_domain or 'unknown',
            status='success'
        ).inc()
        
        return {
            'intent': predicted_intent,
            'intent_confidence': intent_confidence,
            'domain': predicted_domain,
            'domain_confidence': domain_confidence,
            'embeddings': outputs['embeddings'].cpu().numpy().tolist()
        }
    
    async def _handle_policy_creation(self, user_input: str,
                                    context: ConversationContext) -> Dict[str, Any]:
        """Handle policy creation requests"""
        
        with POLICY_SYNTHESIS_DURATION.time():
            # Extract policy requirements from user input
            policy_description = user_input
            domain = context.current_domain or 'general'
            
            # Synthesize policy
            policy_result = self.policy_synthesizer.synthesize_policy(
                policy_description, domain
            )
            
            if 'error' in policy_result:
                return {
                    'response': f"I couldn't generate a policy: {policy_result['error']}",
                    'intent': context.current_intent.value,
                    'success': False
                }
            
            # Generate response
            response = f"I've created a {domain} policy based on your requirements. Here's the generated policy:\n\n{policy_result['policy_text']}"
            
            # Update conversation context
            context.state = ConversationState.COMPLETED
            context.conversation_history[-1]['response'] = response
            context.conversation_history[-1]['policy_generated'] = policy_result
            
            return {
                'response': response,
                'intent': context.current_intent.value,
                'entities': context.extracted_entities,
                'policy_result': policy_result,
                'success': True
            }
    
    async def _handle_governance_query(self, user_input: str,
                                     context: ConversationContext) -> Dict[str, Any]:
        """Handle general governance queries"""
        
        # Translate query to API call
        translation_result = self.query_translator.translate_query(
            context.current_intent,
            context.extracted_entities,
            user_input
        )
        
        if not translation_result.get('success', False):
            return {
                'response': f"I couldn't process your query: {translation_result.get('error')}",
                'intent': context.current_intent.value,
                'success': False
            }
        
        # Simulate API call execution (in production, make actual API calls)
        query_result = await self._execute_simulated_query(
            translation_result['api_call'],
            context
        )
        
        # Generate response
        response_result = self.response_generator.generate_response(
            query_result, context
        )
        
        # Update conversation context
        context.state = ConversationState.COMPLETED
        context.conversation_history[-1]['response'] = response_result['response']
        context.conversation_history[-1]['api_call'] = translation_result['api_call']
        context.conversation_history[-1]['query_result'] = query_result
        
        return {
            'response': response_result['response'],
            'intent': context.current_intent.value,
            'entities': context.extracted_entities,
            'api_call': translation_result['api_call'],
            'data': response_result.get('data', {}),
            'success': response_result.get('success', True)
        }
    
    async def _execute_simulated_query(self, api_call: Dict[str, Any],
                                     context: ConversationContext) -> Dict[str, Any]:
        """Simulate API call execution (replace with actual API calls in production)"""
        
        # Simulate different response types based on endpoint
        endpoint = api_call['endpoint']
        
        if '/policies' in endpoint:
            return {
                'success': True,
                'data': {
                    'policies': [
                        {'name': 'Security Policy 1', 'type': 'network', 'status': 'active'},
                        {'name': 'Compliance Policy 2', 'type': 'data', 'status': 'active'}
                    ]
                }
            }
        
        elif '/compliance' in endpoint:
            return {
                'success': True,
                'data': {
                    'standard': 'GDPR',
                    'overall_status': 'Compliant',
                    'compliant_controls': 45,
                    'non_compliant_controls': 3
                }
            }
        
        elif '/security/analyze' in endpoint:
            return {
                'success': True,
                'data': {
                    'insights': [
                        'No critical vulnerabilities found',
                        'Network security groups properly configured',
                        'Identity access management needs review'
                    ],
                    'recommendations': [
                        'Enable multi-factor authentication',
                        'Update security group rules'
                    ]
                }
            }
        
        else:
            return {
                'success': True,
                'data': {
                    'items': ['Item 1', 'Item 2', 'Item 3']
                }
            }
    
    def _get_or_create_context(self, session_id: str, user_id: str) -> ConversationContext:
        """Get existing conversation context or create new one"""
        
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            
            # Check if session has expired
            if (datetime.utcnow() - context.last_interaction).total_seconds() > \
               (self.config.session_timeout_minutes * 60):
                del self.active_sessions[session_id]
                return ConversationContext(session_id=session_id, user_id=user_id)
            
            return context
        else:
            context = ConversationContext(session_id=session_id, user_id=user_id)
            self.active_sessions[session_id] = context
            return context
    
    def _check_clarification_needed(self, context: ConversationContext,
                                  entities: Dict[str, Any]) -> Optional[List[str]]:
        """Check if clarification is needed for the current query"""
        
        missing_info = []
        
        # Check based on intent
        if context.current_intent == GovernanceIntent.COMPLIANCE_CHECK:
            if not entities.get('compliance_standard'):
                missing_info.append("which compliance standard you want to check (e.g., GDPR, SOX)")
        
        elif context.current_intent == GovernanceIntent.POLICY_QUERY:
            if not entities.get('resource_type') and not entities.get('azure_service'):
                missing_info.append("which resources or services you're asking about")
        
        elif context.current_intent == GovernanceIntent.SECURITY_ANALYSIS:
            if not entities.get('time_period'):
                missing_info.append("the time period for the analysis (e.g., last 30 days)")
        
        return missing_info if missing_info else None
    
    async def _handle_clarification(self, missing_info: List[str],
                                  context: ConversationContext) -> Dict[str, Any]:
        """Handle clarification requests"""
        
        missing_text = " and ".join(missing_info)
        response = f"I need more information to help you better. Could you please specify {missing_text}?"
        
        return {
            'response': response,
            'intent': context.current_intent.value if context.current_intent else 'clarification',
            'entities': context.extracted_entities,
            'clarification_needed': missing_info,
            'success': True
        }
    
    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session"""
        
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            return {
                'session_id': session_id,
                'user_id': context.user_id,
                'history': context.conversation_history,
                'current_state': context.state.value,
                'entities': context.extracted_entities
            }
        else:
            return {
                'error': 'Session not found',
                'success': False
            }

# Factory function for creating conversational intelligence
def create_conversational_intelligence(config: Optional[ConversationConfig] = None) -> ConversationalGovernanceIntelligence:
    """Create a configured Conversational Governance Intelligence instance"""
    
    if config is None:
        config = ConversationConfig()
    
    return ConversationalGovernanceIntelligence(config)

# Global instance for the service
conversational_intelligence = create_conversational_intelligence()

if __name__ == "__main__":
    # Example usage and testing
    async def test_conversational_ai():
        """Test the conversational AI system"""
        
        test_queries = [
            "What are the current security policies for virtual machines?",
            "Check our GDPR compliance status",
            "How can we reduce Azure costs for compute resources?",
            "Create a network security policy that blocks all unauthorized access",
            "Show me security vulnerabilities from last month"
        ]
        
        session_id = "test_session_001"
        user_id = "test_user"
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Query {i+1}: {query} ---")
            
            result = await conversational_intelligence.process_conversation(
                query, session_id, user_id
            )
            
            print(f"Intent: {result.get('intent')}")
            print(f"Response: {result.get('response')}")
            print(f"Success: {result.get('success')}")
            
            if 'entities' in result and result['entities']:
                print(f"Entities: {result['entities']}")
    
    # Run test
    asyncio.run(test_conversational_ai())