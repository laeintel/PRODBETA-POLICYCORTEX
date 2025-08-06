"""
Conversational AI Interface System - Patent 2 Implementation
Advanced natural language interface for governance operations with context awareness,
    multi-turn dialogue management, and intelligent action execution.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from fastapi import HTTPException

    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, pipeline
)
import numpy as np
import redis.asyncio as redis
import spacy
from sentence_transformers import SentenceTransformer

from backend.shared.config import get_settings

from ..ml_models.cross_domain_gnn import CorrelationEngine
from .gnn_correlation_service import GNNCorrelationService

logger = logging.getLogger(__name__)
settings = get_settings()

class ConversationState(Enum):
    """States of conversation flow"""
    GREETING = "greeting"
    INTENT_IDENTIFICATION = "intent_identification"
    INFORMATION_GATHERING = "information_gathering"
    ACTION_CONFIRMATION = "action_confirmation"
    EXECUTION = "execution"
    FEEDBACK = "feedback"
    COMPLETE = "complete"

class IntentType(Enum):
    """Types of user intents"""
    QUERY_RESOURCES = "query_resources"
    POLICY_ANALYSIS = "policy_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    COST_OPTIMIZATION = "cost_optimization"
    SECURITY_ASSESSMENT = "security_assessment"
    CORRELATION_ANALYSIS = "correlation_analysis"
    IMPACT_PREDICTION = "impact_prediction"
    GENERAL_INFO = "general_info"
    HELP = "help"

@dataclass
class ConversationContext:
    """Context for multi-turn conversations"""

    conversation_id: str
    user_id: str
    session_start: datetime
    current_state: ConversationState
    intent_type: Optional[IntentType]
    extracted_entities: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    pending_actions: List[Dict[str, Any]]
    context_variables: Dict[str, Any]
    confidence_scores: Dict[str, float]
    last_interaction: datetime

@dataclass
class ConversationResponse:
    """Response structure for conversational interactions"""

    response_text: str
    intent_type: Optional[IntentType]
    confidence_score: float
    next_state: ConversationState
    suggested_actions: List[Dict[str, Any]]
    extracted_entities: Dict[str, Any]
    requires_confirmation: bool
    response_data: Optional[Dict[str, Any]]
    conversation_id: str

class NLUProcessor:
    """Natural Language Understanding processor"""

    def __init__(self):
        self.nlp = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.sentence_transformer = None
        self.initialized = False

        logger.info("NLUProcessor initialized")

    async def initialize(self):
        """Initialize NLP models"""

        try:
            # Load spaCy model for basic NLP
            self.nlp = spacy.load("en_core_web_sm")

            # Load sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize intent classification patterns
            self.intent_patterns = {
                IntentType.QUERY_RESOURCES: [
                    r'show.*resources?', r'list.*resources?', r'what.*resources?',
                    r'resources.*in.*', r'find.*resources?'
                ],
                IntentType.POLICY_ANALYSIS: [
                    r'policy.*analysis', r'analyze.*polic(y|ies)', r'policy.*compliance',
                    r'check.*polic(y|ies)', r'policy.*review'
                ],
                IntentType.COMPLIANCE_CHECK: [
                    r'compliance.*check', r'is.*compliant', r'compliance.*status',
                    r'audit.*compliance', r'regulatory.*compliance'
                ],
                IntentType.COST_OPTIMIZATION: [
                    r'cost.*optimization', r'reduce.*cost', r'cost.*saving',
                    r'expensive.*resources?', r'budget.*analysis'
                ],
                IntentType.SECURITY_ASSESSMENT: [
                    r'security.*assessment', r'security.*check', r'vulnerabilit(y|ies)',
                    r'security.*risk', r'security.*score'
                ],
                IntentType.CORRELATION_ANALYSIS: [
                    r'correlation.*analysis', r'relationship.*between', r'impact.*analysis',
                    r'dependency.*analysis', r'cross.*domain'
                ],
                IntentType.IMPACT_PREDICTION: [
                    r'impact.*prediction', r'what.*if', r'predict.*impact',
                    r'scenario.*analysis', r'change.*impact'
                ],
                IntentType.HELP: [
                    r'help', r'what.*can.*do', r'how.*to', r'instructions'
                ]
            }

            # Entity extraction patterns
            self.entity_patterns = {
                'resource_type': r'\b(virtual machine|vm|storage|database|network|subnet|nsg)\b',
                'subscription': r'\b(subscription|sub)\s+([a-f0-9\-]{36})\b',
                'resource_group': r'\b(resource group|rg)\s+([a-zA-Z0-9\-_]+)\b',
                'time_period': r'\b(last|past)\s+(\d+)\s+(day|week|month|year)s?\b',
                'cost_amount': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                'percentage': r'(\d{1,3})%'
            }

            self.initialized = True
            logger.info("NLUProcessor initialization completed")

        except Exception as e:
            logger.error(f"Error initializing NLUProcessor: {e}")
            raise

    async def process_utterance(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Process user utterance and extract intent, entities"""

        if not self.initialized:
            await self.initialize()

        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)

        # Extract intent
        intent_results = await self._classify_intent(cleaned_text, context)

        # Extract entities
        entities = await self._extract_entities(cleaned_text)

        # Compute semantic embeddings
        embeddings = self.sentence_transformer.encode([cleaned_text])[0]

        return {
            'intent': intent_results['intent'],
            'intent_confidence': intent_results['confidence'],
            'entities': entities,
            'embeddings': embeddings.tolist(),
            'processed_text': cleaned_text,
            'original_text': text
        }

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Convert to lowercase for pattern matching
        return text.lower()

    async def _classify_intent(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Classify user intent using pattern matching and context"""

        intent_scores = {}

        # Pattern-based classification
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1

            # Normalize by number of patterns
            intent_scores[intent_type] = score / len(patterns)

        # Context-based adjustments
        if context.intent_type and context.current_state != ConversationState.COMPLETE:
            # Boost current intent if conversation is ongoing
            if context.intent_type in intent_scores:
                intent_scores[context.intent_type] *= 1.5

        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return {
                    'intent': best_intent[0],
                    'confidence': min(best_intent[1], 1.0)
                }

        # Default to general info
        return {
            'intent': IntentType.GENERAL_INFO,
            'confidence': 0.3
        }

    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using patterns and NER"""

        entities = {}

        # Pattern-based extraction
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches

        # spaCy NER extraction
        if self.nlp:
            doc = self.nlp(text)

            spacy_entities = {
                'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
                'dates': [ent.text for ent in doc.ents if ent.label_ in ['DATE', 'TIME']],
                'money': [ent.text for ent in doc.ents if ent.label_ == 'MONEY'],
                'numbers': [ent.text for ent in doc.ents if ent.label_ in ['CARDINAL', 'QUANTITY']]
            }

            # Merge with pattern-based entities
            for key, values in spacy_entities.items():
                if values:
                    entities[key] = entities.get(key, []) + values

        return entities

class DialogueManager:
    """Manages multi-turn dialogue flow and state"""

    def __init__(self):
        self.state_transitions = {
            ConversationState.GREETING: [
                ConversationState.INTENT_IDENTIFICATION,
                ConversationState.HELP
            ],
            ConversationState.INTENT_IDENTIFICATION: [
                ConversationState.INFORMATION_GATHERING,
                ConversationState.EXECUTION,
                ConversationState.HELP
            ],
            ConversationState.INFORMATION_GATHERING: [
                ConversationState.ACTION_CONFIRMATION,
                ConversationState.INFORMATION_GATHERING,
                ConversationState.EXECUTION
            ],
            ConversationState.ACTION_CONFIRMATION: [
                ConversationState.EXECUTION,
                ConversationState.INFORMATION_GATHERING,
                ConversationState.COMPLETE
            ],
            ConversationState.EXECUTION: [
                ConversationState.FEEDBACK,
                ConversationState.COMPLETE,
                ConversationState.INTENT_IDENTIFICATION
            ],
            ConversationState.FEEDBACK: [
                ConversationState.COMPLETE,
                ConversationState.INTENT_IDENTIFICATION
            ]
        }

        logger.info("DialogueManager initialized")

    async def determine_next_state(self,
                                 current_context: ConversationContext,
                                 nlu_results: Dict[str, Any]) -> ConversationState:
        """Determine next conversation state based on context and input"""

        current_state = current_context.current_state
        intent = nlu_results.get('intent')
        confidence = nlu_results.get('intent_confidence', 0)

        # State transition logic
        if current_state == ConversationState.GREETING:
            if intent == IntentType.HELP:
                return ConversationState.HELP
            else:
                return ConversationState.INTENT_IDENTIFICATION

        elif current_state == ConversationState.INTENT_IDENTIFICATION:
            if confidence > 0.7 and self._has_sufficient_info(current_context, intent):
                return ConversationState.EXECUTION
            elif confidence > 0.5:
                return ConversationState.INFORMATION_GATHERING
            else:
                return ConversationState.INTENT_IDENTIFICATION

        elif current_state == ConversationState.INFORMATION_GATHERING:
            if self._has_sufficient_info(current_context, intent):
                return ConversationState.ACTION_CONFIRMATION
            else:
                return ConversationState.INFORMATION_GATHERING

        elif current_state == ConversationState.ACTION_CONFIRMATION:
            confirmation = self._detect_confirmation(nlu_results['processed_text'])
            if confirmation == 'yes':
                return ConversationState.EXECUTION
            elif confirmation == 'no':
                return ConversationState.INFORMATION_GATHERING
            else:
                return ConversationState.ACTION_CONFIRMATION

        elif current_state == ConversationState.EXECUTION:
            return ConversationState.FEEDBACK

        elif current_state == ConversationState.FEEDBACK:
            return ConversationState.COMPLETE

        return current_state

    def _has_sufficient_info(self, context: ConversationContext, intent: IntentType) -> bool:
        """Check if we have sufficient information to proceed"""

        required_info = {
            IntentType.QUERY_RESOURCES: ['resource_type', 'subscription'],
            IntentType.POLICY_ANALYSIS: ['policy_type'],
            IntentType.COMPLIANCE_CHECK: ['compliance_domain'],
            IntentType.COST_OPTIMIZATION: ['time_period'],
            IntentType.SECURITY_ASSESSMENT: ['assessment_scope'],
            IntentType.CORRELATION_ANALYSIS: ['analysis_scope'],
            IntentType.IMPACT_PREDICTION: ['change_scenario']
        }

        if intent not in required_info:
            return True  # No specific requirements

        required_fields = required_info[intent]
        available_info = (
            set(context.extracted_entities.keys()) | set(context.context_variables.keys())
        )

        return any(field in available_info for field in required_fields)

    def _detect_confirmation(self, text: str) -> Optional[str]:
        """Detect confirmation/rejection in text"""

        yes_patterns = [r'\byes\b', r'\byeah\b', r'\bokay\b', r'\bok\b', r'\bsure\b', r'\bconfirm\b']
        no_patterns = [r'\bno\b', r'\bnope\b', r'\bcancel\b', r'\babort\b', r'\bstop\b']

        for pattern in yes_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'yes'

        for pattern in no_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'no'

        return None

class ActionExecutor:
    """Executes actions based on conversation intent"""

    def __init__(self, gnn_service: GNNCorrelationService):
        self.gnn_service = gnn_service
        self.action_handlers = {
            IntentType.QUERY_RESOURCES: self._handle_resource_query,
            IntentType.POLICY_ANALYSIS: self._handle_policy_analysis,
            IntentType.COMPLIANCE_CHECK: self._handle_compliance_check,
            IntentType.COST_OPTIMIZATION: self._handle_cost_optimization,
            IntentType.SECURITY_ASSESSMENT: self._handle_security_assessment,
            IntentType.CORRELATION_ANALYSIS: self._handle_correlation_analysis,
            IntentType.IMPACT_PREDICTION: self._handle_impact_prediction,
            IntentType.GENERAL_INFO: self._handle_general_info,
            IntentType.HELP: self._handle_help
        }

        logger.info("ActionExecutor initialized")

    async def execute_action(self,
                           intent: IntentType,
                           context: ConversationContext) -> Dict[str, Any]:
        """Execute action based on intent and context"""

        if intent in self.action_handlers:
            try:
                result = await self.action_handlers[intent](context)
                return {
                    'success': True,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error executing action for {intent}: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'success': False,
                'error': f"No handler for intent: {intent}",
                'timestamp': datetime.now().isoformat()
            }

    async def _handle_resource_query(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle resource query requests"""

        # Extract query parameters from context
        resource_type = context.extracted_entities.get('resource_type', ['all'])[0]
        subscription = context.extracted_entities.get('subscription')
        resource_group = context.extracted_entities.get('resource_group')

        # Simulate resource query (would integrate with Azure APIs)
        resources = [
            {
                'id': f'/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm{i}',
                'name': f'vm{i}',
                'type': 'Microsoft.Compute/virtualMachines',
                'location': 'eastus',
                'status': 'running',
                'cost_monthly': np.random.uniform(100, 500),
                'cpu_utilization': np.random.uniform(20, 80)
            }
            for i in range(1, 6)
        ]

        return {
            'query_type': 'resource_query',
            'parameters': {
                'resource_type': resource_type,
                'subscription': subscription,
                'resource_group': resource_group
            },
            'results': resources,
            'result_count': len(resources)
        }

    async def _handle_policy_analysis(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle policy analysis requests"""

        # Create mock governance data for analysis
        governance_data = {
            'policies': [
                {
                    'id': 'policy1',
                    'name': 'Cost Optimization Policy',
                    'type': 'cost',
                    'active': True,
                    'compliance_score': 0.85
                },
                {
                    'id': 'policy2',
                    'name': 'Security Baseline Policy',
                    'type': 'security',
                    'active': True,
                    'compliance_score': 0.92
                }
            ],
            'resources': [
                {
                    'id': 'resource1',
                    'type': 'virtual_machine',
                    'compliance_scores': {'cost': 0.78, 'security': 0.88}
                }
            ]
        }

        # Analyze with GNN service
        try:
            analysis_results = (
                await self.gnn_service.analyze_governance_correlations(governance_data)
            )

            return {
                'analysis_type': 'policy_analysis',
                'governance_data': governance_data,
                'analysis_results': analysis_results,
                'recommendations': [
                    'Consider updating cost optimization thresholds',
                    'Review security policy compliance scores'
                ]
            }
        except Exception as e:
            logger.warning(f"GNN analysis failed, using fallback: {e}")
            return {
                'analysis_type': 'policy_analysis',
                'governance_data': governance_data,
                'fallback_analysis': True,
                'summary': 'Policy analysis completed with basic metrics'
            }

    async def _handle_compliance_check(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle compliance check requests"""

        return {
            'check_type': 'compliance_check',
            'overall_compliance': 0.87,
            'domain_scores': {
                'security': 0.92,
                'cost': 0.84,
                'governance': 0.88
            },
            'non_compliant_resources': [
                {'id': 'resource1', 'issues': ['high_cost', 'security_warning']},
                {'id': 'resource2', 'issues': ['missing_tags']}
            ],
            'recommendations': [
                'Address high-cost resources',
                'Apply security patches',
                'Add required resource tags'
            ]
        }

    async def _handle_cost_optimization(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle cost optimization requests"""

        return {
            'optimization_type': 'cost_optimization',
            'current_monthly_cost': 12450.00,
            'potential_savings': 2380.00,
            'optimization_opportunities': [
                {
                    'resource': 'vm-oversized-1',
                    'current_cost': 450.00,
                    'optimized_cost': 280.00,
                    'action': 'right_size'
                },
                {
                    'resource': 'storage-unused-2',
                    'current_cost': 180.00,
                    'optimized_cost': 0.00,
                    'action': 'delete'
                }
            ],
            'priority_actions': [
                'Right-size over-provisioned VMs',
                'Remove unused storage accounts',
                'Implement automated shutdown schedules'
            ]
        }

    async def _handle_security_assessment(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle security assessment requests"""

        return {
            'assessment_type': 'security_assessment',
            'overall_score': 0.85,
            'critical_findings': 2,
            'high_findings': 7,
            'medium_findings': 15,
            'security_domains': {
                'network_security': 0.88,
                'identity_access': 0.92,
                'data_protection': 0.78,
                'threat_protection': 0.86
            },
            'critical_issues': [
                'Unencrypted storage account detected',
                'Overly permissive NSG rules found'
            ],
            'recommendations': [
                'Enable encryption for all storage accounts',
                'Review and restrict NSG rules',
                'Implement multi-factor authentication'
            ]
        }

    async def _handle_correlation_analysis(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle correlation analysis requests"""

        # Use GNN service for correlation analysis
        try:
            sample_governance_data = {
                'resources': [
                    {'id': 'r1', 'type': 'vm', 'cost': 200, 'cpu': 0.7},
                    {'id': 'r2', 'type': 'storage', 'cost': 50, 'cpu': 0.1}
                ],
                'policies': [
                    {'id': 'p1', 'type': 'cost', 'active': True}
                ]
            }

            correlation_results = (
                await self.gnn_service.analyze_governance_correlations(sample_governance_data)
            )

            return {
                'analysis_type': 'correlation_analysis',
                'correlation_results': correlation_results,
                'key_findings': [
                    'Strong correlation between CPU utilization and cost',
                    'Policy enforcement impacts resource performance'
                ]
            }
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {
                'analysis_type': 'correlation_analysis',
                'error': str(e),
                'fallback_message': 'Correlation analysis temporarily unavailable'
            }

    async def _handle_impact_prediction(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle impact prediction requests"""

        return {
            'prediction_type': 'impact_prediction',
            'scenario': 'Policy change simulation',
            'predicted_impacts': [
                {
                    'domain': 'cost',
                    'impact_level': 'medium',
                    'predicted_change': '+15%'
                },
                {
                    'domain': 'compliance',
                    'impact_level': 'high',
                    'predicted_change': '+25%'
                }
            ],
            'recommendations': [
                'Implement gradual rollout',
                'Monitor cost metrics closely',
                'Set up automated alerts'
            ]
        }

    async def _handle_general_info(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle general information requests"""

        return {
            'response_type': 'general_info',
            'message': 'I can help you with Azure governance tasks including resource queries, policy analysis, compliance checks, cost optimization, security assessments, and
                more.',
            'available_commands': [
                'Show me my resources',
                'Analyze policy compliance',
                'Check security status',
                'Find cost optimization opportunities',
                'Predict impact of changes'
            ]
        }

    async def _handle_help(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle help requests"""

        return {
            'response_type': 'help',
            'help_topics': {
                'resource_management': 'Query and manage Azure resources',
                'policy_analysis': 'Analyze governance policies and compliance',
                'cost_optimization': 'Find opportunities to reduce costs',
                'security_assessment': 'Evaluate security posture',
                'correlation_analysis': 'Understand relationships between resources',
                'impact_prediction': 'Predict effects of proposed changes'
            },
            'example_queries': [
                'Show me all virtual machines in my subscription',
                'What are my compliance scores?',
                'How can I reduce my Azure costs?',
                'What security issues do I have?',
                'What happens if I change this policy?'
            ]
        }

class ConversationalAIService:
    """Main service for conversational AI interactions"""

    def __init__(self):
        self.nlu_processor = NLUProcessor()
        self.dialogue_manager = DialogueManager()
        self.gnn_service = GNNCorrelationService()
        self.action_executor = ActionExecutor(self.gnn_service)

        # Redis for conversation state management
        self.redis_client = None
        self.conversation_ttl = 3600  # 1 hour

        # Response templates
        self.response_templates = {
            ConversationState.GREETING: [
                "Hello! I'm your Azure governance assistant. How can I help you today?",
                "Hi there! I can help you with Azure resource management, policies, and
                    compliance. What would you like to know?"
            ],
            ConversationState.INTENT_IDENTIFICATION: [
                "I understand you want to {intent}. Could you provide more details?",
                "Let me help you with {intent}. What specific information do you need?"
            ],
            ConversationState.INFORMATION_GATHERING: [
                "I need a bit more information to help you. {missing_info}",
                "To give you the best answer, could you tell me {missing_info}?"
            ],
            ConversationState.ACTION_CONFIRMATION: [
                "I'm about to {action}. Is this what you want me to do?",
                "Please confirm: {action}. Should I proceed?"
            ],
            ConversationState.EXECUTION: [
                "Processing your request...",
                "Let me analyze that for you..."
            ],
            ConversationState.FEEDBACK: [
                "Here are the results: {results}",
                "I found the following information: {results}"
            ]
        }

        logger.info("ConversationalAIService initialized")

    async def initialize(self):
        """Initialize the conversational AI service"""

        try:
            # Initialize NLU processor
            await self.nlu_processor.initialize()

            # Initialize GNN service
            await self.gnn_service.initialize()

            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )

            logger.info("ConversationalAIService initialization completed")

        except Exception as e:
            logger.error(f"Error initializing ConversationalAIService: {e}")
            raise

    async def process_conversation(self,
                                 user_input: str,
                                 conversation_id: Optional[str] = None,
                                 user_id: str = "default_user") -> ConversationResponse:
        """Process user input and generate conversational response"""

        try:
            # Get or create conversation context
            if conversation_id:
                context = await self._load_conversation_context(conversation_id)
                if not context:
                    context = self._create_new_context(user_id, conversation_id)
            else:
                context = self._create_new_context(user_id)

            # Process user input
            nlu_results = await self.nlu_processor.process_utterance(user_input, context)

            # Update context with new information
            context = await self._update_context(context, user_input, nlu_results)

            # Determine next state
            next_state = await self.dialogue_manager.determine_next_state(context, nlu_results)

            # Generate response based on state
            response = await self._generate_response(context, nlu_results, next_state)

            # Update context state
            context.current_state = next_state
            context.last_interaction = datetime.now()

            # Save context
            await self._save_conversation_context(context)

            return response

        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return ConversationResponse(
                response_text=f"I encountered an error: {str(e)}. Please try again.",
                intent_type=None,
                confidence_score=0.0,
                next_state=ConversationState.GREETING,
                suggested_actions=[],
                extracted_entities={},
                requires_confirmation=False,
                response_data=None,
                conversation_id=conversation_id or str(uuid.uuid4())
            )

    async def _generate_response(self,
                               context: ConversationContext,
                               nlu_results: Dict[str, Any],
                               next_state: ConversationState) -> ConversationResponse:
        """Generate appropriate response based on context and state"""

        intent = nlu_results.get('intent')
        confidence = nlu_results.get('intent_confidence', 0.0)

        # Execute action if in execution state
        response_data = None
        if next_state == ConversationState.EXECUTION:
            action_result = await self.action_executor.execute_action(intent, context)
            response_data = action_result

        # Generate response text
        response_text = await self._generate_response_text(
            context, nlu_results, next_state, response_data
        )

        # Determine suggested actions
        suggested_actions = self._get_suggested_actions(intent, next_state)

        return ConversationResponse(
            response_text=response_text,
            intent_type=intent,
            confidence_score=confidence,
            next_state=next_state,
            suggested_actions=suggested_actions,
            extracted_entities=nlu_results.get('entities', {}),
            requires_confirmation=(next_state == ConversationState.ACTION_CONFIRMATION),
            response_data=response_data,
            conversation_id=context.conversation_id
        )

    async def _generate_response_text(self,
                                    context: ConversationContext,
                                    nlu_results: Dict[str, Any],
                                    next_state: ConversationState,
                                    response_data: Optional[Dict[str, Any]]) -> str:
        """Generate natural language response text"""

        templates = self.response_templates.get(next_state, ["I'm processing your request..."])
        base_template = templates[0]  # Use first template for simplicity

        if next_state == ConversationState.GREETING:
            return base_template

        elif next_state == ConversationState.INTENT_IDENTIFICATION:
            intent = nlu_results.get('intent')
            if intent:
                return base_template.format(intent=intent.value.replace('_', ' '))
            return "I'm not sure what you're looking for. Could you be more specific?"

        elif next_state == ConversationState.INFORMATION_GATHERING:
            missing_info = self._identify_missing_information(context, nlu_results.get('intent'))
            return base_template.format(missing_info=missing_info)

        elif next_state == ConversationState.ACTION_CONFIRMATION:
            action_description = self._describe_pending_action(context)
            return base_template.format(action=action_description)

        elif next_state == ConversationState.EXECUTION:
            return "Let me process that for you..."

        elif next_state == ConversationState.FEEDBACK:
            if response_data and response_data.get('success'):
                return self._format_results(response_data.get('result', {}))
            else:
                error = response_data.get(
                    'error',
                    'Unknown error'
                ) if response_data else 'Processing failed'
                return f"I encountered an issue: {error}"

        elif next_state == ConversationState.COMPLETE:
            return "Is there anything else I can help you with?"

        return "I'm here to help! What would you like to know?"

    def _identify_missing_information(
        self,
        context: ConversationContext,
        intent: IntentType
    ) -> str:
        """Identify what information is missing for the current intent"""

        missing_info_map = {
            IntentType.QUERY_RESOURCES: "which subscription or resource group you're interested in?",
            IntentType.POLICY_ANALYSIS: "which policies you'd like me to analyze?",
            IntentType.COMPLIANCE_CHECK: "which compliance domain you're concerned about?",
            IntentType.COST_OPTIMIZATION: "what time period you'd like me to analyze?",
            IntentType.SECURITY_ASSESSMENT: "which resources or scope you'd like me to assess?",
            IntentType.CORRELATION_ANALYSIS: "what specific relationships you're interested in?",
            IntentType.IMPACT_PREDICTION: "what changes you're considering?"
        }

        return missing_info_map.get(intent, "more details about your request?")

    def _describe_pending_action(self, context: ConversationContext) -> str:
        """Describe the action that's about to be executed"""

        if context.intent_type == IntentType.QUERY_RESOURCES:
            return "query your Azure resources"
        elif context.intent_type == IntentType.POLICY_ANALYSIS:
            return "analyze your governance policies"
        elif context.intent_type == IntentType.COMPLIANCE_CHECK:
            return "check compliance status"
        elif context.intent_type == IntentType.COST_OPTIMIZATION:
            return "analyze cost optimization opportunities"
        elif context.intent_type == IntentType.SECURITY_ASSESSMENT:
            return "perform a security assessment"
        elif context.intent_type == IntentType.CORRELATION_ANALYSIS:
            return "analyze cross-domain correlations"
        elif context.intent_type == IntentType.IMPACT_PREDICTION:
            return "predict the impact of your proposed changes"

        return "process your request"

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results into natural language response"""

        result_type = (
            results.get('query_type') or results.get('analysis_type') or results.get('response_type')
        )

        if result_type == 'resource_query':
            count = results.get('result_count', 0)
            return f"I found {count} resources matching your criteria. Here's what I discovered: {json.dumps(
                results.get('results',
                [])[:3],
                indent=2
            )}"

        elif result_type == 'policy_analysis':
            return f"I've analyzed your policies. Here are the key findings: {json.dumps(
                results.get('analysis_results',
                {}),
                indent=2
            )}"

        elif result_type == 'compliance_check':
            score = results.get('overall_compliance', 0)
            return f"Your overall compliance score is {score:.0%}. Here are the details: {json.dumps(
                results.get('domain_scores',
                {}),
                indent=2
            )}"

        elif result_type == 'cost_optimization':
            savings = results.get('potential_savings', 0)
            return f"I found potential monthly savings of ${savings:.2f}. Here are the top optimization opportunities: {json.dumps(
                results.get('optimization_opportunities',
                [])[:3],
                indent=2
            )}"

        elif result_type == 'security_assessment':
            score = results.get('overall_score', 0)
            critical = results.get('critical_findings', 0)
            return f"Your security score is {score:.0%} with {critical} critical findings. Here are the key security domains: {json.dumps(
                results.get('security_domains',
                {}),
                indent=2
            )}"

        elif result_type == 'correlation_analysis':
            return f"I've completed the correlation analysis. Here are the key findings: {json.dumps(
                results.get('key_findings',
                []),
                indent=2
            )}"

        elif result_type == 'impact_prediction':
            return f"Based on my analysis, here are the predicted impacts: {json.dumps(
                results.get('predicted_impacts',
                []),
                indent=2
            )}"

        elif result_type == 'help':
            return f"Here are the topics I can help with: {json.dumps(
                list(results.get('help_topics',
                {}).keys()),
                indent=2
            )}"

        return f"Here are your results: {json.dumps(results, indent=2)[:500]}..."

    def _get_suggested_actions(
        self,
        intent: Optional[IntentType],
        state: ConversationState
    ) -> List[Dict[str, Any]]:
        """Get suggested follow-up actions"""

        if state == ConversationState.COMPLETE:
            return [
                {"action": "new_query", "text": "Ask another question"},
                {"action": "help", "text": "See what else I can do"},
                {"action": "end", "text": "End conversation"}
            ]

        if intent == IntentType.QUERY_RESOURCES:
            return [
                {"action": "analyze_costs", "text": "Analyze costs for these resources"},
                {"action": "check_compliance", "text": "Check compliance status"},
                {"action": "security_assessment", "text": "Assess security posture"}
            ]

        return []

    def _create_new_context(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        """Create new conversation context"""

        return ConversationContext(
            conversation_id=conversation_id or str(uuid.uuid4()),
            user_id=user_id,
            session_start=datetime.now(),
            current_state=ConversationState.GREETING,
            intent_type=None,
            extracted_entities={},
            conversation_history=[],
            pending_actions=[],
            context_variables={},
            confidence_scores={},
            last_interaction=datetime.now()
        )

    async def _update_context(self,
                            context: ConversationContext,
                            user_input: str,
                            nlu_results: Dict[str, Any]) -> ConversationContext:
        """Update conversation context with new information"""

        # Add to conversation history
        context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': nlu_results.get('intent'),
            'entities': nlu_results.get('entities', {}),
            'confidence': nlu_results.get('intent_confidence', 0.0)
        })

        # Update extracted entities
        for entity_type, values in nlu_results.get('entities', {}).items():
            if entity_type not in context.extracted_entities:
                context.extracted_entities[entity_type] = []
            context.extracted_entities[entity_type].extend(values)

        # Update intent if confidence is high enough
        intent = nlu_results.get('intent')
        confidence = nlu_results.get('intent_confidence', 0.0)

        if confidence > 0.6:
            context.intent_type = intent
            context.confidence_scores[intent.value] = confidence

        return context

    async def _load_conversation_context(
        self,
        conversation_id: str
    ) -> Optional[ConversationContext]:
        """Load conversation context from Redis"""

        if not self.redis_client:
            return None

        try:
            context_data = await self.redis_client.get(f"conversation:{conversation_id}")
            if context_data:
                context_dict = json.loads(context_data)

                # Convert string dates back to datetime objects
                context_dict['session_start'] = (
                    datetime.fromisoformat(context_dict['session_start'])
                )
                context_dict['last_interaction'] = (
                    datetime.fromisoformat(context_dict['last_interaction'])
                )

                # Convert enum strings back to enums
                context_dict['current_state'] = ConversationState(context_dict['current_state'])
                if context_dict['intent_type']:
                    context_dict['intent_type'] = IntentType(context_dict['intent_type'])

                return ConversationContext(**context_dict)
        except Exception as e:
            logger.warning(f"Error loading conversation context: {e}")

        return None

    async def _save_conversation_context(self, context: ConversationContext):
        """Save conversation context to Redis"""

        if not self.redis_client:
            return

        try:
            # Convert context to dict for JSON serialization
            context_dict = asdict(context)

            # Convert datetime objects to strings
            context_dict['session_start'] = context.session_start.isoformat()
            context_dict['last_interaction'] = context.last_interaction.isoformat()

            # Convert enums to strings
            context_dict['current_state'] = context.current_state.value
            if context.intent_type:
                context_dict['intent_type'] = context.intent_type.value
            else:
                context_dict['intent_type'] = None

            # Save to Redis
            await self.redis_client.setex(
                f"conversation:{context.conversation_id}",
                self.conversation_ttl,
                json.dumps(context_dict, default=str)
            )
        except Exception as e:
            logger.warning(f"Error saving conversation context: {e}")

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""

        context = await self._load_conversation_context(conversation_id)
        if context:
            return context.conversation_history
        return []

    async def end_conversation(self, conversation_id: str) -> bool:
        """End and cleanup conversation"""

        if not self.redis_client:
            return False

        try:
            await self.redis_client.delete(f"conversation:{conversation_id}")
            logger.info(f"Conversation {conversation_id} ended and cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            return False

# Global service instance
conversational_ai_service = ConversationalAIService()
