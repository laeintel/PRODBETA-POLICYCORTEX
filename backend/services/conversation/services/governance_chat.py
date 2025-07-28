"""
Governance Chat Service for PolicyCortex.
Implements Patent 3: Natural Language Processing for Conversational Cloud Governance.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import structlog
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

logger = structlog.get_logger(__name__)


@dataclass
class ConversationContext:
    """Represents the current conversation context."""
    user_id: str
    session_id: str
    entities: Dict[str, Any]
    intents: List[str]
    history: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    current_domain: Optional[str] = None
    last_activity: Optional[datetime] = None


@dataclass
class ChatResponse:
    """Represents a chat response."""
    message: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    actions: List[Dict[str, Any]]
    context: Dict[str, Any]
    suggestions: List[str]


class GovernanceNLU:
    """
    Natural Language Understanding for Governance Queries.
    Implements domain-specific entity recognition and intent classification.
    """

    def __init__(self):
        self.intent_classifier = None
        self.entity_recognizer = None
        self.tokenizer = None
        self.model = None

        # Governance-specific intents
        self.governance_intents = {
            'policy_query': [
                'show policies', 'list policies', 'policy status', 'compliance check',
                'policy violations', 'policy details'
            ],
            'rbac_query': [
                'user permissions', 'role assignments', 'access review', 'who has access',
                'role details', 'permission audit'
            ],
            'cost_query': [
                'cost analysis', 'spending report', 'budget status', 'cost optimization',
                'resource costs', 'billing'
            ],
            'network_query': [
                'network security', 'firewall rules', 'network topology', 'security groups',
                'network access', 'connectivity'
            ],
            'resource_query': [
                'resource inventory', 'resource health', 'resource usage', 'resource status',
                'resource details', 'resource metrics'
            ],
            'compliance_query': [
                'compliance status', 'audit report', 'compliance violations', 'regulatory check',
                'compliance score', 'drift detection'
            ],
            'optimization_request': [
                'optimize costs', 'improve security', 'reduce risks', 'automate policy',
                'remediate issues', 'fix violations'
            ],
            'prediction_request': [
                'predict violations', 'forecast costs', 'risk assessment', 'trend analysis',
                'future compliance', 'upcoming issues'
            ]
        }

        # Governance entities
        self.entity_patterns = {
            'resource_type': [
                'virtual machine', 'vm', 'storage account', 'database', 'network',
                'resource group', 'subscription', 'key vault', 'app service'
            ],
            'azure_service': [
                'azure policy', 'rbac', 'azure ad', 'cost management', 'monitor',
                'security center', 'network watcher', 'resource manager'
            ],
            'compliance_framework': [
                'soc2', 'iso 27001', 'hipaa', 'gdpr', 'pci dss', 'nist'
            ],
            'time_period': [
                'today', 'yesterday', 'last week', 'last month', 'last 30 days',
                'this quarter', 'last quarter', 'this year'
            ],
            'metric_type': [
                'cost', 'usage', 'compliance', 'security', 'performance', 'availability'
            ]
        }

    async def initialize(self):
        """Initialize NLU models."""
        logger.info("initializing_governance_nlu")

        try:
            # Initialize lightweight models for development
            # In production, these would be fine-tuned governance-specific models
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )

            self.entity_recognizer = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )

            # For development, use a simple model
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            logger.info("governance_nlu_initialized")

        except Exception as e:
            logger.error("nlu_initialization_failed", error=str(e))
            # Use rule-based fallbacks
            self.intent_classifier = None
            self.entity_recognizer = None

    async def analyze_query(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze a governance query for intent and entities."""
        try:
            # Classify intent
            intent, confidence = await self._classify_intent(query)

            # Extract entities
            entities = await self._extract_entities(query)

            # Resolve context-dependent entities
            resolved_entities = await self._resolve_entities(entities, context)

            return {
                'intent': intent,
                'confidence': confidence,
                'entities': resolved_entities,
                'query_type': self._determine_query_type(intent),
                'domain': self._determine_domain(intent, entities)
            }

        except Exception as e:
            logger.error("query_analysis_failed", error=str(e))
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'entities': {},
                'query_type': 'information',
                'domain': 'general'
            }

    async def _classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify the intent of a query."""
        query_lower = query.lower()

        # Rule-based intent classification
        best_intent = 'unknown'
        best_score = 0.0

        for intent, patterns in self.governance_intents.items():
            score = 0.0
            for pattern in patterns:
                if pattern in query_lower:
                    score = max(score, 0.8)  # High confidence for exact matches
                elif any(word in query_lower for word in pattern.split()):
                    score = max(score, 0.6)  # Medium confidence for partial matches

            if score > best_score:
                best_score = score
                best_intent = intent

        # Use ML model if available
        if self.intent_classifier and best_score < 0.7:
            try:
                results = self.intent_classifier(query)
                if results and len(results) > 0:
                    top_result = max(results, key=lambda x: x['score'])
                    ml_confidence = top_result['score']

                    if ml_confidence > best_score:
                        # Map ML result to governance intent
                        best_intent = self._map_ml_intent(top_result['label'])
                        best_score = ml_confidence
            except Exception as e:
                logger.warning("ml_intent_classification_failed", error=str(e))

        return best_intent, best_score

    async def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""
        entities = {}
        query_lower = query.lower()

        # Rule-based entity extraction
        for entity_type, patterns in self.entity_patterns.items():
            found_entities = []
            for pattern in patterns:
                if pattern in query_lower:
                    found_entities.append(pattern)

            if found_entities:
                entities[entity_type] = found_entities

        # Use ML model if available
        if self.entity_recognizer:
            try:
                ml_entities = self.entity_recognizer(query)
                for entity in ml_entities:
                    entity_type = entity['entity_group'].lower()
                    entity_text = entity['word']

                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(entity_text)
            except Exception as e:
                logger.warning("ml_entity_extraction_failed", error=str(e))

        return entities

    async def _resolve_entities(self, entities: Dict[str, List[str]],
                               context: ConversationContext) -> Dict[str, Any]:
        """Resolve entities using conversation context."""
        resolved = {}

        for entity_type, entity_list in entities.items():
            resolved[entity_type] = []

            for entity in entity_list:
                resolved_entity = {
                    'value': entity,
                    'confidence': 0.8,
                    'source': 'extracted'
                }

                # Check if entity was mentioned in context
                if entity in context.entities:
                    resolved_entity['context'] = context.entities[entity]
                    resolved_entity['confidence'] = 0.9

                resolved[entity_type].append(resolved_entity)

        return resolved

    def _determine_query_type(self, intent: str) -> str:
        """Determine the type of query based on intent."""
        information_intents = [
            'policy_query', 'rbac_query', 'cost_query',
            'network_query', 'resource_query', 'compliance_query'
        ]
        action_intents = ['optimization_request', 'prediction_request']

        if intent in information_intents:
            return 'information'
        elif intent in action_intents:
            return 'action'
        else:
            return 'unknown'

    def _determine_domain(self, intent: str, entities: Dict[str, Any]) -> str:
        """Determine the governance domain."""
        domain_mapping = {
            'policy_query': 'policy',
            'rbac_query': 'rbac',
            'cost_query': 'cost',
            'network_query': 'network',
            'resource_query': 'resource',
            'compliance_query': 'compliance'
        }

        return domain_mapping.get(intent, 'general')

    def _map_ml_intent(self, ml_label: str) -> str:
        """Map ML model output to governance intents."""
        # Simplified mapping - in production, this would be more sophisticated
        if 'question' in ml_label.lower():
            return 'policy_query'
        elif 'request' in ml_label.lower():
            return 'optimization_request'
        else:
            return 'unknown'


class GovernanceResponseGenerator:
    """
    Generates contextual responses for governance queries.
    """

    def __init__(self):
        self.response_templates = {
            'policy_query': [
                "I found {count} policies related to your query. Here are the details:",
                "Based on your request, here's the policy information:",
                "Let me show you the current policy status:"
            ],
            'rbac_query': [
                "Here's the RBAC information you requested:",
                "I found {count} role assignments matching your criteria:",
                "Current access permissions are:"
            ],
            'cost_query': [
                "Here's your cost analysis:",
                "Current spending information:",
                "Cost breakdown for the requested period:"
            ],
            'compliance_query': [
                "Compliance status summary:",
                "Here's the current compliance report:",
                "Detected {count} compliance issues:"
            ],
            'error': [
                "I encountered an issue processing your request. Let me try a different approach.",
                "I'm having trouble with that query. Could you rephrase it?",
                "Something went wrong. Would you like me to help you with something else?"
            ]
        }

        self.suggestions = {
            'policy_query': [
                "Check policy compliance status",
                "Show recent policy changes",
                "List non-compliant resources"
            ],
            'rbac_query': [
                "Review user access",
                "Show role assignments",
                "Audit permissions"
            ],
            'cost_query': [
                "Analyze cost trends",
                "Show cost optimization opportunities",
                "Compare monthly spending"
            ]
        }

    async def generate_response(self, query_analysis: Dict[str, Any],
                               query_results: Dict[str, Any],
                               context: ConversationContext) -> ChatResponse:
        """Generate a contextual response."""
        try:
            intent = query_analysis.get('intent', 'unknown')
            confidence = query_analysis.get('confidence', 0.0)
            entities = query_analysis.get('entities', {})

            # Generate message
            message = await self._generate_message(intent, query_results, entities)

            # Generate actions
            actions = await self._generate_actions(intent, query_results)

            # Generate suggestions
            suggestions = self._get_suggestions(intent)

            return ChatResponse(
                message=message,
                intent=intent,
                confidence=confidence,
                entities=entities,
                actions=actions,
                context={
                    'domain': query_analysis.get('domain'),
                    'query_type': query_analysis.get('query_type')
                },
                suggestions=suggestions
            )

        except Exception as e:
            logger.error("response_generation_failed", error=str(e))
            return self._generate_error_response()

    async def _generate_message(self, intent: str, results: Dict[str, Any],
                               entities: Dict[str, Any]) -> str:
        """Generate the main response message."""
        templates = self.response_templates.get(intent, self.response_templates['error'])
        template = templates[0]  # Use first template for now

        # Format template with results
        try:
            if 'count' in template:
                count = len(results.get('items', []))
                template = template.format(count=count)

            # Add result summary
            if results.get('summary'):
                message = f"{template}\n\n{results['summary']}"
            else:
                message = template

            # Add data if available
            if results.get('items'):
                items = results['items'][:5]  # Show first 5 items
                message += "\n\n"
                for i, item in enumerate(items, 1):
                    item_summary = self._format_item(item, intent)
                    message += f"{i}. {item_summary}\n"

                if len(results['items']) > 5:
                    remaining = len(results['items']) - 5
                    message += f"\n... and {remaining} more items."

            return message

        except Exception as e:
            logger.error("message_formatting_failed", error=str(e))
            return "I have some information for you, but I'm having trouble formatting it properly."

    async def _generate_actions(self, intent: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable items based on results."""
        actions = []

        if intent == 'compliance_query' and results.get('violations'):
            actions.append({
                'type': 'remediate',
                'title': 'Fix Compliance Issues',
                'description': 'Automatically remediate detected violations',
                'endpoint': '/api/v1/compliance/remediate'
            })

        if intent == 'cost_query' and results.get('optimization_opportunities'):
            actions.append({
                'type': 'optimize',
                'title': 'Optimize Costs',
                'description': 'Apply cost optimization recommendations',
                'endpoint': '/api/v1/cost/optimize'
            })

        if results.get('items'):
            actions.append({
                'type': 'export',
                'title': 'Export Results',
                'description': 'Download detailed report',
                'endpoint': '/api/v1/export'
            })

        return actions

    def _format_item(self, item: Dict[str, Any], intent: str) -> str:
        """Format an individual item for display."""
        if intent == 'policy_query':
            return f"{item.get('name', 'Unknown')} - {item.get('status', 'Unknown status')}"
        elif intent == 'cost_query':
            return f"{item.get('resource', 'Unknown')} - ${item.get('cost', 0):.2f}"
        elif intent == 'compliance_query':
            return f"{item.get('resource', 'Unknown')} - {item.get('violation', 'Unknown issue')}"
        else:
            return str(item.get('name', item.get('id', 'Unknown item')))

    def _get_suggestions(self, intent: str) -> List[str]:
        """Get contextual suggestions."""
        return self.suggestions.get(intent, [
            "Ask about policy status",
            "Check compliance",
            "Analyze costs"
        ])

    def _generate_error_response(self) -> ChatResponse:
        """Generate an error response."""
        return ChatResponse(
            message="I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
            intent="error",
            confidence=0.0,
            entities={},
            actions=[],
            context={},
            suggestions=["Try a different question", "Check system status", "Contact support"]
        )


class GovernanceChatService:
    """
    Main conversational interface for governance management.
    Implements natural language processing and query routing.
    """

    def __init__(self):
        self.nlu = GovernanceNLU()
        self.response_generator = GovernanceResponseGenerator()
        self.conversations = {}  # Active conversations
        self.query_router = None

    async def initialize(self):
        """Initialize the chat service."""
        logger.info("initializing_governance_chat_service")

        try:
            await self.nlu.initialize()

            logger.info("governance_chat_service_initialized")

        except Exception as e:
            logger.error("chat_service_initialization_failed", error=str(e))
            raise

    async def process_query(self, query: str, user_id: str, session_id: str,
                           user_profile: Dict[str, Any] = None) -> ChatResponse:
        """Process a governance query and return a response."""
        try:
            logger.info("processing_governance_query",
                       user_id=user_id,
                       session_id=session_id,
                       query_length=len(query))

            # Get or create conversation context
            context = await self._get_conversation_context(user_id, session_id, user_profile)

            # Update context with current query
            context.history.append({
                'timestamp': datetime.utcnow(),
                'type': 'user_query',
                'content': query
            })

            # Analyze the query
            query_analysis = await self.nlu.analyze_query(query, context)

            # Route query to appropriate handler
            query_results = await self._route_query(query_analysis, context)

            # Generate response
            response = await self.response_generator.generate_response(
                query_analysis, query_results, context
            )

            # Update conversation context
            await self._update_conversation_context(context, response)

            logger.info("governance_query_processed",
                       user_id=user_id,
                       intent=response.intent,
                       confidence=response.confidence)

            return response

        except Exception as e:
            logger.error("query_processing_failed",
                        user_id=user_id,
                        error=str(e))
            return self.response_generator._generate_error_response()

    async def _get_conversation_context(self, user_id: str, session_id: str,
                                       user_profile: Dict[str, Any]) -> ConversationContext:
        """Get or create conversation context."""
        context_key = f"{user_id}:{session_id}"

        if context_key not in self.conversations:
            self.conversations[context_key] = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                entities={},
                intents=[],
                history=[],
                user_profile=user_profile or {},
                last_activity=datetime.utcnow()
            )

        return self.conversations[context_key]

    async def _route_query(self, query_analysis: Dict[str, Any],
                          context: ConversationContext) -> Dict[str, Any]:
        """Route query to appropriate backend service."""
        intent = query_analysis.get('intent')
        domain = query_analysis.get('domain')
        entities = query_analysis.get('entities', {})

        # For now, return mock data
        # In production, this would route to actual Azure services

        if intent == 'policy_query':
            return await self._handle_policy_query(entities)
        elif intent == 'rbac_query':
            return await self._handle_rbac_query(entities)
        elif intent == 'cost_query':
            return await self._handle_cost_query(entities)
        elif intent == 'compliance_query':
            return await self._handle_compliance_query(entities)
        elif intent == 'optimization_request':
            return await self._handle_optimization_request(entities)
        elif intent == 'prediction_request':
            return await self._handle_prediction_request(entities)
        else:
            return {
                'summary': 'I understand your query, but I need more specific information.',
                'items': [],
                'total': 0
            }

    async def _handle_policy_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle policy-related queries."""
        # Mock response
        return {
            'summary': 'Found 12 policies, 10 compliant, 2 with violations',
            'items': [
                {'name': 'VM Size Policy', 'status': 'Compliant', 'violations': 0},
                {'name': 'Storage Encryption', 'status': 'Non-Compliant', 'violations': 3},
                {'name': 'Network Security', 'status': 'Compliant', 'violations': 0}
            ],
            'total': 12,
            'violations': 2
        }

    async def _handle_rbac_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RBAC-related queries."""
        return {
            'summary': 'Found 45 role assignments across 15 users',
            'items': [
                {'user': 'john.doe@company.com', 'role': 'Contributor', 'scope': 'ResourceGroup'},
                {'user': 'jane.smith@company.com', 'role': 'Owner', 'scope': 'Subscription'}
            ],
            'total': 45
        }

    async def _handle_cost_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cost-related queries."""
        return {
            'summary': 'Total spending: $12,450 this month (+15% vs last month)',
            'items': [
                {'resource': 'Virtual Machines', 'cost': 8500.00, 'percentage': 68},
                {'resource': 'Storage', 'cost': 2100.50, 'percentage': 17},
                {'resource': 'Networking', 'cost': 1849.50, 'percentage': 15}
            ],
            'total': 12450.00,
            'optimization_opportunities': True
        }

    async def _handle_compliance_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compliance-related queries."""
        return {
            'summary': 'Compliance score: 87% (3 violations detected)',
            'items': [
                {'resource': 'StorageAccount-Prod', 'violation': 'Public access enabled'},
                {'resource': 'VM-WebServer-01', 'violation': 'Missing backup policy'},
                {'resource': 'NSG-Default', 'violation': 'Too permissive rules'}
            ],
            'total': 3,
            'violations': [
                {'severity': 'High', 'count': 1},
                {'severity': 'Medium', 'count': 2}
            ]
        }

    async def _handle_optimization_request(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization requests."""
        return {
            'summary': 'Found 5 optimization opportunities with potential savings of $2,400/month',
            'items': [
                {'type': 'Right-sizing', 'savings': 1200, 'resources': 8},
                {'type': 'Reserved Instances', 'savings': 800, 'resources': 15},
                {'type': 'Unused Resources', 'savings': 400, 'resources': 3}
            ],
            'total_savings': 2400
        }

    async def _handle_prediction_request(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction requests."""
        return {
            'summary': 'Predicted 2 compliance violations in next 7 days (85% confidence)',
            'items': [
                {'prediction': 'Policy violation on VM-Test-02', 'probability': 0.87, 'days': 3},
                {'prediction': 'Cost budget exceed', 'probability': 0.72, 'days': 5}
            ],
            'confidence': 0.85
        }

    async def _update_conversation_context(self, context: ConversationContext,
                                          response: ChatResponse):
        """Update conversation context with response."""
        context.history.append({
            'timestamp': datetime.utcnow(),
            'type': 'assistant_response',
            'content': response.message,
            'intent': response.intent,
            'confidence': response.confidence
        })

        # Update entities
        for entity_type, entity_list in response.entities.items():
            if entity_type not in context.entities:
                context.entities[entity_type] = []
            context.entities[entity_type].extend(entity_list)

        # Update current domain
        context.current_domain = response.context.get('domain')
        context.last_activity = datetime.utcnow()

        # Keep only last 20 history items to prevent memory bloat
        if len(context.history) > 20:
            context.history = context.history[-20:]

    async def get_conversation_history(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        context_key = f"{user_id}:{session_id}"

        if context_key in self.conversations:
            return self.conversations[context_key].history

        return []

    async def clear_conversation(self, user_id: str, session_id: str):
        """Clear conversation context."""
        context_key = f"{user_id}:{session_id}"

        if context_key in self.conversations:
            del self.conversations[context_key]

    def is_ready(self) -> bool:
        """Check if service is ready."""
        return True  # For now, always ready

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("cleaning_up_governance_chat_service")
        self.conversations.clear()


class AdvancedConversationManager:
    """
    Advanced conversation manager with multi-turn support, workflow management,
    and personalized interactions.
    """

    def __init__(self):
        self.active_conversations = {}
        self.user_profiles = {}
        self.conversation_workflows = {}
        self.session_analytics = defaultdict(dict)
        self.nlp_processor = None

    async def initialize(self):
        """Initialize the advanced conversation manager."""
        logger.info("initializing_advanced_conversation_manager")

        try:
            # Initialize spaCy for advanced NLP
            # Note: In production, you'd need to install spacy and download the model
            # python -m spacy download en_core_web_sm
            # self.nlp_processor = spacy.load("en_core_web_sm")

            # Initialize predefined workflows
            await self._initialize_workflows()

            logger.info("advanced_conversation_manager_initialized")

        except Exception as e:
            logger.error("advanced_conversation_manager_initialization_failed", error=str(e))

    async def _initialize_workflows(self):
        """Initialize predefined conversation workflows."""

        # Policy Compliance Investigation Workflow
        compliance_workflow = ConversationWorkflow(
            workflow_id="policy_compliance_investigation",
            name="Policy Compliance Investigation",
            description="Deep dive into policy compliance issues",
            steps=[
                {
                    "step_id": "identify_scope",
                    "prompt": "Which specific policy or resource group would you like me to investigate?",
                    "required_entities": ["policy_name", "resource_group"],
                    "optional": False
                },
                {
                    "step_id": "analyze_violations",
                    "prompt": "I'll analyze compliance violations. What time period should I examine?",
                    "required_entities": ["time_period"],
                    "optional": False
                },
                {
                    "step_id": "suggest_remediation",
                    "prompt": "Based on my analysis, would you like me to suggest remediation actions?",
                    "required_entities": ["confirmation"],
                    "optional": True
                },
                {
                    "step_id": "execute_remediation",
                    "prompt": "Should I create automation workflows to fix these issues?",
                    "required_entities": ["confirmation"],
                    "optional": True
                }
            ]
        )

        # Cost Optimization Analysis Workflow
        cost_workflow = ConversationWorkflow(
            workflow_id="cost_optimization_analysis",
            name="Cost Optimization Analysis",
            description="Comprehensive cost analysis and optimization recommendations",
            steps=[
                {
                    "step_id": "define_scope",
                    "prompt": "What scope should I analyze? (
                        subscription,
                        resource group,
                        or specific services
                    )",
                    "required_entities": ["scope"],
                    "optional": False
                },
                {
                    "step_id": "set_timeframe",
                    "prompt": "What time period should I analyze for cost patterns?",
                    "required_entities": ["time_period"],
                    "optional": False
                },
                {
                    "step_id": "identify_opportunities",
                    "prompt": "I'll identify optimization opportunities. Are you interested in immediate savings or long-term optimization?",
                    "required_entities": ["optimization_type"],
                    "optional": False
                },
                {
                    "step_id": "create_action_plan",
                    "prompt": "Would you like me to create an action plan with prioritized recommendations?",
                    "required_entities": ["confirmation"],
                    "optional": True
                }
            ]
        )

        self.conversation_workflows = {
            "policy_compliance_investigation": compliance_workflow,
            "cost_optimization_analysis": cost_workflow
        }

        logger.info("conversation_workflows_initialized", count=len(self.conversation_workflows))

    async def start_conversation(self, user_id: str, session_id: str,
                               user_profile: Optional[UserProfile] = None) -> ConversationContext:
        """Start a new conversation with enhanced context."""

        # Create or update user profile
        if user_profile:
            self.user_profiles[user_id] = user_profile
        elif user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                name=f"User_{user_id[:8]}",
                role="user",
                permissions=["read"],
                domains_of_interest=[]
            )

        # Create conversation context
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            conversation_state=ConversationState.INITIAL,
            entities={},
            intents=[],
            history=[],
            user_profile=self.user_profiles[user_id],
            current_domain=None,
            last_activity=datetime.utcnow(),
            active_workflow=None,
            context_stack=[],
            pending_actions=[],
            clarification_needed=None,
            session_metadata={}
        )

        context_key = f"{user_id}:{session_id}"
        self.active_conversations[context_key] = context

        # Initialize session analytics
        self.session_analytics[context_key] = {
            'start_time': datetime.utcnow(),
            'message_count': 0,
            'intent_distribution': defaultdict(int),
            'workflow_completions': 0,
            'satisfaction_indicators': []
        }

        logger.info("conversation_started", user_id=user_id, session_id=session_id)
        return context

    async def process_message(self, user_id: str, session_id: str,
                            message: str, nlu_service: GovernanceNLU) -> ChatResponse:
        """Process a message with advanced conversation management."""

        context_key = f"{user_id}:{session_id}"

        # Get or create conversation context
        if context_key not in self.active_conversations:
            context = await self.start_conversation(user_id, session_id)
        else:
            context = self.active_conversations[context_key]

        # Update analytics
        self.session_analytics[context_key]['message_count'] += 1
        context.last_activity = datetime.utcnow()

        try:
            # Analyze message with enhanced NLU
            analysis = await nlu_service.analyze_query(message, context)

            # Update analytics
            self.session_analytics[context_key]['intent_distribution'][analysis['intent'].value] + = (
                1
            )

            # Handle workflow processing
            if context.active_workflow:
                return await self._process_workflow_message(context, message, analysis)

            # Check if message should start a workflow
            workflow_trigger = await self._check_workflow_triggers(analysis, context)
            if workflow_trigger:
                return await self._start_workflow(context, workflow_trigger, analysis)

            # Process regular message
            response = await self._process_regular_message(context, message, analysis)

            # Update conversation context
            await self._update_context(context, message, response, analysis)

            return response

        except Exception as e:
            logger.error(
                "message_processing_failed",
                error=str(e),
                user_id=user_id,
                session_id=session_id
            )
            return ChatResponse(
                message="I apologize, but I encountered an error processing your message. Could you please try again?",
                intent=IntentType.HELP,
                confidence=0.0,
                entities={},
                actions=[],
                context={},
                suggestions=["Can you help me?", "What can you do?"],
                conversation_state=ConversationState.ERROR
            )

    async def _check_workflow_triggers(self, analysis: Dict[str, Any],
                                     context: ConversationContext) -> Optional[str]:
        """Check if the message should trigger a workflow."""

        intent = analysis['intent']
        entities = analysis.get('entities', {})

        # Trigger compliance investigation workflow
        if (intent == IntentType.COMPLIANCE_QUERY and
            analysis.get('semantic_analysis', {}).get('complexity') == 'complex'):
            return "policy_compliance_investigation"

        # Trigger cost optimization workflow
        if (intent == IntentType.COST_QUERY and
            any(keyword in analysis.get('normalized_query', '')
                for keyword in ['optimize', 'reduce', 'save', 'analysis'])):
            return "cost_optimization_analysis"

        return None

    async def _start_workflow(self, context: ConversationContext,
                            workflow_id: str, analysis: Dict[str, Any]) -> ChatResponse:
        """Start a conversation workflow."""

        if workflow_id not in self.conversation_workflows:
            return await self._create_error_response("Workflow not found")

        workflow = self.conversation_workflows[workflow_id]
        context.active_workflow = workflow
        context.conversation_state = ConversationState.GATHERING_INFO

        # Get first step
        first_step = workflow.steps[0]

        return ChatResponse(
            message=f"I'll help you with {workflow.name.lower()}. {first_step['prompt']}",
            intent=analysis['intent'],
            confidence=analysis['confidence'],
            entities=analysis.get('entities', {}),
            actions=[{'type': 'start_workflow', 'workflow_id': workflow_id}],
            context={'workflow_step': first_step['step_id']},
            suggestions=self._generate_workflow_suggestions(first_step),
            conversation_state=ConversationState.GATHERING_INFO,
            workflow_status={
                'workflow_id': workflow_id,
                'current_step': 0,
                'total_steps': len(workflow.steps),
                'progress': 0.0
            }
        )

    async def _process_workflow_message(self, context: ConversationContext,
                                      message: str, analysis: Dict[str, Any]) -> ChatResponse:
        """Process a message within an active workflow."""

        workflow = context.active_workflow
        current_step = workflow.steps[workflow.current_step]

        # Check if user wants to cancel workflow
        if analysis['intent'] == IntentType.CANCEL:
            context.active_workflow = None
            context.conversation_state = ConversationState.INITIAL
            return ChatResponse(
                message="Workflow cancelled. How else can I help you?",
                intent=IntentType.CANCEL,
                confidence=1.0,
                entities={},
                actions=[{'type': 'cancel_workflow'}],
                context={},
                suggestions=["Show me policies", "Check compliance", "Cost analysis"],
                conversation_state=ConversationState.INITIAL
            )

        # Extract required entities for current step
        required_entities = current_step.get('required_entities', [])
        extracted_entities = analysis.get('entities', {})

        # Check if all required entities are present
        missing_entities = [entity for entity in required_entities
                           if entity not in extracted_entities and
                               f'inferred_{entity}' not in extracted_entities]

        if missing_entities and not current_step.get('optional', False):
            clarification = self._generate_entity_clarification(missing_entities[0])
            return ChatResponse(
                message=clarification,
                intent=analysis['intent'],
                confidence=analysis['confidence'],
                entities=extracted_entities,
                actions=[],
                context={'workflow_step': current_step['step_id']},
                suggestions=self._generate_entity_suggestions(missing_entities[0]),
                conversation_state=ConversationState.CLARIFYING,
                requires_clarification=True,
                clarification_question=clarification
            )

        # Store step variables
        for entity_type, entity_list in extracted_entities.items():
            if entity_list:
                workflow.variables[entity_type] = entity_list[0]['value']

        # Mark step as completed
        workflow.completed_steps.append(current_step['step_id'])
        workflow.current_step += 1

        # Check if workflow is complete
        if workflow.current_step >= len(workflow.steps):
            return await self._complete_workflow(context, workflow)

        # Move to next step
        next_step = workflow.steps[workflow.current_step]
        progress = (workflow.current_step / len(workflow.steps)) * 100

        return ChatResponse(
            message=f"Great! {next_step['prompt']}",
            intent=analysis['intent'],
            confidence=analysis['confidence'],
            entities=extracted_entities,
            actions=[{'type': 'workflow_progress', 'step': workflow.current_step}],
            context={'workflow_step': next_step['step_id']},
            suggestions=self._generate_workflow_suggestions(next_step),
            conversation_state=ConversationState.GATHERING_INFO,
            workflow_status={
                'workflow_id': workflow.workflow_id,
                'current_step': workflow.current_step,
                'total_steps': len(workflow.steps),
                'progress': progress
            }
        )

    async def _complete_workflow(self, context: ConversationContext,
                               workflow: ConversationWorkflow) -> ChatResponse:
        """Complete a workflow and provide results."""

        # Execute workflow based on collected variables
        results = await self._execute_workflow_logic(workflow)

        # Update analytics
        context_key = f"{context.user_id}:{context.session_id}"
        self.session_analytics[context_key]['workflow_completions'] += 1

        # Reset workflow state
        context.active_workflow = None
        context.conversation_state = ConversationState.COMPLETED

        return ChatResponse(
            message=f"Workflow completed! {results['summary']}",
            intent=IntentType.ANALYTICS_REQUEST,
            confidence=1.0,
            entities={},
            actions=[{'type': 'workflow_complete', 'results': results}],
            context={'workflow_results': results},
            suggestions=["Start another analysis", "Export results", "Create automation"],
            conversation_state=ConversationState.COMPLETED,
            rich_content=results,
            workflow_status={
                'workflow_id': workflow.workflow_id,
                'status': 'completed',
                'progress': 100.0
            }
        )

    async def _execute_workflow_logic(self, workflow: ConversationWorkflow) -> Dict[str, Any]:
        """Execute the actual workflow logic based on collected variables."""

        if workflow.workflow_id == "policy_compliance_investigation":
            return {
                'summary': f"Completed compliance investigation for {workflow.variables.get(
                    'policy_name',
                    'specified scope'
                )}",
                'violations_found': 3,
                'resources_analyzed': 45,
                'recommendations': [
                    "Update 2 storage accounts to enable encryption",
                    "Apply network security group rules to 1 subnet",
                    "Enable backup policy for 3 virtual machines"
                ],
                'estimated_fix_time': "2-4 hours",
                'risk_level': "Medium"
            }

        elif workflow.workflow_id == "cost_optimization_analysis":
            return {
                'summary': f"Cost optimization analysis completed for {workflow.variables.get(
                    'scope',
                    'specified scope'
                )}",
                'current_monthly_cost': 15420.50,
                'potential_savings': 3240.80,
                'savings_percentage': 21.0,
                'top_recommendations': [
                    "Right-size 8 over-provisioned VMs (save $1,200/month)",
                    "Purchase reserved instances for 5 VMs (save $1,800/month)",
                    "Delete 3 unused storage accounts (save $240/month)"
                ],
                'implementation_effort': "Low to Medium"
            }

        return {'summary': 'Workflow completed successfully'}

    def _generate_workflow_suggestions(self, step: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions for a workflow step."""

        required_entities = step.get('required_entities', [])

        suggestions_map = {
            'policy_name': ["All policies", "Security policies", "Compliance policies"],
            'resource_group': ["Production-RG", "Development-RG", "All resource groups"],
            'time_period': ["Last 30 days", "Last week", "Last 3 months"],
            'scope': ["Current subscription", "Specific resource group", "All subscriptions"],
            'optimization_type': ["Immediate savings", "Long-term optimization", "Both"],
            'confirmation': ["Yes, proceed", "No, skip this step", "Tell me more"]
        }

        for entity in required_entities:
            if entity in suggestions_map:
                return suggestions_map[entity]

        return ["Continue", "Skip", "Cancel"]

    def _generate_entity_clarification(self, entity_type: str) -> str:
        """Generate clarification questions for missing entities."""

        clarifications = {
            'policy_name': "Which specific policy would you like me to investigate?",
            'resource_group': "Which resource group should I analyze?",
            'time_period': "What time period should I examine?",
            'scope': "What scope should I analyze - subscription, resource group, or specific services?",
            'optimization_type': "Are you looking for immediate savings or long-term optimization strategies?",
            'confirmation': "Should I proceed with this step?"
        }

        return clarifications.get(entity_type, "Could you provide more details?")

    def _generate_entity_suggestions(self, entity_type: str) -> List[str]:
        """Generate suggestions for entity clarification."""

        suggestions_map = {
            'policy_name': ["Security Center policies", "Custom policies", "All policies"],
            'resource_group': ["prod-rg", "dev-rg", "test-rg"],
            'time_period': ["Last 30 days", "Last week", "Yesterday"],
            'scope': ["Current subscription", "Resource group", "Specific service"],
            'confirmation': ["Yes", "No", "Tell me more"]
        }

        return suggestions_map.get(entity_type, ["Continue", "Cancel"])

    async def _process_regular_message(self, context: ConversationContext,
                                     message: str, analysis: Dict[str, Any]) -> ChatResponse:
        """Process a regular (non-workflow) message with enhanced features."""

        intent = analysis['intent']
        entities = analysis.get('entities', {})

        # Check if clarification is needed
        if analysis.get('clarification_needed'):
            context.clarification_needed = analysis.get('clarification_question')
            return ChatResponse(
                message=analysis['clarification_question'],
                intent=intent,
                confidence=analysis['confidence'],
                entities=entities,
                actions=[],
                context={},
                suggestions=self._generate_clarification_suggestions(intent),
                conversation_state=ConversationState.CLARIFYING,
                requires_clarification=True,
                clarification_question=analysis['clarification_question']
            )

        # Generate personalized response based on user profile
        response_message = await self._generate_personalized_response(intent, entities, context)

        # Generate contextual suggestions
        suggestions = await self._generate_contextual_suggestions(intent, entities, context)

        # Determine next conversation state
        next_state = self._determine_next_state(intent, analysis)

        return ChatResponse(
            message=response_message,
            intent=intent,
            confidence=analysis['confidence'],
            entities=entities,
            actions=self._generate_actions(intent, entities),
            context=self._build_response_context(intent, entities, analysis),
            suggestions=suggestions,
            conversation_state=next_state,
            quick_replies=self._generate_quick_replies(intent),
            rich_content=await self._generate_rich_content(intent, entities)
        )

    def _generate_clarification_suggestions(self, intent: IntentType) -> List[str]:
        """Generate suggestions for clarification scenarios."""

        base_suggestions = {
            IntentType.POLICY_QUERY: ["Show all policies", "Check specific policy", "Policy violations"],
            IntentType.COST_QUERY: ["Monthly costs", "Cost by service", "Budget status"],
            IntentType.RESOURCE_QUERY: ["All resources", "Specific resource type", "Resource health"]
        }

        return base_suggestions.get(intent, ["Help me", "Cancel", "Start over"])

    async def _generate_personalized_response(self, intent: IntentType, entities: Dict[str, Any],
                                            context: ConversationContext) -> str:
        """Generate personalized response based on user profile and history."""

        user_profile = context.user_profile
        expertise_level = user_profile.expertise_level

        # Adjust response complexity based on expertise
        if expertise_level == "beginner":
            prefix = "Let me explain this in simple terms. "
        elif expertise_level == "expert":
            prefix = "Here's the detailed technical information: "
        else:
            prefix = ""

        # Base responses by intent
        base_responses = {
            IntentType.POLICY_QUERY: f"{prefix}I found policy information for your environment.",
            IntentType.COST_QUERY: f"{prefix}Here's your cost analysis.",
            IntentType.COMPLIANCE_QUERY: f"{prefix}I've analyzed your compliance status.",
            IntentType.RESOURCE_QUERY: f"{prefix}Here are your resource details.",
            IntentType.GREETING: f"Hello {user_profile.name}! How can I help you with governance today?",
            IntentType.HELP: f"{prefix}I can help you with Azure governance tasks including policies, compliance, costs, and
                resources."
        }

        return base_responses.get(intent, f"{prefix}I understand your request and
            I'm processing it.")

    async def _generate_contextual_suggestions(self, intent: IntentType, entities: Dict[str, Any],
                                             context: ConversationContext) -> List[str]:
        """Generate contextual suggestions based on conversation state and user behavior."""

        # Base suggestions by intent
        base_suggestions = {
            IntentType.POLICY_QUERY: ["Show policy details", "Check compliance", "View violations"],
            IntentType.COST_QUERY: ["Cost breakdown", "Optimization tips", "Budget alerts"],
            IntentType.COMPLIANCE_QUERY: ["Remediation steps", "Audit report", "Risk assessment"],
            IntentType.RESOURCE_QUERY: ["Resource metrics", "Dependencies", "Health status"]
        }

        suggestions = base_suggestions.get(intent, ["What else can you do?", "Help"])

        # Add personalized suggestions based on user's domains of interest
        for domain in context.user_profile.domains_of_interest[:2]:
            if domain == "cost" and intent != IntentType.COST_QUERY:
                suggestions.append("Check my costs")
            elif domain == "security" and intent != IntentType.COMPLIANCE_QUERY:
                suggestions.append("Security compliance")

        return suggestions[:4]  # Limit to 4 suggestions

    def _determine_next_state(
        self,
        intent: IntentType,
        analysis: Dict[str,
        Any]
    ) -> ConversationState:
        """Determine the next conversation state."""

        if analysis.get('clarification_needed'):
            return ConversationState.CLARIFYING
        elif intent in [IntentType.OPTIMIZATION_REQUEST, IntentType.REMEDIATION_ACTION]:
            return ConversationState.EXECUTING_ACTION
        elif intent in [IntentType.PREDICTION_REQUEST, IntentType.ANALYTICS_REQUEST]:
            return ConversationState.PROVIDING_RESULTS
        else:
            return ConversationState.PROVIDING_RESULTS

    def _generate_actions(
        self,
        intent: IntentType,
        entities: Dict[str,
        Any]
    ) -> List[Dict[str, Any]]:
        """Generate actions based on intent and entities."""

        actions = []

        if intent == IntentType.OPTIMIZATION_REQUEST:
            actions.append({'type': 'generate_optimization_report', 'scope': 'detected'})
        elif intent == IntentType.REMEDIATION_ACTION:
            actions.append({'type': 'create_remediation_workflow', 'auto_execute': False})
        elif intent == IntentType.PREDICTION_REQUEST:
            actions.append({'type': 'run_predictive_analysis', 'horizon': '7_days'})

        return actions

    def _build_response_context(self, intent: IntentType, entities: Dict[str, Any],
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build response context."""

        return {
            'domain': self._map_intent_to_domain(intent),
            'complexity': analysis.get('semantic_analysis', {}).get('complexity', 'simple'),
            'requires_data': analysis.get('semantic_analysis', {}).get('requires_data', False),
            'entities_count': len(entities),
            'confidence_level': 'high' if analysis['confidence'] > 0.8 else 'medium' if analysis['confidence'] > 0.6 else 'low'
        }

    def _map_intent_to_domain(self, intent: IntentType) -> str:
        """Map intent to governance domain."""

        mapping = {
            IntentType.POLICY_QUERY: 'policy',
            IntentType.COMPLIANCE_QUERY: 'policy',
            IntentType.RBAC_QUERY: 'rbac',
            IntentType.COST_QUERY: 'cost',
            IntentType.NETWORK_QUERY: 'network',
            IntentType.RESOURCE_QUERY: 'resource'
        }

        return mapping.get(intent, 'general')

    def _generate_quick_replies(self, intent: IntentType) -> List[str]:
        """Generate quick reply options."""

        quick_replies = {
            IntentType.POLICY_QUERY: ["Show details", "Check violations", "Next policy"],
            IntentType.COST_QUERY: ["Optimize costs", "Show breakdown", "Set budget alert"],
            IntentType.COMPLIANCE_QUERY: ["Fix issues", "Generate report", "Explain risk"]
        }

        return quick_replies.get(intent, ["Continue", "Help", "More options"])

    async def _generate_rich_content(
        self,
        intent: IntentType,
        entities: Dict[str,
        Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate rich content (charts, tables, etc.) for responses."""

        if intent == IntentType.COST_QUERY:
            return {
                'type': 'chart',
                'chart_type': 'pie',
                'data': {
                    'labels': ['Compute', 'Storage', 'Network'],
                    'values': [65, 25, 10],
                    'title': 'Cost Distribution'
                }
            }
        elif intent == IntentType.COMPLIANCE_QUERY:
            return {
                'type': 'table',
                'headers': ['Resource', 'Status', 'Risk Level'],
                'rows': [
                    ['VM-Web-01', 'Compliant', 'Low'],
                    ['SA-Data-01', 'Non-Compliant', 'High'],
                    ['NSG-Default', 'Warning', 'Medium']
                ]
            }

        return None

    async def _update_context(self, context: ConversationContext, message: str,
                            response: ChatResponse, analysis: Dict[str, Any]):
        """Update conversation context after processing."""

        # Create conversation turn
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_message=message,
            bot_response=response.message,
            intent=response.intent,
            confidence=response.confidence,
            entities=response.entities,
            actions_taken=response.actions,
            context_updates=response.context
        )

        # Add to history
        context.history.append(turn)

        # Update context state
        context.conversation_state = response.conversation_state
        context.intents.append(response.intent)

        # Update entities
        for entity_type, entity_list in response.entities.items():
            if entity_type not in context.entities:
                context.entities[entity_type] = []
            context.entities[entity_type].extend(entity_list)

        # Update current domain
        if response.context.get('domain'):
            context.current_domain = response.context['domain']

            # Update user profile interests
            if context.current_domain not in context.user_profile.domains_of_interest:
                context.user_profile.domains_of_interest.append(context.current_domain)

        # Keep history manageable
        if len(context.history) > 50:
            context.history = context.history[-25:]

        # Update user profile activity
        context.user_profile.last_active = datetime.utcnow()
        context.user_profile.conversation_history.append(context.session_id)

    async def _create_error_response(self, error_message: str) -> ChatResponse:
        """Create a standardized error response."""

        return ChatResponse(
            message=f"I apologize, but {error_message.lower()}. How else can I help you?",
            intent=IntentType.HELP,
            confidence=0.0,
            entities={},
            actions=[],
            context={},
            suggestions=["Show me policies", "Check costs", "Help"],
            conversation_state=ConversationState.ERROR
        )

    async def get_conversation_analytics(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get analytics for a conversation session."""

        context_key = f"{user_id}:{session_id}"

        if context_key not in self.session_analytics:
            return {}

        analytics = self.session_analytics[context_key].copy()

        # Calculate session duration
        if 'start_time' in analytics:
            session_duration = (datetime.utcnow() - analytics['start_time']).total_seconds()
            analytics['session_duration_seconds'] = session_duration

        # Add conversation quality metrics
        if context_key in self.active_conversations:
            context = self.active_conversations[context_key]
            analytics['conversation_quality'] = {
                'turns_count': len(context.history),
                'avg_confidence': np.mean([turn.confidence for turn in context.history]) if context.history else 0,
                'clarifications_needed': len([turn for turn in context.history if 'clarification' in turn.bot_response.lower()]),
                'domains_explored': list(set([turn.context_updates.get('domain') for turn in context.history if turn.context_updates.get('domain')]))
            }

        return analytics

    async def export_conversation(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Export conversation data for analysis or backup."""

        context_key = f"{user_id}:{session_id}"

        if context_key not in self.active_conversations:
            return {}

        context = self.active_conversations[context_key]
        analytics = await self.get_conversation_analytics(user_id, session_id)

        return {
            'conversation_metadata': {
                'user_id': user_id,
                'session_id': session_id,
                'start_time': analytics.get('start_time'),
                'export_time': datetime.utcnow(),
                'message_count': analytics.get('message_count', 0)
            },
            'conversation_history': [
                {
                    'turn_id': turn.turn_id,
                    'timestamp': turn.timestamp,
                    'user_message': turn.user_message,
                    'bot_response': turn.bot_response,
                    'intent': turn.intent.value,
                    'confidence': turn.confidence,
                    'entities': turn.entities
                }
                for turn in context.history
            ],
            'user_profile': {
                'name': context.user_profile.name,
                'role': context.user_profile.role,
                'expertise_level': context.user_profile.expertise_level,
                'domains_of_interest': context.user_profile.domains_of_interest
            },
            'analytics': analytics
        }
