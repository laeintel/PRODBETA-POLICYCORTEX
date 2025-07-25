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