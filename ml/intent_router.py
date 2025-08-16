"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/intent_router.py
# Intent Router for Complex Query Handling in PolicyCortex

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
import re
import json
import logging

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents"""
    # Information Seeking
    QUERY = "query"
    EXPLAIN = "explain"
    COMPARE = "compare"
    
    # Action Oriented
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CONFIGURE = "configure"
    
    # Analysis
    ANALYZE = "analyze"
    INVESTIGATE = "investigate"
    TROUBLESHOOT = "troubleshoot"
    PREDICT = "predict"
    
    # Remediation
    FIX = "fix"
    OPTIMIZE = "optimize"
    REMEDIATE = "remediate"
    
    # Reporting
    REPORT = "report"
    SUMMARIZE = "summarize"
    EXPORT = "export"
    
    # Navigation
    NAVIGATE = "navigate"
    SEARCH = "search"
    FILTER = "filter"
    
    # System
    HELP = "help"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

@dataclass
class Intent:
    """Detected intent with metadata"""
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    parameters: Dict[str, Any]
    sub_intents: List['Intent'] = field(default_factory=list)
    context_required: List[str] = field(default_factory=list)

@dataclass
class IntentRoute:
    """Route for handling an intent"""
    handler: str  # Handler function name
    priority: float
    params: Dict[str, Any]
    requires_confirmation: bool = False
    requires_auth: bool = True
    timeout_seconds: int = 30

@dataclass
class QueryDecomposition:
    """Decomposed complex query"""
    main_intent: Intent
    sub_queries: List[str]
    execution_order: List[int]
    dependencies: Dict[int, List[int]]  # Query dependencies
    aggregation_needed: bool

class IntentRouter:
    """Route complex queries to appropriate handlers"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.multi_intent_classifier = MultiIntentClassifier()
        self.query_decomposer = QueryDecomposer()
        self.entity_extractor = EntityExtractor()
        self.handlers = {}
        self._register_default_handlers()
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize intent detection patterns"""
        return {
            IntentType.QUERY: [
                r'\b(what|which|who|where|when|how many|how much)\b',
                r'\b(show|display|list|get)\b.*\?',
                r'\b(tell me about|information on)\b'
            ],
            IntentType.EXPLAIN: [
                r'\b(why|explain|clarify|describe)\b',
                r'\b(what does.*mean|how does.*work)\b',
                r'\b(reason for|cause of)\b'
            ],
            IntentType.CREATE: [
                r'\b(create|add|new|generate|provision)\b',
                r'\b(set up|establish|initialize)\b'
            ],
            IntentType.UPDATE: [
                r'\b(update|modify|change|edit|alter)\b',
                r'\b(set|configure|adjust)\b'
            ],
            IntentType.DELETE: [
                r'\b(delete|remove|destroy|terminate|drop)\b',
                r'\b(clean up|purge)\b'
            ],
            IntentType.ANALYZE: [
                r'\b(analyze|examine|review|assess|evaluate)\b',
                r'\b(check|inspect|audit)\b'
            ],
            IntentType.INVESTIGATE: [
                r'\b(investigate|debug|diagnose|trace)\b',
                r'\b(find out|look into|research)\b'
            ],
            IntentType.TROUBLESHOOT: [
                r'\b(troubleshoot|fix|resolve|solve)\b',
                r'\b(problem with|issue with|error in)\b'
            ],
            IntentType.PREDICT: [
                r'\b(predict|forecast|project|estimate)\b',
                r'\b(will|going to|future|trend)\b'
            ],
            IntentType.FIX: [
                r'\b(fix|repair|correct|patch)\b',
                r'\b(resolve|address|handle)\b'
            ],
            IntentType.OPTIMIZE: [
                r'\b(optimize|improve|enhance|tune)\b',
                r'\b(reduce cost|increase performance|maximize)\b'
            ],
            IntentType.REPORT: [
                r'\b(report|generate report|create report)\b',
                r'\b(summary of|overview of)\b'
            ],
            IntentType.HELP: [
                r'\b(help|assist|guide|tutorial)\b',
                r'\b(how to|how do I)\b'
            ]
        }
    
    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[IntentRoute]:
        """Route a query to appropriate handlers"""
        
        # Classify intents
        intents = self.multi_intent_classifier.classify(query)
        
        # Extract entities
        entities = self.entity_extractor.extract(query)
        
        # Build routes
        routes = []
        for intent in intents:
            if intent.confidence > 0.5:
                # Enrich intent with entities
                intent.entities = entities
                
                # Get handler for intent
                handler = self.get_handler(intent.type)
                
                # Build route
                route = IntentRoute(
                    handler=handler,
                    priority=intent.confidence,
                    params={
                        'intent': intent,
                        'entities': entities,
                        'context': context or {}
                    },
                    requires_confirmation=self._requires_confirmation(intent),
                    requires_auth=self._requires_auth(intent)
                )
                
                routes.append(route)
        
        # Sort by priority
        routes.sort(key=lambda x: x.priority, reverse=True)
        
        return routes
    
    def decompose_complex_query(self, query: str) -> QueryDecomposition:
        """Decompose complex query into sub-queries"""
        return self.query_decomposer.decompose(query)
    
    def get_handler(self, intent_type: IntentType) -> str:
        """Get handler name for intent type"""
        return self.handlers.get(intent_type, 'handle_unknown')
    
    def register_handler(self, intent_type: IntentType, handler_name: str):
        """Register a handler for an intent type"""
        self.handlers[intent_type] = handler_name
    
    def _register_default_handlers(self):
        """Register default intent handlers"""
        self.handlers = {
            IntentType.QUERY: 'handle_query',
            IntentType.EXPLAIN: 'handle_explanation',
            IntentType.CREATE: 'handle_creation',
            IntentType.UPDATE: 'handle_update',
            IntentType.DELETE: 'handle_deletion',
            IntentType.ANALYZE: 'handle_analysis',
            IntentType.INVESTIGATE: 'handle_investigation',
            IntentType.TROUBLESHOOT: 'handle_troubleshooting',
            IntentType.PREDICT: 'handle_prediction',
            IntentType.FIX: 'handle_remediation',
            IntentType.OPTIMIZE: 'handle_optimization',
            IntentType.REPORT: 'handle_reporting',
            IntentType.HELP: 'handle_help',
            IntentType.GREETING: 'handle_greeting',
            IntentType.GOODBYE: 'handle_goodbye',
            IntentType.UNKNOWN: 'handle_unknown'
        }
    
    def _requires_confirmation(self, intent: Intent) -> bool:
        """Check if intent requires user confirmation"""
        # Destructive or high-impact actions require confirmation
        return intent.type in [
            IntentType.DELETE,
            IntentType.CREATE,
            IntentType.UPDATE,
            IntentType.FIX,
            IntentType.REMEDIATE
        ]
    
    def _requires_auth(self, intent: Intent) -> bool:
        """Check if intent requires authentication"""
        # Most actions require auth except basic queries
        return intent.type not in [
            IntentType.GREETING,
            IntentType.GOODBYE,
            IntentType.HELP
        ]

class MultiIntentClassifier:
    """Classify multiple intents in a query"""
    
    def __init__(self):
        self.intent_keywords = self._initialize_keywords()
        self.intent_patterns = self._initialize_patterns()
    
    def _initialize_keywords(self) -> Dict[IntentType, List[str]]:
        """Initialize intent keywords"""
        return {
            IntentType.QUERY: ['what', 'show', 'list', 'get', 'find'],
            IntentType.EXPLAIN: ['why', 'explain', 'how', 'reason'],
            IntentType.CREATE: ['create', 'add', 'new', 'provision'],
            IntentType.UPDATE: ['update', 'modify', 'change', 'configure'],
            IntentType.DELETE: ['delete', 'remove', 'destroy', 'terminate'],
            IntentType.ANALYZE: ['analyze', 'examine', 'review', 'check'],
            IntentType.INVESTIGATE: ['investigate', 'debug', 'diagnose'],
            IntentType.TROUBLESHOOT: ['troubleshoot', 'fix', 'resolve'],
            IntentType.PREDICT: ['predict', 'forecast', 'will', 'future'],
            IntentType.FIX: ['fix', 'repair', 'correct', 'remediate'],
            IntentType.OPTIMIZE: ['optimize', 'improve', 'enhance'],
            IntentType.REPORT: ['report', 'summary', 'overview'],
            IntentType.HELP: ['help', 'assist', 'how to']
        }
    
    def _initialize_patterns(self) -> Dict[IntentType, re.Pattern]:
        """Initialize compiled regex patterns"""
        patterns = {}
        for intent_type, keywords in self.intent_keywords.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            patterns[intent_type] = re.compile(pattern, re.IGNORECASE)
        return patterns
    
    def classify(self, query: str) -> List[Intent]:
        """Classify intents in query"""
        intents = []
        query_lower = query.lower()
        
        # Check each intent type
        for intent_type, pattern in self.intent_patterns.items():
            matches = pattern.findall(query_lower)
            if matches:
                # Calculate confidence based on keyword matches
                confidence = min(len(matches) * 0.3, 1.0)
                
                # Boost confidence for exact matches
                for keyword in self.intent_keywords[intent_type]:
                    if f" {keyword} " in f" {query_lower} ":
                        confidence = min(confidence + 0.2, 1.0)
                
                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    entities={},
                    parameters={'matches': matches}
                )
                
                intents.append(intent)
        
        # If no intents found, mark as unknown
        if not intents:
            intents.append(Intent(
                type=IntentType.UNKNOWN,
                confidence=1.0,
                entities={},
                parameters={}
            ))
        
        # Sort by confidence
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        return intents

class QueryDecomposer:
    """Decompose complex queries into sub-queries"""
    
    def decompose(self, query: str) -> QueryDecomposition:
        """Decompose a complex query"""
        
        # Identify main intent
        classifier = MultiIntentClassifier()
        intents = classifier.classify(query)
        main_intent = intents[0] if intents else Intent(
            type=IntentType.UNKNOWN,
            confidence=1.0,
            entities={},
            parameters={}
        )
        
        # Split into sub-queries
        sub_queries = self._split_query(query)
        
        # Determine execution order
        execution_order = self._determine_order(sub_queries)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(sub_queries)
        
        # Check if aggregation is needed
        aggregation_needed = self._needs_aggregation(query)
        
        return QueryDecomposition(
            main_intent=main_intent,
            sub_queries=sub_queries,
            execution_order=execution_order,
            dependencies=dependencies,
            aggregation_needed=aggregation_needed
        )
    
    def _split_query(self, query: str) -> List[str]:
        """Split query into sub-queries"""
        # Split by conjunctions and punctuation
        split_patterns = [
            r'\band\b',
            r'\bthen\b',
            r'\bafter\b',
            r'\bbefore\b',
            r';',
            r','
        ]
        
        sub_queries = [query]
        for pattern in split_patterns:
            new_queries = []
            for q in sub_queries:
                parts = re.split(pattern, q, flags=re.IGNORECASE)
                new_queries.extend([p.strip() for p in parts if p.strip()])
            sub_queries = new_queries
        
        return sub_queries
    
    def _determine_order(self, sub_queries: List[str]) -> List[int]:
        """Determine execution order for sub-queries"""
        # Simple ordering - can be enhanced with dependency analysis
        order = []
        
        # Prioritize queries with "first" or "before"
        for i, query in enumerate(sub_queries):
            if any(word in query.lower() for word in ['first', 'before', 'initially']):
                order.append(i)
        
        # Add remaining queries
        for i in range(len(sub_queries)):
            if i not in order:
                order.append(i)
        
        return order
    
    def _identify_dependencies(self, sub_queries: List[str]) -> Dict[int, List[int]]:
        """Identify dependencies between sub-queries"""
        dependencies = {}
        
        for i, query in enumerate(sub_queries):
            deps = []
            
            # Check for references to previous queries
            if any(word in query.lower() for word in ['then', 'after', 'using']):
                # Depends on previous query
                if i > 0:
                    deps.append(i - 1)
            
            # Check for explicit references
            for j, other_query in enumerate(sub_queries):
                if i != j:
                    # Check if query i references results from query j
                    if self._references_query(query, other_query):
                        deps.append(j)
            
            if deps:
                dependencies[i] = deps
        
        return dependencies
    
    def _references_query(self, query1: str, query2: str) -> bool:
        """Check if query1 references results from query2"""
        # Extract key terms from query2
        key_terms = self._extract_key_terms(query2)
        
        # Check if query1 mentions these terms
        query1_lower = query1.lower()
        for term in key_terms:
            if term in query1_lower:
                return True
        
        return False
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple extraction - nouns and important words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = query.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _needs_aggregation(self, query: str) -> bool:
        """Check if query needs result aggregation"""
        aggregation_keywords = [
            'total', 'sum', 'average', 'mean', 'count',
            'maximum', 'minimum', 'all', 'combine', 'aggregate'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in aggregation_keywords)

class EntityExtractor:
    """Extract entities from queries"""
    
    def __init__(self):
        self.entity_patterns = {
            'resource': re.compile(r'\b(vm|storage|database|network|container|function)[-\w]*\b', re.I),
            'policy': re.compile(r'\b(policy|rule|standard|compliance|requirement)[-\w]*\b', re.I),
            'metric': re.compile(r'\b(cpu|memory|disk|network|cost|performance|latency)\b', re.I),
            'time': re.compile(r'\b(today|yesterday|tomorrow|last\s+\w+|next\s+\w+|\d+\s*(hour|day|week|month)s?\s+ago)\b', re.I),
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'percentage': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'user': re.compile(r'\b(user|admin|owner|contributor)[-\w]*\b', re.I),
            'location': re.compile(r'\b(east|west|north|south|central|us|europe|asia)[-\w]*\b', re.I)
        }
    
    def extract(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(query)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities

class IntentValidator:
    """Validate intents and their parameters"""
    
    def validate(self, intent: Intent, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an intent"""
        errors = []
        
        # Check required entities
        required = self._get_required_entities(intent.type)
        for entity_type in required:
            if entity_type not in intent.entities or not intent.entities[entity_type]:
                errors.append(f"Missing required entity: {entity_type}")
        
        # Check permissions
        if not self._check_permissions(intent, context):
            errors.append("Insufficient permissions for this action")
        
        # Check resource availability
        if not self._check_resources(intent, context):
            errors.append("Required resources not available")
        
        valid = len(errors) == 0
        return valid, errors
    
    def _get_required_entities(self, intent_type: IntentType) -> List[str]:
        """Get required entities for intent type"""
        requirements = {
            IntentType.CREATE: ['resource'],
            IntentType.UPDATE: ['resource'],
            IntentType.DELETE: ['resource'],
            IntentType.ANALYZE: ['resource', 'metric'],
            IntentType.PREDICT: ['metric', 'time']
        }
        
        return requirements.get(intent_type, [])
    
    def _check_permissions(self, intent: Intent, context: Dict[str, Any]) -> bool:
        """Check if user has permissions for intent"""
        # Simplified permission check
        user_role = context.get('user_role', 'viewer')
        
        if intent.type in [IntentType.DELETE, IntentType.CREATE, IntentType.UPDATE]:
            return user_role in ['admin', 'owner']
        elif intent.type in [IntentType.FIX, IntentType.REMEDIATE]:
            return user_role in ['admin', 'owner', 'contributor']
        else:
            return True  # Read operations allowed for all
    
    def _check_resources(self, intent: Intent, context: Dict[str, Any]) -> bool:
        """Check if required resources are available"""
        # Simplified resource check
        return True

# Export main components
__all__ = [
    'IntentRouter',
    'Intent',
    'IntentType',
    'IntentRoute',
    'QueryDecomposition',
    'MultiIntentClassifier',
    'QueryDecomposer',
    'EntityExtractor',
    'IntentValidator'
]