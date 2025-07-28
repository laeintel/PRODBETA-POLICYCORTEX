"""
Conversational Governance Intelligence System
Patent 3: Advanced governance-specific conversational AI with context awareness,
    multi-modal analysis, and intelligent recommendation systems.
"""

import asyncio
import uuid
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy

from backend.core.config import settings
from backend.core.redis_client import redis_client
from backend.core.exceptions import APIError

logger = logging.getLogger(__name__)


class ConversationContext(str, Enum):
    """Types of conversation contexts"""
    POLICY_INQUIRY = "policy_inquiry"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    COST_ANALYSIS = "cost_analysis"
    SECURITY_REVIEW = "security_review"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    AUDIT_PREPARATION = "audit_preparation"
    INCIDENT_RESPONSE = "incident_response"
    STRATEGIC_PLANNING = "strategic_planning"
    KNOWLEDGE_EXPLORATION = "knowledge_exploration"


class IntentType(str, Enum):
    """Types of user intents"""
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    COMPLAINT = "complaint"
    SUGGESTION = "suggestion"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    EXPLORATION = "exploration"


class ModalityType(str, Enum):
    """Types of input modalities"""
    TEXT = "text"
    VOICE = "voice"
    DOCUMENT = "document"
    IMAGE = "image"
    STRUCTURED_DATA = "structured_data"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    user_id: str
    timestamp: datetime
    input_text: str
    intent: IntentType
    context: ConversationContext
    modality: ModalityType
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: float = 0.0
    confidence: float = 0.0
    response: Optional[str] = None
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Represents a complete conversation session"""
    session_id: str
    user_id: str
    started_at: datetime
    updated_at: datetime
    turns: List[ConversationTurn] = field(default_factory=list)
    context_history: List[ConversationContext] = field(default_factory=list)
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    session_summary: Optional[str] = None
    outcome: Optional[str] = None
    satisfaction_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceKnowledge:
    """Represents governance knowledge"""
    knowledge_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    embedding: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    confidence: float = 1.0


class ContextAwareNLU:
    """Natural Language Understanding with governance context awareness"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.sentiment_analyzer = None
        self.nlp = None
        self._initialized = False

    async def initialize(self):
        """Initialize NLU models"""
        try:
            # Load governance-specific models
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            self.model = AutoModel.from_pretrained('microsoft/DialoGPT-medium')

            # Intent classification
            self.intent_classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                return_all_scores=True
            )

            # Named entity recognition
            self.nlp = spacy.load("en_core_web_sm")

            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )

            # Governance-specific entity patterns
            self._initialize_governance_patterns()

            self._initialized = True
            logger.info("Context-aware NLU initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize NLU: {str(e)}")
            raise

    def _initialize_governance_patterns(self):
        """Initialize governance-specific entity patterns"""
        self.governance_patterns = {
            'policy_reference': re.compile(
                r'\b(?:policy|procedure|guideline)\s+[\w\-\.]+\b',
                re.IGNORECASE
            ),
            'resource_id': re.compile(r'\b(?:rg-|vm-|st-|kv-|aks-|app-|db-)\w+\b', re.IGNORECASE),
            'compliance_framework': re.compile(
                r'\b(?:SOX|GDPR|HIPAA|PCI\s*DSS|ISO\s*27001|NIST)\b',
                re.IGNORECASE
            ),
            'risk_level': re.compile(r'\b(?:critical|high|medium|low)\s*risk\b', re.IGNORECASE),
            'cost_amount': re.compile(
                r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|USD|cents?)\b',
                re.IGNORECASE
            ),
            'percentage': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'time_period': re.compile(
                r'\b(?:daily|weekly|monthly|quarterly|annually|last\s+\w+|next\s+\w+)\b',
                re.IGNORECASE
            )
        }

    async def analyze_input(self,
                          text: str,
                          conversation_history: List[ConversationTurn],
                          user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user input with context awareness"""

        if not self._initialized:
            await self.initialize()

        try:
            # Extract entities
            entities = await self._extract_entities(text)

            # Classify intent with context
            intent = await self._classify_intent(text, conversation_history)

            # Analyze sentiment
            sentiment = await self._analyze_sentiment(text)

            # Determine conversation context
            context = await self._determine_context(text, conversation_history, entities)

            # Extract governance-specific elements
            governance_elements = await self._extract_governance_elements(text)

            # Calculate confidence
            confidence = self._calculate_confidence(intent, entities, sentiment)

            return {
                'entities': entities,
                'intent': intent,
                'sentiment': sentiment,
                'context': context,
                'governance_elements': governance_elements,
                'confidence': confidence,
                'processed_text': text,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"NLU analysis failed: {str(e)}")
            raise APIError(f"NLU analysis failed: {str(e)}", status_code=500)

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []

        # Use spaCy for standard entities
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.9,
                'type': 'standard'
            })

        # Extract governance-specific entities
        for pattern_name, pattern in self.governance_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'type': 'governance'
                })

        return entities

    async def _classify_intent(self,
                             text: str,
                             conversation_history: List[ConversationTurn]) -> Dict[str, Any]:
        """Classify user intent with conversation context"""

        # Define intent labels for governance scenarios
        intent_labels = [
            "asking question about policy",
            "requesting compliance check",
            "reporting security incident",
            "seeking cost optimization",
            "requesting resource information",
            "making suggestion for improvement",
            "requesting clarification",
            "confirming understanding",
            "exploring governance options"
        ]

        # Add conversation context
        context_text = ""
        if conversation_history:
            recent_turns = conversation_history[-3:]  # Last 3 turns
            context_text = " ".join([turn.input_text for turn in recent_turns])
            context_text = f"Previous context: {context_text}. Current query: {text}"
        else:
            context_text = text

        # Classify intent
        results = self.intent_classifier(context_text, intent_labels)

        # Find best intent
        best_intent = max(results, key=lambda x: x['score'])

        # Map to IntentType
        intent_mapping = {
            "asking question": IntentType.QUESTION,
            "requesting": IntentType.REQUEST,
            "reporting": IntentType.COMMAND,
            "seeking": IntentType.REQUEST,
            "making suggestion": IntentType.SUGGESTION,
            "requesting clarification": IntentType.CLARIFICATION,
            "confirming": IntentType.CONFIRMATION,
            "exploring": IntentType.EXPLORATION
        }

        intent_type = IntentType.QUESTION  # Default
        for key, value in intent_mapping.items():
            if key in best_intent['label'].lower():
                intent_type = value
                break

        return {
            'type': intent_type,
            'confidence': best_intent['score'],
            'raw_classification': best_intent,
            'all_scores': results
        }

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        results = self.sentiment_analyzer(text)

        # Convert to normalized score (-1 to 1)
        sentiment_score = 0.0
        if results:
            best_sentiment = max(results, key=lambda x: x['score'])
            if 'positive' in best_sentiment['label'].lower():
                sentiment_score = best_sentiment['score']
            elif 'negative' in best_sentiment['label'].lower():
                sentiment_score = -best_sentiment['score']

        return {
            'score': sentiment_score,
            'raw_results': results
        }

    async def _determine_context(self,
                               text: str,
                               conversation_history: List[ConversationTurn],
                               entities: List[Dict[str, Any]]) -> ConversationContext:
        """Determine conversation context"""

        # Context keywords mapping
        context_keywords = {
            ConversationContext.POLICY_INQUIRY: ['policy', 'rule', 'guideline', 'procedure', 'standard'],
            ConversationContext.COMPLIANCE_CHECK: ['compliance', 'audit', 'regulation', 'violation', 'SOX', 'GDPR'],
            ConversationContext.RISK_ASSESSMENT: ['risk', 'threat', 'vulnerability', 'security', 'assessment'],
            ConversationContext.COST_ANALYSIS: ['cost', 'budget', 'expense', 'saving', 'optimization', 'price'],
            ConversationContext.SECURITY_REVIEW: ['security', 'breach', 'incident', 'vulnerability', 'malware'],
            ConversationContext.RESOURCE_OPTIMIZATION: ['optimize', 'performance', 'efficiency', 'resource', 'scaling'],
            ConversationContext.AUDIT_PREPARATION: ['audit', 'preparation', 'documentation', 'evidence', 'report'],
            ConversationContext.INCIDENT_RESPONSE: ['incident', 'emergency', 'outage', 'failure', 'response'],
            ConversationContext.STRATEGIC_PLANNING: ['strategy', 'planning', 'roadmap', 'future', 'vision'],
            ConversationContext.KNOWLEDGE_EXPLORATION: ['learn', 'understand', 'explain', 'what', 'how', 'why']
        }

        text_lower = text.lower()
        context_scores = {}

        # Score contexts based on keyword presence
        for context, keywords in context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            context_scores[context] = score

        # Boost score based on entities
        for entity in entities:
            if entity['type'] == 'governance':
                if 'policy' in entity['label']:
                    context_scores[ConversationContext.POLICY_INQUIRY] += 2
                elif 'compliance' in entity['label']:
                    context_scores[ConversationContext.COMPLIANCE_CHECK] += 2
                elif 'cost' in entity['label']:
                    context_scores[ConversationContext.COST_ANALYSIS] += 2

        # Consider conversation history
        if conversation_history:
            recent_context = conversation_history[-1].context
            context_scores[recent_context] += 1  # Boost current context

        # Return highest scoring context
        if context_scores:
            best_context = max(context_scores.items(), key=lambda x: x[1])
            if best_context[1] > 0:
                return best_context[0]

        return ConversationContext.KNOWLEDGE_EXPLORATION  # Default

    async def _extract_governance_elements(self, text: str) -> Dict[str, Any]:
        """Extract governance-specific elements"""
        elements = {
            'policies_mentioned': [],
            'compliance_frameworks': [],
            'resources_mentioned': [],
            'risk_indicators': [],
            'cost_references': [],
            'time_references': [],
            'stakeholders': []
        }

        # Extract using patterns
        for pattern_name, pattern in self.governance_patterns.items():
            matches = [match.group() for match in pattern.finditer(text)]
            if pattern_name == 'compliance_framework':
                elements['compliance_frameworks'].extend(matches)
            elif pattern_name == 'resource_id':
                elements['resources_mentioned'].extend(matches)
            elif pattern_name == 'cost_amount':
                elements['cost_references'].extend(matches)
            elif pattern_name == 'time_period':
                elements['time_references'].extend(matches)
            elif pattern_name == 'risk_level':
                elements['risk_indicators'].extend(matches)

        return elements

    def _calculate_confidence(self,
                            intent: Dict[str, Any],
                            entities: List[Dict[str, Any]],
                            sentiment: Dict[str, Any]) -> float:
        """Calculate overall confidence in analysis"""

        # Weight different components
        intent_confidence = intent.get('confidence', 0.0) * 0.4
        entity_confidence = np.mean(
            [e.get('confidence',
            0.0) for e in entities]
        ) * 0.3 if entities else 0.0
        sentiment_confidence = max(
            [r['score'] for r in sentiment.get('raw_results',
            [])]) * 0.3 if sentiment.get('raw_results'
        ) else 0.0

        return min(1.0, intent_confidence + entity_confidence + sentiment_confidence)


class GovernanceKnowledgeBase:
    """Knowledge base for governance information"""

    def __init__(self):
        self.sentence_transformer = None
        self.knowledge_store: Dict[str, GovernanceKnowledge] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.category_index: Dict[str, List[str]] = defaultdict(list)
        self._initialized = False

    async def initialize(self):
        """Initialize knowledge base"""
        try:
            # Load sentence transformer for semantic search
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Load governance knowledge
            await self._load_governance_knowledge()

            self._initialized = True
            logger.info("Governance knowledge base initialized")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            raise

    async def _load_governance_knowledge(self):
        """Load governance knowledge from various sources"""

        # Sample governance knowledge - in production, load from databases/documents
        sample_knowledge = [
            {
                'title': 'Azure Resource Tagging Policy',
                'content': 'All Azure resources must be tagged with Environment, Owner, CostCenter, and
                    Project tags. Resources without proper tags will be flagged for compliance review.',
                'category': 'tagging',
                'tags': ['azure', 'tagging', 'compliance', 'policy'],
                'source': 'corporate_policies'
            },
            {
                'title': 'Data Classification Standards',
                'content': 'Data must be classified as Public, Internal, Confidential, or Restricted based on sensitivity. Each classification has specific handling requirements.',
                'category': 'data_governance',
                'tags': ['data', 'classification', 'security', 'standards'],
                'source': 'security_policies'
            },
            {
                'title': 'Cost Optimization Guidelines',
                'content': 'Implement auto-scaling, use reserved instances for predictable workloads, and
                    regularly review unutilized resources.',
                'category': 'cost_management',
                'tags': ['cost', 'optimization', 'azure', 'best_practices'],
                'source': 'operational_guidelines'
            },
            {
                'title': 'Security Incident Response',
                'content': 'Security incidents must be reported within 1 hour. Follow the NIST incident response framework: Preparation, Detection, Containment, Eradication, Recovery.',
                'category': 'security',
                'tags': ['security', 'incident', 'response', 'NIST'],
                'source': 'security_procedures'
            },
            {
                'title': 'Compliance Audit Checklist',
                'content': 'Quarterly audits must verify resource configurations, access controls, data handling procedures, and
                    policy adherence.',
                'category': 'compliance',
                'tags': ['audit', 'compliance', 'checklist', 'quarterly'],
                'source': 'audit_procedures'
            }
        ]

        for idx, knowledge_data in enumerate(sample_knowledge):
            knowledge = GovernanceKnowledge(
                knowledge_id=f"gov_knowledge_{idx}",
                title=knowledge_data['title'],
                content=knowledge_data['content'],
                category=knowledge_data['category'],
                tags=knowledge_data['tags'],
                source=knowledge_data['source']
            )

            # Generate embedding
            text_for_embedding = f"{knowledge.title} {knowledge.content}"
            knowledge.embedding = self.sentence_transformer.encode(text_for_embedding)

            # Store knowledge
            self.knowledge_store[knowledge.knowledge_id] = knowledge
            self.category_index[knowledge.category].append(knowledge.knowledge_id)

    async def search_knowledge(self,
                             query: str,
                             context: ConversationContext,
                             top_k: int = 5) -> List[GovernanceKnowledge]:
        """Search for relevant governance knowledge"""

        if not self._initialized:
            await self.initialize()

        try:
            # Generate query embedding
            query_embedding = self.sentence_transformer.encode(query)

            # Calculate similarities
            similarities = []
            for knowledge_id, knowledge in self.knowledge_store.items():
                if knowledge.embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        knowledge.embedding.reshape(1, -1)
                    )[0][0]

                    # Boost based on context relevance
                    context_boost = self._calculate_context_boost(knowledge, context)
                    final_score = similarity + context_boost

                    similarities.append((knowledge_id, final_score))

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for knowledge_id, score in similarities[:top_k]:
                knowledge = self.knowledge_store[knowledge_id]
                knowledge.relevance_score = score
                results.append(knowledge)

            return results

        except Exception as e:
            logger.error(f"Knowledge search failed: {str(e)}")
            return []

    def _calculate_context_boost(self,
                               knowledge: GovernanceKnowledge,
                               context: ConversationContext) -> float:
        """Calculate context-based relevance boost"""

        context_mapping = {
            ConversationContext.POLICY_INQUIRY: ['policy', 'procedure', 'guideline'],
            ConversationContext.COMPLIANCE_CHECK: ['compliance', 'audit', 'regulation'],
            ConversationContext.SECURITY_REVIEW: ['security', 'incident', 'threat'],
            ConversationContext.COST_ANALYSIS: ['cost', 'optimization', 'budget'],
            ConversationContext.RESOURCE_OPTIMIZATION: ['performance', 'scaling', 'efficiency']
        }

        boost = 0.0
        if context in context_mapping:
            relevant_keywords = context_mapping[context]
            for keyword in relevant_keywords:
                if keyword in knowledge.tags or keyword in knowledge.content.lower():
                    boost += 0.1

        return min(0.5, boost)  # Max boost of 0.5


class IntelligentResponseGenerator:
    """Generates intelligent responses for governance conversations"""

    def __init__(self):
        self.response_templates = {}
        self.personalization_engine = None
        self._initialized = False

    async def initialize(self):
        """Initialize response generator"""
        try:
            self._initialize_response_templates()
            self._initialized = True
            logger.info("Intelligent response generator initialized")

        except Exception as e:
            logger.error(f"Failed to initialize response generator: {str(e)}")
            raise

    def _initialize_response_templates(self):
        """Initialize response templates for different contexts"""

        self.response_templates = {
            ConversationContext.POLICY_INQUIRY: {
                'greeting': "I'll help you understand our governance policies.",
                'information_template': "Based on our policies, {information}. Here are the key points:\n{details}",
                'clarification': "To provide more specific guidance, could you clarify {clarification_needed}?",
                'action_suggestion': "I recommend {action}. Would you like me to {next_step}?"
            },
            ConversationContext.COMPLIANCE_CHECK: {
                'greeting': "I'll help you check compliance requirements.",
                'assessment_template': "Compliance Assessment:\n✓ {compliant_items}\n⚠️ {attention_items}\n❌ {non_compliant_items}",
                'recommendation': "To ensure compliance, I recommend: {recommendations}",
                'next_steps': "Next steps: {steps}"
            },
            ConversationContext.COST_ANALYSIS: {
                'greeting': "I'll help analyze cost implications and optimization opportunities.",
                'analysis_template': "Cost Analysis:\n💰 Current costs: {current_costs}\n📈 Trends: {trends}\n💡 Optimization opportunities: {optimizations}",
                'savings_potential': "Potential savings: {savings} through {methods}",
                'action_plan': "Recommended actions: {actions}"
            },
            ConversationContext.SECURITY_REVIEW: {
                'greeting': "I'll help with security assessment and recommendations.",
                'security_template': "Security Review:\n🔒 Secure: {secure_items}\n⚠️ Needs attention: {attention_items}\n🚨 Critical: {critical_items}",
                'threat_analysis': "Threat Analysis: {threats}",
                'mitigation': "Recommended mitigations: {mitigations}"
            }
        }

    async def generate_response(self,
                              conversation_analysis: Dict[str, Any],
                              knowledge_results: List[GovernanceKnowledge],
                              conversation_history: List[ConversationTurn],
                              user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent response"""

        if not self._initialized:
            await self.initialize()

        try:
            context = ConversationContext(conversation_analysis['context'])
            intent = conversation_analysis['intent']
            entities = conversation_analysis['entities']

            # Generate contextual response
            response_parts = []

            # Add greeting if first interaction or context change
            if self._should_add_greeting(conversation_history, context):
                greeting = self.response_templates.get(
                    context,
                    {}).get('greeting',
                    'How can I help you today?'
                )
                response_parts.append(greeting)

            # Add main content based on knowledge
            main_content = await self._generate_main_content(
                knowledge_results, context, intent, entities
            )
            response_parts.append(main_content)

            # Add recommendations
            recommendations = await self._generate_recommendations(
                context, knowledge_results, entities
            )
            if recommendations:
                response_parts.append(f"\n**Recommendations:**\n{recommendations}")

            # Add follow-up questions
            follow_up = await self._generate_follow_up(context, intent, entities)
            if follow_up:
                response_parts.append(f"\n{follow_up}")

            # Combine response
            full_response = "\n\n".join(response_parts)

            # Generate action suggestions
            suggested_actions = await self._suggest_actions(context, intent, entities)

            return {
                'response_text': full_response,
                'suggested_actions': suggested_actions,
                'confidence': conversation_analysis.get('confidence', 0.0),
                'context': context.value,
                'knowledge_sources': [k.knowledge_id for k in knowledge_results],
                'response_metadata': {
                    'response_type': 'contextual',
                    'template_used': context.value,
                    'knowledge_count': len(knowledge_results)
                }
            }

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return {
                'response_text': "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?",
                'suggested_actions': [],
                'confidence': 0.0,
                'context': 'error',
                'knowledge_sources': [],
                'response_metadata': {'error': str(e)}
            }

    def _should_add_greeting(self,
                           conversation_history: List[ConversationTurn],
                           current_context: ConversationContext) -> bool:
        """Determine if greeting should be added"""

        if not conversation_history:
            return True

        # Add greeting if context changed
        if conversation_history:
            last_context = conversation_history[-1].context
            if last_context != current_context:
                return True

        return False

    async def _generate_main_content(self,
                                   knowledge_results: List[GovernanceKnowledge],
                                   context: ConversationContext,
                                   intent: Dict[str, Any],
                                   entities: List[Dict[str, Any]]) -> str:
        """Generate main response content"""

        if not knowledge_results:
            return "I don't have specific information about your query, but I can help you find the right resources or connect you with the appropriate team."

        # Use most relevant knowledge
        primary_knowledge = knowledge_results[0]

        content_parts = []

        # Add primary information
        content_parts.append(f"**{primary_knowledge.title}**")
        content_parts.append(primary_knowledge.content)

        # Add related information if available
        if len(knowledge_results) > 1:
            content_parts.append("\n**Related Information:**")
            for knowledge in knowledge_results[1:3]:  # Top 2 additional
                content_parts.append(f"• {knowledge.title}: {knowledge.content[:100]}...")

        return "\n".join(content_parts)

    async def _generate_recommendations(self,
                                      context: ConversationContext,
                                      knowledge_results: List[GovernanceKnowledge],
                                      entities: List[Dict[str, Any]]) -> Optional[str]:
        """Generate contextual recommendations"""

        recommendations = []

        if context == ConversationContext.POLICY_INQUIRY:
            recommendations.append("Review the complete policy documentation")
            recommendations.append("Consult with your governance team for specific implementations")

        elif context == ConversationContext.COMPLIANCE_CHECK:
            recommendations.append("Run automated compliance scans")
            recommendations.append("Document any exceptions or remediation plans")

        elif context == ConversationContext.COST_ANALYSIS:
            recommendations.append("Set up cost alerts for budget monitoring")
            recommendations.append("Schedule regular cost optimization reviews")

        elif context == ConversationContext.SECURITY_REVIEW:
            recommendations.append("Implement recommended security controls")
            recommendations.append("Schedule regular security assessments")

        return "\n".join([f"• {rec}" for rec in recommendations]) if recommendations else None

    async def _generate_follow_up(self,
                                context: ConversationContext,
                                intent: Dict[str, Any],
                                entities: List[Dict[str, Any]]) -> Optional[str]:
        """Generate follow-up questions"""

        follow_ups = []

        if intent['type'] == IntentType.QUESTION:
            follow_ups.append("Would you like more details about any specific aspect?")

        elif intent['type'] == IntentType.REQUEST:
            follow_ups.append("Should I proceed with this request or would you like to review the details first?")

        if context == ConversationContext.POLICY_INQUIRY:
            follow_ups.append("Are there any specific scenarios or use cases you'd like me to address?")

        return " ".join(follow_ups) if follow_ups else None

    async def _suggest_actions(self,
                             context: ConversationContext,
                             intent: Dict[str, Any],
                             entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest possible actions"""

        actions = []

        if context == ConversationContext.COMPLIANCE_CHECK:
            actions.append({
                'action': 'run_compliance_scan',
                'label': 'Run Compliance Scan',
                'description': 'Execute automated compliance checking'
            })

        elif context == ConversationContext.COST_ANALYSIS:
            actions.append({
                'action': 'generate_cost_report',
                'label': 'Generate Cost Report',
                'description': 'Create detailed cost analysis report'
            })

        elif context == ConversationContext.SECURITY_REVIEW:
            actions.append({
                'action': 'security_assessment',
                'label': 'Security Assessment',
                'description': 'Run security vulnerability assessment'
            })

        # Always offer to connect with experts
        actions.append({
            'action': 'connect_expert',
            'label': 'Connect with Expert',
            'description': 'Connect with a governance specialist'
        })

        return actions


class ConversationGovernanceIntelligence:
    """Main conversational governance intelligence system"""

    def __init__(self):
        self.nlu = ContextAwareNLU()
        self.knowledge_base = GovernanceKnowledgeBase()
        self.response_generator = IntelligentResponseGenerator()
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.conversation_history: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self._initialized = False

    async def initialize(self):
        """Initialize the governance intelligence system"""
        try:
            await self.nlu.initialize()
            await self.knowledge_base.initialize()
            await self.response_generator.initialize()

            self._initialized = True
            logger.info("Conversational Governance Intelligence initialized")

        except Exception as e:
            logger.error(f"Failed to initialize governance intelligence: {str(e)}")
            raise

    async def process_conversation(self,
                                 user_id: str,
                                 input_text: str,
                                 session_id: Optional[str] = None,
                                 modality: ModalityType = ModalityType.TEXT,
                                 user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a conversation turn"""

        if not self._initialized:
            await self.initialize()

        try:
            # Get or create session
            if not session_id:
                session_id = str(uuid.uuid4())

            session = await self._get_or_create_session(user_id, session_id)

            # Analyze input
            conversation_analysis = await self.nlu.analyze_input(
                input_text,
                session.turns,
                user_context or {}
            )

            # Search knowledge base
            knowledge_results = await self.knowledge_base.search_knowledge(
                input_text,
                ConversationContext(conversation_analysis['context'])
            )

            # Generate response
            response_data = await self.response_generator.generate_response(
                conversation_analysis,
                knowledge_results,
                session.turns,
                user_context or {}
            )

            # Create conversation turn
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                user_id=user_id,
                timestamp=datetime.now(),
                input_text=input_text,
                intent=IntentType(conversation_analysis['intent']['type']),
                context=ConversationContext(conversation_analysis['context']),
                modality=modality,
                entities=conversation_analysis['entities'],
                sentiment=conversation_analysis['sentiment']['score'],
                confidence=conversation_analysis['confidence'],
                response=response_data['response_text'],
                metadata={
                    'nlu_analysis': conversation_analysis,
                    'knowledge_results': [k.knowledge_id for k in knowledge_results],
                    'response_metadata': response_data['response_metadata']
                }
            )

            # Update session
            session.turns.append(turn)
            session.context_history.append(turn.context)
            session.updated_at = datetime.now()

            # Cache session
            await self._cache_session(session)

            return {
                'session_id': session_id,
                'turn_id': turn.turn_id,
                'response': response_data['response_text'],
                'suggested_actions': response_data['suggested_actions'],
                'context': conversation_analysis['context'],
                'intent': conversation_analysis['intent']['type'].value,
                'confidence': conversation_analysis['confidence'],
                'entities': conversation_analysis['entities'],
                'knowledge_sources': response_data['knowledge_sources'],
                'metadata': response_data['response_metadata']
            }

        except Exception as e:
            logger.error(f"Conversation processing failed: {str(e)}")
            raise APIError(f"Conversation processing failed: {str(e)}", status_code=500)

    async def _get_or_create_session(self, user_id: str, session_id: str) -> ConversationSession:
        """Get or create conversation session"""

        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try to load from cache
        cached_session = await self._load_session_from_cache(session_id)
        if cached_session:
            self.active_sessions[session_id] = cached_session
            return cached_session

        # Create new session
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.active_sessions[session_id] = session
        return session

    async def _cache_session(self, session: ConversationSession):
        """Cache session data"""
        cache_key = f"conversation_session:{session.session_id}"
        session_data = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'started_at': session.started_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'turns_count': len(session.turns),
            'context_history': [ctx.value for ctx in session.context_history],
            'metadata': session.metadata
        }

        await redis_client.setex(
            cache_key,
            timedelta(hours=24),
            json.dumps(session_data)
        )

    async def _load_session_from_cache(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from cache"""
        cache_key = f"conversation_session:{session_id}"

        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # Note: This is simplified - in production, reconstruct full session
                return None  # For now, return None to create new session
        except Exception:
            pass

        return None

    async def get_conversation_insights(self, user_id: str) -> Dict[str, Any]:
        """Get conversation insights for a user"""

        user_sessions = [s for s in self.active_sessions.values() if s.user_id == user_id]

        if not user_sessions:
            return {'total_sessions': 0, 'insights': []}

        # Analyze conversation patterns
        total_turns = sum(len(s.turns) for s in user_sessions)
        contexts = [turn.context for s in user_sessions for turn in s.turns]
        context_distribution = {ctx.value: contexts.count(ctx) for ctx in set(contexts)}

        # Sentiment analysis
        sentiments = [turn.sentiment for s in user_sessions for turn in s.turns]
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        return {
            'total_sessions': len(user_sessions),
            'total_turns': total_turns,
            'context_distribution': context_distribution,
            'average_sentiment': avg_sentiment,
            'most_common_context': max(
                context_distribution.items(),
                key=lambda x: x[1]
            )[0] if context_distribution else None,
            'insights': [
                f"User has {total_turns} conversation turns across {len(user_sessions)} sessions",
                f"Most common topic: {max(
                    context_distribution.items(),
                    key=lambda x: x[1]
                )[0] if context_distribution else 'N/A'}",
                f"Average sentiment: {'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}"
            ]
        }


# Global instance
governance_intelligence = ConversationGovernanceIntelligence()
