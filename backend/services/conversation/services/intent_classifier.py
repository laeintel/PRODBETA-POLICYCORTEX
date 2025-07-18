"""
Intent Classifier and Entity Extractor Service.
Uses Azure OpenAI for intent classification and entity extraction.
"""

import re
import json
from typing import Dict, Any, List, Optional
import structlog
from openai import AsyncAzureOpenAI
import spacy
from spacy.matcher import Matcher

from ....shared.config import get_settings
from ..models import (
    ConversationIntent,
    EntityType,
    Entity,
    IntentClassificationResult
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class IntentClassifier:
    """Intent classification and entity extraction service."""
    
    def __init__(self):
        self.settings = settings
        self.client = None
        self.nlp = None
        self.matcher = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP pipeline."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize matcher for entity extraction
            self.matcher = Matcher(self.nlp.vocab)
            
            # Add patterns for Azure resources
            self._add_azure_patterns()
            
        except Exception as e:
            logger.warning("spacy_model_not_available", error=str(e))
            self.nlp = None
            self.matcher = None
    
    def _add_azure_patterns(self):
        """Add patterns for Azure entity matching."""
        if not self.matcher:
            return
        
        # Resource group patterns
        rg_patterns = [
            [{"LOWER": "resource"}, {"LOWER": "group"}, {"IS_ALPHA": True}],
            [{"LOWER": "rg"}, {"IS_ALPHA": True}],
            [{"TEXT": {"REGEX": r"^rg-[\w-]+$"}}]
        ]
        self.matcher.add("RESOURCE_GROUP", rg_patterns)
        
        # Subscription patterns
        sub_patterns = [
            [{"LOWER": "subscription"}, {"IS_ALPHA": True}],
            [{"LOWER": "sub"}, {"IS_ALPHA": True}],
            [{"TEXT": {"REGEX": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"}}]
        ]
        self.matcher.add("SUBSCRIPTION", sub_patterns)
        
        # Cost patterns
        cost_patterns = [
            [{"LOWER": {"IN": ["$", "dollar", "dollars", "cost", "price", "spend"]}},
             {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["dollars", "usd", "cost"]}}]
        ]
        self.matcher.add("COST_THRESHOLD", cost_patterns)
        
        # Date patterns
        date_patterns = [
            [{"LOWER": {"IN": ["last", "past"]}}, {"LIKE_NUM": True},
             {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months"]}}],
            [{"LOWER": {"IN": ["today", "yesterday", "this"]}},
             {"LOWER": {"IN": ["week", "month", "year"]}}]
        ]
        self.matcher.add("DATE_RANGE", date_patterns)
    
    async def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Get Azure OpenAI client."""
        if not self.client:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.ai.azure_openai_key,
                api_version=self.settings.ai.azure_openai_api_version,
                azure_endpoint=self.settings.ai.azure_openai_endpoint
            )
        return self.client
    
    async def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentClassificationResult:
        """Classify intent of user message."""
        try:
            # First try rule-based classification
            rule_based_intent = self._classify_intent_rules(text)
            
            # If high confidence from rules, use it
            if rule_based_intent["confidence"] > 0.8:
                entities = await self.extract_entities(text, rule_based_intent["intent"])
                return IntentClassificationResult(
                    intent=rule_based_intent["intent"],
                    confidence=rule_based_intent["confidence"],
                    entities=entities,
                    sub_intents=rule_based_intent.get("sub_intents", [])
                )
            
            # Otherwise use AI classification
            ai_intent = await self._classify_intent_ai(text, context)
            entities = await self.extract_entities(text, ai_intent["intent"])
            
            return IntentClassificationResult(
                intent=ai_intent["intent"],
                confidence=ai_intent["confidence"],
                entities=entities,
                sub_intents=ai_intent.get("sub_intents", [])
            )
            
        except Exception as e:
            logger.error("intent_classification_failed", error=str(e))
            
            # Fallback to GENERAL_QUERY
            entities = await self.extract_entities(text, ConversationIntent.GENERAL_QUERY)
            return IntentClassificationResult(
                intent=ConversationIntent.GENERAL_QUERY,
                confidence=0.5,
                entities=entities,
                sub_intents=[]
            )
    
    def _classify_intent_rules(self, text: str) -> Dict[str, Any]:
        """Rule-based intent classification."""
        text_lower = text.lower()
        
        # Greeting patterns
        greeting_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(how are you|what\'s up|greetings)\b'
        ]
        
        # Help patterns
        help_patterns = [
            r'\b(help|assist|support|guide|tutorial)\b',
            r'\b(how to|what is|what are|explain)\b'
        ]
        
        # Cost analysis patterns
        cost_patterns = [
            r'\b(cost|price|spend|budget|bill|expense)\b',
            r'\b(how much|total cost|monthly cost|daily cost)\b',
            r'\b(save money|reduce cost|optimize cost)\b'
        ]
        
        # Policy patterns
        policy_patterns = [
            r'\b(policy|policies|compliance|compliant)\b',
            r'\b(govern|governance|rule|rules)\b',
            r'\b(violat|violation|non-compliant)\b'
        ]
        
        # Resource management patterns
        resource_patterns = [
            r'\b(resource|resources|vm|virtual machine|storage)\b',
            r'\b(create|delete|modify|update|manage)\b',
            r'\b(list|show|display|view)\b.*\b(resource|vm|storage)\b'
        ]
        
        # Security patterns
        security_patterns = [
            r'\b(security|secure|permission|access|role)\b',
            r'\b(vulnerability|threat|risk|attack)\b',
            r'\b(rbac|role-based|access control)\b'
        ]
        
        # RBAC patterns
        rbac_patterns = [
            r'\b(rbac|role|permission|access|authorization)\b',
            r'\b(user|group|assign|grant|revoke)\b',
            r'\b(who has access|who can|permissions for)\b'
        ]
        
        # Network patterns
        network_patterns = [
            r'\b(network|networking|vpc|vnet|subnet)\b',
            r'\b(firewall|nsg|security group|route)\b',
            r'\b(connectivity|connection|endpoint)\b'
        ]
        
        # Optimization patterns
        optimization_patterns = [
            r'\b(optimize|optimization|improve|enhance)\b',
            r'\b(recommend|suggestion|best practice)\b',
            r'\b(performance|efficiency|utilization)\b'
        ]
        
        # Check patterns and calculate confidence
        patterns = [
            (greeting_patterns, ConversationIntent.GREETING, 0.95),
            (help_patterns, ConversationIntent.HELP, 0.9),
            (cost_patterns, ConversationIntent.COST_ANALYSIS, 0.85),
            (policy_patterns, ConversationIntent.POLICY_QUERY, 0.85),
            (resource_patterns, ConversationIntent.RESOURCE_MANAGEMENT, 0.8),
            (security_patterns, ConversationIntent.SECURITY_ANALYSIS, 0.8),
            (rbac_patterns, ConversationIntent.RBAC_QUERY, 0.85),
            (network_patterns, ConversationIntent.NETWORK_ANALYSIS, 0.8),
            (optimization_patterns, ConversationIntent.OPTIMIZATION_SUGGESTION, 0.8)
        ]
        
        max_confidence = 0.0
        best_intent = ConversationIntent.GENERAL_QUERY
        sub_intents = []
        
        for pattern_list, intent, base_confidence in patterns:
            matches = sum(1 for pattern in pattern_list if re.search(pattern, text_lower))
            if matches > 0:
                confidence = min(base_confidence + (matches - 1) * 0.05, 1.0)
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_intent = intent
                elif confidence > 0.7:
                    sub_intents.append(intent.value)
        
        return {
            "intent": best_intent,
            "confidence": max_confidence,
            "sub_intents": sub_intents
        }
    
    async def _classify_intent_ai(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """AI-based intent classification using Azure OpenAI."""
        try:
            client = await self._get_openai_client()
            
            # Build context information
            context_info = ""
            if context:
                recent_intents = context.get("recent_intents", [])
                if recent_intents:
                    context_info = f"Recent conversation intents: {', '.join(recent_intents[-3:])}"
            
            prompt = f"""
            Classify the user's intent from the following message in the context of Azure cloud governance.
            
            User message: "{text}"
            {context_info}
            
            Available intents:
            - cost_analysis: Questions about costs, billing, spending, budgets
            - policy_query: Questions about Azure policies, compliance, governance rules
            - resource_management: Managing Azure resources (VMs, storage, etc.)
            - security_analysis: Security-related questions, vulnerabilities, access
            - compliance_check: Compliance status, regulations, audit requirements
            - rbac_query: Role-based access control, permissions, user access
            - network_analysis: Networking, connectivity, VPC, subnets
            - optimization_suggestion: Performance optimization, best practices
            - general_query: General Azure questions not fitting other categories
            - greeting: Greetings, hello, good morning, etc.
            - help: Requests for help, assistance, how-to questions
            - unknown: Cannot determine intent
            
            Return a JSON object with:
            - intent: the primary intent
            - confidence: confidence score (0.0 to 1.0)
            - sub_intents: list of secondary intents (if any)
            - reasoning: brief explanation of the classification
            """
            
            response = await client.chat.completions.create(
                model=self.settings.ai.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": "You are an Azure governance expert assistant. Classify user intents accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                intent_str = result.get("intent", "general_query")
                
                # Map string to enum
                intent_map = {intent.value: intent for intent in ConversationIntent}
                intent = intent_map.get(intent_str, ConversationIntent.GENERAL_QUERY)
                
                return {
                    "intent": intent,
                    "confidence": result.get("confidence", 0.7),
                    "sub_intents": result.get("sub_intents", []),
                    "reasoning": result.get("reasoning", "")
                }
                
            except json.JSONDecodeError:
                logger.error("failed_to_parse_ai_intent_response", response=result_text)
                return {
                    "intent": ConversationIntent.GENERAL_QUERY,
                    "confidence": 0.5,
                    "sub_intents": [],
                    "reasoning": "Failed to parse AI response"
                }
                
        except Exception as e:
            logger.error("ai_intent_classification_failed", error=str(e))
            return {
                "intent": ConversationIntent.GENERAL_QUERY,
                "confidence": 0.5,
                "sub_intents": [],
                "reasoning": f"AI classification failed: {str(e)}"
            }
    
    async def extract_entities(
        self,
        text: str,
        intent: ConversationIntent
    ) -> List[Entity]:
        """Extract entities from text based on intent."""
        try:
            entities = []
            
            # Rule-based entity extraction
            rule_entities = self._extract_entities_rules(text, intent)
            entities.extend(rule_entities)
            
            # spaCy-based entity extraction
            if self.nlp:
                spacy_entities = self._extract_entities_spacy(text)
                entities.extend(spacy_entities)
            
            # AI-based entity extraction for complex cases
            if intent in [ConversationIntent.COST_ANALYSIS, ConversationIntent.POLICY_QUERY]:
                ai_entities = await self._extract_entities_ai(text, intent)
                entities.extend(ai_entities)
            
            # Remove duplicates and merge similar entities
            entities = self._deduplicate_entities(entities)
            
            return entities
            
        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            return []
    
    def _extract_entities_rules(self, text: str, intent: ConversationIntent) -> List[Entity]:
        """Rule-based entity extraction."""
        entities = []
        
        # Azure subscription ID pattern
        subscription_pattern = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
        for match in re.finditer(subscription_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.SUBSCRIPTION,
                value=match.group(),
                confidence=0.95,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Resource group pattern
        rg_pattern = r'\b(?:resource[- ]?group[- ]?|rg[- ]?)([a-zA-Z0-9][a-zA-Z0-9_-]*)\b'
        for match in re.finditer(rg_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.RESOURCE_GROUP,
                value=match.group(1),
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Cost/money patterns
        cost_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd|per\s+month|per\s+day)?'
        for match in re.finditer(cost_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.COST_THRESHOLD,
                value=match.group(1),
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Azure location pattern
        azure_locations = [
            "eastus", "westus", "centralus", "northcentralus", "southcentralus",
            "eastus2", "westus2", "westus3", "eastasia", "southeastasia",
            "northeurope", "westeurope", "japaneast", "japanwest", "australiaeast",
            "australiasoutheast", "brazilsouth", "canadacentral", "canadaeast",
            "centralindia", "southindia", "westindia", "francecentral", "francesouth"
        ]
        
        for location in azure_locations:
            if location in text.lower():
                start_pos = text.lower().find(location)
                entities.append(Entity(
                    type=EntityType.LOCATION,
                    value=location,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(location)
                ))
        
        # Date range patterns
        date_patterns = [
            (r'\b(last|past)\s+(\d+)\s+(days?|weeks?|months?|years?)\b', 'relative'),
            (r'\b(today|yesterday|this\s+(?:week|month|year))\b', 'relative'),
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'absolute'),
            (r'\b\d{4}-\d{2}-\d{2}\b', 'absolute')
        ]
        
        for pattern, date_type in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type=EntityType.DATE_RANGE,
                    value=match.group(),
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"date_type": date_type}
                ))
        
        return entities
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """spaCy-based entity extraction."""
        if not self.nlp or not self.matcher:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        # Use matcher for custom patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            entity_type_map = {
                "RESOURCE_GROUP": EntityType.RESOURCE_GROUP,
                "SUBSCRIPTION": EntityType.SUBSCRIPTION,
                "COST_THRESHOLD": EntityType.COST_THRESHOLD,
                "DATE_RANGE": EntityType.DATE_RANGE
            }
            
            if label in entity_type_map:
                entities.append(Entity(
                    type=entity_type_map[label],
                    value=span.text,
                    confidence=0.8,
                    start_pos=span.start_char,
                    end_pos=span.end_char
                ))
        
        # Named entity recognition
        for ent in doc.ents:
            if ent.label_ in ["MONEY", "CARDINAL", "DATE", "TIME", "PERCENT"]:
                entity_type = EntityType.METRIC
                if ent.label_ == "MONEY":
                    entity_type = EntityType.COST_THRESHOLD
                elif ent.label_ in ["DATE", "TIME"]:
                    entity_type = EntityType.DATE_RANGE
                
                entities.append(Entity(
                    type=entity_type,
                    value=ent.text,
                    confidence=0.7,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    metadata={"spacy_label": ent.label_}
                ))
        
        return entities
    
    async def _extract_entities_ai(
        self,
        text: str,
        intent: ConversationIntent
    ) -> List[Entity]:
        """AI-based entity extraction for complex cases."""
        try:
            client = await self._get_openai_client()
            
            prompt = f"""
            Extract Azure-related entities from the following text for a {intent.value} query.
            
            Text: "{text}"
            
            Extract the following types of entities if present:
            - resource_group: Azure resource group names
            - subscription: Azure subscription IDs or names
            - resource_type: Types of Azure resources (vm, storage, etc.)
            - location: Azure regions/locations
            - tag: Azure resource tags
            - policy: Azure policy names or references
            - role: Azure roles or RBAC roles
            - user: User names or identifiers
            - date_range: Date ranges or time periods
            - cost_threshold: Cost amounts or budget limits
            - metric: Performance metrics or measurements
            - service: Azure services
            
            Return a JSON array of entities with:
            - type: entity type from the list above
            - value: the actual entity value
            - confidence: confidence score (0.0 to 1.0)
            - start_pos: start position in text (if determinable)
            - end_pos: end position in text (if determinable)
            """
            
            response = await client.chat.completions.create(
                model=self.settings.ai.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": "You are an Azure expert. Extract entities accurately from user queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            try:
                ai_entities = json.loads(result_text)
                entities = []
                
                for entity_data in ai_entities:
                    entity_type_str = entity_data.get("type", "").upper()
                    
                    # Map string to enum
                    entity_type_map = {etype.value.upper(): etype for etype in EntityType}
                    entity_type = entity_type_map.get(entity_type_str, EntityType.METRIC)
                    
                    entities.append(Entity(
                        type=entity_type,
                        value=entity_data.get("value", ""),
                        confidence=entity_data.get("confidence", 0.7),
                        start_pos=entity_data.get("start_pos"),
                        end_pos=entity_data.get("end_pos"),
                        metadata={"source": "ai"}
                    ))
                
                return entities
                
            except json.JSONDecodeError:
                logger.error("failed_to_parse_ai_entities", response=result_text)
                return []
                
        except Exception as e:
            logger.error("ai_entity_extraction_failed", error=str(e))
            return []
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and merge similar ones."""
        if not entities:
            return []
        
        # Group by type and value
        entity_groups = {}
        for entity in entities:
            key = (entity.type, entity.value.lower())
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Keep the highest confidence entity from each group
        deduplicated = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda e: e.confidence)
            deduplicated.append(best_entity)
        
        return deduplicated