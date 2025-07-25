"""
Context Manager Service.
Manages conversation context, history, and state for better response generation.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import redis.asyncio as redis
import structlog

from ....shared.config import get_settings
from ..models import (
    ConversationIntent,
    Entity,
    ConversationMessage,
    ResponseGenerationContext
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class ContextManager:
    """Manages conversation context and history for enhanced responses."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.max_context_age = timedelta(hours=2)  # Maximum age for context relevance
        self.max_context_items = 10  # Maximum number of context items to keep
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for context storage."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
        return self.redis_client
    
    async def update_context(
        self,
        session_id: str,
        message: str,
        intent: ConversationIntent,
        entities: List[Entity]
    ) -> None:
        """Update conversation context with new information."""
        try:
            redis_client = await self._get_redis_client()
            context_key = f"conversation_context:{session_id}"
            
            # Get current context
            current_context = await self._get_context(session_id)
            
            # Update with new information
            now = datetime.utcnow()
            
            # Add current message to context
            current_context["messages"].append({
                "message": message,
                "intent": intent.value,
                "entities": [entity.dict() for entity in entities],
                "timestamp": now.isoformat()
            })
            
            # Keep only recent messages
            current_context["messages"] = current_context["messages"][-self.max_context_items:]
            
            # Update intent tracking
            if "intent_history" not in current_context:
                current_context["intent_history"] = []
            
            current_context["intent_history"].append({
                "intent": intent.value,
                "timestamp": now.isoformat()
            })
            current_context["intent_history"] = current_context["intent_history"][-5:]  # Keep last 5 intents
            
            # Update entity tracking
            await self._update_entity_context(current_context, entities)
            
            # Update conversation state
            await self._update_conversation_state(current_context, intent, entities)
            
            # Save updated context
            await redis_client.set(
                context_key,
                json.dumps(current_context),
                ex=int(self.max_context_age.total_seconds())
            )
            
            logger.info(
                "conversation_context_updated",
                session_id=session_id,
                intent=intent.value,
                entity_count=len(entities)
            )
            
        except Exception as e:
            logger.error(
                "update_context_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def _get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current conversation context."""
        try:
            redis_client = await self._get_redis_client()
            context_key = f"conversation_context:{session_id}"
            
            context_data = await redis_client.get(context_key)
            if context_data:
                return json.loads(context_data)
            
            # Initialize new context
            return {
                "session_id": session_id,
                "messages": [],
                "intent_history": [],
                "entities": {},
                "azure_context": {},
                "user_preferences": {},
                "conversation_state": {
                    "current_topic": None,
                    "follow_up_expected": False,
                    "pending_actions": [],
                    "last_response_type": None
                },
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("get_context_failed", error=str(e))
            return {}
    
    async def _update_entity_context(
        self,
        context: Dict[str, Any],
        entities: List[Entity]
    ) -> None:
        """Update entity context with new entities."""
        if "entities" not in context:
            context["entities"] = {}
        
        for entity in entities:
            entity_type = entity.type.value
            entity_value = entity.value
            
            if entity_type not in context["entities"]:
                context["entities"][entity_type] = {}
            
            # Update entity with frequency and recency
            if entity_value in context["entities"][entity_type]:
                context["entities"][entity_type][entity_value]["frequency"] += 1
                context["entities"][entity_type][entity_value]["last_seen"] = datetime.utcnow().isoformat()
            else:
                context["entities"][entity_type][entity_value] = {
                    "frequency": 1,
                    "first_seen": datetime.utcnow().isoformat(),
                    "last_seen": datetime.utcnow().isoformat(),
                    "confidence": entity.confidence,
                    "metadata": entity.metadata or {}
                }
    
    async def _update_conversation_state(
        self,
        context: Dict[str, Any],
        intent: ConversationIntent,
        entities: List[Entity]
    ) -> None:
        """Update conversation state based on intent and entities."""
        state = context.get("conversation_state", {})
        
        # Update current topic
        topic_mapping = {
            ConversationIntent.COST_ANALYSIS: "cost_management",
            ConversationIntent.POLICY_QUERY: "governance_policy",
            ConversationIntent.RESOURCE_MANAGEMENT: "resource_management",
            ConversationIntent.SECURITY_ANALYSIS: "security",
            ConversationIntent.RBAC_QUERY: "access_control",
            ConversationIntent.NETWORK_ANALYSIS: "networking",
            ConversationIntent.OPTIMIZATION_SUGGESTION: "optimization"
        }
        
        if intent in topic_mapping:
            state["current_topic"] = topic_mapping[intent]
        
        # Determine if follow-up is expected
        follow_up_intents = [
            ConversationIntent.COST_ANALYSIS,
            ConversationIntent.RESOURCE_MANAGEMENT,
            ConversationIntent.SECURITY_ANALYSIS
        ]
        
        state["follow_up_expected"] = intent in follow_up_intents
        
        # Update Azure context based on entities
        await self._update_azure_context(context, entities)
        
        context["conversation_state"] = state
    
    async def _update_azure_context(
        self,
        context: Dict[str, Any],
        entities: List[Entity]
    ) -> None:
        """Update Azure-specific context."""
        if "azure_context" not in context:
            context["azure_context"] = {}
        
        azure_context = context["azure_context"]
        
        # Update subscription context
        subscription_entities = [e for e in entities if e.type.value == "subscription"]
        if subscription_entities:
            azure_context["current_subscription"] = subscription_entities[0].value
        
        # Update resource group context
        rg_entities = [e for e in entities if e.type.value == "resource_group"]
        if rg_entities:
            azure_context["current_resource_group"] = rg_entities[0].value
        
        # Update location context
        location_entities = [e for e in entities if e.type.value == "location"]
        if location_entities:
            azure_context["current_location"] = location_entities[0].value
        
        # Update service context
        service_entities = [e for e in entities if e.type.value == "service"]
        if service_entities:
            if "active_services" not in azure_context:
                azure_context["active_services"] = []
            
            for service in service_entities:
                if service.value not in azure_context["active_services"]:
                    azure_context["active_services"].append(service.value)
    
    async def get_context_for_response(
        self,
        session_id: str,
        include_history: bool = True
    ) -> ResponseGenerationContext:
        """Get context formatted for response generation."""
        try:
            context = await self._get_context(session_id)
            
            # Get conversation history
            conversation_history = []
            if include_history and "messages" in context:
                for msg_data in context["messages"][-5:]:  # Last 5 messages
                    # Convert to ConversationMessage format
                    conversation_history.append({
                        "content": msg_data["message"],
                        "intent": msg_data["intent"],
                        "entities": msg_data["entities"],
                        "timestamp": msg_data["timestamp"]
                    })
            
            # Extract user preferences
            user_preferences = context.get("user_preferences", {})
            
            # Extract Azure context
            azure_context = context.get("azure_context", {})
            
            # Build intent context
            intent_context = {
                "current_topic": context.get("conversation_state", {}).get("current_topic"),
                "follow_up_expected": context.get("conversation_state", {}).get("follow_up_expected", False),
                "recent_intents": [
                    item["intent"] for item in context.get("intent_history", [])[-3:]
                ],
                "frequent_entities": await self._get_frequent_entities(context)
            }
            
            return ResponseGenerationContext(
                session_id=session_id,
                conversation_history=conversation_history,
                user_preferences=user_preferences,
                azure_context=azure_context,
                intent_context=intent_context
            )
            
        except Exception as e:
            logger.error(
                "get_context_for_response_failed",
                error=str(e),
                session_id=session_id
            )
            # Return minimal context
            return ResponseGenerationContext(
                session_id=session_id,
                conversation_history=[],
                user_preferences={},
                azure_context={},
                intent_context={}
            )
    
    async def _get_frequent_entities(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get frequently mentioned entities."""
        frequent_entities = {}
        
        entities = context.get("entities", {})
        for entity_type, entity_values in entities.items():
            # Sort by frequency and recency
            sorted_entities = sorted(
                entity_values.items(),
                key=lambda x: (x[1]["frequency"], x[1]["last_seen"]),
                reverse=True
            )
            
            # Take top 3 for each type
            frequent_entities[entity_type] = [
                entity for entity, _ in sorted_entities[:3]
            ]
        
        return frequent_entities
    
    async def add_user_preference(
        self,
        session_id: str,
        preference_key: str,
        preference_value: Any
    ) -> None:
        """Add or update user preference."""
        try:
            context = await self._get_context(session_id)
            
            if "user_preferences" not in context:
                context["user_preferences"] = {}
            
            context["user_preferences"][preference_key] = preference_value
            
            # Save updated context
            redis_client = await self._get_redis_client()
            context_key = f"conversation_context:{session_id}"
            
            await redis_client.set(
                context_key,
                json.dumps(context),
                ex=int(self.max_context_age.total_seconds())
            )
            
            logger.info(
                "user_preference_added",
                session_id=session_id,
                preference_key=preference_key
            )
            
        except Exception as e:
            logger.error(
                "add_user_preference_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation context."""
        try:
            context = await self._get_context(session_id)
            
            # Count messages by intent
            intent_counts = {}
            for msg in context.get("messages", []):
                intent = msg.get("intent", "unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # Get most frequent entities
            frequent_entities = await self._get_frequent_entities(context)
            
            # Get conversation duration
            messages = context.get("messages", [])
            if messages:
                start_time = datetime.fromisoformat(messages[0]["timestamp"])
                end_time = datetime.fromisoformat(messages[-1]["timestamp"])
                duration_minutes = (end_time - start_time).total_seconds() / 60
            else:
                duration_minutes = 0
            
            return {
                "session_id": session_id,
                "message_count": len(messages),
                "duration_minutes": duration_minutes,
                "intent_distribution": intent_counts,
                "frequent_entities": frequent_entities,
                "current_topic": context.get("conversation_state", {}).get("current_topic"),
                "azure_context": context.get("azure_context", {}),
                "created_at": context.get("created_at")
            }
            
        except Exception as e:
            logger.error(
                "get_conversation_summary_failed",
                error=str(e),
                session_id=session_id
            )
            return {}
    
    async def clear_context(self, session_id: str) -> None:
        """Clear conversation context."""
        try:
            redis_client = await self._get_redis_client()
            context_key = f"conversation_context:{session_id}"
            
            await redis_client.delete(context_key)
            
            logger.info("conversation_context_cleared", session_id=session_id)
            
        except Exception as e:
            logger.error(
                "clear_context_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def get_context_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights about conversation patterns."""
        try:
            context = await self._get_context(session_id)
            
            # Analyze intent transitions
            intent_transitions = {}
            intent_history = context.get("intent_history", [])
            
            for i in range(1, len(intent_history)):
                prev_intent = intent_history[i-1]["intent"]
                curr_intent = intent_history[i]["intent"]
                transition = f"{prev_intent} -> {curr_intent}"
                
                intent_transitions[transition] = intent_transitions.get(transition, 0) + 1
            
            # Analyze entity patterns
            entity_patterns = {}
            for entity_type, entities in context.get("entities", {}).items():
                entity_patterns[entity_type] = {
                    "unique_count": len(entities),
                    "total_mentions": sum(e["frequency"] for e in entities.values()),
                    "most_frequent": max(entities.items(), key=lambda x: x[1]["frequency"])[0] if entities else None
                }
            
            # User behavior insights
            messages = context.get("messages", [])
            if messages:
                avg_message_length = sum(len(msg["message"]) for msg in messages) / len(messages)
                question_count = sum(1 for msg in messages if "?" in msg["message"])
                question_ratio = question_count / len(messages) if messages else 0
            else:
                avg_message_length = 0
                question_ratio = 0
            
            return {
                "session_id": session_id,
                "intent_transitions": intent_transitions,
                "entity_patterns": entity_patterns,
                "user_behavior": {
                    "avg_message_length": avg_message_length,
                    "question_ratio": question_ratio,
                    "message_count": len(messages)
                },
                "conversation_flow": {
                    "topic_switches": len(set(h["intent"] for h in intent_history)),
                    "follow_up_pattern": context.get("conversation_state", {}).get("follow_up_expected", False)
                }
            }
            
        except Exception as e:
            logger.error(
                "get_context_insights_failed",
                error=str(e),
                session_id=session_id
            )
            return {}