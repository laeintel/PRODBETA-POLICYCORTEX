"""
Conversation Manager Service.
Handles conversation session management, message storage, and session state.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
import structlog
from sqlalchemy import select, insert, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from ....shared.config import get_settings
from ....shared.database import get_async_db
from ..models import (
    ConversationSession,
    ConversationMessage,
    ConversationHistory,
    ConversationStatus,
    MessageType,
    ConversationIntent
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class ConversationManager:
    """Manages conversation sessions and message storage."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.session_timeout = timedelta(hours=4)  # Default session timeout
        self.max_context_length = 20  # Maximum messages in context
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for session caching."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
        return self.redis_client
    
    async def create_or_get_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> ConversationSession:
        """Create a new conversation session or get existing one."""
        try:
            if session_id:
                # Try to get existing session
                session = await self._get_session_from_cache(session_id)
                if session and session.user_id == user_id:
                    return session
            
            # Create new session
            session_id = session_id or f"conv_{uuid.uuid4().hex[:16]}"
            now = datetime.utcnow()
            
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                title=title or f"Conversation {now.strftime('%Y-%m-%d %H:%M')}",
                status=ConversationStatus.ACTIVE,
                context={
                    "user_preferences": {},
                    "conversation_state": {},
                    "last_intent": None,
                    "entities": [],
                    "azure_context": {}
                },
                created_at=now,
                updated_at=now,
                expires_at=now + self.session_timeout,
                message_count=0,
                last_message_preview=None
            )
            
            # Save to cache
            await self._save_session_to_cache(session)
            
            # Save to database
            await self._save_session_to_db(session)
            
            logger.info(
                "conversation_session_created",
                session_id=session_id,
                user_id=user_id
            )
            
            return session
            
        except Exception as e:
            logger.error(
                "create_conversation_session_failed",
                error=str(e),
                user_id=user_id,
                session_id=session_id
            )
            raise Exception(f"Failed to create conversation session: {str(e)}")
    
    async def _get_session_from_cache(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session from cache."""
        try:
            redis_client = await self._get_redis_client()
            session_key = f"conversation_session:{session_id}"
            
            session_data = await redis_client.get(session_key)
            if session_data:
                session_dict = json.loads(session_data)
                return ConversationSession(**session_dict)
            
            return None
            
        except Exception as e:
            logger.error("get_session_from_cache_failed", error=str(e))
            return None
    
    async def _save_session_to_cache(self, session: ConversationSession) -> None:
        """Save conversation session to cache."""
        try:
            redis_client = await self._get_redis_client()
            session_key = f"conversation_session:{session.session_id}"
            
            # Convert datetime objects to ISO format
            session_dict = session.dict()
            session_dict["created_at"] = session.created_at.isoformat()
            session_dict["updated_at"] = session.updated_at.isoformat()
            if session.expires_at:
                session_dict["expires_at"] = session.expires_at.isoformat()
            
            await redis_client.set(
                session_key,
                json.dumps(session_dict),
                ex=int(self.session_timeout.total_seconds())
            )
            
        except Exception as e:
            logger.error("save_session_to_cache_failed", error=str(e))
    
    async def _save_session_to_db(self, session: ConversationSession) -> None:
        """Save conversation session to database."""
        try:
            async with get_async_db() as db:
                # This would use actual database tables
                # For now, we'll just log the operation
                logger.info(
                    "session_saved_to_db",
                    session_id=session.session_id,
                    user_id=session.user_id
                )
        except Exception as e:
            logger.error("save_session_to_db_failed", error=str(e))
    
    async def add_message(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: ConversationIntent,
        entities: List[Dict[str, Any]]
    ) -> None:
        """Add a message exchange to the conversation."""
        try:
            # Get session
            session = await self._get_session_from_cache(session_id)
            if not session:
                raise Exception(f"Session {session_id} not found")
            
            now = datetime.utcnow()
            
            # Create user message
            user_msg = ConversationMessage(
                id=f"msg_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                type=MessageType.USER,
                content=user_message,
                intent=intent,
                entities=entities,
                timestamp=now,
                user_id=session.user_id
            )
            
            # Create assistant message
            assistant_msg = ConversationMessage(
                id=f"msg_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                type=MessageType.ASSISTANT,
                content=assistant_message,
                intent=intent,
                entities=entities,
                timestamp=now,
                user_id=session.user_id
            )
            
            # Save messages
            await self._save_message_to_cache(user_msg)
            await self._save_message_to_cache(assistant_msg)
            await self._save_message_to_db(user_msg)
            await self._save_message_to_db(assistant_msg)
            
            # Update session
            session.message_count += 2
            session.updated_at = now
            session.last_message_preview = user_message[:100] + "..." if len(user_message) > 100 else user_message
            session.context["last_intent"] = intent.value
            session.context["entities"] = entities
            
            await self._save_session_to_cache(session)
            await self._save_session_to_db(session)
            
            logger.info(
                "message_added_to_conversation",
                session_id=session_id,
                user_id=session.user_id,
                intent=intent.value
            )
            
        except Exception as e:
            logger.error(
                "add_message_failed",
                error=str(e),
                session_id=session_id
            )
            raise Exception(f"Failed to add message: {str(e)}")
    
    async def _save_message_to_cache(self, message: ConversationMessage) -> None:
        """Save message to cache."""
        try:
            redis_client = await self._get_redis_client()
            message_key = f"conversation_message:{message.id}"
            
            # Convert to dict and handle datetime
            message_dict = message.dict()
            message_dict["timestamp"] = message.timestamp.isoformat()
            
            await redis_client.set(
                message_key,
                json.dumps(message_dict),
                ex=int(self.session_timeout.total_seconds())
            )
            
            # Add to session message list
            session_messages_key = f"conversation_messages:{message.session_id}"
            await redis_client.lpush(session_messages_key, message.id)
            await redis_client.expire(
                session_messages_key,
                int(self.session_timeout.total_seconds())
            )
            
        except Exception as e:
            logger.error("save_message_to_cache_failed", error=str(e))
    
    async def _save_message_to_db(self, message: ConversationMessage) -> None:
        """Save message to database."""
        try:
            async with get_async_db() as db:
                # This would use actual database tables
                logger.info(
                    "message_saved_to_db",
                    message_id=message.id,
                    session_id=message.session_id
                )
        except Exception as e:
            logger.error("save_message_to_db_failed", error=str(e))
    
    async def get_conversation_history(
        self,
        session_id: str,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> ConversationHistory:
        """Get conversation history for a session."""
        try:
            # Verify session access
            session = await self._get_session_from_cache(session_id)
            if not session or session.user_id != user_id:
                raise Exception("Session not found or access denied")
            
            # Get messages from cache
            messages = await self._get_messages_from_cache(session_id, limit, offset)
            
            # If not enough messages in cache, try database
            if len(messages) < limit and offset == 0:
                db_messages = await self._get_messages_from_db(session_id, limit, offset)
                messages.extend(db_messages)
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.timestamp)
            
            # Apply limit
            messages = messages[offset:offset + limit]
            
            # Get total count
            total_count = await self._get_total_message_count(session_id)
            has_more = (offset + len(messages)) < total_count
            
            return ConversationHistory(
                messages=messages,
                total_count=total_count,
                has_more=has_more
            )
            
        except Exception as e:
            logger.error(
                "get_conversation_history_failed",
                error=str(e),
                session_id=session_id,
                user_id=user_id
            )
            raise Exception(f"Failed to get conversation history: {str(e)}")
    
    async def _get_messages_from_cache(
        self,
        session_id: str,
        limit: int,
        offset: int
    ) -> List[ConversationMessage]:
        """Get messages from cache."""
        try:
            redis_client = await self._get_redis_client()
            session_messages_key = f"conversation_messages:{session_id}"
            
            # Get message IDs
            message_ids = await redis_client.lrange(
                session_messages_key,
                offset,
                offset + limit - 1
            )
            
            messages = []
            for message_id in message_ids:
                message_key = f"conversation_message:{message_id}"
                message_data = await redis_client.get(message_key)
                if message_data:
                    message_dict = json.loads(message_data)
                    message_dict["timestamp"] = datetime.fromisoformat(message_dict["timestamp"])
                    messages.append(ConversationMessage(**message_dict))
            
            return messages
            
        except Exception as e:
            logger.error("get_messages_from_cache_failed", error=str(e))
            return []
    
    async def _get_messages_from_db(
        self,
        session_id: str,
        limit: int,
        offset: int
    ) -> List[ConversationMessage]:
        """Get messages from database."""
        try:
            async with get_async_db() as db:
                # This would query actual database tables
                logger.info(
                    "messages_fetched_from_db",
                    session_id=session_id,
                    limit=limit,
                    offset=offset
                )
                return []
        except Exception as e:
            logger.error("get_messages_from_db_failed", error=str(e))
            return []
    
    async def _get_total_message_count(self, session_id: str) -> int:
        """Get total message count for a session."""
        try:
            redis_client = await self._get_redis_client()
            session_messages_key = f"conversation_messages:{session_id}"
            
            count = await redis_client.llen(session_messages_key)
            return count or 0
            
        except Exception as e:
            logger.error("get_total_message_count_failed", error=str(e))
            return 0
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[ConversationSession]:
        """Get all conversation sessions for a user."""
        try:
            # In a real implementation, this would query the database
            # For now, we'll return sessions from cache
            redis_client = await self._get_redis_client()
            
            # Get user session keys
            user_sessions_key = f"user_sessions:{user_id}"
            session_ids = await redis_client.lrange(user_sessions_key, offset, offset + limit - 1)
            
            sessions = []
            for session_id in session_ids:
                session = await self._get_session_from_cache(session_id)
                if session:
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(
                "get_user_sessions_failed",
                error=str(e),
                user_id=user_id
            )
            return []
    
    async def delete_session(self, session_id: str, user_id: str) -> None:
        """Delete a conversation session."""
        try:
            # Verify session access
            session = await self._get_session_from_cache(session_id)
            if not session or session.user_id != user_id:
                raise Exception("Session not found or access denied")
            
            redis_client = await self._get_redis_client()
            
            # Delete session from cache
            session_key = f"conversation_session:{session_id}"
            await redis_client.delete(session_key)
            
            # Delete messages from cache
            session_messages_key = f"conversation_messages:{session_id}"
            message_ids = await redis_client.lrange(session_messages_key, 0, -1)
            
            for message_id in message_ids:
                message_key = f"conversation_message:{message_id}"
                await redis_client.delete(message_key)
            
            await redis_client.delete(session_messages_key)
            
            # Remove from user sessions
            user_sessions_key = f"user_sessions:{user_id}"
            await redis_client.lrem(user_sessions_key, 0, session_id)
            
            # Delete from database
            await self._delete_session_from_db(session_id)
            
            logger.info(
                "conversation_session_deleted",
                session_id=session_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(
                "delete_session_failed",
                error=str(e),
                session_id=session_id,
                user_id=user_id
            )
            raise Exception(f"Failed to delete session: {str(e)}")
    
    async def _delete_session_from_db(self, session_id: str) -> None:
        """Delete session from database."""
        try:
            async with get_async_db() as db:
                # This would delete from actual database tables
                logger.info("session_deleted_from_db", session_id=session_id)
        except Exception as e:
            logger.error("delete_session_from_db_failed", error=str(e))
    
    async def update_session_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> None:
        """Update session context."""
        try:
            session = await self._get_session_from_cache(session_id)
            if not session:
                raise Exception(f"Session {session_id} not found")
            
            # Update context
            session.context.update(context_updates)
            session.updated_at = datetime.utcnow()
            
            # Save updated session
            await self._save_session_to_cache(session)
            await self._save_session_to_db(session)
            
            logger.info(
                "session_context_updated",
                session_id=session_id,
                updates=list(context_updates.keys())
            )
            
        except Exception as e:
            logger.error(
                "update_session_context_failed",
                error=str(e),
                session_id=session_id
            )
            raise Exception(f"Failed to update session context: {str(e)}")
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = None
    ) -> Dict[str, Any]:
        """Get conversation context for response generation."""
        try:
            session = await self._get_session_from_cache(session_id)
            if not session:
                raise Exception(f"Session {session_id} not found")
            
            # Get recent messages for context
            max_messages = max_messages or self.max_context_length
            messages = await self._get_messages_from_cache(session_id, max_messages, 0)
            
            # Build context
            context = {
                "session_id": session_id,
                "user_id": session.user_id,
                "session_context": session.context,
                "recent_messages": [
                    {
                        "type": msg.type,
                        "content": msg.content,
                        "intent": msg.intent,
                        "entities": msg.entities,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in messages[-max_messages:]
                ],
                "last_intent": session.context.get("last_intent"),
                "entities": session.context.get("entities", []),
                "azure_context": session.context.get("azure_context", {})
            }
            
            return context
            
        except Exception as e:
            logger.error(
                "get_conversation_context_failed",
                error=str(e),
                session_id=session_id
            )
            return {}
    
    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions (background task)."""
        try:
            redis_client = await self._get_redis_client()
            
            # This would be implemented as a background task
            # to periodically clean up expired sessions
            
            logger.info("expired_sessions_cleanup_completed")
            
        except Exception as e:
            logger.error("cleanup_expired_sessions_failed", error=str(e))