"""
Analytics Service for Conversation Monitoring.
Tracks conversation metrics, user behavior, and system performance.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import aioredis
import structlog
from collections import defaultdict, Counter

from ....shared.config import get_settings
from ..models import (
    ConversationAnalytics,
    ConversationMetrics,
    ConversationIntent
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class AnalyticsService:
    """Service for conversation analytics and monitoring."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.analytics_retention_days = 90
        self.metrics_update_interval = 300  # 5 minutes
    
    async def _get_redis_client(self) -> aioredis.Redis:
        """Get Redis client for analytics storage."""
        if self.redis_client is None:
            self.redis_client = aioredis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
        return self.redis_client
    
    async def track_conversation_start(
        self,
        session_id: str,
        user_id: str,
        intent: ConversationIntent
    ) -> None:
        """Track conversation start event."""
        try:
            redis_client = await self._get_redis_client()
            now = datetime.utcnow()
            
            # Track conversation start
            conversation_start = {
                "session_id": session_id,
                "user_id": user_id,
                "intent": intent.value,
                "timestamp": now.isoformat(),
                "event_type": "conversation_start"
            }
            
            # Store in daily analytics
            daily_key = f"analytics:daily:{now.strftime('%Y-%m-%d')}"
            await redis_client.lpush(daily_key, json.dumps(conversation_start))
            await redis_client.expire(daily_key, self.analytics_retention_days * 86400)
            
            # Update counters
            await self._update_counters("conversation_start", intent.value, now)
            
            logger.info(
                "conversation_start_tracked",
                session_id=session_id,
                user_id=user_id,
                intent=intent.value
            )
            
        except Exception as e:
            logger.error(
                "track_conversation_start_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def track_message_exchange(
        self,
        session_id: str,
        user_id: str,
        intent: ConversationIntent,
        user_message_length: int,
        response_generation_time: float,
        confidence_score: float
    ) -> None:
        """Track message exchange metrics."""
        try:
            redis_client = await self._get_redis_client()
            now = datetime.utcnow()
            
            # Track message exchange
            message_exchange = {
                "session_id": session_id,
                "user_id": user_id,
                "intent": intent.value,
                "user_message_length": user_message_length,
                "response_generation_time": response_generation_time,
                "confidence_score": confidence_score,
                "timestamp": now.isoformat(),
                "event_type": "message_exchange"
            }
            
            # Store in daily analytics
            daily_key = f"analytics:daily:{now.strftime('%Y-%m-%d')}"
            await redis_client.lpush(daily_key, json.dumps(message_exchange))
            await redis_client.expire(daily_key, self.analytics_retention_days * 86400)
            
            # Update performance metrics
            await self._update_performance_metrics(
                intent.value,
                response_generation_time,
                confidence_score,
                now
            )
            
            # Update counters
            await self._update_counters("message_exchange", intent.value, now)
            
            logger.info(
                "message_exchange_tracked",
                session_id=session_id,
                intent=intent.value,
                response_time=response_generation_time
            )
            
        except Exception as e:
            logger.error(
                "track_message_exchange_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def track_conversation_end(
        self,
        session_id: str,
        user_id: str,
        duration_minutes: float,
        message_count: int,
        satisfaction_score: Optional[float] = None
    ) -> None:
        """Track conversation end event."""
        try:
            redis_client = await self._get_redis_client()
            now = datetime.utcnow()
            
            # Track conversation end
            conversation_end = {
                "session_id": session_id,
                "user_id": user_id,
                "duration_minutes": duration_minutes,
                "message_count": message_count,
                "satisfaction_score": satisfaction_score,
                "timestamp": now.isoformat(),
                "event_type": "conversation_end"
            }
            
            # Store in daily analytics
            daily_key = f"analytics:daily:{now.strftime('%Y-%m-%d')}"
            await redis_client.lpush(daily_key, json.dumps(conversation_end))
            await redis_client.expire(daily_key, self.analytics_retention_days * 86400)
            
            # Update session metrics
            await self._update_session_metrics(
                duration_minutes,
                message_count,
                satisfaction_score,
                now
            )
            
            # Update counters
            await self._update_counters("conversation_end", "all", now)
            
            logger.info(
                "conversation_end_tracked",
                session_id=session_id,
                duration_minutes=duration_minutes,
                message_count=message_count
            )
            
        except Exception as e:
            logger.error(
                "track_conversation_end_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def track_error_event(
        self,
        session_id: str,
        user_id: str,
        error_type: str,
        error_message: str,
        intent: Optional[ConversationIntent] = None
    ) -> None:
        """Track error events."""
        try:
            redis_client = await self._get_redis_client()
            now = datetime.utcnow()
            
            # Track error event
            error_event = {
                "session_id": session_id,
                "user_id": user_id,
                "error_type": error_type,
                "error_message": error_message,
                "intent": intent.value if intent else "unknown",
                "timestamp": now.isoformat(),
                "event_type": "error"
            }
            
            # Store in daily analytics
            daily_key = f"analytics:daily:{now.strftime('%Y-%m-%d')}"
            await redis_client.lpush(daily_key, json.dumps(error_event))
            await redis_client.expire(daily_key, self.analytics_retention_days * 86400)
            
            # Update error counters
            await self._update_counters("error", error_type, now)
            
            logger.info(
                "error_event_tracked",
                session_id=session_id,
                error_type=error_type
            )
            
        except Exception as e:
            logger.error(
                "track_error_event_failed",
                error=str(e),
                session_id=session_id
            )
    
    async def _update_counters(
        self,
        event_type: str,
        category: str,
        timestamp: datetime
    ) -> None:
        """Update event counters."""
        try:
            redis_client = await self._get_redis_client()
            
            # Daily counters
            daily_key = f"counter:daily:{timestamp.strftime('%Y-%m-%d')}:{event_type}:{category}"
            await redis_client.incr(daily_key)
            await redis_client.expire(daily_key, self.analytics_retention_days * 86400)
            
            # Hourly counters
            hourly_key = f"counter:hourly:{timestamp.strftime('%Y-%m-%d-%H')}:{event_type}:{category}"
            await redis_client.incr(hourly_key)
            await redis_client.expire(hourly_key, 7 * 86400)  # Keep for 7 days
            
            # Monthly counters
            monthly_key = f"counter:monthly:{timestamp.strftime('%Y-%m')}:{event_type}:{category}"
            await redis_client.incr(monthly_key)
            await redis_client.expire(monthly_key, 365 * 86400)  # Keep for 1 year
            
        except Exception as e:
            logger.error("update_counters_failed", error=str(e))
    
    async def _update_performance_metrics(
        self,
        intent: str,
        response_time: float,
        confidence_score: float,
        timestamp: datetime
    ) -> None:
        """Update performance metrics."""
        try:
            redis_client = await self._get_redis_client()
            
            # Store performance data
            performance_data = {
                "intent": intent,
                "response_time": response_time,
                "confidence_score": confidence_score,
                "timestamp": timestamp.isoformat()
            }
            
            # Daily performance metrics
            daily_perf_key = f"performance:daily:{timestamp.strftime('%Y-%m-%d')}"
            await redis_client.lpush(daily_perf_key, json.dumps(performance_data))
            await redis_client.expire(daily_perf_key, self.analytics_retention_days * 86400)
            
            # Update running averages
            await self._update_running_averages(
                intent,
                response_time,
                confidence_score,
                timestamp
            )
            
        except Exception as e:
            logger.error("update_performance_metrics_failed", error=str(e))
    
    async def _update_running_averages(
        self,
        intent: str,
        response_time: float,
        confidence_score: float,
        timestamp: datetime
    ) -> None:
        """Update running averages for performance metrics."""
        try:
            redis_client = await self._get_redis_client()
            
            # Update response time average
            response_time_key = f"avg:response_time:{intent}"
            current_avg = await redis_client.get(response_time_key)
            current_count = await redis_client.get(f"count:response_time:{intent}")
            
            if current_avg and current_count:
                current_avg = float(current_avg)
                current_count = int(current_count)
                new_avg = (current_avg * current_count + response_time) / (current_count + 1)
                new_count = current_count + 1
            else:
                new_avg = response_time
                new_count = 1
            
            await redis_client.set(response_time_key, str(new_avg), ex=86400)
            await redis_client.set(f"count:response_time:{intent}", str(new_count), ex=86400)
            
            # Update confidence score average
            confidence_key = f"avg:confidence:{intent}"
            current_avg = await redis_client.get(confidence_key)
            current_count = await redis_client.get(f"count:confidence:{intent}")
            
            if current_avg and current_count:
                current_avg = float(current_avg)
                current_count = int(current_count)
                new_avg = (current_avg * current_count + confidence_score) / (current_count + 1)
                new_count = current_count + 1
            else:
                new_avg = confidence_score
                new_count = 1
            
            await redis_client.set(confidence_key, str(new_avg), ex=86400)
            await redis_client.set(f"count:confidence:{intent}", str(new_count), ex=86400)
            
        except Exception as e:
            logger.error("update_running_averages_failed", error=str(e))
    
    async def _update_session_metrics(
        self,
        duration_minutes: float,
        message_count: int,
        satisfaction_score: Optional[float],
        timestamp: datetime
    ) -> None:
        """Update session-level metrics."""
        try:
            redis_client = await self._get_redis_client()
            
            # Update duration averages
            duration_key = "avg:session_duration"
            current_avg = await redis_client.get(duration_key)
            current_count = await redis_client.get("count:session_duration")
            
            if current_avg and current_count:
                current_avg = float(current_avg)
                current_count = int(current_count)
                new_avg = (current_avg * current_count + duration_minutes) / (current_count + 1)
                new_count = current_count + 1
            else:
                new_avg = duration_minutes
                new_count = 1
            
            await redis_client.set(duration_key, str(new_avg), ex=86400)
            await redis_client.set("count:session_duration", str(new_count), ex=86400)
            
            # Update message count averages
            message_count_key = "avg:message_count"
            current_avg = await redis_client.get(message_count_key)
            current_count = await redis_client.get("count:message_count")
            
            if current_avg and current_count:
                current_avg = float(current_avg)
                current_count = int(current_count)
                new_avg = (current_avg * current_count + message_count) / (current_count + 1)
                new_count = current_count + 1
            else:
                new_avg = message_count
                new_count = 1
            
            await redis_client.set(message_count_key, str(new_avg), ex=86400)
            await redis_client.set("count:message_count", str(new_count), ex=86400)
            
            # Update satisfaction score if provided
            if satisfaction_score is not None:
                satisfaction_key = "avg:satisfaction_score"
                current_avg = await redis_client.get(satisfaction_key)
                current_count = await redis_client.get("count:satisfaction_score")
                
                if current_avg and current_count:
                    current_avg = float(current_avg)
                    current_count = int(current_count)
                    new_avg = (current_avg * current_count + satisfaction_score) / (current_count + 1)
                    new_count = current_count + 1
                else:
                    new_avg = satisfaction_score
                    new_count = 1
                
                await redis_client.set(satisfaction_key, str(new_avg), ex=86400)
                await redis_client.set("count:satisfaction_score", str(new_count), ex=86400)
            
        except Exception as e:
            logger.error("update_session_metrics_failed", error=str(e))
    
    async def get_conversation_analytics(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ConversationAnalytics:
        """Get conversation analytics for a time period."""
        try:
            redis_client = await self._get_redis_client()
            
            # Set default date range
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get analytics data
            analytics_data = await self._collect_analytics_data(
                redis_client, user_id, start_date, end_date
            )
            
            # Calculate metrics
            total_conversations = analytics_data["conversation_counts"]["total"]
            total_messages = analytics_data["message_counts"]["total"]
            
            average_conversation_length = (
                analytics_data["averages"]["message_count"]
                if analytics_data["averages"]["message_count"] > 0
                else 0
            )
            
            intent_distribution = analytics_data["intent_distribution"]
            
            # Get response times
            response_times = {
                "avg": analytics_data["averages"]["response_time"],
                "p95": analytics_data["percentiles"]["response_time_p95"],
                "p99": analytics_data["percentiles"]["response_time_p99"]
            }
            
            # Calculate error rate
            error_count = analytics_data["error_counts"]["total"]
            error_rate = (error_count / total_messages) if total_messages > 0 else 0
            
            # Get active sessions
            active_sessions = await self._get_active_sessions_count(redis_client)
            
            return ConversationAnalytics(
                total_conversations=total_conversations,
                total_messages=total_messages,
                average_conversation_length=average_conversation_length,
                intent_distribution=intent_distribution,
                common_entities=analytics_data["common_entities"],
                user_satisfaction=analytics_data["averages"]["satisfaction_score"],
                response_times=response_times,
                error_rate=error_rate,
                active_sessions=active_sessions,
                period_start=start_date,
                period_end=end_date
            )
            
        except Exception as e:
            logger.error(
                "get_conversation_analytics_failed",
                error=str(e),
                user_id=user_id
            )
            
            # Return empty analytics
            return ConversationAnalytics(
                total_conversations=0,
                total_messages=0,
                average_conversation_length=0,
                intent_distribution={},
                common_entities=[],
                user_satisfaction=None,
                response_times={"avg": 0, "p95": 0, "p99": 0},
                error_rate=0,
                active_sessions=0,
                period_start=start_date or datetime.utcnow(),
                period_end=end_date or datetime.utcnow()
            )
    
    async def _collect_analytics_data(
        self,
        redis_client: aioredis.Redis,
        user_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect analytics data from Redis."""
        analytics_data = {
            "conversation_counts": {"total": 0},
            "message_counts": {"total": 0},
            "intent_distribution": {},
            "error_counts": {"total": 0},
            "averages": {
                "response_time": 0,
                "confidence_score": 0,
                "message_count": 0,
                "satisfaction_score": None
            },
            "percentiles": {
                "response_time_p95": 0,
                "response_time_p99": 0
            },
            "common_entities": []
        }
        
        # Collect daily data
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Get conversation starts
            conversation_start_key = f"counter:daily:{date_str}:conversation_start:*"
            conversation_keys = await redis_client.keys(conversation_start_key)
            
            for key in conversation_keys:
                count = await redis_client.get(key)
                if count:
                    intent = key.split(":")[-1]
                    analytics_data["conversation_counts"]["total"] += int(count)
                    analytics_data["intent_distribution"][intent] = (
                        analytics_data["intent_distribution"].get(intent, 0) + int(count)
                    )
            
            # Get message exchanges
            message_exchange_key = f"counter:daily:{date_str}:message_exchange:*"
            message_keys = await redis_client.keys(message_exchange_key)
            
            for key in message_keys:
                count = await redis_client.get(key)
                if count:
                    analytics_data["message_counts"]["total"] += int(count)
            
            # Get errors
            error_key = f"counter:daily:{date_str}:error:*"
            error_keys = await redis_client.keys(error_key)
            
            for key in error_keys:
                count = await redis_client.get(key)
                if count:
                    analytics_data["error_counts"]["total"] += int(count)
            
            current_date += timedelta(days=1)
        
        # Get current averages
        avg_response_time = await redis_client.get("avg:response_time:all")
        if avg_response_time:
            analytics_data["averages"]["response_time"] = float(avg_response_time)
        
        avg_confidence = await redis_client.get("avg:confidence:all")
        if avg_confidence:
            analytics_data["averages"]["confidence_score"] = float(avg_confidence)
        
        avg_session_duration = await redis_client.get("avg:session_duration")
        if avg_session_duration:
            analytics_data["averages"]["session_duration"] = float(avg_session_duration)
        
        avg_message_count = await redis_client.get("avg:message_count")
        if avg_message_count:
            analytics_data["averages"]["message_count"] = float(avg_message_count)
        
        avg_satisfaction = await redis_client.get("avg:satisfaction_score")
        if avg_satisfaction:
            analytics_data["averages"]["satisfaction_score"] = float(avg_satisfaction)
        
        return analytics_data
    
    async def _get_active_sessions_count(self, redis_client: aioredis.Redis) -> int:
        """Get count of active conversation sessions."""
        try:
            # Count active session keys
            session_keys = await redis_client.keys("conversation_session:*")
            return len(session_keys)
        except Exception as e:
            logger.error("get_active_sessions_count_failed", error=str(e))
            return 0
    
    async def get_user_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        try:
            redis_client = await self._get_redis_client()
            
            # Set default date range
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get user-specific data
            user_analytics = {
                "user_id": user_id,
                "conversation_count": 0,
                "message_count": 0,
                "avg_session_duration": 0,
                "favorite_intents": [],
                "activity_pattern": {},
                "satisfaction_scores": []
            }
            
            # Collect user data from daily analytics
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                daily_key = f"analytics:daily:{date_str}"
                
                # Get all events for this day
                events = await redis_client.lrange(daily_key, 0, -1)
                
                for event_str in events:
                    try:
                        event = json.loads(event_str)
                        if event.get("user_id") == user_id:
                            event_type = event.get("event_type")
                            
                            if event_type == "conversation_start":
                                user_analytics["conversation_count"] += 1
                                intent = event.get("intent")
                                if intent:
                                    user_analytics["favorite_intents"].append(intent)
                            
                            elif event_type == "message_exchange":
                                user_analytics["message_count"] += 1
                            
                            elif event_type == "conversation_end":
                                duration = event.get("duration_minutes", 0)
                                user_analytics["avg_session_duration"] += duration
                                
                                satisfaction = event.get("satisfaction_score")
                                if satisfaction:
                                    user_analytics["satisfaction_scores"].append(satisfaction)
                            
                            # Track activity by hour
                            timestamp = datetime.fromisoformat(event.get("timestamp"))
                            hour = timestamp.hour
                            user_analytics["activity_pattern"][hour] = (
                                user_analytics["activity_pattern"].get(hour, 0) + 1
                            )
                    
                    except json.JSONDecodeError:
                        continue
                
                current_date += timedelta(days=1)
            
            # Calculate averages
            if user_analytics["conversation_count"] > 0:
                user_analytics["avg_session_duration"] /= user_analytics["conversation_count"]
            
            # Get most common intents
            intent_counter = Counter(user_analytics["favorite_intents"])
            user_analytics["favorite_intents"] = [
                intent for intent, count in intent_counter.most_common(5)
            ]
            
            # Calculate average satisfaction
            if user_analytics["satisfaction_scores"]:
                user_analytics["avg_satisfaction"] = sum(user_analytics["satisfaction_scores"]) / len(user_analytics["satisfaction_scores"])
            else:
                user_analytics["avg_satisfaction"] = None
            
            return user_analytics
            
        except Exception as e:
            logger.error(
                "get_user_analytics_failed",
                error=str(e),
                user_id=user_id
            )
            return {}
    
    async def cleanup_old_analytics(self) -> None:
        """Clean up old analytics data (background task)."""
        try:
            redis_client = await self._get_redis_client()
            
            cutoff_date = datetime.utcnow() - timedelta(days=self.analytics_retention_days)
            
            # Clean up daily analytics
            date_str = cutoff_date.strftime('%Y-%m-%d')
            old_keys = await redis_client.keys(f"analytics:daily:{date_str}")
            
            if old_keys:
                await redis_client.delete(*old_keys)
                logger.info(f"cleaned_up_old_analytics", keys_deleted=len(old_keys))
            
        except Exception as e:
            logger.error("cleanup_old_analytics_failed", error=str(e))