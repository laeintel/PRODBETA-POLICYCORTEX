"""
Pydantic models for Conversation service.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field


class ConversationIntent(str, Enum):
    """Conversation intent enumeration."""

    COST_ANALYSIS = "cost_analysis"
    POLICY_QUERY = "policy_query"
    RESOURCE_MANAGEMENT = "resource_management"
    SECURITY_ANALYSIS = "security_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    RBAC_QUERY = "rbac_query"
    NETWORK_ANALYSIS = "network_analysis"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    GENERAL_QUERY = "general_query"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Entity type enumeration."""

    RESOURCE_GROUP = "resource_group"
    SUBSCRIPTION = "subscription"
    RESOURCE_TYPE = "resource_type"
    LOCATION = "location"
    TAG = "tag"
    POLICY = "policy"
    ROLE = "role"
    USER = "user"
    DATE_RANGE = "date_range"
    COST_THRESHOLD = "cost_threshold"
    METRIC = "metric"
    SERVICE = "service"


class ConversationStatus(str, Enum):
    """Conversation status enumeration."""

    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class MessageType(str, Enum):
    """Message type enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class WebSocketMessageType(str, Enum):
    """WebSocket message type enumeration."""

    CONVERSATION = "conversation"
    TYPING = "typing"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"


class Entity(BaseModel):
    """Entity extraction model."""

    type: EntityType = Field(..., description="Entity type")
    value: str = Field(..., description="Entity value")
    confidence: float = Field(..., description="Confidence score")
    start_pos: Optional[int] = Field(None, description="Start position in text")
    end_pos: Optional[int] = Field(None, description="End position in text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ConversationMessage(BaseModel):
    """Conversation message model."""

    id: str = Field(..., description="Message ID")
    session_id: str = Field(..., description="Session ID")
    type: MessageType = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    intent: Optional[ConversationIntent] = Field(None, description="Detected intent")
    entities: Optional[List[Entity]] = Field(None, description="Extracted entities")
    confidence: Optional[float] = Field(None, description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(..., description="Message timestamp")
    user_id: str = Field(..., description="User ID")


class ConversationRequest(BaseModel):
    """Conversation request model."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")
    intent: Optional[ConversationIntent] = Field(None, description="Hint for intent")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")


class ConversationResponse(BaseModel):
    """Conversation response model."""

    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Assistant response")
    intent: ConversationIntent = Field(..., description="Detected intent")
    entities: List[Entity] = Field(..., description="Extracted entities")
    confidence: float = Field(..., description="Confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    follow_up_questions: Optional[List[str]] = Field(None, description="Follow-up questions")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    timestamp: datetime = Field(..., description="Response timestamp")


class ConversationSession(BaseModel):
    """Conversation session model."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Session title")
    status: ConversationStatus = Field(..., description="Session status")
    context: Dict[str, Any] = Field(..., description="Session context")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")
    message_count: int = Field(0, description="Number of messages")
    last_message_preview: Optional[str] = Field(None, description="Preview of last message")


class ConversationSessionResponse(BaseModel):
    """Conversation session response model."""

    session_id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    message_count: int = Field(..., description="Number of messages")
    last_message_preview: Optional[str] = Field(None, description="Preview of last message")


class ConversationHistory(BaseModel):
    """Conversation history model."""

    messages: List[ConversationMessage] = Field(..., description="Conversation messages")
    total_count: int = Field(..., description="Total message count")
    has_more: bool = Field(..., description="Whether there are more messages")


class ConversationHistoryResponse(BaseModel):
    """Conversation history response model."""

    session_id: str = Field(..., description="Session ID")
    messages: List[ConversationMessage] = Field(..., description="Conversation messages")
    total_count: int = Field(..., description="Total message count")
    has_more: bool = Field(..., description="Whether there are more messages")


class IntentClassificationResult(BaseModel):
    """Intent classification result model."""

    intent: ConversationIntent = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    entities: List[Entity] = Field(..., description="Extracted entities")
    sub_intents: Optional[List[str]] = Field(None, description="Sub-intents")


class IntentClassificationResponse(BaseModel):
    """Intent classification response model."""

    intent: ConversationIntent = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    entities: List[Entity] = Field(..., description="Extracted entities")
    sub_intents: Optional[List[str]] = Field(None, description="Sub-intents")


class EntityExtractionResponse(BaseModel):
    """Entity extraction response model."""

    entities: List[Entity] = Field(..., description="Extracted entities")
    message: str = Field(..., description="Original message")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: WebSocketMessageType = Field(..., description="Message type")
    content: Union[str, Dict[str, Any]] = Field(..., description="Message content")
    session_id: Optional[str] = Field(None, description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ResponseGenerationContext(BaseModel):
    """Response generation context model."""

    session_id: str = Field(..., description="Session ID")
    conversation_history: List[ConversationMessage] = Field(..., description="Conversation history")
    user_preferences: Dict[str, Any] = Field(..., description="User preferences")
    azure_context: Dict[str, Any] = Field(..., description="Azure context information")
    intent_context: Dict[str, Any] = Field(..., description="Intent-specific context")


class ResponseGenerationResult(BaseModel):
    """Response generation result model."""

    message: str = Field(..., description="Generated response message")
    suggestions: List[str] = Field(..., description="Follow-up suggestions")
    follow_up_questions: List[str] = Field(..., description="Follow-up questions")
    confidence: float = Field(..., description="Response confidence")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class ConversationAnalytics(BaseModel):
    """Conversation analytics model."""

    total_conversations: int = Field(..., description="Total number of conversations")
    total_messages: int = Field(..., description="Total number of messages")
    average_conversation_length: float = Field(..., description="Average conversation length")
    intent_distribution: Dict[ConversationIntent, int] = Field(
        ..., description="Intent distribution"
    )
    common_entities: List[Dict[str, Any]] = Field(..., description="Common entities")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction score")
    response_times: Dict[str, float] = Field(..., description="Response time statistics")
    error_rate: float = Field(..., description="Error rate")
    active_sessions: int = Field(..., description="Number of active sessions")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")


class QueryRouterResult(BaseModel):
    """Query router result model."""

    service: str = Field(..., description="Target service")
    endpoint: str = Field(..., description="Service endpoint")
    parameters: Dict[str, Any] = Field(..., description="Service parameters")
    confidence: float = Field(..., description="Routing confidence")
    data: Optional[Dict[str, Any]] = Field(None, description="Service response data")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ConversationContextUpdate(BaseModel):
    """Conversation context update model."""

    session_id: str = Field(..., description="Session ID")
    updates: Dict[str, Any] = Field(..., description="Context updates")
    merge_strategy: str = Field("merge", description="Update strategy: merge, replace, append")


class ConversationMetrics(BaseModel):
    """Conversation metrics model."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    message_count: int = Field(..., description="Number of messages")
    duration_minutes: float = Field(..., description="Conversation duration in minutes")
    intent_switches: int = Field(..., description="Number of intent switches")
    entities_extracted: int = Field(..., description="Number of entities extracted")
    average_response_time: float = Field(..., description="Average response time")
    user_satisfaction: Optional[float] = Field(None, description="User satisfaction score")
    errors_encountered: int = Field(..., description="Number of errors encountered")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class ConversationSummary(BaseModel):
    """Conversation summary model."""

    session_id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Conversation title")
    summary: str = Field(..., description="Conversation summary")
    key_topics: List[str] = Field(..., description="Key topics discussed")
    actions_taken: List[str] = Field(..., description="Actions taken")
    recommendations: List[str] = Field(..., description="Recommendations provided")
    outcome: str = Field(..., description="Conversation outcome")
    created_at: datetime = Field(..., description="Summary creation time")


class ConversationFeedback(BaseModel):
    """Conversation feedback model."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    helpful_responses: List[str] = Field(..., description="Helpful response IDs")
    issues_encountered: List[str] = Field(..., description="Issues encountered")
    suggestions: Optional[str] = Field(None, description="User suggestions")
    timestamp: datetime = Field(..., description="Feedback timestamp")


class ConversationExport(BaseModel):
    """Conversation export model."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Conversation title")
    messages: List[ConversationMessage] = Field(..., description="All messages")
    summary: Optional[ConversationSummary] = Field(None, description="Conversation summary")
    analytics: Optional[ConversationMetrics] = Field(None, description="Conversation metrics")
    exported_at: datetime = Field(..., description="Export timestamp")
    format: str = Field(..., description="Export format")


class ConversationSettings(BaseModel):
    """Conversation settings model."""

    user_id: str = Field(..., description="User ID")
    language: str = Field("en", description="Preferred language")
    response_style: str = Field("professional", description="Response style")
    max_context_length: int = Field(10, description="Maximum context length")
    enable_suggestions: bool = Field(True, description="Enable suggestions")
    enable_follow_up_questions: bool = Field(True, description="Enable follow-up questions")
    notification_preferences: Dict[str, bool] = Field(..., description="Notification preferences")
    privacy_settings: Dict[str, bool] = Field(..., description="Privacy settings")
    updated_at: datetime = Field(..., description="Settings update time")
