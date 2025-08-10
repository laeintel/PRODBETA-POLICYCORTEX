"""
Event Sourcing implementation with EventStore for complete audit trail and event-driven architecture
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Type, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod

# EventStore client
try:
    from esdbclient import EventStoreDBClient, NewEvent, StreamState
    EVENTSTORE_AVAILABLE = True
except ImportError:
    EVENTSTORE_AVAILABLE = False
    logging.warning("EventStore client not available, using in-memory implementation")

# PostgreSQL for projections
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# Type variables for generic event handling
T = TypeVar('T')

class EventType(Enum):
    """Standard event types for the system"""
    # Resource events
    RESOURCE_CREATED = "resource.created"
    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"
    RESOURCE_TAGGED = "resource.tagged"
    
    # Policy events
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_APPLIED = "policy.applied"
    POLICY_VIOLATED = "policy.violated"
    POLICY_EXEMPTED = "policy.exempted"
    
    # Compliance events
    COMPLIANCE_CHECKED = "compliance.checked"
    COMPLIANCE_PASSED = "compliance.passed"
    COMPLIANCE_FAILED = "compliance.failed"
    COMPLIANCE_REMEDIATED = "compliance.remediated"
    
    # Security events
    SECURITY_ALERT = "security.alert"
    SECURITY_INCIDENT = "security.incident"
    SECURITY_RESOLVED = "security.resolved"
    
    # Cost events
    COST_THRESHOLD_EXCEEDED = "cost.threshold_exceeded"
    COST_OPTIMIZATION_FOUND = "cost.optimization_found"
    COST_OPTIMIZATION_APPLIED = "cost.optimization_applied"
    
    # User events
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_ACTION = "user.action"
    USER_PERMISSION_CHANGED = "user.permission_changed"
    
    # Approval events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_ESCALATED = "approval.escalated"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_ERROR = "system.error"
    SYSTEM_CONFIGURATION_CHANGED = "system.configuration_changed"


@dataclass
class Event:
    """Base event class"""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    timestamp: datetime
    version: int
    correlation_id: Optional[str]
    causation_id: Optional[str]
    user_id: Optional[str]
    tenant_id: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        return cls(**data)


class EventStore:
    """Event store interface"""
    
    @abstractmethod
    async def append_event(self, stream_name: str, event: Event) -> None:
        """Append event to stream"""
        pass
    
    @abstractmethod
    async def get_events(self, stream_name: str, from_version: int = 0) -> List[Event]:
        """Get events from stream"""
        pass
    
    @abstractmethod
    async def get_all_events(self, from_position: Optional[str] = None) -> List[Event]:
        """Get all events from all streams"""
        pass
    
    @abstractmethod
    async def create_subscription(self, stream_name: str, handler: callable) -> None:
        """Create subscription to stream"""
        pass


class EventStoreDB(EventStore):
    """EventStore database implementation"""
    
    def __init__(self, connection_string: str = "esdb://localhost:2113?tls=false"):
        if not EVENTSTORE_AVAILABLE:
            raise RuntimeError("EventStore client not available")
        
        self.client = EventStoreDBClient(uri=connection_string)
        self.subscriptions = {}
        
    async def append_event(self, stream_name: str, event: Event) -> None:
        """Append event to EventStore stream"""
        new_event = NewEvent(
            type=event.event_type,
            data=json.dumps(event.to_dict()).encode(),
            metadata=json.dumps(event.metadata).encode()
        )
        
        self.client.append_to_stream(
            stream_name=stream_name,
            current_version=StreamState.ANY,
            events=[new_event]
        )
        
    async def get_events(self, stream_name: str, from_version: int = 0) -> List[Event]:
        """Get events from EventStore stream"""
        events = []
        
        for recorded_event in self.client.read_stream(
            stream_name=stream_name,
            stream_position=from_version
        ):
            event_data = json.loads(recorded_event.data)
            events.append(Event.from_dict(event_data))
            
        return events
    
    async def get_all_events(self, from_position: Optional[str] = None) -> List[Event]:
        """Get all events from EventStore"""
        events = []
        
        for recorded_event in self.client.read_all(
            commit_position=from_position or 0
        ):
            if not recorded_event.type.startswith("$"):  # Skip system events
                event_data = json.loads(recorded_event.data)
                events.append(Event.from_dict(event_data))
                
        return events
    
    async def create_subscription(self, stream_name: str, handler: callable) -> None:
        """Create persistent subscription to stream"""
        subscription_id = f"sub-{stream_name}-{uuid.uuid4().hex[:8]}"
        
        async def process_events():
            for event in self.client.subscribe_to_stream(
                stream_name=stream_name,
                stream_position=0
            ):
                if not event.type.startswith("$"):
                    event_data = json.loads(event.data)
                    await handler(Event.from_dict(event_data))
        
        self.subscriptions[subscription_id] = asyncio.create_task(process_events())


class InMemoryEventStore(EventStore):
    """In-memory event store for development/testing"""
    
    def __init__(self):
        self.streams: Dict[str, List[Event]] = {}
        self.all_events: List[Event] = []
        self.subscriptions: Dict[str, List[callable]] = {}
        
    async def append_event(self, stream_name: str, event: Event) -> None:
        """Append event to in-memory stream"""
        if stream_name not in self.streams:
            self.streams[stream_name] = []
            
        self.streams[stream_name].append(event)
        self.all_events.append(event)
        
        # Notify subscribers
        if stream_name in self.subscriptions:
            for handler in self.subscriptions[stream_name]:
                await handler(event)
                
    async def get_events(self, stream_name: str, from_version: int = 0) -> List[Event]:
        """Get events from in-memory stream"""
        if stream_name not in self.streams:
            return []
            
        return self.streams[stream_name][from_version:]
    
    async def get_all_events(self, from_position: Optional[str] = None) -> List[Event]:
        """Get all events from in-memory store"""
        if from_position:
            try:
                position = int(from_position)
                return self.all_events[position:]
            except (ValueError, IndexError):
                return self.all_events
        return self.all_events
    
    async def create_subscription(self, stream_name: str, handler: callable) -> None:
        """Create subscription to in-memory stream"""
        if stream_name not in self.subscriptions:
            self.subscriptions[stream_name] = []
            
        self.subscriptions[stream_name].append(handler)


class Aggregate(ABC):
    """Base aggregate root for event sourcing"""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.pending_events: List[Event] = []
        
    @abstractmethod
    def apply_event(self, event: Event) -> None:
        """Apply event to aggregate state"""
        pass
    
    def add_event(self, event: Event) -> None:
        """Add event to pending events and apply it"""
        event.version = self.version + 1
        self.pending_events.append(event)
        self.apply_event(event)
        self.version += 1
        
    def mark_events_committed(self) -> None:
        """Clear pending events after commit"""
        self.pending_events.clear()
        
    @classmethod
    @abstractmethod
    def from_events(cls, events: List[Event]) -> 'Aggregate':
        """Rebuild aggregate from events"""
        pass


class ResourceAggregate(Aggregate):
    """Resource aggregate for event sourcing"""
    
    def __init__(self, resource_id: str):
        super().__init__(resource_id)
        self.resource_type = None
        self.name = None
        self.provider = None
        self.region = None
        self.tags = {}
        self.configuration = {}
        self.compliance_status = None
        self.deleted = False
        
    def apply_event(self, event: Event) -> None:
        """Apply event to resource state"""
        if event.event_type == EventType.RESOURCE_CREATED.value:
            self.resource_type = event.data.get("resource_type")
            self.name = event.data.get("name")
            self.provider = event.data.get("provider")
            self.region = event.data.get("region")
            self.tags = event.data.get("tags", {})
            self.configuration = event.data.get("configuration", {})
            
        elif event.event_type == EventType.RESOURCE_UPDATED.value:
            if "configuration" in event.data:
                self.configuration.update(event.data["configuration"])
            if "tags" in event.data:
                self.tags.update(event.data["tags"])
                
        elif event.event_type == EventType.RESOURCE_DELETED.value:
            self.deleted = True
            
        elif event.event_type == EventType.COMPLIANCE_CHECKED.value:
            self.compliance_status = event.data.get("status")
            
    @classmethod
    def from_events(cls, events: List[Event]) -> 'ResourceAggregate':
        """Rebuild resource from events"""
        if not events:
            return None
            
        aggregate = cls(events[0].aggregate_id)
        for event in events:
            aggregate.apply_event(event)
            aggregate.version = event.version
            
        return aggregate


class EventProjection(ABC):
    """Base class for event projections"""
    
    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle event and update projection"""
        pass


class ResourceProjection(EventProjection):
    """Resource projection for read model"""
    
    def __init__(self, db_session: AsyncSession):
        self.session = db_session
        
    async def handle_event(self, event: Event) -> None:
        """Update resource projection based on event"""
        if event.event_type == EventType.RESOURCE_CREATED.value:
            # Create resource in read model
            await self._create_resource(event)
            
        elif event.event_type == EventType.RESOURCE_UPDATED.value:
            # Update resource in read model
            await self._update_resource(event)
            
        elif event.event_type == EventType.RESOURCE_DELETED.value:
            # Mark resource as deleted in read model
            await self._delete_resource(event)
            
    async def _create_resource(self, event: Event) -> None:
        """Create resource in database"""
        # Implementation depends on your database schema
        pass
        
    async def _update_resource(self, event: Event) -> None:
        """Update resource in database"""
        pass
        
    async def _delete_resource(self, event: Event) -> None:
        """Delete resource from database"""
        pass


class EventSourcingService:
    """Main event sourcing service"""
    
    def __init__(self, event_store: EventStore, db_session: Optional[AsyncSession] = None):
        self.event_store = event_store
        self.db_session = db_session
        self.projections: List[EventProjection] = []
        self.event_handlers: Dict[str, List[callable]] = {}
        
    async def save_aggregate(self, aggregate: Aggregate, stream_name: str) -> None:
        """Save aggregate events to event store"""
        for event in aggregate.pending_events:
            await self.event_store.append_event(stream_name, event)
            
            # Update projections
            for projection in self.projections:
                await projection.handle_event(event)
                
            # Call event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    await handler(event)
                    
        aggregate.mark_events_committed()
        
    async def load_aggregate(self, aggregate_class: Type[Aggregate], stream_name: str) -> Optional[Aggregate]:
        """Load aggregate from event store"""
        events = await self.event_store.get_events(stream_name)
        if not events:
            return None
            
        return aggregate_class.from_events(events)
    
    def register_projection(self, projection: EventProjection) -> None:
        """Register projection for automatic updates"""
        self.projections.append(projection)
        
    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """Register handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler)
        
    async def replay_events(self, from_position: Optional[str] = None) -> None:
        """Replay all events to rebuild projections"""
        events = await self.event_store.get_all_events(from_position)
        
        for event in events:
            for projection in self.projections:
                await projection.handle_event(event)
                
    async def get_audit_trail(self, aggregate_id: str, event_types: Optional[List[str]] = None) -> List[Event]:
        """Get audit trail for an aggregate"""
        stream_name = f"aggregate-{aggregate_id}"
        events = await self.event_store.get_events(stream_name)
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
            
        return events


class EventBuilder:
    """Helper class to build events"""
    
    @staticmethod
    def create_resource_event(
        event_type: EventType,
        resource_id: str,
        resource_type: str,
        user_id: str,
        tenant_id: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Event:
        """Create a resource event"""
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type.value,
            aggregate_id=resource_id,
            aggregate_type="Resource",
            timestamp=datetime.utcnow(),
            version=0,  # Will be set by aggregate
            correlation_id=correlation_id or str(uuid.uuid4()),
            causation_id=None,
            user_id=user_id,
            tenant_id=tenant_id,
            data={
                "resource_id": resource_id,
                "resource_type": resource_type,
                **data
            },
            metadata={
                "source": "PolicyCortex",
                "version": "2.0.0"
            }
        )
    
    @staticmethod
    def create_compliance_event(
        resource_id: str,
        policy_id: str,
        status: str,
        user_id: str,
        tenant_id: str,
        details: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Event:
        """Create a compliance event"""
        event_type = EventType.COMPLIANCE_PASSED if status == "passed" else EventType.COMPLIANCE_FAILED
        
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type.value,
            aggregate_id=resource_id,
            aggregate_type="Resource",
            timestamp=datetime.utcnow(),
            version=0,
            correlation_id=correlation_id or str(uuid.uuid4()),
            causation_id=None,
            user_id=user_id,
            tenant_id=tenant_id,
            data={
                "resource_id": resource_id,
                "policy_id": policy_id,
                "status": status,
                "details": details
            },
            metadata={
                "source": "PolicyCortex",
                "version": "2.0.0"
            }
        )


class SnapshotStore:
    """Store and retrieve aggregate snapshots for performance"""
    
    def __init__(self, cache_client=None):
        self.cache = cache_client or {}  # Use dict if no cache provided
        
    async def save_snapshot(self, aggregate_id: str, aggregate: Aggregate, version: int) -> None:
        """Save aggregate snapshot"""
        snapshot_key = f"snapshot:{aggregate_id}:{version}"
        snapshot_data = {
            "aggregate_id": aggregate_id,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "state": aggregate.__dict__
        }
        
        if hasattr(self.cache, 'set'):
            await self.cache.set(snapshot_key, json.dumps(snapshot_data), ex=86400)  # 24h TTL
        else:
            self.cache[snapshot_key] = snapshot_data
            
    async def get_latest_snapshot(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get latest snapshot for aggregate"""
        # In production, query cache for latest version
        # For now, return None to always rebuild from events
        return None


# Global event sourcing service instance
event_sourcing_service = None

def initialize_event_sourcing(
    connection_string: Optional[str] = None,
    use_eventstore: bool = True,
    db_session: Optional[AsyncSession] = None
) -> EventSourcingService:
    """Initialize event sourcing service"""
    global event_sourcing_service
    
    if use_eventstore and EVENTSTORE_AVAILABLE and connection_string:
        event_store = EventStoreDB(connection_string)
    else:
        event_store = InMemoryEventStore()
        
    event_sourcing_service = EventSourcingService(event_store, db_session)
    
    logger.info(f"Event sourcing initialized with {type(event_store).__name__}")
    
    return event_sourcing_service


# Export key components
__all__ = [
    "Event",
    "EventType",
    "EventStore",
    "EventStoreDB",
    "InMemoryEventStore",
    "Aggregate",
    "ResourceAggregate",
    "EventProjection",
    "ResourceProjection",
    "EventSourcingService",
    "EventBuilder",
    "SnapshotStore",
    "initialize_event_sourcing",
    "event_sourcing_service"
]