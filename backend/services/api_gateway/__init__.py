"""
PolicyCortex API Gateway Package
"""

from .main import app
from .auth_middleware import (
    AuthContext,
    get_auth_context,
    require_auth,
    require_roles,
    require_admin,
    TenantIsolation,
    ResourceAuthorization
)
from .rate_limiter import (
    rate_limiter,
    rate_limit,
    circuit_breaker,
    rate_limit_middleware
)
from .observability import (
    observability,
    CorrelationIdMiddleware,
    MetricsMiddleware,
    trace,
    timed,
    counted
)
from .websocket_manager import (
    websocket_manager,
    WebSocketManager,
    WebSocketMessage,
    MessageType
)
from .event_sourcing import (
    Event,
    EventType,
    EventStore,
    EventSourcingService,
    initialize_event_sourcing
)
from .caching_strategy import (
    cache_manager,
    MultiTierCache,
    CacheDecorator,
    initialize_cache
)
from .data_validation import (
    InputSanitizer,
    DataValidator,
    RequestLimiter,
    ValidationMiddleware
)

__all__ = [
    'app',
    'AuthContext',
    'get_auth_context',
    'require_auth',
    'require_roles',
    'require_admin',
    'rate_limiter',
    'rate_limit',
    'circuit_breaker',
    'observability',
    'websocket_manager',
    'cache_manager',
    'initialize_event_sourcing',
    'initialize_cache'
]

__version__ = '3.0.0'