"""
PolicyCortex API Gateway Package
"""

"""
Lightweight package initializer for the API gateway.
Only guarantees that `app` is importable; all other modules are optional.
"""

# Import app lazily in uvicorn path; avoid heavy imports at package import time
from .main import app  # noqa: F401
try:
    from .auth_middleware import (
        AuthContext,
        get_auth_context,
        require_auth,
        require_roles,
        require_admin,
        TenantIsolation,
        ResourceAuthorization
    )
except Exception:
    AuthContext = None
    def get_auth_context(*args, **kwargs):
        return {}
    def require_auth(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def require_roles(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def require_admin(*args, **kwargs):
        def deco(f):
            return f
        return deco
    TenantIsolation = None
    ResourceAuthorization = None
try:
    from .rate_limiter import (
        rate_limiter,
        rate_limit,
        circuit_breaker,
        rate_limit_middleware
    )
except Exception:
    rate_limiter = None
    def rate_limit(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def circuit_breaker(*args, **kwargs):
        def deco(f):
            return f
        return deco
    async def rate_limit_middleware(request, call_next):
        return await call_next(request)
try:
    from .observability import (
        observability,
        CorrelationIdMiddleware,
        MetricsMiddleware,
        trace,
        timed,
        counted
    )
except Exception:
    observability = None
    CorrelationIdMiddleware = None
    MetricsMiddleware = None
    def trace(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def timed(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def counted(*args, **kwargs):
        def deco(f):
            return f
        return deco
try:
    from .websocket_manager import (
        websocket_manager,
        WebSocketManager,
        WebSocketMessage,
        MessageType
    )
except Exception:
    websocket_manager = None
    WebSocketManager = None
    WebSocketMessage = None
    MessageType = None
try:
    from .event_sourcing import (
        Event,
        EventType,
        EventStore,
        EventSourcingService,
        initialize_event_sourcing
    )
except Exception:
    Event = None
    EventType = None
    EventStore = None
    EventSourcingService = None
    def initialize_event_sourcing(*args, **kwargs):
        return None
try:
    from .caching_strategy import (
        cache_manager,
        MultiTierCache,
        CacheDecorator,
        initialize_cache
    )
except Exception:
    cache_manager = None
    MultiTierCache = None
    CacheDecorator = None
    def initialize_cache(*args, **kwargs):
        return None
try:
    from .data_validation import (
        InputSanitizer,
        DataValidator,
        RequestLimiter,
        ValidationMiddleware
    )
except Exception:
    InputSanitizer = None
    DataValidator = None
    RequestLimiter = None
    ValidationMiddleware = None

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