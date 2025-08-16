#!/usr/bin/env python3
"""
PolicyCortex Usage Metering Service
Per-API usage tracking with tiered plans and quota enforcement
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import uuid
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyCortex Usage Metering Service", version="1.0.0")

class TierType(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class UsageType(str, Enum):
    API_CALL = "api_call"
    PREDICTION = "prediction"
    ANALYSIS = "analysis"
    STORAGE = "storage"
    COMPUTE = "compute"

class BillingEvent(BaseModel):
    event_id: str
    tenant_id: str
    user_id: Optional[str]
    api_endpoint: str
    usage_type: UsageType
    quantity: float
    unit: str  # calls, mb, seconds, etc.
    cost: float
    tier: TierType
    timestamp: datetime
    metadata: Dict[str, Any]

class UsageQuota(BaseModel):
    tenant_id: str
    tier: TierType
    usage_type: UsageType
    monthly_limit: int
    current_usage: int
    period_start: datetime
    period_end: datetime
    overage_allowed: bool = False
    overage_rate: float = 0.0

class TierConfig(BaseModel):
    tier: TierType
    name: str
    monthly_cost: float
    quotas: Dict[UsageType, int]
    overage_rates: Dict[UsageType, float]
    features: List[str]
    support_level: str

class UsageMeter(BaseModel):
    tenant_id: str
    api_endpoint: str
    usage_type: UsageType
    quantity: float
    unit: str
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    timestamp: datetime = None

class UsageMeteringService:
    """Main usage metering service"""
    
    def __init__(self):
        self.config = self._load_config()
        self.db_pool = None
        self.redis_client = None
        self.tier_configs = self._initialize_tier_configs()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        return {
            'database': {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'policycortex'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', 'postgres'),
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 0)),
            },
            'billing': {
                'service_url': os.getenv('BILLING_SERVICE_URL', 'http://localhost:8084'),
                'batch_size': int(os.getenv('BILLING_BATCH_SIZE', 100)),
                'flush_interval': int(os.getenv('BILLING_FLUSH_INTERVAL', 60)),
            },
            'quotas': {
                'enforcement_enabled': os.getenv('QUOTA_ENFORCEMENT', 'true').lower() == 'true',
                'grace_period_hours': int(os.getenv('QUOTA_GRACE_PERIOD', 24)),
                'warning_threshold': float(os.getenv('QUOTA_WARNING_THRESHOLD', 0.8)),
            }
        }
    
    def _initialize_tier_configs(self) -> Dict[TierType, TierConfig]:
        """Initialize tier configurations"""
        return {
            TierType.FREE: TierConfig(
                tier=TierType.FREE,
                name="Free Tier",
                monthly_cost=0.0,
                quotas={
                    UsageType.API_CALL: 1000,
                    UsageType.PREDICTION: 100,
                    UsageType.ANALYSIS: 10,
                    UsageType.STORAGE: 1024,  # MB
                    UsageType.COMPUTE: 3600,  # seconds
                },
                overage_rates={
                    UsageType.API_CALL: 0.001,
                    UsageType.PREDICTION: 0.01,
                    UsageType.ANALYSIS: 0.1,
                    UsageType.STORAGE: 0.0001,
                    UsageType.COMPUTE: 0.01,
                },
                features=["Basic API access", "Standard support"],
                support_level="Community"
            ),
            TierType.PRO: TierConfig(
                tier=TierType.PRO,
                name="Professional",
                monthly_cost=99.0,
                quotas={
                    UsageType.API_CALL: 50000,
                    UsageType.PREDICTION: 10000,
                    UsageType.ANALYSIS: 1000,
                    UsageType.STORAGE: 10240,  # MB
                    UsageType.COMPUTE: 36000,  # seconds
                },
                overage_rates={
                    UsageType.API_CALL: 0.0008,
                    UsageType.PREDICTION: 0.008,
                    UsageType.ANALYSIS: 0.08,
                    UsageType.STORAGE: 0.00008,
                    UsageType.COMPUTE: 0.008,
                },
                features=["Full API access", "Advanced analytics", "Priority support"],
                support_level="Business"
            ),
            TierType.ENTERPRISE: TierConfig(
                tier=TierType.ENTERPRISE,
                name="Enterprise",
                monthly_cost=999.0,
                quotas={
                    UsageType.API_CALL: 1000000,
                    UsageType.PREDICTION: 100000,
                    UsageType.ANALYSIS: 10000,
                    UsageType.STORAGE: 102400,  # MB
                    UsageType.COMPUTE: 360000,  # seconds
                },
                overage_rates={
                    UsageType.API_CALL: 0.0005,
                    UsageType.PREDICTION: 0.005,
                    UsageType.ANALYSIS: 0.05,
                    UsageType.STORAGE: 0.00005,
                    UsageType.COMPUTE: 0.005,
                },
                features=[
                    "Unlimited API access", 
                    "Custom analytics", 
                    "Dedicated support",
                    "SLA guarantees",
                    "Custom integrations"
                ],
                support_level="Premium"
            )
        }
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing usage metering service...")
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            min_size=2,
            max_size=10
        )
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            decode_responses=True
        )
        
        # Initialize database tables
        await self._initialize_database()
        
        # Start background tasks
        asyncio.create_task(self._billing_event_processor())
        asyncio.create_task(self._quota_checker())
        
        logger.info("Usage metering service initialized")
    
    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            async with self.db_pool.acquire() as conn:
                # Usage events table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_events (
                        event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id VARCHAR NOT NULL,
                        user_id VARCHAR,
                        api_endpoint VARCHAR NOT NULL,
                        usage_type VARCHAR NOT NULL,
                        quantity DECIMAL NOT NULL,
                        unit VARCHAR NOT NULL,
                        cost DECIMAL DEFAULT 0,
                        tier VARCHAR NOT NULL,
                        request_size_bytes INTEGER,
                        response_size_bytes INTEGER,
                        processing_time_ms DECIMAL,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        metadata JSONB DEFAULT '{}'
                    )
                """)
                
                # Usage quotas table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_quotas (
                        quota_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id VARCHAR NOT NULL,
                        tier VARCHAR NOT NULL,
                        usage_type VARCHAR NOT NULL,
                        monthly_limit INTEGER NOT NULL,
                        current_usage INTEGER DEFAULT 0,
                        period_start TIMESTAMP NOT NULL,
                        period_end TIMESTAMP NOT NULL,
                        overage_allowed BOOLEAN DEFAULT FALSE,
                        overage_rate DECIMAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(tenant_id, usage_type, period_start)
                    )
                """)
                
                # Tenant tiers table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS tenant_tiers (
                        tenant_id VARCHAR PRIMARY KEY,
                        tier VARCHAR NOT NULL,
                        tier_start_date TIMESTAMP DEFAULT NOW(),
                        tier_end_date TIMESTAMP,
                        billing_cycle_day INTEGER DEFAULT 1,
                        payment_method VARCHAR,
                        billing_address JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Billing events table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS billing_events (
                        event_id UUID PRIMARY KEY,
                        tenant_id VARCHAR NOT NULL,
                        user_id VARCHAR,
                        api_endpoint VARCHAR NOT NULL,
                        usage_type VARCHAR NOT NULL,
                        quantity DECIMAL NOT NULL,
                        unit VARCHAR NOT NULL,
                        cost DECIMAL NOT NULL,
                        tier VARCHAR NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        processed BOOLEAN DEFAULT FALSE,
                        processed_at TIMESTAMP,
                        metadata JSONB DEFAULT '{}'
                    )
                """)
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_time ON usage_events(tenant_id, timestamp)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_endpoint ON usage_events(api_endpoint)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_quotas_tenant ON usage_quotas(tenant_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_billing_events_tenant ON billing_events(tenant_id)")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def record_usage(self, usage: UsageMeter) -> BillingEvent:
        """Record usage event"""
        if not usage.timestamp:
            usage.timestamp = datetime.utcnow()
        
        # Get tenant tier
        tier = await self._get_tenant_tier(usage.tenant_id)
        
        # Check quotas
        await self._check_and_update_quota(usage.tenant_id, usage.usage_type, usage.quantity, tier)
        
        # Calculate cost
        cost = await self._calculate_cost(usage, tier)
        
        # Create billing event
        billing_event = BillingEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=usage.tenant_id,
            user_id=None,  # Would be extracted from request context
            api_endpoint=usage.api_endpoint,
            usage_type=usage.usage_type,
            quantity=usage.quantity,
            unit=usage.unit,
            cost=cost,
            tier=tier,
            timestamp=usage.timestamp,
            metadata={
                'request_size_bytes': usage.request_size_bytes,
                'response_size_bytes': usage.response_size_bytes,
                'processing_time_ms': usage.processing_time_ms
            }
        )
        
        # Store usage event
        await self._store_usage_event(usage, billing_event)
        
        # Queue for billing
        await self._queue_billing_event(billing_event)
        
        return billing_event
    
    async def _get_tenant_tier(self, tenant_id: str) -> TierType:
        """Get tenant's current tier"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT tier FROM tenant_tiers WHERE tenant_id = $1",
                    tenant_id
                )
                if row:
                    return TierType(row['tier'])
                else:
                    # Default to free tier
                    await self._set_tenant_tier(tenant_id, TierType.FREE)
                    return TierType.FREE
        except Exception as e:
            logger.error(f"Error getting tenant tier: {e}")
            return TierType.FREE
    
    async def _set_tenant_tier(self, tenant_id: str, tier: TierType):
        """Set tenant tier"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_tiers (tenant_id, tier)
                VALUES ($1, $2)
                ON CONFLICT (tenant_id) DO UPDATE SET
                    tier = $2,
                    tier_start_date = NOW(),
                    updated_at = NOW()
            """, tenant_id, tier.value)
    
    async def _check_and_update_quota(self, tenant_id: str, usage_type: UsageType, quantity: float, tier: TierType):
        """Check and update usage quota"""
        if not self.config['quotas']['enforcement_enabled']:
            return
        
        # Get current period
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, 1)
        if now.month == 12:
            period_end = datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            period_end = datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)
        
        async with self.db_pool.acquire() as conn:
            # Get or create quota
            quota = await conn.fetchrow("""
                SELECT * FROM usage_quotas 
                WHERE tenant_id = $1 AND usage_type = $2 AND period_start = $3
            """, tenant_id, usage_type.value, period_start)
            
            if not quota:
                # Create new quota for this period
                tier_config = self.tier_configs[tier]
                monthly_limit = tier_config.quotas.get(usage_type, 0)
                
                await conn.execute("""
                    INSERT INTO usage_quotas 
                    (tenant_id, tier, usage_type, monthly_limit, current_usage, 
                     period_start, period_end, overage_allowed, overage_rate)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, tenant_id, tier.value, usage_type.value, monthly_limit, 0,
                    period_start, period_end, True, tier_config.overage_rates.get(usage_type, 0))
                
                current_usage = 0
                monthly_limit = monthly_limit
            else:
                current_usage = quota['current_usage']
                monthly_limit = quota['monthly_limit']
            
            # Check if usage would exceed quota
            new_usage = current_usage + quantity
            if new_usage > monthly_limit and not quota.get('overage_allowed', True):
                raise HTTPException(
                    status_code=429,
                    detail=f"Usage quota exceeded for {usage_type.value}. Limit: {monthly_limit}, Current: {current_usage}"
                )
            
            # Update usage
            await conn.execute("""
                UPDATE usage_quotas 
                SET current_usage = current_usage + $1, updated_at = NOW()
                WHERE tenant_id = $2 AND usage_type = $3 AND period_start = $4
            """, quantity, tenant_id, usage_type.value, period_start)
            
            # Check for warning threshold
            warning_threshold = monthly_limit * self.config['quotas']['warning_threshold']
            if new_usage >= warning_threshold and current_usage < warning_threshold:
                await self._send_quota_warning(tenant_id, usage_type, new_usage, monthly_limit)
    
    async def _calculate_cost(self, usage: UsageMeter, tier: TierType) -> float:
        """Calculate cost for usage"""
        tier_config = self.tier_configs[tier]
        
        # For now, use overage rates as base rates (simplified)
        rate = tier_config.overage_rates.get(usage.usage_type, 0.0)
        
        cost = usage.quantity * rate
        
        # Apply processing time multiplier for compute
        if usage.usage_type == UsageType.COMPUTE and usage.processing_time_ms:
            cost = (usage.processing_time_ms / 1000.0) * rate
        
        # Apply size-based pricing for storage/data transfer
        if usage.usage_type == UsageType.STORAGE and usage.request_size_bytes:
            cost = (usage.request_size_bytes / (1024 * 1024)) * rate  # Per MB
        
        return round(cost, 6)
    
    async def _store_usage_event(self, usage: UsageMeter, billing_event: BillingEvent):
        """Store usage event in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO usage_events 
                (event_id, tenant_id, api_endpoint, usage_type, quantity, unit, cost, tier,
                 request_size_bytes, response_size_bytes, processing_time_ms, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, 
                billing_event.event_id,
                usage.tenant_id,
                usage.api_endpoint,
                usage.usage_type.value,
                usage.quantity,
                usage.unit,
                billing_event.cost,
                billing_event.tier.value,
                usage.request_size_bytes,
                usage.response_size_bytes,
                usage.processing_time_ms,
                usage.timestamp,
                json.dumps(billing_event.metadata)
            )
    
    async def _queue_billing_event(self, billing_event: BillingEvent):
        """Queue billing event for processing"""
        await self.redis_client.lpush(
            "billing_events_queue",
            json.dumps(billing_event.dict(), default=str)
        )
    
    async def _billing_event_processor(self):
        """Process billing events in background"""
        while True:
            try:
                # Get batch of events
                events = await self.redis_client.lrange("billing_events_queue", 0, self.config['billing']['batch_size'] - 1)
                if events:
                    await self.redis_client.ltrim("billing_events_queue", len(events), -1)
                    
                    # Process events
                    await self._process_billing_events(events)
                
                await asyncio.sleep(self.config['billing']['flush_interval'])
                
            except Exception as e:
                logger.error(f"Error processing billing events: {e}")
                await asyncio.sleep(10)
    
    async def _process_billing_events(self, events: List[str]):
        """Process batch of billing events"""
        billing_events = []
        for event_data in events:
            try:
                event_dict = json.loads(event_data)
                billing_event = BillingEvent(**event_dict)
                billing_events.append(billing_event)
            except Exception as e:
                logger.error(f"Failed to parse billing event: {e}")
        
        if not billing_events:
            return
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            for event in billing_events:
                await conn.execute("""
                    INSERT INTO billing_events 
                    (event_id, tenant_id, user_id, api_endpoint, usage_type, quantity, 
                     unit, cost, tier, timestamp, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (event_id) DO NOTHING
                """,
                    event.event_id,
                    event.tenant_id,
                    event.user_id,
                    event.api_endpoint,
                    event.usage_type.value,
                    event.quantity,
                    event.unit,
                    event.cost,
                    event.tier.value,
                    event.timestamp,
                    json.dumps(event.metadata)
                )
        
        # Send to billing service
        await self._send_to_billing_service(billing_events)
    
    async def _send_to_billing_service(self, events: List[BillingEvent]):
        """Send events to external billing service"""
        try:
            billing_data = {
                'events': [event.dict() for event in events],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config['billing']['service_url']}/events",
                    json=billing_data,
                    timeout=30.0
                )
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Failed to send events to billing service: {e}")
    
    async def _quota_checker(self):
        """Background task to check quotas"""
        while True:
            try:
                await self._check_all_quotas()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in quota checker: {e}")
                await asyncio.sleep(300)
    
    async def _check_all_quotas(self):
        """Check all tenant quotas"""
        async with self.db_pool.acquire() as conn:
            quotas = await conn.fetch("""
                SELECT tenant_id, usage_type, current_usage, monthly_limit
                FROM usage_quotas
                WHERE period_end > NOW()
            """)
            
            for quota in quotas:
                usage_percentage = quota['current_usage'] / quota['monthly_limit']
                if usage_percentage >= 1.0:
                    await self._handle_quota_exceeded(quota)
                elif usage_percentage >= self.config['quotas']['warning_threshold']:
                    await self._send_quota_warning(
                        quota['tenant_id'],
                        UsageType(quota['usage_type']),
                        quota['current_usage'],
                        quota['monthly_limit']
                    )
    
    async def _handle_quota_exceeded(self, quota: Dict):
        """Handle quota exceeded"""
        logger.warning(f"Quota exceeded for tenant {quota['tenant_id']}, type {quota['usage_type']}")
        # Would implement actual quota enforcement here
    
    async def _send_quota_warning(self, tenant_id: str, usage_type: UsageType, current: int, limit: int):
        """Send quota warning"""
        logger.info(f"Quota warning for tenant {tenant_id}: {usage_type.value} at {current}/{limit}")
        # Would implement actual notification here

# Global service instance
usage_service = UsageMeteringService()

class UsageMeteringMiddleware(BaseHTTPMiddleware):
    """Middleware to track API usage"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract tenant ID from request (simplified)
        tenant_id = request.headers.get('X-Tenant-ID', 'default')
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Get response size (approximation)
        response_size = len(response.body) if hasattr(response, 'body') else 0
        
        # Determine usage type based on endpoint
        usage_type = self._determine_usage_type(str(request.url))
        
        # Record usage
        try:
            usage = UsageMeter(
                tenant_id=tenant_id,
                api_endpoint=str(request.url.path),
                usage_type=usage_type,
                quantity=1.0,
                unit="call",
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                processing_time_ms=processing_time
            )
            
            await usage_service.record_usage(usage)
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
        
        return response
    
    def _determine_usage_type(self, url: str) -> UsageType:
        """Determine usage type from URL"""
        if '/predictions' in url:
            return UsageType.PREDICTION
        elif '/analysis' in url:
            return UsageType.ANALYSIS
        elif '/storage' in url:
            return UsageType.STORAGE
        else:
            return UsageType.API_CALL

# Add middleware
app.add_middleware(UsageMeteringMiddleware)

@app.on_event("startup")
async def startup_event():
    await usage_service.initialize()

@app.post("/usage/record")
async def record_usage_manual(usage: UsageMeter):
    """Manually record usage"""
    return await usage_service.record_usage(usage)

@app.get("/usage/{tenant_id}")
async def get_tenant_usage(tenant_id: str, usage_type: Optional[UsageType] = None):
    """Get tenant usage"""
    async with usage_service.db_pool.acquire() as conn:
        query = """
            SELECT usage_type, SUM(quantity) as total_quantity, SUM(cost) as total_cost
            FROM usage_events 
            WHERE tenant_id = $1 
            AND timestamp >= date_trunc('month', NOW())
        """
        params = [tenant_id]
        
        if usage_type:
            query += " AND usage_type = $2"
            params.append(usage_type.value)
        
        query += " GROUP BY usage_type"
        
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

@app.get("/quotas/{tenant_id}")
async def get_tenant_quotas(tenant_id: str):
    """Get tenant quotas"""
    async with usage_service.db_pool.acquire() as conn:
        quotas = await conn.fetch("""
            SELECT usage_type, monthly_limit, current_usage, 
                   (current_usage::float / monthly_limit * 100) as usage_percentage
            FROM usage_quotas 
            WHERE tenant_id = $1 AND period_end > NOW()
        """, tenant_id)
        return [dict(quota) for quota in quotas]

@app.post("/tiers/{tenant_id}")
async def set_tenant_tier(tenant_id: str, tier: TierType):
    """Set tenant tier"""
    await usage_service._set_tenant_tier(tenant_id, tier)
    return {"status": "updated", "tier": tier}

@app.get("/tiers")
async def get_tier_configs():
    """Get available tier configurations"""
    return {tier.value: config.dict() for tier, config in usage_service.tier_configs.items()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "usage-metering"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)