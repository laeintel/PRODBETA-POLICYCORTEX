"""
Evidence Collector Service for PolicyCortex PROVE Pillar
Gathers evidence from all policy checks and stores with immutability guarantees
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Compliance status enumeration"""
    COMPLIANT = "Compliant"
    NON_COMPLIANT = "NonCompliant"
    WARNING = "Warning"
    ERROR = "Error"
    PENDING = "Pending"


class EvidenceType(str, Enum):
    """Evidence type enumeration"""
    POLICY_CHECK = "PolicyCheck"
    CONFIGURATION_CHANGE = "ConfigurationChange"
    ACCESS_CONTROL = "AccessControl"
    DATA_PROTECTION = "DataProtection"
    INCIDENT_RESPONSE = "IncidentResponse"
    AUDIT_LOG = "AuditLog"
    COMPLIANCE_SCAN = "ComplianceScan"
    SECURITY_ASSESSMENT = "SecurityAssessment"
    REMEDIATION_ACTION = "RemediationAction"


@dataclass
class Evidence:
    """Evidence data model"""
    id: str
    timestamp: datetime
    event_type: str
    resource_id: str
    policy_id: str
    policy_name: str
    compliance_status: ComplianceStatus
    actor: str
    subscription_id: str
    resource_group: str
    resource_type: str
    details: Dict[str, Any]
    metadata: Dict[str, str]
    hash: str = ""
    block_index: Optional[int] = None
    immutable: bool = False


class EvidenceRequest(BaseModel):
    """API request model for evidence collection"""
    event_type: EvidenceType
    resource_id: str = Field(..., description="Azure resource ID")
    policy_id: str = Field(..., description="Policy definition ID")
    policy_name: str = Field(..., description="Policy display name")
    compliance_status: ComplianceStatus
    actor: str = Field(..., description="Identity that triggered the event")
    subscription_id: str
    resource_group: str
    resource_type: str = Field(..., description="Azure resource type")
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)


class EvidenceCollector:
    """Main evidence collector implementation"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize database connections and create tables"""
        # Create PostgreSQL connection pool
        self.db_pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Create Redis connection
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Create evidence table with immutability constraints
        await self._create_tables()
        
    async def _create_tables(self):
        """Create database tables with immutability constraints"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS evidence (
                    id UUID PRIMARY KEY,
                    hash VARCHAR(64) NOT NULL UNIQUE,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    resource_id TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    compliance_status VARCHAR(20) NOT NULL,
                    actor TEXT NOT NULL,
                    subscription_id UUID NOT NULL,
                    resource_group TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    details JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    block_index BIGINT,
                    block_hash VARCHAR(64),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    immutable BOOLEAN DEFAULT FALSE,
                    CONSTRAINT evidence_immutable CHECK (
                        immutable = FALSE OR 
                        (block_index IS NOT NULL AND block_hash IS NOT NULL)
                    )
                );
                
                CREATE INDEX IF NOT EXISTS idx_evidence_timestamp 
                    ON evidence(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_evidence_resource 
                    ON evidence(resource_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_policy 
                    ON evidence(policy_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_compliance 
                    ON evidence(compliance_status);
                CREATE INDEX IF NOT EXISTS idx_evidence_subscription 
                    ON evidence(subscription_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_block 
                    ON evidence(block_index) WHERE block_index IS NOT NULL;
                
                -- Create immutability trigger
                CREATE OR REPLACE FUNCTION enforce_evidence_immutability()
                RETURNS TRIGGER AS $$
                BEGIN
                    IF OLD.immutable = TRUE THEN
                        RAISE EXCEPTION 'Cannot modify immutable evidence record %', OLD.id;
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                DROP TRIGGER IF EXISTS evidence_immutability_trigger ON evidence;
                CREATE TRIGGER evidence_immutability_trigger
                BEFORE UPDATE OR DELETE ON evidence
                FOR EACH ROW
                EXECUTE FUNCTION enforce_evidence_immutability();
                
                -- CQRS event store integration
                CREATE TABLE IF NOT EXISTS evidence_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    aggregate_id UUID NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB NOT NULL,
                    event_metadata JSONB NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    sequence_number BIGSERIAL,
                    FOREIGN KEY (aggregate_id) REFERENCES evidence(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_evidence_events_aggregate 
                    ON evidence_events(aggregate_id);
                CREATE INDEX IF NOT EXISTS idx_evidence_events_timestamp 
                    ON evidence_events(timestamp DESC);
            ''')
            
            logger.info("Evidence tables created successfully")
    
    async def collect_evidence(self, request: EvidenceRequest) -> Evidence:
        """Collect and store evidence"""
        # Create evidence object
        evidence = Evidence(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=request.event_type.value,
            resource_id=request.resource_id,
            policy_id=request.policy_id,
            policy_name=request.policy_name,
            compliance_status=request.compliance_status,
            actor=request.actor,
            subscription_id=request.subscription_id,
            resource_group=request.resource_group,
            resource_type=request.resource_type,
            details=request.details,
            metadata=request.metadata
        )
        
        # Calculate hash
        evidence.hash = self._calculate_hash(evidence)
        
        # Store in database
        await self._store_evidence(evidence)
        
        # Publish to event queue for chain processing
        await self.event_queue.put(evidence)
        
        # Cache recent evidence
        await self._cache_evidence(evidence)
        
        # Emit CQRS event
        await self._emit_event(evidence, "EvidenceCollected")
        
        return evidence
    
    def _calculate_hash(self, evidence: Evidence) -> str:
        """Calculate SHA3-256 hash of evidence"""
        hasher = hashlib.sha3_256()
        
        # Hash all fields in deterministic order
        hash_data = {
            'id': evidence.id,
            'timestamp': evidence.timestamp.isoformat(),
            'event_type': evidence.event_type,
            'resource_id': evidence.resource_id,
            'policy_id': evidence.policy_id,
            'compliance_status': evidence.compliance_status.value,
            'actor': evidence.actor,
            'details': json.dumps(evidence.details, sort_keys=True),
            'metadata': json.dumps(evidence.metadata, sort_keys=True)
        }
        
        hasher.update(json.dumps(hash_data, sort_keys=True).encode())
        return hasher.hexdigest()
    
    async def _store_evidence(self, evidence: Evidence):
        """Store evidence in PostgreSQL"""
        async with self.db_pool.acquire() as conn:
            try:
                await conn.execute('''
                    INSERT INTO evidence (
                        id, hash, timestamp, event_type, resource_id,
                        policy_id, policy_name, compliance_status, actor,
                        subscription_id, resource_group, resource_type,
                        details, metadata, immutable
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ''',
                    uuid.UUID(evidence.id),
                    evidence.hash,
                    evidence.timestamp,
                    evidence.event_type,
                    evidence.resource_id,
                    evidence.policy_id,
                    evidence.policy_name,
                    evidence.compliance_status.value,
                    evidence.actor,
                    uuid.UUID(evidence.subscription_id),
                    evidence.resource_group,
                    evidence.resource_type,
                    json.dumps(evidence.details),
                    json.dumps(evidence.metadata),
                    evidence.immutable
                )
                logger.info(f"Evidence stored: {evidence.id}")
            except asyncpg.UniqueViolationError:
                logger.warning(f"Evidence already exists: {evidence.hash}")
                raise
    
    async def _cache_evidence(self, evidence: Evidence):
        """Cache recent evidence in Redis"""
        key = f"evidence:{evidence.id}"
        value = json.dumps(asdict(evidence), default=str)
        
        # Store with 24-hour TTL
        await self.redis.setex(key, 86400, value)
        
        # Add to recent evidence list
        await self.redis.lpush("evidence:recent", evidence.id)
        await self.redis.ltrim("evidence:recent", 0, 999)  # Keep last 1000
    
    async def _emit_event(self, evidence: Evidence, event_type: str):
        """Emit CQRS event for evidence collection"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO evidence_events (
                    aggregate_id, event_type, event_data, event_metadata
                ) VALUES ($1, $2, $3, $4)
            ''',
                uuid.UUID(evidence.id),
                event_type,
                json.dumps(asdict(evidence), default=str),
                json.dumps({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'version': '1.0'
                })
            )
    
    async def mark_evidence_immutable(self, evidence_id: str, block_index: int, block_hash: str):
        """Mark evidence as immutable after adding to blockchain"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                UPDATE evidence 
                SET block_index = $1, block_hash = $2, immutable = TRUE
                WHERE id = $3 AND immutable = FALSE
            ''',
                block_index,
                block_hash,
                uuid.UUID(evidence_id)
            )
            logger.info(f"Evidence marked immutable: {evidence_id} in block {block_index}")
    
    async def get_evidence_by_id(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence by ID"""
        # Check cache first
        cached = await self.redis.get(f"evidence:{evidence_id}")
        if cached:
            return json.loads(cached)
        
        # Query database
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM evidence WHERE id = $1
            ''', uuid.UUID(evidence_id))
            
            if row:
                return dict(row)
        
        return None
    
    async def get_evidence_by_resource(self, resource_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get evidence for a specific resource"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM evidence 
                WHERE resource_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            ''', resource_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_evidence_by_policy(self, policy_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get evidence for a specific policy"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM evidence 
                WHERE policy_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            ''', policy_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_compliance_summary(self, subscription_id: str) -> Dict[str, Any]:
        """Get compliance summary for a subscription"""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetch('''
                SELECT 
                    compliance_status,
                    COUNT(*) as count,
                    COUNT(DISTINCT resource_id) as resource_count,
                    COUNT(DISTINCT policy_id) as policy_count
                FROM evidence
                WHERE subscription_id = $1
                    AND timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY compliance_status
            ''', uuid.UUID(subscription_id))
            
            summary = {
                'subscription_id': subscription_id,
                'period': '30_days',
                'compliance_breakdown': {}
            }
            
            total_checks = 0
            for row in result:
                status = row['compliance_status']
                summary['compliance_breakdown'][status] = {
                    'count': row['count'],
                    'resource_count': row['resource_count'],
                    'policy_count': row['policy_count']
                }
                total_checks += row['count']
            
            summary['total_checks'] = total_checks
            
            # Calculate compliance score
            compliant = summary['compliance_breakdown'].get('Compliant', {}).get('count', 0)
            if total_checks > 0:
                summary['compliance_score'] = (compliant / total_checks) * 100
            else:
                summary['compliance_score'] = 0
            
            return summary
    
    async def verify_evidence_integrity(self, evidence_id: str) -> Dict[str, Any]:
        """Verify integrity of stored evidence"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM evidence WHERE id = $1
            ''', uuid.UUID(evidence_id))
            
            if not row:
                return {'valid': False, 'error': 'Evidence not found'}
            
            # Reconstruct evidence object
            evidence = Evidence(
                id=str(row['id']),
                timestamp=row['timestamp'],
                event_type=row['event_type'],
                resource_id=row['resource_id'],
                policy_id=row['policy_id'],
                policy_name=row['policy_name'],
                compliance_status=ComplianceStatus(row['compliance_status']),
                actor=row['actor'],
                subscription_id=str(row['subscription_id']),
                resource_group=row['resource_group'],
                resource_type=row['resource_type'],
                details=json.loads(row['details']),
                metadata=json.loads(row['metadata'])
            )
            
            # Recalculate hash
            calculated_hash = self._calculate_hash(evidence)
            stored_hash = row['hash']
            
            return {
                'valid': calculated_hash == stored_hash,
                'stored_hash': stored_hash,
                'calculated_hash': calculated_hash,
                'immutable': row['immutable'],
                'block_index': row['block_index'],
                'block_hash': row['block_hash']
            }
    
    async def close(self):
        """Close database connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis:
            await self.redis.close()


# FastAPI application
app = FastAPI(title="Evidence Collector API", version="1.0.0")
collector: Optional[EvidenceCollector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the evidence collector on startup"""
    global collector
    
    # Get connection strings from environment or use defaults
    import os
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/policycortex")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    collector = EvidenceCollector(db_url, redis_url)
    await collector.initialize()
    logger.info("Evidence collector initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global collector
    if collector:
        await collector.close()
    logger.info("Evidence collector shutdown")


@app.post("/api/v1/evidence/collect", response_model=Dict[str, Any])
async def collect_evidence(request: EvidenceRequest, background_tasks: BackgroundTasks):
    """Collect and store evidence endpoint"""
    try:
        evidence = await collector.collect_evidence(request)
        
        return {
            "success": True,
            "evidence_id": evidence.id,
            "hash": evidence.hash,
            "timestamp": evidence.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting evidence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evidence/{evidence_id}")
async def get_evidence(evidence_id: str):
    """Get evidence by ID"""
    evidence = await collector.get_evidence_by_id(evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return evidence


@app.get("/api/v1/evidence/verify/{evidence_id}")
async def verify_evidence(evidence_id: str):
    """Verify evidence integrity"""
    result = await collector.verify_evidence_integrity(evidence_id)
    return result


@app.get("/api/v1/evidence/resource/{resource_id}")
async def get_evidence_by_resource(resource_id: str, limit: int = 100):
    """Get evidence for a resource"""
    evidence = await collector.get_evidence_by_resource(resource_id, limit)
    return {"resource_id": resource_id, "count": len(evidence), "evidence": evidence}


@app.get("/api/v1/evidence/policy/{policy_id}")
async def get_evidence_by_policy(policy_id: str, limit: int = 100):
    """Get evidence for a policy"""
    evidence = await collector.get_evidence_by_policy(policy_id, limit)
    return {"policy_id": policy_id, "count": len(evidence), "evidence": evidence}


@app.get("/api/v1/evidence/compliance/summary/{subscription_id}")
async def get_compliance_summary(subscription_id: str):
    """Get compliance summary for a subscription"""
    summary = await collector.get_compliance_summary(subscription_id)
    return summary


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)