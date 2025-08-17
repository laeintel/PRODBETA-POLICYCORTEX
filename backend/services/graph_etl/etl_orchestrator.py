"""
PATENT NOTICE: Knowledge Graph ETL Orchestrator
Implements Patent #1: Cross-Domain Governance Correlation Engine
© 2024 PolicyCortex. All rights reserved.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction
from azure.identity.aio import DefaultAzureCredentialAsync
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.security import SecurityCenter
from azure.monitor.query import LogsQueryClient
from redis.asyncio import Redis
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure structured logging
logger = structlog.get_logger()

# Metrics
etl_runs = Counter('graph_etl_runs_total', 'Total ETL runs', ['status', 'source'])
etl_duration = Histogram('graph_etl_duration_seconds', 'ETL duration', ['source'])
entities_processed = Counter('graph_etl_entities_total', 'Entities processed', ['type'])
graph_nodes = Gauge('graph_nodes_total', 'Total nodes in graph', ['type'])
graph_edges = Gauge('graph_edges_total', 'Total edges in graph', ['type'])

class DataSource(Enum):
    """ETL data sources"""
    RESOURCE_GRAPH = "resource_graph"
    POLICY_INSIGHTS = "policy_insights"
    COST_MANAGEMENT = "cost_management"
    SECURITY_CENTER = "security_center"
    ACTIVITY_LOGS = "activity_logs"
    DEFENDER = "defender"

@dataclass
class ETLConfig:
    """ETL configuration"""
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    redis_url: str
    subscription_id: str
    batch_size: int = 1000
    parallel_workers: int = 5
    refresh_interval_minutes: int = 15
    checkpoint_enabled: bool = True
    incremental_mode: bool = True

@dataclass
class GraphEntity:
    """Generic graph entity"""
    id: str
    type: str
    properties: Dict[str, Any]
    tenant_id: str
    timestamp: datetime
    source: DataSource

@dataclass
class GraphRelationship:
    """Graph relationship"""
    from_id: str
    to_id: str
    type: str
    properties: Dict[str, Any]
    timestamp: datetime

class KnowledgeGraphETL:
    """Main ETL orchestrator for Knowledge Graph"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.redis_client: Optional[Redis] = None
        self.azure_credential: Optional[DefaultAzureCredentialAsync] = None
        self.checkpoint_key = "graph_etl:checkpoint"
        self.lock_key = "graph_etl:lock"
        self.lock_timeout = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize connections"""
        logger.info("Initializing Knowledge Graph ETL")
        
        # Neo4j connection
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
            max_connection_pool_size=50
        )
        
        # Redis connection
        self.redis_client = Redis.from_url(
            self.config.redis_url,
            decode_responses=True
        )
        
        # Azure credential
        self.azure_credential = DefaultAzureCredentialAsync()
        
        # Create indexes
        await self._create_graph_indexes()
        
        logger.info("ETL initialization complete")
    
    async def _create_graph_indexes(self):
        """Create Neo4j indexes for performance"""
        async with self.neo4j_driver.session() as session:
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Resource) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Resource) ON (n.type)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Resource) ON (n.tenant_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Policy) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Compliance) ON (n.resource_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Cost) ON (n.resource_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Security) ON (n.resource_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Activity) ON (n.timestamp)",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.timestamp)",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:VIOLATES]-() ON (r.severity)",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:COSTS]-() ON (r.amount)",
            ]
            
            for index in indexes:
                await session.run(index)
    
    async def acquire_lock(self) -> bool:
        """Acquire distributed lock for ETL"""
        lock_value = f"{os.getpid()}:{datetime.utcnow().isoformat()}"
        acquired = await self.redis_client.set(
            self.lock_key,
            lock_value,
            nx=True,
            ex=self.lock_timeout
        )
        return bool(acquired)
    
    async def release_lock(self):
        """Release ETL lock"""
        await self.redis_client.delete(self.lock_key)
    
    async def get_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get last checkpoint from Redis"""
        if not self.config.checkpoint_enabled:
            return None
            
        checkpoint_data = await self.redis_client.get(self.checkpoint_key)
        if checkpoint_data:
            return json.loads(checkpoint_data)
        return None
    
    async def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save checkpoint to Redis"""
        if not self.config.checkpoint_enabled:
            return
            
        await self.redis_client.set(
            self.checkpoint_key,
            json.dumps(checkpoint, default=str),
            ex=86400  # 24 hours
        )
    
    async def extract_resource_graph(self) -> List[GraphEntity]:
        """Extract from Azure Resource Graph"""
        logger.info("Extracting Resource Graph data")
        entities = []
        
        async with httpx.AsyncClient() as client:
            token = await self.azure_credential.get_token("https://management.azure.com/.default")
            
            query = """
            Resources
            | project id, name, type, kind, location, resourceGroup, 
                     subscriptionId, tenantId, tags, properties, identity, sku
            | order by id asc
            | limit 5000
            """
            
            response = await client.post(
                f"https://management.azure.com/providers/Microsoft.ResourceGraph/resources?api-version=2021-03-01",
                headers={
                    "Authorization": f"Bearer {token.token}",
                    "Content-Type": "application/json"
                },
                json={
                    "subscriptions": [self.config.subscription_id],
                    "query": query
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                for row in data.get("data", []):
                    entities.append(GraphEntity(
                        id=row["id"],
                        type="Resource",
                        properties={
                            "name": row.get("name"),
                            "resource_type": row.get("type"),
                            "location": row.get("location"),
                            "resource_group": row.get("resourceGroup"),
                            "tags": row.get("tags", {}),
                            "sku": row.get("sku"),
                        },
                        tenant_id=row.get("tenantId", ""),
                        timestamp=datetime.utcnow(),
                        source=DataSource.RESOURCE_GRAPH
                    ))
                    entities_processed.labels(type="resource").inc()
        
        logger.info(f"Extracted {len(entities)} resources")
        return entities
    
    async def extract_policy_compliance(self) -> List[GraphEntity]:
        """Extract policy compliance data"""
        logger.info("Extracting Policy Compliance data")
        entities = []
        
        async with httpx.AsyncClient() as client:
            token = await self.azure_credential.get_token("https://management.azure.com/.default")
            
            # Get policy states
            response = await client.get(
                f"https://management.azure.com/subscriptions/{self.config.subscription_id}/providers/Microsoft.PolicyInsights/policyStates/latest/summarize?api-version=2019-10-01",
                headers={"Authorization": f"Bearer {token.token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                for summary in data.get("value", []):
                    entities.append(GraphEntity(
                        id=f"compliance:{summary.get('policyAssignmentId')}",
                        type="Compliance",
                        properties={
                            "policy_assignment": summary.get("policyAssignmentId"),
                            "compliance_state": summary.get("complianceState"),
                            "non_compliant_count": summary.get("results", {}).get("nonCompliantResources", 0),
                            "compliant_count": summary.get("results", {}).get("compliantResources", 0),
                        },
                        tenant_id=self.config.subscription_id,
                        timestamp=datetime.utcnow(),
                        source=DataSource.POLICY_INSIGHTS
                    ))
                    entities_processed.labels(type="compliance").inc()
        
        logger.info(f"Extracted {len(entities)} compliance records")
        return entities
    
    async def extract_cost_data(self) -> List[GraphEntity]:
        """Extract cost management data"""
        logger.info("Extracting Cost Management data")
        entities = []
        
        async with httpx.AsyncClient() as client:
            token = await self.azure_credential.get_token("https://management.azure.com/.default")
            
            # Get cost data for last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            response = await client.post(
                f"https://management.azure.com/subscriptions/{self.config.subscription_id}/providers/Microsoft.CostManagement/query?api-version=2021-10-01",
                headers={
                    "Authorization": f"Bearer {token.token}",
                    "Content-Type": "application/json"
                },
                json={
                    "type": "Usage",
                    "timeframe": "Custom",
                    "timePeriod": {
                        "from": start_date.isoformat(),
                        "to": end_date.isoformat()
                    },
                    "dataset": {
                        "granularity": "Daily",
                        "aggregation": {
                            "totalCost": {
                                "name": "Cost",
                                "function": "Sum"
                            }
                        },
                        "grouping": [
                            {"type": "Dimension", "name": "ResourceId"},
                            {"type": "Dimension", "name": "ResourceType"}
                        ]
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                for row in data.get("properties", {}).get("rows", []):
                    if len(row) >= 3:
                        entities.append(GraphEntity(
                            id=f"cost:{row[1]}:{start_date.date()}",
                            type="Cost",
                            properties={
                                "resource_id": row[1],
                                "resource_type": row[2],
                                "cost": float(row[0]),
                                "currency": "USD",
                                "period": start_date.date().isoformat()
                            },
                            tenant_id=self.config.subscription_id,
                            timestamp=datetime.utcnow(),
                            source=DataSource.COST_MANAGEMENT
                        ))
                        entities_processed.labels(type="cost").inc()
        
        logger.info(f"Extracted {len(entities)} cost records")
        return entities
    
    async def extract_security_alerts(self) -> List[GraphEntity]:
        """Extract security center alerts"""
        logger.info("Extracting Security Center alerts")
        entities = []
        
        async with httpx.AsyncClient() as client:
            token = await self.azure_credential.get_token("https://management.azure.com/.default")
            
            response = await client.get(
                f"https://management.azure.com/subscriptions/{self.config.subscription_id}/providers/Microsoft.Security/alerts?api-version=2022-01-01",
                headers={"Authorization": f"Bearer {token.token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                for alert in data.get("value", []):
                    entities.append(GraphEntity(
                        id=alert["name"],
                        type="SecurityAlert",
                        properties={
                            "alert_type": alert.get("properties", {}).get("alertType"),
                            "severity": alert.get("properties", {}).get("severity"),
                            "status": alert.get("properties", {}).get("status"),
                            "description": alert.get("properties", {}).get("description"),
                            "resource_ids": alert.get("properties", {}).get("resourceIdentifiers", []),
                            "tactics": alert.get("properties", {}).get("tactics", []),
                            "techniques": alert.get("properties", {}).get("techniques", [])
                        },
                        tenant_id=self.config.subscription_id,
                        timestamp=datetime.fromisoformat(
                            alert.get("properties", {}).get("timeGeneratedUtc", datetime.utcnow().isoformat())
                        ),
                        source=DataSource.SECURITY_CENTER
                    ))
                    entities_processed.labels(type="security_alert").inc()
        
        logger.info(f"Extracted {len(entities)} security alerts")
        return entities
    
    async def transform_entities(self, entities: List[GraphEntity]) -> tuple[List[Dict], List[Dict]]:
        """Transform entities into nodes and relationships"""
        logger.info(f"Transforming {len(entities)} entities")
        
        nodes = []
        relationships = []
        
        # Group entities by type for relationship creation
        entities_by_type = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        # Create nodes
        for entity in entities:
            node = {
                "id": entity.id,
                "labels": [entity.type, entity.source.value],
                "properties": {
                    **entity.properties,
                    "tenant_id": entity.tenant_id,
                    "last_updated": entity.timestamp.isoformat(),
                    "source": entity.source.value
                }
            }
            nodes.append(node)
        
        # Create relationships
        # Resource -> Compliance
        if "Resource" in entities_by_type and "Compliance" in entities_by_type:
            for resource in entities_by_type["Resource"]:
                for compliance in entities_by_type["Compliance"]:
                    if resource.id in str(compliance.properties.get("policy_assignment", "")):
                        relationships.append({
                            "from_id": resource.id,
                            "to_id": compliance.id,
                            "type": "HAS_COMPLIANCE",
                            "properties": {
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        })
        
        # Resource -> Cost
        if "Resource" in entities_by_type and "Cost" in entities_by_type:
            for resource in entities_by_type["Resource"]:
                for cost in entities_by_type["Cost"]:
                    if resource.id == cost.properties.get("resource_id"):
                        relationships.append({
                            "from_id": resource.id,
                            "to_id": cost.id,
                            "type": "INCURS_COST",
                            "properties": {
                                "amount": cost.properties.get("cost", 0),
                                "currency": cost.properties.get("currency", "USD")
                            }
                        })
        
        # Resource -> Security Alert
        if "Resource" in entities_by_type and "SecurityAlert" in entities_by_type:
            for resource in entities_by_type["Resource"]:
                for alert in entities_by_type["SecurityAlert"]:
                    resource_ids = alert.properties.get("resource_ids", [])
                    if any(resource.id in str(rid) for rid in resource_ids):
                        relationships.append({
                            "from_id": alert.id,
                            "to_id": resource.id,
                            "type": "AFFECTS_RESOURCE",
                            "properties": {
                                "severity": alert.properties.get("severity"),
                                "status": alert.properties.get("status")
                            }
                        })
        
        logger.info(f"Transformed to {len(nodes)} nodes and {len(relationships)} relationships")
        return nodes, relationships
    
    async def load_to_neo4j(self, nodes: List[Dict], relationships: List[Dict]):
        """Load data into Neo4j"""
        logger.info(f"Loading {len(nodes)} nodes and {len(relationships)} relationships to Neo4j")
        
        async with self.neo4j_driver.session() as session:
            # Batch load nodes
            for i in range(0, len(nodes), self.config.batch_size):
                batch = nodes[i:i + self.config.batch_size]
                await session.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (n {id: node.id})
                    SET n += node.properties
                    WITH n, node.labels AS labels
                    FOREACH (label IN labels | 
                        SET n :`` + label
                    )
                    """,
                    nodes=batch
                )
            
            # Batch load relationships
            for i in range(0, len(relationships), self.config.batch_size):
                batch = relationships[i:i + self.config.batch_size]
                await session.run(
                    """
                    UNWIND $relationships AS rel
                    MATCH (from {id: rel.from_id})
                    MATCH (to {id: rel.to_id})
                    MERGE (from)-[r:`` + rel.type]->(to)
                    SET r += rel.properties
                    """,
                    relationships=batch
                )
            
            # Update metrics
            result = await session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count")
            async for record in result:
                for label in record["labels"]:
                    graph_nodes.labels(type=label).set(record["count"])
            
            result = await session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")
            async for record in result:
                graph_edges.labels(type=record["type"]).set(record["count"])
        
        logger.info("Data loaded to Neo4j successfully")
    
    async def run_etl_pipeline(self):
        """Run complete ETL pipeline"""
        start_time = datetime.utcnow()
        
        try:
            # Acquire lock
            if not await self.acquire_lock():
                logger.warning("Could not acquire ETL lock, another instance may be running")
                return
            
            logger.info("Starting ETL pipeline run")
            etl_runs.labels(status="started", source="all").inc()
            
            # Get checkpoint for incremental processing
            checkpoint = await self.get_checkpoint() if self.config.incremental_mode else None
            last_run = datetime.fromisoformat(checkpoint["last_run"]) if checkpoint else None
            
            # Extract from all sources in parallel
            extract_tasks = [
                self.extract_resource_graph(),
                self.extract_policy_compliance(),
                self.extract_cost_data(),
                self.extract_security_alerts()
            ]
            
            all_entities = []
            results = await asyncio.gather(*extract_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Extraction failed: {result}")
                else:
                    all_entities.extend(result)
            
            # Filter for incremental updates if checkpoint exists
            if last_run and self.config.incremental_mode:
                all_entities = [e for e in all_entities if e.timestamp > last_run]
                logger.info(f"Processing {len(all_entities)} new/updated entities since {last_run}")
            
            if all_entities:
                # Transform
                nodes, relationships = await self.transform_entities(all_entities)
                
                # Load
                await self.load_to_neo4j(nodes, relationships)
                
                # Save checkpoint
                await self.save_checkpoint({
                    "last_run": datetime.utcnow().isoformat(),
                    "entities_processed": len(all_entities),
                    "nodes_created": len(nodes),
                    "relationships_created": len(relationships)
                })
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            etl_duration.labels(source="all").observe(duration)
            etl_runs.labels(status="completed", source="all").inc()
            
            logger.info(f"ETL pipeline completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}", exc_info=True)
            etl_runs.labels(status="failed", source="all").inc()
            raise
        
        finally:
            await self.release_lock()
    
    async def run_continuous(self):
        """Run ETL continuously with configured interval"""
        logger.info(f"Starting continuous ETL with {self.config.refresh_interval_minutes} minute interval")
        
        while True:
            try:
                await self.run_etl_pipeline()
            except Exception as e:
                logger.error(f"ETL run failed: {e}")
            
            # Wait for next run
            await asyncio.sleep(self.config.refresh_interval_minutes * 60)
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.azure_credential:
            await self.azure_credential.close()


async def main():
    """Main entry point"""
    config = ETLConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        batch_size=int(os.getenv("ETL_BATCH_SIZE", "1000")),
        parallel_workers=int(os.getenv("ETL_WORKERS", "5")),
        refresh_interval_minutes=int(os.getenv("ETL_REFRESH_MINUTES", "15"))
    )
    
    etl = KnowledgeGraphETL(config)
    await etl.initialize()
    
    try:
        # Run continuous ETL
        await etl.run_continuous()
    finally:
        await etl.cleanup()


if __name__ == "__main__":
    asyncio.run(main())