#!/usr/bin/env python3
"""
Knowledge Graph ETL Processor
Ensures 15-minute data freshness for cross-domain correlation
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yaml

import aiohttp
import redis.asyncio as redis
from azure.identity.aio import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.costmanagement import CostManagementClient
from neo4j import AsyncGraphDatabase
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import structlog

# Configure structured logging
logger = structlog.get_logger()

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")))
)

meter = metrics.get_meter(__name__)
etl_lag_metric = meter.create_histogram(
    name="etl_lag_seconds",
    description="ETL processing lag in seconds",
    unit="s"
)
records_processed_metric = meter.create_counter(
    name="etl_records_processed",
    description="Number of records processed"
)


class KnowledgeGraphETL:
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.batch_size = int(os.getenv("BATCH_SIZE", "1000"))
        self.checkpoint_enabled = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
        
        self.credential = None
        self.neo4j_driver = None
        self.redis_client = None
        self.config = None
        self.last_run_timestamp = None
        
    async def initialize(self):
        """Initialize connections and load configuration"""
        logger.info("Initializing Knowledge Graph ETL processor")
        
        # Load ETL configuration
        with open("/app/config/etl-config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Azure credential
        self.credential = DefaultAzureCredential()
        
        # Initialize Neo4j connection
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize Redis for checkpointing
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Load last checkpoint
        if self.checkpoint_enabled:
            checkpoint = await self.redis_client.get("etl:last_run")
            if checkpoint:
                self.last_run_timestamp = datetime.fromisoformat(checkpoint.decode())
                logger.info(f"Loaded checkpoint: {self.last_run_timestamp}")
            else:
                # Default to 15 minutes ago for first run
                self.last_run_timestamp = datetime.utcnow() - timedelta(minutes=15)
    
    async def extract_azure_resources(self) -> List[Dict[str, Any]]:
        """Extract resources from Azure Resource Graph"""
        with tracer.start_as_current_span("extract_azure_resources"):
            resources = []
            
            async with aiohttp.ClientSession() as session:
                token = await self.credential.get_token("https://management.azure.com/.default")
                headers = {"Authorization": f"Bearer {token.token}"}
                
                # Resource Graph query
                query = self.config["sources"][0]["query"].replace(
                    "${AZURE_SUBSCRIPTION_ID}", 
                    self.subscription_id
                )
                
                url = "https://management.azure.com/providers/Microsoft.ResourceGraph/resources"
                params = {"api-version": "2021-03-01"}
                body = {
                    "subscriptions": [self.subscription_id],
                    "query": query,
                    "options": {"$top": self.batch_size}
                }
                
                async with session.post(url, headers=headers, params=params, json=body) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        resources = data.get("data", [])
                        logger.info(f"Extracted {len(resources)} resources from Azure")
                    else:
                        logger.error(f"Failed to extract resources: {resp.status}")
            
            return resources
    
    async def extract_policy_compliance(self) -> List[Dict[str, Any]]:
        """Extract policy compliance data"""
        with tracer.start_as_current_span("extract_policy_compliance"):
            compliance_data = []
            
            async with aiohttp.ClientSession() as session:
                token = await self.credential.get_token("https://management.azure.com/.default")
                headers = {"Authorization": f"Bearer {token.token}"}
                
                url = f"https://management.azure.com/subscriptions/{self.subscription_id}"
                url += "/providers/Microsoft.PolicyInsights/policyStates/latest/summarize"
                params = {"api-version": "2019-10-01"}
                
                # Filter for changes since last run
                if self.last_run_timestamp:
                    params["$filter"] = f"timestamp ge '{self.last_run_timestamp.isoformat()}Z'"
                
                async with session.post(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        compliance_data = data.get("value", [])
                        logger.info(f"Extracted {len(compliance_data)} compliance records")
                    else:
                        logger.error(f"Failed to extract compliance data: {resp.status}")
            
            return compliance_data
    
    async def extract_cost_data(self) -> List[Dict[str, Any]]:
        """Extract cost management data"""
        with tracer.start_as_current_span("extract_cost_data"):
            cost_data = []
            
            async with aiohttp.ClientSession() as session:
                token = await self.credential.get_token("https://management.azure.com/.default")
                headers = {"Authorization": f"Bearer {token.token}"}
                
                url = f"https://management.azure.com/subscriptions/{self.subscription_id}"
                url += "/providers/Microsoft.CostManagement/query"
                params = {"api-version": "2021-10-01"}
                
                # Query for last 7 days of cost data
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=7)
                
                body = {
                    "type": "Usage",
                    "timeframe": "Custom",
                    "timePeriod": {
                        "from": start_date.strftime("%Y-%m-%d"),
                        "to": end_date.strftime("%Y-%m-%d")
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
                
                async with session.post(url, headers=headers, params=params, json=body) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cost_data = data.get("properties", {}).get("rows", [])
                        logger.info(f"Extracted {len(cost_data)} cost records")
                    else:
                        logger.error(f"Failed to extract cost data: {resp.status}")
            
            return cost_data
    
    async def transform_data(self, resources: List, compliance: List, costs: List) -> Dict[str, List]:
        """Transform and enrich extracted data"""
        with tracer.start_as_current_span("transform_data"):
            transformed = {
                "nodes": [],
                "relationships": []
            }
            
            # Create resource nodes with enriched properties
            resource_map = {}
            for resource in resources:
                node = {
                    "label": "Resource",
                    "id": resource.get("id"),
                    "properties": {
                        "name": resource.get("name"),
                        "type": resource.get("type"),
                        "location": resource.get("location"),
                        "subscription": self.subscription_id,
                        "tags": json.dumps(resource.get("tags", {})),
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }
                transformed["nodes"].append(node)
                resource_map[resource["id"]] = node
            
            # Add compliance relationships
            for comp in compliance:
                if comp.get("resourceId") in resource_map:
                    policy_node = {
                        "label": "Policy",
                        "id": comp.get("policyAssignmentId"),
                        "properties": {
                            "name": comp.get("policyAssignmentName"),
                            "definition": comp.get("policyDefinitionId"),
                            "compliance_state": comp.get("complianceState")
                        }
                    }
                    transformed["nodes"].append(policy_node)
                    
                    # Create VIOLATES relationship if non-compliant
                    if comp.get("complianceState") != "Compliant":
                        transformed["relationships"].append({
                            "type": "VIOLATES",
                            "from_id": comp.get("resourceId"),
                            "to_id": comp.get("policyAssignmentId"),
                            "properties": {
                                "timestamp": comp.get("timestamp"),
                                "state": comp.get("complianceState")
                            }
                        })
            
            # Add cost relationships
            for cost_row in costs:
                if len(cost_row) >= 3:
                    resource_id = cost_row[0]
                    cost_amount = cost_row[2]
                    
                    if resource_id in resource_map:
                        cost_node = {
                            "label": "CostCenter",
                            "id": f"cost_{resource_id}",
                            "properties": {
                                "resource_id": resource_id,
                                "amount": cost_amount,
                                "currency": "USD",
                                "period": "daily"
                            }
                        }
                        transformed["nodes"].append(cost_node)
                        
                        transformed["relationships"].append({
                            "type": "INCURS_COST",
                            "from_id": resource_id,
                            "to_id": f"cost_{resource_id}",
                            "properties": {
                                "amount": cost_amount,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        })
            
            logger.info(f"Transformed {len(transformed['nodes'])} nodes and {len(transformed['relationships'])} relationships")
            return transformed
    
    async def load_to_neo4j(self, data: Dict[str, List]):
        """Load transformed data to Neo4j"""
        with tracer.start_as_current_span("load_to_neo4j"):
            async with self.neo4j_driver.session() as session:
                # Create nodes
                for node in data["nodes"]:
                    query = f"""
                    MERGE (n:{node['label']} {{id: $id}})
                    SET n += $properties
                    """
                    await session.run(query, id=node["id"], properties=node["properties"])
                
                # Create relationships
                for rel in data["relationships"]:
                    query = f"""
                    MATCH (a {{id: $from_id}})
                    MATCH (b {{id: $to_id}})
                    MERGE (a)-[r:{rel['type']}]->(b)
                    SET r += $properties
                    """
                    await session.run(
                        query,
                        from_id=rel["from_id"],
                        to_id=rel["to_id"],
                        properties=rel["properties"]
                    )
                
                logger.info(f"Loaded {len(data['nodes'])} nodes and {len(data['relationships'])} relationships to Neo4j")
                records_processed_metric.add(len(data["nodes"]))
    
    async def save_checkpoint(self):
        """Save ETL checkpoint"""
        if self.checkpoint_enabled:
            timestamp = datetime.utcnow().isoformat()
            await self.redis_client.set("etl:last_run", timestamp, ex=86400)  # 24h expiry
            logger.info(f"Saved checkpoint: {timestamp}")
    
    async def calculate_metrics(self):
        """Calculate and export ETL metrics"""
        if self.last_run_timestamp:
            lag = (datetime.utcnow() - self.last_run_timestamp).total_seconds()
            etl_lag_metric.record(lag)
            
            if lag > 900:  # Alert if lag > 15 minutes
                logger.warning(f"ETL lag exceeds 15 minutes: {lag}s")
    
    async def run(self):
        """Main ETL pipeline execution"""
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Extract phase
            logger.info("Starting extraction phase")
            resources, compliance, costs = await asyncio.gather(
                self.extract_azure_resources(),
                self.extract_policy_compliance(),
                self.extract_cost_data()
            )
            
            # Transform phase
            logger.info("Starting transformation phase")
            transformed_data = await self.transform_data(resources, compliance, costs)
            
            # Load phase
            logger.info("Starting load phase")
            await self.load_to_neo4j(transformed_data)
            
            # Save checkpoint
            await self.save_checkpoint()
            
            # Calculate metrics
            await self.calculate_metrics()
            
            elapsed = time.time() - start_time
            logger.info(f"ETL pipeline completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.credential:
                await self.credential.close()


if __name__ == "__main__":
    etl = KnowledgeGraphETL()
    asyncio.run(etl.run())