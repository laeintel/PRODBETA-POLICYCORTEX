#!/usr/bin/env python3
"""
PolicyCortex Knowledge Graph ETL Service
Real-time governance data extraction, transformation, and loading
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import asyncpg
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.monitor import MonitorManagementClient
import pandas as pd
import neo4j
from gremlin_python.driver import client, serializer
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphETLProcessor:
    """Main ETL processor for knowledge graph"""
    
    def __init__(self):
        self.config = self._load_config()
        self.azure_credential = DefaultAzureCredential()
        self.resource_client = None
        self.monitor_client = None
        self.graph_client = None
        self.db_pool = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            'azure': {
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'tenant_id': os.getenv('AZURE_TENANT_ID'),
                'client_id': os.getenv('AZURE_CLIENT_ID'),
            },
            'graph': {
                'type': os.getenv('GRAPH_TYPE', 'neo4j'),  # neo4j or cosmos
                'endpoint': os.getenv('GRAPH_ENDPOINT'),
                'username': os.getenv('GRAPH_USERNAME', 'neo4j'),
                'password': os.getenv('GRAPH_PASSWORD'),
                'database': os.getenv('GRAPH_DATABASE', 'neo4j'),
            },
            'database': {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'policycortex'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', 'postgres'),
            },
            'etl': {
                'batch_size': int(os.getenv('ETL_BATCH_SIZE', 1000)),
                'parallel_workers': int(os.getenv('ETL_WORKERS', 4)),
                'schedule_interval': int(os.getenv('ETL_INTERVAL_MINUTES', 15)),
            }
        }
    
    async def initialize(self):
        """Initialize all connections"""
        logger.info("Initializing ETL processor...")
        
        # Initialize Azure clients
        self.resource_client = ResourceManagementClient(
            self.azure_credential,
            self.config['azure']['subscription_id']
        )
        self.monitor_client = MonitorManagementClient(
            self.azure_credential,
            self.config['azure']['subscription_id']
        )
        
        # Initialize graph database
        await self._init_graph_client()
        
        # Initialize PostgreSQL pool
        await self._init_db_pool()
        
        logger.info("ETL processor initialized successfully")
    
    async def _init_graph_client(self):
        """Initialize graph database client"""
        if self.config['graph']['type'] == 'neo4j':
            self.graph_client = neo4j.AsyncGraphDatabase.driver(
                self.config['graph']['endpoint'],
                auth=(
                    self.config['graph']['username'],
                    self.config['graph']['password']
                )
            )
        elif self.config['graph']['type'] == 'cosmos':
            self.graph_client = client.Client(
                self.config['graph']['endpoint'],
                'g',
                username=self.config['graph']['username'],
                password=self.config['graph']['password'],
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
        else:
            raise ValueError(f"Unsupported graph type: {self.config['graph']['type']}")
    
    async def _init_db_pool(self):
        """Initialize PostgreSQL connection pool"""
        self.db_pool = await asyncpg.create_pool(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            min_size=2,
            max_size=10
        )
    
    async def run_etl_cycle(self):
        """Run a complete ETL cycle"""
        start_time = time.time()
        logger.info("Starting ETL cycle...")
        
        try:
            # Extract data from multiple sources
            azure_data = await self._extract_azure_data()
            compliance_data = await self._extract_compliance_data()
            policy_data = await self._extract_policy_data()
            metrics_data = await self._extract_metrics_data()
            
            # Transform data into graph format
            graph_nodes, graph_edges = await self._transform_data(
                azure_data, compliance_data, policy_data, metrics_data
            )
            
            # Load data into graph database
            await self._load_graph_data(graph_nodes, graph_edges)
            
            # Update materialized views and statistics
            await self._update_views_and_stats()
            
            duration = time.time() - start_time
            logger.info(f"ETL cycle completed in {duration:.2f} seconds")
            
            # Record ETL metrics
            await self._record_etl_metrics(duration, len(graph_nodes), len(graph_edges))
            
        except Exception as e:
            logger.error(f"ETL cycle failed: {e}")
            await self._handle_etl_error(e)
    
    async def _extract_azure_data(self) -> List[Dict[str, Any]]:
        """Extract data from Azure Resource Manager"""
        logger.info("Extracting Azure resource data...")
        
        resources = []
        try:
            # Get all resources in subscription
            for resource in self.resource_client.resources.list():
                resource_data = {
                    'id': resource.id,
                    'name': resource.name,
                    'type': resource.type,
                    'location': resource.location,
                    'resource_group': resource.id.split('/')[4],
                    'tags': resource.tags or {},
                    'properties': resource.additional_properties,
                    'extracted_at': datetime.utcnow()
                }
                resources.append(resource_data)
                
                # Get resource metrics
                if hasattr(resource, 'id'):
                    metrics = await self._get_resource_metrics(resource.id)
                    resource_data['metrics'] = metrics
            
            logger.info(f"Extracted {len(resources)} Azure resources")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to extract Azure data: {e}")
            return []
    
    async def _get_resource_metrics(self, resource_id: str) -> Dict[str, Any]:
        """Get metrics for a specific resource"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            metrics = self.monitor_client.metrics.list(
                resource_uri=resource_id,
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                interval='PT1M',
                aggregation='Average'
            )
            
            metric_data = {}
            for metric in metrics.value:
                if metric.timeseries:
                    values = [point.average for point in metric.timeseries[0].data if point.average is not None]
                    if values:
                        metric_data[metric.name.value] = {
                            'average': sum(values) / len(values),
                            'max': max(values),
                            'min': min(values),
                            'count': len(values)
                        }
            
            return metric_data
            
        except Exception as e:
            logger.warning(f"Failed to get metrics for {resource_id}: {e}")
            return {}
    
    async def _extract_compliance_data(self) -> List[Dict[str, Any]]:
        """Extract compliance and policy data"""
        logger.info("Extracting compliance data...")
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT p.id, p.name, p.type, p.scope, p.rules,
                       v.resource_id, v.compliance_state, v.evaluation_time
                FROM policies p
                LEFT JOIN policy_evaluations v ON p.id = v.policy_id
                WHERE v.evaluation_time > NOW() - INTERVAL '1 hour'
            """
            rows = await conn.fetch(query)
            
            compliance_data = []
            for row in rows:
                compliance_data.append({
                    'policy_id': row['id'],
                    'policy_name': row['name'],
                    'policy_type': row['type'],
                    'scope': row['scope'],
                    'rules': row['rules'],
                    'resource_id': row['resource_id'],
                    'compliance_state': row['compliance_state'],
                    'evaluation_time': row['evaluation_time'],
                    'extracted_at': datetime.utcnow()
                })
            
            logger.info(f"Extracted {len(compliance_data)} compliance records")
            return compliance_data
    
    async def _extract_policy_data(self) -> List[Dict[str, Any]]:
        """Extract policy definitions and assignments"""
        logger.info("Extracting policy data...")
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT id, name, display_name, description, policy_type,
                       mode, parameters, policy_rule, metadata
                FROM policy_definitions
                WHERE updated_at > NOW() - INTERVAL '1 hour'
            """
            rows = await conn.fetch(query)
            
            policy_data = []
            for row in rows:
                policy_data.append({
                    'id': row['id'],
                    'name': row['name'],
                    'display_name': row['display_name'],
                    'description': row['description'],
                    'type': row['policy_type'],
                    'mode': row['mode'],
                    'parameters': row['parameters'],
                    'rule': row['policy_rule'],
                    'metadata': row['metadata'],
                    'extracted_at': datetime.utcnow()
                })
            
            logger.info(f"Extracted {len(policy_data)} policy definitions")
            return policy_data
    
    async def _extract_metrics_data(self) -> List[Dict[str, Any]]:
        """Extract governance metrics"""
        logger.info("Extracting metrics data...")
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT resource_id, metric_name, metric_value, 
                       timestamp, tenant_id, domain
                FROM governance_metrics
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """
            rows = await conn.fetch(query)
            
            metrics_data = []
            for row in rows:
                metrics_data.append({
                    'resource_id': row['resource_id'],
                    'metric_name': row['metric_name'],
                    'metric_value': row['metric_value'],
                    'timestamp': row['timestamp'],
                    'tenant_id': row['tenant_id'],
                    'domain': row['domain'],
                    'extracted_at': datetime.utcnow()
                })
            
            logger.info(f"Extracted {len(metrics_data)} metric records")
            return metrics_data
    
    async def _transform_data(self, azure_data: List[Dict], compliance_data: List[Dict],
                            policy_data: List[Dict], metrics_data: List[Dict]) -> tuple:
        """Transform extracted data into graph nodes and edges"""
        logger.info("Transforming data for graph loading...")
        
        nodes = []
        edges = []
        
        # Transform Azure resources into nodes
        for resource in azure_data:
            node = {
                'id': f"resource:{resource['id']}",
                'labels': ['Resource', 'AzureResource'],
                'properties': {
                    'name': resource['name'],
                    'type': resource['type'],
                    'location': resource['location'],
                    'resource_group': resource['resource_group'],
                    'tags': resource['tags'],
                    'extracted_at': resource['extracted_at'].isoformat(),
                    'metrics': resource.get('metrics', {})
                }
            }
            nodes.append(node)
        
        # Transform policies into nodes
        for policy in policy_data:
            node = {
                'id': f"policy:{policy['id']}",
                'labels': ['Policy', 'Governance'],
                'properties': {
                    'name': policy['name'],
                    'display_name': policy['display_name'],
                    'description': policy['description'],
                    'type': policy['type'],
                    'mode': policy['mode'],
                    'parameters': policy['parameters'],
                    'rule': policy['rule'],
                    'extracted_at': policy['extracted_at'].isoformat()
                }
            }
            nodes.append(node)
        
        # Create compliance edges
        for compliance in compliance_data:
            edge = {
                'id': f"compliance:{compliance['policy_id']}:{compliance['resource_id']}",
                'from_id': f"policy:{compliance['policy_id']}",
                'to_id': f"resource:{compliance['resource_id']}",
                'label': 'EVALUATES',
                'properties': {
                    'compliance_state': compliance['compliance_state'],
                    'evaluation_time': compliance['evaluation_time'].isoformat(),
                    'extracted_at': compliance['extracted_at'].isoformat()
                },
                'weight': 1.0 if compliance['compliance_state'] == 'Compliant' else 0.5
            }
            edges.append(edge)
        
        # Create metric relationships
        resource_metrics = {}
        for metric in metrics_data:
            resource_id = f"resource:{metric['resource_id']}"
            if resource_id not in resource_metrics:
                resource_metrics[resource_id] = []
            resource_metrics[resource_id].append(metric)
        
        # Add metric aggregation edges
        for resource_id, metrics in resource_metrics.items():
            if len(metrics) > 1:
                for i in range(len(metrics) - 1):
                    edge = {
                        'id': f"metric_flow:{metrics[i]['resource_id']}:{i}",
                        'from_id': resource_id,
                        'to_id': resource_id,
                        'label': 'METRIC_CORRELATION',
                        'properties': {
                            'correlation_type': 'temporal',
                            'strength': 0.8,
                            'extracted_at': datetime.utcnow().isoformat()
                        },
                        'weight': 0.8
                    }
                    edges.append(edge)
        
        logger.info(f"Transformed to {len(nodes)} nodes and {len(edges)} edges")
        return nodes, edges
    
    async def _load_graph_data(self, nodes: List[Dict], edges: List[Dict]):
        """Load transformed data into graph database"""
        logger.info("Loading data into graph database...")
        
        if self.config['graph']['type'] == 'neo4j':
            await self._load_neo4j_data(nodes, edges)
        elif self.config['graph']['type'] == 'cosmos':
            await self._load_cosmos_data(nodes, edges)
    
    async def _load_neo4j_data(self, nodes: List[Dict], edges: List[Dict]):
        """Load data into Neo4j"""
        async with self.graph_client.session() as session:
            # Batch insert nodes
            for i in range(0, len(nodes), self.config['etl']['batch_size']):
                batch = nodes[i:i + self.config['etl']['batch_size']]
                await session.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (n {id: node.id})
                    SET n += node.properties
                    SET n:Resource:AzureResource
                    """,
                    nodes=batch
                )
            
            # Batch insert edges
            for i in range(0, len(edges), self.config['etl']['batch_size']):
                batch = edges[i:i + self.config['etl']['batch_size']]
                await session.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (from {id: edge.from_id})
                    MATCH (to {id: edge.to_id})
                    MERGE (from)-[r:EVALUATES {id: edge.id}]->(to)
                    SET r += edge.properties
                    """,
                    edges=batch
                )
        
        logger.info("Data loaded into Neo4j successfully")
    
    async def _load_cosmos_data(self, nodes: List[Dict], edges: List[Dict]):
        """Load data into Cosmos DB Gremlin"""
        # Implementation for Cosmos DB Gremlin API
        for node in nodes:
            query = f"g.addV('{node['labels'][0]}').property('id', '{node['id']}')"
            for key, value in node['properties'].items():
                query += f".property('{key}', '{value}')"
            
            self.graph_client.submit(query)
        
        for edge in edges:
            query = f"g.V().has('id', '{edge['from_id']}').addE('{edge['label']}').to(g.V().has('id', '{edge['to_id']}'))"
            for key, value in edge['properties'].items():
                query += f".property('{key}', '{value}')"
            
            self.graph_client.submit(query)
        
        logger.info("Data loaded into Cosmos DB successfully")
    
    async def _update_views_and_stats(self):
        """Update materialized views and graph statistics"""
        logger.info("Updating views and statistics...")
        
        async with self.db_pool.acquire() as conn:
            # Refresh materialized views
            await conn.execute("REFRESH MATERIALIZED VIEW governance_summary;")
            await conn.execute("REFRESH MATERIALIZED VIEW compliance_trends;")
            await conn.execute("REFRESH MATERIALIZED VIEW resource_dependencies;")
            
            # Update graph statistics
            if self.config['graph']['type'] == 'neo4j':
                async with self.graph_client.session() as session:
                    await session.run("CALL db.stats.retrieve('GRAPH')")
    
    async def _record_etl_metrics(self, duration: float, nodes_count: int, edges_count: int):
        """Record ETL performance metrics"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO etl_metrics (timestamp, duration_seconds, nodes_processed, edges_processed, status)
                VALUES ($1, $2, $3, $4, $5)
                """,
                datetime.utcnow(), duration, nodes_count, edges_count, 'success'
            )
    
    async def _handle_etl_error(self, error: Exception):
        """Handle ETL errors"""
        logger.error(f"ETL error: {error}")
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO etl_metrics (timestamp, duration_seconds, nodes_processed, edges_processed, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                datetime.utcnow(), 0, 0, 0, 'error', str(error)
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.graph_client:
            await self.graph_client.close()
        if self.db_pool:
            await self.db_pool.close()

async def main():
    """Main ETL service entry point"""
    processor = GraphETLProcessor()
    
    try:
        await processor.initialize()
        
        # Run ETL cycle immediately
        await processor.run_etl_cycle()
        
        # Schedule periodic runs
        schedule.every(processor.config['etl']['schedule_interval']).minutes.do(
            lambda: asyncio.create_task(processor.run_etl_cycle())
        )
        
        logger.info(f"ETL service started, running every {processor.config['etl']['schedule_interval']} minutes")
        
        # Keep service running
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("ETL service stopping...")
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())