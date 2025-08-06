"""
Data lineage tracking service for PolicyCortex.
"""

import json
import uuid
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog
from sqlalchemy import text

from ....shared.config import get_settings
from ....shared.database import get_async_db
from ..models import DataSourceConfig
from ..models import DataTargetConfig
from ..models import LineageEdge
from ..models import LineageNode

settings = get_settings()
logger = structlog.get_logger(__name__)


class LineageTrackerService:
    """Service for tracking data lineage."""

    def __init__(self):
        self.settings = settings
        self.lineage_cache = {}

    async def track_pipeline_creation(self, pipeline_id: str, source_config: DataSourceConfig,
                                    target_config: DataTargetConfig, user_id: Optional[str] = None) -> None:
        """Track ETL pipeline creation."""
        try:
            # Create source node
            source_node = LineageNode(
                entity_id=f"source_{pipeline_id}",
                entity_type="data_source",
                name = (
                    f"{source_config.source_type}_{source_config.database or source_config.container}",
                )
                description=f"Source: {source_config.source_type}",
                metadata={
                    "source_type": source_config.source_type,
                    "server": source_config.server,
                    "database": source_config.database,
                    "table": source_config.table,
                    "container": source_config.container
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Create target node
            target_node = LineageNode(
                entity_id=f"target_{pipeline_id}",
                entity_type="data_target",
                name = (
                    f"{target_config.target_type}_{target_config.database or target_config.container}",
                )
                description=f"Target: {target_config.target_type}",
                metadata={
                    "target_type": target_config.target_type,
                    "server": target_config.server,
                    "database": target_config.database,
                    "table": target_config.table,
                    "container": target_config.container
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Create pipeline node
            pipeline_node = LineageNode(
                entity_id=pipeline_id,
                entity_type="etl_pipeline",
                name=f"Pipeline_{pipeline_id}",
                description="ETL Pipeline",
                metadata={
                    "pipeline_type": "etl",
                    "created_by": user_id
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Create relationships
            source_to_pipeline = LineageEdge(
                source_id=source_node.entity_id,
                target_id=pipeline_node.entity_id,
                relationship_type="feeds_into",
                created_at=datetime.utcnow()
            )

            pipeline_to_target = LineageEdge(
                source_id=pipeline_node.entity_id,
                target_id=target_node.entity_id,
                relationship_type="outputs_to",
                created_at=datetime.utcnow()
            )

            # Save to database
            await self._save_lineage_nodes([source_node, target_node, pipeline_node])
            await self._save_lineage_edges([source_to_pipeline, pipeline_to_target])

            logger.info("pipeline_lineage_tracked", pipeline_id=pipeline_id)

        except Exception as e:
            logger.error("track_pipeline_creation_failed", error=str(e))

    async def track_stream_processor_creation(self, processor_id: str, source_config: DataSourceConfig,
                                           output_config: DataTargetConfig, user_id: Optional[str] = None) -> None:
        """Track stream processor creation."""
        try:
            # Similar to pipeline creation but for stream processors
            await self.track_pipeline_creation(processor_id, source_config, output_config, user_id)

        except Exception as e:
            logger.error("track_stream_processor_creation_failed", error=str(e))

    async def track_data_transformation(self, transformation_id: str, input_data: Any,
                                      rules: List[Dict[str, Any]], output_data: Any,
                                      user_id: Optional[str] = None) -> None:
        """Track data transformation."""
        try:
            # Create transformation node
            transformation_node = LineageNode(
                entity_id=transformation_id,
                entity_type="data_transformation",
                name=f"Transformation_{transformation_id}",
                description="Data Transformation",
                metadata={
                    "transformation_type": "data_transformation",
                    "rules_count": len(rules),
                    "created_by": user_id
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database
            await self._save_lineage_nodes([transformation_node])

            logger.info("data_transformation_lineage_tracked", transformation_id=transformation_id)

        except Exception as e:
            logger.error("track_data_transformation_failed", error=str(e))

    async def track_data_validation(self, validation_id: str, data: Any, rules: List[Dict[str, Any]],
                                  results: List[Dict[str, Any]], user_id: Optional[str] = None) -> None:
        """Track data validation."""
        try:
            # Create validation node
            validation_node = LineageNode(
                entity_id=validation_id,
                entity_type="data_validation",
                name=f"Validation_{validation_id}",
                description="Data Validation",
                metadata={
                    "validation_type": "data_validation",
                    "rules_count": len(rules),
                    "results_count": len(results),
                    "created_by": user_id
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database
            await self._save_lineage_nodes([validation_node])

            logger.info("data_validation_lineage_tracked", validation_id=validation_id)

        except Exception as e:
            logger.error("track_data_validation_failed", error=str(e))

    async def track_data_aggregation(self, aggregation_id: str, input_data: Any,
                                   rules: List[Dict[str, Any]], output_data: Any,
                                   user_id: Optional[str] = None) -> None:
        """Track data aggregation."""
        try:
            # Create aggregation node
            aggregation_node = LineageNode(
                entity_id=aggregation_id,
                entity_type="data_aggregation",
                name=f"Aggregation_{aggregation_id}",
                description="Data Aggregation",
                metadata={
                    "aggregation_type": "data_aggregation",
                    "rules_count": len(rules),
                    "created_by": user_id
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database
            await self._save_lineage_nodes([aggregation_node])

            logger.info("data_aggregation_lineage_tracked", aggregation_id=aggregation_id)

        except Exception as e:
            logger.error("track_data_aggregation_failed", error=str(e))

    async def track_data_export(self, export_id: str, source_config: DataSourceConfig,
                              destination_config: DataTargetConfig, user_id: Optional[str] = None) -> None:
        """Track data export."""
        try:
            # Create export node
            export_node = LineageNode(
                entity_id=export_id,
                entity_type="data_export",
                name=f"Export_{export_id}",
                description="Data Export",
                metadata={
                    "export_type": "data_export",
                    "source_type": source_config.source_type,
                    "destination_type": destination_config.target_type,
                    "created_by": user_id
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database
            await self._save_lineage_nodes([export_node])

            logger.info("data_export_lineage_tracked", export_id=export_id)

        except Exception as e:
            logger.error("track_data_export_failed", error=str(e))

    async def get_lineage(self, entity_id: str, entity_type: str, depth: int = 3) -> Dict[str, Any]:
        """Get data lineage for an entity."""
        try:
            # Check cache first
            cache_key = f"{entity_id}_{entity_type}_{depth}"
            if cache_key in self.lineage_cache:
                return self.lineage_cache[cache_key]

            # Get lineage from database
            lineage_graph = await self._build_lineage_graph(entity_id, entity_type, depth)

            # Cache result
            self.lineage_cache[cache_key] = lineage_graph

            return lineage_graph

        except Exception as e:
            logger.error("get_lineage_failed", error=str(e))
            raise

    async def _build_lineage_graph(
        self,
        entity_id: str,
        entity_type: str,
        depth: int
    ) -> Dict[str, Any]:
        """Build lineage graph from database."""
        try:
            db = await get_async_db()

            # Get nodes
            nodes_result = await db.execute(
                text("SELECT * FROM lineage_nodes WHERE entity_id = :entity_id"),
                {"entity_id": entity_id}
            )
            nodes = nodes_result.fetchall()

            # Get upstream dependencies
            upstream_result = await db.execute(
                text("""
                    SELECT ln.* FROM lineage_nodes ln
                    JOIN lineage_edges le ON ln.entity_id = le.source_id
                    WHERE le.target_id = :entity_id
                """),
                {"entity_id": entity_id}
            )
            upstream_nodes = upstream_result.fetchall()

            # Get downstream dependencies
            downstream_result = await db.execute(
                text("""
                    SELECT ln.* FROM lineage_nodes ln
                    JOIN lineage_edges le ON ln.entity_id = le.target_id
                    WHERE le.source_id = :entity_id
                """),
                {"entity_id": entity_id}
            )
            downstream_nodes = downstream_result.fetchall()

            # Build graph structure
            lineage_graph = {
                "nodes": [self._node_to_dict(node) for node in nodes],
                "edges": []
            }

            upstream_dependencies = [self._node_to_dict(node) for node in upstream_nodes]
            downstream_dependencies = [self._node_to_dict(node) for node in downstream_nodes]

            return {
                "lineage_graph": lineage_graph,
                "upstream_dependencies": upstream_dependencies,
                "downstream_dependencies": downstream_dependencies
            }

        except Exception as e:
            logger.error("build_lineage_graph_failed", error=str(e))
            raise

    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Convert database node to dictionary."""
        return {
            "entity_id": node.entity_id,
            "entity_type": node.entity_type,
            "name": node.name,
            "description": node.description,
            "metadata": json.loads(node.metadata) if node.metadata else {},
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None
        }

    async def _save_lineage_nodes(self, nodes: List[LineageNode]) -> None:
        """Save lineage nodes to database."""
        try:
            db = await get_async_db()

            for node in nodes:
                await db.execute(
                    text("""
                        INSERT OR REPLACE INTO lineage_nodes (
                            entity_id, entity_type, name, description, metadata, created_at, updated_at
                        ) VALUES (
                            :entity_id, :entity_type, :name, :description, :metadata, :created_at, :updated_at
                        )
                    """),
                    {
                        "entity_id": node.entity_id,
                        "entity_type": node.entity_type,
                        "name": node.name,
                        "description": node.description,
                        "metadata": json.dumps(node.metadata) if node.metadata else None,
                        "created_at": node.created_at.isoformat(),
                        "updated_at": node.updated_at.isoformat()
                    }
                )

            await db.commit()

        except Exception as e:
            logger.error("save_lineage_nodes_failed", error=str(e))
            raise

    async def _save_lineage_edges(self, edges: List[LineageEdge]) -> None:
        """Save lineage edges to database."""
        try:
            db = await get_async_db()

            for edge in edges:
                await db.execute(
                    text("""
                        INSERT OR REPLACE INTO lineage_edges (
                            source_id, target_id, relationship_type, transformation_info, created_at
                        ) VALUES (
                            :source_id, :target_id, :relationship_type, :transformation_info, :created_at
                        )
                    """),
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relationship_type": edge.relationship_type,
                        "transformation_info": json.dumps(edge.transformation_info) if edge.transformation_info else None,
                        "created_at": edge.created_at.isoformat()
                    }
                )

            await db.commit()

        except Exception as e:
            logger.error("save_lineage_edges_failed", error=str(e))
            raise
