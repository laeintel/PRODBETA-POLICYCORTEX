"""
Data export service for PolicyCortex.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
from sqlalchemy import text

from ....shared.config import get_settings
from ....shared.database import get_async_db
from ..models import DataSourceConfig, DataTargetConfig, DataFormat, PipelineStatus
from .azure_connectors import AzureConnectorService

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataExporterService:
    """Service for data export operations."""
    
    def __init__(self):
        self.settings = settings
        self.azure_connector = AzureConnectorService()
        self.export_jobs = {}
    
    async def create_export_job(self, source_config: DataSourceConfig, destination_config: DataTargetConfig,
                              export_format: DataFormat, filters: Optional[Dict[str, Any]] = None,
                              user_id: Optional[str] = None) -> str:
        """Create a new export job."""
        try:
            export_id = str(uuid.uuid4())
            
            # Store export job configuration
            export_config = {
                "export_id": export_id,
                "source_config": source_config.dict(),
                "destination_config": destination_config.dict(),
                "export_format": export_format.value,
                "filters": filters or {},
                "status": PipelineStatus.CREATED.value,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": user_id
            }
            
            # Save to database
            db = await get_async_db()
            await db.execute(
                text("""
                    INSERT INTO data_exports (
                        export_id, source_config, destination_config, export_format,
                        filters, status, created_at, created_by
                    ) VALUES (
                        :export_id, :source_config, :destination_config, :export_format,
                        :filters, :status, :created_at, :created_by
                    )
                """),
                {
                    "export_id": export_id,
                    "source_config": json.dumps(export_config["source_config"]),
                    "destination_config": json.dumps(export_config["destination_config"]),
                    "export_format": export_config["export_format"],
                    "filters": json.dumps(export_config["filters"]),
                    "status": export_config["status"],
                    "created_at": export_config["created_at"],
                    "created_by": export_config["created_by"]
                }
            )
            await db.commit()
            
            self.export_jobs[export_id] = export_config
            
            logger.info("export_job_created", export_id=export_id)
            
            return export_id
            
        except Exception as e:
            logger.error("create_export_job_failed", error=str(e))
            raise
    
    async def execute_export(self, export_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute an export job."""
        try:
            # Get export configuration
            export_config = await self._get_export_config(export_id)
            
            if not export_config:
                raise ValueError(f"Export job {export_id} not found")
            
            # Update status
            await self._update_export_status(export_id, PipelineStatus.RUNNING)
            
            start_time = datetime.utcnow()
            
            # Parse configurations
            source_config = DataSourceConfig(**export_config["source_config"])
            destination_config = DataTargetConfig(**export_config["destination_config"])
            export_format = DataFormat(export_config["export_format"])
            filters = export_config.get("filters", {})
            
            # Read data from source
            logger.info("reading_data_for_export", export_id=export_id)
            data = await self.azure_connector.read_data(source_config)
            
            # Apply filters if any
            if filters:
                data = self._apply_export_filters(data, filters)
            
            # Set export format in destination config
            destination_config.additional_config = destination_config.additional_config or {}
            destination_config.additional_config["file_format"] = export_format.value
            
            # Export data
            logger.info("exporting_data", export_id=export_id, records=len(data))
            result = await self.azure_connector.write_data(destination_config, data, "overwrite")
            
            # Update export job
            completion_time = datetime.utcnow()
            processing_time = (completion_time - start_time).total_seconds()
            
            await self._update_export_completion(
                export_id, 
                PipelineStatus.COMPLETED, 
                len(data), 
                result.get("file_size", 0),
                processing_time
            )
            
            logger.info(
                "export_job_completed",
                export_id=export_id,
                records_exported=len(data),
                processing_time=processing_time
            )
            
            return {
                "export_id": export_id,
                "status": "completed",
                "records_exported": len(data),
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            logger.error("execute_export_failed", error=str(e))
            await self._update_export_status(export_id, PipelineStatus.FAILED)
            raise
    
    def _apply_export_filters(self, data, filters: Dict[str, Any]):
        """Apply filters to export data."""
        try:
            # Simple filtering implementation
            filtered_data = data.copy()
            
            for column, filter_value in filters.items():
                if column in filtered_data.columns:
                    if isinstance(filter_value, dict):
                        # Complex filter
                        operator = filter_value.get("operator", "eq")
                        value = filter_value.get("value")
                        
                        if operator == "eq":
                            filtered_data = filtered_data[filtered_data[column] == value]
                        elif operator == "ne":
                            filtered_data = filtered_data[filtered_data[column] != value]
                        elif operator == "gt":
                            filtered_data = filtered_data[filtered_data[column] > value]
                        elif operator == "lt":
                            filtered_data = filtered_data[filtered_data[column] < value]
                        elif operator == "in":
                            filtered_data = filtered_data[filtered_data[column].isin(value)]
                    else:
                        # Simple equality filter
                        filtered_data = filtered_data[filtered_data[column] == filter_value]
            
            return filtered_data
            
        except Exception as e:
            logger.error("apply_export_filters_failed", error=str(e))
            return data
    
    async def _get_export_config(self, export_id: str) -> Optional[Dict[str, Any]]:
        """Get export configuration."""
        try:
            db = await get_async_db()
            result = await db.execute(
                text("SELECT * FROM data_exports WHERE export_id = :export_id"),
                {"export_id": export_id}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "export_id": row.export_id,
                    "source_config": json.loads(row.source_config),
                    "destination_config": json.loads(row.destination_config),
                    "export_format": row.export_format,
                    "filters": json.loads(row.filters),
                    "status": row.status,
                    "created_at": row.created_at,
                    "created_by": row.created_by
                }
            
            return None
            
        except Exception as e:
            logger.error("get_export_config_failed", error=str(e))
            return None
    
    async def _update_export_status(self, export_id: str, status: PipelineStatus) -> None:
        """Update export status."""
        try:
            db = await get_async_db()
            await db.execute(
                text("""
                    UPDATE data_exports 
                    SET status = :status, updated_at = :updated_at
                    WHERE export_id = :export_id
                """),
                {
                    "export_id": export_id,
                    "status": status.value,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            await db.commit()
            
        except Exception as e:
            logger.error("update_export_status_failed", error=str(e))
    
    async def _update_export_completion(self, export_id: str, status: PipelineStatus,
                                      record_count: int, file_size: int, processing_time: float) -> None:
        """Update export completion details."""
        try:
            db = await get_async_db()
            await db.execute(
                text("""
                    UPDATE data_exports 
                    SET status = :status, record_count = :record_count, file_size = :file_size,
                        processing_time = :processing_time, completed_at = :completed_at, updated_at = :updated_at
                    WHERE export_id = :export_id
                """),
                {
                    "export_id": export_id,
                    "status": status.value,
                    "record_count": record_count,
                    "file_size": file_size,
                    "processing_time": processing_time,
                    "completed_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            await db.commit()
            
        except Exception as e:
            logger.error("update_export_completion_failed", error=str(e))
    
    async def get_export_info(self, export_id: str) -> Dict[str, Any]:
        """Get export information."""
        try:
            export_config = await self._get_export_config(export_id)
            if not export_config:
                raise ValueError(f"Export job {export_id} not found")
            
            return {
                "export_id": export_id,
                "status": export_config["status"],
                "created_at": export_config["created_at"],
                "message": f"Export job {export_id} information retrieved successfully"
            }
            
        except Exception as e:
            logger.error("get_export_info_failed", error=str(e))
            raise