"""
Data Processing Pipeline for PolicyCortex.
Implements real-time ETL for Azure governance data ingestion and transformation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import structlog
import pandas as pd
import numpy as np
from azure.storage.blob.aio import BlobServiceClient
from azure.eventhub.aio import EventHubConsumerClient
from azure.servicebus.aio import ServiceBusClient

logger = structlog.get_logger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    AZURE_EVENTS = "azure_events"
    POLICY_DATA = "policy_data"
    RBAC_DATA = "rbac_data"
    COST_DATA = "cost_data"
    NETWORK_DATA = "network_data"
    RESOURCE_DATA = "resource_data"


@dataclass
class DataQualityMetrics:
    """Data quality metrics for processed data."""
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    processing_timestamp: datetime


@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    source_type: DataSourceType
    records_processed: int
    records_transformed: int
    records_stored: int
    processing_time_ms: float
    quality_metrics: DataQualityMetrics
    errors: List[str]


class DataTransformer:
    """
    Transforms raw Azure governance data into standardized format.
    """
    
    def __init__(self):
        self.transformation_rules = {
            DataSourceType.AZURE_EVENTS: self._transform_azure_events,
            DataSourceType.POLICY_DATA: self._transform_policy_data,
            DataSourceType.RBAC_DATA: self._transform_rbac_data,
            DataSourceType.COST_DATA: self._transform_cost_data,
            DataSourceType.NETWORK_DATA: self._transform_network_data,
            DataSourceType.RESOURCE_DATA: self._transform_resource_data
        }
    
    async def transform(self, data: List[Dict[str, Any]], 
                       source_type: DataSourceType) -> List[Dict[str, Any]]:
        """Transform data based on source type."""
        try:
            transformer = self.transformation_rules.get(source_type)
            if not transformer:
                raise ValueError(f"No transformer found for {source_type}")
            
            transformed_data = []
            for record in data:
                try:
                    transformed_record = await transformer(record)
                    if transformed_record:
                        transformed_data.append(transformed_record)
                except Exception as e:
                    logger.warning("record_transformation_failed", 
                                 source_type=source_type.value,
                                 error=str(e))
            
            return transformed_data
            
        except Exception as e:
            logger.error("data_transformation_failed", 
                        source_type=source_type.value,
                        error=str(e))
            raise
    
    async def _transform_azure_events(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Azure event data."""
        return {
            "id": record.get("id"),
            "event_type": "azure_event",
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "source": "azure",
            "category": record.get("type", "unknown"),
            "operation": record.get("operation"),
            "status": record.get("status"),
            "resource_id": record.get("resource_id"),
            "subscription_id": record.get("subscription_id"),
            "resource_group": record.get("resource_group"),
            "correlation_id": record.get("correlation_id"),
            "properties": record.get("properties", {}),
            "severity": record.get("severity", "medium"),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _transform_policy_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform policy data."""
        return {
            "id": record.get("policy_id") or record.get("id"),
            "data_type": "policy",
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "policy_name": record.get("name"),
            "policy_type": record.get("type"),
            "compliance_state": record.get("compliance_state", "unknown"),
            "effect": record.get("effect"),
            "scope": record.get("scope"),
            "parameters": record.get("parameters", {}),
            "resource_count": record.get("resource_count", 0),
            "violation_count": record.get("violation_count", 0),
            "last_evaluated": self._parse_timestamp(record.get("last_evaluated")),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _transform_rbac_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform RBAC data."""
        return {
            "id": record.get("assignment_id") or record.get("id"),
            "data_type": "rbac",
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "principal_id": record.get("principal_id"),
            "principal_type": record.get("principal_type"),
            "role_definition_id": record.get("role_definition_id"),
            "role_name": record.get("role_name"),
            "scope": record.get("scope"),
            "created_on": self._parse_timestamp(record.get("created_on")),
            "updated_on": self._parse_timestamp(record.get("updated_on")),
            "permissions": record.get("permissions", []),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _transform_cost_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform cost data."""
        return {
            "id": record.get("usage_id") or record.get("id"),
            "data_type": "cost",
            "timestamp": self._parse_timestamp(record.get("date")),
            "subscription_id": record.get("subscription_id"),
            "resource_group": record.get("resource_group"),
            "resource_id": record.get("resource_id"),
            "service_name": record.get("service_name"),
            "meter_category": record.get("meter_category"),
            "meter_name": record.get("meter_name"),
            "cost": float(record.get("cost", 0)),
            "quantity": float(record.get("quantity", 0)),
            "unit_price": float(record.get("unit_price", 0)),
            "currency": record.get("currency", "USD"),
            "billing_period": record.get("billing_period"),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _transform_network_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform network data."""
        return {
            "id": record.get("flow_id") or record.get("id"),
            "data_type": "network",
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "source_ip": record.get("source_ip"),
            "destination_ip": record.get("destination_ip"),
            "source_port": record.get("source_port"),
            "destination_port": record.get("destination_port"),
            "protocol": record.get("protocol"),
            "action": record.get("action"),
            "flow_state": record.get("flow_state"),
            "nsg_name": record.get("nsg_name"),
            "rule_name": record.get("rule_name"),
            "bytes_sent": int(record.get("bytes_sent", 0)),
            "packets_sent": int(record.get("packets_sent", 0)),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _transform_resource_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform resource data."""
        return {
            "id": record.get("resource_id") or record.get("id"),
            "data_type": "resource",
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "resource_name": record.get("name"),
            "resource_type": record.get("type"),
            "location": record.get("location"),
            "subscription_id": record.get("subscription_id"),
            "resource_group": record.get("resource_group"),
            "tags": record.get("tags", {}),
            "sku": record.get("sku", {}),
            "properties": record.get("properties", {}),
            "provisioning_state": record.get("provisioning_state"),
            "created_time": self._parse_timestamp(record.get("created_time")),
            "changed_time": self._parse_timestamp(record.get("changed_time")),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[str]:
        """Parse timestamp string to ISO format."""
        if not timestamp_str:
            return None
        
        try:
            if isinstance(timestamp_str, datetime):
                return timestamp_str.isoformat()
            elif isinstance(timestamp_str, str):
                # Try to parse common timestamp formats
                dt = pd.to_datetime(timestamp_str)
                return dt.isoformat()
            else:
                return str(timestamp_str)
        except Exception:
            return None


class DataQualityEngine:
    """
    Validates and assesses data quality for processed governance data.
    """
    
    def __init__(self):
        self.validation_rules = {
            "required_fields": ["id", "data_type", "timestamp"],
            "data_types": {
                "cost": ["cost", "quantity"],
                "policy": ["policy_name", "compliance_state"],
                "rbac": ["principal_id", "role_name"],
                "network": ["source_ip", "destination_ip"],
                "resource": ["resource_name", "resource_type"]
            }
        }
    
    async def validate_data(self, data: List[Dict[str, Any]], 
                           source_type: DataSourceType) -> DataQualityMetrics:
        """Validate data quality and return metrics."""
        total_records = len(data)
        valid_records = 0
        invalid_records = 0
        duplicate_records = 0
        
        seen_ids = set()
        valid_data = []
        
        for record in data:
            is_valid = True
            
            # Check required fields
            for field in self.validation_rules["required_fields"]:
                if field not in record or record[field] is None:
                    is_valid = False
                    break
            
            # Check data type specific fields
            data_type = record.get("data_type", source_type.value.replace("_data", ""))
            type_fields = self.validation_rules["data_types"].get(data_type, [])
            for field in type_fields:
                if field not in record or record[field] is None:
                    is_valid = False
                    break
            
            # Check for duplicates
            record_id = record.get("id")
            if record_id in seen_ids:
                duplicate_records += 1
                is_valid = False
            else:
                seen_ids.add(record_id)
            
            if is_valid:
                valid_records += 1
                valid_data.append(record)
            else:
                invalid_records += 1
        
        # Calculate quality scores
        completeness_score = (valid_records / total_records) if total_records > 0 else 0
        accuracy_score = self._calculate_accuracy_score(valid_data)
        consistency_score = self._calculate_consistency_score(valid_data)
        
        return DataQualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_records,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            processing_timestamp=datetime.utcnow()
        )
    
    def _calculate_accuracy_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score based on data consistency."""
        if not data:
            return 0.0
        
        accuracy_checks = 0
        passed_checks = 0
        
        for record in data:
            # Check timestamp validity
            accuracy_checks += 1
            timestamp = record.get("timestamp")
            if timestamp and self._is_valid_timestamp(timestamp):
                passed_checks += 1
            
            # Check numeric fields are actually numeric
            for field, value in record.items():
                if field in ["cost", "quantity", "bytes_sent", "packets_sent"]:
                    accuracy_checks += 1
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        passed_checks += 1
        
        return (passed_checks / accuracy_checks) if accuracy_checks > 0 else 1.0
    
    def _calculate_consistency_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across records."""
        if len(data) < 2:
            return 1.0
        
        # Check field consistency
        field_consistency = {}
        for record in data:
            for field, value in record.items():
                if field not in field_consistency:
                    field_consistency[field] = set()
                field_consistency[field].add(type(value).__name__)
        
        consistent_fields = 0
        total_fields = 0
        
        for field, types in field_consistency.items():
            total_fields += 1
            if len(types) == 1:  # All records have same type for this field
                consistent_fields += 1
        
        return (consistent_fields / total_fields) if total_fields > 0 else 1.0
    
    def _is_valid_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp string is valid."""
        try:
            pd.to_datetime(timestamp_str)
            return True
        except Exception:
            return False


class DataPipeline:
    """
    Main data processing pipeline for PolicyCortex.
    Handles ingestion, transformation, validation, and storage.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.transformer = DataTransformer()
        self.quality_engine = DataQualityEngine()
        self.blob_client = None
        self.event_hub_client = None
        self.service_bus_client = None
        self.is_running = False
        self.processing_tasks = {}
        
        # Data storage
        self.processed_data_store = {}
        self.quality_metrics_store = {}
    
    async def initialize(self):
        """Initialize the data pipeline."""
        logger.info("initializing_data_pipeline")
        
        try:
            # Initialize Azure clients
            if hasattr(self.settings.azure, 'storage_connection_string'):
                self.blob_client = BlobServiceClient.from_connection_string(
                    self.settings.azure.storage_connection_string
                )
            
            if hasattr(self.settings.azure, 'event_hub_connection_string'):
                self.event_hub_client = EventHubConsumerClient.from_connection_string(
                    self.settings.azure.event_hub_connection_string
                )
            
            if hasattr(self.settings.azure, 'service_bus_connection_string'):
                self.service_bus_client = ServiceBusClient.from_connection_string(
                    self.settings.azure.service_bus_connection_string
                )
            
            logger.info("data_pipeline_initialized")
            
        except Exception as e:
            logger.error("data_pipeline_initialization_failed", error=str(e))
            raise
    
    async def start_processing(self):
        """Start data processing pipeline."""
        if self.is_running:
            logger.warning("data_pipeline_already_running")
            return
        
        logger.info("starting_data_pipeline")
        self.is_running = True
        
        # Start processing tasks for different data sources
        self.processing_tasks = {
            "azure_events": asyncio.create_task(self._process_azure_events()),
            "batch_processor": asyncio.create_task(self._run_batch_processor())
        }
        
        logger.info("data_pipeline_started")
    
    async def stop_processing(self):
        """Stop data processing pipeline."""
        logger.info("stopping_data_pipeline")
        self.is_running = False
        
        # Cancel all processing tasks
        for task_name, task in self.processing_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("processing_task_cancelled", task=task_name)
        
        self.processing_tasks.clear()
        logger.info("data_pipeline_stopped")
    
    async def process_data(self, data: List[Dict[str, Any]], 
                          source_type: DataSourceType) -> ProcessingResult:
        """Process a batch of data."""
        start_time = datetime.utcnow()
        errors = []
        
        try:
            logger.info("processing_data_batch", 
                       source_type=source_type.value,
                       record_count=len(data))
            
            # Transform data
            transformed_data = await self.transformer.transform(data, source_type)
            
            # Validate data quality
            quality_metrics = await self.quality_engine.validate_data(
                transformed_data, source_type
            )
            
            # Store processed data
            stored_count = await self._store_processed_data(
                transformed_data, source_type
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = ProcessingResult(
                source_type=source_type,
                records_processed=len(data),
                records_transformed=len(transformed_data),
                records_stored=stored_count,
                processing_time_ms=processing_time,
                quality_metrics=quality_metrics,
                errors=errors
            )
            
            # Store quality metrics
            self.quality_metrics_store[source_type] = quality_metrics
            
            logger.info("data_processing_completed",
                       source_type=source_type.value,
                       processing_time_ms=processing_time,
                       quality_score=quality_metrics.completeness_score)
            
            return result
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            errors.append(error_msg)
            logger.error("data_processing_failed",
                        source_type=source_type.value,
                        error=str(e))
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProcessingResult(
                source_type=source_type,
                records_processed=len(data),
                records_transformed=0,
                records_stored=0,
                processing_time_ms=processing_time,
                quality_metrics=DataQualityMetrics(
                    total_records=len(data),
                    valid_records=0,
                    invalid_records=len(data),
                    duplicate_records=0,
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    consistency_score=0.0,
                    processing_timestamp=datetime.utcnow()
                ),
                errors=errors
            )
    
    async def _process_azure_events(self):
        """Process Azure events from Event Hub."""
        while self.is_running:
            try:
                if not self.event_hub_client:
                    await asyncio.sleep(60)
                    continue
                
                logger.debug("processing_azure_events")
                
                # Mock event processing for development
                # In production, this would consume from Event Hub
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("azure_event_processing_failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _run_batch_processor(self):
        """Run periodic batch processing."""
        while self.is_running:
            try:
                logger.debug("running_batch_processor")
                
                # Process any queued data
                await self._process_queued_data()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next batch
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_processing_failed", error=str(e))
                await asyncio.sleep(300)
    
    async def _store_processed_data(self, data: List[Dict[str, Any]], 
                                   source_type: DataSourceType) -> int:
        """Store processed data."""
        try:
            # Store in memory for development
            if source_type not in self.processed_data_store:
                self.processed_data_store[source_type] = []
            
            self.processed_data_store[source_type].extend(data)
            
            # Limit memory usage - keep only recent data
            max_records = 10000
            if len(self.processed_data_store[source_type]) > max_records:
                self.processed_data_store[source_type] = \
                    self.processed_data_store[source_type][-max_records:]
            
            # In production, store to Azure Blob Storage or database
            if self.blob_client:
                await self._store_to_blob_storage(data, source_type)
            
            return len(data)
            
        except Exception as e:
            logger.error("data_storage_failed", 
                        source_type=source_type.value,
                        error=str(e))
            return 0
    
    async def _store_to_blob_storage(self, data: List[Dict[str, Any]], 
                                    source_type: DataSourceType):
        """Store data to Azure Blob Storage."""
        try:
            container_name = "processed-data"
            blob_name = f"{source_type.value}/{datetime.utcnow().strftime('%Y/%m/%d/%H')}/data.json"
            
            # Convert data to JSON
            json_data = json.dumps(data, default=str)
            
            # Upload to blob storage
            blob_client = self.blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            await blob_client.upload_blob(json_data, overwrite=True)
            
            logger.debug("data_stored_to_blob",
                        source_type=source_type.value,
                        blob_name=blob_name)
            
        except Exception as e:
            logger.error("blob_storage_failed",
                        source_type=source_type.value,
                        error=str(e))
    
    async def _process_queued_data(self):
        """Process any queued data."""
        # Implementation for processing queued data
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old processed data."""
        try:
            # Clean up in-memory data older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for source_type, data_list in self.processed_data_store.items():
                filtered_data = []
                for record in data_list:
                    processed_at = record.get("processed_at")
                    if processed_at:
                        try:
                            processed_time = datetime.fromisoformat(
                                processed_at.replace('Z', '+00:00')
                            )
                            if processed_time > cutoff_time:
                                filtered_data.append(record)
                        except Exception:
                            # Keep record if we can't parse timestamp
                            filtered_data.append(record)
                
                self.processed_data_store[source_type] = filtered_data
            
            logger.debug("old_data_cleanup_completed")
            
        except Exception as e:
            logger.error("data_cleanup_failed", error=str(e))
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get data processing statistics."""
        stats = {
            "is_running": self.is_running,
            "active_tasks": len([t for t in self.processing_tasks.values() if not t.done()]),
            "data_sources": {}
        }
        
        for source_type, data_list in self.processed_data_store.items():
            stats["data_sources"][source_type.value] = {
                "record_count": len(data_list),
                "latest_timestamp": self._get_latest_timestamp(data_list)
            }
        
        # Add quality metrics
        stats["quality_metrics"] = {}
        for source_type, metrics in self.quality_metrics_store.items():
            stats["quality_metrics"][source_type.value] = {
                "completeness_score": metrics.completeness_score,
                "accuracy_score": metrics.accuracy_score,
                "consistency_score": metrics.consistency_score,
                "last_updated": metrics.processing_timestamp.isoformat()
            }
        
        return stats
    
    def _get_latest_timestamp(self, data_list: List[Dict[str, Any]]) -> Optional[str]:
        """Get the latest timestamp from a data list."""
        if not data_list:
            return None
        
        latest = None
        for record in data_list:
            timestamp = record.get("processed_at")
            if timestamp and (not latest or timestamp > latest):
                latest = timestamp
        
        return latest
    
    async def cleanup(self):
        """Cleanup pipeline resources."""
        logger.info("cleaning_up_data_pipeline")
        
        await self.stop_processing()
        
        if self.blob_client:
            await self.blob_client.close()
        
        if self.event_hub_client:
            await self.event_hub_client.close()
        
        if self.service_bus_client:
            await self.service_bus_client.close()
        
        self.processed_data_store.clear()
        self.quality_metrics_store.clear()
        
        logger.info("data_pipeline_cleanup_completed")