"""
Pydantic models for Data Processing service.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum


class PipelineStatus(str, Enum):
    """Pipeline status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class DataSourceType(str, Enum):
    """Data source type enumeration."""
    AZURE_SQL = "azure_sql"
    COSMOS_DB = "cosmos_db"
    BLOB_STORAGE = "blob_storage"
    EVENT_HUB = "event_hub"
    SERVICE_BUS = "service_bus"
    TABLE_STORAGE = "table_storage"
    DATA_LAKE = "data_lake"
    SYNAPSE = "synapse"
    EXTERNAL_API = "external_api"


class DataFormat(str, Enum):
    """Data format enumeration."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    XML = "xml"
    EXCEL = "excel"
    YAML = "yaml"


class ProcessingEngineType(str, Enum):
    """Processing engine type enumeration."""
    PANDAS = "pandas"
    SPARK = "spark"
    AZURE_SYNAPSE = "azure_synapse"
    DATABRICKS = "databricks"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class DataSourceConfig(BaseModel):
    """Data source configuration model."""
    source_type: DataSourceType = Field(..., description="Type of data source")
    connection_string: Optional[str] = Field(None, description="Connection string")
    server: Optional[str] = Field(None, description="Server address")
    database: Optional[str] = Field(None, description="Database name")
    container: Optional[str] = Field(None, description="Container name")
    table: Optional[str] = Field(None, description="Table name")
    query: Optional[str] = Field(None, description="SQL query or filter")
    authentication: Optional[Dict[str, Any]] = Field(None, description="Authentication config")
    additional_config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")


class DataTargetConfig(BaseModel):
    """Data target configuration model."""
    target_type: DataSourceType = Field(..., description="Type of data target")
    connection_string: Optional[str] = Field(None, description="Connection string")
    server: Optional[str] = Field(None, description="Server address")
    database: Optional[str] = Field(None, description="Database name")
    container: Optional[str] = Field(None, description="Container name")
    table: Optional[str] = Field(None, description="Table name")
    write_mode: str = Field("append", description="Write mode (append, overwrite, upsert)")
    authentication: Optional[Dict[str, Any]] = Field(None, description="Authentication config")
    additional_config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")


class TransformationRule(BaseModel):
    """Data transformation rule model."""
    rule_type: str = Field(..., description="Type of transformation rule")
    source_field: Optional[str] = Field(None, description="Source field name")
    target_field: Optional[str] = Field(None, description="Target field name")
    expression: Optional[str] = Field(None, description="Transformation expression")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Rule parameters")
    condition: Optional[str] = Field(None, description="Condition for applying rule")


class ValidationRule(BaseModel):
    """Data validation rule model."""
    rule_type: str = Field(..., description="Type of validation rule")
    field: str = Field(..., description="Field to validate")
    parameters: Dict[str, Any] = Field(..., description="Validation parameters")
    error_message: Optional[str] = Field(None, description="Custom error message")
    severity: str = Field("error", description="Severity level (error, warning, info)")


class AggregationRule(BaseModel):
    """Data aggregation rule model."""
    field: str = Field(..., description="Field to aggregate")
    function: str = Field(..., description="Aggregation function")
    alias: Optional[str] = Field(None, description="Alias for aggregated field")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Function parameters")


class ScheduleConfig(BaseModel):
    """Schedule configuration model."""
    schedule_type: str = Field(..., description="Type of schedule (cron, interval)")
    expression: str = Field(..., description="Schedule expression")
    timezone: str = Field("UTC", description="Timezone for schedule")
    start_date: Optional[datetime] = Field(None, description="Schedule start date")
    end_date: Optional[datetime] = Field(None, description="Schedule end date")
    enabled: bool = Field(True, description="Schedule enabled status")


# ETL Pipeline Models
class ETLPipelineRequest(BaseModel):
    """ETL pipeline request model."""
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    source_config: DataSourceConfig = Field(..., description="Source configuration")
    target_config: DataTargetConfig = Field(..., description="Target configuration")
    transformation_rules: List[TransformationRule] = Field(default_factory=list, description="Transformation rules")
    validation_rules: List[ValidationRule] = Field(default_factory=list, description="Validation rules")
    schedule: Optional[ScheduleConfig] = Field(None, description="Schedule configuration")
    processing_engine: ProcessingEngineType = Field(ProcessingEngineType.PANDAS, description="Processing engine")
    batch_size: int = Field(1000, description="Batch size for processing")
    parallel_processing: bool = Field(False, description="Enable parallel processing")
    execute_immediately: bool = Field(False, description="Execute pipeline immediately")
    tags: List[str] = Field(default_factory=list, description="Pipeline tags")


class ETLPipelineResponse(BaseModel):
    """ETL pipeline response model."""
    pipeline_id: str = Field(..., description="Pipeline identifier")
    status: PipelineStatus = Field(..., description="Pipeline status")
    message: str = Field(..., description="Response message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Stream Processing Models
class StreamProcessingRequest(BaseModel):
    """Stream processing request model."""
    name: str = Field(..., description="Stream processor name")
    description: Optional[str] = Field(None, description="Stream processor description")
    source_config: DataSourceConfig = Field(..., description="Stream source configuration")
    processing_rules: List[TransformationRule] = Field(..., description="Processing rules")
    output_config: DataTargetConfig = Field(..., description="Output configuration")
    window_size: Optional[int] = Field(None, description="Window size for processing")
    window_type: str = Field("tumbling", description="Window type (tumbling, sliding)")
    checkpoint_interval: int = Field(30, description="Checkpoint interval in seconds")
    parallelism: int = Field(1, description="Processing parallelism")
    tags: List[str] = Field(default_factory=list, description="Processor tags")


class StreamProcessingResponse(BaseModel):
    """Stream processing response model."""
    processor_id: str = Field(..., description="Processor identifier")
    status: PipelineStatus = Field(..., description="Processor status")
    message: str = Field(..., description="Response message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Data Transformation Models
class DataTransformationRequest(BaseModel):
    """Data transformation request model."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Data to transform")
    transformation_rules: List[TransformationRule] = Field(..., description="Transformation rules")
    output_format: DataFormat = Field(DataFormat.JSON, description="Output format")
    processing_engine: ProcessingEngineType = Field(ProcessingEngineType.PANDAS, description="Processing engine")
    validate_output: bool = Field(True, description="Validate output data")


class DataTransformationResponse(BaseModel):
    """Data transformation response model."""
    transformation_id: str = Field(..., description="Transformation identifier")
    transformed_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Transformed data")
    status: str = Field(..., description="Transformation status")
    message: str = Field(..., description="Response message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Transformation metadata")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


# Data Validation Models
class DataValidationRequest(BaseModel):
    """Data validation request model."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Data to validate")
    validation_rules: List[ValidationRule] = Field(..., description="Validation rules")
    quality_threshold: float = Field(0.8, description="Quality threshold (0-1)")
    fail_on_error: bool = Field(False, description="Fail validation on first error")
    generate_report: bool = Field(True, description="Generate validation report")


class ValidationResult(BaseModel):
    """Validation result model."""
    rule_name: str = Field(..., description="Validation rule name")
    field: str = Field(..., description="Field validated")
    status: str = Field(..., description="Validation status")
    error_count: int = Field(0, description="Number of errors")
    warning_count: int = Field(0, description="Number of warnings")
    message: Optional[str] = Field(None, description="Validation message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class DataValidationResponse(BaseModel):
    """Data validation response model."""
    validation_id: str = Field(..., description="Validation identifier")
    validation_results: List[ValidationResult] = Field(..., description="Validation results")
    quality_score: float = Field(..., description="Overall quality score")
    status: str = Field(..., description="Validation status")
    message: str = Field(..., description="Response message")
    total_records: int = Field(0, description="Total records validated")
    valid_records: int = Field(0, description="Valid records count")
    invalid_records: int = Field(0, description="Invalid records count")


# Data Aggregation Models
class DataAggregationRequest(BaseModel):
    """Data aggregation request model."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Data to aggregate")
    aggregation_rules: List[AggregationRule] = Field(..., description="Aggregation rules")
    group_by_fields: List[str] = Field(default_factory=list, description="Fields to group by")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    sort_by: Optional[List[str]] = Field(None, description="Sort fields")
    limit: Optional[int] = Field(None, description="Result limit")


class DataAggregationResponse(BaseModel):
    """Data aggregation response model."""
    aggregation_id: str = Field(..., description="Aggregation identifier")
    aggregated_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Aggregated data")
    status: str = Field(..., description="Aggregation status")
    message: str = Field(..., description="Response message")
    record_count: int = Field(0, description="Number of aggregated records")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


# Data Lineage Models
class LineageNode(BaseModel):
    """Data lineage node model."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    description: Optional[str] = Field(None, description="Entity description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Entity metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class LineageEdge(BaseModel):
    """Data lineage edge model."""
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Relationship type")
    transformation_info: Optional[Dict[str, Any]] = Field(None, description="Transformation information")
    created_at: datetime = Field(..., description="Creation timestamp")


class DataLineageRequest(BaseModel):
    """Data lineage request model."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Entity type")
    depth: int = Field(3, description="Lineage depth")
    direction: str = Field("both", description="Lineage direction (upstream, downstream, both)")


class DataLineageResponse(BaseModel):
    """Data lineage response model."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Entity type")
    lineage_graph: Dict[str, Any] = Field(..., description="Lineage graph")
    upstream_dependencies: List[LineageNode] = Field(..., description="Upstream dependencies")
    downstream_dependencies: List[LineageNode] = Field(..., description="Downstream dependencies")
    status: str = Field(..., description="Lineage status")
    message: str = Field(..., description="Response message")


# Data Export Models
class DataExportRequest(BaseModel):
    """Data export request model."""
    name: str = Field(..., description="Export job name")
    description: Optional[str] = Field(None, description="Export job description")
    source_config: DataSourceConfig = Field(..., description="Source configuration")
    destination_config: DataTargetConfig = Field(..., description="Destination configuration")
    export_format: DataFormat = Field(..., description="Export format")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    compression: Optional[str] = Field(None, description="Compression type")
    encryption: Optional[Dict[str, Any]] = Field(None, description="Encryption settings")
    schedule: Optional[ScheduleConfig] = Field(None, description="Schedule configuration")
    tags: List[str] = Field(default_factory=list, description="Export tags")


class DataExportResponse(BaseModel):
    """Data export response model."""
    export_id: str = Field(..., description="Export identifier")
    status: str = Field(..., description="Export status")
    message: str = Field(..., description="Response message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    file_path: Optional[str] = Field(None, description="Exported file path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    record_count: Optional[int] = Field(None, description="Number of exported records")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


# Monitoring and Metrics Models
class ProcessingMetrics(BaseModel):
    """Processing metrics model."""
    active_pipelines: int = Field(0, description="Number of active pipelines")
    completed_pipelines: int = Field(0, description="Number of completed pipelines")
    failed_pipelines: int = Field(0, description="Number of failed pipelines")
    data_volume_processed: int = Field(0, description="Total data volume processed (bytes)")
    average_processing_time: float = Field(0.0, description="Average processing time (seconds)")
    quality_score_average: float = Field(0.0, description="Average quality score")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class PipelineMetrics(BaseModel):
    """Pipeline metrics model."""
    pipeline_id: str = Field(..., description="Pipeline identifier")
    execution_count: int = Field(0, description="Number of executions")
    success_count: int = Field(0, description="Number of successful executions")
    failure_count: int = Field(0, description="Number of failed executions")
    average_duration: float = Field(0.0, description="Average execution duration (seconds)")
    last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")
    data_volume_processed: int = Field(0, description="Total data volume processed (bytes)")
    quality_score_average: float = Field(0.0, description="Average quality score")


class QualityMetrics(BaseModel):
    """Data quality metrics model."""
    total_records: int = Field(0, description="Total records processed")
    valid_records: int = Field(0, description="Valid records count")
    invalid_records: int = Field(0, description="Invalid records count")
    quality_score: float = Field(0.0, description="Overall quality score")
    completeness_score: float = Field(0.0, description="Completeness score")
    accuracy_score: float = Field(0.0, description="Accuracy score")
    consistency_score: float = Field(0.0, description="Consistency score")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class SparkConfig(BaseModel):
    """Spark configuration model."""
    app_name: str = Field(..., description="Spark application name")
    master: str = Field("local[*]", description="Spark master URL")
    executor_memory: str = Field("2g", description="Executor memory")
    executor_cores: int = Field(2, description="Executor cores")
    driver_memory: str = Field("1g", description="Driver memory")
    max_result_size: str = Field("1g", description="Max result size")
    sql_adaptive_enabled: bool = Field(True, description="Enable adaptive query execution")
    sql_adaptive_coalesce_partitions: bool = Field(True, description="Enable partition coalescing")
    additional_config: Optional[Dict[str, Any]] = Field(None, description="Additional Spark configuration")


class DataConnectorHealth(BaseModel):
    """Data connector health model."""
    connector_type: str = Field(..., description="Connector type")
    status: str = Field(..., description="Health status")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ServiceHealth(BaseModel):
    """Service health model."""
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Overall health status")
    connectors: List[DataConnectorHealth] = Field(..., description="Connector health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    uptime_seconds: int = Field(0, description="Service uptime in seconds")
    memory_usage_mb: Optional[int] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")