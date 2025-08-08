"""
ETL Pipeline service for Data Processing.
Handles creation, execution, and management of ETL pipelines.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ....shared.config import get_settings
from ....shared.database import get_async_db

    DataSourceConfig,
    DataTargetConfig,
    TransformationRule,
    ValidationRule,
    ScheduleConfig,
    PipelineStatus,
    ProcessingEngineType,
    SparkConfig
)
from .azure_connectors import AzureConnectorService
from .data_transformer import DataTransformerService
from .data_validator import DataValidatorService

settings = get_settings()
logger = structlog.get_logger(__name__)


class ETLPipelineService:
    """Service for managing ETL pipelines."""

    def __init__(self):
        self.settings = settings
        self.azure_connector = AzureConnectorService()
        self.data_transformer = DataTransformerService()
        self.data_validator = DataValidatorService()
        self.spark_session = None
        self.active_pipelines = {}
        self.pipeline_metrics = {}

    async def _get_spark_session(self, config: Optional[SparkConfig] = None) -> SparkSession:
        """Get or create Spark session."""
        if self.spark_session is None or self.spark_session.sparkContext._jsc is None:
            try:
                from pyspark.sql import SparkSession

                # Default Spark configuration
                default_config = SparkConfig(
                    app_name="PolicyCortex-DataProcessing",
                    master="local[*]",
                    executor_memory="2g",
                    executor_cores=2,
                    driver_memory="1g",
                    max_result_size="1g",
                    sql_adaptive_enabled=True,
                    sql_adaptive_coalesce_partitions=True
                )

                spark_config = config or default_config

                # Create Spark session
                builder = SparkSession.builder \
                    .appName(spark_config.app_name) \
                    .master(spark_config.master) \
                    .config("spark.executor.memory", spark_config.executor_memory) \
                    .config("spark.executor.cores", str(spark_config.executor_cores)) \
                    .config("spark.driver.memory", spark_config.driver_memory) \
                    .config("spark.driver.maxResultSize", spark_config.max_result_size) \
                    .config(
                        "spark.sql.adaptive.enabled",
                        str(spark_config.sql_adaptive_enabled).lower()
                    ) \
                    .config(
                        "spark.sql.adaptive.coalescePartitions.enabled",
                        str(spark_config.sql_adaptive_coalesce_partitions).lower()
                    )

                # Add additional configurations
                if spark_config.additional_config:
                    for key, value in spark_config.additional_config.items():
                        builder = builder.config(key, str(value))

                self.spark_session = builder.getOrCreate()

                logger.info("spark_session_created", app_name=spark_config.app_name)

            except Exception as e:
                logger.error("spark_session_creation_failed", error=str(e))
                raise

        return self.spark_session

    async def create_pipeline(self, source_config: DataSourceConfig, target_config: DataTargetConfig,
                            transformation_rules: List[TransformationRule],
                            schedule: Optional[ScheduleConfig] = None,
                            validation_rules: Optional[List[ValidationRule]] = None,
                            processing_engine: ProcessingEngineType = ProcessingEngineType.PANDAS,
                            batch_size: int = 1000,
                            parallel_processing: bool = False,
                            user_id: Optional[str] = None) -> str:
        """Create a new ETL pipeline."""
        try:
            pipeline_id = str(uuid.uuid4())

            # Store pipeline configuration
            pipeline_config = {
                "pipeline_id": pipeline_id,
                "source_config": source_config.dict(),
                "target_config": target_config.dict(),
                "transformation_rules": [rule.dict() for rule in transformation_rules],
                "validation_rules": [rule.dict() for rule in validation_rules] if validation_rules else [],
                "schedule": schedule.dict() if schedule else None,
                "processing_engine": processing_engine.value,
                "batch_size": batch_size,
                "parallel_processing": parallel_processing,
                "status": PipelineStatus.CREATED.value,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": user_id,
                "last_updated": datetime.utcnow().isoformat(),
                "execution_history": []
            }

            # Save to database
            db = await get_async_db()
            await db.execute(
                text("""
                    INSERT INTO etl_pipelines (
                        pipeline_id, source_config, target_config, transformation_rules,
                        validation_rules, schedule_config, processing_engine, batch_size,
                        parallel_processing, status, created_at, created_by, last_updated
                    ) VALUES (
                        :pipeline_id, :source_config, :target_config, :transformation_rules,
                        :validation_rules, :schedule_config, :processing_engine, :batch_size,
                        :parallel_processing, :status, :created_at, :created_by, :last_updated
                    )
                """),
                {
                    "pipeline_id": pipeline_id,
                    "source_config": json.dumps(pipeline_config["source_config"]),
                    "target_config": json.dumps(pipeline_config["target_config"]),
                    "transformation_rules": json.dumps(pipeline_config["transformation_rules"]),
                    "validation_rules": json.dumps(pipeline_config["validation_rules"]),
                    "schedule_config": json.dumps(pipeline_config["schedule"]),
                    "processing_engine": pipeline_config["processing_engine"],
                    "batch_size": pipeline_config["batch_size"],
                    "parallel_processing": pipeline_config["parallel_processing"],
                    "status": pipeline_config["status"],
                    "created_at": pipeline_config["created_at"],
                    "created_by": pipeline_config["created_by"],
                    "last_updated": pipeline_config["last_updated"]
                }
            )
            await db.commit()

            # Initialize pipeline metrics
            self.pipeline_metrics[pipeline_id] = {
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_processing_time": 0,
                "total_records_processed": 0,
                "average_quality_score": 0.0
            }

            logger.info(
                "etl_pipeline_created",
                pipeline_id=pipeline_id,
                source_type=source_config.source_type,
                target_type=target_config.target_type,
                processing_engine=processing_engine.value
            )

            return pipeline_id

        except Exception as e:
            logger.error("etl_pipeline_creation_failed", error=str(e))
            raise

    async def execute_pipeline(
        self,
        pipeline_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute an ETL pipeline."""
        try:
            # Get pipeline configuration
            pipeline_config = await self._get_pipeline_config(pipeline_id)

            if not pipeline_config:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            # Update pipeline status
            await self._update_pipeline_status(pipeline_id, PipelineStatus.RUNNING)

            execution_start = datetime.utcnow()
            execution_id = str(uuid.uuid4())

            # Create execution record
            execution_record = {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "started_at": execution_start.isoformat(),
                "started_by": user_id,
                "status": "running",
                "records_processed": 0,
                "quality_score": 0.0,
                "error_message": None,
                "details": {}
            }

            try:
                # Execute pipeline based on processing engine
                processing_engine = ProcessingEngineType(pipeline_config["processing_engine"])

                if processing_engine == ProcessingEngineType.SPARK:
                    result = await self._execute_spark_pipeline(pipeline_config, execution_record)
                else:
                    result = await self._execute_pandas_pipeline(pipeline_config, execution_record)

                # Update execution record
                execution_record.update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "processing_time_seconds": (datetime.utcnow() - execution_start).total_seconds(),
                    "records_processed": result["records_processed"],
                    "quality_score": result["quality_score"],
                    "details": result["details"]
                })

                # Update pipeline status
                await self._update_pipeline_status(pipeline_id, PipelineStatus.COMPLETED)

                # Update metrics
                await self._update_pipeline_metrics(pipeline_id, execution_record, success=True)

                logger.info(
                    "etl_pipeline_executed",
                    pipeline_id=pipeline_id,
                    execution_id=execution_id,
                    records_processed=result["records_processed"],
                    processing_time=execution_record["processing_time_seconds"]
                )

                return {
                    "execution_id": execution_id,
                    "status": "completed",
                    "records_processed": result["records_processed"],
                    "quality_score": result["quality_score"],
                    "processing_time_seconds": execution_record["processing_time_seconds"]
                }

            except Exception as e:
                # Update execution record with error
                execution_record.update({
                    "status": "failed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "processing_time_seconds": (datetime.utcnow() - execution_start).total_seconds(),
                    "error_message": str(e)
                })

                # Update pipeline status
                await self._update_pipeline_status(pipeline_id, PipelineStatus.FAILED)

                # Update metrics
                await self._update_pipeline_metrics(pipeline_id, execution_record, success=False)

                logger.error(
                    "etl_pipeline_execution_failed",
                    pipeline_id=pipeline_id,
                    execution_id=execution_id,
                    error=str(e)
                )

                raise

            finally:
                # Save execution record
                await self._save_execution_record(execution_record)

        except Exception as e:
            logger.error("etl_pipeline_execution_error", pipeline_id=pipeline_id, error=str(e))
            raise

    async def _execute_pandas_pipeline(self, pipeline_config: Dict[str, Any],
                                    execution_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline using Pandas."""
        try:
            # Parse configurations
            source_config = DataSourceConfig(**pipeline_config["source_config"])
            target_config = DataTargetConfig(**pipeline_config["target_config"])
            transformation_rules = (
                [TransformationRule(**rule) for rule in pipeline_config["transformation_rules"]]
            )
            validation_rules = (
                [ValidationRule(**rule) for rule in pipeline_config["validation_rules"]]
            )

            # Read data from source
            logger.info("reading_data_from_source", source_type=source_config.source_type)
            data = await self.azure_connector.read_data(source_config)

            execution_record["details"]["source_records"] = len(data)

            # Apply transformations
            if transformation_rules:
                logger.info("applying_transformations", rule_count=len(transformation_rules))
                transformed_data = await self.data_transformer.transform_data(
                    data=data.to_dict(orient="records"),
                    transformation_rules=transformation_rules,
                    output_format="dataframe"
                )
                data = transformed_data["transformed_data"]

            # Apply validation
            quality_score = 1.0
            if validation_rules:
                logger.info("applying_validation_rules", rule_count=len(validation_rules))
                validation_result = await self.data_validator.validate_data(
                    data=data.to_dict(orient="records"),
                    validation_rules=validation_rules,
                    quality_threshold=0.8
                )
                quality_score = validation_result["quality_score"]
                execution_record["details"]["validation_results"] = (
                    validation_result["validation_results"]
                )

            # Write data to target
            logger.info("writing_data_to_target", target_type=target_config.target_type)
            write_result = await self.azure_connector.write_data(
                target_config=target_config,
                data=data,
                write_mode=target_config.write_mode
            )

            execution_record["details"]["write_result"] = write_result

            return {
                "records_processed": len(data),
                "quality_score": quality_score,
                "details": execution_record["details"]
            }

        except Exception as e:
            logger.error("pandas_pipeline_execution_failed", error=str(e))
            raise

    async def _execute_spark_pipeline(self, pipeline_config: Dict[str, Any],
                                    execution_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline using Apache Spark."""
        try:
            # Get Spark session
            spark = await self._get_spark_session()

            # Parse configurations
            source_config = DataSourceConfig(**pipeline_config["source_config"])
            target_config = DataTargetConfig(**pipeline_config["target_config"])
            transformation_rules = (
                [TransformationRule(**rule) for rule in pipeline_config["transformation_rules"]]
            )
            validation_rules = (
                [ValidationRule(**rule) for rule in pipeline_config["validation_rules"]]
            )

            # Read data from source using Spark
            logger.info("reading_data_from_source_spark", source_type=source_config.source_type)
            df = await self._read_data_with_spark(spark, source_config)

            execution_record["details"]["source_records"] = df.count()

            # Apply transformations using Spark
            if transformation_rules:
                logger.info("applying_transformations_spark", rule_count=len(transformation_rules))
                df = await self._apply_spark_transformations(spark, df, transformation_rules)

            # Apply validation using Spark
            quality_score = 1.0
            if validation_rules:
                logger.info("applying_validation_rules_spark", rule_count=len(validation_rules))
                quality_score = await self._apply_spark_validation(spark, df, validation_rules)

            # Write data to target using Spark
            logger.info("writing_data_to_target_spark", target_type=target_config.target_type)
            records_written = await self._write_data_with_spark(spark, df, target_config)

            execution_record["details"]["records_written"] = records_written

            return {
                "records_processed": records_written,
                "quality_score": quality_score,
                "details": execution_record["details"]
            }

        except Exception as e:
            logger.error("spark_pipeline_execution_failed", error=str(e))
            raise

    async def _read_data_with_spark(self, spark: SparkSession, source_config: DataSourceConfig):
        """Read data using Spark."""
        try:
            source_type = source_config.source_type

            if source_type == "azure_sql":
                # Read from Azure SQL using Spark
                df = spark.read \
                    .format("jdbc") \
                    .option(
                        "url",
                        f"jdbc:sqlserver://{source_config.server}:1433;databaseName={source_config.database}"
                    ) \
                    .option("dbtable", source_config.table or f"({source_config.query}) AS query") \
                    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
                    .load()

            elif source_type == "blob_storage":
                # Read from Blob Storage using Spark
                file_format = source_config.additional_config.get("file_format", "csv")
                storage_account = self.settings.azure.storage_account_name
                container = source_config.container
                blob_name = source_config.additional_config.get("blob_name")

                path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/{blob_name}"

                if file_format.lower() == "csv":
                    df = spark.read.csv(path, header=True, inferSchema=True)
                elif file_format.lower() == "parquet":
                    df = spark.read.parquet(path)
                elif file_format.lower() == "json":
                    df = spark.read.json(path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

            else:
                raise ValueError(f"Unsupported source type for Spark: {source_type}")

            return df

        except Exception as e:
            logger.error("spark_read_data_failed", error=str(e))
            raise

    async def _apply_spark_transformations(
        self,
        spark: SparkSession,
        df,
        transformation_rules: List[TransformationRule]
    ):
        """Apply transformations using Spark."""
        try:
            for rule in transformation_rules:
                if rule.rule_type == "select_columns":
                    columns = rule.parameters.get("columns", [])
                    df = df.select(*columns)

                elif rule.rule_type == "filter":
                    condition = rule.expression
                    df = df.filter(condition)

                elif rule.rule_type == "add_column":
                    column_name = rule.target_field
                    expression = rule.expression
                    df = df.withColumn(column_name, expr(expression))

                elif rule.rule_type == "rename_column":
                    old_name = rule.source_field
                    new_name = rule.target_field
                    df = df.withColumnRenamed(old_name, new_name)

                elif rule.rule_type == "drop_column":
                    column_name = rule.source_field
                    df = df.drop(column_name)

                elif rule.rule_type == "cast_column":
                    column_name = rule.source_field
                    data_type = rule.parameters.get("data_type", "string")
                    df = df.withColumn(column_name, df[column_name].cast(data_type))

                elif rule.rule_type == "aggregate":
                    group_by_columns = rule.parameters.get("group_by", [])
                    agg_functions = rule.parameters.get("agg_functions", {})

                    if group_by_columns:
                        df = df.groupBy(*group_by_columns).agg(**agg_functions)
                    else:
                        df = df.agg(**agg_functions)

            return df

        except Exception as e:
            logger.error("spark_transformations_failed", error=str(e))
            raise

    async def _apply_spark_validation(
        self,
        spark: SparkSession,
        df,
        validation_rules: List[ValidationRule]
    ) -> float:
        """Apply validation rules using Spark."""
        try:
            total_records = df.count()
            valid_records = total_records

            for rule in validation_rules:
                if rule.rule_type == "not_null":
                    field = rule.field
                    null_count = df.filter(df[field].isNull()).count()
                    valid_records -= null_count

                elif rule.rule_type == "range_check":
                    field = rule.field
                    min_val = rule.parameters.get("min_value")
                    max_val = rule.parameters.get("max_value")

                    if min_val is not None and max_val is not None:
                        invalid_count = (
                            df.filter(~((df[field] >= min_val) & (df[field] <= max_val))).count()
                        )
                        valid_records -= invalid_count

                elif rule.rule_type == "pattern_match":
                    field = rule.field
                    pattern = rule.parameters.get("pattern")

                    if pattern:
                        invalid_count = df.filter(~df[field].rlike(pattern)).count()
                        valid_records -= invalid_count

            quality_score = valid_records / total_records if total_records > 0 else 0.0
            return quality_score

        except Exception as e:
            logger.error("spark_validation_failed", error=str(e))
            return 0.0

    async def _write_data_with_spark(
        self,
        spark: SparkSession,
        df,
        target_config: DataTargetConfig
    ) -> int:
        """Write data using Spark."""
        try:
            target_type = target_config.target_type
            records_count = df.count()

            if target_type == "azure_sql":
                # Write to Azure SQL using Spark
                df.write \
                    .format("jdbc") \
                    .option(
                        "url",
                        f"jdbc:sqlserver://{target_config.server}:1433;databaseName={target_config.database}"
                    ) \
                    .option("dbtable", target_config.table) \
                    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
                    .mode(target_config.write_mode) \
                    .save()

            elif target_type == "blob_storage":
                # Write to Blob Storage using Spark
                file_format = target_config.additional_config.get("file_format", "parquet")
                storage_account = self.settings.azure.storage_account_name
                container = target_config.container
                blob_name = target_config.additional_config.get(
                    "blob_name",
                    f"data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                )

                path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/{blob_name}"

                if file_format.lower() == "parquet":
                    df.write.mode(target_config.write_mode).parquet(path)
                elif file_format.lower() == "csv":
                    df.write.mode(target_config.write_mode).csv(path, header=True)
                elif file_format.lower() == "json":
                    df.write.mode(target_config.write_mode).json(path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

            else:
                raise ValueError(f"Unsupported target type for Spark: {target_type}")

            return records_count

        except Exception as e:
            logger.error("spark_write_data_failed", error=str(e))
            raise

    async def _get_pipeline_config(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline configuration from database."""
        try:
            db = await get_async_db()
            result = await db.execute(
                text("SELECT * FROM etl_pipelines WHERE pipeline_id = :pipeline_id"),
                {"pipeline_id": pipeline_id}
            )
            row = result.fetchone()

            if row:
                return {
                    "pipeline_id": row.pipeline_id,
                    "source_config": json.loads(row.source_config),
                    "target_config": json.loads(row.target_config),
                    "transformation_rules": json.loads(row.transformation_rules),
                    "validation_rules": json.loads(row.validation_rules),
                    "schedule": json.loads(row.schedule_config) if row.schedule_config else None,
                    "processing_engine": row.processing_engine,
                    "batch_size": row.batch_size,
                    "parallel_processing": row.parallel_processing,
                    "status": row.status,
                    "created_at": row.created_at,
                    "created_by": row.created_by,
                    "last_updated": row.last_updated
                }

            return None

        except Exception as e:
            logger.error("get_pipeline_config_failed", error=str(e))
            return None

    async def _update_pipeline_status(self, pipeline_id: str, status: PipelineStatus) -> None:
        """Update pipeline status."""
        try:
            db = await get_async_db()
            await db.execute(
                text("""
                    UPDATE etl_pipelines
                    SET status = :status, last_updated = :last_updated
                    WHERE pipeline_id = :pipeline_id
                """),
                {
                    "pipeline_id": pipeline_id,
                    "status": status.value,
                    "last_updated": datetime.utcnow().isoformat()
                }
            )
            await db.commit()

        except Exception as e:
            logger.error("update_pipeline_status_failed", error=str(e))

    async def _save_execution_record(self, execution_record: Dict[str, Any]) -> None:
        """Save pipeline execution record."""
        try:
            db = await get_async_db()
            await db.execute(
                text("""
                    INSERT INTO pipeline_executions (
                        execution_id, pipeline_id, started_at, started_by, status,
                        completed_at, processing_time_seconds, records_processed,
                        quality_score, error_message, details
                    ) VALUES (
                        :execution_id, :pipeline_id, :started_at, :started_by, :status,
                        :completed_at, :processing_time_seconds, :records_processed,
                        :quality_score, :error_message, :details
                    )
                """),
                {
                    "execution_id": execution_record["execution_id"],
                    "pipeline_id": execution_record["pipeline_id"],
                    "started_at": execution_record["started_at"],
                    "started_by": execution_record["started_by"],
                    "status": execution_record["status"],
                    "completed_at": execution_record.get("completed_at"),
                    "processing_time_seconds": execution_record.get("processing_time_seconds"),
                    "records_processed": execution_record.get("records_processed", 0),
                    "quality_score": execution_record.get("quality_score", 0.0),
                    "error_message": execution_record.get("error_message"),
                    "details": json.dumps(execution_record.get("details", {}))
                }
            )
            await db.commit()

        except Exception as e:
            logger.error("save_execution_record_failed", error=str(e))

    async def _update_pipeline_metrics(
        self,
        pipeline_id: str,
        execution_record: Dict[str,
        Any],
        success: bool
    ) -> None:
        """Update pipeline metrics."""
        try:
            if pipeline_id not in self.pipeline_metrics:
                self.pipeline_metrics[pipeline_id] = {
                    "execution_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_processing_time": 0,
                    "total_records_processed": 0,
                    "average_quality_score": 0.0
                }

            metrics = self.pipeline_metrics[pipeline_id]
            metrics["execution_count"] += 1

            if success:
                metrics["success_count"] += 1
                metrics["total_processing_time"] += execution_record.get(
                    "processing_time_seconds",
                    0
                )
                metrics["total_records_processed"] += execution_record.get("records_processed", 0)

                # Update average quality score
                current_avg = metrics["average_quality_score"]
                new_score = execution_record.get("quality_score", 0.0)
                metrics["average_quality_score"] = (
                    (current_avg * (metrics["success_count"] - 1) + new_score) / metrics["success_count"]
                )
            else:
                metrics["failure_count"] += 1

        except Exception as e:
            logger.error("update_pipeline_metrics_failed", error=str(e))

    async def get_pipeline_info(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline information."""
        try:
            pipeline_config = await self._get_pipeline_config(pipeline_id)
            if not pipeline_config:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            # Get execution history
            db = await get_async_db()
            result = await db.execute(
                text("""
                    SELECT * FROM pipeline_executions
                    WHERE pipeline_id = :pipeline_id
                    ORDER BY started_at DESC
                    LIMIT 10
                """),
                {"pipeline_id": pipeline_id}
            )
            executions = result.fetchall()

            execution_history = []
            for execution in executions:
                execution_history.append({
                    "execution_id": execution.execution_id,
                    "started_at": execution.started_at,
                    "completed_at": execution.completed_at,
                    "status": execution.status,
                    "records_processed": execution.records_processed,
                    "quality_score": execution.quality_score,
                    "processing_time_seconds": execution.processing_time_seconds,
                    "error_message": execution.error_message
                })

            return {
                "pipeline_id": pipeline_id,
                "status": pipeline_config["status"],
                "created_at": pipeline_config["created_at"],
                "last_updated": pipeline_config["last_updated"],
                "execution_history": execution_history,
                "metrics": self.pipeline_metrics.get(pipeline_id, {}),
                "message": f"Pipeline {pipeline_id} information retrieved successfully"
            }

        except Exception as e:
            logger.error("get_pipeline_info_failed", error=str(e))
            raise

    async def delete_pipeline(self, pipeline_id: str) -> None:
        """Delete a pipeline."""
        try:
            db = await get_async_db()

            # Delete execution records
            await db.execute(
                text("DELETE FROM pipeline_executions WHERE pipeline_id = :pipeline_id"),
                {"pipeline_id": pipeline_id}
            )

            # Delete pipeline
            await db.execute(
                text("DELETE FROM etl_pipelines WHERE pipeline_id = :pipeline_id"),
                {"pipeline_id": pipeline_id}
            )

            await db.commit()

            # Clean up metrics
            if pipeline_id in self.pipeline_metrics:
                del self.pipeline_metrics[pipeline_id]

            logger.info("etl_pipeline_deleted", pipeline_id=pipeline_id)

        except Exception as e:
            logger.error("delete_pipeline_failed", error=str(e))
            raise

    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get overall processing metrics."""
        try:
            db = await get_async_db()

            # Get pipeline counts
            result = await db.execute(
                text("SELECT status, COUNT(*) as count FROM etl_pipelines GROUP BY status")
            )
            status_counts = {row.status: row.count for row in result.fetchall()}

            # Get execution metrics
            result = await db.execute(
                text("""
                    SELECT
                        COUNT(*) as total_executions,
                        AVG(processing_time_seconds) as avg_processing_time,
                        SUM(records_processed) as total_records,
                        AVG(quality_score) as avg_quality_score
                    FROM pipeline_executions
                    WHERE status = 'completed'
                """)
            )
            execution_metrics = result.fetchone()

            return {
                "active_pipelines": status_counts.get("running", 0),
                "completed_pipelines": status_counts.get("completed", 0),
                "failed_pipelines": status_counts.get("failed", 0),
                "total_pipelines": sum(status_counts.values()),
                "data_volume_processed": execution_metrics.total_records or 0,
                "average_processing_time": execution_metrics.avg_processing_time or 0.0,
                "quality_score_average": execution_metrics.avg_quality_score or 0.0,
                "total_executions": execution_metrics.total_executions or 0
            }

        except Exception as e:
            logger.error("get_processing_metrics_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close resources."""
        try:
            if self.spark_session:
                self.spark_session.stop()
                self.spark_session = None

            await self.azure_connector.close_connections()

            logger.info("etl_pipeline_service_closed")

        except Exception as e:
            logger.error("etl_pipeline_service_close_failed", error=str(e))
