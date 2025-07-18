"""
Data transformation service for PolicyCortex.
Handles data transformations using Pandas and Spark.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import structlog
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import col, expr, lit, when, regexp_replace, upper, lower, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType, DateType, TimestampType

from ....shared.config import get_settings
from ..models import TransformationRule, DataFormat, ProcessingEngineType

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataTransformerService:
    """Service for data transformation operations."""
    
    def __init__(self):
        self.settings = settings
        self.transformation_history = {}
    
    async def transform_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                           transformation_rules: List[TransformationRule],
                           output_format: str = "json",
                           processing_engine: ProcessingEngineType = ProcessingEngineType.PANDAS,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Transform data using specified rules."""
        try:
            transformation_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            # Convert data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a dict or list of dicts")
            
            logger.info(
                "data_transformation_started",
                transformation_id=transformation_id,
                input_rows=len(df),
                rule_count=len(transformation_rules)
            )
            
            # Apply transformations
            if processing_engine == ProcessingEngineType.PANDAS:
                transformed_df = await self._apply_pandas_transformations(df, transformation_rules)
            else:
                transformed_df = await self._apply_spark_transformations(df, transformation_rules)
            
            # Convert output format
            if output_format == "dataframe":
                transformed_data = transformed_df
            elif output_format == "json":
                transformed_data = transformed_df.to_dict(orient="records")
            elif output_format == "csv":
                transformed_data = transformed_df.to_csv(index=False)
            else:
                transformed_data = transformed_df.to_dict(orient="records")
            
            # Record transformation history
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.transformation_history[transformation_id] = {
                "transformation_id": transformation_id,
                "input_rows": len(df),
                "output_rows": len(transformed_df) if hasattr(transformed_df, '__len__') else 0,
                "processing_time": processing_time,
                "rules_applied": len(transformation_rules),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "data_transformation_completed",
                transformation_id=transformation_id,
                output_rows=len(transformed_df) if hasattr(transformed_df, '__len__') else 0,
                processing_time=processing_time
            )
            
            return {
                "transformation_id": transformation_id,
                "transformed_data": transformed_data,
                "processing_time_ms": int(processing_time * 1000),
                "input_rows": len(df),
                "output_rows": len(transformed_df) if hasattr(transformed_df, '__len__') else 0
            }
            
        except Exception as e:
            logger.error("data_transformation_failed", error=str(e))
            raise
    
    async def _apply_pandas_transformations(self, df: pd.DataFrame, rules: List[TransformationRule]) -> pd.DataFrame:
        """Apply transformations using Pandas."""
        try:
            result_df = df.copy()
            
            for rule in rules:
                if rule.rule_type == "select_columns":
                    columns = rule.parameters.get("columns", [])
                    result_df = result_df[columns]
                    
                elif rule.rule_type == "drop_columns":
                    columns = rule.parameters.get("columns", [])
                    result_df = result_df.drop(columns=columns, errors='ignore')
                    
                elif rule.rule_type == "rename_column":
                    old_name = rule.source_field
                    new_name = rule.target_field
                    result_df = result_df.rename(columns={old_name: new_name})
                    
                elif rule.rule_type == "add_column":
                    column_name = rule.target_field
                    value = rule.parameters.get("value", None)
                    if rule.expression:
                        result_df[column_name] = result_df.eval(rule.expression)
                    else:
                        result_df[column_name] = value
                        
                elif rule.rule_type == "filter":
                    condition = rule.expression
                    result_df = result_df.query(condition)
                    
                elif rule.rule_type == "cast_column":
                    column_name = rule.source_field
                    data_type = rule.parameters.get("data_type", "str")
                    
                    if data_type == "int":
                        result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                    elif data_type == "float":
                        result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce')
                    elif data_type == "datetime":
                        result_df[column_name] = pd.to_datetime(result_df[column_name], errors='coerce')
                    elif data_type == "bool":
                        result_df[column_name] = result_df[column_name].astype('bool')
                    else:
                        result_df[column_name] = result_df[column_name].astype('str')
                        
                elif rule.rule_type == "string_operations":
                    column_name = rule.source_field
                    operation = rule.parameters.get("operation", "upper")
                    
                    if operation == "upper":
                        result_df[column_name] = result_df[column_name].str.upper()
                    elif operation == "lower":
                        result_df[column_name] = result_df[column_name].str.lower()
                    elif operation == "trim":
                        result_df[column_name] = result_df[column_name].str.strip()
                    elif operation == "replace":
                        old_value = rule.parameters.get("old_value", "")
                        new_value = rule.parameters.get("new_value", "")
                        result_df[column_name] = result_df[column_name].str.replace(old_value, new_value)
                        
                elif rule.rule_type == "aggregate":
                    group_by = rule.parameters.get("group_by", [])
                    agg_functions = rule.parameters.get("agg_functions", {})
                    
                    if group_by:
                        result_df = result_df.groupby(group_by).agg(agg_functions).reset_index()
                    else:
                        result_df = result_df.agg(agg_functions).to_frame().T
                        
                elif rule.rule_type == "sort":
                    columns = rule.parameters.get("columns", [])
                    ascending = rule.parameters.get("ascending", True)
                    result_df = result_df.sort_values(by=columns, ascending=ascending)
                    
                elif rule.rule_type == "fill_null":
                    column_name = rule.source_field
                    fill_value = rule.parameters.get("fill_value", "")
                    result_df[column_name] = result_df[column_name].fillna(fill_value)
                    
                elif rule.rule_type == "custom_function":
                    function_code = rule.expression
                    # Execute custom function (be careful with security)
                    exec(function_code, {"df": result_df, "pd": pd, "np": np})
            
            return result_df
            
        except Exception as e:
            logger.error("pandas_transformations_failed", error=str(e))
            raise
    
    async def _apply_spark_transformations(self, df: pd.DataFrame, rules: List[TransformationRule]) -> pd.DataFrame:
        """Apply transformations using Spark."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd use SparkSession and convert pandas to Spark DataFrame
            return await self._apply_pandas_transformations(df, rules)
            
        except Exception as e:
            logger.error("spark_transformations_failed", error=str(e))
            raise
    
    async def get_transformation_history(self, transformation_id: str) -> Dict[str, Any]:
        """Get transformation history."""
        try:
            if transformation_id in self.transformation_history:
                return self.transformation_history[transformation_id]
            else:
                raise ValueError(f"Transformation {transformation_id} not found")
                
        except Exception as e:
            logger.error("get_transformation_history_failed", error=str(e))
            raise