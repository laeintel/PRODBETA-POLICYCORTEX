"""
Data aggregation service for PolicyCortex.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import structlog

from ....shared.config import get_settings
from ..models import AggregationRule

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataAggregatorService:
    """Service for data aggregation operations."""
    
    def __init__(self):
        self.settings = settings
        self.aggregation_history = {}
    
    async def aggregate_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                           aggregation_rules: List[AggregationRule],
                           group_by_fields: List[str] = None,
                           filters: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate data using specified rules."""
        try:
            aggregation_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            # Convert data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a dict or list of dicts")
            
            logger.info(
                "data_aggregation_started",
                aggregation_id=aggregation_id,
                input_rows=len(df),
                rule_count=len(aggregation_rules)
            )
            
            # Apply filters
            if filters:
                df = self._apply_filters(df, filters)
            
            # Perform aggregation
            aggregated_df = await self._perform_aggregation(df, aggregation_rules, group_by_fields)
            
            # Convert to output format
            aggregated_data = aggregated_df.to_dict(orient="records")
            
            # Record aggregation history
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.aggregation_history[aggregation_id] = {
                "aggregation_id": aggregation_id,
                "input_rows": len(df),
                "output_rows": len(aggregated_df),
                "processing_time": processing_time,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "data_aggregation_completed",
                aggregation_id=aggregation_id,
                output_rows=len(aggregated_df),
                processing_time=processing_time
            )
            
            return {
                "aggregation_id": aggregation_id,
                "aggregated_data": aggregated_data,
                "record_count": len(aggregated_df),
                "processing_time_ms": int(processing_time * 1000)
            }
            
        except Exception as e:
            logger.error("data_aggregation_failed", error=str(e))
            raise
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        try:
            filtered_df = df.copy()
            
            for field, filter_config in filters.items():
                if isinstance(filter_config, dict):
                    operator = filter_config.get("operator", "eq")
                    value = filter_config.get("value")
                    
                    if operator == "eq":
                        filtered_df = filtered_df[filtered_df[field] == value]
                    elif operator == "ne":
                        filtered_df = filtered_df[filtered_df[field] != value]
                    elif operator == "gt":
                        filtered_df = filtered_df[filtered_df[field] > value]
                    elif operator == "gte":
                        filtered_df = filtered_df[filtered_df[field] >= value]
                    elif operator == "lt":
                        filtered_df = filtered_df[filtered_df[field] < value]
                    elif operator == "lte":
                        filtered_df = filtered_df[filtered_df[field] <= value]
                    elif operator == "in":
                        filtered_df = filtered_df[filtered_df[field].isin(value)]
                    elif operator == "not_in":
                        filtered_df = filtered_df[~filtered_df[field].isin(value)]
                    elif operator == "contains":
                        filtered_df = filtered_df[filtered_df[field].str.contains(value, na=False)]
                    elif operator == "startswith":
                        filtered_df = filtered_df[filtered_df[field].str.startswith(value, na=False)]
                    elif operator == "endswith":
                        filtered_df = filtered_df[filtered_df[field].str.endswith(value, na=False)]
                else:
                    # Simple equality filter
                    filtered_df = filtered_df[filtered_df[field] == filter_config]
            
            return filtered_df
            
        except Exception as e:
            logger.error("apply_filters_failed", error=str(e))
            raise
    
    async def _perform_aggregation(self, df: pd.DataFrame, rules: List[AggregationRule], 
                                 group_by_fields: List[str] = None) -> pd.DataFrame:
        """Perform aggregation based on rules."""
        try:
            # Build aggregation dictionary
            agg_dict = {}
            
            for rule in rules:
                field = rule.field
                function = rule.function
                alias = rule.alias or f"{field}_{function}"
                
                if function == "count":
                    agg_dict[alias] = (field, "count")
                elif function == "sum":
                    agg_dict[alias] = (field, "sum")
                elif function == "avg" or function == "mean":
                    agg_dict[alias] = (field, "mean")
                elif function == "min":
                    agg_dict[alias] = (field, "min")
                elif function == "max":
                    agg_dict[alias] = (field, "max")
                elif function == "std":
                    agg_dict[alias] = (field, "std")
                elif function == "var":
                    agg_dict[alias] = (field, "var")
                elif function == "median":
                    agg_dict[alias] = (field, "median")
                elif function == "first":
                    agg_dict[alias] = (field, "first")
                elif function == "last":
                    agg_dict[alias] = (field, "last")
                elif function == "nunique":
                    agg_dict[alias] = (field, "nunique")
                elif function == "size":
                    agg_dict[alias] = (field, "size")
                else:
                    # Default to count
                    agg_dict[alias] = (field, "count")
            
            # Perform aggregation
            if group_by_fields:
                # Group by aggregation
                result_df = df.groupby(group_by_fields).agg(**agg_dict).reset_index()
            else:
                # Overall aggregation
                result_df = df.agg(**agg_dict).to_frame().T
            
            return result_df
            
        except Exception as e:
            logger.error("perform_aggregation_failed", error=str(e))
            raise
    
    async def get_aggregation_history(self, aggregation_id: str) -> Dict[str, Any]:
        """Get aggregation history."""
        try:
            if aggregation_id in self.aggregation_history:
                return self.aggregation_history[aggregation_id]
            else:
                raise ValueError(f"Aggregation {aggregation_id} not found")
                
        except Exception as e:
            logger.error("get_aggregation_history_failed", error=str(e))
            raise