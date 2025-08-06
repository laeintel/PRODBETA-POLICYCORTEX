"""
Data Transformer Module
Handles data transformation and cleaning operations
"""

import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class TransformationType(str, Enum):
    """Types of transformations"""

    CLEAN = "clean"
    NORMALIZE = "normalize"
    AGGREGATE = "aggregate"
    PIVOT = "pivot"
    MERGE = "merge"
    FILTER = "filter"
    ENRICH = "enrich"
    VALIDATE = "validate"


@dataclass
class TransformationRule:
    """Represents a data transformation rule"""

    id: str
    name: str
    type: TransformationType
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 0


class DataTransformer:
    """
    Handles data transformation, cleaning, and enrichment
    """

    def __init__(self):
        self.transformation_rules = []
        self.data_types_cache = {}
        self.transformation_stats = {}

    async def transform_data(
        self, data: pd.DataFrame, transformations: List[TransformationRule]
    ) -> pd.DataFrame:
        """
        Apply transformations to data

        Args:
            data: Input DataFrame
            transformations: List of transformation rules

        Returns:
            Transformed DataFrame
        """

        transformed_data = data.copy()

        # Sort transformations by priority
        transformations = sorted(transformations, key=lambda x: x.priority)

        for rule in transformations:
            if not rule.enabled:
                continue

            try:
                start_time = datetime.utcnow()

                if rule.type == TransformationType.CLEAN:
                    transformed_data = await self._clean_data(transformed_data, rule.config)
                elif rule.type == TransformationType.NORMALIZE:
                    transformed_data = await self._normalize_data(transformed_data, rule.config)
                elif rule.type == TransformationType.AGGREGATE:
                    transformed_data = await self._aggregate_data(transformed_data, rule.config)
                elif rule.type == TransformationType.PIVOT:
                    transformed_data = await self._pivot_data(transformed_data, rule.config)
                elif rule.type == TransformationType.FILTER:
                    transformed_data = await self._filter_data(transformed_data, rule.config)
                elif rule.type == TransformationType.ENRICH:
                    transformed_data = await self._enrich_data(transformed_data, rule.config)
                elif rule.type == TransformationType.VALIDATE:
                    transformed_data = await self._validate_data(transformed_data, rule.config)

                # Track transformation performance
                duration = (datetime.utcnow() - start_time).total_seconds()
                self.transformation_stats[rule.id] = {
                    "duration": duration,
                    "rows_before": len(data),
                    "rows_after": len(transformed_data),
                    "last_run": datetime.utcnow().isoformat(),
                }

                logger.info(f"Applied transformation {rule.name} ({rule.type})")

            except Exception as e:
                logger.error(f"Transformation {rule.name} failed: {e}")
                # Continue with other transformations

        return transformed_data

    async def _clean_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data based on configuration"""

        cleaned_data = data.copy()

        # Remove duplicates
        if config.get("remove_duplicates", False):
            subset = config.get("duplicate_columns")
            cleaned_data = cleaned_data.drop_duplicates(subset=subset)

        # Handle missing values
        if "missing_values" in config:
            missing_config = config["missing_values"]

            if missing_config.get("strategy") == "drop":
                cleaned_data = cleaned_data.dropna(subset=missing_config.get("columns"))
            elif missing_config.get("strategy") == "fill":
                fill_values = missing_config.get("fill_values", {})
                for column, value in fill_values.items():
                    if column in cleaned_data.columns:
                        cleaned_data[column] = cleaned_data[column].fillna(value)

        # Remove outliers
        if config.get("remove_outliers", False):
            for column in config.get("outlier_columns", []):
                if column in cleaned_data.columns and cleaned_data[column].dtype in [
                    "int64",
                    "float64",
                ]:
                    Q1 = cleaned_data[column].quantile(0.25)
                    Q3 = cleaned_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_data = cleaned_data[
                        (cleaned_data[column] >= lower_bound)
                        & (cleaned_data[column] <= upper_bound)
                    ]

        # Standardize text
        if config.get("standardize_text", False):
            text_columns = config.get("text_columns", [])
            for column in text_columns:
                if column in cleaned_data.columns:
                    cleaned_data[column] = cleaned_data[column].str.lower().str.strip()

        return cleaned_data

    async def _normalize_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Normalize data values"""

        normalized_data = data.copy()

        # Min-max normalization
        if "min_max_columns" in config:
            for column in config["min_max_columns"]:
                if column in normalized_data.columns:
                    min_val = normalized_data[column].min()
                    max_val = normalized_data[column].max()
                    if max_val != min_val:
                        normalized_data[column] = (normalized_data[column] - min_val) / (
                            max_val - min_val
                        )

        # Z-score normalization
        if "zscore_columns" in config:
            for column in config["zscore_columns"]:
                if column in normalized_data.columns:
                    mean_val = normalized_data[column].mean()
                    std_val = normalized_data[column].std()
                    if std_val != 0:
                        normalized_data[column] = (normalized_data[column] - mean_val) / std_val

        # Unit scaling
        if "unit_scale_columns" in config:
            for column in config["unit_scale_columns"]:
                if column in normalized_data.columns:
                    max_val = abs(normalized_data[column]).max()
                    if max_val != 0:
                        normalized_data[column] = normalized_data[column] / max_val

        return normalized_data

    async def _aggregate_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate data based on grouping"""

        group_by = config.get("group_by", [])
        if not group_by:
            return data

        aggregations = config.get("aggregations", {})
        if not aggregations:
            return data

        try:
            # Perform grouping and aggregation
            grouped = data.groupby(group_by)

            agg_result = pd.DataFrame()
            for column, agg_funcs in aggregations.items():
                if column in data.columns:
                    if isinstance(agg_funcs, str):
                        agg_funcs = [agg_funcs]

                    for func in agg_funcs:
                        if func == "count":
                            agg_result[f"{column}_{func}"] = grouped[column].count()
                        elif func == "sum":
                            agg_result[f"{column}_{func}"] = grouped[column].sum()
                        elif func == "mean":
                            agg_result[f"{column}_{func}"] = grouped[column].mean()
                        elif func == "median":
                            agg_result[f"{column}_{func}"] = grouped[column].median()
                        elif func == "std":
                            agg_result[f"{column}_{func}"] = grouped[column].std()
                        elif func == "min":
                            agg_result[f"{column}_{func}"] = grouped[column].min()
                        elif func == "max":
                            agg_result[f"{column}_{func}"] = grouped[column].max()

            return agg_result.reset_index()

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return data

    async def _pivot_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Pivot data based on configuration"""

        try:
            index = config.get("index")
            columns = config.get("columns")
            values = config.get("values")

            if not all([index, columns, values]):
                return data

            pivoted = data.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=config.get("aggfunc", "mean"),
                fill_value=config.get("fill_value", 0),
            )

            # Flatten multi-level columns if needed
            if config.get("flatten_columns", True):
                pivoted.columns = ["_".join(str(col).strip() for col in pivoted.columns.values)]

            return pivoted.reset_index()

        except Exception as e:
            logger.error(f"Pivot failed: {e}")
            return data

    async def _filter_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on conditions"""

        filtered_data = data.copy()

        # Apply filters
        filters = config.get("filters", [])
        for filter_config in filters:
            column = filter_config.get("column")
            operator = filter_config.get("operator")
            value = filter_config.get("value")

            if not all([column, operator]) or column not in filtered_data.columns:
                continue

            if operator == "equals":
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == "not_equals":
                filtered_data = filtered_data[filtered_data[column] != value]
            elif operator == "greater_than":
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == "less_than":
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == "greater_equal":
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif operator == "less_equal":
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif operator == "contains":
                filtered_data = filtered_data[
                    filtered_data[column].str.contains(str(value), na=False)
                ]
            elif operator == "in":
                if isinstance(value, list):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
            elif operator == "not_null":
                filtered_data = filtered_data[filtered_data[column].notna()]
            elif operator == "null":
                filtered_data = filtered_data[filtered_data[column].isna()]

        # Date range filters
        if "date_range" in config:
            date_config = config["date_range"]
            date_column = date_config.get("column")
            start_date = date_config.get("start_date")
            end_date = date_config.get("end_date")

            if date_column in filtered_data.columns:
                filtered_data[date_column] = pd.to_datetime(filtered_data[date_column])

                if start_date:
                    filtered_data = filtered_data[
                        filtered_data[date_column] >= pd.to_datetime(start_date)
                    ]
                if end_date:
                    filtered_data = filtered_data[
                        filtered_data[date_column] <= pd.to_datetime(end_date)
                    ]

        return filtered_data

    async def _enrich_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Enrich data with additional information"""

        enriched_data = data.copy()

        # Add calculated columns
        if "calculated_columns" in config:
            for calc_config in config["calculated_columns"]:
                column_name = calc_config.get("name")
                formula = calc_config.get("formula")

                if column_name and formula:
                    try:
                        # Simple formula evaluation (extend as needed)
                        if formula == "current_timestamp":
                            enriched_data[column_name] = datetime.utcnow()
                        elif "+" in formula:
                            parts = formula.split("+")
                            if len(parts) == 2 and all(
                                p.strip() in enriched_data.columns for p in parts
                            ):
                                enriched_data[column_name] = (
                                    enriched_data[parts[0].strip()]
                                    + enriched_data[parts[1].strip()]
                                )
                        elif "*" in formula:
                            parts = formula.split("*")
                            if len(parts) == 2 and all(
                                p.strip() in enriched_data.columns for p in parts
                            ):
                                enriched_data[column_name] = (
                                    enriched_data[parts[0].strip()]
                                    * enriched_data[parts[1].strip()]
                                )

                    except Exception as e:
                        logger.warning(f"Failed to calculate column {column_name}: {e}")

        # Add lookup data
        if "lookups" in config:
            for lookup_config in config["lookups"]:
                lookup_column = lookup_config.get("column")
                lookup_data = lookup_config.get("data", {})
                new_column = lookup_config.get("new_column")

                if lookup_column in enriched_data.columns and new_column:
                    enriched_data[new_column] = enriched_data[lookup_column].map(lookup_data)

        # Add row numbers
        if config.get("add_row_number", False):
            enriched_data["row_number"] = range(1, len(enriched_data) + 1)

        # Add hash columns for deduplication
        if config.get("add_hash", False):
            hash_columns = config.get("hash_columns", enriched_data.columns.tolist())
            enriched_data["record_hash"] = pd.util.hash_pandas_object(
                enriched_data[hash_columns], index=False
            )

        return enriched_data

    async def _validate_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Validate data quality and add validation results"""

        validated_data = data.copy()
        validation_results = []

        # Schema validation
        if "schema" in config:
            schema = config["schema"]
            for column, rules in schema.items():
                if column in validated_data.columns:
                    # Data type validation
                    expected_type = rules.get("type")
                    if expected_type:
                        try:
                            if expected_type == "int":
                                validated_data[column] = pd.to_numeric(
                                    validated_data[column], errors="coerce"
                                )
                            elif expected_type == "float":
                                validated_data[column] = pd.to_numeric(
                                    validated_data[column], errors="coerce"
                                )
                            elif expected_type == "datetime":
                                validated_data[column] = pd.to_datetime(
                                    validated_data[column], errors="coerce"
                                )
                        except Exception as e:
                            validation_results.append(
                                {"column": column, "rule": "type_conversion", "error": str(e)}
                            )

                    # Range validation
                    if "min_value" in rules or "max_value" in rules:
                        min_val = rules.get("min_value")
                        max_val = rules.get("max_value")

                        if min_val is not None:
                            invalid_rows = validated_data[validated_data[column] < min_val]
                            if not invalid_rows.empty:
                                validation_results.append(
                                    {
                                        "column": column,
                                        "rule": "min_value",
                                        "invalid_count": len(invalid_rows),
                                    }
                                )

                        if max_val is not None:
                            invalid_rows = validated_data[validated_data[column] > max_val]
                            if not invalid_rows.empty:
                                validation_results.append(
                                    {
                                        "column": column,
                                        "rule": "max_value",
                                        "invalid_count": len(invalid_rows),
                                    }
                                )

        # Business rule validation
        if "business_rules" in config:
            for rule in config["business_rules"]:
                rule_name = rule.get("name")
                condition = rule.get("condition")

                # Simple condition evaluation (extend as needed)
                try:
                    if condition and rule_name:
                        # Example: "column1 > column2"
                        invalid_rows = validated_data.query(f"not ({condition})")
                        if not invalid_rows.empty:
                            validation_results.append(
                                {
                                    "rule": rule_name,
                                    "invalid_count": len(invalid_rows),
                                    "condition": condition,
                                }
                            )
                except Exception as e:
                    validation_results.append(
                        {"rule": rule_name, "error": f"Rule evaluation failed: {e}"}
                    )

        # Add validation metadata
        if config.get("add_validation_metadata", False):
            validated_data["_validation_timestamp"] = datetime.utcnow()
            validated_data["_validation_passed"] = True  # Mark all as passed for now

        # Store validation results
        self.transformation_stats["validation_results"] = validation_results

        return validated_data

    def create_transformation_rule(
        self,
        rule_id: str,
        name: str,
        transformation_type: TransformationType,
        config: Dict[str, Any],
        priority: int = 0,
    ) -> TransformationRule:
        """Create a new transformation rule"""

        return TransformationRule(
            id=rule_id, name=name, type=transformation_type, config=config, priority=priority
        )

    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation performance statistics"""
        return self.transformation_stats

    async def detect_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect and suggest optimal data types"""

        suggestions = {}

        for column in data.columns:
            current_type = str(data[column].dtype)

            # Check if can be converted to numeric
            if current_type == "object":
                try:
                    pd.to_numeric(data[column], errors="raise")
                    if data[column].str.contains(r"\.").any():
                        suggestions[column] = "float64"
                    else:
                        suggestions[column] = "int64"
                    continue
                except (ValueError, AttributeError):
                    pass

                # Check if can be converted to datetime
                try:
                    pd.to_datetime(data[column], errors="raise")
                    suggestions[column] = "datetime64[ns]"
                    continue
                except (ValueError, AttributeError):
                    pass

                # Check if boolean
                unique_values = data[column].str.lower().unique()
                if len(unique_values) <= 2 and all(
                    v in ["true", "false", "1", "0", "yes", "no"]
                    for v in unique_values
                    if pd.notna(v)
                ):
                    suggestions[column] = "bool"
                    continue

            suggestions[column] = current_type

        return suggestions

    async def profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality profile"""

        profile = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "columns": {},
        }

        for column in data.columns:
            col_profile = {
                "dtype": str(data[column].dtype),
                "null_count": data[column].isnull().sum(),
                "null_percentage": (data[column].isnull().sum() / len(data)) * 100,
                "unique_count": data[column].nunique(),
                "unique_percentage": (data[column].nunique() / len(data)) * 100,
            }

            # Add stats for numeric columns
            if data[column].dtype in ["int64", "float64"]:
                col_profile.update(
                    {
                        "min": data[column].min(),
                        "max": data[column].max(),
                        "mean": data[column].mean(),
                        "median": data[column].median(),
                        "std": data[column].std(),
                    }
                )

            # Add stats for text columns
            elif data[column].dtype == "object":
                col_profile.update(
                    {
                        "avg_length": data[column].astype(str).str.len().mean(),
                        "min_length": data[column].astype(str).str.len().min(),
                        "max_length": data[column].astype(str).str.len().max(),
                    }
                )

            profile["columns"][column] = col_profile

        return profile
