"""
Data validation and quality service for PolicyCortex.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import structlog
import re

from ....shared.config import get_settings
from ..models import ValidationRule, ValidationResult

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataValidatorService:
    """Service for data validation and quality checks."""
    
    def __init__(self):
        self.settings = settings
        self.validation_history = {}
    
    async def validate_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                          validation_rules: List[ValidationRule],
                          quality_threshold: float = 0.8,
                          fail_on_error: bool = False,
                          user_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate data using specified rules."""
        try:
            validation_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            # Convert data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a dict or list of dicts")
            
            logger.info(
                "data_validation_started",
                validation_id=validation_id,
                input_rows=len(df),
                rule_count=len(validation_rules)
            )
            
            # Apply validation rules
            validation_results = []
            total_errors = 0
            total_warnings = 0
            
            for rule in validation_rules:
                result = await self._apply_validation_rule(df, rule)
                validation_results.append(result)
                total_errors += result.error_count
                total_warnings += result.warning_count
                
                if fail_on_error and result.error_count > 0:
                    raise ValueError(f"Validation failed for rule {rule.rule_type}: {result.message}")
            
            # Calculate quality score
            total_records = len(df)
            quality_score = max(0.0, 1.0 - (total_errors / total_records)) if total_records > 0 else 0.0
            
            # Record validation history
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.validation_history[validation_id] = {
                "validation_id": validation_id,
                "total_records": total_records,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "quality_score": quality_score,
                "processing_time": processing_time,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "data_validation_completed",
                validation_id=validation_id,
                quality_score=quality_score,
                total_errors=total_errors,
                processing_time=processing_time
            )
            
            return {
                "validation_id": validation_id,
                "validation_results": validation_results,
                "quality_score": quality_score,
                "total_records": total_records,
                "valid_records": total_records - total_errors,
                "invalid_records": total_errors
            }
            
        except Exception as e:
            logger.error("data_validation_failed", error=str(e))
            raise
    
    async def _apply_validation_rule(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule."""
        try:
            field = rule.field
            rule_type = rule.rule_type
            parameters = rule.parameters
            
            error_count = 0
            warning_count = 0
            message = ""
            details = {}
            
            if rule_type == "not_null":
                null_count = df[field].isnull().sum()
                error_count = null_count
                message = f"Found {null_count} null values in {field}"
                details["null_count"] = int(null_count)
                
            elif rule_type == "unique":
                duplicate_count = df[field].duplicated().sum()
                error_count = duplicate_count
                message = f"Found {duplicate_count} duplicate values in {field}"
                details["duplicate_count"] = int(duplicate_count)
                
            elif rule_type == "range_check":
                min_val = parameters.get("min_value")
                max_val = parameters.get("max_value")
                
                if min_val is not None and max_val is not None:
                    invalid_count = (~df[field].between(min_val, max_val)).sum()
                    error_count = invalid_count
                    message = f"Found {invalid_count} values outside range [{min_val}, {max_val}] in {field}"
                    details["invalid_count"] = int(invalid_count)
                    details["min_value"] = min_val
                    details["max_value"] = max_val
                    
            elif rule_type == "pattern_match":
                pattern = parameters.get("pattern")
                if pattern:
                    invalid_count = (~df[field].astype(str).str.match(pattern)).sum()
                    error_count = invalid_count
                    message = f"Found {invalid_count} values not matching pattern {pattern} in {field}"
                    details["invalid_count"] = int(invalid_count)
                    details["pattern"] = pattern
                    
            elif rule_type == "length_check":
                min_length = parameters.get("min_length", 0)
                max_length = parameters.get("max_length", float('inf'))
                
                lengths = df[field].astype(str).str.len()
                invalid_count = (~lengths.between(min_length, max_length)).sum()
                error_count = invalid_count
                message = f"Found {invalid_count} values with invalid length in {field}"
                details["invalid_count"] = int(invalid_count)
                details["min_length"] = min_length
                details["max_length"] = max_length
                
            elif rule_type == "data_type_check":
                expected_type = parameters.get("expected_type", "str")
                invalid_count = 0
                
                if expected_type == "int":
                    invalid_count = (~pd.to_numeric(df[field], errors='coerce').notna()).sum()
                elif expected_type == "float":
                    invalid_count = (~pd.to_numeric(df[field], errors='coerce').notna()).sum()
                elif expected_type == "datetime":
                    invalid_count = (~pd.to_datetime(df[field], errors='coerce').notna()).sum()
                elif expected_type == "bool":
                    invalid_count = (~df[field].astype(str).str.lower().isin(['true', 'false', '1', '0'])).sum()
                
                error_count = invalid_count
                message = f"Found {invalid_count} values with invalid type in {field}"
                details["invalid_count"] = int(invalid_count)
                details["expected_type"] = expected_type
                
            elif rule_type == "enum_check":
                valid_values = parameters.get("valid_values", [])
                invalid_count = (~df[field].isin(valid_values)).sum()
                error_count = invalid_count
                message = f"Found {invalid_count} values not in allowed set for {field}"
                details["invalid_count"] = int(invalid_count)
                details["valid_values"] = valid_values
                
            elif rule_type == "completeness_check":
                threshold = parameters.get("threshold", 0.9)
                completeness = (df[field].notna().sum() / len(df)) if len(df) > 0 else 0
                
                if completeness < threshold:
                    warning_count = 1
                    message = f"Completeness {completeness:.2f} below threshold {threshold} for {field}"
                else:
                    message = f"Completeness {completeness:.2f} meets threshold for {field}"
                
                details["completeness"] = completeness
                details["threshold"] = threshold
                
            elif rule_type == "statistical_outlier":
                method = parameters.get("method", "iqr")
                multiplier = parameters.get("multiplier", 1.5)
                
                numeric_values = pd.to_numeric(df[field], errors='coerce').dropna()
                
                if method == "iqr":
                    Q1 = numeric_values.quantile(0.25)
                    Q3 = numeric_values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    outlier_count = (~numeric_values.between(lower_bound, upper_bound)).sum()
                    
                elif method == "zscore":
                    z_scores = np.abs((numeric_values - numeric_values.mean()) / numeric_values.std())
                    outlier_count = (z_scores > multiplier).sum()
                    
                warning_count = outlier_count
                message = f"Found {outlier_count} statistical outliers in {field}"
                details["outlier_count"] = int(outlier_count)
                details["method"] = method
                
            # Determine status
            if error_count > 0:
                status = "failed"
            elif warning_count > 0:
                status = "warning"
            else:
                status = "passed"
            
            return ValidationResult(
                rule_name=rule_type,
                field=field,
                status=status,
                error_count=error_count,
                warning_count=warning_count,
                message=message,
                details=details
            )
            
        except Exception as e:
            logger.error("validation_rule_failed", rule_type=rule_type, field=field, error=str(e))
            return ValidationResult(
                rule_name=rule_type,
                field=field,
                status="error",
                error_count=1,
                warning_count=0,
                message=f"Validation rule failed: {str(e)}"
            )
    
    async def get_validation_history(self, validation_id: str) -> Dict[str, Any]:
        """Get validation history."""
        try:
            if validation_id in self.validation_history:
                return self.validation_history[validation_id]
            else:
                raise ValueError(f"Validation {validation_id} not found")
                
        except Exception as e:
            logger.error("get_validation_history_failed", error=str(e))
            raise