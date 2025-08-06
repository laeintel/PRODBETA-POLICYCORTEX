"""
Data Pipeline Module
Orchestrates data processing workflows
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import structlog
import json

from .data_connector import DataConnector
from .data_transformer import DataTransformer, TransformationRule
from .data_synchronizer import DataSynchronizer, SyncRule

logger = structlog.get_logger(__name__)

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepType(str, Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    SYNC = "sync"
    CUSTOM = "custom"

@dataclass
class PipelineStep:
    """Represents a step in a data pipeline"""
    id: str
    name: str
    type: StepType
    config: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    timeout_minutes: int = 30

@dataclass
class PipelineRun:
    """Represents a pipeline execution"""
    id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Pipeline:
    """Represents a data processing pipeline"""
    id: str
    name: str
    description: str
    steps: List[PipelineStep] = field(default_factory=list)
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

class DataPipeline:
    """
    Orchestrates complex data processing workflows
    """
    
    def __init__(self,
                 data_connector: DataConnector,
                 data_transformer: DataTransformer,
                 data_synchronizer: DataSynchronizer):
        self.data_connector = data_connector
        self.data_transformer = data_transformer
        self.data_synchronizer = data_synchronizer
        self.pipelines = {}
        self.pipeline_runs = []
        self.running_pipelines = set()
        self.custom_functions = {}
        
    async def create_pipeline(self, pipeline: Pipeline) -> bool:
        """Create a new pipeline"""
        try:
            # Validate pipeline
            validation_errors = await self._validate_pipeline(pipeline)
            if validation_errors:
                logger.error(f"Pipeline validation failed: {validation_errors}")
                return False
                
            self.pipelines[pipeline.id] = pipeline
            logger.info(f"Created pipeline: {pipeline.name} ({pipeline.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline.id}: {e}")
            return False
            
    async def run_pipeline(self, pipeline_id: str, params: Optional[Dict[str, Any]] = None) -> PipelineRun:
        """Execute a pipeline"""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        if pipeline_id in self.running_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} is already running")
            
        pipeline = self.pipelines[pipeline_id]
        
        if not pipeline.enabled:
            raise ValueError(f"Pipeline {pipeline_id} is disabled")
            
        # Create pipeline run
        run = PipelineRun(
            id=f"{pipeline_id}_{int(datetime.utcnow().timestamp())}",
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            metadata=params or {}
        )
        
        self.running_pipelines.add(pipeline_id)
        self.pipeline_runs.append(run)
        
        try:
            # Execute pipeline steps
            await self._execute_pipeline_steps(pipeline, run)
            run.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            run.status = PipelineStatus.FAILED
            run.errors.append(str(e))
            
        finally:
            run.end_time = datetime.utcnow()
            self.running_pipelines.discard(pipeline_id)
            
        return run
        
    async def _execute_pipeline_steps(self, pipeline: Pipeline, run: PipelineRun):
        """Execute pipeline steps in dependency order"""
        
        # Build dependency graph
        step_map = {step.id: step for step in pipeline.steps}
        dependency_graph = self._build_dependency_graph(pipeline.steps)
        
        # Execute steps in topological order
        executed_steps = set()
        step_data = {}  # Store data between steps
        
        while len(executed_steps) < len(pipeline.steps):
            # Find steps ready to execute
            ready_steps = []
            for step_id, deps in dependency_graph.items():
                if step_id not in executed_steps and all(dep in executed_steps for dep in deps):
                    step = step_map[step_id]
                    if step.enabled:
                        ready_steps.append(step)
                        
            if not ready_steps:
                # Check for circular dependencies
                remaining_steps = [s for s in pipeline.steps if s.id not in executed_steps and s.enabled]
                if remaining_steps:
                    raise Exception(f"Circular dependency detected or unresolvable dependencies in steps: {[s.id for s in remaining_steps]}")
                break
                
            # Execute ready steps (can be parallel if no data dependencies)
            for step in ready_steps:
                run.current_step = step.id
                
                try:
                    step_result = await self._execute_step(step, step_data, run)
                    step_data[step.id] = step_result
                    executed_steps.add(step.id)
                    run.steps_completed.append(step.id)
                    
                    logger.info(f"Completed step {step.name} ({step.id})")
                    
                except Exception as e:
                    logger.error(f"Step {step.name} ({step.id}) failed: {e}")
                    
                    # Retry if configured
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        logger.info(f"Retrying step {step.id} (attempt {step.retry_count}/{step.max_retries})")
                        # Add step back to be retried
                        continue
                    else:
                        run.steps_failed.append(step.id)
                        run.errors.append(f"Step {step.id}: {str(e)}")
                        raise e
                        
        run.current_step = None
        
    async def _execute_step(self, step: PipelineStep, step_data: Dict[str, Any], run: PipelineRun) -> Any:
        """Execute a single pipeline step"""
        
        logger.info(f"Executing step: {step.name} ({step.type.value})")
        
        # Set timeout
        timeout = step.timeout_minutes * 60
        
        try:
            if step.type == StepType.EXTRACT:
                return await asyncio.wait_for(self._execute_extract_step(step, step_data), timeout)
            elif step.type == StepType.TRANSFORM:
                return await asyncio.wait_for(self._execute_transform_step(step, step_data), timeout)
            elif step.type == StepType.LOAD:
                return await asyncio.wait_for(self._execute_load_step(step, step_data), timeout)
            elif step.type == StepType.VALIDATE:
                return await asyncio.wait_for(self._execute_validate_step(step, step_data), timeout)
            elif step.type == StepType.SYNC:
                return await asyncio.wait_for(self._execute_sync_step(step, step_data), timeout)
            elif step.type == StepType.CUSTOM:
                return await asyncio.wait_for(self._execute_custom_step(step, step_data), timeout)
            else:
                raise ValueError(f"Unknown step type: {step.type}")
                
        except asyncio.TimeoutError:
            raise Exception(f"Step {step.id} timed out after {step.timeout_minutes} minutes")
            
    async def _execute_extract_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> pd.DataFrame:
        """Execute data extraction step"""
        
        source = step.config.get('source')
        if not source:
            raise ValueError("Extract step requires 'source' in config")
            
        query = step.config.get('query')
        filters = step.config.get('filters', {})
        limit = step.config.get('limit')
        
        # Support referencing data from previous steps
        if filters and 'from_step' in filters:
            from_step = filters.pop('from_step')
            if from_step in step_data:
                previous_data = step_data[from_step]
                if isinstance(previous_data, pd.DataFrame):
                    # Use values from previous step as filters
                    filter_column = filters.get('filter_column')
                    if filter_column and filter_column in previous_data.columns:
                        filters['values'] = previous_data[filter_column].unique().tolist()
                        
        data = await self.data_connector.read_data(source, query, filters, limit)
        
        logger.info(f"Extracted {len(data)} records from {source}")
        return data
        
    async def _execute_transform_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> pd.DataFrame:
        """Execute data transformation step"""
        
        input_step = step.config.get('input_step')
        if not input_step or input_step not in step_data:
            raise ValueError(f"Transform step requires valid 'input_step' in config. Available: {list(step_data.keys())}")
            
        input_data = step_data[input_step]
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Transform step input must be a DataFrame")
            
        # Build transformation rules from config
        transformation_rules = []
        for transform_config in step.config.get('transformations', []):
            rule = self.data_transformer.create_transformation_rule(
                rule_id=f"{step.id}_{transform_config.get('id', len(transformation_rules))}",
                name=transform_config.get('name', 'Unnamed Transformation'),
                transformation_type=transform_config.get('type'),
                config=transform_config.get('config', {}),
                priority=transform_config.get('priority', 0)
            )
            transformation_rules.append(rule)
            
        transformed_data = await self.data_transformer.transform_data(input_data, transformation_rules)
        
        logger.info(f"Transformed data: {len(input_data)} -> {len(transformed_data)} records")
        return transformed_data
        
    async def _execute_load_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> bool:
        """Execute data loading step"""
        
        input_step = step.config.get('input_step')
        if not input_step or input_step not in step_data:
            raise ValueError(f"Load step requires valid 'input_step' in config. Available: {list(step_data.keys())}")
            
        input_data = step_data[input_step]
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Load step input must be a DataFrame")
            
        target = step.config.get('target')
        if not target:
            raise ValueError("Load step requires 'target' in config")
            
        table_name = step.config.get('table_name')
        mode = step.config.get('mode', 'append')
        
        success = await self.data_connector.write_data(target, input_data, table_name, mode)
        
        if success:
            logger.info(f"Loaded {len(input_data)} records to {target}")
        else:
            raise Exception(f"Failed to load data to {target}")
            
        return success
        
    async def _execute_validate_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation step"""
        
        input_step = step.config.get('input_step')
        if not input_step or input_step not in step_data:
            raise ValueError(f"Validate step requires valid 'input_step' in config. Available: {list(step_data.keys())}")
            
        input_data = step_data[input_step]
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Validate step input must be a DataFrame")
            
        # Perform validation
        validation_rules = step.config.get('validation_rules', [])
        validation_results = []
        
        for rule in validation_rules:
            rule_name = rule.get('name')
            rule_type = rule.get('type')
            
            if rule_type == 'not_null':
                columns = rule.get('columns', [])
                for column in columns:
                    if column in input_data.columns:
                        null_count = input_data[column].isnull().sum()
                        validation_results.append({
                            'rule': rule_name,
                            'type': rule_type,
                            'column': column,
                            'passed': null_count == 0,
                            'null_count': null_count
                        })
                        
            elif rule_type == 'unique':
                columns = rule.get('columns', [])
                for column in columns:
                    if column in input_data.columns:
                        duplicate_count = input_data.duplicated(subset=[column]).sum()
                        validation_results.append({
                            'rule': rule_name,
                            'type': rule_type,
                            'column': column,
                            'passed': duplicate_count == 0,
                            'duplicate_count': duplicate_count
                        })
                        
            elif rule_type == 'range':
                column = rule.get('column')
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                
                if column in input_data.columns:
                    out_of_range = 0
                    if min_val is not None:
                        out_of_range += (input_data[column] < min_val).sum()
                    if max_val is not None:
                        out_of_range += (input_data[column] > max_val).sum()
                        
                    validation_results.append({
                        'rule': rule_name,
                        'type': rule_type,
                        'column': column,
                        'passed': out_of_range == 0,
                        'out_of_range_count': out_of_range
                    })
                    
        # Check if validation passed
        failed_validations = [r for r in validation_results if not r['passed']]
        
        if failed_validations and step.config.get('fail_on_validation_error', True):
            raise Exception(f"Validation failed: {failed_validations}")
            
        return {
            'validation_results': validation_results,
            'validation_passed': len(failed_validations) == 0,
            'total_records': len(input_data)
        }
        
    async def _execute_sync_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data synchronization step"""
        
        sync_rule_id = step.config.get('sync_rule_id')
        if not sync_rule_id:
            raise ValueError("Sync step requires 'sync_rule_id' in config")
            
        sync_result = await self.data_synchronizer.sync_data(sync_rule_id)
        
        return {
            'sync_result': {
                'status': sync_result.status.value,
                'records_processed': sync_result.records_processed,
                'records_created': sync_result.records_created,
                'records_updated': sync_result.records_updated,
                'records_deleted': sync_result.records_deleted,
                'errors': sync_result.errors
            }
        }
        
    async def _execute_custom_step(self, step: PipelineStep, step_data: Dict[str, Any]) -> Any:
        """Execute custom function step"""
        
        function_name = step.config.get('function')
        if not function_name or function_name not in self.custom_functions:
            raise ValueError(f"Custom step requires valid 'function' in config. Available: {list(self.custom_functions.keys())}")
            
        custom_function = self.custom_functions[function_name]
        function_args = step.config.get('args', {})
        
        # Pass step data and run context to custom function
        context = {
            'step_data': step_data,
            'step_config': step.config,
            'data_connector': self.data_connector,
            'data_transformer': self.data_transformer,
            'data_synchronizer': self.data_synchronizer
        }
        
        return await custom_function(context, **function_args)
        
    def _build_dependency_graph(self, steps: List[PipelineStep]) -> Dict[str, List[str]]:
        """Build dependency graph for steps"""
        
        graph = {}
        for step in steps:
            graph[step.id] = step.depends_on.copy()
            
        return graph
        
    async def _validate_pipeline(self, pipeline: Pipeline) -> List[str]:
        """Validate pipeline configuration"""
        
        errors = []
        
        if not pipeline.steps:
            errors.append("Pipeline must have at least one step")
            return errors
            
        # Check for duplicate step IDs
        step_ids = [step.id for step in pipeline.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
            
        # Check dependencies
        for step in pipeline.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} depends on non-existent step {dep}")
                    
        # Check for circular dependencies
        if self._has_circular_dependencies(pipeline.steps):
            errors.append("Circular dependencies detected")
            
        return errors
        
    def _has_circular_dependencies(self, steps: List[PipelineStep]) -> bool:
        """Check for circular dependencies in steps"""
        
        graph = self._build_dependency_graph(steps)
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
                    
        return False
        
    def register_custom_function(self, name: str, function: Callable):
        """Register a custom function for use in pipelines"""
        self.custom_functions[name] = function
        logger.info(f"Registered custom function: {name}")
        
    def get_pipeline_status(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get pipeline status"""
        
        if pipeline_id:
            if pipeline_id not in self.pipelines:
                return {}
                
            pipeline = self.pipelines[pipeline_id]
            recent_runs = [r for r in self.pipeline_runs if r.pipeline_id == pipeline_id][-5:]
            
            return {
                'pipeline': {
                    'id': pipeline.id,
                    'name': pipeline.name,
                    'enabled': pipeline.enabled,
                    'step_count': len(pipeline.steps),
                    'running': pipeline_id in self.running_pipelines
                },
                'recent_runs': [
                    {
                        'id': r.id,
                        'status': r.status.value,
                        'start_time': r.start_time.isoformat(),
                        'end_time': r.end_time.isoformat() if r.end_time else None,
                        'steps_completed': len(r.steps_completed),
                        'steps_failed': len(r.steps_failed),
                        'current_step': r.current_step,
                        'errors': r.errors
                    }
                    for r in recent_runs
                ]
            }
        else:
            # Return overall status
            total_pipelines = len(self.pipelines)
            enabled_pipelines = sum(1 for p in self.pipelines.values() if p.enabled)
            running_pipelines = len(self.running_pipelines)
            
            recent_runs = [r for r in self.pipeline_runs if r.start_time > datetime.utcnow() - timedelta(hours=24)]
            successful_runs = sum(1 for r in recent_runs if r.status == PipelineStatus.COMPLETED)
            failed_runs = sum(1 for r in recent_runs if r.status == PipelineStatus.FAILED)
            
            return {
                'total_pipelines': total_pipelines,
                'enabled_pipelines': enabled_pipelines,
                'running_pipelines': running_pipelines,
                'last_24h': {
                    'total_runs': len(recent_runs),
                    'successful_runs': successful_runs,
                    'failed_runs': failed_runs,
                    'success_rate': successful_runs / len(recent_runs) if recent_runs else 0
                }
            }
            
    def create_pipeline_from_template(self, template_name: str, **params) -> Pipeline:
        """Create pipeline from predefined template"""
        
        templates = {
            'etl_basic': self._create_basic_etl_template,
            'data_validation': self._create_validation_template,
            'sync_to_warehouse': self._create_sync_template
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        return templates[template_name](**params)
        
    def _create_basic_etl_template(self, **params) -> Pipeline:
        """Create basic ETL pipeline template"""
        
        pipeline_id = params.get('pipeline_id', f'etl_{int(datetime.utcnow().timestamp())}')
        source = params.get('source', 'source_db')
        target = params.get('target', 'target_db')
        
        steps = [
            PipelineStep(
                id='extract',
                name='Extract Data',
                type=StepType.EXTRACT,
                config={
                    'source': source,
                    'query': params.get('query'),
                    'filters': params.get('filters', {})
                }
            ),
            PipelineStep(
                id='transform',
                name='Transform Data',
                type=StepType.TRANSFORM,
                config={
                    'input_step': 'extract',
                    'transformations': params.get('transformations', [])
                },
                depends_on=['extract']
            ),
            PipelineStep(
                id='load',
                name='Load Data',
                type=StepType.LOAD,
                config={
                    'input_step': 'transform',
                    'target': target,
                    'mode': params.get('load_mode', 'append')
                },
                depends_on=['transform']
            )
        ]
        
        return Pipeline(
            id=pipeline_id,
            name=params.get('name', 'Basic ETL Pipeline'),
            description=params.get('description', 'Extract, transform, and load data'),
            steps=steps
        )
        
    def _create_validation_template(self, **params) -> Pipeline:
        """Create data validation pipeline template"""
        
        pipeline_id = params.get('pipeline_id', f'validation_{int(datetime.utcnow().timestamp())}')
        
        steps = [
            PipelineStep(
                id='extract',
                name='Extract Data for Validation',
                type=StepType.EXTRACT,
                config={
                    'source': params.get('source'),
                    'query': params.get('query')
                }
            ),
            PipelineStep(
                id='validate',
                name='Validate Data Quality',
                type=StepType.VALIDATE,
                config={
                    'input_step': 'extract',
                    'validation_rules': params.get('validation_rules', []),
                    'fail_on_validation_error': params.get('fail_on_error', True)
                },
                depends_on=['extract']
            )
        ]
        
        return Pipeline(
            id=pipeline_id,
            name=params.get('name', 'Data Validation Pipeline'),
            description=params.get('description', 'Validate data quality'),
            steps=steps
        )
        
    def _create_sync_template(self, **params) -> Pipeline:
        """Create data synchronization pipeline template"""
        
        pipeline_id = params.get('pipeline_id', f'sync_{int(datetime.utcnow().timestamp())}')
        
        steps = [
            PipelineStep(
                id='sync',
                name='Synchronize Data',
                type=StepType.SYNC,
                config={
                    'sync_rule_id': params.get('sync_rule_id')
                }
            )
        ]
        
        return Pipeline(
            id=pipeline_id,
            name=params.get('name', 'Data Sync Pipeline'),
            description=params.get('description', 'Synchronize data between sources'),
            steps=steps
        )