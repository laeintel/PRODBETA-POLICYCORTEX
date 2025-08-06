"""
Data Integration Service Main Entry Point
Phase 4: Data Integration Layer
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

import structlog
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.shared.config import get_settings
from backend.shared.database import get_async_db

from .data_connector import DataConnector, DataSourceType
from .data_transformer import DataTransformer, TransformationRule, TransformationType
from .data_synchronizer import DataSynchronizer, SyncRule, SyncMode, SyncDirection
from .data_pipeline import DataPipeline, Pipeline, PipelineStep, StepType

settings = get_settings()
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PolicyCortex Data Integration Service",
    description="Universal Data Integration and Processing Layer",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data integration components
data_connector = DataConnector()
data_transformer = DataTransformer()
data_synchronizer = DataSynchronizer(data_connector)
data_pipeline = DataPipeline(data_connector, data_transformer, data_synchronizer)

# Pydantic models
class DataSourceConfig(BaseModel):
    name: str
    type: DataSourceType
    connection_config: Dict[str, Any]

class TransformationConfig(BaseModel):
    rule_id: str
    name: str
    type: TransformationType
    config: Dict[str, Any]
    priority: int = 0

class SyncRuleConfig(BaseModel):
    rule_id: str
    name: str
    source: str
    target: str
    mode: SyncMode = SyncMode.INCREMENTAL
    direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET
    schedule: Optional[str] = None
    filters: Dict[str, Any] = {}
    field_mapping: Dict[str, str] = {}

class PipelineConfig(BaseModel):
    pipeline_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    schedule: Optional[str] = None
    enabled: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Data Integration Service starting up...")
    
    # Register custom pipeline functions
    await register_custom_functions()
    
    logger.info("Data Integration Service initialized")

async def register_custom_functions():
    """Register custom functions for pipelines"""
    
    async def data_quality_check(context: Dict[str, Any], **kwargs):
        """Custom function for comprehensive data quality checks"""
        step_data = context['step_data']
        input_step = kwargs.get('input_step')
        
        if input_step not in step_data:
            raise ValueError(f"Input step {input_step} not found")
            
        data = step_data[input_step]
        
        quality_report = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'missing_data_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict()
        }
        
        return quality_report
    
    async def data_sampling(context: Dict[str, Any], **kwargs):
        """Custom function for data sampling"""
        step_data = context['step_data']
        input_step = kwargs.get('input_step')
        sample_size = kwargs.get('sample_size', 1000)
        method = kwargs.get('method', 'random')
        
        if input_step not in step_data:
            raise ValueError(f"Input step {input_step} not found")
            
        data = step_data[input_step]
        
        if method == 'random':
            sampled_data = data.sample(n=min(sample_size, len(data)))
        elif method == 'stratified':
            stratify_column = kwargs.get('stratify_column')
            if stratify_column and stratify_column in data.columns:
                sampled_data = data.groupby(stratify_column).apply(
                    lambda x: x.sample(n=min(sample_size // data[stratify_column].nunique(), len(x)))
                ).reset_index(drop=True)
            else:
                sampled_data = data.sample(n=min(sample_size, len(data)))
        else:
            sampled_data = data.head(sample_size)
            
        return sampled_data
    
    data_pipeline.register_custom_function('data_quality_check', data_quality_check)
    data_pipeline.register_custom_function('data_sampling', data_sampling)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-integration",
        "timestamp": datetime.utcnow().isoformat()
    }

# Data Source Management Endpoints
@app.post("/api/v1/datasources/connect")
async def connect_data_source(config: DataSourceConfig):
    """Connect to a data source"""
    try:
        success = await data_connector.connect(
            config.name,
            config.type,
            config.connection_config
        )
        
        if success:
            return {"message": f"Connected to {config.name}", "status": "connected"}
        else:
            raise HTTPException(status_code=400, detail="Failed to connect to data source")
            
    except Exception as e:
        logger.error(f"Failed to connect to data source {config.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/datasources/{source_name}")
async def disconnect_data_source(source_name: str):
    """Disconnect from a data source"""
    try:
        success = await data_connector.disconnect(source_name)
        
        if success:
            return {"message": f"Disconnected from {source_name}", "status": "disconnected"}
        else:
            raise HTTPException(status_code=404, detail="Data source not found")
            
    except Exception as e:
        logger.error(f"Failed to disconnect from data source {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/datasources")
async def list_data_sources():
    """List connected data sources"""
    try:
        connections = {}
        for name, conn_info in data_connector.connections.items():
            connections[name] = {
                'type': conn_info['type'].value,
                'connected_at': conn_info['connected_at'].isoformat(),
                'config_keys': list(conn_info['config'].keys())
            }
        return connections
        
    except Exception as e:
        logger.error(f"Failed to list data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Operations Endpoints
@app.post("/api/v1/data/read")
async def read_data(source_name: str,
                   query: Optional[str] = None,
                   filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None):
    """Read data from a connected source"""
    try:
        data = await data_connector.read_data(source_name, query, filters, limit)
        
        return {
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": data.columns.tolist(),
            "data": data.head(100).to_dict('records'),  # First 100 rows
            "preview": True if len(data) > 100 else False
        }
        
    except Exception as e:
        logger.error(f"Failed to read data from {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/write")
async def write_data(source_name: str,
                    data: List[Dict[str, Any]],
                    table_name: Optional[str] = None,
                    mode: str = "append"):
    """Write data to a connected source"""
    try:
        df = pd.DataFrame(data)
        success = await data_connector.write_data(source_name, df, table_name, mode)
        
        if success:
            return {"message": f"Successfully wrote {len(data)} records", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Failed to write data")
            
    except Exception as e:
        logger.error(f"Failed to write data to {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/profile")
async def profile_data(source_name: str,
                      query: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None):
    """Generate data quality profile"""
    try:
        data = await data_connector.read_data(source_name, query, filters)
        profile = await data_transformer.profile_data(data)
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to profile data from {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Transformation Endpoints
@app.post("/api/v1/transform/apply")
async def apply_transformations(source_name: str,
                              transformations: List[TransformationConfig],
                              output_source: Optional[str] = None):
    """Apply transformations to data"""
    try:
        # Read source data
        data = await data_connector.read_data(source_name)
        
        # Create transformation rules
        rules = []
        for transform_config in transformations:
            rule = data_transformer.create_transformation_rule(
                transform_config.rule_id,
                transform_config.name,
                transform_config.type,
                transform_config.config,
                transform_config.priority
            )
            rules.append(rule)
            
        # Apply transformations
        transformed_data = await data_transformer.transform_data(data, rules)
        
        # Write to output source if specified
        if output_source:
            success = await data_connector.write_data(output_source, transformed_data)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to write transformed data")
                
        return {
            "message": "Transformations applied successfully",
            "input_rows": len(data),
            "output_rows": len(transformed_data),
            "transformations_applied": len(rules),
            "stats": data_transformer.get_transformation_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to apply transformations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/transform/stats")
async def get_transformation_stats():
    """Get transformation performance statistics"""
    return data_transformer.get_transformation_stats()

# Data Synchronization Endpoints
@app.post("/api/v1/sync/rules")
async def create_sync_rule(config: SyncRuleConfig):
    """Create a data synchronization rule"""
    try:
        rule = data_synchronizer.create_sync_rule(
            config.rule_id,
            config.name,
            config.source,
            config.target,
            config.mode,
            config.direction,
            schedule=config.schedule,
            filters=config.filters,
            field_mapping=config.field_mapping
        )
        
        success = await data_synchronizer.add_sync_rule(rule)
        
        if success:
            return {"message": f"Sync rule {config.rule_id} created", "rule_id": config.rule_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to create sync rule")
            
    except Exception as e:
        logger.error(f"Failed to create sync rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sync/execute/{rule_id}")
async def execute_sync(rule_id: str, background_tasks: BackgroundTasks):
    """Execute a synchronization rule"""
    try:
        # Run sync in background
        background_tasks.add_task(run_sync_task, rule_id)
        
        return {"message": f"Sync {rule_id} initiated", "status": "running"}
        
    except Exception as e:
        logger.error(f"Failed to execute sync {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_sync_task(rule_id: str):
    """Background task for running sync"""
    try:
        result = await data_synchronizer.sync_data(rule_id)
        logger.info(f"Sync {rule_id} completed with status {result.status}")
    except Exception as e:
        logger.error(f"Sync {rule_id} failed: {e}")

@app.get("/api/v1/sync/status")
async def get_sync_status(rule_id: Optional[str] = None):
    """Get synchronization status"""
    return data_synchronizer.get_sync_status(rule_id)

@app.get("/api/v1/sync/history")
async def get_sync_history(rule_id: Optional[str] = None, limit: int = 100):
    """Get synchronization history"""
    return data_synchronizer.get_sync_history(rule_id, limit)

# Data Pipeline Endpoints
@app.post("/api/v1/pipelines")
async def create_pipeline(config: PipelineConfig):
    """Create a data pipeline"""
    try:
        # Convert step configs to PipelineStep objects
        steps = []
        for step_config in config.steps:
            step = PipelineStep(
                id=step_config['id'],
                name=step_config['name'],
                type=StepType(step_config['type']),
                config=step_config.get('config', {}),
                depends_on=step_config.get('depends_on', []),
                enabled=step_config.get('enabled', True),
                max_retries=step_config.get('max_retries', 3),
                timeout_minutes=step_config.get('timeout_minutes', 30)
            )
            steps.append(step)
            
        pipeline = Pipeline(
            id=config.pipeline_id,
            name=config.name,
            description=config.description,
            steps=steps,
            schedule=config.schedule,
            enabled=config.enabled
        )
        
        success = await data_pipeline.create_pipeline(pipeline)
        
        if success:
            return {"message": f"Pipeline {config.pipeline_id} created", "pipeline_id": config.pipeline_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to create pipeline")
            
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/pipelines/{pipeline_id}/run")
async def run_pipeline(pipeline_id: str, 
                      background_tasks: BackgroundTasks,
                      params: Optional[Dict[str, Any]] = None):
    """Execute a pipeline"""
    try:
        # Run pipeline in background
        background_tasks.add_task(run_pipeline_task, pipeline_id, params)
        
        return {"message": f"Pipeline {pipeline_id} initiated", "status": "running"}
        
    except Exception as e:
        logger.error(f"Failed to run pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_pipeline_task(pipeline_id: str, params: Optional[Dict[str, Any]] = None):
    """Background task for running pipeline"""
    try:
        result = await data_pipeline.run_pipeline(pipeline_id, params)
        logger.info(f"Pipeline {pipeline_id} completed with status {result.status}")
    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} failed: {e}")

@app.get("/api/v1/pipelines/status")
async def get_pipeline_status(pipeline_id: Optional[str] = None):
    """Get pipeline status"""
    return data_pipeline.get_pipeline_status(pipeline_id)

@app.post("/api/v1/pipelines/template/{template_name}")
async def create_pipeline_from_template(template_name: str, **params):
    """Create pipeline from template"""
    try:
        pipeline = data_pipeline.create_pipeline_from_template(template_name, **params)
        success = await data_pipeline.create_pipeline(pipeline)
        
        if success:
            return {
                "message": f"Pipeline created from template {template_name}",
                "pipeline_id": pipeline.id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create pipeline from template")
            
    except Exception as e:
        logger.error(f"Failed to create pipeline from template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility Endpoints
@app.get("/api/v1/datatypes/detect")
async def detect_data_types(source_name: str, sample_size: int = 1000):
    """Detect optimal data types for a dataset"""
    try:
        data = await data_connector.read_data(source_name, limit=sample_size)
        suggestions = await data_transformer.detect_data_types(data)
        
        return {
            "source": source_name,
            "sample_size": len(data),
            "type_suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Failed to detect data types for {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/upload")
async def upload_data_file(file: UploadFile = File(...),
                          source_name: Optional[str] = None):
    """Upload and process data file"""
    try:
        contents = await file.read()
        
        # Determine file type and parse
        if file.filename.endswith('.csv'):
            import io
            data = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            import json
            data = pd.DataFrame(json.loads(contents))
        elif file.filename.endswith('.xlsx'):
            import io
            data = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # If source_name provided, write to that source
        if source_name:
            success = await data_connector.write_data(source_name, data)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to write uploaded data")
                
        return {
            "filename": file.filename,
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": data.columns.tolist(),
            "preview": data.head(10).to_dict('records'),
            "written_to_source": source_name if source_name else None
        }
        
    except Exception as e:
        logger.error(f"Failed to upload data file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)