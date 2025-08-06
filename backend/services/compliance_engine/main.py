"""
Compliance Engine Service Main Entry Point
Phase 2: Policy Compliance Engine with Document Processing and NLP
"""

import asyncio
from datetime import datetime
from typing import Optional

import structlog
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.shared.config import get_settings
from backend.shared.database import get_async_db

from .compliance_analyzer import ComplianceAnalyzer
from .document_processor import DocumentProcessor
from .document_processor import DocumentStatus
from .nlp_extractor import NLPPolicyExtractor
from .rule_engine import ComplianceRule
from .rule_engine import RuleEngine
from .visual_rule_builder import router as rule_builder_router

settings = get_settings()
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PolicyCortex Compliance Engine",
    description="AI-Powered Policy Compliance Engine with Document Processing",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = None
nlp_extractor = None
compliance_analyzer = ComplianceAnalyzer()
rule_engine = RuleEngine()

# Include routers
app.include_router(rule_builder_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global document_processor, nlp_extractor

    # Initialize document processor
    document_processor = DocumentProcessor(
        storage_account_name=settings.azure.storage_account_name, container_name="policy-documents"
    )
    await document_processor.initialize()

    # Initialize NLP extractor
    nlp_extractor = NLPPolicyExtractor(
        azure_openai_endpoint=settings.ai.openai_endpoint,
        azure_openai_key=settings.ai.openai_api_key,
        azure_openai_deployment=settings.ai.openai_deployment_name,
        text_analytics_endpoint=settings.ai.text_analytics_endpoint,
        text_analytics_key=settings.ai.text_analytics_key,
    )

    logger.info("Compliance Engine services initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "compliance-engine",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Document Processing Endpoints
@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = "default",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload and process a policy document"""
    try:
        # Process document
        processed_doc = await document_processor.upload_document(
            file_content=file.file,
            filename=file.filename,
            tenant_id=tenant_id,
            metadata={"content_type": file.content_type, "uploaded_by": "api"},
        )

        # Extract policies in background
        background_tasks.add_task(
            extract_policies_from_document, processed_doc.document_id, tenant_id
        )

        return processed_doc.dict()

    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details and extracted policies"""
    try:
        document = await document_processor.get_document(document_id)

        # Get extracted policies if available
        policies = []
        if document.status == DocumentStatus.ANALYZED and document.extracted_text:
            policies = await nlp_extractor.extract_policies(document.extracted_text)

        return {"document": document.dict(), "policies": [p.dict() for p in policies]}

    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/documents")
async def list_documents(
    tenant_id: str = "default", status: Optional[DocumentStatus] = None, limit: int = 100
):
    """List documents for a tenant"""
    try:
        documents = await document_processor.list_documents(
            tenant_id=tenant_id, status=status, limit=limit
        )
        return [d.dict() for d in documents]

    except Exception as e:
        logger.error(f"Document listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Policy Extraction Endpoints
@app.post("/api/v1/policies/extract")
async def extract_policies_from_text(text: str):
    """Extract policies from provided text"""
    try:
        policies = await nlp_extractor.extract_policies(text)
        rules = await nlp_extractor.extract_compliance_rules(policies)

        return {"policies": [p.dict() for p in policies], "rules": [r.dict() for r in rules]}

    except Exception as e:
        logger.error(f"Policy extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Compliance Analysis Endpoints
class ComplianceAnalysisRequest(BaseModel):
    resources: list
    policies: list
    tenant_id: str = "default"
    real_time: bool = True


@app.post("/api/v1/compliance/analyze")
async def analyze_compliance(request: ComplianceAnalysisRequest):
    """Analyze compliance of resources against policies"""
    try:
        report = await compliance_analyzer.analyze_compliance(
            resources=request.resources,
            policies=request.policies,
            tenant_id=request.tenant_id,
            real_time=request.real_time,
        )
        return report.dict()

    except Exception as e:
        logger.error(f"Compliance analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/compliance/predict/{tenant_id}")
async def predict_compliance(tenant_id: str, days_ahead: int = 7):
    """Predict future compliance trends"""
    try:
        prediction = await compliance_analyzer.predict_future_compliance(
            tenant_id=tenant_id, days_ahead=days_ahead
        )
        return prediction

    except Exception as e:
        logger.error(f"Compliance prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rule Engine Endpoints
@app.post("/api/v1/rules")
async def create_rule(rule: ComplianceRule):
    """Create a new compliance rule"""
    try:
        rule_engine.add_rule(rule)
        return {"rule_id": rule.rule_id, "message": "Rule created successfully"}

    except Exception as e:
        logger.error(f"Rule creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/rules/{rule_id}")
async def update_rule(rule_id: str, updates: dict):
    """Update an existing rule"""
    try:
        success = rule_engine.update_rule(rule_id, updates)
        if success:
            return {"message": "Rule updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Rule not found")

    except Exception as e:
        logger.error(f"Rule update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a rule"""
    try:
        success = rule_engine.remove_rule(rule_id)
        if success:
            return {"message": "Rule deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Rule not found")

    except Exception as e:
        logger.error(f"Rule deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RuleEvaluationRequest(BaseModel):
    resource: dict
    rule_ids: Optional[list] = None
    execute_actions: bool = True


@app.post("/api/v1/rules/evaluate")
async def evaluate_rules(request: RuleEvaluationRequest):
    """Evaluate rules against a resource"""
    try:
        results = await rule_engine.evaluate_rules(
            resource=request.resource,
            rule_ids=request.rule_ids,
            execute_actions=request.execute_actions,
        )
        return [r.dict() for r in results]

    except Exception as e:
        logger.error(f"Rule evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rules/statistics")
async def get_rule_statistics():
    """Get rule execution statistics"""
    try:
        stats = rule_engine.get_execution_statistics()
        return stats

    except Exception as e:
        logger.error(f"Statistics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rules/export")
async def export_rules(format: str = "json"):
    """Export all rules"""
    try:
        exported = rule_engine.export_rules(format)
        return JSONResponse(content={"data": exported})

    except Exception as e:
        logger.error(f"Rule export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rules/import")
async def import_rules(data: str, format: str = "json"):
    """Import rules from data"""
    try:
        count = rule_engine.import_rules(data, format)
        return {"imported_count": count, "message": f"Imported {count} rules"}

    except Exception as e:
        logger.error(f"Rule import error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background Tasks
async def extract_policies_from_document(document_id: str, tenant_id: str):
    """Background task to extract policies from document"""
    try:
        # Get document
        document = await document_processor.get_document(document_id)

        if document.extracted_text:
            # Extract policies
            policies = await nlp_extractor.extract_policies(
                document.extracted_text,
                document_context={
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "filename": document.filename,
                },
            )

            # Convert to compliance rules
            rules = await nlp_extractor.extract_compliance_rules(policies)

            # Add rules to engine
            for rule in rules:
                compliance_rule = ComplianceRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    description=rule.description,
                    rule_type="policy",
                    conditions=[],  # Will be populated from evaluation_criteria
                    actions=[{"type": "alert", "level": "medium"}],
                    severity="medium",
                )
                rule_engine.add_rule(compliance_rule)

            logger.info(
                f"Extracted {len(policies)} policies and {len(rules)} rules from document {document_id}"
            )

    except Exception as e:
        logger.error(f"Background policy extraction error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
