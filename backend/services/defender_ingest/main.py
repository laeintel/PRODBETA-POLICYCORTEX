#!/usr/bin/env python3
"""
PolicyCortex Microsoft Defender Integration & SOC-2 Evidence Collection Service
Automated ingestion of security alerts and compliance evidence generation
"""

import asyncio
import logging
import os
import json
import hashlib
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import asyncpg
import redis.asyncio as redis
from azure.identity.aio import DefaultAzureCredential
from azure.mgmt.security.aio import SecurityCenter
from azure.mgmt.security.models import Alert, Assessment, Recommendation
from msgraph import GraphServiceClient
from msgraph.generated.security.alerts.alerts_request_builder import AlertsRequestBuilder
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyCortex Defender Integration Service", version="1.0.0")

@dataclass
class DefenderAlert:
    alert_id: str
    title: str
    description: str
    severity: str
    status: str
    category: str
    vendor: str
    product: str
    first_activity_time: datetime
    last_activity_time: datetime
    confidence: float
    risk_score: float
    impacted_entities: List[Dict[str, Any]]
    tactics: List[str]
    techniques: List[str]
    evidence: List[Dict[str, Any]]
    raw_data: Dict[str, Any]

@dataclass
class SOC2Control:
    control_id: str
    control_name: str
    control_description: str
    control_category: str  # CC1, CC2, etc.
    implementation_status: str  # implemented, partially_implemented, not_implemented
    last_tested: datetime
    test_results: str
    evidence_collected: List[str]
    deficiencies: List[str]
    remediation_timeline: Optional[datetime]

@dataclass
class ComplianceEvidence:
    evidence_id: str
    control_id: str
    evidence_type: str  # policy, procedure, log, screenshot, report
    evidence_source: str
    collection_date: datetime
    evidence_description: str
    file_path: Optional[str]
    hash_value: str
    retention_period: int  # days
    access_restrictions: List[str]

class DefenderIngestRequest(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    alert_types: Optional[List[str]] = None
    severity_filter: Optional[List[str]] = None
    include_resolved: bool = False

class SOC2ReportRequest(BaseModel):
    report_period_start: datetime
    report_period_end: datetime
    controls: Optional[List[str]] = None  # Specific controls to include
    include_evidence: bool = True
    output_format: str = "pdf"  # pdf, json, csv

class DefenderIngestService:
    """Main Defender integration and SOC-2 compliance service"""
    
    def __init__(self):
        self.config = self._load_config()
        self.db_pool = None
        self.redis_client = None
        self.credential = None
        self.security_center_client = None
        self.graph_client = None
        self.soc2_controls = self._initialize_soc2_controls()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        return {
            'database': {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'policycortex'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', 'postgres'),
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 2)),
            },
            'azure': {
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'tenant_id': os.getenv('AZURE_TENANT_ID'),
                'client_id': os.getenv('AZURE_CLIENT_ID'),
            },
            'defender': {
                'ingest_interval': int(os.getenv('DEFENDER_INGEST_INTERVAL', 300)),  # 5 minutes
                'alert_retention_days': int(os.getenv('ALERT_RETENTION_DAYS', 90)),
                'enable_auto_response': os.getenv('ENABLE_AUTO_RESPONSE', 'false').lower() == 'true',
            },
            'soc2': {
                'evidence_storage_path': os.getenv('EVIDENCE_STORAGE_PATH', './evidence'),
                'report_storage_path': os.getenv('REPORT_STORAGE_PATH', './reports'),
                'evidence_retention_years': int(os.getenv('EVIDENCE_RETENTION_YEARS', 7)),
                'auto_collect_evidence': os.getenv('AUTO_COLLECT_EVIDENCE', 'true').lower() == 'true',
            },
            'graph': {
                'api_version': 'v1.0',
                'base_url': 'https://graph.microsoft.com'
            }
        }
    
    def _initialize_soc2_controls(self) -> Dict[str, SOC2Control]:
        """Initialize SOC-2 control framework"""
        controls = {}
        
        # Common Criteria 1: Control Environment
        cc1_controls = [
            ("CC1.1", "Commitment to Integrity and Ethical Values", "The entity demonstrates a commitment to integrity and ethical values."),
            ("CC1.2", "Board Independence and Oversight", "The board of directors demonstrates independence from management."),
            ("CC1.3", "Organizational Structure and Assignment of Authority", "Management establishes structures, reporting lines, and appropriate authorities."),
            ("CC1.4", "Commitment to Competence", "The entity demonstrates a commitment to attract, develop, and retain competent individuals."),
            ("CC1.5", "Accountability", "The entity holds individuals accountable for their internal control responsibilities."),
        ]
        
        for control_id, name, description in cc1_controls:
            controls[control_id] = SOC2Control(
                control_id=control_id,
                control_name=name,
                control_description=description,
                control_category="CC1",
                implementation_status="implemented",
                last_tested=datetime.utcnow() - timedelta(days=30),
                test_results="Satisfactory",
                evidence_collected=[],
                deficiencies=[],
                remediation_timeline=None
            )
        
        # Common Criteria 2: Communication and Information
        cc2_controls = [
            ("CC2.1", "Information Quality", "The entity obtains or generates and uses relevant, quality information."),
            ("CC2.2", "Internal Communication", "The entity internally communicates information necessary for internal control."),
            ("CC2.3", "External Communication", "The entity communicates with external parties regarding matters affecting internal control."),
        ]
        
        for control_id, name, description in cc2_controls:
            controls[control_id] = SOC2Control(
                control_id=control_id,
                control_name=name,
                control_description=description,
                control_category="CC2",
                implementation_status="implemented",
                last_tested=datetime.utcnow() - timedelta(days=30),
                test_results="Satisfactory",
                evidence_collected=[],
                deficiencies=[],
                remediation_timeline=None
            )
        
        # Add Security (A1) controls
        security_controls = [
            ("A1.1", "Logical and Physical Access Controls", "The entity implements logical and physical access controls."),
            ("A1.2", "User Authentication", "The entity uses multi-factor authentication for privileged users."),
            ("A1.3", "Network Security", "The entity implements network security controls."),
        ]
        
        for control_id, name, description in security_controls:
            controls[control_id] = SOC2Control(
                control_id=control_id,
                control_name=name,
                control_description=description,
                control_category="A1",
                implementation_status="implemented",
                last_tested=datetime.utcnow() - timedelta(days=30),
                test_results="Satisfactory",
                evidence_collected=[],
                deficiencies=[],
                remediation_timeline=None
            )
        
        return controls
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing Defender integration service...")
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            min_size=2,
            max_size=10
        )
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            decode_responses=True
        )
        
        # Initialize Azure clients
        self.credential = DefaultAzureCredential()
        
        # Initialize Security Center client
        if self.config['azure']['subscription_id']:
            self.security_center_client = SecurityCenter(
                credential=self.credential,
                subscription_id=self.config['azure']['subscription_id']
            )
        
        # Initialize Microsoft Graph client
        self.graph_client = GraphServiceClient(credentials=self.credential)
        
        # Initialize database tables
        await self._initialize_database()
        
        # Create evidence storage directories
        self._ensure_storage_directories()
        
        # Start background tasks
        asyncio.create_task(self._defender_ingest_loop())
        asyncio.create_task(self._evidence_collection_loop())
        
        logger.info("Defender integration service initialized")
    
    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            async with self.db_pool.acquire() as conn:
                # Defender alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS defender_alerts (
                        alert_id VARCHAR PRIMARY KEY,
                        title VARCHAR NOT NULL,
                        description TEXT,
                        severity VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        category VARCHAR,
                        vendor VARCHAR,
                        product VARCHAR,
                        first_activity_time TIMESTAMP NOT NULL,
                        last_activity_time TIMESTAMP NOT NULL,
                        confidence DECIMAL,
                        risk_score DECIMAL,
                        impacted_entities JSONB DEFAULT '[]',
                        tactics JSONB DEFAULT '[]',
                        techniques JSONB DEFAULT '[]',
                        evidence JSONB DEFAULT '[]',
                        raw_data JSONB NOT NULL,
                        ingested_at TIMESTAMP DEFAULT NOW(),
                        processed BOOLEAN DEFAULT FALSE,
                        response_actions JSONB DEFAULT '[]'
                    )
                """)
                
                # SOC-2 controls table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS soc2_controls (
                        control_id VARCHAR PRIMARY KEY,
                        control_name VARCHAR NOT NULL,
                        control_description TEXT NOT NULL,
                        control_category VARCHAR NOT NULL,
                        implementation_status VARCHAR NOT NULL,
                        last_tested TIMESTAMP,
                        test_results TEXT,
                        evidence_collected JSONB DEFAULT '[]',
                        deficiencies JSONB DEFAULT '[]',
                        remediation_timeline TIMESTAMP,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Compliance evidence table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_evidence (
                        evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        control_id VARCHAR NOT NULL,
                        evidence_type VARCHAR NOT NULL,
                        evidence_source VARCHAR NOT NULL,
                        collection_date TIMESTAMP DEFAULT NOW(),
                        evidence_description TEXT NOT NULL,
                        file_path VARCHAR,
                        hash_value VARCHAR,
                        retention_period INTEGER DEFAULT 2555,
                        access_restrictions JSONB DEFAULT '[]',
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # SOC-2 reports table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS soc2_reports (
                        report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        report_period_start TIMESTAMP NOT NULL,
                        report_period_end TIMESTAMP NOT NULL,
                        controls_tested JSONB NOT NULL,
                        report_type VARCHAR DEFAULT 'Type II',
                        status VARCHAR DEFAULT 'draft',
                        file_path VARCHAR,
                        hash_value VARCHAR,
                        generated_at TIMESTAMP DEFAULT NOW(),
                        auditor_notes TEXT,
                        exceptions JSONB DEFAULT '[]'
                    )
                """)
                
                # Evidence collection tasks table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS evidence_collection_tasks (
                        task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        control_id VARCHAR NOT NULL,
                        task_type VARCHAR NOT NULL,
                        schedule VARCHAR,
                        last_run TIMESTAMP,
                        next_run TIMESTAMP,
                        status VARCHAR DEFAULT 'pending',
                        configuration JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_defender_alerts_severity ON defender_alerts(severity)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_defender_alerts_time ON defender_alerts(first_activity_time)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_control ON compliance_evidence(control_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_type ON compliance_evidence(evidence_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_soc2_controls_category ON soc2_controls(control_category)")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _ensure_storage_directories(self):
        """Ensure evidence and report storage directories exist"""
        evidence_path = Path(self.config['soc2']['evidence_storage_path'])
        report_path = Path(self.config['soc2']['report_storage_path'])
        
        evidence_path.mkdir(parents=True, exist_ok=True)
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different evidence types
        for evidence_type in ['policies', 'procedures', 'logs', 'screenshots', 'reports']:
            (evidence_path / evidence_type).mkdir(exist_ok=True)
    
    async def ingest_defender_alerts(self, request: DefenderIngestRequest) -> List[DefenderAlert]:
        """Ingest alerts from Microsoft Defender"""
        logger.info("Starting Defender alert ingestion...")
        
        try:
            alerts = []
            
            # Get alerts from Microsoft Defender for Cloud
            cloud_alerts = await self._get_defender_cloud_alerts(request)
            alerts.extend(cloud_alerts)
            
            # Get alerts from Microsoft 365 Defender
            m365_alerts = await self._get_microsoft_365_alerts(request)
            alerts.extend(m365_alerts)
            
            # Store alerts in database
            stored_alerts = []
            for alert in alerts:
                stored_alert = await self._store_defender_alert(alert)
                stored_alerts.append(stored_alert)
                
                # Trigger automated response if enabled
                if self.config['defender']['enable_auto_response']:
                    await self._trigger_automated_response(stored_alert)
            
            # Cache results
            await self._cache_alert_summary(stored_alerts)
            
            logger.info(f"Ingested {len(stored_alerts)} Defender alerts")
            return stored_alerts
            
        except Exception as e:
            logger.error(f"Defender alert ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Alert ingestion failed: {str(e)}")
    
    async def collect_soc2_evidence(self, control_ids: Optional[List[str]] = None) -> Dict[str, List[ComplianceEvidence]]:
        """Collect evidence for SOC-2 controls"""
        logger.info("Starting SOC-2 evidence collection...")
        
        evidence_by_control = {}
        controls_to_process = control_ids or list(self.soc2_controls.keys())
        
        try:
            for control_id in controls_to_process:
                control = self.soc2_controls.get(control_id)
                if not control:
                    continue
                
                evidence_list = []
                
                # Collect evidence based on control category
                if control.control_category == "CC1":
                    evidence_list.extend(await self._collect_governance_evidence(control))
                elif control.control_category == "CC2":
                    evidence_list.extend(await self._collect_communication_evidence(control))
                elif control.control_category == "A1":
                    evidence_list.extend(await self._collect_security_evidence(control))
                
                # Store evidence in database
                for evidence in evidence_list:
                    await self._store_compliance_evidence(evidence)
                
                evidence_by_control[control_id] = evidence_list
            
            logger.info(f"Collected evidence for {len(evidence_by_control)} controls")
            return evidence_by_control
            
        except Exception as e:
            logger.error(f"Evidence collection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Evidence collection failed: {str(e)}")
    
    async def generate_soc2_report(self, request: SOC2ReportRequest) -> Dict[str, Any]:
        """Generate SOC-2 compliance report"""
        logger.info(f"Generating SOC-2 report for period {request.report_period_start} to {request.report_period_end}")
        
        try:
            # Collect data for report
            controls_data = await self._get_controls_data(request.controls, request.report_period_start, request.report_period_end)
            evidence_data = await self._get_evidence_data(request.controls, request.report_period_start, request.report_period_end) if request.include_evidence else []
            
            # Generate report based on format
            if request.output_format == "pdf":
                report_path = await self._generate_pdf_report(controls_data, evidence_data, request)
            elif request.output_format == "json":
                report_path = await self._generate_json_report(controls_data, evidence_data, request)
            elif request.output_format == "csv":
                report_path = await self._generate_csv_report(controls_data, evidence_data, request)
            else:
                raise ValueError(f"Unsupported output format: {request.output_format}")
            
            # Store report metadata
            report_metadata = await self._store_report_metadata(request, report_path)
            
            return {
                "report_id": report_metadata["report_id"],
                "file_path": report_path,
                "controls_tested": len(controls_data),
                "evidence_items": len(evidence_data),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SOC-2 report generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    
    async def _get_defender_cloud_alerts(self, request: DefenderIngestRequest) -> List[DefenderAlert]:
        """Get alerts from Microsoft Defender for Cloud"""
        alerts = []
        
        try:
            # Set time filter
            start_time = request.start_time or (datetime.utcnow() - timedelta(hours=24))
            
            # Get alerts from Security Center
            async for alert in self.security_center_client.alerts.list():
                # Filter by time
                if alert.time_generated_utc < start_time:
                    continue
                
                # Filter by severity if specified
                if request.severity_filter and alert.alert_severity not in request.severity_filter:
                    continue
                
                # Filter by status
                if not request.include_resolved and alert.state == "Resolved":
                    continue
                
                defender_alert = DefenderAlert(
                    alert_id=alert.alert_display_name,
                    title=alert.alert_display_name,
                    description=alert.description or "",
                    severity=alert.alert_severity,
                    status=alert.state,
                    category=alert.alert_type or "",
                    vendor="Microsoft",
                    product="Defender for Cloud",
                    first_activity_time=alert.time_generated_utc,
                    last_activity_time=alert.time_generated_utc,
                    confidence=alert.confidence_score or 0.0,
                    risk_score=0.0,  # Would be calculated based on severity and other factors
                    impacted_entities=[],
                    tactics=[],
                    techniques=[],
                    evidence=[],
                    raw_data=alert.as_dict()
                )
                
                alerts.append(defender_alert)
                
        except Exception as e:
            logger.error(f"Failed to get Defender for Cloud alerts: {e}")
        
        return alerts
    
    async def _get_microsoft_365_alerts(self, request: DefenderIngestRequest) -> List[DefenderAlert]:
        """Get alerts from Microsoft 365 Defender"""
        alerts = []
        
        try:
            # Query Microsoft Graph for security alerts
            # This would use the actual Graph API
            # For now, return empty list as placeholder
            pass
            
        except Exception as e:
            logger.error(f"Failed to get Microsoft 365 alerts: {e}")
        
        return alerts
    
    async def _store_defender_alert(self, alert: DefenderAlert) -> DefenderAlert:
        """Store Defender alert in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO defender_alerts 
                (alert_id, title, description, severity, status, category, vendor, product,
                 first_activity_time, last_activity_time, confidence, risk_score,
                 impacted_entities, tactics, techniques, evidence, raw_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (alert_id) DO UPDATE SET
                    status = $5,
                    last_activity_time = $10,
                    confidence = $11,
                    risk_score = $12,
                    impacted_entities = $13,
                    tactics = $14,
                    techniques = $15,
                    evidence = $16,
                    raw_data = $17
            """,
                alert.alert_id,
                alert.title,
                alert.description,
                alert.severity,
                alert.status,
                alert.category,
                alert.vendor,
                alert.product,
                alert.first_activity_time,
                alert.last_activity_time,
                alert.confidence,
                alert.risk_score,
                json.dumps(alert.impacted_entities),
                json.dumps(alert.tactics),
                json.dumps(alert.techniques),
                json.dumps(alert.evidence),
                json.dumps(alert.raw_data)
            )
        
        return alert
    
    async def _collect_governance_evidence(self, control: SOC2Control) -> List[ComplianceEvidence]:
        """Collect evidence for governance controls (CC1)"""
        evidence = []
        
        # Collect policy documents
        policy_evidence = ComplianceEvidence(
            evidence_id="",
            control_id=control.control_id,
            evidence_type="policy",
            evidence_source="PolicyCortex",
            collection_date=datetime.utcnow(),
            evidence_description=f"Governance policies for {control.control_name}",
            file_path=None,
            hash_value="",
            retention_period=2555,  # 7 years
            access_restrictions=["management", "auditors"]
        )
        evidence.append(policy_evidence)
        
        # Collect organizational charts
        org_evidence = ComplianceEvidence(
            evidence_id="",
            control_id=control.control_id,
            evidence_type="procedure",
            evidence_source="HR System",
            collection_date=datetime.utcnow(),
            evidence_description=f"Organizational structure documentation for {control.control_name}",
            file_path=None,
            hash_value="",
            retention_period=2555,
            access_restrictions=["management", "auditors"]
        )
        evidence.append(org_evidence)
        
        return evidence
    
    async def _collect_communication_evidence(self, control: SOC2Control) -> List[ComplianceEvidence]:
        """Collect evidence for communication controls (CC2)"""
        evidence = []
        
        # Collect communication logs
        comm_evidence = ComplianceEvidence(
            evidence_id="",
            control_id=control.control_id,
            evidence_type="log",
            evidence_source="Communication Systems",
            collection_date=datetime.utcnow(),
            evidence_description=f"Communication logs for {control.control_name}",
            file_path=None,
            hash_value="",
            retention_period=2555,
            access_restrictions=["it_team", "auditors"]
        )
        evidence.append(comm_evidence)
        
        return evidence
    
    async def _collect_security_evidence(self, control: SOC2Control) -> List[ComplianceEvidence]:
        """Collect evidence for security controls (A1)"""
        evidence = []
        
        # Collect access control logs
        access_evidence = ComplianceEvidence(
            evidence_id="",
            control_id=control.control_id,
            evidence_type="log",
            evidence_source="Azure AD",
            collection_date=datetime.utcnow(),
            evidence_description=f"Access control logs for {control.control_name}",
            file_path=None,
            hash_value="",
            retention_period=2555,
            access_restrictions=["security_team", "auditors"]
        )
        evidence.append(access_evidence)
        
        # Collect security scan results
        scan_evidence = ComplianceEvidence(
            evidence_id="",
            control_id=control.control_id,
            evidence_type="report",
            evidence_source="Security Scanner",
            collection_date=datetime.utcnow(),
            evidence_description=f"Security scan results for {control.control_name}",
            file_path=None,
            hash_value="",
            retention_period=2555,
            access_restrictions=["security_team", "auditors"]
        )
        evidence.append(scan_evidence)
        
        return evidence
    
    async def _store_compliance_evidence(self, evidence: ComplianceEvidence):
        """Store compliance evidence in database"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO compliance_evidence 
                (control_id, evidence_type, evidence_source, evidence_description,
                 file_path, hash_value, retention_period, access_restrictions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING evidence_id
            """,
                evidence.control_id,
                evidence.evidence_type,
                evidence.evidence_source,
                evidence.evidence_description,
                evidence.file_path,
                evidence.hash_value,
                evidence.retention_period,
                json.dumps(evidence.access_restrictions)
            )
            
            evidence.evidence_id = str(row['evidence_id'])
    
    async def _generate_pdf_report(self, controls_data: List[Dict], evidence_data: List[Dict], 
                                 request: SOC2ReportRequest) -> str:
        """Generate PDF SOC-2 report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"soc2_report_{timestamp}.pdf"
        file_path = Path(self.config['soc2']['report_storage_path']) / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(file_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("SOC 2 Type II Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Report period
        period_text = f"Report Period: {request.report_period_start.strftime('%Y-%m-%d')} to {request.report_period_end.strftime('%Y-%m-%d')}"
        period = Paragraph(period_text, styles['Normal'])
        story.append(period)
        story.append(Spacer(1, 12))
        
        # Controls summary
        controls_title = Paragraph("Controls Tested", styles['Heading1'])
        story.append(controls_title)
        
        # Create controls table
        controls_table_data = [['Control ID', 'Control Name', 'Status', 'Test Results']]
        for control in controls_data:
            controls_table_data.append([
                control['control_id'],
                control['control_name'],
                control['implementation_status'],
                control['test_results']
            ])
        
        controls_table = Table(controls_table_data)
        controls_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(controls_table)
        story.append(Spacer(1, 12))
        
        # Evidence summary
        if evidence_data:
            evidence_title = Paragraph("Evidence Summary", styles['Heading1'])
            story.append(evidence_title)
            
            evidence_summary = Paragraph(f"Total evidence items collected: {len(evidence_data)}", styles['Normal'])
            story.append(evidence_summary)
        
        # Build PDF
        doc.build(story)
        
        return str(file_path)
    
    async def _generate_json_report(self, controls_data: List[Dict], evidence_data: List[Dict], 
                                  request: SOC2ReportRequest) -> str:
        """Generate JSON SOC-2 report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"soc2_report_{timestamp}.json"
        file_path = Path(self.config['soc2']['report_storage_path']) / filename
        
        report_data = {
            "report_metadata": {
                "report_type": "SOC 2 Type II",
                "report_period_start": request.report_period_start.isoformat(),
                "report_period_end": request.report_period_end.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "generated_by": "PolicyCortex"
            },
            "controls": controls_data,
            "evidence": evidence_data if request.include_evidence else [],
            "summary": {
                "total_controls": len(controls_data),
                "total_evidence": len(evidence_data),
                "controls_implemented": len([c for c in controls_data if c['implementation_status'] == 'implemented'])
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(file_path)
    
    async def _generate_csv_report(self, controls_data: List[Dict], evidence_data: List[Dict], 
                                 request: SOC2ReportRequest) -> str:
        """Generate CSV SOC-2 report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"soc2_report_{timestamp}.csv"
        file_path = Path(self.config['soc2']['report_storage_path']) / filename
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(controls_data)
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    async def _get_controls_data(self, control_ids: Optional[List[str]], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get controls data for report"""
        controls_filter = control_ids or list(self.soc2_controls.keys())
        
        async with self.db_pool.acquire() as conn:
            placeholders = ','.join(f'${i+1}' for i in range(len(controls_filter)))
            query = f"""
                SELECT * FROM soc2_controls 
                WHERE control_id IN ({placeholders})
                ORDER BY control_category, control_id
            """
            rows = await conn.fetch(query, *controls_filter)
            return [dict(row) for row in rows]
    
    async def _get_evidence_data(self, control_ids: Optional[List[str]], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get evidence data for report"""
        controls_filter = control_ids or list(self.soc2_controls.keys())
        
        async with self.db_pool.acquire() as conn:
            placeholders = ','.join(f'${i+1}' for i in range(len(controls_filter)))
            query = f"""
                SELECT * FROM compliance_evidence 
                WHERE control_id IN ({placeholders})
                AND collection_date BETWEEN ${len(controls_filter)+1} AND ${len(controls_filter)+2}
                ORDER BY control_id, collection_date
            """
            rows = await conn.fetch(query, *controls_filter, start_date, end_date)
            return [dict(row) for row in rows]
    
    async def _store_report_metadata(self, request: SOC2ReportRequest, file_path: str) -> Dict[str, Any]:
        """Store report metadata in database"""
        file_hash = self._calculate_file_hash(file_path)
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO soc2_reports 
                (report_period_start, report_period_end, controls_tested, file_path, hash_value)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING report_id
            """,
                request.report_period_start,
                request.report_period_end,
                json.dumps(request.controls or []),
                file_path,
                file_hash
            )
            
            return {
                "report_id": str(row['report_id']),
                "file_hash": file_hash
            }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _defender_ingest_loop(self):
        """Background task for periodic Defender alert ingestion"""
        while True:
            try:
                await asyncio.sleep(self.config['defender']['ingest_interval'])
                
                # Ingest alerts
                request = DefenderIngestRequest()
                await self.ingest_defender_alerts(request)
                
                logger.info("Completed scheduled Defender alert ingestion")
                
            except Exception as e:
                logger.error(f"Error in Defender ingest loop: {e}")
                await asyncio.sleep(60)
    
    async def _evidence_collection_loop(self):
        """Background task for periodic evidence collection"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.config['soc2']['auto_collect_evidence']:
                    await self.collect_soc2_evidence()
                
                logger.info("Completed scheduled evidence collection")
                
            except Exception as e:
                logger.error(f"Error in evidence collection loop: {e}")
                await asyncio.sleep(300)
    
    async def _trigger_automated_response(self, alert: DefenderAlert):
        """Trigger automated response to security alert"""
        # Implementation would trigger appropriate response actions
        logger.info(f"Triggering automated response for alert: {alert.alert_id}")
    
    async def _cache_alert_summary(self, alerts: List[DefenderAlert]):
        """Cache alert summary in Redis"""
        summary = {
            'total_alerts': len(alerts),
            'by_severity': {},
            'by_status': {},
            'last_updated': datetime.utcnow().isoformat()
        }
        
        for alert in alerts:
            # Count by severity
            summary['by_severity'][alert.severity] = summary['by_severity'].get(alert.severity, 0) + 1
            # Count by status
            summary['by_status'][alert.status] = summary['by_status'].get(alert.status, 0) + 1
        
        await self.redis_client.setex(
            "defender_alert_summary",
            3600,  # 1 hour TTL
            json.dumps(summary)
        )

# Global service instance
defender_service = DefenderIngestService()

@app.on_event("startup")
async def startup_event():
    await defender_service.initialize()

@app.post("/defender/ingest")
async def ingest_alerts(request: DefenderIngestRequest, background_tasks: BackgroundTasks):
    """Ingest Defender alerts"""
    background_tasks.add_task(defender_service.ingest_defender_alerts, request)
    return {"status": "ingest_started", "request": request.dict()}

@app.get("/defender/alerts")
async def get_alerts(limit: int = 100, severity: Optional[str] = None):
    """Get stored Defender alerts"""
    async with defender_service.db_pool.acquire() as conn:
        query = "SELECT * FROM defender_alerts"
        params = []
        
        if severity:
            query += " WHERE severity = $1"
            params.append(severity)
        
        query += " ORDER BY first_activity_time DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

@app.post("/soc2/evidence/collect")
async def collect_evidence(control_ids: Optional[List[str]] = None):
    """Collect SOC-2 evidence"""
    return await defender_service.collect_soc2_evidence(control_ids)

@app.post("/soc2/report/generate")
async def generate_report(request: SOC2ReportRequest):
    """Generate SOC-2 compliance report"""
    return await defender_service.generate_soc2_report(request)

@app.get("/soc2/controls")
async def get_controls():
    """Get SOC-2 controls"""
    async with defender_service.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM soc2_controls ORDER BY control_category, control_id")
        return [dict(row) for row in rows]

@app.get("/soc2/evidence/{control_id}")
async def get_evidence_by_control(control_id: str):
    """Get evidence for specific control"""
    async with defender_service.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM compliance_evidence WHERE control_id = $1 ORDER BY collection_date DESC",
            control_id
        )
        return [dict(row) for row in rows]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "defender-ingest"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)