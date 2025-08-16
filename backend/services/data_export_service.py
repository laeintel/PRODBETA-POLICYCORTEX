"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Data Export and Integration Service for PolicyCortex
Provides comprehensive data export, import, and integration capabilities
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import asyncio
import aiohttp
from io import BytesIO, StringIO
import zipfile
import yaml
import xml.etree.ElementTree as ET
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import base64

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    YAML = "yaml"
    PARQUET = "parquet"
    HTML = "html"
    PDF = "pdf"
    ARCHIVE = "archive"  # ZIP with multiple formats

class IntegrationType(Enum):
    """Integration types"""
    WEBHOOK = "webhook"
    API = "api"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    FTP = "ftp"
    EMAIL = "email"
    KAFKA = "kafka"
    EVENT_HUB = "event_hub"
    DATABASE = "database"

@dataclass
class ExportJob:
    """Export job definition"""
    id: str
    format: ExportFormat
    data_types: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    id: str
    name: str
    type: IntegrationType
    enabled: bool
    config: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataTransformation:
    """Data transformation rule"""
    id: str
    name: str
    source_field: str
    target_field: str
    transformation: str  # Expression or function name
    parameters: Dict[str, Any] = field(default_factory=dict)

class DataExportService:
    """Data export and integration service"""
    
    def __init__(self):
        """Initialize data export service"""
        self.export_jobs: Dict[str, ExportJob] = {}
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.transformations: Dict[str, DataTransformation] = {}
        self.export_templates: Dict[str, Dict] = {}
        self.export_path = os.getenv("EXPORT_PATH", "/tmp/exports")
        
        # Ensure export directory exists
        os.makedirs(self.export_path, exist_ok=True)
        
        # Initialize templates
        self._initialize_export_templates()
    
    def _initialize_export_templates(self):
        """Initialize export templates"""
        self.export_templates = {
            "compliance_audit": {
                "data_types": ["policies", "compliance", "violations", "exceptions"],
                "format": ExportFormat.EXCEL,
                "transformations": ["format_dates", "mask_sensitive_data"]
            },
            "cost_report": {
                "data_types": ["costs", "budgets", "forecasts", "recommendations"],
                "format": ExportFormat.EXCEL,
                "transformations": ["aggregate_by_service", "calculate_trends"]
            },
            "security_assessment": {
                "data_types": ["security_findings", "vulnerabilities", "incidents", "remediation"],
                "format": ExportFormat.JSON,
                "transformations": ["severity_mapping", "risk_scoring"]
            },
            "resource_inventory": {
                "data_types": ["resources", "tags", "configurations", "dependencies"],
                "format": ExportFormat.CSV,
                "transformations": ["flatten_nested", "normalize_tags"]
            },
            "executive_dashboard": {
                "data_types": ["metrics", "kpis", "trends", "alerts"],
                "format": ExportFormat.HTML,
                "transformations": ["calculate_percentages", "generate_charts"]
            }
        }
    
    async def export_data(
        self,
        data_types: List[str],
        format: ExportFormat,
        filters: Optional[Dict[str, Any]] = None,
        transformations: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> ExportJob:
        """Export data in specified format"""
        
        # Use template if provided
        if template and template in self.export_templates:
            template_config = self.export_templates[template]
            data_types = template_config.get("data_types", data_types)
            format = template_config.get("format", format)
            transformations = template_config.get("transformations", transformations)
        
        # Create export job
        job_id = f"export-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        job = ExportJob(
            id=job_id,
            format=format,
            data_types=data_types,
            filters=filters or {},
            metadata={"template": template} if template else {}
        )
        
        self.export_jobs[job_id] = job
        
        # Start export asynchronously
        asyncio.create_task(self._process_export(job, transformations))
        
        return job
    
    async def _process_export(self, job: ExportJob, transformations: Optional[List[str]]):
        """Process export job"""
        try:
            job.status = "processing"
            job.progress = 10.0
            
            # Fetch data
            data = await self._fetch_data(job.data_types, job.filters)
            job.progress = 40.0
            
            # Apply transformations
            if transformations:
                data = await self._apply_transformations(data, transformations)
            job.progress = 60.0
            
            # Export to format
            file_path = await self._export_to_format(data, job.format, job.id)
            job.progress = 90.0
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Update job
            job.file_path = file_path
            job.file_size = file_size
            job.status = "completed"
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            
            logger.info(f"Export job {job.id} completed: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Export job {job.id} failed: {e}")
    
    async def _fetch_data(self, data_types: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data based on types and filters"""
        data = {}
        
        for data_type in data_types:
            # Simulate fetching data from different sources
            if data_type == "policies":
                data["policies"] = await self._fetch_policies(filters)
            elif data_type == "compliance":
                data["compliance"] = await self._fetch_compliance(filters)
            elif data_type == "costs":
                data["costs"] = await self._fetch_costs(filters)
            elif data_type == "resources":
                data["resources"] = await self._fetch_resources(filters)
            elif data_type == "security_findings":
                data["security_findings"] = await self._fetch_security_findings(filters)
            else:
                # Generic data fetch
                data[data_type] = []
        
        return data
    
    async def _fetch_policies(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch policy data"""
        # Simulate fetching from database
        policies = [
            {
                "id": f"policy-{i}",
                "name": f"Policy {i}",
                "type": "compliance",
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "compliance_rate": 85 + i
            }
            for i in range(1, 11)
        ]
        
        # Apply filters
        if "status" in filters:
            policies = [p for p in policies if p["status"] == filters["status"]]
        
        return policies
    
    async def _fetch_compliance(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch compliance data"""
        compliance = [
            {
                "resource_id": f"resource-{i}",
                "policy_id": f"policy-{i % 5 + 1}",
                "compliant": i % 3 != 0,
                "checked_at": datetime.utcnow().isoformat(),
                "score": 70 + (i * 3) % 30
            }
            for i in range(1, 21)
        ]
        
        return compliance
    
    async def _fetch_costs(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch cost data"""
        costs = [
            {
                "service": ["Compute", "Storage", "Network", "Database"][i % 4],
                "amount": 1000 + i * 100,
                "currency": "USD",
                "date": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "provider": "Azure"
            }
            for i in range(30)
        ]
        
        return costs
    
    async def _fetch_resources(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch resource data"""
        resources = [
            {
                "id": f"resource-{i}",
                "name": f"Resource {i}",
                "type": ["VM", "Storage", "Database", "Network"][i % 4],
                "location": ["eastus", "westus", "europe"][i % 3],
                "tags": {"env": "prod" if i % 2 == 0 else "dev"}
            }
            for i in range(1, 16)
        ]
        
        return resources
    
    async def _fetch_security_findings(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch security findings"""
        findings = [
            {
                "id": f"finding-{i}",
                "severity": ["critical", "high", "medium", "low"][i % 4],
                "type": "vulnerability",
                "resource_id": f"resource-{i % 10 + 1}",
                "detected_at": datetime.utcnow().isoformat()
            }
            for i in range(1, 11)
        ]
        
        return findings
    
    async def _apply_transformations(self, data: Dict[str, Any], transformations: List[str]) -> Dict[str, Any]:
        """Apply data transformations"""
        for transformation in transformations:
            if transformation == "format_dates":
                data = self._transform_format_dates(data)
            elif transformation == "mask_sensitive_data":
                data = self._transform_mask_sensitive(data)
            elif transformation == "aggregate_by_service":
                data = self._transform_aggregate_service(data)
            elif transformation == "flatten_nested":
                data = self._transform_flatten(data)
            elif transformation == "calculate_percentages":
                data = self._transform_percentages(data)
        
        return data
    
    def _transform_format_dates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format date fields"""
        for key, values in data.items():
            if isinstance(values, list):
                for item in values:
                    for field, value in item.items():
                        if "date" in field or "at" in field:
                            if isinstance(value, str):
                                try:
                                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    item[field] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                except:
                                    pass
        return data
    
    def _transform_mask_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data"""
        sensitive_fields = ["email", "password", "token", "key", "secret"]
        
        for key, values in data.items():
            if isinstance(values, list):
                for item in values:
                    for field in item:
                        if any(sensitive in field.lower() for sensitive in sensitive_fields):
                            if isinstance(item[field], str):
                                item[field] = "***MASKED***"
        return data
    
    def _transform_aggregate_service(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data by service"""
        if "costs" in data and isinstance(data["costs"], list):
            df = pd.DataFrame(data["costs"])
            if "service" in df.columns:
                aggregated = df.groupby("service")["amount"].sum().to_dict()
                data["costs_by_service"] = [
                    {"service": k, "total": v} for k, v in aggregated.items()
                ]
        return data
    
    def _transform_flatten(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested structures"""
        for key, values in data.items():
            if isinstance(values, list):
                flattened = []
                for item in values:
                    flat_item = {}
                    for field, value in item.items():
                        if isinstance(value, dict):
                            for sub_field, sub_value in value.items():
                                flat_item[f"{field}_{sub_field}"] = sub_value
                        else:
                            flat_item[field] = value
                    flattened.append(flat_item)
                data[key] = flattened
        return data
    
    def _transform_percentages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate percentage values"""
        for key, values in data.items():
            if isinstance(values, list):
                for item in values:
                    if "score" in item and isinstance(item["score"], (int, float)):
                        item["score_percentage"] = f"{item['score']:.1f}%"
                    if "compliance_rate" in item and isinstance(item["compliance_rate"], (int, float)):
                        item["compliance_percentage"] = f"{item['compliance_rate']:.1f}%"
        return data
    
    async def _export_to_format(self, data: Dict[str, Any], format: ExportFormat, job_id: str) -> str:
        """Export data to specified format"""
        file_path = os.path.join(self.export_path, f"{job_id}.{format.value}")
        
        if format == ExportFormat.JSON:
            await self._export_json(data, file_path)
        elif format == ExportFormat.CSV:
            await self._export_csv(data, file_path)
        elif format == ExportFormat.EXCEL:
            await self._export_excel(data, file_path)
        elif format == ExportFormat.XML:
            await self._export_xml(data, file_path)
        elif format == ExportFormat.YAML:
            await self._export_yaml(data, file_path)
        elif format == ExportFormat.HTML:
            await self._export_html(data, file_path)
        elif format == ExportFormat.ARCHIVE:
            await self._export_archive(data, file_path)
        else:
            # Default to JSON
            await self._export_json(data, file_path)
        
        return file_path
    
    async def _export_json(self, data: Dict[str, Any], file_path: str):
        """Export to JSON"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def _export_csv(self, data: Dict[str, Any], file_path: str):
        """Export to CSV (multiple files if needed)"""
        base_path = file_path.rsplit('.', 1)[0]
        
        for key, values in data.items():
            if isinstance(values, list) and values:
                csv_path = f"{base_path}_{key}.csv"
                df = pd.DataFrame(values)
                df.to_csv(csv_path, index=False)
    
    async def _export_excel(self, data: Dict[str, Any], file_path: str):
        """Export to Excel with multiple sheets"""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for key, values in data.items():
                if isinstance(values, list) and values:
                    df = pd.DataFrame(values)
                    sheet_name = key[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format the sheet
                    worksheet = writer.sheets[sheet_name]
                    
                    # Header formatting
                    for cell in worksheet[1]:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                        cell.font = Font(color="FFFFFF", bold=True)
                        cell.alignment = Alignment(horizontal="center")
                    
                    # Auto-adjust column width
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
    
    async def _export_xml(self, data: Dict[str, Any], file_path: str):
        """Export to XML"""
        root = ET.Element("export")
        root.set("timestamp", datetime.utcnow().isoformat())
        
        for key, values in data.items():
            section = ET.SubElement(root, key)
            if isinstance(values, list):
                for item in values:
                    record = ET.SubElement(section, "record")
                    for field, value in item.items():
                        elem = ET.SubElement(record, field)
                        elem.text = str(value)
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    async def _export_yaml(self, data: Dict[str, Any], file_path: str):
        """Export to YAML"""
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    async def _export_html(self, data: Dict[str, Any], file_path: str):
        """Export to HTML"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PolicyCortex Data Export</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
                th { background-color: #f2f2f2; border: 1px solid #ddd; padding: 12px; text-align: left; }
                td { border: 1px solid #ddd; padding: 8px; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .timestamp { color: #999; font-size: 12px; }
            </style>
        </head>
        <body>
            <h1>PolicyCortex Data Export</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        """.format(timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        
        for key, values in data.items():
            if isinstance(values, list) and values:
                html_content += f"<h2>{key.replace('_', ' ').title()}</h2>"
                html_content += "<table>"
                
                # Headers
                html_content += "<tr>"
                for field in values[0].keys():
                    html_content += f"<th>{field.replace('_', ' ').title()}</th>"
                html_content += "</tr>"
                
                # Data rows
                for item in values:
                    html_content += "<tr>"
                    for value in item.values():
                        html_content += f"<td>{value}</td>"
                    html_content += "</tr>"
                
                html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(file_path, 'w') as f:
            f.write(html_content)
    
    async def _export_archive(self, data: Dict[str, Any], file_path: str):
        """Export to ZIP archive with multiple formats"""
        base_path = file_path.rsplit('.', 1)[0]
        
        # Create individual format exports
        json_path = f"{base_path}.json"
        await self._export_json(data, json_path)
        
        excel_path = f"{base_path}.xlsx"
        await self._export_excel(data, excel_path)
        
        # Create ZIP archive
        zip_path = f"{base_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_path, os.path.basename(json_path))
            zipf.write(excel_path, os.path.basename(excel_path))
            
            # Add CSV files if they exist
            for key in data.keys():
                csv_path = f"{base_path}_{key}.csv"
                if os.path.exists(csv_path):
                    zipf.write(csv_path, os.path.basename(csv_path))
        
        # Clean up individual files
        for path in [json_path, excel_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return zip_path
    
    async def import_data(self, file_path: str, format: ExportFormat, data_type: str) -> Dict[str, Any]:
        """Import data from file"""
        if format == ExportFormat.JSON:
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif format == ExportFormat.CSV:
            df = pd.read_csv(file_path)
            return {data_type: df.to_dict('records')}
        
        elif format == ExportFormat.EXCEL:
            dfs = pd.read_excel(file_path, sheet_name=None)
            result = {}
            for sheet_name, df in dfs.items():
                result[sheet_name] = df.to_dict('records')
            return result
        
        else:
            raise ValueError(f"Import not supported for format: {format}")
    
    async def setup_integration(
        self,
        name: str,
        type: IntegrationType,
        config: Dict[str, Any],
        schedule: Optional[str] = None
    ) -> IntegrationConfig:
        """Setup a new integration"""
        integration_id = f"integration-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        integration = IntegrationConfig(
            id=integration_id,
            name=name,
            type=type,
            enabled=True,
            config=config,
            schedule=schedule
        )
        
        self.integrations[integration_id] = integration
        
        logger.info(f"Integration {name} ({type.value}) configured")
        
        return integration
    
    async def send_to_integration(self, integration_id: str, data: Dict[str, Any]):
        """Send data to integration"""
        if integration_id not in self.integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = self.integrations[integration_id]
        
        if not integration.enabled:
            logger.warning(f"Integration {integration.name} is disabled")
            return
        
        if integration.type == IntegrationType.WEBHOOK:
            await self._send_webhook(integration.config, data)
        elif integration.type == IntegrationType.S3:
            await self._send_s3(integration.config, data)
        elif integration.type == IntegrationType.AZURE_BLOB:
            await self._send_azure_blob(integration.config, data)
        elif integration.type == IntegrationType.EMAIL:
            await self._send_email(integration.config, data)
        else:
            logger.warning(f"Integration type {integration.type} not implemented")
        
        integration.last_run = datetime.utcnow()
    
    async def _send_webhook(self, config: Dict[str, Any], data: Dict[str, Any]):
        """Send data to webhook"""
        url = config.get("url")
        headers = config.get("headers", {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Webhook failed: {response.status}")
                logger.info(f"Data sent to webhook: {url}")
    
    async def _send_s3(self, config: Dict[str, Any], data: Dict[str, Any]):
        """Send data to S3"""
        # Implementation would use boto3
        logger.info("S3 integration not fully implemented")
    
    async def _send_azure_blob(self, config: Dict[str, Any], data: Dict[str, Any]):
        """Send data to Azure Blob Storage"""
        # Implementation would use azure-storage-blob
        logger.info("Azure Blob integration not fully implemented")
    
    async def _send_email(self, config: Dict[str, Any], data: Dict[str, Any]):
        """Send data via email"""
        # Implementation would use email service
        logger.info("Email integration not fully implemented")
    
    def get_export_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job status"""
        return self.export_jobs.get(job_id)
    
    def get_export_jobs(self, limit: int = 10) -> List[ExportJob]:
        """Get recent export jobs"""
        jobs = sorted(self.export_jobs.values(), key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]
    
    def get_integrations(self) -> List[IntegrationConfig]:
        """Get all integrations"""
        return list(self.integrations.values())
    
    async def download_export(self, job_id: str) -> Optional[bytes]:
        """Download export file"""
        job = self.export_jobs.get(job_id)
        
        if not job or not job.file_path:
            return None
        
        with open(job.file_path, 'rb') as f:
            return f.read()

# Singleton instance
data_export_service = DataExportService()