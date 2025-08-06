"""
Compliance Engine Proxy for API Gateway
Routes compliance-related requests to the Phase 2 Compliance Engine service
"""

import json
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class ComplianceEngineProxy:
    """
    Proxy for routing requests to the Compliance Engine service
    """

    def __init__(self, compliance_engine_url: str):
        self.compliance_engine_url = compliance_engine_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to compliance engine"""
        await self._ensure_session()

        url = f"{self.compliance_engine_url}{endpoint}"

        try:
            async with self.session.request(
                method=method, url=url, json=data, params=params, headers=headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"Compliance engine error: {response.status} - {error_text}")
                    raise Exception(f"Compliance engine error: {response.status}")

                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to compliance engine: {e}")
            raise Exception(f"Compliance engine unavailable: {e}")

    # Document Processing Methods
    async def upload_document(
        self, file_content: bytes, filename: str, tenant_id: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Upload document for policy extraction"""
        await self._ensure_session()

        url = f"{self.compliance_engine_url}/api/v1/documents/upload"

        data = aiohttp.FormData()
        data.add_field(
            "file", file_content, filename=filename, content_type="application/octet-stream"
        )
        data.add_field("tenant_id", tenant_id)
        if metadata:
            data.add_field("metadata", json.dumps(metadata))

        try:
            async with self.session.post(url, data=data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Document upload failed: {response.status}")

                return await response.json()

        except Exception as e:
            logger.error(f"Document upload error: {e}")
            raise

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get document details and extracted policies"""
        return await self._make_request("GET", f"/api/v1/documents/{document_id}")

    async def list_documents(
        self, tenant_id: str, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List documents for a tenant"""
        params = {"tenant_id": tenant_id, "limit": limit}
        if status:
            params["status"] = status

        return await self._make_request("GET", "/api/v1/documents", params=params)

    # Policy Extraction Methods
    async def extract_policies(self, text: str) -> Dict[str, Any]:
        """Extract policies from text"""
        return await self._make_request("POST", "/api/v1/policies/extract", data={"text": text})

    # Compliance Analysis Methods
    async def analyze_compliance(
        self, resources: List[Dict], policies: List[Dict], tenant_id: str, real_time: bool = True
    ) -> Dict[str, Any]:
        """Analyze resource compliance"""
        return await self._make_request(
            "POST",
            "/api/v1/compliance/analyze",
            data={
                "resources": resources,
                "policies": policies,
                "tenant_id": tenant_id,
                "real_time": real_time,
            },
        )

    async def predict_compliance(self, tenant_id: str, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future compliance trends"""
        return await self._make_request(
            "GET", f"/api/v1/compliance/predict/{tenant_id}", params={"days_ahead": days_ahead}
        )

    async def get_compliance_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get compliance metrics for tenant"""
        # Mock implementation for now
        return {
            "overallScore": 85.5,
            "complianceLevel": "good",
            "totalResources": 150,
            "compliantResources": 128,
            "nonCompliantResources": 22,
            "criticalViolations": 2,
            "highViolations": 5,
            "mediumViolations": 10,
            "lowViolations": 5,
        }

    async def get_compliance_resources(
        self, tenant_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get resource compliance details"""
        # Mock implementation
        return [
            {
                "resourceId": "res-001",
                "resourceName": "Storage Account",
                "resourceType": "Storage",
                "complianceStatus": "compliant",
                "complianceScore": 95.0,
                "violations": [],
                "lastChecked": datetime.utcnow().isoformat(),
            },
            {
                "resourceId": "res-002",
                "resourceName": "Key Vault",
                "resourceType": "Security",
                "complianceStatus": "non_compliant",
                "complianceScore": 60.0,
                "violations": [{"severity": "high", "description": "Missing firewall rules"}],
                "lastChecked": datetime.utcnow().isoformat(),
            },
        ]

    async def get_compliance_coverage(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get policy coverage analysis"""
        # Mock implementation
        return [
            {"policyName": "Encryption at Rest", "coverage": 95},
            {"policyName": "Network Security", "coverage": 88},
            {"policyName": "Access Control", "coverage": 92},
            {"policyName": "Data Classification", "coverage": 78},
            {"policyName": "Backup Policy", "coverage": 85},
        ]

    async def get_compliance_trends(
        self, tenant_id: str, range: str = "7d"
    ) -> List[Dict[str, Any]]:
        """Get compliance trend data"""
        # Mock implementation
        trends = []
        for i in range(7):
            trends.append({"date": f"Day {i+1}", "score": 80 + (i * 2), "violations": 20 - i})
        return trends

    # Rule Engine Methods
    async def create_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compliance rule"""
        return await self._make_request("POST", "/api/v1/rules", data=rule)

    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a compliance rule"""
        return await self._make_request("PUT", f"/api/v1/rules/{rule_id}", data=updates)

    async def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete a compliance rule"""
        return await self._make_request("DELETE", f"/api/v1/rules/{rule_id}")

    async def evaluate_rules(
        self,
        resource: Dict[str, Any],
        rule_ids: Optional[List[str]] = None,
        execute_actions: bool = True,
    ) -> List[Dict[str, Any]]:
        """Evaluate rules against a resource"""
        return await self._make_request(
            "POST",
            "/api/v1/rules/evaluate",
            data={"resource": resource, "rule_ids": rule_ids, "execute_actions": execute_actions},
        )

    async def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule execution statistics"""
        return await self._make_request("GET", "/api/v1/rules/statistics")

    async def export_rules(self, format: str = "json") -> str:
        """Export all rules"""
        response = await self._make_request(
            "GET", "/api/v1/rules/export", params={"format": format}
        )
        return response.get("data", "")

    async def import_rules(self, data: str, format: str = "json") -> Dict[str, Any]:
        """Import rules"""
        return await self._make_request(
            "POST", "/api/v1/rules/import", data={"data": data, "format": format}
        )

    # Visual Rule Builder Methods
    async def create_rule_builder_session(self) -> str:
        """Create a new rule builder session"""
        response = await self._make_request("POST", "/api/v1/rule-builder/sessions")
        return response.get("session_id")

    async def get_rule_builder_session(self, session_id: str) -> Dict[str, Any]:
        """Get rule builder session state"""
        return await self._make_request("GET", f"/api/v1/rule-builder/sessions/{session_id}")

    async def get_rule_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available rule templates"""
        params = {"category": category} if category else {}
        return await self._make_request("GET", "/api/v1/rule-builder/templates", params=params)

    async def get_rule_components(self) -> Dict[str, Any]:
        """Get component library for rule builder"""
        return await self._make_request("GET", "/api/v1/rule-builder/components")

    async def add_rule_component(self, session_id: str, component: Dict[str, Any]) -> str:
        """Add component to rule builder"""
        response = await self._make_request(
            "POST", f"/api/v1/rule-builder/sessions/{session_id}/components", data=component
        )
        return response.get("component_id")

    async def validate_rule_builder(self, session_id: str) -> Dict[str, Any]:
        """Validate rule configuration"""
        return await self._make_request(
            "POST", f"/api/v1/rule-builder/sessions/{session_id}/validate"
        )

    async def compile_rule(self, session_id: str) -> Dict[str, Any]:
        """Compile visual rule to compliance rule"""
        return await self._make_request(
            "POST", f"/api/v1/rule-builder/sessions/{session_id}/compile"
        )

    async def save_rule_from_builder(self, session_id: str) -> str:
        """Save rule from builder"""
        response = await self._make_request(
            "POST", f"/api/v1/rule-builder/sessions/{session_id}/save"
        )
        return response.get("rule_id")

    async def export_compliance_report(self, tenant_id: str) -> bytes:
        """Export compliance report as PDF"""
        # Mock implementation - return empty bytes for now
        return b"Mock PDF content"

    async def health_check(self) -> Dict[str, Any]:
        """Check compliance engine health"""
        try:
            return await self._make_request("GET", "/health")
        except:
            return {
                "status": "unhealthy",
                "service": "compliance-engine",
                "timestamp": datetime.utcnow().isoformat(),
            }
