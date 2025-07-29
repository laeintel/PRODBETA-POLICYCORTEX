"""
Query Router Service.
Routes queries to appropriate backend services based on intent and entities.
"""

import json
from typing import Dict, Any, List, Optional
import structlog
import httpx
import asyncio

from ....shared.config import get_settings
    ConversationIntent,
    Entity,
    QueryRouterResult
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class QueryRouter:
    """Routes queries to appropriate backend services."""

    def __init__(self):
        self.settings = settings
        self.service_endpoints = self._initialize_service_endpoints()
        self.timeout = 30
        self.max_retries = 2

    def _initialize_service_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize service endpoint configurations."""
        return {
            "azure_integration": {
                "url": self.settings.azure_integration_url,
                "endpoints": {
                    "cost_analysis": "/api/v1/cost/analyze",
                    "resource_list": "/api/v1/resources",
                    "policy_check": "/api/v1/policy/check",
                    "security_scan": "/api/v1/security/scan",
                    "rbac_analysis": "/api/v1/rbac/analyze",
                    "network_topology": "/api/v1/network/topology",
                    "compliance_check": "/api/v1/compliance/check"
                },
                "timeout": 30
            },
            "ai_engine": {
                "url": self.settings.ai_engine_url,
                "endpoints": {
                    "cost_optimization": "/api/v1/optimize/cost",
                    "anomaly_detection": "/api/v1/detect/anomaly",
                    "predictive_analysis": "/api/v1/predict",
                    "recommendations": "/api/v1/recommend",
                    "sentiment_analysis": "/api/v1/analyze/sentiment"
                },
                "timeout": 60
            },
            "data_processing": {
                "url": self.settings.data_processing_url,
                "endpoints": {
                    "data_aggregation": "/api/v1/aggregate",
                    "metrics_collection": "/api/v1/metrics",
                    "trend_analysis": "/api/v1/trends",
                    "report_generation": "/api/v1/reports"
                },
                "timeout": 45
            }
        }

    async def route_query(
        self,
        intent: ConversationIntent,
        entities: List[Entity],
        message: str,
        user_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Route query to appropriate service and return response."""
        try:
            # Determine routing based on intent
            routing_plan = self._create_routing_plan(intent, entities, message)

            if not routing_plan:
                logger.warning(
                    "no_routing_plan_created",
                    intent=intent.value,
                    message=message
                )
                return None

            # Execute routing plan
            results = await self._execute_routing_plan(routing_plan, entities, user_info)

            # Aggregate results
            aggregated_results = self._aggregate_results(results, intent)

            return aggregated_results

        except Exception as e:
            logger.error(
                "query_routing_failed",
                error=str(e),
                intent=intent.value,
                message=message
            )
            return None

    def _create_routing_plan(
        self,
        intent: ConversationIntent,
        entities: List[Entity],
        message: str
    ) -> List[Dict[str, Any]]:
        """Create routing plan based on intent and entities."""
        routing_plan = []

        if intent == ConversationIntent.COST_ANALYSIS:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "cost_analysis",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "ai_engine",
                    "endpoint": "cost_optimization",
                    "priority": 2,
                    "required": False
                },
                {
                    "service": "data_processing",
                    "endpoint": "trend_analysis",
                    "priority": 3,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.POLICY_QUERY:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "policy_check",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "azure_integration",
                    "endpoint": "compliance_check",
                    "priority": 2,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.RESOURCE_MANAGEMENT:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "resource_list",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "ai_engine",
                    "endpoint": "recommendations",
                    "priority": 2,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.SECURITY_ANALYSIS:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "security_scan",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "ai_engine",
                    "endpoint": "anomaly_detection",
                    "priority": 2,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.RBAC_QUERY:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "rbac_analysis",
                    "priority": 1,
                    "required": True
                }
            ])

        elif intent == ConversationIntent.NETWORK_ANALYSIS:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "network_topology",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "ai_engine",
                    "endpoint": "recommendations",
                    "priority": 2,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.OPTIMIZATION_SUGGESTION:
            routing_plan.extend([
                {
                    "service": "ai_engine",
                    "endpoint": "recommendations",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "azure_integration",
                    "endpoint": "resource_list",
                    "priority": 2,
                    "required": False
                }
            ])

        elif intent == ConversationIntent.COMPLIANCE_CHECK:
            routing_plan.extend([
                {
                    "service": "azure_integration",
                    "endpoint": "compliance_check",
                    "priority": 1,
                    "required": True
                },
                {
                    "service": "azure_integration",
                    "endpoint": "policy_check",
                    "priority": 2,
                    "required": False
                }
            ])

        # Sort by priority
        routing_plan.sort(key=lambda x: x["priority"])

        return routing_plan

    async def _execute_routing_plan(
        self,
        routing_plan: List[Dict[str, Any]],
        entities: List[Entity],
        user_info: Dict[str, Any]
    ) -> List[QueryRouterResult]:
        """Execute routing plan and collect results."""
        results = []

        # Execute required services first
        required_tasks = [
            self._call_service(route, entities, user_info)
            for route in routing_plan
            if route["required"]
        ]

        if required_tasks:
            required_results = await asyncio.gather(*required_tasks, return_exceptions=True)

            for result in required_results:
                if isinstance(result, QueryRouterResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error("required_service_call_failed", error=str(result))

        # Execute optional services
        optional_tasks = [
            self._call_service(route, entities, user_info)
            for route in routing_plan
            if not route["required"]
        ]

        if optional_tasks:
            optional_results = await asyncio.gather(*optional_tasks, return_exceptions=True)

            for result in optional_results:
                if isinstance(result, QueryRouterResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.warning("optional_service_call_failed", error=str(result))

        return results

    async def _call_service(
        self,
        route: Dict[str, Any],
        entities: List[Entity],
        user_info: Dict[str, Any]
    ) -> QueryRouterResult:
        """Call a specific service endpoint."""
        try:
            service_name = route["service"]
            endpoint_name = route["endpoint"]

            service_config = self.service_endpoints[service_name]
            endpoint_path = service_config["endpoints"][endpoint_name]
            service_url = service_config["url"]
            timeout = service_config["timeout"]

            # Build request parameters
            request_params = self._build_request_params(
                endpoint_name, entities, user_info
            )

            # Make HTTP request
            full_url = f"{service_url}{endpoint_path}"

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    full_url,
                    json=request_params,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {user_info.get('token', '')}",
                        "X-User-ID": user_info.get("id", ""),
                        "X-Tenant-ID": user_info.get("tenant_id", "")
                    }
                )

                response.raise_for_status()
                response_data = response.json()

            # Calculate confidence based on response quality
            confidence = self._calculate_service_confidence(
                response_data, endpoint_name
            )

            return QueryRouterResult(
                service=service_name,
                endpoint=endpoint_path,
                parameters=request_params,
                confidence=confidence,
                data=response_data
            )

        except httpx.TimeoutException:
            logger.error(
                "service_call_timeout",
                service=service_name,
                endpoint=endpoint_name
            )
            raise Exception(f"Service call timeout: {service_name}/{endpoint_name}")

        except httpx.HTTPStatusError as e:
            logger.error(
                "service_call_http_error",
                service=service_name,
                endpoint=endpoint_name,
                status_code=e.response.status_code
            )
            raise Exception(f"Service call HTTP error: {e.response.status_code}")

        except Exception as e:
            logger.error(
                "service_call_failed",
                service=service_name,
                endpoint=endpoint_name,
                error=str(e)
            )
            raise Exception(f"Service call failed: {str(e)}")

    def _build_request_params(
        self,
        endpoint_name: str,
        entities: List[Entity],
        user_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build request parameters for service calls."""
        params = {
            "user_id": user_info.get("id"),
            "tenant_id": user_info.get("tenant_id"),
            "subscription_ids": user_info.get("subscription_ids", [])
        }

        # Add entity-based parameters
        entity_params = {}
        for entity in entities:
            entity_type = entity.type.value
            entity_value = entity.value

            if entity_type == "subscription":
                entity_params["subscription_id"] = entity_value
            elif entity_type == "resource_group":
                entity_params["resource_group"] = entity_value
            elif entity_type == "location":
                entity_params["location"] = entity_value
            elif entity_type == "resource_type":
                entity_params["resource_type"] = entity_value
            elif entity_type == "date_range":
                entity_params["date_range"] = entity_value
            elif entity_type == "cost_threshold":
                entity_params["cost_threshold"] = entity_value
            elif entity_type == "tag":
                if "tags" not in entity_params:
                    entity_params["tags"] = []
                entity_params["tags"].append(entity_value)

        params.update(entity_params)

        # Add endpoint-specific parameters
        if endpoint_name == "cost_analysis":
            params.update({
                "include_forecasting": True,
                "granularity": "daily",
                "group_by": ["resource_type", "location"]
            })

        elif endpoint_name == "policy_check":
            params.update({
                "include_violations": True,
                "policy_types": ["security", "compliance", "cost"]
            })

        elif endpoint_name == "security_scan":
            params.update({
                "scan_types": ["vulnerability", "configuration", "access"],
                "include_recommendations": True
            })

        elif endpoint_name == "rbac_analysis":
            params.update({
                "include_permissions": True,
                "analyze_access_patterns": True
            })

        elif endpoint_name == "resource_list":
            params.update({
                "include_metadata": True,
                "include_tags": True,
                "include_dependencies": True
            })

        elif endpoint_name == "recommendations":
            params.update({
                "recommendation_types": ["cost", "security", "performance"],
                "priority_threshold": "medium"
            })

        return params

    def _calculate_service_confidence(
        self,
        response_data: Dict[str, Any],
        endpoint_name: str
    ) -> float:
        """Calculate confidence score for service response."""
        confidence = 0.7  # Base confidence

        # Check response completeness
        if response_data.get("success", False):
            confidence += 0.1

        # Check data quality
        data = response_data.get("data", {})
        if data:
            confidence += 0.1

            # Check for specific data indicators
            if endpoint_name == "cost_analysis":
                if "total_cost" in data and "breakdown" in data:
                    confidence += 0.1

            elif endpoint_name == "policy_check":
                if "compliant" in data and "violations" in data:
                    confidence += 0.1

            elif endpoint_name == "security_scan":
                if "findings" in data and "score" in data:
                    confidence += 0.1

        # Check for errors
        if response_data.get("error"):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _aggregate_results(
        self,
        results: List[QueryRouterResult],
        intent: ConversationIntent
    ) -> Dict[str, Any]:
        """Aggregate results from multiple services."""
        if not results:
            return {}

        aggregated = {
            "intent": intent.value,
            "services_called": len(results),
            "total_confidence": sum(r.confidence for r in results) / len(results),
            "data": {},
            "summary": {}
        }

        # Aggregate by service
        for result in results:
            service_name = result.service
            aggregated["data"][service_name] = result.data

        # Create intent-specific summaries
        if intent == ConversationIntent.COST_ANALYSIS:
            aggregated["summary"] = self._summarize_cost_analysis(results)

        elif intent == ConversationIntent.POLICY_QUERY:
            aggregated["summary"] = self._summarize_policy_analysis(results)

        elif intent == ConversationIntent.SECURITY_ANALYSIS:
            aggregated["summary"] = self._summarize_security_analysis(results)

        elif intent == ConversationIntent.RESOURCE_MANAGEMENT:
            aggregated["summary"] = self._summarize_resource_analysis(results)

        return aggregated

    def _summarize_cost_analysis(self, results: List[QueryRouterResult]) -> Dict[str, Any]:
        """Summarize cost analysis results."""
        summary = {
            "total_cost": 0,
            "cost_breakdown": {},
            "optimization_opportunities": [],
            "trends": {}
        }

        for result in results:
            if result.service == "azure_integration":
                data = result.data.get("data", {})
                if "total_cost" in data:
                    summary["total_cost"] = data["total_cost"]
                if "breakdown" in data:
                    summary["cost_breakdown"] = data["breakdown"]

            elif result.service == "ai_engine":
                data = result.data.get("data", {})
                if "recommendations" in data:
                    summary["optimization_opportunities"] = data["recommendations"]

            elif result.service == "data_processing":
                data = result.data.get("data", {})
                if "trends" in data:
                    summary["trends"] = data["trends"]

        return summary

    def _summarize_policy_analysis(self, results: List[QueryRouterResult]) -> Dict[str, Any]:
        """Summarize policy analysis results."""
        summary = {
            "compliance_status": "unknown",
            "violations": [],
            "recommendations": []
        }

        for result in results:
            if result.service == "azure_integration":
                data = result.data.get("data", {})
                if "compliant" in data:
                    summary["compliance_status"] = (
                        "compliant" if data["compliant"] else "non_compliant"
                    )
                if "violations" in data:
                    summary["violations"] = data["violations"]
                if "recommendations" in data:
                    summary["recommendations"] = data["recommendations"]

        return summary

    def _summarize_security_analysis(self, results: List[QueryRouterResult]) -> Dict[str, Any]:
        """Summarize security analysis results."""
        summary = {
            "security_score": 0,
            "vulnerabilities": [],
            "recommendations": [],
            "anomalies": []
        }

        for result in results:
            if result.service == "azure_integration":
                data = result.data.get("data", {})
                if "score" in data:
                    summary["security_score"] = data["score"]
                if "findings" in data:
                    summary["vulnerabilities"] = data["findings"]

            elif result.service == "ai_engine":
                data = result.data.get("data", {})
                if "anomalies" in data:
                    summary["anomalies"] = data["anomalies"]
                if "recommendations" in data:
                    summary["recommendations"] = data["recommendations"]

        return summary

    def _summarize_resource_analysis(self, results: List[QueryRouterResult]) -> Dict[str, Any]:
        """Summarize resource analysis results."""
        summary = {
            "resource_count": 0,
            "resource_types": {},
            "recommendations": [],
            "optimization_opportunities": []
        }

        for result in results:
            if result.service == "azure_integration":
                data = result.data.get("data", {})
                if "resources" in data:
                    resources = data["resources"]
                    summary["resource_count"] = len(resources)

                    # Count by type
                    for resource in resources:
                        resource_type = resource.get("type", "unknown")
                        summary["resource_types"][resource_type] = summary["resource_types"].get(
                            resource_type,
                            0
                        ) + 1

            elif result.service == "ai_engine":
                data = result.data.get("data", {})
                if "recommendations" in data:
                    summary["recommendations"] = data["recommendations"]

        return summary
