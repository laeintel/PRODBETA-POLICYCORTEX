"""
Azure Resource Management service for handling Azure resources and resource groups.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.resource import ResourceManagementClient
from shared.config import get_settings

from ..models import ResourceGroupResponse, ResourceMetrics, ResourceResponse
from .azure_auth import AzureAuthService

settings = get_settings()
logger = structlog.get_logger(__name__)


class ResourceManagementService:
    """Service for managing Azure resources and resource groups."""

    def __init__(self):
        self.settings = settings
        self.auth_service = AzureAuthService()
        self.resource_clients = {}
        self.monitor_clients = {}

    async def _get_resource_client(self, subscription_id: str) -> ResourceManagementClient:
        """Get or create Resource Management client for subscription."""
        if subscription_id not in self.resource_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.resource_clients[subscription_id] = ResourceManagementClient(
                credential, subscription_id
            )
        return self.resource_clients[subscription_id]

    async def _get_monitor_client(self, subscription_id: str) -> MonitorManagementClient:
        """Get or create Monitor client for subscription."""
        if subscription_id not in self.monitor_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.monitor_clients[subscription_id] = MonitorManagementClient(
                credential, subscription_id
            )
        return self.monitor_clients[subscription_id]

    async def list_resources(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None,
        resource_type: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[ResourceResponse]:
        """List all resources in subscription or resource group."""
        try:
            client = await self._get_resource_client(subscription_id)
            resources = []

            # Build filter
            filter_str = []
            if resource_type:
                filter_str.append(f"resourceType eq '{resource_type}'")
            if tags:
                for key, value in tags.items():
                    filter_str.append(f"tagname eq '{key}' and tagvalue eq '{value}'")

            filter_query = " and ".join(filter_str) if filter_str else None

            # List resources
            if resource_group:
                resource_list = client.resources.list_by_resource_group(
                    resource_group_name=resource_group,
                    filter=filter_query
                )
            else:
                resource_list = client.resources.list(filter=filter_query)

            async for resource in resource_list:
                resources.append(ResourceResponse(
                    id=resource.id,
                    name=resource.name,
                    type=resource.type,
                    location=resource.location,
                    resource_group=resource.id.split('/')[4],  # Extract RG from resource ID
                    subscription_id=subscription_id,
                    kind=resource.kind,
                    sku=resource.sku.dict() if resource.sku else None,
                    tags=resource.tags,
                    properties=resource.properties,
                    provisioning_state = (
                        resource.properties.get("provisioningState") if resource.properties else None,
                    )
                    created_time=resource.created_time,
                    changed_time=resource.changed_time
                ))

            logger.info(
                "resources_listed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                resource_type=resource_type,
                count=len(resources)
            )

            return resources

        except AzureError as e:
            logger.error(
                "list_resources_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list resources: {str(e)}")

    async def get_resource(
        self,
        resource_id: str
    ) -> ResourceResponse:
        """Get a specific resource by ID."""
        try:
            # Extract subscription ID from resource ID
            resource_id_parts = resource_id.split('/')
            subscription_id = resource_id_parts[2]

            client = await self._get_resource_client(subscription_id)

            # Get resource
            resource = await client.resources.get_by_id(
                resource_id=resource_id,
                api_version="2021-04-01"
            )

            logger.info(
                "resource_retrieved",
                resource_id=resource_id
            )

            return ResourceResponse(
                id=resource.id,
                name=resource.name,
                type=resource.type,
                location=resource.location,
                resource_group=resource.id.split('/')[4],  # Extract RG from resource ID
                subscription_id=subscription_id,
                kind=resource.kind,
                sku=resource.sku.dict() if resource.sku else None,
                tags=resource.tags,
                properties=resource.properties,
                provisioning_state = (
                    resource.properties.get("provisioningState") if resource.properties else None,
                )
                created_time=resource.created_time,
                changed_time=resource.changed_time
            )

        except ResourceNotFoundError:
            logger.error(
                "resource_not_found",
                resource_id=resource_id
            )
            raise Exception(f"Resource {resource_id} not found")
        except AzureError as e:
            logger.error(
                "get_resource_failed",
                error=str(e),
                resource_id=resource_id
            )
            raise Exception(f"Failed to get resource: {str(e)}")

    async def list_resource_groups(
        self,
        subscription_id: str
    ) -> List[Dict[str, Any]]:
        """List all resource groups in subscription."""
        try:
            client = await self._get_resource_client(subscription_id)
            resource_groups = []

            # List resource groups
            rg_list = client.resource_groups.list()

            async for rg in rg_list:
                resource_groups.append({
                    "id": rg.id,
                    "name": rg.name,
                    "type": rg.type,
                    "location": rg.location,
                    "subscription_id": subscription_id,
                    "tags": rg.tags,
                    "properties": rg.properties.dict() if rg.properties else {},
                    "managed_by": rg.managed_by
                })

            logger.info(
                "resource_groups_listed",
                subscription_id=subscription_id,
                count=len(resource_groups)
            )

            return resource_groups

        except AzureError as e:
            logger.error(
                "list_resource_groups_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list resource groups: {str(e)}")

    async def create_resource_group(
        self,
        subscription_id: str,
        resource_group_name: str,
        location: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a new resource group."""
        try:
            client = await self._get_resource_client(subscription_id)

            # Create resource group parameters
            rg_params = {
                "location": location,
                "tags": tags or {}
            }

            # Create resource group
            rg = await client.resource_groups.create_or_update(
                resource_group_name=resource_group_name,
                parameters=rg_params
            )

            logger.info(
                "resource_group_created",
                subscription_id=subscription_id,
                resource_group_name=resource_group_name,
                location=location
            )

            return {
                "id": rg.id,
                "name": rg.name,
                "type": rg.type,
                "location": rg.location,
                "subscription_id": subscription_id,
                "tags": rg.tags,
                "properties": rg.properties.dict() if rg.properties else {},
                "managed_by": rg.managed_by
            }

        except AzureError as e:
            logger.error(
                "create_resource_group_failed",
                error=str(e),
                subscription_id=subscription_id,
                resource_group_name=resource_group_name
            )
            raise Exception(f"Failed to create resource group: {str(e)}")

    async def delete_resource_group(
        self,
        subscription_id: str,
        resource_group_name: str
    ) -> None:
        """Delete a resource group."""
        try:
            client = await self._get_resource_client(subscription_id)

            # Delete resource group (async operation)
            await client.resource_groups.begin_delete(resource_group_name)

            logger.info(
                "resource_group_deleted",
                subscription_id=subscription_id,
                resource_group_name=resource_group_name
            )

        except ResourceNotFoundError:
            logger.error(
                "resource_group_not_found",
                subscription_id=subscription_id,
                resource_group_name=resource_group_name
            )
            raise Exception(f"Resource group {resource_group_name} not found")
        except AzureError as e:
            logger.error(
                "delete_resource_group_failed",
                error=str(e),
                subscription_id=subscription_id,
                resource_group_name=resource_group_name
            )
            raise Exception(f"Failed to delete resource group: {str(e)}")

    async def update_resource_tags(
        self,
        resource_id: str,
        tags: Dict[str, str]
    ) -> None:
        """Update tags for a resource."""
        try:
            # Extract subscription ID from resource ID
            resource_id_parts = resource_id.split('/')
            subscription_id = resource_id_parts[2]

            client = await self._get_resource_client(subscription_id)

            # Get current resource
            resource = await client.resources.get_by_id(
                resource_id=resource_id,
                api_version="2021-04-01"
            )

            # Update tags
            resource.tags = tags

            # Update resource
            await client.resources.begin_update_by_id(
                resource_id=resource_id,
                api_version="2021-04-01",
                parameters=resource
            )

            logger.info(
                "resource_tags_updated",
                resource_id=resource_id,
                tags=tags
            )

        except ResourceNotFoundError:
            logger.error(
                "resource_not_found",
                resource_id=resource_id
            )
            raise Exception(f"Resource {resource_id} not found")
        except AzureError as e:
            logger.error(
                "update_resource_tags_failed",
                error=str(e),
                resource_id=resource_id
            )
            raise Exception(f"Failed to update resource tags: {str(e)}")

    async def get_resource_metrics(
        self,
        resource_id: str,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        time_grain: str = "PT1M"
    ) -> List[ResourceMetrics]:
        """Get metrics for a resource."""
        try:
            # Extract subscription ID from resource ID
            resource_id_parts = resource_id.split('/')
            subscription_id = resource_id_parts[2]

            monitor_client = await self._get_monitor_client(subscription_id)
            metrics = []

            for metric_name in metric_names:
                try:
                    # Get metrics
                    metrics_data = await monitor_client.metrics.list(
                        resource_uri=resource_id,
                        metricnames=metric_name,
                        timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                        interval=time_grain
                    )

                    # Process metrics data
                    for metric in metrics_data.value:
                        data_points = []
                        for timeseries in metric.timeseries:
                            for data_point in timeseries.data:
                                data_points.append({
                                    "timestamp": data_point.time_stamp.isoformat(),
                                    "value": data_point.average or data_point.total or data_point.count,
                                    "unit": metric.unit.value if metric.unit else None
                                })

                        metrics.append(ResourceMetrics(
                            resource_id=resource_id,
                            metric_name=metric.name.value,
                            time_grain=time_grain,
                            unit=metric.unit.value if metric.unit else "None",
                            data_points=data_points,
                            start_time=start_time,
                            end_time=end_time
                        ))

                except Exception as e:
                    logger.warning(
                        "metric_retrieval_failed",
                        resource_id=resource_id,
                        metric_name=metric_name,
                        error=str(e)
                    )
                    continue

            logger.info(
                "resource_metrics_retrieved",
                resource_id=resource_id,
                metric_count=len(metrics)
            )

            return metrics

        except Exception as e:
            logger.error(
                "get_resource_metrics_failed",
                error=str(e),
                resource_id=resource_id
            )
            raise Exception(f"Failed to get resource metrics: {str(e)}")

    async def get_resource_health(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get health status of resources."""
        try:
            # Get all resources
            resources = await self.list_resources(subscription_id, resource_group)

            # Analyze resource health
            health_summary = {
                "total_resources": len(resources),
                "healthy_resources": 0,
                "unhealthy_resources": 0,
                "unknown_resources": 0,
                "resource_types": {},
                "unhealthy_details": []
            }

            for resource in resources:
                # Check provisioning state
                provisioning_state = resource.provisioning_state
                if provisioning_state == "Succeeded":
                    health_summary["healthy_resources"] += 1
                elif provisioning_state in ["Failed", "Canceled"]:
                    health_summary["unhealthy_resources"] += 1
                    health_summary["unhealthy_details"].append({
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "resource_type": resource.type,
                        "provisioning_state": provisioning_state
                    })
                else:
                    health_summary["unknown_resources"] += 1

                # Count resource types
                if resource.type not in health_summary["resource_types"]:
                    health_summary["resource_types"][resource.type] = 0
                health_summary["resource_types"][resource.type] += 1

            logger.info(
                "resource_health_analyzed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                total_resources=health_summary["total_resources"],
                healthy_resources=health_summary["healthy_resources"],
                unhealthy_resources=health_summary["unhealthy_resources"]
            )

            return health_summary

        except Exception as e:
            logger.error(
                "get_resource_health_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to get resource health: {str(e)}")

    async def get_resource_recommendations(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get resource optimization recommendations."""
        try:
            recommendations = []

            # Get all resources
            resources = await self.list_resources(subscription_id, resource_group)

            # Analyze resources for optimization opportunities
            untagged_resources = []
            old_resources = []
            unused_resources = []

            for resource in resources:
                # Check for untagged resources
                if not resource.tags or len(resource.tags) == 0:
                    untagged_resources.append(resource)

                # Check for old resources (older than 90 days)
                if resource.created_time:
                    days_old = (datetime.now() - resource.created_time.replace(tzinfo=None)).days
                    if days_old > 90:
                        old_resources.append(resource)

                # Check for potentially unused resources
                if resource.type in ["Microsoft.Storage/storageAccounts", "Microsoft.Network/publicIPAddresses"]:
                    # These resource types often become unused
                    unused_resources.append(resource)

            # Generate recommendations
            if untagged_resources:
                recommendations.append({
                    "type": "governance",
                    "priority": "medium",
                    "title": "Add tags to resources",
                    "description": f"Found {len(untagged_resources)} resources without tags. Add tags for better resource management.",
                    "affected_resources": [r.id for r in untagged_resources[:10]],
                    "actions": [
                        "Add environment tags (dev, staging, prod)",
                        "Add cost center tags",
                        "Add owner tags"
                    ]
                })

            if old_resources:
                recommendations.append({
                    "type": "lifecycle",
                    "priority": "low",
                    "title": "Review old resources",
                    "description": f"Found {len(old_resources)} resources older than 90 days. Review if they are still needed.",
                    "affected_resources": [r.id for r in old_resources[:10]],
                    "actions": [
                        "Review resource necessity",
                        "Archive unused resources",
                        "Consider resource lifecycle policies"
                    ]
                })

            if unused_resources:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "medium",
                    "title": "Review potentially unused resources",
                    "description": f"Found {len(unused_resources)} resources that might be unused. Review and
                        clean up.",
                    "affected_resources": [r.id for r in unused_resources[:10]],
                    "actions": [
                        "Check resource utilization",
                        "Delete unused resources",
                        "Consolidate similar resources"
                    ]
                })

            logger.info(
                "resource_recommendations_generated",
                subscription_id=subscription_id,
                resource_group=resource_group,
                recommendation_count=len(recommendations)
            )

            return recommendations

        except Exception as e:
            logger.error(
                "get_resource_recommendations_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            return []
