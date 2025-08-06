"""
Azure Policy management service for handling policy operations.
"""

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog
from azure.core.exceptions import AzureError
from azure.core.exceptions import ResourceNotFoundError
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.resource import PolicyClient
from services.azure_integration.models import PolicyComplianceResponse
from services.azure_integration.models import PolicyComplianceState
from services.azure_integration.models import PolicyResponse
from services.azure_integration.services.azure_auth import AzureAuthService
from shared.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class PolicyManagementService:
    """Service for managing Azure policies."""

    def __init__(self):
        self.settings = settings
        self.auth_service = AzureAuthService()
        self.policy_clients = {}
        self.insights_clients = {}

    async def _get_policy_client(self, subscription_id: str) -> PolicyClient:
        """Get or create Policy client for subscription."""
        if subscription_id not in self.policy_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.policy_clients[subscription_id] = PolicyClient(credential, subscription_id)
        return self.policy_clients[subscription_id]

    async def _get_insights_client(self, subscription_id: str) -> PolicyInsightsClient:
        """Get or create Policy Insights client for subscription."""
        if subscription_id not in self.insights_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.insights_clients[subscription_id] = PolicyInsightsClient(
                credential, subscription_id
            )
        return self.insights_clients[subscription_id]

    async def list_policies(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None,
        policy_type: Optional[str] = None,
    ) -> List[PolicyResponse]:
        """List all policies in subscription or resource group."""
        try:
            client = await self._get_policy_client(subscription_id)
            policies = []

            # List policies
            if resource_group:
                # List policies in resource group
                policy_list = client.policy_definitions.list_by_management_group(resource_group)
            else:
                # List all policies in subscription
                policy_list = client.policy_definitions.list()

            async for policy in policy_list:
                # Filter by policy type if specified
                if policy_type and policy.policy_type != policy_type:
                    continue

                policies.append(
                    PolicyResponse(
                        id=policy.id,
                        name=policy.name,
                        type=policy.type,
                        display_name=policy.display_name or policy.name,
                        description=policy.description,
                        policy_type=policy.policy_type,
                        mode=policy.mode,
                        metadata=policy.metadata,
                        parameters=policy.parameters,
                        policy_rule=policy.policy_rule,
                    )
                )

            logger.info(
                "policies_listed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                count=len(policies),
            )

            return policies

        except AzureError as e:
            logger.error("list_policies_failed", error=str(e), subscription_id=subscription_id)
            raise Exception(f"Failed to list policies: {str(e)}")

    async def get_policy(self, subscription_id: str, policy_id: str) -> PolicyResponse:
        """Get a specific policy by ID."""
        try:
            client = await self._get_policy_client(subscription_id)

            # Extract policy name from ID if full resource ID provided
            policy_name = policy_id.split("/")[-1] if "/" in policy_id else policy_id

            policy = await client.policy_definitions.get(policy_name)

            logger.info("policy_retrieved", subscription_id=subscription_id, policy_id=policy_id)

            return PolicyResponse(
                id=policy.id,
                name=policy.name,
                type=policy.type,
                display_name=policy.display_name or policy.name,
                description=policy.description,
                policy_type=policy.policy_type,
                mode=policy.mode,
                metadata=policy.metadata,
                parameters=policy.parameters,
                policy_rule=policy.policy_rule,
            )

        except ResourceNotFoundError:
            logger.error("policy_not_found", subscription_id=subscription_id, policy_id=policy_id)
            raise Exception(f"Policy {policy_id} not found")
        except AzureError as e:
            logger.error(
                "get_policy_failed",
                error=str(e),
                subscription_id=subscription_id,
                policy_id=policy_id,
            )
            raise Exception(f"Failed to get policy: {str(e)}")

    async def create_policy(
        self, subscription_id: str, policy_data: Dict[str, Any]
    ) -> PolicyResponse:
        """Create a new policy."""
        try:
            client = await self._get_policy_client(subscription_id)

            # Create policy definition
            policy_def = {
                "display_name": policy_data["display_name"],
                "description": policy_data.get("description"),
                "policy_type": policy_data.get("policy_type", "Custom"),
                "mode": policy_data.get("mode", "All"),
                "metadata": policy_data.get("metadata", {}),
                "parameters": policy_data.get("parameters", {}),
                "policy_rule": policy_data["policy_rule"],
            }

            # Create policy
            policy = await client.policy_definitions.create_or_update(
                policy_definition_name=policy_data["name"], parameters=policy_def
            )

            logger.info(
                "policy_created", subscription_id=subscription_id, policy_name=policy_data["name"]
            )

            return PolicyResponse(
                id=policy.id,
                name=policy.name,
                type=policy.type,
                display_name=policy.display_name,
                description=policy.description,
                policy_type=policy.policy_type,
                mode=policy.mode,
                metadata=policy.metadata,
                parameters=policy.parameters,
                policy_rule=policy.policy_rule,
                created_on=datetime.utcnow(),
            )

        except AzureError as e:
            logger.error(
                "create_policy_failed",
                error=str(e),
                subscription_id=subscription_id,
                policy_name=policy_data.get("name"),
            )
            raise Exception(f"Failed to create policy: {str(e)}")

    async def update_policy(
        self, subscription_id: str, policy_id: str, policy_data: Dict[str, Any]
    ) -> PolicyResponse:
        """Update an existing policy."""
        try:
            client = await self._get_policy_client(subscription_id)

            # Extract policy name from ID
            policy_name = policy_id.split("/")[-1] if "/" in policy_id else policy_id

            # Get existing policy
            existing_policy = await client.policy_definitions.get(policy_name)

            # Update policy definition
            policy_def = {
                "display_name": policy_data.get("display_name", existing_policy.display_name),
                "description": policy_data.get("description", existing_policy.description),
                "policy_type": existing_policy.policy_type,  # Can't change type
                "mode": policy_data.get("mode", existing_policy.mode),
                "metadata": policy_data.get("metadata", existing_policy.metadata),
                "parameters": policy_data.get("parameters", existing_policy.parameters),
                "policy_rule": policy_data.get("policy_rule", existing_policy.policy_rule),
            }

            # Update policy
            policy = await client.policy_definitions.create_or_update(
                policy_definition_name=policy_name, parameters=policy_def
            )

            logger.info("policy_updated", subscription_id=subscription_id, policy_id=policy_id)

            return PolicyResponse(
                id=policy.id,
                name=policy.name,
                type=policy.type,
                display_name=policy.display_name,
                description=policy.description,
                policy_type=policy.policy_type,
                mode=policy.mode,
                metadata=policy.metadata,
                parameters=policy.parameters,
                policy_rule=policy.policy_rule,
                updated_on=datetime.utcnow(),
            )

        except ResourceNotFoundError:
            logger.error("policy_not_found", subscription_id=subscription_id, policy_id=policy_id)
            raise Exception(f"Policy {policy_id} not found")
        except AzureError as e:
            logger.error(
                "update_policy_failed",
                error=str(e),
                subscription_id=subscription_id,
                policy_id=policy_id,
            )
            raise Exception(f"Failed to update policy: {str(e)}")

    async def delete_policy(self, subscription_id: str, policy_id: str) -> None:
        """Delete a policy."""
        try:
            client = await self._get_policy_client(subscription_id)

            # Extract policy name from ID
            policy_name = policy_id.split("/")[-1] if "/" in policy_id else policy_id

            # Delete policy
            await client.policy_definitions.delete(policy_name)

            logger.info("policy_deleted", subscription_id=subscription_id, policy_id=policy_id)

        except ResourceNotFoundError:
            logger.error("policy_not_found", subscription_id=subscription_id, policy_id=policy_id)
            raise Exception(f"Policy {policy_id} not found")
        except AzureError as e:
            logger.error(
                "delete_policy_failed",
                error=str(e),
                subscription_id=subscription_id,
                policy_id=policy_id,
            )
            raise Exception(f"Failed to delete policy: {str(e)}")

    async def get_policy_compliance(self, subscription_id: str, policy_id: str) -> Dict[str, Any]:
        """Get compliance status for a policy."""
        try:
            insights_client = await self._get_insights_client(subscription_id)

            # Get policy states
            scope = f"/subscriptions/{subscription_id}"
            policy_states = insights_client.policy_states.list_query_results_for_subscription(
                subscription_id=subscription_id,
                policy_states_resource="latest",
                query_options={"$filter": f"policyDefinitionId eq '{policy_id}'"},
            )

            # Calculate compliance statistics
            total_resources = 0
            compliant = 0
            non_compliant = 0
            conflicting = 0
            exempt = 0

            compliance_details = []

            async for state in policy_states:
                total_resources += 1

                if state.compliance_state == "Compliant":
                    compliant += 1
                elif state.compliance_state == "NonCompliant":
                    non_compliant += 1
                elif state.compliance_state == "Conflict":
                    conflicting += 1
                elif state.compliance_state == "Exempt":
                    exempt += 1

                compliance_details.append(
                    {
                        "resource_id": state.resource_id,
                        "resource_type": state.resource_type,
                        "compliance_state": state.compliance_state,
                        "timestamp": state.timestamp.isoformat() if state.timestamp else None,
                    }
                )

            # Calculate compliance percentage
            compliance_percentage = (
                (compliant / total_resources * 100) if total_resources > 0 else 0
            )

            # Determine overall compliance state
            if total_resources == 0:
                overall_state = PolicyComplianceState.UNKNOWN
            elif non_compliant > 0:
                overall_state = PolicyComplianceState.NON_COMPLIANT
            elif conflicting > 0:
                overall_state = PolicyComplianceState.CONFLICT
            else:
                overall_state = PolicyComplianceState.COMPLIANT

            logger.info(
                "policy_compliance_retrieved",
                subscription_id=subscription_id,
                policy_id=policy_id,
                total_resources=total_resources,
                compliance_percentage=compliance_percentage,
            )

            return PolicyComplianceResponse(
                policy_id=policy_id,
                policy_name=policy_id.split("/")[-1],
                compliance_state=overall_state,
                compliant_resources=compliant,
                non_compliant_resources=non_compliant,
                conflicting_resources=conflicting,
                exempt_resources=exempt,
                total_resources=total_resources,
                compliance_percentage=compliance_percentage,
                last_evaluated=datetime.utcnow(),
                details=compliance_details[:100],  # Limit details to first 100 resources
            ).dict()

        except AzureError as e:
            logger.error(
                "get_policy_compliance_failed",
                error=str(e),
                subscription_id=subscription_id,
                policy_id=policy_id,
            )
            raise Exception(f"Failed to get policy compliance: {str(e)}")

    async def get_policy_assignments(
        self, subscription_id: str, resource_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get policy assignments."""
        try:
            client = await self._get_policy_client(subscription_id)
            assignments = []

            # List assignments
            if resource_group:
                assignment_list = client.policy_assignments.list_for_resource_group(resource_group)
            else:
                assignment_list = client.policy_assignments.list()

            async for assignment in assignment_list:
                assignments.append(
                    {
                        "id": assignment.id,
                        "name": assignment.name,
                        "type": assignment.type,
                        "display_name": assignment.display_name,
                        "description": assignment.description,
                        "policy_definition_id": assignment.policy_definition_id,
                        "scope": assignment.scope,
                        "parameters": assignment.parameters,
                        "enforcement_mode": assignment.enforcement_mode,
                        "metadata": assignment.metadata,
                    }
                )

            return assignments

        except AzureError as e:
            logger.error(
                "get_policy_assignments_failed", error=str(e), subscription_id=subscription_id
            )
            raise Exception(f"Failed to get policy assignments: {str(e)}")
