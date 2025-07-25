"""
Azure RBAC management service for handling role-based access control operations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import structlog
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.core.exceptions import AzureError, ResourceNotFoundError

from shared.config import get_settings
from ..models import RBACResponse, RoleAssignmentResponse
from .azure_auth import AzureAuthService

settings = get_settings()
logger = structlog.get_logger(__name__)


class RBACManagementService:
    """Service for managing Azure RBAC."""
    
    def __init__(self):
        self.settings = settings
        self.auth_service = AzureAuthService()
        self.auth_clients = {}
    
    async def _get_auth_client(self, subscription_id: str) -> AuthorizationManagementClient:
        """Get or create Authorization client for subscription."""
        if subscription_id not in self.auth_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.auth_clients[subscription_id] = AuthorizationManagementClient(
                credential, subscription_id
            )
        return self.auth_clients[subscription_id]
    
    async def list_roles(
        self,
        subscription_id: str,
        scope: Optional[str] = None
    ) -> List[RBACResponse]:
        """List all role definitions."""
        try:
            client = await self._get_auth_client(subscription_id)
            roles = []
            
            # Default scope to subscription if not provided
            if not scope:
                scope = f"/subscriptions/{subscription_id}"
            
            # List role definitions
            role_list = client.role_definitions.list(scope)
            
            async for role in role_list:
                roles.append(RBACResponse(
                    id=role.id,
                    name=role.name,
                    type=role.type,
                    role_name=role.role_name,
                    description=role.description,
                    role_type=role.role_type,
                    permissions=[
                        {
                            "actions": perm.actions,
                            "not_actions": perm.not_actions,
                            "data_actions": perm.data_actions if hasattr(perm, 'data_actions') else [],
                            "not_data_actions": perm.not_data_actions if hasattr(perm, 'not_data_actions') else []
                        }
                        for perm in role.permissions
                    ],
                    assignable_scopes=role.assignable_scopes
                ))
            
            logger.info(
                "roles_listed",
                subscription_id=subscription_id,
                scope=scope,
                count=len(roles)
            )
            
            return roles
            
        except AzureError as e:
            logger.error(
                "list_roles_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list roles: {str(e)}")
    
    async def get_role_definition(
        self,
        subscription_id: str,
        role_id: str,
        scope: Optional[str] = None
    ) -> RBACResponse:
        """Get a specific role definition."""
        try:
            client = await self._get_auth_client(subscription_id)
            
            # Default scope to subscription if not provided
            if not scope:
                scope = f"/subscriptions/{subscription_id}"
            
            # Get role definition
            role = await client.role_definitions.get(scope, role_id)
            
            logger.info(
                "role_definition_retrieved",
                subscription_id=subscription_id,
                role_id=role_id
            )
            
            return RBACResponse(
                id=role.id,
                name=role.name,
                type=role.type,
                role_name=role.role_name,
                description=role.description,
                role_type=role.role_type,
                permissions=[
                    {
                        "actions": perm.actions,
                        "not_actions": perm.not_actions,
                        "data_actions": perm.data_actions if hasattr(perm, 'data_actions') else [],
                        "not_data_actions": perm.not_data_actions if hasattr(perm, 'not_data_actions') else []
                    }
                    for perm in role.permissions
                ],
                assignable_scopes=role.assignable_scopes
            )
            
        except ResourceNotFoundError:
            logger.error(
                "role_definition_not_found",
                subscription_id=subscription_id,
                role_id=role_id
            )
            raise Exception(f"Role definition {role_id} not found")
        except AzureError as e:
            logger.error(
                "get_role_definition_failed",
                error=str(e),
                subscription_id=subscription_id,
                role_id=role_id
            )
            raise Exception(f"Failed to get role definition: {str(e)}")
    
    async def list_role_assignments(
        self,
        subscription_id: str,
        principal_id: Optional[str] = None,
        scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List role assignments."""
        try:
            client = await self._get_auth_client(subscription_id)
            assignments = []
            
            # Default scope to subscription if not provided
            if not scope:
                scope = f"/subscriptions/{subscription_id}"
            
            # Build filter
            filter_str = None
            if principal_id:
                filter_str = f"principalId eq '{principal_id}'"
            
            # List role assignments
            assignment_list = client.role_assignments.list_for_scope(
                scope=scope,
                filter=filter_str
            )
            
            async for assignment in assignment_list:
                # Get role definition details
                role_def_id = assignment.role_definition_id.split('/')[-1]
                try:
                    role_def = await self.get_role_definition(
                        subscription_id, role_def_id, scope
                    )
                    role_name = role_def.role_name
                except Exception:
                    role_name = "Unknown"
                
                assignments.append({
                    "id": assignment.id,
                    "name": assignment.name,
                    "type": assignment.type,
                    "principal_id": assignment.principal_id,
                    "principal_type": assignment.principal_type,
                    "role_definition_id": assignment.role_definition_id,
                    "role_definition_name": role_name,
                    "scope": assignment.scope,
                    "created_on": assignment.created_on.isoformat() if assignment.created_on else None,
                    "updated_on": assignment.updated_on.isoformat() if assignment.updated_on else None,
                    "created_by": assignment.created_by
                })
            
            logger.info(
                "role_assignments_listed",
                subscription_id=subscription_id,
                scope=scope,
                principal_id=principal_id,
                count=len(assignments)
            )
            
            return assignments
            
        except AzureError as e:
            logger.error(
                "list_role_assignments_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list role assignments: {str(e)}")
    
    async def create_role_assignment(
        self,
        subscription_id: str,
        assignment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new role assignment."""
        try:
            client = await self._get_auth_client(subscription_id)
            
            # Generate assignment name
            assignment_name = str(uuid.uuid4())
            
            # Create role assignment parameters
            parameters = {
                "role_definition_id": assignment_data["role_definition_id"],
                "principal_id": assignment_data["principal_id"],
                "principal_type": assignment_data.get("principal_type", "ServicePrincipal"),
                "description": assignment_data.get("description")
            }
            
            # Create role assignment
            assignment = await client.role_assignments.create(
                scope=assignment_data["scope"],
                role_assignment_name=assignment_name,
                parameters=parameters
            )
            
            # Get role definition details
            role_def_id = assignment.role_definition_id.split('/')[-1]
            try:
                role_def = await self.get_role_definition(
                    subscription_id, role_def_id, assignment_data["scope"]
                )
                role_name = role_def.role_name
            except Exception:
                role_name = "Unknown"
            
            logger.info(
                "role_assignment_created",
                subscription_id=subscription_id,
                assignment_id=assignment.id,
                principal_id=assignment_data["principal_id"],
                role_name=role_name
            )
            
            return {
                "id": assignment.id,
                "name": assignment.name,
                "type": assignment.type,
                "principal_id": assignment.principal_id,
                "principal_type": assignment.principal_type,
                "role_definition_id": assignment.role_definition_id,
                "role_definition_name": role_name,
                "scope": assignment.scope,
                "created_on": assignment.created_on.isoformat() if assignment.created_on else datetime.utcnow().isoformat(),
                "created_by": assignment.created_by
            }
            
        except AzureError as e:
            logger.error(
                "create_role_assignment_failed",
                error=str(e),
                subscription_id=subscription_id,
                principal_id=assignment_data.get("principal_id")
            )
            raise Exception(f"Failed to create role assignment: {str(e)}")
    
    async def delete_role_assignment(
        self,
        subscription_id: str,
        assignment_id: str
    ) -> None:
        """Delete a role assignment."""
        try:
            client = await self._get_auth_client(subscription_id)
            
            # Extract assignment name from ID
            assignment_name = assignment_id.split('/')[-1]
            
            # Get assignment details first
            assignments = await self.list_role_assignments(subscription_id)
            assignment = next((a for a in assignments if a["id"] == assignment_id), None)
            
            if not assignment:
                raise Exception(f"Role assignment {assignment_id} not found")
            
            # Delete role assignment
            await client.role_assignments.delete(
                scope=assignment["scope"],
                role_assignment_name=assignment_name
            )
            
            logger.info(
                "role_assignment_deleted",
                subscription_id=subscription_id,
                assignment_id=assignment_id
            )
            
        except AzureError as e:
            logger.error(
                "delete_role_assignment_failed",
                error=str(e),
                subscription_id=subscription_id,
                assignment_id=assignment_id
            )
            raise Exception(f"Failed to delete role assignment: {str(e)}")
    
    async def check_access(
        self,
        subscription_id: str,
        principal_id: str,
        action: str,
        scope: str
    ) -> bool:
        """Check if a principal has access to perform an action."""
        try:
            # Get role assignments for principal
            assignments = await self.list_role_assignments(
                subscription_id=subscription_id,
                principal_id=principal_id,
                scope=scope
            )
            
            # Check each assignment
            for assignment in assignments:
                role_def_id = assignment["role_definition_id"].split('/')[-1]
                role_def = await self.get_role_definition(
                    subscription_id, role_def_id, scope
                )
                
                # Check permissions
                for perm in role_def.permissions:
                    # Check if action is allowed
                    for allowed_action in perm["actions"]:
                        if self._action_matches(action, allowed_action):
                            # Check if not explicitly denied
                            for denied_action in perm["not_actions"]:
                                if self._action_matches(action, denied_action):
                                    break
                            else:
                                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "check_access_failed",
                error=str(e),
                subscription_id=subscription_id,
                principal_id=principal_id,
                action=action
            )
            return False
    
    def _action_matches(self, action: str, pattern: str) -> bool:
        """Check if an action matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        if "*" in pattern:
            # Convert pattern to regex
            import re
            regex_pattern = pattern.replace("*", ".*")
            return re.match(regex_pattern, action) is not None
        
        return action == pattern
    
    async def get_rbac_recommendations(
        self,
        subscription_id: str
    ) -> List[Dict[str, Any]]:
        """Get RBAC best practice recommendations."""
        try:
            recommendations = []
            
            # Get all role assignments
            assignments = await self.list_role_assignments(subscription_id)
            
            # Check for overly broad permissions
            owner_assignments = [a for a in assignments if "Owner" in a.get("role_definition_name", "")]
            if len(owner_assignments) > 5:
                recommendations.append({
                    "type": "security",
                    "severity": "high",
                    "title": "Too many Owner role assignments",
                    "description": f"Found {len(owner_assignments)} Owner role assignments. Consider using more restrictive roles.",
                    "affected_resources": [a["principal_id"] for a in owner_assignments[:5]]
                })
            
            # Check for service principals with Owner access
            sp_owners = [a for a in owner_assignments if a.get("principal_type") == "ServicePrincipal"]
            if sp_owners:
                recommendations.append({
                    "type": "security",
                    "severity": "high",
                    "title": "Service principals with Owner access",
                    "description": f"Found {len(sp_owners)} service principals with Owner access. Use least privilege principle.",
                    "affected_resources": [a["principal_id"] for a in sp_owners]
                })
            
            # Check for stale assignments (older than 90 days without activity)
            # This would require activity log analysis in production
            
            logger.info(
                "rbac_recommendations_generated",
                subscription_id=subscription_id,
                recommendation_count=len(recommendations)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "get_rbac_recommendations_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            return []