"""
Intelligent Automation Engine for PolicyCortex.
Implements automated governance actions, remediation, and policy management.
"""

import asyncio
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import structlog

logger = structlog.get_logger(__name__)


class AutomationTrigger(Enum):
    """Types of automation triggers."""

    POLICY_VIOLATION = "policy_violation"
    SECURITY_INCIDENT = "security_incident"
    COST_THRESHOLD = "cost_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLIANCE_DRIFT = "compliance_drift"
    RESOURCE_ANOMALY = "resource_anomaly"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class AutomationStatus(Enum):
    """Status of automation execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class ActionPriority(Enum):
    """Priority levels for automation actions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AutomationAction:
    """Represents an automated action."""

    action_id: str
    name: str
    action_type: str
    target_resource: str
    parameters: Dict[str, Any]
    priority: ActionPriority
    timeout_seconds: int = 300
    retry_count: int = 3
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AutomationWorkflow:
    """Represents an automation workflow."""

    workflow_id: str
    name: str
    description: str
    trigger: AutomationTrigger
    conditions: List[Dict[str, Any]]
    actions: List[AutomationAction]
    approval_required: bool = False
    auto_rollback: bool = True
    max_execution_time: int = 1800  # 30 minutes
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"


@dataclass
class AutomationExecution:
    """Represents an automation execution instance."""

    execution_id: str
    workflow_id: str
    trigger_event: Dict[str, Any]
    status: AutomationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    actions_completed: int = 0
    actions_total: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rollback_executed: bool = False


class ActionExecutor:
    """
    Executes individual automation actions.
    """

    def __init__(self, azure_client=None):
        self.azure_client = azure_client
        self.action_handlers = {
            "azure_policy_assign": self._handle_policy_assignment,
            "azure_policy_remediate": self._handle_policy_remediation,
            "azure_rbac_update": self._handle_rbac_update,
            "azure_resource_tag": self._handle_resource_tagging,
            "azure_resource_stop": self._handle_resource_stop,
            "azure_resource_restart": self._handle_resource_restart,
            "azure_resource_scale": self._handle_resource_scaling,
            "azure_cost_alert": self._handle_cost_alert,
            "azure_security_rule": self._handle_security_rule,
            "notification_send": self._handle_notification,
            "approval_request": self._handle_approval_request,
            "custom_script": self._handle_custom_script,
        }

    async def execute_action(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single automation action."""
        try:
            logger.info(
                "executing_automation_action",
                action_id=action.action_id,
                action_type=action.action_type,
                target=action.target_resource,
            )

            # Validate action
            validation_result = await self._validate_action(action, context)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Action validation failed: {validation_result['reason']}",
                    "action_id": action.action_id,
                }

            # Get action handler
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                return {
                    "success": False,
                    "error": f"No handler found for action type: {action.action_type}",
                    "action_id": action.action_id,
                }

            # Execute action with timeout and retries
            result = await self._execute_with_retry(handler, action, context)

            logger.info(
                "automation_action_completed", action_id=action.action_id, success=result["success"]
            )

            return result

        except Exception as e:
            logger.error("automation_action_failed", action_id=action.action_id, error=str(e))
            return {"success": False, "error": str(e), "action_id": action.action_id}

    async def _validate_action(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate action before execution."""
        try:
            # Check basic validation rules
            for rule in action.validation_rules:
                rule_type = rule.get("type")

                if rule_type == "resource_exists":
                    # Check if target resource exists
                    resource_id = action.target_resource
                    if not await self._check_resource_exists(resource_id):
                        return {
                            "valid": False,
                            "reason": f"Target resource does not exist: {resource_id}",
                        }

                elif rule_type == "permission_check":
                    # Check if user has required permissions
                    required_permissions = rule.get("permissions", [])
                    user_permissions = context.get("user_permissions", [])

                    if not all(perm in user_permissions for perm in required_permissions):
                        return {"valid": False, "reason": "Insufficient permissions for action"}

                elif rule_type == "cost_limit":
                    # Check if action exceeds cost limits
                    estimated_cost = rule.get("estimated_cost", 0)
                    max_cost = rule.get("max_cost", 1000)

                    if estimated_cost > max_cost:
                        return {
                            "valid": False,
                            "reason": f"Estimated cost ${estimated_cost} exceeds limit ${max_cost}",
                        }

                elif rule_type == "time_window":
                    # Check if action is within allowed time window
                    allowed_hours = rule.get("allowed_hours", [])
                    current_hour = datetime.utcnow().hour

                    if allowed_hours and current_hour not in allowed_hours:
                        return {
                            "valid": False,
                            "reason": f"Action not allowed during current time: {current_hour}:00",
                        }

            return {"valid": True, "reason": "All validations passed"}

        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    async def _execute_with_retry(
        self, handler: Callable, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action with retry logic."""
        last_error = None

        for attempt in range(action.retry_count + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(action, context), timeout=action.timeout_seconds
                )

                if result.get("success", False):
                    result["attempts"] = attempt + 1
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    if attempt < action.retry_count:
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            except asyncio.TimeoutError:
                last_error = f"Action timed out after {action.timeout_seconds} seconds"
                logger.warning("action_timeout", action_id=action.action_id, attempt=attempt + 1)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "action_attempt_failed",
                    action_id=action.action_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

        return {
            "success": False,
            "error": f"Action failed after {action.retry_count + 1} attempts. Last error: {last_error}",
            "action_id": action.action_id,
            "attempts": action.retry_count + 1,
        }

    async def _check_resource_exists(self, resource_id: str) -> bool:
        """Check if Azure resource exists."""
        try:
            # Mock implementation - in production, this would call Azure APIs
            if self.azure_client:
                # return await self.azure_client.resource_exists(resource_id)
                pass

            # For development, assume resource exists if ID follows pattern
            return bool(re.match(r"^/subscriptions/.+/resourceGroups/.+", resource_id))

        except Exception:
            return False

    # Action Handlers
    async def _handle_policy_assignment(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Azure policy assignment."""
        try:
            parameters = action.parameters
            policy_definition_id = parameters.get("policy_definition_id")
            scope = parameters.get("scope", action.target_resource)
            assignment_name = parameters.get("name", f"auto-assignment-{uuid.uuid4().hex[:8]}")

            logger.info(
                "assigning_azure_policy",
                policy_id=policy_definition_id,
                scope=scope,
                assignment_name=assignment_name,
            )

            # Mock implementation - in production, call Azure Policy API
            result = {
                "assignment_id": f"/subscriptions/sub123/providers/Microsoft.Authorization/policyAssignments/{assignment_name}",
                "policy_definition_id": policy_definition_id,
                "scope": scope,
                "status": "assigned",
            }

            return {
                "success": True,
                "result": result,
                "message": f"Policy assigned successfully: {assignment_name}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_policy_remediation(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle policy violation remediation."""
        try:
            parameters = action.parameters
            policy_assignment_id = parameters.get("policy_assignment_id")
            resource_ids = parameters.get("resource_ids", [action.target_resource])

            logger.info(
                "remediating_policy_violations",
                policy_assignment=policy_assignment_id,
                resources=len(resource_ids),
            )

            # Mock remediation results
            remediated_resources = []
            failed_resources = []

            for resource_id in resource_ids:
                # Simulate remediation success/failure
                if resource_id.endswith("test"):
                    failed_resources.append(
                        {"resource_id": resource_id, "error": "Test resource - remediation skipped"}
                    )
                else:
                    remediated_resources.append(
                        {"resource_id": resource_id, "status": "remediated"}
                    )

            return {
                "success": len(failed_resources) == 0,
                "result": {
                    "remediated_count": len(remediated_resources),
                    "failed_count": len(failed_resources),
                    "remediated_resources": remediated_resources,
                    "failed_resources": failed_resources,
                },
                "message": f"Remediated {len(remediated_resources)} resources, "
                f"{len(failed_resources)} failed",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_rbac_update(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle RBAC role assignment updates."""
        try:
            parameters = action.parameters
            principal_id = parameters.get("principal_id")
            role_definition_id = parameters.get("role_definition_id")
            scope = parameters.get("scope", action.target_resource)
            operation = parameters.get("operation", "assign")  # assign or remove

            logger.info(
                "updating_rbac_assignment",
                principal_id=principal_id,
                role=role_definition_id,
                scope=scope,
                operation=operation,
            )

            # Mock RBAC operation
            assignment_id = f"/subscriptions/sub123/providers/Microsoft.Authorization/roleAssignments/{uuid.uuid4()}"

            result = {
                "assignment_id": assignment_id,
                "principal_id": principal_id,
                "role_definition_id": role_definition_id,
                "scope": scope,
                "operation": operation,
                "status": "completed",
            }

            return {
                "success": True,
                "result": result,
                "message": f"RBAC {operation} completed successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_resource_tagging(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource tagging operations."""
        try:
            parameters = action.parameters
            tags = parameters.get("tags", {})
            operation = parameters.get("operation", "merge")  # merge, replace, or remove

            logger.info(
                "updating_resource_tags",
                resource=action.target_resource,
                tags=tags,
                operation=operation,
            )

            # Mock tagging operation
            current_tags = {"Environment": "Development", "Owner": "TeamA"}

            if operation == "merge":
                updated_tags = {**current_tags, **tags}
            elif operation == "replace":
                updated_tags = tags
            elif operation == "remove":
                updated_tags = {k: v for k, v in current_tags.items() if k not in tags}
            else:
                updated_tags = current_tags

            return {
                "success": True,
                "result": {
                    "resource_id": action.target_resource,
                    "previous_tags": current_tags,
                    "updated_tags": updated_tags,
                    "operation": operation,
                },
                "message": f"Resource tags {operation}d successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_resource_stop(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource stop operations."""
        try:
            parameters = action.parameters
            force = parameters.get("force", False)

            logger.info("stopping_resource", resource=action.target_resource, force=force)

            # Mock resource stop
            return {
                "success": True,
                "result": {
                    "resource_id": action.target_resource,
                    "previous_state": "running",
                    "new_state": "stopped",
                    "force": force,
                },
                "message": "Resource stopped successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_resource_restart(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource restart operations."""
        try:
            logger.info("restarting_resource", resource=action.target_resource)

            # Mock resource restart
            await asyncio.sleep(1)  # Simulate restart time

            return {
                "success": True,
                "result": {
                    "resource_id": action.target_resource,
                    "operation": "restart",
                    "status": "completed",
                },
                "message": "Resource restarted successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_resource_scaling(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource scaling operations."""
        try:
            parameters = action.parameters
            scale_type = parameters.get("scale_type", "manual")  # manual, auto
            target_capacity = parameters.get("target_capacity")
            min_capacity = parameters.get("min_capacity")
            max_capacity = parameters.get("max_capacity")

            logger.info(
                "scaling_resource",
                resource=action.target_resource,
                scale_type=scale_type,
                target_capacity=target_capacity,
            )

            # Mock scaling operation
            return {
                "success": True,
                "result": {
                    "resource_id": action.target_resource,
                    "previous_capacity": 2,
                    "new_capacity": target_capacity,
                    "scale_type": scale_type,
                    "min_capacity": min_capacity,
                    "max_capacity": max_capacity,
                },
                "message": f"Resource scaled to {target_capacity} instances",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_cost_alert(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle cost alert creation/updates."""
        try:
            parameters = action.parameters
            threshold = parameters.get("threshold")
            alert_type = parameters.get("alert_type", "budget")

            logger.info(
                "creating_cost_alert",
                resource=action.target_resource,
                threshold=threshold,
                alert_type=alert_type,
            )

            return {
                "success": True,
                "result": {
                    "alert_id": f"cost-alert-{uuid.uuid4().hex[:8]}",
                    "scope": action.target_resource,
                    "threshold": threshold,
                    "alert_type": alert_type,
                    "status": "active",
                },
                "message": f"Cost alert created with threshold ${threshold}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_security_rule(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle security rule updates."""
        try:
            parameters = action.parameters
            rule_type = parameters.get("rule_type", "nsg")
            rule_action = parameters.get("action", "allow")  # allow, deny
            priority = parameters.get("priority", 1000)

            logger.info(
                "updating_security_rule",
                resource=action.target_resource,
                rule_type=rule_type,
                action=rule_action,
            )

            return {
                "success": True,
                "result": {
                    "rule_id": f"security-rule-{uuid.uuid4().hex[:8]}",
                    "resource_id": action.target_resource,
                    "rule_type": rule_type,
                    "action": rule_action,
                    "priority": priority,
                    "status": "applied",
                },
                "message": f"Security rule applied: {rule_action} with priority {priority}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_notification(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle notification sending."""
        try:
            parameters = action.parameters
            notification_type = parameters.get("type", "email")
            recipients = parameters.get("recipients", [])
            message = parameters.get("message", "Automation action completed")

            logger.info("sending_notification", type=notification_type, recipients=len(recipients))

            return {
                "success": True,
                "result": {
                    "notification_id": f"notification-{uuid.uuid4().hex[:8]}",
                    "type": notification_type,
                    "recipients": recipients,
                    "status": "sent",
                },
                "message": f"Notification sent to {len(recipients)} recipients",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_approval_request(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle approval request creation."""
        try:
            parameters = action.parameters
            approvers = parameters.get("approvers", [])
            timeout_hours = parameters.get("timeout_hours", 24)

            logger.info(
                "creating_approval_request", approvers=len(approvers), timeout_hours=timeout_hours
            )

            return {
                "success": True,
                "result": {
                    "approval_id": f"approval-{uuid.uuid4().hex[:8]}",
                    "approvers": approvers,
                    "timeout_hours": timeout_hours,
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat(),
                },
                "message": f"Approval request created for {len(approvers)} approvers",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_custom_script(
        self, action: AutomationAction, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle custom script execution."""
        try:
            parameters = action.parameters
            script_type = parameters.get("script_type", "powershell")
            script_content = parameters.get("script_content", "")

            logger.info(
                "executing_custom_script", script_type=script_type, resource=action.target_resource
            )

            # Mock script execution
            return {
                "success": True,
                "result": {
                    "script_id": f"script-{uuid.uuid4().hex[:8]}",
                    "script_type": script_type,
                    "exit_code": 0,
                    "output": "Script executed successfully",
                    "execution_time": 2.5,
                },
                "message": f"Custom {script_type} script executed successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
