"""
Intelligent Automation Orchestrator for PolicyCortex.
Orchestrates complex automation workflows for governance scenarios.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import structlog
from .automation_engine import (
    AutomationAction, AutomationWorkflow, AutomationExecution, 
    AutomationTrigger, AutomationStatus, ActionPriority, ActionExecutor
)

logger = structlog.get_logger(__name__)


class WorkflowEngine:
    """
    Orchestrates automation workflows with dependency management and rollback capabilities.
    """
    
    def __init__(self, azure_client=None):
        self.azure_client = azure_client
        self.action_executor = ActionExecutor(azure_client)
        self.active_executions: Dict[str, AutomationExecution] = {}
        self.workflow_registry: Dict[str, AutomationWorkflow] = {}
        self.execution_history: List[AutomationExecution] = []
        self.max_concurrent_executions = 10
        self.executor_pool = None
        
        # Pre-built governance workflows
        self._initialize_predefined_workflows()
    
    def _initialize_predefined_workflows(self):
        """Initialize common governance automation workflows."""
        try:
            # Compliance Violation Response Workflow
            compliance_workflow = AutomationWorkflow(
                workflow_id="compliance_violation_response",
                name="Compliance Violation Response",
                description="Automated response to policy compliance violations",
                trigger=AutomationTrigger.POLICY_VIOLATION,
                conditions=[
                    {"type": "severity", "operator": ">=", "value": "medium"},
                    {"type": "resource_type", "operator": "in", "value": ["Microsoft.Compute/virtualMachines", "Microsoft.Storage/storageAccounts"]}
                ],
                actions=[
                    AutomationAction(
                        action_id="tag_non_compliant_resource",
                        name="Tag Non-Compliant Resource",
                        action_type="azure_resource_tag",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "tags": {
                                "ComplianceStatus": "Non-Compliant",
                                "AutoRemediation": "Required",
                                "LastViolation": "{{event.timestamp}}"
                            },
                            "operation": "merge"
                        },
                        priority=ActionPriority.HIGH,
                        validation_rules=[
                            {"type": "resource_exists"},
                            {"type": "permission_check", "permissions": ["Microsoft.Resources/tags/write"]}
                        ]
                    ),
                    AutomationAction(
                        action_id="notify_compliance_team",
                        name="Notify Compliance Team",
                        action_type="notification_send",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "type": "email",
                            "recipients": ["compliance-team@company.com"],
                            "message": "Compliance violation detected on resource: {{event.resource_id}}",
                            "priority": "high"
                        },
                        priority=ActionPriority.MEDIUM,
                        dependencies=["tag_non_compliant_resource"]
                    ),
                    AutomationAction(
                        action_id="create_remediation_task",
                        name="Create Remediation Task",
                        action_type="azure_policy_remediate",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "policy_assignment_id": "{{event.policy_assignment_id}}",
                            "resource_ids": ["{{event.resource_id}}"]
                        },
                        priority=ActionPriority.HIGH,
                        dependencies=["tag_non_compliant_resource"],
                        rollback_actions=[
                            {"action_type": "azure_resource_tag", "parameters": {"tags": {"ComplianceStatus": "Pending"}, "operation": "merge"}}
                        ]
                    )
                ],
                approval_required=False,
                auto_rollback=True
            )
            
            # Security Incident Response Workflow
            security_workflow = AutomationWorkflow(
                workflow_id="security_incident_response",
                name="Security Incident Response",
                description="Automated response to security incidents",
                trigger=AutomationTrigger.SECURITY_INCIDENT,
                conditions=[
                    {"type": "severity", "operator": ">=", "value": "high"},
                    {"type": "incident_type", "operator": "in", "value": ["unauthorized_access", "privilege_escalation", "data_exfiltration"]}
                ],
                actions=[
                    AutomationAction(
                        action_id="isolate_affected_resource",
                        name="Isolate Affected Resource",
                        action_type="azure_security_rule",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "rule_type": "nsg",
                            "action": "deny",
                            "priority": 100,
                            "source": "*",
                            "destination": "{{event.resource_id}}"
                        },
                        priority=ActionPriority.CRITICAL,
                        timeout_seconds=60
                    ),
                    AutomationAction(
                        action_id="notify_security_team",
                        name="Notify Security Team",
                        action_type="notification_send",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "type": "slack",
                            "recipients": ["#security-alerts"],
                            "message": "SECURITY ALERT: {{event.incident_type}} detected on {{event.resource_id}}",
                            "priority": "critical"
                        },
                        priority=ActionPriority.CRITICAL
                    ),
                    AutomationAction(
                        action_id="create_security_log",
                        name="Create Security Log Entry",
                        action_type="custom_script",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "script_type": "powershell",
                            "script_content": "Write-EventLog -LogName Security -Source 'PolicyCortex' -EventId 1001 -Message 'Automated security response executed for incident: {{event.incident_id}}'"
                        },
                        priority=ActionPriority.MEDIUM,
                        dependencies=["isolate_affected_resource"]
                    )
                ],
                approval_required=False,
                auto_rollback=False  # Security actions should not auto-rollback
            )
            
            # Cost Optimization Workflow
            cost_optimization_workflow = AutomationWorkflow(
                workflow_id="cost_optimization_response",
                name="Cost Optimization Response",
                description="Automated cost optimization actions",
                trigger=AutomationTrigger.COST_THRESHOLD,
                conditions=[
                    {"type": "cost_variance", "operator": ">", "value": 20},  # 20% over budget
                    {"type": "resource_utilization", "operator": "<", "value": 50}  # Low utilization
                ],
                actions=[
                    AutomationAction(
                        action_id="analyze_resource_utilization",
                        name="Analyze Resource Utilization",
                        action_type="custom_script",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "script_type": "powershell",
                            "script_content": "Get-AzMetric -ResourceId {{event.resource_id}} -MetricName 'Percentage CPU' -TimeGrain 01:00:00"
                        },
                        priority=ActionPriority.MEDIUM
                    ),
                    AutomationAction(
                        action_id="right_size_resource",
                        name="Right-Size Resource",
                        action_type="azure_resource_scale",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "scale_type": "manual",
                            "target_capacity": "{{calculated_optimal_size}}",
                            "min_capacity": 1,
                            "max_capacity": 10
                        },
                        priority=ActionPriority.HIGH,
                        dependencies=["analyze_resource_utilization"],
                        approval_required=True
                    ),
                    AutomationAction(
                        action_id="setup_cost_alert",
                        name="Setup Cost Alert",
                        action_type="azure_cost_alert",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "threshold": "{{event.budget_threshold}}",
                            "alert_type": "budget"
                        },
                        priority=ActionPriority.MEDIUM,
                        dependencies=["right_size_resource"]
                    )
                ],
                approval_required=True,
                auto_rollback=True
            )
            
            # Performance Degradation Workflow
            performance_workflow = AutomationWorkflow(
                workflow_id="performance_degradation_response",
                name="Performance Degradation Response",
                description="Automated response to performance issues",
                trigger=AutomationTrigger.PERFORMANCE_DEGRADATION,
                conditions=[
                    {"type": "response_time", "operator": ">", "value": 5000},  # 5 seconds
                    {"type": "error_rate", "operator": ">", "value": 5}  # 5% error rate
                ],
                actions=[
                    AutomationAction(
                        action_id="restart_unhealthy_instances",
                        name="Restart Unhealthy Instances",
                        action_type="azure_resource_restart",
                        target_resource="{{event.resource_id}}",
                        parameters={},
                        priority=ActionPriority.HIGH,
                        timeout_seconds=300
                    ),
                    AutomationAction(
                        action_id="scale_out_resources",
                        name="Scale Out Resources",
                        action_type="azure_resource_scale",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "scale_type": "auto",
                            "target_capacity": "{{current_capacity + 2}}",
                            "min_capacity": 2,
                            "max_capacity": 20
                        },
                        priority=ActionPriority.HIGH,
                        dependencies=["restart_unhealthy_instances"]
                    ),
                    AutomationAction(
                        action_id="notify_operations_team",
                        name="Notify Operations Team",
                        action_type="notification_send",
                        target_resource="{{event.resource_id}}",
                        parameters={
                            "type": "teams",
                            "recipients": ["operations-team"],
                            "message": "Performance degradation detected and auto-scaling initiated for {{event.resource_id}}"
                        },
                        priority=ActionPriority.MEDIUM
                    )
                ],
                approval_required=False,
                auto_rollback=True,
                max_execution_time=900  # 15 minutes
            )
            
            # Register workflows
            self.workflow_registry = {
                compliance_workflow.workflow_id: compliance_workflow,
                security_workflow.workflow_id: security_workflow,
                cost_optimization_workflow.workflow_id: cost_optimization_workflow,
                performance_workflow.workflow_id: performance_workflow
            }
            
            logger.info("predefined_workflows_initialized", 
                       workflow_count=len(self.workflow_registry))
        
        except Exception as e:
            logger.error("workflow_initialization_failed", error=str(e))
    
    async def initialize(self):
        """Initialize the workflow engine."""
        try:
            logger.info("initializing_workflow_engine")
            
            # Initialize thread pool for concurrent execution
            self.executor_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_executions)
            
            logger.info("workflow_engine_initialized")
        
        except Exception as e:
            logger.error("workflow_engine_initialization_failed", error=str(e))
            raise
    
    async def trigger_workflow(self, trigger_event: Dict[str, Any], 
                              workflow_id: Optional[str] = None) -> Optional[str]:
        """Trigger a workflow based on an event."""
        try:
            trigger_type = AutomationTrigger(trigger_event.get('trigger_type', 'manual'))
            
            logger.info("triggering_workflow",
                       trigger_type=trigger_type.value,
                       workflow_id=workflow_id,
                       event_id=trigger_event.get('event_id'))
            
            # Find appropriate workflow
            if workflow_id:
                workflow = self.workflow_registry.get(workflow_id)
                if not workflow:
                    logger.error("workflow_not_found", workflow_id=workflow_id)
                    return None
            else:
                workflow = await self._find_matching_workflow(trigger_event, trigger_type)
                if not workflow:
                    logger.warning("no_matching_workflow_found", trigger_type=trigger_type.value)
                    return None
            
            # Check if conditions are met
            if not await self._evaluate_conditions(workflow.conditions, trigger_event):
                logger.info("workflow_conditions_not_met", workflow_id=workflow.workflow_id)
                return None
            
            # Check concurrent execution limits
            if len(self.active_executions) >= self.max_concurrent_executions:
                logger.warning("max_concurrent_executions_reached")
                return None
            
            # Create execution instance
            execution = AutomationExecution(
                execution_id=str(uuid.uuid4()),
                workflow_id=workflow.workflow_id,
                trigger_event=trigger_event,
                status=AutomationStatus.PENDING,
                started_at=datetime.utcnow(),
                actions_total=len(workflow.actions)
            )
            
            # Start execution
            self.active_executions[execution.execution_id] = execution
            
            # Execute workflow asynchronously
            asyncio.create_task(self._execute_workflow(workflow, execution))
            
            logger.info("workflow_triggered",
                       execution_id=execution.execution_id,
                       workflow_id=workflow.workflow_id)
            
            return execution.execution_id
        
        except Exception as e:
            logger.error("workflow_trigger_failed", error=str(e))
            return None
    
    async def _find_matching_workflow(self, trigger_event: Dict[str, Any], 
                                     trigger_type: AutomationTrigger) -> Optional[AutomationWorkflow]:
        """Find workflow that matches the trigger event."""
        try:
            for workflow in self.workflow_registry.values():
                if workflow.trigger == trigger_type:
                    return workflow
            
            return None
        
        except Exception as e:
            logger.error("workflow_matching_failed", error=str(e))
            return None
    
    async def _evaluate_conditions(self, conditions: List[Dict[str, Any]], 
                                  trigger_event: Dict[str, Any]) -> bool:
        """Evaluate if workflow conditions are met."""
        try:
            for condition in conditions:
                condition_type = condition.get('type')
                operator = condition.get('operator')
                expected_value = condition.get('value')
                
                actual_value = trigger_event.get(condition_type)
                
                if not self._evaluate_condition(actual_value, operator, expected_value):
                    return False
            
            return True
        
        except Exception as e:
            logger.error("condition_evaluation_failed", error=str(e))
            return False
    
    def _evaluate_condition(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        """Evaluate a single condition."""
        try:
            if operator == "==":
                return actual_value == expected_value
            elif operator == "!=":
                return actual_value != expected_value
            elif operator == ">":
                return float(actual_value) > float(expected_value)
            elif operator == ">=":
                return float(actual_value) >= float(expected_value)
            elif operator == "<":
                return float(actual_value) < float(expected_value)
            elif operator == "<=":
                return float(actual_value) <= float(expected_value)
            elif operator == "in":
                return actual_value in expected_value
            elif operator == "not_in":
                return actual_value not in expected_value
            elif operator == "contains":
                return expected_value in str(actual_value)
            else:
                return False
        
        except Exception:
            return False
    
    async def _execute_workflow(self, workflow: AutomationWorkflow, execution: AutomationExecution):
        """Execute a workflow with all its actions."""
        try:
            execution.status = AutomationStatus.RUNNING
            logger.info("executing_workflow",
                       execution_id=execution.execution_id,
                       workflow_id=workflow.workflow_id)
            
            # Build action dependency graph
            action_graph = self._build_action_graph(workflow.actions)
            
            # Execute actions respecting dependencies
            executed_actions = set()
            failed_actions = set()
            
            while len(executed_actions) < len(workflow.actions) and not failed_actions:
                # Check for timeout
                if (datetime.utcnow() - execution.started_at).total_seconds() > workflow.max_execution_time:
                    execution.status = AutomationStatus.FAILED
                    execution.errors.append("Workflow execution timeout")
                    break
                
                # Find actions ready to execute
                ready_actions = self._get_ready_actions(
                    workflow.actions, action_graph, executed_actions, failed_actions
                )
                
                if not ready_actions:
                    # No more actions can be executed
                    break
                
                # Execute ready actions in parallel (by priority)
                ready_actions.sort(key=lambda a: self._get_priority_value(a.priority), reverse=True)
                
                tasks = []
                for action in ready_actions:
                    if workflow.approval_required and action.approval_required:
                        # Handle approval workflow
                        approval_result = await self._handle_approval(action, execution)
                        if not approval_result:
                            failed_actions.add(action.action_id)
                            continue
                    
                    # Create execution context
                    context = self._create_execution_context(execution, workflow)
                    
                    # Execute action
                    task = asyncio.create_task(
                        self._execute_action_with_tracking(action, context, execution)
                    )
                    tasks.append((action, task))
                
                # Wait for all actions to complete
                for action, task in tasks:
                    try:
                        result = await task
                        if result['success']:
                            executed_actions.add(action.action_id)
                            execution.actions_completed += 1
                        else:
                            failed_actions.add(action.action_id)
                            execution.errors.append(f"Action {action.action_id} failed: {result.get('error')}")
                    
                    except Exception as e:
                        failed_actions.add(action.action_id)
                        execution.errors.append(f"Action {action.action_id} failed: {str(e)}")
            
            # Determine final status
            if failed_actions:
                if executed_actions:
                    execution.status = AutomationStatus.PARTIALLY_COMPLETED
                else:
                    execution.status = AutomationStatus.FAILED
                
                # Execute rollback if enabled
                if workflow.auto_rollback:
                    await self._execute_rollback(workflow, execution, executed_actions)
            else:
                execution.status = AutomationStatus.COMPLETED
            
            execution.completed_at = datetime.utcnow()
            
            # Move to history and cleanup
            self.execution_history.append(execution)
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            
            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            logger.info("workflow_execution_completed",
                       execution_id=execution.execution_id,
                       status=execution.status.value,
                       actions_completed=execution.actions_completed,
                       actions_total=execution.actions_total)
        
        except Exception as e:
            execution.status = AutomationStatus.FAILED
            execution.errors.append(f"Workflow execution failed: {str(e)}")
            execution.completed_at = datetime.utcnow()
            
            logger.error("workflow_execution_failed",
                        execution_id=execution.execution_id,
                        error=str(e))
    
    def _build_action_graph(self, actions: List[AutomationAction]) -> Dict[str, List[str]]:
        """Build action dependency graph."""
        graph = {}
        for action in actions:
            graph[action.action_id] = action.dependencies.copy()
        return graph
    
    def _get_ready_actions(self, actions: List[AutomationAction], 
                          action_graph: Dict[str, List[str]],
                          executed_actions: Set[str], 
                          failed_actions: Set[str]) -> List[AutomationAction]:
        """Get actions that are ready to execute."""
        ready = []
        
        for action in actions:
            if action.action_id in executed_actions or action.action_id in failed_actions:
                continue
            
            # Check if all dependencies are satisfied
            dependencies = action_graph.get(action.action_id, [])
            if all(dep in executed_actions for dep in dependencies):
                ready.append(action)
        
        return ready
    
    def _get_priority_value(self, priority: ActionPriority) -> int:
        """Convert priority to numeric value for sorting."""
        priority_map = {
            ActionPriority.CRITICAL: 4,
            ActionPriority.HIGH: 3,
            ActionPriority.MEDIUM: 2,
            ActionPriority.LOW: 1
        }
        return priority_map.get(priority, 1)
    
    async def _execute_action_with_tracking(self, action: AutomationAction, 
                                          context: Dict[str, Any],
                                          execution: AutomationExecution) -> Dict[str, Any]:
        """Execute action and track results."""
        try:
            # Substitute template variables in action
            resolved_action = self._resolve_action_templates(action, context)
            
            # Execute action
            result = await self.action_executor.execute_action(resolved_action, context)
            
            # Track result
            execution.results.append({
                'action_id': action.action_id,
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return result
        
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'action_id': action.action_id
            }
            
            execution.results.append({
                'action_id': action.action_id,
                'result': error_result,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return error_result
    
    def _create_execution_context(self, execution: AutomationExecution, 
                                 workflow: AutomationWorkflow) -> Dict[str, Any]:
        """Create execution context for actions."""
        return {
            'execution_id': execution.execution_id,
            'workflow_id': workflow.workflow_id,
            'trigger_event': execution.trigger_event,
            'started_at': execution.started_at,
            'user_permissions': ['*'],  # For development - in production, get from auth
            'azure_client': self.azure_client
        }
    
    def _resolve_action_templates(self, action: AutomationAction, 
                                 context: Dict[str, Any]) -> AutomationAction:
        """Resolve template variables in action parameters."""
        try:
            # Simple template resolution - in production, use a proper template engine
            resolved_action = AutomationAction(
                action_id=action.action_id,
                name=action.name,
                action_type=action.action_type,
                target_resource=self._resolve_template_string(action.target_resource, context),
                parameters=self._resolve_template_dict(action.parameters, context),
                priority=action.priority,
                timeout_seconds=action.timeout_seconds,
                retry_count=action.retry_count,
                rollback_actions=action.rollback_actions,
                validation_rules=action.validation_rules,
                dependencies=action.dependencies
            )
            
            return resolved_action
        
        except Exception as e:
            logger.error("template_resolution_failed", 
                        action_id=action.action_id,
                        error=str(e))
            return action
    
    def _resolve_template_string(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve template variables in a string."""
        try:
            # Simple template resolution
            result = template
            
            # Replace event variables
            event = context.get('trigger_event', {})
            for key, value in event.items():
                result = result.replace(f"{{{{event.{key}}}}}", str(value))
            
            # Replace context variables
            for key, value in context.items():
                if key != 'trigger_event':
                    result = result.replace(f"{{{{{key}}}}}", str(value))
            
            return result
        
        except Exception:
            return template
    
    def _resolve_template_dict(self, template_dict: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables in a dictionary."""
        try:
            result = {}
            
            for key, value in template_dict.items():
                if isinstance(value, str):
                    result[key] = self._resolve_template_string(value, context)
                elif isinstance(value, dict):
                    result[key] = self._resolve_template_dict(value, context)
                elif isinstance(value, list):
                    result[key] = [
                        self._resolve_template_string(item, context) if isinstance(item, str) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            
            return result
        
        except Exception:
            return template_dict
    
    async def _handle_approval(self, action: AutomationAction, 
                              execution: AutomationExecution) -> bool:
        """Handle approval workflow for actions."""
        try:
            logger.info("approval_required",
                       action_id=action.action_id,
                       execution_id=execution.execution_id)
            
            # For development, auto-approve non-critical actions
            # In production, this would integrate with approval systems
            if action.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]:
                return False  # Require manual approval
            else:
                return True  # Auto-approve medium/low priority
        
        except Exception as e:
            logger.error("approval_handling_failed", error=str(e))
            return False
    
    async def _execute_rollback(self, workflow: AutomationWorkflow, 
                               execution: AutomationExecution,
                               executed_actions: Set[str]):
        """Execute rollback actions for failed workflow."""
        try:
            logger.info("executing_rollback",
                       execution_id=execution.execution_id,
                       executed_actions=len(executed_actions))
            
            rollback_tasks = []
            
            # Execute rollback actions for completed actions (in reverse order)
            for action in reversed(workflow.actions):
                if action.action_id in executed_actions and action.rollback_actions:
                    for rollback_action_data in action.rollback_actions:
                        rollback_action = AutomationAction(
                            action_id=f"rollback_{action.action_id}_{uuid.uuid4().hex[:8]}",
                            name=f"Rollback {action.name}",
                            action_type=rollback_action_data['action_type'],
                            target_resource=action.target_resource,
                            parameters=rollback_action_data.get('parameters', {}),
                            priority=ActionPriority.HIGH,
                            timeout_seconds=120
                        )
                        
                        context = self._create_execution_context(execution, workflow)
                        task = asyncio.create_task(
                            self.action_executor.execute_action(rollback_action, context)
                        )
                        rollback_tasks.append(task)
            
            # Wait for all rollback actions
            if rollback_tasks:
                rollback_results = await asyncio.gather(*rollback_tasks, return_exceptions=True)
                execution.rollback_executed = True
                
                successful_rollbacks = sum(1 for result in rollback_results 
                                         if isinstance(result, dict) and result.get('success'))
                
                logger.info("rollback_completed",
                           execution_id=execution.execution_id,
                           successful_rollbacks=successful_rollbacks,
                           total_rollbacks=len(rollback_tasks))
        
        except Exception as e:
            logger.error("rollback_execution_failed",
                        execution_id=execution.execution_id,
                        error=str(e))
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution."""
        try:
            # Check active executions
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                return self._execution_to_dict(execution)
            
            # Check history
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return self._execution_to_dict(execution)
            
            return None
        
        except Exception as e:
            logger.error("get_execution_status_failed", error=str(e))
            return None
    
    def _execution_to_dict(self, execution: AutomationExecution) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            'execution_id': execution.execution_id,
            'workflow_id': execution.workflow_id,
            'status': execution.status.value,
            'started_at': execution.started_at.isoformat(),
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'actions_completed': execution.actions_completed,
            'actions_total': execution.actions_total,
            'progress_percentage': (execution.actions_completed / execution.actions_total * 100) if execution.actions_total > 0 else 0,
            'errors': execution.errors,
            'rollback_executed': execution.rollback_executed,
            'results': execution.results
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = AutomationStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                
                logger.info("execution_cancelled", execution_id=execution_id)
                return True
            
            return False
        
        except Exception as e:
            logger.error("cancel_execution_failed", error=str(e))
            return False
    
    async def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get analytics about workflow executions."""
        try:
            total_executions = len(self.execution_history) + len(self.active_executions)
            
            status_counts = {}
            for execution in self.execution_history:
                status = execution.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_executions': total_executions,
                'active_executions': len(self.active_executions),
                'completed_executions': len(self.execution_history),
                'status_distribution': status_counts,
                'available_workflows': len(self.workflow_registry),
                'workflow_types': list(self.workflow_registry.keys())
            }
        
        except Exception as e:
            logger.error("get_workflow_analytics_failed", error=str(e))
            return {}
    
    async def cleanup(self):
        """Cleanup workflow engine resources."""
        try:
            logger.info("cleaning_up_workflow_engine")
            
            # Cancel active executions
            for execution_id in list(self.active_executions.keys()):
                await self.cancel_execution(execution_id)
            
            # Shutdown thread pool
            if self.executor_pool:
                self.executor_pool.shutdown(wait=True)
            
            logger.info("workflow_engine_cleanup_completed")
        
        except Exception as e:
            logger.error("workflow_engine_cleanup_failed", error=str(e))