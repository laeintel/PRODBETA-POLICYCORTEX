"""
Automation Orchestrator
    Part of Patent 2: Unified AI-Driven Platform with Multi-Objective Optimization
Enhanced with multi-objective optimization integration
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import networkx as nx
import numpy as np
from backend.core.config import settings
from backend.core.exceptions import APIError
from backend.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class AutomationState(str, Enum):
    """States of automation workflows"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class ActionType(str, Enum):
    """Types of automation actions"""
    RESOURCE_SCALING = "resource_scaling"
    POLICY_UPDATE = "policy_update"
    SECURITY_REMEDIATION = "security_remediation"
    COMPLIANCE_ENFORCEMENT = "compliance_enforcement"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TUNING = "performance_tuning"
    BACKUP_RESTORE = "backup_restore"
    ALERT_RESPONSE = "alert_response"
    WORKFLOW_TRIGGER = "workflow_trigger"
    CUSTOM_SCRIPT = "custom_script"
    OPTIMIZATION_APPLY = "optimization_apply"


class TriggerType(str, Enum):
    """Types of automation triggers"""
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    THRESHOLD_BASED = "threshold_based"
    MANUAL = "manual"
    OPTIMIZATION_RESULT = "optimization_result"
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    COST_THRESHOLD = "cost_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class ActionPriority(str, Enum):
    """Priority levels for actions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AutomationAction:
    """Represents an automation action"""
    action_id: str
    type: ActionType
    name: str
    description: str
    parameters: Dict[str, Any]
    target_resource: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 3
    rollback_action: Optional[str] = None
    priority: ActionPriority = ActionPriority.MEDIUM
    approval_required: bool = False
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationWorkflow:
    """Represents an automation workflow"""
    workflow_id: str
    name: str
    description: str
    trigger: Dict[str, Any]
    actions: List[AutomationAction]
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    state: AutomationState = AutomationState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_execution_time: int = 3600  # seconds
    auto_rollback: bool = True
    approval_required: bool = False


@dataclass
class ExecutionContext:
    """Context for workflow execution"""
    workflow_id: str
    execution_id: str
    started_at: datetime
    parameters: Dict[str, Any]
    trigger_event: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    completed_actions: Set[str] = field(default_factory=set)
    failed_actions: Set[str] = field(default_factory=set)
    rollback_executed: bool = False


class WorkflowValidator:
    """Validates automation workflows"""

    @staticmethod
    def validate_workflow(workflow: AutomationWorkflow) -> Tuple[bool, List[str]]:
        """Validate workflow configuration"""
        errors = []

        # Check for circular dependencies
        if WorkflowValidator._has_circular_dependencies(workflow.actions):
            errors.append("Circular dependencies detected in workflow")

        # Validate action parameters
        for action in workflow.actions:
            action_errors = WorkflowValidator._validate_action(action)
            errors.extend(action_errors)

        # Validate trigger configuration
        trigger_errors = WorkflowValidator._validate_trigger(workflow.trigger)
        errors.extend(trigger_errors)

        return len(errors) == 0, errors

    @staticmethod
    def _has_circular_dependencies(actions: List[AutomationAction]) -> bool:
        """Check for circular dependencies using topological sort"""
        graph = nx.DiGraph()

        for action in actions:
            graph.add_node(action.action_id)
            for dep in action.dependencies:
                graph.add_edge(dep, action.action_id)

        try:
            nx.topological_sort(graph)
            return False
        except nx.NetworkXError:
            return True

    @staticmethod
    def _validate_action(action: AutomationAction) -> List[str]:
        """Validate individual action"""
        errors = []

        if not action.name:
            errors.append(f"Action {action.action_id} missing name")

        if action.timeout <= 0:
            errors.append(f"Action {action.action_id} has invalid timeout")

        # Validate action type specific parameters
        required_params = {
            ActionType.RESOURCE_SCALING: ['resource_type', 'scale_factor'],
            ActionType.POLICY_UPDATE: ['policy_id', 'updates'],
            ActionType.SECURITY_REMEDIATION: ['threat_id', 'remediation_steps'],
            ActionType.COMPLIANCE_ENFORCEMENT: ['compliance_rule', 'enforcement_action'],
            ActionType.OPTIMIZATION_APPLY: ['optimization_id', 'solution_id']
        }

        if action.type in required_params:
            for param in required_params[action.type]:
                if param not in action.parameters:
                    errors.append(f"Action {action.action_id} missing required parameter: {param}")

        return errors

    @staticmethod
    def _validate_trigger(trigger: Dict[str, Any]) -> List[str]:
        """Validate trigger configuration"""
        errors = []

        if 'type' not in trigger:
            errors.append("Trigger missing type")
            return errors

        trigger_type = trigger.get('type')

        if trigger_type == TriggerType.SCHEDULED.value:
            if 'cron_expression' not in trigger:
                errors.append("Scheduled trigger missing cron_expression")

        elif trigger_type == TriggerType.THRESHOLD_BASED.value:
            required = ['metric', 'threshold', 'operator']
            for field in required:
                if field not in trigger:
                    errors.append(f"Threshold trigger missing {field}")

        elif trigger_type == TriggerType.EVENT_BASED.value:
            if 'event_type' not in trigger:
                errors.append("Event trigger missing event_type")

        return errors


class ActionExecutor:
    """Executes automation actions"""

    def __init__(
        self,
        resource_manager=None,
        policy_engine=None,
        security_service=None,
        azure_client=None
    ):
        self.resource_manager = resource_manager
        self.policy_engine = policy_engine
        self.security_service = security_service
        self.azure_client = azure_client
        self.execution_handlers = {
            ActionType.RESOURCE_SCALING: self._execute_resource_scaling,
            ActionType.POLICY_UPDATE: self._execute_policy_update,
            ActionType.SECURITY_REMEDIATION: self._execute_security_remediation,
            ActionType.COMPLIANCE_ENFORCEMENT: self._execute_compliance_enforcement,
            ActionType.COST_OPTIMIZATION: self._execute_cost_optimization,
            ActionType.PERFORMANCE_TUNING: self._execute_performance_tuning,
            ActionType.BACKUP_RESTORE: self._execute_backup_restore,
            ActionType.ALERT_RESPONSE: self._execute_alert_response,
            ActionType.WORKFLOW_TRIGGER: self._execute_workflow_trigger,
            ActionType.CUSTOM_SCRIPT: self._execute_custom_script,
            ActionType.OPTIMIZATION_APPLY: self._execute_optimization_apply
        }

    async def execute_action(self,
                           action: AutomationAction,
                           context: ExecutionContext) -> Dict[str, Any]:
        """Execute a single automation action"""

        logger.info(f"Executing action {action.action_id} of type {action.type}")

        try:
            # Check dependencies
            for dep in action.dependencies:
                if dep not in context.completed_actions:
                    raise APIError(f"Dependency {dep} not completed", status_code=400)

            # Get handler
            handler = self.execution_handlers.get(action.type)
            if not handler:
                raise APIError(f"No handler for action type {action.type}", status_code=400)

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(action, context),
                timeout=action.timeout
            )

            # Mark as completed
            context.completed_actions.add(action.action_id)
            context.results[action.action_id] = result

            return result

        except asyncio.TimeoutError:
            logger.error(f"Action {action.action_id} timed out")
            context.failed_actions.add(action.action_id)
            context.errors.append({
                'action_id': action.action_id,
                'error': 'Action timed out',
                'timestamp': datetime.now().isoformat()
            })
            raise

        except Exception as e:
            logger.error(f"Action {action.action_id} failed: {str(e)}")
            context.failed_actions.add(action.action_id)
            context.errors.append({
                'action_id': action.action_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

            # Attempt rollback if configured
            if action.rollback_action:
                await self._execute_rollback(action.rollback_action, context)

            raise

    async def _execute_resource_scaling(self,
                                      action: AutomationAction,
                                      context: ExecutionContext) -> Dict[str, Any]:
        """Execute resource scaling action"""
        params = action.parameters

        # Simulated implementation - would call actual Azure APIs
        return {
            'success': True,
            'action_type': 'resource_scaling',
            'resource_type': params.get('resource_type'),
            'scale_factor': params.get('scale_factor'),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_policy_update(self,
                                   action: AutomationAction,
                                   context: ExecutionContext) -> Dict[str, Any]:
        """Execute policy update action"""
        params = action.parameters

        # Simulated implementation
        return {
            'success': True,
            'action_type': 'policy_update',
            'policy_id': params.get('policy_id'),
            'updates': params.get('updates'),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_security_remediation(self,
                                          action: AutomationAction,
                                          context: ExecutionContext) -> Dict[str, Any]:
        """Execute security remediation action"""
        params = action.parameters

        # Simulated implementation
        return {
            'success': True,
            'action_type': 'security_remediation',
            'threat_id': params.get('threat_id'),
            'remediation_steps': params.get('remediation_steps'),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_compliance_enforcement(self,
                                            action: AutomationAction,
                                            context: ExecutionContext) -> Dict[str, Any]:
        """Execute compliance enforcement action"""
        params = action.parameters

        # Simulated implementation
        return {
            'success': True,
            'action_type': 'compliance_enforcement',
            'compliance_rule': params.get('compliance_rule'),
            'enforcement_action': params.get('enforcement_action'),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_optimization_apply(self,
                                        action: AutomationAction,
                                        context: ExecutionContext) -> Dict[str, Any]:
        """Apply optimization solution from multi-objective optimizer"""
        params = action.parameters
        optimization_id = params.get('optimization_id')
        solution_id = params.get('solution_id')

        # Integrate with multi-objective optimizer
        from .multi_objective_optimizer import multi_objective_optimizer

        result = await multi_objective_optimizer.apply_solution(
            solution_id=solution_id,
            dry_run=params.get('dry_run', False)
        )

        return {
            'success': True,
            'action_type': 'optimization_apply',
            'optimization_id': optimization_id,
            'solution_id': solution_id,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_cost_optimization(self,
                                       action: AutomationAction,
                                       context: ExecutionContext) -> Dict[str, Any]:
        """Execute cost optimization action"""
        return {
            'success': True,
            'action_type': 'cost_optimization',
            'optimizations_applied': [],
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_performance_tuning(self,
                                        action: AutomationAction,
                                        context: ExecutionContext) -> Dict[str, Any]:
        """Execute performance tuning action"""
        return {
            'success': True,
            'action_type': 'performance_tuning',
            'tunings_applied': [],
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_backup_restore(self,
                                    action: AutomationAction,
                                    context: ExecutionContext) -> Dict[str, Any]:
        """Execute backup/restore action"""
        return {
            'success': True,
            'action_type': 'backup_restore',
            'operation': action.parameters.get('operation', 'backup'),
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_alert_response(self,
                                    action: AutomationAction,
                                    context: ExecutionContext) -> Dict[str, Any]:
        """Execute alert response action"""
        return {
            'success': True,
            'action_type': 'alert_response',
            'alerts_handled': [],
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_workflow_trigger(self,
                                      action: AutomationAction,
                                      context: ExecutionContext) -> Dict[str, Any]:
        """Execute workflow trigger action"""
        target_workflow = action.parameters.get('target_workflow_id')

        return {
            'success': True,
            'action_type': 'workflow_trigger',
            'triggered_workflow': target_workflow,
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_custom_script(self,
                                   action: AutomationAction,
                                   context: ExecutionContext) -> Dict[str, Any]:
        """Execute custom script action"""
        return {
            'success': True,
            'action_type': 'custom_script',
            'script': action.parameters.get('script', ''),
            'output': '',
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_rollback(self, rollback_action_id: str, context: ExecutionContext):
        """Execute rollback action"""
        logger.info(f"Executing rollback action {rollback_action_id}")
        # Implement rollback logic


class AutomationOrchestrator:
    """Main automation orchestration service - integrates with multi-objective optimizer"""

    def __init__(self):
        self.workflows: Dict[str, AutomationWorkflow] = {}
        self.executor = None
        self.validator = WorkflowValidator()
        self.execution_queue = asyncio.Queue()
        self.active_executions: Dict[str, ExecutionContext] = {}
        self._initialized = False
        self.executor_pool = None
        self.max_concurrent_executions = 10

        # Integration with multi-objective optimizer
        self.optimization_workflows = {}

    async def initialize(
        self,
        resource_manager=None,
        policy_engine=None,
        security_service=None,
        azure_client=None
    ):
        """Initialize the orchestrator"""
        self.executor = ActionExecutor(
            resource_manager,
            policy_engine,
            security_service,
            azure_client
        )
        self._initialized = True

        # Initialize thread pool
        self.executor_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_executions)

        # Start execution worker
        asyncio.create_task(self._execution_worker())

        # Initialize predefined workflows
        self._initialize_optimization_workflows()

        logger.info("Automation orchestrator initialized")

    def _initialize_optimization_workflows(self):
        """Initialize workflows that integrate with multi-objective optimizer"""

        # Cost-Security Optimization Workflow
        cost_security_workflow = AutomationWorkflow(
            workflow_id="cost_security_optimization",
            name="Cost-Security Multi-Objective Optimization",
            description="Optimizes cost and security objectives simultaneously",
            trigger={
                'type': TriggerType.SCHEDULED.value,
                'cron_expression': '0 0 * * *'  # Daily
            },
            actions=[
                AutomationAction(
                    action_id="run_optimization",
                    type=ActionType.OPTIMIZATION_APPLY,
                    name="Run Multi-Objective Optimization",
                    description="Execute cost-security optimization",
                    parameters={
                        'objectives': ['minimize_cost', 'maximize_security'],
                        'constraints': ['compliance', 'performance'],
                        'algorithm': 'nsga2'
                    },
                    priority=ActionPriority.HIGH
                ),
                AutomationAction(
                    action_id="apply_optimal_solution",
                    type=ActionType.OPTIMIZATION_APPLY,
                    name="Apply Optimal Solution",
                    description="Apply the selected optimal solution",
                    parameters={
                        'optimization_id': '{{run_optimization.result.optimization_id}}',
                        'solution_id': '{{run_optimization.result.solution_id}}',
                        'dry_run': False
                    },
                    dependencies=['run_optimization'],
                    priority=ActionPriority.HIGH,
                    approval_required=True
                )
            ]
        )

        # Performance-Cost-Compliance Optimization Workflow
        perf_cost_compliance_workflow = AutomationWorkflow(
            workflow_id="performance_cost_compliance_optimization",
            name="Performance-Cost-Compliance Optimization",
            description="Balances performance, cost, and compliance requirements",
            trigger={
                'type': TriggerType.PERFORMANCE_DEGRADATION.value,
                'threshold': 80  # 80% performance degradation
            },
            actions=[
                AutomationAction(
                    action_id="analyze_current_state",
                    type=ActionType.CUSTOM_SCRIPT,
                    name="Analyze Current State",
                    description="Gather current metrics",
                    parameters={
                        'script': 'analyze_metrics.py'
                    },
                    priority=ActionPriority.HIGH
                ),
                AutomationAction(
                    action_id="run_multi_objective_optimization",
                    type=ActionType.OPTIMIZATION_APPLY,
                    name="Run Multi-Objective Optimization",
                    description="Find optimal balance",
                    parameters={
                        'objectives': ['maximize_performance', 'minimize_cost', 'minimize_compliance_risk'],
                        'constraints': ['budget', 'availability'],
                        'algorithm': 'nsga3',
                        'selection_method': 'topsis'
                    },
                    dependencies=['analyze_current_state'],
                    priority=ActionPriority.CRITICAL
                )
            ]
        )

        self.optimization_workflows = {
            cost_security_workflow.workflow_id: cost_security_workflow,
            perf_cost_compliance_workflow.workflow_id: perf_cost_compliance_workflow
        }

        # Add to main workflow registry
        self.workflows.update(self.optimization_workflows)

    async def create_workflow(self, workflow: AutomationWorkflow) -> str:
        """Create a new automation workflow"""
        # Validate workflow
        is_valid, errors = self.validator.validate_workflow(workflow)
        if not is_valid:
            raise APIError(f"Invalid workflow: {', '.join(errors)}", status_code=400)

        # Generate ID if not provided
        if not workflow.workflow_id:
            workflow.workflow_id = str(uuid.uuid4())

        # Store workflow
        self.workflows[workflow.workflow_id] = workflow

        # Cache workflow
        await self._cache_workflow(workflow)

        # Schedule if needed
        if workflow.trigger['type'] == TriggerType.SCHEDULED.value:
            await self._schedule_workflow(workflow)

        return workflow.workflow_id

    async def execute_workflow(self,
                             workflow_id: str,
                             parameters: Dict[str, Any] = None) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise APIError(f"Workflow {workflow_id} not found", status_code=404)

        workflow = self.workflows[workflow_id]

        # Create execution context
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            workflow_id=workflow_id,
            execution_id=execution_id,
            started_at=datetime.now(),
            parameters=parameters or {}
        )

        # Add to active executions
        self.active_executions[execution_id] = context

        # Queue for execution
        await self.execution_queue.put((workflow, context))

        return execution_id

    async def trigger_optimization_workflow(self,
                                          objectives: List[str],
                                          constraints: List[str],
                                          parameters: Dict[str, Any] = None) -> str:
        """Trigger a workflow based on optimization objectives"""
        # Create dynamic optimization workflow
        workflow = AutomationWorkflow(
            workflow_id=f"opt_workflow_{uuid.uuid4().hex[:8]}",
            name="Dynamic Optimization Workflow",
            description=f"Optimize {', '.join(objectives)}",
            trigger={'type': TriggerType.OPTIMIZATION_RESULT.value},
            actions=[
                AutomationAction(
                    action_id="run_optimization",
                    type=ActionType.OPTIMIZATION_APPLY,
                    name="Run Optimization",
                    description="Execute multi-objective optimization",
                    parameters={
                        'objectives': objectives,
                        'constraints': constraints,
                        'algorithm': parameters.get('algorithm', 'nsga2'),
                        'selection_method': parameters.get('selection_method', 'weighted_sum')
                    },
                    priority=ActionPriority.HIGH
                ),
                AutomationAction(
                    action_id="apply_solution",
                    type=ActionType.OPTIMIZATION_APPLY,
                    name="Apply Solution",
                    description="Apply optimization results",
                    parameters={
                        'solution_id': '{{run_optimization.result.solution_id}}',
                        'dry_run': parameters.get('dry_run', True)
                    },
                    dependencies=['run_optimization'],
                    priority=ActionPriority.HIGH,
                    approval_required=not parameters.get('auto_apply', False)
                )
            ]
        )

        # Create and execute
        workflow_id = await self.create_workflow(workflow)
        execution_id = await self.execute_workflow(workflow_id, parameters)

        return execution_id

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        if workflow_id not in self.workflows:
            raise APIError(f"Workflow {workflow_id} not found", status_code=404)

        workflow = self.workflows[workflow_id]

        # Get active executions
        active_executions = [
            {
                'execution_id': ctx.execution_id,
                'started_at': ctx.started_at.isoformat(),
                'completed_actions': list(ctx.completed_actions),
                'failed_actions': list(ctx.failed_actions)
            }
            for ctx in self.active_executions.values()
            if ctx.workflow_id == workflow_id
        ]

        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'state': workflow.state.value,
            'created_at': workflow.created_at.isoformat(),
            'updated_at': workflow.updated_at.isoformat(),
            'active_executions': active_executions,
            'execution_history': workflow.execution_history[-10:]  # Last 10 executions
        }

    async def cancel_execution(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a workflow execution"""
        if execution_id not in self.active_executions:
            raise APIError(f"Execution {execution_id} not found", status_code=404)

        context = self.active_executions[execution_id]

        # Mark as cancelled
        context.errors.append({
            'error': 'Execution cancelled by user',
            'timestamp': datetime.now().isoformat()
        })

        # Remove from active executions
        del self.active_executions[execution_id]

        return {
            'execution_id': execution_id,
            'status': 'cancelled',
            'completed_actions': list(context.completed_actions),
            'failed_actions': list(context.failed_actions)
        }

    async def _execution_worker(self):
        """Worker to process workflow executions"""
        while True:
            try:
                # Get next workflow to execute
                workflow, context = await self.execution_queue.get()

                # Execute workflow
                await self._execute_workflow(workflow, context)

            except Exception as e:
                logger.error(f"Execution worker error: {str(e)}")
                await asyncio.sleep(1)

    async def _execute_workflow(self,
                              workflow: AutomationWorkflow,
                              context: ExecutionContext):
        """Execute a complete workflow"""
        logger.info(f"Starting execution of workflow {workflow.workflow_id}")

        try:
            # Update workflow state
            workflow.state = AutomationState.RUNNING
            workflow.updated_at = datetime.now()

            # Get execution order (topological sort)
            execution_order = self._get_execution_order(workflow.actions)

            # Execute actions in order
            for action_id in execution_order:
                action = next(a for a in workflow.actions if a.action_id == action_id)

                # Check if should continue
                if context.execution_id not in self.active_executions:
                    logger.info(f"Execution {context.execution_id} cancelled")
                    break

                # Handle approval if required
                if action.approval_required:
                    approval = await self._handle_approval(action, context)
                    if not approval:
                        context.failed_actions.add(action.action_id)
                        continue

                # Execute action with retries
                for attempt in range(action.retry_count):
                    try:
                        await self.executor.execute_action(action, context)
                        break
                    except Exception as e:
                        if attempt == action.retry_count - 1:
                            raise
                        logger.warning(f"Action {action_id} failed, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

            # Update workflow state
            if context.failed_actions:
                workflow.state = AutomationState.PARTIALLY_COMPLETED
            else:
                workflow.state = AutomationState.COMPLETED

            # Record execution
            execution_record = {
                'execution_id': context.execution_id,
                'started_at': context.started_at.isoformat(),
                'completed_at': datetime.now().isoformat(),
                'completed_actions': list(context.completed_actions),
                'failed_actions': list(context.failed_actions),
                'results': context.results,
                'errors': context.errors
            }

            workflow.execution_history.append(execution_record)

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            workflow.state = AutomationState.FAILED

            # Execute rollback if configured
            if workflow.auto_rollback and context.completed_actions:
                await self._execute_rollback(workflow, context)

        finally:
            # Remove from active executions
            if context.execution_id in self.active_executions:
                del self.active_executions[context.execution_id]

            # Update workflow
            workflow.updated_at = datetime.now()
            await self._cache_workflow(workflow)

    def _get_execution_order(self, actions: List[AutomationAction]) -> List[str]:
        """Get execution order using topological sort"""
        graph = nx.DiGraph()

        for action in actions:
            graph.add_node(action.action_id)
            for dep in action.dependencies:
                graph.add_edge(dep, action.action_id)

        return list(nx.topological_sort(graph))

    async def _handle_approval(self, action: AutomationAction, context: ExecutionContext) -> bool:
        """Handle approval workflow"""
        # In production, integrate with approval system
        # For now, auto-approve based on priority
        if action.priority == ActionPriority.CRITICAL:
            return False  # Require manual approval
        return True

    async def _execute_rollback(self, workflow: AutomationWorkflow, context: ExecutionContext):
        """Execute rollback for failed workflow"""
        logger.info(f"Executing rollback for workflow {workflow.workflow_id}")

        # Execute rollback actions in reverse order
        for action in reversed(workflow.actions):
            if action.action_id in context.completed_actions and action.rollback_action:
                # Create rollback action
                rollback_action = AutomationAction(
                    action_id=f"rollback_{action.action_id}",
                    type=ActionType.CUSTOM_SCRIPT,
                    name=f"Rollback {action.name}",
                    description=f"Rollback action for {action.name}",
                    parameters={'rollback_config': action.rollback_action},
                    priority=ActionPriority.HIGH
                )

                try:
                    await self.executor.execute_action(rollback_action, context)
                    context.rollback_executed = True
                except Exception as e:
                    logger.error(f"Rollback failed for action {action.action_id}: {str(e)}")

    async def _schedule_workflow(self, workflow: AutomationWorkflow):
        """Schedule workflow execution"""
        # Implement cron-based scheduling
        pass

    async def _cache_workflow(self, workflow: AutomationWorkflow):
        """Cache workflow configuration"""
        cache_key = f"workflow:{workflow.workflow_id}"
        cache_data = {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'state': workflow.state.value,
            'trigger': workflow.trigger,
            'updated_at': workflow.updated_at.isoformat()
        }

        await redis_client.setex(
            cache_key,
            timedelta(hours=24),
            json.dumps(cache_data)
        )

    async def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization workflows"""
        insights = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'cost_savings': 0,
            'performance_improvements': 0,
            'security_enhancements': 0,
            'optimization_trends': []
        }

        # Analyze optimization workflow executions
        for workflow in self.optimization_workflows.values():
            for execution in workflow.execution_history:
                insights['total_optimizations'] += 1

                if not execution.get('failed_actions'):
                    insights['successful_optimizations'] += 1

                # Extract optimization results
                results = execution.get('results', {})
                for action_id, result in results.items():
                    if result.get('action_type') == 'optimization_apply':
                        opt_result = result.get('result', {})
                        if 'cost_savings' in opt_result:
                            insights['cost_savings'] += opt_result['cost_savings']
                        if 'performance_improvement' in opt_result:
                            insights['performance_improvements'] + = (
                                opt_result['performance_improvement']
                            )
                        if 'security_score' in opt_result:
                            insights['security_enhancements'] += opt_result['security_score']

        return insights


# Global instances
automation_orchestrator = AutomationOrchestrator()

# For backward compatibility
workflow_engine = None  # Will be initialized when needed


class WorkflowEngine:
    """
    Legacy orchestration engine - maintained for backward compatibility
    New implementations should use AutomationOrchestrator
    """

    def __init__(self, azure_client=None):
        self.azure_client = azure_client
        self.action_executor = ActionExecutor(azure_client=azure_client)
        self.active_executions: Dict[str, ExecutionContext] = {}
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
                            "message": "Performance degradation detected and
                                auto-scaling initiated for {{event.resource_id}}"
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
                        self._resolve_template_string(
                            item,
                            context) if isinstance(item,
                            str
                        ) else item
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
