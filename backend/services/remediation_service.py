"""
Automated Remediation Workflows Service for PolicyCortex
Provides intelligent automated remediation with approval workflows
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

class RemediationType(Enum):
    """Types of remediation actions"""
    CONFIGURATION_CHANGE = "configuration_change"
    RESOURCE_RESTART = "resource_restart"
    SCALE_RESOURCE = "scale_resource"
    APPLY_PATCH = "apply_patch"
    ENABLE_FEATURE = "enable_feature"
    DISABLE_FEATURE = "disable_feature"
    TAG_RESOURCE = "tag_resource"
    BACKUP_RESOURCE = "backup_resource"
    RESTORE_RESOURCE = "restore_resource"
    DELETE_RESOURCE = "delete_resource"
    QUARANTINE_RESOURCE = "quarantine_resource"
    NETWORK_ISOLATION = "network_isolation"
    ACCESS_REVOKE = "access_revoke"
    POLICY_ENFORCEMENT = "policy_enforcement"
    CUSTOM_SCRIPT = "custom_script"

class RemediationStatus(Enum):
    """Remediation workflow status"""
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

class RemediationPriority(Enum):
    """Remediation priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class RemediationStep:
    """Individual remediation step"""
    id: str
    name: str
    action_type: RemediationType
    parameters: Dict[str, Any]
    timeout_seconds: int = 300
    retry_count: int = 3
    rollback_on_failure: bool = True
    validation_required: bool = True
    dependencies: List[str] = field(default_factory=list)

@dataclass
class RemediationWorkflow:
    """Complete remediation workflow"""
    id: str
    name: str
    description: str
    trigger: Dict[str, Any]  # What triggers this workflow
    steps: List[RemediationStep]
    priority: RemediationPriority
    approval_required: bool
    approvers: List[str] = field(default_factory=list)
    auto_rollback: bool = True
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RemediationExecution:
    """Execution instance of a remediation workflow"""
    id: str
    workflow_id: str
    resource_id: str
    status: RemediationStatus
    priority: RemediationPriority
    current_step: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approval_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    rollback_performed: bool = False
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RemediationTemplate:
    """Reusable remediation template"""
    id: str
    name: str
    category: str
    description: str
    steps: List[RemediationStep]
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class RemediationService:
    """Automated remediation workflow service"""
    
    def __init__(self):
        """Initialize remediation service"""
        self.workflows: Dict[str, RemediationWorkflow] = {}
        self.templates: Dict[str, RemediationTemplate] = {}
        self.executions: Dict[str, RemediationExecution] = {}
        self.execution_queue: deque = deque()
        self.approval_queue: Dict[str, RemediationExecution] = {}
        self.action_handlers: Dict[RemediationType, Callable] = {}
        self.validation_handlers: Dict[str, Callable] = {}
        
        # Initialize default templates
        self._initialize_templates()
        
        # Initialize default workflows
        self._initialize_workflows()
        
        # Register default action handlers
        self._register_default_handlers()
        
        # Start execution worker
        self.worker_task = None
    
    def _initialize_templates(self):
        """Initialize default remediation templates"""
        templates = [
            RemediationTemplate(
                id="template-security-baseline",
                name="Security Baseline Enforcement",
                category="security",
                description="Apply security baseline configuration",
                steps=[
                    RemediationStep(
                        id="step-1",
                        name="Enable encryption",
                        action_type=RemediationType.CONFIGURATION_CHANGE,
                        parameters={"setting": "encryption", "value": "enabled"}
                    ),
                    RemediationStep(
                        id="step-2",
                        name="Configure firewall",
                        action_type=RemediationType.CONFIGURATION_CHANGE,
                        parameters={"setting": "firewall", "rules": "baseline"}
                    ),
                    RemediationStep(
                        id="step-3",
                        name="Enable monitoring",
                        action_type=RemediationType.ENABLE_FEATURE,
                        parameters={"feature": "monitoring"}
                    )
                ]
            ),
            RemediationTemplate(
                id="template-cost-optimization",
                name="Cost Optimization",
                category="cost",
                description="Optimize resource costs",
                steps=[
                    RemediationStep(
                        id="step-1",
                        name="Rightsize resources",
                        action_type=RemediationType.SCALE_RESOURCE,
                        parameters={"direction": "down", "factor": 0.8}
                    ),
                    RemediationStep(
                        id="step-2",
                        name="Stop idle resources",
                        action_type=RemediationType.RESOURCE_RESTART,
                        parameters={"action": "stop", "condition": "idle"}
                    ),
                    RemediationStep(
                        id="step-3",
                        name="Apply reserved pricing",
                        action_type=RemediationType.CONFIGURATION_CHANGE,
                        parameters={"pricing": "reserved"}
                    )
                ]
            ),
            RemediationTemplate(
                id="template-compliance-fix",
                name="Compliance Violation Fix",
                category="compliance",
                description="Fix compliance violations",
                steps=[
                    RemediationStep(
                        id="step-1",
                        name="Apply required tags",
                        action_type=RemediationType.TAG_RESOURCE,
                        parameters={"tags": {"compliance": "required"}}
                    ),
                    RemediationStep(
                        id="step-2",
                        name="Update configuration",
                        action_type=RemediationType.CONFIGURATION_CHANGE,
                        parameters={"compliance": "enabled"}
                    ),
                    RemediationStep(
                        id="step-3",
                        name="Validate compliance",
                        action_type=RemediationType.CUSTOM_SCRIPT,
                        parameters={"script": "validate_compliance.py"}
                    )
                ]
            ),
            RemediationTemplate(
                id="template-incident-response",
                name="Security Incident Response",
                category="security",
                description="Respond to security incidents",
                steps=[
                    RemediationStep(
                        id="step-1",
                        name="Isolate resource",
                        action_type=RemediationType.NETWORK_ISOLATION,
                        parameters={"isolation": "full"}
                    ),
                    RemediationStep(
                        id="step-2",
                        name="Backup state",
                        action_type=RemediationType.BACKUP_RESOURCE,
                        parameters={"type": "snapshot"}
                    ),
                    RemediationStep(
                        id="step-3",
                        name="Revoke access",
                        action_type=RemediationType.ACCESS_REVOKE,
                        parameters={"scope": "all"}
                    ),
                    RemediationStep(
                        id="step-4",
                        name="Apply patches",
                        action_type=RemediationType.APPLY_PATCH,
                        parameters={"patches": "critical"}
                    )
                ]
            )
        ]
        
        for template in templates:
            self.templates[template.id] = template
    
    def _initialize_workflows(self):
        """Initialize default remediation workflows"""
        workflows = [
            RemediationWorkflow(
                id="workflow-auto-security",
                name="Automatic Security Remediation",
                description="Automatically fix security vulnerabilities",
                trigger={"type": "security_alert", "severity": "critical"},
                steps=self.templates["template-security-baseline"].steps,
                priority=RemediationPriority.CRITICAL,
                approval_required=False,
                auto_rollback=True,
                notification_channels=["security-team"]
            ),
            RemediationWorkflow(
                id="workflow-cost-approval",
                name="Cost Optimization with Approval",
                description="Optimize costs with approval workflow",
                trigger={"type": "cost_threshold", "amount": 10000},
                steps=self.templates["template-cost-optimization"].steps,
                priority=RemediationPriority.MEDIUM,
                approval_required=True,
                approvers=["cost-manager", "finance-team"],
                auto_rollback=True,
                notification_channels=["finance-team"]
            ),
            RemediationWorkflow(
                id="workflow-compliance-auto",
                name="Compliance Auto-Fix",
                description="Automatically fix compliance violations",
                trigger={"type": "compliance_violation", "auto_fix": True},
                steps=self.templates["template-compliance-fix"].steps,
                priority=RemediationPriority.HIGH,
                approval_required=False,
                auto_rollback=False,
                notification_channels=["compliance-team"]
            )
        ]
        
        for workflow in workflows:
            self.workflows[workflow.id] = workflow
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.action_handlers = {
            RemediationType.CONFIGURATION_CHANGE: self._handle_configuration_change,
            RemediationType.RESOURCE_RESTART: self._handle_resource_restart,
            RemediationType.SCALE_RESOURCE: self._handle_scale_resource,
            RemediationType.TAG_RESOURCE: self._handle_tag_resource,
            RemediationType.ENABLE_FEATURE: self._handle_enable_feature,
            RemediationType.DISABLE_FEATURE: self._handle_disable_feature,
            RemediationType.NETWORK_ISOLATION: self._handle_network_isolation,
            RemediationType.ACCESS_REVOKE: self._handle_access_revoke,
            RemediationType.CUSTOM_SCRIPT: self._handle_custom_script
        }
    
    async def start_worker(self):
        """Start the remediation execution worker"""
        if not self.worker_task:
            self.worker_task = asyncio.create_task(self._execution_worker())
            logger.info("Remediation worker started")
    
    async def stop_worker(self):
        """Stop the remediation execution worker"""
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
            self.worker_task = None
            logger.info("Remediation worker stopped")
    
    async def _execution_worker(self):
        """Main execution worker loop"""
        while True:
            try:
                # Process execution queue
                if self.execution_queue:
                    execution_id = self.execution_queue.popleft()
                    if execution_id in self.executions:
                        execution = self.executions[execution_id]
                        
                        # Check if approval is needed
                        if execution.status == RemediationStatus.AWAITING_APPROVAL:
                            continue
                        
                        # Execute if approved or no approval needed
                        if execution.status in [RemediationStatus.PENDING, RemediationStatus.APPROVED]:
                            await self._execute_workflow(execution)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution worker: {e}")
                await asyncio.sleep(5)
    
    async def trigger_remediation(
        self,
        workflow_id: str,
        resource_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RemediationExecution:
        """Trigger a remediation workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = f"exec-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(resource_id.encode()).hexdigest()[:8]}"
        
        execution = RemediationExecution(
            id=execution_id,
            workflow_id=workflow_id,
            resource_id=resource_id,
            status=RemediationStatus.PENDING,
            priority=workflow.priority,
            results={"context": context or {}}
        )
        
        self.executions[execution_id] = execution
        
        # Check if approval is required
        if workflow.approval_required:
            execution.status = RemediationStatus.AWAITING_APPROVAL
            self.approval_queue[execution_id] = execution
            await self._send_approval_request(execution, workflow)
        else:
            # Add to execution queue
            self.execution_queue.append(execution_id)
        
        # Log execution start
        self._log_execution(execution, "Remediation triggered", {"workflow": workflow.name})
        
        return execution
    
    async def approve_remediation(
        self,
        execution_id: str,
        approver: str,
        comments: Optional[str] = None
    ) -> bool:
        """Approve a remediation execution"""
        if execution_id not in self.approval_queue:
            return False
        
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        # Check if approver is authorized
        if approver not in workflow.approvers and "admin" not in approver.lower():
            logger.warning(f"Unauthorized approver: {approver}")
            return False
        
        # Update execution
        execution.status = RemediationStatus.APPROVED
        execution.approved_by = approver
        execution.approval_time = datetime.utcnow()
        
        # Remove from approval queue and add to execution queue
        del self.approval_queue[execution_id]
        self.execution_queue.append(execution_id)
        
        # Log approval
        self._log_execution(execution, "Remediation approved", {
            "approver": approver,
            "comments": comments
        })
        
        return True
    
    async def reject_remediation(
        self,
        execution_id: str,
        rejector: str,
        reason: str
    ) -> bool:
        """Reject a remediation execution"""
        if execution_id not in self.approval_queue:
            return False
        
        execution = self.executions[execution_id]
        
        # Update execution
        execution.status = RemediationStatus.REJECTED
        execution.completed_at = datetime.utcnow()
        execution.error = f"Rejected by {rejector}: {reason}"
        
        # Remove from approval queue
        del self.approval_queue[execution_id]
        
        # Log rejection
        self._log_execution(execution, "Remediation rejected", {
            "rejector": rejector,
            "reason": reason
        })
        
        return True
    
    async def _execute_workflow(self, execution: RemediationExecution):
        """Execute a remediation workflow"""
        workflow = self.workflows[execution.workflow_id]
        
        try:
            execution.status = RemediationStatus.IN_PROGRESS
            execution.started_at = datetime.utcnow()
            
            # Execute steps in order
            for step in workflow.steps:
                # Check dependencies
                if not await self._check_dependencies(step, execution):
                    raise Exception(f"Dependencies not met for step {step.name}")
                
                # Execute step
                execution.current_step = step.id
                execution.progress = (workflow.steps.index(step) / len(workflow.steps)) * 100
                
                self._log_execution(execution, f"Executing step: {step.name}", {
                    "step_id": step.id,
                    "action_type": step.action_type.value
                })
                
                # Execute with retries
                success = False
                for attempt in range(step.retry_count):
                    try:
                        result = await self._execute_step(step, execution.resource_id)
                        execution.results[step.id] = result
                        success = True
                        break
                    except Exception as e:
                        if attempt == step.retry_count - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # Validate if required
                if step.validation_required:
                    validation_result = await self._validate_step(step, execution.resource_id)
                    if not validation_result["valid"]:
                        raise Exception(f"Validation failed: {validation_result.get('reason')}")
            
            # Mark as completed
            execution.status = RemediationStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.progress = 100.0
            
            self._log_execution(execution, "Remediation completed successfully", {
                "duration": (execution.completed_at - execution.started_at).total_seconds()
            })
            
            # Send notifications
            await self._send_notifications(workflow, execution, "completed")
            
        except Exception as e:
            execution.status = RemediationStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            
            self._log_execution(execution, "Remediation failed", {"error": str(e)})
            
            # Perform rollback if configured
            if workflow.auto_rollback:
                await self._rollback_execution(execution, workflow)
            
            # Send notifications
            await self._send_notifications(workflow, execution, "failed")
    
    async def _execute_step(self, step: RemediationStep, resource_id: str) -> Dict[str, Any]:
        """Execute a single remediation step"""
        handler = self.action_handlers.get(step.action_type)
        
        if not handler:
            raise ValueError(f"No handler for action type: {step.action_type}")
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                handler(resource_id, step.parameters),
                timeout=step.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Step {step.name} timed out after {step.timeout_seconds} seconds")
    
    async def _validate_step(self, step: RemediationStep, resource_id: str) -> Dict[str, Any]:
        """Validate a remediation step"""
        # Check if custom validator exists
        validator_key = f"{step.action_type.value}_{step.id}"
        if validator_key in self.validation_handlers:
            return await self.validation_handlers[validator_key](resource_id, step.parameters)
        
        # Default validation (always passes)
        return {"valid": True, "message": "Default validation passed"}
    
    async def _check_dependencies(self, step: RemediationStep, execution: RemediationExecution) -> bool:
        """Check if step dependencies are met"""
        for dep_id in step.dependencies:
            if dep_id not in execution.results:
                return False
            if not execution.results[dep_id].get("success", False):
                return False
        return True
    
    async def _rollback_execution(self, execution: RemediationExecution, workflow: RemediationWorkflow):
        """Rollback a failed execution"""
        try:
            self._log_execution(execution, "Starting rollback", {})
            
            # Execute rollback steps in reverse order
            for step in reversed(workflow.steps):
                if step.id in execution.results:
                    # Create rollback step
                    rollback_step = RemediationStep(
                        id=f"rollback-{step.id}",
                        name=f"Rollback: {step.name}",
                        action_type=RemediationType.CUSTOM_SCRIPT,
                        parameters={"action": "rollback", "original_step": step.id}
                    )
                    
                    await self._execute_step(rollback_step, execution.resource_id)
            
            execution.rollback_performed = True
            execution.status = RemediationStatus.ROLLED_BACK
            
            self._log_execution(execution, "Rollback completed", {})
            
        except Exception as e:
            self._log_execution(execution, "Rollback failed", {"error": str(e)})
    
    async def _send_approval_request(self, execution: RemediationExecution, workflow: RemediationWorkflow):
        """Send approval request notifications"""
        logger.info(f"Approval required for {execution.id} - notifying {workflow.approvers}")
        # Implementation would send actual notifications
    
    async def _send_notifications(self, workflow: RemediationWorkflow, execution: RemediationExecution, status: str):
        """Send workflow notifications"""
        for channel in workflow.notification_channels:
            logger.info(f"Sending {status} notification to {channel} for {execution.id}")
            # Implementation would send actual notifications
    
    def _log_execution(self, execution: RemediationExecution, message: str, details: Dict[str, Any]):
        """Log execution event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "details": details
        }
        execution.execution_log.append(log_entry)
        logger.info(f"[{execution.id}] {message}")
    
    # Default action handlers
    async def _handle_configuration_change(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration change"""
        logger.info(f"Changing configuration for {resource_id}: {parameters}")
        # Implementation would make actual configuration changes
        return {"success": True, "changes": parameters}
    
    async def _handle_resource_restart(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource restart"""
        action = parameters.get("action", "restart")
        logger.info(f"Performing {action} on {resource_id}")
        # Implementation would restart actual resource
        return {"success": True, "action": action}
    
    async def _handle_scale_resource(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource scaling"""
        direction = parameters.get("direction", "up")
        factor = parameters.get("factor", 1.0)
        logger.info(f"Scaling {resource_id} {direction} by factor {factor}")
        # Implementation would scale actual resource
        return {"success": True, "scaled": direction, "factor": factor}
    
    async def _handle_tag_resource(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource tagging"""
        tags = parameters.get("tags", {})
        logger.info(f"Applying tags to {resource_id}: {tags}")
        # Implementation would apply actual tags
        return {"success": True, "tags_applied": tags}
    
    async def _handle_enable_feature(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature enablement"""
        feature = parameters.get("feature")
        logger.info(f"Enabling feature {feature} on {resource_id}")
        # Implementation would enable actual feature
        return {"success": True, "feature_enabled": feature}
    
    async def _handle_disable_feature(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature disablement"""
        feature = parameters.get("feature")
        logger.info(f"Disabling feature {feature} on {resource_id}")
        # Implementation would disable actual feature
        return {"success": True, "feature_disabled": feature}
    
    async def _handle_network_isolation(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network isolation"""
        isolation_type = parameters.get("isolation", "full")
        logger.info(f"Isolating {resource_id} with {isolation_type} isolation")
        # Implementation would isolate actual resource
        return {"success": True, "isolation": isolation_type}
    
    async def _handle_access_revoke(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle access revocation"""
        scope = parameters.get("scope", "all")
        logger.info(f"Revoking {scope} access to {resource_id}")
        # Implementation would revoke actual access
        return {"success": True, "access_revoked": scope}
    
    async def _handle_custom_script(self, resource_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom script execution"""
        script = parameters.get("script", "")
        logger.info(f"Executing custom script {script} on {resource_id}")
        # Implementation would execute actual script
        return {"success": True, "script_executed": script}
    
    # Management methods
    def get_execution(self, execution_id: str) -> Optional[RemediationExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def get_executions(self, status: Optional[RemediationStatus] = None) -> List[RemediationExecution]:
        """Get executions by status"""
        executions = list(self.executions.values())
        if status:
            executions = [e for e in executions if e.status == status]
        return sorted(executions, key=lambda x: x.created_at if hasattr(x, 'created_at') else datetime.utcnow(), reverse=True)
    
    def get_approval_queue(self) -> List[RemediationExecution]:
        """Get pending approvals"""
        return list(self.approval_queue.values())
    
    def register_action_handler(self, action_type: RemediationType, handler: Callable):
        """Register custom action handler"""
        self.action_handlers[action_type] = handler
    
    def register_validation_handler(self, key: str, handler: Callable):
        """Register custom validation handler"""
        self.validation_handlers[key] = handler

# Singleton instance
remediation_service = RemediationService()