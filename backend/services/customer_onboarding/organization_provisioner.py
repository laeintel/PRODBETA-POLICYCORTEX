"""
Organization Provisioner Module
Handles automated provisioning of customer organizations
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import uuid

logger = structlog.get_logger(__name__)

class ProvisioningStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class ResourceType(str, Enum):
    DATABASE = "database"
    STORAGE = "storage"
    KEY_VAULT = "key_vault"
    IDENTITY = "identity"
    NETWORK = "network"
    COMPUTE = "compute"
    MONITORING = "monitoring"
    POLICIES = "policies"

@dataclass
class ProvisioningTask:
    """Represents a provisioning task"""
    task_id: str
    resource_type: ResourceType
    action: str
    status: ProvisioningStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    rollback_action: Optional[str] = None

@dataclass
class OrganizationResources:
    """Resources provisioned for an organization"""
    tenant_id: str
    database_url: Optional[str] = None
    storage_account: Optional[str] = None
    key_vault_url: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    service_principals: List[str] = field(default_factory=list)
    resource_group: Optional[str] = None
    custom_domain: Optional[str] = None
    ssl_certificate: Optional[str] = None

class OrganizationProvisioner:
    """
    Handles automated provisioning of customer organizations
    """
    
    def __init__(self):
        self.provisioning_tasks = {}
        self.organization_resources = {}
        self.provisioning_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load provisioning templates"""
        return {
            'starter': {
                'name': 'Starter',
                'resources': {
                    'database': {
                        'type': 'shared',
                        'size': 'small',
                        'backup_enabled': False
                    },
                    'storage': {
                        'type': 'shared',
                        'quota_gb': 10,
                        'redundancy': 'lrs'
                    },
                    'compute': {
                        'type': 'serverless',
                        'max_instances': 1
                    },
                    'monitoring': {
                        'retention_days': 7,
                        'alerts_enabled': True
                    }
                }
            },
            'professional': {
                'name': 'Professional',
                'resources': {
                    'database': {
                        'type': 'dedicated',
                        'size': 'medium',
                        'backup_enabled': True,
                        'geo_redundant': False
                    },
                    'storage': {
                        'type': 'dedicated',
                        'quota_gb': 100,
                        'redundancy': 'zrs'
                    },
                    'compute': {
                        'type': 'dedicated',
                        'min_instances': 1,
                        'max_instances': 5
                    },
                    'monitoring': {
                        'retention_days': 30,
                        'alerts_enabled': True,
                        'custom_metrics': True
                    },
                    'network': {
                        'private_endpoint': True,
                        'firewall_enabled': True
                    }
                }
            },
            'enterprise': {
                'name': 'Enterprise',
                'resources': {
                    'database': {
                        'type': 'dedicated',
                        'size': 'large',
                        'backup_enabled': True,
                        'geo_redundant': True,
                        'read_replicas': 2
                    },
                    'storage': {
                        'type': 'premium',
                        'quota_gb': 1000,
                        'redundancy': 'grs'
                    },
                    'compute': {
                        'type': 'dedicated',
                        'min_instances': 2,
                        'max_instances': 20,
                        'auto_scaling': True
                    },
                    'monitoring': {
                        'retention_days': 365,
                        'alerts_enabled': True,
                        'custom_metrics': True,
                        'log_analytics': True
                    },
                    'network': {
                        'private_endpoint': True,
                        'firewall_enabled': True,
                        'ddos_protection': True,
                        'custom_domain': True
                    },
                    'security': {
                        'key_vault': True,
                        'managed_identity': True,
                        'encryption_at_rest': True
                    }
                }
            }
        }
        
    async def provision_organization(self,
                                   tenant_id: str,
                                   organization_name: str,
                                   template: str,
                                   configuration: Dict[str, Any]) -> OrganizationResources:
        """
        Provision resources for a new organization
        
        Args:
            tenant_id: Unique tenant identifier
            organization_name: Name of the organization
            template: Provisioning template to use
            configuration: Additional configuration options
            
        Returns:
            Provisioned organization resources
        """
        
        logger.info(f"Starting provisioning for tenant {tenant_id} using template {template}")
        
        if template not in self.provisioning_templates:
            raise ValueError(f"Unknown provisioning template: {template}")
            
        template_config = self.provisioning_templates[template]
        
        # Create provisioning plan
        tasks = await self._create_provisioning_plan(
            tenant_id,
            organization_name,
            template_config,
            configuration
        )
        
        # Execute provisioning tasks
        resources = await self._execute_provisioning(tenant_id, tasks)
        
        self.organization_resources[tenant_id] = resources
        
        logger.info(f"Completed provisioning for tenant {tenant_id}")
        
        return resources
        
    async def _create_provisioning_plan(self,
                                      tenant_id: str,
                                      organization_name: str,
                                      template_config: Dict[str, Any],
                                      configuration: Dict[str, Any]) -> List[ProvisioningTask]:
        """Create provisioning task plan"""
        
        tasks = []
        
        # Resource group (Azure-specific, but demonstrates the concept)
        rg_task = ProvisioningTask(
            task_id=f"{tenant_id}_resource_group",
            resource_type=ResourceType.NETWORK,
            action=f"create_resource_group_{organization_name.lower().replace(' ', '_')}",
            status=ProvisioningStatus.PENDING,
            rollback_action="delete_resource_group"
        )
        tasks.append(rg_task)
        
        # Database provisioning
        if 'database' in template_config['resources']:
            db_config = template_config['resources']['database']
            db_task = ProvisioningTask(
                task_id=f"{tenant_id}_database",
                resource_type=ResourceType.DATABASE,
                action=f"provision_database_{db_config['type']}_{db_config['size']}",
                status=ProvisioningStatus.PENDING,
                dependencies=[rg_task.task_id],
                rollback_action="delete_database"
            )
            tasks.append(db_task)
            
        # Storage provisioning
        if 'storage' in template_config['resources']:
            storage_config = template_config['resources']['storage']
            storage_task = ProvisioningTask(
                task_id=f"{tenant_id}_storage",
                resource_type=ResourceType.STORAGE,
                action=f"provision_storage_{storage_config['type']}_{storage_config['quota_gb']}gb",
                status=ProvisioningStatus.PENDING,
                dependencies=[rg_task.task_id],
                rollback_action="delete_storage"
            )
            tasks.append(storage_task)
            
        # Key Vault provisioning (for enterprise)
        if 'security' in template_config['resources'] and template_config['resources']['security'].get('key_vault'):
            kv_task = ProvisioningTask(
                task_id=f"{tenant_id}_keyvault",
                resource_type=ResourceType.KEY_VAULT,
                action="provision_key_vault",
                status=ProvisioningStatus.PENDING,
                dependencies=[rg_task.task_id],
                rollback_action="delete_key_vault"
            )
            tasks.append(kv_task)
            
        # Identity provisioning
        identity_task = ProvisioningTask(
            task_id=f"{tenant_id}_identity",
            resource_type=ResourceType.IDENTITY,
            action="create_service_principals",
            status=ProvisioningStatus.PENDING,
            dependencies=[rg_task.task_id],
            rollback_action="delete_service_principals"
        )
        tasks.append(identity_task)
        
        # Monitoring setup
        if 'monitoring' in template_config['resources']:
            monitor_config = template_config['resources']['monitoring']
            monitor_task = ProvisioningTask(
                task_id=f"{tenant_id}_monitoring",
                resource_type=ResourceType.MONITORING,
                action=f"setup_monitoring_{monitor_config['retention_days']}days",
                status=ProvisioningStatus.PENDING,
                dependencies=[rg_task.task_id],
                rollback_action="delete_monitoring"
            )
            tasks.append(monitor_task)
            
        # Default policies
        policy_task = ProvisioningTask(
            task_id=f"{tenant_id}_policies",
            resource_type=ResourceType.POLICIES,
            action="apply_default_policies",
            status=ProvisioningStatus.PENDING,
            dependencies=[db_task.task_id] if 'database' in template_config['resources'] else [],
            rollback_action="remove_policies"
        )
        tasks.append(policy_task)
        
        self.provisioning_tasks[tenant_id] = tasks
        
        return tasks
        
    async def _execute_provisioning(self,
                                  tenant_id: str,
                                  tasks: List[ProvisioningTask]) -> OrganizationResources:
        """Execute provisioning tasks"""
        
        resources = OrganizationResources(tenant_id=tenant_id)
        completed_tasks = set()
        failed_task = None
        
        try:
            while len(completed_tasks) < len(tasks):
                # Find tasks ready to execute
                ready_tasks = [
                    task for task in tasks
                    if task.task_id not in completed_tasks
                    and task.status == ProvisioningStatus.PENDING
                    and all(dep in completed_tasks for dep in task.dependencies)
                ]
                
                if not ready_tasks and len(completed_tasks) < len(tasks):
                    # Check for failures
                    failed_tasks = [t for t in tasks if t.status == ProvisioningStatus.FAILED]
                    if failed_tasks:
                        failed_task = failed_tasks[0]
                        raise Exception(f"Task {failed_task.task_id} failed: {failed_task.error_message}")
                    
                    # No tasks ready and no failures - might be a dependency issue
                    await asyncio.sleep(1)
                    continue
                    
                # Execute ready tasks in parallel
                execution_tasks = []
                for task in ready_tasks:
                    task.status = ProvisioningStatus.IN_PROGRESS
                    task.started_at = datetime.utcnow()
                    execution_tasks.append(self._execute_task(task, resources))
                    
                # Wait for tasks to complete
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Process results
                for task, result in zip(ready_tasks, results):
                    if isinstance(result, Exception):
                        task.status = ProvisioningStatus.FAILED
                        task.error_message = str(result)
                        
                        # Retry if possible
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = ProvisioningStatus.PENDING
                            logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                        else:
                            failed_task = task
                            raise result
                    else:
                        task.status = ProvisioningStatus.COMPLETED
                        task.completed_at = datetime.utcnow()
                        completed_tasks.add(task.task_id)
                        logger.info(f"Completed task {task.task_id}")
                        
        except Exception as e:
            logger.error(f"Provisioning failed for tenant {tenant_id}: {e}")
            
            # Rollback completed tasks
            await self._rollback_provisioning(tenant_id, tasks, completed_tasks)
            
            raise Exception(f"Provisioning failed and rolled back: {e}")
            
        return resources
        
    async def _execute_task(self,
                          task: ProvisioningTask,
                          resources: OrganizationResources) -> None:
        """Execute a single provisioning task"""
        
        logger.info(f"Executing task {task.task_id}: {task.action}")
        
        # Simulate task execution with different resource types
        if task.resource_type == ResourceType.DATABASE:
            await self._provision_database(task, resources)
        elif task.resource_type == ResourceType.STORAGE:
            await self._provision_storage(task, resources)
        elif task.resource_type == ResourceType.KEY_VAULT:
            await self._provision_key_vault(task, resources)
        elif task.resource_type == ResourceType.IDENTITY:
            await self._provision_identity(task, resources)
        elif task.resource_type == ResourceType.NETWORK:
            await self._provision_network(task, resources)
        elif task.resource_type == ResourceType.MONITORING:
            await self._provision_monitoring(task, resources)
        elif task.resource_type == ResourceType.POLICIES:
            await self._provision_policies(task, resources)
        else:
            raise ValueError(f"Unknown resource type: {task.resource_type}")
            
    async def _provision_database(self,
                                task: ProvisioningTask,
                                resources: OrganizationResources) -> None:
        """Provision database resources"""
        
        # Simulate database provisioning
        await asyncio.sleep(2)
        
        # Parse database configuration from action
        if 'shared' in task.action:
            resources.database_url = f"postgresql://shared.db.policycortex.com/{resources.tenant_id}"
        else:
            resources.database_url = f"postgresql://{resources.tenant_id}.db.policycortex.com/governance"
            
        logger.info(f"Provisioned database: {resources.database_url}")
        
    async def _provision_storage(self,
                               task: ProvisioningTask,
                               resources: OrganizationResources) -> None:
        """Provision storage resources"""
        
        # Simulate storage provisioning
        await asyncio.sleep(1.5)
        
        storage_account = f"stpc{resources.tenant_id[:8]}".lower()
        resources.storage_account = storage_account
        
        logger.info(f"Provisioned storage account: {storage_account}")
        
    async def _provision_key_vault(self,
                                 task: ProvisioningTask,
                                 resources: OrganizationResources) -> None:
        """Provision key vault"""
        
        # Simulate key vault provisioning
        await asyncio.sleep(1)
        
        resources.key_vault_url = f"https://kv-{resources.tenant_id[:8]}.vault.azure.net/"
        
        logger.info(f"Provisioned key vault: {resources.key_vault_url}")
        
    async def _provision_identity(self,
                                task: ProvisioningTask,
                                resources: OrganizationResources) -> None:
        """Provision identity resources"""
        
        # Simulate identity provisioning
        await asyncio.sleep(1)
        
        # Create API keys
        for i in range(2):
            api_key = f"pk_{resources.tenant_id}_{uuid.uuid4().hex[:16]}"
            resources.api_keys.append(api_key)
            
        # Create service principals
        sp_id = str(uuid.uuid4())
        resources.service_principals.append(sp_id)
        
        logger.info(f"Provisioned {len(resources.api_keys)} API keys and {len(resources.service_principals)} service principals")
        
    async def _provision_network(self,
                               task: ProvisioningTask,
                               resources: OrganizationResources) -> None:
        """Provision network resources"""
        
        # Simulate network provisioning
        await asyncio.sleep(1)
        
        if 'resource_group' in task.action:
            resources.resource_group = f"rg-policycortex-{resources.tenant_id[:8]}"
            
        logger.info(f"Provisioned network resources: {resources.resource_group}")
        
    async def _provision_monitoring(self,
                                  task: ProvisioningTask,
                                  resources: OrganizationResources) -> None:
        """Provision monitoring resources"""
        
        # Simulate monitoring setup
        await asyncio.sleep(0.5)
        
        logger.info("Configured monitoring and alerting")
        
    async def _provision_policies(self,
                                task: ProvisioningTask,
                                resources: OrganizationResources) -> None:
        """Apply default policies"""
        
        # Simulate policy application
        await asyncio.sleep(0.5)
        
        logger.info("Applied default governance policies")
        
    async def _rollback_provisioning(self,
                                   tenant_id: str,
                                   tasks: List[ProvisioningTask],
                                   completed_tasks: set) -> None:
        """Rollback provisioned resources"""
        
        logger.info(f"Starting rollback for tenant {tenant_id}")
        
        # Execute rollback in reverse order
        rollback_tasks = [
            task for task in reversed(tasks)
            if task.task_id in completed_tasks and task.rollback_action
        ]
        
        for task in rollback_tasks:
            try:
                task.status = ProvisioningStatus.ROLLING_BACK
                await self._execute_rollback(task)
                task.status = ProvisioningStatus.ROLLED_BACK
                logger.info(f"Rolled back task {task.task_id}")
            except Exception as e:
                logger.error(f"Failed to rollback task {task.task_id}: {e}")
                
    async def _execute_rollback(self, task: ProvisioningTask) -> None:
        """Execute rollback for a task"""
        
        # Simulate rollback
        await asyncio.sleep(0.5)
        
        logger.info(f"Executed rollback action: {task.rollback_action}")
        
    async def deprovision_organization(self, tenant_id: str) -> bool:
        """
        Deprovision an organization's resources
        
        Args:
            tenant_id: Tenant to deprovision
            
        Returns:
            Success status
        """
        
        logger.info(f"Starting deprovisioning for tenant {tenant_id}")
        
        if tenant_id not in self.provisioning_tasks:
            logger.warning(f"No provisioning tasks found for tenant {tenant_id}")
            return False
            
        tasks = self.provisioning_tasks[tenant_id]
        
        # Execute rollback for all completed tasks
        completed_tasks = {
            task.task_id for task in tasks
            if task.status == ProvisioningStatus.COMPLETED
        }
        
        await self._rollback_provisioning(tenant_id, tasks, completed_tasks)
        
        # Clean up records
        del self.provisioning_tasks[tenant_id]
        if tenant_id in self.organization_resources:
            del self.organization_resources[tenant_id]
            
        logger.info(f"Completed deprovisioning for tenant {tenant_id}")
        
        return True
        
    def get_provisioning_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get provisioning status for a tenant"""
        
        if tenant_id not in self.provisioning_tasks:
            return {'status': 'not_found'}
            
        tasks = self.provisioning_tasks[tenant_id]
        
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == ProvisioningStatus.COMPLETED)
        failed_tasks = sum(1 for t in tasks if t.status == ProvisioningStatus.FAILED)
        in_progress_tasks = sum(1 for t in tasks if t.status == ProvisioningStatus.IN_PROGRESS)
        
        overall_status = 'completed'
        if failed_tasks > 0:
            overall_status = 'failed'
        elif in_progress_tasks > 0:
            overall_status = 'in_progress'
        elif completed_tasks < total_tasks:
            overall_status = 'pending'
            
        return {
            'tenant_id': tenant_id,
            'status': overall_status,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'in_progress_tasks': in_progress_tasks,
            'progress_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'resource_type': task.resource_type.value,
                    'action': task.action,
                    'status': task.status.value,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count
                }
                for task in tasks
            ]
        }
        
    def get_organization_resources(self, tenant_id: str) -> Optional[OrganizationResources]:
        """Get provisioned resources for an organization"""
        return self.organization_resources.get(tenant_id)
        
    async def update_resources(self,
                             tenant_id: str,
                             updates: Dict[str, Any]) -> bool:
        """
        Update organization resources
        
        Args:
            tenant_id: Tenant to update
            updates: Resource updates to apply
            
        Returns:
            Success status
        """
        
        if tenant_id not in self.organization_resources:
            logger.warning(f"No resources found for tenant {tenant_id}")
            return False
            
        resources = self.organization_resources[tenant_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(resources, key):
                setattr(resources, key, value)
                logger.info(f"Updated {key} for tenant {tenant_id}")
                
        return True
        
    def estimate_provisioning_time(self, template: str) -> Dict[str, Any]:
        """Estimate provisioning time for a template"""
        
        estimates = {
            'starter': {
                'minimum_minutes': 2,
                'typical_minutes': 3,
                'maximum_minutes': 5
            },
            'professional': {
                'minimum_minutes': 5,
                'typical_minutes': 8,
                'maximum_minutes': 12
            },
            'enterprise': {
                'minimum_minutes': 10,
                'typical_minutes': 15,
                'maximum_minutes': 25
            }
        }
        
        return estimates.get(template, {
            'minimum_minutes': 5,
            'typical_minutes': 10,
            'maximum_minutes': 20
        })