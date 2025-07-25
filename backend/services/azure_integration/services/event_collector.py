"""
Real-time Azure Event Collection Service for PolicyCortex.
Collects governance events from Azure services for AI analysis.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import structlog
from azure.identity.aio import DefaultAzureCredential
try:
    from azure.monitor.query.aio import LogsQueryClient
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    LogsQueryClient = None
    AZURE_MONITOR_AVAILABLE = False

try:
    from azure.eventgrid.aio import EventGridPublisherClient
    from azure.eventgrid import EventGridEvent
    AZURE_EVENTGRID_AVAILABLE = True
except ImportError:
    EventGridPublisherClient = None
    EventGridEvent = None
    AZURE_EVENTGRID_AVAILABLE = False
from azure.core.credentials import AccessToken

logger = structlog.get_logger(__name__)


class AzureEventCollector:
    """
    Collects real-time governance events from Azure services.
    Implements event deduplication and filtering for AI processing.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.credential = None
        self.logs_client = None
        self.event_grid_client = None
        self.event_cache = {}  # For deduplication
        self.subscribers = []  # Event subscribers
        self.collection_tasks = {}
        self.is_running = False
        
        # Bloom filter for deduplication (simplified)
        self.bloom_filter = set()
        self.bloom_filter_size = 100000
        
    async def initialize(self):
        """Initialize the event collector."""
        logger.info("initializing_azure_event_collector")
        
        try:
            # Initialize Azure credentials
            self.credential = DefaultAzureCredential()
            
            # Initialize Azure Monitor Logs client
            self.logs_client = LogsQueryClient(self.credential)
            
            # Initialize Event Grid client for publishing
            if hasattr(self.settings.azure, 'event_grid_endpoint'):
                self.event_grid_client = EventGridPublisherClient(
                    self.settings.azure.event_grid_endpoint,
                    self.credential
                )
            
            logger.info("azure_event_collector_initialized")
            
        except Exception as e:
            logger.error("failed_to_initialize_event_collector", error=str(e))
            raise
    
    async def start_collection(self):
        """Start real-time event collection."""
        if self.is_running:
            logger.warning("event_collection_already_running")
            return
        
        logger.info("starting_event_collection")
        self.is_running = True
        
        # Start collection tasks for different event types
        self.collection_tasks = {
            'policy_events': asyncio.create_task(self._collect_policy_events()),
            'rbac_events': asyncio.create_task(self._collect_rbac_events()),
            'network_events': asyncio.create_task(self._collect_network_events()),
            'cost_events': asyncio.create_task(self._collect_cost_events()),
            'resource_events': asyncio.create_task(self._collect_resource_events())
        }
        
        logger.info("event_collection_started", tasks=list(self.collection_tasks.keys()))
    
    async def stop_collection(self):
        """Stop event collection."""
        logger.info("stopping_event_collection")
        self.is_running = False
        
        # Cancel all collection tasks
        for task_name, task in self.collection_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("collection_task_cancelled", task=task_name)
        
        self.collection_tasks.clear()
        logger.info("event_collection_stopped")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None], event_types: List[str] = None):
        """Subscribe to governance events."""
        subscriber = {
            'callback': callback,
            'event_types': event_types or ['all'],
            'subscribed_at': datetime.utcnow()
        }
        self.subscribers.append(subscriber)
        logger.info("event_subscriber_added", event_types=event_types)
    
    async def _collect_policy_events(self):
        """Collect Azure Policy events."""
        while self.is_running:
            try:
                logger.debug("collecting_policy_events")
                
                # Query Azure Activity Log for policy-related events
                query = """
                AzureActivity
                | where TimeGenerated > ago(5m)
                | where CategoryValue == "Policy" or 
                        OperationNameValue contains "policy" or
                        OperationNameValue contains "Policy"
                | project TimeGenerated, CorrelationId, OperationNameValue, 
                         ActivityStatusValue, Caller, ResourceGroup, 
                         SubscriptionId, Properties
                | order by TimeGenerated desc
                """
                
                events = await self._execute_logs_query(query, "policy_events")
                
                for event in events:
                    processed_event = await self._process_policy_event(event)
                    if processed_event:
                        await self._publish_event(processed_event, "policy")
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("policy_event_collection_failed", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_rbac_events(self):
        """Collect RBAC and access management events."""
        while self.is_running:
            try:
                logger.debug("collecting_rbac_events")
                
                # Query for RBAC-related events
                query = """
                AzureActivity
                | where TimeGenerated > ago(5m)
                | where OperationNameValue contains "role" or
                        OperationNameValue contains "permission" or
                        OperationNameValue contains "access"
                | union (
                    AuditLogs
                    | where TimeGenerated > ago(5m)
                    | where Category == "RoleManagement" or
                            Category == "DirectoryManagement"
                    | project TimeGenerated, CorrelationId, OperationName,
                             Result, InitiatedBy, TargetResources
                )
                | order by TimeGenerated desc
                """
                
                events = await self._execute_logs_query(query, "rbac_events")
                
                for event in events:
                    processed_event = await self._process_rbac_event(event)
                    if processed_event:
                        await self._publish_event(processed_event, "rbac")
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("rbac_event_collection_failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_network_events(self):
        """Collect network security events."""
        while self.is_running:
            try:
                logger.debug("collecting_network_events")
                
                # Query for network-related events
                query = """
                AzureActivity
                | where TimeGenerated > ago(5m)
                | where OperationNameValue contains "network" or
                        OperationNameValue contains "firewall" or
                        OperationNameValue contains "security"
                | union (
                    AzureDiagnostics
                    | where TimeGenerated > ago(5m)
                    | where Category == "NetworkSecurityGroupEvent" or
                            Category == "NetworkSecurityGroupRuleCounter"
                )
                | order by TimeGenerated desc
                """
                
                events = await self._execute_logs_query(query, "network_events")
                
                for event in events:
                    processed_event = await self._process_network_event(event)
                    if processed_event:
                        await self._publish_event(processed_event, "network")
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("network_event_collection_failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_cost_events(self):
        """Collect cost management events."""
        while self.is_running:
            try:
                logger.debug("collecting_cost_events")
                
                # Query for cost-related events
                query = """
                AzureActivity
                | where TimeGenerated > ago(5m)
                | where OperationNameValue contains "budget" or
                        OperationNameValue contains "cost" or
                        OperationNameValue contains "billing"
                | order by TimeGenerated desc
                """
                
                events = await self._execute_logs_query(query, "cost_events")
                
                for event in events:
                    processed_event = await self._process_cost_event(event)
                    if processed_event:
                        await self._publish_event(processed_event, "cost")
                
                await asyncio.sleep(60)  # Cost events less frequent
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cost_event_collection_failed", error=str(e))
                await asyncio.sleep(120)
    
    async def _collect_resource_events(self):
        """Collect general resource management events."""
        while self.is_running:
            try:
                logger.debug("collecting_resource_events")
                
                # Query for resource lifecycle events
                query = """
                AzureActivity
                | where TimeGenerated > ago(5m)
                | where OperationNameValue contains "create" or
                        OperationNameValue contains "delete" or
                        OperationNameValue contains "update" or
                        OperationNameValue contains "modify"
                | where ActivityStatusValue == "Success"
                | order by TimeGenerated desc
                """
                
                events = await self._execute_logs_query(query, "resource_events")
                
                for event in events:
                    processed_event = await self._process_resource_event(event)
                    if processed_event:
                        await self._publish_event(processed_event, "resource")
                
                await asyncio.sleep(45)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("resource_event_collection_failed", error=str(e))
                await asyncio.sleep(90)
    
    async def _execute_logs_query(self, query: str, event_type: str) -> List[Dict[str, Any]]:
        """Execute a logs query and return results."""
        try:
            if not self.logs_client:
                return []
            
            # Use the default workspace (subscription level)
            workspace_id = getattr(self.settings.azure, 'log_analytics_workspace_id', None)
            if not workspace_id:
                logger.warning("no_log_analytics_workspace_configured")
                return []
            
            response = await self.logs_client.query_workspace(
                workspace_id=workspace_id,
                query=query,
                timespan=timedelta(minutes=5)
            )
            
            events = []
            for table in response.tables:
                for row in table.rows:
                    # Convert table row to dictionary
                    event = dict(zip([col.name for col in table.columns], row))
                    events.append(event)
            
            logger.debug("logs_query_executed", 
                        event_type=event_type, 
                        events_found=len(events))
            
            return events
            
        except Exception as e:
            logger.error("logs_query_failed", 
                        event_type=event_type, 
                        error=str(e))
            return []
    
    async def _process_policy_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw policy event."""
        try:
            event_id = self._generate_event_id(raw_event)
            
            # Check for duplicates
            if await self._is_duplicate_event(event_id):
                return None
            
            processed_event = {
                'id': event_id,
                'type': 'policy',
                'timestamp': raw_event.get('TimeGenerated'),
                'correlation_id': raw_event.get('CorrelationId'),
                'operation': raw_event.get('OperationNameValue'),
                'status': raw_event.get('ActivityStatusValue'),
                'caller': raw_event.get('Caller'),
                'resource_group': raw_event.get('ResourceGroup'),
                'subscription_id': raw_event.get('SubscriptionId'),
                'properties': raw_event.get('Properties', {}),
                'domain': 'policy',
                'severity': self._calculate_event_severity(raw_event),
                'metadata': {
                    'collected_at': datetime.utcnow().isoformat(),
                    'source': 'azure_activity_log'
                }
            }
            
            return processed_event
            
        except Exception as e:
            logger.error("policy_event_processing_failed", error=str(e))
            return None
    
    async def _process_rbac_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw RBAC event."""
        try:
            event_id = self._generate_event_id(raw_event)
            
            if await self._is_duplicate_event(event_id):
                return None
            
            processed_event = {
                'id': event_id,
                'type': 'rbac',
                'timestamp': raw_event.get('TimeGenerated'),
                'correlation_id': raw_event.get('CorrelationId'),
                'operation': raw_event.get('OperationName') or raw_event.get('OperationNameValue'),
                'result': raw_event.get('Result') or raw_event.get('ActivityStatusValue'),
                'initiated_by': raw_event.get('InitiatedBy') or raw_event.get('Caller'),
                'target_resources': raw_event.get('TargetResources', []),
                'domain': 'rbac',
                'severity': self._calculate_event_severity(raw_event),
                'metadata': {
                    'collected_at': datetime.utcnow().isoformat(),
                    'source': 'azure_audit_log'
                }
            }
            
            return processed_event
            
        except Exception as e:
            logger.error("rbac_event_processing_failed", error=str(e))
            return None
    
    async def _process_network_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw network event."""
        try:
            event_id = self._generate_event_id(raw_event)
            
            if await self._is_duplicate_event(event_id):
                return None
            
            processed_event = {
                'id': event_id,
                'type': 'network',
                'timestamp': raw_event.get('TimeGenerated'),
                'correlation_id': raw_event.get('CorrelationId'),
                'operation': raw_event.get('OperationNameValue'),
                'category': raw_event.get('Category'),
                'resource_id': raw_event.get('ResourceId'),
                'properties': raw_event.get('Properties', {}),
                'domain': 'network',
                'severity': self._calculate_event_severity(raw_event),
                'metadata': {
                    'collected_at': datetime.utcnow().isoformat(),
                    'source': 'azure_diagnostics'
                }
            }
            
            return processed_event
            
        except Exception as e:
            logger.error("network_event_processing_failed", error=str(e))
            return None
    
    async def _process_cost_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw cost event."""
        try:
            event_id = self._generate_event_id(raw_event)
            
            if await self._is_duplicate_event(event_id):
                return None
            
            processed_event = {
                'id': event_id,
                'type': 'cost',
                'timestamp': raw_event.get('TimeGenerated'),
                'correlation_id': raw_event.get('CorrelationId'),
                'operation': raw_event.get('OperationNameValue'),
                'status': raw_event.get('ActivityStatusValue'),
                'resource_group': raw_event.get('ResourceGroup'),
                'subscription_id': raw_event.get('SubscriptionId'),
                'domain': 'cost',
                'severity': self._calculate_event_severity(raw_event),
                'metadata': {
                    'collected_at': datetime.utcnow().isoformat(),
                    'source': 'azure_activity_log'
                }
            }
            
            return processed_event
            
        except Exception as e:
            logger.error("cost_event_processing_failed", error=str(e))
            return None
    
    async def _process_resource_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw resource event."""
        try:
            event_id = self._generate_event_id(raw_event)
            
            if await self._is_duplicate_event(event_id):
                return None
            
            processed_event = {
                'id': event_id,
                'type': 'resource',
                'timestamp': raw_event.get('TimeGenerated'),
                'correlation_id': raw_event.get('CorrelationId'),
                'operation': raw_event.get('OperationNameValue'),
                'status': raw_event.get('ActivityStatusValue'),
                'caller': raw_event.get('Caller'),
                'resource_group': raw_event.get('ResourceGroup'),
                'resource_id': raw_event.get('ResourceId'),
                'subscription_id': raw_event.get('SubscriptionId'),
                'domain': 'resource',
                'severity': self._calculate_event_severity(raw_event),
                'metadata': {
                    'collected_at': datetime.utcnow().isoformat(),
                    'source': 'azure_activity_log'
                }
            }
            
            return processed_event
            
        except Exception as e:
            logger.error("resource_event_processing_failed", error=str(e))
            return None
    
    def _generate_event_id(self, event: Dict[str, Any]) -> str:
        """Generate a unique event ID for deduplication."""
        # Create hash from key event attributes
        key_attrs = [
            str(event.get('TimeGenerated', '')),
            str(event.get('CorrelationId', '')),
            str(event.get('OperationNameValue', '')),
            str(event.get('Caller', ''))
        ]
        
        event_key = '|'.join(key_attrs)
        import hashlib
        return hashlib.md5(event_key.encode()).hexdigest()
    
    async def _is_duplicate_event(self, event_id: str) -> bool:
        """Check if event is a duplicate using bloom filter."""
        if event_id in self.bloom_filter:
            return True
        
        # Add to bloom filter
        self.bloom_filter.add(event_id)
        
        # Clean bloom filter if it gets too large
        if len(self.bloom_filter) > self.bloom_filter_size:
            # Remove oldest 20% of entries (simplified)
            remove_count = int(self.bloom_filter_size * 0.2)
            items_to_remove = list(self.bloom_filter)[:remove_count]
            for item in items_to_remove:
                self.bloom_filter.discard(item)
        
        return False
    
    def _calculate_event_severity(self, event: Dict[str, Any]) -> str:
        """Calculate event severity based on operation and status."""
        operation = event.get('OperationNameValue', '').lower()
        status = event.get('ActivityStatusValue', '').lower()
        
        # High severity operations
        high_severity_ops = [
            'delete', 'remove', 'disable', 'deny', 'block', 'fail'
        ]
        
        # Medium severity operations
        medium_severity_ops = [
            'create', 'add', 'enable', 'modify', 'update', 'change'
        ]
        
        if any(op in operation for op in high_severity_ops) or status == 'failed':
            return 'high'
        elif any(op in operation for op in medium_severity_ops):
            return 'medium'
        else:
            return 'low'
    
    async def _publish_event(self, event: Dict[str, Any], event_type: str):
        """Publish processed event to subscribers."""
        try:
            # Notify local subscribers
            for subscriber in self.subscribers:
                event_types = subscriber.get('event_types', ['all'])
                if 'all' in event_types or event_type in event_types:
                    try:
                        await subscriber['callback'](event)
                    except Exception as e:
                        logger.error("subscriber_callback_failed", 
                                   event_type=event_type, 
                                   error=str(e))
            
            # Publish to Event Grid if configured
            if self.event_grid_client:
                try:
                    event_grid_event = EventGridEvent(
                        subject=f"governance/{event_type}",
                        event_type=f"PolicyCortex.Governance.{event_type.capitalize()}",
                        data=event,
                        data_version="1.0"
                    )
                    
                    await self.event_grid_client.send([event_grid_event])
                    
                except Exception as e:
                    logger.error("event_grid_publish_failed", 
                               event_type=event_type, 
                               error=str(e))
            
            logger.debug("event_published", 
                        event_type=event_type, 
                        event_id=event.get('id'))
            
        except Exception as e:
            logger.error("event_publish_failed", 
                        event_type=event_type, 
                        error=str(e))
    
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get event collection statistics."""
        return {
            'is_running': self.is_running,
            'active_tasks': len([t for t in self.collection_tasks.values() if not t.done()]),
            'subscribers': len(self.subscribers),
            'bloom_filter_size': len(self.bloom_filter),
            'collection_tasks': list(self.collection_tasks.keys())
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("cleaning_up_event_collector")
        
        await self.stop_collection()
        
        if self.logs_client:
            await self.logs_client.close()
        
        if self.event_grid_client:
            await self.event_grid_client.close()
        
        if self.credential:
            await self.credential.close()
        
        self.subscribers.clear()
        self.bloom_filter.clear()
        
        logger.info("event_collector_cleanup_completed")