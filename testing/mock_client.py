"""
Stateful mock HTTP client for integration tests
Maintains state across requests to simulate real API behavior
"""
import uuid
from typing import Dict, Any, Optional
from unittest.mock import Mock

class StatefulMockClient:
    """Mock HTTP client that maintains state across requests"""
    
    def __init__(self):
        self.state = {
            'alerts': {},
            'policies': {},
            'sessions': {},
            'tenants': {},
            'connectors': {},
            'pipelines': {},
            'datasets': {},
            'models': {},
            'jobs': {}
        }
        
        # Track alert status changes
        self.alert_status_changes = {}
        
        # Track tenant-specific policies
        self.tenant_policies = {}
    
    async def post(self, url: str, **kwargs) -> Mock:
        """Handle POST requests with state management"""
        mock_response = Mock()
        json_data = kwargs.get('json', {})
        
        if 'onboarding/start' in url:
            session_id = f"test-session-{uuid.uuid4().hex[:8]}"
            tenant_id = "test-tenant-123"
            
            self.state['sessions'][session_id] = {
                'tenant_id': tenant_id,
                'status': 'active'
            }
            
            mock_response.status_code = 200
            mock_response.json = lambda: {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "subscription_id": "test-subscription-id",
                "status": "success",
                "message": "Onboarding session created"
            }
            
        elif 'policies' in url and 'batch' not in url:
            # Extract tenant_id from URL params
            tenant_id = self._extract_tenant_id(url)
            policy_id = f"test-policy-{uuid.uuid4().hex[:8]}"
            
            # Store policy with tenant association
            if tenant_id not in self.tenant_policies:
                self.tenant_policies[tenant_id] = []
            
            policy_data = {
                "policy_id": policy_id,
                "id": policy_id,
                "name": json_data.get('name', 'Test Policy'),
                "description": json_data.get('description', ''),
                "category": json_data.get('category', 'security'),
                "rules": json_data.get('rules', [{"condition": "valid", "requirement": "valid"}]),
                "tenant_id": tenant_id
            }
            
            self.state['policies'][policy_id] = policy_data
            self.tenant_policies[tenant_id].append(policy_id)
            
            mock_response.status_code = 201
            mock_response.json = lambda: policy_data
            
        elif 'alerts/create' in url or ('alerts' in url and 'acknowledge' not in url and 'resolve' not in url):
            alert_id = f"test-alert-{uuid.uuid4().hex[:8]}"
            
            self.state['alerts'][alert_id] = {
                "alert_id": alert_id,
                "status": "open",
                "title": json_data.get('title', 'Test Alert'),
                "severity": json_data.get('severity', 'high')
            }
            
            mock_response.status_code = 200
            mock_response.json = lambda: {
                "data": {
                    "alert_id": alert_id,
                    "severity": "high",
                    "priority": "urgent"
                },
                "status": "success"
            }
            
        elif '/acknowledge' in url:
            alert_id = self._extract_alert_id(url)
            if alert_id in self.state['alerts']:
                self.state['alerts'][alert_id]['status'] = 'acknowledged'
            
            mock_response.status_code = 200
            mock_response.json = lambda: {"status": "acknowledged"}
            
        elif '/resolve' in url:
            alert_id = self._extract_alert_id(url)
            if alert_id in self.state['alerts']:
                self.state['alerts'][alert_id]['status'] = 'resolved'
            
            mock_response.status_code = 200
            mock_response.json = lambda: {"status": "resolved"}
            
        elif 'connectors' in url or 'transformations' in url or 'pipelines' in url:
            entity_type = 'connector' if 'connectors' in url else 'transformation' if 'transformations' in url else 'pipeline'
            entity_id = f"test-{entity_type}-{uuid.uuid4().hex[:8]}"
            
            mock_response.status_code = 201
            mock_response.json = lambda: {
                f"{entity_type}_id": entity_id,
                "status": "created"
            }
            
        elif 'training-data' in url or 'models/create' in url or 'models/train' in url:
            if 'training-data' in url:
                dataset_id = f"test-dataset-{uuid.uuid4().hex[:8]}"
                mock_response.status_code = 201
                mock_response.json = lambda: {"dataset_id": dataset_id}
            elif 'models/create' in url:
                model_id = f"test-model-{uuid.uuid4().hex[:8]}"
                mock_response.status_code = 201
                mock_response.json = lambda: {"model_id": model_id}
            else:
                job_id = f"test-job-{uuid.uuid4().hex[:8]}"
                mock_response.status_code = 200
                mock_response.json = lambda: {"job_id": job_id}
                
        else:
            # Default response
            mock_response.status_code = 200
            mock_response.json = lambda: {
                "status": "success",
                "message": "Operation completed"
            }
        
        return mock_response
    
    async def get(self, url: str, **kwargs) -> Mock:
        """Handle GET requests with state management"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        if '/alerts/' in url and not url.endswith('/alerts'):
            alert_id = self._extract_alert_id(url)
            alert_data = self.state['alerts'].get(alert_id, {
                "alert_id": alert_id,
                "status": "open",
                "title": "Test Alert"
            })
            
            mock_response.json = lambda: {
                "data": {
                    "alert": alert_data
                }
            }
            
        elif 'policies' in url and '/policies/' in url:
            # Get specific policy
            policy_id = self._extract_policy_id(url)
            tenant_id = self._extract_tenant_id(url)
            
            # Check if policy belongs to tenant
            if tenant_id and tenant_id in self.tenant_policies:
                if policy_id in self.tenant_policies[tenant_id]:
                    policy_data = self.state['policies'].get(policy_id)
                    mock_response.json = lambda: policy_data if policy_data else {"error": "not_found"}
                    if not policy_data:
                        mock_response.status_code = 404
                else:
                    mock_response.status_code = 404
                    mock_response.json = lambda: {"error": "not_found"}
            else:
                policy_data = self.state['policies'].get(policy_id, {
                    "policy_id": policy_id,
                    "rules": [{"condition": "valid", "requirement": "valid"}]
                })
                mock_response.json = lambda: policy_data
                
        elif 'policies' in url:
            # List policies for tenant
            tenant_id = self._extract_tenant_id(url)
            
            if tenant_id and tenant_id in self.tenant_policies:
                # Return only policies for this tenant
                tenant_policy_ids = self.tenant_policies[tenant_id]
                policies = [
                    self.state['policies'][pid] 
                    for pid in tenant_policy_ids 
                    if pid in self.state['policies']
                ]
            else:
                # Return generic policies for other tests
                policies = [{"policy_id": f"test-policy-{i}", "id": f"policy-{i}", "name": f"Policy {i}"} for i in range(20)]
            
            mock_response.json = lambda: {
                "policies": policies,
                "total": len(policies),
                "status": "success"
            }
            
        elif 'training' in url and '/status' in url:
            mock_response.json = lambda: {"status": "completed"}
            
        elif 'models' in url and '/metrics' in url:
            mock_response.json = lambda: {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88
            }
            
        elif 'pipelines' in url and '/runs/' in url:
            mock_response.json = lambda: {"status": "completed"}
            
        elif 'quality/metrics' in url:
            mock_response.json = lambda: {
                "completeness_score": 0.95,
                "accuracy_score": 0.92
            }
            
        else:
            # Default response
            mock_response.json = lambda: {
                "data": [],
                "total": 0,
                "status": "success"
            }
        
        return mock_response
    
    async def put(self, url: str, **kwargs) -> Mock:
        """Handle PUT requests with state management"""
        mock_response = Mock()
        json_data = kwargs.get('json', {})
        
        if 'policies' in url:
            policy_id = self._extract_policy_id(url)
            
            # Check for invalid update (empty rules)
            if isinstance(json_data, dict) and json_data.get('rules') == []:
                mock_response.status_code = 400
                mock_response.json = lambda: {
                    "error": "validation_error",
                    "message": "Rules cannot be empty"
                }
            else:
                # Update policy in state
                if policy_id in self.state['policies']:
                    self.state['policies'][policy_id].update(json_data)
                
                mock_response.status_code = 200
                mock_response.json = lambda: {
                    "status": "success",
                    "message": "Updated successfully"
                }
        else:
            mock_response.status_code = 200
            mock_response.json = lambda: {"status": "success"}
        
        return mock_response
    
    async def delete(self, url: str, **kwargs) -> Mock:
        """Handle DELETE requests"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.json = lambda: {}
        return mock_response
    
    def _extract_tenant_id(self, url: str) -> Optional[str]:
        """Extract tenant_id from URL parameters"""
        if 'tenant_id=' in url:
            parts = url.split('tenant_id=')
            if len(parts) > 1:
                return parts[1].split('&')[0]
        return None
    
    def _extract_alert_id(self, url: str) -> str:
        """Extract alert ID from URL"""
        parts = url.split('/alerts/')
        if len(parts) > 1:
            return parts[1].split('/')[0].split('?')[0]
        return "test-alert-123"
    
    def _extract_policy_id(self, url: str) -> str:
        """Extract policy ID from URL"""
        parts = url.split('/policies/')
        if len(parts) > 1:
            return parts[1].split('/')[0].split('?')[0]
        return "test-policy-123"