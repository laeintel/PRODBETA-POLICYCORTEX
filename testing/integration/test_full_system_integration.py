"""
Full System Integration Tests
Tests the complete PolicyCortex system with all services working together
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any, List
import uuid
from datetime import datetime, timedelta

# Test data
SAMPLE_TENANT_ID = "integration-test-tenant"
SAMPLE_USER_ID = "integration-test-user"

class TestFullSystemIntegration:
    """Test complete system integration across all services"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_onboarding_to_compliance_flow(
        self,
        api_client: httpx.AsyncClient,
        sample_user: Dict[str, Any],
        sample_tenant: Dict[str, Any]
    ):
        """Test complete flow from customer onboarding to compliance monitoring"""
        
        # Step 1: Customer Onboarding
        onboarding_request = {
            "user_email": sample_user["email"],
            "company_name": sample_tenant["name"],
            "template": "professional"
        }
        
        response = await api_client.post("/api/v1/onboarding/start", json=onboarding_request)
        assert response.status_code == 200
        
        onboarding_data = response.json()
        session_id = onboarding_data["session_id"]
        tenant_id = onboarding_data["tenant_id"]
        
        # Step 2: Complete onboarding steps
        steps_data = [
            {
                "organization_type": "enterprise",
                "industry": "technology",
                "employee_count": 500,
                "cloud_environments": ["azure"]
            },
            {
                "azure_subscription_id": "test-subscription-id",
                "azure_tenant_id": "test-tenant-id",
                "azure_client_id": "test-client-id",
                "azure_client_secret": "test-secret"
            },
            {
                "selected_features": ["compliance", "analytics", "notifications"]
            }
        ]
        
        for step_data in steps_data:
            response = await api_client.post(
                f"/api/v1/onboarding/{session_id}/step",
                json={"step_data": step_data}
            )
            assert response.status_code == 200
        
        # Step 3: Create subscription
        subscription_request = {
            "plan_id": "professional",
            "billing_cycle": "monthly"
        }
        
        response = await api_client.post(
            f"/api/v1/subscriptions?tenant_id={tenant_id}",
            json=subscription_request
        )
        assert response.status_code == 200
        
        subscription_data = response.json()
        subscription_id = subscription_data["subscription_id"]
        
        # Step 4: Initialize Azure integration
        response = await api_client.post(
            f"/api/v1/azure/initialize?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        # Step 5: Create compliance policy
        policy_request = {
            "name": "Storage HTTPS Policy",
            "description": "Ensure all storage accounts use HTTPS",
            "category": "security",
            "severity": "high",
            "rules": [
                {
                    "condition": "resource.type == 'Microsoft.Storage/storageAccounts'",
                    "requirement": "properties.supportsHttpsTrafficOnly == true",
                    "message": "Storage accounts must use HTTPS only"
                }
            ]
        }
        
        response = await api_client.post(
            f"/api/v1/policies?tenant_id={tenant_id}",
            json=policy_request
        )
        assert response.status_code == 201
        
        policy_data = response.json()
        policy_id = policy_data["policy_id"]
        
        # Step 6: Run compliance analysis
        response = await api_client.post(
            f"/api/v1/azure/compliance/analyze?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        # Step 7: Check compliance results
        response = await api_client.get(
            f"/api/v1/azure/compliance/summary?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        compliance_data = response.json()
        assert "total_resources" in compliance_data
        assert "compliant_resources" in compliance_data
        assert "compliance_percentage" in compliance_data
        
        # Step 8: Generate AI insights
        response = await api_client.post(
            f"/api/v1/ai/insights/generate?tenant_id={tenant_id}",
            json={
                "analysis_type": "compliance_trends",
                "time_range": "7d"
            }
        )
        assert response.status_code == 200
        
        insights_data = response.json()
        assert "insights" in insights_data
        assert len(insights_data["insights"]) > 0
        
        # Step 9: Create notification subscription
        notification_subscription = {
            "user_id": sample_user["user_id"],
            "event_types": ["compliance_violation", "policy_update"],
            "channels": ["email"],
            "preferences": {
                "email": sample_user["email"],
                "frequency": "immediate"
            }
        }
        
        response = await api_client.post(
            "/api/v1/notifications/subscriptions",
            json=notification_subscription
        )
        assert response.status_code == 200
        
        # Step 10: Test conversation interface
        conversation_request = {
            "message": "What is our current compliance status?",
            "context": {
                "user_id": sample_user["user_id"],
                "tenant_id": tenant_id
            }
        }
        
        response = await api_client.post(
            "/api/v1/chat/conversation",
            json=conversation_request
        )
        assert response.status_code == 200
        
        conversation_data = response.json()
        assert "response" in conversation_data
        assert "confidence" in conversation_data
        assert conversation_data["confidence"] > 0.7
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_alert_escalation_flow(
        self,
        api_client: httpx.AsyncClient,
        sample_alert: Dict[str, Any]
    ):
        """Test complete alert creation and escalation flow"""
        
        # Step 1: Create alert
        response = await api_client.post(
            "/api/v1/notifications/alerts/create",
            json={
                "title": sample_alert["title"],
                "description": sample_alert["description"],
                "severity": sample_alert["severity"],
                "source": sample_alert["source"],
                "tenant_id": sample_alert["tenant_id"],
                "metadata": sample_alert["metadata"]
            }
        )
        assert response.status_code == 200
        
        alert_data = response.json()
        alert_id = alert_data["data"]["alert_id"]
        
        # Step 2: Verify alert was created
        response = await api_client.get(f"/api/v1/notifications/alerts/{alert_id}")
        assert response.status_code == 200
        
        alert_details = response.json()
        assert alert_details["data"]["alert"]["status"] == "open"
        
        # Step 3: Acknowledge alert
        response = await api_client.post(
            f"/api/v1/notifications/alerts/{alert_id}/acknowledge",
            json={
                "acknowledged_by": "test@example.com",
                "notes": "Investigating the issue"
            }
        )
        assert response.status_code == 200
        
        # Step 4: Verify acknowledgment
        response = await api_client.get(f"/api/v1/notifications/alerts/{alert_id}")
        assert response.status_code == 200
        
        alert_details = response.json()
        assert alert_details["data"]["alert"]["status"] == "acknowledged"
        
        # Step 5: Resolve alert
        response = await api_client.post(
            f"/api/v1/notifications/alerts/{alert_id}/resolve",
            json={
                "resolved_by": "test@example.com",
                "resolution_notes": "Issue has been resolved"
            }
        )
        assert response.status_code == 200
        
        # Step 6: Verify resolution
        response = await api_client.get(f"/api/v1/notifications/alerts/{alert_id}")
        assert response.status_code == 200
        
        alert_details = response.json()
        assert alert_details["data"]["alert"]["status"] == "resolved"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_processing_pipeline(
        self,
        api_client: httpx.AsyncClient,
        sample_tenant: Dict[str, Any]
    ):
        """Test complete data processing pipeline"""
        
        tenant_id = sample_tenant["tenant_id"]
        
        # Step 1: Create data connector
        connector_request = {
            "name": "Azure Resource Connector",
            "type": "azure_resource_manager",
            "config": {
                "subscription_id": "test-subscription",
                "resource_groups": ["rg-test"],
                "resource_types": ["Microsoft.Storage/storageAccounts"]
            },
            "schedule": "0 */6 * * *"  # Every 6 hours
        }
        
        response = await api_client.post(
            f"/api/v1/data/connectors?tenant_id={tenant_id}",
            json=connector_request
        )
        assert response.status_code == 201
        
        connector_data = response.json()
        connector_id = connector_data["connector_id"]
        
        # Step 2: Create transformation rule
        transformation_request = {
            "name": "Resource Standardization",
            "type": "normalization",
            "config": {
                "rules": [
                    {
                        "field": "location",
                        "operation": "lowercase"
                    },
                    {
                        "field": "tags",
                        "operation": "merge_defaults",
                        "defaults": {
                            "Environment": "Unknown",
                            "Owner": "Unassigned"
                        }
                    }
                ]
            }
        }
        
        response = await api_client.post(
            f"/api/v1/data/transformations?tenant_id={tenant_id}",
            json=transformation_request
        )
        assert response.status_code == 201
        
        transformation_data = response.json()
        transformation_id = transformation_data["transformation_id"]
        
        # Step 3: Create data pipeline
        pipeline_request = {
            "name": "Azure Resource Processing Pipeline",
            "steps": [
                {
                    "type": "extract",
                    "connector_id": connector_id
                },
                {
                    "type": "transform",
                    "transformation_id": transformation_id
                },
                {
                    "type": "validate",
                    "validation_rules": ["required_fields", "data_types"]
                },
                {
                    "type": "load",
                    "destination": "analytics_warehouse"
                }
            ],
            "schedule": "0 */4 * * *"  # Every 4 hours
        }
        
        response = await api_client.post(
            f"/api/v1/data/pipelines?tenant_id={tenant_id}",
            json=pipeline_request
        )
        assert response.status_code == 201
        
        pipeline_data = response.json()
        pipeline_id = pipeline_data["pipeline_id"]
        
        # Step 4: Execute pipeline
        response = await api_client.post(
            f"/api/v1/data/pipelines/{pipeline_id}/execute?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        execution_data = response.json()
        run_id = execution_data["run_id"]
        
        # Step 5: Monitor pipeline execution
        max_attempts = 30
        for _ in range(max_attempts):
            response = await api_client.get(
                f"/api/v1/data/pipelines/{pipeline_id}/runs/{run_id}?tenant_id={tenant_id}"
            )
            assert response.status_code == 200
            
            run_data = response.json()
            status = run_data["status"]
            
            if status in ["completed", "failed"]:
                break
            
            await asyncio.sleep(2)
        
        assert status == "completed"
        
        # Step 6: Verify data quality metrics
        response = await api_client.get(
            f"/api/v1/data/quality/metrics?tenant_id={tenant_id}&pipeline_id={pipeline_id}"
        )
        assert response.status_code == 200
        
        quality_data = response.json()
        assert "completeness_score" in quality_data
        assert "accuracy_score" in quality_data
        assert quality_data["completeness_score"] > 0.8
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ml_model_training_and_inference(
        self,
        api_client: httpx.AsyncClient,
        sample_tenant: Dict[str, Any],
        sample_ml_data: Dict[str, Any]
    ):
        """Test ML model training and inference pipeline"""
        
        tenant_id = sample_tenant["tenant_id"]
        
        # Step 1: Upload training data
        response = await api_client.post(
            f"/api/v1/ai/models/training-data?tenant_id={tenant_id}",
            json={
                "dataset_name": "compliance_prediction_data",
                "features": sample_ml_data["features"],
                "labels": sample_ml_data["labels"],
                "feature_names": sample_ml_data["feature_names"],
                "model_type": sample_ml_data["model_type"]
            }
        )
        assert response.status_code == 201
        
        dataset_data = response.json()
        dataset_id = dataset_data["dataset_id"]
        
        # Step 2: Create model configuration
        model_config = {
            "name": "Compliance Predictor Model",
            "algorithm": "random_forest",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "features": sample_ml_data["feature_names"],
            "target": "compliance_status"
        }
        
        response = await api_client.post(
            f"/api/v1/ai/models/create?tenant_id={tenant_id}",
            json=model_config
        )
        assert response.status_code == 201
        
        model_data = response.json()
        model_id = model_data["model_id"]
        
        # Step 3: Train model
        training_request = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "validation_split": 0.2,
            "cross_validation_folds": 5
        }
        
        response = await api_client.post(
            f"/api/v1/ai/models/train?tenant_id={tenant_id}",
            json=training_request
        )
        assert response.status_code == 200
        
        training_data = response.json()
        training_job_id = training_data["job_id"]
        
        # Step 4: Monitor training progress
        max_attempts = 60  # Training can take longer
        for _ in range(max_attempts):
            response = await api_client.get(
                f"/api/v1/ai/models/training/{training_job_id}/status?tenant_id={tenant_id}"
            )
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            if status in ["completed", "failed"]:
                break
            
            await asyncio.sleep(5)
        
        assert status == "completed"
        
        # Step 5: Get model metrics
        response = await api_client.get(
            f"/api/v1/ai/models/{model_id}/metrics?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert "accuracy" in metrics_data
        assert "precision" in metrics_data
        assert "recall" in metrics_data
        assert metrics_data["accuracy"] > 0.5
        
        # Step 6: Make predictions
        prediction_request = {
            "features": sample_ml_data["features"][0],  # Use first sample
            "feature_names": sample_ml_data["feature_names"]
        }
        
        response = await api_client.post(
            f"/api/v1/ai/models/{model_id}/predict?tenant_id={tenant_id}",
            json=prediction_request
        )
        assert response.status_code == 200
        
        prediction_data = response.json()
        assert "prediction" in prediction_data
        assert "confidence" in prediction_data
        assert 0 <= prediction_data["confidence"] <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_tenant_isolation(
        self,
        api_client: httpx.AsyncClient
    ):
        """Test that data is properly isolated between tenants"""
        
        # Create two separate tenants
        tenant1_id = f"tenant-1-{uuid.uuid4()}"
        tenant2_id = f"tenant-2-{uuid.uuid4()}"
        
        # Step 1: Create policies for each tenant
        policy1_request = {
            "name": "Tenant 1 Policy",
            "description": "Policy for tenant 1",
            "category": "security",
            "rules": [{"condition": "test1", "requirement": "test1"}]
        }
        
        policy2_request = {
            "name": "Tenant 2 Policy", 
            "description": "Policy for tenant 2",
            "category": "compliance",
            "rules": [{"condition": "test2", "requirement": "test2"}]
        }
        
        response1 = await api_client.post(
            f"/api/v1/policies?tenant_id={tenant1_id}",
            json=policy1_request
        )
        assert response1.status_code == 201
        
        response2 = await api_client.post(
            f"/api/v1/policies?tenant_id={tenant2_id}",
            json=policy2_request
        )
        assert response2.status_code == 201
        
        policy1_id = response1.json()["policy_id"]
        policy2_id = response2.json()["policy_id"]
        
        # Step 2: Verify tenant 1 can only see their policies
        response = await api_client.get(f"/api/v1/policies?tenant_id={tenant1_id}")
        assert response.status_code == 200
        
        tenant1_policies = response.json()["policies"]
        policy_ids = [p["policy_id"] for p in tenant1_policies]
        
        assert policy1_id in policy_ids
        assert policy2_id not in policy_ids
        
        # Step 3: Verify tenant 2 can only see their policies
        response = await api_client.get(f"/api/v1/policies?tenant_id={tenant2_id}")
        assert response.status_code == 200
        
        tenant2_policies = response.json()["policies"]
        policy_ids = [p["policy_id"] for p in tenant2_policies]
        
        assert policy2_id in policy_ids
        assert policy1_id not in policy_ids
        
        # Step 4: Verify cross-tenant access is denied
        response = await api_client.get(
            f"/api/v1/policies/{policy1_id}?tenant_id={tenant2_id}"
        )
        assert response.status_code in [403, 404]  # Forbidden or Not Found
        
        response = await api_client.get(
            f"/api/v1/policies/{policy2_id}?tenant_id={tenant1_id}"
        )
        assert response.status_code in [403, 404]  # Forbidden or Not Found
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_system_resilience_under_load(
        self,
        api_client: httpx.AsyncClient,
        sample_tenant: Dict[str, Any]
    ):
        """Test system behavior under concurrent load"""
        
        tenant_id = sample_tenant["tenant_id"]
        
        # Create multiple concurrent requests
        async def create_policy(policy_name: str) -> Dict[str, Any]:
            policy_request = {
                "name": f"{policy_name} Policy",
                "description": f"Test policy {policy_name}",
                "category": "test",
                "rules": [{"condition": f"test_{policy_name}", "requirement": "test"}]
            }
            
            response = await api_client.post(
                f"/api/v1/policies?tenant_id={tenant_id}",
                json=policy_request
            )
            return response.json()
        
        # Execute concurrent policy creations
        policy_names = [f"concurrent_policy_{i}" for i in range(20)]
        
        tasks = [create_policy(name) for name in policy_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify most requests succeeded
        successful_requests = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_requests) >= 15  # Allow for some failures under load
        
        # Verify system is still responsive after load
        response = await api_client.get(f"/api/v1/policies?tenant_id={tenant_id}")
        assert response.status_code == 200
        
        policies = response.json()["policies"]
        created_policy_names = [p["name"] for p in policies]
        
        # Verify policies were created
        for result in successful_requests:
            policy_name = result.get("name", "")
            if policy_name:
                assert policy_name in created_policy_names
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_and_rollback(
        self,
        api_client: httpx.AsyncClient,
        sample_tenant: Dict[str, Any]
    ):
        """Test system error recovery and transaction rollback"""
        
        tenant_id = sample_tenant["tenant_id"]
        
        # Step 1: Create valid policy
        valid_policy = {
            "name": "Valid Policy",
            "description": "Valid policy for rollback test",
            "category": "security",
            "rules": [{"condition": "valid", "requirement": "valid"}]
        }
        
        response = await api_client.post(
            f"/api/v1/policies?tenant_id={tenant_id}",
            json=valid_policy
        )
        assert response.status_code == 201
        
        policy_id = response.json()["policy_id"]
        
        # Step 2: Attempt invalid update that should rollback
        invalid_update = {
            "rules": []  # Invalid: empty rules should cause validation error
        }
        
        response = await api_client.put(
            f"/api/v1/policies/{policy_id}?tenant_id={tenant_id}",
            json=invalid_update
        )
        assert response.status_code == 400  # Bad Request
        
        # Step 3: Verify original policy is unchanged
        response = await api_client.get(
            f"/api/v1/policies/{policy_id}?tenant_id={tenant_id}"
        )
        assert response.status_code == 200
        
        policy_data = response.json()
        assert len(policy_data["rules"]) == 1  # Original rule still exists
        assert policy_data["rules"][0]["condition"] == "valid"
        
        # Step 4: Test partial failure in batch operation
        batch_request = {
            "policies": [
                {
                    "name": "Batch Policy 1",
                    "description": "Valid batch policy",
                    "category": "security",
                    "rules": [{"condition": "batch1", "requirement": "batch1"}]
                },
                {
                    "name": "",  # Invalid: empty name
                    "description": "Invalid batch policy",
                    "category": "security",
                    "rules": [{"condition": "batch2", "requirement": "batch2"}]
                },
                {
                    "name": "Batch Policy 3",
                    "description": "Another valid batch policy",
                    "category": "security",
                    "rules": [{"condition": "batch3", "requirement": "batch3"}]
                }
            ]
        }
        
        response = await api_client.post(
            f"/api/v1/policies/batch?tenant_id={tenant_id}",
            json=batch_request
        )
        
        # System should handle partial failure gracefully
        assert response.status_code in [207, 400]  # Multi-Status or Bad Request
        
        if response.status_code == 207:
            batch_results = response.json()["results"]
            assert len(batch_results) == 3
            
            # First and third should succeed, second should fail
            assert batch_results[0]["status"] == "created"
            assert batch_results[1]["status"] == "error"
            assert batch_results[2]["status"] == "created"