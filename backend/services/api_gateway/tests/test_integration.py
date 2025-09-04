"""
Integration Tests for PolicyCortex API Gateway
Tests API contracts, service integration, and end-to-end workflows
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any
from httpx import AsyncClient
from unittest.mock import Mock, patch
import redis
import psycopg2
from datetime import datetime, timedelta
import os
import socket

# Test configuration
API_BASE_URL = "http://localhost:8000"
RUST_API_URL = "http://localhost:8080"
GRAPHQL_URL = "http://localhost:4000/graphql"

# Helper functions to check service availability
def is_service_running(host, port):
    """Check if a service is running on the given host and port"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

# Check if services are available
POSTGRES_AVAILABLE = is_service_running('localhost', 5432)
REDIS_AVAILABLE = is_service_running('localhost', 6379)
API_AVAILABLE = is_service_running('localhost', 8000)
RUST_AVAILABLE = is_service_running('localhost', 8080)
GRAPHQL_AVAILABLE = is_service_running('localhost', 4000)

# Skip markers
skipif_no_postgres = pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
skipif_no_redis = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
skipif_no_api = pytest.mark.skipif(not API_AVAILABLE, reason="API Gateway not running")
skipif_no_rust = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust Core API not running")
skipif_no_graphql = pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not running")

class TestAPIIntegration:
    """Integration tests for API Gateway"""
    
    @pytest.fixture
    async def client(self):
        """Create async HTTP client"""
        async with AsyncClient(base_url=API_BASE_URL) as client:
            yield client
    
    @pytest.fixture
    def auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers"""
        return {
            "Authorization": "Bearer test-token-123",
            "X-Tenant-ID": "test-tenant",
            "X-Request-ID": "test-request-123"
        }
    
    @pytest.fixture
    @skipif_no_postgres
    def db_connection(self):
        """PostgreSQL database connection"""
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="policycortex",
            user="postgres",
            password="postgres"
        )
        yield conn
        conn.close()
    
    @pytest.fixture
    @skipif_no_redis
    def redis_client(self):
        """Redis client for cache testing"""
        client = redis.Redis(host='localhost', port=6379, db=0)
        yield client
        client.flushdb()  # Clean up after tests
    
    # Health and Status Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_health_check(self, client: AsyncClient):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_service_dependencies(self, client: AsyncClient):
        """Test all service dependencies are accessible"""
        response = await client.get("/api/v1/status/dependencies")
        assert response.status_code == 200
        
        dependencies = response.json()
        assert dependencies["postgres"]["status"] == "connected"
        assert dependencies["redis"]["status"] == "connected"
        assert dependencies["rust_core"]["status"] == "healthy"
        assert dependencies["eventstore"]["status"] == "connected"
    
    # Authentication and Authorization Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_authentication_required(self, client: AsyncClient):
        """Test endpoints require authentication"""
        endpoints = [
            "/api/v1/metrics",
            "/api/v1/resources",
            "/api/v1/policies"
        ]
        
        for endpoint in endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 401
            assert "authentication required" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_authorization_enforcement(self, client: AsyncClient):
        """Test role-based access control"""
        # Admin endpoints
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/tenants",
            "/api/v1/admin/settings"
        ]
        
        # Test with user token (non-admin)
        user_headers = {"Authorization": "Bearer user-token-456"}
        
        for endpoint in admin_endpoints:
            response = await client.get(endpoint, headers=user_headers)
            assert response.status_code == 403
            assert "insufficient permissions" in response.json()["detail"].lower()
    
    # Cross-Service Integration Tests
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_rust
    async def test_rust_api_integration(self, client: AsyncClient, auth_headers):
        """Test integration with Rust core service"""
        # Create a resource through Python API
        resource_data = {
            "name": "test-vm-integration",
            "type": "Microsoft.Compute/virtualMachines",
            "location": "eastus",
            "tags": {"env": "test"}
        }
        
        response = await client.post(
            "/api/v1/resources",
            json=resource_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        resource_id = response.json()["id"]
        
        # Verify resource is accessible through Rust API
        async with AsyncClient(base_url=RUST_API_URL) as rust_client:
            rust_response = await rust_client.get(
                f"/api/v1/resources/{resource_id}",
                headers=auth_headers
            )
            assert rust_response.status_code == 200
            assert rust_response.json()["name"] == resource_data["name"]
    
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_graphql
    async def test_graphql_integration(self, client: AsyncClient, auth_headers):
        """Test GraphQL gateway integration"""
        query = """
        query GetMetrics {
            metrics {
                totalResources
                complianceScore
                activeIncidents
                costTrend
            }
        }
        """
        
        async with AsyncClient(base_url=GRAPHQL_URL) as graphql_client:
            response = await graphql_client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "metrics" in data["data"]
    
    # Database Integration Tests
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_postgres
    async def test_database_transactions(self, client: AsyncClient, auth_headers, db_connection):
        """Test database transaction integrity"""
        # Start a transaction
        response = await client.post(
            "/api/v1/policies",
            json={
                "name": "test-policy-transaction",
                "rules": ["rule1", "rule2"],
                "effect": "deny"
            },
            headers=auth_headers
        )
        assert response.status_code == 201
        policy_id = response.json()["id"]
        
        # Verify in database
        cursor = db_connection.cursor()
        cursor.execute(
            "SELECT name, effect FROM policies WHERE id = %s",
            (policy_id,)
        )
        result = cursor.fetchone()
        assert result[0] == "test-policy-transaction"
        assert result[1] == "deny"
        cursor.close()
        
        # Test rollback on error
        with patch('api.policies.validate_policy', side_effect=Exception("Validation error")):
            response = await client.post(
                "/api/v1/policies",
                json={"name": "should-rollback", "rules": [], "effect": "allow"},
                headers=auth_headers
            )
            assert response.status_code == 500
        
        # Verify rollback occurred
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM policies WHERE name = 'should-rollback'")
        count = cursor.fetchone()[0]
        assert count == 0
        cursor.close()
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_event_sourcing(self, client: AsyncClient, auth_headers):
        """Test event sourcing integration"""
        # Perform an action that generates events
        response = await client.put(
            "/api/v1/resources/test-resource-123/tags",
            json={"environment": "production", "owner": "team-a"},
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # Verify event was stored
        events_response = await client.get(
            "/api/v1/events?resource_id=test-resource-123",
            headers=auth_headers
        )
        assert events_response.status_code == 200
        events = events_response.json()
        
        # Should have a TagsUpdated event
        tag_events = [e for e in events if e["type"] == "TagsUpdated"]
        assert len(tag_events) > 0
        assert tag_events[0]["data"]["tags"]["environment"] == "production"
    
    # Cache Integration Tests
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_redis
    async def test_cache_integration(self, client: AsyncClient, auth_headers, redis_client):
        """Test caching behavior"""
        # First request - should hit database
        start_time = time.time()
        response1 = await client.get("/api/v1/metrics", headers=auth_headers)
        first_request_time = time.time() - start_time
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request - should hit cache
        start_time = time.time()
        response2 = await client.get("/api/v1/metrics", headers=auth_headers)
        second_request_time = time.time() - start_time
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Cache should make second request faster
        assert second_request_time < first_request_time * 0.5
        
        # Data should be identical
        assert data1 == data2
        
        # Verify cache key exists
        cache_key = "metrics:test-tenant"
        assert redis_client.exists(cache_key)
    
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_redis
    async def test_cache_invalidation(self, client: AsyncClient, auth_headers, redis_client):
        """Test cache invalidation on updates"""
        # Get initial metrics
        response1 = await client.get("/api/v1/metrics", headers=auth_headers)
        initial_resources = response1.json()["totalResources"]
        
        # Create a new resource (should invalidate cache)
        await client.post(
            "/api/v1/resources",
            json={"name": "cache-test", "type": "storage"},
            headers=auth_headers
        )
        
        # Get metrics again
        response2 = await client.get("/api/v1/metrics", headers=auth_headers)
        updated_resources = response2.json()["totalResources"]
        
        # Should reflect the change
        assert updated_resources == initial_resources + 1
    
    # Performance Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_api_performance(self, client: AsyncClient, auth_headers):
        """Test API response times meet SLA"""
        endpoints = [
            ("/api/v1/metrics", 500),  # 500ms threshold
            ("/api/v1/resources?limit=10", 300),  # 300ms threshold
            ("/api/v1/policies", 200),  # 200ms threshold
        ]
        
        for endpoint, threshold_ms in endpoints:
            start_time = time.time()
            response = await client.get(endpoint, headers=auth_headers)
            response_time_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert response_time_ms < threshold_ms, f"{endpoint} took {response_time_ms}ms, threshold is {threshold_ms}ms"
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_concurrent_requests(self, client: AsyncClient, auth_headers):
        """Test handling of concurrent requests"""
        async def make_request(index: int):
            response = await client.get(
                f"/api/v1/resources?page={index}",
                headers=auth_headers
            )
            return response.status_code
        
        # Make 50 concurrent requests
        tasks = [make_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(status == 200 for status in results)
    
    # Error Handling Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_error_handling(self, client: AsyncClient, auth_headers):
        """Test graceful error handling"""
        # Invalid resource ID
        response = await client.get(
            "/api/v1/resources/invalid-uuid",
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "invalid resource id" in response.json()["detail"].lower()
        
        # Non-existent resource
        response = await client.get(
            "/api/v1/resources/00000000-0000-0000-0000-000000000000",
            headers=auth_headers
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_circuit_breaker(self, client: AsyncClient, auth_headers):
        """Test circuit breaker pattern"""
        # Simulate downstream service failure
        with patch('api.external.azure_service.get_resources', side_effect=Exception("Service unavailable")):
            # Make multiple requests to trip circuit breaker
            for _ in range(5):
                response = await client.get(
                    "/api/v1/azure/resources",
                    headers=auth_headers
                )
            
            # Circuit should be open
            response = await client.get(
                "/api/v1/azure/resources",
                headers=auth_headers
            )
            assert response.status_code == 503
            assert "circuit breaker open" in response.json()["detail"].lower()
    
    # Data Consistency Tests
    @pytest.mark.asyncio
    @skipif_no_api
    @skipif_no_postgres
    async def test_data_consistency(self, client: AsyncClient, auth_headers, db_connection):
        """Test data consistency across services"""
        # Create resource through API
        resource_data = {
            "name": "consistency-test",
            "type": "storage",
            "tags": {"test": "true"}
        }
        
        create_response = await client.post(
            "/api/v1/resources",
            json=resource_data,
            headers=auth_headers
        )
        resource_id = create_response.json()["id"]
        
        # Verify through different endpoints
        # Direct API
        api_response = await client.get(
            f"/api/v1/resources/{resource_id}",
            headers=auth_headers
        )
        
        # GraphQL
        graphql_query = f"""
        query {{
            resource(id: "{resource_id}") {{
                name
                type
                tags
            }}
        }}
        """
        
        async with AsyncClient(base_url=GRAPHQL_URL) as graphql_client:
            graphql_response = await graphql_client.post(
                "/graphql",
                json={"query": graphql_query},
                headers=auth_headers
            )
        
        # Database
        cursor = db_connection.cursor()
        cursor.execute(
            "SELECT name, type, tags FROM resources WHERE id = %s",
            (resource_id,)
        )
        db_result = cursor.fetchone()
        cursor.close()
        
        # All should have consistent data
        api_data = api_response.json()
        graphql_data = graphql_response.json()["data"]["resource"]
        
        assert api_data["name"] == graphql_data["name"] == db_result[0]
        assert api_data["type"] == graphql_data["type"] == db_result[1]
        assert api_data["tags"] == graphql_data["tags"] == db_result[2]
    
    # Pagination and Filtering Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_pagination(self, client: AsyncClient, auth_headers):
        """Test pagination functionality"""
        # Get first page
        page1 = await client.get(
            "/api/v1/resources?page=1&limit=10",
            headers=auth_headers
        )
        assert page1.status_code == 200
        data1 = page1.json()
        
        assert "items" in data1
        assert "total" in data1
        assert "page" in data1
        assert "pages" in data1
        assert data1["page"] == 1
        assert len(data1["items"]) <= 10
        
        # Get second page
        if data1["pages"] > 1:
            page2 = await client.get(
                "/api/v1/resources?page=2&limit=10",
                headers=auth_headers
            )
            assert page2.status_code == 200
            data2 = page2.json()
            
            # Items should be different
            ids1 = {item["id"] for item in data1["items"]}
            ids2 = {item["id"] for item in data2["items"]}
            assert ids1.isdisjoint(ids2)
    
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_filtering(self, client: AsyncClient, auth_headers):
        """Test filtering functionality"""
        # Filter by type
        response = await client.get(
            "/api/v1/resources?type=compute",
            headers=auth_headers
        )
        assert response.status_code == 200
        resources = response.json()["items"]
        
        # All resources should be compute type
        assert all(r["type"].lower().contains("compute") for r in resources)
        
        # Filter by tags
        response = await client.get(
            "/api/v1/resources?tag=environment:production",
            headers=auth_headers
        )
        assert response.status_code == 200
        resources = response.json()["items"]
        
        # All resources should have production tag
        assert all(r.get("tags", {}).get("environment") == "production" for r in resources)


class TestPatentAPIs:
    """Integration tests for patent-specific APIs"""
    
    @pytest.fixture
    async def client(self):
        async with AsyncClient(base_url=API_BASE_URL) as client:
            yield client
    
    @pytest.fixture
    def auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer test-token-123",
            "X-Tenant-ID": "test-tenant"
        }
    
    # Patent #1: Cross-Domain Correlation Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_correlation_api_performance(self, client: AsyncClient, auth_headers):
        """Test correlation API meets <100ms requirement"""
        start_time = time.time()
        
        response = await client.post(
            "/api/v1/correlations",
            json={
                "domains": ["security", "compliance", "cost"],
                "timeRange": "24h"
            },
            headers=auth_headers
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 100, f"Correlation API took {response_time_ms}ms, requirement is <100ms"
        
        data = response.json()
        assert "correlations" in data
        assert "confidence" in data
        assert data["confidence"] >= 0.8
    
    # Patent #2: Conversational AI Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_conversation_api_accuracy(self, client: AsyncClient, auth_headers):
        """Test conversation API intent classification accuracy"""
        test_cases = [
            ("Show me non-compliant resources", "compliance_query"),
            ("Create a backup policy", "policy_creation"),
            ("What's my cloud spend?", "cost_query"),
            ("List security vulnerabilities", "security_query"),
            ("Schedule a maintenance window", "maintenance_request")
        ]
        
        correct_classifications = 0
        
        for message, expected_intent in test_cases:
            response = await client.post(
                "/api/v1/conversation",
                json={"message": message},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            if data["intent"] == expected_intent:
                correct_classifications += 1
        
        accuracy = (correct_classifications / len(test_cases)) * 100
        assert accuracy >= 95, f"Intent classification accuracy is {accuracy}%, requirement is >=95%"
    
    # Patent #3: Unified Platform Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_unified_metrics_performance(self, client: AsyncClient, auth_headers):
        """Test unified platform API meets <500ms requirement"""
        start_time = time.time()
        
        response = await client.get(
            "/api/v1/metrics/unified",
            headers=auth_headers
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Unified metrics API took {response_time_ms}ms, requirement is <500ms"
        
        data = response.json()
        # Should include metrics from all domains
        required_domains = ["security", "compliance", "cost", "operations", "performance"]
        for domain in required_domains:
            assert domain in data
    
    # Patent #4: Predictive Compliance Tests
    @pytest.mark.asyncio
    @skipif_no_api
    async def test_prediction_api_performance(self, client: AsyncClient, auth_headers):
        """Test prediction API meets <100ms requirement"""
        start_time = time.time()
        
        response = await client.get(
            "/api/v1/predictions",
            headers=auth_headers
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 100, f"Prediction API took {response_time_ms}ms, requirement is <100ms"
        
        data = response.json()
        assert "predictions" in data
        assert "accuracy" in data
        assert data["accuracy"] >= 99.2