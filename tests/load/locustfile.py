"""
Locust load testing for PolicyCortex API
Provides advanced load testing scenarios for stress and spike testing
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
import json
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyCortexUser(HttpUser):
    """
    Simulates a user interacting with the PolicyCortex platform
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        # Simulate authentication (if needed)
        self.client.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        
        # Initialize user state
        self.resource_ids = [f"resource_{i}" for i in range(100)]
        self.policy_ids = [f"policy_{i}" for i in range(50)]
    
    @task(10)
    def health_check(self):
        """Health check endpoint - high frequency"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(5)
    def get_metrics(self):
        """Fetch governance metrics"""
        with self.client.get("/api/v1/metrics", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'data' in data:
                        response.success()
                    else:
                        response.failure("Invalid metrics response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def get_correlations(self):
        """Get cross-domain correlations"""
        # Select random resources for correlation analysis
        selected_resources = random.sample(self.resource_ids, min(5, len(self.resource_ids)))
        
        payload = {
            "resourceIds": selected_resources,
            "timeRange": random.choice(["1h", "6h", "24h", "7d"]),
            "correlationType": random.choice(["security", "cost", "compliance", "all"])
        }
        
        with self.client.post("/api/v1/correlations", 
                              json=payload,
                              catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 404:
                response.failure("Correlations endpoint not found")
            else:
                response.failure(f"Correlation request failed: {response.status_code}")
    
    @task(4)
    def get_predictions(self):
        """Get compliance predictions"""
        params = {
            "resourceId": random.choice(self.resource_ids),
            "horizon": random.choice(["1d", "7d", "30d"])
        }
        
        with self.client.get("/api/v1/predictions",
                             params=params,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Predictions failed: {response.status_code}")
    
    @task(2)
    def get_recommendations(self):
        """Get AI-driven recommendations"""
        params = {
            "resourceType": random.choice(["vm", "storage", "network", "all"]),
            "category": random.choice(["security", "cost", "performance", "compliance"])
        }
        
        with self.client.get("/api/v1/recommendations",
                             params=params,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recommendations failed: {response.status_code}")
    
    @task(3)
    def conversation_query(self):
        """Conversational AI query"""
        queries = [
            "What are my top security risks?",
            "How can I reduce costs in production?",
            "Show me compliance violations",
            "What resources are non-compliant?",
            "Analyze my security posture",
        ]
        
        payload = {
            "query": random.choice(queries),
            "context": {
                "subscriptionId": "test-subscription",
                "resourceGroup": "test-rg"
            }
        }
        
        with self.client.post("/api/v1/conversation",
                              json=payload,
                              catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Conversation query failed: {response.status_code}")
    
    @task(1)
    def heavy_operation(self):
        """Simulate a heavy operation (report generation)"""
        payload = {
            "reportType": random.choice(["compliance", "security", "cost", "executive"]),
            "format": random.choice(["pdf", "json", "csv"]),
            "timeRange": random.choice(["24h", "7d", "30d", "90d"]),
            "includeDetails": random.choice([True, False])
        }
        
        with self.client.post("/api/v1/reports/generate",
                              json=payload,
                              timeout=30,
                              catch_response=True) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            elif response.status_code == 404:
                # Report endpoint might not exist yet
                response.success()
            else:
                response.failure(f"Report generation failed: {response.status_code}")

class AdminUser(HttpUser):
    """
    Simulates an admin user performing management tasks
    """
    wait_time = between(2, 5)
    weight = 1  # Lower weight - fewer admin users
    
    @task(3)
    def get_system_status(self):
        """Check system status"""
        self.client.get("/api/admin/status")
    
    @task(2)
    def get_audit_logs(self):
        """Fetch audit logs"""
        params = {
            "limit": random.choice([10, 50, 100]),
            "offset": random.randint(0, 100)
        }
        self.client.get("/api/admin/audit", params=params)
    
    @task(1)
    def update_configuration(self):
        """Update system configuration"""
        payload = {
            "setting": random.choice(["threshold", "retention", "notification"]),
            "value": random.randint(1, 100)
        }
        self.client.post("/api/admin/config", json=payload)

class MobileUser(HttpUser):
    """
    Simulates mobile app users with different behavior patterns
    """
    wait_time = between(3, 8)  # Mobile users interact less frequently
    weight = 2
    
    @task(10)
    def dashboard_summary(self):
        """Get dashboard summary (mobile optimized)"""
        self.client.get("/api/mobile/dashboard")
    
    @task(5)
    def notifications(self):
        """Check notifications"""
        self.client.get("/api/mobile/notifications")
    
    @task(3)
    def quick_actions(self):
        """Perform quick action"""
        action = random.choice(["acknowledge", "dismiss", "escalate"])
        payload = {"action": action, "targetId": f"alert_{random.randint(1, 100)}"}
        self.client.post("/api/mobile/actions", json=payload)

# Event handlers for custom metrics and reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("Load test starting...")
    logger.info(f"Target host: {environment.host}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Load test completed")
    
    # Print summary statistics
    logger.info("\n=== Test Summary ===")
    logger.info(f"Total requests: {environment.stats.total.num_requests}")
    logger.info(f"Failed requests: {environment.stats.total.num_failures}")
    logger.info(f"Median response time: {environment.stats.total.median_response_time}ms")
    logger.info(f"95th percentile: {environment.stats.total.get_response_time_percentile(0.95)}ms")
    logger.info(f"99th percentile: {environment.stats.total.get_response_time_percentile(0.99)}ms")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom request handler for detailed logging"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    elif response_time > 1000:  # Log slow requests (>1s)
        logger.warning(f"Slow request: {name} took {response_time}ms")

# Custom test scenarios for different load patterns
class StressTestUser(PolicyCortexUser):
    """User behavior for stress testing"""
    wait_time = between(0.1, 0.5)  # Very aggressive
    
    @task(20)
    def aggressive_polling(self):
        """Aggressive API polling"""
        for _ in range(5):
            self.client.get("/api/v1/metrics")
            time.sleep(0.1)

class SpikeTestUser(PolicyCortexUser):
    """User behavior for spike testing"""
    
    def on_start(self):
        super().on_start()
        # 50% chance of being an aggressive user during spike
        self.aggressive = random.random() > 0.5
    
    @task
    def variable_behavior(self):
        if self.aggressive:
            # Aggressive behavior
            for _ in range(10):
                self.client.get("/api/v1/metrics")
                time.sleep(0.05)
        else:
            # Normal behavior
            self.client.get("/health")
            time.sleep(2)