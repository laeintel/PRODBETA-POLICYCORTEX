"""
ML System Integration Tests
Tests the complete ML pipeline end-to-end
"""

import asyncio
import json
import time
import numpy as np
import requests
import websockets
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Test configuration
API_BASE = "http://localhost:8080/api/v1"
WS_URL = "ws://localhost:8765"
TEST_TIMEOUT = 30

class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with color"""
    if passed:
        print(f"{Colors.GREEN}✓{Colors.ENDC} {name}")
        if details:
            print(f"  {Colors.BLUE}{details}{Colors.ENDC}")
    else:
        print(f"{Colors.RED}✗{Colors.ENDC} {name}")
        if details:
            print(f"  {Colors.YELLOW}{details}{Colors.ENDC}")

class MLIntegrationTests:
    """Integration tests for ML system"""
    
    def __init__(self):
        self.results = {"passed": 0, "failed": 0}
        self.api_base = API_BASE
        self.ws_url = WS_URL
    
    def test_api_health(self) -> bool:
        """Test if ML API is healthy"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            passed = response.status_code == 200
            print_test("API Health Check", passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            print_test("API Health Check", False, str(e))
            return False
    
    def test_create_prediction(self) -> Dict[str, Any]:
        """Test creating a new prediction"""
        try:
            payload = {
                "resource_id": f"test-resource-{int(time.time())}",
                "tenant_id": "test-tenant",
                "configuration": {
                    "encryption": {"enabled": False},
                    "public_access": True,
                    "mfa": {"enabled": False}
                }
            }
            
            response = requests.post(
                f"{self.api_base}/predictions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = [
                    "prediction_id", "resource_id", "violation_probability",
                    "confidence_score", "risk_level", "recommendations"
                ]
                
                all_fields_present = all(field in data for field in required_fields)
                
                if all_fields_present:
                    print_test(
                        "Create Prediction",
                        True,
                        f"ID: {data['prediction_id']}, "
                        f"Risk: {data['risk_level']}, "
                        f"Prob: {data['violation_probability']:.2%}"
                    )
                    return data
                else:
                    print_test("Create Prediction", False, "Missing required fields")
                    return {}
            else:
                print_test("Create Prediction", False, f"Status: {response.status_code}")
                return {}
                
        except Exception as e:
            print_test("Create Prediction", False, str(e))
            return {}
    
    def test_batch_predictions(self) -> bool:
        """Test batch prediction performance"""
        try:
            batch_size = 10
            predictions = []
            start_time = time.time()
            
            for i in range(batch_size):
                payload = {
                    "resource_id": f"batch-test-{i}",
                    "tenant_id": "test-tenant",
                    "configuration": {
                        "encryption": {"enabled": i % 2 == 0},
                        "public_access": i % 3 == 0
                    }
                }
                
                response = requests.post(
                    f"{self.api_base}/predictions",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    predictions.append(response.json())
            
            elapsed_time = (time.time() - start_time) * 1000
            avg_time = elapsed_time / batch_size
            
            # Check if average time is under 100ms (patent requirement)
            passed = len(predictions) == batch_size and avg_time < 100
            
            print_test(
                "Batch Predictions",
                passed,
                f"{len(predictions)}/{batch_size} successful, "
                f"Avg time: {avg_time:.2f}ms"
            )
            
            return passed
            
        except Exception as e:
            print_test("Batch Predictions", False, str(e))
            return False
    
    def test_get_risk_assessment(self, resource_id: str) -> bool:
        """Test risk assessment endpoint"""
        try:
            response = requests.get(
                f"{self.api_base}/predictions/risk-score/{resource_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate risk assessment structure
                has_score = "risk_score" in data
                has_factors = "impact_factors" in data
                has_recommendations = "recommendations" in data
                
                passed = has_score and has_factors and has_recommendations
                
                if passed:
                    print_test(
                        "Risk Assessment",
                        True,
                        f"Score: {data['risk_score']:.2f}, "
                        f"Level: {data['risk_level']}"
                    )
                else:
                    print_test("Risk Assessment", False, "Missing required fields")
                
                return passed
            else:
                print_test("Risk Assessment", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            print_test("Risk Assessment", False, str(e))
            return False
    
    def test_model_metrics(self) -> bool:
        """Test model metrics endpoint"""
        try:
            response = requests.get(f"{self.api_base}/ml/metrics", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                
                # Check patent requirements
                accuracy = metrics.get("accuracy", 0)
                fpr = metrics.get("false_positive_rate", 1)
                latency_p95 = metrics.get("inference_time_p95_ms", 1000)
                meets_requirements = metrics.get("meets_patent_requirements", False)
                
                print_test(
                    "Model Metrics",
                    meets_requirements,
                    f"Accuracy: {accuracy:.3f} (req: 0.992), "
                    f"FPR: {fpr:.3f} (req: <0.02), "
                    f"P95: {latency_p95:.1f}ms (req: <100ms)"
                )
                
                return meets_requirements
            else:
                print_test("Model Metrics", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            print_test("Model Metrics", False, str(e))
            return False
    
    def test_drift_detection(self) -> bool:
        """Test drift detection endpoint"""
        try:
            payload = {
                "resource_id": "drift-test-001",
                "configuration": {
                    "encryption": {"enabled": False},
                    "public_access": True,
                    "firewall_rules": []
                }
            }
            
            response = requests.post(
                f"{self.api_base}/configurations/drift-analysis",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                has_detection = "drift_detected" in data
                has_score = "drift_score" in data
                has_velocity = "drift_velocity" in data
                
                passed = has_detection and has_score and has_velocity
                
                print_test(
                    "Drift Detection",
                    passed,
                    f"Detected: {data.get('drift_detected', False)}, "
                    f"Score: {data.get('drift_score', 0):.2f}"
                )
                
                return passed
            else:
                print_test("Drift Detection", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            print_test("Drift Detection", False, str(e))
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and real-time updates"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send authentication
                auth_message = json.dumps({
                    "tenant_id": "test-tenant",
                    "auth_token": "test-token"
                })
                await websocket.send(auth_message)
                
                # Wait for connection confirmation
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5
                )
                
                data = json.loads(response)
                connected = data.get("type") in ["connected", "error"]
                
                if connected and data.get("type") == "connected":
                    # Subscribe to predictions
                    subscribe_message = json.dumps({
                        "type": "subscribe",
                        "resource_ids": ["all"],
                        "prediction_types": ["all"]
                    })
                    await websocket.send(subscribe_message)
                    
                    # Wait for subscription confirmation
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=5
                    )
                    
                    sub_data = json.loads(response)
                    subscribed = sub_data.get("type") == "subscribed"
                    
                    print_test(
                        "WebSocket Connection",
                        subscribed,
                        "Connected and subscribed to updates"
                    )
                    
                    return subscribed
                else:
                    print_test("WebSocket Connection", False, "Authentication failed")
                    return False
                    
        except asyncio.TimeoutError:
            print_test("WebSocket Connection", False, "Connection timeout")
            return False
        except Exception as e:
            print_test("WebSocket Connection", False, str(e))
            return False
    
    def test_latency_requirements(self) -> bool:
        """Test that inference latency meets patent requirements"""
        try:
            latencies = []
            
            for i in range(20):
                start_time = time.time()
                
                payload = {
                    "resource_id": f"latency-test-{i}",
                    "tenant_id": "test-tenant",
                    "configuration": {"test": True}
                }
                
                response = requests.post(
                    f"{self.api_base}/predictions",
                    json=payload,
                    timeout=10
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    # Use reported inference time if available
                    if "inference_time_ms" in data:
                        latencies.append(data["inference_time_ms"])
                    else:
                        latencies.append(latency)
            
            if latencies:
                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)
                
                # Patent requirement: <100ms
                passed = p95 < 100
                
                print_test(
                    "Latency Requirements",
                    passed,
                    f"P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms"
                )
                
                return passed
            else:
                print_test("Latency Requirements", False, "No successful predictions")
                return False
                
        except Exception as e:
            print_test("Latency Requirements", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print(f"\n{Colors.BOLD}ML System Integration Tests{Colors.ENDC}")
        print("=" * 50)
        
        # Test API health
        if self.test_api_health():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Test prediction creation
        prediction = self.test_create_prediction()
        if prediction:
            self.results["passed"] += 1
            
            # Test risk assessment with created resource
            if self.test_get_risk_assessment(prediction["resource_id"]):
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
        else:
            self.results["failed"] += 2
        
        # Test batch predictions
        if self.test_batch_predictions():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Test model metrics
        if self.test_model_metrics():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Test drift detection
        if self.test_drift_detection():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Test WebSocket connection
        if await self.test_websocket_connection():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Test latency requirements
        if self.test_latency_requirements():
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"{Colors.BOLD}Test Summary{Colors.ENDC}")
        print(f"{Colors.GREEN}Passed: {self.results['passed']}{Colors.ENDC}")
        print(f"{Colors.RED}Failed: {self.results['failed']}{Colors.ENDC}")
        
        if self.results["failed"] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All integration tests passed!{Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.YELLOW}⚠️  Some tests failed. Check the output above.{Colors.ENDC}")
            return False


def main():
    """Main test runner"""
    tests = MLIntegrationTests()
    
    # Check if services are running
    print(f"{Colors.BOLD}Checking ML services...{Colors.ENDC}")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        print(f"{Colors.GREEN}✓{Colors.ENDC} ML API is running")
    except:
        print(f"{Colors.RED}✗{Colors.ENDC} ML API is not running")
        print(f"{Colors.YELLOW}Start the API server first:{Colors.ENDC}")
        print("  python -m backend.services.ml_models.prediction_serving")
        return False
    
    # Run tests
    success = asyncio.run(tests.run_all_tests())
    
    if success:
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("1. Deploy to Kubernetes: kubectl apply -f infrastructure/k8s/ml-deployment.yaml")
        print("2. Run performance tests: python tests/ml/test_performance_validation.py")
        print("3. Start frontend: cd frontend && npm run dev")
        print("4. Access ML dashboard: http://localhost:3000/tactical/ml")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)