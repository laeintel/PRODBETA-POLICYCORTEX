#!/usr/bin/env python3
"""
Comprehensive ML API Endpoint Test Suite for PolicyCortex Patent #4
Tests all ML endpoints with performance validation and response structure verification
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Tuple, List
from datetime import datetime
import statistics

# Configuration
import sys
BASE_URL = "http://localhost:8081" if "--mock" in sys.argv else "http://localhost:8080"
PERFORMANCE_THRESHOLD_MS = 100  # Maximum allowed response time in milliseconds

# Test results tracking
test_results = []
performance_metrics = {}

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_test_start(test_name: str):
    """Print test start message"""
    print(f"{Colors.YELLOW}Testing:{Colors.RESET} {test_name}")

def print_success(message: str, response_time_ms: float = None):
    """Print success message"""
    time_str = f" ({response_time_ms:.2f}ms)" if response_time_ms else ""
    print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {message}{time_str}")

def print_error(message: str):
    """Print error message"""
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {message}")

def measure_request(method: str, url: str, **kwargs) -> Tuple[requests.Response, float]:
    """Execute request and measure response time"""
    start_time = time.perf_counter()
    try:
        response = requests.request(method, url, **kwargs)
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        return response, response_time_ms
    except Exception as e:
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        raise e

def validate_response_structure(response_data: Any, expected_fields: List[str], test_name: str) -> bool:
    """Validate that response contains expected fields"""
    if isinstance(response_data, dict):
        missing_fields = [field for field in expected_fields if field not in response_data]
        if missing_fields:
            print_error(f"Missing fields in response: {missing_fields}")
            return False
        return True
    elif isinstance(response_data, list) and len(response_data) > 0:
        # Check first item if it's a list
        return validate_response_structure(response_data[0], expected_fields, test_name)
    return True

def test_get_all_predictions():
    """Test GET /api/v1/predictions"""
    test_name = "GET /api/v1/predictions - Get all predictions"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/predictions")
        
        # Check response status
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            if isinstance(data, list):
                print_success(f"Response is a list with {len(data)} predictions")
                if len(data) > 0:
                    expected_fields = ['id', 'resource_id', 'prediction_type', 'risk_score', 'confidence', 'timestamp']
                    if validate_response_structure(data, expected_fields, test_name):
                        print_success("Response structure validation passed")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_risk_score():
    """Test GET /api/v1/predictions/risk-score/{resource_id}"""
    test_name = "GET /api/v1/predictions/risk-score/{resource_id}"
    resource_id = "test-resource-1"
    print_test_start(f"{test_name} (resource_id: {resource_id})")
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/predictions/risk-score/{resource_id}")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['resource_id', 'risk_score', 'risk_level', 'contributing_factors', 'recommendations']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Risk score: {data.get('risk_score', 'N/A')}, Level: {data.get('risk_level', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_create_prediction():
    """Test POST /api/v1/predictions"""
    test_name = "POST /api/v1/predictions - Create new prediction"
    print_test_start(test_name)
    
    payload = {
        "resource_id": f"test-resource-{int(time.time())}",
        "resource_type": "VirtualMachine",
        "prediction_type": "compliance_drift",
        "features": {
            "cpu_utilization": 75.5,
            "memory_usage": 82.3,
            "disk_io": 45.2,
            "network_throughput": 120.5,
            "error_rate": 0.02,
            "compliance_score": 85.0
        },
        "metadata": {
            "region": "eastus",
            "environment": "production",
            "tags": ["critical", "monitored"]
        }
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST", 
            f"{BASE_URL}/api/v1/predictions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['id', 'resource_id', 'prediction_type', 'risk_score', 'confidence', 'timestamp']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Created prediction ID: {data.get('id', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_trigger_remediation():
    """Test POST /api/v1/predictions/remediate/{resource_id}"""
    test_name = "POST /api/v1/predictions/remediate/{resource_id}"
    resource_id = "test-resource-1"
    print_test_start(f"{test_name} (resource_id: {resource_id})")
    
    payload = {
        "action_type": "auto_remediate",
        "priority": "high",
        "approval_required": False,
        "remediation_steps": [
            "Update security group rules",
            "Apply compliance patches",
            "Restart service"
        ]
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST",
            f"{BASE_URL}/api/v1/predictions/remediate/{resource_id}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201, 202]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['remediation_id', 'resource_id', 'status', 'initiated_at']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Remediation ID: {data.get('remediation_id', 'N/A')}, Status: {data.get('status', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_ml_metrics():
    """Test GET /api/v1/ml/metrics"""
    test_name = "GET /api/v1/ml/metrics - Get ML model metrics"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/metrics")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['accuracy', 'precision', 'recall', 'f1_score', 'model_version', 'last_trained']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Model accuracy: {data.get('accuracy', 'N/A')}, Version: {data.get('model_version', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_submit_feedback():
    """Test POST /api/v1/ml/feedback"""
    test_name = "POST /api/v1/ml/feedback - Submit feedback"
    print_test_start(test_name)
    
    payload = {
        "prediction_id": f"pred-{int(time.time())}",
        "resource_id": "test-resource-1",
        "feedback_type": "accuracy",
        "rating": 4,
        "correct_label": "compliant",
        "predicted_label": "non_compliant",
        "comments": "False positive - resource was actually compliant",
        "user_id": "test-user-1",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST",
            f"{BASE_URL}/api/v1/ml/feedback",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201, 202]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['feedback_id', 'status', 'processed_at']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Feedback ID: {data.get('feedback_id', 'N/A')}, Status: {data.get('status', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_drift_analysis():
    """Test POST /api/v1/configurations/drift-analysis"""
    test_name = "POST /api/v1/configurations/drift-analysis"
    print_test_start(test_name)
    
    payload = {
        "resource_id": "test-resource-1",
        "current_config": {
            "vm_size": "Standard_D4s_v3",
            "os_disk_size": 128,
            "network_security_group": "nsg-prod-01",
            "tags": {
                "environment": "production",
                "owner": "team-alpha"
            }
        },
        "baseline_config": {
            "vm_size": "Standard_D2s_v3",
            "os_disk_size": 128,
            "network_security_group": "nsg-prod-01",
            "tags": {
                "environment": "production",
                "owner": "team-alpha",
                "compliance": "required"
            }
        },
        "analysis_type": "comprehensive"
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST",
            f"{BASE_URL}/api/v1/configurations/drift-analysis",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['drift_score', 'drift_detected', 'drift_details', 'recommendations']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                print_success(f"Drift score: {data.get('drift_score', 'N/A')}, Detected: {data.get('drift_detected', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_feature_importance():
    """Test GET /api/v1/ml/feature-importance"""
    test_name = "GET /api/v1/ml/feature-importance - Get SHAP analysis"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/feature-importance")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response structure
            data = response.json()
            expected_fields = ['global_importance', 'feature_impacts', 'model_interpretation']
            if validate_response_structure(data, expected_fields, test_name):
                print_success("Response structure validation passed")
                if 'global_importance' in data and isinstance(data['global_importance'], dict):
                    top_features = list(data['global_importance'].keys())[:3]
                    print_success(f"Top features: {', '.join(top_features)}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, passed, _ in test_results if passed)
    failed_tests = total_tests - passed_tests
    
    print(f"{Colors.BOLD}Total Tests:{Colors.RESET} {total_tests}")
    print(f"{Colors.GREEN}Passed:{Colors.RESET} {passed_tests}")
    print(f"{Colors.RED}Failed:{Colors.RESET} {failed_tests}")
    
    if performance_metrics:
        response_times = [time for time in performance_metrics.values() if time is not None]
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n{Colors.BOLD}Performance Metrics:{Colors.RESET}")
            print(f"  Average Response Time: {avg_time:.2f}ms")
            print(f"  Minimum Response Time: {min_time:.2f}ms")
            print(f"  Maximum Response Time: {max_time:.2f}ms")
            
            under_threshold = sum(1 for time in response_times if time < PERFORMANCE_THRESHOLD_MS)
            print(f"  Under {PERFORMANCE_THRESHOLD_MS}ms: {under_threshold}/{len(response_times)} tests")
    
    print(f"\n{Colors.BOLD}Test Results:{Colors.RESET}")
    for test_name, passed, response_time in test_results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        time_str = f" ({response_time:.2f}ms)" if response_time else ""
        print(f"  [{status}] {test_name}{time_str}")
    
    # Overall result
    print(f"\n{Colors.BOLD}Overall Result:{Colors.RESET}")
    if failed_tests == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL TESTS PASSED{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}[FAILURE] SOME TESTS FAILED{Colors.RESET}")
        return 1

def main():
    """Main test execution"""
    print_header("PolicyCortex ML API Endpoint Test Suite")
    print(f"Testing against: {BASE_URL}")
    print(f"Performance threshold: {PERFORMANCE_THRESHOLD_MS}ms")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    print(f"\n{Colors.YELLOW}Checking server availability...{Colors.RESET}")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Server is running and healthy")
        else:
            print_warning(f"Server returned status {response.status_code} on health check")
    except Exception as e:
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_error(f"Error: {str(e)}")
        print(f"\n{Colors.YELLOW}Please ensure the Rust core service is running on localhost:8080{Colors.RESET}")
        return 1
    
    # Run all tests
    tests = [
        test_get_all_predictions,
        test_get_risk_score,
        test_create_prediction,
        test_trigger_remediation,
        test_get_ml_metrics,
        test_submit_feedback,
        test_drift_analysis,
        test_feature_importance
    ]
    
    for test in tests:
        print()  # Add spacing between tests
        test()
    
    # Print summary
    return print_summary()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)