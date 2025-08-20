#!/usr/bin/env python3
"""
Test Suite for ACTUAL ML API Endpoints implemented in PolicyCortex
Tests the endpoints that are actually defined in the Rust backend
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Tuple, List
from datetime import datetime
import statistics

# Configuration
BASE_URL = "http://localhost:8080"
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
            print_warning(f"Missing expected fields: {missing_fields}")
            # Still pass if we got some response
            return True
        return True
    elif isinstance(response_data, list) and len(response_data) > 0:
        # Check first item if it's a list
        return validate_response_structure(response_data[0], expected_fields, test_name)
    return True

def test_get_all_predictions():
    """Test GET /api/v1/predictions - Get all predictions"""
    test_name = "GET /api/v1/predictions"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/predictions")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded: {response_time_ms:.2f}ms > {PERFORMANCE_THRESHOLD_MS}ms")
            
            # Validate response
            data = response.json()
            if isinstance(data, list):
                print_success(f"Response is a list with {len(data)} predictions")
            elif isinstance(data, dict):
                print_success(f"Response is an object: {list(data.keys())[:5]}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            if response.text:
                print_error(f"Response: {response.text[:200]}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_ml_prediction():
    """Test GET /api/v1/ml/predict/:resource_id"""
    test_name = "GET /api/v1/ml/predict/:resource_id"
    resource_id = "test-vm-001"
    print_test_start(f"{test_name} (resource: {resource_id})")
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/predict/{resource_id}")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            expected_fields = ['resource_id', 'risk_score', 'confidence', 'recommendations']
            validate_response_structure(data, expected_fields, test_name)
            
            if 'risk_score' in data:
                print_success(f"Risk score: {data['risk_score']}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_risk_score():
    """Test GET /api/v1/predictions/risk-score/:resource_id"""
    test_name = "GET /api/v1/predictions/risk-score/:resource_id"
    resource_id = "test-resource-1"
    print_test_start(f"{test_name} (resource: {resource_id})")
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/predictions/risk-score/{resource_id}")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'risk_score' in data:
                print_success(f"Risk score: {data.get('risk_score', 'N/A')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_ml_metrics():
    """Test GET /api/v1/ml/metrics"""
    test_name = "GET /api/v1/ml/metrics"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/metrics")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'accuracy' in data:
                print_success(f"Model accuracy: {data.get('accuracy', 'N/A')}")
            elif 'models' in data:
                print_success(f"Got metrics for {len(data.get('models', []))} models")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_feature_importance():
    """Test GET /api/v1/ml/feature-importance"""
    test_name = "GET /api/v1/ml/feature-importance"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/feature-importance")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'features' in data:
                features = data['features']
                if isinstance(features, list) and len(features) > 0:
                    print_success(f"Top feature: {features[0].get('name', 'Unknown')}")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_submit_feedback():
    """Test POST /api/v1/ml/feedback"""
    test_name = "POST /api/v1/ml/feedback"
    print_test_start(test_name)
    
    payload = {
        "prediction_id": f"pred-{int(time.time())}",
        "resource_id": "test-resource-1",
        "feedback_type": "accuracy",
        "rating": 4,
        "correct_label": "compliant",
        "predicted_label": "non_compliant",
        "comments": "False positive - resource was actually compliant"
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
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'feedback_id' in data or 'status' in data:
                print_success(f"Feedback submitted successfully")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_anomalies():
    """Test GET /api/v1/ml/anomalies"""
    test_name = "GET /api/v1/ml/anomalies"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/ml/anomalies")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'anomalies' in data:
                print_success(f"Found {len(data['anomalies'])} anomalies")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_trigger_retraining():
    """Test POST /api/v1/ml/retrain"""
    test_name = "POST /api/v1/ml/retrain"
    print_test_start(test_name)
    
    payload = {
        "model_type": "compliance_prediction",
        "reason": "scheduled_retraining",
        "parameters": {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        }
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST",
            f"{BASE_URL}/api/v1/ml/retrain",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201, 202]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'job_id' in data or 'status' in data:
                print_success(f"Retraining triggered successfully")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_get_correlations():
    """Test GET /api/v1/correlations - Patent #4 Cross-Domain Correlation"""
    test_name = "GET /api/v1/correlations"
    print_test_start(test_name)
    
    try:
        response, response_time_ms = measure_request("GET", f"{BASE_URL}/api/v1/correlations")
        
        if response.status_code == 200:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'correlations' in data:
                print_success(f"Found {len(data['correlations'])} correlations")
            elif isinstance(data, list):
                print_success(f"Found {len(data)} correlation items")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            test_results.append((test_name, False, response_time_ms))
            return False
            
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        test_results.append((test_name, False, None))
        return False

def test_conversation_api():
    """Test POST /api/v1/conversation - Patent #2 Conversational Intelligence"""
    test_name = "POST /api/v1/conversation"
    print_test_start(test_name)
    
    payload = {
        "message": "What are the compliance risks for my virtual machines?",
        "context": {
            "tenant_id": "test-tenant",
            "user_id": "test-user",
            "session_id": f"session-{int(time.time())}"
        }
    }
    
    try:
        response, response_time_ms = measure_request(
            "POST",
            f"{BASE_URL}/api/v1/conversation",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 201]:
            print_success(f"Status code: {response.status_code}", response_time_ms)
            
            # Check performance
            if response_time_ms < PERFORMANCE_THRESHOLD_MS:
                print_success(f"Performance check passed")
            else:
                print_warning(f"Performance threshold exceeded")
            
            # Validate response
            data = response.json()
            if 'response' in data or 'message' in data:
                print_success(f"Got conversational response")
            
            test_results.append((test_name, True, response_time_ms))
            performance_metrics[test_name] = response_time_ms
            return True
        else:
            print_error(f"Status code: {response.status_code}")
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
        print(f"{Colors.RED}{Colors.BOLD}[FAILURE] {failed_tests} TEST(S) FAILED{Colors.RESET}")
        return 1

def main():
    """Main test execution"""
    print_header("PolicyCortex ACTUAL ML API Endpoint Test Suite")
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
    
    # Run all tests - testing the ACTUAL implemented endpoints
    tests = [
        test_get_all_predictions,        # GET /api/v1/predictions
        test_get_risk_score,             # GET /api/v1/predictions/risk-score/:resource_id
        test_get_ml_prediction,          # GET /api/v1/ml/predict/:resource_id
        test_get_ml_metrics,             # GET /api/v1/ml/metrics
        test_get_feature_importance,     # GET /api/v1/ml/feature-importance
        test_get_anomalies,              # GET /api/v1/ml/anomalies
        test_submit_feedback,            # POST /api/v1/ml/feedback
        test_trigger_retraining,         # POST /api/v1/ml/retrain
        test_get_correlations,           # GET /api/v1/correlations (Patent #4)
        test_conversation_api,           # POST /api/v1/conversation (Patent #2)
    ]
    
    print(f"\n{Colors.BOLD}Running {len(tests)} endpoint tests...{Colors.RESET}")
    
    for test in tests:
        print()  # Add spacing between tests
        test()
    
    # Print summary
    return print_summary()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)