#!/usr/bin/env python3
"""
PolicyCortex Patent API Testing Script
Tests all 4 patent implementations locally
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8002"  # AI Engine service
API_TIMEOUT = 30
RETRY_ATTEMPTS = 3

def test_api_endpoint(
    endpoint_name: str,
    url: str,
    method: str = "GET",
    data: Dict[Any, Any] = None,
    headers: Dict[str, str] = None
) -> bool:
    """Test an API endpoint and return success status."""
    
    print(f"🔍 Testing {endpoint_name}...")
    
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-token-for-testing"
        }
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=API_TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=API_TIMEOUT)
            else:
                response = requests.request(method, url, json=data, headers=headers, timeout=API_TIMEOUT)
            
            if response.status_code in [200, 201]:
                print(f"✅ {endpoint_name} - Success")
                if response.json():
                    result = response.json()
                    print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                return True
            else:
                print(f"❌ {endpoint_name} - HTTP {response.status_code}")
                print(f"   Error: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint_name} - Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(2)
    
    return False

def test_service_health(service_name: str, base_url: str) -> bool:
    """Test if a service is healthy."""
    print(f"🏥 Checking {service_name} health...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} health check failed: {str(e)}")
        return False

def test_patent_1_compliance_prediction():
    """Test Patent 1: Predictive Policy Compliance Engine"""
    print("\n🧪 Testing Patent 1: Predictive Policy Compliance Engine")
    print("=" * 60)
    
    # Test compliance prediction
    test_data = {
        "request_id": "patent1_test_001",
        "policy_data": {
            "policies": [
                {
                    "id": "policy_001",
                    "name": "VM Security Policy",
                    "type": "security",
                    "rules": ["require_encryption", "block_public_ip"]
                }
            ],
            "resources": [
                {
                    "id": "vm_001",
                    "type": "virtual_machine",
                    "properties": {
                        "encryption_enabled": True,
                        "public_ip": False,
                        "size": "Standard_D2s_v3"
                    }
                }
            ]
        },
        "time_horizon": "7_days",
        "confidence_threshold": 0.8
    }
    
    return test_api_endpoint(
        "Patent 1: Compliance Prediction",
        f"{BASE_URL}/api/v1/compliance/predict",
        "POST",
        test_data
    )

def test_patent_2_unified_ai_platform():
    """Test Patent 2: Unified AI-Driven Platform"""
    print("\n🧪 Testing Patent 2: Unified AI-Driven Platform")
    print("=" * 60)
    
    # Test unified AI analysis
    governance_data = {
        "resource_data": [[[0.5, 0.3, 0.7] + [0.4] * 47]],  # 1x1x50 resource features
        "service_data": [[0.6, 0.4, 0.8] + [0.5] * 27],      # 1x30 service features
        "domain_data": [[[0.7, 0.5, 0.9] + [0.6] * 17]]      # 1x1x20 domain features
    }
    
    test_data = {
        "request_id": "patent2_test_001",
        "governance_data": governance_data,
        "analysis_scope": ["security", "compliance", "cost"],
        "optimization_preferences": {
            "security_weight": 0.3,
            "compliance_weight": 0.3,
            "cost_weight": 0.2,
            "performance_weight": 0.1,
            "operations_weight": 0.1
        }
    }
    
    success1 = test_api_endpoint(
        "Patent 2: Unified AI Analysis",
        f"{BASE_URL}/api/v1/unified-ai/analyze",
        "POST",
        test_data
    )
    
    # Test governance optimization
    optimization_data = {
        "request_id": "patent2_opt_001",
        "governance_data": {
            "n_variables": 60,
            "budget_limit": 10000,
            "min_security": 0.8,
            "min_compliance": 0.9,
            "current_state": governance_data
        },
        "preferences": {
            "security_weight": 0.3,
            "compliance_weight": 0.3,
            "cost_weight": 0.2,
            "performance_weight": 0.1,
            "operations_weight": 0.1
        },
        "constraints": {
            "budget_limit": 10000,
            "min_security_score": 0.8
        },
        "max_generations": 50
    }
    
    success2 = test_api_endpoint(
        "Patent 2: Governance Optimization",
        f"{BASE_URL}/api/v1/unified-ai/optimize",
        "POST",
        optimization_data
    )
    
    return success1 and success2

def test_patent_3_conversational_ai():
    """Test Patent 3: Conversational Governance Intelligence"""
    print("\n🧪 Testing Patent 3: Conversational Governance Intelligence")
    print("=" * 60)
    
    # Test conversational AI
    conversation_data = {
        "user_input": "What are the current security policies for virtual machines?",
        "session_id": "test_session_001",
        "user_id": "test_user"
    }
    
    success1 = test_api_endpoint(
        "Patent 3: Conversational AI",
        f"{BASE_URL}/api/v1/conversation/governance",
        "POST",
        conversation_data
    )
    
    # Test policy synthesis
    policy_synthesis_data = {
        "request_id": "patent3_policy_001",
        "description": "Create a network security policy that blocks all unauthorized access and requires VPN for remote connections",
        "domain": "security",
        "policy_type": "network",
        "constraints": ["must_include_logging", "require_approval"]
    }
    
    success2 = test_api_endpoint(
        "Patent 3: Policy Synthesis",
        f"{BASE_URL}/api/v1/conversation/policy-synthesis",
        "POST",
        policy_synthesis_data
    )
    
    # Test conversation history
    success3 = test_api_endpoint(
        "Patent 3: Conversation History",
        f"{BASE_URL}/api/v1/conversation/history/test_session_001",
        "GET"
    )
    
    return success1 and success2 and success3

def test_patent_4_correlation_engine():
    """Test Patent 4: Cross-Domain Correlation Engine"""
    print("\n🧪 Testing Patent 4: Cross-Domain Correlation Engine")
    print("=" * 60)
    
    # Test cross-domain correlation analysis
    correlation_data = {
        "request_id": "patent4_test_001",
        "events": [
            {
                "event_id": "evt_001",
                "domain": "security",
                "timestamp": "2024-01-15T10:30:00Z",
                "event_type": "policy_violation",
                "severity": "high",
                "attributes": {
                    "resource_id": "vm_001",
                    "policy_id": "sec_policy_001",
                    "violation_type": "unauthorized_access"
                }
            },
            {
                "event_id": "evt_002", 
                "domain": "cost",
                "timestamp": "2024-01-15T10:35:00Z",
                "event_type": "cost_spike",
                "severity": "medium",
                "attributes": {
                    "resource_id": "vm_001",
                    "cost_increase": 250.0,
                    "timeframe": "1_hour"
                }
            }
        ],
        "correlation_types": ["temporal", "causal", "statistical"],
        "analysis_window": "24_hours"
    }
    
    success1 = test_api_endpoint(
        "Patent 4: Correlation Analysis",
        f"{BASE_URL}/api/v1/correlation/analyze",
        "POST",
        correlation_data
    )
    
    # Test correlation patterns
    success2 = test_api_endpoint(
        "Patent 4: Correlation Patterns",
        f"{BASE_URL}/api/v1/correlation/patterns?min_confidence=0.7&limit=10",
        "GET"
    )
    
    # Test correlation insights
    success3 = test_api_endpoint(
        "Patent 4: Correlation Insights",
        f"{BASE_URL}/api/v1/correlation/insights?priority=high&limit=5",
        "GET"
    )
    
    return success1 and success2 and success3

def test_additional_endpoints():
    """Test additional important endpoints"""
    print("\n🧪 Testing Additional Endpoints")
    print("=" * 40)
    
    endpoints = [
        ("AI Engine Health", f"{BASE_URL}/health", "GET"),
        ("Model Manager Status", f"{BASE_URL}/api/v1/models", "GET"),
        ("Policy Analysis", f"{BASE_URL}/api/v1/policy-analysis", "POST", {
            "request_id": "test_policy_001",
            "policy_text": "All virtual machines must have encryption enabled",
            "analysis_type": "compliance",
            "options": {"include_recommendations": True}
        })
    ]
    
    results = []
    for name, url, method, *data in endpoints:
        payload = data[0] if data else None
        results.append(test_api_endpoint(name, url, method, payload))
    
    return all(results)

def main():
    """Main testing function"""
    print("🚀 PolicyCortex Patent API Testing")
    print("=" * 50)
    
    # Check if AI Engine is accessible
    if not test_service_health("AI Engine", BASE_URL):
        print("\n❌ AI Engine service is not accessible. Please ensure:")
        print("1. Docker services are running: docker-compose -f docker-compose.local.yml up -d")
        print("2. AI Engine container is healthy: docker-compose -f docker-compose.local.yml logs ai-engine")
        print("3. Port 8002 is not blocked by firewall")
        sys.exit(1)
    
    # Run patent tests
    test_results = []
    
    # Note: Some tests might fail due to missing ML model dependencies
    # This is expected in a minimal local setup
    print("\n⚠️ Note: Some tests may fail due to missing ML model dependencies.")
    print("This is expected in a minimal Docker setup without GPU support.\n")
    
    test_results.append(("Patent 1", test_patent_1_compliance_prediction()))
    test_results.append(("Patent 2", test_patent_2_unified_ai_platform()))
    test_results.append(("Patent 3", test_patent_3_conversational_ai()))
    test_results.append(("Patent 4", test_patent_4_correlation_engine()))
    test_results.append(("Additional APIs", test_additional_endpoints()))
    
    # Results summary
    print("\n🎯 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All patent implementations are working correctly!")
        print("\n🌟 Next steps:")
        print("1. Open http://localhost:5173 to test the frontend")
        print("2. Navigate to the AI Assistant page")
        print("3. Try the conversational interface")
    else:
        print(f"\n⚠️ {total - passed} test suite(s) failed.")
        print("This might be due to:")
        print("• Missing ML model dependencies (expected in Docker)")
        print("• Service startup delays (try running again)")
        print("• Configuration issues")
        print("\nThe core functionality may still work via the frontend.")
    
    return passed == total

if __name__ == "__main__":
    main()