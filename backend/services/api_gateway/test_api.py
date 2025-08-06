#!/usr/bin/env python
"""
Test script to verify API Gateway is running and returning policy data
"""
import json
from datetime import datetime

import requests

BASE_URL = "http://localhost:8010"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_policies():
    """Test policies endpoint"""
    print("\nTesting policies endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/policies")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total policies: {data.get('total', 0)}")
        if data.get("policies"):
            print(f"First policy: {data['policies'][0].get('displayName', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_policy_details():
    """Test policy details endpoint"""
    print("\nTesting policy details endpoint...")
    policy_id = "ASC Default (subscription: 205b477d-17e7-4b3b-92c1-32cf02626b78)"
    try:
        response = requests.get(f"{BASE_URL}/api/v1/policies/{policy_id}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Policy: {data.get('displayName', 'N/A')}")
            print(f"Compliance: {data.get('compliancePercentage', 0):.1f}%")
            print(f"Total resources: {data.get('totalResources', 0)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print(f"Testing PolicyCortex API at {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Check if server is running
    server_running = False
    try:
        requests.get(BASE_URL, timeout=2)
        server_running = True
    except:
        print("⚠️  Server is not running on port 8010!")
        print("\nTo start the server, run one of these commands:")
        print("1. From api_gateway directory: python start_server.py")
        print("2. From api_gateway directory: start-api-8010.bat")
        print("3. From project root: powershell scripts\\start-local-development.ps1")
        exit(1)

    # Run tests
    all_passed = True
    all_passed &= test_health()
    all_passed &= test_policies()
    all_passed &= test_policy_details()

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
