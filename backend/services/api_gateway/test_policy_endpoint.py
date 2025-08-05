#!/usr/bin/env python
"""
Test the policy details endpoint
"""
import time
import subprocess
import json

def test_policy_endpoint():
    """Test the policy details endpoint"""
    print("Waiting for server to start...")
    time.sleep(5)
    
    url = "http://localhost:8010/api/v1/policies/SecurityCenterBuiltIn"
    
    try:
        # Use curl to test the endpoint
        result = subprocess.run([
            "curl", "-s", url
        ], capture_output=True, text=True, check=True)
        
        print(f"Response: {result.stdout}")
        
        # Try to parse as JSON
        try:
            data = json.loads(result.stdout)
            print(f"Policy name: {data.get('displayName', 'unknown')}")
            print(f"Compliance: {data.get('complianceState', 'unknown')}")
            print(f"Resources: {data.get('totalResources', 0)}")
        except:
            print("Response is not JSON")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_policy_endpoint()