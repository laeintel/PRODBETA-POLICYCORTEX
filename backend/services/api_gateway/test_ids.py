#!/usr/bin/env python
"""
Test the policy IDs from the API endpoint
"""
import requests
import json

def test_policy_ids():
    url = "http://localhost:8010/api/v1/policies"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            policies = data.get('policies', [])
            print(f"Found {len(policies)} policies")
            
            for i, policy in enumerate(policies):
                print(f"Policy {i+1}:")
                print(f"  ID: {policy.get('id')}")
                print(f"  Name: {policy.get('name')}")
                print(f"  Display Name: {policy.get('displayName')}")
                print()
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_policy_ids()