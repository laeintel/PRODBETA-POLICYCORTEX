#!/usr/bin/env python
"""
Test that we're getting real Azure policies without fallback
"""
import json

import requests


def test_real_policies():
    """Test the policies endpoint returns real data"""
    url = "http://localhost:8010/api/v1/policies"

    print("Testing policies endpoint...")
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nTotal policies: {data.get('total_policies', 0)}")
            print(f"Data source: {data.get('data_source', 'unknown')}")

            policies = data.get("policy_assignments", [])
            if policies:
                print(f"\nFound {len(policies)} policies:")
                for policy in policies:
                    print(f"\n- {policy.get('displayName', 'Unknown')}")
                    print(f"  Subscription: {policy.get('subscriptionName', 'Unknown')}")
                    print(
                        f"  Compliance: {policy.get('complianceState', 'Unknown')} ({policy.get('resourceCompliance', '0%')})"
                    )
                    print(f"  Type: {policy.get('type', 'Unknown')}")
            else:
                print("\n⚠️  No policies found!")
                if "error" in data:
                    print(f"Error: {data['error']}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Error calling API: {e}")


if __name__ == "__main__":
    test_real_policies()
