#!/usr/bin/env python
"""
Test the dynamic policy fetching
"""
import asyncio
from azure_policy_fetcher import get_all_policy_assignments

async def test_dynamic_policies():
    print("Testing dynamic policy fetching...")
    
    # Test the fetcher directly
    assignments = get_all_policy_assignments()
    
    print(f"\nFound {len(assignments)} policy assignments:")
    for assignment in assignments:
        print(f"\n- Name: {assignment['displayName']}")
        print(f"  Subscription: {assignment['subscriptionName']}")
        print(f"  Compliance: {assignment['complianceState']} ({assignment['resourceCompliance']})")
        print(f"  Total Resources: {assignment['totalResources']}")
        print(f"  Compliant: {assignment['compliantResources']}")
        print(f"  Non-Compliant: {assignment['nonCompliantResources']}")

if __name__ == "__main__":
    asyncio.run(test_dynamic_policies())