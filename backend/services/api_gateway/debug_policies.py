#!/usr/bin/env python
"""
Debug the policy fetching
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_policy_fetcher import get_all_policy_assignments


async def debug_policies():
    print("=== Direct Test of Dynamic Policy Fetcher ===")

    # Test the fetcher directly
    assignments = get_all_policy_assignments()

    print(f"\nRaw assignments returned: {len(assignments)}")
    for i, assignment in enumerate(assignments):
        print(f"\nAssignment {i+1}:")
        print(f"  Name: {assignment.get('name')}")
        print(f"  Display Name: {assignment.get('displayName')}")
        print(f"  Type: {assignment.get('type')}")
        print(f"  Subscription: {assignment.get('subscriptionName')}")

    # Now test through the main_simple function
    print("\n\n=== Test Through main_simple.py ===")
    from main_simple import get_azure_policies

    result = await get_azure_policies()
    print(f"\nResult from get_azure_policies:")
    print(f"  Total policies: {result.get('total_policies', 0)}")
    print(f"  Data source: {result.get('data_source')}")
    print(f"  Policy count: {len(result.get('policy_assignments', []))}")

    if result.get("policy_assignments"):
        print("\nPolicies returned:")
        for policy in result["policy_assignments"]:
            print(f"  - {policy.get('displayName')}")


if __name__ == "__main__":
    asyncio.run(debug_policies())
