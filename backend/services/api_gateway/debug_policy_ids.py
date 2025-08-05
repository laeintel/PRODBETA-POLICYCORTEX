#!/usr/bin/env python
"""
Debug the policy IDs being returned
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_simple import get_azure_policies

async def debug_policy_ids():
    print("=== Debug Policy IDs ===")
    
    policies_data = await get_azure_policies()
    print(f"Total policies from get_azure_policies: {len(policies_data.get('policy_assignments', []))}")
    
    for i, policy in enumerate(policies_data.get("policy_assignments", [])):
        print(f"\nPolicy {i+1}:")
        print(f"  name: {policy.get('name')}")
        print(f"  id: {policy.get('id')}")
        print(f"  displayName: {policy.get('displayName')}")
        print(f"  subscriptionId: {policy.get('subscriptionId')}")
        print(f"  subscriptionName: {policy.get('subscriptionName')}")

if __name__ == "__main__":
    asyncio.run(debug_policy_ids())