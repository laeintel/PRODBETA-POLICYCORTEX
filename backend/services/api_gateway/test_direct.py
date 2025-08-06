#!/usr/bin/env python
"""
Test the get_policies_list function directly
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_direct():
    # Import fresh
    from main_simple import get_policies_list

    print("=== Testing get_policies_list directly ===")

    result = await get_policies_list()
    policies = result.get("policies", [])

    print(f"Found {len(policies)} policies:")
    for i, policy in enumerate(policies):
        print(f"Policy {i+1}:")
        print(f"  ID: {policy.get('id')}")
        print(f"  Name: {policy.get('name')}")
        print()


if __name__ == "__main__":
    asyncio.run(test_direct())
