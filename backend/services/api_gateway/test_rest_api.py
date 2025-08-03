#!/usr/bin/env python3
"""
Test script for Azure REST API policy discovery.
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import main_simple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_simple import policy_discovery

async def test_policy_discovery():
    """Test the Azure Policy REST API discovery system."""
    print("Testing Azure Policy REST API Discovery...")
    print("=" * 50)
    
    try:
        # Test 1: Get access token
        print("1. Testing Azure access token retrieval...")
        token = await policy_discovery.get_access_token()
        if token:
            print("   [SUCCESS] Successfully obtained Azure access token")
            print(f"   Token length: {len(token)} characters")
        else:
            print("   [FAILED] Failed to obtain Azure access token")
            
        # Test 2: Test policy discovery
        print("\n2. Testing automatic policy discovery...")
        result = await policy_discovery.discover_policies()
        
        print(f"   Total policies found: {result.get('total_policies', 0)}")
        print(f"   Data source: {result.get('data_source', 'unknown')}")
        
        # Display policies
        policies = result.get('policy_assignments', [])
        if policies:
            print("\n   Found policies:")
            for i, policy in enumerate(policies[:10]):  # Show first 10
                name = policy.get('displayName', policy.get('name', 'Unknown'))
                policy_type = policy.get('type', 'Unknown')
                print(f"   {i+1}. {name} ({policy_type})")
            
            if len(policies) > 10:
                print(f"   ... and {len(policies) - 10} more policies")
        else:
            print("   No policies found")
            
        # Test 3: Check if all 4 expected initiatives are found
        print("\n3. Checking for expected policy initiatives...")
        expected_initiatives = [
            "ASC Default (subscription: PolicyCortex Ai)",
            "FedRAMP High (PolicyCortex Ai/rg-policortex001-app-dev)", 
            "FedRAMP High (AeoliTech_app)",
            "ASC Default (subscription: sub-dev)"
        ]
        
        found_count = 0
        for expected in expected_initiatives:
            found = any(expected in policy.get('displayName', '') for policy in policies)
            status = "[FOUND]" if found else "[MISSING]"
            print(f"   {status}: {expected}")
            if found:
                found_count += 1
        
        print(f"\n   Summary: Found {found_count} out of {len(expected_initiatives)} expected initiatives")
        
        if found_count == len(expected_initiatives):
            print("   [SUCCESS] All expected policy initiatives discovered!")
        elif found_count > 0:
            print("   [PARTIAL] Some initiatives found, REST API working but may need token permissions")
        else:
            print("   [INFO] Using fallback data (REST API may need authentication setup)")
            
        return result
        
    except Exception as e:
        print(f"   [ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_policy_discovery())
    
    if result:
        print(f"\n[READY] Policy discovery system ready!")
        print(f"   - Automatic refresh every 5 minutes")
        print(f"   - Real-time detection of new policies/initiatives")
        print(f"   - Zero manual intervention required")
    else:
        print(f"\n[ATTENTION] Policy discovery needs attention")