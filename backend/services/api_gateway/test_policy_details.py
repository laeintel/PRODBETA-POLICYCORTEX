#!/usr/bin/env python3
"""
Test script for policy details resource display.
"""

import asyncio
import os
import sys

# Add the parent directory to the path so we can import main_simple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_simple import get_detailed_compliance_resources


def test_policy_details():
    """Test the policy details resource retrieval for all 4 initiatives."""
    print("Testing Policy Details Resource Display...")
    print("=" * 50)

    # Test all 4 policy initiative IDs
    policy_ids = [
        "SecurityCenterBuiltIn-PolicyCortexAi",
        "FedRAMP-High-rg-policortex001-app-dev",
        "FedRAMP-High-AeoliTech-app",
        "SecurityCenterBuiltIn-sub-dev",
    ]

    for policy_id in policy_ids:
        print(f"\n--- Testing Policy: {policy_id} ---")

        try:
            resources = get_detailed_compliance_resources(policy_id)

            if resources:
                compliant = [r for r in resources if r["status"] == "Compliant"]
                non_compliant = [r for r in resources if r["status"] == "NonCompliant"]

                print(f"[SUCCESS] Found {len(resources)} total resources")
                print(f"  - Compliant: {len(compliant)}")
                print(f"  - Non-compliant: {len(non_compliant)}")

                # Show sample resources
                if compliant:
                    print(
                        f"  Sample compliant resource: {compliant[0]['name']} ({compliant[0]['type']})"
                    )
                if non_compliant:
                    print(
                        f"  Sample non-compliant resource: {non_compliant[0]['name']} ({non_compliant[0]['type']})"
                    )
                    print(f"  Reason: {non_compliant[0]['complianceReasonCode']}")
            else:
                print(f"[FAILED] No resources found for {policy_id}")

        except Exception as e:
            print(f"[ERROR] Failed to get resources for {policy_id}: {e}")

    print(f"\n[COMPLETE] Policy details resource test completed!")


if __name__ == "__main__":
    test_policy_details()
