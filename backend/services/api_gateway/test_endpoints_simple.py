#!/usr/bin/env python
"""
Simple test to check which cost endpoints are available
"""
import json

import requests


def test_endpoints():
    base_url = "http://localhost:8010"

    endpoints = ["/api/v1/costs/overview", "/api/v1/costs/trends", "/api/v1/costs/budgets"]

    print("Quick endpoint test:")
    print("=" * 30)

    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"{endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Data source: {data.get('data_source', 'unknown')}")
            else:
                print(f"  ✗ Error: {response.text[:50]}")
        except Exception as e:
            print(f"{endpoint}: ERROR - {str(e)[:50]}")

    print("=" * 30)


if __name__ == "__main__":
    test_endpoints()
