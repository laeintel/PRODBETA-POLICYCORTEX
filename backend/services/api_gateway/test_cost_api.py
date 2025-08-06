#!/usr/bin/env python
"""
Test the cost API endpoint
"""
import json

import requests


def test_cost_api():
    url = "http://localhost:8010/api/v1/costs/overview"

    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nCost Overview:")
            print(f"  Monthly Cost: ${data['current']['monthlyCost']}")
            print(f"  Daily Cost: ${data['current']['dailyCost']}")
            print(f"  Currency: {data['current']['currency']}")
            print(f"  Data Source: {data['data_source']}")

            print(f"\nTop Services by Cost:")
            for service in data["breakdown"]["byService"][:5]:
                print(f"  - {service['service']}: ${service['cost']} ({service['percentage']}%)")

            print(f"\nTop Resource Groups by Cost:")
            for rg in data["breakdown"]["byResourceGroup"][:3]:
                print(f"  - {rg['resourceGroup']}: ${rg['cost']} ({rg['percentage']}%)")

            print(f"\nCost Recommendations:")
            for rec in data["recommendations"]:
                print(f"  - {rec['type']}: Save ${rec['estimatedSavings']} on {rec['resource']}")
                print(f"    {rec['description']}")
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_cost_api()
