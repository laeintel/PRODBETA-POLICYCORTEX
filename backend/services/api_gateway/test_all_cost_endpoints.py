#!/usr/bin/env python
"""
Test all cost API endpoints
"""
import requests
import json

def test_cost_endpoints():
    base_url = "http://localhost:8010"
    
    endpoints = [
        "/api/v1/costs/overview",
        "/api/v1/costs/trends", 
        "/api/v1/costs/budgets",
        "/api/v1/costs/details/205b477d-17e7-4b3b-92c1-32cf02626b78"
    ]
    
    print("Testing Cost Management API Endpoints:")
    print("=" * 50)
    
    for endpoint in endpoints:
        print(f"\nTesting: {endpoint}")
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'data_source' in data:
                    print(f"  Data Source: {data['data_source']}")
                
                # Show key metrics based on endpoint
                if "overview" in endpoint:
                    print(f"  Monthly Cost: ${data['current']['monthlyCost']}")
                    print(f"  Services: {len(data['breakdown']['byService'])}")
                    print(f"  Recommendations: {len(data['recommendations'])}")
                
                elif "trends" in endpoint:
                    print(f"  Historical Months: {len(data['historical_data'])}")
                    print(f"  Trend: {data['trend_analysis']['overall_trend']}")
                    print(f"  Next Month Projection: ${data['projections']['next_month']}")
                
                elif "budgets" in endpoint:
                    print(f"  Total Budgets: {data['summary']['total_budgets']}")
                    print(f"  Total Allocated: ${data['summary']['total_allocated']}")
                    print(f"  Overall Utilization: {data['summary']['overall_utilization']}%")
                
                elif "details" in endpoint:
                    if 'cost_summary' in data:
                        print(f"  Subscription Cost: ${data['cost_summary']['total_cost']}")
                        print(f"  Services: {len(data['cost_summary']['service_breakdown'])}")
                        print(f"  Resource Groups: {len(data['resource_group_breakdown'])}")
                    else:
                        print(f"  Error: {data.get('error', 'Unknown error')}")
                
            else:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Exception: {e}")
    
    print(f"\n{'=' * 50}")
    print("Cost API Testing Complete")

if __name__ == "__main__":
    test_cost_endpoints()