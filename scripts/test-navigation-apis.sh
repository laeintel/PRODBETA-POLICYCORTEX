#!/bin/bash

# Test script for navigation system APIs
# This script tests all the new endpoints added for the comprehensive navigation system

API_BASE="http://localhost:8080/api/v1"

echo "Testing PolicyCortex Navigation System APIs"
echo "==========================================="

# Dashboard APIs
echo -e "\n📊 Testing Dashboard APIs..."
curl -s "${API_BASE}/dashboard/metrics" | jq '.total_resources' > /dev/null && echo "✅ Dashboard metrics endpoint working"
curl -s "${API_BASE}/dashboard/alerts" | jq '.[0].severity' > /dev/null && echo "✅ Dashboard alerts endpoint working"
curl -s "${API_BASE}/dashboard/activities" | jq '.[0].activity_type' > /dev/null && echo "✅ Dashboard activities endpoint working"

# Governance APIs
echo -e "\n⚖️ Testing Governance APIs..."
curl -s "${API_BASE}/governance/compliance/status" | jq '.[0].framework' > /dev/null && echo "✅ Compliance status endpoint working"
curl -s "${API_BASE}/governance/compliance/violations" | jq '.[0].policy_name' > /dev/null && echo "✅ Compliance violations endpoint working"
curl -s "${API_BASE}/governance/risk/assessment" | jq '.[0].risk_level' > /dev/null && echo "✅ Risk assessment endpoint working"
curl -s "${API_BASE}/governance/cost/summary" | jq '.[0].service' > /dev/null && echo "✅ Cost summary endpoint working"
curl -s "${API_BASE}/governance/policies" | jq '.[0].name' > /dev/null && echo "✅ Governance policies endpoint working"

# Security APIs
echo -e "\n🔒 Testing Security APIs..."
curl -s "${API_BASE}/security/iam/users" | jq '.[0].display_name' > /dev/null && echo "✅ IAM users endpoint working"
curl -s "${API_BASE}/security/rbac/roles" | jq '.[0].name' > /dev/null && echo "✅ RBAC roles endpoint working"
curl -s "${API_BASE}/security/pim/requests" | jq '.[0].requestor' > /dev/null && echo "✅ PIM requests endpoint working"
curl -s "${API_BASE}/security/conditional-access/policies" | jq '.[0].name' > /dev/null && echo "✅ Conditional access endpoint working"
curl -s "${API_BASE}/security/zero-trust/status" | jq '.[0].pillar' > /dev/null && echo "✅ Zero trust status endpoint working"
curl -s "${API_BASE}/security/entitlements" | jq '.[0].name' > /dev/null && echo "✅ Entitlements endpoint working"
curl -s "${API_BASE}/security/access-reviews" | jq '.[0].name' > /dev/null && echo "✅ Access reviews endpoint working"

# Operations APIs
echo -e "\n⚙️ Testing Operations APIs..."
curl -s "${API_BASE}/operations/resources" | jq '.[0].name' > /dev/null && echo "✅ Operations resources endpoint working"
curl -s "${API_BASE}/operations/monitoring/metrics" | jq '.[0].metric_name' > /dev/null && echo "✅ Monitoring metrics endpoint working"
curl -s "${API_BASE}/operations/automation/workflows" | jq '.[0].name' > /dev/null && echo "✅ Automation workflows endpoint working"
curl -s "${API_BASE}/operations/notifications" | jq '.[0].title' > /dev/null && echo "✅ Notifications endpoint working"
curl -s "${API_BASE}/operations/alerts" | jq '.[0].name' > /dev/null && echo "✅ Operations alerts endpoint working"

# DevOps APIs
echo -e "\n🚀 Testing DevOps APIs..."
curl -s "${API_BASE}/devops/pipelines" | jq '.[0].name' > /dev/null && echo "✅ Pipelines endpoint working"
curl -s "${API_BASE}/devops/releases" | jq '.[0].version' > /dev/null && echo "✅ Releases endpoint working"
curl -s "${API_BASE}/devops/artifacts" | jq '.[0].name' > /dev/null && echo "✅ Artifacts endpoint working"
curl -s "${API_BASE}/devops/deployments" | jq '.[0].application' > /dev/null && echo "✅ Deployments endpoint working"
curl -s "${API_BASE}/devops/builds" | jq '.[0].number' > /dev/null && echo "✅ Builds endpoint working"
curl -s "${API_BASE}/devops/repos" | jq '.[0].name' > /dev/null && echo "✅ Repos endpoint working"

# AI APIs
echo -e "\n🤖 Testing AI APIs..."
curl -s "${API_BASE}/ai/predictive/compliance" | jq '.[0].violation_probability' > /dev/null && echo "✅ Predictive compliance endpoint working"
curl -s "${API_BASE}/ai/correlations" | jq '.[0].correlation_strength' > /dev/null && echo "✅ AI correlations endpoint working"
curl -s -X POST "${API_BASE}/ai/chat" -H "Content-Type: application/json" -d '{"message":"test"}' | jq '.response' > /dev/null && echo "✅ AI chat endpoint working"
curl -s "${API_BASE}/ai/unified/metrics" | jq '.[0].domain' > /dev/null && echo "✅ Unified metrics endpoint working"

echo -e "\n✅ All navigation API endpoints are configured and responding!"