# Test script for navigation system APIs
# This script tests all the new endpoints added for the comprehensive navigation system

$API_BASE = "http://localhost:8080/api/v1"

Write-Host "Testing PolicyCortex Navigation System APIs" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# Dashboard APIs
Write-Host "`n📊 Testing Dashboard APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/dashboard/metrics" -Method Get
    if ($response.total_resources) { Write-Host "✅ Dashboard metrics endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Dashboard metrics endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/dashboard/alerts" -Method Get
    if ($response[0].severity) { Write-Host "✅ Dashboard alerts endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Dashboard alerts endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/dashboard/activities" -Method Get
    if ($response[0].activity_type) { Write-Host "✅ Dashboard activities endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Dashboard activities endpoint failed" -ForegroundColor Red }

# Governance APIs
Write-Host "`n⚖️ Testing Governance APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/governance/compliance/status" -Method Get
    if ($response[0].framework) { Write-Host "✅ Compliance status endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Compliance status endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/governance/compliance/violations" -Method Get
    if ($response[0].policy_name) { Write-Host "✅ Compliance violations endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Compliance violations endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/governance/risk/assessment" -Method Get
    if ($response[0].risk_level) { Write-Host "✅ Risk assessment endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Risk assessment endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/governance/cost/summary" -Method Get
    if ($response[0].service) { Write-Host "✅ Cost summary endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Cost summary endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/governance/policies" -Method Get
    if ($response[0].name) { Write-Host "✅ Governance policies endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Governance policies endpoint failed" -ForegroundColor Red }

# Security APIs
Write-Host "`n🔒 Testing Security APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/iam/users" -Method Get
    if ($response[0].display_name) { Write-Host "✅ IAM users endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ IAM users endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/rbac/roles" -Method Get
    if ($response[0].name) { Write-Host "✅ RBAC roles endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ RBAC roles endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/pim/requests" -Method Get
    if ($response[0].requestor) { Write-Host "✅ PIM requests endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ PIM requests endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/conditional-access/policies" -Method Get
    if ($response[0].name) { Write-Host "✅ Conditional access endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Conditional access endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/zero-trust/status" -Method Get
    if ($response[0].pillar) { Write-Host "✅ Zero trust status endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Zero trust status endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/entitlements" -Method Get
    if ($response[0].name) { Write-Host "✅ Entitlements endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Entitlements endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/security/access-reviews" -Method Get
    if ($response[0].name) { Write-Host "✅ Access reviews endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Access reviews endpoint failed" -ForegroundColor Red }

# Operations APIs
Write-Host "`n⚙️ Testing Operations APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/operations/resources" -Method Get
    if ($response[0].name) { Write-Host "✅ Operations resources endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Operations resources endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/operations/monitoring/metrics" -Method Get
    if ($response[0].metric_name) { Write-Host "✅ Monitoring metrics endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Monitoring metrics endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/operations/automation/workflows" -Method Get
    if ($response[0].name) { Write-Host "✅ Automation workflows endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Automation workflows endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/operations/notifications" -Method Get
    if ($response[0].title) { Write-Host "✅ Notifications endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Notifications endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/operations/alerts" -Method Get
    if ($response[0].name) { Write-Host "✅ Operations alerts endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Operations alerts endpoint failed" -ForegroundColor Red }

# DevOps APIs
Write-Host "`n🚀 Testing DevOps APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/pipelines" -Method Get
    if ($response[0].name) { Write-Host "✅ Pipelines endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Pipelines endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/releases" -Method Get
    if ($response[0].version) { Write-Host "✅ Releases endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Releases endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/artifacts" -Method Get
    if ($response[0].name) { Write-Host "✅ Artifacts endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Artifacts endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/deployments" -Method Get
    if ($response[0].application) { Write-Host "✅ Deployments endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Deployments endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/builds" -Method Get
    if ($response[0].number) { Write-Host "✅ Builds endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Builds endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/devops/repos" -Method Get
    if ($response[0].name) { Write-Host "✅ Repos endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Repos endpoint failed" -ForegroundColor Red }

# AI APIs
Write-Host "`n🤖 Testing AI APIs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_BASE/ai/predictive/compliance" -Method Get
    if ($response[0].violation_probability) { Write-Host "✅ Predictive compliance endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Predictive compliance endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/ai/correlations" -Method Get
    if ($response[0].correlation_strength) { Write-Host "✅ AI correlations endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ AI correlations endpoint failed" -ForegroundColor Red }

try {
    $body = @{ message = "test" } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$API_BASE/ai/chat" -Method Post -Body $body -ContentType "application/json"
    if ($response.response) { Write-Host "✅ AI chat endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ AI chat endpoint failed" -ForegroundColor Red }

try {
    $response = Invoke-RestMethod -Uri "$API_BASE/ai/unified/metrics" -Method Get
    if ($response[0].domain) { Write-Host "✅ Unified metrics endpoint working" -ForegroundColor Green }
} catch { Write-Host "❌ Unified metrics endpoint failed" -ForegroundColor Red }

Write-Host "`n✅ Navigation API endpoint testing complete!" -ForegroundColor Green