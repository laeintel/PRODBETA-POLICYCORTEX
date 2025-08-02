# PolicyCortex Local Testing Setup Script
# This script sets up and tests the complete PolicyCortex system locally

Write-Host "🚀 PolicyCortex Local Testing Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Function to check if a service is running
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [string]$Url,
        [int]$MaxRetries = 10
    )
    
    Write-Host "⏳ Checking $ServiceName health..." -ForegroundColor Yellow
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            $response = Invoke-RestMethod -Uri "$Url/health" -Method Get -TimeoutSec 5
            if ($response) {
                Write-Host "✅ $ServiceName is healthy" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Host "❌ Attempt $i/$MaxRetries failed for $ServiceName" -ForegroundColor Red
            Start-Sleep -Seconds 5
        }
    }
    
    Write-Host "💥 $ServiceName failed to start properly" -ForegroundColor Red
    return $false
}

# Function to test API endpoints
function Test-ApiEndpoint {
    param(
        [string]$EndpointName,
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Body = @{},
        [bool]$ExpectSuccess = $true
    )
    
    Write-Host "🔍 Testing $EndpointName..." -ForegroundColor Yellow
    
    try {
        $headers = @{
            "Content-Type" = "application/json"
            "Authorization" = "Bearer dummy-token-for-testing"
        }
        
        if ($Method -eq "GET") {
            $response = Invoke-RestMethod -Uri $Url -Method $Method -Headers $headers -TimeoutSec 10
        } else {
            $jsonBody = $Body | ConvertTo-Json -Depth 5
            $response = Invoke-RestMethod -Uri $Url -Method $Method -Body $jsonBody -Headers $headers -TimeoutSec 30
        }
        
        if ($ExpectSuccess) {
            Write-Host "✅ $EndpointName passed" -ForegroundColor Green
            return $true
        }
    }
    catch {
        if (-not $ExpectSuccess) {
            Write-Host "✅ $EndpointName correctly failed (expected)" -ForegroundColor Green
            return $true
        }
        Write-Host "❌ $EndpointName failed: $($_.ErrorDetails.Message)" -ForegroundColor Red
        return $false
    }
    
    return $false
}

# Step 1: Check Prerequisites
Write-Host "`n📋 Step 1: Checking Prerequisites" -ForegroundColor Cyan

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not installed or not running" -ForegroundColor Red
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker-compose --version
    Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker Compose is not installed" -ForegroundColor Red
    exit 1
}

# Step 2: Install Python Dependencies (for local AI engine testing)
Write-Host "`n📦 Step 2: Installing Python Dependencies for AI Engine" -ForegroundColor Cyan

try {
    Push-Location "backend"
    
    # Check if virtual environment exists
    if (-not (Test-Path "venv")) {
        Write-Host "🔧 Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    
    # Install requirements
    Write-Host "🔧 Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Try to install ML dependencies
    Write-Host "🔧 Installing ML dependencies..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers sentence-transformers scikit-learn pymoo
    
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
    Pop-Location
}
catch {
    Write-Host "⚠️ Python dependency installation failed, continuing with Docker..." -ForegroundColor Yellow
    Pop-Location
}

# Step 3: Start Services
Write-Host "`n🐳 Step 3: Starting Docker Services" -ForegroundColor Cyan

# Stop any existing containers
Write-Host "🔧 Stopping existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.local.yml down

# Build and start services
Write-Host "🔧 Building and starting services..." -ForegroundColor Yellow
docker-compose -f docker-compose.local.yml up -d --build

# Wait for services to be ready
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Step 4: Health Checks
Write-Host "`n🏥 Step 4: Service Health Checks" -ForegroundColor Cyan

$services = @(
    @{ Name = "API Gateway"; Url = "http://localhost:8000" },
    @{ Name = "Azure Integration"; Url = "http://localhost:8001" },
    @{ Name = "AI Engine"; Url = "http://localhost:8002" },
    @{ Name = "Data Processing"; Url = "http://localhost:8003" },
    @{ Name = "Conversation"; Url = "http://localhost:8004" },
    @{ Name = "Notification"; Url = "http://localhost:8005" }
)

$healthyServices = 0
foreach ($service in $services) {
    if (Test-ServiceHealth -ServiceName $service.Name -Url $service.Url) {
        $healthyServices++
    }
}

Write-Host "`n📊 Health Check Results: $healthyServices/$($services.Count) services healthy" -ForegroundColor Cyan

# Step 5: Test Patent Implementations
Write-Host "`n🧪 Step 5: Testing Patent Implementations" -ForegroundColor Cyan

# Test Patent 2: Unified AI Analysis
$unifiedAiTest = @{
    request_id = "test_001"
    governance_data = @{
        resource_data = @(@(@(0.5, 0.3, 0.7) * 50)) # Mock resource data
        service_data = @(@(0.6, 0.4, 0.8) * 30) # Mock service data  
        domain_data = @(@(@(0.7, 0.5, 0.9) * 20)) # Mock domain data
    }
    analysis_scope = @("security", "compliance", "cost")
    optimization_preferences = @{
        security_weight = 0.3
        compliance_weight = 0.3
        cost_weight = 0.2
        performance_weight = 0.1
        operations_weight = 0.1
    }
}

Test-ApiEndpoint -EndpointName "Patent 2: Unified AI Analysis" -Url "http://localhost:8002/api/v1/unified-ai/analyze" -Method "POST" -Body $unifiedAiTest

# Test Patent 3: Conversational AI
$conversationTest = @{
    user_input = "What are the current security policies for virtual machines?"
    session_id = "test_session_001"
    user_id = "test_user"
}

Test-ApiEndpoint -EndpointName "Patent 3: Conversational AI" -Url "http://localhost:8002/api/v1/conversation/governance" -Method "POST" -Body $conversationTest

# Test Patent 3: Policy Synthesis
$policySynthesisTest = @{
    request_id = "policy_test_001"
    description = "Create a security policy that blocks all unauthorized network access"
    domain = "security"
    policy_type = "network"
}

Test-ApiEndpoint -EndpointName "Patent 3: Policy Synthesis" -Url "http://localhost:8002/api/v1/conversation/policy-synthesis" -Method "POST" -Body $policySynthesisTest

# Step 6: Test Frontend Access
Write-Host "`n🌐 Step 6: Testing Frontend Access" -ForegroundColor Cyan

try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 10
    if ($frontendResponse.StatusCode -eq 200) {
        Write-Host "✅ Frontend is accessible at http://localhost:5173" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ Frontend is not accessible" -ForegroundColor Red
}

# Step 7: Integration Test
Write-Host "`n🔗 Step 7: Frontend-Backend Integration Test" -ForegroundColor Cyan

# Test API Gateway routing
$apiGatewayTests = @(
    @{ Name = "API Gateway Health"; Url = "http://localhost:8000/health" },
    @{ Name = "AI Engine via Gateway"; Url = "http://localhost:8000/api/v1/ai/health" },
    @{ Name = "Conversation via Gateway"; Url = "http://localhost:8000/api/v1/conversation/health" }
)

foreach ($test in $apiGatewayTests) {
    Test-ApiEndpoint -EndpointName $test.Name -Url $test.Url
}

# Final Results
Write-Host "`n🎯 Local Testing Results Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "• Services Health: $healthyServices/$($services.Count)" -ForegroundColor White
Write-Host "• Frontend: $(if (Test-Path "http://localhost:5173") { 'Accessible' } else { 'Check manually' })" -ForegroundColor White
Write-Host "• Patent APIs: Check test results above" -ForegroundColor White

Write-Host "`n🌟 Next Steps:" -ForegroundColor Green
Write-Host "1. Open http://localhost:5173 in your browser" -ForegroundColor White
Write-Host "2. Navigate to the AI Assistant page to test conversational AI" -ForegroundColor White
Write-Host "3. Check Docker logs if any service failed: 'docker-compose -f docker-compose.local.yml logs'" -ForegroundColor White
Write-Host "4. Use Postman/curl to test individual patent endpoints" -ForegroundColor White

Write-Host "`nLocal testing setup complete!" -ForegroundColor Green