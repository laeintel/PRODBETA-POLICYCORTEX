# PowerShell script to generate Software Bill of Materials (SBOM) in CycloneDX format
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("json", "xml")]
    [string]$Format = "json",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath = ".",
    
    [Parameter(Mandatory=$false)]
    [switch]$SignSBOM,
    
    [Parameter(Mandatory=$false)]
    [switch]$UploadToRegistry
)

Write-Host "PolicyCortex SBOM Generation" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SBOMVersion = "1.5"

# Install required tools if not present
function Install-RequiredTools {
    Write-Host "Checking for required tools..." -ForegroundColor Yellow
    
    # Check for syft
    if (-not (Get-Command syft -ErrorAction SilentlyContinue)) {
        Write-Host "Installing Syft for SBOM generation..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/anchore/syft/main/install.sh" -OutFile "install-syft.sh"
        bash install-syft.sh
        Remove-Item "install-syft.sh"
    }
    
    # Check for cyclonedx-cli
    if (-not (Get-Command cyclonedx -ErrorAction SilentlyContinue)) {
        Write-Host "Installing CycloneDX CLI..." -ForegroundColor Yellow
        dotnet tool install --global CycloneDX.CLI
    }
    
    # Check for cosign
    if ($SignSBOM -and -not (Get-Command cosign -ErrorAction SilentlyContinue)) {
        Write-Host "Installing Cosign for signing..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://github.com/sigstore/cosign/releases/latest/download/cosign-windows-amd64.exe" -OutFile "cosign.exe"
        Move-Item "cosign.exe" "$env:USERPROFILE\.local\bin\" -Force
    }
}

Install-RequiredTools

# Generate SBOM for different components
$Components = @(
    @{Name="Frontend"; Path=".\frontend"; Type="npm"},
    @{Name="Core"; Path=".\core"; Type="cargo"},
    @{Name="API-Gateway"; Path=".\backend\services\api_gateway"; Type="python"},
    @{Name="AI-Engine"; Path=".\backend\services\ai_engine"; Type="python"},
    @{Name="GraphQL"; Path=".\graphql"; Type="npm"}
)

$AllSBOMs = @()

foreach ($Component in $Components) {
    Write-Host "`nGenerating SBOM for $($Component.Name)..." -ForegroundColor Green
    
    $ComponentSBOM = Join-Path $OutputPath "sbom-$($Component.Name.ToLower())-$Timestamp.$Format"
    
    switch ($Component.Type) {
        "npm" {
            # Generate SBOM for Node.js projects
            Push-Location $Component.Path
            
            if (Test-Path "package.json") {
                # Use npm to generate SBOM
                npm list --json --all > npm-deps.json
                
                # Convert to CycloneDX format
                npx @cyclonedx/cyclonedx-npm --output-file $ComponentSBOM
                
                Remove-Item "npm-deps.json" -ErrorAction SilentlyContinue
            }
            
            Pop-Location
        }
        "cargo" {
            # Generate SBOM for Rust projects
            Push-Location $Component.Path
            
            if (Test-Path "Cargo.toml") {
                # Use cargo-cyclonedx if available, otherwise use syft
                if (Get-Command cargo-cyclonedx -ErrorAction SilentlyContinue) {
                    cargo cyclonedx --format $Format --output $ComponentSBOM
                } else {
                    syft . -o cyclonedx-$Format=$ComponentSBOM
                }
            }
            
            Pop-Location
        }
        "python" {
            # Generate SBOM for Python projects
            Push-Location $Component.Path
            
            if (Test-Path "requirements.txt") {
                # Use pip to generate dependency list
                pip list --format json > pip-deps.json
                
                # Use syft for Python SBOM
                syft . -o cyclonedx-$Format=$ComponentSBOM
                
                Remove-Item "pip-deps.json" -ErrorAction SilentlyContinue
            }
            
            Pop-Location
        }
    }
    
    if (Test-Path $ComponentSBOM) {
        Write-Host "  SBOM generated: $ComponentSBOM" -ForegroundColor Green
        $AllSBOMs += $ComponentSBOM
    }
}

# Generate container SBOMs
Write-Host "`nGenerating Container SBOMs..." -ForegroundColor Green

$ContainerImages = @(
    "policycortex-frontend:latest",
    "policycortex-core:latest",
    "policycortex-api-gateway:latest",
    "policycortex-ai-engine:latest"
)

foreach ($Image in $ContainerImages) {
    $ContainerSBOM = Join-Path $OutputPath "sbom-container-$($Image.Replace(':', '-'))-$Timestamp.$Format"
    
    # Check if image exists locally
    $ImageExists = docker image ls $Image --format "{{.Repository}}:{{.Tag}}" 2>$null
    
    if ($ImageExists) {
        Write-Host "  Generating SBOM for container: $Image" -ForegroundColor Yellow
        syft $Image -o cyclonedx-$Format=$ContainerSBOM
        
        if (Test-Path $ContainerSBOM) {
            Write-Host "  Container SBOM generated: $ContainerSBOM" -ForegroundColor Green
            $AllSBOMs += $ContainerSBOM
        }
    } else {
        Write-Host "  Container image not found: $Image" -ForegroundColor Red
    }
}

# Merge all SBOMs into a single comprehensive SBOM
$MergedSBOM = Join-Path $OutputPath "sbom-policycortex-complete-$Timestamp.$Format"
Write-Host "`nMerging all SBOMs into comprehensive SBOM..." -ForegroundColor Green

if ($AllSBOMs.Count -gt 0) {
    # Use CycloneDX CLI to merge
    $MergeCommand = "cyclonedx merge --input-files $($AllSBOMs -join ',') --output-file $MergedSBOM"
    Invoke-Expression $MergeCommand
    
    Write-Host "Comprehensive SBOM created: $MergedSBOM" -ForegroundColor Green
}

# Generate SaaSBOM (for cloud services dependencies)
Write-Host "`nGenerating SaaSBOM for cloud services..." -ForegroundColor Green

$SaaSBOM = @{
    bomFormat = "CycloneDX"
    specVersion = $SBOMVersion
    serialNumber = "urn:uuid:$(New-Guid)"
    version = 1
    metadata = @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        tools = @(
            @{
                vendor = "PolicyCortex"
                name = "SBOM Generator"
                version = "1.0.0"
            }
        )
    }
    services = @(
        @{
            bom_ref = "azure-subscription"
            provider = @{
                name = "Microsoft Azure"
                url = @("https://azure.microsoft.com")
            }
            name = "Azure Subscription"
            version = "2024"
            description = "Azure cloud services subscription"
            data = @(
                @{
                    flow = "bi-directional"
                    classification = "PII,Business"
                }
            )
        },
        @{
            bom_ref = "azure-ad"
            provider = @{
                name = "Microsoft"
                url = @("https://azure.microsoft.com/services/active-directory")
            }
            name = "Azure Active Directory"
            version = "2.0"
            description = "Identity and access management"
        },
        @{
            bom_ref = "sendgrid"
            provider = @{
                name = "SendGrid"
                url = @("https://sendgrid.com")
            }
            name = "SendGrid Email Service"
            version = "3.0"
            description = "Transactional email service"
        },
        @{
            bom_ref = "pagerduty"
            provider = @{
                name = "PagerDuty"
                url = @("https://pagerduty.com")
            }
            name = "PagerDuty"
            version = "2.0"
            description = "Incident management platform"
        }
    )
    dependencies = @(
        @{
            ref = "policycortex-app"
            dependsOn = @("azure-subscription", "azure-ad", "sendgrid", "pagerduty")
        }
    )
}

$SaaSBOMPath = Join-Path $OutputPath "saasbom-policycortex-$Timestamp.json"
$SaaSBOM | ConvertTo-Json -Depth 10 | Set-Content $SaaSBOMPath
Write-Host "SaaSBOM created: $SaaSBOMPath" -ForegroundColor Green

# Sign SBOMs if requested
if ($SignSBOM) {
    Write-Host "`nSigning SBOMs with Cosign..." -ForegroundColor Green
    
    foreach ($SBOM in $AllSBOMs + $MergedSBOM + $SaaSBOMPath) {
        if (Test-Path $SBOM) {
            Write-Host "  Signing: $SBOM" -ForegroundColor Yellow
            
            # Use keyless signing with Fulcio
            cosign sign-blob --yes $SBOM --output-signature "$SBOM.sig" --output-certificate "$SBOM.crt"
            
            if (Test-Path "$SBOM.sig") {
                Write-Host "  Signature created: $SBOM.sig" -ForegroundColor Green
            }
        }
    }
}

# Generate vulnerability report
Write-Host "`nScanning for vulnerabilities..." -ForegroundColor Green

$VulnReport = Join-Path $OutputPath "vulnerability-report-$Timestamp.json"

if (Test-Path $MergedSBOM) {
    # Use Grype to scan SBOM for vulnerabilities
    if (Get-Command grype -ErrorAction SilentlyContinue) {
        grype sbom:$MergedSBOM -o json > $VulnReport
        
        # Parse and display summary
        $Vulns = Get-Content $VulnReport | ConvertFrom-Json
        $Critical = ($Vulns.matches | Where-Object { $_.vulnerability.severity -eq "Critical" }).Count
        $High = ($Vulns.matches | Where-Object { $_.vulnerability.severity -eq "High" }).Count
        $Medium = ($Vulns.matches | Where-Object { $_.vulnerability.severity -eq "Medium" }).Count
        $Low = ($Vulns.matches | Where-Object { $_.vulnerability.severity -eq "Low" }).Count
        
        Write-Host "`nVulnerability Summary:" -ForegroundColor Cyan
        Write-Host "  Critical: $Critical" -ForegroundColor Red
        Write-Host "  High: $High" -ForegroundColor Magenta
        Write-Host "  Medium: $Medium" -ForegroundColor Yellow
        Write-Host "  Low: $Low" -ForegroundColor Green
        Write-Host "  Full report: $VulnReport" -ForegroundColor White
    } else {
        Write-Host "Grype not installed. Skipping vulnerability scan." -ForegroundColor Yellow
    }
}

# Upload to registry if requested
if ($UploadToRegistry) {
    Write-Host "`nUploading SBOMs to container registry..." -ForegroundColor Green
    
    $RegistryUrl = "crcortexdev.azurecr.io"
    
    # Login to registry
    az acr login --name crcortexdev
    
    # Upload SBOMs as OCI artifacts
    foreach ($SBOM in @($MergedSBOM, $SaaSBOMPath)) {
        if (Test-Path $SBOM) {
            $ArtifactName = [System.IO.Path]::GetFileNameWithoutExtension($SBOM)
            
            # Use ORAS to push SBOM as OCI artifact
            oras push "$RegistryUrl/sbom/$ArtifactName:$Timestamp" $SBOM
            
            Write-Host "  Uploaded: $RegistryUrl/sbom/$ArtifactName:$Timestamp" -ForegroundColor Green
        }
    }
}

Write-Host "`nSBOM generation complete!" -ForegroundColor Cyan
Write-Host "Generated files:" -ForegroundColor Yellow
$AllSBOMs + $MergedSBOM + $SaaSBOMPath | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "  - $_" -ForegroundColor White
    }
}

exit 0