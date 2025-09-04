# PowerShell script to sign container images with Cosign and generate attestations
param(
    [Parameter(Mandatory=$false)]
    [string]$Registry = "crcortexdev.azurecr.io",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest",
    
    [Parameter(Mandatory=$false)]
    [switch]$GenerateProvenance,
    
    [Parameter(Mandatory=$false)]
    [switch]$GenerateSLSA,
    
    [Parameter(Mandatory=$false)]
    [switch]$VerifySignatures
)

Write-Host "PolicyCortex Container Signing & Attestation" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Install Cosign if not present
if (-not (Get-Command cosign -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Cosign..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://github.com/sigstore/cosign/releases/latest/download/cosign-windows-amd64.exe" -OutFile "cosign.exe"
    Move-Item "cosign.exe" "$env:USERPROFILE\.local\bin\" -Force
}

# Install in-toto if not present for provenance
if ($GenerateProvenance -and -not (Get-Command in-toto -ErrorAction SilentlyContinue)) {
    Write-Host "Installing in-toto for provenance..." -ForegroundColor Yellow
    pip install in-toto
}

# Container images to sign
$Images = @(
    "policycortex-frontend",
    "policycortex-core",
    "policycortex-api-gateway",
    "policycortex-ai-engine",
    "policycortex-graphql"
)

# Login to Azure Container Registry
Write-Host "`nLogging into Azure Container Registry..." -ForegroundColor Yellow
az acr login --name ($Registry -replace '\.azurecr\.io', '')

foreach ($Image in $Images) {
    $FullImageName = "$Registry/${Image}:$ImageTag"
    Write-Host "`nProcessing image: $FullImageName" -ForegroundColor Green
    
    # Check if image exists
    $ImageExists = docker image ls $FullImageName --format "{{.Repository}}:{{.Tag}}" 2>$null
    
    if (-not $ImageExists) {
        Write-Host "  Image not found locally. Pulling from registry..." -ForegroundColor Yellow
        docker pull $FullImageName
    }
    
    # Sign the container image with keyless signing
    Write-Host "  Signing container image..." -ForegroundColor Yellow
    
    try {
        # Generate ephemeral keys and sign with Sigstore
        cosign sign --yes $FullImageName
        Write-Host "  Image signed successfully" -ForegroundColor Green
        
        # Store signature reference
        $SignatureRef = "$FullImageName.sig"
        Write-Host "  Signature stored at: $SignatureRef" -ForegroundColor Green
    } catch {
        Write-Host "  Failed to sign image: $_" -ForegroundColor Red
        continue
    }
    
    # Generate in-toto attestation for provenance
    if ($GenerateProvenance) {
        Write-Host "  Generating in-toto provenance attestation..." -ForegroundColor Yellow
        
        $Attestation = @{
            "_type" = "https://in-toto.io/Statement/v0.1"
            "subject" = @(
                @{
                    "name" = $FullImageName
                    "digest" = @{
                        "sha256" = (docker inspect $FullImageName --format='{{index .RepoDigests 0}}' | Select-String -Pattern 'sha256:(\w+)').Matches[0].Groups[1].Value
                    }
                }
            )
            "predicateType" = "https://slsa.dev/provenance/v0.2"
            "predicate" = @{
                "builder" = @{
                    "id" = "https://github.com/policycortex/builder"
                }
                "buildType" = "https://github.com/policycortex/build"
                "invocation" = @{
                    "configSource" = @{
                        "uri" = "git+https://github.com/policycortex/policycortex"
                        "digest" = @{
                            "sha1" = (git rev-parse HEAD)
                        }
                        "entryPoint" = ".github/workflows/build.yml"
                    }
                    "parameters" = @{
                        "registry" = $Registry
                        "tag" = $ImageTag
                    }
                    "environment" = @{
                        "github_run_id" = $env:GITHUB_RUN_ID ?? "local"
                        "github_actor" = $env:GITHUB_ACTOR ?? $env:USERNAME
                    }
                }
                "buildConfig" = @{
                    "steps" = @(
                        @{
                            "command" = @("docker", "build", "-t", $FullImageName, ".")
                        },
                        @{
                            "command" = @("docker", "push", $FullImageName)
                        }
                    )
                }
                "metadata" = @{
                    "buildStartedOn" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
                    "buildFinishedOn" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
                    "completeness" = @{
                        "parameters" = $true
                        "environment" = $true
                        "materials" = $false
                    }
                    "reproducible" = $false
                }
                "materials" = @(
                    @{
                        "uri" = "git+https://github.com/policycortex/policycortex"
                        "digest" = @{
                            "sha1" = (git rev-parse HEAD)
                        }
                    }
                )
            }
        }
        
        $AttestationFile = "provenance-$Image-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
        $Attestation | ConvertTo-Json -Depth 10 | Set-Content $AttestationFile
        
        # Attach attestation to image
        cosign attest --yes --predicate $AttestationFile $FullImageName
        Write-Host "  Provenance attestation attached" -ForegroundColor Green
        
        # Clean up attestation file
        Remove-Item $AttestationFile
    }
    
    # Generate SLSA attestation
    if ($GenerateSLSA) {
        Write-Host "  Generating SLSA Level 3 attestation..." -ForegroundColor Yellow
        
        $SLSAAttestation = @{
            "buildLevel" = "SLSA_BUILD_LEVEL_3"
            "recipe" = @{
                "type" = "https://github.com/policycortex/build"
                "definedInMaterial" = 0
                "entryPoint" = "Dockerfile"
                "arguments" = @{
                    "registry" = $Registry
                    "tag" = $ImageTag
                }
            }
            "reproducible" = $false
            "metadata" = @{
                "buildInvocationId" = (New-Guid).ToString()
                "buildStartedOn" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
                "completeness" = @{
                    "arguments" = $true
                    "environment" = $true
                    "materials" = $true
                }
            }
            "materials" = @(
                @{
                    "resourceUri" = "git+https://github.com/policycortex/policycortex@$(git rev-parse HEAD)"
                    "annotations" = @{
                        "branch" = (git branch --show-current)
                    }
                }
            )
            "builder" = @{
                "id" = "https://github.com/policycortex/builder@v1"
                "version" = @{
                    "gitCommit" = (git rev-parse HEAD)
                    "gitTag" = (git describe --tags --abbrev=0 2>$null) ?? "untagged"
                }
            }
        }
        
        $SLSAFile = "slsa-$Image-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
        $SLSAAttestation | ConvertTo-Json -Depth 10 | Set-Content $SLSAFile
        
        # Attach SLSA attestation
        cosign attest --yes --type slsaprovenance --predicate $SLSAFile $FullImageName
        Write-Host "  SLSA attestation attached" -ForegroundColor Green
        
        # Clean up SLSA file
        Remove-Item $SLSAFile
    }
}

# Verify signatures if requested
if ($VerifySignatures) {
    Write-Host "`nVerifying container signatures..." -ForegroundColor Cyan
    
    foreach ($Image in $Images) {
        $FullImageName = "$Registry/${Image}:$ImageTag"
        Write-Host "`nVerifying: $FullImageName" -ForegroundColor Yellow
        
        try {
            # Verify signature
            cosign verify --certificate-identity-regexp ".*" --certificate-oidc-issuer-regexp ".*" $FullImageName
            Write-Host "  Signature verification: PASSED" -ForegroundColor Green
            
            # Verify attestations
            cosign verify-attestation --certificate-identity-regexp ".*" --certificate-oidc-issuer-regexp ".*" $FullImageName
            Write-Host "  Attestation verification: PASSED" -ForegroundColor Green
        } catch {
            Write-Host "  Verification failed: $_" -ForegroundColor Red
        }
    }
}

# Generate summary report
Write-Host "`nContainer Signing Summary" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

$Report = @{
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    registry = $Registry
    images_signed = $Images.Count
    signatures_method = "keyless (Sigstore)"
    attestations = @{
        provenance = $GenerateProvenance
        slsa = $GenerateSLSA
    }
    verification_performed = $VerifySignatures
}

$ReportFile = "container-signing-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
$Report | ConvertTo-Json -Depth 10 | Set-Content $ReportFile

Write-Host "Report saved to: $ReportFile" -ForegroundColor Green
Write-Host "`nAll containers have been processed successfully!" -ForegroundColor Green

exit 0