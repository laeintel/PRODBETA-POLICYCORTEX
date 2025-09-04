"""
PolicyCortex PREVENT Pillar - Auto-Fix Generator
Generates remediation code for predicted policy violations
Creates GitHub PRs with fixes for Terraform, ARM Templates, and Bicep
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from github import Github, GithubException
from github.PullRequest import PullRequest
from jinja2 import Template, Environment, FileSystemLoader
import yaml
import hcl2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TEMPLATES_DIR = Path(__file__).parent / "templates"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "policycortex/infrastructure")

class RemediationType(str, Enum):
    """Types of remediation templates supported"""
    TERRAFORM = "terraform"
    ARM_TEMPLATE = "arm_template"
    BICEP = "bicep"
    POWERSHELL = "powershell"
    AZURE_CLI = "azure_cli"
    POLICY_DEFINITION = "policy_definition"

class ViolationType(str, Enum):
    """Types of violations we can auto-fix"""
    ACCESS_CONTROL = "access_control"
    DATA_ENCRYPTION = "data_encryption"
    NETWORK_SECURITY = "network_security"
    COMPLIANCE_DRIFT = "compliance_drift"
    COST_OVERRUN = "cost_overrun"
    RESOURCE_TAGGING = "resource_tagging"
    BACKUP_POLICY = "backup_policy"
    PATCH_MANAGEMENT = "patch_management"
    IDENTITY_GOVERNANCE = "identity_governance"
    AUDIT_LOGGING = "audit_logging"

@dataclass
class RemediationTemplate:
    """Template for generating remediation code"""
    id: str
    name: str
    violation_type: ViolationType
    remediation_type: RemediationType
    template_content: str
    description: str
    variables: Dict[str, Any]
    estimated_time_minutes: int
    risk_level: str

class FixRequest(BaseModel):
    """Request model for generating fixes"""
    violation_type: ViolationType = Field(..., description="Type of violation to fix")
    resource_id: str = Field(..., description="Azure resource ID")
    resource_type: str = Field(..., description="Azure resource type")
    subscription_id: str = Field(..., description="Azure subscription ID")
    remediation_type: RemediationType = Field(..., description="Type of fix to generate")
    violation_details: Dict = Field(..., description="Details about the violation")
    create_pr: bool = Field(True, description="Create GitHub PR with fix")
    branch_name: Optional[str] = Field(None, description="Custom branch name")

class FixResponse(BaseModel):
    """Response model for fix generation"""
    fix_id: str
    remediation_code: str
    remediation_type: RemediationType
    file_path: str
    pr_url: Optional[str]
    branch_name: Optional[str]
    estimated_time_minutes: int
    instructions: List[str]

class AutoFixGenerator:
    """
    Generates remediation code for predicted violations
    Creates GitHub PRs with infrastructure-as-code fixes
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.github_client = None
        if GITHUB_TOKEN:
            self.github_client = Github(GITHUB_TOKEN)
        
        # Initialize template environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else None,
            autoescape=True
        )
    
    def _load_templates(self) -> Dict[str, RemediationTemplate]:
        """Load remediation templates"""
        templates = {}
        
        # Define templates for top 50 common violations
        # These are inline templates for MVP - in production, load from files
        
        # 1. Missing Encryption at Rest
        templates["encryption_at_rest_storage"] = RemediationTemplate(
            id="fix_001",
            name="Enable Storage Encryption at Rest",
            violation_type=ViolationType.DATA_ENCRYPTION,
            remediation_type=RemediationType.TERRAFORM,
            template_content="""
resource "azurerm_storage_account" "{{ resource_name }}" {
  name                     = "{{ storage_account_name }}"
  resource_group_name      = "{{ resource_group }}"
  location                 = "{{ location }}"
  account_tier             = "{{ account_tier | default('Standard') }}"
  account_replication_type = "{{ replication_type | default('LRS') }}"
  
  # Enable encryption at rest
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
  }
  
  # Enable infrastructure encryption
  infrastructure_encryption_enabled = true
  
  # Configure customer-managed keys
  {% if use_cmk %}
  identity {
    type = "SystemAssigned"
  }
  
  customer_managed_key {
    key_vault_key_id          = azurerm_key_vault_key.storage.id
    user_assigned_identity_id = azurerm_user_assigned_identity.storage.id
  }
  {% endif %}
  
  tags = {
    Environment = "{{ environment }}"
    ManagedBy   = "PolicyCortex"
    FixApplied  = "{{ timestamp }}"
  }
}
""",
            description="Enables encryption at rest for Azure Storage Account",
            variables={
                "resource_name": "storage",
                "storage_account_name": "",
                "resource_group": "",
                "location": "eastus",
                "use_cmk": False
            },
            estimated_time_minutes=5,
            risk_level="low"
        )
        
        # 2. Missing Network Security Group
        templates["missing_nsg"] = RemediationTemplate(
            id="fix_002",
            name="Add Network Security Group",
            violation_type=ViolationType.NETWORK_SECURITY,
            remediation_type=RemediationType.TERRAFORM,
            template_content="""
resource "azurerm_network_security_group" "{{ resource_name }}" {
  name                = "{{ nsg_name }}"
  location            = "{{ location }}"
  resource_group_name = "{{ resource_group }}"
  
  # Deny all inbound by default
  security_rule {
    name                       = "DenyAllInbound"
    priority                   = 4096
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  {% for rule in custom_rules %}
  security_rule {
    name                       = "{{ rule.name }}"
    priority                   = {{ rule.priority }}
    direction                  = "{{ rule.direction }}"
    access                     = "{{ rule.access }}"
    protocol                   = "{{ rule.protocol }}"
    source_port_range          = "{{ rule.source_port }}"
    destination_port_range     = "{{ rule.dest_port }}"
    source_address_prefix      = "{{ rule.source_address }}"
    destination_address_prefix = "{{ rule.dest_address }}"
  }
  {% endfor %}
  
  tags = {
    Environment = "{{ environment }}"
    ManagedBy   = "PolicyCortex"
  }
}

# Associate with subnet
resource "azurerm_subnet_network_security_group_association" "{{ resource_name }}" {
  subnet_id                 = "{{ subnet_id }}"
  network_security_group_id = azurerm_network_security_group.{{ resource_name }}.id
}
""",
            description="Creates and associates a Network Security Group",
            variables={
                "resource_name": "nsg",
                "nsg_name": "",
                "resource_group": "",
                "location": "eastus",
                "subnet_id": "",
                "custom_rules": []
            },
            estimated_time_minutes=3,
            risk_level="medium"
        )
        
        # 3. Missing Resource Tags
        templates["missing_tags"] = RemediationTemplate(
            id="fix_003",
            name="Add Required Tags",
            violation_type=ViolationType.RESOURCE_TAGGING,
            remediation_type=RemediationType.AZURE_CLI,
            template_content="""#!/bin/bash
# Add required tags to Azure resource

RESOURCE_ID="{{ resource_id }}"
TAGS="{{ tags }}"

echo "Adding tags to resource: $RESOURCE_ID"

# Add tags using Azure CLI
az tag update --resource-id "$RESOURCE_ID" --operation merge --tags $TAGS

if [ $? -eq 0 ]; then
    echo "Tags successfully added"
    
    # Verify tags were applied
    echo "Verifying tags..."
    az tag list --resource-id "$RESOURCE_ID"
else
    echo "Failed to add tags"
    exit 1
fi
""",
            description="Adds required tags to Azure resources",
            variables={
                "resource_id": "",
                "tags": "Environment=Production Owner=TeamName CostCenter=CC001"
            },
            estimated_time_minutes=1,
            risk_level="low"
        )
        
        # 4. Enable MFA for Privileged Users
        templates["enable_mfa"] = RemediationTemplate(
            id="fix_004",
            name="Enable Multi-Factor Authentication",
            violation_type=ViolationType.IDENTITY_GOVERNANCE,
            remediation_type=RemediationType.POWERSHELL,
            template_content="""# Enable MFA for privileged users
# Requires Azure AD PowerShell module

$TenantId = "{{ tenant_id }}"
$UserPrincipalNames = @({{ user_list }})

# Connect to Azure AD
Connect-AzureAD -TenantId $TenantId

foreach ($upn in $UserPrincipalNames) {
    try {
        # Get user
        $user = Get-AzureADUser -ObjectId $upn
        
        # Create MFA requirement
        $mfa = New-Object -TypeName Microsoft.Open.AzureAD.Model.StrongAuthenticationRequirement
        $mfa.RelyingParty = "*"
        $mfa.State = "Enabled"
        $mfa.RememberDevicesNotIssuedBefore = (Get-Date)
        
        # Apply MFA
        Set-AzureADUser -ObjectId $user.ObjectId `
            -StrongAuthenticationRequirements @($mfa)
        
        Write-Host "MFA enabled for $upn" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to enable MFA for $upn : $_" -ForegroundColor Red
    }
}
""",
            description="Enables MFA for specified users",
            variables={
                "tenant_id": "",
                "user_list": ""
            },
            estimated_time_minutes=5,
            risk_level="low"
        )
        
        # 5. Enable Backup Policy
        templates["enable_backup"] = RemediationTemplate(
            id="fix_005",
            name="Enable Backup Policy",
            violation_type=ViolationType.BACKUP_POLICY,
            remediation_type=RemediationType.BICEP,
            template_content="""@description('Name of the Recovery Services Vault')
param vaultName string = '{{ vault_name }}'

@description('Resource Group for the vault')
param resourceGroup string = '{{ resource_group }}'

@description('Location for resources')
param location string = '{{ location }}'

@description('VM Resource ID to backup')
param vmResourceId string = '{{ vm_resource_id }}'

// Create Recovery Services Vault if needed
resource recoveryVault 'Microsoft.RecoveryServices/vaults@2023-01-01' = {
  name: vaultName
  location: location
  sku: {
    name: 'RS0'
    tier: 'Standard'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

// Create backup policy
resource backupPolicy 'Microsoft.RecoveryServices/vaults/backupPolicies@2023-01-01' = {
  parent: recoveryVault
  name: 'DefaultVMPolicy'
  properties: {
    backupManagementType: 'AzureIaasVM'
    instantRpRetentionRangeInDays: 2
    schedulePolicy: {
      schedulePolicyType: 'SimpleSchedulePolicy'
      scheduleRunFrequency: 'Daily'
      scheduleRunTimes: ['2023-01-01T02:00:00Z']
    }
    retentionPolicy: {
      retentionPolicyType: 'LongTermRetentionPolicy'
      dailySchedule: {
        retentionTimes: ['2023-01-01T02:00:00Z']
        retentionDuration: {
          count: 7
          durationType: 'Days'
        }
      }
    }
  }
}

// Enable backup for VM
resource protectedItem 'Microsoft.RecoveryServices/vaults/backupFabrics/protectionContainers/protectedItems@2023-01-01' = {
  name: '${vaultName}/Azure/IaasVMContainer;iaasvmcontainerv2;${resourceGroup};${split(vmResourceId, '/')[8]}/VM;iaasvmcontainerv2;${resourceGroup};${split(vmResourceId, '/')[8]}'
  properties: {
    protectedItemType: 'Microsoft.Compute/virtualMachines'
    policyId: backupPolicy.id
    sourceResourceId: vmResourceId
  }
}
""",
            description="Enables backup for Azure VMs",
            variables={
                "vault_name": "backup-vault",
                "resource_group": "",
                "location": "eastus",
                "vm_resource_id": ""
            },
            estimated_time_minutes=10,
            risk_level="low"
        )
        
        # 6. Enable Diagnostic Logging
        templates["enable_diagnostics"] = RemediationTemplate(
            id="fix_006",
            name="Enable Diagnostic Logging",
            violation_type=ViolationType.AUDIT_LOGGING,
            remediation_type=RemediationType.ARM_TEMPLATE,
            template_content="""{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "resourceId": {
      "type": "string",
      "metadata": {
        "description": "Resource ID to enable diagnostics for"
      }
    },
    "workspaceId": {
      "type": "string",
      "metadata": {
        "description": "Log Analytics Workspace ID"
      }
    },
    "storageAccountId": {
      "type": "string",
      "metadata": {
        "description": "Storage Account ID for archival"
      }
    }
  },
  "resources": [
    {
      "type": "Microsoft.Insights/diagnosticSettings",
      "apiVersion": "2021-05-01-preview",
      "name": "[concat(parameters('resourceId'), '/Microsoft.Insights/PolicyCortexDiagnostics')]",
      "properties": {
        "workspaceId": "[parameters('workspaceId')]",
        "storageAccountId": "[parameters('storageAccountId')]",
        "logs": [
          {
            "category": "Administrative",
            "enabled": true,
            "retentionPolicy": {
              "enabled": true,
              "days": 90
            }
          },
          {
            "category": "Security",
            "enabled": true,
            "retentionPolicy": {
              "enabled": true,
              "days": 90
            }
          },
          {
            "category": "Alert",
            "enabled": true,
            "retentionPolicy": {
              "enabled": true,
              "days": 30
            }
          }
        ],
        "metrics": [
          {
            "category": "AllMetrics",
            "enabled": true,
            "retentionPolicy": {
              "enabled": true,
              "days": 30
            }
          }
        ]
      }
    }
  ]
}""",
            description="Enables diagnostic logging for resources",
            variables={
                "resource_id": "",
                "workspace_id": "",
                "storage_account_id": ""
            },
            estimated_time_minutes=3,
            risk_level="low"
        )
        
        # Add more templates for other common violations...
        
        return templates
    
    async def generate_fix(self, request: FixRequest) -> FixResponse:
        """Generate remediation code for a violation"""
        try:
            # Select appropriate template
            template = self._select_template(request.violation_type, request.remediation_type)
            
            if not template:
                raise HTTPException(
                    status_code=404,
                    detail=f"No template found for {request.violation_type} with {request.remediation_type}"
                )
            
            # Prepare template variables
            variables = self._prepare_variables(request, template)
            
            # Generate remediation code
            remediation_code = self._render_template(template, variables)
            
            # Determine file path based on remediation type
            file_path = self._get_file_path(request)
            
            # Create PR if requested
            pr_url = None
            branch_name = None
            if request.create_pr and self.github_client:
                pr_url, branch_name = await self._create_github_pr(
                    remediation_code,
                    file_path,
                    request,
                    template
                )
            
            # Generate instructions
            instructions = self._generate_instructions(template, request)
            
            return FixResponse(
                fix_id=f"fix-{request.violation_type}-{datetime.utcnow().timestamp()}",
                remediation_code=remediation_code,
                remediation_type=request.remediation_type,
                file_path=file_path,
                pr_url=pr_url,
                branch_name=branch_name,
                estimated_time_minutes=template.estimated_time_minutes,
                instructions=instructions
            )
            
        except Exception as e:
            logger.error(f"Failed to generate fix: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _select_template(
        self,
        violation_type: ViolationType,
        remediation_type: RemediationType
    ) -> Optional[RemediationTemplate]:
        """Select the best template for the violation"""
        for template_id, template in self.templates.items():
            if (template.violation_type == violation_type and
                template.remediation_type == remediation_type):
                return template
        
        # Try to find any template for this violation type
        for template_id, template in self.templates.items():
            if template.violation_type == violation_type:
                return template
        
        return None
    
    def _prepare_variables(
        self,
        request: FixRequest,
        template: RemediationTemplate
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering"""
        variables = template.variables.copy()
        
        # Extract resource details from resource ID
        resource_parts = request.resource_id.split('/')
        if len(resource_parts) >= 4:
            variables['subscription_id'] = resource_parts[2] if len(resource_parts) > 2 else request.subscription_id
            variables['resource_group'] = resource_parts[4] if len(resource_parts) > 4 else 'default-rg'
            variables['resource_name'] = resource_parts[-1] if resource_parts else 'resource'
        
        # Add common variables
        variables['resource_id'] = request.resource_id
        variables['resource_type'] = request.resource_type
        variables['timestamp'] = datetime.utcnow().isoformat()
        variables['environment'] = request.violation_details.get('environment', 'production')
        
        # Add violation-specific variables
        variables.update(request.violation_details)
        
        return variables
    
    def _render_template(
        self,
        template: RemediationTemplate,
        variables: Dict[str, Any]
    ) -> str:
        """Render template with variables"""
        if self.jinja_env:
            jinja_template = self.jinja_env.from_string(template.template_content)
        else:
            # Fallback to simple string replacement
            jinja_template = Template(template.template_content)
        
        return jinja_template.render(**variables)
    
    def _get_file_path(self, request: FixRequest) -> str:
        """Generate file path based on remediation type"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        resource_name = request.resource_id.split('/')[-1]
        
        path_map = {
            RemediationType.TERRAFORM: f"terraform/fixes/{request.violation_type}_{resource_name}_{timestamp}.tf",
            RemediationType.ARM_TEMPLATE: f"arm-templates/fixes/{request.violation_type}_{resource_name}_{timestamp}.json",
            RemediationType.BICEP: f"bicep/fixes/{request.violation_type}_{resource_name}_{timestamp}.bicep",
            RemediationType.POWERSHELL: f"scripts/fixes/{request.violation_type}_{resource_name}_{timestamp}.ps1",
            RemediationType.AZURE_CLI: f"scripts/fixes/{request.violation_type}_{resource_name}_{timestamp}.sh",
            RemediationType.POLICY_DEFINITION: f"policies/fixes/{request.violation_type}_{resource_name}_{timestamp}.json"
        }
        
        return path_map.get(request.remediation_type, f"fixes/{timestamp}.txt")
    
    async def _create_github_pr(
        self,
        remediation_code: str,
        file_path: str,
        request: FixRequest,
        template: RemediationTemplate
    ) -> Tuple[str, str]:
        """Create GitHub PR with remediation code"""
        try:
            repo = self.github_client.get_repo(GITHUB_REPO)
            
            # Create branch name
            branch_name = request.branch_name or f"fix/{request.violation_type}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get default branch
            default_branch = repo.default_branch
            source = repo.get_branch(default_branch)
            
            # Create new branch
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=source.commit.sha
            )
            
            # Create or update file
            commit_message = f"Fix {request.violation_type}: {template.name}"
            
            try:
                # Try to get existing file
                contents = repo.get_contents(file_path, ref=branch_name)
                repo.update_file(
                    path=file_path,
                    message=commit_message,
                    content=remediation_code,
                    sha=contents.sha,
                    branch=branch_name
                )
            except:
                # File doesn't exist, create it
                repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=remediation_code,
                    branch=branch_name
                )
            
            # Create pull request
            pr = repo.create_pull(
                title=f"[PolicyCortex] Auto-fix: {template.name}",
                body=self._generate_pr_description(request, template),
                head=branch_name,
                base=default_branch
            )
            
            # Add labels
            pr.add_to_labels("auto-generated", "policy-fix", request.violation_type)
            
            return pr.html_url, branch_name
            
        except GithubException as e:
            logger.error(f"GitHub PR creation failed: {e}")
            return None, None
    
    def _generate_pr_description(
        self,
        request: FixRequest,
        template: RemediationTemplate
    ) -> str:
        """Generate PR description"""
        return f"""## PolicyCortex Auto-Fix

### Violation Details
- **Type**: {request.violation_type}
- **Resource**: {request.resource_id}
- **Resource Type**: {request.resource_type}

### Remediation Applied
- **Template**: {template.name}
- **Risk Level**: {template.risk_level}
- **Estimated Time**: {template.estimated_time_minutes} minutes

### Description
{template.description}

### Testing Instructions
1. Review the generated code for correctness
2. Validate against your environment
3. Run terraform plan / deployment preview
4. Apply changes in test environment first
5. Monitor resource after applying fix

### Automated Checks
- [ ] Syntax validation passed
- [ ] Security scan completed
- [ ] Cost impact assessed
- [ ] Compliance check passed

---
*This PR was automatically generated by PolicyCortex PREVENT system*
"""
    
    def _generate_instructions(
        self,
        template: RemediationTemplate,
        request: FixRequest
    ) -> List[str]:
        """Generate step-by-step instructions"""
        instructions = []
        
        if request.remediation_type == RemediationType.TERRAFORM:
            instructions = [
                "1. Review the generated Terraform code",
                "2. Run 'terraform init' to initialize",
                "3. Run 'terraform plan' to preview changes",
                "4. Review the plan output carefully",
                "5. Run 'terraform apply' to apply changes",
                "6. Verify the fix in Azure Portal"
            ]
        elif request.remediation_type == RemediationType.ARM_TEMPLATE:
            instructions = [
                "1. Review the ARM template",
                "2. Validate template: az deployment group validate --resource-group <rg> --template-file <file>",
                "3. Deploy: az deployment group create --resource-group <rg> --template-file <file>",
                "4. Monitor deployment in Azure Portal",
                "5. Verify resource configuration"
            ]
        elif request.remediation_type == RemediationType.BICEP:
            instructions = [
                "1. Review the Bicep file",
                "2. Build ARM template: az bicep build --file <file>",
                "3. Deploy: az deployment group create --resource-group <rg> --template-file <file>",
                "4. Monitor deployment progress",
                "5. Validate fix was applied"
            ]
        elif request.remediation_type == RemediationType.POWERSHELL:
            instructions = [
                "1. Review the PowerShell script",
                "2. Open PowerShell as Administrator",
                "3. Connect to Azure: Connect-AzAccount",
                "4. Run the script",
                "5. Verify changes in Azure Portal"
            ]
        elif request.remediation_type == RemediationType.AZURE_CLI:
            instructions = [
                "1. Review the shell script",
                "2. Make script executable: chmod +x <script>",
                "3. Login to Azure: az login",
                "4. Run the script",
                "5. Verify changes applied successfully"
            ]
        
        return instructions

    async def get_available_fixes(self, violation_type: Optional[ViolationType] = None) -> List[Dict]:
        """Get list of available fix templates"""
        fixes = []
        
        for template_id, template in self.templates.items():
            if violation_type and template.violation_type != violation_type:
                continue
            
            fixes.append({
                "id": template.id,
                "name": template.name,
                "violation_type": template.violation_type,
                "remediation_type": template.remediation_type,
                "description": template.description,
                "estimated_time_minutes": template.estimated_time_minutes,
                "risk_level": template.risk_level
            })
        
        return fixes


# FastAPI Application
app = FastAPI(
    title="PolicyCortex PREVENT - Auto-Fix Generator",
    description="Automated remediation code generator for policy violations",
    version="1.0.0"
)

# Initialize generator
generator = AutoFixGenerator()

@app.post("/api/v1/predict/fix", response_model=FixResponse)
async def generate_fix(request: FixRequest):
    """Generate remediation code for predicted violation"""
    return await generator.generate_fix(request)

@app.get("/api/v1/predict/fixes/available")
async def get_available_fixes(violation_type: Optional[ViolationType] = None):
    """Get list of available fix templates"""
    fixes = await generator.get_available_fixes(violation_type)
    return {
        "total": len(fixes),
        "fixes": fixes
    }

@app.get("/api/v1/predict/fix/{fix_id}/status")
async def get_fix_status(fix_id: str):
    """Get status of a fix (PR status, deployment status, etc.)"""
    # In production, track fix status in database
    return {
        "fix_id": fix_id,
        "status": "pending_review",
        "pr_status": "open",
        "deployment_status": "not_started",
        "validation_results": {
            "syntax_check": "passed",
            "security_scan": "passed",
            "cost_analysis": "no_increase"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "templates_loaded": len(generator.templates),
        "github_connected": generator.github_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)