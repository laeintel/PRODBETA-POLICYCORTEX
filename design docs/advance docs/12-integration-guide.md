# Integration Guide

## Table of Contents
1. [Integration Architecture](#integration-architecture)
2. [Azure Integration](#azure-integration)
3. [Third-Party Integrations](#third-party-integrations)
4. [API Integration](#api-integration)
5. [Webhook Integration](#webhook-integration)
6. [SDK and Client Libraries](#sdk-and-client-libraries)
7. [Enterprise Integrations](#enterprise-integrations)
8. [Custom Extensions](#custom-extensions)
9. [Migration Strategies](#migration-strategies)

## Integration Architecture

PolicyCortex provides comprehensive integration capabilities designed for enterprise environments with multiple systems and complex workflows:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     Azure       │  │   Third-Party   │  │   Enterprise    │  │
│  │  Integration    │  │  Integrations   │  │  Systems        │  │
│  │                 │  │                 │  │                 │  │
│  │ • ARM Templates │  │ • Terraform     │  │ • ServiceNow    │  │
│  │ • Resource Mgr  │  │ • Ansible       │  │ • Jira          │  │
│  │ • Security Ctr  │  │ • Jenkins       │  │ • Slack/Teams   │  │
│  │ • Monitor       │  │ • Splunk        │  │ • PagerDuty     │  │
│  │ • Cost Mgmt     │  │ • Datadog       │  │ • Active Directory│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │        │
│           └─────────────────────┼─────────────────────┘        │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              PolicyCortex Integration Layer               │  │
│  │                                                             │  │
│  │ • REST APIs        • GraphQL        • Webhooks             │  │
│  │ • SDK/Libraries    • Message Queue  • Event Streaming      │  │
│  │ • SCIM            • SAML/OIDC       • Custom Connectors    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Patterns

1. **Pull-based Integration**: Regular polling for data updates
2. **Push-based Integration**: Real-time webhooks and events
3. **Batch Integration**: Scheduled bulk data processing
4. **Stream Integration**: Real-time event streaming
5. **Hybrid Integration**: Combination of multiple patterns

## Azure Integration

### Azure Resource Manager Integration

```rust
// core/src/integrations/azure/resource_manager.rs
use azure_mgmt_resources::Client as ResourceClient;
use azure_identity::DefaultAzureCredential;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct AzureResourceManagerIntegration {
    client: ResourceClient,
    subscription_cache: HashMap<String, SubscriptionInfo>,
    resource_cache: HashMap<String, Vec<AzureResource>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    pub subscription_id: String,
    pub display_name: String,
    pub state: String,
    pub tenant_id: String,
    pub last_synced: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub resource_group: String,
    pub subscription_id: String,
    pub properties: serde_json::Value,
    pub tags: HashMap<String, String>,
    pub sku: Option<ResourceSku>,
    pub identity: Option<ResourceIdentity>,
}

impl AzureResourceManagerIntegration {
    pub async fn new() -> Result<Self, AzureError> {
        let credential = DefaultAzureCredential::default();
        let client = ResourceClient::new(credential);
        
        Ok(Self {
            client,
            subscription_cache: HashMap::new(),
            resource_cache: HashMap::new(),
        })
    }

    // Comprehensive resource discovery
    pub async fn discover_resources(&mut self, subscription_id: &str) -> Result<Vec<AzureResource>, AzureError> {
        let mut all_resources = Vec::new();

        // Get all resource groups first
        let resource_groups = self.client
            .resource_groups()
            .list(subscription_id)
            .await?;

        // Discover resources in each resource group in parallel
        let discovery_tasks = resource_groups.into_iter().map(|rg| {
            let client = self.client.clone();
            let subscription_id = subscription_id.to_string();
            let resource_group_name = rg.name().to_string();
            
            tokio::spawn(async move {
                client.resources()
                    .list_by_resource_group(&subscription_id, &resource_group_name)
                    .await
            })
        });

        let results = futures::future::join_all(discovery_tasks).await;
        
        for task_result in results {
            match task_result {
                Ok(Ok(resources)) => {
                    for resource in resources {
                        let azure_resource = AzureResource {
                            id: resource.id().to_string(),
                            name: resource.name().to_string(),
                            resource_type: resource.type_().to_string(),
                            location: resource.location().to_string(),
                            resource_group: extract_resource_group(&resource.id())?,
                            subscription_id: subscription_id.to_string(),
                            properties: resource.properties().clone(),
                            tags: resource.tags().clone(),
                            sku: resource.sku().cloned(),
                            identity: resource.identity().cloned(),
                        };
                        all_resources.push(azure_resource);
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!("Failed to discover resources in resource group: {}", e);
                }
                Err(e) => {
                    tracing::error!("Task execution failed: {}", e);
                }
            }
        }

        // Cache discovered resources
        self.resource_cache.insert(subscription_id.to_string(), all_resources.clone());

        Ok(all_resources)
    }

    // Real-time resource monitoring with Change Feed
    pub async fn start_resource_monitoring(&mut self, subscription_id: &str) -> Result<(), AzureError> {
        let change_feed_client = self.create_change_feed_client().await?;
        
        tokio::spawn({
            let subscription_id = subscription_id.to_string();
            let mut client = change_feed_client;
            
            async move {
                loop {
                    match client.get_changes(&subscription_id).await {
                        Ok(changes) => {
                            for change in changes {
                                match change.change_type.as_str() {
                                    "Microsoft.Resources/resourceGroups/write" => {
                                        tracing::info!("Resource created/updated: {}", change.resource_id);
                                        // Handle resource creation/update
                                    }
                                    "Microsoft.Resources/resourceGroups/delete" => {
                                        tracing::info!("Resource deleted: {}", change.resource_id);
                                        // Handle resource deletion
                                    }
                                    _ => {
                                        tracing::debug!("Other change detected: {}", change.change_type);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to get resource changes: {}", e);
                            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                        }
                    }
                    
                    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                }
            }
        });

        Ok(())
    }

    // Deploy ARM templates
    pub async fn deploy_arm_template(
        &self,
        subscription_id: &str,
        resource_group_name: &str,
        template: ArmTemplate,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Result<DeploymentResult, AzureError> {
        let deployment_name = format!("policycortex-{}", Uuid::new_v4());
        
        let deployment = azure_mgmt_resources::models::Deployment::builder()
            .properties(
                azure_mgmt_resources::models::DeploymentProperties::builder()
                    .template(template.template)
                    .parameters(parameters)
                    .mode(azure_mgmt_resources::models::DeploymentMode::Incremental)
                    .build()
            )
            .build();

        let deployment_result = self.client
            .deployments()
            .begin_create_or_update(
                subscription_id,
                resource_group_name,
                &deployment_name,
                deployment,
            )
            .await?
            .wait_for_completion()
            .await?;

        Ok(DeploymentResult {
            deployment_name,
            status: deployment_result.properties().provisioning_state().to_string(),
            outputs: deployment_result.properties().outputs().clone(),
            correlation_id: deployment_result.properties().correlation_id().to_string(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmTemplate {
    pub template: serde_json::Value,
    pub metadata: TemplateMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub parameters: Vec<TemplateParameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub default_value: Option<serde_json::Value>,
    pub allowed_values: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentResult {
    pub deployment_name: String,
    pub status: String,
    pub outputs: Option<serde_json::Value>,
    pub correlation_id: String,
}
```

### Azure Security Center Integration

```rust
// core/src/integrations/azure/security_center.rs
use azure_security_center::Client as SecurityCenterClient;
use azure_identity::DefaultAzureCredential;

pub struct AzureSecurityCenterIntegration {
    client: SecurityCenterClient,
    alert_processors: Vec<Box<dyn AlertProcessor + Send + Sync>>,
}

impl AzureSecurityCenterIntegration {
    pub async fn new() -> Result<Self, AzureError> {
        let credential = DefaultAzureCredential::default();
        let client = SecurityCenterClient::new(credential);
        
        Ok(Self {
            client,
            alert_processors: vec![
                Box::new(ComplianceAlertProcessor::new()),
                Box::new(SecurityAlertProcessor::new()),
                Box::new(PolicyViolationProcessor::new()),
            ],
        })
    }

    // Sync security recommendations
    pub async fn sync_security_recommendations(
        &self,
        subscription_id: &str,
    ) -> Result<Vec<SecurityRecommendation>, AzureError> {
        let recommendations = self.client
            .tasks()
            .list_by_subscription(subscription_id)
            .await?;

        let mut processed_recommendations = Vec::new();

        for recommendation in recommendations {
            let processed = SecurityRecommendation {
                id: recommendation.id().to_string(),
                name: recommendation.name().to_string(),
                description: recommendation.properties().description().to_string(),
                severity: map_severity(recommendation.properties().severity()),
                resource_id: recommendation.properties().resource_details().id().to_string(),
                remediation_steps: extract_remediation_steps(recommendation.properties()),
                estimated_effort: estimate_remediation_effort(&recommendation),
                compliance_frameworks: map_compliance_frameworks(&recommendation),
                created_at: recommendation.properties().created_time_utc(),
            };

            processed_recommendations.push(processed);
        }

        Ok(processed_recommendations)
    }

    // Real-time security alerts processing
    pub async fn process_security_alerts(&self, subscription_id: &str) -> Result<(), AzureError> {
        let alerts = self.client
            .alerts()
            .list_by_subscription(subscription_id)
            .await?;

        for alert in alerts {
            for processor in &self.alert_processors {
                if processor.can_process(&alert) {
                    match processor.process(&alert).await {
                        Ok(result) => {
                            tracing::info!("Alert processed successfully: {:?}", result);
                        }
                        Err(e) => {
                            tracing::error!("Failed to process alert {}: {}", alert.id(), e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // Compliance assessment integration
    pub async fn get_compliance_assessment(
        &self,
        subscription_id: &str,
        assessment_name: &str,
    ) -> Result<ComplianceAssessment, AzureError> {
        let assessment = self.client
            .assessments()
            .get(subscription_id, assessment_name)
            .await?;

        Ok(ComplianceAssessment {
            id: assessment.id().to_string(),
            name: assessment.name().to_string(),
            status: assessment.properties().status().to_string(),
            score: assessment.properties().score().clone(),
            compliance_frameworks: extract_compliance_frameworks(&assessment),
            findings: extract_findings(&assessment),
            remediation_guidance: extract_remediation_guidance(&assessment),
            last_updated: assessment.properties().time_generated(),
        })
    }
}
```

## Third-Party Integrations

### Terraform Integration

```rust
// core/src/integrations/terraform.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use tokio::process::Command as AsyncCommand;

#[derive(Debug, Clone)]
pub struct TerraformIntegration {
    workspace_path: String,
    terraform_binary: String,
    backend_config: BackendConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub backend_type: String,
    pub config: HashMap<String, String>,
}

impl TerraformIntegration {
    pub fn new(workspace_path: String, backend_config: BackendConfig) -> Self {
        Self {
            workspace_path,
            terraform_binary: "terraform".to_string(),
            backend_config,
        }
    }

    // Parse Terraform state to discover resources
    pub async fn parse_terraform_state(&self) -> Result<Vec<TerraformResource>, TerraformError> {
        let output = AsyncCommand::new(&self.terraform_binary)
            .args(&["show", "-json"])
            .current_dir(&self.workspace_path)
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(TerraformError::CommandFailed(error.to_string()));
        }

        let state: TerraformState = serde_json::from_slice(&output.stdout)?;
        let mut resources = Vec::new();

        for resource in state.values.root_module.resources {
            resources.push(TerraformResource {
                address: resource.address,
                resource_type: resource.resource_type,
                name: resource.name,
                provider_name: resource.provider_name,
                values: resource.values,
                depends_on: resource.depends_on.unwrap_or_default(),
                sensitive_attributes: resource.sensitive_attributes.unwrap_or_default(),
            });
        }

        Ok(resources)
    }

    // Generate PolicyCortex compliance checks for Terraform resources
    pub async fn generate_compliance_checks(
        &self,
        resources: &[TerraformResource],
    ) -> Result<Vec<ComplianceCheck>, TerraformError> {
        let mut checks = Vec::new();

        for resource in resources {
            match resource.resource_type.as_str() {
                "azurerm_virtual_machine" => {
                    checks.extend(self.generate_vm_compliance_checks(resource));
                }
                "azurerm_storage_account" => {
                    checks.extend(self.generate_storage_compliance_checks(resource));
                }
                "azurerm_sql_database" => {
                    checks.extend(self.generate_sql_compliance_checks(resource));
                }
                "azurerm_key_vault" => {
                    checks.extend(self.generate_keyvault_compliance_checks(resource));
                }
                _ => {
                    checks.extend(self.generate_generic_compliance_checks(resource));
                }
            }
        }

        Ok(checks)
    }

    // Apply compliance policies to Terraform configuration
    pub async fn apply_compliance_policies(
        &self,
        policies: &[TerraformPolicy],
    ) -> Result<ApplyResult, TerraformError> {
        // Generate Terraform files for compliance policies
        for policy in policies {
            let terraform_code = self.generate_terraform_code(policy)?;
            let filename = format!("{}/compliance_{}.tf", self.workspace_path, policy.name);
            tokio::fs::write(&filename, terraform_code).await?;
        }

        // Run terraform plan
        let plan_output = AsyncCommand::new(&self.terraform_binary)
            .args(&["plan", "-out=compliance.tfplan"])
            .current_dir(&self.workspace_path)
            .output()
            .await?;

        if !plan_output.status.success() {
            let error = String::from_utf8_lossy(&plan_output.stderr);
            return Err(TerraformError::PlanFailed(error.to_string()));
        }

        // Parse plan to understand changes
        let plan_json_output = AsyncCommand::new(&self.terraform_binary)
            .args(&["show", "-json", "compliance.tfplan"])
            .current_dir(&self.workspace_path)
            .output()
            .await?;

        let plan: TerraformPlan = serde_json::from_slice(&plan_json_output.stdout)?;
        
        Ok(ApplyResult {
            changes_count: plan.resource_changes.len(),
            planned_changes: plan.resource_changes,
            plan_output: String::from_utf8_lossy(&plan_output.stdout).to_string(),
        })
    }

    fn generate_vm_compliance_checks(&self, resource: &TerraformResource) -> Vec<ComplianceCheck> {
        let mut checks = Vec::new();

        // Check for disk encryption
        if let Some(storage_os_disk) = resource.values.get("storage_os_disk") {
            if let Some(disk_config) = storage_os_disk.as_object() {
                if disk_config.get("disk_encryption_key").is_none() {
                    checks.push(ComplianceCheck {
                        resource_address: resource.address.clone(),
                        check_type: "disk_encryption".to_string(),
                        severity: "HIGH".to_string(),
                        message: "OS disk should be encrypted".to_string(),
                        remediation: "Add disk_encryption_key configuration".to_string(),
                    });
                }
            }
        }

        // Check for network security group
        if resource.values.get("network_interface_ids").is_none() {
            checks.push(ComplianceCheck {
                resource_address: resource.address.clone(),
                check_type: "network_security".to_string(),
                severity: "MEDIUM".to_string(),
                message: "Virtual machine should have network security group attached".to_string(),
                remediation: "Associate a network security group with the VM".to_string(),
            });
        }

        checks
    }

    fn generate_terraform_code(&self, policy: &TerraformPolicy) -> Result<String, TerraformError> {
        match policy.policy_type.as_str() {
            "network_security_group" => {
                Ok(format!(
                    r#"
resource "azurerm_network_security_group" "{}" {{
  name                = "{}"
  location            = var.location
  resource_group_name = var.resource_group_name

  {}

  tags = var.tags
}}
"#,
                    policy.resource_name,
                    policy.resource_name,
                    policy.configuration
                ))
            }
            "storage_account_encryption" => {
                Ok(format!(
                    r#"
resource "azurerm_storage_account" "{}" {{
  name                     = "{}"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  enable_https_traffic_only = true
  min_tls_version          = "TLS1_2"
  
  blob_properties {{
    delete_retention_policy {{
      days = 7
    }}
  }}

  tags = var.tags
}}
"#,
                    policy.resource_name,
                    policy.resource_name
                ))
            }
            _ => Err(TerraformError::UnsupportedPolicyType(policy.policy_type.clone())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerraformResource {
    pub address: String,
    pub resource_type: String,
    pub name: String,
    pub provider_name: String,
    pub values: serde_json::Value,
    pub depends_on: Vec<String>,
    pub sensitive_attributes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerraformPolicy {
    pub name: String,
    pub policy_type: String,
    pub resource_name: String,
    pub configuration: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub resource_address: String,
    pub check_type: String,
    pub severity: String,
    pub message: String,
    pub remediation: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TerraformState {
    pub values: StateValues,
}

#[derive(Debug, Serialize, Deserialize)]
struct StateValues {
    pub root_module: RootModule,
}

#[derive(Debug, Serialize, Deserialize)]
struct RootModule {
    pub resources: Vec<StateResource>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StateResource {
    pub address: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub name: String,
    pub provider_name: String,
    pub values: serde_json::Value,
    pub depends_on: Option<Vec<String>>,
    pub sensitive_attributes: Option<Vec<String>>,
}

#[derive(Debug, thiserror::Error)]
pub enum TerraformError {
    #[error("Command failed: {0}")]
    CommandFailed(String),
    #[error("Plan failed: {0}")]
    PlanFailed(String),
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Unsupported policy type: {0}")]
    UnsupportedPolicyType(String),
}
```

### ServiceNow Integration

```python
# backend/services/integrations/servicenow.py
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ServiceNowIncident:
    sys_id: str
    number: str
    short_description: str
    description: str
    state: str
    priority: str
    severity: str
    assigned_to: Optional[str]
    assignment_group: Optional[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class ServiceNowChangeRequest:
    sys_id: str
    number: str
    short_description: str
    description: str
    state: str
    type: str
    risk: str
    priority: str
    requested_by: str
    start_date: datetime
    end_date: datetime

class ServiceNowIntegration:
    def __init__(self, instance_url: str, username: str, password: str):
        self.instance_url = instance_url.rstrip('/')
        self.auth = aiohttp.BasicAuth(username, password)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(auth=self.auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # Create incident for compliance violations
    async def create_compliance_incident(
        self,
        resource_id: str,
        policy_name: str,
        violation_details: Dict[str, Any]
    ) -> ServiceNowIncident:
        
        incident_data = {
            'short_description': f'Compliance violation: {policy_name}',
            'description': f"""
Compliance Violation Details:
- Resource ID: {resource_id}
- Policy: {policy_name}
- Severity: {violation_details.get('severity', 'Medium')}
- Details: {violation_details.get('description', '')}
- Recommendations: {', '.join(violation_details.get('recommendations', []))}

This incident was automatically created by PolicyCortex.
            """.strip(),
            'category': 'Security',
            'subcategory': 'Compliance',
            'priority': self._map_priority(violation_details.get('severity', 'Medium')),
            'assignment_group': 'Cloud Security Team',
            'u_resource_id': resource_id,  # Custom field
            'u_policy_cortex_id': violation_details.get('evaluation_id', ''),  # Custom field
        }

        async with self.session.post(
            f'{self.instance_url}/api/now/table/incident',
            json=incident_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return ServiceNowIncident(
                sys_id=result['result']['sys_id'],
                number=result['result']['number'],
                short_description=result['result']['short_description'],
                description=result['result']['description'],
                state=result['result']['state'],
                priority=result['result']['priority'],
                severity=result['result']['severity'],
                assigned_to=result['result'].get('assigned_to'),
                assignment_group=result['result'].get('assignment_group'),
                created_at=datetime.fromisoformat(result['result']['sys_created_on'].replace(' ', 'T')),
                updated_at=datetime.fromisoformat(result['result']['sys_updated_on'].replace(' ', 'T')),
            )

    # Create change request for remediation actions
    async def create_remediation_change_request(
        self,
        resource_id: str,
        remediation_plan: Dict[str, Any]
    ) -> ServiceNowChangeRequest:
        
        change_data = {
            'short_description': f'Automated remediation: {remediation_plan["title"]}',
            'description': f"""
Automated Remediation Plan:
- Resource ID: {resource_id}
- Action: {remediation_plan['action_type']}
- Description: {remediation_plan.get('description', '')}
- Estimated Impact: {remediation_plan.get('estimated_impact', 'Low')}
- Rollback Plan: {remediation_plan.get('rollback_plan', 'Standard rollback procedures')}

This change request was automatically created by PolicyCortex.
            """.strip(),
            'type': 'Standard',
            'risk': self._map_risk(remediation_plan.get('risk_level', 'Low')),
            'priority': self._map_priority(remediation_plan.get('priority', 'Medium')),
            'requested_by': 'PolicyCortex System',
            'assignment_group': 'Cloud Operations Team',
            'implementation_plan': remediation_plan.get('implementation_steps', ''),
            'backout_plan': remediation_plan.get('rollback_plan', ''),
            'u_resource_id': resource_id,  # Custom field
            'u_automation_approved': remediation_plan.get('auto_approved', False),  # Custom field
        }

        async with self.session.post(
            f'{self.instance_url}/api/now/table/change_request',
            json=change_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return ServiceNowChangeRequest(
                sys_id=result['result']['sys_id'],
                number=result['result']['number'],
                short_description=result['result']['short_description'],
                description=result['result']['description'],
                state=result['result']['state'],
                type=result['result']['type'],
                risk=result['result']['risk'],
                priority=result['result']['priority'],
                requested_by=result['result']['requested_by'],
                start_date=datetime.fromisoformat(result['result']['start_date'].replace(' ', 'T')),
                end_date=datetime.fromisoformat(result['result']['end_date'].replace(' ', 'T')),
            )

    # Update incident status
    async def update_incident_status(
        self,
        incident_sys_id: str,
        state: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        
        update_data = {'state': state}
        if resolution_notes:
            update_data['close_notes'] = resolution_notes

        async with self.session.patch(
            f'{self.instance_url}/api/now/table/incident/{incident_sys_id}',
            json=update_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            response.raise_for_status()
            return True

    # Get incidents related to PolicyCortex
    async def get_policycortex_incidents(
        self,
        limit: int = 100,
        state_filter: Optional[str] = None
    ) -> List[ServiceNowIncident]:
        
        query_params = {
            'sysparm_limit': limit,
            'sysparm_query': 'categoryLIKECompliance^ORshort_descriptionLIKEPolicyCortex'
        }
        
        if state_filter:
            query_params['sysparm_query'] += f'^state={state_filter}'

        async with self.session.get(
            f'{self.instance_url}/api/now/table/incident',
            params=query_params
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            incidents = []
            for incident_data in result['result']:
                incidents.append(ServiceNowIncident(
                    sys_id=incident_data['sys_id'],
                    number=incident_data['number'],
                    short_description=incident_data['short_description'],
                    description=incident_data['description'],
                    state=incident_data['state'],
                    priority=incident_data['priority'],
                    severity=incident_data.get('severity', ''),
                    assigned_to=incident_data.get('assigned_to'),
                    assignment_group=incident_data.get('assignment_group'),
                    created_at=datetime.fromisoformat(incident_data['sys_created_on'].replace(' ', 'T')),
                    updated_at=datetime.fromisoformat(incident_data['sys_updated_on'].replace(' ', 'T')),
                ))
            
            return incidents

    def _map_priority(self, severity: str) -> str:
        priority_map = {
            'Critical': '1 - Critical',
            'High': '2 - High',
            'Medium': '3 - Moderate',
            'Low': '4 - Low'
        }
        return priority_map.get(severity, '3 - Moderate')

    def _map_risk(self, risk_level: str) -> str:
        risk_map = {
            'Critical': 'High',
            'High': 'High',
            'Medium': 'Moderate',
            'Low': 'Low'
        }
        return risk_map.get(risk_level, 'Low')
```

## API Integration

### REST API Client Implementation

```python
# python-sdk/policycortex/client.py
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PolicyCortexConfig:
    base_url: str
    api_key: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class PolicyCortexClient:
    def __init__(self, config: PolicyCortexConfig):
        self.config = config
        self.session = None
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PolicyCortex-SDK/1.0'
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # Resources API
    async def list_resources(
        self,
        subscription_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        location: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        params = {'page': page, 'limit': limit}
        if subscription_id:
            params['subscriptionId'] = subscription_id
        if resource_type:
            params['resourceType'] = resource_type
        if location:
            params['location'] = location

        return await self._make_request('GET', '/api/v1/resources', params=params)

    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        return await self._make_request('GET', f'/api/v1/resources/{resource_id}')

    async def create_resource(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._make_request('POST', '/api/v1/resources', json=resource_data)

    async def update_resource(self, resource_id: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._make_request('PUT', f'/api/v1/resources/{resource_id}', json=resource_data)

    async def delete_resource(self, resource_id: str) -> bool:
        response = await self._make_request('DELETE', f'/api/v1/resources/{resource_id}')
        return response is not None

    # Policies API
    async def list_policies(
        self,
        policy_type: Optional[str] = None,
        category: Optional[str] = None,
        enabled: Optional[bool] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        params = {'page': page, 'limit': limit}
        if policy_type:
            params['type'] = policy_type
        if category:
            params['category'] = category
        if enabled is not None:
            params['enabled'] = enabled

        return await self._make_request('GET', '/api/v1/policies', params=params)

    async def get_policy(self, policy_id: str) -> Dict[str, Any]:
        return await self._make_request('GET', f'/api/v1/policies/{policy_id}')

    async def create_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._make_request('POST', '/api/v1/policies', json=policy_data)

    async def evaluate_policy(self, policy_id: str, resource_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        evaluation_data = {'async': True}
        if resource_ids:
            evaluation_data['resourceIds'] = resource_ids

        return await self._make_request('POST', f'/api/v1/policies/{policy_id}/evaluate', json=evaluation_data)

    # Patent Feature APIs
    async def generate_policy_ai(
        self,
        resource_type: str,
        category: str,
        requirements: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        request_data = {
            'resourceType': resource_type,
            'category': category,
            'requirements': requirements
        }
        if constraints:
            request_data['constraints'] = constraints

        return await self._make_request('POST', '/api/v1/ai/generate-policy', json=request_data)

    async def conversational_query(
        self,
        query: str,
        context_id: Optional[str] = None,
        include_resources: bool = False
    ) -> Dict[str, Any]:
        request_data = {
            'query': query,
            'includeResources': include_resources
        }
        if context_id:
            request_data['contextId'] = context_id

        return await self._make_request('POST', '/api/v1/ai/conversational', json=request_data)

    async def get_compliance_predictions(
        self,
        resource_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        time_horizon: int = 30
    ) -> Dict[str, Any]:
        params = {'timeHorizon': time_horizon}
        if resource_id:
            params['resourceId'] = resource_id
        if policy_id:
            params['policyId'] = policy_id

        return await self._make_request('GET', '/api/v1/predictions', params=params)

    async def get_cross_domain_correlations(
        self,
        domains: Optional[List[str]] = None,
        time_range: str = '24h'
    ) -> Dict[str, Any]:
        params = {'timeRange': time_range}
        if domains:
            params['domains'] = domains

        return await self._make_request('GET', '/api/v1/correlations', params=params)

    async def get_unified_metrics(
        self,
        subscription_id: Optional[str] = None,
        aggregation: str = 'subscription',
        time_range: str = '24h'
    ) -> Dict[str, Any]:
        params = {'aggregation': aggregation, 'timeRange': time_range}
        if subscription_id:
            params['subscriptionId'] = subscription_id

        return await self._make_request('GET', '/api/v1/metrics/unified', params=params)

    # Streaming and real-time features
    async def stream_resource_updates(
        self,
        resource_id: Optional[str] = None,
        subscription_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        params = {}
        if resource_id:
            params['resourceId'] = resource_id
        if subscription_id:
            params['subscriptionId'] = subscription_id

        async with self.session.ws_connect(
            f"{self.config.base_url.replace('http', 'ws')}/api/v1/stream/resources",
            params=params
        ) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    yield json.loads(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    # Batch operations
    async def batch_create_resources(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self._make_request('POST', '/api/v1/resources/batch', json={'resources': resources})

    async def batch_evaluate_policies(
        self,
        policy_ids: List[str],
        resource_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        request_data = {'policyIds': policy_ids}
        if resource_ids:
            request_data['resourceIds'] = resource_ids

        return await self._make_request('POST', '/api/v1/policies/batch-evaluate', json=request_data)

    # Helper methods
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(
                    method,
                    url,
                    params=params,
                    json=json
                ) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    
                    if response.status == 204:  # No content
                        return None
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise PolicyCortexAPIError(f"Request failed after {self.config.max_retries} attempts: {e}")
                
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

    def health_check(self) -> bool:
        """Synchronous health check"""
        import requests
        try:
            response = requests.get(f"{self.config.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Custom exceptions
class PolicyCortexAPIError(Exception):
    pass

class PolicyCortexAuthError(PolicyCortexAPIError):
    pass

class PolicyCortexRateLimitError(PolicyCortexAPIError):
    pass

# Usage example
async def example_usage():
    config = PolicyCortexConfig(
        base_url='https://api.policycortex.com',
        api_key='your-api-key-here'
    )
    
    async with PolicyCortexClient(config) as client:
        # List resources
        resources = await client.list_resources(
            subscription_id='12345678-1234-1234-1234-123456789012',
            resource_type='virtual_machine'
        )
        print(f"Found {len(resources['data'])} virtual machines")
        
        # Generate policy using AI
        policy = await client.generate_policy_ai(
            resource_type='storage_account',
            category='security',
            requirements=['encrypt at rest', 'https only', 'no public access']
        )
        print(f"Generated policy: {policy['name']}")
        
        # Conversational query
        response = await client.conversational_query(
            "Show me all non-compliant storage accounts and how to fix them"
        )
        print(f"AI Response: {response['text']}")
```

## Webhook Integration

### Webhook Management System

```rust
// core/src/webhooks/manager.rs
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use tokio::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Webhook {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub name: String,
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub secret: String,
    pub active: bool,
    pub retry_config: RetryConfig,
    pub filters: Option<WebhookFilters>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookEvent {
    ResourceCreated,
    ResourceUpdated,
    ResourceDeleted,
    PolicyEvaluated,
    ComplianceViolated,
    ComplianceResolved,
    ActionExecuted,
    AlertTriggered,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_backoff: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookFilters {
    pub subscription_ids: Option<Vec<String>>,
    pub resource_types: Option<Vec<String>>,
    pub severity_levels: Option<Vec<String>>,
    pub custom_filters: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookPayload {
    pub event_id: Uuid,
    pub event_type: WebhookEvent,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tenant_id: Uuid,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

pub struct WebhookManager {
    webhooks: HashMap<Uuid, Webhook>,
    http_client: reqwest::Client,
    delivery_queue: tokio::sync::mpsc::UnboundedSender<WebhookDelivery>,
}

#[derive(Debug, Clone)]
struct WebhookDelivery {
    webhook_id: Uuid,
    payload: WebhookPayload,
    attempt: u32,
    next_attempt_at: Instant,
}

impl WebhookManager {
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        let manager = Self {
            webhooks: HashMap::new(),
            http_client: reqwest::Client::new(),
            delivery_queue: tx,
        };

        // Start delivery worker
        tokio::spawn(Self::delivery_worker(rx, manager.http_client.clone()));

        manager
    }

    // Register new webhook
    pub async fn register_webhook(
        &mut self,
        webhook_request: CreateWebhookRequest,
        tenant_id: Uuid,
    ) -> Result<Webhook, WebhookError> {
        // Validate webhook URL
        if !self.validate_webhook_url(&webhook_request.url).await? {
            return Err(WebhookError::InvalidUrl);
        }

        let webhook = Webhook {
            id: Uuid::new_v4(),
            tenant_id,
            name: webhook_request.name,
            url: webhook_request.url,
            events: webhook_request.events,
            secret: self.generate_webhook_secret(),
            active: true,
            retry_config: webhook_request.retry_config.unwrap_or_default(),
            filters: webhook_request.filters,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Store webhook
        self.webhooks.insert(webhook.id, webhook.clone());

        // Send test webhook
        self.send_test_webhook(&webhook).await?;

        Ok(webhook)
    }

    // Send webhook for event
    pub async fn send_webhook(
        &self,
        event_type: WebhookEvent,
        tenant_id: Uuid,
        data: serde_json::Value,
        metadata: HashMap<String, String>,
    ) -> Result<(), WebhookError> {
        let payload = WebhookPayload {
            event_id: Uuid::new_v4(),
            event_type: event_type.clone(),
            timestamp: chrono::Utc::now(),
            tenant_id,
            data,
            metadata,
        };

        // Find matching webhooks
        for webhook in self.webhooks.values() {
            if webhook.tenant_id == tenant_id 
                && webhook.active 
                && webhook.events.contains(&event_type)
                && self.matches_filters(webhook, &payload) {
                
                let delivery = WebhookDelivery {
                    webhook_id: webhook.id,
                    payload: payload.clone(),
                    attempt: 0,
                    next_attempt_at: Instant::now(),
                };

                self.delivery_queue.send(delivery).map_err(|_| WebhookError::DeliveryQueueFull)?;
            }
        }

        Ok(())
    }

    // Delivery worker
    async fn delivery_worker(
        mut rx: tokio::sync::mpsc::UnboundedReceiver<WebhookDelivery>,
        http_client: reqwest::Client,
    ) {
        let mut pending_deliveries = Vec::new();
        let mut retry_timer = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                // New delivery
                delivery = rx.recv() => {
                    if let Some(delivery) = delivery {
                        Self::attempt_delivery(&http_client, delivery, &mut pending_deliveries).await;
                    } else {
                        break; // Channel closed
                    }
                }
                
                // Retry timer
                _ = retry_timer.tick() => {
                    let now = Instant::now();
                    let mut i = 0;
                    while i < pending_deliveries.len() {
                        if pending_deliveries[i].next_attempt_at <= now {
                            let delivery = pending_deliveries.remove(i);
                            Self::attempt_delivery(&http_client, delivery, &mut pending_deliveries).await;
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
    }

    async fn attempt_delivery(
        http_client: &reqwest::Client,
        mut delivery: WebhookDelivery,
        pending_deliveries: &mut Vec<WebhookDelivery>,
    ) {
        // Get webhook configuration (in real implementation, this would be from database)
        // For now, using default retry config
        let retry_config = RetryConfig::default();
        
        delivery.attempt += 1;

        // Create signature
        let signature = Self::create_signature(&delivery.payload, "webhook_secret"); // Use actual secret
        
        // Send webhook
        let result = http_client
            .post(&format!("https://example.com/webhook")) // Use actual webhook URL
            .header("Content-Type", "application/json")
            .header("X-PolicyCortex-Signature", signature)
            .header("X-PolicyCortex-Event", format!("{:?}", delivery.payload.event_type))
            .header("X-PolicyCortex-Delivery", delivery.payload.event_id.to_string())
            .json(&delivery.payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await;

        match result {
            Ok(response) if response.status().is_success() => {
                tracing::info!(
                    "Webhook delivered successfully: event_id={}, attempt={}",
                    delivery.payload.event_id,
                    delivery.attempt
                );
            }
            Ok(response) => {
                tracing::warn!(
                    "Webhook delivery failed: event_id={}, status={}, attempt={}",
                    delivery.payload.event_id,
                    response.status(),
                    delivery.attempt
                );
                Self::schedule_retry(delivery, &retry_config, pending_deliveries);
            }
            Err(e) => {
                tracing::error!(
                    "Webhook delivery error: event_id={}, error={}, attempt={}",
                    delivery.payload.event_id,
                    e,
                    delivery.attempt
                );
                Self::schedule_retry(delivery, &retry_config, pending_deliveries);
            }
        }
    }

    fn schedule_retry(
        delivery: WebhookDelivery,
        retry_config: &RetryConfig,
        pending_deliveries: &mut Vec<WebhookDelivery>,
    ) {
        if delivery.attempt >= retry_config.max_retries {
            tracing::error!(
                "Webhook delivery permanently failed after {} attempts: event_id={}",
                delivery.attempt,
                delivery.payload.event_id
            );
            return;
        }

        let delay_ms = if retry_config.exponential_backoff {
            std::cmp::min(
                retry_config.initial_delay_ms * (2_u64.pow(delivery.attempt - 1)),
                retry_config.max_delay_ms,
            )
        } else {
            retry_config.initial_delay_ms
        };

        let mut retry_delivery = delivery;
        retry_delivery.next_attempt_at = Instant::now() + Duration::from_millis(delay_ms);
        pending_deliveries.push(retry_delivery);
    }

    fn create_signature(payload: &WebhookPayload, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let payload_json = serde_json::to_string(payload).unwrap();
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(payload_json.as_bytes());
        let signature = mac.finalize().into_bytes();
        
        format!("sha256={}", hex::encode(signature))
    }

    async fn validate_webhook_url(&self, url: &str) -> Result<bool, WebhookError> {
        // Send a validation request
        let validation_payload = serde_json::json!({
            "event_type": "webhook.validation",
            "challenge": Uuid::new_v4().to_string(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        match self.http_client
            .post(url)
            .header("Content-Type", "application/json")
            .header("X-PolicyCortex-Event", "webhook.validation")
            .json(&validation_payload)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false), // Consider validation failed but don't error
        }
    }

    async fn send_test_webhook(&self, webhook: &Webhook) -> Result<(), WebhookError> {
        let test_payload = WebhookPayload {
            event_id: Uuid::new_v4(),
            event_type: WebhookEvent::Custom("webhook.test".to_string()),
            timestamp: chrono::Utc::now(),
            tenant_id: webhook.tenant_id,
            data: serde_json::json!({
                "message": "This is a test webhook from PolicyCortex",
                "webhook_id": webhook.id,
                "webhook_name": webhook.name
            }),
            metadata: HashMap::from([
                ("test".to_string(), "true".to_string()),
            ]),
        };

        let signature = Self::create_signature(&test_payload, &webhook.secret);
        
        self.http_client
            .post(&webhook.url)
            .header("Content-Type", "application/json")
            .header("X-PolicyCortex-Signature", signature)
            .header("X-PolicyCortex-Event", "webhook.test")
            .header("X-PolicyCortex-Delivery", test_payload.event_id.to_string())
            .json(&test_payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|_| WebhookError::TestWebhookFailed)?;

        Ok(())
    }

    fn matches_filters(&self, webhook: &Webhook, payload: &WebhookPayload) -> bool {
        if let Some(filters) = &webhook.filters {
            // Check subscription filter
            if let Some(subscription_ids) = &filters.subscription_ids {
                if let Some(subscription_id) = payload.data.get("subscription_id") {
                    if let Some(subscription_id) = subscription_id.as_str() {
                        if !subscription_ids.contains(&subscription_id.to_string()) {
                            return false;
                        }
                    }
                }
            }

            // Check resource type filter
            if let Some(resource_types) = &filters.resource_types {
                if let Some(resource_type) = payload.data.get("resource_type") {
                    if let Some(resource_type) = resource_type.as_str() {
                        if !resource_types.contains(&resource_type.to_string()) {
                            return false;
                        }
                    }
                }
            }

            // Check severity filter
            if let Some(severity_levels) = &filters.severity_levels {
                if let Some(severity) = payload.data.get("severity") {
                    if let Some(severity) = severity.as_str() {
                        if !severity_levels.contains(&severity.to_string()) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    fn generate_webhook_secret(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..32)
            .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
            .collect()
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            exponential_backoff: true,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateWebhookRequest {
    pub name: String,
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub retry_config: Option<RetryConfig>,
    pub filters: Option<WebhookFilters>,
}

#[derive(Debug, thiserror::Error)]
pub enum WebhookError {
    #[error("Invalid webhook URL")]
    InvalidUrl,
    #[error("Test webhook failed")]
    TestWebhookFailed,
    #[error("Delivery queue is full")]
    DeliveryQueueFull,
    #[error("Webhook not found")]
    NotFound,
}
```

This comprehensive integration guide provides detailed implementations and examples for integrating PolicyCortex with various external systems, from Azure native services to third-party platforms and custom applications. The integration patterns support both real-time and batch processing scenarios while maintaining reliability and security.