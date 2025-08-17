// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Remediation Template Library Manager
// Loads, validates, and manages remediation templates from YAML files

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateLibrary {
    templates: HashMap<String, RemediationTemplate>,
    categories: HashMap<String, Vec<String>>,
    resource_type_index: HashMap<String, Vec<String>>,
    violation_type_index: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlTemplateFile {
    pub templates: Vec<YamlTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub resource_type: String,
    pub violation_types: Vec<String>,
    pub arm_template: Option<String>,
    pub powershell_script: Option<String>,
    pub azure_cli_commands: Option<Vec<CliCommand>>,
    pub validation_rules: Option<Vec<YamlValidationRule>>,
    pub rollback_template: Option<String>,
    pub success_criteria: Option<YamlSuccessCriteria>,
    pub created_by: Option<String>,
    pub created_date: Option<String>,
    pub last_modified: Option<String>,
    pub tags: Option<Vec<String>>,
    pub estimated_duration_minutes: Option<u32>,
    pub risk_level: Option<String>,
    pub required_permissions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliCommand {
    pub command: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlValidationRule {
    pub rule_id: String,
    pub rule_type: String,
    pub condition: String,
    pub error_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlSuccessCriteria {
    pub compliance_check: bool,
    pub health_check: bool,
    pub performance_check: bool,
    pub custom_validations: Option<Vec<String>>,
    pub min_success_percentage: f64,
}

pub struct TemplateLibraryManager {
    library: Arc<RwLock<TemplateLibrary>>,
    template_paths: Vec<String>,
}

impl TemplateLibraryManager {
    pub fn new() -> Self {
        Self {
            library: Arc::new(RwLock::new(TemplateLibrary::new())),
            template_paths: vec![
                "templates/remediation/".to_string(),
                "./templates/remediation/".to_string(),
                "../templates/remediation/".to_string(),
                "../../templates/remediation/".to_string(),
            ],
        }
    }

    pub async fn load_templates(&self) -> Result<usize, String> {
        let mut loaded_count = 0;
        
        for template_path in &self.template_paths {
            if Path::new(template_path).exists() {
                match self.load_templates_from_directory(template_path).await {
                    Ok(count) => {
                        loaded_count += count;
                        break; // Use first existing path
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load templates from {}: {}", template_path, e);
                        continue;
                    }
                }
            }
        }

        if loaded_count == 0 {
            // Load built-in templates if no files found
            self.load_builtin_templates().await?;
            loaded_count = self.library.read().await.templates.len();
        }

        tracing::info!("Loaded {} remediation templates", loaded_count);
        Ok(loaded_count)
    }

    async fn load_templates_from_directory(&self, dir_path: &str) -> Result<usize, String> {
        let mut loaded_count = 0;
        
        let entries = fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {}: {}", dir_path, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("yaml") ||
               path.extension().and_then(|s| s.to_str()) == Some("yml") {
                
                match self.load_template_file(&path).await {
                    Ok(count) => loaded_count += count,
                    Err(e) => {
                        tracing::error!("Failed to load template file {:?}: {}", path, e);
                        continue;
                    }
                }
            }
        }

        Ok(loaded_count)
    }

    async fn load_template_file(&self, file_path: &Path) -> Result<usize, String> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file {:?}: {}", file_path, e))?;

        let yaml_file: YamlTemplateFile = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML file {:?}: {}", file_path, e))?;

        let mut library = self.library.write().await;
        let mut loaded_count = 0;

        for yaml_template in yaml_file.templates {
            match self.convert_yaml_to_template(yaml_template) {
                Ok(template) => {
                    library.add_template(template);
                    loaded_count += 1;
                }
                Err(e) => {
                    tracing::error!("Failed to convert template: {}", e);
                    continue;
                }
            }
        }

        Ok(loaded_count)
    }

    fn convert_yaml_to_template(&self, yaml: YamlTemplate) -> Result<RemediationTemplate, String> {
        let arm_template = if let Some(arm_str) = yaml.arm_template {
            Some(serde_json::from_str(&arm_str)
                .map_err(|e| format!("Invalid ARM template JSON: {}", e))?)
        } else {
            None
        };

        let validation_rules = yaml.validation_rules
            .unwrap_or_default()
            .into_iter()
            .map(|rule| ValidationRule {
                rule_id: rule.rule_id,
                rule_type: match rule.rule_type.as_str() {
                    "pre_condition" => ValidationType::PreCondition,
                    "post_condition" => ValidationType::PostCondition,
                    "custom" => ValidationType::Custom,
                    _ => ValidationType::PostCondition,
                },
                condition: rule.condition,
                error_message: rule.error_message,
            })
            .collect();

        let success_criteria = yaml.success_criteria
            .map(|sc| SuccessCriteria {
                compliance_check: sc.compliance_check,
                health_check: sc.health_check,
                performance_check: sc.performance_check,
                custom_validations: sc.custom_validations.unwrap_or_default(),
                min_success_percentage: sc.min_success_percentage,
            })
            .unwrap_or_default();

        let azure_cli_commands = yaml.azure_cli_commands
            .unwrap_or_default()
            .into_iter()
            .map(|cmd| cmd.command)
            .collect();

        Ok(RemediationTemplate {
            template_id: yaml.id,
            name: yaml.name,
            description: yaml.description,
            violation_types: yaml.violation_types,
            resource_types: vec![yaml.resource_type],
            arm_template,
            powershell_script: yaml.powershell_script,
            azure_cli_commands,
            validation_rules,
            rollback_template: yaml.rollback_template.and_then(|s| serde_json::from_str(&s).ok()),
            success_criteria,
        })
    }

    async fn load_builtin_templates(&self) -> Result<(), String> {
        let mut library = self.library.write().await;
        
        // Storage Encryption Template
        let storage_encryption = RemediationTemplate {
            template_id: "enable-storage-encryption".to_string(),
            name: "Enable Storage Account Encryption".to_string(),
            description: "Enables encryption at rest for Azure Storage Accounts".to_string(),
            violation_types: vec![
                "EncryptionNotEnabled".to_string(),
                "EncryptionDisabled".to_string(),
                "MissingEncryption".to_string(),
            ],
            resource_types: vec!["Microsoft.Storage/storageAccounts".to_string()],
            arm_template: Some(serde_json::json!({
                "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "parameters": {
                    "storageAccountName": { "type": "string" }
                },
                "resources": [{
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2021-04-01",
                    "name": "[parameters('storageAccountName')]",
                    "properties": {
                        "encryption": {
                            "services": {
                                "blob": { "enabled": true },
                                "file": { "enabled": true }
                            },
                            "keySource": "Microsoft.Storage"
                        },
                        "supportsHttpsTrafficOnly": true
                    }
                }]
            })),
            powershell_script: Some(r#"
param([string]$StorageAccountName, [string]$ResourceGroupName)
Set-AzStorageAccount -ResourceGroupName $ResourceGroupName -Name $StorageAccountName -EnableEncryptionService Blob,File
Set-AzStorageAccount -ResourceGroupName $ResourceGroupName -Name $StorageAccountName -EnableHttpsTrafficOnly $true
"#.to_string()),
            azure_cli_commands: vec![
                "az storage account update --name {storageAccountName} --resource-group {resourceGroupName} --encryption-services blob file".to_string(),
                "az storage account update --name {storageAccountName} --resource-group {resourceGroupName} --https-only true".to_string(),
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_id: "check-encryption".to_string(),
                    rule_type: ValidationType::PostCondition,
                    condition: "resource.properties.encryption.services.blob.enabled == true".to_string(),
                    error_message: "Encryption was not successfully enabled".to_string(),
                }
            ],
            rollback_template: None,
            success_criteria: SuccessCriteria {
                compliance_check: true,
                health_check: true,
                performance_check: false,
                custom_validations: vec!["Encryption enabled".to_string()],
                min_success_percentage: 100.0,
            },
        };

        // Network Security Template
        let network_security = RemediationTemplate {
            template_id: "secure-network-access".to_string(),
            name: "Secure Network Access".to_string(),
            description: "Applies security rules to Network Security Groups".to_string(),
            violation_types: vec![
                "InsecureNetwork".to_string(),
                "PublicAccessEnabled".to_string(),
                "MissingSecurityRules".to_string(),
            ],
            resource_types: vec!["Microsoft.Network/networkSecurityGroups".to_string()],
            arm_template: Some(serde_json::json!({
                "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "parameters": {
                    "nsgName": { "type": "string" }
                },
                "resources": [{
                    "type": "Microsoft.Network/networkSecurityGroups",
                    "apiVersion": "2021-02-01",
                    "name": "[parameters('nsgName')]",
                    "properties": {
                        "securityRules": [{
                            "name": "DenyInternetInbound",
                            "properties": {
                                "priority": 100,
                                "direction": "Inbound",
                                "access": "Deny",
                                "protocol": "*",
                                "sourcePortRange": "*",
                                "destinationPortRange": "*",
                                "sourceAddressPrefix": "Internet",
                                "destinationAddressPrefix": "*"
                            }
                        }]
                    }
                }]
            })),
            powershell_script: Some(r#"
param([string]$NetworkSecurityGroupName, [string]$ResourceGroupName)
$nsg = Get-AzNetworkSecurityGroup -Name $NetworkSecurityGroupName -ResourceGroupName $ResourceGroupName
Add-AzNetworkSecurityRuleConfig -NetworkSecurityGroup $nsg -Name "DenyInternetInbound" -Access Deny -Protocol * -Direction Inbound -Priority 100 -SourceAddressPrefix Internet -SourcePortRange * -DestinationAddressPrefix * -DestinationPortRange *
Set-AzNetworkSecurityGroup -NetworkSecurityGroup $nsg
"#.to_string()),
            azure_cli_commands: vec![
                "az network nsg rule create --resource-group {resourceGroupName} --nsg-name {nsgName} --name DenyInternetInbound --priority 100 --source-address-prefixes Internet --access Deny --protocol '*'".to_string(),
            ],
            validation_rules: vec![],
            rollback_template: None,
            success_criteria: SuccessCriteria::default(),
        };

        // Tagging Template
        let tagging_compliance = RemediationTemplate {
            template_id: "apply-required-tags".to_string(),
            name: "Apply Required Tags".to_string(),
            description: "Applies mandatory governance tags to Azure resources".to_string(),
            violation_types: vec![
                "MissingTags".to_string(),
                "InvalidTags".to_string(),
                "IncompleteTagging".to_string(),
            ],
            resource_types: vec!["*".to_string()],
            arm_template: None,
            powershell_script: Some(r#"
param([string]$ResourceId, [hashtable]$RequiredTags)
$resource = Get-AzResource -ResourceId $ResourceId
$mergedTags = $resource.Tags + $RequiredTags
Set-AzResource -ResourceId $ResourceId -Tag $mergedTags -Force
"#.to_string()),
            azure_cli_commands: vec![
                "az tag create --resource-id {resourceId} --tags Environment=Production Owner=IT-Team".to_string(),
            ],
            validation_rules: vec![],
            rollback_template: None,
            success_criteria: SuccessCriteria::default(),
        };

        library.add_template(storage_encryption);
        library.add_template(network_security);
        library.add_template(tagging_compliance);

        Ok(())
    }

    pub async fn get_template(&self, template_id: &str) -> Option<RemediationTemplate> {
        self.library.read().await.get_template(template_id)
    }

    pub async fn get_templates_for_violation(&self, violation_type: &str) -> Vec<RemediationTemplate> {
        self.library.read().await.get_templates_for_violation(violation_type)
    }

    pub async fn get_templates_for_resource_type(&self, resource_type: &str) -> Vec<RemediationTemplate> {
        self.library.read().await.get_templates_for_resource_type(resource_type)
    }

    pub async fn get_templates_by_category(&self, category: &str) -> Vec<RemediationTemplate> {
        self.library.read().await.get_templates_by_category(category)
    }

    pub async fn list_all_templates(&self) -> Vec<RemediationTemplate> {
        self.library.read().await.list_all_templates()
    }

    pub async fn search_templates(&self, query: &str) -> Vec<RemediationTemplate> {
        self.library.read().await.search_templates(query)
    }

    pub async fn get_template_statistics(&self) -> TemplateStatistics {
        self.library.read().await.get_statistics()
    }
}

impl TemplateLibrary {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            categories: HashMap::new(),
            resource_type_index: HashMap::new(),
            violation_type_index: HashMap::new(),
        }
    }

    pub fn add_template(&mut self, template: RemediationTemplate) {
        let template_id = template.template_id.clone();
        
        // Update category index
        let category = self.extract_category(&template);
        self.categories.entry(category.clone()).or_default().push(template_id.clone());

        // Update resource type index
        for resource_type in &template.resource_types {
            self.resource_type_index
                .entry(resource_type.clone())
                .or_default()
                .push(template_id.clone());
        }

        // Update violation type index
        for violation_type in &template.violation_types {
            self.violation_type_index
                .entry(violation_type.clone())
                .or_default()
                .push(template_id.clone());
        }

        self.templates.insert(template_id, template);
    }

    fn extract_category(&self, template: &RemediationTemplate) -> String {
        // Determine category based on template characteristics
        if template.template_id.contains("encryption") || template.template_id.contains("security") {
            "security".to_string()
        } else if template.template_id.contains("tag") {
            "governance".to_string()
        } else if template.template_id.contains("cost") || template.template_id.contains("rightsize") {
            "cost-optimization".to_string()
        } else if template.template_id.contains("network") {
            "networking".to_string()
        } else {
            "general".to_string()
        }
    }

    pub fn get_template(&self, template_id: &str) -> Option<RemediationTemplate> {
        self.templates.get(template_id).cloned()
    }

    pub fn get_templates_for_violation(&self, violation_type: &str) -> Vec<RemediationTemplate> {
        self.violation_type_index
            .get(violation_type)
            .map(|template_ids| {
                template_ids
                    .iter()
                    .filter_map(|id| self.templates.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_templates_for_resource_type(&self, resource_type: &str) -> Vec<RemediationTemplate> {
        self.resource_type_index
            .get(resource_type)
            .or_else(|| self.resource_type_index.get("*"))
            .map(|template_ids| {
                template_ids
                    .iter()
                    .filter_map(|id| self.templates.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_templates_by_category(&self, category: &str) -> Vec<RemediationTemplate> {
        self.categories
            .get(category)
            .map(|template_ids| {
                template_ids
                    .iter()
                    .filter_map(|id| self.templates.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn list_all_templates(&self) -> Vec<RemediationTemplate> {
        self.templates.values().cloned().collect()
    }

    pub fn search_templates(&self, query: &str) -> Vec<RemediationTemplate> {
        let query = query.to_lowercase();
        self.templates
            .values()
            .filter(|template| {
                template.name.to_lowercase().contains(&query) ||
                template.description.to_lowercase().contains(&query) ||
                template.template_id.to_lowercase().contains(&query) ||
                template.violation_types.iter().any(|vt| vt.to_lowercase().contains(&query))
            })
            .cloned()
            .collect()
    }

    pub fn get_statistics(&self) -> TemplateStatistics {
        TemplateStatistics {
            total_templates: self.templates.len(),
            categories: self.categories.keys().cloned().collect(),
            resource_types: self.resource_type_index.keys().cloned().collect(),
            violation_types: self.violation_type_index.keys().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStatistics {
    pub total_templates: usize,
    pub categories: Vec<String>,
    pub resource_types: Vec<String>,
    pub violation_types: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_template_library_creation() {
        let library = TemplateLibraryManager::new();
        let count = library.load_templates().await.unwrap();
        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_template_search() {
        let library = TemplateLibraryManager::new();
        library.load_templates().await.unwrap();
        
        let templates = library.search_templates("encryption").await;
        assert!(!templates.is_empty());
    }

    #[tokio::test]
    async fn test_violation_type_lookup() {
        let library = TemplateLibraryManager::new();
        library.load_templates().await.unwrap();
        
        let templates = library.get_templates_for_violation("EncryptionNotEnabled").await;
        assert!(!templates.is_empty());
    }
}