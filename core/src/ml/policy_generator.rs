// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Policy Generation from Natural Language
// Translates natural language requirements into Azure Policy JSON definitions

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use regex::Regex;

/// Policy generator that converts natural language to Azure Policy
pub struct PolicyGenerator {
    template_library: PolicyTemplateLibrary,
    requirement_parser: RequirementParser,
    policy_builder: PolicyBuilder,
}

impl PolicyGenerator {
    pub fn new() -> Self {
        Self {
            template_library: PolicyTemplateLibrary::new(),
            requirement_parser: RequirementParser::new(),
            policy_builder: PolicyBuilder::new(),
        }
    }
    
    /// Generate Azure Policy from natural language description
    pub async fn generate_from_nl(&self, description: &str) -> Result<GeneratedPolicy, String> {
        // Parse requirements from natural language
        let requirements = self.requirement_parser.parse(description)?;
        
        // Find matching template or build custom
        let base_policy = if let Some(template) = self.template_library.find_matching_template(&requirements) {
            template.to_policy()
        } else {
            self.policy_builder.build_custom(&requirements)?
        };
        
        // Enhance with parsed requirements
        let enhanced_policy = self.enhance_policy(base_policy, &requirements);
        
        // Validate the generated policy
        if let Err(e) = self.validate_policy(&enhanced_policy) {
            return Err(format!("Policy validation failed: {}", e));
        }
        
        Ok(GeneratedPolicy {
            policy_definition: enhanced_policy,
            explanation: self.generate_explanation(&requirements),
            warnings: self.generate_warnings(&requirements),
            confidence: requirements.confidence,
        })
    }
    
    fn enhance_policy(&self, mut policy: Value, requirements: &ParsedRequirements) -> Value {
        // Add metadata
        if let Some(obj) = policy["properties"]["metadata"].as_object_mut() {
            obj.insert("generatedFrom".to_string(), json!(requirements.original_text));
            obj.insert("generatedAt".to_string(), json!(chrono::Utc::now().to_rfc3339()));
            obj.insert("confidence".to_string(), json!(requirements.confidence));
        }
        
        // Add parameters if detected
        if !requirements.parameters.is_empty() {
            if let Some(params) = policy["properties"]["parameters"].as_object_mut() {
                for param in &requirements.parameters {
                    params.insert(param.name.clone(), json!({
                        "type": param.param_type,
                        "metadata": {
                            "displayName": param.display_name,
                            "description": param.description
                        },
                        "defaultValue": param.default_value
                    }));
                }
            }
        }
        
        policy
    }
    
    fn validate_policy(&self, policy: &Value) -> Result<(), String> {
        // Check required fields
        if !policy["properties"]["policyRule"].is_object() {
            return Err("Missing policyRule".to_string());
        }
        
        if !policy["properties"]["policyRule"]["if"].is_object() {
            return Err("Missing if condition".to_string());
        }
        
        if !policy["properties"]["policyRule"]["then"].is_object() {
            return Err("Missing then effect".to_string());
        }
        
        Ok(())
    }
    
    fn generate_explanation(&self, requirements: &ParsedRequirements) -> String {
        format!(
            "Generated {} policy that {} for {}. The policy will {} when conditions are not met.",
            requirements.policy_type,
            requirements.conditions.iter()
                .map(|c| c.description.clone())
                .collect::<Vec<_>>()
                .join(" and "),
            requirements.resource_types.join(", "),
            requirements.effect
        )
    }
    
    fn generate_warnings(&self, requirements: &ParsedRequirements) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if requirements.confidence < 0.7 {
            warnings.push("Low confidence in interpretation. Please review the generated policy carefully.".to_string());
        }
        
        if requirements.effect == "deny" {
            warnings.push("This policy will block resource creation/updates. Test in audit mode first.".to_string());
        }
        
        if requirements.resource_types.is_empty() {
            warnings.push("No specific resource types detected. Policy may apply broadly.".to_string());
        }
        
        warnings
    }
}

/// Parsed requirements from natural language
#[derive(Debug, Clone)]
pub struct ParsedRequirements {
    pub original_text: String,
    pub policy_type: String,
    pub resource_types: Vec<String>,
    pub conditions: Vec<Condition>,
    pub effect: String,
    pub parameters: Vec<PolicyParameter>,
    pub confidence: f64,
}

/// Policy condition
#[derive(Debug, Clone)]
pub struct Condition {
    pub field: String,
    pub operator: String,
    pub value: Value,
    pub description: String,
}

/// Policy parameter
#[derive(Debug, Clone)]
pub struct PolicyParameter {
    pub name: String,
    pub param_type: String,
    pub display_name: String,
    pub description: String,
    pub default_value: Option<Value>,
}

/// Requirement parser
pub struct RequirementParser {
    resource_patterns: HashMap<String, String>,
    condition_patterns: Vec<ConditionPattern>,
    effect_keywords: HashMap<String, String>,
}

impl RequirementParser {
    pub fn new() -> Self {
        let mut parser = Self {
            resource_patterns: HashMap::new(),
            condition_patterns: Vec::new(),
            effect_keywords: HashMap::new(),
        };
        
        parser.initialize_patterns();
        parser
    }
    
    fn initialize_patterns(&mut self) {
        // Resource type patterns
        self.resource_patterns.insert("storage".to_string(), "Microsoft.Storage/storageAccounts".to_string());
        self.resource_patterns.insert("vm".to_string(), "Microsoft.Compute/virtualMachines".to_string());
        self.resource_patterns.insert("database".to_string(), "Microsoft.Sql/servers".to_string());
        self.resource_patterns.insert("network".to_string(), "Microsoft.Network/*".to_string());
        self.resource_patterns.insert("key vault".to_string(), "Microsoft.KeyVault/vaults".to_string());
        
        // Condition patterns
        self.condition_patterns.push(ConditionPattern {
            pattern: Regex::new(r"require[s]?\s+encryption").unwrap(),
            condition: Condition {
                field: "encryption.enabled".to_string(),
                operator: "equals".to_string(),
                value: json!(true),
                description: "requires encryption".to_string(),
            },
        });
        
        self.condition_patterns.push(ConditionPattern {
            pattern: Regex::new(r"require[s]?\s+tag[s]?\s+(\w+)").unwrap(),
            condition: Condition {
                field: "tags".to_string(),
                operator: "exists".to_string(),
                value: json!(true),
                description: "requires tags".to_string(),
            },
        });
        
        self.condition_patterns.push(ConditionPattern {
            pattern: Regex::new(r"block\s+public\s+access").unwrap(),
            condition: Condition {
                field: "publicNetworkAccess".to_string(),
                operator: "notEquals".to_string(),
                value: json!("Enabled"),
                description: "blocks public access".to_string(),
            },
        });
        
        // Effect keywords
        self.effect_keywords.insert("require".to_string(), "deny".to_string());
        self.effect_keywords.insert("enforce".to_string(), "deny".to_string());
        self.effect_keywords.insert("block".to_string(), "deny".to_string());
        self.effect_keywords.insert("prevent".to_string(), "deny".to_string());
        self.effect_keywords.insert("audit".to_string(), "audit".to_string());
        self.effect_keywords.insert("monitor".to_string(), "audit".to_string());
        self.effect_keywords.insert("deploy".to_string(), "deployIfNotExists".to_string());
    }
    
    pub fn parse(&self, description: &str) -> Result<ParsedRequirements, String> {
        let lower = description.to_lowercase();
        
        // Extract resource types
        let mut resource_types = Vec::new();
        for (keyword, resource_type) in &self.resource_patterns {
            if lower.contains(keyword) {
                resource_types.push(resource_type.clone());
            }
        }
        
        // Extract conditions
        let mut conditions = Vec::new();
        for pattern in &self.condition_patterns {
            if pattern.pattern.is_match(&lower) {
                conditions.push(pattern.condition.clone());
            }
        }
        
        // Determine effect
        let mut effect = "audit".to_string();
        for (keyword, effect_type) in &self.effect_keywords {
            if lower.contains(keyword) {
                effect = effect_type.clone();
                break;
            }
        }
        
        // Extract parameters (simplified)
        let parameters = self.extract_parameters(description);
        
        // Calculate confidence
        let confidence = self.calculate_confidence(&resource_types, &conditions);
        
        Ok(ParsedRequirements {
            original_text: description.to_string(),
            policy_type: if conditions.is_empty() { "audit" } else { "compliance" }.to_string(),
            resource_types,
            conditions,
            effect,
            parameters,
            confidence,
        })
    }
    
    fn extract_parameters(&self, description: &str) -> Vec<PolicyParameter> {
        let mut parameters = Vec::new();
        
        // Look for parameter-like patterns
        if description.contains("tag") {
            if let Some(captures) = Regex::new(r#"tag[s]?\s+['"]?(\w+)['"]?"#).unwrap().captures(description) {
                if let Some(tag_name) = captures.get(1) {
                    parameters.push(PolicyParameter {
                        name: format!("required_{}", tag_name.as_str()),
                        param_type: "String".to_string(),
                        display_name: format!("Required {} Tag", tag_name.as_str()),
                        description: format!("The required value for the {} tag", tag_name.as_str()),
                        default_value: None,
                    });
                }
            }
        }
        
        parameters
    }
    
    fn calculate_confidence(&self, resource_types: &[String], conditions: &[Condition]) -> f64 {
        let mut confidence: f64 = 0.5;
        
        if !resource_types.is_empty() {
            confidence += 0.2;
        }
        
        if !conditions.is_empty() {
            confidence += 0.2;
        }
        
        if resource_types.len() == 1 && conditions.len() >= 1 {
            confidence += 0.1;
        }
        
        confidence.min(1.0)
    }
}

/// Condition pattern
struct ConditionPattern {
    pattern: Regex,
    condition: Condition,
}

/// Policy template library
pub struct PolicyTemplateLibrary {
    templates: Vec<PolicyTemplate>,
}

impl PolicyTemplateLibrary {
    pub fn new() -> Self {
        Self {
            templates: Self::initialize_templates(),
        }
    }
    
    fn initialize_templates() -> Vec<PolicyTemplate> {
        vec![
            PolicyTemplate {
                name: "Require Storage Encryption".to_string(),
                keywords: vec!["storage", "encryption"],
                template: json!({
                    "properties": {
                        "displayName": "Require Storage Encryption",
                        "policyType": "Custom",
                        "mode": "All",
                        "metadata": {
                            "category": "Security"
                        },
                        "parameters": {},
                        "policyRule": {
                            "if": {
                                "allOf": [
                                    {
                                        "field": "type",
                                        "equals": "Microsoft.Storage/storageAccounts"
                                    },
                                    {
                                        "field": "Microsoft.Storage/storageAccounts/encryption.services.blob.enabled",
                                        "notEquals": "true"
                                    }
                                ]
                            },
                            "then": {
                                "effect": "deny"
                            }
                        }
                    }
                }),
            },
            PolicyTemplate {
                name: "Require Tags".to_string(),
                keywords: vec!["tag", "label", "metadata"],
                template: json!({
                    "properties": {
                        "displayName": "Require Required Tags",
                        "policyType": "Custom",
                        "mode": "All",
                        "metadata": {
                            "category": "Governance"
                        },
                        "parameters": {
                            "tagName": {
                                "type": "String",
                                "metadata": {
                                    "displayName": "Tag Name",
                                    "description": "Name of the required tag"
                                }
                            }
                        },
                        "policyRule": {
                            "if": {
                                "field": "[concat('tags[', parameters('tagName'), ']')]",
                                "exists": "false"
                            },
                            "then": {
                                "effect": "deny"
                            }
                        }
                    }
                }),
            },
        ]
    }
    
    pub fn find_matching_template(&self, requirements: &ParsedRequirements) -> Option<&PolicyTemplate> {
        let desc_lower = requirements.original_text.to_lowercase();
        
        self.templates.iter()
            .find(|template| {
                template.keywords.iter()
                    .filter(|keyword| desc_lower.contains(*keyword))
                    .count() >= 2
            })
    }
}

/// Policy template
pub struct PolicyTemplate {
    pub name: String,
    pub keywords: Vec<&'static str>,
    pub template: Value,
}

impl PolicyTemplate {
    pub fn to_policy(&self) -> Value {
        self.template.clone()
    }
}

/// Policy builder for custom policies
pub struct PolicyBuilder {
    // Builder configuration
}

impl PolicyBuilder {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn build_custom(&self, requirements: &ParsedRequirements) -> Result<Value, String> {
        let mut policy = json!({
            "properties": {
                "displayName": "Custom Policy",
                "policyType": "Custom",
                "mode": "All",
                "description": requirements.original_text.clone(),
                "metadata": {
                    "category": "Custom",
                    "version": "1.0.0"
                },
                "parameters": {},
                "policyRule": {}
            }
        });
        
        // Build if conditions
        let if_conditions = self.build_if_conditions(requirements);
        policy["properties"]["policyRule"]["if"] = if_conditions;
        
        // Build then effect
        policy["properties"]["policyRule"]["then"] = json!({
            "effect": requirements.effect.clone()
        });
        
        Ok(policy)
    }
    
    fn build_if_conditions(&self, requirements: &ParsedRequirements) -> Value {
        if requirements.conditions.is_empty() && requirements.resource_types.is_empty() {
            // Default condition
            json!({
                "field": "type",
                "like": "Microsoft.*"
            })
        } else if requirements.conditions.len() == 1 && requirements.resource_types.len() <= 1 {
            // Single condition
            let condition = &requirements.conditions[0];
            let mut cond = json!({
                "field": condition.field.clone(),
                condition.operator.clone(): condition.value.clone()
            });
            
            if !requirements.resource_types.is_empty() {
                cond = json!({
                    "allOf": [
                        {
                            "field": "type",
                            "equals": requirements.resource_types[0].clone()
                        },
                        cond
                    ]
                });
            }
            
            cond
        } else {
            // Multiple conditions
            let mut all_conditions = Vec::new();
            
            // Add resource type conditions
            for resource_type in &requirements.resource_types {
                all_conditions.push(json!({
                    "field": "type",
                    "equals": resource_type
                }));
            }
            
            // Add other conditions
            for condition in &requirements.conditions {
                all_conditions.push(json!({
                    "field": condition.field.clone(),
                    condition.operator.clone(): condition.value.clone()
                }));
            }
            
            json!({
                "allOf": all_conditions
            })
        }
    }
}

/// Generated policy result
#[derive(Debug, Serialize)]
pub struct GeneratedPolicy {
    pub policy_definition: Value,
    pub explanation: String,
    pub warnings: Vec<String>,
    pub confidence: f64,
}

/// Policy validation result
#[derive(Debug, Serialize)]
pub struct PolicyValidation {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

impl PolicyGenerator {
    /// Validate an existing policy definition
    pub async fn validate_policy_definition(&self, policy: &Value) -> PolicyValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check structure
        if !policy["properties"].is_object() {
            errors.push("Missing 'properties' object".to_string());
        }
        
        if !policy["properties"]["policyRule"].is_object() {
            errors.push("Missing 'policyRule' object".to_string());
        }
        
        // Check effect
        if let Some(effect) = policy["properties"]["policyRule"]["then"]["effect"].as_str() {
            if !["audit", "deny", "append", "modify", "deployIfNotExists", "auditIfNotExists"].contains(&effect) {
                errors.push(format!("Invalid effect: {}", effect));
            }
            
            if effect == "deny" {
                warnings.push("Deny effect will block resource operations".to_string());
                suggestions.push("Consider testing with 'audit' effect first".to_string());
            }
        } else {
            errors.push("Missing effect in 'then' clause".to_string());
        }
        
        // Check if conditions
        if !policy["properties"]["policyRule"]["if"].is_object() && !policy["properties"]["policyRule"]["if"].is_array() {
            errors.push("Invalid 'if' condition structure".to_string());
        }
        
        PolicyValidation {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }
}