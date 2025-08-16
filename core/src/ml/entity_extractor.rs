// Entity Extraction for Natural Language Governance Queries
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub entity_type: EntityType,
    pub value: String,
    pub confidence: f64,
    pub start_index: usize,
    pub end_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    ResourceType,
    ResourceName,
    AzureService,
    Policy,
    Location,
    TimeRange,
    Severity,
    Compliance,
    Cost,
    Action,
    Number,
    Date,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub entities: Vec<Entity>,
    pub normalized_query: String,
    pub domain: String,
    pub intent_hints: Vec<String>,
}

pub struct EntityExtractor {
    patterns: HashMap<EntityType, Vec<Regex>>,
    azure_services: Vec<String>,
    resource_types: Vec<String>,
    policies: Vec<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let mut extractor = Self {
            patterns: HashMap::new(),
            azure_services: Self::initialize_azure_services(),
            resource_types: Self::initialize_resource_types(),
            policies: Self::initialize_policies(),
        };
        
        extractor.initialize_patterns();
        extractor
    }

    fn initialize_patterns(&mut self) {
        // Resource Type patterns
        self.patterns.insert(
            EntityType::ResourceType,
            vec![
                Regex::new(r"(?i)\b(virtual machines?|vms?)\b").unwrap(),
                Regex::new(r"(?i)\b(storage accounts?)\b").unwrap(),
                Regex::new(r"(?i)\b(sql databases?)\b").unwrap(),
                Regex::new(r"(?i)\b(app services?)\b").unwrap(),
                Regex::new(r"(?i)\b(key vaults?)\b").unwrap(),
                Regex::new(r"(?i)\b(network security groups?|nsgs?)\b").unwrap(),
                Regex::new(r"(?i)\b(load balancers?)\b").unwrap(),
                Regex::new(r"(?i)\b(function apps?)\b").unwrap(),
            ]
        );

        // Azure Service patterns
        self.patterns.insert(
            EntityType::AzureService,
            vec![
                Regex::new(r"(?i)\b(azure active directory|aad|entra id)\b").unwrap(),
                Regex::new(r"(?i)\b(azure monitor)\b").unwrap(),
                Regex::new(r"(?i)\b(azure security center)\b").unwrap(),
                Regex::new(r"(?i)\b(azure policy)\b").unwrap(),
                Regex::new(r"(?i)\b(azure rbac)\b").unwrap(),
            ]
        );

        // Policy patterns
        self.patterns.insert(
            EntityType::Policy,
            vec![
                Regex::new(r"(?i)\b(encryption|encrypted?)\b").unwrap(),
                Regex::new(r"(?i)\b(backup|backups?)\b").unwrap(),
                Regex::new(r"(?i)\b(public access|public network)\b").unwrap(),
                Regex::new(r"(?i)\b(https only|ssl|tls)\b").unwrap(),
                Regex::new(r"(?i)\b(diagnostic settings?)\b").unwrap(),
                Regex::new(r"(?i)\b(audit logs?)\b").unwrap(),
            ]
        );

        // Location patterns
        self.patterns.insert(
            EntityType::Location,
            vec![
                Regex::new(r"(?i)\b(east us|west us|central us|north central us|south central us)\b").unwrap(),
                Regex::new(r"(?i)\b(west europe|north europe)\b").unwrap(),
                Regex::new(r"(?i)\b(asia pacific|southeast asia|east asia)\b").unwrap(),
                Regex::new(r"(?i)\b(australia east|australia southeast)\b").unwrap(),
            ]
        );

        // Time Range patterns
        self.patterns.insert(
            EntityType::TimeRange,
            vec![
                Regex::new(r"(?i)\b(last|past|previous)\s+(\d+)\s+(days?|weeks?|months?|hours?)\b").unwrap(),
                Regex::new(r"(?i)\b(today|yesterday|this week|this month)\b").unwrap(),
                Regex::new(r"(?i)\b(in the (next|coming))\s+(\d+)\s+(days?|weeks?|months?|hours?)\b").unwrap(),
            ]
        );

        // Severity patterns
        self.patterns.insert(
            EntityType::Severity,
            vec![
                Regex::new(r"(?i)\b(critical|high|medium|low)\s+(priority|severity|risk)\b").unwrap(),
                Regex::new(r"(?i)\b(urgent|important|minor)\b").unwrap(),
            ]
        );

        // Cost patterns
        self.patterns.insert(
            EntityType::Cost,
            vec![
                Regex::new(r"\$\d+(?:\.\d{2})?|\d+\s*dollars?").unwrap(),
                Regex::new(r"(?i)\b(expensive|cheap|cost|budget|spend|spending)\b").unwrap(),
                Regex::new(r"(?i)\b(\d+%|percent)\s*(more|less|increase|decrease|savings?)\b").unwrap(),
            ]
        );

        // Action patterns
        self.patterns.insert(
            EntityType::Action,
            vec![
                Regex::new(r"(?i)\b(enable|disable|create|delete|update|modify|configure)\b").unwrap(),
                Regex::new(r"(?i)\b(fix|repair|remediate|resolve)\b").unwrap(),
                Regex::new(r"(?i)\b(list|show|display|find|search)\b").unwrap(),
                Regex::new(r"(?i)\b(monitor|check|verify|validate)\b").unwrap(),
            ]
        );

        // Number patterns
        self.patterns.insert(
            EntityType::Number,
            vec![
                Regex::new(r"\b\d+(?:\.\d+)?\b").unwrap(),
            ]
        );

        // Date patterns
        self.patterns.insert(
            EntityType::Date,
            vec![
                Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap(),
                Regex::new(r"\b\d{1,2}/\d{1,2}/\d{4}\b").unwrap(),
                Regex::new(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b").unwrap(),
            ]
        );
    }

    pub fn extract_entities(&self, query: &str) -> ExtractionResult {
        let mut entities = Vec::new();

        // Extract entities using pattern matching
        for (entity_type, patterns) in &self.patterns {
            for pattern in patterns {
                for mat in pattern.find_iter(query) {
                    entities.push(Entity {
                        entity_type: entity_type.clone(),
                        value: mat.as_str().to_string(),
                        confidence: self.calculate_confidence(&entity_type, mat.as_str()),
                        start_index: mat.start(),
                        end_index: mat.end(),
                    });
                }
            }
        }

        // Named entity recognition for specific Azure resources
        entities.extend(self.extract_named_entities(query));

        // Sort by position in text
        entities.sort_by_key(|e| e.start_index);

        // Deduplicate overlapping entities
        entities = self.deduplicate_entities(entities);

        // Generate normalized query
        let normalized_query = self.normalize_query(query, &entities);

        // Determine domain
        let domain = self.determine_domain(&entities);

        // Generate intent hints
        let intent_hints = self.generate_intent_hints(&entities);

        ExtractionResult {
            entities,
            normalized_query,
            domain,
            intent_hints,
        }
    }

    fn extract_named_entities(&self, query: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let query_lower = query.to_lowercase();

        // Extract Azure service names
        for service in &self.azure_services {
            if query_lower.contains(&service.to_lowercase()) {
                if let Some(start) = query_lower.find(&service.to_lowercase()) {
                    entities.push(Entity {
                        entity_type: EntityType::AzureService,
                        value: service.clone(),
                        confidence: 0.9,
                        start_index: start,
                        end_index: start + service.len(),
                    });
                }
            }
        }

        // Extract resource type names
        for resource_type in &self.resource_types {
            if query_lower.contains(&resource_type.to_lowercase()) {
                if let Some(start) = query_lower.find(&resource_type.to_lowercase()) {
                    entities.push(Entity {
                        entity_type: EntityType::ResourceType,
                        value: resource_type.clone(),
                        confidence: 0.85,
                        start_index: start,
                        end_index: start + resource_type.len(),
                    });
                }
            }
        }

        entities
    }

    fn calculate_confidence(&self, entity_type: &EntityType, value: &str) -> f64 {
        match entity_type {
            EntityType::ResourceType => {
                if self.resource_types.iter().any(|rt| rt.to_lowercase() == value.to_lowercase()) {
                    0.95
                } else {
                    0.7
                }
            },
            EntityType::AzureService => {
                if self.azure_services.iter().any(|s| s.to_lowercase() == value.to_lowercase()) {
                    0.95
                } else {
                    0.7
                }
            },
            EntityType::Date => 0.9,
            EntityType::Number => 0.8,
            EntityType::Location => 0.9,
            _ => 0.75,
        }
    }

    fn deduplicate_entities(&self, mut entities: Vec<Entity>) -> Vec<Entity> {
        let mut deduplicated = Vec::new();
        
        for entity in entities {
            let is_overlap = deduplicated.iter().any(|existing: &Entity| {
                (entity.start_index < existing.end_index && entity.end_index > existing.start_index)
            });
            
            if !is_overlap {
                deduplicated.push(entity);
            } else {
                // Keep the entity with higher confidence
                if let Some(pos) = deduplicated.iter().position(|existing| {
                    entity.start_index < existing.end_index && entity.end_index > existing.start_index
                }) {
                    if entity.confidence > deduplicated[pos].confidence {
                        deduplicated[pos] = entity;
                    }
                }
            }
        }
        
        deduplicated
    }

    fn normalize_query(&self, query: &str, entities: &[Entity]) -> String {
        let mut normalized = query.to_string();
        
        // Replace entities with placeholders
        for entity in entities.iter().rev() { // Reverse to maintain indices
            let placeholder = format!("[{}]", self.entity_type_to_string(&entity.entity_type));
            normalized.replace_range(entity.start_index..entity.end_index, &placeholder);
        }
        
        normalized
    }

    fn determine_domain(&self, entities: &[Entity]) -> String {
        let mut domains = HashMap::new();
        
        for entity in entities {
            match &entity.entity_type {
                EntityType::ResourceType | EntityType::AzureService => {
                    *domains.entry("resource_management".to_string()).or_insert(0) += 2;
                },
                EntityType::Policy | EntityType::Compliance => {
                    *domains.entry("governance".to_string()).or_insert(0) += 2;
                },
                EntityType::Cost => {
                    *domains.entry("finops".to_string()).or_insert(0) += 2;
                },
                EntityType::Severity => {
                    *domains.entry("security".to_string()).or_insert(0) += 1;
                },
                _ => {},
            }
        }
        
        domains.into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(domain, _)| domain)
            .unwrap_or_else(|| "general".to_string())
    }

    fn generate_intent_hints(&self, entities: &[Entity]) -> Vec<String> {
        let mut hints = Vec::new();
        
        let has_action = entities.iter().any(|e| matches!(e.entity_type, EntityType::Action));
        let has_resource = entities.iter().any(|e| matches!(e.entity_type, EntityType::ResourceType | EntityType::AzureService));
        let has_time = entities.iter().any(|e| matches!(e.entity_type, EntityType::TimeRange | EntityType::Date));
        let has_cost = entities.iter().any(|e| matches!(e.entity_type, EntityType::Cost));
        
        if has_action && has_resource {
            hints.push("resource_action".to_string());
        }
        
        if has_time {
            hints.push("temporal_query".to_string());
        }
        
        if has_cost {
            hints.push("cost_analysis".to_string());
        }
        
        if entities.iter().any(|e| e.value.to_lowercase().contains("compliance") || e.value.to_lowercase().contains("policy")) {
            hints.push("compliance_query".to_string());
        }
        
        hints
    }

    fn entity_type_to_string(&self, entity_type: &EntityType) -> String {
        match entity_type {
            EntityType::ResourceType => "RESOURCE_TYPE",
            EntityType::ResourceName => "RESOURCE_NAME",
            EntityType::AzureService => "AZURE_SERVICE",
            EntityType::Policy => "POLICY",
            EntityType::Location => "LOCATION",
            EntityType::TimeRange => "TIME_RANGE",
            EntityType::Severity => "SEVERITY",
            EntityType::Compliance => "COMPLIANCE",
            EntityType::Cost => "COST",
            EntityType::Action => "ACTION",
            EntityType::Number => "NUMBER",
            EntityType::Date => "DATE",
        }.to_string()
    }

    fn initialize_azure_services() -> Vec<String> {
        vec![
            "Azure Active Directory".to_string(),
            "Azure Monitor".to_string(),
            "Azure Security Center".to_string(),
            "Azure Policy".to_string(),
            "Azure RBAC".to_string(),
            "Azure Resource Manager".to_string(),
            "Azure Key Vault".to_string(),
            "Azure Storage".to_string(),
            "Azure SQL Database".to_string(),
            "Azure App Service".to_string(),
            "Azure Functions".to_string(),
            "Azure Virtual Machines".to_string(),
            "Azure Load Balancer".to_string(),
            "Azure Application Gateway".to_string(),
            "Azure Firewall".to_string(),
            "Azure Sentinel".to_string(),
            "Azure Defender".to_string(),
        ]
    }

    fn initialize_resource_types() -> Vec<String> {
        vec![
            "Microsoft.Compute/virtualMachines".to_string(),
            "Microsoft.Storage/storageAccounts".to_string(),
            "Microsoft.Sql/servers".to_string(),
            "Microsoft.Web/sites".to_string(),
            "Microsoft.KeyVault/vaults".to_string(),
            "Microsoft.Network/networkSecurityGroups".to_string(),
            "Microsoft.Network/loadBalancers".to_string(),
            "Microsoft.Network/virtualNetworks".to_string(),
            "Microsoft.Authorization/policyDefinitions".to_string(),
            "Microsoft.Authorization/roleDefinitions".to_string(),
        ]
    }

    fn initialize_policies() -> Vec<String> {
        vec![
            "Encryption at Rest".to_string(),
            "Encryption in Transit".to_string(),
            "Backup Required".to_string(),
            "Public Access Restricted".to_string(),
            "HTTPS Only".to_string(),
            "Diagnostic Settings Required".to_string(),
            "Audit Logs Enabled".to_string(),
            "MFA Required".to_string(),
            "Password Policy".to_string(),
            "Network Security Group Required".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction() {
        let extractor = EntityExtractor::new();
        let result = extractor.extract_entities("Show me all virtual machines in East US that don't have encryption enabled");
        
        assert!(!result.entities.is_empty());
        assert!(result.entities.iter().any(|e| matches!(e.entity_type, EntityType::ResourceType)));
        assert!(result.entities.iter().any(|e| matches!(e.entity_type, EntityType::Location)));
        assert!(result.entities.iter().any(|e| matches!(e.entity_type, EntityType::Policy)));
    }

    #[test]
    fn test_cost_entity_extraction() {
        let extractor = EntityExtractor::new();
        let result = extractor.extract_entities("Find resources that cost more than $100 per month");
        
        assert!(result.entities.iter().any(|e| matches!(e.entity_type, EntityType::Cost)));
        assert!(result.intent_hints.contains(&"cost_analysis".to_string()));
    }

    #[test]
    fn test_temporal_query() {
        let extractor = EntityExtractor::new();
        let result = extractor.extract_entities("Show violations from the last 30 days");
        
        assert!(result.entities.iter().any(|e| matches!(e.entity_type, EntityType::TimeRange)));
        assert!(result.intent_hints.contains(&"temporal_query".to_string()));
    }
}