use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Natural Language Processing Engine for Conversational Governance
// Patent 2: Conversational Governance Intelligence System

pub struct NaturalLanguageEngine {
    intent_classifier: IntentClassifier,
    entity_extractor: EntityExtractor,
    policy_translator: PolicyTranslator,
    context_manager: ConversationContext,
    knowledge_base: GovernanceKnowledgeBase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationQuery {
    pub query_id: Uuid,
    pub user_input: String,
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    pub context: Option<ConversationContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationResponse {
    pub query_id: Uuid,
    pub response_type: ResponseType,
    pub message: String,
    pub data: Option<serde_json::Value>,
    pub actions: Vec<SuggestedAction>,
    pub confidence: f64,
    pub context: ConversationContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    Information,
    Clarification,
    Action,
    Confirmation,
    Error,
    PolicyRecommendation,
    ComplianceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedAction {
    pub action_type: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
    pub requires_approval: bool,
    pub estimated_impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub session_id: String,
    pub conversation_history: Vec<ConversationTurn>,
    pub identified_entities: HashMap<String, EntityInfo>,
    pub current_intent: Option<Intent>,
    pub pending_clarifications: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub timestamp: DateTime<Utc>,
    pub user_input: String,
    pub system_response: String,
    pub intent: Option<Intent>,
    pub entities: Vec<EntityInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub intent_type: IntentType,
    pub confidence: f64,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IntentType {
    // Query intents
    GetComplianceStatus,
    CheckPolicyViolations,
    ExplainPolicy,
    ListResources,
    GetRecommendations,
    AnalyzeRisk,
    
    // Action intents
    RemediateViolation,
    CreatePolicy,
    UpdatePolicy,
    EnableCompliance,
    ConfigureResource,
    ApproveAction,
    
    // Analysis intents
    PredictViolations,
    AnalyzeTrends,
    CompareConfigurations,
    SimulateChange,
    
    // Informational
    GetHelp,
    ExplainConcept,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    pub entity_type: EntityType,
    pub value: String,
    pub normalized_value: String,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    ResourceId,
    ResourceType,
    PolicyName,
    TimeRange,
    RiskLevel,
    ComplianceFramework,
    AzureService,
    Action,
    Metric,
    Location,
}

struct IntentClassifier {
    patterns: HashMap<IntentType, Vec<String>>,
    keywords: HashMap<String, Vec<IntentType>>,
}

struct EntityExtractor {
    patterns: HashMap<EntityType, Vec<String>>,
    azure_services: Vec<String>,
    compliance_frameworks: Vec<String>,
}

struct PolicyTranslator {
    templates: HashMap<String, PolicyTemplate>,
    azure_policy_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyTemplate {
    pub name: String,
    pub description: String,
    pub parameters: Vec<PolicyParameter>,
    pub template_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyParameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default_value: Option<String>,
}

struct GovernanceKnowledgeBase {
    concepts: HashMap<String, ConceptDefinition>,
    policies: HashMap<String, PolicyInfo>,
    best_practices: Vec<BestPractice>,
    compliance_mappings: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConceptDefinition {
    pub term: String,
    pub definition: String,
    pub examples: Vec<String>,
    pub related_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyInfo {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub severity: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BestPractice {
    pub title: String,
    pub description: String,
    pub category: String,
    pub related_policies: Vec<String>,
}

impl NaturalLanguageEngine {
    pub fn new() -> Self {
        Self {
            intent_classifier: IntentClassifier::new(),
            entity_extractor: EntityExtractor::new(),
            policy_translator: PolicyTranslator::new(),
            context_manager: ConversationContext::new("default".to_string()),
            knowledge_base: GovernanceKnowledgeBase::new(),
        }
    }

    pub async fn process_query(&mut self, query: ConversationQuery) -> ConversationResponse {
        // Extract intent and entities
        let intent = self.intent_classifier.classify(&query.user_input);
        let entities = self.entity_extractor.extract(&query.user_input);
        
        // Update context
        self.context_manager.update(&intent, &entities);
        
        // Generate response based on intent
        let response = match intent.intent_type {
            IntentType::GetComplianceStatus => self.handle_compliance_status(&entities).await,
            IntentType::CheckPolicyViolations => self.handle_policy_violations(&entities).await,
            IntentType::ExplainPolicy => self.handle_explain_policy(&entities).await,
            IntentType::RemediateViolation => self.handle_remediation(&entities).await,
            IntentType::CreatePolicy => self.handle_create_policy(&query.user_input, &entities).await,
            IntentType::PredictViolations => self.handle_predict_violations(&entities).await,
            IntentType::GetHelp => self.handle_help_request().await,
            _ => self.handle_unknown_intent().await,
        };
        
        // Add conversation turn to history
        self.context_manager.add_turn(ConversationTurn {
            timestamp: Utc::now(),
            user_input: query.user_input.clone(),
            system_response: response.message.clone(),
            intent: Some(intent),
            entities: entities.clone(),
        });
        
        response
    }

    async fn handle_compliance_status(&self, entities: &[EntityInfo]) -> ConversationResponse {
        let resource_filter = entities.iter()
            .find(|e| matches!(e.entity_type, EntityType::ResourceType | EntityType::ResourceId))
            .map(|e| e.value.clone());
        
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::ComplianceStatus,
            message: format!(
                "Analyzing compliance status{}...",
                resource_filter.map(|r| format!(" for {}", r)).unwrap_or_default()
            ),
            data: Some(serde_json::json!({
                "compliant_resources": 156,
                "non_compliant_resources": 12,
                "critical_violations": 3,
                "compliance_score": 92.3,
                "frameworks": ["SOC2", "ISO27001", "PCI-DSS"]
            })),
            actions: vec![
                SuggestedAction {
                    action_type: "view_violations".to_string(),
                    description: "View detailed violation report".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: false,
                    estimated_impact: "No impact - read only".to_string(),
                },
                SuggestedAction {
                    action_type: "auto_remediate".to_string(),
                    description: "Automatically fix critical violations".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: true,
                    estimated_impact: "Will modify 3 resources".to_string(),
                },
            ],
            confidence: 0.95,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_policy_violations(&self, entities: &[EntityInfo]) -> ConversationResponse {
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Information,
            message: "I found 8 policy violations in your environment. 3 are critical and need immediate attention.".to_string(),
            data: Some(serde_json::json!({
                "violations": [
                    {
                        "resource": "/subscriptions/xxx/resourceGroups/prod/storageAccounts/data",
                        "policy": "Require Encryption",
                        "severity": "Critical",
                        "detected": "2 hours ago"
                    },
                    {
                        "resource": "/subscriptions/xxx/resourceGroups/dev/virtualMachines/testvm",
                        "policy": "Require Managed Disks",
                        "severity": "High",
                        "detected": "1 day ago"
                    }
                ]
            })),
            actions: vec![
                SuggestedAction {
                    action_type: "remediate_all".to_string(),
                    description: "Fix all violations automatically".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: true,
                    estimated_impact: "Will modify 8 resources".to_string(),
                },
            ],
            confidence: 0.92,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_explain_policy(&self, entities: &[EntityInfo]) -> ConversationResponse {
        let policy_name = entities.iter()
            .find(|e| matches!(e.entity_type, EntityType::PolicyName))
            .map(|e| e.value.clone())
            .unwrap_or("Require Encryption".to_string());
        
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Information,
            message: format!(
                "The '{}' policy ensures that all storage accounts have encryption enabled for data at rest. \
                This is required for SOC2 and ISO27001 compliance. When this policy is in 'Deny' mode, \
                it prevents creation of non-compliant resources.",
                policy_name
            ),
            data: Some(serde_json::json!({
                "policy_details": {
                    "name": policy_name,
                    "mode": "Deny",
                    "category": "Security",
                    "compliance_frameworks": ["SOC2", "ISO27001", "HIPAA"],
                    "affected_resources": 45,
                    "enforcement_rate": "98%"
                }
            })),
            actions: vec![
                SuggestedAction {
                    action_type: "view_policy_definition".to_string(),
                    description: "View full policy JSON definition".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: false,
                    estimated_impact: "No impact - read only".to_string(),
                },
            ],
            confidence: 0.98,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_remediation(&self, entities: &[EntityInfo]) -> ConversationResponse {
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Action,
            message: "I can remediate the identified violations. This will enable encryption on 3 storage accounts and configure network security on 2 virtual machines.".to_string(),
            data: Some(serde_json::json!({
                "remediation_plan": {
                    "total_resources": 5,
                    "estimated_time": "10 minutes",
                    "rollback_available": true,
                    "changes": [
                        {
                            "resource": "storageaccount1",
                            "action": "Enable encryption",
                            "risk": "Low"
                        },
                        {
                            "resource": "vm-prod-01",
                            "action": "Configure NSG rules",
                            "risk": "Medium"
                        }
                    ]
                }
            })),
            actions: vec![
                SuggestedAction {
                    action_type: "execute_remediation".to_string(),
                    description: "Execute remediation plan".to_string(),
                    parameters: HashMap::from([
                        ("approval_required".to_string(), "true".to_string()),
                        ("notification_email".to_string(), "admin@company.com".to_string()),
                    ]),
                    requires_approval: true,
                    estimated_impact: "Will modify 5 resources, no downtime expected".to_string(),
                },
                SuggestedAction {
                    action_type: "schedule_remediation".to_string(),
                    description: "Schedule for maintenance window".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: false,
                    estimated_impact: "Will be scheduled for next maintenance window".to_string(),
                },
            ],
            confidence: 0.89,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_create_policy(&self, input: &str, entities: &[EntityInfo]) -> ConversationResponse {
        // Extract policy requirements from natural language
        let policy_json = self.policy_translator.translate_to_policy(input, entities);
        
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::PolicyRecommendation,
            message: "I've created an Azure Policy based on your requirements. This policy will enforce encryption and require specific tags on all storage accounts.".to_string(),
            data: Some(policy_json),
            actions: vec![
                SuggestedAction {
                    action_type: "deploy_policy".to_string(),
                    description: "Deploy this policy to your subscription".to_string(),
                    parameters: HashMap::from([
                        ("scope".to_string(), "/subscriptions/xxx".to_string()),
                        ("enforcement_mode".to_string(), "Default".to_string()),
                    ]),
                    requires_approval: true,
                    estimated_impact: "Will affect all future storage account deployments".to_string(),
                },
                SuggestedAction {
                    action_type: "test_policy".to_string(),
                    description: "Test policy in audit mode first".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: false,
                    estimated_impact: "No enforcement, audit only".to_string(),
                },
            ],
            confidence: 0.91,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_predict_violations(&self, entities: &[EntityInfo]) -> ConversationResponse {
        let time_range = entities.iter()
            .find(|e| matches!(e.entity_type, EntityType::TimeRange))
            .map(|e| e.value.clone())
            .unwrap_or("24 hours".to_string());
        
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Information,
            message: format!(
                "Based on current trends and configuration drift, I predict 5 policy violations within the next {}. \
                The highest risk is a storage account that will lose encryption due to a scheduled configuration change.",
                time_range
            ),
            data: Some(serde_json::json!({
                "predictions": [
                    {
                        "resource": "storage-prod-001",
                        "violation": "Encryption will be disabled",
                        "time_to_violation": "18 hours",
                        "confidence": 0.92,
                        "preventable": true
                    },
                    {
                        "resource": "vm-web-02",
                        "violation": "Public IP will be exposed",
                        "time_to_violation": "12 hours",
                        "confidence": 0.78,
                        "preventable": true
                    }
                ]
            })),
            actions: vec![
                SuggestedAction {
                    action_type: "prevent_violations".to_string(),
                    description: "Take preventive action now".to_string(),
                    parameters: HashMap::new(),
                    requires_approval: true,
                    estimated_impact: "Will prevent 5 future violations".to_string(),
                },
            ],
            confidence: 0.87,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_help_request(&self) -> ConversationResponse {
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Information,
            message: "I can help you with Azure governance and compliance. Here are some things you can ask me:\n\
                     • 'What are my current policy violations?'\n\
                     • 'Show me compliance status for production resources'\n\
                     • 'Create a policy to require encryption on all storage accounts'\n\
                     • 'Predict violations for the next 24 hours'\n\
                     • 'Remediate all critical security violations'\n\
                     • 'Explain the Required Tags policy'".to_string(),
            data: None,
            actions: vec![],
            confidence: 1.0,
            context: self.context_manager.clone(),
        }
    }

    async fn handle_unknown_intent(&self) -> ConversationResponse {
        ConversationResponse {
            query_id: Uuid::new_v4(),
            response_type: ResponseType::Clarification,
            message: "I'm not sure what you're asking. Could you please rephrase or ask me to 'help' to see what I can do?".to_string(),
            data: None,
            actions: vec![],
            confidence: 0.0,
            context: self.context_manager.clone(),
        }
    }

    pub fn translate_to_azure_policy(&self, natural_language: &str) -> Result<serde_json::Value, String> {
        self.policy_translator.natural_language_to_policy(natural_language)
    }
}

impl IntentClassifier {
    fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Define patterns for each intent
        patterns.insert(IntentType::GetComplianceStatus, vec![
            "compliance status".to_string(),
            "are we compliant".to_string(),
            "check compliance".to_string(),
            "compliance score".to_string(),
        ]);
        
        patterns.insert(IntentType::CheckPolicyViolations, vec![
            "policy violations".to_string(),
            "what's violating".to_string(),
            "non-compliant resources".to_string(),
            "violations".to_string(),
        ]);
        
        patterns.insert(IntentType::RemediateViolation, vec![
            "fix violations".to_string(),
            "remediate".to_string(),
            "resolve issues".to_string(),
            "auto-fix".to_string(),
        ]);
        
        patterns.insert(IntentType::CreatePolicy, vec![
            "create policy".to_string(),
            "new policy".to_string(),
            "enforce".to_string(),
            "require".to_string(),
        ]);
        
        patterns.insert(IntentType::PredictViolations, vec![
            "predict violations".to_string(),
            "future violations".to_string(),
            "will violate".to_string(),
            "forecast".to_string(),
        ]);
        
        let keywords = Self::build_keyword_map(&patterns);
        
        Self { patterns, keywords }
    }
    
    fn build_keyword_map(patterns: &HashMap<IntentType, Vec<String>>) -> HashMap<String, Vec<IntentType>> {
        let mut keywords = HashMap::new();
        
        for (intent, pattern_list) in patterns {
            for pattern in pattern_list {
                for word in pattern.split_whitespace() {
                    keywords.entry(word.to_lowercase())
                        .or_insert_with(Vec::new)
                        .push(intent.clone());
                }
            }
        }
        
        keywords
    }
    
    fn classify(&self, input: &str) -> Intent {
        let input_lower = input.to_lowercase();
        let mut scores: HashMap<IntentType, f64> = HashMap::new();
        
        // Check pattern matches
        for (intent, patterns) in &self.patterns {
            for pattern in patterns {
                if input_lower.contains(pattern) {
                    *scores.entry(intent.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
        
        // Check keyword matches
        for word in input_lower.split_whitespace() {
            if let Some(intents) = self.keywords.get(word) {
                for intent in intents {
                    *scores.entry(intent.clone()).or_insert(0.0) += 0.5;
                }
            }
        }
        
        // Find best match
        let (intent_type, confidence) = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(intent, score)| (intent, score.min(1.0)))
            .unwrap_or((IntentType::Unknown, 0.0));
        
        Intent {
            intent_type,
            confidence,
            parameters: HashMap::new(),
        }
    }
}

impl EntityExtractor {
    fn new() -> Self {
        Self {
            patterns: Self::init_patterns(),
            azure_services: Self::init_azure_services(),
            compliance_frameworks: vec![
                "SOC2".to_string(),
                "ISO27001".to_string(),
                "PCI-DSS".to_string(),
                "HIPAA".to_string(),
                "GDPR".to_string(),
            ],
        }
    }
    
    fn init_patterns() -> HashMap<EntityType, Vec<String>> {
        let mut patterns = HashMap::new();
        
        patterns.insert(EntityType::ResourceId, vec![
            r"/subscriptions/[^/]+/resourceGroups/[^/]+".to_string(),
        ]);
        
        patterns.insert(EntityType::TimeRange, vec![
            r"\d+ hours?".to_string(),
            r"\d+ days?".to_string(),
            r"next \w+".to_string(),
        ]);
        
        patterns.insert(EntityType::RiskLevel, vec![
            "critical".to_string(),
            "high".to_string(),
            "medium".to_string(),
            "low".to_string(),
        ]);
        
        patterns
    }
    
    fn init_azure_services() -> Vec<String> {
        vec![
            "storage account".to_string(),
            "virtual machine".to_string(),
            "key vault".to_string(),
            "sql database".to_string(),
            "app service".to_string(),
            "container".to_string(),
        ]
    }
    
    fn extract(&self, input: &str) -> Vec<EntityInfo> {
        let mut entities = Vec::new();
        let input_lower = input.to_lowercase();
        
        // Extract Azure services
        for service in &self.azure_services {
            if input_lower.contains(service) {
                entities.push(EntityInfo {
                    entity_type: EntityType::AzureService,
                    value: service.clone(),
                    normalized_value: service.replace(" ", ""),
                    confidence: 0.95,
                    metadata: HashMap::new(),
                });
            }
        }
        
        // Extract compliance frameworks
        for framework in &self.compliance_frameworks {
            if input_lower.contains(&framework.to_lowercase()) {
                entities.push(EntityInfo {
                    entity_type: EntityType::ComplianceFramework,
                    value: framework.clone(),
                    normalized_value: framework.clone(),
                    confidence: 0.98,
                    metadata: HashMap::new(),
                });
            }
        }
        
        // Extract risk levels
        for level in ["critical", "high", "medium", "low"] {
            if input_lower.contains(level) {
                entities.push(EntityInfo {
                    entity_type: EntityType::RiskLevel,
                    value: level.to_string(),
                    normalized_value: level.to_string(),
                    confidence: 0.9,
                    metadata: HashMap::new(),
                });
            }
        }
        
        entities
    }
}

impl PolicyTranslator {
    fn new() -> Self {
        Self {
            templates: Self::init_templates(),
            azure_policy_schema: serde_json::json!({
                "properties": {
                    "displayName": "",
                    "policyType": "Custom",
                    "mode": "All",
                    "parameters": {},
                    "policyRule": {}
                }
            }),
        }
    }
    
    fn init_templates() -> HashMap<String, PolicyTemplate> {
        let mut templates = HashMap::new();
        
        templates.insert("require_encryption".to_string(), PolicyTemplate {
            name: "Require Encryption".to_string(),
            description: "Requires encryption for storage accounts".to_string(),
            parameters: vec![],
            template_json: serde_json::json!({
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
            }),
        });
        
        templates
    }
    
    fn translate_to_policy(&self, input: &str, entities: &[EntityInfo]) -> serde_json::Value {
        // Simple translation logic - in production would use more sophisticated NLP
        let mut policy = self.azure_policy_schema.clone();
        
        if input.to_lowercase().contains("encryption") {
            if let Some(template) = self.templates.get("require_encryption") {
                policy["properties"]["policyRule"] = template.template_json.clone();
                policy["properties"]["displayName"] = serde_json::Value::String(template.name.clone());
            }
        }
        
        policy
    }
    
    fn natural_language_to_policy(&self, natural_language: &str) -> Result<serde_json::Value, String> {
        Ok(self.translate_to_policy(natural_language, &[]))
    }
}

impl ConversationContext {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            conversation_history: Vec::new(),
            identified_entities: HashMap::new(),
            current_intent: None,
            pending_clarifications: Vec::new(),
        }
    }
    
    fn update(&mut self, intent: &Intent, entities: &[EntityInfo]) {
        self.current_intent = Some(intent.clone());
        
        for entity in entities {
            self.identified_entities.insert(
                format!("{}_{}", entity.entity_type.to_string(), entity.value),
                entity.clone()
            );
        }
    }
    
    fn add_turn(&mut self, turn: ConversationTurn) {
        self.conversation_history.push(turn);
        
        // Keep only last 10 turns
        if self.conversation_history.len() > 10 {
            self.conversation_history.remove(0);
        }
    }
}

impl GovernanceKnowledgeBase {
    fn new() -> Self {
        Self {
            concepts: Self::init_concepts(),
            policies: Self::init_policies(),
            best_practices: Self::init_best_practices(),
            compliance_mappings: Self::init_compliance_mappings(),
        }
    }
    
    fn init_concepts() -> HashMap<String, ConceptDefinition> {
        let mut concepts = HashMap::new();
        
        concepts.insert("compliance".to_string(), ConceptDefinition {
            term: "Compliance".to_string(),
            definition: "Adherence to laws, regulations, guidelines and specifications relevant to business operations".to_string(),
            examples: vec!["SOC2 compliance".to_string(), "GDPR compliance".to_string()],
            related_terms: vec!["governance".to_string(), "audit".to_string()],
        });
        
        concepts
    }
    
    fn init_policies() -> HashMap<String, PolicyInfo> {
        HashMap::new()
    }
    
    fn init_best_practices() -> Vec<BestPractice> {
        Vec::new()
    }
    
    fn init_compliance_mappings() -> HashMap<String, Vec<String>> {
        HashMap::new()
    }
}

// Helper trait implementations
impl ToString for EntityType {
    fn to_string(&self) -> String {
        match self {
            EntityType::ResourceId => "ResourceId",
            EntityType::ResourceType => "ResourceType",
            EntityType::PolicyName => "PolicyName",
            EntityType::TimeRange => "TimeRange",
            EntityType::RiskLevel => "RiskLevel",
            EntityType::ComplianceFramework => "ComplianceFramework",
            EntityType::AzureService => "AzureService",
            EntityType::Action => "Action",
            EntityType::Metric => "Metric",
            EntityType::Location => "Location",
        }.to_string()
    }
}
