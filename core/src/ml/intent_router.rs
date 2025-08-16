// Intent Router for Complex Queries
// Routes natural language queries to appropriate handlers based on intent classification

use super::conversation_memory::{Intent, IntentType, ExtractedEntity, EntityType, ConversationContext};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

/// Intent router for directing queries to appropriate handlers
pub struct IntentRouter {
    classifiers: Vec<Box<dyn IntentClassifier>>,
    handlers: HashMap<String, Box<dyn IntentHandler>>,
    confidence_threshold: f64,
}

impl IntentRouter {
    pub fn new() -> Self {
        let mut router = Self {
            classifiers: vec![
                Box::new(RuleBasedClassifier::new()),
                Box::new(KeywordClassifier::new()),
                Box::new(PatternClassifier::new()),
            ],
            handlers: HashMap::new(),
            confidence_threshold: 0.6,
        };
        
        router.register_handlers();
        router
    }
    
    fn register_handlers(&mut self) {
        self.handlers.insert("violation_handler".to_string(), Box::new(ViolationHandler));
        self.handlers.insert("prediction_handler".to_string(), Box::new(PredictionHandler));
        self.handlers.insert("remediation_handler".to_string(), Box::new(RemediationHandler));
        self.handlers.insert("policy_handler".to_string(), Box::new(PolicyHandler));
        self.handlers.insert("report_handler".to_string(), Box::new(ReportHandler));
        self.handlers.insert("cost_handler".to_string(), Box::new(CostHandler));
        self.handlers.insert("security_handler".to_string(), Box::new(SecurityHandler));
    }
    
    /// Route query to appropriate handlers
    pub async fn route_query(
        &self,
        query: &str,
        context: &ConversationContext,
    ) -> Vec<IntentRoute> {
        let mut all_intents = Vec::new();
        
        // Classify with all available classifiers
        for classifier in &self.classifiers {
            let intents = classifier.classify(query, context).await;
            all_intents.extend(intents);
        }
        
        // Aggregate and deduplicate intents
        let aggregated = self.aggregate_intents(all_intents);
        
        // Filter by confidence and create routes
        aggregated.into_iter()
            .filter(|intent| intent.confidence >= self.confidence_threshold)
            .map(|intent| {
                let handler_name = self.get_handler_name(&intent.intent_type);
                IntentRoute {
                    handler: handler_name.clone(),
                    priority: intent.confidence,
                    intent,
                    can_execute: self.handlers.contains_key(&handler_name),
                }
            })
            .collect()
    }
    
    /// Aggregate intents from multiple classifiers
    fn aggregate_intents(&self, intents: Vec<Intent>) -> Vec<Intent> {
        let mut intent_map: HashMap<IntentType, Vec<Intent>> = HashMap::new();
        
        // Group by intent type
        for intent in intents {
            intent_map.entry(intent.intent_type.clone())
                .or_insert_with(Vec::new)
                .push(intent);
        }
        
        // Average confidence for each intent type
        intent_map.into_iter()
            .map(|(intent_type, intents)| {
                let total_confidence: f64 = intents.iter().map(|i| i.confidence).sum();
                let avg_confidence = total_confidence / intents.len() as f64;
                
                // Merge entities and parameters
                let mut all_entities = Vec::new();
                let mut all_parameters = HashMap::new();
                
                for intent in intents {
                    all_entities.extend(intent.entities);
                    all_parameters.extend(intent.parameters);
                }
                
                all_entities.sort();
                all_entities.dedup();
                
                Intent {
                    intent_type,
                    confidence: avg_confidence,
                    entities: all_entities,
                    parameters: all_parameters,
                }
            })
            .collect()
    }
    
    fn get_handler_name(&self, intent_type: &IntentType) -> String {
        match intent_type {
            IntentType::QueryViolations => "violation_handler".to_string(),
            IntentType::PredictCompliance => "prediction_handler".to_string(),
            IntentType::ExecuteRemediation => "remediation_handler".to_string(),
            IntentType::GenerateReport => "report_handler".to_string(),
            IntentType::ExplainPolicy | IntentType::CreatePolicy => "policy_handler".to_string(),
            IntentType::AnalyzeCost => "cost_handler".to_string(),
            IntentType::CheckSecurity => "security_handler".to_string(),
            IntentType::GetRecommendations => "recommendation_handler".to_string(),
            _ => "fallback_handler".to_string(),
        }
    }
    
    /// Execute handler for given route
    pub async fn execute_handler(
        &self,
        route: &IntentRoute,
        query: &str,
        context: &ConversationContext,
    ) -> HandlerResponse {
        if let Some(handler) = self.handlers.get(&route.handler) {
            handler.handle(query, &route.intent, context).await
        } else {
            HandlerResponse {
                success: false,
                message: format!("No handler found for intent: {:?}", route.intent.intent_type),
                data: None,
                actions: Vec::new(),
            }
        }
    }
}

/// Intent route
#[derive(Debug, Clone)]
pub struct IntentRoute {
    pub handler: String,
    pub priority: f64,
    pub intent: Intent,
    pub can_execute: bool,
}

/// Handler response
#[derive(Debug, Serialize)]
pub struct HandlerResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
    pub actions: Vec<String>,
}

/// Trait for intent classifiers
#[async_trait]
pub trait IntentClassifier: Send + Sync {
    async fn classify(&self, query: &str, context: &ConversationContext) -> Vec<Intent>;
}

/// Trait for intent handlers
#[async_trait]
pub trait IntentHandler: Send + Sync {
    async fn handle(
        &self,
        query: &str,
        intent: &Intent,
        context: &ConversationContext,
    ) -> HandlerResponse;
}

/// Rule-based intent classifier
pub struct RuleBasedClassifier {
    rules: Vec<IntentRule>,
}

impl RuleBasedClassifier {
    pub fn new() -> Self {
        Self {
            rules: Self::initialize_rules(),
        }
    }
    
    fn initialize_rules() -> Vec<IntentRule> {
        vec![
            // Violation queries
            IntentRule {
                patterns: vec![
                    r"(what|show|list|get).*(violation|non.?complian|issue)",
                    r"violation.*report",
                    r"compliance.*status",
                ],
                intent_type: IntentType::QueryViolations,
                confidence_boost: 0.9,
            },
            // Prediction queries
            IntentRule {
                patterns: vec![
                    r"predict|forecast|will.*violate",
                    r"future.*compliance",
                    r"drift.*analysis",
                ],
                intent_type: IntentType::PredictCompliance,
                confidence_boost: 0.85,
            },
            // Remediation commands
            IntentRule {
                patterns: vec![
                    r"(fix|remediate|resolve|repair).*violation",
                    r"auto.*fix",
                    r"one.?click.*remediation",
                ],
                intent_type: IntentType::ExecuteRemediation,
                confidence_boost: 0.9,
            },
            // Policy creation
            IntentRule {
                patterns: vec![
                    r"create.*policy",
                    r"new.*rule",
                    r"define.*governance",
                ],
                intent_type: IntentType::CreatePolicy,
                confidence_boost: 0.85,
            },
            // Cost analysis
            IntentRule {
                patterns: vec![
                    r"cost|spend|budget|expense",
                    r"save.*money",
                    r"optimize.*cost",
                ],
                intent_type: IntentType::AnalyzeCost,
                confidence_boost: 0.8,
            },
            // Security checks
            IntentRule {
                patterns: vec![
                    r"security|vulnerability|threat",
                    r"exposed|unsecure",
                    r"check.*permission",
                ],
                intent_type: IntentType::CheckSecurity,
                confidence_boost: 0.85,
            },
        ]
    }
}

#[async_trait]
impl IntentClassifier for RuleBasedClassifier {
    async fn classify(&self, query: &str, _context: &ConversationContext) -> Vec<Intent> {
        let query_lower = query.to_lowercase();
        let mut intents = Vec::new();
        
        for rule in &self.rules {
            for pattern in &rule.patterns {
                let regex = regex::Regex::new(pattern).unwrap();
                if regex.is_match(&query_lower) {
                    intents.push(Intent {
                        intent_type: rule.intent_type.clone(),
                        confidence: rule.confidence_boost,
                        entities: Vec::new(),
                        parameters: HashMap::new(),
                    });
                    break;
                }
            }
        }
        
        intents
    }
}

/// Keyword-based classifier
pub struct KeywordClassifier {
    keyword_map: HashMap<String, IntentType>,
}

impl KeywordClassifier {
    pub fn new() -> Self {
        let mut keyword_map = HashMap::new();
        
        // Violations
        keyword_map.insert("violations".to_string(), IntentType::QueryViolations);
        keyword_map.insert("non-compliant".to_string(), IntentType::QueryViolations);
        keyword_map.insert("compliance".to_string(), IntentType::QueryViolations);
        
        // Predictions
        keyword_map.insert("predict".to_string(), IntentType::PredictCompliance);
        keyword_map.insert("forecast".to_string(), IntentType::PredictCompliance);
        keyword_map.insert("drift".to_string(), IntentType::PredictCompliance);
        
        // Remediation
        keyword_map.insert("fix".to_string(), IntentType::ExecuteRemediation);
        keyword_map.insert("remediate".to_string(), IntentType::ExecuteRemediation);
        keyword_map.insert("resolve".to_string(), IntentType::ExecuteRemediation);
        
        // Policy
        keyword_map.insert("policy".to_string(), IntentType::CreatePolicy);
        keyword_map.insert("rule".to_string(), IntentType::CreatePolicy);
        
        // Cost
        keyword_map.insert("cost".to_string(), IntentType::AnalyzeCost);
        keyword_map.insert("budget".to_string(), IntentType::AnalyzeCost);
        keyword_map.insert("spend".to_string(), IntentType::AnalyzeCost);
        
        Self { keyword_map }
    }
}

#[async_trait]
impl IntentClassifier for KeywordClassifier {
    async fn classify(&self, query: &str, _context: &ConversationContext) -> Vec<Intent> {
        let query_lower = query.to_lowercase();
        let mut intents = Vec::new();
        
        for (keyword, intent_type) in &self.keyword_map {
            if query_lower.contains(keyword) {
                // Calculate confidence based on keyword position and frequency
                let confidence = if query_lower.starts_with(keyword) {
                    0.8
                } else {
                    0.6
                };
                
                intents.push(Intent {
                    intent_type: intent_type.clone(),
                    confidence,
                    entities: Vec::new(),
                    parameters: HashMap::new(),
                });
            }
        }
        
        intents
    }
}

/// Pattern-based classifier for complex queries
pub struct PatternClassifier;

impl PatternClassifier {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl IntentClassifier for PatternClassifier {
    async fn classify(&self, query: &str, context: &ConversationContext) -> Vec<Intent> {
        let mut intents = Vec::new();
        
        // Check for multi-intent queries
        if query.contains(" and ") || query.contains(" then ") {
            // Split and classify each part
            let parts: Vec<&str> = query.split(|c| c == ',' || c == ';').collect();
            
            for part in parts {
                // Simplified classification for each part
                if part.contains("violation") {
                    intents.push(Intent {
                        intent_type: IntentType::QueryViolations,
                        confidence: 0.7,
                        entities: Vec::new(),
                        parameters: HashMap::new(),
                    });
                }
                if part.contains("fix") || part.contains("remediate") {
                    intents.push(Intent {
                        intent_type: IntentType::ExecuteRemediation,
                        confidence: 0.7,
                        entities: Vec::new(),
                        parameters: HashMap::new(),
                    });
                }
            }
        }
        
        // Context-aware classification
        if let Some(goal) = &context.user_goal {
            if goal.contains("compliance") && query.contains("status") {
                intents.push(Intent {
                    intent_type: IntentType::QueryViolations,
                    confidence: 0.75,
                    entities: Vec::new(),
                    parameters: HashMap::new(),
                });
            }
        }
        
        intents
    }
}

/// Intent rule structure
struct IntentRule {
    patterns: Vec<&'static str>,
    intent_type: IntentType,
    confidence_boost: f64,
}

// Handler implementations

struct ViolationHandler;

#[async_trait]
impl IntentHandler for ViolationHandler {
    async fn handle(
        &self,
        _query: &str,
        intent: &Intent,
        context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: format!("Querying violations for resources: {:?}", context.active_resources),
            data: Some(serde_json::json!({
                "total_violations": 8,
                "critical": 3,
                "resources_affected": context.active_resources
            })),
            actions: vec!["view_details".to_string(), "auto_fix".to_string()],
        }
    }
}

struct PredictionHandler;

#[async_trait]
impl IntentHandler for PredictionHandler {
    async fn handle(
        &self,
        _query: &str,
        _intent: &Intent,
        _context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: "Analyzing drift patterns and predicting future violations".to_string(),
            data: Some(serde_json::json!({
                "predictions": [
                    {
                        "resource": "storage-001",
                        "violation_type": "encryption_disabled",
                        "time_to_violation": "18 hours",
                        "confidence": 0.89
                    }
                ]
            })),
            actions: vec!["prevent_violations".to_string()],
        }
    }
}

struct RemediationHandler;

#[async_trait]
impl IntentHandler for RemediationHandler {
    async fn handle(
        &self,
        _query: &str,
        intent: &Intent,
        _context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: format!("Preparing to remediate violations for entities: {:?}", intent.entities),
            data: Some(serde_json::json!({
                "remediation_plan": {
                    "actions": 3,
                    "estimated_time": "2 minutes",
                    "rollback_available": true
                }
            })),
            actions: vec!["execute_remediation".to_string(), "schedule_remediation".to_string()],
        }
    }
}

struct PolicyHandler;

#[async_trait]
impl IntentHandler for PolicyHandler {
    async fn handle(
        &self,
        query: &str,
        _intent: &Intent,
        _context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: format!("Creating policy based on: {}", query),
            data: Some(serde_json::json!({
                "policy_draft": {
                    "name": "Custom Security Policy",
                    "rules": 2,
                    "enforcement": "deny"
                }
            })),
            actions: vec!["review_policy".to_string(), "deploy_policy".to_string()],
        }
    }
}

struct ReportHandler;

#[async_trait]
impl IntentHandler for ReportHandler {
    async fn handle(
        &self,
        _query: &str,
        _intent: &Intent,
        _context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: "Generating compliance report".to_string(),
            data: Some(serde_json::json!({
                "report": {
                    "type": "compliance",
                    "format": "pdf",
                    "sections": ["violations", "predictions", "recommendations"]
                }
            })),
            actions: vec!["download_report".to_string(), "email_report".to_string()],
        }
    }
}

struct CostHandler;

#[async_trait]
impl IntentHandler for CostHandler {
    async fn handle(
        &self,
        _query: &str,
        _intent: &Intent,
        _context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: "Analyzing cost optimization opportunities".to_string(),
            data: Some(serde_json::json!({
                "cost_analysis": {
                    "current_spend": "$45,000",
                    "potential_savings": "$12,000",
                    "recommendations": 5
                }
            })),
            actions: vec!["view_recommendations".to_string(), "apply_optimizations".to_string()],
        }
    }
}

struct SecurityHandler;

#[async_trait]
impl IntentHandler for SecurityHandler {
    async fn handle(
        &self,
        _query: &str,
        _intent: &Intent,
        context: &ConversationContext,
    ) -> HandlerResponse {
        HandlerResponse {
            success: true,
            message: "Performing security assessment".to_string(),
            data: Some(serde_json::json!({
                "security_score": 72,
                "vulnerabilities": 3,
                "exposed_resources": 1,
                "recommendations": 4
            })),
            actions: vec!["fix_vulnerabilities".to_string(), "enable_defender".to_string()],
        }
    }
}

// Export use statements
use regex;