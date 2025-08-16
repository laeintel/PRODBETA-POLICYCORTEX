// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Advanced Query Understanding System for Natural Language Governance
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::ml::entity_extractor::{EntityExtractor, ExtractionResult, EntityType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstanding {
    pub intent: Intent,
    pub entities: ExtractionResult,
    pub semantic_parse: SemanticParse,
    pub execution_plan: ExecutionPlan,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub primary: IntentType,
    pub secondary: Vec<IntentType>,
    pub confidence: f64,
    pub domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum IntentType {
    // Query Intents
    List,
    Show,
    Find,
    Search,
    
    // Action Intents
    Create,
    Update,
    Delete,
    Fix,
    Remediate,
    
    // Analysis Intents
    Analyze,
    Compare,
    Predict,
    Explain,
    
    // Compliance Intents
    CheckCompliance,
    ValidatePolicy,
    AuditResources,
    
    // Cost Intents
    CostAnalysis,
    OptimizeCosts,
    BudgetTracking,
    
    // Security Intents
    SecurityAnalysis,
    ThreatDetection,
    AccessReview,
    
    // Monitoring Intents
    Monitor,
    Alert,
    Report,
    
    // Unknown
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticParse {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub modifiers: Vec<Modifier>,
    pub relationships: Vec<Relationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modifier {
    pub modifier_type: ModifierType,
    pub value: String,
    pub applies_to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModifierType {
    Temporal,
    Spatial,
    Conditional,
    Quantitative,
    Qualitative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: String,
    pub relation_type: RelationType,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationType {
    DependsOn,
    Contains,
    LocatedIn,
    OwnedBy,
    ConnectedTo,
    SimilarTo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub estimated_time: f64, // seconds
    pub complexity: Complexity,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_type: StepType,
    pub description: String,
    pub api_endpoint: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub depends_on: Vec<usize>, // indices of prerequisite steps
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    DataRetrieval,
    Analysis,
    Computation,
    Remediation,
    Notification,
    Validation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Complexity {
    Simple,
    Moderate,
    Complex,
    Critical,
}

pub struct QueryUnderstandingEngine {
    entity_extractor: EntityExtractor,
    intent_patterns: HashMap<IntentType, Vec<String>>,
    domain_keywords: HashMap<String, Vec<String>>,
}

impl QueryUnderstandingEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            entity_extractor: EntityExtractor::new(),
            intent_patterns: HashMap::new(),
            domain_keywords: HashMap::new(),
        };
        
        engine.initialize_intent_patterns();
        engine.initialize_domain_keywords();
        engine
    }

    pub fn understand_query(&self, query: &str) -> QueryUnderstanding {
        // Extract entities
        let entities = self.entity_extractor.extract_entities(query);
        
        // Determine intent
        let intent = self.classify_intent(query, &entities);
        
        // Parse semantics
        let semantic_parse = self.parse_semantics(query, &entities);
        
        // Generate execution plan
        let execution_plan = self.generate_execution_plan(&intent, &entities, &semantic_parse);
        
        // Calculate overall confidence
        let confidence = self.calculate_confidence(&intent, &entities, &semantic_parse);
        
        QueryUnderstanding {
            intent,
            entities,
            semantic_parse,
            execution_plan,
            confidence,
        }
    }

    fn classify_intent(&self, query: &str, entities: &ExtractionResult) -> Intent {
        let query_lower = query.to_lowercase();
        let mut intent_scores = HashMap::new();
        
        // Score based on pattern matching
        for (intent_type, patterns) in &self.intent_patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if query_lower.contains(pattern) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                intent_scores.insert(intent_type.clone(), score);
            }
        }
        
        // Boost scores based on entities
        for entity in &entities.entities {
            match entity.entity_type {
                EntityType::Action => {
                    if entity.value.to_lowercase().contains("show") || entity.value.to_lowercase().contains("list") {
                        *intent_scores.entry(IntentType::List).or_insert(0.0) += 0.5;
                    }
                    if entity.value.to_lowercase().contains("fix") || entity.value.to_lowercase().contains("remediate") {
                        *intent_scores.entry(IntentType::Remediate).or_insert(0.0) += 0.5;
                    }
                    if entity.value.to_lowercase().contains("create") {
                        *intent_scores.entry(IntentType::Create).or_insert(0.0) += 0.5;
                    }
                },
                EntityType::Cost => {
                    *intent_scores.entry(IntentType::CostAnalysis).or_insert(0.0) += 0.3;
                },
                _ => {},
            }
        }
        
        // Determine primary intent
        let (primary_intent, primary_score) = intent_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(intent, score)| (intent.clone(), *score))
            .unwrap_or((IntentType::Unknown, 0.0));
        
        // Determine secondary intents
        let secondary: Vec<IntentType> = intent_scores.iter()
            .filter(|(intent, score)| intent != &&primary_intent && **score > 0.3)
            .map(|(intent, _)| intent.clone())
            .collect();
        
        let confidence = (primary_score / (1.0 + primary_score)).min(1.0);
        let domain = entities.domain.clone();
        
        Intent {
            primary: primary_intent,
            secondary,
            confidence,
            domain,
        }
    }

    fn parse_semantics(&self, query: &str, entities: &ExtractionResult) -> SemanticParse {
        let mut subject = None;
        let mut predicate = None;
        let mut object = None;
        let mut modifiers = Vec::new();
        let mut relationships = Vec::new();
        
        // Simple semantic parsing based on entities and patterns
        for entity in &entities.entities {
            match entity.entity_type {
                EntityType::ResourceType | EntityType::AzureService => {
                    if subject.is_none() {
                        subject = Some(entity.value.clone());
                    }
                },
                EntityType::Action => {
                    if predicate.is_none() {
                        predicate = Some(entity.value.clone());
                    }
                },
                EntityType::Policy => {
                    if object.is_none() {
                        object = Some(entity.value.clone());
                    }
                },
                EntityType::TimeRange | EntityType::Date => {
                    modifiers.push(Modifier {
                        modifier_type: ModifierType::Temporal,
                        value: entity.value.clone(),
                        applies_to: subject.clone().unwrap_or_else(|| "query".to_string()),
                    });
                },
                EntityType::Location => {
                    modifiers.push(Modifier {
                        modifier_type: ModifierType::Spatial,
                        value: entity.value.clone(),
                        applies_to: subject.clone().unwrap_or_else(|| "query".to_string()),
                    });
                },
                EntityType::Number => {
                    modifiers.push(Modifier {
                        modifier_type: ModifierType::Quantitative,
                        value: entity.value.clone(),
                        applies_to: subject.clone().unwrap_or_else(|| "query".to_string()),
                    });
                },
                _ => {},
            }
        }
        
        // Extract relationships from query patterns
        if query.to_lowercase().contains("that depend on") || query.to_lowercase().contains("dependent on") {
            if let (Some(ref subj), Some(ref obj)) = (&subject, &object) {
                relationships.push(Relationship {
                    source: subj.clone(),
                    relation_type: RelationType::DependsOn,
                    target: obj.clone(),
                });
            }
        }
        
        if query.to_lowercase().contains("in") || query.to_lowercase().contains("located in") {
            if let Some(location) = entities.entities.iter().find(|e| matches!(e.entity_type, EntityType::Location)) {
                if let Some(ref subj) = subject {
                    relationships.push(Relationship {
                        source: subj.clone(),
                        relation_type: RelationType::LocatedIn,
                        target: location.value.clone(),
                    });
                }
            }
        }
        
        SemanticParse {
            subject,
            predicate,
            object,
            modifiers,
            relationships,
        }
    }

    fn generate_execution_plan(&self, intent: &Intent, entities: &ExtractionResult, semantic_parse: &SemanticParse) -> ExecutionPlan {
        let mut steps = Vec::new();
        let mut estimated_time = 0.0;
        let mut requires_approval = false;
        
        match intent.primary {
            IntentType::List | IntentType::Show | IntentType::Find => {
                steps.push(ExecutionStep {
                    step_type: StepType::DataRetrieval,
                    description: "Retrieve resource data".to_string(),
                    api_endpoint: "/api/v1/resources".to_string(),
                    parameters: self.build_query_parameters(entities, semantic_parse),
                    depends_on: vec![],
                });
                estimated_time += 1.0;
                
                if intent.domain == "governance" {
                    steps.push(ExecutionStep {
                        step_type: StepType::Analysis,
                        description: "Analyze compliance status".to_string(),
                        api_endpoint: "/api/v1/compliance/analyze".to_string(),
                        parameters: HashMap::new(),
                        depends_on: vec![0],
                    });
                    estimated_time += 2.0;
                }
            },
            
            IntentType::Remediate | IntentType::Fix => {
                steps.push(ExecutionStep {
                    step_type: StepType::DataRetrieval,
                    description: "Identify issues to remediate".to_string(),
                    api_endpoint: "/api/v1/compliance/violations".to_string(),
                    parameters: self.build_query_parameters(entities, semantic_parse),
                    depends_on: vec![],
                });
                
                steps.push(ExecutionStep {
                    step_type: StepType::Validation,
                    description: "Validate remediation safety".to_string(),
                    api_endpoint: "/api/v1/remediation/validate".to_string(),
                    parameters: HashMap::new(),
                    depends_on: vec![0],
                });
                
                steps.push(ExecutionStep {
                    step_type: StepType::Remediation,
                    description: "Execute remediation".to_string(),
                    api_endpoint: "/api/v1/remediation/execute".to_string(),
                    parameters: HashMap::new(),
                    depends_on: vec![0, 1],
                });
                
                estimated_time += 5.0;
                requires_approval = true;
            },
            
            IntentType::CostAnalysis => {
                steps.push(ExecutionStep {
                    step_type: StepType::DataRetrieval,
                    description: "Retrieve cost data".to_string(),
                    api_endpoint: "/api/v1/finops/costs".to_string(),
                    parameters: self.build_query_parameters(entities, semantic_parse),
                    depends_on: vec![],
                });
                
                steps.push(ExecutionStep {
                    step_type: StepType::Analysis,
                    description: "Analyze cost patterns".to_string(),
                    api_endpoint: "/api/v1/finops/analyze".to_string(),
                    parameters: HashMap::new(),
                    depends_on: vec![0],
                });
                
                estimated_time += 3.0;
            },
            
            IntentType::Predict => {
                steps.push(ExecutionStep {
                    step_type: StepType::DataRetrieval,
                    description: "Retrieve historical data".to_string(),
                    api_endpoint: "/api/v1/metrics/historical".to_string(),
                    parameters: self.build_query_parameters(entities, semantic_parse),
                    depends_on: vec![],
                });
                
                steps.push(ExecutionStep {
                    step_type: StepType::Computation,
                    description: "Generate predictions".to_string(),
                    api_endpoint: "/api/v1/predictions/generate".to_string(),
                    parameters: HashMap::new(),
                    depends_on: vec![0],
                });
                
                estimated_time += 4.0;
            },
            
            _ => {
                steps.push(ExecutionStep {
                    step_type: StepType::DataRetrieval,
                    description: "General data retrieval".to_string(),
                    api_endpoint: "/api/v1/query".to_string(),
                    parameters: self.build_query_parameters(entities, semantic_parse),
                    depends_on: vec![],
                });
                estimated_time += 1.0;
            },
        }
        
        let complexity = match steps.len() {
            1 => Complexity::Simple,
            2..=3 => Complexity::Moderate,
            4..=6 => Complexity::Complex,
            _ => Complexity::Critical,
        };
        
        ExecutionPlan {
            steps,
            estimated_time,
            complexity,
            requires_approval,
        }
    }

    fn build_query_parameters(&self, entities: &ExtractionResult, semantic_parse: &SemanticParse) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();
        
        for entity in &entities.entities {
            match entity.entity_type {
                EntityType::ResourceType => {
                    params.insert("resource_type".to_string(), serde_json::Value::String(entity.value.clone()));
                },
                EntityType::Location => {
                    params.insert("location".to_string(), serde_json::Value::String(entity.value.clone()));
                },
                EntityType::TimeRange => {
                    params.insert("time_range".to_string(), serde_json::Value::String(entity.value.clone()));
                },
                _ => {},
            }
        }
        
        if let Some(ref subject) = semantic_parse.subject {
            params.insert("subject".to_string(), serde_json::Value::String(subject.clone()));
        }
        
        params
    }

    fn calculate_confidence(&self, intent: &Intent, entities: &ExtractionResult, _semantic_parse: &SemanticParse) -> f64 {
        let entity_confidence: f64 = entities.entities.iter().map(|e| e.confidence).sum::<f64>() / entities.entities.len().max(1) as f64;
        let intent_confidence = intent.confidence;
        
        (entity_confidence + intent_confidence) / 2.0
    }

    fn initialize_intent_patterns(&mut self) {
        self.intent_patterns.insert(IntentType::List, vec![
            "list".to_string(), "show all".to_string(), "display".to_string(), "enumerate".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Find, vec![
            "find".to_string(), "search".to_string(), "look for".to_string(), "locate".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Fix, vec![
            "fix".to_string(), "repair".to_string(), "resolve".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Remediate, vec![
            "remediate".to_string(), "correct".to_string(), "address".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Create, vec![
            "create".to_string(), "make".to_string(), "build".to_string(), "generate".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Analyze, vec![
            "analyze".to_string(), "examine".to_string(), "review".to_string(), "assess".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::Predict, vec![
            "predict".to_string(), "forecast".to_string(), "anticipate".to_string(), "estimate".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::CostAnalysis, vec![
            "cost".to_string(), "expense".to_string(), "spending".to_string(), "budget".to_string()
        ]);
        
        self.intent_patterns.insert(IntentType::CheckCompliance, vec![
            "compliance".to_string(), "compliant".to_string(), "violat".to_string(), "policy".to_string()
        ]);
    }

    fn initialize_domain_keywords(&mut self) {
        self.domain_keywords.insert("governance".to_string(), vec![
            "policy".to_string(), "compliance".to_string(), "violation".to_string(), "audit".to_string()
        ]);
        
        self.domain_keywords.insert("finops".to_string(), vec![
            "cost".to_string(), "budget".to_string(), "expense".to_string(), "savings".to_string()
        ]);
        
        self.domain_keywords.insert("security".to_string(), vec![
            "security".to_string(), "threat".to_string(), "vulnerability".to_string(), "access".to_string()
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_understanding() {
        let engine = QueryUnderstandingEngine::new();
        let understanding = engine.understand_query("Show me all virtual machines in East US that are not compliant with encryption policies");
        
        assert!(matches!(understanding.intent.primary, IntentType::List | IntentType::Show));
        assert!(!understanding.execution_plan.steps.is_empty());
        assert!(understanding.confidence > 0.5);
    }

    #[test]
    fn test_remediation_intent() {
        let engine = QueryUnderstandingEngine::new();
        let understanding = engine.understand_query("Fix all storage accounts that don't have encryption enabled");
        
        assert!(matches!(understanding.intent.primary, IntentType::Fix | IntentType::Remediate));
        assert!(understanding.execution_plan.requires_approval);
    }

    #[test]
    fn test_cost_analysis() {
        let engine = QueryUnderstandingEngine::new();
        let understanding = engine.understand_query("Analyze costs for resources in the last 30 days");
        
        assert!(matches!(understanding.intent.primary, IntentType::CostAnalysis | IntentType::Analyze));
        assert_eq!(understanding.intent.domain, "finops");
    }
}