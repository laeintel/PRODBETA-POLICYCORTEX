// Multi-Turn Conversation Memory System
// Maintains conversation state, context, and entity tracking across sessions

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Maximum history size per session
const MAX_HISTORY_SIZE: usize = 50;
/// Session timeout in hours
const SESSION_TIMEOUT_HOURS: i64 = 24;

/// Conversation memory manager
pub struct ConversationMemory {
    sessions: Arc<RwLock<HashMap<String, ConversationSession>>>,
    entity_store: Arc<RwLock<EntityStore>>,
    context_analyzer: ContextAnalyzer,
}

impl ConversationMemory {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            entity_store: Arc::new(RwLock::new(EntityStore::new())),
            context_analyzer: ContextAnalyzer::new(),
        }
    }
    
    /// Get or create a conversation session
    pub async fn get_or_create_session(&self, session_id: &str) -> ConversationSession {
        let mut sessions = self.sessions.write().await;
        
        // Clean up expired sessions
        sessions.retain(|_, session| {
            session.last_interaction > Utc::now() - Duration::hours(SESSION_TIMEOUT_HOURS)
        });
        
        sessions.entry(session_id.to_string())
            .or_insert_with(|| ConversationSession::new(session_id))
            .clone()
    }
    
    /// Update session with new exchange
    pub async fn update_session(
        &self,
        session_id: &str,
        user_input: &str,
        assistant_response: &str,
        entities: Vec<ExtractedEntity>,
        intent: Intent,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            // Add to conversation history
            session.add_exchange(ConversationExchange {
                timestamp: Utc::now(),
                user_input: user_input.to_string(),
                assistant_response: assistant_response.to_string(),
                entities: entities.clone(),
                intent: intent.clone(),
                confidence: intent.confidence,
            });
            
            // Update entity memory
            for entity in entities {
                session.update_entity(entity);
                self.entity_store.write().await.add_entity(session_id, entity).await;
            }
            
            // Update context
            session.context = self.context_analyzer.analyze(session).await;
            session.last_interaction = Utc::now();
            
            Ok(())
        } else {
            Err("Session not found".to_string())
        }
    }
    
    /// Get conversation context for a session
    pub async fn get_context(&self, session_id: &str) -> ConversationContext {
        let sessions = self.sessions.read().await;
        
        if let Some(session) = sessions.get(session_id) {
            session.context.clone()
        } else {
            ConversationContext::default()
        }
    }
    
    /// Get recent history for a session
    pub async fn get_history(&self, session_id: &str, limit: usize) -> Vec<ConversationExchange> {
        let sessions = self.sessions.read().await;
        
        if let Some(session) = sessions.get(session_id) {
            session.history.iter()
                .rev()
                .take(limit)
                .rev()
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Clear session memory
    pub async fn clear_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        
        self.entity_store.write().await.clear_session(session_id).await;
    }
    
    /// Get relevant entities for context
    pub async fn get_relevant_entities(&self, session_id: &str, query: &str) -> Vec<ExtractedEntity> {
        self.entity_store.read().await
            .get_relevant_entities(session_id, query)
            .await
    }
}

/// Conversation session
#[derive(Debug, Clone)]
pub struct ConversationSession {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_interaction: DateTime<Utc>,
    pub history: VecDeque<ConversationExchange>,
    pub entities: HashMap<String, ExtractedEntity>,
    pub context: ConversationContext,
    pub user_preferences: UserPreferences,
    pub topic_stack: Vec<ConversationTopic>,
}

impl ConversationSession {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            created_at: Utc::now(),
            last_interaction: Utc::now(),
            history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            entities: HashMap::new(),
            context: ConversationContext::default(),
            user_preferences: UserPreferences::default(),
            topic_stack: Vec::new(),
        }
    }
    
    pub fn add_exchange(&mut self, exchange: ConversationExchange) {
        // Maintain max history size
        if self.history.len() >= MAX_HISTORY_SIZE {
            self.history.pop_front();
        }
        
        // Update topic stack
        if exchange.intent.intent_type != IntentType::Unknown {
            self.update_topic_stack(&exchange.intent);
        }
        
        self.history.push_back(exchange);
    }
    
    pub fn update_entity(&mut self, entity: ExtractedEntity) {
        self.entities.insert(entity.id.clone(), entity);
    }
    
    fn update_topic_stack(&mut self, intent: &Intent) {
        let topic = ConversationTopic {
            topic_type: format!("{:?}", intent.intent_type),
            started_at: Utc::now(),
            entities: intent.entities.clone(),
        };
        
        // Keep only recent topics (last 5)
        if self.topic_stack.len() >= 5 {
            self.topic_stack.remove(0);
        }
        
        self.topic_stack.push(topic);
    }
    
    pub fn get_current_topic(&self) -> Option<&ConversationTopic> {
        self.topic_stack.last()
    }
}

/// Conversation exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExchange {
    pub timestamp: DateTime<Utc>,
    pub user_input: String,
    pub assistant_response: String,
    pub entities: Vec<ExtractedEntity>,
    pub intent: Intent,
    pub confidence: f64,
}

/// Conversation topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTopic {
    pub topic_type: String,
    pub started_at: DateTime<Utc>,
    pub entities: Vec<String>,
}

/// Conversation context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationContext {
    pub session_id: String,
    pub active_resources: Vec<String>,
    pub discussed_policies: Vec<String>,
    pub pending_actions: Vec<PendingAction>,
    pub user_goal: Option<String>,
    pub conversation_state: ConversationState,
    pub clarifications_needed: Vec<String>,
}

/// Conversation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationState {
    Greeting,
    GatheringInfo,
    ProvidingSolution,
    AwaitingConfirmation,
    ExecutingAction,
    Complete,
}

impl Default for ConversationState {
    fn default() -> Self {
        ConversationState::Greeting
    }
}

/// Pending action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAction {
    pub action_id: String,
    pub action_type: String,
    pub description: String,
    pub requires_approval: bool,
    pub created_at: DateTime<Utc>,
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub response_style: ResponseStyle,
    pub detail_level: DetailLevel,
    pub auto_execute: bool,
    pub notification_preferences: NotificationPreferences,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            response_style: ResponseStyle::Balanced,
            detail_level: DetailLevel::Medium,
            auto_execute: false,
            notification_preferences: NotificationPreferences::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStyle {
    Concise,
    Balanced,
    Detailed,
    Technical,
    Executive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub email: bool,
    pub teams: bool,
    pub slack: bool,
    pub in_app: bool,
}

/// Entity store for managing extracted entities
pub struct EntityStore {
    entities: HashMap<String, HashMap<String, ExtractedEntity>>,
    entity_index: HashMap<String, Vec<String>>, // entity_type -> entity_ids
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_index: HashMap::new(),
        }
    }
    
    pub async fn add_entity(&mut self, session_id: &str, entity: ExtractedEntity) {
        let session_entities = self.entities.entry(session_id.to_string()).or_insert_with(HashMap::new);
        session_entities.insert(entity.id.clone(), entity.clone());
        
        // Update index
        let entity_type = format!("{:?}", entity.entity_type);
        self.entity_index.entry(entity_type).or_insert_with(Vec::new).push(entity.id);
    }
    
    pub async fn get_relevant_entities(&self, session_id: &str, query: &str) -> Vec<ExtractedEntity> {
        if let Some(session_entities) = self.entities.get(session_id) {
            // Simple relevance: return entities mentioned in query
            session_entities.values()
                .filter(|e| query.to_lowercase().contains(&e.value.to_lowercase()))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    pub async fn clear_session(&mut self, session_id: &str) {
        self.entities.remove(session_id);
    }
}

/// Context analyzer with advanced cognitive understanding
pub struct ContextAnalyzer {
    semantic_patterns: HashMap<String, Vec<String>>,
    entity_relationships: HashMap<String, Vec<String>>,
    conversation_patterns: HashMap<ConversationState, Vec<String>>,
}

impl ContextAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self {
            semantic_patterns: HashMap::new(),
            entity_relationships: HashMap::new(),
            conversation_patterns: HashMap::new(),
        };
        analyzer.initialize_patterns();
        analyzer
    }
    
    fn initialize_patterns(&mut self) {
        // Semantic patterns for goal inference
        self.semantic_patterns.insert("troubleshooting".to_string(), vec![
            "fix".to_string(), "issue".to_string(), "problem".to_string(), "error".to_string()
        ]);
        
        self.semantic_patterns.insert("compliance_check".to_string(), vec![
            "compliance".to_string(), "violation".to_string(), "policy".to_string(), "audit".to_string()
        ]);
        
        self.semantic_patterns.insert("cost_optimization".to_string(), vec![
            "cost".to_string(), "expensive".to_string(), "optimize".to_string(), "budget".to_string()
        ]);
        
        // Entity relationships
        self.entity_relationships.insert("StorageAccount".to_string(), vec![
            "encryption".to_string(), "backup".to_string(), "access_tier".to_string()
        ]);
        
        self.entity_relationships.insert("VirtualMachine".to_string(), vec![
            "disk_encryption".to_string(), "backup".to_string(), "monitoring".to_string()
        ]);
        
        // Conversation state patterns
        self.conversation_patterns.insert(ConversationState::GatheringInfo, vec![
            "tell me".to_string(), "show me".to_string(), "what".to_string(), "how".to_string()
        ]);
        
        self.conversation_patterns.insert(ConversationState::AwaitingConfirmation, vec![
            "should i".to_string(), "can you".to_string(), "proceed".to_string(), "confirm".to_string()
        ]);
    }
    
    pub async fn analyze(&self, session: &ConversationSession) -> ConversationContext {
        let mut context = ConversationContext {
            session_id: session.id.clone(),
            active_resources: Vec::new(),
            discussed_policies: Vec::new(),
            pending_actions: Vec::new(),
            user_goal: None,
            conversation_state: ConversationState::GatheringInfo,
            clarifications_needed: Vec::new(),
        };
        
        // Enhanced context analysis
        self.analyze_entities(session, &mut context).await;
        self.analyze_conversation_flow(session, &mut context).await;
        self.analyze_user_intent(session, &mut context).await;
        self.identify_clarifications_needed(session, &mut context).await;
        
        context
    }
    
    async fn analyze_entities(&self, session: &ConversationSession, context: &mut ConversationContext) {
        let mut entity_frequency: HashMap<String, i32> = HashMap::new();
        
        // Analyze recent history with weighted importance (recent = more important)
        for (i, exchange) in session.history.iter().rev().take(15).enumerate() {
            let weight = 15 - i; // Recent exchanges have higher weight
            
            for entity in &exchange.entities {
                // Track entity frequency with weight
                *entity_frequency.entry(entity.value.clone()).or_insert(0) += weight as i32;
                
                match entity.entity_type {
                    EntityType::ResourceGroup | EntityType::StorageAccount | 
                    EntityType::VirtualMachine | EntityType::Database | EntityType::Network => {
                        if !context.active_resources.contains(&entity.value) {
                            context.active_resources.push(entity.value.clone());
                        }
                    },
                    EntityType::Policy => {
                        if !context.discussed_policies.contains(&entity.value) {
                            context.discussed_policies.push(entity.value.clone());
                        }
                    },
                    _ => {}
                }
            }
        }
        
        // Sort resources by importance (frequency * recency)
        context.active_resources.sort_by(|a, b| {
            let freq_a = entity_frequency.get(a).unwrap_or(&0);
            let freq_b = entity_frequency.get(b).unwrap_or(&0);
            freq_b.cmp(freq_a)
        });
        
        // Keep only top 10 most relevant resources
        context.active_resources.truncate(10);
    }
    
    async fn analyze_conversation_flow(&self, session: &ConversationSession, context: &mut ConversationContext) {
        if let Some(last_exchange) = session.history.back() {
            // Determine conversation state based on last interaction and patterns
            let user_input = &last_exchange.user_input.to_lowercase();
            
            if self.contains_patterns(user_input, &self.conversation_patterns.get(&ConversationState::AwaitingConfirmation).unwrap_or(&vec![])) {
                context.conversation_state = ConversationState::AwaitingConfirmation;
            } else if last_exchange.intent.intent_type == IntentType::ExecuteRemediation {
                context.conversation_state = ConversationState::ExecutingAction;
            } else if last_exchange.assistant_response.contains("recommendation") || 
                     last_exchange.assistant_response.contains("solution") {
                context.conversation_state = ConversationState::ProvidingSolution;
            } else {
                context.conversation_state = ConversationState::GatheringInfo;
            }
        }
        
        // Analyze pending actions from conversation flow
        for exchange in session.history.iter().rev().take(5) {
            if exchange.intent.intent_type == IntentType::ExecuteRemediation && 
               exchange.assistant_response.contains("pending") {
                context.pending_actions.push(PendingAction {
                    action_id: Uuid::new_v4().to_string(),
                    action_type: "remediation".to_string(),
                    description: format!("Remediation from: {}", &exchange.user_input[..50.min(exchange.user_input.len())]),
                    requires_approval: true,
                    created_at: exchange.timestamp,
                });
            }
        }
    }
    
    async fn analyze_user_intent(&self, session: &ConversationSession, context: &mut ConversationContext) {
        let mut goal_scores: HashMap<String, f64> = HashMap::new();
        
        // Analyze semantic patterns across conversation
        for exchange in &session.history {
            let combined_text = format!("{} {}", exchange.user_input, exchange.assistant_response).to_lowercase();
            
            for (goal, patterns) in &self.semantic_patterns {
                let mut score = 0.0;
                for pattern in patterns {
                    if combined_text.contains(pattern) {
                        score += 1.0;
                    }
                }
                if score > 0.0 {
                    *goal_scores.entry(goal.clone()).or_insert(0.0) += score;
                }
            }
        }
        
        // Extract dominant goal
        if let Some((goal, _)) = goal_scores.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
            context.user_goal = Some(format!("Primary goal: {}", goal.replace("_", " ")));
        }
        
        // Enhance with current topic
        if let Some(current_topic) = session.get_current_topic() {
            if let Some(ref existing_goal) = context.user_goal {
                context.user_goal = Some(format!("{}, Current focus: {}", existing_goal, current_topic.topic_type));
            } else {
                context.user_goal = Some(format!("Current focus: {}", current_topic.topic_type));
            }
        }
    }
    
    async fn identify_clarifications_needed(&self, session: &ConversationSession, context: &mut ConversationContext) {
        if let Some(last_exchange) = session.history.back() {
            let user_input = &last_exchange.user_input.to_lowercase();
            
            // Check for ambiguous requests
            if user_input.contains("all") && !user_input.contains("specific") {
                context.clarifications_needed.push("Scope clarification: Which specific resources or categories?".to_string());
            }
            
            if user_input.contains("fix") && !context.active_resources.is_empty() && context.active_resources.len() > 5 {
                context.clarifications_needed.push("Target clarification: Which resources should be prioritized?".to_string());
            }
            
            if user_input.contains("cost") && !user_input.contains("time") && !user_input.contains("period") {
                context.clarifications_needed.push("Time range clarification: What time period should be analyzed?".to_string());
            }
            
            // Check for incomplete entity information
            let has_action = last_exchange.entities.iter().any(|e| e.entity_type == EntityType::Action);
            let has_resource = last_exchange.entities.iter().any(|e| matches!(e.entity_type, 
                EntityType::ResourceGroup | EntityType::StorageAccount | EntityType::VirtualMachine));
            
            if has_action && !has_resource && context.active_resources.is_empty() {
                context.clarifications_needed.push("Resource specification: Which resources should this action apply to?".to_string());
            }
        }
        
        // Check conversation coherence
        if session.history.len() > 3 {
            let recent_intents: Vec<&IntentType> = session.history.iter()
                .rev()
                .take(3)
                .map(|e| &e.intent.intent_type)
                .collect();
            
            // If intents are very different, might need clarification
            let intent_consistency = self.calculate_intent_consistency(&recent_intents);
            if intent_consistency < 0.5 {
                context.clarifications_needed.push("Goal clarification: It seems we've covered multiple topics. What's the main objective?".to_string());
            }
        }
    }
    
    fn contains_patterns(&self, text: &str, patterns: &[String]) -> bool {
        patterns.iter().any(|pattern| text.contains(pattern))
    }
    
    fn calculate_intent_consistency(&self, intents: &[&IntentType]) -> f64 {
        if intents.len() < 2 {
            return 1.0;
        }
        
        let mut consistency_score = 0.0;
        let mut comparisons = 0;
        
        for i in 0..intents.len() {
            for j in i+1..intents.len() {
                comparisons += 1;
                if self.intents_are_related(intents[i], intents[j]) {
                    consistency_score += 1.0;
                }
            }
        }
        
        if comparisons > 0 {
            consistency_score / comparisons as f64
        } else {
            1.0
        }
    }
    
    fn intents_are_related(&self, intent1: &IntentType, intent2: &IntentType) -> bool {
        match (intent1, intent2) {
            (IntentType::QueryViolations, IntentType::ExecuteRemediation) => true,
            (IntentType::PredictCompliance, IntentType::QueryViolations) => true,
            (IntentType::AnalyzeCost, IntentType::GetRecommendations) => true,
            (IntentType::ExplainPolicy, IntentType::CreatePolicy) => true,
            (IntentType::CheckSecurity, IntentType::QueryViolations) => true,
            _ => intent1 == intent2,
        }
    }
}

/// Extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub value: String,
    pub confidence: f64,
    pub context: Option<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    ResourceGroup,
    StorageAccount,
    VirtualMachine,
    Database,
    Network,
    Policy,
    Subscription,
    TimeRange,
    Metric,
    Action,
    User,
    Location,
    Tag,
}

/// Intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub intent_type: IntentType,
    pub confidence: f64,
    pub entities: Vec<String>,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IntentType {
    QueryViolations,
    PredictCompliance,
    ExecuteRemediation,
    GenerateReport,
    ExplainPolicy,
    CreatePolicy,
    AnalyzeCost,
    CheckSecurity,
    GetRecommendations,
    Greeting,
    Farewell,
    Unknown,
}

/// Conversation metrics for analysis
#[derive(Debug, Clone, Serialize)]
pub struct ConversationMetrics {
    pub total_exchanges: usize,
    pub avg_confidence: f64,
    pub entities_extracted: usize,
    pub successful_actions: usize,
    pub clarifications_needed: usize,
    pub session_duration: i64,
}

impl ConversationMemory {
    /// Get conversation metrics
    pub async fn get_metrics(&self, session_id: &str) -> ConversationMetrics {
        let sessions = self.sessions.read().await;
        
        if let Some(session) = sessions.get(session_id) {
            let total_confidence: f64 = session.history.iter()
                .map(|e| e.confidence)
                .sum();
            
            let total_entities: usize = session.history.iter()
                .map(|e| e.entities.len())
                .sum();
            
            ConversationMetrics {
                total_exchanges: session.history.len(),
                avg_confidence: if session.history.is_empty() { 0.0 } else { total_confidence / session.history.len() as f64 },
                entities_extracted: total_entities,
                successful_actions: 0, // Would track completed actions
                clarifications_needed: 0, // Would track clarification requests
                session_duration: (session.last_interaction - session.created_at).num_seconds(),
            }
        } else {
            ConversationMetrics {
                total_exchanges: 0,
                avg_confidence: 0.0,
                entities_extracted: 0,
                successful_actions: 0,
                clarifications_needed: 0,
                session_duration: 0,
            }
        }
    }
}