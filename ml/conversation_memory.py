"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/conversation_memory.py
# Multi-turn Conversation Memory System for PolicyCortex

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import uuid
import json
import pickle
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """State of the conversation"""
    GREETING = "greeting"
    INQUIRY = "inquiry"
    TROUBLESHOOTING = "troubleshooting"
    REMEDIATION = "remediation"
    REPORTING = "reporting"
    COMPLETED = "completed"

@dataclass
class Entity:
    """Extracted entity from conversation"""
    type: str  # 'resource', 'policy', 'user', 'time', 'metric'
    value: str
    confidence: float
    context: str
    mentions: List[int] = field(default_factory=list)  # Turn numbers where mentioned

@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    turn_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    intent: str
    entities: List[Entity]
    confidence: float
    context_used: Dict[str, Any]
    state: ConversationState

@dataclass
class ConversationContext:
    """Full conversation context"""
    session_id: str
    user_id: str
    started_at: datetime
    turns: List[ConversationTurn]
    entities: Dict[str, Entity]  # Aggregated entities
    facts: Dict[str, Any]  # Important facts learned
    goals: List[str]  # User goals identified
    current_state: ConversationState
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationMemory:
    """Advanced conversation memory management"""
    
    def __init__(self, max_turns: int = 10, max_sessions: int = 100):
        self.short_term = deque(maxlen=max_turns)
        self.long_term = {}  # Persistent context
        self.sessions = {}  # Active sessions
        self.max_sessions = max_sessions
        self.entity_resolver = EntityResolver()
        self.fact_extractor = FactExtractor()
        
    def create_session(self, user_id: str) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            turns=[],
            entities={},
            facts={},
            goals=[],
            current_state=ConversationState.GREETING
        )
        
        self.sessions[session_id] = context
        
        # Clean up old sessions if needed
        if len(self.sessions) > self.max_sessions:
            self._cleanup_old_sessions()
        
        return session_id
    
    def update_context(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        intent: str,
        confidence: float = 1.0
    ) -> ConversationContext:
        """Update conversation context with new turn"""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.sessions[session_id]
        
        # Extract entities from this turn
        entities = self.entity_resolver.extract_entities(user_input)
        
        # Create turn record
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities=entities,
            confidence=confidence,
            context_used=self._get_relevant_context(context),
            state=self._determine_state(intent, context)
        )
        
        # Add to conversation
        context.turns.append(turn)
        self.short_term.append(turn)
        
        # Update aggregated entities
        for entity in entities:
            key = f"{entity.type}:{entity.value}"
            if key not in context.entities or entity.confidence > context.entities[key].confidence:
                context.entities[key] = entity
            context.entities[key].mentions.append(len(context.turns) - 1)
        
        # Extract and store important facts
        facts = self.fact_extractor.extract_facts(assistant_response, entities)
        context.facts.update(facts)
        
        # Update goals if new ones identified
        if intent in ['query', 'request', 'investigate']:
            goal = self._extract_goal(user_input, intent)
            if goal and goal not in context.goals:
                context.goals.append(goal)
        
        # Update conversation state
        context.current_state = turn.state
        
        # Update long-term memory with important information
        if self._is_important(turn):
            self._update_long_term_memory(context.user_id, turn)
        
        return context
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get current conversation context"""
        return self.sessions.get(session_id)
    
    def get_relevant_history(
        self,
        session_id: str,
        query: str,
        max_turns: int = 5
    ) -> List[ConversationTurn]:
        """Get relevant conversation history for a query"""
        
        if session_id not in self.sessions:
            return []
        
        context = self.sessions[session_id]
        
        # Score each turn for relevance
        scored_turns = []
        query_entities = self.entity_resolver.extract_entities(query)
        
        for turn in context.turns[-20:]:  # Look at last 20 turns
            score = self._calculate_relevance_score(turn, query, query_entities)
            scored_turns.append((score, turn))
        
        # Sort by relevance and return top turns
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        
        return [turn for score, turn in scored_turns[:max_turns] if score > 0.3]
    
    def _calculate_relevance_score(
        self,
        turn: ConversationTurn,
        query: str,
        query_entities: List[Entity]
    ) -> float:
        """Calculate relevance score between turn and query"""
        score = 0.0
        
        # Entity overlap
        turn_entities = {f"{e.type}:{e.value}" for e in turn.entities}
        query_entity_set = {f"{e.type}:{e.value}" for e in query_entities}
        
        if turn_entities and query_entity_set:
            overlap = len(turn_entities & query_entity_set)
            score += overlap / max(len(turn_entities), len(query_entity_set)) * 0.4
        
        # Intent similarity
        if turn.intent == self._extract_intent(query):
            score += 0.3
        
        # Keyword overlap
        turn_keywords = set(turn.user_input.lower().split())
        query_keywords = set(query.lower().split())
        
        if turn_keywords and query_keywords:
            keyword_overlap = len(turn_keywords & query_keywords)
            score += keyword_overlap / max(len(turn_keywords), len(query_keywords)) * 0.3
        
        return score
    
    def _determine_state(
        self,
        intent: str,
        context: ConversationContext
    ) -> ConversationState:
        """Determine conversation state based on intent and context"""
        
        # State transition logic
        if intent in ['greeting', 'hello']:
            return ConversationState.GREETING
        elif intent in ['query', 'question', 'investigate']:
            return ConversationState.INQUIRY
        elif intent in ['troubleshoot', 'debug', 'analyze']:
            return ConversationState.TROUBLESHOOTING
        elif intent in ['fix', 'remediate', 'resolve']:
            return ConversationState.REMEDIATION
        elif intent in ['report', 'summary', 'export']:
            return ConversationState.REPORTING
        elif intent in ['goodbye', 'done', 'complete']:
            return ConversationState.COMPLETED
        else:
            # Continue with current state
            return context.current_state
    
    def _get_relevant_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Get relevant context for current turn"""
        relevant = {}
        
        # Include recent entities
        recent_entities = {}
        for turn in context.turns[-3:]:  # Last 3 turns
            for entity in turn.entities:
                key = f"{entity.type}:{entity.value}"
                recent_entities[key] = entity.value
        
        relevant['recent_entities'] = recent_entities
        
        # Include active goals
        relevant['goals'] = context.goals[-2:] if context.goals else []
        
        # Include important facts
        relevant['facts'] = dict(list(context.facts.items())[-5:])
        
        # Include current state
        relevant['state'] = context.current_state.value
        
        return relevant
    
    def _is_important(self, turn: ConversationTurn) -> bool:
        """Determine if a turn contains important information"""
        
        # High confidence responses are important
        if turn.confidence > 0.9:
            return True
        
        # Turns with many entities are important
        if len(turn.entities) > 3:
            return True
        
        # Remediation and troubleshooting turns are important
        if turn.state in [ConversationState.REMEDIATION, ConversationState.TROUBLESHOOTING]:
            return True
        
        # Turns that change state are important
        if len(self.short_term) > 1 and turn.state != self.short_term[-2].state:
            return True
        
        return False
    
    def _update_long_term_memory(self, user_id: str, turn: ConversationTurn):
        """Update long-term memory with important information"""
        
        if user_id not in self.long_term:
            self.long_term[user_id] = {
                'preferences': {},
                'common_queries': [],
                'resolved_issues': [],
                'entities': {}
            }
        
        memory = self.long_term[user_id]
        
        # Update entity memory
        for entity in turn.entities:
            key = f"{entity.type}:{entity.value}"
            if key not in memory['entities']:
                memory['entities'][key] = {
                    'count': 0,
                    'last_seen': None,
                    'contexts': []
                }
            
            memory['entities'][key]['count'] += 1
            memory['entities'][key]['last_seen'] = turn.timestamp
            memory['entities'][key]['contexts'].append(turn.intent)
        
        # Track resolved issues
        if turn.state == ConversationState.REMEDIATION:
            memory['resolved_issues'].append({
                'timestamp': turn.timestamp,
                'issue': turn.user_input,
                'solution': turn.assistant_response
            })
        
        # Keep only recent data
        memory['resolved_issues'] = memory['resolved_issues'][-50:]
    
    def _extract_goal(self, user_input: str, intent: str) -> Optional[str]:
        """Extract user goal from input"""
        
        # Simple goal extraction based on keywords
        goal_keywords = {
            'find': 'locate',
            'fix': 'remediate',
            'check': 'verify',
            'analyze': 'investigate',
            'optimize': 'improve',
            'reduce': 'minimize',
            'increase': 'maximize'
        }
        
        input_lower = user_input.lower()
        for keyword, goal_type in goal_keywords.items():
            if keyword in input_lower:
                # Extract object of the goal
                words = input_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    if idx < len(words) - 1:
                        return f"{goal_type} {' '.join(words[idx+1:idx+3])}"
        
        return None
    
    def _extract_intent(self, text: str) -> str:
        """Extract intent from text"""
        # Simplified intent extraction
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return 'greeting'
        elif any(word in text_lower for word in ['fix', 'remediate', 'resolve']):
            return 'remediate'
        elif any(word in text_lower for word in ['why', 'what', 'how', 'when']):
            return 'query'
        elif any(word in text_lower for word in ['troubleshoot', 'debug', 'investigate']):
            return 'troubleshoot'
        else:
            return 'general'
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions to manage memory"""
        # Remove sessions older than 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        
        to_remove = []
        for session_id, context in self.sessions.items():
            if context.started_at < cutoff:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
    
    def summarize_conversation(self, session_id: str) -> Dict[str, Any]:
        """Generate conversation summary"""
        
        if session_id not in self.sessions:
            return {}
        
        context = self.sessions[session_id]
        
        summary = {
            'session_id': session_id,
            'duration': (datetime.now() - context.started_at).total_seconds(),
            'turn_count': len(context.turns),
            'entities_discussed': list(context.entities.keys()),
            'goals_identified': context.goals,
            'facts_learned': context.facts,
            'final_state': context.current_state.value,
            'topics': self._extract_topics(context)
        }
        
        return summary
    
    def _extract_topics(self, context: ConversationContext) -> List[str]:
        """Extract main topics from conversation"""
        topics = set()
        
        # Extract from intents
        for turn in context.turns:
            if turn.intent not in ['greeting', 'general']:
                topics.add(turn.intent)
        
        # Extract from entity types
        for entity_key in context.entities:
            entity_type = entity_key.split(':')[0]
            topics.add(entity_type)
        
        return list(topics)

class EntityResolver:
    """Resolve and extract entities from text"""
    
    def __init__(self):
        self.entity_patterns = {
            'resource': ['vm', 'storage', 'database', 'network', 'resource'],
            'policy': ['policy', 'rule', 'compliance', 'standard'],
            'metric': ['cpu', 'memory', 'cost', 'performance', 'utilization'],
            'time': ['yesterday', 'today', 'hour', 'day', 'week', 'month'],
            'action': ['create', 'delete', 'update', 'modify', 'configure']
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Extract context around the pattern
                    start = text_lower.find(pattern)
                    context_start = max(0, start - 20)
                    context_end = min(len(text), start + len(pattern) + 20)
                    context = text[context_start:context_end]
                    
                    entities.append(Entity(
                        type=entity_type,
                        value=pattern,
                        confidence=0.8,
                        context=context,
                        mentions=[]
                    ))
        
        return entities

class FactExtractor:
    """Extract facts from conversation"""
    
    def extract_facts(self, text: str, entities: List[Entity]) -> Dict[str, Any]:
        """Extract facts from assistant response"""
        facts = {}
        
        # Extract numeric facts
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers and entities:
            for entity in entities:
                if entity.type == 'metric' and numbers:
                    facts[f"{entity.value}_value"] = float(numbers[0])
        
        # Extract status facts
        if 'enabled' in text.lower():
            facts['status'] = 'enabled'
        elif 'disabled' in text.lower():
            facts['status'] = 'disabled'
        
        # Extract compliance facts
        if 'compliant' in text.lower():
            facts['compliance'] = True
        elif 'violation' in text.lower() or 'non-compliant' in text.lower():
            facts['compliance'] = False
        
        return facts

class ConversationAnalyzer:
    """Analyze conversation patterns and user behavior"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze patterns in user conversations"""
        
        if user_id not in self.memory.long_term:
            return {}
        
        user_memory = self.memory.long_term[user_id]
        
        analysis = {
            'frequent_entities': self._get_frequent_entities(user_memory),
            'common_issues': self._get_common_issues(user_memory),
            'interaction_style': self._determine_interaction_style(user_memory),
            'expertise_level': self._assess_expertise_level(user_memory)
        }
        
        return analysis
    
    def _get_frequent_entities(self, user_memory: Dict) -> List[Tuple[str, int]]:
        """Get most frequently discussed entities"""
        entities = user_memory.get('entities', {})
        
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [(k, v['count']) for k, v in sorted_entities[:10]]
    
    def _get_common_issues(self, user_memory: Dict) -> List[str]:
        """Get common issues user faces"""
        resolved_issues = user_memory.get('resolved_issues', [])
        
        # Extract issue types
        issue_types = []
        for issue in resolved_issues[-20:]:  # Last 20 issues
            if 'compliance' in issue['issue'].lower():
                issue_types.append('compliance')
            elif 'cost' in issue['issue'].lower():
                issue_types.append('cost')
            elif 'performance' in issue['issue'].lower():
                issue_types.append('performance')
            elif 'security' in issue['issue'].lower():
                issue_types.append('security')
        
        # Count occurrences
        from collections import Counter
        return [issue for issue, count in Counter(issue_types).most_common(5)]
    
    def _determine_interaction_style(self, user_memory: Dict) -> str:
        """Determine user's interaction style"""
        # Simplified - in production would use more sophisticated analysis
        resolved_count = len(user_memory.get('resolved_issues', []))
        
        if resolved_count > 20:
            return 'technical'
        elif resolved_count > 10:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _assess_expertise_level(self, user_memory: Dict) -> str:
        """Assess user's expertise level"""
        entities = user_memory.get('entities', {})
        
        # Count technical entities
        technical_count = sum(
            1 for k in entities.keys()
            if any(term in k for term in ['policy', 'compliance', 'metric'])
        )
        
        if technical_count > 50:
            return 'expert'
        elif technical_count > 20:
            return 'advanced'
        elif technical_count > 10:
            return 'intermediate'
        else:
            return 'beginner'

# Export main components
__all__ = [
    'ConversationMemory',
    'ConversationContext',
    'ConversationTurn',
    'Entity',
    'ConversationState',
    'EntityResolver',
    'FactExtractor',
    'ConversationAnalyzer'
]