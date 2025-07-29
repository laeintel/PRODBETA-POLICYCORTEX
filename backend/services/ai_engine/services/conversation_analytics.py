"""
Advanced Conversation Analytics
Part of Patent 3: Conversational Governance Intelligence
Provides deep analytics, learning capabilities, and conversation optimization
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

from backend.core.config import settings
from backend.core.redis_client import redis_client
from backend.core.exceptions import APIError

logger = logging.getLogger(__name__)


class AnalyticsType(str, Enum):
    """Types of conversation analytics"""
    CONVERSATION_FLOW = "conversation_flow"
    TOPIC_MODELING = "topic_modeling"
    USER_BEHAVIOR = "user_behavior"
    SENTIMENT_TRENDS = "sentiment_trends"
    KNOWLEDGE_GAPS = "knowledge_gaps"
    EFFICIENCY_METRICS = "efficiency_metrics"
    PREDICTIVE_ANALYTICS = "predictive_analytics"


class MetricType(str, Enum):
    """Types of metrics to track"""
    RESOLUTION_RATE = "resolution_rate"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    KNOWLEDGE_COVERAGE = "knowledge_coverage"
    CONTEXT_ACCURACY = "context_accuracy"
    ESCALATION_RATE = "escalation_rate"


@dataclass
class ConversationMetrics:
    """Metrics for conversation analysis"""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsInsight:
    """Represents an analytics insight"""
    insight_id: str
    title: str
    description: str
    category: str
    priority: str  # high, medium, low
    confidence: float
    actionable: bool
    recommendations: List[str]
    data_source: str
    generated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationFlowAnalyzer:
    """Analyzes conversation flows and patterns"""

    def __init__(self):
        self.flow_graphs: Dict[str, nx.DiGraph] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize flow analyzer"""
        self._initialized = True
        logger.info("Conversation flow analyzer initialized")

    async def analyze_conversation_flows(self,
                                       conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation flow patterns"""

        if not self._initialized:
            await self.initialize()

        try:
            # Build conversation flow graph
            flow_graph = nx.DiGraph()
            context_transitions = defaultdict(int)
            intent_transitions = defaultdict(int)

            for conversation in conversations:
                turns = conversation.get('turns', [])

                for i in range(len(turns) - 1):
                    current_turn = turns[i]
                    next_turn = turns[i + 1]

                    # Track context transitions
                    current_context = current_turn.get('context', 'unknown')
                    next_context = next_turn.get('context', 'unknown')
                    context_transitions[(current_context, next_context)] += 1

                    # Track intent transitions
                    current_intent = current_turn.get('intent', 'unknown')
                    next_intent = next_turn.get('intent', 'unknown')
                    intent_transitions[(current_intent, next_intent)] += 1

                    # Add to graph
                    flow_graph.add_edge(
                        f"{current_context}_{current_intent}",
                        f"{next_context}_{next_intent}",
                        weight=1
                    )

            # Analyze flow patterns
            most_common_flows = Counter(context_transitions).most_common(10)
            dead_ends = self._find_dead_ends(flow_graph)
            circular_flows = self._find_circular_flows(flow_graph)

            # Calculate flow efficiency
            efficiency_metrics = self._calculate_flow_efficiency(conversations)

            return {
                'total_flows': len(context_transitions),
                'most_common_flows': most_common_flows,
                'dead_ends': dead_ends,
                'circular_flows': circular_flows,
                'efficiency_metrics': efficiency_metrics,
                'flow_graph_stats': {
                    'nodes': flow_graph.number_of_nodes(),
                    'edges': flow_graph.number_of_edges(),
                    'density': nx.density(flow_graph)
                }
            }

        except Exception as e:
            logger.error(f"Flow analysis failed: {str(e)}")
            return {}

    def _find_dead_ends(self, graph: nx.DiGraph) -> List[str]:
        """Find nodes with no outgoing edges (dead ends)"""
        return [node for node in graph.nodes() if graph.out_degree(node) == 0]

    def _find_circular_flows(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find circular conversation flows"""
        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles[:10]  # Return top 10 cycles
        except:
            return []

    def _calculate_flow_efficiency(self, conversations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate conversation flow efficiency metrics"""

        total_turns = 0
        resolved_conversations = 0
        escalated_conversations = 0

        for conversation in conversations:
            turns = conversation.get('turns', [])
            total_turns += len(turns)

            # Check if conversation was resolved
            if conversation.get('outcome') == 'resolved':
                resolved_conversations += 1
            elif conversation.get('outcome') == 'escalated':
                escalated_conversations += 1

        total_conversations = len(conversations)

        return {
            'average_turns_per_conversation': total_turns / total_conversations if total_conversations > 0 else 0,
            'resolution_rate': resolved_conversations / total_conversations if total_conversations > 0 else 0,
            'escalation_rate': escalated_conversations / total_conversations if total_conversations > 0 else 0,
            'efficiency_score': (resolved_conversations * 2 - escalated_conversations) / total_conversations if total_conversations > 0 else 0
        }


class TopicModelingEngine:
    """Advanced topic modeling for conversation analysis"""

    def __init__(self):
        self.topic_models: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize topic modeling engine"""
        self._initialized = True
        logger.info("Topic modeling engine initialized")

    async def analyze_conversation_topics(self,
                                        conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topics in conversations"""

        if not self._initialized:
            await self.initialize()

        try:
            # Extract text from conversations
            conversation_texts = []
            for conversation in conversations:
                turns = conversation.get('turns', [])
                conversation_text = " ".join([turn.get('input_text', '') for turn in turns])
                conversation_texts.append(conversation_text)

            if not conversation_texts:
                return {'topics': [], 'topic_distribution': {}}

            # Simple topic clustering using keywords
            topics = await self._extract_topics_by_keywords(conversation_texts)

            # Topic distribution analysis
            topic_distribution = await self._analyze_topic_distribution(conversations, topics)

            # Trending topics
            trending_topics = await self._identify_trending_topics(conversations)

            # Topic evolution over time
            topic_evolution = await self._analyze_topic_evolution(conversations)

            return {
                'topics': topics,
                'topic_distribution': topic_distribution,
                'trending_topics': trending_topics,
                'topic_evolution': topic_evolution
            }

        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
            return {}

    async def _extract_topics_by_keywords(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract topics using keyword-based approach"""

        # Define governance topic keywords
        topic_keywords = {
            'Policy Management': ['policy', 'policies', 'procedure', 'guideline', 'standard', 'rule'],
            'Compliance': ['compliance', 'audit', 'regulation', 'violation', 'requirement', 'mandate'],
            'Security': ['security', 'threat', 'vulnerability', 'incident', 'breach', 'risk'],
            'Cost Management': ['cost', 'budget', 'expense', 'optimization', 'savings', 'financial'],
            'Resource Management': ['resource', 'allocation', 'scaling', 'capacity', 'utilization'],
            'Access Control': ['access', 'permission', 'authentication', 'authorization', 'identity'],
            'Data Governance': ['data', 'privacy', 'classification', 'retention', 'backup'],
            'Risk Management': ['risk', 'assessment', 'mitigation', 'impact', 'likelihood']
        }

        topics = []

        for topic_name, keywords in topic_keywords.items():
            keyword_scores = []

            for text in texts:
                text_lower = text.lower()
                score = sum(1 for keyword in keywords if keyword in text_lower)
                keyword_scores.append(score)

            if keyword_scores:
                avg_score = np.mean(keyword_scores)
                total_mentions = sum(keyword_scores)

                topics.append({
                    'topic_name': topic_name,
                    'keywords': keywords,
                    'average_score': avg_score,
                    'total_mentions': total_mentions,
                    'prevalence': total_mentions / len(texts) if texts else 0
                })

        # Sort by prevalence
        topics.sort(key=lambda x: x['prevalence'], reverse=True)

        return topics

    async def _analyze_topic_distribution(self,
                                        conversations: List[Dict[str, Any]],
                                        topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of topics across conversations"""

        topic_counts = defaultdict(int)
        context_topic_mapping = defaultdict(lambda: defaultdict(int))

        for conversation in conversations:
            turns = conversations.get('turns', [])
            conversation_text = " ".join([turn.get('input_text', '') for turn in turns])
            text_lower = conversation_text.lower()

            # Find matching topics
            for topic in topics:
                topic_name = topic['topic_name']
                keywords = topic['keywords']

                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    topic_counts[topic_name] += 1

                    # Map to conversation contexts
                    for turn in turns:
                        context = turn.get('context', 'unknown')
                        context_topic_mapping[context][topic_name] += 1

        return {
            'topic_counts': dict(topic_counts),
            'context_topic_mapping': {k: dict(v) for k, v in context_topic_mapping.items()}
        }

    async def _identify_trending_topics(
        self,
        conversations: List[Dict[str,
        Any]]
    ) -> List[Dict[str, Any]]:
        """Identify trending topics over time"""

        # Group conversations by time periods
        time_buckets = defaultdict(list)

        for conversation in conversations:
            # Use conversation start time
            start_time = conversation.get('started_at')
            if start_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

                # Group by day
                date_key = start_time.date().isoformat()
                time_buckets[date_key].append(conversation)

        # Analyze topic trends
        trending_topics = []

        # This is simplified - in production, use more sophisticated trend analysis
        recent_dates = sorted(time_buckets.keys())[-7:]  # Last 7 days

        for topic_name in ['Policy Management', 'Compliance', 'Security', 'Cost Management']:
            trend_data = []

            for date_key in recent_dates:
                daily_conversations = time_buckets.get(date_key, [])
                topic_mentions = 0

                for conversation in daily_conversations:
                    turns = conversation.get('turns', [])
                    conversation_text = " ".join([turn.get('input_text', '') for turn in turns])

                    if topic_name.lower() in conversation_text.lower():
                        topic_mentions += 1

                trend_data.append(topic_mentions)

            if len(trend_data) > 1:
                # Calculate trend direction
                trend_slope = np.polyfit(range(len(trend_data)), trend_data, 1)[0]

                trending_topics.append({
                    'topic': topic_name,
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'trend_strength': abs(trend_slope),
                    'recent_mentions': sum(trend_data),
                    'daily_data': trend_data
                })

        return trending_topics

    async def _analyze_topic_evolution(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how topics evolve over time"""

        # This is a simplified implementation
        # In production, use more sophisticated temporal analysis

        return {
            'evolution_summary': 'Topic evolution analysis requires more data points',
            'time_periods_analyzed': 0,
            'topic_lifecycle_stages': []
        }


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns in conversations"""

    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize behavior analyzer"""
        self._initialized = True
        logger.info("User behavior analyzer initialized")

    async def analyze_user_behavior(self,
                                  conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""

        if not self._initialized:
            await self.initialize()

        try:
            # Group conversations by user
            user_conversations = defaultdict(list)
            for conversation in conversations:
                user_id = conversation.get('user_id')
                if user_id:
                    user_conversations[user_id].append(conversation)

            # Analyze each user
            user_behaviors = {}

            for user_id, user_convs in user_conversations.items():
                behavior_profile = await self._analyze_individual_user(user_id, user_convs)
                user_behaviors[user_id] = behavior_profile

            # Aggregate behavior patterns
            behavior_patterns = await self._identify_behavior_patterns(user_behaviors)

            # User segmentation
            user_segments = await self._segment_users(user_behaviors)

            return {
                'total_users': len(user_behaviors),
                'behavior_patterns': behavior_patterns,
                'user_segments': user_segments,
                'individual_profiles': user_behaviors
            }

        except Exception as e:
            logger.error(f"User behavior analysis failed: {str(e)}")
            return {}

    async def _analyze_individual_user(self,
                                     user_id: str,
                                     conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual user behavior"""

        total_conversations = len(conversations)
        total_turns = sum(len(conv.get('turns', [])) for conv in conversations)

        # Conversation contexts
        contexts = []
        for conv in conversations:
            turns = conv.get('turns', [])
            contexts.extend([turn.get('context') for turn in turns])

        context_distribution = Counter(contexts)

        # Sentiment analysis
        sentiments = []
        for conv in conversations:
            turns = conv.get('turns', [])
            sentiments.extend([turn.get('sentiment', 0) for turn in turns])

        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        # Session patterns
        session_durations = []
        for conv in conversations:
            start_time = conv.get('started_at')
            end_time = conv.get('updated_at', start_time)

            if start_time and end_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

                duration = (end_time - start_time).total_seconds() / 60  # minutes
                session_durations.append(duration)

        avg_session_duration = np.mean(session_durations) if session_durations else 0.0

        # Engagement level
        engagement_score = self._calculate_engagement_score(
            total_conversations, total_turns, avg_session_duration, avg_sentiment
        )

        return {
            'user_id': user_id,
            'total_conversations': total_conversations,
            'total_turns': total_turns,
            'average_turns_per_conversation': total_turns / total_conversations if total_conversations > 0 else 0,
            'context_distribution': dict(context_distribution),
            'preferred_context': context_distribution.most_common(1)[0][0] if context_distribution else None,
            'average_sentiment': avg_sentiment,
            'average_session_duration': avg_session_duration,
            'engagement_score': engagement_score,
            'user_type': self._classify_user_type(engagement_score, context_distribution)
        }

    def _calculate_engagement_score(self,
                                  conversations: int,
                                  turns: int,
                                  avg_duration: float,
                                  avg_sentiment: float) -> float:
        """Calculate user engagement score"""

        # Normalize components (0-1 scale)
        conv_score = min(1.0, conversations / 10)  # Max score at 10 conversations
        turn_score = min(1.0, turns / 50)  # Max score at 50 turns
        duration_score = min(1.0, avg_duration / 30)  # Max score at 30 minutes
        sentiment_score = (avg_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]

        # Weighted average
        engagement = (
            (conv_score * 0.3 + turn_score * 0.3 + duration_score * 0.2 + sentiment_score * 0.2)
        )

        return engagement

    def _classify_user_type(self,
                          engagement_score: float,
                          context_distribution: Counter) -> str:
        """Classify user type based on behavior"""

        if engagement_score > 0.7:
            return "power_user"
        elif engagement_score > 0.4:
            return "regular_user"
        elif engagement_score > 0.2:
            return "occasional_user"
        else:
            return "new_user"

    async def _identify_behavior_patterns(
        self,
        user_behaviors: Dict[str,
        Dict[str,
        Any]]
    ) -> Dict[str, Any]:
        """Identify common behavior patterns"""

        patterns = {
            'most_common_user_type': None,
            'average_engagement': 0.0,
            'common_contexts': [],
            'behavior_clusters': []
        }

        if not user_behaviors:
            return patterns

        # User type distribution
        user_types = [profile['user_type'] for profile in user_behaviors.values()]
        patterns['most_common_user_type'] = Counter(user_types).most_common(1)[0][0]

        # Average engagement
        engagements = [profile['engagement_score'] for profile in user_behaviors.values()]
        patterns['average_engagement'] = np.mean(engagements)

        # Common contexts across all users
        all_contexts = []
        for profile in user_behaviors.values():
            context_dist = profile.get('context_distribution', {})
            all_contexts.extend(context_dist.keys())

        patterns['common_contexts'] = [ctx for ctx, count in Counter(all_contexts).most_common(5)]

        return patterns

    async def _segment_users(self, user_behaviors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Segment users based on behavior"""

        if len(user_behaviors) < 2:
            return {'segments': [], 'total_segments': 0}

        # Extract features for clustering
        features = []
        user_ids = []

        for user_id, profile in user_behaviors.items():
            feature_vector = [
                profile.get('total_conversations', 0),
                profile.get('total_turns', 0),
                profile.get('average_session_duration', 0),
                profile.get('engagement_score', 0),
                profile.get('average_sentiment', 0)
            ]
            features.append(feature_vector)
            user_ids.append(user_id)

        features = np.array(features)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform clustering
        try:
            n_clusters = min(4, len(user_behaviors))  # Max 4 segments
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # Organize segments
            segments = defaultdict(list)
            for user_id, label in zip(user_ids, cluster_labels):
                segments[f"segment_{label}"].append({
                    'user_id': user_id,
                    'profile': user_behaviors[user_id]
                })

            return {
                'segments': dict(segments),
                'total_segments': n_clusters,
                'clustering_quality': silhouette_score(
                    features_scaled,
                    cluster_labels
                ) if n_clusters > 1 else 0
            }

        except Exception as e:
            logger.error(f"User segmentation failed: {str(e)}")
            return {'segments': [], 'total_segments': 0}


class ConversationAnalyticsEngine:
    """Main conversation analytics engine"""

    def __init__(self):
        self.flow_analyzer = ConversationFlowAnalyzer()
        self.topic_engine = TopicModelingEngine()
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.metrics_store: Dict[str, ConversationMetrics] = {}
        self.insights_store: Dict[str, AnalyticsInsight] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize analytics engine"""
        try:
            await self.flow_analyzer.initialize()
            await self.topic_engine.initialize()
            await self.behavior_analyzer.initialize()

            self._initialized = True
            logger.info("Conversation analytics engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize analytics engine: {str(e)}")
            raise

    async def analyze_conversations(self,
                                  conversations: List[Dict[str, Any]],
                                  analysis_types: List[AnalyticsType] = None) -> Dict[str, Any]:
        """Perform comprehensive conversation analysis"""

        if not self._initialized:
            await self.initialize()

        if analysis_types is None:
            analysis_types = list(AnalyticsType)

        results = {}

        try:
            # Conversation flow analysis
            if AnalyticsType.CONVERSATION_FLOW in analysis_types:
                results['conversation_flow'] = (
                    await self.flow_analyzer.analyze_conversation_flows(conversations)
                )

            # Topic modeling
            if AnalyticsType.TOPIC_MODELING in analysis_types:
                results['topic_analysis'] = (
                    await self.topic_engine.analyze_conversation_topics(conversations)
                )

            # User behavior analysis
            if AnalyticsType.USER_BEHAVIOR in analysis_types:
                results['user_behavior'] = (
                    await self.behavior_analyzer.analyze_user_behavior(conversations)
                )

            # Sentiment trends
            if AnalyticsType.SENTIMENT_TRENDS in analysis_types:
                results['sentiment_trends'] = await self._analyze_sentiment_trends(conversations)

            # Knowledge gaps
            if AnalyticsType.KNOWLEDGE_GAPS in analysis_types:
                results['knowledge_gaps'] = await self._identify_knowledge_gaps(conversations)

            # Efficiency metrics
            if AnalyticsType.EFFICIENCY_METRICS in analysis_types:
                results['efficiency_metrics'] = (
                    await self._calculate_efficiency_metrics(conversations)
                )

            # Generate insights
            insights = await self._generate_insights(results)
            results['insights'] = insights

            # Cache results
            await self._cache_analytics_results(results)

            return results

        except Exception as e:
            logger.error(f"Conversation analysis failed: {str(e)}")
            raise APIError(f"Conversation analysis failed: {str(e)}", status_code=500)

    async def _analyze_sentiment_trends(
        self,
        conversations: List[Dict[str,
        Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""

        # Group by time periods
        daily_sentiments = defaultdict(list)

        for conversation in conversations:
            turns = conversation.get('turns', [])
            for turn in turns:
                timestamp = turn.get('timestamp')
                sentiment = turn.get('sentiment', 0)

                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

                    date_key = timestamp.date().isoformat()
                    daily_sentiments[date_key].append(sentiment)

        # Calculate daily averages
        sentiment_trends = {}
        for date_key, sentiments in daily_sentiments.items():
            sentiment_trends[date_key] = {
                'average_sentiment': np.mean(sentiments),
                'sentiment_std': np.std(sentiments),
                'total_interactions': len(sentiments)
            }

        return {
            'daily_trends': sentiment_trends,
            'overall_trend': self._calculate_overall_sentiment_trend(sentiment_trends)
        }

    def _calculate_overall_sentiment_trend(self, daily_trends: Dict[str, Dict[str, float]]) -> str:
        """Calculate overall sentiment trend direction"""

        if len(daily_trends) < 2:
            return 'insufficient_data'

        dates = sorted(daily_trends.keys())
        sentiments = [daily_trends[date]['average_sentiment'] for date in dates]

        # Simple trend calculation
        if len(sentiments) >= 2:
            trend_slope = np.polyfit(range(len(sentiments)), sentiments, 1)[0]

            if trend_slope > 0.05:
                return 'improving'
            elif trend_slope < -0.05:
                return 'declining'
            else:
                return 'stable'

        return 'stable'

    async def _identify_knowledge_gaps(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify knowledge gaps based on conversation patterns"""

        unresolved_queries = []
        low_confidence_responses = []
        escalated_conversations = []

        for conversation in conversations:
            turns = conversation.get('turns', [])

            # Check for unresolved queries
            if conversation.get('outcome') != 'resolved':
                unresolved_queries.append(conversation)

            # Check for low confidence responses
            for turn in turns:
                confidence = turn.get('confidence', 1.0)
                if confidence < 0.5:
                    low_confidence_responses.append({
                        'conversation_id': conversation.get('session_id'),
                        'query': turn.get('input_text', ''),
                        'confidence': confidence,
                        'context': turn.get('context')
                    })

            # Check for escalations
            if conversation.get('outcome') == 'escalated':
                escalated_conversations.append(conversation)

        # Identify common themes in gaps
        gap_themes = await self._identify_gap_themes(
            unresolved_queries + [{'turns': [lc]} for lc in low_confidence_responses]
        )

        return {
            'unresolved_queries': len(unresolved_queries),
            'low_confidence_responses': len(low_confidence_responses),
            'escalated_conversations': len(escalated_conversations),
            'gap_themes': gap_themes,
            'knowledge_coverage': 1.0 - (len(unresolved_queries) / len(conversations)) if conversations else 1.0
        }

    async def _identify_gap_themes(
        self,
        problematic_conversations: List[Dict[str,
        Any]]
    ) -> List[Dict[str, Any]]:
        """Identify themes in knowledge gaps"""

        # Extract text from problematic conversations
        gap_texts = []
        for conversation in problematic_conversations:
            turns = conversation.get('turns', [])
            conversation_text = " ".join([turn.get('input_text', '') for turn in turns])
            gap_texts.append(conversation_text)

        # Simple keyword-based theme identification
        gap_keywords = defaultdict(int)

        for text in gap_texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    gap_keywords[word] += 1

        # Return most common themes
        common_themes = gap_keywords.most_common(10)

        return [{'theme': theme, 'frequency': freq} for theme, freq in common_themes]

    async def _calculate_efficiency_metrics(
        self,
        conversations: List[Dict[str,
        Any]]
    ) -> Dict[str, Any]:
        """Calculate conversation efficiency metrics"""

        total_conversations = len(conversations)
        if total_conversations == 0:
            return {}

        # Resolution metrics
        resolved = sum(1 for conv in conversations if conv.get('outcome') == 'resolved')
        escalated = sum(1 for conv in conversations if conv.get('outcome') == 'escalated')

        # Turn metrics
        turn_counts = [len(conv.get('turns', [])) for conv in conversations]
        avg_turns = np.mean(turn_counts) if turn_counts else 0

        # Duration metrics
        durations = []
        for conv in conversations:
            start_time = conv.get('started_at')
            end_time = conv.get('updated_at', start_time)

            if start_time and end_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

                duration = (end_time - start_time).total_seconds() / 60  # minutes
                durations.append(duration)

        avg_duration = np.mean(durations) if durations else 0

        # Confidence metrics
        confidences = []
        for conv in conversations:
            turns = conv.get('turns', [])
            conv_confidences = [turn.get('confidence', 0) for turn in turns]
            if conv_confidences:
                confidences.extend(conv_confidences)

        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'resolution_rate': resolved / total_conversations,
            'escalation_rate': escalated / total_conversations,
            'average_turns_per_conversation': avg_turns,
            'average_duration_minutes': avg_duration,
            'average_confidence': avg_confidence,
            'efficiency_score': (resolved / total_conversations) * (1 - escalated / total_conversations) * avg_confidence
        }

    async def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate actionable insights from analysis results"""

        insights = []

        # Flow analysis insights
        if 'conversation_flow' in analysis_results:
            flow_data = analysis_results['conversation_flow']

            if flow_data.get('efficiency_metrics', {}).get('escalation_rate', 0) > 0.2:
                insights.append(AnalyticsInsight(
                    insight_id=str(uuid.uuid4()),
                    title="High Escalation Rate Detected",
                    description=f"Escalation rate is {flow_data['efficiency_metrics']['escalation_rate']:.1%}, which is above the recommended 20% threshold.",
                    category="efficiency",
                    priority="high",
                    confidence=0.9,
                    actionable=True,
                    recommendations=[
                        "Review knowledge base coverage for frequently escalated topics",
                        "Improve automated response accuracy",
                        "Provide additional training for common scenarios"
                    ],
                    data_source="conversation_flow",
                    generated_at=datetime.now()
                ))

        # User behavior insights
        if 'user_behavior' in analysis_results:
            behavior_data = analysis_results['user_behavior']

            avg_engagement = behavior_data.get('behavior_patterns', {}).get('average_engagement', 0)
            if avg_engagement < 0.5:
                insights.append(AnalyticsInsight(
                    insight_id=str(uuid.uuid4()),
                    title="Low User Engagement Detected",
                    description=f"Average user engagement is {avg_engagement:.1%}, indicating potential UX issues.",
                    category="user_experience",
                    priority="medium",
                    confidence=0.8,
                    actionable=True,
                    recommendations=[
                        "Improve response relevance and accuracy",
                        "Reduce conversation length for simple queries",
                        "Add more interactive elements to conversations"
                    ],
                    data_source="user_behavior",
                    generated_at=datetime.now()
                ))

        # Knowledge gap insights
        if 'knowledge_gaps' in analysis_results:
            gap_data = analysis_results['knowledge_gaps']

            if gap_data.get('knowledge_coverage', 1.0) < 0.8:
                insights.append(AnalyticsInsight(
                    insight_id=str(uuid.uuid4()),
                    title="Knowledge Coverage Gap Identified",
                    description=f"Knowledge coverage is {gap_data['knowledge_coverage']:.1%}, indicating missing information.",
                    category="knowledge_management",
                    priority="high",
                    confidence=0.9,
                    actionable=True,
                    recommendations=[
                        "Expand knowledge base with missing topics",
                        "Review and update existing knowledge articles",
                        "Train models on additional governance scenarios"
                    ],
                    data_source="knowledge_gaps",
                    generated_at=datetime.now()
                ))

        return insights

    async def _cache_analytics_results(self, results: Dict[str, Any]):
        """Cache analytics results"""
        cache_key = f"conversation_analytics:{datetime.now().date().isoformat()}"

        try:
            await redis_client.setex(
                cache_key,
                timedelta(hours=24),
                json.dumps(results, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache analytics results: {str(e)}")


# Global instance
conversation_analytics = ConversationAnalyticsEngine()
