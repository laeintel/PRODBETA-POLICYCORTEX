"""
Cross-Domain Correlation Engine
Patent 4: Advanced Cross-Domain Correlation and Pattern Recognition System
Provides comprehensive correlation analysis across multiple Azure governance domains
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import networkx as nx
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import mutual_info_score, adjusted_rand_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE, GAT
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings

from backend.core.config import settings
from backend.core.redis_client import redis_client
from backend.core.exceptions import APIError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CorrelationType(str, Enum):
    """Types of correlation analysis"""
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    ANOMALY = "anomaly"
    PREDICTIVE = "predictive"


class DomainType(str, Enum):
    """Azure governance domains"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    COST_MANAGEMENT = "cost_management"
    PERFORMANCE = "performance"
    RESOURCE_MANAGEMENT = "resource_management"
    ACCESS_CONTROL = "access_control"
    DATA_GOVERNANCE = "data_governance"
    NETWORK = "network"
    MONITORING = "monitoring"


class CorrelationStrength(str, Enum):
    """Correlation strength levels"""
    VERY_WEAK = "very_weak"      # 0.0 - 0.2
    WEAK = "weak"                # 0.2 - 0.4
    MODERATE = "moderate"        # 0.4 - 0.6
    STRONG = "strong"            # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0


@dataclass
class CorrelationEvent:
    """Represents a single correlation event"""
    event_id: str
    domain: DomainType
    timestamp: datetime
    event_type: str
    severity: float
    attributes: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class CorrelationPattern:
    """Represents a discovered correlation pattern"""
    pattern_id: str
    correlation_type: CorrelationType
    domains: List[DomainType]
    strength: CorrelationStrength
    confidence: float
    events: List[CorrelationEvent]
    temporal_window: Optional[timedelta] = None
    frequency: int = 1
    description: str = ""
    rule_template: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationInsight:
    """Actionable insight from correlation analysis"""
    insight_id: str
    title: str
    description: str
    category: str
    priority: str  # critical, high, medium, low
    confidence: float
    affected_domains: List[DomainType]
    patterns: List[CorrelationPattern]
    recommendations: List[str]
    potential_impact: str
    risk_score: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphNeuralCorrelator(nn.Module):
    """Graph Neural Network for cross-domain correlation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        self.convs.append(GAT(input_dim, hidden_dim, heads=4, concat=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GAT(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        
        self.convs.append(GAT(hidden_dim * 4, output_dim, heads=1, concat=False))
        
        # Correlation prediction layers
        self.correlation_head = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_pairs=None):
        # Graph convolution
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # If edge pairs provided, predict correlations
        if edge_pairs is not None:
            correlations = []
            for i, j in edge_pairs:
                pair_features = torch.cat([x[i], x[j]], dim=0)
                correlation = self.correlation_head(pair_features)
                correlations.append(correlation)
            return x, torch.stack(correlations)
        
        return x


class TemporalCorrelationAnalyzer:
    """Analyzes temporal correlations between events"""
    
    def __init__(self, window_size: int = 3600):  # 1 hour default
        self.window_size = window_size
        self.event_buffer = defaultdict(lambda: deque(maxlen=1000))
    
    async def analyze_temporal_correlations(self, 
                                          events: List[CorrelationEvent]) -> List[CorrelationPattern]:
        """Analyze temporal patterns in events"""
        patterns = []
        
        # Group events by domain
        domain_events = defaultdict(list)
        for event in events:
            domain_events[event.domain].append(event)
        
        # Sort events by timestamp within each domain
        for domain in domain_events:
            domain_events[domain].sort(key=lambda x: x.timestamp)
        
        # Find cross-domain temporal correlations
        domain_pairs = self._get_domain_pairs(list(domain_events.keys()))
        
        for domain1, domain2 in domain_pairs:
            events1 = domain_events[domain1]
            events2 = domain_events[domain2]
            
            correlation_strength = await self._calculate_temporal_correlation(events1, events2)
            
            if correlation_strength > 0.3:  # Threshold for meaningful correlation
                pattern = CorrelationPattern(
                    pattern_id=f"temporal_{domain1}_{domain2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    correlation_type=CorrelationType.TEMPORAL,
                    domains=[domain1, domain2],
                    strength=self._get_strength_category(correlation_strength),
                    confidence=correlation_strength,
                    events=self._get_correlated_events(events1, events2),
                    temporal_window=timedelta(seconds=self.window_size),
                    description=f"Temporal correlation between {domain1} and {domain2} events"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _calculate_temporal_correlation(self, 
                                            events1: List[CorrelationEvent], 
                                            events2: List[CorrelationEvent]) -> float:
        """Calculate temporal correlation coefficient"""
        if len(events1) < 2 or len(events2) < 2:
            return 0.0
        
        # Create time series for both event streams
        timestamps1 = [event.timestamp.timestamp() for event in events1]
        timestamps2 = [event.timestamp.timestamp() for event in events2]
        
        # Bin events into time windows
        min_time = min(min(timestamps1), min(timestamps2))
        max_time = max(max(timestamps1), max(timestamps2))
        
        bins = np.arange(min_time, max_time + self.window_size, self.window_size)
        
        # Count events in each bin
        counts1, _ = np.histogram(timestamps1, bins=bins)
        counts2, _ = np.histogram(timestamps2, bins=bins)
        
        # Calculate correlation
        if np.std(counts1) > 0 and np.std(counts2) > 0:
            correlation = np.corrcoef(counts1, counts2)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _get_domain_pairs(self, domains: List[DomainType]) -> List[Tuple[DomainType, DomainType]]:
        """Get all pairs of domains for correlation analysis"""
        pairs = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                pairs.append((domains[i], domains[j]))
        return pairs
    
    def _get_correlated_events(self, 
                              events1: List[CorrelationEvent], 
                              events2: List[CorrelationEvent]) -> List[CorrelationEvent]:
        """Get events that are temporally correlated"""
        correlated = []
        
        for event1 in events1:
            for event2 in events2:
                time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                if time_diff <= self.window_size:
                    correlated.extend([event1, event2])
        
        return list(set(correlated))
    
    def _get_strength_category(self, correlation: float) -> CorrelationStrength:
        """Convert correlation coefficient to strength category"""
        if correlation >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif correlation >= 0.6:
            return CorrelationStrength.STRONG
        elif correlation >= 0.4:
            return CorrelationStrength.MODERATE
        elif correlation >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK


class CausalInferenceEngine:
    """Infers causal relationships between events"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_models = {}
    
    async def infer_causal_relationships(self, 
                                       events: List[CorrelationEvent]) -> List[CorrelationPattern]:
        """Infer causal relationships using Granger causality and other methods"""
        patterns = []
        
        # Group events by domain and create time series
        domain_series = self._create_domain_time_series(events)
        
        # Test for Granger causality between domains
        domain_pairs = self._get_domain_pairs(list(domain_series.keys()))
        
        for domain1, domain2 in domain_pairs:
            series1 = domain_series[domain1]
            series2 = domain_series[domain2]
            
            # Test both directions for causality
            causality_12 = await self._test_granger_causality(series1, series2)
            causality_21 = await self._test_granger_causality(series2, series1)
            
            if causality_12['significant'] or causality_21['significant']:
                if causality_12['p_value'] < causality_21['p_value']:
                    cause_domain, effect_domain = domain1, domain2
                    p_value = causality_12['p_value']
                else:
                    cause_domain, effect_domain = domain2, domain1
                    p_value = causality_21['p_value']
                
                confidence = 1 - p_value
                
                pattern = CorrelationPattern(
                    pattern_id=f"causal_{cause_domain}_{effect_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    correlation_type=CorrelationType.CAUSAL,
                    domains=[cause_domain, effect_domain],
                    strength=self._get_strength_category(confidence),
                    confidence=confidence,
                    events=self._get_causal_events(events, cause_domain, effect_domain),
                    description=f"Causal relationship: {cause_domain} causes {effect_domain}"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _create_domain_time_series(self, events: List[CorrelationEvent]) -> Dict[DomainType, pd.Series]:
        """Create time series for each domain"""
        domain_series = {}
        
        for domain in DomainType:
            domain_events = [e for e in events if e.domain == domain]
            if not domain_events:
                continue
            
            # Create time series with event counts per hour
            timestamps = [e.timestamp for e in domain_events]
            min_time = min(timestamps).replace(minute=0, second=0, microsecond=0)
            max_time = max(timestamps).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            
            time_range = pd.date_range(start=min_time, end=max_time, freq='H')
            counts = []
            
            for timestamp in time_range:
                count = sum(1 for e in domain_events 
                           if timestamp <= e.timestamp < timestamp + timedelta(hours=1))
                counts.append(count)
            
            domain_series[domain] = pd.Series(counts, index=time_range)
        
        return domain_series
    
    async def _test_granger_causality(self, cause_series: pd.Series, effect_series: pd.Series) -> Dict[str, Any]:
        """Test for Granger causality between two time series"""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Align series
            common_index = cause_series.index.intersection(effect_series.index)
            if len(common_index) < 10:  # Need sufficient data
                return {'significant': False, 'p_value': 1.0}
            
            cause_aligned = cause_series.loc[common_index]
            effect_aligned = effect_series.loc[common_index]
            
            # Prepare data for Granger test
            data = pd.DataFrame({
                'cause': cause_aligned.values,
                'effect': effect_aligned.values
            })
            
            # Test with lag of 1-3 periods
            max_lag = min(3, len(data) // 4)
            if max_lag < 1:
                return {'significant': False, 'p_value': 1.0}
            
            results = grangercausalitytests(data[['effect', 'cause']], maxlag=max_lag, verbose=False)
            
            # Get minimum p-value across lags
            p_values = []
            for lag in range(1, max_lag + 1):
                if lag in results:
                    f_test = results[lag][0]['ssr_ftest']
                    p_values.append(f_test[1])  # p-value
            
            if p_values:
                min_p_value = min(p_values)
                return {
                    'significant': min_p_value < 0.05,
                    'p_value': min_p_value
                }
            
        except Exception as e:
            logger.warning(f"Granger causality test failed: {str(e)}")
        
        return {'significant': False, 'p_value': 1.0}
    
    def _get_causal_events(self, 
                          events: List[CorrelationEvent], 
                          cause_domain: DomainType, 
                          effect_domain: DomainType) -> List[CorrelationEvent]:
        """Get events that demonstrate the causal relationship"""
        causal_events = []
        
        cause_events = [e for e in events if e.domain == cause_domain]
        effect_events = [e for e in events if e.domain == effect_domain]
        
        # Find effect events that occur after cause events within a reasonable window
        for cause_event in cause_events:
            for effect_event in effect_events:
                time_diff = (effect_event.timestamp - cause_event.timestamp).total_seconds()
                if 0 < time_diff <= 3600:  # Effect within 1 hour of cause
                    causal_events.extend([cause_event, effect_event])
        
        return causal_events
    
    def _get_domain_pairs(self, domains: List[DomainType]) -> List[Tuple[DomainType, DomainType]]:
        """Get all pairs of domains for causality analysis"""
        pairs = []
        for i in range(len(domains)):
            for j in range(len(domains)):
                if i != j:
                    pairs.append((domains[i], domains[j]))
        return pairs
    
    def _get_strength_category(self, confidence: float) -> CorrelationStrength:
        """Convert confidence to strength category"""
        if confidence >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif confidence >= 0.6:
            return CorrelationStrength.STRONG
        elif confidence >= 0.4:
            return CorrelationStrength.MODERATE
        elif confidence >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK


class AnomalyCorrelationDetector:
    """Detects correlations in anomalous events"""
    
    def __init__(self):
        self.anomaly_detectors = {}
        self.baseline_models = {}
    
    async def detect_anomaly_correlations(self, 
                                        events: List[CorrelationEvent]) -> List[CorrelationPattern]:
        """Detect correlations in anomalous events"""
        patterns = []
        
        # Detect anomalies in each domain
        domain_anomalies = {}
        for domain in DomainType:
            domain_events = [e for e in events if e.domain == domain]
            if len(domain_events) > 10:  # Need sufficient data
                anomalies = await self._detect_domain_anomalies(domain, domain_events)
                if anomalies:
                    domain_anomalies[domain] = anomalies
        
        # Find correlations between anomalous periods
        domain_pairs = self._get_domain_pairs(list(domain_anomalies.keys()))
        
        for domain1, domain2 in domain_pairs:
            anomalies1 = domain_anomalies[domain1]
            anomalies2 = domain_anomalies[domain2]
            
            correlation = await self._calculate_anomaly_correlation(anomalies1, anomalies2)
            
            if correlation > 0.5:  # Threshold for significant anomaly correlation
                pattern = CorrelationPattern(
                    pattern_id=f"anomaly_{domain1}_{domain2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    correlation_type=CorrelationType.ANOMALY,
                    domains=[domain1, domain2],
                    strength=self._get_strength_category(correlation),
                    confidence=correlation,
                    events=anomalies1 + anomalies2,
                    description=f"Correlated anomalies between {domain1} and {domain2}"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_domain_anomalies(self, 
                                     domain: DomainType, 
                                     events: List[CorrelationEvent]) -> List[CorrelationEvent]:
        """Detect anomalies in domain events"""
        if len(events) < 10:
            return []
        
        # Extract features for anomaly detection
        features = []
        for event in events:
            feature_vector = [
                event.severity,
                event.timestamp.hour,
                event.timestamp.weekday(),
                len(event.attributes),
                sum(1 for v in event.attributes.values() if isinstance(v, (int, float)))
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Use Isolation Forest for anomaly detection
        if domain not in self.anomaly_detectors:
            self.anomaly_detectors[domain] = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
        
        detector = self.anomaly_detectors[domain]
        
        try:
            anomaly_labels = detector.fit_predict(features)
            anomalies = [events[i] for i in range(len(events)) if anomaly_labels[i] == -1]
            return anomalies
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {domain}: {str(e)}")
            return []
    
    async def _calculate_anomaly_correlation(self, 
                                           anomalies1: List[CorrelationEvent], 
                                           anomalies2: List[CorrelationEvent]) -> float:
        """Calculate correlation between anomalous periods"""
        if len(anomalies1) < 2 or len(anomalies2) < 2:
            return 0.0
        
        # Create binary time series indicating anomalous periods
        all_timestamps = [e.timestamp for e in anomalies1 + anomalies2]
        min_time = min(all_timestamps).replace(minute=0, second=0, microsecond=0)
        max_time = max(all_timestamps).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        time_range = pd.date_range(start=min_time, end=max_time, freq='H')
        
        # Create binary series
        series1 = []
        series2 = []
        
        for timestamp in time_range:
            has_anomaly1 = any(timestamp <= e.timestamp < timestamp + timedelta(hours=1) 
                              for e in anomalies1)
            has_anomaly2 = any(timestamp <= e.timestamp < timestamp + timedelta(hours=1) 
                              for e in anomalies2)
            
            series1.append(1 if has_anomaly1 else 0)
            series2.append(1 if has_anomaly2 else 0)
        
        # Calculate correlation
        if np.std(series1) > 0 and np.std(series2) > 0:
            correlation = np.corrcoef(series1, series2)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _get_domain_pairs(self, domains: List[DomainType]) -> List[Tuple[DomainType, DomainType]]:
        """Get all pairs of domains for correlation analysis"""
        pairs = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                pairs.append((domains[i], domains[j]))
        return pairs
    
    def _get_strength_category(self, correlation: float) -> CorrelationStrength:
        """Convert correlation coefficient to strength category"""
        if correlation >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif correlation >= 0.6:
            return CorrelationStrength.STRONG
        elif correlation >= 0.4:
            return CorrelationStrength.MODERATE
        elif correlation >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK


class CrossDomainCorrelationEngine:
    """Main cross-domain correlation engine"""
    
    def __init__(self):
        self.temporal_analyzer = TemporalCorrelationAnalyzer()
        self.causal_engine = CausalInferenceEngine()
        self.anomaly_detector = AnomalyCorrelationDetector()
        self.gnn_model = None
        self.pattern_store: Dict[str, CorrelationPattern] = {}
        self.insight_store: Dict[str, CorrelationInsight] = {}
        self.correlation_graph = nx.MultiDiGraph()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the correlation engine"""
        try:
            # Initialize GNN model
            self.gnn_model = GraphNeuralCorrelator(
                input_dim=64,  # Feature dimension
                hidden_dim=128,
                output_dim=64,
                num_layers=3
            )
            
            self._initialized = True
            logger.info("Cross-domain correlation engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize correlation engine: {str(e)}")
            raise
    
    async def analyze_correlations(self, 
                                 events: List[CorrelationEvent],
                                 correlation_types: List[CorrelationType] = None) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        
        if not self._initialized:
            await self.initialize()
        
        if correlation_types is None:
            correlation_types = [CorrelationType.TEMPORAL, CorrelationType.CAUSAL, CorrelationType.ANOMALY]
        
        all_patterns = []
        analysis_results = {}
        
        try:
            # Temporal correlation analysis
            if CorrelationType.TEMPORAL in correlation_types:
                temporal_patterns = await self.temporal_analyzer.analyze_temporal_correlations(events)
                all_patterns.extend(temporal_patterns)
                analysis_results['temporal'] = {
                    'patterns_found': len(temporal_patterns),
                    'patterns': [self._pattern_to_dict(p) for p in temporal_patterns]
                }
            
            # Causal inference
            if CorrelationType.CAUSAL in correlation_types:
                causal_patterns = await self.causal_engine.infer_causal_relationships(events)
                all_patterns.extend(causal_patterns)
                analysis_results['causal'] = {
                    'patterns_found': len(causal_patterns),
                    'patterns': [self._pattern_to_dict(p) for p in causal_patterns]
                }
            
            # Anomaly correlation
            if CorrelationType.ANOMALY in correlation_types:
                anomaly_patterns = await self.anomaly_detector.detect_anomaly_correlations(events)
                all_patterns.extend(anomaly_patterns)
                analysis_results['anomaly'] = {
                    'patterns_found': len(anomaly_patterns),
                    'patterns': [self._pattern_to_dict(p) for p in anomaly_patterns]
                }
            
            # Statistical correlations
            if CorrelationType.STATISTICAL in correlation_types:
                statistical_patterns = await self._analyze_statistical_correlations(events)
                all_patterns.extend(statistical_patterns)
                analysis_results['statistical'] = {
                    'patterns_found': len(statistical_patterns),
                    'patterns': [self._pattern_to_dict(p) for p in statistical_patterns]
                }
            
            # Graph-based analysis using GNN
            if CorrelationType.STRUCTURAL in correlation_types:
                structural_patterns = await self._analyze_structural_correlations(events)
                all_patterns.extend(structural_patterns)
                analysis_results['structural'] = {
                    'patterns_found': len(structural_patterns),
                    'patterns': [self._pattern_to_dict(p) for p in structural_patterns]
                }
            
            # Store patterns
            for pattern in all_patterns:
                self.pattern_store[pattern.pattern_id] = pattern
                await self._update_correlation_graph(pattern)
            
            # Generate insights
            insights = await self._generate_correlation_insights(all_patterns)
            for insight in insights:
                self.insight_store[insight.insight_id] = insight
            
            # Calculate summary statistics
            summary = await self._calculate_summary_statistics(all_patterns, events)
            
            analysis_results.update({
                'summary': summary,
                'insights': [self._insight_to_dict(i) for i in insights],
                'total_patterns': len(all_patterns),
                'correlation_graph_stats': {
                    'nodes': self.correlation_graph.number_of_nodes(),
                    'edges': self.correlation_graph.number_of_edges(),
                    'domains_connected': len(set(data['domain'] for _, data in self.correlation_graph.nodes(data=True)))
                }
            })
            
            # Cache results
            await self._cache_analysis_results(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            raise APIError(f"Correlation analysis failed: {str(e)}", status_code=500)
    
    async def _analyze_statistical_correlations(self, 
                                              events: List[CorrelationEvent]) -> List[CorrelationPattern]:
        """Analyze statistical correlations between domains"""
        patterns = []
        
        # Create feature matrix for each domain
        domain_features = {}
        for domain in DomainType:
            domain_events = [e for e in events if e.domain == domain]
            if len(domain_events) < 5:
                continue
            
            # Extract statistical features
            features = self._extract_statistical_features(domain_events)
            if features is not None:
                domain_features[domain] = features
        
        # Calculate correlations between domain features
        domain_pairs = self._get_domain_pairs(list(domain_features.keys()))
        
        for domain1, domain2 in domain_pairs:
            features1 = domain_features[domain1]
            features2 = domain_features[domain2]
            
            # Calculate various correlation metrics
            pearson_corr = self._calculate_pearson_correlation(features1, features2)
            spearman_corr = self._calculate_spearman_correlation(features1, features2)
            mutual_info = self._calculate_mutual_information(features1, features2)
            
            # Use the strongest correlation
            max_correlation = max(abs(pearson_corr), abs(spearman_corr), mutual_info)
            
            if max_correlation > 0.3:
                pattern = CorrelationPattern(
                    pattern_id=f"statistical_{domain1}_{domain2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    correlation_type=CorrelationType.STATISTICAL,
                    domains=[domain1, domain2],
                    strength=self._get_strength_category(max_correlation),
                    confidence=max_correlation,
                    events=[e for e in events if e.domain in [domain1, domain2]],
                    description=f"Statistical correlation between {domain1} and {domain2}",
                    metadata={
                        'pearson_correlation': pearson_corr,
                        'spearman_correlation': spearman_corr,
                        'mutual_information': mutual_info
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_structural_correlations(self, 
                                             events: List[CorrelationEvent]) -> List[CorrelationPattern]:
        """Analyze structural correlations using graph neural networks"""
        patterns = []
        
        try:
            # Create graph representation
            graph_data = await self._create_event_graph(events)
            
            if graph_data is None:
                return patterns
            
            # Use GNN to find correlations
            with torch.no_grad():
                node_embeddings = self.gnn_model(graph_data.x, graph_data.edge_index)
                
                # Find strongly connected components
                components = await self._find_structural_components(node_embeddings, events)
                
                for component in components:
                    if len(component['domains']) > 1:
                        pattern = CorrelationPattern(
                            pattern_id=f"structural_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                            correlation_type=CorrelationType.STRUCTURAL,
                            domains=component['domains'],
                            strength=component['strength'],
                            confidence=component['confidence'],
                            events=component['events'],
                            description=f"Structural correlation cluster with {len(component['domains'])} domains"
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Structural correlation analysis failed: {str(e)}")
        
        return patterns
    
    def _extract_statistical_features(self, events: List[CorrelationEvent]) -> Optional[np.ndarray]:
        """Extract statistical features from events"""
        if not events:
            return None
        
        # Extract various features
        severities = [e.severity for e in events]
        timestamps = [e.timestamp.timestamp() for e in events]
        
        if len(severities) < 2:
            return None
        
        features = [
            np.mean(severities),
            np.std(severities),
            np.median(severities),
            np.percentile(severities, 75) - np.percentile(severities, 25),  # IQR
            len(events),
            (max(timestamps) - min(timestamps)) / 3600,  # Duration in hours
        ]
        
        return np.array(features)
    
    def _calculate_pearson_correlation(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Pearson correlation"""
        try:
            corr, _ = stats.pearsonr(features1, features2)
            return abs(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _calculate_spearman_correlation(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Spearman correlation"""
        try:
            corr, _ = stats.spearmanr(features1, features2)
            return abs(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _calculate_mutual_information(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate mutual information"""
        try:
            # Discretize continuous features
            bins = 10
            features1_discrete = np.digitize(features1, np.linspace(features1.min(), features1.max(), bins))
            features2_discrete = np.digitize(features2, np.linspace(features2.min(), features2.max(), bins))
            
            mi = mutual_info_score(features1_discrete, features2_discrete)
            return mi / np.log(bins)  # Normalize
        except:
            return 0.0
    
    async def _create_event_graph(self, events: List[CorrelationEvent]) -> Optional[Data]:
        """Create graph representation of events for GNN"""
        if len(events) < 5:
            return None
        
        try:
            # Create node features (one node per event)
            node_features = []
            for event in events:
                feature_vector = [
                    event.severity,
                    event.timestamp.hour / 24.0,  # Normalize hour
                    event.timestamp.weekday() / 7.0,  # Normalize weekday
                    len(event.attributes),
                    hash(event.domain.value) % 1000 / 1000.0,  # Domain encoding
                ] + [0.0] * 59  # Pad to 64 dimensions
                
                node_features.append(feature_vector[:64])
            
            # Create edges based on temporal proximity and domain relationships
            edge_indices = []
            for i in range(len(events)):
                for j in range(i + 1, len(events)):
                    event1, event2 = events[i], events[j]
                    
                    # Connect if events are temporally close or in related domains
                    time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                    
                    if (time_diff <= 3600 or  # Within 1 hour
                        self._are_domains_related(event1.domain, event2.domain)):
                        edge_indices.append([i, j])
                        edge_indices.append([j, i])  # Undirected graph
            
            if not edge_indices:
                return None
            
            # Convert to tensors
            x = torch.FloatTensor(node_features)
            edge_index = torch.LongTensor(edge_indices).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            logger.warning(f"Graph creation failed: {str(e)}")
            return None
    
    def _are_domains_related(self, domain1: DomainType, domain2: DomainType) -> bool:
        """Check if two domains are related"""
        related_pairs = {
            (DomainType.SECURITY, DomainType.ACCESS_CONTROL),
            (DomainType.SECURITY, DomainType.COMPLIANCE),
            (DomainType.COST_MANAGEMENT, DomainType.RESOURCE_MANAGEMENT),
            (DomainType.PERFORMANCE, DomainType.RESOURCE_MANAGEMENT),
            (DomainType.DATA_GOVERNANCE, DomainType.COMPLIANCE),
            (DomainType.NETWORK, DomainType.SECURITY),
            (DomainType.MONITORING, DomainType.PERFORMANCE),
        }
        
        return (domain1, domain2) in related_pairs or (domain2, domain1) in related_pairs
    
    async def _find_structural_components(self, 
                                        embeddings: torch.Tensor, 
                                        events: List[CorrelationEvent]) -> List[Dict[str, Any]]:
        """Find structural components in the graph"""
        components = []
        
        try:
            # Use clustering on embeddings to find components
            embeddings_np = embeddings.numpy()
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(embeddings_np)
            
            # Process each cluster
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Noise cluster
                    continue
                
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_indices) < 2:
                    continue
                
                cluster_events = [events[i] for i in cluster_indices]
                cluster_domains = list(set(e.domain for e in cluster_events))
                
                if len(cluster_domains) > 1:
                    # Calculate cluster coherence as confidence
                    cluster_embeddings = embeddings_np[cluster_indices]
                    coherence = 1.0 / (1.0 + np.std(cluster_embeddings))
                    
                    components.append({
                        'domains': cluster_domains,
                        'events': cluster_events,
                        'confidence': coherence,
                        'strength': self._get_strength_category(coherence)
                    })
        
        except Exception as e:
            logger.warning(f"Structural component finding failed: {str(e)}")
        
        return components
    
    async def _update_correlation_graph(self, pattern: CorrelationPattern):
        """Update the correlation graph with new pattern"""
        for domain in pattern.domains:
            if not self.correlation_graph.has_node(domain.value):
                self.correlation_graph.add_node(
                    domain.value,
                    domain=domain,
                    pattern_count=0
                )
            
            self.correlation_graph.nodes[domain.value]['pattern_count'] += 1
        
        # Add edges between correlated domains
        for i in range(len(pattern.domains)):
            for j in range(i + 1, len(pattern.domains)):
                domain1 = pattern.domains[i].value
                domain2 = pattern.domains[j].value
                
                self.correlation_graph.add_edge(
                    domain1,
                    domain2,
                    pattern_id=pattern.pattern_id,
                    correlation_type=pattern.correlation_type,
                    strength=pattern.strength,
                    confidence=pattern.confidence
                )
    
    async def _generate_correlation_insights(self, 
                                           patterns: List[CorrelationPattern]) -> List[CorrelationInsight]:
        """Generate actionable insights from correlation patterns"""
        insights = []
        
        # Group patterns by strength and type
        strong_patterns = [p for p in patterns if p.strength in [CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG]]
        causal_patterns = [p for p in patterns if p.correlation_type == CorrelationType.CAUSAL]
        anomaly_patterns = [p for p in patterns if p.correlation_type == CorrelationType.ANOMALY]
        
        # High-priority insight: Strong causal relationships
        if causal_patterns:
            insights.append(CorrelationInsight(
                insight_id=f"causal_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title="Critical Causal Relationships Detected",
                description=f"Found {len(causal_patterns)} causal relationships that require immediate attention.",
                category="causal_analysis",
                priority="high",
                confidence=np.mean([p.confidence for p in causal_patterns]),
                affected_domains=list(set(d for p in causal_patterns for d in p.domains)),
                patterns=causal_patterns,
                recommendations=[
                    "Investigate root causes in upstream domains",
                    "Implement preventive measures to break causal chains",
                    "Set up monitoring for causal triggers"
                ],
                potential_impact="High - Causal relationships can lead to cascading failures",
                risk_score=np.mean([p.confidence for p in causal_patterns]) * 0.9
            ))
        
        # Medium-priority insight: Anomaly correlations
        if anomaly_patterns:
            insights.append(CorrelationInsight(
                insight_id=f"anomaly_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title="Correlated Anomalies Detected",
                description=f"Found {len(anomaly_patterns)} patterns of correlated anomalies across domains.",
                category="anomaly_detection",
                priority="medium",
                confidence=np.mean([p.confidence for p in anomaly_patterns]),
                affected_domains=list(set(d for p in anomaly_patterns for d in p.domains)),
                patterns=anomaly_patterns,
                recommendations=[
                    "Investigate common causes of correlated anomalies",
                    "Update monitoring thresholds based on correlation patterns",
                    "Implement cross-domain anomaly detection rules"
                ],
                potential_impact="Medium - Correlated anomalies may indicate systemic issues",
                risk_score=np.mean([p.confidence for p in anomaly_patterns]) * 0.7
            ))
        
        # General insight: Strong correlations
        if strong_patterns:
            insights.append(CorrelationInsight(
                insight_id=f"strong_corr_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title="Strong Cross-Domain Correlations",
                description=f"Identified {len(strong_patterns)} strong correlations requiring optimization.",
                category="correlation_optimization",
                priority="medium",
                confidence=np.mean([p.confidence for p in strong_patterns]),
                affected_domains=list(set(d for p in strong_patterns for d in p.domains)),
                patterns=strong_patterns,
                recommendations=[
                    "Optimize correlated processes for better efficiency",
                    "Consider unified management for strongly correlated domains",
                    "Implement correlation-aware alerting rules"
                ],
                potential_impact="Medium - Strong correlations indicate optimization opportunities",
                risk_score=np.mean([p.confidence for p in strong_patterns]) * 0.6
            ))
        
        return insights
    
    async def _calculate_summary_statistics(self, 
                                          patterns: List[CorrelationPattern], 
                                          events: List[CorrelationEvent]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not patterns:
            return {
                'total_events': len(events),
                'patterns_found': 0,
                'domains_analyzed': len(set(e.domain for e in events)),
                'correlation_strength_distribution': {},
                'correlation_type_distribution': {}
            }
        
        strength_dist = defaultdict(int)
        type_dist = defaultdict(int)
        
        for pattern in patterns:
            strength_dist[pattern.strength.value] += 1
            type_dist[pattern.correlation_type.value] += 1
        
        return {
            'total_events': len(events),
            'patterns_found': len(patterns),
            'domains_analyzed': len(set(e.domain for e in events)),
            'domains_correlated': len(set(d for p in patterns for d in p.domains)),
            'correlation_strength_distribution': dict(strength_dist),
            'correlation_type_distribution': dict(type_dist),
            'average_confidence': np.mean([p.confidence for p in patterns]),
            'max_confidence': max([p.confidence for p in patterns]),
            'cross_domain_coverage': len(set(d for p in patterns for d in p.domains)) / len(DomainType) * 100
        }
    
    def _pattern_to_dict(self, pattern: CorrelationPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            'pattern_id': pattern.pattern_id,
            'correlation_type': pattern.correlation_type.value,
            'domains': [d.value for d in pattern.domains],
            'strength': pattern.strength.value,
            'confidence': pattern.confidence,
            'event_count': len(pattern.events),
            'temporal_window': pattern.temporal_window.total_seconds() if pattern.temporal_window else None,
            'frequency': pattern.frequency,
            'description': pattern.description,
            'created_at': pattern.created_at.isoformat(),
            'last_seen': pattern.last_seen.isoformat(),
            'metadata': pattern.metadata
        }
    
    def _insight_to_dict(self, insight: CorrelationInsight) -> Dict[str, Any]:
        """Convert insight to dictionary"""
        return {
            'insight_id': insight.insight_id,
            'title': insight.title,
            'description': insight.description,
            'category': insight.category,
            'priority': insight.priority,
            'confidence': insight.confidence,
            'affected_domains': [d.value for d in insight.affected_domains],
            'pattern_count': len(insight.patterns),
            'recommendations': insight.recommendations,
            'potential_impact': insight.potential_impact,
            'risk_score': insight.risk_score,
            'created_at': insight.created_at.isoformat(),
            'metadata': insight.metadata
        }
    
    def _get_domain_pairs(self, domains: List[DomainType]) -> List[Tuple[DomainType, DomainType]]:
        """Get all pairs of domains for correlation analysis"""
        pairs = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                pairs.append((domains[i], domains[j]))
        return pairs
    
    def _get_strength_category(self, correlation: float) -> CorrelationStrength:
        """Convert correlation coefficient to strength category"""
        if correlation >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif correlation >= 0.6:
            return CorrelationStrength.STRONG
        elif correlation >= 0.4:
            return CorrelationStrength.MODERATE
        elif correlation >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK
    
    async def _cache_analysis_results(self, results: Dict[str, Any]):
        """Cache analysis results"""
        cache_key = f"correlation_analysis:{datetime.now().date().isoformat()}"
        
        try:
            await redis_client.setex(
                cache_key,
                timedelta(hours=12),
                json.dumps(results, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache correlation results: {str(e)}")
    
    async def get_correlation_patterns(self, 
                                     correlation_type: Optional[CorrelationType] = None,
                                     min_confidence: float = 0.0,
                                     limit: int = 50) -> List[CorrelationPattern]:
        """Get correlation patterns with filtering"""
        patterns = list(self.pattern_store.values())
        
        # Apply filters
        if correlation_type:
            patterns = [p for p in patterns if p.correlation_type == correlation_type]
        
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # Sort by confidence and recency
        patterns.sort(key=lambda x: (x.confidence, x.last_seen), reverse=True)
        
        return patterns[:limit]
    
    async def get_correlation_insights(self, 
                                     priority: Optional[str] = None,
                                     category: Optional[str] = None,
                                     limit: int = 20) -> List[CorrelationInsight]:
        """Get correlation insights with filtering"""
        insights = list(self.insight_store.values())
        
        # Apply filters
        if priority:
            insights = [i for i in insights if i.priority == priority]
        
        if category:
            insights = [i for i in insights if i.category == category]
        
        # Sort by risk score and recency
        insights.sort(key=lambda x: (x.risk_score, x.created_at), reverse=True)
        
        return insights[:limit]
    
    async def get_domain_correlation_summary(self, domain: DomainType) -> Dict[str, Any]:
        """Get correlation summary for a specific domain"""
        domain_patterns = [p for p in self.pattern_store.values() if domain in p.domains]
        
        if not domain_patterns:
            return {
                'domain': domain.value,
                'total_patterns': 0,
                'correlation_types': {},
                'connected_domains': [],
                'average_confidence': 0.0
            }
        
        # Calculate statistics
        type_counts = defaultdict(int)
        connected_domains = set()
        
        for pattern in domain_patterns:
            type_counts[pattern.correlation_type.value] += 1
            connected_domains.update(d for d in pattern.domains if d != domain)
        
        return {
            'domain': domain.value,
            'total_patterns': len(domain_patterns),
            'correlation_types': dict(type_counts),
            'connected_domains': [d.value for d in connected_domains],
            'average_confidence': np.mean([p.confidence for p in domain_patterns]),
            'strongest_correlations': sorted(
                [(p.pattern_id, p.confidence, [d.value for d in p.domains if d != domain])
                 for p in domain_patterns],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Global instance
cross_domain_correlator = CrossDomainCorrelationEngine()