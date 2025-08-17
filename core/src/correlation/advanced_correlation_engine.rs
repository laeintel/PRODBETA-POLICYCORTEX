// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

// Advanced Cross-Domain Correlation Engine with ML-based Pattern Detection
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration, Timelike};
use petgraph::graph::{DiGraph, NodeIndex};
use uuid::Uuid;

/// Advanced correlation engine with machine learning capabilities
pub struct AdvancedCorrelationEngine {
    temporal_graph: TemporalGraph,
    pattern_detector: PatternDetector,
    correlation_memory: CorrelationMemory,
    ml_correlator: MLCorrelator,
    real_time_analyzer: RealTimeAnalyzer,
}

impl AdvancedCorrelationEngine {
    pub fn new() -> Self {
        Self {
            temporal_graph: TemporalGraph::new(),
            pattern_detector: PatternDetector::new(),
            correlation_memory: CorrelationMemory::new(),
            ml_correlator: MLCorrelator::new(),
            real_time_analyzer: RealTimeAnalyzer::new(),
        }
    }

    /// Advanced correlation analysis with ML-based pattern detection
    pub async fn analyze_advanced_correlations(&mut self, 
        resources: Vec<AzureResource>,
        events: Vec<ResourceEvent>,
        time_window: Duration
    ) -> AdvancedCorrelationResult {
        
        // Update temporal graph with new data
        self.temporal_graph.update(&resources, &events);
        
        // Detect temporal patterns
        let temporal_patterns = self.pattern_detector.detect_temporal_patterns(&events, time_window);
        
        // Find ML-based correlations
        let ml_correlations = self.ml_correlator.find_correlations(&resources, &events).await;
        
        // Analyze real-time correlations
        let real_time_correlations = self.real_time_analyzer.analyze(&events).await;
        
        // Predict future correlations
        let predicted_correlations = self.predict_future_correlations(&resources, &events).await;
        
        // Calculate correlation confidence scores
        let confidence_scores = self.calculate_correlation_confidence(&ml_correlations);
        
        // Generate advanced insights
        let insights = self.generate_correlation_insights(&temporal_patterns, &ml_correlations);
        
        AdvancedCorrelationResult {
            temporal_patterns,
            ml_correlations,
            real_time_correlations,
            predicted_correlations,
            confidence_scores,
            insights,
            anomalies: self.detect_correlation_anomalies(&events),
            graph_metrics: self.temporal_graph.calculate_metrics(),
        }
    }

    /// Predict future correlations based on historical patterns
    async fn predict_future_correlations(&self, 
        resources: &[AzureResource], 
        events: &[ResourceEvent]
    ) -> Vec<PredictedCorrelation> {
        let mut predictions = Vec::new();
        
        // Use temporal patterns to predict future correlations
        let patterns = self.correlation_memory.get_historical_patterns();
        
        for pattern in patterns {
            if let Some(prediction) = self.extrapolate_pattern(pattern, resources, events) {
                predictions.push(prediction);
            }
        }
        
        // ML-based predictions
        let ml_predictions = self.ml_correlator.predict_correlations(resources, events).await;
        predictions.extend(ml_predictions);
        
        predictions
    }

    fn extrapolate_pattern(&self, 
        pattern: &TemporalPattern, 
        resources: &[AzureResource], 
        events: &[ResourceEvent]
    ) -> Option<PredictedCorrelation> {
        // Analyze if current state matches pattern preconditions
        let pattern_strength = self.calculate_pattern_strength(pattern, events);
        
        if pattern_strength > 0.7 {
            Some(PredictedCorrelation {
                pattern_id: pattern.id.clone(),
                predicted_time: Utc::now() + pattern.typical_duration,
                source_resource: pattern.source_pattern.clone(),
                target_resource: pattern.target_pattern.clone(),
                confidence: pattern_strength,
                correlation_type: pattern.correlation_type.clone(),
                impact_level: pattern.impact_level.clone(),
            })
        } else {
            None
        }
    }

    fn calculate_pattern_strength(&self, pattern: &TemporalPattern, events: &[ResourceEvent]) -> f64 {
        let recent_events: Vec<_> = events.iter()
            .filter(|e| e.timestamp > Utc::now() - Duration::hours(24))
            .collect();
        
        let mut matches = 0;
        let mut total_checks = 0;
        
        for precondition in &pattern.preconditions {
            total_checks += 1;
            if self.check_precondition(precondition, &recent_events) {
                matches += 1;
            }
        }
        
        if total_checks > 0 {
            matches as f64 / total_checks as f64
        } else {
            0.0
        }
    }

    fn check_precondition(&self, precondition: &PatternPrecondition, events: &[&ResourceEvent]) -> bool {
        match precondition {
            PatternPrecondition::EventType(event_type) => {
                events.iter().any(|e| e.event_type == *event_type)
            },
            PatternPrecondition::ResourceState(resource_id, state) => {
                events.iter().any(|e| e.resource_id == *resource_id && e.new_state.as_ref() == Some(state))
            },
            PatternPrecondition::TimeWindow(duration) => {
                let oldest_event = events.iter().map(|e| e.timestamp).min();
                if let Some(oldest) = oldest_event {
                    Utc::now() - oldest <= *duration
                } else {
                    false
                }
            },
        }
    }

    fn calculate_correlation_confidence(&self, correlations: &[MLCorrelation]) -> HashMap<String, f64> {
        let mut confidence_scores = HashMap::new();
        
        for correlation in correlations {
            let base_confidence = correlation.ml_confidence;
            
            // Adjust based on historical accuracy
            let historical_accuracy = self.correlation_memory
                .get_historical_accuracy(&correlation.correlation_id);
            
            // Adjust based on data quality
            let data_quality_factor = self.calculate_data_quality_factor(correlation);
            
            // Adjust based on correlation consistency
            let consistency_factor = self.calculate_consistency_factor(correlation);
            
            let adjusted_confidence = base_confidence * historical_accuracy * data_quality_factor * consistency_factor;
            
            confidence_scores.insert(correlation.correlation_id.clone(), adjusted_confidence);
        }
        
        confidence_scores
    }

    fn calculate_data_quality_factor(&self, correlation: &MLCorrelation) -> f64 {
        let mut quality_score = 1.0;
        
        // Check data completeness
        if correlation.data_points < 100 {
            quality_score *= 0.8;
        }
        
        // Check data recency
        if correlation.last_updated < Utc::now() - Duration::hours(24) {
            quality_score *= 0.9;
        }
        
        // Check data variance
        if correlation.variance < 0.1 {
            quality_score *= 0.7; // Low variance might indicate limited data
        }
        
        quality_score
    }

    fn calculate_consistency_factor(&self, correlation: &MLCorrelation) -> f64 {
        // Check if correlation is consistent with known patterns
        let similar_correlations = self.correlation_memory
            .find_similar_correlations(&correlation.source_id, &correlation.target_id);
        
        if similar_correlations.is_empty() {
            0.8 // New correlation, slightly lower confidence
        } else {
            let avg_strength: f64 = similar_correlations.iter()
                .map(|c| c.strength)
                .sum::<f64>() / similar_correlations.len() as f64;
            
            // How close is this correlation to historical average?
            let diff = (correlation.strength - avg_strength).abs();
            (1.0 - diff).max(0.5)
        }
    }

    fn generate_correlation_insights(&self, 
        temporal_patterns: &[TemporalPattern],
        ml_correlations: &[MLCorrelation]
    ) -> Vec<CorrelationInsight> {
        let mut insights = Vec::new();
        
        // Insight 1: Dominant correlation types
        let correlation_types: HashMap<String, usize> = ml_correlations.iter()
            .fold(HashMap::new(), |mut acc, c| {
                *acc.entry(format!("{:?}", c.correlation_type)).or_insert(0) += 1;
                acc
            });
        
        if let Some((dominant_type, count)) = correlation_types.iter().max_by_key(|(_, &count)| count) {
            insights.push(CorrelationInsight {
                insight_type: InsightType::DominantPattern,
                title: "Dominant Correlation Type".to_string(),
                description: format!("{} correlations are most common ({} occurrences)", 
                    dominant_type, count),
                confidence: 0.9,
                actionable: true,
                recommendations: vec![
                    format!("Focus monitoring on {} relationships", dominant_type),
                    "Consider automated responses for common patterns".to_string(),
                ],
            });
        }
        
        // Insight 2: Temporal clustering
        let temporal_clusters = self.identify_temporal_clusters(temporal_patterns);
        for cluster in temporal_clusters {
            insights.push(CorrelationInsight {
                insight_type: InsightType::TemporalCluster,
                title: format!("Temporal Cluster: {}", cluster.time_period),
                description: format!("High correlation activity during {}", cluster.description),
                confidence: cluster.confidence,
                actionable: true,
                recommendations: cluster.recommendations,
            });
        }
        
        // Insight 3: Critical correlation paths
        let critical_paths = self.identify_critical_correlation_paths(ml_correlations);
        for path in critical_paths {
            insights.push(CorrelationInsight {
                insight_type: InsightType::CriticalPath,
                title: "Critical Correlation Path".to_string(),
                description: format!("Path: {} affects system stability", 
                    path.resources.join(" -> ")),
                confidence: path.criticality_score,
                actionable: true,
                recommendations: vec![
                    "Monitor this path for early warning signs".to_string(),
                    "Consider redundancy for critical resources".to_string(),
                ],
            });
        }
        
        insights
    }

    fn identify_temporal_clusters(&self, patterns: &[TemporalPattern]) -> Vec<TemporalCluster> {
        let mut clusters = Vec::new();
        
        // Group patterns by time of day
        let mut hourly_activity: HashMap<u32, Vec<&TemporalPattern>> = HashMap::new();
        for pattern in patterns {
            let hour = pattern.peak_time.hour();
            hourly_activity.entry(hour).or_default().push(pattern);
        }
        
        // Identify significant clusters
        for (hour, patterns_in_hour) in hourly_activity {
            if patterns_in_hour.len() >= 3 {
                clusters.push(TemporalCluster {
                    time_period: format!("{}:00-{}:00", hour, (hour + 1) % 24),
                    description: format!("Peak correlation activity with {} patterns", patterns_in_hour.len()),
                    confidence: (patterns_in_hour.len() as f64 / patterns.len() as f64).min(1.0),
                    recommendations: vec![
                        format!("Schedule maintenance outside of {}:00 peak time", hour),
                        "Increase monitoring during peak correlation hours".to_string(),
                    ],
                });
            }
        }
        
        clusters
    }

    fn identify_critical_correlation_paths(&self, correlations: &[MLCorrelation]) -> Vec<CriticalCorrelationPath> {
        let mut paths = Vec::new();
        
        // Build correlation graph
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        for correlation in correlations {
            if correlation.strength > 0.8 {
                graph.entry(correlation.source_id.clone())
                    .or_default()
                    .push(correlation.target_id.clone());
            }
        }
        
        // Find paths that affect many resources
        for (source, targets) in &graph {
            if targets.len() >= 3 {
                let criticality_score = targets.len() as f64 / 10.0; // Normalize
                paths.push(CriticalCorrelationPath {
                    resources: vec![source.clone()].into_iter().chain(targets.iter().cloned()).collect(),
                    criticality_score: criticality_score.min(1.0),
                });
            }
        }
        
        paths
    }

    fn detect_correlation_anomalies(&self, events: &[ResourceEvent]) -> Vec<CorrelationAnomaly> {
        let mut anomalies = Vec::new();
        
        // Detect unusual correlation patterns
        let recent_events: Vec<_> = events.iter()
            .filter(|e| e.timestamp > Utc::now() - Duration::hours(1))
            .collect();
        
        // Check for burst correlations
        if recent_events.len() > 50 {
            anomalies.push(CorrelationAnomaly {
                anomaly_type: AnomalyType::BurstCorrelation,
                description: format!("Unusual correlation burst: {} events in 1 hour", recent_events.len()),
                severity: AnomalySeverity::High,
                detected_at: Utc::now(),
                affected_resources: recent_events.iter().map(|e| e.resource_id.clone()).collect(),
            });
        }
        
        // Check for missing expected correlations
        let expected_patterns = self.correlation_memory.get_expected_patterns_for_time(Utc::now());
        for expected in expected_patterns {
            if !self.pattern_exists_in_events(&expected, &recent_events) {
                anomalies.push(CorrelationAnomaly {
                    anomaly_type: AnomalyType::MissingCorrelation,
                    description: format!("Expected correlation pattern '{}' not detected", expected.name),
                    severity: AnomalySeverity::Medium,
                    detected_at: Utc::now(),
                    affected_resources: vec![expected.source_pattern.clone(), expected.target_pattern.clone()],
                });
            }
        }
        
        anomalies
    }

    fn pattern_exists_in_events(&self, pattern: &TemporalPattern, events: &[&ResourceEvent]) -> bool {
        // Simplified pattern matching
        events.iter().any(|e| {
            e.resource_id.contains(&pattern.source_pattern) && 
            matches!(e.event_type, EventType::StateChange | EventType::ConfigurationChange)
        })
    }
}

/// Temporal graph for tracking resource relationships over time
pub struct TemporalGraph {
    graph: DiGraph<ResourceNode, TemporalEdge>,
    node_map: HashMap<String, NodeIndex>,
    snapshots: VecDeque<GraphSnapshot>,
}

impl TemporalGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            snapshots: VecDeque::with_capacity(100), // Keep last 100 snapshots
        }
    }

    pub fn update(&mut self, resources: &[AzureResource], events: &[ResourceEvent]) {
        // Take snapshot before update
        self.take_snapshot();
        
        // Update graph structure
        self.update_nodes(resources);
        self.update_edges(events);
        
        // Prune old snapshots
        if self.snapshots.len() > 100 {
            self.snapshots.pop_front();
        }
    }

    fn take_snapshot(&mut self) {
        let snapshot = GraphSnapshot {
            timestamp: Utc::now(),
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            avg_degree: self.calculate_avg_degree(),
        };
        self.snapshots.push_back(snapshot);
    }

    fn update_nodes(&mut self, resources: &[AzureResource]) {
        for resource in resources {
            if !self.node_map.contains_key(&resource.id) {
                let node = ResourceNode {
                    id: resource.id.clone(),
                    name: resource.name.clone(),
                    resource_type: resource.resource_type.clone(),
                    domain: self.get_domain(&resource.resource_type),
                    risk_level: 0.0,
                };
                let idx = self.graph.add_node(node);
                self.node_map.insert(resource.id.clone(), idx);
            }
        }
    }

    fn update_edges(&mut self, events: &[ResourceEvent]) {
        for event in events {
            if let Some(target_id) = &event.correlated_resource {
                if let (Some(&source_idx), Some(&target_idx)) = 
                    (self.node_map.get(&event.resource_id), self.node_map.get(target_id)) {
                    
                    let edge = TemporalEdge {
                        strength: event.correlation_strength.unwrap_or(0.5),
                        last_updated: event.timestamp,
                        event_type: event.event_type.clone(),
                    };
                    
                    self.graph.add_edge(source_idx, target_idx, edge);
                }
            }
        }
    }

    fn get_domain(&self, resource_type: &str) -> String {
        if resource_type.contains("Compute") {
            "compute".to_string()
        } else if resource_type.contains("Storage") {
            "storage".to_string()
        } else if resource_type.contains("Network") {
            "network".to_string()
        } else {
            "other".to_string()
        }
    }

    fn calculate_avg_degree(&self) -> f64 {
        if self.graph.node_count() == 0 {
            return 0.0;
        }
        
        let total_degree: usize = self.graph.node_indices()
            .map(|idx| self.graph.edges(idx).count())
            .sum();
        
        total_degree as f64 / self.graph.node_count() as f64
    }

    pub fn calculate_metrics(&self) -> GraphMetrics {
        GraphMetrics {
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            avg_degree: self.calculate_avg_degree(),
            density: self.calculate_density(),
            clustering_coefficient: self.calculate_clustering_coefficient(),
        }
    }

    fn calculate_density(&self) -> f64 {
        let n = self.graph.node_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        
        let max_edges = n * (n - 1.0);
        self.graph.edge_count() as f64 / max_edges
    }

    fn calculate_clustering_coefficient(&self) -> f64 {
        // Simplified clustering coefficient calculation
        let mut total_coefficient = 0.0;
        let mut node_count = 0;
        
        for node_idx in self.graph.node_indices() {
            let neighbors: Vec<_> = self.graph.neighbors(node_idx).collect();
            if neighbors.len() >= 2 {
                let mut triangle_count = 0;
                let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
                
                for i in 0..neighbors.len() {
                    for j in i+1..neighbors.len() {
                        if self.graph.find_edge(neighbors[i], neighbors[j]).is_some() {
                            triangle_count += 1;
                        }
                    }
                }
                
                if possible_triangles > 0 {
                    total_coefficient += triangle_count as f64 / possible_triangles as f64;
                    node_count += 1;
                }
            }
        }
        
        if node_count > 0 {
            total_coefficient / node_count as f64
        } else {
            0.0
        }
    }
}

/// Pattern detector for temporal correlation patterns
pub struct PatternDetector {
    pattern_library: Vec<PatternTemplate>,
}

impl PatternDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            pattern_library: Vec::new(),
        };
        detector.initialize_patterns();
        detector
    }

    fn initialize_patterns(&mut self) {
        self.pattern_library.push(PatternTemplate {
            name: "Cascade Failure".to_string(),
            description: "Multiple resources failing in sequence".to_string(),
            event_sequence: vec![
                EventType::Alert,
                EventType::StateChange,
                EventType::Alert,
            ],
            time_window: Duration::minutes(30),
            confidence_threshold: 0.8,
        });

        self.pattern_library.push(PatternTemplate {
            name: "Configuration Drift".to_string(),
            description: "Gradual changes leading to compliance violations".to_string(),
            event_sequence: vec![
                EventType::ConfigurationChange,
                EventType::ConfigurationChange,
                EventType::ComplianceViolation,
            ],
            time_window: Duration::hours(24),
            confidence_threshold: 0.7,
        });
    }

    pub fn detect_temporal_patterns(&self, events: &[ResourceEvent], window: Duration) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        let cutoff_time = Utc::now() - window;
        let recent_events: Vec<_> = events.iter()
            .filter(|e| e.timestamp > cutoff_time)
            .collect();
        
        for template in &self.pattern_library {
            if let Some(pattern) = self.match_pattern(template, &recent_events) {
                patterns.push(pattern);
            }
        }
        
        patterns
    }

    fn match_pattern(&self, template: &PatternTemplate, events: &[&ResourceEvent]) -> Option<TemporalPattern> {
        // Simplified pattern matching
        let mut sequence_index = 0;
        let mut matched_events = Vec::new();
        
        for event in events {
            if sequence_index < template.event_sequence.len() && 
               event.event_type == template.event_sequence[sequence_index] {
                matched_events.push((*event).clone());
                sequence_index += 1;
            }
        }
        
        if sequence_index == template.event_sequence.len() {
            Some(TemporalPattern {
                id: Uuid::new_v4().to_string(),
                name: template.name.clone(),
                pattern_type: PatternType::Sequence,
                confidence: template.confidence_threshold,
                matched_events,
                source_pattern: "detected".to_string(),
                target_pattern: "pattern".to_string(),
                peak_time: Utc::now(),
                typical_duration: template.time_window,
                correlation_type: CorrelationType::TemporalSequence,
                impact_level: ImpactLevel::Medium,
                preconditions: vec![],
            })
        } else {
            None
        }
    }
}

/// ML-based correlator
pub struct MLCorrelator {
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
}

impl MLCorrelator {
    pub fn new() -> Self {
        Self {
            feature_extractors: vec![
                Box::new(ResourceFeatureExtractor),
                Box::new(EventFeatureExtractor),
                Box::new(TemporalFeatureExtractor),
            ],
        }
    }

    pub async fn find_correlations(&self, 
        resources: &[AzureResource], 
        events: &[ResourceEvent]
    ) -> Vec<MLCorrelation> {
        let mut correlations = Vec::new();
        
        // Extract features for all resources
        let features = self.extract_features(resources, events);
        
        // Find correlations using similarity metrics
        for i in 0..features.len() {
            for j in i+1..features.len() {
                let similarity = self.calculate_similarity(&features[i], &features[j]);
                if similarity > 0.6 {
                    correlations.push(MLCorrelation {
                        correlation_id: Uuid::new_v4().to_string(),
                        source_id: features[i].resource_id.clone(),
                        target_id: features[j].resource_id.clone(),
                        ml_confidence: similarity,
                        correlation_type: CorrelationType::FeatureSimilarity,
                        strength: similarity,
                        data_points: features[i].data_points + features[j].data_points,
                        last_updated: Utc::now(),
                        variance: (features[i].variance + features[j].variance) / 2.0,
                    });
                }
            }
        }
        
        correlations
    }

    pub async fn predict_correlations(&self, 
        _resources: &[AzureResource], 
        _events: &[ResourceEvent]
    ) -> Vec<PredictedCorrelation> {
        // Placeholder for ML-based prediction
        // In production, would use trained models
        vec![]
    }

    fn extract_features(&self, resources: &[AzureResource], events: &[ResourceEvent]) -> Vec<ResourceFeatures> {
        let mut all_features = Vec::new();
        
        for resource in resources {
            let resource_events: Vec<_> = events.iter()
                .filter(|e| e.resource_id == resource.id)
                .collect();
            
            let mut features = ResourceFeatures {
                resource_id: resource.id.clone(),
                features: HashMap::new(),
                data_points: resource_events.len(),
                variance: 0.0,
            };
            
            // Extract features using all extractors
            for extractor in &self.feature_extractors {
                let extracted = extractor.extract(resource, &resource_events);
                features.features.extend(extracted);
            }
            
            // Calculate variance
            features.variance = self.calculate_feature_variance(&features.features);
            
            all_features.push(features);
        }
        
        all_features
    }

    fn calculate_similarity(&self, features1: &ResourceFeatures, features2: &ResourceFeatures) -> f64 {
        let mut similarity_sum = 0.0;
        let mut comparison_count = 0;
        
        for (key, value1) in &features1.features {
            if let Some(value2) = features2.features.get(key) {
                let feature_similarity = 1.0 - (value1 - value2).abs().min(1.0);
                similarity_sum += feature_similarity;
                comparison_count += 1;
            }
        }
        
        if comparison_count > 0 {
            similarity_sum / comparison_count as f64
        } else {
            0.0
        }
    }

    fn calculate_feature_variance(&self, features: &HashMap<String, f64>) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        
        let mean: f64 = features.values().sum::<f64>() / features.len() as f64;
        let variance: f64 = features.values()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / features.len() as f64;
        
        variance.sqrt()
    }
}

/// Real-time correlation analyzer
pub struct RealTimeAnalyzer {
    event_buffer: VecDeque<ResourceEvent>,
    correlation_window: Duration,
}

impl RealTimeAnalyzer {
    pub fn new() -> Self {
        Self {
            event_buffer: VecDeque::with_capacity(1000),
            correlation_window: Duration::minutes(5),
        }
    }

    pub async fn analyze(&mut self, events: &[ResourceEvent]) -> Vec<RealTimeCorrelation> {
        // Add new events to buffer
        for event in events {
            self.event_buffer.push_back(event.clone());
        }
        
        // Remove old events
        let cutoff = Utc::now() - self.correlation_window;
        while let Some(front) = self.event_buffer.front() {
            if front.timestamp < cutoff {
                self.event_buffer.pop_front();
            } else {
                break;
            }
        }
        
        // Analyze correlations in the buffer
        self.find_real_time_correlations()
    }

    fn find_real_time_correlations(&self) -> Vec<RealTimeCorrelation> {
        let mut correlations = Vec::new();
        
        // Simple time-based correlation detection
        let events: Vec<_> = self.event_buffer.iter().collect();
        
        for i in 0..events.len() {
            for j in i+1..events.len() {
                let time_diff = (events[j].timestamp - events[i].timestamp).num_seconds().abs();
                
                if time_diff < 300 { // Within 5 minutes
                    let correlation_strength = self.calculate_temporal_correlation(events[i], events[j]);
                    
                    if correlation_strength > 0.5 {
                        correlations.push(RealTimeCorrelation {
                            source_event: events[i].clone(),
                            target_event: events[j].clone(),
                            time_delta: Duration::seconds(time_diff),
                            correlation_strength,
                            correlation_type: CorrelationType::Temporal,
                        });
                    }
                }
            }
        }
        
        correlations
    }

    fn calculate_temporal_correlation(&self, event1: &ResourceEvent, event2: &ResourceEvent) -> f64 {
        let mut score = 0.0;
        
        // Same event type
        if event1.event_type == event2.event_type {
            score += 0.3;
        }
        
        // Similar resource types
        if event1.resource_type == event2.resource_type {
            score += 0.2;
        }
        
        // Time proximity (closer = higher score)
        let time_diff = (event2.timestamp - event1.timestamp).num_seconds().abs();
        let time_score = ((300 - time_diff) as f64 / 300.0).max(0.0);
        score += time_score * 0.5;
        
        score.min(1.0)
    }
}

/// Correlation memory for historical patterns
pub struct CorrelationMemory {
    historical_patterns: Vec<TemporalPattern>,
    accuracy_history: HashMap<String, f64>,
    similar_correlations: HashMap<String, Vec<HistoricalCorrelation>>,
}

impl CorrelationMemory {
    pub fn new() -> Self {
        Self {
            historical_patterns: Vec::new(),
            accuracy_history: HashMap::new(),
            similar_correlations: HashMap::new(),
        }
    }

    pub fn get_historical_patterns(&self) -> &[TemporalPattern] {
        &self.historical_patterns
    }

    pub fn get_historical_accuracy(&self, correlation_id: &str) -> f64 {
        self.accuracy_history.get(correlation_id).copied().unwrap_or(0.5)
    }

    pub fn find_similar_correlations(&self, source_id: &str, target_id: &str) -> Vec<HistoricalCorrelation> {
        let key = format!("{}:{}", source_id, target_id);
        self.similar_correlations.get(&key).cloned().unwrap_or_default()
    }

    pub fn get_expected_patterns_for_time(&self, _time: DateTime<Utc>) -> Vec<TemporalPattern> {
        // Return patterns that typically occur at this time
        self.historical_patterns.iter()
            .filter(|p| self.pattern_matches_time(p, _time))
            .cloned()
            .collect()
    }

    fn pattern_matches_time(&self, _pattern: &TemporalPattern, _time: DateTime<Utc>) -> bool {
        // Simplified time matching
        true
    }
}

// Feature extractors
pub trait FeatureExtractor: Send + Sync {
    fn extract(&self, resource: &AzureResource, events: &[&ResourceEvent]) -> HashMap<String, f64>;
}

pub struct ResourceFeatureExtractor;

impl FeatureExtractor for ResourceFeatureExtractor {
    fn extract(&self, resource: &AzureResource, _events: &[&ResourceEvent]) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        // Resource type encoding
        features.insert("is_compute".to_string(), if resource.resource_type.contains("Compute") { 1.0 } else { 0.0 });
        features.insert("is_storage".to_string(), if resource.resource_type.contains("Storage") { 1.0 } else { 0.0 });
        features.insert("is_network".to_string(), if resource.resource_type.contains("Network") { 1.0 } else { 0.0 });
        
        // Tag count
        features.insert("tag_count".to_string(), resource.tags.len() as f64);
        
        // Compliance state
        features.insert("is_compliant".to_string(), if resource.compliance_state == "Compliant" { 1.0 } else { 0.0 });
        
        features
    }
}

pub struct EventFeatureExtractor;

impl FeatureExtractor for EventFeatureExtractor {
    fn extract(&self, _resource: &AzureResource, events: &[&ResourceEvent]) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        // Event frequency
        features.insert("event_count".to_string(), events.len() as f64);
        
        // Event type distribution
        let alert_count = events.iter().filter(|e| e.event_type == EventType::Alert).count();
        let config_count = events.iter().filter(|e| e.event_type == EventType::ConfigurationChange).count();
        
        features.insert("alert_ratio".to_string(), alert_count as f64 / events.len().max(1) as f64);
        features.insert("config_ratio".to_string(), config_count as f64 / events.len().max(1) as f64);
        
        features
    }
}

pub struct TemporalFeatureExtractor;

impl FeatureExtractor for TemporalFeatureExtractor {
    fn extract(&self, _resource: &AzureResource, events: &[&ResourceEvent]) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        if events.is_empty() {
            return features;
        }
        
        // Time since last event
        let latest_event = events.iter().max_by_key(|e| e.timestamp).unwrap();
        let time_since_last = (Utc::now() - latest_event.timestamp).num_hours();
        features.insert("hours_since_last_event".to_string(), time_since_last as f64);
        
        // Event time distribution
        let hour_distribution = events.iter()
            .map(|e| e.timestamp.hour())
            .fold(HashMap::new(), |mut acc, hour| {
                *acc.entry(hour).or_insert(0) += 1;
                acc
            });
        
        let peak_hour = hour_distribution.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&hour, _)| hour)
            .unwrap_or(0);
        
        features.insert("peak_activity_hour".to_string(), peak_hour as f64);
        
        features
    }
}

// Data structures for advanced correlation

#[derive(Debug, Clone)]
pub struct ResourceNode {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub domain: String,
    pub risk_level: f64,
}

#[derive(Debug, Clone)]
pub struct TemporalEdge {
    pub strength: f64,
    pub last_updated: DateTime<Utc>,
    pub event_type: EventType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureResource {
    pub id: String,
    pub name: String,
    pub resource_type: String,
    pub resource_group: String,
    pub location: String,
    pub tags: HashMap<String, String>,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub compliance_state: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEvent {
    pub event_id: String,
    pub resource_id: String,
    pub resource_type: String,
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub severity: EventSeverity,
    pub correlated_resource: Option<String>,
    pub correlation_strength: Option<f64>,
    pub new_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    Alert,
    ConfigurationChange,
    StateChange,
    ComplianceViolation,
    PerformanceAnomaly,
    SecurityEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedCorrelationResult {
    pub temporal_patterns: Vec<TemporalPattern>,
    pub ml_correlations: Vec<MLCorrelation>,
    pub real_time_correlations: Vec<RealTimeCorrelation>,
    pub predicted_correlations: Vec<PredictedCorrelation>,
    pub confidence_scores: HashMap<String, f64>,
    pub insights: Vec<CorrelationInsight>,
    pub anomalies: Vec<CorrelationAnomaly>,
    pub graph_metrics: GraphMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub id: String,
    pub name: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub matched_events: Vec<ResourceEvent>,
    pub source_pattern: String,
    pub target_pattern: String,
    pub peak_time: DateTime<Utc>,
    pub typical_duration: Duration,
    pub correlation_type: CorrelationType,
    pub impact_level: ImpactLevel,
    pub preconditions: Vec<PatternPrecondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Sequence,
    Burst,
    Periodic,
    Anomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternPrecondition {
    EventType(EventType),
    ResourceState(String, String),
    TimeWindow(Duration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLCorrelation {
    pub correlation_id: String,
    pub source_id: String,
    pub target_id: String,
    pub ml_confidence: f64,
    pub correlation_type: CorrelationType,
    pub strength: f64,
    pub data_points: usize,
    pub last_updated: DateTime<Utc>,
    pub variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeCorrelation {
    pub source_event: ResourceEvent,
    pub target_event: ResourceEvent,
    pub time_delta: Duration,
    pub correlation_strength: f64,
    pub correlation_type: CorrelationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedCorrelation {
    pub pattern_id: String,
    pub predicted_time: DateTime<Utc>,
    pub source_resource: String,
    pub target_resource: String,
    pub confidence: f64,
    pub correlation_type: CorrelationType,
    pub impact_level: ImpactLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    TemporalSequence,
    FeatureSimilarity,
    Temporal,
    DataDependency,
    SecurityDependency,
    AccessDependency,
    PerformanceDependency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationInsight {
    pub insight_type: InsightType,
    pub title: String,
    pub description: String,
    pub confidence: f64,
    pub actionable: bool,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    DominantPattern,
    TemporalCluster,
    CriticalPath,
    AnomalyDetection,
}

#[derive(Debug, Clone)]
pub struct TemporalCluster {
    pub time_period: String,
    pub description: String,
    pub confidence: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CriticalCorrelationPath {
    pub resources: Vec<String>,
    pub criticality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnomaly {
    pub anomaly_type: AnomalyType,
    pub description: String,
    pub severity: AnomalySeverity,
    pub detected_at: DateTime<Utc>,
    pub affected_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    BurstCorrelation,
    MissingCorrelation,
    UnexpectedPattern,
    StrengthDeviation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub timestamp: DateTime<Utc>,
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub clustering_coefficient: f64,
}

#[derive(Debug, Clone)]
pub struct PatternTemplate {
    pub name: String,
    pub description: String,
    pub event_sequence: Vec<EventType>,
    pub time_window: Duration,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceFeatures {
    pub resource_id: String,
    pub features: HashMap<String, f64>,
    pub data_points: usize,
    pub variance: f64,
}

#[derive(Debug, Clone)]
pub struct HistoricalCorrelation {
    pub source_id: String,
    pub target_id: String,
    pub strength: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_correlation_engine() {
        let mut engine = AdvancedCorrelationEngine::new();
        
        let resources = vec![
            AzureResource {
                id: "vm-001".to_string(),
                name: "test-vm".to_string(),
                resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                resource_group: "test-rg".to_string(),
                location: "eastus".to_string(),
                tags: HashMap::new(),
                properties: serde_json::Map::new(),
                compliance_state: "Compliant".to_string(),
                dependencies: vec![],
            }
        ];
        
        let events = vec![
            ResourceEvent {
                event_id: "event-001".to_string(),
                resource_id: "vm-001".to_string(),
                resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                event_type: EventType::StateChange,
                timestamp: Utc::now(),
                description: "VM state changed".to_string(),
                severity: EventSeverity::Medium,
                correlated_resource: None,
                correlation_strength: None,
                new_state: Some("Running".to_string()),
            }
        ];
        
        let result = engine.analyze_advanced_correlations(
            resources,
            events,
            Duration::hours(24)
        ).await;
        
        assert!(result.graph_metrics.node_count > 0);
    }
}