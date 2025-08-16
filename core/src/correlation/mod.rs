// Correlation Module
// Cross-domain correlation and impact analysis

pub mod cross_domain_engine;
pub mod resource_mapper;
pub mod impact_analyzer;

// Day 10: Advanced correlation capabilities
pub mod advanced_correlation_engine;

// Day 11: Predictive impact analysis
pub mod predictive_impact_analyzer;

// Day 12: Smart dependency mapping
pub mod smart_dependency_mapper;

pub use cross_domain_engine::{CrossDomainEngine, CorrelationAnalysis, AzureResource};
pub use resource_mapper::{ResourceMapper, ResourceMap, DependencyMap};
pub use impact_analyzer::{ImpactAnalyzer, ImpactAssessment, CascadeEffect};
pub use advanced_correlation_engine::{AdvancedCorrelationEngine, AdvancedCorrelationResult};
pub use predictive_impact_analyzer::{PredictiveImpactAnalyzer, PredictiveImpactResult, WhatIfAnalysisResult};
pub use smart_dependency_mapper::{SmartDependencyMapper, SmartDependencyMap, SmartDependencyInfo};