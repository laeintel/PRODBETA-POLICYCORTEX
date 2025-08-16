// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

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

// Knowledge graph driver
pub mod graph_driver;

pub use cross_domain_engine::{CrossDomainEngine, CorrelationAnalysis, AzureResource};
pub use resource_mapper::{ResourceMapper, ResourceMap, DependencyMap};
pub use impact_analyzer::{ImpactAnalyzer, ImpactAssessment, CascadeEffect};
pub use advanced_correlation_engine::{AdvancedCorrelationEngine, AdvancedCorrelationResult};
pub use predictive_impact_analyzer::{PredictiveImpactAnalyzer, PredictiveImpactResult, WhatIfAnalysisResult};
pub use smart_dependency_mapper::{SmartDependencyMapper, SmartDependencyMap, SmartDependencyInfo};
pub use graph_driver::{GraphDriver, GraphNode, GraphEdge, WhatIfScenario, SimulationResult};