// Correlation Module
// Cross-domain correlation and impact analysis

pub mod cross_domain_engine;
pub mod resource_mapper;
pub mod impact_analyzer;

pub use cross_domain_engine::{CrossDomainEngine, CorrelationAnalysis, AzureResource};
pub use resource_mapper::{ResourceMapper, ResourceMap, DependencyMap};
pub use impact_analyzer::{ImpactAnalyzer, ImpactAssessment, CascadeEffect};