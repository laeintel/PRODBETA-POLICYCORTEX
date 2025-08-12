pub mod aws_collector;
pub mod gcp_collector;

pub use aws_collector::{
    AuditLog, AwsCollector, CloudCollector, CloudPolicy, CloudProvider, CloudResource,
    ComplianceStatus, ResourceType,
};
pub use gcp_collector::GcpCollector;
