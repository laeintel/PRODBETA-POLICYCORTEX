// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Simulated data provider for development and testing
pub struct SimulatedDataProvider;

impl SimulatedDataProvider {
    /// Get simulated policies
    pub fn get_policies() -> Vec<Policy> {
        vec![
            Policy {
                id: "pol-001".to_string(),
                name: "Require HTTPS for Storage".to_string(),
                description: "Ensures all storage accounts require HTTPS connections".to_string(),
                category: "Security".to_string(),
                severity: "High".to_string(),
                cloud_provider: "Azure".to_string(),
                resource_types: vec!["Microsoft.Storage/storageAccounts".to_string()],
                policy_type: "BuiltIn".to_string(),
                enforcement_mode: "Default".to_string(),
                compliance_percentage: 87.5,
                affected_resources: 24,
                last_evaluated: Utc::now(),
                metadata: HashMap::from([
                    ("version".to_string(), "1.0.0".to_string()),
                    ("author".to_string(), "Platform Team".to_string()),
                ]),
            },
            Policy {
                id: "pol-002".to_string(),
                name: "VM Backup Required".to_string(),
                description: "All production VMs must have backup enabled".to_string(),
                category: "Reliability".to_string(),
                severity: "Medium".to_string(),
                cloud_provider: "Azure".to_string(),
                resource_types: vec!["Microsoft.Compute/virtualMachines".to_string()],
                policy_type: "Custom".to_string(),
                enforcement_mode: "Default".to_string(),
                compliance_percentage: 92.3,
                affected_resources: 156,
                last_evaluated: Utc::now(),
                metadata: HashMap::new(),
            },
            Policy {
                id: "pol-003".to_string(),
                name: "Tag Compliance".to_string(),
                description: "Resources must have required tags: Environment, Owner, CostCenter"
                    .to_string(),
                category: "Governance".to_string(),
                severity: "Low".to_string(),
                cloud_provider: "Azure".to_string(),
                resource_types: vec!["*".to_string()],
                policy_type: "Custom".to_string(),
                enforcement_mode: "Audit".to_string(),
                compliance_percentage: 76.8,
                affected_resources: 512,
                last_evaluated: Utc::now(),
                metadata: HashMap::new(),
            },
            Policy {
                id: "pol-004".to_string(),
                name: "Network Security Groups Required".to_string(),
                description: "All subnets must have an NSG attached".to_string(),
                category: "Security".to_string(),
                severity: "High".to_string(),
                cloud_provider: "Azure".to_string(),
                resource_types: vec!["Microsoft.Network/virtualNetworks/subnets".to_string()],
                policy_type: "BuiltIn".to_string(),
                enforcement_mode: "Deny".to_string(),
                compliance_percentage: 95.2,
                affected_resources: 42,
                last_evaluated: Utc::now(),
                metadata: HashMap::new(),
            },
            Policy {
                id: "pol-005".to_string(),
                name: "Cost Center Tagging".to_string(),
                description: "All resources must have a valid cost center tag".to_string(),
                category: "FinOps".to_string(),
                severity: "Medium".to_string(),
                cloud_provider: "Azure".to_string(),
                resource_types: vec!["*".to_string()],
                policy_type: "Custom".to_string(),
                enforcement_mode: "Audit".to_string(),
                compliance_percentage: 68.9,
                affected_resources: 892,
                last_evaluated: Utc::now(),
                metadata: HashMap::new(),
            },
        ]
    }

    /// Get simulated resources
    pub fn get_resources() -> Vec<Resource> {
        vec![
            Resource {
                id: "/subscriptions/sub-123/resourceGroups/prod-rg/providers/Microsoft.Compute/virtualMachines/web-vm-01".to_string(),
                name: "web-vm-01".to_string(),
                type_name: "Microsoft.Compute/virtualMachines".to_string(),
                location: "eastus".to_string(),
                resource_group: "prod-rg".to_string(),
                subscription_id: "sub-123".to_string(),
                tags: HashMap::from([
                    ("Environment".to_string(), "Production".to_string()),
                    ("Owner".to_string(), "WebTeam".to_string()),
                    ("CostCenter".to_string(), "CC-100".to_string()),
                ]),
                compliance_status: "Compliant".to_string(),
                created_time: Utc::now() - chrono::Duration::days(30),
                properties: HashMap::from([
                    ("vmSize".to_string(), "Standard_D4s_v3".to_string()),
                    ("osType".to_string(), "Linux".to_string()),
                ]),
            },
            Resource {
                id: "/subscriptions/sub-123/resourceGroups/prod-rg/providers/Microsoft.Storage/storageAccounts/prodstorage01".to_string(),
                name: "prodstorage01".to_string(),
                type_name: "Microsoft.Storage/storageAccounts".to_string(),
                location: "eastus".to_string(),
                resource_group: "prod-rg".to_string(),
                subscription_id: "sub-123".to_string(),
                tags: HashMap::from([
                    ("Environment".to_string(), "Production".to_string()),
                    ("Owner".to_string(), "DataTeam".to_string()),
                ]),
                compliance_status: "NonCompliant".to_string(),
                created_time: Utc::now() - chrono::Duration::days(90),
                properties: HashMap::from([
                    ("httpsOnly".to_string(), "false".to_string()),
                    ("encryption".to_string(), "enabled".to_string()),
                ]),
            },
            Resource {
                id: "/subscriptions/sub-123/resourceGroups/dev-rg/providers/Microsoft.Sql/servers/devsql01".to_string(),
                name: "devsql01".to_string(),
                type_name: "Microsoft.Sql/servers".to_string(),
                location: "westus2".to_string(),
                resource_group: "dev-rg".to_string(),
                subscription_id: "sub-123".to_string(),
                tags: HashMap::from([
                    ("Environment".to_string(), "Development".to_string()),
                ]),
                compliance_status: "NonCompliant".to_string(),
                created_time: Utc::now() - chrono::Duration::days(45),
                properties: HashMap::new(),
            },
        ]
    }

    /// Get simulated RBAC data
    pub fn get_rbac_data() -> RbacData {
        RbacData {
            users: vec![
                User {
                    id: "user-001".to_string(),
                    display_name: "Alice Admin".to_string(),
                    email: "alice@company.com".to_string(),
                    roles: vec!["Global Administrator".to_string()],
                    last_login: Some(Utc::now() - chrono::Duration::hours(2)),
                },
                User {
                    id: "user-002".to_string(),
                    display_name: "Bob Developer".to_string(),
                    email: "bob@company.com".to_string(),
                    roles: vec!["Contributor".to_string()],
                    last_login: Some(Utc::now() - chrono::Duration::days(1)),
                },
            ],
            roles: vec![
                Role {
                    id: "role-001".to_string(),
                    name: "Global Administrator".to_string(),
                    description: "Full access to all resources".to_string(),
                    permissions: vec!["*".to_string()],
                    assigned_count: 2,
                },
                Role {
                    id: "role-002".to_string(),
                    name: "Contributor".to_string(),
                    description: "Can create and manage resources".to_string(),
                    permissions: vec!["read".to_string(), "write".to_string()],
                    assigned_count: 15,
                },
            ],
            total_users: 150,
            total_roles: 25,
            violations_detected: 3,
        }
    }

    /// Get simulated cost data
    pub fn get_cost_data() -> CostData {
        CostData {
            current_month_spend: 125000.50,
            projected_month_spend: 142000.00,
            last_month_spend: 118500.25,
            year_to_date: 1450000.00,
            top_services: vec![
                ServiceCost {
                    service: "Virtual Machines".to_string(),
                    cost: 45000.00,
                    percentage: 36.0,
                },
                ServiceCost {
                    service: "Storage".to_string(),
                    cost: 28000.00,
                    percentage: 22.4,
                },
                ServiceCost {
                    service: "Databases".to_string(),
                    cost: 22000.00,
                    percentage: 17.6,
                },
            ],
            savings_opportunities: vec![
                SavingOpportunity {
                    description: "Right-size underutilized VMs".to_string(),
                    potential_savings: 8500.00,
                    effort: "Low".to_string(),
                },
                SavingOpportunity {
                    description: "Delete unattached disks".to_string(),
                    potential_savings: 2200.00,
                    effort: "Low".to_string(),
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub severity: String,
    pub cloud_provider: String,
    pub resource_types: Vec<String>,
    pub policy_type: String,
    pub enforcement_mode: String,
    pub compliance_percentage: f64,
    pub affected_resources: u32,
    pub last_evaluated: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    pub name: String,
    pub type_name: String,
    pub location: String,
    pub resource_group: String,
    pub subscription_id: String,
    pub tags: HashMap<String, String>,
    pub compliance_status: String,
    pub created_time: DateTime<Utc>,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacData {
    pub users: Vec<User>,
    pub roles: Vec<Role>,
    pub total_users: u32,
    pub total_roles: u32,
    pub violations_detected: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub display_name: String,
    pub email: String,
    pub roles: Vec<String>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: Vec<String>,
    pub assigned_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostData {
    pub current_month_spend: f64,
    pub projected_month_spend: f64,
    pub last_month_spend: f64,
    pub year_to_date: f64,
    pub top_services: Vec<ServiceCost>,
    pub savings_opportunities: Vec<SavingOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCost {
    pub service: String,
    pub cost: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingOpportunity {
    pub description: String,
    pub potential_savings: f64,
    pub effort: String,
}
