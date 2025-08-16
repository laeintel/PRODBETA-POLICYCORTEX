// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

use super::{ResourceCategory, AzureResource, ResourceStatus, ResourceHealth, HealthStatus, QuickAction, ActionType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDefinition {
    pub resource_type: String,
    pub display_name: String,
    pub category: ResourceCategory,
    pub icon: String,
    pub description: String,
    pub quick_actions: Vec<QuickActionTemplate>,
    pub required_permissions: Vec<String>,
    pub cost_factors: Vec<String>,
    pub compliance_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickActionTemplate {
    pub action_type: ActionType,
    pub label: String,
    pub icon: String,
    pub confirmation_required: bool,
}

pub struct ResourceCatalog {
    definitions: HashMap<String, ResourceDefinition>,
}

impl ResourceCatalog {
    pub fn new() -> Self {
        let mut catalog = ResourceCatalog {
            definitions: HashMap::new(),
        };
        catalog.initialize_resources();
        catalog
    }

    fn initialize_resources(&mut self) {
        self.init_policy_resources();
        self.init_cost_resources();
        self.init_security_resources();
        self.init_compute_storage_resources();
        self.init_network_resources();
    }

    fn init_policy_resources(&mut self) {
        let policy_resources = vec![
            ResourceDefinition {
                resource_type: "Microsoft.Authorization/policyDefinitions".to_string(),
                display_name: "Azure Policy".to_string(),
                category: ResourceCategory::Policy,
                icon: "shield-check".to_string(),
                description: "Define and enforce organizational standards".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Policy".to_string(),
                        icon: "eye".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Edit Rules".to_string(),
                        icon: "edit".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Policy.Read".to_string(), "Policy.Write".to_string()],
                cost_factors: vec![],
                compliance_policies: vec!["ISO27001".to_string(), "SOC2".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Blueprint/blueprints".to_string(),
                display_name: "Azure Blueprints".to_string(),
                category: ResourceCategory::Policy,
                icon: "template".to_string(),
                description: "Repeatable environment deployment templates".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Blueprint".to_string(),
                        icon: "eye".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Blueprint.Read".to_string()],
                cost_factors: vec![],
                compliance_policies: vec!["CIS".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Resources/templateSpecs".to_string(),
                display_name: "ARM Templates".to_string(),
                category: ResourceCategory::Policy,
                icon: "code".to_string(),
                description: "Infrastructure as Code templates".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Template".to_string(),
                        icon: "code".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Resources.Read".to_string()],
                cost_factors: vec![],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Advisor/recommendations".to_string(),
                display_name: "Azure Advisor".to_string(),
                category: ResourceCategory::Policy,
                icon: "lightbulb".to_string(),
                description: "Best practices and optimization recommendations".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Recommendations".to_string(),
                        icon: "list".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Advisor.Read".to_string()],
                cost_factors: vec![],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.ServiceHealth/events".to_string(),
                display_name: "Service Health".to_string(),
                category: ResourceCategory::Policy,
                icon: "heart-pulse".to_string(),
                description: "Azure service availability and incidents".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Status".to_string(),
                        icon: "activity".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["ServiceHealth.Read".to_string()],
                cost_factors: vec![],
                compliance_policies: vec![],
            },
        ];

        for resource in policy_resources {
            self.definitions.insert(resource.resource_type.clone(), resource);
        }
    }

    fn init_cost_resources(&mut self) {
        let cost_resources = vec![
            ResourceDefinition {
                resource_type: "Microsoft.CostManagement/views".to_string(),
                display_name: "Cost Management".to_string(),
                category: ResourceCategory::CostManagement,
                icon: "dollar-sign".to_string(),
                description: "Monitor and optimize cloud spending".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Costs".to_string(),
                        icon: "chart-line".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Optimize,
                        label: "Optimize".to_string(),
                        icon: "trending-down".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Cost.Read".to_string()],
                cost_factors: vec!["usage".to_string(), "reservations".to_string()],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Compute/virtualMachines/spotVMs".to_string(),
                display_name: "Spot VMs".to_string(),
                category: ResourceCategory::CostManagement,
                icon: "zap".to_string(),
                description: "Cost-effective compute with flexible availability".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::Start,
                        label: "Start".to_string(),
                        icon: "play".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Stop,
                        label: "Stop".to_string(),
                        icon: "square".to_string(),
                        confirmation_required: true,
                    },
                ],
                required_permissions: vec!["VM.Manage".to_string()],
                cost_factors: vec!["spot_pricing".to_string(), "eviction_rate".to_string()],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Capacity/reservations".to_string(),
                display_name: "Reserved Instances".to_string(),
                category: ResourceCategory::CostManagement,
                icon: "bookmark".to_string(),
                description: "Pre-purchased compute capacity for savings".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Utilization".to_string(),
                        icon: "percent".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Reservations.Read".to_string()],
                cost_factors: vec!["utilization".to_string(), "term_length".to_string()],
                compliance_policies: vec![],
            },
        ];

        for resource in cost_resources {
            self.definitions.insert(resource.resource_type.clone(), resource);
        }
    }

    fn init_security_resources(&mut self) {
        let security_resources = vec![
            ResourceDefinition {
                resource_type: "Microsoft.AAD/domainServices".to_string(),
                display_name: "Microsoft Entra ID".to_string(),
                category: ResourceCategory::SecurityControls,
                icon: "users".to_string(),
                description: "Identity and access management".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Users".to_string(),
                        icon: "users".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Manage Access".to_string(),
                        icon: "key".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Directory.Read".to_string()],
                cost_factors: vec!["user_count".to_string(), "mfa_licenses".to_string()],
                compliance_policies: vec!["GDPR".to_string(), "HIPAA".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Security/defenderForCloud".to_string(),
                display_name: "Defender for Cloud".to_string(),
                category: ResourceCategory::SecurityControls,
                icon: "shield".to_string(),
                description: "Cloud security posture management".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "Security Score".to_string(),
                        icon: "shield-check".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::RunDiagnostics,
                        label: "Run Scan".to_string(),
                        icon: "search".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Security.Read".to_string()],
                cost_factors: vec!["protected_resources".to_string()],
                compliance_policies: vec!["PCI-DSS".to_string(), "ISO27001".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.KeyVault/vaults".to_string(),
                display_name: "Key Vault".to_string(),
                category: ResourceCategory::SecurityControls,
                icon: "lock".to_string(),
                description: "Secure secrets and key management".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Secrets".to_string(),
                        icon: "key".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Access Policies".to_string(),
                        icon: "settings".to_string(),
                        confirmation_required: true,
                    },
                ],
                required_permissions: vec!["KeyVault.Read".to_string()],
                cost_factors: vec!["operations".to_string(), "hsm_keys".to_string()],
                compliance_policies: vec!["FIPS140-2".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Sentinel/workspaces".to_string(),
                display_name: "Microsoft Sentinel".to_string(),
                category: ResourceCategory::SecurityControls,
                icon: "eye".to_string(),
                description: "Security information and event management".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Incidents".to_string(),
                        icon: "alert-triangle".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::RunDiagnostics,
                        label: "Hunt Threats".to_string(),
                        icon: "target".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Sentinel.Read".to_string()],
                cost_factors: vec!["data_ingestion".to_string(), "retention".to_string()],
                compliance_policies: vec!["SOC2".to_string()],
            },
        ];

        for resource in security_resources {
            self.definitions.insert(resource.resource_type.clone(), resource);
        }
    }

    fn init_compute_storage_resources(&mut self) {
        let compute_storage_resources = vec![
            ResourceDefinition {
                resource_type: "Microsoft.Compute/virtualMachines".to_string(),
                display_name: "Virtual Machines".to_string(),
                category: ResourceCategory::ComputeStorage,
                icon: "server".to_string(),
                description: "Scalable compute infrastructure".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::Start,
                        label: "Start".to_string(),
                        icon: "play".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Stop,
                        label: "Stop".to_string(),
                        icon: "square".to_string(),
                        confirmation_required: true,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Restart,
                        label: "Restart".to_string(),
                        icon: "refresh-cw".to_string(),
                        confirmation_required: true,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Scale,
                        label: "Resize".to_string(),
                        icon: "maximize".to_string(),
                        confirmation_required: true,
                    },
                ],
                required_permissions: vec!["VM.Manage".to_string()],
                cost_factors: vec!["size".to_string(), "disk".to_string(), "network".to_string()],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Web/sites".to_string(),
                display_name: "App Service".to_string(),
                category: ResourceCategory::ComputeStorage,
                icon: "globe".to_string(),
                description: "Managed web application hosting".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::Start,
                        label: "Start".to_string(),
                        icon: "play".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Stop,
                        label: "Stop".to_string(),
                        icon: "square".to_string(),
                        confirmation_required: true,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Metrics".to_string(),
                        icon: "activity".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Web.Manage".to_string()],
                cost_factors: vec!["plan_tier".to_string(), "instances".to_string()],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.ContainerService/managedClusters".to_string(),
                display_name: "AKS Clusters".to_string(),
                category: ResourceCategory::ComputeStorage,
                icon: "box".to_string(),
                description: "Managed Kubernetes orchestration".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::Scale,
                        label: "Scale Nodes".to_string(),
                        icon: "git-branch".to_string(),
                        confirmation_required: true,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Workloads".to_string(),
                        icon: "layers".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["AKS.Manage".to_string()],
                cost_factors: vec!["node_count".to_string(), "node_size".to_string()],
                compliance_policies: vec!["CIS-Kubernetes".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Storage/storageAccounts".to_string(),
                display_name: "Storage Accounts".to_string(),
                category: ResourceCategory::ComputeStorage,
                icon: "database".to_string(),
                description: "Scalable cloud storage".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Containers".to_string(),
                        icon: "folder".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Access Keys".to_string(),
                        icon: "key".to_string(),
                        confirmation_required: true,
                    },
                ],
                required_permissions: vec!["Storage.Read".to_string()],
                cost_factors: vec!["capacity".to_string(), "transactions".to_string(), "redundancy".to_string()],
                compliance_policies: vec!["GDPR".to_string()],
            },
        ];

        for resource in compute_storage_resources {
            self.definitions.insert(resource.resource_type.clone(), resource);
        }
    }

    fn init_network_resources(&mut self) {
        let network_resources = vec![
            ResourceDefinition {
                resource_type: "Microsoft.Network/virtualNetworks".to_string(),
                display_name: "Virtual Networks".to_string(),
                category: ResourceCategory::NetworksFirewalls,
                icon: "git-merge".to_string(),
                description: "Private network infrastructure".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Subnets".to_string(),
                        icon: "git-branch".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Peering".to_string(),
                        icon: "link".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["Network.Read".to_string()],
                cost_factors: vec!["peering".to_string(), "gateway".to_string()],
                compliance_policies: vec![],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Network/azureFirewalls".to_string(),
                display_name: "Azure Firewall".to_string(),
                category: ResourceCategory::NetworksFirewalls,
                icon: "shield".to_string(),
                description: "Managed network security service".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Rules".to_string(),
                        icon: "list".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Edit Policies".to_string(),
                        icon: "edit".to_string(),
                        confirmation_required: true,
                    },
                ],
                required_permissions: vec!["Firewall.Manage".to_string()],
                cost_factors: vec!["data_processed".to_string(), "deployment_hours".to_string()],
                compliance_policies: vec!["PCI-DSS".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Network/applicationGateways".to_string(),
                display_name: "Application Gateway".to_string(),
                category: ResourceCategory::NetworksFirewalls,
                icon: "filter".to_string(),
                description: "Web traffic load balancer with WAF".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Backend".to_string(),
                        icon: "server".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "WAF Rules".to_string(),
                        icon: "shield".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["AppGateway.Manage".to_string()],
                cost_factors: vec!["capacity_units".to_string(), "data_processed".to_string()],
                compliance_policies: vec!["OWASP".to_string()],
            },
            ResourceDefinition {
                resource_type: "Microsoft.Network/frontDoors".to_string(),
                display_name: "Azure Front Door".to_string(),
                category: ResourceCategory::NetworksFirewalls,
                icon: "globe".to_string(),
                description: "Global load balancer and CDN".to_string(),
                quick_actions: vec![
                    QuickActionTemplate {
                        action_type: ActionType::ViewDetails,
                        label: "View Origins".to_string(),
                        icon: "map-pin".to_string(),
                        confirmation_required: false,
                    },
                    QuickActionTemplate {
                        action_type: ActionType::Configure,
                        label: "Routing Rules".to_string(),
                        icon: "route".to_string(),
                        confirmation_required: false,
                    },
                ],
                required_permissions: vec!["FrontDoor.Manage".to_string()],
                cost_factors: vec!["requests".to_string(), "data_transfer".to_string()],
                compliance_policies: vec![],
            },
        ];

        for resource in network_resources {
            self.definitions.insert(resource.resource_type.clone(), resource);
        }
    }

    pub fn get_definition(&self, resource_type: &str) -> Option<&ResourceDefinition> {
        self.definitions.get(resource_type)
    }

    pub fn get_by_category(&self, category: ResourceCategory) -> Vec<&ResourceDefinition> {
        self.definitions
            .values()
            .filter(|d| matches!(d.category, ref c if *c == category))
            .collect()
    }

    pub fn get_all_definitions(&self) -> Vec<&ResourceDefinition> {
        self.definitions.values().collect()
    }
}