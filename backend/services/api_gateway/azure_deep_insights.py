"""
Azure Deep Insights Module
Provides comprehensive drill-down capabilities for all Azure resources
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.security import SecurityCenter
from azure.mgmt.monitor import MonitorManagementClient
import logging

logger = logging.getLogger(__name__)

class AzureDeepInsights:
    def __init__(self, subscription_id: str = "205b477d-17e7-4b3b-92c1-32cf02626b78"):
        """Initialize Azure clients for deep insights"""
        try:
            self.credential = AzureCliCredential()
            self.subscription_id = subscription_id
            
            # Initialize all Azure management clients
            self.resource_client = ResourceManagementClient(self.credential, subscription_id)
            self.policy_insights = PolicyInsightsClient(self.credential, subscription_id)
            self.auth_client = AuthorizationManagementClient(self.credential, subscription_id)
            self.cost_client = CostManagementClient(self.credential)
            self.network_client = NetworkManagementClient(self.credential, subscription_id)
            self.security_client = SecurityCenter(self.credential, subscription_id)
            self.monitor_client = MonitorManagementClient(self.credential, subscription_id)
            
            self.connected = True
            logger.info("Successfully connected to Azure with deep insights capabilities")
        except Exception as e:
            logger.error(f"Failed to connect to Azure: {e}")
            self.connected = False
    
    async def get_policy_compliance_deep(self) -> Dict[str, Any]:
        """Get deep policy compliance data with full drill-down"""
        if not self.connected:
            return self._get_mock_policy_data()
        
        try:
            # Get all policy assignments
            assignments = []
            for assignment in self.policy_insights.policy_assignments.list():
                assignments.append({
                    "id": assignment.id,
                    "name": assignment.name,
                    "displayName": assignment.display_name,
                    "description": assignment.description,
                    "policyDefinitionId": assignment.policy_definition_id,
                    "scope": assignment.scope,
                    "parameters": assignment.parameters,
                    "enforcementMode": assignment.enforcement_mode
                })
            
            # Get compliance states for each assignment
            compliance_results = []
            for assignment in assignments[:10]:  # Limit for performance
                states = self.policy_insights.policy_states.list_query_results_for_subscription(
                    policy_states_resource="latest",
                    query_options={"filter": f"policyAssignmentId eq '{assignment['id']}'"}
                )
                
                compliant_count = 0
                non_compliant_count = 0
                non_compliant_resources = []
                
                for state in states:
                    if state.compliance_state == "Compliant":
                        compliant_count += 1
                    else:
                        non_compliant_count += 1
                        non_compliant_resources.append({
                            "resourceId": state.resource_id,
                            "resourceType": state.resource_type,
                            "resourceLocation": state.resource_location,
                            "resourceGroup": state.resource_group,
                            "complianceState": state.compliance_state,
                            "complianceReason": state.compliance_state_reason,
                            "policyDefinitionAction": state.policy_definition_action,
                            "policyDefinitionReferenceId": state.policy_definition_reference_id,
                            "timestamp": state.timestamp.isoformat() if state.timestamp else None,
                            "isCompliant": False,
                            "remediationOptions": self._get_remediation_options(state)
                        })
                
                compliance_results.append({
                    "assignment": assignment,
                    "summary": {
                        "totalResources": compliant_count + non_compliant_count,
                        "compliantResources": compliant_count,
                        "nonCompliantResources": non_compliant_count,
                        "compliancePercentage": (compliant_count / (compliant_count + non_compliant_count) * 100) if (compliant_count + non_compliant_count) > 0 else 0
                    },
                    "nonCompliantResources": non_compliant_resources
                })
            
            return {
                "success": True,
                "totalAssignments": len(assignments),
                "complianceResults": compliance_results,
                "recommendations": self._generate_compliance_recommendations(compliance_results)
            }
            
        except Exception as e:
            logger.error(f"Error fetching deep policy compliance: {e}")
            return self._get_mock_policy_data()
    
    async def get_rbac_deep_analysis(self) -> Dict[str, Any]:
        """Get deep RBAC analysis with privilege insights"""
        if not self.connected:
            return self._get_mock_rbac_data()
        
        try:
            # Get all role assignments
            role_assignments = []
            for assignment in self.auth_client.role_assignments.list_for_subscription():
                # Get role definition details
                role_def = self.auth_client.role_definitions.get_by_id(assignment.role_definition_id)
                
                role_assignments.append({
                    "id": assignment.id,
                    "principalId": assignment.principal_id,
                    "principalType": assignment.principal_type,
                    "roleDefinitionId": assignment.role_definition_id,
                    "roleName": role_def.role_name,
                    "roleType": role_def.role_type,
                    "scope": assignment.scope,
                    "permissions": [
                        {
                            "actions": perm.actions,
                            "notActions": perm.not_actions,
                            "dataActions": perm.data_actions,
                            "notDataActions": perm.not_data_actions
                        } for perm in role_def.permissions
                    ],
                    "isBuiltIn": role_def.role_type == "BuiltInRole",
                    "isPrivileged": self._is_privileged_role(role_def.role_name),
                    "riskLevel": self._calculate_rbac_risk(role_def)
                })
            
            # Analyze RBAC risks
            risks = self._analyze_rbac_risks(role_assignments)
            
            return {
                "success": True,
                "totalAssignments": len(role_assignments),
                "roleAssignments": role_assignments[:50],  # Limit for UI
                "riskAnalysis": risks,
                "recommendations": self._generate_rbac_recommendations(risks)
            }
            
        except Exception as e:
            logger.error(f"Error fetching RBAC analysis: {e}")
            return self._get_mock_rbac_data()
    
    async def get_cost_analysis_deep(self) -> Dict[str, Any]:
        """Get deep cost analysis with optimization opportunities"""
        if not self.connected:
            return self._get_mock_cost_data()
        
        try:
            # Query cost data
            query = {
                "type": "Usage",
                "timeframe": "MonthToDate",
                "dataset": {
                    "granularity": "Daily",
                    "aggregation": {
                        "totalCost": {
                            "name": "Cost",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {"type": "Dimension", "name": "ResourceGroup"},
                        {"type": "Dimension", "name": "ServiceName"}
                    ]
                }
            }
            
            result = self.cost_client.query.usage(
                scope=f"/subscriptions/{self.subscription_id}",
                parameters=query
            )
            
            # Process cost data
            cost_breakdown = []
            total_cost = 0
            
            for row in result.rows:
                cost = row[0]  # Cost value
                resource_group = row[1] if len(row) > 1 else "Unknown"
                service = row[2] if len(row) > 2 else "Unknown"
                
                total_cost += cost
                cost_breakdown.append({
                    "resourceGroup": resource_group,
                    "service": service,
                    "cost": cost,
                    "currency": "USD",
                    "optimizationPotential": self._calculate_optimization_potential(service, cost)
                })
            
            # Get cost anomalies
            anomalies = self._detect_cost_anomalies(cost_breakdown)
            
            # Get optimization recommendations
            optimizations = self._generate_cost_optimizations(cost_breakdown)
            
            return {
                "success": True,
                "totalCost": total_cost,
                "currency": "USD",
                "period": "Month to Date",
                "breakdown": cost_breakdown,
                "anomalies": anomalies,
                "optimizations": optimizations,
                "savingsPotential": sum(opt["savingsAmount"] for opt in optimizations)
            }
            
        except Exception as e:
            logger.error(f"Error fetching cost analysis: {e}")
            return self._get_mock_cost_data()
    
    async def get_network_security_deep(self) -> Dict[str, Any]:
        """Get deep network security analysis"""
        if not self.connected:
            return self._get_mock_network_data()
        
        try:
            # Get all NSGs
            nsgs = []
            for nsg in self.network_client.network_security_groups.list_all():
                security_rules = []
                for rule in nsg.security_rules or []:
                    risk_level = self._assess_rule_risk(rule)
                    security_rules.append({
                        "name": rule.name,
                        "priority": rule.priority,
                        "direction": rule.direction,
                        "access": rule.access,
                        "protocol": rule.protocol,
                        "sourcePortRange": rule.source_port_range,
                        "destinationPortRange": rule.destination_port_range,
                        "sourceAddressPrefix": rule.source_address_prefix,
                        "destinationAddressPrefix": rule.destination_address_prefix,
                        "riskLevel": risk_level,
                        "issues": self._identify_rule_issues(rule)
                    })
                
                nsgs.append({
                    "id": nsg.id,
                    "name": nsg.name,
                    "location": nsg.location,
                    "resourceGroup": nsg.id.split('/')[4],
                    "securityRules": security_rules,
                    "totalRules": len(security_rules),
                    "highRiskRules": len([r for r in security_rules if r["riskLevel"] == "High"]),
                    "recommendations": self._generate_nsg_recommendations(security_rules)
                })
            
            # Get network watchers
            network_insights = self._get_network_insights()
            
            return {
                "success": True,
                "networkSecurityGroups": nsgs,
                "totalNSGs": len(nsgs),
                "securityInsights": network_insights,
                "overallRisk": self._calculate_overall_network_risk(nsgs),
                "recommendations": self._generate_network_recommendations(nsgs)
            }
            
        except Exception as e:
            logger.error(f"Error fetching network security: {e}")
            return self._get_mock_network_data()
    
    async def get_resource_insights_deep(self) -> Dict[str, Any]:
        """Get deep resource insights with health and performance"""
        if not self.connected:
            return self._get_mock_resource_data()
        
        try:
            resources = []
            for resource in self.resource_client.resources.list():
                # Get resource health
                try:
                    health = self.monitor_client.activity_logs.list(
                        filter=f"resourceId eq '{resource.id}'",
                        select="eventName,status,level"
                    )
                    health_status = "Healthy"  # Default
                    for event in health:
                        if event.level in ["Error", "Critical"]:
                            health_status = "Unhealthy"
                            break
                        elif event.level == "Warning":
                            health_status = "Warning"
                except:
                    health_status = "Unknown"
                
                resources.append({
                    "id": resource.id,
                    "name": resource.name,
                    "type": resource.type,
                    "location": resource.location,
                    "resourceGroup": resource.id.split('/')[4] if len(resource.id.split('/')) > 4 else "Unknown",
                    "tags": resource.tags or {},
                    "sku": resource.sku.name if resource.sku else None,
                    "kind": resource.kind,
                    "managedBy": resource.managed_by,
                    "healthStatus": health_status,
                    "complianceStatus": "Compliant",  # Would need to cross-reference with policy
                    "costEstimate": self._estimate_resource_cost(resource),
                    "recommendations": self._get_resource_recommendations(resource)
                })
            
            # Group resources by type
            resource_summary = {}
            for resource in resources:
                res_type = resource["type"]
                if res_type not in resource_summary:
                    resource_summary[res_type] = {
                        "count": 0,
                        "locations": set(),
                        "healthSummary": {"Healthy": 0, "Warning": 0, "Unhealthy": 0, "Unknown": 0}
                    }
                resource_summary[res_type]["count"] += 1
                resource_summary[res_type]["locations"].add(resource["location"])
                resource_summary[res_type]["healthSummary"][resource["healthStatus"]] += 1
            
            # Convert sets to lists for JSON serialization
            for res_type in resource_summary:
                resource_summary[res_type]["locations"] = list(resource_summary[res_type]["locations"])
            
            return {
                "success": True,
                "totalResources": len(resources),
                "resources": resources[:100],  # Limit for UI
                "resourceSummary": resource_summary,
                "healthOverview": {
                    "healthy": len([r for r in resources if r["healthStatus"] == "Healthy"]),
                    "warning": len([r for r in resources if r["healthStatus"] == "Warning"]),
                    "unhealthy": len([r for r in resources if r["healthStatus"] == "Unhealthy"]),
                    "unknown": len([r for r in resources if r["healthStatus"] == "Unknown"])
                },
                "recommendations": self._generate_resource_recommendations(resources)
            }
            
        except Exception as e:
            logger.error(f"Error fetching resource insights: {e}")
            return self._get_mock_resource_data()
    
    # Helper methods for generating recommendations and risk assessments
    
    def _get_remediation_options(self, policy_state) -> List[Dict[str, str]]:
        """Generate remediation options for non-compliant resources"""
        options = []
        
        if policy_state.policy_definition_action == "deny":
            options.append({
                "action": "modify",
                "description": "Modify resource to meet policy requirements",
                "automated": True
            })
        
        options.append({
            "action": "exception",
            "description": "Create policy exception for this resource",
            "automated": True
        })
        
        options.append({
            "action": "delete",
            "description": "Delete non-compliant resource",
            "automated": False
        })
        
        return options
    
    def _generate_compliance_recommendations(self, compliance_results) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for result in compliance_results:
            if result["summary"]["compliancePercentage"] < 80:
                recommendations.append(
                    f"Policy '{result['assignment']['displayName']}' has low compliance ({result['summary']['compliancePercentage']:.1f}%). "
                    f"Review {result['summary']['nonCompliantResources']} non-compliant resources."
                )
        
        return recommendations
    
    def _is_privileged_role(self, role_name: str) -> bool:
        """Check if role is privileged"""
        privileged_roles = [
            "Owner", "Contributor", "User Access Administrator",
            "Security Admin", "Global Administrator"
        ]
        return role_name in privileged_roles
    
    def _calculate_rbac_risk(self, role_def) -> str:
        """Calculate RBAC risk level"""
        high_risk_actions = ["*", "Microsoft.Authorization/*", "Microsoft.Security/*"]
        
        for perm in role_def.permissions:
            for action in perm.actions:
                if action in high_risk_actions:
                    return "High"
        
        if self._is_privileged_role(role_def.role_name):
            return "Medium"
        
        return "Low"
    
    def _analyze_rbac_risks(self, role_assignments) -> Dict[str, Any]:
        """Analyze RBAC risks"""
        return {
            "privilegedAccounts": len([r for r in role_assignments if r["isPrivileged"]]),
            "highRiskAssignments": len([r for r in role_assignments if r["riskLevel"] == "High"]),
            "staleAssignments": 0,  # Would need activity data
            "serviceAccounts": len([r for r in role_assignments if r["principalType"] == "ServicePrincipal"])
        }
    
    def _generate_rbac_recommendations(self, risks) -> List[str]:
        """Generate RBAC recommendations"""
        recommendations = []
        
        if risks["privilegedAccounts"] > 5:
            recommendations.append(f"Review {risks['privilegedAccounts']} privileged accounts for least privilege principle")
        
        if risks["highRiskAssignments"] > 0:
            recommendations.append(f"Address {risks['highRiskAssignments']} high-risk role assignments")
        
        return recommendations
    
    def _calculate_optimization_potential(self, service: str, cost: float) -> float:
        """Calculate cost optimization potential"""
        # Simple heuristic - real implementation would use more sophisticated analysis
        optimization_percentages = {
            "Virtual Machines": 0.3,
            "Storage": 0.2,
            "SQL Database": 0.25,
            "App Service": 0.15
        }
        return cost * optimization_percentages.get(service, 0.1)
    
    def _detect_cost_anomalies(self, cost_breakdown) -> List[Dict[str, Any]]:
        """Detect cost anomalies"""
        anomalies = []
        avg_cost = sum(item["cost"] for item in cost_breakdown) / len(cost_breakdown) if cost_breakdown else 0
        
        for item in cost_breakdown:
            if item["cost"] > avg_cost * 2:
                anomalies.append({
                    "resourceGroup": item["resourceGroup"],
                    "service": item["service"],
                    "cost": item["cost"],
                    "severity": "High" if item["cost"] > avg_cost * 3 else "Medium",
                    "recommendation": f"Investigate high cost in {item['service']}"
                })
        
        return anomalies
    
    def _generate_cost_optimizations(self, cost_breakdown) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations"""
        optimizations = []
        
        for item in cost_breakdown:
            if item["optimizationPotential"] > 100:
                optimizations.append({
                    "resourceGroup": item["resourceGroup"],
                    "service": item["service"],
                    "currentCost": item["cost"],
                    "savingsAmount": item["optimizationPotential"],
                    "savingsPercentage": (item["optimizationPotential"] / item["cost"] * 100) if item["cost"] > 0 else 0,
                    "recommendation": self._get_optimization_recommendation(item["service"])
                })
        
        return optimizations
    
    def _get_optimization_recommendation(self, service: str) -> str:
        """Get service-specific optimization recommendation"""
        recommendations = {
            "Virtual Machines": "Consider using Reserved Instances or Spot VMs",
            "Storage": "Review storage tiers and implement lifecycle policies",
            "SQL Database": "Optimize DTU allocation or switch to serverless",
            "App Service": "Review App Service Plan sizing"
        }
        return recommendations.get(service, "Review resource utilization and sizing")
    
    def _assess_rule_risk(self, rule) -> str:
        """Assess network security rule risk"""
        if rule.source_address_prefix == "*" and rule.access == "Allow":
            return "High"
        elif rule.destination_port_range in ["3389", "22", "445"] and rule.access == "Allow":
            return "High"
        elif rule.source_address_prefix == "Internet" and rule.access == "Allow":
            return "Medium"
        return "Low"
    
    def _identify_rule_issues(self, rule) -> List[str]:
        """Identify issues with network security rule"""
        issues = []
        
        if rule.source_address_prefix == "*":
            issues.append("Rule allows traffic from any source")
        
        if rule.destination_port_range in ["3389", "22"]:
            issues.append("Management ports exposed")
        
        if rule.priority > 4000:
            issues.append("Low priority rule may be overridden")
        
        return issues
    
    def _generate_nsg_recommendations(self, rules) -> List[str]:
        """Generate NSG recommendations"""
        recommendations = []
        
        high_risk_rules = [r for r in rules if r["riskLevel"] == "High"]
        if high_risk_rules:
            recommendations.append(f"Review {len(high_risk_rules)} high-risk security rules")
        
        return recommendations
    
    def _get_network_insights(self) -> Dict[str, Any]:
        """Get network security insights"""
        return {
            "exposedEndpoints": 0,  # Would need to analyze public IPs
            "unusedNSGs": 0,  # Would need to check associations
            "complianceStatus": "Compliant"
        }
    
    def _calculate_overall_network_risk(self, nsgs) -> str:
        """Calculate overall network risk"""
        high_risk_count = sum(nsg["highRiskRules"] for nsg in nsgs)
        if high_risk_count > 10:
            return "High"
        elif high_risk_count > 5:
            return "Medium"
        return "Low"
    
    def _generate_network_recommendations(self, nsgs) -> List[str]:
        """Generate network recommendations"""
        recommendations = []
        
        total_high_risk = sum(nsg["highRiskRules"] for nsg in nsgs)
        if total_high_risk > 0:
            recommendations.append(f"Address {total_high_risk} high-risk network security rules")
        
        return recommendations
    
    def _estimate_resource_cost(self, resource) -> float:
        """Estimate resource cost"""
        # Simple estimation based on resource type
        cost_map = {
            "Microsoft.Compute/virtualMachines": 150.0,
            "Microsoft.Storage/storageAccounts": 50.0,
            "Microsoft.Sql/servers/databases": 200.0,
            "Microsoft.Web/sites": 75.0
        }
        return cost_map.get(resource.type, 25.0)
    
    def _get_resource_recommendations(self, resource) -> List[str]:
        """Get resource-specific recommendations"""
        recommendations = []
        
        if not resource.tags:
            recommendations.append("Add tags for better organization and cost tracking")
        
        if resource.type == "Microsoft.Compute/virtualMachines":
            recommendations.append("Review VM size for optimization opportunities")
        
        return recommendations
    
    def _generate_resource_recommendations(self, resources) -> List[str]:
        """Generate overall resource recommendations"""
        recommendations = []
        
        untagged = len([r for r in resources if not r["tags"]])
        if untagged > 10:
            recommendations.append(f"Tag {untagged} resources for better management")
        
        unhealthy = len([r for r in resources if r["healthStatus"] == "Unhealthy"])
        if unhealthy > 0:
            recommendations.append(f"Investigate {unhealthy} unhealthy resources")
        
        return recommendations
    
    # Mock data methods (fallbacks)
    
    def _get_mock_policy_data(self) -> Dict[str, Any]:
        """Return mock policy data when Azure connection fails"""
        return {
            "success": False,
            "message": "Using mock data - Azure connection not available",
            "totalAssignments": 5,
            "complianceResults": [
                {
                    "assignment": {
                        "name": "require-tag-environment",
                        "displayName": "Require Environment Tag",
                        "description": "All resources must have an environment tag"
                    },
                    "summary": {
                        "totalResources": 100,
                        "compliantResources": 75,
                        "nonCompliantResources": 25,
                        "compliancePercentage": 75.0
                    },
                    "nonCompliantResources": [
                        {
                            "resourceId": "/subscriptions/xxx/resourceGroups/rg-test/providers/Microsoft.Compute/virtualMachines/vm-test",
                            "resourceType": "Microsoft.Compute/virtualMachines",
                            "resourceLocation": "eastus",
                            "resourceGroup": "rg-test",
                            "complianceState": "NonCompliant",
                            "complianceReason": "Missing required tag: environment",
                            "remediationOptions": [
                                {"action": "modify", "description": "Add environment tag", "automated": True},
                                {"action": "exception", "description": "Create policy exception", "automated": True}
                            ]
                        }
                    ]
                }
            ],
            "recommendations": ["Review non-compliant resources and apply remediation"]
        }
    
    def _get_mock_rbac_data(self) -> Dict[str, Any]:
        """Return mock RBAC data"""
        return {
            "success": False,
            "message": "Using mock data - Azure connection not available",
            "totalAssignments": 25,
            "roleAssignments": [
                {
                    "principalId": "user1@company.com",
                    "principalType": "User",
                    "roleName": "Owner",
                    "scope": "/subscriptions/xxx",
                    "isPrivileged": True,
                    "riskLevel": "High"
                }
            ],
            "riskAnalysis": {
                "privilegedAccounts": 5,
                "highRiskAssignments": 2,
                "staleAssignments": 3,
                "serviceAccounts": 10
            },
            "recommendations": ["Review privileged access", "Implement PIM"]
        }
    
    def _get_mock_cost_data(self) -> Dict[str, Any]:
        """Return mock cost data"""
        return {
            "success": False,
            "message": "Using mock data - Azure connection not available",
            "totalCost": 15000.0,
            "currency": "USD",
            "period": "Month to Date",
            "breakdown": [
                {
                    "resourceGroup": "rg-production",
                    "service": "Virtual Machines",
                    "cost": 5000.0,
                    "optimizationPotential": 1500.0
                }
            ],
            "anomalies": [],
            "optimizations": [
                {
                    "service": "Virtual Machines",
                    "currentCost": 5000.0,
                    "savingsAmount": 1500.0,
                    "recommendation": "Use Reserved Instances"
                }
            ],
            "savingsPotential": 3000.0
        }
    
    def _get_mock_network_data(self) -> Dict[str, Any]:
        """Return mock network data"""
        return {
            "success": False,
            "message": "Using mock data - Azure connection not available",
            "networkSecurityGroups": [],
            "totalNSGs": 0,
            "securityInsights": {},
            "overallRisk": "Low",
            "recommendations": []
        }
    
    def _get_mock_resource_data(self) -> Dict[str, Any]:
        """Return mock resource data"""
        return {
            "success": False,
            "message": "Using mock data - Azure connection not available",
            "totalResources": 0,
            "resources": [],
            "resourceSummary": {},
            "healthOverview": {
                "healthy": 0,
                "warning": 0,
                "unhealthy": 0,
                "unknown": 0
            },
            "recommendations": []
        }