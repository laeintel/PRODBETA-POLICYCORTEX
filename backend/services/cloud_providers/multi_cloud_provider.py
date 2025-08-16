"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Multi-Cloud Provider Orchestrator for PolicyCortex
Provides unified interface for managing resources across Azure, AWS, and GCP
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud providers"""
    AZURE = "Azure"
    AWS = "AWS"
    GCP = "GCP"
    ALL = "All"

class MultiCloudProvider:
    """Multi-cloud provider orchestrator"""
    
    def __init__(self):
        """Initialize multi-cloud provider"""
        self.providers = {}
        self.enabled_providers = []
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize enabled cloud providers"""
        
        # Initialize Azure provider
        if os.getenv("AZURE_SUBSCRIPTION_ID"):
            try:
                from azure_real_data import AzureRealDataCollector
                self.providers[CloudProvider.AZURE] = AzureRealDataCollector()
                self.enabled_providers.append(CloudProvider.AZURE)
                logger.info("Azure provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure provider: {e}")
        
        # Initialize AWS provider
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCOUNT_ID"):
            try:
                from cloud_providers.aws_provider import AWSProvider
                self.providers[CloudProvider.AWS] = AWSProvider()
                self.enabled_providers.append(CloudProvider.AWS)
                logger.info("AWS provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS provider: {e}")
        
        # Initialize GCP provider
        if os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from cloud_providers.gcp_provider import GCPProvider
                self.providers[CloudProvider.GCP] = GCPProvider()
                self.enabled_providers.append(CloudProvider.GCP)
                logger.info("GCP provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GCP provider: {e}")
        
        if not self.enabled_providers:
            logger.warning("No cloud providers enabled. Check your environment configuration.")
    
    async def get_resources(self, provider: CloudProvider = CloudProvider.ALL, 
                           resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resources from specified or all providers"""
        all_resources = []
        
        providers_to_query = self._get_providers_to_query(provider)
        
        # Gather resources from all providers concurrently
        tasks = []
        for cloud_provider in providers_to_query:
            if cloud_provider in self.providers:
                provider_instance = self.providers[cloud_provider]
                
                if cloud_provider == CloudProvider.AZURE:
                    # Azure uses different method signature
                    tasks.append(self._get_azure_resources(provider_instance, resource_type))
                else:
                    tasks.append(provider_instance.get_resources(resource_type))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error getting resources: {result}")
                elif isinstance(result, list):
                    all_resources.extend(result)
        
        return all_resources
    
    async def get_policies(self, provider: CloudProvider = CloudProvider.ALL) -> List[Dict[str, Any]]:
        """Get policies from specified or all providers"""
        all_policies = []
        
        providers_to_query = self._get_providers_to_query(provider)
        
        tasks = []
        for cloud_provider in providers_to_query:
            if cloud_provider in self.providers:
                provider_instance = self.providers[cloud_provider]
                
                if cloud_provider == CloudProvider.AZURE:
                    tasks.append(self._get_azure_policies(provider_instance))
                else:
                    tasks.append(provider_instance.get_policies())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error getting policies: {result}")
                elif isinstance(result, list):
                    all_policies.extend(result)
        
        return all_policies
    
    async def get_costs(self, provider: CloudProvider = CloudProvider.ALL,
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get cost information from specified or all providers"""
        costs_by_provider = {}
        total_spend = 0
        total_forecast = 0
        
        providers_to_query = self._get_providers_to_query(provider)
        
        tasks = []
        provider_names = []
        
        for cloud_provider in providers_to_query:
            if cloud_provider in self.providers:
                provider_instance = self.providers[cloud_provider]
                provider_names.append(cloud_provider.value)
                
                if cloud_provider == CloudProvider.AZURE:
                    tasks.append(self._get_azure_costs(provider_instance))
                else:
                    tasks.append(provider_instance.get_costs(start_date, end_date))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                provider_name = provider_names[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error getting costs from {provider_name}: {result}")
                    costs_by_provider[provider_name] = {"error": str(result)}
                elif isinstance(result, dict):
                    costs_by_provider[provider_name] = result
                    total_spend += result.get("current_spend", 0)
                    total_forecast += result.get("forecasted_spend", 0)
        
        return {
            "total_current_spend": round(total_spend, 2),
            "total_forecasted_spend": round(total_forecast, 2),
            "providers": costs_by_provider,
            "currency": "USD",  # Standardize to USD
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_compliance_status(self, provider: CloudProvider = CloudProvider.ALL) -> Dict[str, Any]:
        """Get compliance status from specified or all providers"""
        compliance_by_provider = {}
        total_compliant = 0
        total_non_compliant = 0
        all_violations = []
        
        providers_to_query = self._get_providers_to_query(provider)
        
        tasks = []
        provider_names = []
        
        for cloud_provider in providers_to_query:
            if cloud_provider in self.providers:
                provider_instance = self.providers[cloud_provider]
                provider_names.append(cloud_provider.value)
                
                if cloud_provider == CloudProvider.AZURE:
                    tasks.append(self._get_azure_compliance(provider_instance))
                else:
                    tasks.append(provider_instance.get_compliance_status())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                provider_name = provider_names[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error getting compliance from {provider_name}: {result}")
                    compliance_by_provider[provider_name] = {"error": str(result)}
                elif isinstance(result, dict):
                    compliance_by_provider[provider_name] = result
                    total_compliant += result.get("compliant_resources", 0)
                    total_non_compliant += result.get("non_compliant_resources", 0)
                    
                    # Aggregate violations
                    provider_violations = result.get("violations", [])
                    for violation in provider_violations:
                        violation["provider"] = provider_name
                        all_violations.append(violation)
        
        # Calculate overall compliance score
        total_resources = total_compliant + total_non_compliant
        overall_score = 0
        if total_resources > 0:
            overall_score = round((total_compliant / total_resources) * 100, 2)
        
        return {
            "overall_compliance_score": overall_score,
            "total_compliant_resources": total_compliant,
            "total_non_compliant_resources": total_non_compliant,
            "violations": all_violations[:50],  # Limit to 50 violations
            "providers": compliance_by_provider,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_security_findings(self, provider: CloudProvider = CloudProvider.ALL) -> List[Dict[str, Any]]:
        """Get security findings from specified or all providers"""
        all_findings = []
        
        providers_to_query = self._get_providers_to_query(provider)
        
        tasks = []
        for cloud_provider in providers_to_query:
            if cloud_provider in self.providers:
                provider_instance = self.providers[cloud_provider]
                
                if cloud_provider == CloudProvider.AZURE:
                    tasks.append(self._get_azure_security_findings(provider_instance))
                else:
                    tasks.append(provider_instance.get_security_findings())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error getting security findings: {result}")
                elif isinstance(result, list):
                    all_findings.extend(result)
        
        # Sort by severity and return top findings
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        all_findings.sort(key=lambda x: severity_order.get(str(x.get("severity", "")).upper(), 5))
        
        return all_findings[:100]  # Limit to 100 findings
    
    async def apply_governance_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply governance action to resource in appropriate cloud"""
        try:
            resource_id = action.get("resource_id")
            provider_name = action.get("provider")
            
            # Determine provider from resource ID if not specified
            if not provider_name:
                if resource_id.startswith("/subscriptions/"):
                    provider_name = CloudProvider.AZURE.value
                elif resource_id.startswith("arn:aws:") or resource_id.startswith("i-"):
                    provider_name = CloudProvider.AWS.value
                elif resource_id.startswith("projects/") or resource_id.startswith("gs://"):
                    provider_name = CloudProvider.GCP.value
                else:
                    return {
                        "success": False,
                        "error": "Could not determine cloud provider from resource ID"
                    }
            
            # Get the appropriate provider
            cloud_provider = CloudProvider(provider_name)
            if cloud_provider not in self.providers:
                return {
                    "success": False,
                    "error": f"Provider {provider_name} not available"
                }
            
            provider_instance = self.providers[cloud_provider]
            
            # Apply the action
            if cloud_provider == CloudProvider.AZURE:
                return await self._apply_azure_action(provider_instance, action)
            else:
                return await provider_instance.apply_governance_action(action)
            
        except Exception as e:
            logger.error(f"Failed to apply governance action: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled cloud providers"""
        return [provider.value for provider in self.enabled_providers]
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all cloud providers"""
        status = {
            "enabled_providers": self.get_enabled_providers(),
            "provider_details": {}
        }
        
        for provider in [CloudProvider.AZURE, CloudProvider.AWS, CloudProvider.GCP]:
            is_enabled = provider in self.enabled_providers
            status["provider_details"][provider.value] = {
                "enabled": is_enabled,
                "configured": self._is_provider_configured(provider),
                "healthy": is_enabled  # Could add health checks here
            }
        
        return status
    
    # Helper methods
    def _get_providers_to_query(self, provider: CloudProvider) -> List[CloudProvider]:
        """Get list of providers to query based on request"""
        if provider == CloudProvider.ALL:
            return self.enabled_providers
        elif provider in self.enabled_providers:
            return [provider]
        else:
            logger.warning(f"Provider {provider.value} not enabled")
            return []
    
    def _is_provider_configured(self, provider: CloudProvider) -> bool:
        """Check if provider is configured"""
        if provider == CloudProvider.AZURE:
            return bool(os.getenv("AZURE_SUBSCRIPTION_ID"))
        elif provider == CloudProvider.AWS:
            return bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCOUNT_ID"))
        elif provider == CloudProvider.GCP:
            return bool(os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        return False
    
    # Azure adapter methods (to work with existing Azure implementation)
    async def _get_azure_resources(self, provider_instance, resource_type: Optional[str]) -> List[Dict[str, Any]]:
        """Adapt Azure provider to common interface"""
        try:
            # Use existing Azure data collector
            data = provider_instance.get_complete_governance_data()
            resources = []
            
            # Convert Azure format to common format
            if "resources" in data:
                for res in data["resources"][:100]:  # Limit to 100 resources
                    resources.append({
                        "id": res.get("id"),
                        "name": res.get("name"),
                        "type": res.get("type"),
                        "provider": "Azure",
                        "region": res.get("location"),
                        "state": "active",
                        "tags": res.get("tags", {}),
                        "metadata": res
                    })
            
            return resources
        except Exception as e:
            logger.error(f"Error getting Azure resources: {e}")
            return []
    
    async def _get_azure_policies(self, provider_instance) -> List[Dict[str, Any]]:
        """Adapt Azure provider to common interface"""
        try:
            data = provider_instance.get_complete_governance_data()
            policies = []
            
            if "policies" in data and "assignments" in data["policies"]:
                for policy in data["policies"]["assignments"][:50]:
                    policies.append({
                        "id": policy.get("id"),
                        "name": policy.get("name"),
                        "type": "Azure::Policy",
                        "provider": "Azure",
                        "description": policy.get("description", ""),
                        "state": policy.get("enforcement_mode", "Enabled")
                    })
            
            return policies
        except Exception as e:
            logger.error(f"Error getting Azure policies: {e}")
            return []
    
    async def _get_azure_costs(self, provider_instance) -> Dict[str, Any]:
        """Adapt Azure provider to common interface"""
        try:
            data = provider_instance.get_complete_governance_data()
            costs = data.get("costs", {})
            
            return {
                "provider": "Azure",
                "current_spend": costs.get("current_spend", 0),
                "forecasted_spend": costs.get("predicted_spend", 0),
                "savings_identified": costs.get("savings_identified", 0),
                "currency": "USD"
            }
        except Exception as e:
            logger.error(f"Error getting Azure costs: {e}")
            return {"provider": "Azure", "current_spend": 0, "forecasted_spend": 0}
    
    async def _get_azure_compliance(self, provider_instance) -> Dict[str, Any]:
        """Adapt Azure provider to common interface"""
        try:
            data = provider_instance.get_complete_governance_data()
            policies = data.get("policies", {})
            
            return {
                "provider": "Azure",
                "compliant_resources": policies.get("compliant_resources", 0),
                "non_compliant_resources": policies.get("non_compliant_resources", 0),
                "compliance_score": policies.get("compliance_rate", 0),
                "violations": []  # Would need to extract from detailed data
            }
        except Exception as e:
            logger.error(f"Error getting Azure compliance: {e}")
            return {"provider": "Azure", "compliance_score": 0}
    
    async def _get_azure_security_findings(self, provider_instance) -> List[Dict[str, Any]]:
        """Adapt Azure provider to common interface"""
        try:
            data = provider_instance.get_complete_governance_data()
            security = data.get("security", {})
            findings = []
            
            # Convert Azure security data to common format
            if security.get("alerts"):
                for alert in security["alerts"][:20]:
                    findings.append({
                        "id": f"azure-alert-{hash(str(alert))}",
                        "title": alert,
                        "description": alert,
                        "severity": "MEDIUM",
                        "provider": "Azure",
                        "service": "Security Center",
                        "created_at": datetime.utcnow().isoformat()
                    })
            
            return findings
        except Exception as e:
            logger.error(f"Error getting Azure security findings: {e}")
            return []
    
    async def _apply_azure_action(self, provider_instance, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action to Azure resource"""
        # This would need to be implemented based on Azure capabilities
        return {
            "success": False,
            "error": "Azure governance actions not yet implemented"
        }

# Singleton instance
multi_cloud_provider = MultiCloudProvider()