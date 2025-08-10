"""
Google Cloud Platform Provider Implementation for PolicyCortex
Provides GCP resource management and governance capabilities
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
# GCP imports with fallback
try:
    from google.cloud import storage
    from google.cloud import billing_v1
    from google.cloud import bigquery
    from google.oauth2 import service_account
    from google.api_core import exceptions
    GCP_AVAILABLE = True
    # Optional imports - will check individually
    try:
        from google.cloud import compute_v1
        COMPUTE_AVAILABLE = True
    except ImportError:
        COMPUTE_AVAILABLE = False
        compute_v1 = None
    
    try:
        from google.cloud import monitoring_v3
        MONITORING_AVAILABLE = True
    except ImportError:
        MONITORING_AVAILABLE = False
        monitoring_v3 = None
    
    try:
        from google.cloud import asset_v1
        ASSET_AVAILABLE = True
    except ImportError:
        ASSET_AVAILABLE = False
        asset_v1 = None
    
    try:
        from google.cloud import resourcemanager_v3
        RESOURCE_MANAGER_AVAILABLE = True
    except ImportError:
        RESOURCE_MANAGER_AVAILABLE = False
        resourcemanager_v3 = None
    
    try:
        from google.cloud import securitycenter_v1
        SECURITY_CENTER_AVAILABLE = True
    except ImportError:
        SECURITY_CENTER_AVAILABLE = False
        securitycenter_v1 = None
        
except ImportError:
    GCP_AVAILABLE = False
    logger.warning("GCP libraries not available")

logger = logging.getLogger(__name__)

class GCPProvider:
    """Google Cloud Platform provider implementation"""
    
    def __init__(self):
        """Initialize GCP provider with credentials"""
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.region = os.getenv("GCP_REGION", "us-central1")
        
        # Initialize GCP clients
        self.credentials = None
        self.compute_client = None
        self.storage_client = None
        self.monitoring_client = None
        self.billing_client = None
        self.asset_client = None
        self.resource_manager = None
        self.security_center = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize GCP service clients"""
        try:
            # Load credentials
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            
            # Initialize service clients (only if available)
            if self.credentials:
                if COMPUTE_AVAILABLE and compute_v1:
                    self.compute_client = compute_v1.InstancesClient(credentials=self.credentials)
                self.storage_client = storage.Client(credentials=self.credentials, project=self.project_id) if GCP_AVAILABLE else None
                if MONITORING_AVAILABLE and monitoring_v3:
                    self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=self.credentials)
                self.billing_client = billing_v1.CloudBillingClient(credentials=self.credentials) if GCP_AVAILABLE else None
                if ASSET_AVAILABLE and asset_v1:
                    self.asset_client = asset_v1.AssetServiceClient(credentials=self.credentials)
                if RESOURCE_MANAGER_AVAILABLE and resourcemanager_v3:
                    self.resource_manager = resourcemanager_v3.ProjectsClient(credentials=self.credentials)
                if SECURITY_CENTER_AVAILABLE and securitycenter_v1:
                    self.security_center = securitycenter_v1.SecurityCenterClient(credentials=self.credentials)
            else:
                # Use default credentials (application default credentials) if available
                if COMPUTE_AVAILABLE and compute_v1:
                    self.compute_client = compute_v1.InstancesClient()
                self.storage_client = storage.Client(project=self.project_id) if GCP_AVAILABLE else None
                if MONITORING_AVAILABLE and monitoring_v3:
                    self.monitoring_client = monitoring_v3.MetricServiceClient()
                self.billing_client = billing_v1.CloudBillingClient() if GCP_AVAILABLE else None
                if ASSET_AVAILABLE and asset_v1:
                    self.asset_client = asset_v1.AssetServiceClient()
                if RESOURCE_MANAGER_AVAILABLE and resourcemanager_v3:
                    self.resource_manager = resourcemanager_v3.ProjectsClient()
                if SECURITY_CENTER_AVAILABLE and securitycenter_v1:
                    self.security_center = securitycenter_v1.SecurityCenterClient()
            
            logger.info(f"GCP clients initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
    
    async def get_resources(self, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get GCP resources"""
        resources = []
        
        try:
            # Get Compute Engine instances
            if (not resource_type or resource_type == "compute") and COMPUTE_AVAILABLE and compute_v1:
                zones_client = compute_v1.ZonesClient(credentials=self.credentials)
                zones = zones_client.list(project=self.project_id)
                
                for zone in zones:
                    try:
                        instances = self.compute_client.list(
                            project=self.project_id,
                            zone=zone.name
                        )
                        
                        for instance in instances:
                            resources.append({
                                "id": f"projects/{self.project_id}/zones/{zone.name}/instances/{instance.name}",
                                "name": instance.name,
                                "type": "GCP::Compute::Instance",
                                "provider": "GCP",
                                "region": zone.name,
                                "state": instance.status,
                                "tags": instance.labels if hasattr(instance, 'labels') else {},
                                "metadata": {
                                    "machine_type": instance.machine_type.split('/')[-1] if instance.machine_type else None,
                                    "creation_timestamp": instance.creation_timestamp,
                                    "network_interfaces": len(instance.network_interfaces) if instance.network_interfaces else 0
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error fetching instances in zone {zone.name}: {e}")
            
            # Get Cloud Storage buckets
            if not resource_type or resource_type == "storage":
                buckets = self.storage_client.list_buckets()
                
                for bucket in buckets:
                    resources.append({
                        "id": f"gs://{bucket.name}",
                        "name": bucket.name,
                        "type": "GCP::Storage::Bucket",
                        "provider": "GCP",
                        "region": bucket.location,
                        "state": "active",
                        "tags": bucket.labels if bucket.labels else {},
                        "metadata": {
                            "storage_class": bucket.storage_class,
                            "time_created": bucket.time_created.isoformat() if bucket.time_created else None,
                            "versioning_enabled": bucket.versioning_enabled if hasattr(bucket, 'versioning_enabled') else False
                        }
                    })
            
            # Get Cloud SQL instances
            if not resource_type or resource_type == "database":
                from google.cloud import sql_v1
                sql_client = sql_v1.SqlInstancesServiceClient(credentials=self.credentials)
                
                try:
                    sql_instances = sql_client.list(project=self.project_id)
                    
                    for db_instance in sql_instances:
                        resources.append({
                            "id": f"projects/{self.project_id}/instances/{db_instance.name}",
                            "name": db_instance.name,
                            "type": "GCP::CloudSQL::Instance",
                            "provider": "GCP",
                            "region": db_instance.region if hasattr(db_instance, 'region') else self.region,
                            "state": db_instance.state.name if hasattr(db_instance, 'state') else "UNKNOWN",
                            "tags": db_instance.user_labels if hasattr(db_instance, 'user_labels') else {},
                            "metadata": {
                                "database_version": db_instance.database_version if hasattr(db_instance, 'database_version') else None,
                                "tier": db_instance.settings.tier if hasattr(db_instance.settings, 'tier') else None
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error fetching Cloud SQL instances: {e}")
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to get GCP resources: {e}")
            return []
    
    async def get_policies(self) -> List[Dict[str, Any]]:
        """Get GCP policies and organization policies"""
        policies = []
        
        try:
            # Get organization policies
            from google.cloud import orgpolicy_v2
            org_policy_client = orgpolicy_v2.OrgPolicyClient(credentials=self.credentials)
            
            # List policies for the project
            parent = f"projects/{self.project_id}"
            
            try:
                org_policies = org_policy_client.list_policies(parent=parent)
                
                for policy in org_policies:
                    policies.append({
                        "id": policy.name,
                        "name": policy.name.split('/')[-1],
                        "type": "OrgPolicy::Policy",
                        "provider": "GCP",
                        "description": f"Organization policy for {policy.name}",
                        "spec": policy.spec if hasattr(policy, 'spec') else None
                    })
            except Exception as e:
                logger.warning(f"Could not fetch organization policies: {e}")
            
            # Get IAM policies
            from google.cloud import iam_v1
            iam_client = iam_v1.IAMClient(credentials=self.credentials)
            
            try:
                # Get project IAM policy
                resource = f"projects/{self.project_id}"
                project = self.resource_manager.get_project(name=resource)
                
                if hasattr(project, 'iam_policy'):
                    policies.append({
                        "id": f"projects/{self.project_id}/iamPolicy",
                        "name": "Project IAM Policy",
                        "type": "IAM::Policy",
                        "provider": "GCP",
                        "description": f"IAM policy for project {self.project_id}",
                        "bindings": len(project.iam_policy.bindings) if hasattr(project.iam_policy, 'bindings') else 0
                    })
            except Exception as e:
                logger.warning(f"Could not fetch IAM policies: {e}")
            
            return policies
            
        except Exception as e:
            logger.error(f"Failed to get GCP policies: {e}")
            return []
    
    async def get_costs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get GCP cost information"""
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(credentials=self.credentials, project=self.project_id)
            
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Query billing data from BigQuery (requires billing export to be configured)
            query = f"""
                SELECT 
                    service.description as service,
                    SUM(cost) as total_cost,
                    currency
                FROM `{self.project_id}.billing.gcp_billing_export_v1`
                WHERE DATE(usage_start_time) >= '{start_date}'
                    AND DATE(usage_start_time) <= '{end_date}'
                GROUP BY service.description, currency
                ORDER BY total_cost DESC
            """
            
            try:
                query_job = bq_client.query(query)
                results = query_job.result()
                
                total_cost = 0
                service_costs = {}
                currency = "USD"
                
                for row in results:
                    service = row.service
                    cost = row.total_cost
                    currency = row.currency
                    total_cost += cost
                    service_costs[service] = round(cost, 2)
                
                return {
                    "provider": "GCP",
                    "current_spend": round(total_cost, 2),
                    "forecasted_spend": round(total_cost * 1.1, 2),  # Simple 10% increase forecast
                    "service_breakdown": service_costs,
                    "currency": currency,
                    "period": {
                        "start": start_date,
                        "end": end_date
                    }
                }
                
            except Exception as e:
                logger.warning(f"Could not query billing data from BigQuery: {e}")
                
                # Fallback to billing API
                billing_account = None
                if self.billing_client:
                    accounts = self.billing_client.list_billing_accounts()
                    
                    for account in accounts:
                        if account.open:
                            billing_account = account.name
                            break
                
                if billing_account:
                    # Note: Budget API may not be available
                    try:
                        from google.cloud import billing_budgets_v1
                        budget_client = billing_budgets_v1.BudgetServiceClient(credentials=self.credentials)
                        
                        budgets = budget_client.list_budgets(parent=billing_account)
                        total_budget = 0
                        
                        for budget in budgets:
                            if hasattr(budget.amount, 'specified_amount'):
                                total_budget += budget.amount.specified_amount.units
                        
                        return {
                            "provider": "GCP",
                            "current_spend": 0,  # Cannot get actual spend without BigQuery export
                            "budget": total_budget,
                            "currency": "USD",
                            "note": "Enable billing export to BigQuery for detailed cost data"
                        }
                    except ImportError:
                        logger.warning("GCP billing budgets API not available")
                        return {
                            "provider": "GCP",
                            "current_spend": 0,
                            "budget": 0,
                            "currency": "USD",
                            "note": "Budget API not available. Enable billing export to BigQuery for cost data"
                        }
                
                return {
                    "provider": "GCP",
                    "current_spend": 0,
                    "forecasted_spend": 0,
                    "error": "Billing data not available"
                }
                
        except Exception as e:
            logger.error(f"Failed to get GCP costs: {e}")
            return {
                "provider": "GCP",
                "current_spend": 0,
                "forecasted_spend": 0,
                "error": str(e)
            }
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get GCP compliance status"""
        try:
            compliance_data = {
                "provider": "GCP",
                "compliant_resources": 0,
                "non_compliant_resources": 0,
                "security_findings": 0,
                "compliance_score": 0,
                "violations": []
            }
            
            # Get Security Command Center findings
            try:
                org_name = f"organizations/{os.getenv('GCP_ORGANIZATION_ID', '123456789')}"
                source_name = f"{org_name}/sources/-"
                
                findings = self.security_center.list_findings(
                    request={
                        "parent": source_name,
                        "filter": 'state="ACTIVE"'
                    }
                )
                
                finding_count = 0
                for finding in findings:
                    finding_count += 1
                    if finding_count <= 10:  # Limit to first 10 violations
                        compliance_data["violations"].append({
                            "rule": finding.finding.category,
                            "resource": finding.finding.resource_name,
                            "severity": finding.finding.severity.name if hasattr(finding.finding, 'severity') else "MEDIUM"
                        })
                
                compliance_data["security_findings"] = finding_count
                compliance_data["non_compliant_resources"] = finding_count
                
                # Calculate compliance score (simple formula)
                total_resources = len(await self.get_resources())
                if total_resources > 0:
                    compliance_data["compliant_resources"] = max(0, total_resources - finding_count)
                    compliance_data["compliance_score"] = round(
                        (compliance_data["compliant_resources"] / total_resources) * 100, 2
                    )
                
            except Exception as e:
                logger.warning(f"Could not fetch Security Command Center findings: {e}")
            
            return compliance_data
            
        except Exception as e:
            logger.error(f"Failed to get GCP compliance status: {e}")
            return {
                "provider": "GCP",
                "compliance_score": 0,
                "error": str(e)
            }
    
    async def get_security_findings(self) -> List[Dict[str, Any]]:
        """Get GCP security findings"""
        findings = []
        
        try:
            org_name = f"organizations/{os.getenv('GCP_ORGANIZATION_ID', '123456789')}"
            source_name = f"{org_name}/sources/-"
            
            # Get Security Command Center findings
            scc_findings = self.security_center.list_findings(
                request={
                    "parent": source_name,
                    "filter": 'state="ACTIVE"',
                    "page_size": 100
                }
            )
            
            for finding_result in scc_findings:
                finding = finding_result.finding
                findings.append({
                    "id": finding.name,
                    "title": finding.category,
                    "description": finding.finding.get('description', '') if hasattr(finding, 'finding') else '',
                    "severity": finding.severity.name if hasattr(finding, 'severity') else "MEDIUM",
                    "provider": "GCP",
                    "service": "Security Command Center",
                    "resource": finding.resource_name,
                    "recommendation": finding.finding.get('recommendation', '') if hasattr(finding, 'finding') else '',
                    "created_at": finding.create_time.isoformat() if hasattr(finding, 'create_time') else None
                })
            
            return findings
            
        except Exception as e:
            logger.error(f"Failed to get GCP security findings: {e}")
            return []
    
    async def apply_governance_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply governance action to GCP resources"""
        try:
            action_type = action.get("type")
            resource_id = action.get("resource_id")
            
            if action_type == "label_resource":
                return await self._label_resource(resource_id, action.get("labels", {}))
            elif action_type == "stop_instance":
                return await self._stop_instance(resource_id)
            elif action_type == "delete_resource":
                return await self._delete_resource(resource_id)
            elif action_type == "enable_versioning":
                return await self._enable_bucket_versioning(resource_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
                
        except Exception as e:
            logger.error(f"Failed to apply GCP governance action: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Helper methods
    async def _label_resource(self, resource_id: str, labels: Dict[str, str]) -> Dict[str, Any]:
        """Apply labels to resource"""
        try:
            if "/instances/" in resource_id:  # Compute instance
                parts = resource_id.split('/')
                zone = parts[parts.index('zones') + 1]
                instance_name = parts[-1]
                
                instance = self.compute_client.get(
                    project=self.project_id,
                    zone=zone,
                    instance=instance_name
                )
                
                # Update labels
                instance.labels = {**instance.labels, **labels} if instance.labels else labels
                
                operation = self.compute_client.set_labels(
                    project=self.project_id,
                    zone=zone,
                    instance=instance_name,
                    instances_set_labels_request_resource={
                        "labels": instance.labels,
                        "label_fingerprint": instance.label_fingerprint
                    }
                )
                
                return {"success": True, "message": f"Labels applied to {resource_id}"}
                
            elif resource_id.startswith("gs://"):  # Storage bucket
                bucket_name = resource_id.replace("gs://", "")
                bucket = self.storage_client.bucket(bucket_name)
                bucket.labels = {**bucket.labels, **labels} if bucket.labels else labels
                bucket.patch()
                
                return {"success": True, "message": f"Labels applied to {resource_id}"}
            
            return {"success": False, "error": "Resource type not supported for labeling"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stop_instance(self, resource_id: str) -> Dict[str, Any]:
        """Stop compute instance"""
        try:
            parts = resource_id.split('/')
            zone = parts[parts.index('zones') + 1]
            instance_name = parts[-1]
            
            operation = self.compute_client.stop(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            
            return {"success": True, "message": f"Instance {instance_name} stopped"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _delete_resource(self, resource_id: str) -> Dict[str, Any]:
        """Delete resource (with safety checks)"""
        # Implement with appropriate safety checks
        return {"success": False, "error": "Delete operation requires additional confirmation"}
    
    async def _enable_bucket_versioning(self, resource_id: str) -> Dict[str, Any]:
        """Enable versioning on storage bucket"""
        try:
            if resource_id.startswith("gs://"):
                bucket_name = resource_id.replace("gs://", "")
                bucket = self.storage_client.bucket(bucket_name)
                bucket.versioning_enabled = True
                bucket.patch()
                
                return {"success": True, "message": f"Versioning enabled for {resource_id}"}
            
            return {"success": False, "error": "Resource is not a storage bucket"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}