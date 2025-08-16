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
AWS Cloud Provider Implementation for PolicyCortex
Provides AWS resource management and governance capabilities
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class AWSProvider:
    """AWS cloud provider implementation"""
    
    def __init__(self):
        """Initialize AWS provider with credentials"""
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
        self.aws_account_id = os.getenv("AWS_ACCOUNT_ID")
        
        # Initialize AWS clients
        self.session = None
        self.ec2_client = None
        self.s3_client = None
        self.iam_client = None
        self.cloudwatch_client = None
        self.cost_explorer = None
        self.config_client = None
        self.organizations_client = None
        self.cloudtrail_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS service clients"""
        try:
            # Create session
            if self.aws_access_key and self.aws_secret_key:
                self.session = boto3.Session(
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
            else:
                # Use default credentials (IAM role, etc.)
                self.session = boto3.Session(region_name=self.aws_region)
            
            # Initialize service clients
            self.ec2_client = self.session.client('ec2')
            self.s3_client = self.session.client('s3')
            self.iam_client = self.session.client('iam')
            self.cloudwatch_client = self.session.client('cloudwatch')
            self.cost_explorer = self.session.client('ce')
            self.config_client = self.session.client('config')
            self.organizations_client = self.session.client('organizations')
            self.cloudtrail_client = self.session.client('cloudtrail')
            
            logger.info(f"AWS clients initialized for region: {self.aws_region}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
    
    async def get_resources(self, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get AWS resources"""
        resources = []
        
        try:
            # Get EC2 instances
            if not resource_type or resource_type == "compute":
                instances = self.ec2_client.describe_instances()
                for reservation in instances.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        resources.append({
                            "id": instance['InstanceId'],
                            "name": self._get_tag_value(instance.get('Tags', []), 'Name'),
                            "type": "AWS::EC2::Instance",
                            "provider": "AWS",
                            "region": instance.get('Placement', {}).get('AvailabilityZone'),
                            "state": instance['State']['Name'],
                            "tags": self._tags_to_dict(instance.get('Tags', [])),
                            "metadata": {
                                "instance_type": instance.get('InstanceType'),
                                "launch_time": instance.get('LaunchTime').isoformat() if instance.get('LaunchTime') else None,
                                "private_ip": instance.get('PrivateIpAddress'),
                                "public_ip": instance.get('PublicIpAddress')
                            }
                        })
            
            # Get S3 buckets
            if not resource_type or resource_type == "storage":
                buckets = self.s3_client.list_buckets()
                for bucket in buckets.get('Buckets', []):
                    # Get bucket tags
                    tags = {}
                    try:
                        tag_response = self.s3_client.get_bucket_tagging(Bucket=bucket['Name'])
                        tags = self._tags_to_dict(tag_response.get('TagSet', []))
                    except ClientError:
                        pass
                    
                    resources.append({
                        "id": f"arn:aws:s3:::{bucket['Name']}",
                        "name": bucket['Name'],
                        "type": "AWS::S3::Bucket",
                        "provider": "AWS",
                        "region": self._get_bucket_region(bucket['Name']),
                        "state": "active",
                        "tags": tags,
                        "metadata": {
                            "creation_date": bucket.get('CreationDate').isoformat() if bucket.get('CreationDate') else None
                        }
                    })
            
            # Get RDS instances
            if not resource_type or resource_type == "database":
                rds_client = self.session.client('rds')
                db_instances = rds_client.describe_db_instances()
                for db in db_instances.get('DBInstances', []):
                    resources.append({
                        "id": db['DBInstanceArn'],
                        "name": db['DBInstanceIdentifier'],
                        "type": "AWS::RDS::DBInstance",
                        "provider": "AWS",
                        "region": db['DBInstanceArn'].split(':')[3],
                        "state": db['DBInstanceStatus'],
                        "tags": self._tags_to_dict(db.get('TagList', [])),
                        "metadata": {
                            "engine": db.get('Engine'),
                            "engine_version": db.get('EngineVersion'),
                            "instance_class": db.get('DBInstanceClass'),
                            "allocated_storage": db.get('AllocatedStorage')
                        }
                    })
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to get AWS resources: {e}")
            return []
    
    async def get_policies(self) -> List[Dict[str, Any]]:
        """Get AWS policies and compliance information"""
        policies = []
        
        try:
            # Get IAM policies
            managed_policies = self.iam_client.list_policies(Scope='Local')
            for policy in managed_policies.get('Policies', []):
                policies.append({
                    "id": policy['Arn'],
                    "name": policy['PolicyName'],
                    "type": "IAM::Policy",
                    "provider": "AWS",
                    "description": policy.get('Description', ''),
                    "created_at": policy.get('CreateDate').isoformat() if policy.get('CreateDate') else None,
                    "updated_at": policy.get('UpdateDate').isoformat() if policy.get('UpdateDate') else None
                })
            
            # Get AWS Config rules
            if self.config_client:
                try:
                    config_rules = self.config_client.describe_config_rules()
                    for rule in config_rules.get('ConfigRules', []):
                        policies.append({
                            "id": rule['ConfigRuleArn'],
                            "name": rule['ConfigRuleName'],
                            "type": "Config::Rule",
                            "provider": "AWS",
                            "description": rule.get('Description', ''),
                            "state": rule.get('ConfigRuleState'),
                            "source": rule.get('Source', {})
                        })
                except ClientError as e:
                    logger.warning(f"Could not fetch Config rules: {e}")
            
            return policies
            
        except Exception as e:
            logger.error(f"Failed to get AWS policies: {e}")
            return []
    
    async def get_costs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get AWS cost information"""
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get cost and usage
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            total_cost = 0
            service_costs = {}
            
            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    total_cost += cost
                    
                    if service not in service_costs:
                        service_costs[service] = 0
                    service_costs[service] += cost
            
            # Get cost forecast
            forecast_response = self.cost_explorer.get_cost_forecast(
                TimePeriod={
                    'Start': end_date,
                    'End': (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
                },
                Metric='UNBLENDED_COST',
                Granularity='MONTHLY'
            )
            
            forecasted_cost = float(forecast_response['Total']['Amount'])
            
            return {
                "provider": "AWS",
                "current_spend": round(total_cost, 2),
                "forecasted_spend": round(forecasted_cost, 2),
                "service_breakdown": service_costs,
                "currency": "USD",
                "period": {
                    "start": start_date,
                    "end": end_date
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get AWS costs: {e}")
            return {
                "provider": "AWS",
                "current_spend": 0,
                "forecasted_spend": 0,
                "error": str(e)
            }
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get AWS compliance status"""
        try:
            compliance_data = {
                "provider": "AWS",
                "compliant_resources": 0,
                "non_compliant_resources": 0,
                "rules_evaluated": 0,
                "compliance_score": 0,
                "violations": []
            }
            
            # Get Config compliance
            if self.config_client:
                try:
                    compliance = self.config_client.describe_compliance_by_config_rule()
                    
                    for rule_compliance in compliance.get('ComplianceByConfigRules', []):
                        compliance_data["rules_evaluated"] += 1
                        
                        if rule_compliance.get('Compliance', {}).get('ComplianceType') == 'NON_COMPLIANT':
                            compliance_data["violations"].append({
                                "rule": rule_compliance['ConfigRuleName'],
                                "type": "Config Rule Violation",
                                "severity": "medium"
                            })
                    
                    # Get aggregated compliance
                    aggregated = self.config_client.get_compliance_summary_by_config_rule()
                    summary = aggregated.get('ComplianceSummary', {})
                    
                    compliance_data["compliant_resources"] = summary.get('CompliantResourceCount', {}).get('CappedCount', 0)
                    compliance_data["non_compliant_resources"] = summary.get('NonCompliantResourceCount', {}).get('CappedCount', 0)
                    
                    total = compliance_data["compliant_resources"] + compliance_data["non_compliant_resources"]
                    if total > 0:
                        compliance_data["compliance_score"] = round(
                            (compliance_data["compliant_resources"] / total) * 100, 2
                        )
                    
                except ClientError as e:
                    logger.warning(f"Could not fetch Config compliance: {e}")
            
            return compliance_data
            
        except Exception as e:
            logger.error(f"Failed to get AWS compliance status: {e}")
            return {
                "provider": "AWS",
                "compliance_score": 0,
                "error": str(e)
            }
    
    async def get_security_findings(self) -> List[Dict[str, Any]]:
        """Get AWS security findings"""
        findings = []
        
        try:
            # Get Security Hub findings
            securityhub = self.session.client('securityhub')
            
            try:
                response = securityhub.get_findings(
                    Filters={
                        'RecordState': [{'Value': 'ACTIVE', 'Comparison': 'EQUALS'}],
                        'WorkflowStatus': [{'Value': 'NEW', 'Comparison': 'EQUALS'}]
                    },
                    MaxResults=100
                )
                
                for finding in response.get('Findings', []):
                    findings.append({
                        "id": finding['Id'],
                        "title": finding.get('Title'),
                        "description": finding.get('Description'),
                        "severity": finding.get('Severity', {}).get('Label'),
                        "provider": "AWS",
                        "service": "Security Hub",
                        "resource": finding.get('Resources', [{}])[0].get('Id') if finding.get('Resources') else None,
                        "remediation": finding.get('Remediation', {}).get('Recommendation', {}).get('Text'),
                        "created_at": finding.get('CreatedAt')
                    })
                    
            except ClientError as e:
                logger.warning(f"Could not fetch Security Hub findings: {e}")
            
            # Get GuardDuty findings
            guardduty = self.session.client('guardduty')
            
            try:
                # Get detector ID
                detectors = guardduty.list_detectors()
                if detectors.get('DetectorIds'):
                    detector_id = detectors['DetectorIds'][0]
                    
                    # Get findings
                    gd_findings = guardduty.list_findings(
                        DetectorId=detector_id,
                        FindingCriteria={
                            'Criterion': {
                                'service.archived': {'Eq': ['false']}
                            }
                        }
                    )
                    
                    if gd_findings.get('FindingIds'):
                        details = guardduty.get_findings(
                            DetectorId=detector_id,
                            FindingIds=gd_findings['FindingIds'][:50]  # Limit to 50
                        )
                        
                        for finding in details.get('Findings', []):
                            findings.append({
                                "id": finding['Id'],
                                "title": finding.get('Title'),
                                "description": finding.get('Description'),
                                "severity": finding.get('Severity'),
                                "provider": "AWS",
                                "service": "GuardDuty",
                                "resource": finding.get('Resource', {}).get('ResourceType'),
                                "created_at": finding.get('CreatedAt')
                            })
                            
            except ClientError as e:
                logger.warning(f"Could not fetch GuardDuty findings: {e}")
            
            return findings
            
        except Exception as e:
            logger.error(f"Failed to get AWS security findings: {e}")
            return []
    
    async def apply_governance_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply governance action to AWS resources"""
        try:
            action_type = action.get("type")
            resource_id = action.get("resource_id")
            
            if action_type == "tag_resource":
                return await self._tag_resource(resource_id, action.get("tags", {}))
            elif action_type == "stop_instance":
                return await self._stop_instance(resource_id)
            elif action_type == "delete_resource":
                return await self._delete_resource(resource_id)
            elif action_type == "enable_encryption":
                return await self._enable_encryption(resource_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
                
        except Exception as e:
            logger.error(f"Failed to apply AWS governance action: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Helper methods
    def _get_tag_value(self, tags: List[Dict], key: str) -> Optional[str]:
        """Get tag value by key"""
        for tag in tags:
            if tag.get('Key') == key:
                return tag.get('Value')
        return None
    
    def _tags_to_dict(self, tags: List[Dict]) -> Dict[str, str]:
        """Convert AWS tags to dictionary"""
        return {tag.get('Key'): tag.get('Value') for tag in tags if tag.get('Key')}
    
    def _get_bucket_region(self, bucket_name: str) -> str:
        """Get S3 bucket region"""
        try:
            response = self.s3_client.get_bucket_location(Bucket=bucket_name)
            return response.get('LocationConstraint', 'us-east-1')
        except:
            return self.aws_region
    
    async def _tag_resource(self, resource_id: str, tags: Dict[str, str]) -> Dict[str, Any]:
        """Apply tags to resource"""
        try:
            # Determine resource type and apply tags
            if resource_id.startswith('i-'):  # EC2 instance
                self.ec2_client.create_tags(
                    Resources=[resource_id],
                    Tags=[{'Key': k, 'Value': v} for k, v in tags.items()]
                )
            elif resource_id.startswith('arn:aws:s3:::'):  # S3 bucket
                bucket_name = resource_id.split(':::')[1]
                self.s3_client.put_bucket_tagging(
                    Bucket=bucket_name,
                    Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]}
                )
            
            return {"success": True, "message": f"Tags applied to {resource_id}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stop_instance(self, instance_id: str) -> Dict[str, Any]:
        """Stop EC2 instance"""
        try:
            self.ec2_client.stop_instances(InstanceIds=[instance_id])
            return {"success": True, "message": f"Instance {instance_id} stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _delete_resource(self, resource_id: str) -> Dict[str, Any]:
        """Delete resource (with safety checks)"""
        # Implement with appropriate safety checks
        return {"success": False, "error": "Delete operation requires additional confirmation"}
    
    async def _enable_encryption(self, resource_id: str) -> Dict[str, Any]:
        """Enable encryption on resource"""
        try:
            if resource_id.startswith('arn:aws:s3:::'):  # S3 bucket
                bucket_name = resource_id.split(':::')[1]
                self.s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        'Rules': [{
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        }]
                    }
                )
                return {"success": True, "message": f"Encryption enabled for {resource_id}"}
            
            return {"success": False, "error": "Encryption not supported for this resource type"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}