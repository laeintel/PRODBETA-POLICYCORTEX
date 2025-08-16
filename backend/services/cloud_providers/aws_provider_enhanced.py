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
Enhanced AWS Cloud Provider Implementation for PolicyCortex
Complete end-to-end integration with AWS services
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AWSConfig:
    """AWS configuration"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    region: str = "us-west-2"
    account_id: Optional[str] = None
    role_arn: Optional[str] = None

class EnhancedAWSProvider:
    """Enhanced AWS cloud provider with full integration"""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        """Initialize AWS provider with configuration"""
        if config:
            self.config = config
        else:
            self.config = AWSConfig(
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                session_token=os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
                account_id=os.getenv("AWS_ACCOUNT_ID"),
                role_arn=os.getenv("AWS_ROLE_ARN")
            )
        
        self.session = None
        self.clients = {}
        self._initialized = False
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize AWS session and clients"""
        try:
            # Create session with credentials or use default chain
            session_kwargs = {"region_name": self.config.region}
            
            if self.config.access_key_id and self.config.secret_access_key:
                session_kwargs.update({
                    "aws_access_key_id": self.config.access_key_id,
                    "aws_secret_access_key": self.config.secret_access_key
                })
                if self.config.session_token:
                    session_kwargs["aws_session_token"] = self.config.session_token
            
            self.session = boto3.Session(**session_kwargs)
            
            # Assume role if specified
            if self.config.role_arn:
                sts = self.session.client('sts')
                assumed_role = sts.assume_role(
                    RoleArn=self.config.role_arn,
                    RoleSessionName='PolicyCortex'
                )
                
                credentials = assumed_role['Credentials']
                self.session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.config.region
                )
            
            # Get account ID if not provided
            if not self.config.account_id:
                sts = self.session.client('sts')
                self.config.account_id = sts.get_caller_identity()['Account']
            
            self._initialized = True
            logger.info(f"AWS provider initialized for account {self.config.account_id} in region {self.config.region}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize AWS session: {e}")
            self._initialized = False
    
    def _get_client(self, service: str):
        """Get or create AWS service client"""
        if not self._initialized:
            return None
            
        if service not in self.clients:
            try:
                self.clients[service] = self.session.client(service)
            except Exception as e:
                logger.error(f"Failed to create {service} client: {e}")
                return None
        
        return self.clients[service]
    
    async def get_resources(self, resource_type: Optional[str] = None, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get AWS resources with optional filtering"""
        if not self._initialized:
            return []
        
        resources = []
        tasks = []
        
        # Determine which resource types to fetch
        fetch_compute = not resource_type or resource_type in ["compute", "ec2", "all"]
        fetch_storage = not resource_type or resource_type in ["storage", "s3", "all"]
        fetch_database = not resource_type or resource_type in ["database", "rds", "all"]
        fetch_network = not resource_type or resource_type in ["network", "vpc", "all"]
        fetch_containers = not resource_type or resource_type in ["containers", "ecs", "eks", "all"]
        
        # Fetch resources in parallel
        if fetch_compute:
            tasks.append(self._get_ec2_instances(filters))
            tasks.append(self._get_lambda_functions(filters))
        
        if fetch_storage:
            tasks.append(self._get_s3_buckets(filters))
            tasks.append(self._get_ebs_volumes(filters))
        
        if fetch_database:
            tasks.append(self._get_rds_instances(filters))
            tasks.append(self._get_dynamodb_tables(filters))
        
        if fetch_network:
            tasks.append(self._get_vpcs(filters))
            tasks.append(self._get_load_balancers(filters))
        
        if fetch_containers:
            tasks.append(self._get_ecs_clusters(filters))
            tasks.append(self._get_eks_clusters(filters))
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    resources.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Error fetching resources: {result}")
        
        return resources
    
    async def _get_ec2_instances(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get EC2 instances"""
        resources = []
        ec2 = self._get_client('ec2')
        if not ec2:
            return resources
        
        try:
            paginator = ec2.get_paginator('describe_instances')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for reservation in page.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        # Apply filters if provided
                        if filters:
                            if 'state' in filters and instance['State']['Name'] != filters['state']:
                                continue
                            if 'tags' in filters:
                                instance_tags = {t['Key']: t['Value'] for t in instance.get('Tags', [])}
                                if not all(k in instance_tags and instance_tags[k] == v for k, v in filters['tags'].items()):
                                    continue
                        
                        resources.append({
                            "id": instance['InstanceId'],
                            "name": self._get_tag_value(instance.get('Tags', []), 'Name') or instance['InstanceId'],
                            "type": "AWS::EC2::Instance",
                            "provider": "AWS",
                            "region": self.config.region,
                            "state": instance['State']['Name'],
                            "tags": {t['Key']: t['Value'] for t in instance.get('Tags', [])},
                            "metadata": {
                                "instance_type": instance.get('InstanceType'),
                                "launch_time": instance.get('LaunchTime').isoformat() if instance.get('LaunchTime') else None,
                                "private_ip": instance.get('PrivateIpAddress'),
                                "public_ip": instance.get('PublicIpAddress'),
                                "vpc_id": instance.get('VpcId'),
                                "subnet_id": instance.get('SubnetId'),
                                "security_groups": [sg['GroupId'] for sg in instance.get('SecurityGroups', [])]
                            },
                            "cost_data": {
                                "hourly_cost": self._estimate_ec2_cost(instance.get('InstanceType')),
                                "state_since": instance.get('LaunchTime').isoformat() if instance.get('LaunchTime') else None
                            }
                        })
        except Exception as e:
            logger.error(f"Failed to get EC2 instances: {e}")
        
        return resources
    
    async def _get_s3_buckets(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get S3 buckets"""
        resources = []
        s3 = self._get_client('s3')
        if not s3:
            return resources
        
        try:
            response = s3.list_buckets()
            
            for bucket in response.get('Buckets', []):
                bucket_name = bucket['Name']
                
                # Get bucket details in parallel
                try:
                    # Get bucket location
                    location = s3.get_bucket_location(Bucket=bucket_name)
                    region = location.get('LocationConstraint', 'us-east-1') or 'us-east-1'
                    
                    # Get bucket tags
                    tags = {}
                    try:
                        tag_response = s3.get_bucket_tagging(Bucket=bucket_name)
                        tags = {t['Key']: t['Value'] for t in tag_response.get('TagSet', [])}
                    except ClientError:
                        pass
                    
                    # Get bucket size and object count
                    size_data = await self._get_bucket_size(bucket_name)
                    
                    # Get bucket encryption
                    encryption = "None"
                    try:
                        enc_response = s3.get_bucket_encryption(Bucket=bucket_name)
                        if enc_response.get('ServerSideEncryptionConfiguration'):
                            encryption = "Enabled"
                    except ClientError:
                        pass
                    
                    # Get bucket versioning
                    versioning = "Disabled"
                    try:
                        ver_response = s3.get_bucket_versioning(Bucket=bucket_name)
                        versioning = ver_response.get('Status', 'Disabled')
                    except ClientError:
                        pass
                    
                    resources.append({
                        "id": f"arn:aws:s3:::{bucket_name}",
                        "name": bucket_name,
                        "type": "AWS::S3::Bucket",
                        "provider": "AWS",
                        "region": region,
                        "state": "active",
                        "tags": tags,
                        "metadata": {
                            "creation_date": bucket.get('CreationDate').isoformat() if bucket.get('CreationDate') else None,
                            "encryption": encryption,
                            "versioning": versioning,
                            "size_bytes": size_data['size'],
                            "object_count": size_data['count']
                        },
                        "cost_data": {
                            "storage_cost": self._estimate_s3_cost(size_data['size']),
                            "request_cost": 0  # Would need CloudWatch metrics for accurate request costs
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Error getting details for bucket {bucket_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to get S3 buckets: {e}")
        
        return resources
    
    async def _get_bucket_size(self, bucket_name: str) -> Dict[str, int]:
        """Get S3 bucket size and object count"""
        cloudwatch = self._get_client('cloudwatch')
        if not cloudwatch:
            return {"size": 0, "count": 0}
        
        try:
            # Get bucket size
            size_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=datetime.now() - timedelta(days=1),
                EndTime=datetime.now(),
                Period=86400,
                Statistics=['Average']
            )
            
            size = 0
            if size_response['Datapoints']:
                size = int(size_response['Datapoints'][0]['Average'])
            
            # Get object count
            count_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='NumberOfObjects',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=datetime.now() - timedelta(days=1),
                EndTime=datetime.now(),
                Period=86400,
                Statistics=['Average']
            )
            
            count = 0
            if count_response['Datapoints']:
                count = int(count_response['Datapoints'][0]['Average'])
            
            return {"size": size, "count": count}
            
        except Exception as e:
            logger.warning(f"Could not get size for bucket {bucket_name}: {e}")
            return {"size": 0, "count": 0}
    
    async def _get_lambda_functions(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get Lambda functions"""
        resources = []
        lambda_client = self._get_client('lambda')
        if not lambda_client:
            return resources
        
        try:
            paginator = lambda_client.get_paginator('list_functions')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for function in page.get('Functions', []):
                    resources.append({
                        "id": function['FunctionArn'],
                        "name": function['FunctionName'],
                        "type": "AWS::Lambda::Function",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": "active",
                        "tags": {},  # Would need separate API call for tags
                        "metadata": {
                            "runtime": function.get('Runtime'),
                            "handler": function.get('Handler'),
                            "code_size": function.get('CodeSize'),
                            "memory_size": function.get('MemorySize'),
                            "timeout": function.get('Timeout'),
                            "last_modified": function.get('LastModified')
                        },
                        "cost_data": {
                            "invocations_per_month": 0,  # Would need CloudWatch metrics
                            "estimated_cost": 0
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to get Lambda functions: {e}")
        
        return resources
    
    async def _get_rds_instances(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get RDS instances"""
        resources = []
        rds = self._get_client('rds')
        if not rds:
            return resources
        
        try:
            paginator = rds.get_paginator('describe_db_instances')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for db in page.get('DBInstances', []):
                    resources.append({
                        "id": db['DBInstanceArn'],
                        "name": db['DBInstanceIdentifier'],
                        "type": "AWS::RDS::DBInstance",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": db['DBInstanceStatus'],
                        "tags": {t['Key']: t['Value'] for t in db.get('TagList', [])},
                        "metadata": {
                            "engine": db.get('Engine'),
                            "engine_version": db.get('EngineVersion'),
                            "instance_class": db.get('DBInstanceClass'),
                            "allocated_storage": db.get('AllocatedStorage'),
                            "multi_az": db.get('MultiAZ'),
                            "publicly_accessible": db.get('PubliclyAccessible'),
                            "backup_retention_period": db.get('BackupRetentionPeriod'),
                            "encryption_enabled": db.get('StorageEncrypted')
                        },
                        "cost_data": {
                            "hourly_cost": self._estimate_rds_cost(db.get('DBInstanceClass')),
                            "storage_cost": db.get('AllocatedStorage', 0) * 0.115  # Approximate $/GB/month
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to get RDS instances: {e}")
        
        return resources
    
    async def _get_dynamodb_tables(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get DynamoDB tables"""
        resources = []
        dynamodb = self._get_client('dynamodb')
        if not dynamodb:
            return resources
        
        try:
            paginator = dynamodb.get_paginator('list_tables')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for table_name in page.get('TableNames', []):
                    try:
                        table = dynamodb.describe_table(TableName=table_name)['Table']
                        
                        resources.append({
                            "id": table['TableArn'],
                            "name": table['TableName'],
                            "type": "AWS::DynamoDB::Table",
                            "provider": "AWS",
                            "region": self.config.region,
                            "state": table['TableStatus'],
                            "tags": {},  # Would need separate API call for tags
                            "metadata": {
                                "item_count": table.get('ItemCount'),
                                "size_bytes": table.get('TableSizeBytes'),
                                "billing_mode": table.get('BillingModeSummary', {}).get('BillingMode'),
                                "creation_date": table.get('CreationDateTime').isoformat() if table.get('CreationDateTime') else None
                            },
                            "cost_data": {
                                "estimated_monthly_cost": self._estimate_dynamodb_cost(table)
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Error describing table {table_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to get DynamoDB tables: {e}")
        
        return resources
    
    async def _get_ebs_volumes(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get EBS volumes"""
        resources = []
        ec2 = self._get_client('ec2')
        if not ec2:
            return resources
        
        try:
            paginator = ec2.get_paginator('describe_volumes')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for volume in page.get('Volumes', []):
                    resources.append({
                        "id": volume['VolumeId'],
                        "name": self._get_tag_value(volume.get('Tags', []), 'Name') or volume['VolumeId'],
                        "type": "AWS::EC2::Volume",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": volume['State'],
                        "tags": {t['Key']: t['Value'] for t in volume.get('Tags', [])},
                        "metadata": {
                            "size_gb": volume.get('Size'),
                            "volume_type": volume.get('VolumeType'),
                            "iops": volume.get('Iops'),
                            "encrypted": volume.get('Encrypted'),
                            "create_time": volume.get('CreateTime').isoformat() if volume.get('CreateTime') else None,
                            "attachments": [a['InstanceId'] for a in volume.get('Attachments', [])]
                        },
                        "cost_data": {
                            "monthly_cost": self._estimate_ebs_cost(volume.get('VolumeType'), volume.get('Size', 0))
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to get EBS volumes: {e}")
        
        return resources
    
    async def _get_vpcs(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get VPCs"""
        resources = []
        ec2 = self._get_client('ec2')
        if not ec2:
            return resources
        
        try:
            response = ec2.describe_vpcs()
            
            for vpc in response.get('Vpcs', []):
                # Get subnets for this VPC
                subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc['VpcId']]}])
                subnet_count = len(subnets.get('Subnets', []))
                
                resources.append({
                    "id": vpc['VpcId'],
                    "name": self._get_tag_value(vpc.get('Tags', []), 'Name') or vpc['VpcId'],
                    "type": "AWS::EC2::VPC",
                    "provider": "AWS",
                    "region": self.config.region,
                    "state": vpc['State'],
                    "tags": {t['Key']: t['Value'] for t in vpc.get('Tags', [])},
                    "metadata": {
                        "cidr_block": vpc.get('CidrBlock'),
                        "is_default": vpc.get('IsDefault'),
                        "subnet_count": subnet_count,
                        "enable_dns_support": vpc.get('EnableDnsSupport'),
                        "enable_dns_hostnames": vpc.get('EnableDnsHostnames')
                    }
                })
        except Exception as e:
            logger.error(f"Failed to get VPCs: {e}")
        
        return resources
    
    async def _get_load_balancers(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get Load Balancers (ALB/NLB)"""
        resources = []
        elbv2 = self._get_client('elbv2')
        if not elbv2:
            return resources
        
        try:
            paginator = elbv2.get_paginator('describe_load_balancers')
            page_iterator = paginator.paginate()
            
            for page in page_iterator:
                for lb in page.get('LoadBalancers', []):
                    resources.append({
                        "id": lb['LoadBalancerArn'],
                        "name": lb['LoadBalancerName'],
                        "type": f"AWS::ElasticLoadBalancingV2::{lb['Type']}",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": lb['State']['Code'],
                        "tags": {},  # Would need separate API call for tags
                        "metadata": {
                            "dns_name": lb.get('DNSName'),
                            "scheme": lb.get('Scheme'),
                            "vpc_id": lb.get('VpcId'),
                            "type": lb.get('Type'),
                            "created_time": lb.get('CreatedTime').isoformat() if lb.get('CreatedTime') else None
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to get load balancers: {e}")
        
        return resources
    
    async def _get_ecs_clusters(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get ECS clusters"""
        resources = []
        ecs = self._get_client('ecs')
        if not ecs:
            return resources
        
        try:
            cluster_arns = ecs.list_clusters()['clusterArns']
            
            if cluster_arns:
                clusters = ecs.describe_clusters(clusters=cluster_arns)
                
                for cluster in clusters.get('clusters', []):
                    resources.append({
                        "id": cluster['clusterArn'],
                        "name": cluster['clusterName'],
                        "type": "AWS::ECS::Cluster",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": cluster['status'],
                        "tags": cluster.get('tags', []),
                        "metadata": {
                            "running_tasks_count": cluster.get('runningTasksCount'),
                            "pending_tasks_count": cluster.get('pendingTasksCount'),
                            "active_services_count": cluster.get('activeServicesCount'),
                            "registered_container_instances_count": cluster.get('registeredContainerInstancesCount')
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to get ECS clusters: {e}")
        
        return resources
    
    async def _get_eks_clusters(self, filters: Optional[Dict]) -> List[Dict[str, Any]]:
        """Get EKS clusters"""
        resources = []
        eks = self._get_client('eks')
        if not eks:
            return resources
        
        try:
            clusters = eks.list_clusters()
            
            for cluster_name in clusters.get('clusters', []):
                try:
                    cluster = eks.describe_cluster(name=cluster_name)['cluster']
                    
                    resources.append({
                        "id": cluster['arn'],
                        "name": cluster['name'],
                        "type": "AWS::EKS::Cluster",
                        "provider": "AWS",
                        "region": self.config.region,
                        "state": cluster['status'],
                        "tags": cluster.get('tags', {}),
                        "metadata": {
                            "version": cluster.get('version'),
                            "endpoint": cluster.get('endpoint'),
                            "role_arn": cluster.get('roleArn'),
                            "created_at": cluster.get('createdAt').isoformat() if cluster.get('createdAt') else None
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error describing EKS cluster {cluster_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to get EKS clusters: {e}")
        
        return resources
    
    # Cost estimation methods
    def _estimate_ec2_cost(self, instance_type: str) -> float:
        """Estimate EC2 instance hourly cost"""
        # Simplified cost estimates - in production, use AWS Pricing API
        cost_map = {
            "t2.micro": 0.0116, "t2.small": 0.023, "t2.medium": 0.0464,
            "t3.micro": 0.0104, "t3.small": 0.0208, "t3.medium": 0.0416,
            "m5.large": 0.096, "m5.xlarge": 0.192, "m5.2xlarge": 0.384,
            "c5.large": 0.085, "c5.xlarge": 0.17, "c5.2xlarge": 0.34
        }
        return cost_map.get(instance_type, 0.1)  # Default to $0.10/hour
    
    def _estimate_s3_cost(self, size_bytes: int) -> float:
        """Estimate S3 storage monthly cost"""
        size_gb = size_bytes / (1024 ** 3)
        return size_gb * 0.023  # $0.023 per GB for standard storage
    
    def _estimate_rds_cost(self, instance_class: str) -> float:
        """Estimate RDS instance hourly cost"""
        cost_map = {
            "db.t2.micro": 0.017, "db.t2.small": 0.034, "db.t2.medium": 0.068,
            "db.t3.micro": 0.017, "db.t3.small": 0.034, "db.t3.medium": 0.068,
            "db.m5.large": 0.171, "db.m5.xlarge": 0.342, "db.m5.2xlarge": 0.684
        }
        return cost_map.get(instance_class, 0.1)
    
    def _estimate_ebs_cost(self, volume_type: str, size_gb: int) -> float:
        """Estimate EBS volume monthly cost"""
        cost_per_gb = {
            "gp2": 0.10, "gp3": 0.08, "io1": 0.125, "io2": 0.125,
            "st1": 0.045, "sc1": 0.025, "standard": 0.05
        }
        return size_gb * cost_per_gb.get(volume_type, 0.10)
    
    def _estimate_dynamodb_cost(self, table: Dict) -> float:
        """Estimate DynamoDB monthly cost"""
        # Simplified - actual cost depends on read/write capacity units
        if table.get('BillingModeSummary', {}).get('BillingMode') == 'PAY_PER_REQUEST':
            return 5.0  # Base estimate for on-demand
        else:
            return 10.0  # Base estimate for provisioned
    
    def _get_tag_value(self, tags: List[Dict], key: str) -> str:
        """Extract tag value by key"""
        for tag in tags:
            if tag.get('Key') == key:
                return tag.get('Value', '')
        return ''
    
    def is_configured(self) -> bool:
        """Check if AWS provider is properly configured"""
        return self._initialized
    
    def get_account_info(self) -> Dict[str, str]:
        """Get AWS account information"""
        if not self._initialized:
            return {}
        
        return {
            "account_id": self.config.account_id,
            "region": self.config.region,
            "provider": "AWS"
        }