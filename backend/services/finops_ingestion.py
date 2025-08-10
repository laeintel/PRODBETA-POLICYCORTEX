"""
Real FinOps Data Ingestion Service for PolicyCortex
Ingests and processes financial operations data from multiple cloud providers
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

# Set up logger first
logger = logging.getLogger(__name__)

# Azure imports
try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.costmanagement import CostManagementClient
    from azure.mgmt.consumption import ConsumptionManagementClient
    from azure.mgmt.billing import BillingManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure FinOps libraries not available")

# AWS imports
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("AWS FinOps libraries not available")

# GCP imports
try:
    from google.cloud import bigquery
    from google.cloud import billing_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logger.warning("GCP FinOps libraries not available")


@dataclass
class CostData:
    """Cost data structure"""
    provider: str
    service: str
    resource_id: Optional[str]
    resource_name: Optional[str]
    resource_type: Optional[str]
    region: str
    cost: Decimal
    currency: str
    usage_quantity: Decimal
    usage_unit: str
    date: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class BudgetData:
    """Budget data structure"""
    provider: str
    budget_name: str
    budget_amount: Decimal
    spent_amount: Decimal
    remaining_amount: Decimal
    percentage_used: float
    currency: str
    period_start: datetime
    period_end: datetime
    alerts: List[Dict[str, Any]]

@dataclass
class SavingsRecommendation:
    """Savings recommendation structure"""
    provider: str
    recommendation_type: str
    resource_id: Optional[str]
    description: str
    estimated_savings: Decimal
    currency: str
    impact: str  # High, Medium, Low
    effort: str  # High, Medium, Low
    confidence: float
    actions: List[str]

class FinOpsIngestion:
    """Real FinOps data ingestion service"""
    
    def __init__(self):
        """Initialize FinOps ingestion service"""
        self.azure_credential = None
        self.aws_session = None
        self.gcp_client = None
        self.cost_data_cache = []
        self.budget_data_cache = []
        self.recommendations_cache = []
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize cloud provider clients"""
        # Azure
        if os.getenv("AZURE_SUBSCRIPTION_ID") and AZURE_AVAILABLE:
            try:
                self.azure_credential = DefaultAzureCredential()
                self.azure_subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
                self.azure_cost_client = CostManagementClient(
                    self.azure_credential,
                    self.azure_subscription_id
                )
                self.azure_consumption_client = ConsumptionManagementClient(
                    self.azure_credential,
                    self.azure_subscription_id
                )
                logger.info("Azure FinOps clients initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure FinOps clients: {e}")
        
        # AWS
        if os.getenv("AWS_ACCESS_KEY_ID") and AWS_AVAILABLE:
            try:
                self.aws_session = boto3.Session(
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2")
                )
                self.aws_ce_client = self.aws_session.client('ce')  # Cost Explorer
                self.aws_budgets_client = self.aws_session.client('budgets')
                logger.info("AWS FinOps clients initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AWS FinOps clients: {e}")
        
        # GCP
        if os.getenv("GCP_PROJECT_ID") and GCP_AVAILABLE:
            try:
                self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
                self.gcp_bq_client = bigquery.Client(project=self.gcp_project_id)
                # Note: billing_budgets_v1 is not available, using billing_v1 only
                logger.info("GCP FinOps clients initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GCP FinOps clients: {e}")
    
    async def ingest_all_costs(self, start_date: datetime, end_date: datetime) -> List[CostData]:
        """Ingest cost data from all providers"""
        tasks = []
        
        if self.azure_credential:
            tasks.append(self.ingest_azure_costs(start_date, end_date))
        
        if self.aws_session:
            tasks.append(self.ingest_aws_costs(start_date, end_date))
        
        if self.gcp_bq_client:
            tasks.append(self.ingest_gcp_costs(start_date, end_date))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_costs = []
            for result in results:
                if isinstance(result, list):
                    all_costs.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error ingesting costs: {result}")
            
            self.cost_data_cache = all_costs
            return all_costs
        
        return []
    
    async def ingest_azure_costs(self, start_date: datetime, end_date: datetime) -> List[CostData]:
        """Ingest Azure cost data"""
        costs = []
        
        try:
            # Query cost management API
            scope = f"/subscriptions/{self.azure_subscription_id}"
            
            query = {
                "type": "Usage",
                "timeframe": "Custom",
                "time_period": {
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                },
                "dataset": {
                    "granularity": "Daily",
                    "aggregation": {
                        "totalCost": {
                            "name": "Cost",
                            "function": "Sum"
                        },
                        "totalQuantity": {
                            "name": "Quantity",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {"type": "Dimension", "name": "ServiceName"},
                        {"type": "Dimension", "name": "ResourceLocation"},
                        {"type": "Dimension", "name": "ResourceId"}
                    ]
                }
            }
            
            result = self.azure_cost_client.query.usage(scope, query)
            
            for row in result.rows:
                cost_data = CostData(
                    provider="Azure",
                    service=row[0] if len(row) > 0 else "Unknown",
                    resource_id=row[2] if len(row) > 2 else None,
                    resource_name=None,
                    resource_type=None,
                    region=row[1] if len(row) > 1 else "Unknown",
                    cost=Decimal(str(row[3])) if len(row) > 3 else Decimal("0"),
                    currency="USD",
                    usage_quantity=Decimal(str(row[4])) if len(row) > 4 else Decimal("0"),
                    usage_unit="Units",
                    date=datetime.fromisoformat(row[5]) if len(row) > 5 else datetime.now(),
                    tags={},
                    metadata={}
                )
                costs.append(cost_data)
            
            logger.info(f"Ingested {len(costs)} Azure cost records")
            
        except Exception as e:
            logger.error(f"Failed to ingest Azure costs: {e}")
        
        return costs
    
    async def ingest_aws_costs(self, start_date: datetime, end_date: datetime) -> List[CostData]:
        """Ingest AWS cost data"""
        costs = []
        
        try:
            response = self.aws_ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'},
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
                ]
            )
            
            for result in response.get('ResultsByTime', []):
                date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
                
                for group in result.get('Groups', []):
                    keys = group['Keys']
                    metrics = group['Metrics']
                    
                    cost_data = CostData(
                        provider="AWS",
                        service=keys[0] if len(keys) > 0 else "Unknown",
                        resource_id=None,
                        resource_name=None,
                        resource_type=keys[2] if len(keys) > 2 else None,
                        region=keys[1] if len(keys) > 1 else "Unknown",
                        cost=Decimal(metrics['UnblendedCost']['Amount']),
                        currency=metrics['UnblendedCost']['Unit'],
                        usage_quantity=Decimal(metrics.get('UsageQuantity', {}).get('Amount', '0')),
                        usage_unit=metrics.get('UsageQuantity', {}).get('Unit', 'Units'),
                        date=date,
                        tags={},
                        metadata={}
                    )
                    costs.append(cost_data)
            
            logger.info(f"Ingested {len(costs)} AWS cost records")
            
        except ClientError as e:
            logger.error(f"Failed to ingest AWS costs: {e}")
        
        return costs
    
    async def ingest_gcp_costs(self, start_date: datetime, end_date: datetime) -> List[CostData]:
        """Ingest GCP cost data from BigQuery billing export"""
        costs = []
        
        try:
            # Query BigQuery billing export table
            query = f"""
                SELECT 
                    service.description as service,
                    location.location as region,
                    sku.description as resource_type,
                    resource.name as resource_name,
                    resource.global_name as resource_id,
                    cost,
                    currency,
                    usage.amount as usage_quantity,
                    usage.unit as usage_unit,
                    DATE(usage_start_time) as date,
                    labels
                FROM `{self.gcp_project_id}.billing.gcp_billing_export_v1`
                WHERE DATE(usage_start_time) >= '{start_date.strftime('%Y-%m-%d')}'
                    AND DATE(usage_start_time) <= '{end_date.strftime('%Y-%m-%d')}'
            """
            
            query_job = self.gcp_bq_client.query(query)
            results = query_job.result()
            
            for row in results:
                cost_data = CostData(
                    provider="GCP",
                    service=row.service or "Unknown",
                    resource_id=row.resource_id,
                    resource_name=row.resource_name,
                    resource_type=row.resource_type,
                    region=row.region or "Unknown",
                    cost=Decimal(str(row.cost)),
                    currency=row.currency,
                    usage_quantity=Decimal(str(row.usage_quantity or 0)),
                    usage_unit=row.usage_unit or "Units",
                    date=row.date,
                    tags=dict(row.labels) if row.labels else {},
                    metadata={}
                )
                costs.append(cost_data)
            
            logger.info(f"Ingested {len(costs)} GCP cost records")
            
        except Exception as e:
            logger.error(f"Failed to ingest GCP costs: {e}")
        
        return costs
    
    async def ingest_budgets(self) -> List[BudgetData]:
        """Ingest budget data from all providers"""
        budgets = []
        
        # Azure budgets
        if self.azure_consumption_client:
            try:
                azure_budgets = self.azure_consumption_client.budgets.list(
                    scope=f"/subscriptions/{self.azure_subscription_id}"
                )
                
                for budget in azure_budgets:
                    budget_data = BudgetData(
                        provider="Azure",
                        budget_name=budget.name,
                        budget_amount=Decimal(str(budget.amount)),
                        spent_amount=Decimal(str(budget.current_spend.amount if budget.current_spend else 0)),
                        remaining_amount=Decimal(str(budget.amount - (budget.current_spend.amount if budget.current_spend else 0))),
                        percentage_used=float((budget.current_spend.amount if budget.current_spend else 0) / budget.amount * 100),
                        currency=budget.currency if hasattr(budget, 'currency') else "USD",
                        period_start=budget.time_period.start_date,
                        period_end=budget.time_period.end_date,
                        alerts=[]
                    )
                    budgets.append(budget_data)
                    
            except Exception as e:
                logger.error(f"Failed to ingest Azure budgets: {e}")
        
        # AWS budgets
        if self.aws_budgets_client:
            try:
                aws_budgets = self.aws_budgets_client.describe_budgets(
                    AccountId=os.getenv("AWS_ACCOUNT_ID", "")
                )
                
                for budget in aws_budgets.get('Budgets', []):
                    budget_data = BudgetData(
                        provider="AWS",
                        budget_name=budget['BudgetName'],
                        budget_amount=Decimal(budget['BudgetLimit']['Amount']),
                        spent_amount=Decimal(budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', '0')),
                        remaining_amount=Decimal(budget['BudgetLimit']['Amount']) - Decimal(budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', '0')),
                        percentage_used=float(budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', 0)) / float(budget['BudgetLimit']['Amount']) * 100,
                        currency=budget['BudgetLimit']['Unit'],
                        period_start=datetime.strptime(budget['TimePeriod']['Start'], '%Y-%m-%d'),
                        period_end=datetime.strptime(budget['TimePeriod']['End'], '%Y-%m-%d'),
                        alerts=[]
                    )
                    budgets.append(budget_data)
                    
            except ClientError as e:
                logger.error(f"Failed to ingest AWS budgets: {e}")
        
        self.budget_data_cache = budgets
        return budgets
    
    async def generate_savings_recommendations(self) -> List[SavingsRecommendation]:
        """Generate savings recommendations based on cost data"""
        recommendations = []
        
        if not self.cost_data_cache:
            await self.ingest_all_costs(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
        
        # Analyze cost data for recommendations
        df = pd.DataFrame([{
            'provider': c.provider,
            'service': c.service,
            'region': c.region,
            'cost': float(c.cost),
            'usage': float(c.usage_quantity),
            'date': c.date
        } for c in self.cost_data_cache])
        
        if not df.empty:
            # Group by service and calculate statistics
            service_stats = df.groupby(['provider', 'service']).agg({
                'cost': ['sum', 'mean', 'std'],
                'usage': ['sum', 'mean']
            }).reset_index()
            
            # Identify high-cost services
            high_cost_services = service_stats[service_stats[('cost', 'sum')] > service_stats[('cost', 'sum')].quantile(0.75)]
            
            for _, row in high_cost_services.iterrows():
                provider = row[('provider', '')]
                service = row[('service', '')]
                total_cost = row[('cost', 'sum')]
                
                # Generate recommendations based on service type
                if 'storage' in service.lower():
                    recommendations.append(SavingsRecommendation(
                        provider=provider,
                        recommendation_type="Storage Optimization",
                        resource_id=None,
                        description=f"Optimize {service} storage - lifecycle policies and compression",
                        estimated_savings=Decimal(str(total_cost * 0.2)),
                        currency="USD",
                        impact="Medium",
                        effort="Low",
                        confidence=0.75,
                        actions=[
                            "Implement lifecycle policies",
                            "Enable compression",
                            "Delete unused data",
                            "Move to cheaper storage tiers"
                        ]
                    ))
                
                elif 'compute' in service.lower() or 'vm' in service.lower():
                    recommendations.append(SavingsRecommendation(
                        provider=provider,
                        recommendation_type="Compute Rightsizing",
                        resource_id=None,
                        description=f"Rightsize {service} instances based on utilization",
                        estimated_savings=Decimal(str(total_cost * 0.3)),
                        currency="USD",
                        impact="High",
                        effort="Medium",
                        confidence=0.80,
                        actions=[
                            "Analyze CPU/memory utilization",
                            "Identify overprovisioned instances",
                            "Resize to appropriate instance types",
                            "Consider reserved instances or savings plans"
                        ]
                    ))
            
            # Identify idle resources (low usage but consistent cost)
            low_usage = df[df['usage'] < df['usage'].quantile(0.1)]
            if not low_usage.empty:
                idle_cost = low_usage['cost'].sum()
                recommendations.append(SavingsRecommendation(
                    provider="Multi-cloud",
                    recommendation_type="Idle Resource Cleanup",
                    resource_id=None,
                    description="Remove or stop idle resources with minimal usage",
                    estimated_savings=Decimal(str(idle_cost)),
                    currency="USD",
                    impact="High",
                    effort="Low",
                    confidence=0.90,
                    actions=[
                        "Identify resources with < 5% utilization",
                        "Stop development/test resources after hours",
                        "Delete unused resources",
                        "Implement auto-shutdown policies"
                    ]
                ))
        
        self.recommendations_cache = recommendations
        return recommendations
    
    async def get_cost_anomalies(self, threshold: float = 1.5) -> List[Dict[str, Any]]:
        """Detect cost anomalies using statistical analysis"""
        anomalies = []
        
        if not self.cost_data_cache:
            return anomalies
        
        df = pd.DataFrame([{
            'provider': c.provider,
            'service': c.service,
            'cost': float(c.cost),
            'date': c.date
        } for c in self.cost_data_cache])
        
        if not df.empty:
            # Group by service and date
            daily_costs = df.groupby(['provider', 'service', 'date'])['cost'].sum().reset_index()
            
            for (provider, service), group in daily_costs.groupby(['provider', 'service']):
                if len(group) > 7:  # Need at least a week of data
                    # Calculate rolling statistics
                    group = group.sort_values('date')
                    group['rolling_mean'] = group['cost'].rolling(window=7, min_periods=3).mean()
                    group['rolling_std'] = group['cost'].rolling(window=7, min_periods=3).std()
                    
                    # Detect anomalies (costs above threshold * std from mean)
                    group['is_anomaly'] = (group['cost'] > group['rolling_mean'] + threshold * group['rolling_std'])
                    
                    for _, row in group[group['is_anomaly']].iterrows():
                        anomalies.append({
                            'provider': provider,
                            'service': service,
                            'date': row['date'].isoformat(),
                            'cost': float(row['cost']),
                            'expected_cost': float(row['rolling_mean']),
                            'deviation': float(row['cost'] - row['rolling_mean']),
                            'severity': 'High' if row['cost'] > row['rolling_mean'] * 2 else 'Medium'
                        })
        
        return anomalies
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of ingested cost data"""
        if not self.cost_data_cache:
            return {
                'total_cost': 0,
                'providers': {},
                'services': {},
                'period': None
            }
        
        df = pd.DataFrame([{
            'provider': c.provider,
            'service': c.service,
            'cost': float(c.cost),
            'date': c.date
        } for c in self.cost_data_cache])
        
        return {
            'total_cost': float(df['cost'].sum()),
            'providers': df.groupby('provider')['cost'].sum().to_dict(),
            'services': df.groupby('service')['cost'].sum().nlargest(10).to_dict(),
            'period': {
                'start': df['date'].min().isoformat() if not df.empty else None,
                'end': df['date'].max().isoformat() if not df.empty else None
            },
            'daily_average': float(df.groupby('date')['cost'].sum().mean()) if not df.empty else 0,
            'record_count': len(self.cost_data_cache)
        }

# Singleton instance
finops_ingestion = FinOpsIngestion()