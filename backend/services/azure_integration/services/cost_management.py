"""
Azure Cost Management service for handling cost analysis and budgeting operations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.consumption import ConsumptionManagementClient
from azure.core.exceptions import AzureError, ResourceNotFoundError

from shared.config import get_settings
from ..models import CostResponse, BudgetResponse
from .azure_auth import AzureAuthService

settings = get_settings()
logger = structlog.get_logger(__name__)


class CostManagementService:
    """Service for managing Azure costs and budgets."""
    
    def __init__(self):
        self.settings = settings
        self.auth_service = AzureAuthService()
        self.cost_clients = {}
        self.consumption_clients = {}
    
    async def _get_cost_client(self, subscription_id: str) -> CostManagementClient:
        """Get or create Cost Management client for subscription."""
        if subscription_id not in self.cost_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.cost_clients[subscription_id] = CostManagementClient(
                credential, subscription_id
            )
        return self.cost_clients[subscription_id]
    
    async def _get_consumption_client(self, subscription_id: str) -> ConsumptionManagementClient:
        """Get or create Consumption Management client for subscription."""
        if subscription_id not in self.consumption_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.consumption_clients[subscription_id] = ConsumptionManagementClient(
                credential, subscription_id
            )
        return self.consumption_clients[subscription_id]
    
    async def get_usage_details(
        self,
        subscription_id: str,
        start_date: str,
        end_date: str,
        granularity: str = "Daily"
    ) -> CostResponse:
        """Get cost usage details for a subscription."""
        try:
            client = await self._get_cost_client(subscription_id)
            
            # Build query parameters
            scope = f"/subscriptions/{subscription_id}"
            
            # Define the query
            query = {
                "type": "ActualCost",
                "timeframe": "Custom",
                "time_period": {
                    "from": start_date,
                    "to": end_date
                },
                "dataset": {
                    "granularity": granularity,
                    "aggregation": {
                        "totalCost": {
                            "name": "PreTaxCost",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {
                            "type": "Dimension",
                            "name": "ServiceName"
                        }
                    ]
                }
            }
            
            # Execute query
            result = await client.query.usage(scope, query)
            
            # Process results
            total_cost = 0
            cost_breakdown = []
            
            if result.rows:
                for row in result.rows:
                    # Assuming columns are: [Cost, ServiceName, Currency, Date]
                    cost = float(row[0]) if row[0] else 0
                    service_name = row[1] if len(row) > 1 else "Unknown"
                    currency = row[2] if len(row) > 2 else "USD"
                    date_str = row[3] if len(row) > 3 else None
                    
                    total_cost += cost
                    cost_breakdown.append({
                        "service_name": service_name,
                        "cost": cost,
                        "currency": currency,
                        "date": date_str
                    })
            
            # Get budget status
            budget_status = await self._get_budget_status(subscription_id)
            
            logger.info(
                "usage_details_retrieved",
                subscription_id=subscription_id,
                start_date=start_date,
                end_date=end_date,
                total_cost=total_cost
            )
            
            return CostResponse(
                subscription_id=subscription_id,
                time_period={"from": start_date, "to": end_date},
                currency="USD",
                total_cost=total_cost,
                cost_breakdown=cost_breakdown,
                budget_status=budget_status
            )
            
        except AzureError as e:
            logger.error(
                "get_usage_details_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to get usage details: {str(e)}")
    
    async def get_cost_forecast(
        self,
        subscription_id: str,
        forecast_days: int = 30
    ) -> CostResponse:
        """Get cost forecast for a subscription."""
        try:
            client = await self._get_cost_client(subscription_id)
            
            # Calculate forecast period
            start_date = datetime.now().date()
            end_date = start_date + timedelta(days=forecast_days)
            
            scope = f"/subscriptions/{subscription_id}"
            
            # Define the forecast query
            query = {
                "type": "ForecastActualCost",
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
                        }
                    }
                }
            }
            
            # Execute forecast query
            result = await client.query.usage(scope, query)
            
            # Process forecast results
            total_forecast = 0
            forecast_breakdown = []
            
            if result.rows:
                for row in result.rows:
                    cost = float(row[0]) if row[0] else 0
                    date_str = row[1] if len(row) > 1 else None
                    
                    total_forecast += cost
                    forecast_breakdown.append({
                        "date": date_str,
                        "forecast_cost": cost,
                        "currency": "USD"
                    })
            
            logger.info(
                "cost_forecast_retrieved",
                subscription_id=subscription_id,
                forecast_days=forecast_days,
                total_forecast=total_forecast
            )
            
            return CostResponse(
                subscription_id=subscription_id,
                time_period={
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                },
                currency="USD",
                total_cost=total_forecast,
                forecast={
                    "period_days": forecast_days,
                    "total_forecast": total_forecast,
                    "daily_breakdown": forecast_breakdown
                }
            )
            
        except AzureError as e:
            logger.error(
                "get_cost_forecast_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to get cost forecast: {str(e)}")
    
    async def list_budgets(
        self,
        subscription_id: str
    ) -> List[Dict[str, Any]]:
        """List all budgets for a subscription."""
        try:
            consumption_client = await self._get_consumption_client(subscription_id)
            budgets = []
            
            # List budgets
            budget_list = consumption_client.budgets.list()
            
            async for budget in budget_list:
                # Calculate budget usage
                spent = budget.current_spend.amount if budget.current_spend else 0
                amount = budget.amount
                remaining = amount - spent
                percentage_used = (spent / amount * 100) if amount > 0 else 0
                
                budgets.append({
                    "id": budget.id,
                    "name": budget.name,
                    "type": budget.type,
                    "amount": amount,
                    "spent": spent,
                    "remaining": remaining,
                    "percentage_used": percentage_used,
                    "time_grain": budget.time_grain,
                    "time_period": {
                        "start_date": budget.time_period.start_date.isoformat(),
                        "end_date": budget.time_period.end_date.isoformat() if budget.time_period.end_date else None
                    },
                    "notifications": [
                        {
                            "enabled": notif.enabled,
                            "operator": notif.operator,
                            "threshold": notif.threshold,
                            "contact_emails": notif.contact_emails,
                            "contact_groups": notif.contact_groups
                        }
                        for notif in budget.notifications.values()
                    ] if budget.notifications else [],
                    "forecast_spend": budget.forecast_spend.amount if budget.forecast_spend else None
                })
            
            logger.info(
                "budgets_listed",
                subscription_id=subscription_id,
                count=len(budgets)
            )
            
            return budgets
            
        except AzureError as e:
            logger.error(
                "list_budgets_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list budgets: {str(e)}")
    
    async def create_budget(
        self,
        subscription_id: str,
        budget_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new budget."""
        try:
            consumption_client = await self._get_consumption_client(subscription_id)
            
            # Create budget parameters
            budget_params = {
                "category": budget_data.get("category", "Cost"),
                "amount": budget_data["amount"],
                "time_grain": budget_data.get("time_grain", "Monthly"),
                "time_period": {
                    "start_date": budget_data["start_date"],
                    "end_date": budget_data.get("end_date")
                }
            }
            
            # Add filters if provided
            if budget_data.get("filters"):
                budget_params["filters"] = budget_data["filters"]
            
            # Add notifications if provided
            if budget_data.get("notifications"):
                budget_params["notifications"] = {}
                for i, notif in enumerate(budget_data["notifications"]):
                    budget_params["notifications"][f"notification{i}"] = {
                        "enabled": notif.get("enabled", True),
                        "operator": notif.get("operator", "GreaterThan"),
                        "threshold": notif.get("threshold", 90),
                        "contact_emails": notif.get("contact_emails", []),
                        "contact_groups": notif.get("contact_groups", [])
                    }
            
            # Create budget
            budget = await consumption_client.budgets.create_or_update(
                budget_name=budget_data["name"],
                parameters=budget_params
            )
            
            logger.info(
                "budget_created",
                subscription_id=subscription_id,
                budget_name=budget_data["name"],
                amount=budget_data["amount"]
            )
            
            return {
                "id": budget.id,
                "name": budget.name,
                "type": budget.type,
                "amount": budget.amount,
                "category": budget.category,
                "time_grain": budget.time_grain,
                "time_period": {
                    "start_date": budget.time_period.start_date.isoformat(),
                    "end_date": budget.time_period.end_date.isoformat() if budget.time_period.end_date else None
                },
                "notifications": budget.notifications
            }
            
        except AzureError as e:
            logger.error(
                "create_budget_failed",
                error=str(e),
                subscription_id=subscription_id,
                budget_name=budget_data.get("name")
            )
            raise Exception(f"Failed to create budget: {str(e)}")
    
    async def get_cost_recommendations(
        self,
        subscription_id: str
    ) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations."""
        try:
            recommendations = []
            
            # Get usage data for analysis
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            usage_data = await self.get_usage_details(
                subscription_id=subscription_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            # Analyze cost patterns
            if usage_data.cost_breakdown:
                # Find top cost services
                sorted_costs = sorted(
                    usage_data.cost_breakdown,
                    key=lambda x: x.get("cost", 0),
                    reverse=True
                )
                
                # Recommend optimization for top 3 services
                for i, service in enumerate(sorted_costs[:3]):
                    if service.get("cost", 0) > 100:  # $100 threshold
                        recommendations.append({
                            "type": "cost_optimization",
                            "priority": "high" if i == 0 else "medium",
                            "service": service.get("service_name", "Unknown"),
                            "current_cost": service.get("cost", 0),
                            "recommendation": f"Review {service.get('service_name')} usage for optimization opportunities",
                            "potential_savings": service.get("cost", 0) * 0.2,  # Assume 20% savings
                            "actions": [
                                "Right-size resources",
                                "Use Reserved Instances",
                                "Implement auto-scaling"
                            ]
                        })
            
            # Add generic recommendations
            recommendations.extend([
                {
                    "type": "resource_optimization",
                    "priority": "medium",
                    "service": "Virtual Machines",
                    "recommendation": "Consider using Azure Reserved VM Instances for predictable workloads",
                    "potential_savings": usage_data.total_cost * 0.3,
                    "actions": [
                        "Analyze VM usage patterns",
                        "Purchase Reserved Instances for stable workloads",
                        "Use Spot VMs for development/testing"
                    ]
                },
                {
                    "type": "storage_optimization",
                    "priority": "low",
                    "service": "Storage",
                    "recommendation": "Implement lifecycle management policies for blob storage",
                    "potential_savings": usage_data.total_cost * 0.1,
                    "actions": [
                        "Set up blob lifecycle policies",
                        "Move old data to cooler storage tiers",
                        "Delete unused storage accounts"
                    ]
                }
            ])
            
            logger.info(
                "cost_recommendations_generated",
                subscription_id=subscription_id,
                recommendation_count=len(recommendations)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "get_cost_recommendations_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            return []
    
    async def _get_budget_status(self, subscription_id: str) -> Dict[str, Any]:
        """Get budget status for subscription."""
        try:
            budgets = await self.list_budgets(subscription_id)
            
            if not budgets:
                return {
                    "has_budgets": False,
                    "total_budget": 0,
                    "total_spent": 0,
                    "budget_utilization": 0
                }
            
            total_budget = sum(b.get("amount", 0) for b in budgets)
            total_spent = sum(b.get("spent", 0) for b in budgets)
            utilization = (total_spent / total_budget * 100) if total_budget > 0 else 0
            
            return {
                "has_budgets": True,
                "budget_count": len(budgets),
                "total_budget": total_budget,
                "total_spent": total_spent,
                "budget_utilization": utilization,
                "budgets_over_threshold": len([b for b in budgets if b.get("percentage_used", 0) > 80])
            }
            
        except Exception as e:
            logger.error(
                "get_budget_status_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            return {
                "has_budgets": False,
                "error": str(e)
            }