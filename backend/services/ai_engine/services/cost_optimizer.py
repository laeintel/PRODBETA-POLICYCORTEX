"""
Cost Optimization Service for AI Engine.
Provides AI-driven cost optimization recommendations for Azure resources.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from azure.mgmt.consumption.aio import ConsumptionManagementClient
from azure.mgmt.resource.aio import ResourceManagementClient
from azure.identity.aio import DefaultAzureCredential

from ....shared.config import get_settings
from ..models import OptimizationGoal

settings = get_settings()
logger = structlog.get_logger(__name__)


@dataclass
class CostOptimizationRecommendation:
    """Represents a cost optimization recommendation."""
    resource_id: str
    resource_type: str
    recommendation_type: str
    priority: str
    potential_savings: float
    implementation_effort: str
    risk_level: str
    description: str
    detailed_actions: List[str]
    estimated_timeline: str
    impact_analysis: Dict[str, Any]


class CostOptimizer:
    """AI-driven cost optimization service."""

    def __init__(self):
        self.settings = settings
        self.consumption_client = None
        self.resource_client = None
        self.azure_credential = None
        self.optimization_rules = self._load_optimization_rules()
        self.cost_patterns = self._load_cost_patterns()
        self.savings_calculator = self._initialize_savings_calculator()

    def _load_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load cost optimization rules."""
        return {
            "underutilized_vm": {
                "cpu_threshold": 0.05,  # 5% CPU utilization
                "memory_threshold": 0.1,  # 10% memory utilization
                "duration_days": 7,
                "potential_savings": 0.3,  # 30% savings
                "action": "resize_or_shutdown"
            },
            "oversized_vm": {
                "cpu_threshold": 0.95,  # 95% CPU utilization
                "memory_threshold": 0.9,  # 90% memory utilization
                "duration_days": 1,
                "potential_savings": 0.2,  # 20% savings through rightsizing
                "action": "resize_up"
            },
            "unused_disk": {
                "attachment_status": "unattached",
                "duration_days": 30,
                "potential_savings": 1.0,  # 100% savings
                "action": "delete"
            },
            "expensive_storage": {
                "access_frequency": "low",
                "duration_days": 90,
                "potential_savings": 0.5,  # 50% savings
                "action": "move_to_cheaper_tier"
            },
            "idle_database": {
                "connections": 0,
                "duration_days": 7,
                "potential_savings": 0.8,  # 80% savings
                "action": "pause_or_scale_down"
            },
            "redundant_backup": {
                "backup_frequency": "daily",
                "retention_days": 365,
                "potential_savings": 0.4,  # 40% savings
                "action": "optimize_backup_policy"
            }
        }

    def _load_cost_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load cost pattern analysis rules."""
        return {
            "seasonal_usage": {
                "pattern": "seasonal",
                "recommendation": "schedule_scaling",
                "savings_potential": 0.25
            },
            "weekend_idle": {
                "pattern": "weekend_low",
                "recommendation": "weekend_shutdown",
                "savings_potential": 0.15
            },
            "business_hours": {
                "pattern": "business_hours_only",
                "recommendation": "schedule_start_stop",
                "savings_potential": 0.35
            },
            "development_resources": {
                "pattern": "dev_environment",
                "recommendation": "auto_shutdown",
                "savings_potential": 0.50
            }
        }

    def _initialize_savings_calculator(self) -> Dict[str, Any]:
        """Initialize savings calculation parameters."""
        return {
            "pricing_data": {
                "vm_hourly_rates": {
                    "Standard_B1s": 0.0104,
                    "Standard_B2s": 0.0416,
                    "Standard_D2s_v3": 0.096,
                    "Standard_D4s_v3": 0.192,
                    "Standard_D8s_v3": 0.384
                },
                "storage_monthly_rates": {
                    "premium_ssd": 0.135,  # per GB
                    "standard_ssd": 0.05,
                    "standard_hdd": 0.02
                },
                "database_hourly_rates": {
                    "Basic": 0.0067,
                    "Standard_S1": 0.02,
                    "Standard_S2": 0.05,
                    "Premium_P1": 0.25
                }
            },
            "reserved_instance_discounts": {
                "1_year": 0.2,  # 20% discount
                "3_year": 0.35  # 35% discount
            },
            "volume_discounts": {
                "tier_1": {"threshold": 1000, "discount": 0.05},
                "tier_2": {"threshold": 5000, "discount": 0.10},
                "tier_3": {"threshold": 10000, "discount": 0.15}
            }
        }

    async def initialize(self) -> None:
        """Initialize the cost optimizer."""
        try:
            logger.info("Initializing cost optimizer")

            # Initialize Azure clients
            if self.settings.is_production():
                await self._initialize_azure_clients()

            logger.info("Cost optimizer initialized successfully")

        except Exception as e:
            logger.error("Cost optimizer initialization failed", error=str(e))
            raise

    async def _initialize_azure_clients(self) -> None:
        """Initialize Azure clients for cost data."""
        try:
            self.azure_credential = DefaultAzureCredential()

            # Initialize Consumption Management client
            self.consumption_client = ConsumptionManagementClient(
                credential=self.azure_credential,
                subscription_id=self.settings.azure.subscription_id
            )

            # Initialize Resource Management client
            self.resource_client = ResourceManagementClient(
                credential=self.azure_credential,
                subscription_id=self.settings.azure.subscription_id
            )

            logger.info("Azure clients initialized for cost optimization")

        except Exception as e:
            logger.warning("Failed to initialize Azure clients", error=str(e))

    async def optimize_costs(self, resource_data: Dict[str, Any],
                           optimization_goals: List[str],
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate cost optimization recommendations."""
        try:
            logger.info("Starting cost optimization analysis",
                       goals=optimization_goals)

            # Initialize results
            results = {
                "optimization_goals": optimization_goals,
                "constraints": constraints or {},
                "recommendations": [],
                "projected_savings": {
                    "monthly": 0.0,
                    "annual": 0.0,
                    "percentage": 0.0
                },
                "implementation_plan": {
                    "quick_wins": [],
                    "medium_term": [],
                    "long_term": []
                },
                "risk_assessment": {
                    "low_risk": [],
                    "medium_risk": [],
                    "high_risk": []
                },
                "confidence": 0.0
            }

            # Analyze resources for optimization opportunities
            resources = resource_data.get("resources", [])
            current_cost = resource_data.get("current_cost", 0)

            # Generate recommendations based on goals
            for goal in optimization_goals:
                if goal == OptimizationGoal.MINIMIZE_COST:
                    recommendations = (
                        await self._generate_cost_minimization_recommendations(resources)
                    )
                elif goal == OptimizationGoal.MAXIMIZE_PERFORMANCE:
                    recommendations = (
                        await self._generate_performance_optimization_recommendations(resources)
                    )
                elif goal == OptimizationGoal.BALANCE_COST_PERFORMANCE:
                    recommendations = await self._generate_balanced_recommendations(resources)
                elif goal == OptimizationGoal.OPTIMIZE_UTILIZATION:
                    recommendations = (
                        await self._generate_utilization_optimization_recommendations(resources)
                    )
                else:
                    recommendations = []

                results["recommendations"].extend(recommendations)

            # Remove duplicates and prioritize
            results["recommendations"] = await self._prioritize_recommendations(
                results["recommendations"], constraints
            )

            # Calculate projected savings
            results["projected_savings"] = await self._calculate_projected_savings(
                results["recommendations"], current_cost
            )

            # Create implementation plan
            results["implementation_plan"] = await self._create_implementation_plan(
                results["recommendations"]
            )

            # Assess risks
            results["risk_assessment"] = await self._assess_risks(
                results["recommendations"]
            )

            # Calculate confidence score
            results["confidence"] = await self._calculate_confidence_score(
                results["recommendations"], resource_data
            )

            logger.info("Cost optimization analysis completed",
                       recommendations_count=len(results["recommendations"]),
                       projected_monthly_savings=results["projected_savings"]["monthly"])

            return results

        except Exception as e:
            logger.error("Cost optimization failed", error=str(e))
            raise

    async def _generate_cost_minimization_recommendations(self,
                                                        resources: List[Dict[str, Any]]) -> List[CostOptimizationRecommendation]:
        """Generate recommendations focused on cost minimization."""
        recommendations = []

        try:
            for resource in resources:
                resource_type = resource.get("type", "")
                resource_id = resource.get("id", "")

                # Check for underutilized VMs
                if resource_type == "Microsoft.Compute/virtualMachines":
                    if await self._is_underutilized_vm(resource):
                        recommendation = CostOptimizationRecommendation(
                            resource_id=resource_id,
                            resource_type=resource_type,
                            recommendation_type="underutilized_vm",
                            priority="high",
                            potential_savings=await self._calculate_vm_savings(resource, 0.3),
                            implementation_effort="low",
                            risk_level="low",
                            description="VM is underutilized and can be downsized or shut down",
                            detailed_actions=[
                                "Analyze usage patterns over the last 30 days",
                                "Downsize to appropriate VM size",
                                "Consider scheduled shutdown during off-hours",
                                "Implement auto-scaling if workload varies"
                            ],
                            estimated_timeline="1-2 weeks",
                            impact_analysis={
                                "performance_impact": "minimal",
                                "availability_impact": "none",
                                "user_impact": "minimal"
                            }
                        )
                        recommendations.append(recommendation)

                # Check for unused storage
                elif resource_type == "Microsoft.Storage/storageAccounts":
                    if await self._is_unused_storage(resource):
                        recommendation = CostOptimizationRecommendation(
                            resource_id=resource_id,
                            resource_type=resource_type,
                            recommendation_type="unused_storage",
                            priority="medium",
                            potential_savings=await self._calculate_storage_savings(resource, 1.0),
                            implementation_effort="low",
                            risk_level="low",
                            description="Storage account appears to be unused",
                            detailed_actions=[
                                "Verify storage account is not in use",
                                "Backup any important data",
                                "Delete unused storage account",
                                "Update applications to remove references"
                            ],
                            estimated_timeline="1 week",
                            impact_analysis={
                                "performance_impact": "none",
                                "availability_impact": "none",
                                "user_impact": "none"
                            }
                        )
                        recommendations.append(recommendation)

                # Check for idle databases
                elif resource_type == "Microsoft.Sql/servers/databases":
                    if await self._is_idle_database(resource):
                        recommendation = CostOptimizationRecommendation(
                            resource_id=resource_id,
                            resource_type=resource_type,
                            recommendation_type="idle_database",
                            priority="high",
                            potential_savings=await self._calculate_database_savings(resource, 0.8),
                            implementation_effort="medium",
                            risk_level="medium",
                            description="Database has minimal activity and
                                can be paused or scaled down",
                            detailed_actions=[
                                "Analyze database connection patterns",
                                "Implement database pause/resume schedule",
                                "Consider serverless compute tier",
                                "Optimize backup retention policies"
                            ],
                            estimated_timeline="2-3 weeks",
                            impact_analysis={
                                "performance_impact": "low",
                                "availability_impact": "scheduled",
                                "user_impact": "minimal"
                            }
                        )
                        recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error("Cost minimization recommendations failed", error=str(e))
            return []

    async def _generate_performance_optimization_recommendations(self,
                                                               resources: List[Dict[str, Any]]) -> List[CostOptimizationRecommendation]:
        """Generate recommendations focused on performance optimization."""
        recommendations = []

        try:
            for resource in resources:
                resource_type = resource.get("type", "")
                resource_id = resource.get("id", "")

                # Check for oversized VMs
                if resource_type == "Microsoft.Compute/virtualMachines":
                    if await self._is_oversized_vm(resource):
                        recommendation = CostOptimizationRecommendation(
                            resource_id=resource_id,
                            resource_type=resource_type,
                            recommendation_type="oversized_vm",
                            priority="medium",
                            potential_savings=await self._calculate_vm_savings(
                                resource,
                                -0.2
                            ),  # Negative for cost increase
                            implementation_effort="medium",
                            risk_level="low",
                            description="VM is consistently at high utilization and
                                may need scaling up",
                            detailed_actions=[
                                "Monitor performance metrics",
                                "Scale up to higher performance tier",
                                "Consider premium storage",
                                "Implement load balancing"
                            ],
                            estimated_timeline="2-4 weeks",
                            impact_analysis={
                                "performance_impact": "significant improvement",
                                "availability_impact": "brief during scaling",
                                "user_impact": "improved experience"
                            }
                        )
                        recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error("Performance optimization recommendations failed", error=str(e))
            return []

    async def _generate_balanced_recommendations(self,
                                               resources: List[Dict[str, Any]]) -> List[CostOptimizationRecommendation]:
        """Generate recommendations balancing cost and performance."""
        recommendations = []

        try:
            # Combine cost and performance recommendations with balanced priorities
            cost_recs = await self._generate_cost_minimization_recommendations(resources)
            perf_recs = await self._generate_performance_optimization_recommendations(resources)

            # Adjust priorities for balanced approach
            for rec in cost_recs:
                if rec.risk_level == "low" and rec.potential_savings > 100:
                    rec.priority = "high"
                else:
                    rec.priority = "medium"
                recommendations.append(rec)

            for rec in perf_recs:
                if rec.impact_analysis.get("performance_impact") == "significant improvement":
                    rec.priority = "high"
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error("Balanced recommendations failed", error=str(e))
            return []

    async def _generate_utilization_optimization_recommendations(self,
                                                              resources: List[Dict[str, Any]]) -> List[CostOptimizationRecommendation]:
        """Generate recommendations focused on utilization optimization."""
        recommendations = []

        try:
            for resource in resources:
                resource_type = resource.get("type", "")
                resource_id = resource.get("id", "")

                # Check utilization patterns
                utilization = resource.get("utilization", {})

                if resource_type == "Microsoft.Compute/virtualMachines":
                    cpu_avg = utilization.get("cpu_average", 0)
                    memory_avg = utilization.get("memory_average", 0)

                    if cpu_avg < 0.2 or memory_avg < 0.3:  # Low utilization
                        recommendation = CostOptimizationRecommendation(
                            resource_id=resource_id,
                            resource_type=resource_type,
                            recommendation_type="low_utilization",
                            priority="high",
                            potential_savings=await self._calculate_vm_savings(resource, 0.4),
                            implementation_effort="medium",
                            risk_level="low",
                            description="VM utilization is consistently low",
                            detailed_actions=[
                                "Implement auto-scaling",
                                "Consider spot instances",
                                "Consolidate workloads",
                                "Use reserved instances for predictable workloads"
                            ],
                            estimated_timeline="3-4 weeks",
                            impact_analysis={
                                "performance_impact": "minimal",
                                "availability_impact": "improved",
                                "user_impact": "none"
                            }
                        )
                        recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error("Utilization optimization recommendations failed", error=str(e))
            return []

    async def _is_underutilized_vm(self, resource: Dict[str, Any]) -> bool:
        """Check if VM is underutilized."""
        try:
            utilization = resource.get("utilization", {})
            cpu_avg = utilization.get("cpu_average", 0)
            memory_avg = utilization.get("memory_average", 0)

            rule = self.optimization_rules["underutilized_vm"]
            return (cpu_avg < rule["cpu_threshold"] and
                   memory_avg < rule["memory_threshold"])

        except Exception as e:
            logger.error("VM utilization check failed", error=str(e))
            return False

    async def _is_oversized_vm(self, resource: Dict[str, Any]) -> bool:
        """Check if VM is oversized."""
        try:
            utilization = resource.get("utilization", {})
            cpu_avg = utilization.get("cpu_average", 0)
            memory_avg = utilization.get("memory_average", 0)

            rule = self.optimization_rules["oversized_vm"]
            return (cpu_avg > rule["cpu_threshold"] or
                   memory_avg > rule["memory_threshold"])

        except Exception as e:
            logger.error("VM oversize check failed", error=str(e))
            return False

    async def _is_unused_storage(self, resource: Dict[str, Any]) -> bool:
        """Check if storage is unused."""
        try:
            metrics = resource.get("metrics", {})
            transactions = metrics.get("transactions_last_30_days", 0)
            return transactions == 0

        except Exception as e:
            logger.error("Storage usage check failed", error=str(e))
            return False

    async def _is_idle_database(self, resource: Dict[str, Any]) -> bool:
        """Check if database is idle."""
        try:
            metrics = resource.get("metrics", {})
            connections = metrics.get("active_connections", 0)
            queries = metrics.get("queries_last_7_days", 0)
            return connections == 0 and queries == 0

        except Exception as e:
            logger.error("Database idle check failed", error=str(e))
            return False

    async def _calculate_vm_savings(self, resource: Dict[str, Any], savings_rate: float) -> float:
        """Calculate potential VM cost savings."""
        try:
            vm_size = resource.get("size", "Standard_D2s_v3")
            pricing = self.savings_calculator["pricing_data"]["vm_hourly_rates"]
            hourly_rate = pricing.get(vm_size, 0.096)

            # Calculate monthly savings
            monthly_cost = hourly_rate * 24 * 30  # 30 days
            return monthly_cost * savings_rate

        except Exception as e:
            logger.error("VM savings calculation failed", error=str(e))
            return 0.0

    async def _calculate_storage_savings(
        self,
        resource: Dict[str,
        Any],
        savings_rate: float
    ) -> float:
        """Calculate potential storage cost savings."""
        try:
            storage_size = resource.get("size_gb", 100)
            storage_type = resource.get("tier", "standard_ssd")
            pricing = self.savings_calculator["pricing_data"]["storage_monthly_rates"]
            monthly_rate = pricing.get(storage_type, 0.05)

            monthly_cost = storage_size * monthly_rate
            return monthly_cost * savings_rate

        except Exception as e:
            logger.error("Storage savings calculation failed", error=str(e))
            return 0.0

    async def _calculate_database_savings(
        self,
        resource: Dict[str,
        Any],
        savings_rate: float
    ) -> float:
        """Calculate potential database cost savings."""
        try:
            db_tier = resource.get("tier", "Standard_S1")
            pricing = self.savings_calculator["pricing_data"]["database_hourly_rates"]
            hourly_rate = pricing.get(db_tier, 0.02)

            monthly_cost = hourly_rate * 24 * 30
            return monthly_cost * savings_rate

        except Exception as e:
            logger.error("Database savings calculation failed", error=str(e))
            return 0.0

    async def _prioritize_recommendations(self, recommendations: List[CostOptimizationRecommendation],
                                        constraints: Optional[Dict[str, Any]]) -> List[CostOptimizationRecommendation]:
        """Prioritize recommendations based on impact and constraints."""
        try:
            # Sort by priority and potential savings
            def priority_score(rec):
                priority_weights = {"high": 3, "medium": 2, "low": 1}
                effort_weights = {"low": 3, "medium": 2, "high": 1}
                risk_weights = {"low": 3, "medium": 2, "high": 1}

                return (priority_weights.get(rec.priority, 0) *
                       effort_weights.get(rec.implementation_effort, 0) *
                       risk_weights.get(rec.risk_level, 0) *
                       rec.potential_savings)

            sorted_recommendations = sorted(recommendations, key=priority_score, reverse=True)

            # Remove duplicates based on resource_id
            seen_resources = set()
            unique_recommendations = []

            for rec in sorted_recommendations:
                if rec.resource_id not in seen_resources:
                    seen_resources.add(rec.resource_id)
                    unique_recommendations.append(rec)

            return unique_recommendations

        except Exception as e:
            logger.error("Recommendation prioritization failed", error=str(e))
            return recommendations

    async def _calculate_projected_savings(self, recommendations: List[CostOptimizationRecommendation],
                                         current_cost: float) -> Dict[str, float]:
        """Calculate projected savings from recommendations."""
        try:
            total_monthly_savings = sum(rec.potential_savings for rec in recommendations)
            annual_savings = total_monthly_savings * 12

            percentage_savings = 0.0
            if current_cost > 0:
                percentage_savings = (total_monthly_savings / current_cost) * 100

            return {
                "monthly": total_monthly_savings,
                "annual": annual_savings,
                "percentage": percentage_savings
            }

        except Exception as e:
            logger.error("Savings calculation failed", error=str(e))
            return {"monthly": 0.0, "annual": 0.0, "percentage": 0.0}

    async def _create_implementation_plan(
        self,
        recommendations: List[CostOptimizationRecommendation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create implementation plan categorized by timeline."""
        try:
            plan = {"quick_wins": [], "medium_term": [], "long_term": []}

            for rec in recommendations:
                item = {
                    "resource_id": rec.resource_id,
                    "recommendation_type": rec.recommendation_type,
                    "priority": rec.priority,
                    "potential_savings": rec.potential_savings,
                    "estimated_timeline": rec.estimated_timeline,
                    "actions": rec.detailed_actions
                }

                if rec.implementation_effort == "low":
                    plan["quick_wins"].append(item)
                elif rec.implementation_effort == "medium":
                    plan["medium_term"].append(item)
                else:
                    plan["long_term"].append(item)

            return plan

        except Exception as e:
            logger.error("Implementation plan creation failed", error=str(e))
            return {"quick_wins": [], "medium_term": [], "long_term": []}

    async def _assess_risks(
        self,
        recommendations: List[CostOptimizationRecommendation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assess risks for recommendations."""
        try:
            risk_assessment = {"low_risk": [], "medium_risk": [], "high_risk": []}

            for rec in recommendations:
                risk_item = {
                    "resource_id": rec.resource_id,
                    "recommendation_type": rec.recommendation_type,
                    "risk_level": rec.risk_level,
                    "impact_analysis": rec.impact_analysis,
                    "mitigation_strategies": self._get_mitigation_strategies(rec)
                }

                risk_assessment[f"{rec.risk_level}_risk"].append(risk_item)

            return risk_assessment

        except Exception as e:
            logger.error("Risk assessment failed", error=str(e))
            return {"low_risk": [], "medium_risk": [], "high_risk": []}

    def _get_mitigation_strategies(
        self,
        recommendation: CostOptimizationRecommendation
    ) -> List[str]:
        """Get mitigation strategies for a recommendation."""
        strategies = {
            "underutilized_vm": [
                "Create VM snapshots before changes",
                "Test in development environment first",
                "Implement gradual rollout"
            ],
            "idle_database": [
                "Backup database before changes",
                "Test application connections",
                "Plan for restart procedures"
            ],
            "unused_storage": [
                "Verify no dependencies exist",
                "Create final backup",
                "Update documentation"
            ]
        }

        return strategies.get(recommendation.recommendation_type, [
            "Test in non-production environment",
            "Create rollback plan",
            "Monitor after implementation"
        ])

    async def _calculate_confidence_score(self, recommendations: List[CostOptimizationRecommendation],
                                        resource_data: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendations."""
        try:
            if not recommendations:
                return 0.0

            # Base confidence factors
            data_quality = resource_data.get("data_quality", 0.8)
            sample_size = min(len(resource_data.get("resources", [])) / 10, 1.0)

            # Calculate average confidence based on recommendation characteristics
            recommendation_confidence = 0.0
            for rec in recommendations:
                rec_confidence = 0.5  # Base confidence

                # Adjust based on implementation effort
                if rec.implementation_effort == "low":
                    rec_confidence += 0.2
                elif rec.implementation_effort == "medium":
                    rec_confidence += 0.1

                # Adjust based on risk level
                if rec.risk_level == "low":
                    rec_confidence += 0.2
                elif rec.risk_level == "medium":
                    rec_confidence += 0.1

                # Adjust based on potential savings
                if rec.potential_savings > 500:
                    rec_confidence += 0.1

                recommendation_confidence += rec_confidence

            recommendation_confidence /= len(recommendations)

            # Combine factors
            overall_confidence = (data_quality * 0.4 +
                                sample_size * 0.3 +
                                recommendation_confidence * 0.3)

            return min(overall_confidence, 1.0)

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.5

    def is_ready(self) -> bool:
        """Check if cost optimizer is ready."""
        return len(self.optimization_rules) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Close Azure clients
            if self.consumption_client:
                await self.consumption_client.close()

            if self.resource_client:
                await self.resource_client.close()

            logger.info("Cost optimizer cleanup completed")

        except Exception as e:
            logger.error("Cost optimizer cleanup failed", error=str(e))
