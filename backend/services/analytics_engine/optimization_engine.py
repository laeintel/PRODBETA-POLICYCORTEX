"""
Optimization Engine
Provides AI-driven optimization recommendations
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
import structlog
from scipy.optimize import linprog, minimize
from sklearn.ensemble import RandomForestRegressor

logger = structlog.get_logger(__name__)


class OptimizationType(str, Enum):
    COST = "cost"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    CAPACITY = "capacity"
    EFFICIENCY = "efficiency"


@dataclass
class OptimizationRecommendation:
    id: str
    title: str
    type: OptimizationType
    impact: str  # high, medium, low
    estimated_savings: float
    implementation_effort: str  # low, medium, high
    confidence: float
    description: str
    steps: List[str]
    risks: List[str]
    prerequisites: List[str]


class OptimizationEngine:
    """
    AI-driven optimization engine for resource and cost optimization
    """

    def __init__(self):
        self.optimization_history = []
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize optimization models"""
        self.models["cost"] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models["performance"] = RandomForestRegressor(n_estimators=100, random_state=42)

    async def generate_optimizations(
        self, resource_data: pd.DataFrame, cost_data: pd.DataFrame, performance_data: pd.DataFrame
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on current state

        Args:
            resource_data: Current resource utilization
            cost_data: Cost metrics
            performance_data: Performance metrics

        Returns:
            List of optimization recommendations
        """

        recommendations = []

        # Resource rightsizing
        rightsizing = await self._analyze_resource_rightsizing(resource_data)
        recommendations.extend(rightsizing)

        # Cost optimization
        cost_opts = await self._analyze_cost_optimization(cost_data)
        recommendations.extend(cost_opts)

        # Performance optimization
        perf_opts = await self._analyze_performance_optimization(performance_data)
        recommendations.extend(perf_opts)

        # Capacity optimization
        capacity_opts = await self._analyze_capacity_optimization(resource_data)
        recommendations.extend(capacity_opts)

        # Sort by impact and confidence
        recommendations.sort(
            key=lambda x: (self._impact_score(x.impact), x.confidence, x.estimated_savings),
            reverse=True,
        )

        return recommendations

    async def _analyze_resource_rightsizing(
        self, resource_data: pd.DataFrame
    ) -> List[OptimizationRecommendation]:
        """Analyze resources for rightsizing opportunities"""

        recommendations = []

        # Identify underutilized resources
        underutilized = resource_data[resource_data["utilization"] < 30]

        for _, resource in underutilized.iterrows():
            # Calculate potential savings
            current_cost = resource.get("cost", 0)
            recommended_size = self._calculate_optimal_size(resource)
            estimated_savings = current_cost * 0.4  # Assume 40% savings

            recommendations.append(
                OptimizationRecommendation(
                    id=f"rightsize_{resource['id']}",
                    title=f"Rightsize {resource['name']}",
                    type=OptimizationType.RESOURCE,
                    impact="high" if estimated_savings > 1000 else "medium",
                    estimated_savings=estimated_savings,
                    implementation_effort="low",
                    confidence=0.85,
                    description=f"Resource {resource['name']} is only {resource['utilization']:.1f}% utilized",
                    steps=[
                        f"Analyze historical usage patterns",
                        f"Resize to {recommended_size} instance type",
                        "Monitor performance for 7 days",
                        "Adjust if necessary",
                    ],
                    risks=["Potential performance impact during peak times"],
                    prerequisites=["Backup current configuration", "Schedule maintenance window"],
                )
            )

        # Identify overutilized resources
        overutilized = resource_data[resource_data["utilization"] > 85]

        for _, resource in overutilized.iterrows():
            recommendations.append(
                OptimizationRecommendation(
                    id=f"scale_{resource['id']}",
                    title=f"Scale up {resource['name']}",
                    type=OptimizationType.PERFORMANCE,
                    impact="high",
                    estimated_savings=-resource.get("cost", 0) * 0.2,  # Cost increase
                    implementation_effort="medium",
                    confidence=0.90,
                    description=f"Resource {resource['name']} is {resource['utilization']:.1f}% utilized",
                    steps=[
                        "Review performance metrics",
                        "Plan scaling strategy",
                        "Implement auto-scaling if applicable",
                        "Monitor and adjust thresholds",
                    ],
                    risks=["Increased costs", "Configuration complexity"],
                    prerequisites=["Performance baseline", "Budget approval"],
                )
            )

        return recommendations

    async def _analyze_cost_optimization(
        self, cost_data: pd.DataFrame
    ) -> List[OptimizationRecommendation]:
        """Analyze cost optimization opportunities"""

        recommendations = []

        # Analyze spending patterns
        spending_trend = cost_data["daily_cost"].pct_change().mean()

        if spending_trend > 0.05:  # 5% daily increase
            recommendations.append(
                OptimizationRecommendation(
                    id="cost_trend_alert",
                    title="Implement Cost Controls",
                    type=OptimizationType.COST,
                    impact="high",
                    estimated_savings=cost_data["daily_cost"].sum() * 0.15,
                    implementation_effort="medium",
                    confidence=0.80,
                    description=f"Costs are increasing at {spending_trend*100:.1f}% daily",
                    steps=[
                        "Set budget alerts",
                        "Implement tagging strategy",
                        "Review and eliminate unused resources",
                        "Consider reserved instances",
                    ],
                    risks=["Service disruption if not carefully planned"],
                    prerequisites=["Cost allocation tags", "Budget thresholds"],
                )
            )

        # Identify cost anomalies
        cost_std = cost_data["daily_cost"].std()
        cost_mean = cost_data["daily_cost"].mean()
        anomalies = cost_data[cost_data["daily_cost"] > cost_mean + 2 * cost_std]

        if not anomalies.empty:
            recommendations.append(
                OptimizationRecommendation(
                    id="cost_anomaly",
                    title="Investigate Cost Anomalies",
                    type=OptimizationType.COST,
                    impact="medium",
                    estimated_savings=anomalies["daily_cost"].sum() - cost_mean * len(anomalies),
                    implementation_effort="low",
                    confidence=0.75,
                    description=f"Detected {len(anomalies)} days with abnormal costs",
                    steps=[
                        "Review detailed billing for anomaly dates",
                        "Identify root cause",
                        "Implement preventive measures",
                        "Set up anomaly detection alerts",
                    ],
                    risks=["May be legitimate usage spikes"],
                    prerequisites=["Access to detailed billing data"],
                )
            )

        return recommendations

    async def _analyze_performance_optimization(
        self, performance_data: pd.DataFrame
    ) -> List[OptimizationRecommendation]:
        """Analyze performance optimization opportunities"""

        recommendations = []

        # Identify performance bottlenecks
        slow_resources = performance_data[performance_data["response_time"] > 1000]  # > 1 second

        for _, resource in slow_resources.iterrows():
            recommendations.append(
                OptimizationRecommendation(
                    id=f"perf_{resource['id']}",
                    title=f"Optimize {resource['name']} Performance",
                    type=OptimizationType.PERFORMANCE,
                    impact="high" if resource["response_time"] > 2000 else "medium",
                    estimated_savings=0,  # Performance optimization
                    implementation_effort="medium",
                    confidence=0.82,
                    description=f"Response time is {resource['response_time']}ms",
                    steps=[
                        "Profile application performance",
                        "Implement caching strategy",
                        "Optimize database queries",
                        "Consider CDN for static content",
                    ],
                    risks=["Cache invalidation complexity"],
                    prerequisites=["Performance profiling tools", "Baseline metrics"],
                )
            )

        return recommendations

    async def _analyze_capacity_optimization(
        self, resource_data: pd.DataFrame
    ) -> List[OptimizationRecommendation]:
        """Analyze capacity optimization opportunities"""

        recommendations = []

        # Predictive capacity planning
        avg_growth = resource_data["utilization"].pct_change().mean()

        if avg_growth > 0.02:  # 2% daily growth
            days_to_capacity = (100 - resource_data["utilization"].mean()) / (avg_growth * 100)

            if days_to_capacity < 30:
                recommendations.append(
                    OptimizationRecommendation(
                        id="capacity_planning",
                        title="Proactive Capacity Expansion",
                        type=OptimizationType.CAPACITY,
                        impact="high",
                        estimated_savings=0,
                        implementation_effort="high",
                        confidence=0.78,
                        description=f"Capacity will be reached in {days_to_capacity:.0f} days",
                        steps=[
                            "Review growth projections",
                            "Plan capacity expansion",
                            "Implement auto-scaling policies",
                            "Set up capacity monitoring",
                        ],
                        risks=["Over-provisioning if growth slows"],
                        prerequisites=["Growth forecast", "Budget approval"],
                    )
                )

        return recommendations

    async def optimize_resource_allocation(
        self, resources: List[Dict], constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize resource allocation using linear programming

        Args:
            resources: List of resources with costs and capacities
            constraints: Optimization constraints

        Returns:
            Optimal allocation plan
        """

        # Create optimization problem
        prob = pulp.LpProblem("Resource_Allocation", pulp.LpMinimize)

        # Decision variables
        allocation_vars = {}
        for resource in resources:
            var_name = f"alloc_{resource['id']}"
            allocation_vars[resource["id"]] = pulp.LpVariable(
                var_name, lowBound=0, upBound=resource.get("max_capacity", 100), cat="Continuous"
            )

        # Objective function (minimize cost)
        prob += pulp.lpSum([allocation_vars[r["id"]] * r["cost_per_unit"] for r in resources])

        # Constraints
        # Total capacity constraint
        if "min_total_capacity" in constraints:
            prob += (
                pulp.lpSum([allocation_vars[r["id"]] * r["capacity_per_unit"] for r in resources])
                >= constraints["min_total_capacity"]
            )

        # Budget constraint
        if "max_budget" in constraints:
            prob += (
                pulp.lpSum([allocation_vars[r["id"]] * r["cost_per_unit"] for r in resources])
                <= constraints["max_budget"]
            )

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        allocation = {}
        total_cost = 0
        total_capacity = 0

        for resource in resources:
            allocated = allocation_vars[resource["id"]].varValue
            allocation[resource["id"]] = allocated
            total_cost += allocated * resource["cost_per_unit"]
            total_capacity += allocated * resource["capacity_per_unit"]

        return {
            "status": pulp.LpStatus[prob.status],
            "allocation": allocation,
            "total_cost": total_cost,
            "total_capacity": total_capacity,
            "optimization_value": pulp.value(prob.objective),
        }

    async def simulate_optimization_impact(
        self,
        recommendation: OptimizationRecommendation,
        historical_data: pd.DataFrame,
        simulation_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Simulate the impact of applying an optimization

        Args:
            recommendation: Optimization recommendation
            historical_data: Historical metrics
            simulation_days: Days to simulate

        Returns:
            Simulated impact metrics
        """

        # Create baseline projection
        baseline = self._project_baseline(historical_data, simulation_days)

        # Apply optimization impact
        optimized = baseline.copy()

        if recommendation.type == OptimizationType.COST:
            # Reduce costs by estimated savings percentage
            savings_pct = recommendation.estimated_savings / baseline["cost"].sum()
            optimized["cost"] = optimized["cost"] * (1 - savings_pct)

        elif recommendation.type == OptimizationType.PERFORMANCE:
            # Improve performance metrics
            optimized["response_time"] = optimized["response_time"] * 0.7  # 30% improvement
            optimized["throughput"] = optimized["throughput"] * 1.3  # 30% increase

        elif recommendation.type == OptimizationType.RESOURCE:
            # Adjust resource utilization
            optimized["utilization"] = 65  # Target utilization
            optimized["cost"] = optimized["cost"] * 0.8  # 20% cost reduction

        # Calculate impact metrics
        impact = {
            "cost_savings": baseline["cost"].sum() - optimized["cost"].sum(),
            "performance_improvement": {
                "response_time": (
                    baseline["response_time"].mean() - optimized["response_time"].mean()
                )
                / baseline["response_time"].mean()
                * 100,
                "throughput": (optimized["throughput"].mean() - baseline["throughput"].mean())
                / baseline["throughput"].mean()
                * 100,
            },
            "roi": (baseline["cost"].sum() - optimized["cost"].sum())
            / recommendation.estimated_savings
            * 100,
            "payback_period_days": abs(
                recommendation.estimated_savings
                / (baseline["cost"].mean() - optimized["cost"].mean())
            ),
            "risk_score": self._calculate_risk_score(recommendation),
            "confidence_interval": {
                "lower": recommendation.estimated_savings * 0.7,
                "upper": recommendation.estimated_savings * 1.3,
            },
        }

        return impact

    def _calculate_optimal_size(self, resource: pd.Series) -> str:
        """Calculate optimal resource size based on utilization"""

        utilization = resource["utilization"]
        current_size = resource.get("size", "medium")

        size_map = {"micro": 1, "small": 2, "medium": 4, "large": 8, "xlarge": 16}

        current_units = size_map.get(current_size, 4)
        optimal_units = current_units * (utilization / 65)  # Target 65% utilization

        # Find closest size
        for size, units in size_map.items():
            if units >= optimal_units:
                return size

        return "xlarge"

    def _impact_score(self, impact: str) -> int:
        """Convert impact to numeric score"""
        scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return scores.get(impact, 0)

    def _project_baseline(self, historical_data: pd.DataFrame, days: int) -> pd.DataFrame:
        """Project baseline metrics"""

        # Simple linear projection
        trend = historical_data.pct_change().mean()

        projection = pd.DataFrame()
        last_values = historical_data.iloc[-1]

        for i in range(days):
            daily_values = last_values * (1 + trend) ** i
            projection = pd.concat([projection, daily_values.to_frame().T])

        projection.reset_index(drop=True, inplace=True)
        return projection

    def _calculate_risk_score(self, recommendation: OptimizationRecommendation) -> float:
        """Calculate risk score for optimization"""

        # Base risk on effort and impact
        effort_risk = {"low": 0.2, "medium": 0.5, "high": 0.8}

        impact_factor = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

        base_risk = effort_risk.get(recommendation.implementation_effort, 0.5)
        impact_multiplier = impact_factor.get(recommendation.impact, 0.5)

        # Adjust by confidence
        risk_score = base_risk * impact_multiplier * (1 - recommendation.confidence)

        return min(max(risk_score, 0), 1)  # Bound between 0 and 1
