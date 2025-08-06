"""
AI Insight Generator
Generates actionable insights from analytics data
"""

import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class InsightType(str, Enum):
    ANOMALY = "anomaly"
    TREND = "trend"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"


class InsightSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Insight:
    id: str
    type: InsightType
    severity: InsightSeverity
    title: str
    description: str
    impact: str
    confidence: float
    actions: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = None


class InsightGenerator:
    """
    Generates actionable insights from various data sources
    """

    def __init__(self):
        self.insights_cache = []
        self.templates = self._load_insight_templates()

    def _load_insight_templates(self) -> Dict[str, Dict]:
        """Load insight generation templates"""
        return {
            "cost_spike": {
                "type": InsightType.ANOMALY,
                "severity": InsightSeverity.HIGH,
                "title": "Unusual Cost Spike Detected",
                "description": "Costs have increased by {increase_pct}% in the last {period}",
                "impact": "Potential budget overrun of ${amount}",
                "actions": [
                    "Review recent resource provisioning",
                    "Check for unused resources",
                    "Implement cost alerts",
                ],
            },
            "performance_degradation": {
                "type": InsightType.WARNING,
                "severity": InsightSeverity.HIGH,
                "title": "Performance Degradation Detected",
                "description": "Response time has increased by {increase_pct}% over baseline",
                "impact": "User experience may be affected",
                "actions": [
                    "Review application logs",
                    "Check resource utilization",
                    "Consider scaling resources",
                ],
            },
            "capacity_forecast": {
                "type": InsightType.PREDICTION,
                "severity": InsightSeverity.MEDIUM,
                "title": "Capacity Threshold Approaching",
                "description": "Current growth rate will reach capacity in {days} days",
                "impact": "Service availability may be affected",
                "actions": [
                    "Plan capacity expansion",
                    "Implement auto-scaling",
                    "Review growth projections",
                ],
            },
            "optimization_opportunity": {
                "type": InsightType.OPTIMIZATION,
                "severity": InsightSeverity.LOW,
                "title": "Resource Optimization Opportunity",
                "description": "{resource_count} resources are underutilized (<30%)",
                "impact": "Potential savings of ${savings}/month",
                "actions": [
                    "Review resource sizing",
                    "Consider consolidation",
                    "Implement auto-scaling",
                ],
            },
            "compliance_risk": {
                "type": InsightType.WARNING,
                "severity": InsightSeverity.CRITICAL,
                "title": "Compliance Risk Identified",
                "description": "{violation_count} resources violate {policy_name}",
                "impact": "Regulatory compliance at risk",
                "actions": [
                    "Review compliance violations",
                    "Apply remediation immediately",
                    "Update compliance policies",
                ],
            },
        }

    async def generate_insights(
        self,
        metrics_data: pd.DataFrame,
        anomalies: List[Dict],
        predictions: Dict[str, Any],
        correlations: List[Dict],
    ) -> List[Insight]:
        """
        Generate insights from multiple data sources

        Args:
            metrics_data: Current metrics
            anomalies: Detected anomalies
            predictions: Predictive analytics results
            correlations: Correlation analysis results

        Returns:
            List of generated insights
        """

        insights = []

        # Generate anomaly insights
        anomaly_insights = await self._generate_anomaly_insights(anomalies)
        insights.extend(anomaly_insights)

        # Generate trend insights
        trend_insights = await self._generate_trend_insights(metrics_data)
        insights.extend(trend_insights)

        # Generate predictive insights
        predictive_insights = await self._generate_predictive_insights(predictions)
        insights.extend(predictive_insights)

        # Generate correlation insights
        correlation_insights = await self._generate_correlation_insights(correlations)
        insights.extend(correlation_insights)

        # Generate optimization insights
        optimization_insights = await self._generate_optimization_insights(metrics_data)
        insights.extend(optimization_insights)

        # Rank and filter insights
        insights = self._rank_insights(insights)
        insights = self._filter_duplicate_insights(insights)

        # Cache insights
        self.insights_cache.extend(insights)

        return insights[:20]  # Return top 20 insights

    async def _generate_anomaly_insights(self, anomalies: List[Dict]) -> List[Insight]:
        """Generate insights from anomalies"""

        insights = []

        for anomaly in anomalies:
            severity = self._determine_anomaly_severity(anomaly)

            insight = Insight(
                id=f"anomaly_{anomaly.get('id', datetime.utcnow().timestamp())}",
                type=InsightType.ANOMALY,
                severity=severity,
                title=f"Anomaly Detected in {anomaly.get('metric', 'System')}",
                description=f"Unusual pattern detected with {anomaly.get('confidence', 0)*100:.1f}% confidence",
                impact=self._assess_anomaly_impact(anomaly),
                confidence=anomaly.get("confidence", 0.5),
                actions=self._generate_anomaly_actions(anomaly),
                data=anomaly,
                timestamp=datetime.utcnow(),
                tags=["anomaly", anomaly.get("metric", "unknown")],
            )

            insights.append(insight)

        return insights

    async def _generate_trend_insights(self, metrics_data: pd.DataFrame) -> List[Insight]:
        """Generate insights from trends"""

        insights = []

        for column in metrics_data.columns:
            if column == "date":
                continue

            # Calculate trend
            trend = metrics_data[column].pct_change().mean()

            if abs(trend) > 0.1:  # Significant trend (>10% change)
                direction = "increasing" if trend > 0 else "decreasing"

                insight = Insight(
                    id=f"trend_{column}_{datetime.utcnow().timestamp()}",
                    type=InsightType.TREND,
                    severity=self._determine_trend_severity(column, trend),
                    title=f"{column.title()} is {direction} rapidly",
                    description=f"{column} has changed by {abs(trend)*100:.1f}% on average",
                    impact=self._assess_trend_impact(column, trend),
                    confidence=0.75,
                    actions=[
                        f"Review {column} trends",
                        "Identify root causes",
                        "Adjust resource allocation if needed",
                    ],
                    data={"metric": column, "trend": trend},
                    timestamp=datetime.utcnow(),
                    tags=["trend", column],
                )

                insights.append(insight)

        return insights

    async def _generate_predictive_insights(self, predictions: Dict[str, Any]) -> List[Insight]:
        """Generate insights from predictions"""

        insights = []

        if "breach_date" in predictions and predictions["breach_date"]:
            breach_date = datetime.fromisoformat(predictions["breach_date"])
            days_until = (breach_date - datetime.utcnow()).days

            insight = Insight(
                id=f"prediction_capacity_{datetime.utcnow().timestamp()}",
                type=InsightType.PREDICTION,
                severity=InsightSeverity.HIGH if days_until < 30 else InsightSeverity.MEDIUM,
                title="Capacity Limit Approaching",
                description=f"System will reach capacity in {days_until} days",
                impact="Service availability may be affected",
                confidence=predictions.get("confidence", 0.8),
                actions=[
                    "Plan capacity expansion",
                    "Review growth projections",
                    "Implement auto-scaling",
                ],
                data=predictions,
                timestamp=datetime.utcnow(),
                expires_at=breach_date,
                tags=["capacity", "prediction"],
            )

            insights.append(insight)

        # Generate forecast insights
        if "forecast" in predictions:
            forecast_values = predictions["forecast"]
            if len(forecast_values) > 0:
                max_value = max(forecast_values)
                min_value = min(forecast_values)
                volatility = (max_value - min_value) / np.mean(forecast_values)

                if volatility > 0.3:  # High volatility
                    insight = Insight(
                        id=f"prediction_volatility_{datetime.utcnow().timestamp()}",
                        type=InsightType.WARNING,
                        severity=InsightSeverity.MEDIUM,
                        title="High Volatility Predicted",
                        description=f"Expect {volatility*100:.1f}% volatility in coming period",
                        impact="Budget and resource planning may be affected",
                        confidence=predictions.get("confidence", 0.7),
                        actions=[
                            "Prepare for demand fluctuations",
                            "Implement flexible scaling",
                            "Review contingency plans",
                        ],
                        data={"volatility": volatility},
                        timestamp=datetime.utcnow(),
                        tags=["volatility", "prediction"],
                    )

                    insights.append(insight)

        return insights

    async def _generate_correlation_insights(self, correlations: List[Dict]) -> List[Insight]:
        """Generate insights from correlations"""

        insights = []

        for correlation in correlations[:5]:  # Top 5 correlations
            if abs(correlation["correlation"]) > 0.7:
                relationship = "positive" if correlation["correlation"] > 0 else "negative"

                insight = Insight(
                    id=f"correlation_{correlation['metric1']}_{correlation['metric2']}",
                    type=InsightType.OPPORTUNITY,
                    severity=InsightSeverity.INFO,
                    title=f"Strong {relationship} correlation discovered",
                    description=f"{correlation['metric1']} and {correlation['metric2']} are {relationship}ly correlated ({correlation['correlation']:.2f})",
                    impact="Optimization opportunity identified",
                    confidence=1 - correlation.get("p_value", 0.05),
                    actions=[
                        f"Optimize {correlation['metric1']} to impact {correlation['metric2']}",
                        "Consider combined optimization strategy",
                        "Monitor both metrics together",
                    ],
                    data=correlation,
                    timestamp=datetime.utcnow(),
                    tags=["correlation", correlation["metric1"], correlation["metric2"]],
                )

                insights.append(insight)

        return insights

    async def _generate_optimization_insights(self, metrics_data: pd.DataFrame) -> List[Insight]:
        """Generate optimization insights"""

        insights = []

        # Check for optimization opportunities
        if "utilization" in metrics_data.columns:
            low_utilization = metrics_data[metrics_data["utilization"] < 30]

            if len(low_utilization) > 0:
                potential_savings = len(low_utilization) * 100  # Rough estimate

                insight = Insight(
                    id=f"optimization_utilization_{datetime.utcnow().timestamp()}",
                    type=InsightType.OPTIMIZATION,
                    severity=InsightSeverity.MEDIUM,
                    title="Resource Optimization Opportunity",
                    description=f"{len(low_utilization)} resources are underutilized",
                    impact=f"Potential savings of ${potential_savings}/month",
                    confidence=0.85,
                    actions=[
                        "Review underutilized resources",
                        "Consider rightsizing",
                        "Implement auto-scaling",
                    ],
                    data={"resource_count": len(low_utilization)},
                    timestamp=datetime.utcnow(),
                    tags=["optimization", "cost-saving"],
                )

                insights.append(insight)

        return insights

    def _determine_anomaly_severity(self, anomaly: Dict) -> InsightSeverity:
        """Determine severity of anomaly"""

        score = anomaly.get("anomaly_score", 0)

        if score > 0.8:
            return InsightSeverity.CRITICAL
        elif score > 0.6:
            return InsightSeverity.HIGH
        elif score > 0.4:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW

    def _determine_trend_severity(self, metric: str, trend: float) -> InsightSeverity:
        """Determine severity of trend"""

        critical_metrics = ["cost", "errors", "violations"]
        important_metrics = ["utilization", "response_time"]

        if metric in critical_metrics and abs(trend) > 0.2:
            return InsightSeverity.HIGH
        elif metric in important_metrics and abs(trend) > 0.3:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW

    def _assess_anomaly_impact(self, anomaly: Dict) -> str:
        """Assess impact of anomaly"""

        metric = anomaly.get("metric", "unknown")
        severity = anomaly.get("severity", "medium")

        impact_map = {
            "cost": "Budget impact possible",
            "performance": "User experience may be affected",
            "security": "Security posture may be compromised",
            "compliance": "Compliance requirements may be violated",
        }

        return impact_map.get(metric, "System behavior outside normal parameters")

    def _assess_trend_impact(self, metric: str, trend: float) -> str:
        """Assess impact of trend"""

        if metric == "cost" and trend > 0:
            return f"Costs increasing at {trend*100:.1f}% rate"
        elif metric == "utilization" and trend > 0:
            return "Capacity constraints may be approaching"
        elif metric == "errors" and trend > 0:
            return "System reliability may be degrading"
        else:
            return f"{metric} showing significant change"

    def _generate_anomaly_actions(self, anomaly: Dict) -> List[str]:
        """Generate actions for anomaly"""

        metric = anomaly.get("metric", "unknown")

        action_map = {
            "cost": [
                "Review recent billing details",
                "Check for resource leaks",
                "Verify auto-scaling settings",
            ],
            "performance": [
                "Check application logs",
                "Review recent deployments",
                "Monitor system resources",
            ],
            "security": [
                "Review security logs",
                "Check for unauthorized access",
                "Verify security policies",
            ],
        }

        return action_map.get(metric, ["Investigate anomaly", "Review system logs"])

    def _rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """Rank insights by importance"""

        severity_scores = {
            InsightSeverity.CRITICAL: 5,
            InsightSeverity.HIGH: 4,
            InsightSeverity.MEDIUM: 3,
            InsightSeverity.LOW: 2,
            InsightSeverity.INFO: 1,
        }

        type_scores = {
            InsightType.ANOMALY: 3,
            InsightType.WARNING: 3,
            InsightType.PREDICTION: 2,
            InsightType.TREND: 2,
            InsightType.OPTIMIZATION: 1,
            InsightType.RECOMMENDATION: 1,
            InsightType.OPPORTUNITY: 1,
        }

        def score_insight(insight: Insight) -> float:
            severity_score = severity_scores.get(insight.severity, 0)
            type_score = type_scores.get(insight.type, 0)
            confidence_score = insight.confidence

            # Recency bonus (newer insights score higher)
            age_hours = (datetime.utcnow() - insight.timestamp).total_seconds() / 3600
            recency_score = max(0, 1 - age_hours / 24)  # Decay over 24 hours

            return severity_score * 2 + type_score + confidence_score + recency_score

        insights.sort(key=score_insight, reverse=True)
        return insights

    def _filter_duplicate_insights(self, insights: List[Insight]) -> List[Insight]:
        """Filter duplicate or similar insights"""

        filtered = []
        seen_keys = set()

        for insight in insights:
            # Create a key based on type and main metric
            key = f"{insight.type}_{insight.tags[0] if insight.tags else ''}"

            if key not in seen_keys:
                filtered.append(insight)
                seen_keys.add(key)

        return filtered

    async def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of current insights"""

        # Clean expired insights
        self.insights_cache = [
            i for i in self.insights_cache if not i.expires_at or i.expires_at > datetime.utcnow()
        ]

        # Count by severity
        severity_counts = {}
        for severity in InsightSeverity:
            severity_counts[severity.value] = sum(
                1 for i in self.insights_cache if i.severity == severity
            )

        # Count by type
        type_counts = {}
        for insight_type in InsightType:
            type_counts[insight_type.value] = sum(
                1 for i in self.insights_cache if i.type == insight_type
            )

        return {
            "total_insights": len(self.insights_cache),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "critical_count": severity_counts.get(InsightSeverity.CRITICAL.value, 0),
            "high_priority_count": (
                severity_counts.get(InsightSeverity.CRITICAL.value, 0)
                + severity_counts.get(InsightSeverity.HIGH.value, 0)
            ),
            "last_updated": datetime.utcnow().isoformat(),
        }
