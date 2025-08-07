"""
Real-time Compliance Analysis Engine
Analyzes resources against extracted policies and provides compliance scores
"""

import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from pydantic import BaseModel, Field, validator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class ComplianceStatus(str, Enum):
    """Compliance status for resources"""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ComplianceLevel(str, Enum):
    """Overall compliance level"""

    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"  # 80-94%
    FAIR = "fair"  # 60-79%
    POOR = "poor"  # 40-59%
    CRITICAL = "critical"  # <40%


class ResourceCompliance(BaseModel):
    """Compliance status for a single resource"""

    resource_id: str
    resource_type: str
    resource_name: str
    compliance_status: ComplianceStatus
    compliance_score: float = Field(ge=0.0, le=100.0)
    policy_violations: List[Dict[str, Any]] = Field(default_factory=list)
    last_checked: datetime = Field(default_factory=datetime.utcnow)
    remediation_required: bool = False
    remediation_actions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceReport(BaseModel):
    """Comprehensive compliance report"""

    report_id: str
    tenant_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_score: float = Field(ge=0.0, le=100.0)
    compliance_level: ComplianceLevel
    total_resources: int
    compliant_resources: int
    non_compliant_resources: int
    critical_violations: int
    high_violations: int
    medium_violations: int
    low_violations: int
    resource_compliance: List[ResourceCompliance] = Field(default_factory=list)
    policy_coverage: Dict[str, float] = Field(default_factory=dict)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)


class ComplianceAnalyzer:
    """
    Real-time compliance analysis engine that evaluates resources against policies
    """

    def __init__(self):
        self.anomaly_detector = None
        self.compliance_predictor = None
        self.scaler = StandardScaler()
        self.compliance_cache = {}
        self.historical_data = defaultdict(list)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for compliance analysis"""
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42
        )

        # Random Forest for compliance prediction
        self.compliance_predictor = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )

    async def analyze_compliance(
        self,
        resources: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
        tenant_id: str,
        real_time: bool = True,
    ) -> ComplianceReport:
        """
        Analyze compliance of resources against policies

        Args:
            resources: List of resources to analyze
            policies: List of policies to check against
            tenant_id: Tenant identifier
            real_time: Whether to perform real-time analysis

        Returns:
            Comprehensive compliance report
        """
        report_id = f"report_{datetime.utcnow().timestamp()}_{tenant_id}"

        # Analyze each resource
        resource_compliance_list = []
        total_score = 0
        violation_counts = defaultdict(int)

        for resource in resources:
            if real_time:
                # Real-time analysis
                compliance = await self._analyze_resource_real_time(resource, policies)
            else:
                # Batch analysis (from cache if available)
                compliance = self._analyze_resource_batch(resource, policies)

            resource_compliance_list.append(compliance)
            total_score += compliance.compliance_score

            # Count violations by severity
            for violation in compliance.policy_violations:
                severity = violation.get("severity", "medium")
                violation_counts[severity] += 1

        # Calculate overall metrics
        total_resources = len(resources)
        compliant_count = sum(
            1 for r in resource_compliance_list if r.compliance_status == ComplianceStatus.COMPLIANT
        )
        non_compliant_count = sum(
            1
            for r in resource_compliance_list
            if r.compliance_status == ComplianceStatus.NON_COMPLIANT
        )

        overall_score = (total_score / total_resources) if total_resources > 0 else 0

        # Determine compliance level
        compliance_level = self._determine_compliance_level(overall_score)

        # Generate policy coverage analysis
        policy_coverage = self._analyze_policy_coverage(resource_compliance_list, policies)

        # Perform trend analysis
        trend_analysis = await self._analyze_trends(
            tenant_id, overall_score, resource_compliance_list
        )

        # Risk assessment
        risk_assessment = self._assess_risks(resource_compliance_list, violation_counts)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            resource_compliance_list, policy_coverage, risk_assessment
        )

        # Create report
        report = ComplianceReport(
            report_id=report_id,
            tenant_id=tenant_id,
            overall_score=overall_score,
            compliance_level=compliance_level,
            total_resources=total_resources,
            compliant_resources=compliant_count,
            non_compliant_resources=non_compliant_count,
            critical_violations=violation_counts.get("critical", 0),
            high_violations=violation_counts.get("high", 0),
            medium_violations=violation_counts.get("medium", 0),
            low_violations=violation_counts.get("low", 0),
            resource_compliance=resource_compliance_list,
            policy_coverage=policy_coverage,
            trend_analysis=trend_analysis,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
        )

        # Store in historical data
        self.historical_data[tenant_id].append(
            {"timestamp": datetime.utcnow(), "score": overall_score, "report_id": report_id}
        )

        return report

    async def _analyze_resource_real_time(
        self, resource: Dict[str, Any], policies: List[Dict[str, Any]]
    ) -> ResourceCompliance:
        """Perform real-time compliance analysis for a single resource"""
        resource_id = resource.get("id", "unknown")
        resource_type = resource.get("type", "unknown")
        resource_name = resource.get("name", resource_id)

        violations = []
        total_checks = 0
        passed_checks = 0
        remediation_actions = []

        # Check resource against each applicable policy
        for policy in policies:
            if self._is_policy_applicable(resource, policy):
                total_checks += 1

                # Evaluate policy conditions
                evaluation_result = self._evaluate_policy(resource, policy)

                if evaluation_result["compliant"]:
                    passed_checks += 1
                else:
                    violations.append(
                        {
                            "policy_id": policy.get("id"),
                            "policy_name": policy.get("name"),
                            "severity": policy.get("severity", "medium"),
                            "description": evaluation_result.get("description"),
                            "details": evaluation_result.get("details"),
                        }
                    )

                    # Add remediation actions
                    if "remediation" in evaluation_result:
                        remediation_actions.extend(evaluation_result["remediation"])

        # Calculate compliance score
        compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100

        # Determine compliance status
        if compliance_score == 100:
            status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 80:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        elif compliance_score > 0:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.UNKNOWN

        # Check for anomalies
        anomaly_score = await self._detect_anomalies(resource)

        return ResourceCompliance(
            resource_id=resource_id,
            resource_type=resource_type,
            resource_name=resource_name,
            compliance_status=status,
            compliance_score=compliance_score,
            policy_violations=violations,
            remediation_required=len(violations) > 0,
            remediation_actions=list(set(remediation_actions)),
            metadata={
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "anomaly_score": anomaly_score,
            },
        )

    def _analyze_resource_batch(
        self, resource: Dict[str, Any], policies: List[Dict[str, Any]]
    ) -> ResourceCompliance:
        """Batch compliance analysis (synchronous)"""
        # Check cache first
        cache_key = self._generate_cache_key(resource, policies)

        if cache_key in self.compliance_cache:
            cached_result = self.compliance_cache[cache_key]
            # Check if cache is still valid (1 hour)
            if (datetime.utcnow() - cached_result["timestamp"]).seconds < 3600:
                return cached_result["compliance"]

        # Perform analysis (similar to real-time but synchronous)
        compliance = asyncio.run(self._analyze_resource_real_time(resource, policies))

        # Cache result
        self.compliance_cache[cache_key] = {
            "compliance": compliance,
            "timestamp": datetime.utcnow(),
        }

        return compliance

    def _is_policy_applicable(self, resource: Dict[str, Any], policy: Dict[str, Any]) -> bool:
        """Check if a policy applies to a resource"""
        # Check resource type filter
        if "resource_types" in policy:
            if resource.get("type") not in policy["resource_types"]:
                return False

        # Check scope filter
        if "scope" in policy:
            resource_scope = resource.get("scope", "")
            policy_scope = policy["scope"]

            if not resource_scope.startswith(policy_scope):
                return False

        # Check tags filter
        if "required_tags" in policy:
            resource_tags = resource.get("tags", {})
            for tag_key in policy["required_tags"]:
                if tag_key not in resource_tags:
                    return False

        return True

    def _evaluate_policy(self, resource: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a policy against a resource"""
        result = {"compliant": True, "description": "", "details": {}, "remediation": []}

        # Get evaluation criteria from policy
        criteria = policy.get("evaluation_criteria", {})

        if criteria.get("type") == "composite":
            # Composite criteria with multiple conditions
            operator = criteria.get("operator", "AND")
            conditions = criteria.get("conditions", [])

            condition_results = []
            for condition in conditions:
                cond_result = self._evaluate_condition(resource, condition)
                condition_results.append(cond_result["compliant"])

                if not cond_result["compliant"]:
                    result["details"][condition.get("name", "condition")] = cond_result

            # Apply operator
            if operator == "AND":
                result["compliant"] = all(condition_results)
            elif operator == "OR":
                result["compliant"] = any(condition_results)

        else:
            # Simple criteria
            result = self._evaluate_condition(resource, criteria)

        # Add remediation steps if non-compliant
        if not result["compliant"]:
            result["description"] = f"Resource violates policy: {policy.get('name')}"
            result["remediation"] = policy.get(
                "remediation_steps",
                [f"Review and update resource configuration to comply with {policy.get('name')}"],
            )

        return result

    def _evaluate_condition(
        self, resource: Dict[str, Any], condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single condition"""
        result = {"compliant": True, "description": ""}

        condition_type = condition.get("type")

        if condition_type == "property":
            # Check resource property
            property_path = condition.get("property")
            expected_value = condition.get("value")
            operator = condition.get("operator", "equals")

            actual_value = self._get_nested_property(resource, property_path)

            if operator == "equals":
                result["compliant"] = actual_value == expected_value
            elif operator == "not_equals":
                result["compliant"] = actual_value != expected_value
            elif operator == "contains":
                result["compliant"] = expected_value in str(actual_value)
            elif operator == "greater_than":
                result["compliant"] = float(actual_value) > float(expected_value)
            elif operator == "less_than":
                result["compliant"] = float(actual_value) < float(expected_value)

            if not result["compliant"]:
                result[
                    "description"
                ] = f"Property {property_path} is {actual_value}, expected {operator} {expected_value}"

        elif condition_type == "tag":
            # Check resource tags
            tag_key = condition.get("key")
            tag_value = condition.get("value")

            resource_tags = resource.get("tags", {})

            if tag_key not in resource_tags:
                result["compliant"] = False
                result["description"] = f"Required tag '{tag_key}' is missing"
            elif tag_value and resource_tags[tag_key] != tag_value:
                result["compliant"] = False
                result[
                    "description"
                ] = f"Tag '{tag_key}' has value '{resource_tags[tag_key]}', expected '{tag_value}'"

        elif condition_type == "custom":
            # Custom evaluation logic
            result = self._evaluate_custom_condition(resource, condition)

        return result

    def _get_nested_property(self, obj: Dict, path: str) -> Any:
        """Get nested property from object using dot notation"""
        keys = path.split(".")
        value = obj

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    def _evaluate_custom_condition(
        self, resource: Dict[str, Any], condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate custom conditions"""
        result = {"compliant": True, "description": ""}

        custom_type = condition.get("custom_type")

        if custom_type == "encryption":
            # Check if resource is encrypted
            encryption_config = resource.get("properties", {}).get("encryption", {})
            if not encryption_config.get("enabled", False):
                result["compliant"] = False
                result["description"] = "Resource is not encrypted"

        elif custom_type == "public_access":
            # Check for public access
            network_config = resource.get("properties", {}).get("networkAcls", {})
            if network_config.get("defaultAction") == "Allow":
                result["compliant"] = False
                result["description"] = "Resource allows public access"

        elif custom_type == "backup":
            # Check backup configuration
            backup_config = resource.get("properties", {}).get("backup", {})
            if not backup_config.get("enabled", False):
                result["compliant"] = False
                result["description"] = "Backup is not configured"

        return result

    async def _detect_anomalies(self, resource: Dict[str, Any]) -> float:
        """Detect anomalies in resource configuration"""
        try:
            # Extract features for anomaly detection
            features = self._extract_features(resource)

            if len(features) > 0:
                # Reshape for sklearn
                feature_array = np.array(features).reshape(1, -1)

                # Scale features
                scaled_features = self.scaler.fit_transform(feature_array)

                # Detect anomalies (if model is trained)
                if hasattr(self.anomaly_detector, "offset_"):
                    anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
                    # Normalize to 0-1 range
                    return abs(anomaly_score)

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")

        return 0.0

    def _extract_features(self, resource: Dict[str, Any]) -> List[float]:
        """Extract numerical features from resource for ML analysis"""
        features = []

        # Resource age (if creation time available)
        if "created_at" in resource:
            created = datetime.fromisoformat(resource["created_at"])
            age_days = (datetime.utcnow() - created).days
            features.append(age_days)

        # Configuration complexity (number of properties)
        properties = resource.get("properties", {})
        features.append(len(properties))

        # Tag count
        tags = resource.get("tags", {})
        features.append(len(tags))

        # Network rules count
        network_rules = resource.get("properties", {}).get("networkAcls", {}).get("ipRules", [])
        features.append(len(network_rules))

        # Cost (if available)
        cost = resource.get("cost", {}).get("monthly", 0)
        features.append(float(cost))

        return features

    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level based on score"""
        if score >= 95:
            return ComplianceLevel.EXCELLENT
        elif score >= 80:
            return ComplianceLevel.GOOD
        elif score >= 60:
            return ComplianceLevel.FAIR
        elif score >= 40:
            return ComplianceLevel.POOR
        else:
            return ComplianceLevel.CRITICAL

    def _analyze_policy_coverage(
        self, resource_compliance: List[ResourceCompliance], policies: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze policy coverage across resources"""
        coverage = {}

        for policy in policies:
            policy_id = policy.get("id")
            policy_name = policy.get("name", policy_id)

            # Count resources checked against this policy
            checked_count = 0
            compliant_count = 0

            for resource in resource_compliance:
                violations = [
                    v for v in resource.policy_violations if v.get("policy_id") == policy_id
                ]

                if violations:
                    checked_count += 1
                elif any(policy_id in str(resource.metadata) for _ in [1]):
                    checked_count += 1
                    compliant_count += 1

            # Calculate coverage percentage
            if checked_count > 0:
                coverage[policy_name] = (compliant_count / checked_count) * 100
            else:
                coverage[policy_name] = 100.0  # No resources to check

        return coverage

    async def _analyze_trends(
        self, tenant_id: str, current_score: float, resource_compliance: List[ResourceCompliance]
    ) -> Dict[str, Any]:
        """Analyze compliance trends over time"""
        trends = {
            "score_trend": "stable",
            "score_change": 0.0,
            "improvement_rate": 0.0,
            "violation_trend": {},
        }

        # Get historical data
        history = self.historical_data.get(tenant_id, [])

        if len(history) > 1:
            # Calculate score trend
            previous_score = history[-2]["score"]
            score_change = current_score - previous_score

            trends["score_change"] = score_change

            if score_change > 5:
                trends["score_trend"] = "improving"
            elif score_change < -5:
                trends["score_trend"] = "declining"
            else:
                trends["score_trend"] = "stable"

            # Calculate improvement rate (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_history = [h for h in history if h["timestamp"] > thirty_days_ago]

            if len(recent_history) > 1:
                first_score = recent_history[0]["score"]
                improvement = current_score - first_score
                days_elapsed = (datetime.utcnow() - recent_history[0]["timestamp"]).days

                if days_elapsed > 0:
                    trends["improvement_rate"] = improvement / days_elapsed

        # Analyze violation trends
        violation_counts = defaultdict(int)
        for resource in resource_compliance:
            for violation in resource.policy_violations:
                severity = violation.get("severity", "medium")
                violation_counts[severity] += 1

        trends["violation_trend"] = dict(violation_counts)

        return trends

    def _assess_risks(
        self, resource_compliance: List[ResourceCompliance], violation_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Assess compliance risks"""
        risk_assessment = {
            "overall_risk": "low",
            "risk_score": 0,
            "critical_resources": [],
            "risk_factors": [],
        }

        # Calculate risk score
        risk_score = (
            violation_counts.get("critical", 0) * 10
            + violation_counts.get("high", 0) * 5
            + violation_counts.get("medium", 0) * 2
            + violation_counts.get("low", 0) * 1
        )

        risk_assessment["risk_score"] = risk_score

        # Determine overall risk level
        if risk_score >= 50:
            risk_assessment["overall_risk"] = "critical"
        elif risk_score >= 30:
            risk_assessment["overall_risk"] = "high"
        elif risk_score >= 15:
            risk_assessment["overall_risk"] = "medium"
        else:
            risk_assessment["overall_risk"] = "low"

        # Identify critical resources
        for resource in resource_compliance:
            critical_violations = [
                v for v in resource.policy_violations if v.get("severity") == "critical"
            ]

            if critical_violations:
                risk_assessment["critical_resources"].append(
                    {
                        "resource_id": resource.resource_id,
                        "resource_name": resource.resource_name,
                        "violations": len(critical_violations),
                    }
                )

        # Identify risk factors
        if violation_counts.get("critical", 0) > 0:
            risk_assessment["risk_factors"].append(
                f"{violation_counts['critical']} critical violations detected"
            )

        non_compliant_percentage = (
            sum(
                1
                for r in resource_compliance
                if r.compliance_status == ComplianceStatus.NON_COMPLIANT
            )
            / len(resource_compliance)
            * 100
            if resource_compliance
            else 0
        )

        if non_compliant_percentage > 30:
            risk_assessment["risk_factors"].append(
                f"{non_compliant_percentage:.1f}% of resources are non-compliant"
            )

        return risk_assessment

    def _generate_recommendations(
        self,
        resource_compliance: List[ResourceCompliance],
        policy_coverage: Dict[str, float],
        risk_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        # Risk-based recommendations
        if risk_assessment["overall_risk"] in ["critical", "high"]:
            recommendations.append(
                "URGENT: Address critical compliance violations immediately to reduce risk"
            )

        # Coverage-based recommendations
        low_coverage_policies = [
            policy for policy, coverage in policy_coverage.items() if coverage < 70
        ]

        if low_coverage_policies:
            recommendations.append(
                f"Improve compliance for policies: {', '.join(low_coverage_policies[:3])}"
            )

        # Resource-specific recommendations
        resources_needing_remediation = [r for r in resource_compliance if r.remediation_required]

        if len(resources_needing_remediation) > 5:
            recommendations.append(
                f"Implement automated remediation for {len(resources_needing_remediation)} non-compliant resources"
            )

        # Pattern-based recommendations
        common_violations = defaultdict(int)
        for resource in resource_compliance:
            for violation in resource.policy_violations:
                common_violations[violation.get("policy_name", "Unknown")] += 1

        if common_violations:
            most_common = max(common_violations, key=common_violations.get)
            recommendations.append(
                f"Focus on '{most_common}' policy - most frequent violation ({common_violations[most_common]} occurrences)"
            )

        # General recommendations
        if not recommendations:
            if risk_assessment["risk_score"] < 10:
                recommendations.append(
                    "Maintain current compliance practices and monitor for changes"
                )
            else:
                recommendations.append(
                    "Review and update compliance policies to address emerging risks"
                )

        return recommendations

    def _generate_cache_key(self, resource: Dict[str, Any], policies: List[Dict[str, Any]]) -> str:
        """Generate cache key for compliance results"""
        # Create a hash of resource and policies
        resource_str = json.dumps(resource, sort_keys=True)
        policies_str = json.dumps([p.get("id") for p in policies], sort_keys=True)

        combined = f"{resource_str}:{policies_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def predict_future_compliance(
        self, tenant_id: str, days_ahead: int = 7
    ) -> Dict[str, Any]:
        """Predict future compliance trends using ML"""
        history = self.historical_data.get(tenant_id, [])

        if len(history) < 10:
            return {
                "prediction_available": False,
                "message": "Insufficient historical data for prediction",
            }

        # Prepare time series data
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Simple trend prediction (can be enhanced with Prophet or ARIMA)
        recent_scores = df["score"].tail(30).values
        trend = np.polyfit(range(len(recent_scores)), recent_scores, deg=1)

        # Predict future scores
        future_scores = []
        for day in range(1, days_ahead + 1):
            predicted_score = trend[0] * (len(recent_scores) + day) + trend[1]
            # Bound between 0 and 100
            predicted_score = max(0, min(100, predicted_score))
            future_scores.append(predicted_score)

        return {
            "prediction_available": True,
            "current_score": recent_scores[-1] if len(recent_scores) > 0 else 0,
            "predicted_scores": future_scores,
            "trend_direction": "improving" if trend[0] > 0 else "declining",
            "confidence": 0.75,  # Can be calculated based on model metrics
        }
