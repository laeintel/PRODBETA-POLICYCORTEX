"""
PolicyCortex Impact Analyzer
Translates technical metrics to business outcomes with risk-to-money conversion
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class BusinessMetric(Enum):
    """Business metrics that executives care about"""
    REVENUE_IMPACT = "Revenue Impact"
    COST_REDUCTION = "Cost Reduction"
    RISK_MITIGATION = "Risk Mitigation"
    OPERATIONAL_EFFICIENCY = "Operational Efficiency"
    COMPLIANCE_POSTURE = "Compliance Posture"
    CUSTOMER_SATISFACTION = "Customer Satisfaction"
    TIME_TO_MARKET = "Time to Market"
    SECURITY_POSTURE = "Security Posture"

class RiskCategory(Enum):
    """Risk categories with business impact"""
    DATA_BREACH = "Data Breach"
    SERVICE_OUTAGE = "Service Outage"
    COMPLIANCE_VIOLATION = "Compliance Violation"
    REPUTATION_DAMAGE = "Reputation Damage"
    INTELLECTUAL_PROPERTY = "IP Theft"
    OPERATIONAL_FAILURE = "Operational Failure"
    SUPPLY_CHAIN = "Supply Chain Disruption"
    REGULATORY_PENALTY = "Regulatory Penalty"

@dataclass
class BusinessImpact:
    """Quantified business impact"""
    metric: BusinessMetric
    current_value: Decimal
    target_value: Decimal
    improvement: Decimal
    improvement_percentage: float
    financial_impact: Decimal
    confidence_level: float
    time_to_realize: int  # days
    supporting_evidence: List[str] = field(default_factory=list)
    
    def to_executive_summary(self) -> str:
        """Generate executive-friendly summary"""
        return (
            f"{self.metric.value}: ${self.financial_impact:,.0f} impact "
            f"({self.improvement_percentage:.1f}% improvement) - "
            f"Confidence: {self.confidence_level:.0%}"
        )

@dataclass
class RiskAssessment:
    """Risk assessment with financial quantification"""
    risk_category: RiskCategory
    probability: float  # 0-1
    impact_if_realized: Decimal
    annual_risk_exposure: Decimal  # probability * impact
    mitigation_cost: Decimal
    residual_risk: Decimal
    risk_reduction_roi: float
    
    def risk_score(self) -> float:
        """Calculate risk score (0-100)"""
        return min(100, self.probability * 100 * float(
            self.impact_if_realized / Decimal("1000000")
        ))

class ImpactAnalyzer:
    """
    Translates technical governance metrics into business outcomes
    Provides executive-level insights with financial quantification
    """
    
    def __init__(self):
        # Industry benchmarks for business impact calculations
        self.industry_benchmarks = {
            "downtime_cost_per_hour": {
                "small": Decimal("1000"),
                "medium": Decimal("10000"),
                "large": Decimal("100000"),
                "enterprise": Decimal("1000000")
            },
            "data_breach_cost_per_record": Decimal("165"),  # IBM Cost of Data Breach 2024
            "compliance_violation_percentage_revenue": {
                "gdpr": 0.04,  # 4% of annual revenue
                "hipaa": 0.02,  # Variable, using conservative estimate
                "sox": 0.05,   # Including criminal penalties
                "pci_dss": 0.03  # Based on tier
            },
            "customer_churn_cost": Decimal("500"),  # Per customer
            "productivity_hour_value": Decimal("75"),
            "reputation_recovery_cost_multiplier": 3.5,  # Times incident cost
            "market_cap_impact": {  # Percentage impact on valuation
                "minor_incident": 0.001,
                "moderate_incident": 0.01,
                "major_incident": 0.05,
                "catastrophic": 0.20
            }
        }
        
        # Business value drivers
        self.value_drivers = {
            "automation": {
                "hours_saved_multiplier": 1.5,
                "error_reduction": 0.95,
                "speed_improvement": 2.0
            },
            "compliance": {
                "audit_cost_reduction": 0.60,
                "penalty_avoidance": 0.99,
                "certification_speed": 0.50
            },
            "security": {
                "incident_reduction": 0.80,
                "detection_speed": 0.90,
                "response_time": 0.75
            }
        }
        
        logger.info("Impact Analyzer initialized with industry benchmarks")
    
    def analyze_technical_metrics(
        self,
        metrics: Dict[str, Any],
        company_profile: Optional[Dict[str, Any]] = None
    ) -> List[BusinessImpact]:
        """
        Convert technical metrics to business impacts
        
        Args:
            metrics: Technical metrics from governance platform
            company_profile: Company size, industry, revenue etc.
            
        Returns:
            List of quantified business impacts
        """
        impacts = []
        company_size = self._determine_company_size(company_profile)
        
        # Analyze cost reduction opportunities
        if "resource_optimization" in metrics:
            cost_impact = self._analyze_cost_reduction(
                metrics["resource_optimization"],
                company_size
            )
            impacts.append(cost_impact)
        
        # Analyze risk mitigation value
        if "security_posture" in metrics:
            risk_impact = self._analyze_risk_mitigation(
                metrics["security_posture"],
                company_profile
            )
            impacts.append(risk_impact)
        
        # Analyze operational efficiency gains
        if "automation_metrics" in metrics:
            efficiency_impact = self._analyze_operational_efficiency(
                metrics["automation_metrics"],
                company_size
            )
            impacts.append(efficiency_impact)
        
        # Analyze compliance improvements
        if "compliance_score" in metrics:
            compliance_impact = self._analyze_compliance_impact(
                metrics["compliance_score"],
                company_profile
            )
            impacts.append(compliance_impact)
        
        # Analyze time-to-market improvements
        if "deployment_metrics" in metrics:
            ttm_impact = self._analyze_time_to_market(
                metrics["deployment_metrics"],
                company_profile
            )
            impacts.append(ttm_impact)
        
        return impacts
    
    def convert_risk_to_dollars(
        self,
        risk_profile: Dict[str, Any],
        company_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert risk scores to dollar amounts
        
        Args:
            risk_profile: Risk assessment data
            company_profile: Company context for accurate conversion
            
        Returns:
            Financial quantification of risks
        """
        company_size = self._determine_company_size(company_profile)
        annual_revenue = Decimal(str(
            company_profile.get("annual_revenue", 100000000) if company_profile else 100000000
        ))
        
        risk_assessments = []
        total_risk_exposure = Decimal("0")
        
        for risk_type, risk_data in risk_profile.items():
            risk_category = self._map_risk_category(risk_type)
            
            # Calculate probability based on risk score
            risk_score = risk_data.get("score", 50)
            probability = self._calculate_risk_probability(risk_score)
            
            # Calculate financial impact if risk materializes
            impact = self._calculate_risk_impact(
                risk_category,
                risk_data,
                company_size,
                annual_revenue
            )
            
            # Calculate annual risk exposure (probability * impact)
            annual_exposure = impact * Decimal(str(probability))
            
            # Estimate mitigation cost
            mitigation_cost = self._estimate_mitigation_cost(
                risk_category,
                impact
            )
            
            # Calculate residual risk after mitigation
            mitigation_effectiveness = risk_data.get("mitigation_effectiveness", 0.7)
            residual_risk = annual_exposure * Decimal(str(1 - mitigation_effectiveness))
            
            # Calculate ROI of risk mitigation
            risk_reduction = annual_exposure - residual_risk
            risk_roi = float(
                (risk_reduction - mitigation_cost) / mitigation_cost * 100
            ) if mitigation_cost > 0 else 0
            
            assessment = RiskAssessment(
                risk_category=risk_category,
                probability=probability,
                impact_if_realized=impact,
                annual_risk_exposure=annual_exposure,
                mitigation_cost=mitigation_cost,
                residual_risk=residual_risk,
                risk_reduction_roi=risk_roi
            )
            
            risk_assessments.append(assessment)
            total_risk_exposure += annual_exposure
        
        # Sort by exposure
        risk_assessments.sort(
            key=lambda x: x.annual_risk_exposure,
            reverse=True
        )
        
        return {
            "total_annual_risk_exposure": float(total_risk_exposure),
            "risk_assessments": [
                {
                    "category": ra.risk_category.value,
                    "probability": f"{ra.probability:.1%}",
                    "impact_if_realized": float(ra.impact_if_realized),
                    "annual_exposure": float(ra.annual_risk_exposure),
                    "mitigation_cost": float(ra.mitigation_cost),
                    "residual_risk": float(ra.residual_risk),
                    "roi": f"{ra.risk_reduction_roi:.0f}%",
                    "risk_score": ra.risk_score()
                }
                for ra in risk_assessments
            ],
            "recommended_mitigation_budget": float(
                sum(ra.mitigation_cost for ra in risk_assessments[:5])  # Top 5 risks
            ),
            "potential_risk_reduction": float(
                sum(ra.annual_risk_exposure - ra.residual_risk 
                    for ra in risk_assessments[:5])
            )
        }
    
    def calculate_incident_cost(
        self,
        incident_type: str,
        severity: str,
        duration_hours: float,
        affected_systems: int,
        company_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate the total cost of an incident
        
        Args:
            incident_type: Type of incident (outage, breach, etc.)
            severity: Incident severity (minor, moderate, major, critical)
            duration_hours: Duration of the incident
            affected_systems: Number of affected systems
            company_profile: Company context
            
        Returns:
            Breakdown of incident costs
        """
        costs = {}
        company_size = self._determine_company_size(company_profile)
        
        # Direct costs
        downtime_cost = (
            self.industry_benchmarks["downtime_cost_per_hour"][company_size] *
            Decimal(str(duration_hours)) *
            Decimal(str(affected_systems))
        )
        costs["downtime"] = downtime_cost
        
        # Investigation and remediation costs
        remediation_hours = duration_hours * 3  # Typical multiplier
        costs["remediation"] = (
            Decimal(str(remediation_hours)) *
            self.industry_benchmarks["productivity_hour_value"] *
            Decimal("5")  # Typical team size
        )
        
        # Data breach specific costs
        if "breach" in incident_type.lower():
            affected_records = affected_systems * 10000  # Estimate
            costs["breach_notification"] = (
                self.industry_benchmarks["data_breach_cost_per_record"] *
                Decimal(str(affected_records))
            )
        
        # Compliance penalties
        if severity in ["major", "critical"]:
            revenue = Decimal(str(
                company_profile.get("annual_revenue", 100000000)
                if company_profile else 100000000
            ))
            costs["compliance_penalty"] = revenue * Decimal("0.001")  # 0.1% of revenue
        
        # Reputation damage
        reputation_multiplier = {
            "minor": 1,
            "moderate": 2,
            "major": 3.5,
            "critical": 5
        }.get(severity, 2)
        
        costs["reputation"] = downtime_cost * Decimal(str(reputation_multiplier))
        
        # Customer churn
        if severity in ["major", "critical"]:
            estimated_churn = affected_systems * 10  # Customers per system
            costs["customer_churn"] = (
                self.industry_benchmarks["customer_churn_cost"] *
                Decimal(str(estimated_churn))
            )
        
        # Total cost
        costs["total"] = sum(costs.values())
        
        return costs
    
    def measure_productivity_gains(
        self,
        automation_data: Dict[str, Any],
        team_size: int = 10
    ) -> Dict[str, Any]:
        """
        Measure productivity improvements from governance automation
        
        Args:
            automation_data: Data about automated processes
            team_size: Size of the team
            
        Returns:
            Quantified productivity gains
        """
        # Calculate hours saved through automation
        manual_hours = automation_data.get("manual_hours_before", 40) * team_size
        automated_hours = automation_data.get("automated_hours_after", 10) * team_size
        hours_saved = manual_hours - automated_hours
        
        # Apply multiplier for freed-up time value
        productive_hours_gained = hours_saved * self.value_drivers["automation"]["hours_saved_multiplier"]
        
        # Calculate financial value
        hourly_value = self.industry_benchmarks["productivity_hour_value"]
        monthly_value = Decimal(str(productive_hours_gained)) * hourly_value * Decimal("4")  # 4 weeks
        annual_value = monthly_value * Decimal("12")
        
        # Calculate error reduction value
        error_rate_before = automation_data.get("error_rate_before", 0.05)
        error_rate_after = error_rate_before * (1 - self.value_drivers["automation"]["error_reduction"])
        errors_prevented = (error_rate_before - error_rate_after) * 1000  # Per 1000 operations
        
        error_cost = Decimal("500")  # Average cost per error
        error_savings = Decimal(str(errors_prevented)) * error_cost * Decimal("12")  # Annual
        
        # Calculate speed improvements
        process_time_before = automation_data.get("process_time_hours", 24)
        process_time_after = process_time_before / self.value_drivers["automation"]["speed_improvement"]
        time_saved_per_process = process_time_before - process_time_after
        
        # Opportunity cost of faster delivery
        processes_per_month = automation_data.get("processes_per_month", 20)
        opportunity_value = (
            Decimal(str(time_saved_per_process)) *
            Decimal(str(processes_per_month)) *
            Decimal("100")  # Opportunity value per hour
        ) * Decimal("12")  # Annual
        
        return {
            "hours_saved_monthly": hours_saved,
            "productive_hours_gained": productive_hours_gained,
            "monthly_productivity_value": float(monthly_value),
            "annual_productivity_value": float(annual_value),
            "errors_prevented_annually": errors_prevented * 12,
            "error_prevention_value": float(error_savings),
            "process_acceleration": f"{self.value_drivers['automation']['speed_improvement']}x",
            "time_to_value_reduction": f"{time_saved_per_process:.1f} hours",
            "opportunity_value_captured": float(opportunity_value),
            "total_annual_value": float(annual_value + error_savings + opportunity_value),
            "roi_percentage": float(
                ((annual_value + error_savings + opportunity_value) / 
                 (Decimal(str(team_size)) * Decimal("100000"))) * 100  # Assuming 100k per FTE
            )
        }
    
    def project_growth_impact(
        self,
        current_metrics: Dict[str, Any],
        improvements: Dict[str, Any],
        projection_months: int = 12
    ) -> pd.DataFrame:
        """
        Project business growth impact over time
        
        Args:
            current_metrics: Current business metrics
            improvements: Expected improvements from governance
            projection_months: Projection period
            
        Returns:
            DataFrame with projected metrics over time
        """
        projections = []
        
        for month in range(projection_months + 1):
            # Calculate cumulative improvements
            improvement_factor = month / projection_months  # Linear improvement
            
            metrics = {
                "month": month,
                "date": (datetime.now() + timedelta(days=month * 30)).strftime("%Y-%m"),
                "revenue": float(
                    Decimal(str(current_metrics.get("revenue", 1000000))) *
                    (Decimal("1") + Decimal(str(improvements.get("revenue_growth", 0.05) * improvement_factor)))
                ),
                "costs": float(
                    Decimal(str(current_metrics.get("costs", 500000))) *
                    (Decimal("1") - Decimal(str(improvements.get("cost_reduction", 0.10) * improvement_factor)))
                ),
                "efficiency_score": min(100,
                    current_metrics.get("efficiency", 60) +
                    improvements.get("efficiency_gain", 20) * improvement_factor
                ),
                "risk_score": max(0,
                    current_metrics.get("risk", 70) -
                    improvements.get("risk_reduction", 30) * improvement_factor
                ),
                "compliance_score": min(100,
                    current_metrics.get("compliance", 75) +
                    improvements.get("compliance_improvement", 20) * improvement_factor
                ),
                "customer_satisfaction": min(100,
                    current_metrics.get("satisfaction", 80) +
                    improvements.get("satisfaction_increase", 10) * improvement_factor
                )
            }
            
            # Calculate profit margin
            metrics["profit"] = metrics["revenue"] - metrics["costs"]
            metrics["profit_margin"] = (metrics["profit"] / metrics["revenue"] * 100) if metrics["revenue"] > 0 else 0
            
            projections.append(metrics)
        
        return pd.DataFrame(projections)
    
    # Private helper methods
    
    def _determine_company_size(self, profile: Optional[Dict[str, Any]]) -> str:
        """Determine company size category"""
        if not profile:
            return "medium"
        
        revenue = profile.get("annual_revenue", 0)
        employees = profile.get("employees", 0)
        
        if revenue > 1000000000 or employees > 1000:
            return "enterprise"
        elif revenue > 100000000 or employees > 500:
            return "large"
        elif revenue > 10000000 or employees > 100:
            return "medium"
        else:
            return "small"
    
    def _analyze_cost_reduction(
        self,
        optimization_data: Dict[str, Any],
        company_size: str
    ) -> BusinessImpact:
        """Analyze cost reduction impact"""
        current_cost = Decimal(str(optimization_data.get("current_monthly_cost", 50000)))
        optimization_rate = optimization_data.get("optimization_percentage", 15) / 100
        
        target_cost = current_cost * (Decimal("1") - Decimal(str(optimization_rate)))
        savings = current_cost - target_cost
        annual_savings = savings * Decimal("12")
        
        return BusinessImpact(
            metric=BusinessMetric.COST_REDUCTION,
            current_value=current_cost * Decimal("12"),  # Annual
            target_value=target_cost * Decimal("12"),
            improvement=savings * Decimal("12"),
            improvement_percentage=optimization_rate * 100,
            financial_impact=annual_savings,
            confidence_level=0.85,
            time_to_realize=30,
            supporting_evidence=[
                f"Resource rightsizing opportunity: {optimization_data.get('rightsizing_count', 0)} resources",
                f"Reserved instance savings: {optimization_data.get('ri_savings', 0):.0f}%",
                f"Spot instance opportunities: {optimization_data.get('spot_opportunities', 0)} workloads"
            ]
        )
    
    def _analyze_risk_mitigation(
        self,
        security_data: Dict[str, Any],
        company_profile: Optional[Dict[str, Any]]
    ) -> BusinessImpact:
        """Analyze risk mitigation value"""
        current_risk_score = security_data.get("current_risk_score", 70)
        target_risk_score = security_data.get("target_risk_score", 30)
        
        # Convert risk reduction to financial value
        company_size = self._determine_company_size(company_profile)
        annual_revenue = Decimal(str(
            company_profile.get("annual_revenue", 100000000) if company_profile else 100000000
        ))
        
        # Risk exposure as percentage of revenue
        current_exposure = annual_revenue * Decimal(str(current_risk_score / 1000))
        target_exposure = annual_revenue * Decimal(str(target_risk_score / 1000))
        risk_reduction_value = current_exposure - target_exposure
        
        return BusinessImpact(
            metric=BusinessMetric.RISK_MITIGATION,
            current_value=current_exposure,
            target_value=target_exposure,
            improvement=risk_reduction_value,
            improvement_percentage=((current_risk_score - target_risk_score) / current_risk_score * 100),
            financial_impact=risk_reduction_value,
            confidence_level=0.75,
            time_to_realize=90,
            supporting_evidence=[
                f"Security incidents reduced by {security_data.get('incident_reduction', 60):.0f}%",
                f"Mean time to detect: {security_data.get('mttd_hours', 24):.0f} → {security_data.get('target_mttd', 1):.0f} hours",
                f"Compliance violations prevented: {security_data.get('violations_prevented', 0)}"
            ]
        )
    
    def _analyze_operational_efficiency(
        self,
        automation_data: Dict[str, Any],
        company_size: str
    ) -> BusinessImpact:
        """Analyze operational efficiency gains"""
        hours_saved = automation_data.get("hours_saved_monthly", 100)
        hourly_value = self.industry_benchmarks["productivity_hour_value"]
        
        monthly_value = Decimal(str(hours_saved)) * hourly_value
        annual_value = monthly_value * Decimal("12")
        
        current_efficiency = automation_data.get("current_efficiency_score", 60)
        target_efficiency = automation_data.get("target_efficiency_score", 85)
        
        return BusinessImpact(
            metric=BusinessMetric.OPERATIONAL_EFFICIENCY,
            current_value=Decimal(str(current_efficiency)),
            target_value=Decimal(str(target_efficiency)),
            improvement=Decimal(str(target_efficiency - current_efficiency)),
            improvement_percentage=((target_efficiency - current_efficiency) / current_efficiency * 100),
            financial_impact=annual_value,
            confidence_level=0.90,
            time_to_realize=60,
            supporting_evidence=[
                f"Automation rate increased to {automation_data.get('automation_rate', 75):.0f}%",
                f"Manual tasks reduced by {automation_data.get('manual_reduction', 60):.0f}%",
                f"Process cycle time improved by {automation_data.get('cycle_improvement', 40):.0f}%"
            ]
        )
    
    def _analyze_compliance_impact(
        self,
        compliance_data: Dict[str, Any],
        company_profile: Optional[Dict[str, Any]]
    ) -> BusinessImpact:
        """Analyze compliance improvement impact"""
        current_score = compliance_data.get("current_score", 70)
        target_score = compliance_data.get("target_score", 95)
        
        # Calculate penalty avoidance value
        annual_revenue = Decimal(str(
            company_profile.get("annual_revenue", 100000000) if company_profile else 100000000
        ))
        
        # Risk of penalties based on compliance score
        current_penalty_risk = (100 - current_score) / 100
        target_penalty_risk = (100 - target_score) / 100
        
        # Average penalty as percentage of revenue
        avg_penalty_rate = 0.02  # 2% average
        
        current_penalty_exposure = annual_revenue * Decimal(str(current_penalty_risk * avg_penalty_rate))
        target_penalty_exposure = annual_revenue * Decimal(str(target_penalty_risk * avg_penalty_rate))
        penalty_reduction = current_penalty_exposure - target_penalty_exposure
        
        return BusinessImpact(
            metric=BusinessMetric.COMPLIANCE_POSTURE,
            current_value=Decimal(str(current_score)),
            target_value=Decimal(str(target_score)),
            improvement=Decimal(str(target_score - current_score)),
            improvement_percentage=((target_score - current_score) / current_score * 100),
            financial_impact=penalty_reduction,
            confidence_level=0.80,
            time_to_realize=120,
            supporting_evidence=[
                f"Audit findings reduced by {compliance_data.get('audit_finding_reduction', 70):.0f}%",
                f"Compliance frameworks covered: {compliance_data.get('frameworks_covered', 5)}",
                f"Automated compliance checks: {compliance_data.get('automated_checks', 500)}"
            ]
        )
    
    def _analyze_time_to_market(
        self,
        deployment_data: Dict[str, Any],
        company_profile: Optional[Dict[str, Any]]
    ) -> BusinessImpact:
        """Analyze time-to-market improvements"""
        current_deployment_time = deployment_data.get("current_deployment_days", 30)
        target_deployment_time = deployment_data.get("target_deployment_days", 7)
        
        time_saved = current_deployment_time - target_deployment_time
        deployments_per_year = deployment_data.get("deployments_per_year", 12)
        
        # Calculate opportunity value
        revenue_per_day = Decimal(str(
            company_profile.get("daily_revenue", 274000) if company_profile else 274000
        ))
        
        opportunity_value = (
            Decimal(str(time_saved)) *
            Decimal(str(deployments_per_year)) *
            revenue_per_day *
            Decimal("0.01")  # Conservative 1% of daily revenue
        )
        
        return BusinessImpact(
            metric=BusinessMetric.TIME_TO_MARKET,
            current_value=Decimal(str(current_deployment_time)),
            target_value=Decimal(str(target_deployment_time)),
            improvement=Decimal(str(time_saved)),
            improvement_percentage=((time_saved) / current_deployment_time * 100),
            financial_impact=opportunity_value,
            confidence_level=0.70,
            time_to_realize=45,
            supporting_evidence=[
                f"Deployment frequency increased by {deployment_data.get('frequency_increase', 300):.0f}%",
                f"Lead time reduced from {current_deployment_time} to {target_deployment_time} days",
                f"Failed deployments reduced by {deployment_data.get('failure_reduction', 80):.0f}%"
            ]
        )
    
    def _map_risk_category(self, risk_type: str) -> RiskCategory:
        """Map risk type string to RiskCategory enum"""
        mapping = {
            "security": RiskCategory.DATA_BREACH,
            "availability": RiskCategory.SERVICE_OUTAGE,
            "compliance": RiskCategory.COMPLIANCE_VIOLATION,
            "reputation": RiskCategory.REPUTATION_DAMAGE,
            "intellectual_property": RiskCategory.INTELLECTUAL_PROPERTY,
            "operational": RiskCategory.OPERATIONAL_FAILURE,
            "supply_chain": RiskCategory.SUPPLY_CHAIN,
            "regulatory": RiskCategory.REGULATORY_PENALTY
        }
        return mapping.get(risk_type.lower(), RiskCategory.OPERATIONAL_FAILURE)
    
    def _calculate_risk_probability(self, risk_score: float) -> float:
        """Convert risk score to probability"""
        # Using sigmoid function for smooth conversion
        return 1 / (1 + np.exp(-0.1 * (risk_score - 50)))
    
    def _calculate_risk_impact(
        self,
        category: RiskCategory,
        risk_data: Dict[str, Any],
        company_size: str,
        annual_revenue: Decimal
    ) -> Decimal:
        """Calculate financial impact of risk"""
        base_impact = Decimal("0")
        
        if category == RiskCategory.DATA_BREACH:
            records = risk_data.get("potential_records", 10000)
            base_impact = self.industry_benchmarks["data_breach_cost_per_record"] * Decimal(str(records))
        
        elif category == RiskCategory.SERVICE_OUTAGE:
            hours = risk_data.get("potential_downtime_hours", 4)
            base_impact = (
                self.industry_benchmarks["downtime_cost_per_hour"][company_size] *
                Decimal(str(hours))
            )
        
        elif category == RiskCategory.COMPLIANCE_VIOLATION:
            regulation = risk_data.get("regulation", "gdpr")
            penalty_rate = self.industry_benchmarks["compliance_violation_percentage_revenue"].get(
                regulation.lower(), 0.01
            )
            base_impact = annual_revenue * Decimal(str(penalty_rate))
        
        elif category == RiskCategory.REPUTATION_DAMAGE:
            # Market cap impact
            market_cap = annual_revenue * Decimal("5")  # Rough P/S ratio
            severity = risk_data.get("severity", "moderate")
            impact_rate = self.industry_benchmarks["market_cap_impact"].get(
                f"{severity}_incident", 0.01
            )
            base_impact = market_cap * Decimal(str(impact_rate))
        
        else:
            # Default calculation
            base_impact = annual_revenue * Decimal("0.001")  # 0.1% of revenue
        
        return base_impact
    
    def _estimate_mitigation_cost(
        self,
        category: RiskCategory,
        impact: Decimal
    ) -> Decimal:
        """Estimate cost to mitigate risk"""
        # Generally, mitigation costs 5-20% of potential impact
        mitigation_rates = {
            RiskCategory.DATA_BREACH: 0.15,
            RiskCategory.SERVICE_OUTAGE: 0.10,
            RiskCategory.COMPLIANCE_VIOLATION: 0.20,
            RiskCategory.REPUTATION_DAMAGE: 0.25,
            RiskCategory.INTELLECTUAL_PROPERTY: 0.30,
            RiskCategory.OPERATIONAL_FAILURE: 0.10,
            RiskCategory.SUPPLY_CHAIN: 0.15,
            RiskCategory.REGULATORY_PENALTY: 0.20
        }
        
        rate = mitigation_rates.get(category, 0.15)
        return impact * Decimal(str(rate))


# Export main analyzer instance
def create_analyzer() -> ImpactAnalyzer:
    """Create an impact analyzer instance"""
    return ImpactAnalyzer()