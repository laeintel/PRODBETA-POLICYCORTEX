"""
PolicyCortex What-If Financial Simulator
Monte Carlo simulation engine for financial projections with scenario planning
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import logging
from enum import Enum
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of what-if scenarios"""
    POLICY_CHANGE = "policy_change"
    RESOURCE_SCALING = "resource_scaling"
    COMPLIANCE_ENFORCEMENT = "compliance_enforcement"
    AUTOMATION_DEPLOYMENT = "automation_deployment"
    DISASTER_RECOVERY = "disaster_recovery"
    SECURITY_HARDENING = "security_hardening"
    COST_OPTIMIZATION = "cost_optimization"
    MIGRATION = "migration"

@dataclass
class ScenarioParameter:
    """Parameter for scenario simulation"""
    name: str
    current_value: float
    proposed_value: float
    unit: str
    impact_factor: float  # How much this parameter affects costs
    confidence_interval: Tuple[float, float]  # Min, max bounds
    distribution: str = "normal"  # normal, uniform, exponential
    
@dataclass
class SimulationResult:
    """Result of what-if simulation"""
    scenario_id: str
    scenario_type: ScenarioType
    time_horizon_days: int
    base_cost: Decimal
    projected_costs: Dict[int, Decimal]  # Day -> Cost
    confidence_intervals: Dict[int, Tuple[Decimal, Decimal]]  # Day -> (Lower, Upper)
    expected_savings: Decimal
    roi_percentage: float
    payback_period_days: int
    risk_score: float
    confidence_level: float
    key_metrics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "time_horizon_days": self.time_horizon_days,
            "base_cost": float(self.base_cost),
            "projected_costs": {k: float(v) for k, v in self.projected_costs.items()},
            "confidence_intervals": {
                k: (float(v[0]), float(v[1])) 
                for k, v in self.confidence_intervals.items()
            },
            "expected_savings": float(self.expected_savings),
            "roi_percentage": self.roi_percentage,
            "payback_period_days": self.payback_period_days,
            "risk_score": self.risk_score,
            "confidence_level": self.confidence_level,
            "key_metrics": self.key_metrics,
            "recommendations": self.recommendations
        }

class WhatIfSimulator:
    """
    Monte Carlo simulation engine for financial projections
    Provides scenario planning with confidence intervals
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize simulator with optional random seed for reproducibility
        
        Args:
            seed: Random seed for Monte Carlo simulations
        """
        if seed:
            np.random.seed(seed)
        
        self.simulation_runs = 10000  # Number of Monte Carlo iterations
        self.confidence_level = 0.95  # 95% confidence interval
        
        # Cost impact models based on historical data
        self.impact_models = {
            ScenarioType.POLICY_CHANGE: {
                "base_impact": 0.15,  # 15% average impact
                "volatility": 0.05,    # 5% standard deviation
                "lag_days": 7,         # Days before impact is visible
                "ramp_days": 30        # Days to full impact
            },
            ScenarioType.RESOURCE_SCALING: {
                "base_impact": 0.25,
                "volatility": 0.08,
                "lag_days": 1,
                "ramp_days": 14
            },
            ScenarioType.COMPLIANCE_ENFORCEMENT: {
                "base_impact": 0.20,
                "volatility": 0.10,
                "lag_days": 14,
                "ramp_days": 60
            },
            ScenarioType.AUTOMATION_DEPLOYMENT: {
                "base_impact": 0.30,
                "volatility": 0.12,
                "lag_days": 30,
                "ramp_days": 90
            },
            ScenarioType.DISASTER_RECOVERY: {
                "base_impact": -0.10,  # Costs increase initially
                "volatility": 0.15,
                "lag_days": 0,
                "ramp_days": 180
            },
            ScenarioType.SECURITY_HARDENING: {
                "base_impact": -0.05,  # Small cost increase for better security
                "volatility": 0.03,
                "lag_days": 7,
                "ramp_days": 45
            },
            ScenarioType.COST_OPTIMIZATION: {
                "base_impact": 0.35,
                "volatility": 0.10,
                "lag_days": 14,
                "ramp_days": 45
            },
            ScenarioType.MIGRATION: {
                "base_impact": 0.20,
                "volatility": 0.20,
                "lag_days": 60,
                "ramp_days": 120
            }
        }
        
        # Risk factors for different scenarios
        self.risk_factors = {
            "implementation_complexity": 0.2,
            "organizational_readiness": 0.15,
            "technical_debt": 0.25,
            "vendor_dependency": 0.20,
            "regulatory_compliance": 0.20
        }
        
        logger.info("What-If Simulator initialized with %d simulation runs", self.simulation_runs)
    
    async def simulate_scenario(
        self,
        scenario_type: ScenarioType,
        parameters: List[ScenarioParameter],
        current_monthly_cost: Decimal,
        time_horizon_days: int = 90,
        additional_factors: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a what-if scenario
        
        Args:
            scenario_type: Type of scenario to simulate
            parameters: List of parameters being changed
            current_monthly_cost: Current baseline monthly cost
            time_horizon_days: Simulation time horizon (30/60/90 days)
            additional_factors: Additional factors affecting simulation
            
        Returns:
            SimulationResult with projections and confidence intervals
        """
        scenario_id = self._generate_scenario_id(scenario_type, parameters)
        
        # Get impact model for scenario type
        impact_model = self.impact_models.get(
            scenario_type,
            self.impact_models[ScenarioType.POLICY_CHANGE]
        )
        
        # Run Monte Carlo simulation
        simulations = []
        for _ in range(self.simulation_runs):
            sim_result = self._run_single_simulation(
                parameters,
                current_monthly_cost,
                time_horizon_days,
                impact_model,
                additional_factors or {}
            )
            simulations.append(sim_result)
        
        # Aggregate results
        simulations_array = np.array(simulations)
        
        # Calculate projections for each day
        projected_costs = {}
        confidence_intervals = {}
        
        for day in range(1, time_horizon_days + 1):
            day_costs = simulations_array[:, day - 1]
            projected_costs[day] = Decimal(str(np.mean(day_costs)))
            
            # Calculate confidence interval
            lower = np.percentile(day_costs, (1 - self.confidence_level) * 100 / 2)
            upper = np.percentile(day_costs, (1 + self.confidence_level) * 100 / 2)
            confidence_intervals[day] = (Decimal(str(lower)), Decimal(str(upper)))
        
        # Calculate overall metrics
        total_base = current_monthly_cost * Decimal(str(time_horizon_days / 30))
        total_projected = sum(projected_costs.values()) / len(projected_costs)
        expected_savings = total_base - total_projected
        
        # Calculate ROI
        investment = self._estimate_implementation_cost(scenario_type, parameters)
        roi_percentage = float(
            (expected_savings - investment) / investment * 100
            if investment > 0 else 0
        )
        
        # Calculate payback period
        daily_savings = expected_savings / Decimal(str(time_horizon_days))
        payback_period = int(
            (investment / daily_savings) 
            if daily_savings > 0 else 999
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            scenario_type, 
            parameters, 
            additional_factors or {}
        )
        
        # Calculate confidence level based on parameter certainty
        confidence = self._calculate_confidence_level(parameters, simulations_array)
        
        # Generate key metrics
        key_metrics = self._generate_key_metrics(
            simulations_array,
            projected_costs,
            current_monthly_cost,
            time_horizon_days
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            scenario_type,
            expected_savings,
            risk_score,
            payback_period
        )
        
        return SimulationResult(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            time_horizon_days=time_horizon_days,
            base_cost=total_base,
            projected_costs=projected_costs,
            confidence_intervals=confidence_intervals,
            expected_savings=expected_savings,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period,
            risk_score=risk_score,
            confidence_level=confidence,
            key_metrics=key_metrics,
            recommendations=recommendations
        )
    
    async def simulate_combined_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        current_monthly_cost: Decimal,
        time_horizon_days: int = 90
    ) -> Dict[str, Any]:
        """
        Simulate multiple scenarios combined
        
        Args:
            scenarios: List of scenario configurations
            current_monthly_cost: Current baseline monthly cost
            time_horizon_days: Simulation time horizon
            
        Returns:
            Combined simulation results with interaction effects
        """
        combined_results = {
            "individual_scenarios": [],
            "combined_impact": None,
            "interaction_effects": {},
            "optimal_sequence": [],
            "total_savings": Decimal("0"),
            "combined_roi": 0,
            "implementation_roadmap": []
        }
        
        # Simulate each scenario individually
        for scenario_config in scenarios:
            result = await self.simulate_scenario(
                scenario_type=ScenarioType[scenario_config["type"]],
                parameters=scenario_config["parameters"],
                current_monthly_cost=current_monthly_cost,
                time_horizon_days=time_horizon_days,
                additional_factors=scenario_config.get("factors")
            )
            combined_results["individual_scenarios"].append(result.to_dict())
        
        # Calculate interaction effects
        if len(scenarios) > 1:
            interaction_matrix = self._calculate_interaction_effects(scenarios)
            combined_results["interaction_effects"] = interaction_matrix
            
            # Determine optimal implementation sequence
            optimal_sequence = self._optimize_implementation_sequence(
                scenarios,
                interaction_matrix
            )
            combined_results["optimal_sequence"] = optimal_sequence
            
            # Calculate combined impact with interactions
            combined_savings = Decimal("0")
            for i, scenario in enumerate(scenarios):
                base_savings = combined_results["individual_scenarios"][i]["expected_savings"]
                
                # Apply interaction effects
                for j, other_scenario in enumerate(scenarios):
                    if i != j:
                        interaction_factor = interaction_matrix.get(f"{i}_{j}", 1.0)
                        base_savings *= Decimal(str(interaction_factor))
                
                combined_savings += base_savings
            
            combined_results["total_savings"] = float(combined_savings)
            
            # Calculate combined ROI
            total_investment = sum(
                self._estimate_implementation_cost(
                    ScenarioType[s["type"]], 
                    s["parameters"]
                )
                for s in scenarios
            )
            
            combined_results["combined_roi"] = float(
                (combined_savings - total_investment) / total_investment * 100
                if total_investment > 0 else 0
            )
            
            # Generate implementation roadmap
            combined_results["implementation_roadmap"] = self._generate_roadmap(
                optimal_sequence,
                scenarios,
                combined_results["individual_scenarios"]
            )
        
        return combined_results
    
    async def sensitivity_analysis(
        self,
        scenario_type: ScenarioType,
        base_parameters: List[ScenarioParameter],
        current_monthly_cost: Decimal,
        sensitivity_range: float = 0.2  # ±20% variation
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on scenario parameters
        
        Args:
            scenario_type: Type of scenario
            base_parameters: Base parameters for scenario
            current_monthly_cost: Current baseline monthly cost
            sensitivity_range: Range of variation for sensitivity analysis
            
        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {
            "parameter_impacts": {},
            "tornado_chart_data": [],
            "critical_parameters": [],
            "robust_parameters": []
        }
        
        base_result = await self.simulate_scenario(
            scenario_type=scenario_type,
            parameters=base_parameters,
            current_monthly_cost=current_monthly_cost,
            time_horizon_days=90
        )
        
        base_savings = base_result.expected_savings
        
        # Test each parameter's sensitivity
        for param in base_parameters:
            param_results = {
                "parameter": param.name,
                "base_value": param.proposed_value,
                "impacts": []
            }
            
            # Test variations
            variations = np.linspace(
                param.proposed_value * (1 - sensitivity_range),
                param.proposed_value * (1 + sensitivity_range),
                11  # 11 points including center
            )
            
            for variation in variations:
                # Create modified parameter list
                test_params = base_parameters.copy()
                for i, p in enumerate(test_params):
                    if p.name == param.name:
                        test_params[i] = ScenarioParameter(
                            name=p.name,
                            current_value=p.current_value,
                            proposed_value=variation,
                            unit=p.unit,
                            impact_factor=p.impact_factor,
                            confidence_interval=p.confidence_interval,
                            distribution=p.distribution
                        )
                
                # Run simulation with varied parameter
                result = await self.simulate_scenario(
                    scenario_type=scenario_type,
                    parameters=test_params,
                    current_monthly_cost=current_monthly_cost,
                    time_horizon_days=90
                )
                
                impact = float(
                    (result.expected_savings - base_savings) / base_savings * 100
                    if base_savings != 0 else 0
                )
                
                param_results["impacts"].append({
                    "value": variation,
                    "savings": float(result.expected_savings),
                    "impact_percentage": impact
                })
            
            sensitivity_results["parameter_impacts"][param.name] = param_results
            
            # Calculate sensitivity score (slope of impact)
            values = [i["value"] for i in param_results["impacts"]]
            impacts = [i["impact_percentage"] for i in param_results["impacts"]]
            
            if len(values) > 1:
                slope, _ = np.polyfit(values, impacts, 1)
                sensitivity_score = abs(slope)
                
                # Add to tornado chart data
                sensitivity_results["tornado_chart_data"].append({
                    "parameter": param.name,
                    "sensitivity": sensitivity_score,
                    "min_impact": min(impacts),
                    "max_impact": max(impacts)
                })
                
                # Classify parameter
                if sensitivity_score > 10:
                    sensitivity_results["critical_parameters"].append(param.name)
                elif sensitivity_score < 1:
                    sensitivity_results["robust_parameters"].append(param.name)
        
        # Sort tornado chart data by sensitivity
        sensitivity_results["tornado_chart_data"].sort(
            key=lambda x: x["sensitivity"],
            reverse=True
        )
        
        return sensitivity_results
    
    def generate_scenario_report(
        self,
        result: SimulationResult,
        format: str = "json"
    ) -> Any:
        """
        Generate detailed scenario report
        
        Args:
            result: Simulation result to report on
            format: Output format (json, html, pdf)
            
        Returns:
            Formatted report
        """
        if format == "json":
            return result.to_dict()
        
        elif format == "html":
            html_template = """
            <html>
            <head>
                <title>PolicyCortex What-If Scenario Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #0078D4; color: white; padding: 20px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                              border: 1px solid #ddd; border-radius: 5px; }}
                    .savings {{ color: #107C10; font-size: 24px; font-weight: bold; }}
                    .recommendation {{ background: #F3F2F1; padding: 10px; margin: 5px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background: #F3F2F1; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>What-If Scenario Analysis Report</h1>
                    <p>Scenario: {scenario_type} | ID: {scenario_id}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Expected Savings</h3>
                        <div class="savings">${expected_savings:,.2f}</div>
                    </div>
                    <div class="metric">
                        <h3>ROI</h3>
                        <div>{roi_percentage:.1f}%</div>
                    </div>
                    <div class="metric">
                        <h3>Payback Period</h3>
                        <div>{payback_period_days} days</div>
                    </div>
                    <div class="metric">
                        <h3>Confidence Level</h3>
                        <div>{confidence_level:.1%}</div>
                    </div>
                    <div class="metric">
                        <h3>Risk Score</h3>
                        <div>{risk_score:.2f}/10</div>
                    </div>
                </div>
                
                <h2>Key Metrics</h2>
                <table>
                    {metrics_table}
                </table>
                
                <h2>Recommendations</h2>
                {recommendations_html}
                
                <h2>Projected Cost Trajectory</h2>
                <p>Time Horizon: {time_horizon_days} days</p>
                <p>Base Cost: ${base_cost:,.2f}</p>
                
                <div class="footer">
                    <p>Generated: {timestamp}</p>
                    <p>PolicyCortex PAYBACK Engine - Demonstrating Value Through Data</p>
                </div>
            </body>
            </html>
            """
            
            # Build metrics table
            metrics_rows = []
            for key, value in result.key_metrics.items():
                metrics_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            metrics_table = "".join(metrics_rows)
            
            # Build recommendations list
            recommendations_html = "".join([
                f'<div class="recommendation">{r}</div>' 
                for r in result.recommendations
            ])
            
            return html_template.format(
                scenario_type=result.scenario_type.value,
                scenario_id=result.scenario_id,
                expected_savings=result.expected_savings,
                roi_percentage=result.roi_percentage,
                payback_period_days=result.payback_period_days,
                confidence_level=result.confidence_level,
                risk_score=result.risk_score,
                metrics_table=metrics_table,
                recommendations_html=recommendations_html,
                time_horizon_days=result.time_horizon_days,
                base_cost=result.base_cost,
                timestamp=datetime.now().isoformat()
            )
        
        return result.to_dict()
    
    # Private helper methods
    
    def _generate_scenario_id(
        self,
        scenario_type: ScenarioType,
        parameters: List[ScenarioParameter]
    ) -> str:
        """Generate unique scenario ID"""
        param_str = "_".join([f"{p.name}_{p.proposed_value}" for p in parameters])
        hash_input = f"{scenario_type.value}_{param_str}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _run_single_simulation(
        self,
        parameters: List[ScenarioParameter],
        current_monthly_cost: Decimal,
        time_horizon_days: int,
        impact_model: Dict[str, Any],
        additional_factors: Dict[str, Any]
    ) -> np.ndarray:
        """Run a single Monte Carlo simulation iteration"""
        daily_costs = []
        daily_base = float(current_monthly_cost) / 30
        
        for day in range(time_horizon_days):
            # Calculate impact based on ramp-up period
            if day < impact_model["lag_days"]:
                impact_factor = 0
            elif day < impact_model["lag_days"] + impact_model["ramp_days"]:
                progress = (day - impact_model["lag_days"]) / impact_model["ramp_days"]
                impact_factor = impact_model["base_impact"] * progress
            else:
                impact_factor = impact_model["base_impact"]
            
            # Add random variation
            variation = np.random.normal(0, impact_model["volatility"])
            impact_factor += variation
            
            # Apply parameter-specific impacts
            for param in parameters:
                param_impact = (
                    (param.proposed_value - param.current_value) / 
                    param.current_value * param.impact_factor
                )
                
                # Add parameter-specific noise based on distribution
                if param.distribution == "normal":
                    noise = np.random.normal(0, 0.05)
                elif param.distribution == "uniform":
                    noise = np.random.uniform(-0.05, 0.05)
                else:
                    noise = np.random.exponential(0.02) - 0.02
                
                impact_factor += param_impact * (1 + noise)
            
            # Calculate daily cost with impact
            daily_cost = daily_base * (1 - impact_factor)
            
            # Apply additional factors (seasonality, market conditions, etc.)
            if "seasonality" in additional_factors:
                seasonal_factor = additional_factors["seasonality"]
                daily_cost *= (1 + seasonal_factor * np.sin(2 * np.pi * day / 365))
            
            if "market_volatility" in additional_factors:
                market_factor = additional_factors["market_volatility"]
                daily_cost *= (1 + np.random.normal(0, market_factor))
            
            daily_costs.append(max(0, daily_cost))  # Ensure non-negative
        
        return np.array(daily_costs)
    
    def _estimate_implementation_cost(
        self,
        scenario_type: ScenarioType,
        parameters: List[ScenarioParameter]
    ) -> Decimal:
        """Estimate cost of implementing scenario"""
        base_costs = {
            ScenarioType.POLICY_CHANGE: Decimal("5000"),
            ScenarioType.RESOURCE_SCALING: Decimal("10000"),
            ScenarioType.COMPLIANCE_ENFORCEMENT: Decimal("25000"),
            ScenarioType.AUTOMATION_DEPLOYMENT: Decimal("50000"),
            ScenarioType.DISASTER_RECOVERY: Decimal("100000"),
            ScenarioType.SECURITY_HARDENING: Decimal("30000"),
            ScenarioType.COST_OPTIMIZATION: Decimal("15000"),
            ScenarioType.MIGRATION: Decimal("75000")
        }
        
        base_cost = base_costs.get(scenario_type, Decimal("10000"))
        
        # Adjust based on parameter complexity
        complexity_factor = Decimal("1") + (Decimal(str(len(parameters))) * Decimal("0.1"))
        
        return base_cost * complexity_factor
    
    def _calculate_risk_score(
        self,
        scenario_type: ScenarioType,
        parameters: List[ScenarioParameter],
        additional_factors: Dict[str, Any]
    ) -> float:
        """Calculate risk score for scenario (0-10)"""
        base_risk = {
            ScenarioType.POLICY_CHANGE: 3.0,
            ScenarioType.RESOURCE_SCALING: 5.0,
            ScenarioType.COMPLIANCE_ENFORCEMENT: 4.0,
            ScenarioType.AUTOMATION_DEPLOYMENT: 6.0,
            ScenarioType.DISASTER_RECOVERY: 2.0,
            ScenarioType.SECURITY_HARDENING: 3.0,
            ScenarioType.COST_OPTIMIZATION: 4.0,
            ScenarioType.MIGRATION: 8.0
        }
        
        risk = base_risk.get(scenario_type, 5.0)
        
        # Adjust for parameter uncertainty
        for param in parameters:
            interval_width = param.confidence_interval[1] - param.confidence_interval[0]
            uncertainty = interval_width / param.proposed_value if param.proposed_value != 0 else 0
            risk += uncertainty * 2
        
        # Apply risk factors
        for factor, weight in self.risk_factors.items():
            if factor in additional_factors:
                risk += additional_factors[factor] * weight * 10
        
        return min(10, max(0, risk))
    
    def _calculate_confidence_level(
        self,
        parameters: List[ScenarioParameter],
        simulations: np.ndarray
    ) -> float:
        """Calculate confidence level based on simulation convergence"""
        # Check convergence of simulations
        means = []
        for i in range(100, len(simulations), 100):
            means.append(np.mean(simulations[:i]))
        
        if len(means) > 1:
            cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 1
            convergence_score = 1 - min(cv, 1)
        else:
            convergence_score = 0.5
        
        # Check parameter confidence
        param_confidence = np.mean([
            1 - (p.confidence_interval[1] - p.confidence_interval[0]) / 
            (p.proposed_value if p.proposed_value != 0 else 1)
            for p in parameters
        ])
        
        # Combined confidence
        confidence = 0.6 * convergence_score + 0.4 * param_confidence
        
        return min(0.99, max(0.1, confidence))
    
    def _generate_key_metrics(
        self,
        simulations: np.ndarray,
        projected_costs: Dict[int, Decimal],
        current_monthly_cost: Decimal,
        time_horizon_days: int
    ) -> Dict[str, Any]:
        """Generate key metrics from simulation results"""
        final_costs = simulations[:, -1]
        daily_base = float(current_monthly_cost) / 30
        
        return {
            "mean_daily_savings": f"${daily_base - np.mean(final_costs):.2f}",
            "median_daily_savings": f"${daily_base - np.median(final_costs):.2f}",
            "best_case_savings": f"${daily_base - np.percentile(final_costs, 5):.2f}",
            "worst_case_savings": f"${daily_base - np.percentile(final_costs, 95):.2f}",
            "volatility": f"{np.std(final_costs) / np.mean(final_costs) * 100:.1f}%",
            "savings_probability": f"{np.mean(final_costs < daily_base) * 100:.1f}%",
            "break_even_probability": f"{np.mean(simulations[:, :30].sum(axis=1) < daily_base * 30) * 100:.1f}%",
            "cost_reduction_rate": f"{(1 - np.mean(final_costs) / daily_base) * 100:.1f}%"
        }
    
    def _generate_recommendations(
        self,
        scenario_type: ScenarioType,
        expected_savings: Decimal,
        risk_score: float,
        payback_period: int
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Payback period recommendations
        if payback_period < 30:
            recommendations.append("PRIORITY: Quick win with <30 day payback - implement immediately")
        elif payback_period < 90:
            recommendations.append("Strong ROI with <90 day payback - schedule for next quarter")
        elif payback_period < 180:
            recommendations.append("Moderate payback period - consider as part of annual planning")
        else:
            recommendations.append("Long payback period - evaluate strategic alignment carefully")
        
        # Risk-based recommendations
        if risk_score < 3:
            recommendations.append("Low risk scenario - proceed with standard change process")
        elif risk_score < 6:
            recommendations.append("Moderate risk - implement pilot program first")
        else:
            recommendations.append("High risk - extensive testing and phased rollout recommended")
        
        # Scenario-specific recommendations
        scenario_recommendations = {
            ScenarioType.POLICY_CHANGE: [
                "Document policy exceptions and edge cases",
                "Communicate changes to all stakeholders",
                "Monitor compliance metrics closely post-implementation"
            ],
            ScenarioType.RESOURCE_SCALING: [
                "Implement auto-scaling policies where possible",
                "Review resource utilization weekly for first month",
                "Set up cost alerts at 80% of projected savings"
            ],
            ScenarioType.AUTOMATION_DEPLOYMENT: [
                "Start with non-critical processes",
                "Maintain manual fallback procedures",
                "Track automation success rate and adjust accordingly"
            ],
            ScenarioType.COST_OPTIMIZATION: [
                "Prioritize highest-impact optimizations",
                "Review and adjust Reserved Instance coverage",
                "Implement tagging strategy for cost allocation"
            ]
        }
        
        if scenario_type in scenario_recommendations:
            recommendations.extend(scenario_recommendations[scenario_type])
        
        # Savings-based recommendations
        if expected_savings > Decimal("100000"):
            recommendations.append("Significant savings opportunity - consider dedicated project team")
        elif expected_savings > Decimal("50000"):
            recommendations.append("Material savings - assign dedicated owner for tracking")
        
        return recommendations
    
    def _calculate_interaction_effects(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate interaction effects between scenarios"""
        interactions = {}
        
        for i, scenario1 in enumerate(scenarios):
            for j, scenario2 in enumerate(scenarios):
                if i != j:
                    # Calculate synergy or conflict
                    type1 = ScenarioType[scenario1["type"]]
                    type2 = ScenarioType[scenario2["type"]]
                    
                    # Synergistic combinations
                    if (type1 == ScenarioType.AUTOMATION_DEPLOYMENT and 
                        type2 == ScenarioType.COST_OPTIMIZATION):
                        interactions[f"{i}_{j}"] = 1.2  # 20% synergy
                    elif (type1 == ScenarioType.POLICY_CHANGE and 
                          type2 == ScenarioType.COMPLIANCE_ENFORCEMENT):
                        interactions[f"{i}_{j}"] = 1.15  # 15% synergy
                    # Conflicting combinations
                    elif (type1 == ScenarioType.DISASTER_RECOVERY and 
                          type2 == ScenarioType.COST_OPTIMIZATION):
                        interactions[f"{i}_{j}"] = 0.9  # 10% conflict
                    else:
                        interactions[f"{i}_{j}"] = 1.0  # No interaction
        
        return interactions
    
    def _optimize_implementation_sequence(
        self,
        scenarios: List[Dict[str, Any]],
        interaction_matrix: Dict[str, float]
    ) -> List[int]:
        """Determine optimal sequence for implementing scenarios"""
        n = len(scenarios)
        if n == 1:
            return [0]
        
        # Simple greedy algorithm - start with highest impact
        remaining = list(range(n))
        sequence = []
        
        while remaining:
            best_next = None
            best_score = -float('inf')
            
            for idx in remaining:
                score = 0
                # Calculate score based on previous implementations
                for prev_idx in sequence:
                    interaction = interaction_matrix.get(f"{prev_idx}_{idx}", 1.0)
                    score += interaction
                
                if score > best_score:
                    best_score = score
                    best_next = idx
            
            if best_next is not None:
                sequence.append(best_next)
                remaining.remove(best_next)
        
        return sequence
    
    def _generate_roadmap(
        self,
        sequence: List[int],
        scenarios: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate implementation roadmap"""
        roadmap = []
        cumulative_days = 0
        
        for idx in sequence:
            scenario = scenarios[idx]
            result = results[idx]
            
            roadmap.append({
                "phase": len(roadmap) + 1,
                "scenario": scenario["type"],
                "start_day": cumulative_days,
                "duration_days": 30,  # Default implementation duration
                "expected_savings": result["expected_savings"],
                "key_milestones": [
                    f"Day {cumulative_days + 7}: Initial deployment",
                    f"Day {cumulative_days + 14}: Monitoring phase",
                    f"Day {cumulative_days + 21}: Optimization",
                    f"Day {cumulative_days + 30}: Full rollout"
                ]
            })
            
            cumulative_days += 30
        
        return roadmap


# Export main simulator instance
def create_simulator(seed: Optional[int] = None) -> WhatIfSimulator:
    """Create a what-if simulator instance"""
    return WhatIfSimulator(seed=seed)