"""
PolicyCortex ROI API Endpoints
Financial impact and what-if simulation endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import asyncio
import json

# Import ROI components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from core.roi.cost_calculator.calculator import CostCalculator, PolicyCostImpact, ResourceOptimization
    from core.roi.what_if_engine.simulator import WhatIfSimulator, ScenarioType, ScenarioParameter, SimulationResult
    from core.roi.pl_dashboard.metrics import PLDashboardMetrics, MetricPeriod, GovernancePL
    ROI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ROI modules not available: {e}")
    ROI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/roi", tags=["ROI & Financial Impact"])

# Request/Response models
class PolicyImpactRequest(BaseModel):
    policy_id: str
    policy_metrics: Dict[str, Any]
    time_range_days: int = Field(default=30, ge=1, le=365)

class WhatIfScenarioRequest(BaseModel):
    scenario_type: str
    parameters: List[Dict[str, Any]]
    current_monthly_cost: float
    time_horizon_days: int = Field(default=90, ge=30, le=365)
    additional_factors: Optional[Dict[str, Any]] = None

class PLDashboardRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metrics_data: Dict[str, Any]
    detailed: bool = True

class ROIReportRequest(BaseModel):
    period_days: int = Field(default=90, ge=30, le=365)
    format: str = Field(default="json", pattern="^(json|html|pdf)$")
    include_projections: bool = True

# Initialize services (with mock data for demo)
if ROI_AVAILABLE:
    cost_calculator = CostCalculator(
        subscription_id="demo-subscription",
        resource_group="demo-rg"
    )
    what_if_simulator = WhatIfSimulator(seed=42)
    pl_metrics = PLDashboardMetrics()
else:
    cost_calculator = None
    what_if_simulator = None
    pl_metrics = None

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_pl_dashboard(
    period: str = Query(default="monthly", pattern="^(daily|weekly|monthly|quarterly|yearly)$"),
    detailed: bool = Query(default=True)
):
    """
    Get P&L dashboard metrics
    
    Returns comprehensive financial metrics including:
    - Governance P&L statement
    - ROI percentage and payback period
    - Cost savings breakdown
    - KPI tracking
    """
    try:
        if not ROI_AVAILABLE:
            # Return mock data for demo
            return _get_mock_dashboard_data(period)
        
        # Calculate period dates
        end_date = datetime.now()
        if period == "daily":
            start_date = end_date - timedelta(days=1)
        elif period == "weekly":
            start_date = end_date - timedelta(weeks=1)
        elif period == "monthly":
            start_date = end_date - timedelta(days=30)
        elif period == "quarterly":
            start_date = end_date - timedelta(days=90)
        else:  # yearly
            start_date = end_date - timedelta(days=365)
        
        # Prepare metrics data (would come from actual monitoring in production)
        metrics_data = {
            "cost_data": {
                "baseline_monthly": 100000,
                "optimized_monthly": 85000
            },
            "security_data": {
                "prevented_incidents": 47,
                "severity_distribution": {
                    "minor": 0.6,
                    "moderate": 0.3,
                    "major": 0.09,
                    "critical": 0.01
                }
            },
            "automation_data": {
                "hours_saved": 150
            },
            "compliance_data": {
                "violations_prevented": {
                    "gdpr": 3,
                    "hipaa": 5,
                    "pci_dss": 2
                }
            },
            "resource_data": {
                "compute": {
                    "current_cost": 50000,
                    "optimization_rate": 0.7
                },
                "storage": {
                    "current_cost": 20000,
                    "optimization_rate": 0.6
                }
            },
            "policies_enforced": 234,
            "violations_prevented": 47,
            "resources_optimized": 156,
            "automation_hours_saved": 150
        }
        
        # Generate P&L statement
        pl_statement = await pl_metrics.generate_pl_statement(
            start_date=start_date,
            end_date=end_date,
            metrics_data=metrics_data,
            detailed=detailed
        )
        
        # Calculate KPIs
        kpis = await pl_metrics.calculate_kpis(pl_statement)
        
        return {
            "status": "success",
            "period": {
                "type": period,
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "pl_statement": pl_statement.to_dict(),
            "kpis": kpis,
            "summary": {
                "net_benefit": float(pl_statement.net_benefit),
                "roi_percentage": pl_statement.roi_percentage,
                "payback_months": pl_statement.payback_months,
                "total_savings": float(pl_statement.gross_benefit),
                "total_investment": float(pl_statement.total_investment)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating P&L dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate", response_model=Dict[str, Any])
async def simulate_what_if(request: WhatIfScenarioRequest):
    """
    Run what-if financial simulation
    
    Simulates financial impact of proposed changes using Monte Carlo analysis:
    - 30/60/90-day projections
    - Confidence intervals
    - ROI calculations
    - Risk assessment
    """
    try:
        if not ROI_AVAILABLE:
            # Return mock simulation data
            return _get_mock_simulation_data(request.scenario_type)
        
        # Convert parameters to ScenarioParameter objects
        scenario_params = []
        for param in request.parameters:
            scenario_params.append(ScenarioParameter(
                name=param["name"],
                current_value=param["current_value"],
                proposed_value=param["proposed_value"],
                unit=param.get("unit", ""),
                impact_factor=param.get("impact_factor", 0.1),
                confidence_interval=(
                    param.get("min_value", param["proposed_value"] * 0.9),
                    param.get("max_value", param["proposed_value"] * 1.1)
                ),
                distribution=param.get("distribution", "normal")
            ))
        
        # Run simulation
        result = await what_if_simulator.simulate_scenario(
            scenario_type=ScenarioType[request.scenario_type.upper()],
            parameters=scenario_params,
            current_monthly_cost=Decimal(str(request.current_monthly_cost)),
            time_horizon_days=request.time_horizon_days,
            additional_factors=request.additional_factors
        )
        
        # Perform sensitivity analysis
        sensitivity = await what_if_simulator.sensitivity_analysis(
            scenario_type=ScenarioType[request.scenario_type.upper()],
            base_parameters=scenario_params,
            current_monthly_cost=Decimal(str(request.current_monthly_cost))
        )
        
        return {
            "status": "success",
            "simulation": result.to_dict(),
            "sensitivity_analysis": sensitivity,
            "recommendations": result.recommendations,
            "confidence_level": result.confidence_level
        }
        
    except Exception as e:
        logger.error(f"Error running what-if simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/savings", response_model=Dict[str, Any])
async def get_cost_savings(
    days: int = Query(default=30, ge=1, le=365),
    category: Optional[str] = Query(default=None)
):
    """
    Get detailed cost savings data
    
    Returns breakdown of achieved and projected savings:
    - Per-policy impact
    - Resource optimization opportunities
    - Incident prevention value
    - Automation benefits
    """
    try:
        if not ROI_AVAILABLE:
            return _get_mock_savings_data(days)
        
        # Calculate various savings components
        savings_data = {
            "period_days": days,
            "total_savings": 0,
            "categories": {},
            "top_policies": [],
            "optimization_opportunities": [],
            "trends": {}
        }
        
        # Policy-based savings
        sample_policies = [
            {
                "id": "pol-001",
                "name": "VM Rightsizing Policy",
                "metrics": {
                    "violations_prevented": 23,
                    "severity_distribution": {"minor": 15, "moderate": 8},
                    "automation_hours_saved": 45,
                    "optimization_rate": 0.25
                }
            },
            {
                "id": "pol-002",
                "name": "Storage Optimization Policy",
                "metrics": {
                    "violations_prevented": 18,
                    "severity_distribution": {"minor": 12, "moderate": 6},
                    "automation_hours_saved": 30,
                    "optimization_rate": 0.20
                }
            },
            {
                "id": "pol-003",
                "name": "Network Security Policy",
                "metrics": {
                    "violations_prevented": 35,
                    "severity_distribution": {"moderate": 20, "major": 15},
                    "automation_hours_saved": 60,
                    "optimization_rate": 0.15
                }
            }
        ]
        
        for policy in sample_policies:
            impact = await cost_calculator.calculate_policy_impact(
                policy_id=policy["id"],
                policy_metrics=policy["metrics"],
                time_range_days=days
            )
            
            savings_data["top_policies"].append({
                "policy_id": policy["id"],
                "policy_name": policy["name"],
                "savings": float(impact.total_impact),
                "roi_percentage": (float(impact.total_impact) / 1000) * 100,  # Simplified ROI
                "violations_prevented": impact.violations_prevented
            })
            
            savings_data["total_savings"] += float(impact.total_impact)
        
        # Resource optimization opportunities
        sample_resources = [
            {
                "id": "vm-prod-001",
                "type": "Virtual Machine",
                "monthly_cost": 5000,
                "cpu_utilization": 15,
                "memory_utilization": 25,
                "age_days": 120
            },
            {
                "id": "storage-001",
                "type": "Storage Account",
                "monthly_cost": 2000,
                "storage_utilization": 30,
                "age_days": 200
            },
            {
                "id": "sql-db-001",
                "type": "SQL Database",
                "monthly_cost": 8000,
                "cpu_utilization": 10,
                "age_days": 150,
                "reserved_instance": False
            }
        ]
        
        optimizations = await cost_calculator.identify_optimization_opportunities(
            sample_resources
        )
        
        for opt in optimizations[:5]:  # Top 5 opportunities
            savings_data["optimization_opportunities"].append({
                "resource_id": opt.resource_id,
                "resource_type": opt.resource_type,
                "current_cost": float(opt.current_cost),
                "potential_savings": float(opt.savings_potential),
                "actions": opt.optimization_actions,
                "payback_days": opt.payback_period_days,
                "risk_level": opt.risk_level
            })
        
        # Category breakdown
        savings_data["categories"] = {
            "compute": savings_data["total_savings"] * 0.40,
            "storage": savings_data["total_savings"] * 0.20,
            "network": savings_data["total_savings"] * 0.15,
            "database": savings_data["total_savings"] * 0.15,
            "other": savings_data["total_savings"] * 0.10
        }
        
        # Trends (mock data for demo)
        savings_data["trends"] = {
            "daily_average": savings_data["total_savings"] / days,
            "projected_monthly": (savings_data["total_savings"] / days) * 30,
            "projected_quarterly": (savings_data["total_savings"] / days) * 90,
            "growth_rate": 15.5  # 15.5% month-over-month growth
        }
        
        return {
            "status": "success",
            "data": savings_data,
            "summary": {
                "total_savings": savings_data["total_savings"],
                "daily_rate": savings_data["trends"]["daily_average"],
                "top_opportunity": (
                    savings_data["optimization_opportunities"][0] 
                    if savings_data["optimization_opportunities"] else None
                ),
                "policies_contributing": len(savings_data["top_policies"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cost savings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report", response_model=Dict[str, Any])
async def get_executive_report(
    request: ROIReportRequest = Depends()
):
    """
    Generate executive ROI report
    
    Comprehensive financial impact report including:
    - Executive summary
    - P&L statement
    - ROI metrics
    - Projections
    - Recommendations
    """
    try:
        if not ROI_AVAILABLE:
            return _get_mock_report_data(request.period_days)
        
        # Calculate report period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.period_days)
        
        # Generate comprehensive metrics data
        metrics_data = {
            "cost_data": {
                "baseline_monthly": 120000,
                "optimized_monthly": 102000  # 15% savings
            },
            "security_data": {
                "prevented_incidents": 67,
                "severity_distribution": {
                    "minor": 0.55,
                    "moderate": 0.30,
                    "major": 0.12,
                    "critical": 0.03
                }
            },
            "automation_data": {
                "hours_saved": 280
            },
            "compliance_data": {
                "violations_prevented": {
                    "gdpr": 5,
                    "hipaa": 8,
                    "pci_dss": 3,
                    "sox": 2
                }
            },
            "resource_data": {
                "compute": {
                    "current_cost": 60000,
                    "optimization_rate": 0.75
                },
                "storage": {
                    "current_cost": 25000,
                    "optimization_rate": 0.65
                },
                "network": {
                    "current_cost": 15000,
                    "optimization_rate": 0.60
                },
                "database": {
                    "current_cost": 20000,
                    "optimization_rate": 0.70
                }
            },
            "policies_enforced": 412,
            "violations_prevented": 67,
            "resources_optimized": 234,
            "automation_hours_saved": 280,
            "platform_cost": 10000,
            "implementation_cost": 25000,
            "training_cost": 5000,
            "maintenance_cost": 2000
        }
        
        # Generate P&L statement
        pl_statement = await pl_metrics.generate_pl_statement(
            start_date=start_date,
            end_date=end_date,
            metrics_data=metrics_data,
            detailed=True
        )
        
        # Calculate KPIs
        kpis = await pl_metrics.calculate_kpis(pl_statement)
        
        # Generate executive report
        report = await pl_metrics.generate_executive_report(
            pl_statement=pl_statement,
            kpis=kpis,
            format=request.format
        )
        
        # Add projections if requested
        if request.include_projections:
            # Run what-if simulation for next period
            future_params = [
                ScenarioParameter(
                    name="policy_adoption",
                    current_value=0.75,
                    proposed_value=0.90,
                    unit="percentage",
                    impact_factor=0.3,
                    confidence_interval=(0.85, 0.95),
                    distribution="normal"
                ),
                ScenarioParameter(
                    name="automation_scope",
                    current_value=100,
                    proposed_value=150,
                    unit="processes",
                    impact_factor=0.25,
                    confidence_interval=(130, 170),
                    distribution="normal"
                )
            ]
            
            projection = await what_if_simulator.simulate_scenario(
                scenario_type=ScenarioType.COST_OPTIMIZATION,
                parameters=future_params,
                current_monthly_cost=Decimal("102000"),
                time_horizon_days=90
            )
            
            report["projections"] = {
                "next_quarter": projection.to_dict(),
                "annual_projection": {
                    "estimated_savings": float(projection.expected_savings) * 4,
                    "roi_percentage": projection.roi_percentage * 1.5,
                    "confidence": projection.confidence_level * 0.85
                }
            }
        
        # Add achievement highlights
        report["achievements"] = {
            "cost_savings_achieved": f"{((120000 - 102000) / 120000 * 100):.1f}%",
            "target_vs_actual": "Exceeded 15% savings target",
            "key_wins": [
                f"Prevented {pl_statement.violations_prevented} policy violations",
                f"Saved {pl_statement.automation_hours_saved:.0f} hours through automation",
                f"Optimized {pl_statement.resources_optimized} cloud resources",
                f"Achieved {pl_statement.roi_percentage:.0f}% ROI in {request.period_days} days"
            ]
        }
        
        return {
            "status": "success",
            "report": report,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "period_days": request.period_days,
                "format": request.format,
                "confidence_level": 0.92  # High confidence in data
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating executive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mock data generators for demo mode
def _get_mock_dashboard_data(period: str) -> Dict[str, Any]:
    """Generate mock dashboard data for demo"""
    base_savings = 15000 if period == "monthly" else 45000 if period == "quarterly" else 5000
    
    return {
        "status": "success",
        "period": {
            "type": period,
            "start": (datetime.now() - timedelta(days=30)).isoformat(),
            "end": datetime.now().isoformat()
        },
        "pl_statement": {
            "revenue": {
                "direct_cost_savings": base_savings * 0.4,
                "incident_prevention": base_savings * 0.25,
                "productivity_gains": base_savings * 0.2,
                "compliance_avoidance": base_savings * 0.1,
                "optimization_savings": base_savings * 0.05,
                "total": base_savings
            },
            "costs": {
                "platform": 10000,
                "implementation": 5000,
                "training": 2000,
                "maintenance": 1000,
                "total": 18000
            },
            "net_results": {
                "net_benefit": base_savings - 18000,
                "roi_percentage": ((base_savings - 18000) / 18000 * 100) if base_savings > 18000 else 0,
                "payback_months": 3.5
            }
        },
        "kpis": {
            "current_period": {
                "savings_rate": {"value": 12.5, "target": 15, "status": "amber", "unit": "%"},
                "roi": {"value": 250, "target": 300, "status": "amber", "unit": "%"},
                "payback_period": {"value": 3.5, "target": 6, "status": "green", "unit": "months"},
                "automation_hours": {"value": 120, "target": 100, "status": "green", "unit": "hours/month"}
            },
            "health_scores": {
                "financial_health": 85,
                "operational_health": 92,
                "overall_health": 88.5
            }
        },
        "summary": {
            "net_benefit": base_savings - 18000,
            "roi_percentage": ((base_savings - 18000) / 18000 * 100) if base_savings > 18000 else 0,
            "payback_months": 3.5,
            "total_savings": base_savings,
            "total_investment": 18000
        }
    }

def _get_mock_simulation_data(scenario_type: str) -> Dict[str, Any]:
    """Generate mock simulation data for demo"""
    return {
        "status": "success",
        "simulation": {
            "scenario_id": "sim-12345",
            "scenario_type": scenario_type,
            "time_horizon_days": 90,
            "base_cost": 300000,
            "expected_savings": 45000,
            "roi_percentage": 250,
            "payback_period_days": 45,
            "risk_score": 3.5,
            "confidence_level": 0.87,
            "key_metrics": {
                "mean_daily_savings": "$500.00",
                "best_case_savings": "$750.00",
                "worst_case_savings": "$250.00",
                "savings_probability": "92.3%"
            }
        },
        "sensitivity_analysis": {
            "critical_parameters": ["resource_utilization", "policy_compliance"],
            "robust_parameters": ["team_size"],
            "tornado_chart_data": [
                {"parameter": "resource_utilization", "sensitivity": 15.2},
                {"parameter": "policy_compliance", "sensitivity": 12.8},
                {"parameter": "automation_scope", "sensitivity": 8.5}
            ]
        },
        "recommendations": [
            "PRIORITY: Quick win with <30 day payback - implement immediately",
            "Low risk scenario - proceed with standard change process",
            "Implement auto-scaling policies where possible",
            "Track automation success rate and adjust accordingly"
        ],
        "confidence_level": 0.87
    }

def _get_mock_savings_data(days: int) -> Dict[str, Any]:
    """Generate mock savings data for demo"""
    daily_savings = 1500
    total = daily_savings * days
    
    return {
        "status": "success",
        "data": {
            "period_days": days,
            "total_savings": total,
            "categories": {
                "compute": total * 0.40,
                "storage": total * 0.20,
                "network": total * 0.15,
                "database": total * 0.15,
                "other": total * 0.10
            },
            "top_policies": [
                {
                    "policy_id": "pol-001",
                    "policy_name": "VM Rightsizing Policy",
                    "savings": total * 0.35,
                    "roi_percentage": 320,
                    "violations_prevented": 23
                },
                {
                    "policy_id": "pol-002",
                    "policy_name": "Storage Optimization Policy",
                    "savings": total * 0.25,
                    "roi_percentage": 280,
                    "violations_prevented": 18
                }
            ],
            "optimization_opportunities": [
                {
                    "resource_id": "vm-prod-001",
                    "resource_type": "Virtual Machine",
                    "current_cost": 5000,
                    "potential_savings": 1500,
                    "actions": ["Rightsize: CPU at 15%", "Purchase Reserved Instance"],
                    "payback_days": 7,
                    "risk_level": "Low"
                }
            ],
            "trends": {
                "daily_average": daily_savings,
                "projected_monthly": daily_savings * 30,
                "projected_quarterly": daily_savings * 90,
                "growth_rate": 15.5
            }
        },
        "summary": {
            "total_savings": total,
            "daily_rate": daily_savings,
            "top_opportunity": {
                "resource_id": "vm-prod-001",
                "potential_savings": 1500
            },
            "policies_contributing": 5
        }
    }

def _get_mock_report_data(period_days: int) -> Dict[str, Any]:
    """Generate mock executive report data"""
    return {
        "status": "success",
        "report": {
            "executive_summary": (
                f"PolicyCortex delivered $54,000 in net benefits over {period_days} days, "
                f"achieving a 250% ROI with a payback period of 3.5 months. "
                f"The platform prevented 67 policy violations and saved 280 hours through automation."
            ),
            "financial_impact": _get_mock_dashboard_data("quarterly")["pl_statement"],
            "kpis": _get_mock_dashboard_data("quarterly")["kpis"],
            "recommendations": [
                "Expand automation scope to increase ROI - target processes with >20 hours/month manual effort",
                "Implement aggressive rightsizing policies - analysis shows 30% of resources are oversized",
                "SUCCESS: Consider expanding PolicyCortex deployment to additional departments"
            ],
            "risk_mitigation": {
                "security_risk_reduction": 90000,
                "compliance_risk_reduction": 450000,
                "operational_risk_reduction": 67500,
                "total_risk_value": 607500,
                "risk_adjusted_roi": 312.5
            },
            "next_quarter_projection": {
                "estimated_savings": 62100,
                "projected_roi": 287.5,
                "additional_optimizations": 269,
                "automation_hours": 322,
                "confidence": 0.75
            }
        },
        "achievements": {
            "cost_savings_achieved": "15.0%",
            "target_vs_actual": "Exceeded 15% savings target",
            "key_wins": [
                "Prevented 67 policy violations",
                "Saved 280 hours through automation",
                "Optimized 234 cloud resources",
                f"Achieved 250% ROI in {period_days} days"
            ]
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "period_days": period_days,
            "format": "json",
            "confidence_level": 0.92
        }
    }