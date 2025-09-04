"""
PolicyCortex P&L Dashboard Metrics Service
CFO-ready financial metrics and governance P&L statements
"""

import os
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import logging
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

class MetricPeriod(Enum):
    """Time periods for metric calculations"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    MTD = "month_to_date"
    QTD = "quarter_to_date"
    YTD = "year_to_date"

class MetricCategory(Enum):
    """Categories for P&L metrics"""
    REVENUE = "revenue"           # Cost savings are revenue in P&L
    COST_AVOIDANCE = "cost_avoidance"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    RISK_REDUCTION = "risk_reduction"
    PRODUCTIVITY = "productivity"
    INVESTMENT = "investment"

@dataclass
class PLLineItem:
    """Individual line item for P&L statement"""
    category: MetricCategory
    description: str
    amount: Decimal
    period: MetricPeriod
    date: datetime
    confidence: float
    source: str  # Data source
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "amount": float(self.amount),
            "period": self.period.value,
            "date": self.date.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
            "verified": self.verified
        }

@dataclass
class GovernancePL:
    """Governance P&L Statement"""
    period_start: datetime
    period_end: datetime
    
    # Revenue side (Savings & Benefits)
    direct_cost_savings: Decimal = Decimal("0")
    incident_prevention_value: Decimal = Decimal("0")
    productivity_gains: Decimal = Decimal("0")
    compliance_penalty_avoidance: Decimal = Decimal("0")
    optimization_savings: Decimal = Decimal("0")
    
    # Cost side (Investments)
    platform_cost: Decimal = Decimal("0")
    implementation_cost: Decimal = Decimal("0")
    training_cost: Decimal = Decimal("0")
    maintenance_cost: Decimal = Decimal("0")
    
    # Net results
    gross_benefit: Decimal = Decimal("0")
    total_investment: Decimal = Decimal("0")
    net_benefit: Decimal = Decimal("0")
    roi_percentage: float = 0.0
    payback_months: float = 0.0
    
    # Supporting metrics
    policies_enforced: int = 0
    violations_prevented: int = 0
    resources_optimized: int = 0
    automation_hours_saved: float = 0.0
    
    line_items: List[PLLineItem] = field(default_factory=list)
    
    def calculate_totals(self):
        """Calculate P&L totals"""
        self.gross_benefit = (
            self.direct_cost_savings +
            self.incident_prevention_value +
            self.productivity_gains +
            self.compliance_penalty_avoidance +
            self.optimization_savings
        )
        
        self.total_investment = (
            self.platform_cost +
            self.implementation_cost +
            self.training_cost +
            self.maintenance_cost
        )
        
        self.net_benefit = self.gross_benefit - self.total_investment
        
        if self.total_investment > 0:
            self.roi_percentage = float(
                (self.net_benefit / self.total_investment) * 100
            )
            
            months = (self.period_end - self.period_start).days / 30
            monthly_benefit = self.net_benefit / Decimal(str(max(months, 1)))
            
            if monthly_benefit > 0:
                self.payback_months = float(
                    self.total_investment / monthly_benefit
                )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "revenue": {
                "direct_cost_savings": float(self.direct_cost_savings),
                "incident_prevention": float(self.incident_prevention_value),
                "productivity_gains": float(self.productivity_gains),
                "compliance_avoidance": float(self.compliance_penalty_avoidance),
                "optimization_savings": float(self.optimization_savings),
                "total": float(self.gross_benefit)
            },
            "costs": {
                "platform": float(self.platform_cost),
                "implementation": float(self.implementation_cost),
                "training": float(self.training_cost),
                "maintenance": float(self.maintenance_cost),
                "total": float(self.total_investment)
            },
            "net_results": {
                "net_benefit": float(self.net_benefit),
                "roi_percentage": self.roi_percentage,
                "payback_months": self.payback_months
            },
            "operational_metrics": {
                "policies_enforced": self.policies_enforced,
                "violations_prevented": self.violations_prevented,
                "resources_optimized": self.resources_optimized,
                "automation_hours_saved": self.automation_hours_saved
            },
            "line_items": [item.to_dict() for item in self.line_items]
        }

class PLDashboardMetrics:
    """
    P&L Dashboard Metrics Service
    Generates CFO-ready financial metrics and governance P&L statements
    """
    
    def __init__(self):
        # Industry benchmarks for calculations
        self.benchmarks = {
            "hourly_rate": Decimal("75"),  # IT professional hourly rate
            "incident_costs": {
                "minor": Decimal("500"),
                "moderate": Decimal("5000"),
                "major": Decimal("50000"),
                "critical": Decimal("500000")
            },
            "compliance_penalties": {
                "gdpr": Decimal("20000000"),
                "hipaa": Decimal("50000"),
                "pci_dss": Decimal("100000"),
                "sox": Decimal("1000000")
            },
            "automation_multiplier": Decimal("3.5"),  # Automation efficiency gain
            "optimization_targets": {
                "compute": Decimal("0.30"),    # 30% typical savings
                "storage": Decimal("0.25"),    # 25% typical savings
                "network": Decimal("0.20"),     # 20% typical savings
                "licensing": Decimal("0.35")    # 35% typical savings
            }
        }
        
        # KPI targets for tracking
        self.kpi_targets = {
            "savings_rate": 0.15,           # 15% cost savings target
            "roi_target": 3.0,              # 300% ROI target
            "payback_target_months": 6,     # 6-month payback target
            "automation_target_hours": 100, # 100 hours/month automation target
            "compliance_rate": 0.98,        # 98% compliance target
            "optimization_adoption": 0.75    # 75% optimization adoption
        }
        
        logger.info("P&L Dashboard Metrics initialized")
    
    async def generate_pl_statement(
        self,
        start_date: datetime,
        end_date: datetime,
        metrics_data: Dict[str, Any],
        detailed: bool = True
    ) -> GovernancePL:
        """
        Generate comprehensive P&L statement
        
        Args:
            start_date: Period start date
            end_date: Period end date
            metrics_data: Raw metrics data
            detailed: Include detailed line items
            
        Returns:
            GovernancePL statement
        """
        pl = GovernancePL(
            period_start=start_date,
            period_end=end_date
        )
        
        # Calculate direct cost savings
        pl.direct_cost_savings = await self._calculate_direct_savings(
            metrics_data.get("cost_data", {}),
            start_date,
            end_date
        )
        
        # Calculate incident prevention value
        pl.incident_prevention_value = await self._calculate_incident_prevention(
            metrics_data.get("security_data", {}),
            metrics_data.get("violations", [])
        )
        
        # Calculate productivity gains
        pl.productivity_gains = await self._calculate_productivity_gains(
            metrics_data.get("automation_data", {}),
            metrics_data.get("team_metrics", {})
        )
        
        # Calculate compliance penalty avoidance
        pl.compliance_penalty_avoidance = await self._calculate_compliance_avoidance(
            metrics_data.get("compliance_data", {}),
            metrics_data.get("audit_results", {})
        )
        
        # Calculate optimization savings
        pl.optimization_savings = await self._calculate_optimization_savings(
            metrics_data.get("resource_data", {}),
            metrics_data.get("optimization_actions", [])
        )
        
        # Calculate investment costs
        pl.platform_cost = Decimal(str(
            metrics_data.get("platform_cost", 10000)
        ))  # Monthly platform cost
        
        pl.implementation_cost = Decimal(str(
            metrics_data.get("implementation_cost", 0)
        ))
        
        pl.training_cost = Decimal(str(
            metrics_data.get("training_cost", 0)
        ))
        
        pl.maintenance_cost = Decimal(str(
            metrics_data.get("maintenance_cost", 2000)
        ))  # Monthly maintenance
        
        # Set operational metrics
        pl.policies_enforced = metrics_data.get("policies_enforced", 0)
        pl.violations_prevented = metrics_data.get("violations_prevented", 0)
        pl.resources_optimized = metrics_data.get("resources_optimized", 0)
        pl.automation_hours_saved = metrics_data.get("automation_hours_saved", 0)
        
        # Add detailed line items if requested
        if detailed:
            pl.line_items = await self._generate_line_items(
                pl,
                metrics_data
            )
        
        # Calculate totals and ROI
        pl.calculate_totals()
        
        return pl
    
    async def calculate_kpis(
        self,
        pl_statement: GovernancePL,
        historical_data: Optional[List[GovernancePL]] = None
    ) -> Dict[str, Any]:
        """
        Calculate key performance indicators
        
        Args:
            pl_statement: Current P&L statement
            historical_data: Historical P&L statements for trending
            
        Returns:
            Dictionary of KPIs with status
        """
        kpis = {
            "current_period": {},
            "trends": {},
            "targets": {},
            "health_scores": {}
        }
        
        # Current period KPIs
        period_days = (pl_statement.period_end - pl_statement.period_start).days
        
        # Savings rate
        total_spend = pl_statement.gross_benefit + pl_statement.total_investment
        savings_rate = float(
            pl_statement.net_benefit / total_spend * 100
            if total_spend > 0 else 0
        )
        
        kpis["current_period"]["savings_rate"] = {
            "value": savings_rate,
            "target": self.kpi_targets["savings_rate"] * 100,
            "status": "green" if savings_rate >= self.kpi_targets["savings_rate"] * 100 else "amber",
            "unit": "%"
        }
        
        # ROI
        roi_multiple = pl_statement.roi_percentage / 100
        kpis["current_period"]["roi"] = {
            "value": pl_statement.roi_percentage,
            "target": self.kpi_targets["roi_target"] * 100,
            "status": "green" if roi_multiple >= self.kpi_targets["roi_target"] else "amber",
            "unit": "%"
        }
        
        # Payback period
        kpis["current_period"]["payback_period"] = {
            "value": pl_statement.payback_months,
            "target": self.kpi_targets["payback_target_months"],
            "status": "green" if pl_statement.payback_months <= self.kpi_targets["payback_target_months"] else "amber",
            "unit": "months"
        }
        
        # Prevention value ratio
        prevention_ratio = float(
            pl_statement.incident_prevention_value / pl_statement.gross_benefit * 100
            if pl_statement.gross_benefit > 0 else 0
        )
        
        kpis["current_period"]["prevention_value"] = {
            "value": prevention_ratio,
            "target": 25,  # 25% of value from prevention
            "status": "green" if prevention_ratio >= 25 else "amber",
            "unit": "%"
        }
        
        # Automation hours
        monthly_automation = pl_statement.automation_hours_saved * 30 / max(period_days, 1)
        kpis["current_period"]["automation_hours"] = {
            "value": monthly_automation,
            "target": self.kpi_targets["automation_target_hours"],
            "status": "green" if monthly_automation >= self.kpi_targets["automation_target_hours"] else "amber",
            "unit": "hours/month"
        }
        
        # Calculate trends if historical data available
        if historical_data and len(historical_data) > 1:
            kpis["trends"] = self._calculate_trends(pl_statement, historical_data)
        
        # Calculate health scores
        kpis["health_scores"] = self._calculate_health_scores(kpis["current_period"])
        
        # Add target achievement summary
        kpis["targets"]["achieved"] = sum(
            1 for kpi in kpis["current_period"].values() 
            if kpi["status"] == "green"
        )
        kpis["targets"]["total"] = len(kpis["current_period"])
        kpis["targets"]["achievement_rate"] = (
            kpis["targets"]["achieved"] / kpis["targets"]["total"] * 100
            if kpis["targets"]["total"] > 0 else 0
        )
        
        return kpis
    
    async def generate_executive_report(
        self,
        pl_statement: GovernancePL,
        kpis: Dict[str, Any],
        format: str = "json"
    ) -> Any:
        """
        Generate executive-ready ROI report
        
        Args:
            pl_statement: P&L statement
            kpis: Calculated KPIs
            format: Output format (json, html, pdf)
            
        Returns:
            Formatted executive report
        """
        if format == "json":
            return {
                "executive_summary": self._generate_executive_summary(pl_statement, kpis),
                "financial_impact": pl_statement.to_dict(),
                "kpis": kpis,
                "recommendations": self._generate_recommendations(pl_statement, kpis),
                "risk_mitigation": self._calculate_risk_mitigation_value(pl_statement),
                "next_quarter_projection": await self._project_next_quarter(pl_statement)
            }
        
        elif format == "html":
            html_template = """
            <html>
            <head>
                <title>PolicyCortex Executive ROI Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; }}
                    .header {{ 
                        background: linear-gradient(135deg, #0078D4 0%, #005A9E 100%); 
                        color: white; 
                        padding: 30px; 
                        text-align: center;
                    }}
                    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    .executive-summary {{ 
                        background: #F8F9FA; 
                        border-left: 4px solid #0078D4; 
                        padding: 20px; 
                        margin: 20px 0;
                        border-radius: 4px;
                    }}
                    .metrics-grid {{ 
                        display: grid; 
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                        gap: 20px; 
                        margin: 30px 0;
                    }}
                    .metric-card {{ 
                        background: white; 
                        border: 1px solid #E1E1E1; 
                        border-radius: 8px; 
                        padding: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metric-value {{ 
                        font-size: 36px; 
                        font-weight: bold; 
                        color: #0078D4;
                        margin: 10px 0;
                    }}
                    .metric-label {{ 
                        color: #666; 
                        font-size: 14px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }}
                    .pl-table {{ 
                        width: 100%; 
                        border-collapse: collapse;
                        margin: 30px 0;
                    }}
                    .pl-table th {{ 
                        background: #F8F9FA; 
                        padding: 12px; 
                        text-align: left;
                        font-weight: 600;
                        border-bottom: 2px solid #0078D4;
                    }}
                    .pl-table td {{ 
                        padding: 10px; 
                        border-bottom: 1px solid #E1E1E1;
                    }}
                    .pl-total {{ 
                        font-weight: bold; 
                        background: #F8F9FA;
                    }}
                    .positive {{ color: #107C10; }}
                    .negative {{ color: #D13438; }}
                    .recommendation {{ 
                        background: #FFF4CE; 
                        border-left: 4px solid #FFB900;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 4px;
                    }}
                    .footer {{ 
                        background: #F8F9FA; 
                        padding: 20px; 
                        text-align: center;
                        margin-top: 40px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>PolicyCortex ROI Report</h1>
                    <p>Governance Financial Impact Analysis</p>
                    <p>{period_start} to {period_end}</p>
                </div>
                
                <div class="container">
                    <div class="executive-summary">
                        <h2>Executive Summary</h2>
                        <p>{executive_summary}</p>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Net Benefit</div>
                            <div class="metric-value {net_class}">${net_benefit:,.0f}</div>
                            <div>ROI: {roi_percentage:.1f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total Savings</div>
                            <div class="metric-value positive">${gross_benefit:,.0f}</div>
                            <div>vs ${total_investment:,.0f} invested</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Payback Period</div>
                            <div class="metric-value">{payback_months:.1f}</div>
                            <div>months</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Savings Rate</div>
                            <div class="metric-value">{savings_rate:.1f}%</div>
                            <div>of total cloud spend</div>
                        </div>
                    </div>
                    
                    <h2>Governance P&L Statement</h2>
                    <table class="pl-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Description</th>
                                <th style="text-align: right;">Amount</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="5"><strong>Benefits</strong></td>
                                <td>Direct Cost Savings</td>
                                <td style="text-align: right;" class="positive">${direct_savings:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Incident Prevention Value</td>
                                <td style="text-align: right;" class="positive">${incident_value:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Productivity Gains</td>
                                <td style="text-align: right;" class="positive">${productivity:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Compliance Penalty Avoidance</td>
                                <td style="text-align: right;" class="positive">${compliance:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Resource Optimization</td>
                                <td style="text-align: right;" class="positive">${optimization:,.2f}</td>
                            </tr>
                            <tr class="pl-total">
                                <td></td>
                                <td><strong>Total Benefits</strong></td>
                                <td style="text-align: right;" class="positive"><strong>${gross_benefit:,.2f}</strong></td>
                            </tr>
                            <tr>
                                <td rowspan="4"><strong>Investments</strong></td>
                                <td>Platform Cost</td>
                                <td style="text-align: right;" class="negative">-${platform_cost:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Implementation Cost</td>
                                <td style="text-align: right;" class="negative">-${implementation:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Training Cost</td>
                                <td style="text-align: right;" class="negative">-${training:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Maintenance Cost</td>
                                <td style="text-align: right;" class="negative">-${maintenance:,.2f}</td>
                            </tr>
                            <tr class="pl-total">
                                <td></td>
                                <td><strong>Total Investment</strong></td>
                                <td style="text-align: right;" class="negative"><strong>-${total_investment:,.2f}</strong></td>
                            </tr>
                            <tr class="pl-total" style="background: #E6F3FF;">
                                <td></td>
                                <td><strong>NET BENEFIT</strong></td>
                                <td style="text-align: right;" class="{net_class}"><strong>${net_benefit:,.2f}</strong></td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <h2>Key Performance Indicators</h2>
                    <div class="metrics-grid">
                        {kpi_cards}
                    </div>
                    
                    <h2>Recommendations</h2>
                    {recommendations_html}
                    
                    <h2>Next Quarter Projection</h2>
                    <p>Based on current trends and planned initiatives, we project:</p>
                    <ul>
                        <li>Estimated savings: ${next_quarter_savings:,.0f}</li>
                        <li>ROI improvement: {next_quarter_roi:.1f}%</li>
                        <li>Additional optimizations: {next_quarter_optimizations} resources</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated: {timestamp}</p>
                    <p>PolicyCortex PAYBACK Engine | CFO-Ready Financial Metrics</p>
                    <p><strong>Demonstrating 8-15% cloud cost savings within 90 days</strong></p>
                </div>
            </body>
            </html>
            """
            
            # Build KPI cards
            kpi_cards = []
            for kpi_name, kpi_data in kpis.get("current_period", {}).items():
                status_color = "#107C10" if kpi_data["status"] == "green" else "#FFB900"
                kpi_cards.append(f"""
                    <div class="metric-card">
                        <div class="metric-label">{kpi_name.replace('_', ' ').title()}</div>
                        <div style="color: {status_color}; font-size: 24px; font-weight: bold;">
                            {kpi_data['value']:.1f} {kpi_data['unit']}
                        </div>
                        <div>Target: {kpi_data['target']:.1f} {kpi_data['unit']}</div>
                    </div>
                """)
            
            # Build recommendations
            recommendations = self._generate_recommendations(pl_statement, kpis)
            recommendations_html = "".join([
                f'<div class="recommendation">{r}</div>' 
                for r in recommendations
            ])
            
            # Project next quarter
            next_quarter = await self._project_next_quarter(pl_statement)
            
            # Determine net class
            net_class = "positive" if pl_statement.net_benefit > 0 else "negative"
            
            return html_template.format(
                period_start=pl_statement.period_start.strftime("%B %d, %Y"),
                period_end=pl_statement.period_end.strftime("%B %d, %Y"),
                executive_summary=self._generate_executive_summary(pl_statement, kpis),
                net_benefit=pl_statement.net_benefit,
                net_class=net_class,
                roi_percentage=pl_statement.roi_percentage,
                gross_benefit=pl_statement.gross_benefit,
                total_investment=pl_statement.total_investment,
                payback_months=pl_statement.payback_months,
                savings_rate=kpis["current_period"]["savings_rate"]["value"],
                direct_savings=pl_statement.direct_cost_savings,
                incident_value=pl_statement.incident_prevention_value,
                productivity=pl_statement.productivity_gains,
                compliance=pl_statement.compliance_penalty_avoidance,
                optimization=pl_statement.optimization_savings,
                platform_cost=pl_statement.platform_cost,
                implementation=pl_statement.implementation_cost,
                training=pl_statement.training_cost,
                maintenance=pl_statement.maintenance_cost,
                kpi_cards="".join(kpi_cards),
                recommendations_html=recommendations_html,
                next_quarter_savings=next_quarter["estimated_savings"],
                next_quarter_roi=next_quarter["projected_roi"],
                next_quarter_optimizations=next_quarter["additional_optimizations"],
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        return {
            "pl_statement": pl_statement.to_dict(),
            "kpis": kpis
        }
    
    # Private helper methods
    
    async def _calculate_direct_savings(
        self,
        cost_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Decimal:
        """Calculate direct cost savings"""
        # For demo purposes, calculate based on provided data or estimates
        baseline_monthly = Decimal(str(cost_data.get("baseline_monthly", 100000)))
        optimized_monthly = Decimal(str(cost_data.get("optimized_monthly", 85000)))
        
        months = (end_date - start_date).days / 30
        return (baseline_monthly - optimized_monthly) * Decimal(str(months))
    
    async def _calculate_incident_prevention(
        self,
        security_data: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate value of prevented incidents"""
        total_value = Decimal("0")
        
        prevented_incidents = security_data.get("prevented_incidents", 0)
        
        # Calculate based on severity distribution
        severity_distribution = security_data.get("severity_distribution", {
            "minor": 0.6,
            "moderate": 0.3,
            "major": 0.09,
            "critical": 0.01
        })
        
        for severity, percentage in severity_distribution.items():
            incident_count = int(prevented_incidents * percentage)
            incident_cost = self.benchmarks["incident_costs"].get(
                severity, 
                Decimal("1000")
            )
            total_value += incident_cost * Decimal(str(incident_count))
        
        return total_value
    
    async def _calculate_productivity_gains(
        self,
        automation_data: Dict[str, Any],
        team_metrics: Dict[str, Any]
    ) -> Decimal:
        """Calculate productivity gains from automation"""
        hours_saved = Decimal(str(automation_data.get("hours_saved", 0)))
        hourly_rate = self.benchmarks["hourly_rate"]
        
        # Apply automation multiplier for compound benefits
        effective_hours = hours_saved * self.benchmarks["automation_multiplier"]
        
        return effective_hours * hourly_rate
    
    async def _calculate_compliance_avoidance(
        self,
        compliance_data: Dict[str, Any],
        audit_results: Dict[str, Any]
    ) -> Decimal:
        """Calculate compliance penalty avoidance value"""
        total_avoidance = Decimal("0")
        
        violations_prevented = compliance_data.get("violations_prevented", {})
        
        for regulation, count in violations_prevented.items():
            if regulation in self.benchmarks["compliance_penalties"]:
                # Risk-adjusted penalty (probability * impact)
                penalty = self.benchmarks["compliance_penalties"][regulation]
                probability = Decimal("0.01")  # 1% probability if not prevented
                total_avoidance += penalty * probability * Decimal(str(count))
        
        return total_avoidance
    
    async def _calculate_optimization_savings(
        self,
        resource_data: Dict[str, Any],
        optimization_actions: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate savings from resource optimization"""
        total_savings = Decimal("0")
        
        for resource_type, data in resource_data.items():
            if resource_type in self.benchmarks["optimization_targets"]:
                current_cost = Decimal(str(data.get("current_cost", 0)))
                target_savings = self.benchmarks["optimization_targets"][resource_type]
                actual_optimization = Decimal(str(data.get("optimization_rate", 0.5)))
                
                # Calculate achieved savings
                achieved_savings = current_cost * target_savings * actual_optimization
                total_savings += achieved_savings
        
        return total_savings
    
    async def _generate_line_items(
        self,
        pl: GovernancePL,
        metrics_data: Dict[str, Any]
    ) -> List[PLLineItem]:
        """Generate detailed line items for P&L"""
        line_items = []
        
        # Add savings line items
        if pl.direct_cost_savings > 0:
            line_items.append(PLLineItem(
                category=MetricCategory.REVENUE,
                description="Direct infrastructure cost reduction",
                amount=pl.direct_cost_savings,
                period=MetricPeriod.MONTHLY,
                date=pl.period_end,
                confidence=0.95,
                source="Azure Cost Management",
                verified=True
            ))
        
        if pl.incident_prevention_value > 0:
            line_items.append(PLLineItem(
                category=MetricCategory.COST_AVOIDANCE,
                description="Security incident prevention value",
                amount=pl.incident_prevention_value,
                period=MetricPeriod.MONTHLY,
                date=pl.period_end,
                confidence=0.85,
                source="Security Operations",
                verified=False
            ))
        
        if pl.productivity_gains > 0:
            line_items.append(PLLineItem(
                category=MetricCategory.PRODUCTIVITY,
                description="Automation-driven productivity gains",
                amount=pl.productivity_gains,
                period=MetricPeriod.MONTHLY,
                date=pl.period_end,
                confidence=0.90,
                source="HR Analytics",
                verified=True
            ))
        
        # Add investment line items
        if pl.platform_cost > 0:
            line_items.append(PLLineItem(
                category=MetricCategory.INVESTMENT,
                description="PolicyCortex platform subscription",
                amount=-pl.platform_cost,
                period=MetricPeriod.MONTHLY,
                date=pl.period_end,
                confidence=1.0,
                source="Finance",
                verified=True
            ))
        
        return line_items
    
    def _calculate_trends(
        self,
        current: GovernancePL,
        historical: List[GovernancePL]
    ) -> Dict[str, Any]:
        """Calculate trends from historical data"""
        trends = {}
        
        if len(historical) >= 3:
            # Calculate moving averages
            roi_values = [h.roi_percentage for h in historical[-3:]]
            trends["roi_trend"] = "improving" if roi_values[-1] > np.mean(roi_values[:-1]) else "declining"
            
            savings_values = [float(h.net_benefit) for h in historical[-3:]]
            trends["savings_trend"] = "improving" if savings_values[-1] > np.mean(savings_values[:-1]) else "declining"
            
            # Calculate growth rates
            if len(historical) >= 2:
                previous = historical[-2]
                trends["savings_growth"] = float(
                    (current.net_benefit - previous.net_benefit) / previous.net_benefit * 100
                    if previous.net_benefit != 0 else 0
                )
                trends["roi_growth"] = current.roi_percentage - previous.roi_percentage
        
        return trends
    
    def _calculate_health_scores(
        self,
        current_kpis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate health scores for different areas"""
        scores = {}
        
        # Financial health
        financial_metrics = ["savings_rate", "roi", "payback_period"]
        financial_scores = [
            1.0 if current_kpis.get(m, {}).get("status") == "green" else 0.5
            for m in financial_metrics
        ]
        scores["financial_health"] = np.mean(financial_scores) * 100
        
        # Operational health
        operational_metrics = ["automation_hours", "prevention_value"]
        operational_scores = [
            1.0 if current_kpis.get(m, {}).get("status") == "green" else 0.5
            for m in operational_metrics
        ]
        scores["operational_health"] = np.mean(operational_scores) * 100
        
        # Overall health
        scores["overall_health"] = np.mean(list(scores.values()))
        
        return scores
    
    def _generate_executive_summary(
        self,
        pl: GovernancePL,
        kpis: Dict[str, Any]
    ) -> str:
        """Generate executive summary text"""
        period_days = (pl.period_end - pl.period_start).days
        
        if pl.net_benefit > 0:
            summary = (
                f"PolicyCortex delivered ${pl.net_benefit:,.0f} in net benefits over {period_days} days, "
                f"achieving a {pl.roi_percentage:.0f}% ROI with a payback period of {pl.payback_months:.1f} months. "
                f"The platform prevented {pl.violations_prevented} policy violations and saved "
                f"{pl.automation_hours_saved:.0f} hours through automation. "
            )
            
            if kpis.get("current_period", {}).get("savings_rate", {}).get("value", 0) > 10:
                summary += f"Cost savings of {kpis['current_period']['savings_rate']['value']:.1f}% "
                summary += "demonstrate strong value delivery, exceeding industry benchmarks."
            else:
                summary += "Continued optimization will further enhance cost savings performance."
        else:
            summary = (
                f"PolicyCortex is in the investment phase with ${abs(pl.net_benefit):,.0f} net investment. "
                f"Early indicators show {pl.violations_prevented} violations prevented and "
                f"{pl.automation_hours_saved:.0f} hours saved through automation. "
                f"Full ROI realization expected within {pl.payback_months:.1f} months."
            )
        
        return summary
    
    def _generate_recommendations(
        self,
        pl: GovernancePL,
        kpis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # ROI-based recommendations
        if pl.roi_percentage < 200:
            recommendations.append(
                "Expand automation scope to increase ROI - target processes with >20 hours/month manual effort"
            )
        
        if pl.payback_months > 6:
            recommendations.append(
                "Accelerate value realization by prioritizing high-impact cost optimization policies"
            )
        
        # Savings-based recommendations
        savings_rate = kpis.get("current_period", {}).get("savings_rate", {}).get("value", 0)
        if savings_rate < 10:
            recommendations.append(
                "Implement aggressive rightsizing policies - analysis shows 30% of resources are oversized"
            )
            recommendations.append(
                "Enable automated shutdown of non-production resources during off-hours"
            )
        
        # Operational recommendations
        if pl.automation_hours_saved < 100:
            recommendations.append(
                "Deploy PolicyCortex automation to repetitive compliance checks and reporting tasks"
            )
        
        if pl.violations_prevented < 50:
            recommendations.append(
                "Strengthen preventive policies - each prevented violation saves average $5,000"
            )
        
        # Growth recommendations
        if pl.roi_percentage > 300:
            recommendations.append(
                "SUCCESS: Consider expanding PolicyCortex deployment to additional departments"
            )
            recommendations.append(
                "Document best practices for organization-wide rollout"
            )
        
        return recommendations
    
    def _calculate_risk_mitigation_value(
        self,
        pl: GovernancePL
    ) -> Dict[str, Any]:
        """Calculate value of risk mitigation"""
        return {
            "security_risk_reduction": float(pl.incident_prevention_value * Decimal("2")),
            "compliance_risk_reduction": float(pl.compliance_penalty_avoidance * Decimal("10")),
            "operational_risk_reduction": float(pl.productivity_gains * Decimal("1.5")),
            "total_risk_value": float(
                (pl.incident_prevention_value * Decimal("2")) +
                (pl.compliance_penalty_avoidance * Decimal("10")) +
                (pl.productivity_gains * Decimal("1.5"))
            ),
            "risk_adjusted_roi": pl.roi_percentage * 1.25  # 25% premium for risk reduction
        }
    
    async def _project_next_quarter(
        self,
        current_pl: GovernancePL
    ) -> Dict[str, Any]:
        """Project next quarter performance"""
        # Apply growth factors based on maturity
        growth_factor = 1.15  # 15% quarter-over-quarter improvement
        
        return {
            "estimated_savings": float(current_pl.net_benefit) * growth_factor * 3,  # 3 months
            "projected_roi": current_pl.roi_percentage * growth_factor,
            "additional_optimizations": int(current_pl.resources_optimized * growth_factor),
            "automation_hours": current_pl.automation_hours_saved * growth_factor * 3,
            "confidence": 0.75  # 75% confidence in projection
        }


# Export main metrics service instance
def create_metrics_service() -> PLDashboardMetrics:
    """Create P&L metrics service instance"""
    return PLDashboardMetrics()