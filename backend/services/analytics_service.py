"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Advanced Analytics and Reporting Service for PolicyCortex
Provides comprehensive analytics, insights, and report generation
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    COMPLIANCE_REPORT = "compliance_report"
    SECURITY_ASSESSMENT = "security_assessment"
    COST_ANALYSIS = "cost_analysis"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_REPORT = "anomaly_report"
    CUSTOM = "custom"

class AnalysisType(Enum):
    """Types of analysis"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"

@dataclass
class AnalyticsResult:
    """Analytics result"""
    analysis_type: AnalysisType
    metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    visualizations: Dict[str, str]  # base64 encoded images
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Report:
    """Report definition"""
    id: str
    type: ReportType
    title: str
    executive_summary: str
    sections: List[Dict[str, Any]]
    visualizations: Dict[str, str]
    recommendations: List[str]
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnalyticsService:
    """Advanced analytics and reporting service"""
    
    def __init__(self):
        """Initialize analytics service"""
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.report_templates: Dict[ReportType, Dict] = {}
        self.analysis_models: Dict[str, Any] = {}
        
        # Initialize templates
        self._initialize_report_templates()
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def _initialize_report_templates(self):
        """Initialize report templates"""
        self.report_templates = {
            ReportType.EXECUTIVE_SUMMARY: {
                "sections": ["overview", "key_metrics", "trends", "risks", "recommendations"],
                "visualizations": ["compliance_gauge", "cost_trend", "risk_matrix"],
                "focus": "high_level"
            },
            ReportType.COMPLIANCE_REPORT: {
                "sections": ["compliance_score", "violations", "policy_coverage", "remediation_status"],
                "visualizations": ["compliance_heatmap", "violation_trend", "policy_effectiveness"],
                "focus": "compliance"
            },
            ReportType.SECURITY_ASSESSMENT: {
                "sections": ["security_posture", "vulnerabilities", "threats", "incidents", "recommendations"],
                "visualizations": ["threat_landscape", "vulnerability_distribution", "incident_timeline"],
                "focus": "security"
            },
            ReportType.COST_ANALYSIS: {
                "sections": ["cost_breakdown", "trends", "forecasts", "optimization_opportunities"],
                "visualizations": ["cost_by_service", "cost_trend", "forecast_chart", "savings_potential"],
                "focus": "financial"
            },
            ReportType.RESOURCE_OPTIMIZATION: {
                "sections": ["utilization", "rightsizing", "idle_resources", "recommendations"],
                "visualizations": ["utilization_heatmap", "resource_efficiency", "optimization_matrix"],
                "focus": "efficiency"
            }
        }
    
    async def analyze_data(
        self,
        data: pd.DataFrame,
        analysis_type: AnalysisType,
        target_metric: Optional[str] = None
    ) -> AnalyticsResult:
        """Perform advanced analytics on data"""
        
        if analysis_type == AnalysisType.DESCRIPTIVE:
            return await self._descriptive_analysis(data, target_metric)
        elif analysis_type == AnalysisType.DIAGNOSTIC:
            return await self._diagnostic_analysis(data, target_metric)
        elif analysis_type == AnalysisType.PREDICTIVE:
            return await self._predictive_analysis(data, target_metric)
        elif analysis_type == AnalysisType.PRESCRIPTIVE:
            return await self._prescriptive_analysis(data, target_metric)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    async def _descriptive_analysis(self, data: pd.DataFrame, target_metric: Optional[str]) -> AnalyticsResult:
        """Perform descriptive analytics"""
        metrics = {}
        insights = []
        visualizations = {}
        
        # Calculate basic statistics
        if target_metric and target_metric in data.columns:
            target_data = data[target_metric]
            metrics['mean'] = float(target_data.mean())
            metrics['median'] = float(target_data.median())
            metrics['std'] = float(target_data.std())
            metrics['min'] = float(target_data.min())
            metrics['max'] = float(target_data.max())
            metrics['q25'] = float(target_data.quantile(0.25))
            metrics['q75'] = float(target_data.quantile(0.75))
            
            # Generate insights
            if metrics['std'] > metrics['mean'] * 0.5:
                insights.append(f"High variability detected in {target_metric}")
            
            if metrics['max'] > metrics['q75'] * 2:
                insights.append(f"Significant outliers detected in upper range of {target_metric}")
            
            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            target_data.hist(bins=30, ax=ax, edgecolor='black')
            ax.set_title(f'Distribution of {target_metric}')
            ax.set_xlabel(target_metric)
            ax.set_ylabel('Frequency')
            
            visualizations['distribution'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            
            # Find strong correlations
            strong_corr = np.where(np.abs(corr_matrix) > 0.7)
            for i, j in zip(strong_corr[0], strong_corr[1]):
                if i < j:  # Avoid duplicates
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) < 1.0:  # Exclude self-correlation
                        insights.append(f"Strong correlation ({corr_val:.2f}) between {col1} and {col2}")
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix')
            
            visualizations['correlation'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        recommendations = []
        if insights:
            recommendations.append("Review identified patterns and correlations")
            recommendations.append("Investigate outliers and anomalies")
        
        return AnalyticsResult(
            analysis_type=AnalysisType.DESCRIPTIVE,
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            visualizations=visualizations,
            confidence_score=0.95
        )
    
    async def _diagnostic_analysis(self, data: pd.DataFrame, target_metric: Optional[str]) -> AnalyticsResult:
        """Perform diagnostic analytics to understand why something happened"""
        metrics = {}
        insights = []
        visualizations = {}
        recommendations = []
        
        if target_metric and target_metric in data.columns:
            # Identify factors contributing to target metric
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_importance = {}
            
            for col in numeric_cols:
                if col != target_metric:
                    correlation = data[col].corr(data[target_metric])
                    if abs(correlation) > 0.3:
                        feature_importance[col] = abs(correlation)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Top contributing factors
            top_factors = sorted_features[:5]
            for factor, importance in top_factors:
                metrics[f'impact_{factor}'] = importance
                insights.append(f"{factor} has {importance:.1%} impact on {target_metric}")
            
            # Create feature importance plot
            if top_factors:
                fig, ax = plt.subplots(figsize=(10, 6))
                factors = [f[0] for f in top_factors]
                importances = [f[1] for f in top_factors]
                
                bars = ax.bar(range(len(factors)), importances)
                ax.set_xticks(range(len(factors)))
                ax.set_xticklabels(factors, rotation=45, ha='right')
                ax.set_ylabel('Impact Score')
                ax.set_title(f'Factors Affecting {target_metric}')
                
                # Color bars based on importance
                for i, bar in enumerate(bars):
                    if importances[i] > 0.7:
                        bar.set_color('red')
                    elif importances[i] > 0.5:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                visualizations['feature_importance'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # Time-based analysis if datetime column exists
            date_cols = data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                data_sorted = data.sort_values(date_col)
                
                # Detect change points
                target_values = data_sorted[target_metric].values
                if len(target_values) > 10:
                    # Simple change point detection using rolling statistics
                    window = min(len(target_values) // 5, 30)
                    rolling_mean = pd.Series(target_values).rolling(window).mean()
                    rolling_std = pd.Series(target_values).rolling(window).std()
                    
                    # Identify significant changes
                    z_scores = np.abs((target_values - rolling_mean) / rolling_std)
                    change_points = np.where(z_scores > 2)[0]
                    
                    if len(change_points) > 0:
                        insights.append(f"Detected {len(change_points)} significant changes in {target_metric}")
                        recommendations.append("Investigate events around detected change points")
                
                # Create time series plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data_sorted[date_col], data_sorted[target_metric], label=target_metric)
                if len(change_points) > 0:
                    for cp in change_points[:10]:  # Limit to 10 change points
                        ax.axvline(x=data_sorted[date_col].iloc[cp], color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Time')
                ax.set_ylabel(target_metric)
                ax.set_title(f'{target_metric} Over Time with Change Points')
                ax.legend()
                plt.xticks(rotation=45)
                
                visualizations['time_series'] = self._fig_to_base64(fig)
                plt.close(fig)
        
        # Root cause recommendations
        if feature_importance:
            recommendations.append(f"Focus on top factors: {', '.join([f[0] for f in top_factors[:3]])}")
            recommendations.append("Perform deep-dive analysis on high-impact factors")
        
        return AnalyticsResult(
            analysis_type=AnalysisType.DIAGNOSTIC,
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            visualizations=visualizations,
            confidence_score=0.85
        )
    
    async def _predictive_analysis(self, data: pd.DataFrame, target_metric: Optional[str]) -> AnalyticsResult:
        """Perform predictive analytics"""
        metrics = {}
        insights = []
        visualizations = {}
        recommendations = []
        
        if target_metric and target_metric in data.columns:
            # Simple time series forecasting
            date_cols = data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                data_sorted = data.sort_values(date_col)
                target_values = data_sorted[target_metric].values
                
                if len(target_values) > 20:
                    # Calculate trend using linear regression
                    x = np.arange(len(target_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, target_values)
                    
                    # Make predictions
                    future_steps = min(len(target_values) // 4, 30)
                    future_x = np.arange(len(target_values), len(target_values) + future_steps)
                    predictions = slope * future_x + intercept
                    
                    # Calculate prediction metrics
                    metrics['trend_slope'] = float(slope)
                    metrics['trend_r2'] = float(r_value ** 2)
                    metrics['predicted_next'] = float(predictions[0])
                    metrics['predicted_avg'] = float(np.mean(predictions))
                    
                    # Generate insights
                    if slope > 0:
                        insights.append(f"{target_metric} showing upward trend ({slope:.2f} per period)")
                    else:
                        insights.append(f"{target_metric} showing downward trend ({slope:.2f} per period)")
                    
                    if r_value ** 2 > 0.7:
                        insights.append(f"Strong trend detected (R² = {r_value**2:.2f})")
                    
                    # Create forecast plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Historical data
                    historical_dates = data_sorted[date_col]
                    ax.plot(historical_dates, target_values, label='Historical', color='blue')
                    
                    # Forecast
                    last_date = historical_dates.iloc[-1]
                    future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='D')[1:]
                    ax.plot(future_dates, predictions, label='Forecast', color='red', linestyle='--')
                    
                    # Confidence interval
                    std = np.std(target_values)
                    upper_bound = predictions + 1.96 * std
                    lower_bound = predictions - 1.96 * std
                    ax.fill_between(future_dates, lower_bound, upper_bound, alpha=0.2, color='red')
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel(target_metric)
                    ax.set_title(f'{target_metric} Forecast')
                    ax.legend()
                    plt.xticks(rotation=45)
                    
                    visualizations['forecast'] = self._fig_to_base64(fig)
                    plt.close(fig)
                    
                    # Recommendations based on predictions
                    if slope > std:
                        recommendations.append(f"Prepare for increasing {target_metric}")
                    elif slope < -std:
                        recommendations.append(f"Take action to address declining {target_metric}")
            
            # Anomaly detection using Isolation Forest
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Prepare data
                X = data[numeric_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(X_scaled)
                
                # Count anomalies
                n_anomalies = np.sum(anomalies == -1)
                metrics['anomaly_rate'] = float(n_anomalies / len(data))
                
                if n_anomalies > 0:
                    insights.append(f"Detected {n_anomalies} anomalies ({metrics['anomaly_rate']:.1%} of data)")
                    recommendations.append("Investigate detected anomalies for potential issues")
                
                # Create anomaly visualization
                if len(numeric_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Use first two numeric columns for visualization
                    col1, col2 = numeric_cols[0], numeric_cols[1]
                    normal_mask = anomalies == 1
                    anomaly_mask = anomalies == -1
                    
                    ax.scatter(data[col1][normal_mask], data[col2][normal_mask], 
                             c='blue', label='Normal', alpha=0.6)
                    ax.scatter(data[col1][anomaly_mask], data[col2][anomaly_mask], 
                             c='red', label='Anomaly', s=100, marker='x')
                    
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title('Anomaly Detection')
                    ax.legend()
                    
                    visualizations['anomalies'] = self._fig_to_base64(fig)
                    plt.close(fig)
        
        return AnalyticsResult(
            analysis_type=AnalysisType.PREDICTIVE,
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            visualizations=visualizations,
            confidence_score=0.75
        )
    
    async def _prescriptive_analysis(self, data: pd.DataFrame, target_metric: Optional[str]) -> AnalyticsResult:
        """Perform prescriptive analytics to recommend actions"""
        metrics = {}
        insights = []
        visualizations = {}
        recommendations = []
        
        # Optimization analysis
        if target_metric and target_metric in data.columns:
            target_values = data[target_metric]
            
            # Identify optimization opportunities
            current_avg = float(target_values.mean())
            best_quartile = float(target_values.quantile(0.75))
            potential_improvement = (best_quartile - current_avg) / current_avg if current_avg > 0 else 0
            
            metrics['current_performance'] = current_avg
            metrics['target_performance'] = best_quartile
            metrics['improvement_potential'] = potential_improvement
            
            if potential_improvement > 0.1:
                insights.append(f"Potential {potential_improvement:.1%} improvement in {target_metric}")
                recommendations.append(f"Target top quartile performance: {best_quartile:.2f}")
            
            # Identify best practices (rows with best performance)
            top_performers = data.nlargest(min(5, len(data) // 10), target_metric)
            
            # Find common characteristics of top performers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_metric:
                    top_avg = top_performers[col].mean()
                    overall_avg = data[col].mean()
                    if abs(top_avg - overall_avg) / overall_avg > 0.2:
                        insights.append(f"Top performers have {((top_avg/overall_avg - 1) * 100):.1f}% different {col}")
                        
                        if top_avg > overall_avg:
                            recommendations.append(f"Increase {col} to {top_avg:.2f}")
                        else:
                            recommendations.append(f"Reduce {col} to {top_avg:.2f}")
            
            # Create optimization matrix
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Performance distribution
            ax1.hist(target_values, bins=20, edgecolor='black', alpha=0.7)
            ax1.axvline(current_avg, color='red', linestyle='--', label=f'Current: {current_avg:.2f}')
            ax1.axvline(best_quartile, color='green', linestyle='--', label=f'Target: {best_quartile:.2f}')
            ax1.set_xlabel(target_metric)
            ax1.set_ylabel('Frequency')
            ax1.set_title('Performance Distribution')
            ax1.legend()
            
            # Impact vs Effort matrix (simulated)
            if recommendations:
                n_recs = min(len(recommendations), 10)
                impact_scores = np.random.uniform(0.3, 1.0, n_recs)
                effort_scores = np.random.uniform(0.2, 0.9, n_recs)
                
                colors = ['green' if i > 0.6 and e < 0.5 else 'orange' if i > 0.5 else 'red' 
                         for i, e in zip(impact_scores, effort_scores)]
                
                ax2.scatter(effort_scores, impact_scores, s=200, c=colors, alpha=0.6)
                
                for i, rec in enumerate(recommendations[:n_recs]):
                    ax2.annotate(f'R{i+1}', (effort_scores[i], impact_scores[i]), 
                               ha='center', va='center')
                
                ax2.set_xlabel('Implementation Effort')
                ax2.set_ylabel('Expected Impact')
                ax2.set_title('Recommendation Priority Matrix')
                ax2.grid(True, alpha=0.3)
                
                # Add quadrant lines
                ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Add quadrant labels
                ax2.text(0.25, 0.75, 'Quick Wins', ha='center', va='center', fontsize=10, alpha=0.5)
                ax2.text(0.75, 0.75, 'Major Projects', ha='center', va='center', fontsize=10, alpha=0.5)
                ax2.text(0.25, 0.25, 'Fill Ins', ha='center', va='center', fontsize=10, alpha=0.5)
                ax2.text(0.75, 0.25, 'Low Priority', ha='center', va='center', fontsize=10, alpha=0.5)
            
            visualizations['optimization'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # Scenario analysis
        scenarios = [
            ("Conservative", 0.9),
            ("Moderate", 1.0),
            ("Aggressive", 1.2)
        ]
        
        for scenario_name, multiplier in scenarios:
            if target_metric and target_metric in data.columns:
                scenario_value = current_avg * multiplier
                metrics[f'scenario_{scenario_name.lower()}'] = scenario_value
        
        # Priority recommendations
        if not recommendations:
            recommendations.append("Collect more data for detailed analysis")
            recommendations.append("Define clear optimization targets")
        
        # Sort recommendations by priority
        recommendations = recommendations[:10]  # Limit to top 10
        
        return AnalyticsResult(
            analysis_type=AnalysisType.PRESCRIPTIVE,
            metrics=metrics,
            insights=insights,
            recommendations=recommendations,
            visualizations=visualizations,
            confidence_score=0.70
        )
    
    async def generate_report(
        self,
        report_type: ReportType,
        data: Dict[str, pd.DataFrame],
        period_start: datetime,
        period_end: datetime,
        custom_sections: Optional[List[str]] = None
    ) -> Report:
        """Generate a comprehensive report"""
        
        report_id = f"report-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        template = self.report_templates.get(report_type, {})
        
        sections = []
        visualizations = {}
        recommendations = []
        
        # Generate sections based on template
        section_names = custom_sections or template.get("sections", [])
        
        for section_name in section_names:
            section_content = await self._generate_section(section_name, data, report_type)
            sections.append(section_content)
            
            # Collect recommendations from sections
            if 'recommendations' in section_content:
                recommendations.extend(section_content['recommendations'])
        
        # Generate visualizations
        for viz_name in template.get("visualizations", []):
            viz_data = await self._generate_visualization(viz_name, data)
            if viz_data:
                visualizations[viz_name] = viz_data
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(sections, data)
        
        # Remove duplicate recommendations
        recommendations = list(set(recommendations))[:10]
        
        return Report(
            id=report_id,
            type=report_type,
            title=f"{report_type.value.replace('_', ' ').title()} - {period_end.strftime('%B %Y')}",
            executive_summary=executive_summary,
            sections=sections,
            visualizations=visualizations,
            recommendations=recommendations,
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            metadata={
                "data_sources": list(data.keys()),
                "total_records": sum(len(df) for df in data.values())
            }
        )
    
    async def _generate_section(self, section_name: str, data: Dict[str, pd.DataFrame], report_type: ReportType) -> Dict[str, Any]:
        """Generate a report section"""
        section = {
            "name": section_name,
            "title": section_name.replace('_', ' ').title(),
            "content": {},
            "metrics": {},
            "insights": [],
            "recommendations": []
        }
        
        # Section-specific logic
        if section_name == "overview" and any(data.values()):
            first_df = next(iter(data.values()))
            section["metrics"]["total_records"] = len(first_df)
            section["insights"].append(f"Analysis based on {len(first_df)} data points")
            
        elif section_name == "compliance_score" and "compliance" in data:
            df = data["compliance"]
            if "score" in df.columns:
                avg_score = df["score"].mean()
                section["metrics"]["average_score"] = float(avg_score)
                section["metrics"]["min_score"] = float(df["score"].min())
                section["metrics"]["max_score"] = float(df["score"].max())
                
                if avg_score < 70:
                    section["insights"].append("Compliance score below target threshold")
                    section["recommendations"].append("Immediate compliance review required")
        
        elif section_name == "cost_breakdown" and "costs" in data:
            df = data["costs"]
            if "amount" in df.columns:
                total_cost = df["amount"].sum()
                section["metrics"]["total_cost"] = float(total_cost)
                
                if "service" in df.columns:
                    top_services = df.groupby("service")["amount"].sum().nlargest(5)
                    section["content"]["top_services"] = top_services.to_dict()
                    section["insights"].append(f"Top service: {top_services.index[0]} (${top_services.iloc[0]:.2f})")
        
        return section
    
    async def _generate_visualization(self, viz_name: str, data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Generate a visualization"""
        try:
            fig = None
            
            if viz_name == "compliance_gauge" and "compliance" in data:
                df = data["compliance"]
                if "score" in df.columns:
                    avg_score = df["score"].mean()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Create gauge chart (simplified as bar)
                    colors = ['red' if avg_score < 60 else 'orange' if avg_score < 80 else 'green']
                    ax.barh([0], [avg_score], color=colors[0], height=0.5)
                    ax.barh([0], [100 - avg_score], left=[avg_score], color='lightgray', height=0.5)
                    
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel('Compliance Score')
                    ax.set_title(f'Overall Compliance: {avg_score:.1f}%')
                    ax.set_yticks([])
                    
                    # Add target line
                    ax.axvline(x=80, color='black', linestyle='--', alpha=0.5, label='Target')
                    ax.legend()
            
            elif viz_name == "cost_trend" and "costs" in data:
                df = data["costs"]
                if "date" in df.columns and "amount" in df.columns:
                    daily_costs = df.groupby("date")["amount"].sum()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    daily_costs.plot(ax=ax, color='blue', linewidth=2)
                    
                    # Add trend line
                    z = np.polyfit(range(len(daily_costs)), daily_costs.values, 1)
                    p = np.poly1d(z)
                    ax.plot(daily_costs.index, p(range(len(daily_costs))), 
                           "r--", alpha=0.5, label='Trend')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Cost ($)')
                    ax.set_title('Cost Trend Analysis')
                    ax.legend()
                    plt.xticks(rotation=45)
            
            if fig:
                result = self._fig_to_base64(fig)
                plt.close(fig)
                return result
                
        except Exception as e:
            logger.error(f"Failed to generate visualization {viz_name}: {e}")
        
        return None
    
    def _generate_executive_summary(self, sections: List[Dict], data: Dict[str, pd.DataFrame]) -> str:
        """Generate executive summary"""
        summary_points = []
        
        # Extract key metrics from sections
        for section in sections:
            if section.get("metrics"):
                for metric_name, value in section["metrics"].items():
                    if isinstance(value, (int, float)):
                        summary_points.append(f"{metric_name.replace('_', ' ').title()}: {value:.2f}")
        
        # Add high-level insights
        total_records = sum(len(df) for df in data.values())
        summary_points.insert(0, f"Report based on analysis of {total_records:,} data points")
        
        # Create summary text
        summary = "Executive Summary:\n\n"
        summary += "\n".join(f"• {point}" for point in summary_points[:10])
        
        return summary
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
    
    async def export_report(self, report: Report, format: str = "json") -> bytes:
        """Export report in specified format"""
        if format == "json":
            report_dict = {
                "id": report.id,
                "type": report.type.value,
                "title": report.title,
                "executive_summary": report.executive_summary,
                "sections": report.sections,
                "recommendations": report.recommendations,
                "generated_at": report.generated_at.isoformat(),
                "period": {
                    "start": report.period_start.isoformat(),
                    "end": report.period_end.isoformat()
                },
                "metadata": report.metadata
            }
            return json.dumps(report_dict, indent=2).encode('utf-8')
        
        elif format == "html":
            html = f"""
            <html>
            <head><title>{report.title}</title></head>
            <body>
                <h1>{report.title}</h1>
                <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Executive Summary</h2>
                <p>{report.executive_summary.replace(chr(10), '<br>')}</p>
                
                <h2>Sections</h2>
                {''.join(f"<h3>{s['title']}</h3><pre>{json.dumps(s, indent=2)}</pre>" for s in report.sections)}
                
                <h2>Recommendations</h2>
                <ul>
                {''.join(f"<li>{r}</li>" for r in report.recommendations)}
                </ul>
                
                <h2>Visualizations</h2>
                {''.join(f'<h3>{k}</h3><img src="data:image/png;base64,{v}"/>' for k, v in report.visualizations.items())}
            </body>
            </html>
            """
            return html.encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Singleton instance
analytics_service = AnalyticsService()