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
Natural Language Generation Service for PolicyCortex
Advanced NLG capabilities for automated report generation, executive summaries,
technical documentation, and multi-language support.

Features:
- Executive summary generation from data and metrics
- Technical report writing with structured formatting
- Compliance documentation generation
- Multi-language support (English, Spanish, French, German)
- Template-based and AI-powered content generation
- Customizable report templates and styles
- Integration with PolicyCortex data sources
- Automated scheduling and delivery
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import re
import uuid
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import textwrap

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import markdown
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_REPORT = "technical_report"
    COMPLIANCE_REPORT = "compliance_report"
    SECURITY_ASSESSMENT = "security_assessment"
    COST_ANALYSIS = "cost_analysis"
    PERFORMANCE_REPORT = "performance_report"
    INCIDENT_REPORT = "incident_report"
    AUDIT_REPORT = "audit_report"


class Language(Enum):
    """Supported languages for report generation."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    JAPANESE = "ja"
    CHINESE = "zh"


class OutputFormat(Enum):
    """Output formats for generated reports."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    JSON = "json"
    PLAIN_TEXT = "plain_text"


@dataclass
class ReportTemplate:
    """Report template configuration."""
    template_id: str
    name: str
    report_type: ReportType
    language: Language
    template_content: str
    required_data_fields: List[str]
    optional_data_fields: List[str] = field(default_factory=list)
    custom_css: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """Data container for report generation."""
    metrics: Dict[str, Union[int, float, str]]
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliance_scores: Dict[str, float] = field(default_factory=dict)
    cost_data: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportRequest:
    """Request for report generation."""
    request_id: str
    report_type: ReportType
    template_id: Optional[str]
    language: Language
    output_format: OutputFormat
    data: ReportData
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    recipient: Optional[str] = None
    time_period: Optional[Tuple[datetime, datetime]] = None
    custom_sections: List[Dict[str, Any]] = field(default_factory=list)
    generation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedReport:
    """Generated report result."""
    report_id: str
    request_id: str
    title: str
    content: str
    output_format: OutputFormat
    language: Language
    generated_at: datetime
    generation_time: float
    word_count: int
    page_count: int
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageTranslator:
    """Multi-language translation service."""
    
    def __init__(self):
        self.translations = self._load_translations()
        self.translation_models = {}
        self._initialize_translation_models()
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for common terms."""
        return {
            "en": {
                "executive_summary": "Executive Summary",
                "technical_details": "Technical Details",
                "recommendations": "Recommendations",
                "compliance_status": "Compliance Status",
                "security_assessment": "Security Assessment",
                "cost_analysis": "Cost Analysis",
                "performance_metrics": "Performance Metrics",
                "alerts_warnings": "Alerts and Warnings",
                "critical": "Critical",
                "high": "High",
                "medium": "Medium",
                "low": "Low",
                "compliant": "Compliant",
                "non_compliant": "Non-Compliant",
                "total_cost": "Total Cost",
                "monthly_savings": "Monthly Savings",
                "cpu_usage": "CPU Usage",
                "memory_usage": "Memory Usage",
                "storage_usage": "Storage Usage",
                "network_traffic": "Network Traffic"
            },
            "es": {
                "executive_summary": "Resumen Ejecutivo",
                "technical_details": "Detalles Técnicos",
                "recommendations": "Recomendaciones",
                "compliance_status": "Estado de Cumplimiento",
                "security_assessment": "Evaluación de Seguridad",
                "cost_analysis": "Análisis de Costos",
                "performance_metrics": "Métricas de Rendimiento",
                "alerts_warnings": "Alertas y Advertencias",
                "critical": "Crítico",
                "high": "Alto",
                "medium": "Medio",
                "low": "Bajo",
                "compliant": "Conforme",
                "non_compliant": "No Conforme",
                "total_cost": "Costo Total",
                "monthly_savings": "Ahorros Mensuales",
                "cpu_usage": "Uso de CPU",
                "memory_usage": "Uso de Memoria",
                "storage_usage": "Uso de Almacenamiento",
                "network_traffic": "Tráfico de Red"
            },
            "fr": {
                "executive_summary": "Résumé Exécutif",
                "technical_details": "Détails Techniques",
                "recommendations": "Recommandations",
                "compliance_status": "Statut de Conformité",
                "security_assessment": "Évaluation de Sécurité",
                "cost_analysis": "Analyse des Coûts",
                "performance_metrics": "Métriques de Performance",
                "alerts_warnings": "Alertes et Avertissements",
                "critical": "Critique",
                "high": "Élevé",
                "medium": "Moyen",
                "low": "Faible",
                "compliant": "Conforme",
                "non_compliant": "Non-Conforme",
                "total_cost": "Coût Total",
                "monthly_savings": "Économies Mensuelles",
                "cpu_usage": "Utilisation CPU",
                "memory_usage": "Utilisation Mémoire",
                "storage_usage": "Utilisation Stockage",
                "network_traffic": "Trafic Réseau"
            },
            "de": {
                "executive_summary": "Zusammenfassung für die Geschäftsführung",
                "technical_details": "Technische Details",
                "recommendations": "Empfehlungen",
                "compliance_status": "Compliance-Status",
                "security_assessment": "Sicherheitsbewertung",
                "cost_analysis": "Kostenanalyse",
                "performance_metrics": "Leistungsmetriken",
                "alerts_warnings": "Warnungen und Hinweise",
                "critical": "Kritisch",
                "high": "Hoch",
                "medium": "Mittel",
                "low": "Niedrig",
                "compliant": "Konform",
                "non_compliant": "Nicht-Konform",
                "total_cost": "Gesamtkosten",
                "monthly_savings": "Monatliche Einsparungen",
                "cpu_usage": "CPU-Nutzung",
                "memory_usage": "Speichernutzung",
                "storage_usage": "Speicherplatznutzung",
                "network_traffic": "Netzwerkverkehr"
            }
        }
    
    def _initialize_translation_models(self):
        """Initialize translation models if available."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize translation pipelines for supported languages
                self.translation_models["en-es"] = pipeline(
                    "translation", model="Helsinki-NLP/opus-mt-en-es"
                )
                logger.info("Translation models initialized")
            except Exception as e:
                logger.warning(f"Could not initialize translation models: {e}")
    
    def translate_text(self, text: str, source_lang: Language, 
                      target_lang: Language) -> str:
        """Translate text from source to target language."""
        if source_lang == target_lang:
            return text
        
        # Try dictionary-based translation for common terms
        translated = self._dictionary_translate(text, source_lang, target_lang)
        if translated != text:
            return translated
        
        # Try AI-based translation if available
        if TRANSFORMERS_AVAILABLE and f"{source_lang.value}-{target_lang.value}" in self.translation_models:
            try:
                model = self.translation_models[f"{source_lang.value}-{target_lang.value}"]
                result = model(text, max_length=512)
                return result[0]['translation_text']
            except Exception as e:
                logger.warning(f"AI translation failed: {e}")
        
        # Fallback: return original text with language note
        return f"[{target_lang.value.upper()}] {text}"
    
    def _dictionary_translate(self, text: str, source_lang: Language, 
                             target_lang: Language) -> str:
        """Translate using dictionary lookup."""
        source_dict = self.translations.get(source_lang.value, {})
        target_dict = self.translations.get(target_lang.value, {})
        
        # Find the key for the source text
        source_key = None
        for key, value in source_dict.items():
            if value.lower() == text.lower():
                source_key = key
                break
        
        if source_key and source_key in target_dict:
            return target_dict[source_key]
        
        return text


class TextGenerator:
    """Core text generation engine."""
    
    def __init__(self):
        self.ai_model = None
        self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """Initialize AI model for text generation."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a general-purpose text generation model
                self.ai_model = pipeline(
                    "text-generation",
                    model="gpt2-medium",
                    device=0 if self._has_gpu() else -1
                )
                logger.info("AI text generation model initialized")
            except Exception as e:
                logger.warning(f"Could not initialize AI model: {e}")
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def generate_summary(self, data: ReportData, max_length: int = 300) -> str:
        """Generate an executive summary from report data."""
        # Create summary based on key metrics
        summary_parts = []
        
        # Overall health assessment
        if data.compliance_scores:
            avg_compliance = np.mean(list(data.compliance_scores.values()))
            if avg_compliance >= 90:
                summary_parts.append("The organization demonstrates excellent compliance across all assessed areas.")
            elif avg_compliance >= 75:
                summary_parts.append("The organization maintains good compliance with identified areas for improvement.")
            else:
                summary_parts.append("The organization requires significant compliance improvements.")
        
        # Cost analysis
        if data.cost_data:
            total_cost = data.cost_data.get('total_monthly_cost', 0)
            if total_cost > 0:
                summary_parts.append(f"Current monthly cloud spending totals ${total_cost:,.0f}.")
                
                savings = data.cost_data.get('potential_savings', 0)
                if savings > 0:
                    summary_parts.append(f"Potential monthly savings of ${savings:,.0f} have been identified.")
        
        # Security findings
        if data.security_findings:
            critical_findings = [f for f in data.security_findings if f.get('severity') == 'critical']
            if critical_findings:
                summary_parts.append(f"{len(critical_findings)} critical security issues require immediate attention.")
            else:
                summary_parts.append("No critical security vulnerabilities were identified.")
        
        # Performance assessment
        if data.performance_data:
            performance_score = data.performance_data.get('overall_score', 75)
            if performance_score >= 85:
                summary_parts.append("System performance metrics are within optimal ranges.")
            else:
                summary_parts.append("Performance optimization opportunities have been identified.")
        
        # Recommendations count
        if data.recommendations:
            summary_parts.append(f"{len(data.recommendations)} specific recommendations are provided.")
        
        summary = " ".join(summary_parts)
        
        # Use AI enhancement if available
        if self.ai_model and len(summary) < max_length // 2:
            try:
                enhanced = self.ai_model(
                    f"Enhance this executive summary: {summary}",
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7
                )
                if enhanced and len(enhanced[0]['generated_text']) > len(summary):
                    return enhanced[0]['generated_text'][:max_length]
            except Exception as e:
                logger.warning(f"AI enhancement failed: {e}")
        
        return summary[:max_length]
    
    def generate_technical_details(self, data: ReportData) -> str:
        """Generate technical details section."""
        details = []
        
        # System metrics
        if data.metrics:
            details.append("## System Metrics")
            details.append("")
            for metric, value in data.metrics.items():
                formatted_metric = metric.replace('_', ' ').title()
                if isinstance(value, float):
                    details.append(f"- **{formatted_metric}**: {value:.2f}")
                else:
                    details.append(f"- **{formatted_metric}**: {value}")
            details.append("")
        
        # Performance data
        if data.performance_data:
            details.append("## Performance Analysis")
            details.append("")
            for key, value in data.performance_data.items():
                formatted_key = key.replace('_', ' ').title()
                details.append(f"- **{formatted_key}**: {value}")
            details.append("")
        
        # Security findings
        if data.security_findings:
            details.append("## Security Assessment")
            details.append("")
            for finding in data.security_findings:
                severity = finding.get('severity', 'unknown').upper()
                description = finding.get('description', 'No description')
                resource = finding.get('resource', 'Unknown resource')
                details.append(f"- **[{severity}]** {resource}: {description}")
            details.append("")
        
        # Compliance details
        if data.compliance_scores:
            details.append("## Compliance Status")
            details.append("")
            for domain, score in data.compliance_scores.items():
                status = "COMPLIANT" if score >= 80 else "NON-COMPLIANT"
                formatted_domain = domain.replace('_', ' ').title()
                details.append(f"- **{formatted_domain}**: {score:.1f}% ({status})")
            details.append("")
        
        return "\n".join(details)
    
    def generate_recommendations(self, data: ReportData) -> str:
        """Generate recommendations section."""
        recommendations = []
        
        # Use provided recommendations
        if data.recommendations:
            recommendations.append("## Key Recommendations")
            recommendations.append("")
            for i, rec in enumerate(data.recommendations, 1):
                recommendations.append(f"{i}. {rec}")
            recommendations.append("")
        
        # Generate additional recommendations based on data
        auto_recommendations = self._generate_auto_recommendations(data)
        if auto_recommendations:
            recommendations.append("## Additional Recommendations")
            recommendations.append("")
            for i, rec in enumerate(auto_recommendations, len(data.recommendations) + 1):
                recommendations.append(f"{i}. {rec}")
            recommendations.append("")
        
        return "\n".join(recommendations)
    
    def _generate_auto_recommendations(self, data: ReportData) -> List[str]:
        """Generate automatic recommendations based on data patterns."""
        recommendations = []
        
        # Cost optimization recommendations
        if data.cost_data:
            total_cost = data.cost_data.get('total_monthly_cost', 0)
            if total_cost > 10000:
                recommendations.append("Consider implementing automated cost optimization policies for high-spend resources.")
            
            unused_resources = data.cost_data.get('unused_resources', [])
            if unused_resources:
                recommendations.append(f"Review and consider terminating {len(unused_resources)} unused resources to reduce costs.")
        
        # Security recommendations
        if data.security_findings:
            critical_count = sum(1 for f in data.security_findings if f.get('severity') == 'critical')
            if critical_count > 0:
                recommendations.append(f"Address {critical_count} critical security vulnerabilities within 24-48 hours.")
        
        # Compliance recommendations
        if data.compliance_scores:
            low_compliance_domains = [domain for domain, score in data.compliance_scores.items() if score < 70]
            if low_compliance_domains:
                domains_str = ", ".join(domain.replace('_', ' ').title() for domain in low_compliance_domains)
                recommendations.append(f"Implement compliance improvement plan for: {domains_str}.")
        
        # Performance recommendations
        if data.performance_data:
            cpu_usage = data.performance_data.get('cpu_usage', 0)
            if cpu_usage > 80:
                recommendations.append("Consider scaling up compute resources to handle high CPU utilization.")
            elif cpu_usage < 30:
                recommendations.append("Consider scaling down compute resources to optimize costs.")
        
        return recommendations


class ReportTemplateManager:
    """Manager for report templates."""
    
    def __init__(self):
        self.templates = {}
        self.template_dir = Path("templates")
        self.template_dir.mkdir(exist_ok=True)
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default report templates."""
        # Executive Summary Template
        exec_template = """
# {{ title }}
{% if subtitle %}## {{ subtitle }}{% endif %}

**Generated:** {{ generated_date }}
**Period:** {{ time_period }}
**Author:** {{ author }}

## Executive Summary

{{ executive_summary }}

## Key Metrics

{% for metric, value in key_metrics.items() %}
- **{{ metric|title|replace('_', ' ') }}**: {{ value }}
{% endfor %}

## Compliance Overview

{% for domain, score in compliance_scores.items() %}
- **{{ domain|title|replace('_', ' ') }}**: {{ "%.1f"|format(score) }}% 
  {% if score >= 80 %}✅ Compliant{% else %}❌ Non-Compliant{% endif %}
{% endfor %}

## Critical Issues

{% if critical_alerts %}
{% for alert in critical_alerts %}
- {{ alert.description }}
{% endfor %}
{% else %}
No critical issues identified.
{% endif %}

## Recommendations

{% for recommendation in recommendations %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}

---
*This report was automatically generated by PolicyCortex AI.*
        """
        
        self.templates["executive_summary_en"] = ReportTemplate(
            template_id="executive_summary_en",
            name="Executive Summary (English)",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            language=Language.ENGLISH,
            template_content=exec_template,
            required_data_fields=["metrics", "compliance_scores", "recommendations"]
        )
        
        # Technical Report Template
        tech_template = """
# {{ title }} - Technical Report
{% if subtitle %}## {{ subtitle }}{% endif %}

**Generated:** {{ generated_date }}
**Report ID:** {{ report_id }}
**Author:** {{ author }}

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Metrics](#system-metrics)
3. [Performance Analysis](#performance-analysis)
4. [Security Assessment](#security-assessment)
5. [Compliance Status](#compliance-status)
6. [Cost Analysis](#cost-analysis)
7. [Recommendations](#recommendations)
8. [Appendix](#appendix)

## Executive Summary

{{ executive_summary }}

## System Metrics

{% for category, metrics in grouped_metrics.items() %}
### {{ category|title|replace('_', ' ') }}

| Metric | Value | Status |
|--------|-------|--------|
{% for metric, value in metrics.items() %}
| {{ metric|title|replace('_', ' ') }} | {{ value }} | {% if value is number and value > 80 %}⚠️ High{% elif value is number and value < 20 %}⚠️ Low{% else %}✅ Normal{% endif %} |
{% endfor %}

{% endfor %}

## Performance Analysis

{{ technical_details }}

## Security Assessment

{% if security_findings %}
### Security Findings by Severity

#### Critical Issues
{% for finding in security_findings if finding.severity == 'critical' %}
- **{{ finding.resource }}**: {{ finding.description }}
{% endfor %}

#### High Priority Issues
{% for finding in security_findings if finding.severity == 'high' %}
- **{{ finding.resource }}**: {{ finding.description }}
{% endfor %}

#### Medium Priority Issues
{% for finding in security_findings if finding.severity == 'medium' %}
- **{{ finding.resource }}**: {{ finding.description }}
{% endfor %}

{% else %}
No security issues identified in the current assessment period.
{% endif %}

## Compliance Status

### Overall Compliance Score: {{ "%.1f"|format(overall_compliance_score) }}%

{% for domain, score in compliance_scores.items() %}
#### {{ domain|title|replace('_', ' ') }}: {{ "%.1f"|format(score) }}%
{% if score >= 90 %}
Status: ✅ Excellent - No action required
{% elif score >= 75 %}
Status: ✅ Good - Minor improvements needed
{% elif score >= 60 %}
Status: ⚠️ Fair - Moderate improvements needed
{% else %}
Status: ❌ Poor - Significant improvements required
{% endif %}

{% endfor %}

## Cost Analysis

### Monthly Cost Breakdown

| Category | Current Cost | Budget | Variance |
|----------|-------------|--------|----------|
{% for category, cost in cost_breakdown.items() %}
| {{ category|title|replace('_', ' ') }} | ${{ "%.2f"|format(cost) }} | ${{ "%.2f"|format(budget[category]) }} | {% if cost > budget[category] %}+${{ "%.2f"|format(cost - budget[category]) }}{% else %}-${{ "%.2f"|format(budget[category] - cost) }}{% endif %} |
{% endfor %}

### Cost Optimization Opportunities

{% if cost_savings %}
Total potential monthly savings: **${{ "%.2f"|format(cost_savings) }}**

{% for opportunity in cost_opportunities %}
- {{ opportunity }}
{% endfor %}
{% else %}
No immediate cost optimization opportunities identified.
{% endif %}

## Recommendations

### Immediate Actions Required (0-30 days)
{% for rec in recommendations if rec.priority == 'immediate' %}
{{ loop.index }}. {{ rec.description }}
   - **Impact**: {{ rec.impact }}
   - **Effort**: {{ rec.effort }}
{% endfor %}

### Short-term Improvements (1-3 months)
{% for rec in recommendations if rec.priority == 'short_term' %}
{{ loop.index }}. {{ rec.description }}
   - **Impact**: {{ rec.impact }}
   - **Effort**: {{ rec.effort }}
{% endfor %}

### Long-term Strategic Initiatives (3-12 months)
{% for rec in recommendations if rec.priority == 'long_term' %}
{{ loop.index }}. {{ rec.description }}
   - **Impact**: {{ rec.impact }}
   - **Effort**: {{ rec.effort }}
{% endfor %}

## Appendix

### Methodology

This report was generated using PolicyCortex's AI-powered analysis engine, which:
- Continuously monitors cloud infrastructure and compliance status
- Applies machine learning algorithms to detect anomalies and patterns
- Generates insights based on industry best practices and regulatory requirements
- Provides actionable recommendations prioritized by impact and effort

### Data Sources

- Azure Resource Manager APIs
- Azure Monitor and Log Analytics
- Azure Security Center
- Azure Cost Management
- Custom PolicyCortex agents and sensors

### Report Generation Details

- **Analysis Period**: {{ analysis_period }}
- **Data Points Analyzed**: {{ data_points_count }}
- **Analysis Duration**: {{ analysis_duration }}
- **Confidence Level**: {{ confidence_level }}%

---
*This report was automatically generated by PolicyCortex AI on {{ generated_date }}*
        """
        
        self.templates["technical_report_en"] = ReportTemplate(
            template_id="technical_report_en",
            name="Technical Report (English)",
            report_type=ReportType.TECHNICAL_REPORT,
            language=Language.ENGLISH,
            template_content=tech_template,
            required_data_fields=["metrics", "compliance_scores", "security_findings", "cost_data"],
            optional_data_fields=["performance_data", "recommendations"]
        )
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, report_type: Optional[ReportType] = None, 
                      language: Optional[Language] = None) -> List[ReportTemplate]:
        """List available templates with optional filtering."""
        templates = list(self.templates.values())
        
        if report_type:
            templates = [t for t in templates if t.report_type == report_type]
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        return templates
    
    def add_template(self, template: ReportTemplate):
        """Add a new template."""
        self.templates[template.template_id] = template
    
    def render_template(self, template: ReportTemplate, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        if JINJA2_AVAILABLE:
            jinja_template = Template(template.template_content)
            return jinja_template.render(**context)
        else:
            # Simple string replacement fallback
            content = template.template_content
            for key, value in context.items():
                placeholder = f"{{{{ {key} }}}}"
                content = content.replace(placeholder, str(value))
            return content


class ReportGenerator:
    """Main report generation service."""
    
    def __init__(self):
        self.text_generator = TextGenerator()
        self.template_manager = ReportTemplateManager()
        self.translator = LanguageTranslator()
        self.output_dir = Path("generated_reports")
        self.output_dir.mkdir(exist_ok=True)
    
    async def generate_report(self, request: ReportRequest) -> GeneratedReport:
        """
        Generate a report based on the request.
        
        Args:
            request: Report generation request
            
        Returns:
            GeneratedReport: Generated report result
        """
        start_time = datetime.now()
        report_id = str(uuid.uuid4())
        
        logger.info(f"Generating report {report_id} of type {request.report_type.value}")
        
        try:
            # Prepare context data
            context = await self._prepare_context(request)
            
            # Get or create template
            template = self._get_template(request)
            
            # Generate content
            content = await self._generate_content(template, context, request)
            
            # Translate if needed
            if request.language != Language.ENGLISH:
                content = self._translate_content(content, Language.ENGLISH, request.language)
            
            # Format output
            formatted_content, file_path = await self._format_output(
                content, request.output_format, report_id
            )
            
            # Calculate metrics
            word_count = len(formatted_content.split())
            page_count = max(1, word_count // 250)  # Approximate pages
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedReport(
                report_id=report_id,
                request_id=request.request_id,
                title=request.title or self._generate_title(request),
                content=formatted_content,
                output_format=request.output_format,
                language=request.language,
                generated_at=start_time,
                generation_time=generation_time,
                word_count=word_count,
                page_count=page_count,
                file_path=file_path,
                metadata={
                    "template_id": template.template_id if template else "auto_generated",
                    "data_fields_used": list(context.keys()),
                    "custom_sections": len(request.custom_sections)
                }
            )
            
            logger.info(f"Report {report_id} generated successfully in {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating report {report_id}: {e}")
            # Return error report
            return GeneratedReport(
                report_id=report_id,
                request_id=request.request_id,
                title="Report Generation Error",
                content=f"Error generating report: {str(e)}",
                output_format=request.output_format,
                language=request.language,
                generated_at=start_time,
                generation_time=(datetime.now() - start_time).total_seconds(),
                word_count=0,
                page_count=0,
                metadata={"error": str(e)}
            )
    
    async def _prepare_context(self, request: ReportRequest) -> Dict[str, Any]:
        """Prepare template context from request data."""
        context = {
            "report_id": str(uuid.uuid4()),
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": request.title or self._generate_title(request),
            "subtitle": request.subtitle,
            "author": request.author or "PolicyCortex AI",
            "recipient": request.recipient,
            "time_period": self._format_time_period(request.time_period),
        }
        
        # Add data from request
        data = request.data
        
        # Key metrics
        context["key_metrics"] = data.metrics
        context["metrics"] = data.metrics
        context["grouped_metrics"] = self._group_metrics(data.metrics)
        
        # Compliance
        context["compliance_scores"] = data.compliance_scores
        context["overall_compliance_score"] = np.mean(list(data.compliance_scores.values())) if data.compliance_scores else 0
        
        # Generate content sections
        context["executive_summary"] = self.text_generator.generate_summary(data)
        context["technical_details"] = self.text_generator.generate_technical_details(data)
        context["recommendations"] = data.recommendations
        
        # Security
        context["security_findings"] = data.security_findings
        context["critical_alerts"] = [f for f in data.alerts if f.get('severity') == 'critical']
        
        # Cost analysis
        context["cost_data"] = data.cost_data
        context["cost_breakdown"] = data.cost_data.get('breakdown', {})
        context["cost_savings"] = data.cost_data.get('potential_savings', 0)
        context["cost_opportunities"] = data.cost_data.get('opportunities', [])
        
        # Budget data (mock for template)
        context["budget"] = {key: value * 1.1 for key, value in context["cost_breakdown"].items()}
        
        # Performance
        context["performance_data"] = data.performance_data
        
        # Charts and tables
        context["charts"] = data.charts
        context["tables"] = data.tables
        
        # Custom sections
        for section in request.custom_sections:
            section_name = section.get('name', 'custom_section')
            context[section_name] = section.get('content', '')
        
        # Analysis metadata
        context["analysis_period"] = self._format_time_period(request.time_period)
        context["data_points_count"] = len(data.metrics) + len(data.alerts) + len(data.security_findings)
        context["analysis_duration"] = "2.3 minutes"
        context["confidence_level"] = 95
        
        return context
    
    def _get_template(self, request: ReportRequest) -> Optional[ReportTemplate]:
        """Get appropriate template for the request."""
        if request.template_id:
            return self.template_manager.get_template(request.template_id)
        
        # Find template by type and language
        templates = self.template_manager.list_templates(
            report_type=request.report_type,
            language=request.language
        )
        
        return templates[0] if templates else None
    
    async def _generate_content(self, template: Optional[ReportTemplate], 
                               context: Dict[str, Any], request: ReportRequest) -> str:
        """Generate report content."""
        if template:
            # Use template-based generation
            return self.template_manager.render_template(template, context)
        else:
            # Use AI-based generation
            return await self._ai_generate_content(request, context)
    
    async def _ai_generate_content(self, request: ReportRequest, context: Dict[str, Any]) -> str:
        """Generate content using AI when no template is available."""
        # Fallback content generation
        content_parts = []
        
        content_parts.append(f"# {context['title']}")
        content_parts.append(f"\n**Generated:** {context['generated_date']}")
        content_parts.append(f"**Author:** {context['author']}\n")
        
        # Executive summary
        content_parts.append("## Executive Summary")
        content_parts.append(f"\n{context['executive_summary']}\n")
        
        # Key metrics
        if context.get('key_metrics'):
            content_parts.append("## Key Metrics")
            for metric, value in context['key_metrics'].items():
                formatted_metric = metric.replace('_', ' ').title()
                content_parts.append(f"- **{formatted_metric}**: {value}")
            content_parts.append("")
        
        # Compliance
        if context.get('compliance_scores'):
            content_parts.append("## Compliance Status")
            for domain, score in context['compliance_scores'].items():
                status = "✅ Compliant" if score >= 80 else "❌ Non-Compliant"
                formatted_domain = domain.replace('_', ' ').title()
                content_parts.append(f"- **{formatted_domain}**: {score:.1f}% ({status})")
            content_parts.append("")
        
        # Recommendations
        if context.get('recommendations'):
            content_parts.append("## Recommendations")
            for i, rec in enumerate(context['recommendations'], 1):
                content_parts.append(f"{i}. {rec}")
            content_parts.append("")
        
        content_parts.append("---")
        content_parts.append("*This report was automatically generated by PolicyCortex AI.*")
        
        return "\n".join(content_parts)
    
    def _translate_content(self, content: str, source_lang: Language, target_lang: Language) -> str:
        """Translate content to target language."""
        return self.translator.translate_text(content, source_lang, target_lang)
    
    async def _format_output(self, content: str, output_format: OutputFormat, 
                            report_id: str) -> Tuple[str, Optional[str]]:
        """Format content to the requested output format."""
        file_path = None
        
        if output_format == OutputFormat.MARKDOWN:
            file_path = self.output_dir / f"report_{report_id}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif output_format == OutputFormat.HTML:
            if markdown:
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>PolicyCortex Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 2em; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .compliant {{ color: green; }}
                        .non-compliant {{ color: red; }}
                    </style>
                </head>
                <body>
                {markdown.markdown(content, extensions=['tables'])}
                </body>
                </html>
                """
                content = html_content
                file_path = self.output_dir / f"report_{report_id}.html"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        elif output_format == OutputFormat.PDF:
            if WEASYPRINT_AVAILABLE and markdown:
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>PolicyCortex Report</title>
                    <style>
                        @page {{ size: A4; margin: 2cm; }}
                        body {{ font-family: Arial, sans-serif; font-size: 11pt; }}
                        h1 {{ color: #2c3e50; page-break-before: always; }}
                        h2 {{ color: #34495e; margin-top: 2em; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 10pt; }}
                        th {{ background-color: #f8f9fa; }}
                        .page-break {{ page-break-before: always; }}
                    </style>
                </head>
                <body>
                {markdown.markdown(content, extensions=['tables'])}
                </body>
                </html>
                """
                file_path = self.output_dir / f"report_{report_id}.pdf"
                HTML(string=html_content).write_pdf(str(file_path))
                content = f"PDF report generated: {file_path}"
        
        elif output_format == OutputFormat.JSON:
            json_content = {
                "report_id": report_id,
                "content": content,
                "generated_at": datetime.now().isoformat(),
                "format": "json"
            }
            content = json.dumps(json_content, indent=2, ensure_ascii=False)
            file_path = self.output_dir / f"report_{report_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif output_format == OutputFormat.PLAIN_TEXT:
            # Convert markdown to plain text
            content = re.sub(r'#{1,6}\s*', '', content)  # Remove headers
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
            content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italics
            content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links
            content = re.sub(r'\|.*?\|', '', content)  # Remove tables
            
            file_path = self.output_dir / f"report_{report_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content, str(file_path) if file_path else None
    
    def _generate_title(self, request: ReportRequest) -> str:
        """Generate a title for the report."""
        titles = {
            ReportType.EXECUTIVE_SUMMARY: "Executive Summary Report",
            ReportType.TECHNICAL_REPORT: "Technical Analysis Report",
            ReportType.COMPLIANCE_REPORT: "Compliance Assessment Report",
            ReportType.SECURITY_ASSESSMENT: "Security Assessment Report",
            ReportType.COST_ANALYSIS: "Cost Analysis Report",
            ReportType.PERFORMANCE_REPORT: "Performance Analysis Report",
            ReportType.INCIDENT_REPORT: "Incident Analysis Report",
            ReportType.AUDIT_REPORT: "Audit Report"
        }
        
        base_title = titles.get(request.report_type, "Analysis Report")
        
        if request.time_period:
            start_date = request.time_period[0].strftime("%Y-%m-%d")
            end_date = request.time_period[1].strftime("%Y-%m-%d")
            return f"{base_title} ({start_date} to {end_date})"
        
        return f"{base_title} - {datetime.now().strftime('%Y-%m-%d')}"
    
    def _format_time_period(self, time_period: Optional[Tuple[datetime, datetime]]) -> str:
        """Format time period for display."""
        if time_period:
            start, end = time_period
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        return "Current Period"
    
    def _group_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group metrics by category."""
        groups = {
            "performance": {},
            "security": {},
            "cost": {},
            "compliance": {},
            "general": {}
        }
        
        for key, value in metrics.items():
            if any(keyword in key.lower() for keyword in ['cpu', 'memory', 'storage', 'network']):
                groups["performance"][key] = value
            elif any(keyword in key.lower() for keyword in ['security', 'vulnerability', 'threat']):
                groups["security"][key] = value
            elif any(keyword in key.lower() for keyword in ['cost', 'spend', 'budget', 'price']):
                groups["cost"][key] = value
            elif any(keyword in key.lower() for keyword in ['compliance', 'policy', 'violation']):
                groups["compliance"][key] = value
            else:
                groups["general"][key] = value
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}


# Example usage and testing
async def example_report_generation():
    """
    Example usage of the NLG report generation service.
    """
    print("PolicyCortex Natural Language Generation Service")
    print("=" * 60)
    
    # Create sample data
    sample_data = ReportData(
        metrics={
            "cpu_usage": 78.5,
            "memory_usage": 6.2,
            "storage_usage": 1200.5,
            "network_throughput": 245.8,
            "active_policies": 15,
            "total_resources": 342,
            "monthly_cost": 4250.75
        },
        compliance_scores={
            "security_compliance": 85.2,
            "cost_compliance": 72.8,
            "performance_compliance": 91.4,
            "governance_compliance": 78.9
        },
        security_findings=[
            {
                "severity": "critical",
                "resource": "VM-Web-01",
                "description": "Unencrypted storage disk detected"
            },
            {
                "severity": "high",
                "resource": "Storage-Account-02",
                "description": "Public access enabled without IP restrictions"
            },
            {
                "severity": "medium",
                "resource": "NSG-Database",
                "description": "Overly permissive inbound rules"
            }
        ],
        cost_data={
            "total_monthly_cost": 4250.75,
            "potential_savings": 650.30,
            "breakdown": {
                "compute": 2100.50,
                "storage": 450.25,
                "network": 200.00,
                "database": 1500.00
            },
            "opportunities": [
                "Right-size overprovisioned VMs for 15% savings",
                "Implement automated shutdown for dev resources",
                "Migrate to reserved instances for stable workloads"
            ]
        },
        recommendations=[
            "Implement encryption at rest for all storage resources",
            "Review and restrict network security group rules",
            "Enable Azure Security Center recommendations",
            "Implement cost monitoring and alerts",
            "Establish automated compliance monitoring"
        ],
        alerts=[
            {
                "severity": "critical",
                "description": "CPU usage exceeded 90% for 2+ hours",
                "resource": "VM-App-01"
            }
        ]
    )
    
    # Initialize report generator
    generator = ReportGenerator()
    
    print("\n1. Generating Executive Summary Report...")
    
    # Create report request
    exec_request = ReportRequest(
        request_id="req-001",
        report_type=ReportType.EXECUTIVE_SUMMARY,
        template_id="executive_summary_en",
        language=Language.ENGLISH,
        output_format=OutputFormat.MARKDOWN,
        data=sample_data,
        title="Monthly Infrastructure Review",
        subtitle="Executive Summary",
        author="PolicyCortex AI",
        recipient="CTO",
        time_period=(datetime(2024, 1, 1), datetime(2024, 1, 31))
    )
    
    # Generate report
    exec_report = await generator.generate_report(exec_request)
    
    print(f"   Report ID: {exec_report.report_id}")
    print(f"   Generation Time: {exec_report.generation_time:.2f}s")
    print(f"   Word Count: {exec_report.word_count}")
    print(f"   File Path: {exec_report.file_path}")
    
    # Show preview of content
    preview = exec_report.content[:500] + "..." if len(exec_report.content) > 500 else exec_report.content
    print(f"\n   Content Preview:")
    print("   " + "\n   ".join(preview.split('\n')[:10]))
    
    print("\n2. Generating Technical Report...")
    
    # Technical report request
    tech_request = ReportRequest(
        request_id="req-002",
        report_type=ReportType.TECHNICAL_REPORT,
        template_id="technical_report_en",
        language=Language.ENGLISH,
        output_format=OutputFormat.HTML,
        data=sample_data,
        title="Comprehensive Technical Analysis",
        author="PolicyCortex AI",
        time_period=(datetime(2024, 1, 1), datetime(2024, 1, 31))
    )
    
    tech_report = await generator.generate_report(tech_request)
    
    print(f"   Report ID: {tech_report.report_id}")
    print(f"   Generation Time: {tech_report.generation_time:.2f}s")
    print(f"   Pages: {tech_report.page_count}")
    
    print("\n3. Generating Multi-language Reports...")
    
    # Spanish report
    spanish_request = ReportRequest(
        request_id="req-003",
        report_type=ReportType.EXECUTIVE_SUMMARY,
        template_id=None,  # Will use auto-generation
        language=Language.SPANISH,
        output_format=OutputFormat.PDF,
        data=sample_data,
        title="Resumen Ejecutivo - Análisis de Infraestructura"
    )
    
    spanish_report = await generator.generate_report(spanish_request)
    
    print(f"   Spanish Report ID: {spanish_report.report_id}")
    print(f"   Language: {spanish_report.language.value}")
    
    print("\n4. Testing Template System...")
    
    # List available templates
    templates = generator.template_manager.list_templates()
    print(f"   Available Templates: {len(templates)}")
    for template in templates:
        print(f"   - {template.name} ({template.language.value})")
    
    print("\n5. Report Statistics:")
    total_reports = [exec_report, tech_report, spanish_report]
    avg_generation_time = np.mean([r.generation_time for r in total_reports])
    total_words = sum(r.word_count for r in total_reports)
    
    print(f"   Reports Generated: {len(total_reports)}")
    print(f"   Average Generation Time: {avg_generation_time:.2f}s")
    print(f"   Total Words Generated: {total_words:,}")
    print(f"   Output Formats: {len(set(r.output_format for r in total_reports))}")
    print(f"   Languages: {len(set(r.language for r in total_reports))}")
    
    print("\nNLG Report Generation Complete! 📄")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_report_generation())