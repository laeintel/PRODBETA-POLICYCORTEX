"""
Natural Language Processing Service for AI Engine.
Handles policy analysis, text classification, and NLP operations.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential

from ....shared.config import get_settings
from ..models import AnalysisType

settings = get_settings()
logger = structlog.get_logger(__name__)


class NLPService:
    """Natural Language Processing service for policy analysis."""

    def __init__(self):
        self.settings = settings
        self.text_analytics_client = None
        self.azure_credential = None
        self.policy_keywords = self._load_policy_keywords()
        self.compliance_patterns = self._load_compliance_patterns()
        self.risk_indicators = self._load_risk_indicators()

    def _load_policy_keywords(self) -> Dict[str, List[str]]:
        """Load policy-related keywords for analysis."""
        return {
            "security": [
                "access control",
                "authentication",
                "authorization",
                "encryption",
                "firewall",
                "security group",
                "network security",
                "vulnerability",
                "threat",
                "malware",
                "intrusion",
                "breach",
                "compliance",
                "audit",
            ],
            "compliance": [
                "gdpr",
                "hipaa",
                "sox",
                "pci",
                "iso27001",
                "nist",
                "regulation",
                "standard",
                "policy",
                "procedure",
                "requirement",
                "obligation",
                "mandate",
                "governance",
                "framework",
                "control",
            ],
            "cost": [
                "budget",
                "cost",
                "expense",
                "billing",
                "charge",
                "pricing",
                "optimization",
                "saving",
                "reduction",
                "allocation",
                "usage",
                "consumption",
                "spending",
                "financial",
                "economic",
            ],
            "performance": [
                "performance",
                "latency",
                "throughput",
                "scalability",
                "availability",
                "reliability",
                "uptime",
                "downtime",
                "response time",
                "cpu",
                "memory",
                "storage",
                "network",
                "bandwidth",
                "capacity",
            ],
            "governance": [
                "governance",
                "management",
                "oversight",
                "control",
                "monitoring",
                "reporting",
                "accountability",
                "responsibility",
                "delegation",
                "approval",
                "review",
                "assessment",
                "evaluation",
                "audit",
            ],
        }

    def _load_compliance_patterns(self) -> Dict[str, List[str]]:
        """Load compliance-related patterns for analysis."""
        return {
            "data_protection": [
                r"personal\s+data",
                r"sensitive\s+information",
                r"privacy\s+policy",
                r"data\s+retention",
                r"data\s+classification",
                r"data\s+loss\s+prevention",
            ],
            "access_control": [
                r"role\s+based\s+access",
                r"least\s+privilege",
                r"multi\s+factor\s+authentication",
                r"privileged\s+access",
                r"identity\s+management",
                r"access\s+review",
            ],
            "monitoring": [
                r"audit\s+log",
                r"security\s+monitoring",
                r"continuous\s+monitoring",
                r"threat\s+detection",
                r"incident\s+response",
                r"compliance\s+reporting",
            ],
            "risk_management": [
                r"risk\s+assessment",
                r"vulnerability\s+management",
                r"business\s+continuity",
                r"disaster\s+recovery",
                r"risk\s+mitigation",
                r"security\s+controls",
            ],
        }

    def _load_risk_indicators(self) -> Dict[str, List[str]]:
        """Load risk indicator patterns."""
        return {
            "high_risk": [
                "public access",
                "no encryption",
                "weak password",
                "admin access",
                "unrestricted",
                "open firewall",
                "no monitoring",
                "deprecated",
            ],
            "medium_risk": [
                "limited access",
                "basic encryption",
                "standard password",
                "user access",
                "restricted",
                "filtered access",
                "basic monitoring",
            ],
            "low_risk": [
                "private access",
                "strong encryption",
                "complex password",
                "restricted access",
                "secure",
                "monitored",
                "updated",
            ],
        }

    async def initialize(self) -> None:
        """Initialize the NLP service."""
        try:
            logger.info("Initializing NLP service")

            # Initialize Azure Text Analytics client if available
            if self.settings.azure.client_id and self.settings.is_production():
                await self._initialize_azure_text_analytics()

            logger.info("NLP service initialized successfully")

        except Exception as e:
            logger.error("NLP service initialization failed", error=str(e))
            raise

    async def _initialize_azure_text_analytics(self) -> None:
        """Initialize Azure Text Analytics client."""
        try:
            # For production, use Azure Text Analytics
            endpoint = "https://your-text-analytics-resource.cognitiveservices.azure.com/"

            if hasattr(self.settings, "azure_text_analytics_key"):
                credential = AzureKeyCredential(self.settings.azure_text_analytics_key)
            else:
                self.azure_credential = DefaultAzureCredential()
                credential = self.azure_credential

            self.text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, credential=credential
            )

            logger.info("Azure Text Analytics client initialized")

        except Exception as e:
            logger.warning("Failed to initialize Azure Text Analytics", error=str(e))

    async def analyze_policy(
        self, policy_text: str, analysis_type: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze policy document using NLP techniques."""
        try:
            logger.info(
                "Starting policy analysis",
                text_length=len(policy_text),
                analysis_type=analysis_type,
            )

            # Initialize results
            results = {
                "analysis_type": analysis_type,
                "text_length": len(policy_text),
                "processed_at": datetime.utcnow().isoformat(),
                "confidence": 0.0,
                "key_insights": [],
                "recommendations": [],
                "compliance_status": {},
                "risk_assessment": {},
            }

            # Perform different types of analysis
            if analysis_type == AnalysisType.COMPLIANCE:
                results.update(await self._analyze_compliance(policy_text, options))
            elif analysis_type == AnalysisType.SECURITY:
                results.update(await self._analyze_security(policy_text, options))
            elif analysis_type == AnalysisType.COST:
                results.update(await self._analyze_cost(policy_text, options))
            elif analysis_type == AnalysisType.PERFORMANCE:
                results.update(await self._analyze_performance(policy_text, options))
            elif analysis_type == AnalysisType.GOVERNANCE:
                results.update(await self._analyze_governance(policy_text, options))
            else:
                # General analysis
                results.update(await self._analyze_general(policy_text, options))

            # Perform common analysis tasks
            results.update(await self._extract_key_entities(policy_text))
            results.update(await self._analyze_sentiment(policy_text))
            results.update(await self._extract_requirements(policy_text))

            logger.info(
                "Policy analysis completed",
                confidence=results["confidence"],
                insights_count=len(results["key_insights"]),
            )

            return results

        except Exception as e:
            logger.error("Policy analysis failed", error=str(e))
            raise

    async def _analyze_compliance(
        self, text: str, options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze compliance aspects of the policy."""
        try:
            results = {
                "compliance_status": {},
                "violations": [],
                "recommendations": [],
                "confidence": 0.0,
            }

            # Check for compliance framework mentions
            frameworks = ["gdpr", "hipaa", "sox", "pci", "iso27001", "nist"]
            detected_frameworks = []

            text_lower = text.lower()
            for framework in frameworks:
                if framework in text_lower:
                    detected_frameworks.append(framework.upper())

            results["compliance_status"]["frameworks"] = detected_frameworks

            # Check for compliance patterns
            compliance_score = 0
            total_patterns = 0

            for category, patterns in self.compliance_patterns.items():
                category_matches = 0
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        category_matches += 1

                if category_matches > 0:
                    compliance_score += category_matches
                    results["compliance_status"][category] = {
                        "matches": category_matches,
                        "coverage": min(category_matches / len(patterns), 1.0),
                    }

                total_patterns += len(patterns)

            # Calculate confidence
            results["confidence"] = min(compliance_score / total_patterns, 1.0)

            # Generate recommendations
            if results["confidence"] < 0.5:
                results["recommendations"].append(
                    {
                        "type": "compliance_improvement",
                        "priority": "high",
                        "description": "Policy lacks comprehensive compliance coverage",
                        "action": "Review and enhance compliance requirements",
                    }
                )

            return results

        except Exception as e:
            logger.error("Compliance analysis failed", error=str(e))
            return {"compliance_status": {}, "confidence": 0.0}

    async def _analyze_security(
        self, text: str, options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze security aspects of the policy."""
        try:
            results = {
                "security_controls": [],
                "vulnerabilities": [],
                "recommendations": [],
                "confidence": 0.0,
            }

            # Check for security keywords
            security_keywords = self.policy_keywords["security"]
            security_mentions = []

            for keyword in security_keywords:
                if keyword.lower() in text.lower():
                    security_mentions.append(keyword)

            results["security_controls"] = security_mentions

            # Check for risk indicators
            risk_score = 0
            for risk_level, indicators in self.risk_indicators.items():
                for indicator in indicators:
                    if indicator.lower() in text.lower():
                        if risk_level == "high_risk":
                            risk_score += 3
                            results["vulnerabilities"].append(
                                {
                                    "type": "high_risk_indicator",
                                    "description": f"Found high-risk indicator: {indicator}",
                                    "severity": "high",
                                }
                            )
                        elif risk_level == "medium_risk":
                            risk_score += 1
                        else:
                            risk_score -= 1

            # Calculate confidence based on security coverage
            results["confidence"] = min(len(security_mentions) / len(security_keywords), 1.0)

            # Generate security recommendations
            if len(security_mentions) < 5:
                results["recommendations"].append(
                    {
                        "type": "security_enhancement",
                        "priority": "high",
                        "description": "Policy lacks comprehensive security controls",
                        "action": "Add detailed security requirements and controls",
                    }
                )

            return results

        except Exception as e:
            logger.error("Security analysis failed", error=str(e))
            return {"security_controls": [], "confidence": 0.0}

    async def _analyze_cost(self, text: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cost-related aspects of the policy."""
        try:
            results = {
                "cost_controls": [],
                "budget_mentions": [],
                "optimization_opportunities": [],
                "recommendations": [],
                "confidence": 0.0,
            }

            # Check for cost-related keywords
            cost_keywords = self.policy_keywords["cost"]
            cost_mentions = []

            for keyword in cost_keywords:
                if keyword.lower() in text.lower():
                    cost_mentions.append(keyword)

            results["cost_controls"] = cost_mentions

            # Look for budget patterns
            budget_patterns = [
                r"\$[\d,]+",
                r"budget\s+of\s+\$?[\d,]+",
                r"cost\s+limit\s+\$?[\d,]+",
                r"spending\s+cap\s+\$?[\d,]+",
                r"maximum\s+cost\s+\$?[\d,]+",
            ]

            for pattern in budget_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                results["budget_mentions"].extend(matches)

            # Calculate confidence
            results["confidence"] = min(len(cost_mentions) / len(cost_keywords), 1.0)

            # Generate cost optimization recommendations
            if len(cost_mentions) < 3:
                results["recommendations"].append(
                    {
                        "type": "cost_optimization",
                        "priority": "medium",
                        "description": "Policy lacks cost management controls",
                        "action": "Add budget limits and cost optimization requirements",
                    }
                )

            return results

        except Exception as e:
            logger.error("Cost analysis failed", error=str(e))
            return {"cost_controls": [], "confidence": 0.0}

    async def _analyze_performance(
        self, text: str, options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance aspects of the policy."""
        try:
            results = {
                "performance_metrics": [],
                "sla_requirements": [],
                "optimization_areas": [],
                "recommendations": [],
                "confidence": 0.0,
            }

            # Check for performance keywords
            perf_keywords = self.policy_keywords["performance"]
            perf_mentions = []

            for keyword in perf_keywords:
                if keyword.lower() in text.lower():
                    perf_mentions.append(keyword)

            results["performance_metrics"] = perf_mentions

            # Look for SLA patterns
            sla_patterns = [
                r"(\d+)%\s+uptime",
                r"(\d+)\s+seconds?\s+response\s+time",
                r"(\d+)\s+ms\s+latency",
                r"(\d+)\s+minutes?\s+downtime",
            ]

            for pattern in sla_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                results["sla_requirements"].extend(matches)

            # Calculate confidence
            results["confidence"] = min(len(perf_mentions) / len(perf_keywords), 1.0)

            return results

        except Exception as e:
            logger.error("Performance analysis failed", error=str(e))
            return {"performance_metrics": [], "confidence": 0.0}

    async def _analyze_governance(
        self, text: str, options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze governance aspects of the policy."""
        try:
            results = {
                "governance_controls": [],
                "approval_processes": [],
                "monitoring_requirements": [],
                "recommendations": [],
                "confidence": 0.0,
            }

            # Check for governance keywords
            gov_keywords = self.policy_keywords["governance"]
            gov_mentions = []

            for keyword in gov_keywords:
                if keyword.lower() in text.lower():
                    gov_mentions.append(keyword)

            results["governance_controls"] = gov_mentions

            # Look for approval patterns
            approval_patterns = [
                r"approval\s+required",
                r"manager\s+approval",
                r"review\s+process",
                r"oversight\s+committee",
                r"governance\s+board",
            ]

            for pattern in approval_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    results["approval_processes"].append(pattern)

            # Calculate confidence
            results["confidence"] = min(len(gov_mentions) / len(gov_keywords), 1.0)

            return results

        except Exception as e:
            logger.error("Governance analysis failed", error=str(e))
            return {"governance_controls": [], "confidence": 0.0}

    async def _analyze_general(
        self, text: str, options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform general analysis of the policy."""
        try:
            results = {
                "document_structure": {},
                "key_topics": [],
                "complexity_score": 0.0,
                "readability_score": 0.0,
                "confidence": 0.0,
            }

            # Analyze document structure
            results["document_structure"] = {
                "paragraphs": len(text.split("\n\n")),
                "sentences": len(re.findall(r"[.!?]+", text)),
                "words": len(text.split()),
                "characters": len(text),
            }

            # Calculate complexity score (simplified)
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            complexity_score = min(avg_word_length / 10, 1.0)
            results["complexity_score"] = complexity_score

            # Calculate readability score (simplified)
            sentences = len(re.findall(r"[.!?]+", text))
            words_per_sentence = len(words) / sentences if sentences > 0 else 0
            readability_score = max(0, 1.0 - (words_per_sentence / 30))
            results["readability_score"] = readability_score

            # Extract key topics
            all_keywords = []
            for category, keywords in self.policy_keywords.items():
                all_keywords.extend(keywords)

            key_topics = []
            for keyword in all_keywords:
                if keyword.lower() in text.lower():
                    key_topics.append(keyword)

            results["key_topics"] = list(set(key_topics))[:10]  # Top 10 topics

            results["confidence"] = 0.8  # General analysis confidence

            return results

        except Exception as e:
            logger.error("General analysis failed", error=str(e))
            return {"confidence": 0.0}

    async def _extract_key_entities(self, text: str) -> Dict[str, Any]:
        """Extract key entities from the text."""
        try:
            entities = {"organizations": [], "technologies": [], "standards": [], "roles": []}

            # Simple entity extraction (would use NER in production)
            org_patterns = [r"Microsoft", r"Amazon", r"Google", r"Azure", r"AWS", r"GCP"]

            tech_patterns = [
                r"Active Directory",
                r"Kubernetes",
                r"Docker",
                r"SQL Server",
                r"Virtual Machine",
                r"Storage Account",
                r"Key Vault",
            ]

            standard_patterns = [r"ISO\s+\d+", r"NIST\s+\d+", r"RFC\s+\d+"]

            role_patterns = [
                r"administrator",
                r"manager",
                r"analyst",
                r"engineer",
                r"architect",
                r"developer",
                r"operator",
            ]

            for pattern in org_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["organizations"].extend(matches)

            for pattern in tech_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["technologies"].extend(matches)

            for pattern in standard_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["standards"].extend(matches)

            for pattern in role_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["roles"].extend(matches)

            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))

            return {"entities": entities}

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return {"entities": {}}

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the policy text."""
        try:
            # Simple sentiment analysis (would use Azure Text Analytics in production)
            positive_words = [
                "secure",
                "compliant",
                "efficient",
                "optimized",
                "protected",
                "approved",
                "authorized",
                "verified",
                "validated",
                "certified",
            ]

            negative_words = [
                "vulnerable",
                "insecure",
                "non-compliant",
                "unauthorized",
                "deprecated",
                "outdated",
                "risky",
                "exposed",
                "blocked",
            ]

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            total_words = len(text.split())
            positive_score = positive_count / total_words if total_words > 0 else 0
            negative_score = negative_count / total_words if total_words > 0 else 0

            # Determine overall sentiment
            if positive_score > negative_score:
                sentiment = "positive"
            elif negative_score > positive_score:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": {
                    "overall": sentiment,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "confidence": abs(positive_score - negative_score),
                }
            }

        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            return {"sentiment": {"overall": "neutral", "confidence": 0.0}}

    async def _extract_requirements(self, text: str) -> Dict[str, Any]:
        """Extract requirements from the policy text."""
        try:
            requirements = {"mandatory": [], "recommended": [], "prohibited": []}

            # Patterns for different requirement types
            mandatory_patterns = [
                r"must\s+([^.!?]+)",
                r"shall\s+([^.!?]+)",
                r"required\s+to\s+([^.!?]+)",
                r"mandatory\s+([^.!?]+)",
                r"obligated\s+to\s+([^.!?]+)",
            ]

            recommended_patterns = [
                r"should\s+([^.!?]+)",
                r"recommended\s+to\s+([^.!?]+)",
                r"advisable\s+to\s+([^.!?]+)",
                r"suggested\s+([^.!?]+)",
            ]

            prohibited_patterns = [
                r"must\s+not\s+([^.!?]+)",
                r"shall\s+not\s+([^.!?]+)",
                r"prohibited\s+([^.!?]+)",
                r"forbidden\s+([^.!?]+)",
                r"not\s+allowed\s+([^.!?]+)",
            ]

            for pattern in mandatory_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                requirements["mandatory"].extend([match.strip() for match in matches])

            for pattern in recommended_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                requirements["recommended"].extend([match.strip() for match in matches])

            for pattern in prohibited_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                requirements["prohibited"].extend([match.strip() for match in matches])

            return {"requirements": requirements}

        except Exception as e:
            logger.error("Requirements extraction failed", error=str(e))
            return {"requirements": {}}

    def is_ready(self) -> bool:
        """Check if NLP service is ready."""
        return True  # Always ready for basic NLP operations

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            if self.text_analytics_client:
                await self.text_analytics_client.close()

            logger.info("NLP service cleanup completed")

        except Exception as e:
            logger.error("NLP service cleanup failed", error=str(e))
