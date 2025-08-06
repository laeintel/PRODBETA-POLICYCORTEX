"""
Mock implementations of ML models for local testing
These provide the same API as the real models but with simulated responses
"""

import asyncio
import json
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np


class MockUnifiedAIPlatform:
    """Mock implementation of Unified AI Platform for testing"""

    def __init__(self):
        self.initialized = True

    async def analyze_governance_state(self, governance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock governance state analysis"""
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "success": True,
            "optimization_scores": [0.85, 0.72, 0.91, 0.68, 0.77],  # 5 objectives
            "domain_correlations": {
                "security_compliance": 0.78,
                "cost_performance": 0.65,
                "operations_security": 0.82,
                "compliance_cost": 0.59,
            },
            "embeddings": {
                "resource": [[0.1, 0.2, 0.3] * 42 + [0.4, 0.5]],  # 128-dim embedding
                "service": [[0.2, 0.3, 0.4] * 85 + [0.1, 0.2, 0.3]],  # 256-dim embedding
                "domain": [[0.3, 0.4, 0.5] * 170 + [0.6, 0.7]],  # 512-dim embedding
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def optimize_governance_configuration(
        self, governance_data: Dict[str, Any], preferences: Dict[str, float]
    ) -> Dict[str, Any]:
        """Mock governance optimization"""
        await asyncio.sleep(0.2)  # Simulate optimization time

        # Generate mock Pareto front
        pareto_size = random.randint(15, 25)
        pareto_front = []
        pareto_solutions = []

        for i in range(pareto_size):
            # Mock objectives (lower is better for minimization problems)
            objectives = [
                random.uniform(-0.9, -0.6),  # Security (maximized, so negative)
                random.uniform(-0.8, -0.7),  # Compliance (maximized, so negative)
                random.uniform(3000, 8000),  # Cost (minimized)
                random.uniform(-0.7, -0.5),  # Performance (maximized, so negative)
                random.uniform(100, 500),  # Operations complexity (minimized)
            ]
            pareto_front.append(objectives)

            # Mock solution variables
            solution = [random.uniform(0, 1) for _ in range(60)]
            pareto_solutions.append(solution)

        # Select "best" solution based on preferences
        best_idx = random.randint(0, pareto_size - 1)
        best_solution = {
            "solution": pareto_solutions[best_idx],
            "objectives": pareto_front[best_idx],
            "utility_score": random.uniform(0.7, 0.9),
        }

        recommendations = [
            {
                "domain": "security",
                "priority": "high",
                "action": "strengthen_access_controls",
                "description": "Implement multi-factor authentication for administrative access",
                "impact_score": random.uniform(0.6, 0.9),
            },
            {
                "domain": "cost",
                "priority": "medium",
                "action": "optimize_vm_sizing",
                "description": "Right-size underutilized virtual machines",
                "impact_score": random.uniform(0.5, 0.8),
            },
        ]

        return {
            "success": True,
            "optimization_result": {
                "pareto_front": pareto_front,
                "pareto_solutions": pareto_solutions,
                "convergence_history": [
                    {"generation": i, "best_fitness": random.uniform(0.6, 0.9)} for i in range(10)
                ],
                "execution_time": random.uniform(1.5, 3.0),
            },
            "best_solution": best_solution,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }


class MockConversationalIntelligence:
    """Mock implementation of Conversational Governance Intelligence"""

    def __init__(self):
        self.initialized = True
        self.sessions = {}

        # Mock intent responses
        self.intent_responses = {
            "policy_query": [
                "I found {count} policies related to {domain}. Here are the key policies that apply to your query.",
                "Based on your request, I've identified several relevant {domain} policies that might help.",
                "Here are the current {domain} policies that match your criteria.",
            ],
            "compliance_check": [
                "Your {standard} compliance status is currently at {score}%. I've identified {issues} areas that need attention.",
                "Compliance assessment shows {status} for {standard}. Here are the details.",
                "Based on the latest data, your {standard} compliance shows {status} with {recommendations} recommendations.",
            ],
            "security_analysis": [
                "Security analysis complete. I found {findings} security findings across your infrastructure.",
                "Your security posture shows {score} overall rating. Here are the key areas to focus on.",
                "Security assessment reveals {critical} critical and {medium} medium-priority issues.",
            ],
            "cost_optimization": [
                "I've identified ${savings} potential monthly savings through these optimization opportunities.",
                "Cost analysis shows {resources} underutilized resources that could save you ${amount} per month.",
                "Based on your usage patterns, here are {count} cost optimization recommendations.",
            ],
        }

    async def process_conversation(
        self, user_input: str, session_id: str, user_id: str
    ) -> Dict[str, Any]:
        """Mock conversation processing"""
        await asyncio.sleep(0.1)  # Simulate NLU processing

        # Mock intent classification
        intent = self._classify_mock_intent(user_input)
        entities = self._extract_mock_entities(user_input)

        # Generate mock response
        response = self._generate_mock_response(intent, entities, user_input)

        # Store in mock session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "user_id": user_id,
                "history": [],
                "entities": {},
                "state": "active",
            }

        self.sessions[session_id]["history"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "response": response,
                "intent": intent,
                "entities": entities,
            }
        )

        return {
            "success": True,
            "response": response,
            "intent": intent,
            "entities": entities,
            "confidence": random.uniform(0.75, 0.95),
            "api_call": self._generate_mock_api_call(intent, entities),
            "data": {
                "session_id": session_id,
                "turn_count": len(self.sessions[session_id]["history"]),
            },
        }

    def _classify_mock_intent(self, user_input: str) -> str:
        """Mock intent classification"""
        input_lower = user_input.lower()

        if any(word in input_lower for word in ["policy", "policies", "rule", "regulation"]):
            return "policy_query"
        elif any(
            word in input_lower for word in ["compliance", "compliant", "gdpr", "sox", "hipaa"]
        ):
            return "compliance_check"
        elif any(word in input_lower for word in ["security", "secure", "threat", "vulnerability"]):
            return "security_analysis"
        elif any(
            word in input_lower for word in ["cost", "money", "expensive", "optimize", "save"]
        ):
            return "cost_optimization"
        else:
            return "general_query"

    def _extract_mock_entities(self, user_input: str) -> Dict[str, List[str]]:
        """Mock entity extraction"""
        entities = {}
        input_lower = user_input.lower()

        # Resource types
        if "virtual machine" in input_lower or "vm" in input_lower:
            entities["resource_type"] = ["virtual_machine"]
        if "storage" in input_lower:
            entities.setdefault("resource_type", []).append("storage_account")

        # Compliance standards
        for standard in ["gdpr", "sox", "hipaa", "pci dss", "iso 27001"]:
            if standard in input_lower:
                entities.setdefault("compliance_standard", []).append(standard.upper())

        # Time periods
        if "last week" in input_lower:
            entities["time_period"] = ["last_week"]
        elif "last month" in input_lower:
            entities["time_period"] = ["last_month"]

        return entities

    def _generate_mock_response(
        self, intent: str, entities: Dict[str, List[str]], user_input: str
    ) -> str:
        """Generate mock response based on intent"""

        if intent in self.intent_responses:
            template = random.choice(self.intent_responses[intent])

            # Mock data for template filling
            mock_data = {
                "count": random.randint(3, 12),
                "domain": entities.get("resource_type", ["governance"])[0],
                "standard": entities.get("compliance_standard", ["GDPR"])[0],
                "score": random.randint(75, 95),
                "issues": random.randint(2, 8),
                "status": random.choice(["Compliant", "Mostly Compliant", "Needs Attention"]),
                "recommendations": random.randint(3, 7),
                "findings": random.randint(5, 15),
                "critical": random.randint(0, 3),
                "medium": random.randint(2, 8),
                "savings": random.randint(500, 3000),
                "resources": random.randint(8, 25),
                "amount": random.randint(800, 2500),
            }

            try:
                return template.format(**mock_data)
            except KeyError:
                pass

        # Fallback response
        return f"I understand you're asking about {intent.replace('_', ' ')}. Based on the current governance state, I can help you with that. Here's what I found in our analysis."

    def _generate_mock_api_call(
        self, intent: str, entities: Dict[str, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Generate mock API call specification"""

        api_calls = {
            "policy_query": {
                "endpoint": "/api/v1/azure/policies",
                "method": "GET",
                "parameters": {
                    "resource_type": entities.get("resource_type", ["all"])[0],
                    "domain": "governance",
                },
            },
            "compliance_check": {
                "endpoint": "/api/v1/azure/compliance/check",
                "method": "POST",
                "parameters": {
                    "standard": entities.get("compliance_standard", ["GDPR"])[0],
                    "scope": "all_resources",
                },
            },
            "security_analysis": {
                "endpoint": "/api/v1/ai/security/analyze",
                "method": "POST",
                "parameters": {
                    "analysis_type": "comprehensive",
                    "time_range": entities.get("time_period", ["last_30_days"])[0],
                },
            },
        }

        return api_calls.get(intent)

    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get mock conversation history"""
        if session_id in self.sessions:
            return {
                "success": True,
                "user_id": self.sessions[session_id]["user_id"],
                "history": self.sessions[session_id]["history"],
                "current_state": self.sessions[session_id]["state"],
                "entities": self.sessions[session_id]["entities"],
            }
        else:
            return {"success": False, "error": "Session not found"}

    @property
    def policy_synthesizer(self):
        """Mock policy synthesizer"""
        return MockPolicySynthesizer()


class MockPolicySynthesizer:
    """Mock policy synthesis for testing"""

    def synthesize_policy(
        self, description: str, domain: str, policy_type: str = "general"
    ) -> Dict[str, Any]:
        """Mock policy synthesis"""

        # Generate mock policy text
        policy_templates = {
            "security": {
                "network": """
                NETWORK SECURITY POLICY
                
                Purpose: {description}
                
                Policy Statement:
                1. All network traffic must be encrypted using TLS 1.2 or higher
                2. Unauthorized access attempts will be automatically blocked
                3. VPN connections are required for remote access
                4. Network security groups must follow least-privilege principles
                5. All network activity must be logged and monitored
                
                Implementation:
                - Configure NSG rules to restrict traffic
                - Enable Azure DDoS protection
                - Implement Web Application Firewall
                - Regular security assessments required
                
                Compliance: This policy ensures compliance with security frameworks
                """,
                "identity": """
                IDENTITY AND ACCESS MANAGEMENT POLICY
                
                Purpose: {description}
                
                Policy Statement:
                1. Multi-factor authentication required for all administrative access
                2. Role-based access control must be implemented
                3. Regular access reviews conducted quarterly
                4. Privileged access must be time-limited
                5. Zero-trust principles apply to all access decisions
                
                Implementation:
                - Configure Azure AD conditional access
                - Implement Privileged Identity Management
                - Regular user access audits
                - Automated access provisioning/deprovisioning
                """,
            },
            "compliance": {
                "data_protection": """
                DATA PROTECTION POLICY
                
                Purpose: {description}
                
                Policy Statement:
                1. All personal data must be encrypted at rest and in transit
                2. Data retention policies must be clearly defined
                3. Data processing must have legal basis under GDPR
                4. Data subject rights must be honored within 30 days
                5. Regular data protection impact assessments required
                
                Implementation:
                - Enable Azure Storage encryption
                - Implement data classification
                - Deploy data loss prevention controls
                - Regular compliance audits
                """
            },
        }

        # Get template or create generic one
        if domain in policy_templates and policy_type in policy_templates[domain]:
            policy_text = policy_templates[domain][policy_type].format(description=description)
        else:
            policy_text = f"""
            {domain.upper()} POLICY - {policy_type.upper()}
            
            Purpose: {description}
            
            Policy Statement:
            This policy addresses the requirements described in the purpose statement.
            All resources must comply with the following requirements:
            
            1. Security controls must be implemented according to best practices
            2. Regular monitoring and assessment is required
            3. Violations must be reported and remediated promptly
            4. Documentation must be maintained for audit purposes
            
            Implementation:
            - Deploy appropriate Azure security controls
            - Configure monitoring and alerting
            - Establish incident response procedures
            - Regular policy reviews and updates
            """

        # Generate structured policy
        structured_policy = {
            "name": f"{domain}_{policy_type}_policy",
            "domain": domain,
            "type": policy_type,
            "description": description,
            "rules": [
                "Implement appropriate security controls",
                "Monitor compliance continuously",
                "Report violations promptly",
                "Maintain audit documentation",
            ],
            "conditions": [
                "Applies to all Azure resources",
                "Effective immediately upon approval",
                "Subject to regular review",
            ],
            "actions": ["Deploy security controls", "Configure monitoring", "Establish procedures"],
        }

        return {
            "policy_text": policy_text.strip(),
            "structured_policy": structured_policy,
            "domain": domain,
            "confidence_score": random.uniform(0.75, 0.92),
            "generated_at": datetime.utcnow().isoformat(),
        }


# Global mock instances
mock_unified_ai_platform = MockUnifiedAIPlatform()
mock_conversational_intelligence = MockConversationalIntelligence()
