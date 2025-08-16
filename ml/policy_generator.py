"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/policy_generator.py
# Natural Language to Policy Generation System for PolicyCortex

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
import logging

logger = logging.getLogger(__name__)

class PolicyEffect(Enum):
    """Policy effect types"""
    DENY = "Deny"
    ALLOW = "Allow"
    AUDIT = "Audit"
    DEPLOY_IF_NOT_EXISTS = "DeployIfNotExists"
    APPEND = "Append"
    MODIFY = "Modify"

class PolicyMode(Enum):
    """Policy evaluation modes"""
    ALL = "All"
    INDEXED = "Indexed"

@dataclass
class PolicyRequirement:
    """Extracted policy requirement"""
    resource_type: str
    conditions: List[Dict[str, Any]]
    effect: PolicyEffect
    parameters: Dict[str, Any]
    exceptions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedPolicy:
    """Generated Azure Policy"""
    name: str
    display_name: str
    description: str
    policy_rule: Dict[str, Any]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    mode: PolicyMode
    validation_results: Dict[str, Any]

class PolicyGenerator:
    """Generate Azure Policies from natural language descriptions"""
    
    def __init__(self):
        self.requirement_parser = RequirementParser()
        self.condition_builder = ConditionBuilder()
        self.template_library = PolicyTemplateLibrary()
        self.validator = PolicyValidator()
    
    def generate_from_nl(self, description: str, context: Optional[Dict[str, Any]] = None) -> GeneratedPolicy:
        """Generate policy from natural language description"""
        
        # Parse requirements from description
        requirements = self.requirement_parser.parse(description)
        
        # Determine policy type and template
        template = self.template_library.get_best_template(requirements)
        
        # Build policy structure
        policy = self._build_policy(requirements, template, description)
        
        # Validate generated policy
        validation_results = self.validator.validate(policy)
        policy.validation_results = validation_results
        
        return policy
    
    def _build_policy(
        self,
        requirements: PolicyRequirement,
        template: Dict[str, Any],
        description: str
    ) -> GeneratedPolicy:
        """Build policy from requirements and template"""
        
        # Generate policy name
        policy_name = self._generate_policy_name(requirements)
        
        # Build conditions
        conditions = self.condition_builder.build_conditions(requirements.conditions)
        
        # Build policy rule
        policy_rule = {
            "if": conditions,
            "then": {
                "effect": requirements.effect.value,
                **self._build_effect_details(requirements)
            }
        }
        
        # Build parameters
        parameters = self._build_parameters(requirements.parameters)
        
        # Build metadata
        metadata = {
            "version": "1.0.0",
            "category": self._determine_category(requirements),
            "generatedFrom": "natural_language",
            "generatedAt": datetime.now().isoformat(),
            "originalDescription": description
        }
        
        return GeneratedPolicy(
            name=policy_name,
            display_name=self._generate_display_name(requirements),
            description=description,
            policy_rule=policy_rule,
            parameters=parameters,
            metadata=metadata,
            mode=PolicyMode.ALL,
            validation_results={}
        )
    
    def _generate_policy_name(self, requirements: PolicyRequirement) -> str:
        """Generate policy name from requirements"""
        # Create name from resource type and main condition
        resource_part = requirements.resource_type.replace('/', '-').lower()
        
        if requirements.conditions:
            condition_part = self._extract_main_condition(requirements.conditions[0])
        else:
            condition_part = "policy"
        
        return f"{resource_part}-{condition_part}-{datetime.now().strftime('%Y%m%d')}"
    
    def _generate_display_name(self, requirements: PolicyRequirement) -> str:
        """Generate human-readable display name"""
        resource_display = requirements.resource_type.split('/')[-1]
        
        if requirements.effect == PolicyEffect.DENY:
            return f"Deny {resource_display} without required configuration"
        elif requirements.effect == PolicyEffect.AUDIT:
            return f"Audit {resource_display} compliance"
        elif requirements.effect == PolicyEffect.DEPLOY_IF_NOT_EXISTS:
            return f"Deploy configuration for {resource_display}"
        else:
            return f"Policy for {resource_display}"
    
    def _extract_main_condition(self, condition: Dict[str, Any]) -> str:
        """Extract main condition for naming"""
        if 'field' in condition:
            field_parts = condition['field'].split('.')
            return field_parts[-1] if field_parts else 'condition'
        return 'condition'
    
    def _build_effect_details(self, requirements: PolicyRequirement) -> Dict[str, Any]:
        """Build effect-specific details"""
        details = {}
        
        if requirements.effect == PolicyEffect.DEPLOY_IF_NOT_EXISTS:
            # Add deployment template
            details["details"] = {
                "type": requirements.resource_type,
                "roleDefinitionIds": [
                    "/providers/Microsoft.Authorization/roleDefinitions/contributor"
                ],
                "deployment": {
                    "properties": {
                        "mode": "incremental",
                        "template": self._generate_deployment_template(requirements)
                    }
                }
            }
        elif requirements.effect == PolicyEffect.APPEND:
            # Add append details
            details["details"] = [
                {
                    "field": field,
                    "value": value
                }
                for field, value in requirements.parameters.items()
            ]
        elif requirements.effect == PolicyEffect.MODIFY:
            # Add modify operations
            details["details"] = {
                "operations": [
                    {
                        "operation": "add",
                        "field": field,
                        "value": value
                    }
                    for field, value in requirements.parameters.items()
                ]
            }
        
        return details
    
    def _build_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build policy parameters"""
        parameters = {}
        
        for key, value in params.items():
            param_type = self._determine_param_type(value)
            
            parameters[key] = {
                "type": param_type,
                "metadata": {
                    "displayName": self._format_display_name(key),
                    "description": f"Parameter for {key}"
                }
            }
            
            # Add default value if provided
            if value is not None:
                parameters[key]["defaultValue"] = value
        
        return parameters
    
    def _determine_param_type(self, value: Any) -> str:
        """Determine parameter type from value"""
        if isinstance(value, bool):
            return "Boolean"
        elif isinstance(value, int):
            return "Integer"
        elif isinstance(value, list):
            return "Array"
        elif isinstance(value, dict):
            return "Object"
        else:
            return "String"
    
    def _format_display_name(self, key: str) -> str:
        """Format parameter key as display name"""
        # Convert snake_case or camelCase to Title Case
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', key)
        return ' '.join(word.capitalize() for word in words)
    
    def _determine_category(self, requirements: PolicyRequirement) -> str:
        """Determine policy category"""
        resource_type = requirements.resource_type.lower()
        
        if 'storage' in resource_type:
            return "Storage"
        elif 'compute' in resource_type or 'vm' in resource_type:
            return "Compute"
        elif 'network' in resource_type:
            return "Network"
        elif 'keyvault' in resource_type or 'security' in resource_type:
            return "Security"
        elif 'sql' in resource_type or 'database' in resource_type:
            return "Database"
        else:
            return "General"
    
    def _generate_deployment_template(self, requirements: PolicyRequirement) -> Dict[str, Any]:
        """Generate ARM template for DeployIfNotExists"""
        return {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {},
            "resources": [
                {
                    "type": requirements.resource_type,
                    "apiVersion": "2021-04-01",
                    "name": "[parameters('resourceName')]",
                    "properties": requirements.parameters
                }
            ]
        }

class RequirementParser:
    """Parse policy requirements from natural language"""
    
    def __init__(self):
        self.resource_patterns = self._init_resource_patterns()
        self.condition_patterns = self._init_condition_patterns()
        self.effect_patterns = self._init_effect_patterns()
    
    def _init_resource_patterns(self) -> Dict[str, str]:
        """Initialize resource type patterns"""
        return {
            'storage': 'Microsoft.Storage/storageAccounts',
            'vm': 'Microsoft.Compute/virtualMachines',
            'database': 'Microsoft.Sql/servers/databases',
            'network': 'Microsoft.Network/virtualNetworks',
            'keyvault': 'Microsoft.KeyVault/vaults',
            'webapp': 'Microsoft.Web/sites',
            'container': 'Microsoft.ContainerService/managedClusters'
        }
    
    def _init_condition_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Initialize condition extraction patterns"""
        return [
            (re.compile(r'must have (\w+)', re.I), 'exists'),
            (re.compile(r'should be (\w+)', re.I), 'equals'),
            (re.compile(r'must be encrypted', re.I), 'encryption'),
            (re.compile(r'require (\w+) tag', re.I), 'tag'),
            (re.compile(r'only allow (\w+)', re.I), 'allow'),
            (re.compile(r'must not be public', re.I), 'not_public'),
            (re.compile(r'at least (\d+)', re.I), 'minimum'),
            (re.compile(r'no more than (\d+)', re.I), 'maximum')
        ]
    
    def _init_effect_patterns(self) -> Dict[re.Pattern, PolicyEffect]:
        """Initialize effect detection patterns"""
        return {
            re.compile(r'\b(deny|prevent|block|prohibit)\b', re.I): PolicyEffect.DENY,
            re.compile(r'\b(audit|check|monitor|track)\b', re.I): PolicyEffect.AUDIT,
            re.compile(r'\b(deploy|create|add)\b.*if.*not', re.I): PolicyEffect.DEPLOY_IF_NOT_EXISTS,
            re.compile(r'\b(append|add|attach)\b', re.I): PolicyEffect.APPEND,
            re.compile(r'\b(allow|permit|enable)\b', re.I): PolicyEffect.ALLOW,
            re.compile(r'\b(modify|update|change)\b', re.I): PolicyEffect.MODIFY
        }
    
    def parse(self, description: str) -> PolicyRequirement:
        """Parse requirements from description"""
        
        # Extract resource type
        resource_type = self._extract_resource_type(description)
        
        # Extract conditions
        conditions = self._extract_conditions(description)
        
        # Determine effect
        effect = self._determine_effect(description)
        
        # Extract parameters
        parameters = self._extract_parameters(description)
        
        # Extract exceptions
        exceptions = self._extract_exceptions(description)
        
        return PolicyRequirement(
            resource_type=resource_type,
            conditions=conditions,
            effect=effect,
            parameters=parameters,
            exceptions=exceptions
        )
    
    def _extract_resource_type(self, description: str) -> str:
        """Extract resource type from description"""
        description_lower = description.lower()
        
        for keyword, resource_type in self.resource_patterns.items():
            if keyword in description_lower:
                return resource_type
        
        # Default to generic resource
        return "Microsoft.Resources/resources"
    
    def _extract_conditions(self, description: str) -> List[Dict[str, Any]]:
        """Extract conditions from description"""
        conditions = []
        
        for pattern, condition_type in self.condition_patterns:
            matches = pattern.findall(description)
            for match in matches:
                condition = self._build_condition(condition_type, match)
                if condition:
                    conditions.append(condition)
        
        return conditions
    
    def _build_condition(self, condition_type: str, match: Any) -> Optional[Dict[str, Any]]:
        """Build condition from type and match"""
        if condition_type == 'exists':
            return {
                "field": f"properties.{match}",
                "exists": "true"
            }
        elif condition_type == 'equals':
            return {
                "field": f"properties.{match}",
                "equals": "[parameters('required_value')]"
            }
        elif condition_type == 'encryption':
            return {
                "field": "properties.encryption.services.blob.enabled",
                "equals": "true"
            }
        elif condition_type == 'tag':
            return {
                "field": f"tags['{match}']",
                "exists": "true"
            }
        elif condition_type == 'not_public':
            return {
                "field": "properties.publicNetworkAccess",
                "notEquals": "Enabled"
            }
        elif condition_type == 'minimum':
            return {
                "field": "properties.value",
                "greaterOrEquals": int(match) if isinstance(match, str) and match.isdigit() else match
            }
        elif condition_type == 'maximum':
            return {
                "field": "properties.value",
                "lessOrEquals": int(match) if isinstance(match, str) and match.isdigit() else match
            }
        
        return None
    
    def _determine_effect(self, description: str) -> PolicyEffect:
        """Determine policy effect from description"""
        for pattern, effect in self.effect_patterns.items():
            if pattern.search(description):
                return effect
        
        # Default to audit
        return PolicyEffect.AUDIT
    
    def _extract_parameters(self, description: str) -> Dict[str, Any]:
        """Extract parameters from description"""
        parameters = {}
        
        # Extract numeric values
        numbers = re.findall(r'\b(\d+)\b', description)
        if numbers:
            parameters['threshold'] = int(numbers[0])
        
        # Extract quoted strings as parameter values
        quoted = re.findall(r'"([^"]+)"', description)
        for i, value in enumerate(quoted):
            parameters[f'value_{i}'] = value
        
        # Extract specific configurations
        if 'tls' in description.lower():
            parameters['minimumTlsVersion'] = '1.2'
        
        if 'backup' in description.lower():
            parameters['backupEnabled'] = True
        
        return parameters
    
    def _extract_exceptions(self, description: str) -> List[str]:
        """Extract policy exceptions from description"""
        exceptions = []
        
        # Look for exception patterns
        exception_patterns = [
            r'except for ([^,\.]+)',
            r'excluding ([^,\.]+)',
            r'not including ([^,\.]+)'
        ]
        
        for pattern in exception_patterns:
            matches = re.findall(pattern, description, re.I)
            exceptions.extend(matches)
        
        return exceptions

class ConditionBuilder:
    """Build policy conditions from requirements"""
    
    def build_conditions(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete condition structure"""
        
        if not conditions:
            return {"field": "type", "exists": "true"}  # Default condition
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Multiple conditions - combine with allOf
        return {
            "allOf": conditions
        }
    
    def build_complex_condition(
        self,
        operator: str,
        conditions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build complex condition with logical operator"""
        return {
            operator: conditions
        }

class PolicyTemplateLibrary:
    """Library of policy templates"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load policy templates"""
        return {
            'encryption': {
                'name': 'require-encryption',
                'conditions': [
                    {
                        "field": "properties.encryption.services.blob.enabled",
                        "equals": "true"
                    }
                ],
                'effect': PolicyEffect.DENY
            },
            'tagging': {
                'name': 'require-tags',
                'conditions': [
                    {
                        "field": "tags",
                        "exists": "true"
                    }
                ],
                'effect': PolicyEffect.DENY
            },
            'network_security': {
                'name': 'secure-network',
                'conditions': [
                    {
                        "field": "properties.publicNetworkAccess",
                        "notEquals": "Enabled"
                    }
                ],
                'effect': PolicyEffect.AUDIT
            }
        }
    
    def get_best_template(self, requirements: PolicyRequirement) -> Dict[str, Any]:
        """Get best matching template for requirements"""
        
        # Match based on conditions
        for template_name, template in self.templates.items():
            if self._matches_requirements(template, requirements):
                return template
        
        # Return empty template if no match
        return {
            'name': 'custom',
            'conditions': [],
            'effect': requirements.effect
        }
    
    def _matches_requirements(
        self,
        template: Dict[str, Any],
        requirements: PolicyRequirement
    ) -> bool:
        """Check if template matches requirements"""
        # Simple matching logic - can be enhanced
        return template.get('effect') == requirements.effect

class PolicyValidator:
    """Validate generated policies"""
    
    def validate(self, policy: GeneratedPolicy) -> Dict[str, Any]:
        """Validate a generated policy"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check required fields
        if not policy.name:
            results['errors'].append("Policy name is required")
            results['valid'] = False
        
        if not policy.policy_rule:
            results['errors'].append("Policy rule is required")
            results['valid'] = False
        
        # Validate policy rule structure
        if policy.policy_rule:
            rule_validation = self._validate_policy_rule(policy.policy_rule)
            results['errors'].extend(rule_validation['errors'])
            results['warnings'].extend(rule_validation['warnings'])
        
        # Check for best practices
        if not policy.description:
            results['warnings'].append("Policy should have a description")
        
        if not policy.metadata.get('category'):
            results['warnings'].append("Policy should have a category")
        
        # Provide suggestions
        if policy.mode == PolicyMode.ALL:
            results['suggestions'].append(
                "Consider using 'Indexed' mode for better performance if not all resources need evaluation"
            )
        
        return results
    
    def _validate_policy_rule(self, rule: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate policy rule structure"""
        errors = []
        warnings = []
        
        # Check for 'if' and 'then' blocks
        if 'if' not in rule:
            errors.append("Policy rule must have an 'if' block")
        
        if 'then' not in rule:
            errors.append("Policy rule must have a 'then' block")
        
        # Check effect
        if 'then' in rule:
            if 'effect' not in rule['then']:
                errors.append("Policy rule must specify an effect")
            else:
                effect = rule['then']['effect']
                if effect not in [e.value for e in PolicyEffect]:
                    errors.append(f"Invalid effect: {effect}")
        
        return {'errors': errors, 'warnings': warnings}

# Export main components
__all__ = [
    'PolicyGenerator',
    'GeneratedPolicy',
    'PolicyRequirement',
    'PolicyEffect',
    'PolicyMode',
    'RequirementParser',
    'ConditionBuilder',
    'PolicyTemplateLibrary',
    'PolicyValidator'
]