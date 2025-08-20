"""
Policy Translation Engine for Patent #2: Conversational Governance Intelligence System

This module implements the natural language to cloud policy translation system,
converting user requirements into valid Azure Policy, AWS Config Rules, and
GCP Organization Policies with syntax validation and semantic verification.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import uuid

# NLP libraries
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers for policy translation"""
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    MULTI_CLOUD = "multi_cloud"


class PolicyType(Enum):
    """Types of governance policies"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    COST = "cost"
    RESOURCE = "resource"
    ACCESS = "access"
    NETWORK = "network"
    DATA = "data"


@dataclass
class PolicyRequirement:
    """Natural language policy requirement"""
    text: str
    intent: str
    entities: Dict[str, List[str]]
    constraints: List[str]
    compliance_frameworks: List[str]
    severity: str
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslatedPolicy:
    """Translated cloud policy with validation"""
    policy_id: str
    name: str
    description: str
    cloud_provider: CloudProvider
    policy_type: PolicyType
    policy_json: Dict[str, Any]
    validation_status: str
    validation_errors: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    natural_language_source: str


class PolicyTemplateLibrary:
    """Library of policy templates for common governance patterns"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize policy templates for each cloud provider"""
        templates = {
            "azure": {
                "vm_encryption": {
                    "name": "Require VM Disk Encryption",
                    "template": {
                        "properties": {
                            "displayName": "Require disk encryption on VMs",
                            "policyType": "Custom",
                            "mode": "All",
                            "parameters": {},
                            "policyRule": {
                                "if": {
                                    "allOf": [
                                        {
                                            "field": "type",
                                            "equals": "Microsoft.Compute/virtualMachines"
                                        },
                                        {
                                            "field": "Microsoft.Compute/virtualMachines/storageProfile.osDisk.encryptionSettings.enabled",
                                            "notEquals": "true"
                                        }
                                    ]
                                },
                                "then": {
                                    "effect": "deny"
                                }
                            }
                        }
                    }
                },
                "tagging_compliance": {
                    "name": "Require Mandatory Tags",
                    "template": {
                        "properties": {
                            "displayName": "Require mandatory tags on resources",
                            "policyType": "Custom",
                            "mode": "Indexed",
                            "parameters": {
                                "tagName": {
                                    "type": "String",
                                    "metadata": {
                                        "displayName": "Tag Name",
                                        "description": "Name of the required tag"
                                    }
                                }
                            },
                            "policyRule": {
                                "if": {
                                    "field": "[concat('tags[', parameters('tagName'), ']')]",
                                    "exists": "false"
                                },
                                "then": {
                                    "effect": "deny"
                                }
                            }
                        }
                    }
                },
                "network_security": {
                    "name": "Deny Public IP Assignment",
                    "template": {
                        "properties": {
                            "displayName": "Deny public IP addresses",
                            "policyType": "Custom",
                            "mode": "All",
                            "policyRule": {
                                "if": {
                                    "field": "type",
                                    "equals": "Microsoft.Network/publicIPAddresses"
                                },
                                "then": {
                                    "effect": "deny"
                                }
                            }
                        }
                    }
                }
            },
            "aws": {
                "s3_encryption": {
                    "name": "S3 Bucket Server-Side Encryption",
                    "template": {
                        "ConfigRuleName": "s3-bucket-server-side-encryption-enabled",
                        "Source": {
                            "Owner": "AWS",
                            "SourceIdentifier": "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
                        },
                        "Scope": {
                            "ComplianceResourceTypes": [
                                "AWS::S3::Bucket"
                            ]
                        }
                    }
                },
                "ec2_instance_type": {
                    "name": "Approved EC2 Instance Types",
                    "template": {
                        "ConfigRuleName": "approved-ec2-instance-types",
                        "Source": {
                            "Owner": "AWS",
                            "SourceIdentifier": "DESIRED_INSTANCE_TYPE"
                        },
                        "InputParameters": {
                            "instanceType": "t3.micro,t3.small,t3.medium"
                        },
                        "Scope": {
                            "ComplianceResourceTypes": [
                                "AWS::EC2::Instance"
                            ]
                        }
                    }
                },
                "iam_password_policy": {
                    "name": "IAM Password Policy",
                    "template": {
                        "ConfigRuleName": "iam-password-policy",
                        "Source": {
                            "Owner": "AWS",
                            "SourceIdentifier": "IAM_PASSWORD_POLICY"
                        },
                        "InputParameters": {
                            "RequireUppercaseCharacters": "true",
                            "RequireLowercaseCharacters": "true",
                            "RequireNumbers": "true",
                            "MinimumPasswordLength": "14"
                        }
                    }
                }
            },
            "gcp": {
                "compute_encryption": {
                    "name": "Require Compute Disk Encryption",
                    "template": {
                        "constraint": {
                            "displayName": "Require disk encryption",
                            "constraintDefault": {
                                "resourceTypes": [
                                    "compute.googleapis.com/Disk"
                                ],
                                "condition": {
                                    "expression": "resource.diskEncryptionKey != null",
                                    "title": "Disk must be encrypted",
                                    "description": "All disks must have encryption enabled"
                                }
                            }
                        }
                    }
                },
                "resource_location": {
                    "name": "Restrict Resource Locations",
                    "template": {
                        "constraint": {
                            "displayName": "Restrict resource locations",
                            "constraintDefault": {
                                "resourceTypes": [
                                    "compute.googleapis.com/Instance"
                                ],
                                "parameters": {
                                    "allowedLocations": {
                                        "type": "list",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                },
                                "condition": {
                                    "expression": "resource.zone in params.allowedLocations",
                                    "title": "Resource must be in allowed location"
                                }
                            }
                        }
                    }
                }
            }
        }
        return templates
    
    def get_template(
        self,
        provider: CloudProvider,
        pattern: str
    ) -> Optional[Dict[str, Any]]:
        """Get policy template for a specific pattern"""
        provider_templates = self.templates.get(provider.value, {})
        return provider_templates.get(pattern)
    
    def match_template(
        self,
        requirement: PolicyRequirement,
        provider: CloudProvider
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Match requirement to best template"""
        # Extract key terms from requirement
        text_lower = requirement.text.lower()
        
        # Pattern matching rules
        patterns = {
            "vm_encryption": ["encrypt", "vm", "disk", "virtual machine"],
            "s3_encryption": ["s3", "bucket", "encrypt", "server-side"],
            "tagging_compliance": ["tag", "mandatory", "require", "label"],
            "network_security": ["public ip", "deny", "network", "private"],
            "ec2_instance_type": ["ec2", "instance type", "approved", "allowed"],
            "iam_password_policy": ["password", "iam", "policy", "complexity"],
            "compute_encryption": ["compute", "disk", "encrypt", "gcp"],
            "resource_location": ["location", "region", "restrict", "zone"]
        }
        
        # Score each pattern
        scores = {}
        for pattern_name, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[pattern_name] = score
        
        # Get best matching pattern
        if scores:
            best_pattern = max(scores, key=scores.get)
            template = self.get_template(provider, best_pattern)
            if template:
                return best_pattern, template
        
        return None


class PolicySynthesizer:
    """Synthesize cloud policies from natural language requirements"""
    
    def __init__(self):
        self.template_library = PolicyTemplateLibrary()
        self.validators = {
            CloudProvider.AZURE: self._validate_azure_policy,
            CloudProvider.AWS: self._validate_aws_policy,
            CloudProvider.GCP: self._validate_gcp_policy
        }
    
    def synthesize_policy(
        self,
        requirement: PolicyRequirement,
        provider: CloudProvider
    ) -> TranslatedPolicy:
        """
        Synthesize cloud policy from natural language requirement
        
        Patent requirement: Convert NL to valid cloud policy JSON
        """
        # Try to match with template
        template_match = self.template_library.match_template(requirement, provider)
        
        if template_match:
            pattern_name, template = template_match
            policy_json = self._customize_template(template, requirement, provider)
        else:
            # Synthesize from scratch
            policy_json = self._synthesize_from_scratch(requirement, provider)
        
        # Generate policy metadata
        policy_id = f"pol_{uuid.uuid4().hex[:8]}"
        policy_name = self._generate_policy_name(requirement)
        policy_type = self._determine_policy_type(requirement)
        
        # Validate synthesized policy
        validation_status, validation_errors = self.validators[provider](policy_json)
        
        # Create translated policy
        translated_policy = TranslatedPolicy(
            policy_id=policy_id,
            name=policy_name,
            description=requirement.text,
            cloud_provider=provider,
            policy_type=policy_type,
            policy_json=policy_json,
            validation_status=validation_status,
            validation_errors=validation_errors,
            metadata={
                'intent': requirement.intent,
                'entities': requirement.entities,
                'compliance_frameworks': requirement.compliance_frameworks,
                'severity': requirement.severity,
                'synthesized_from': 'template' if template_match else 'scratch'
            },
            created_at=datetime.now(),
            natural_language_source=requirement.text
        )
        
        return translated_policy
    
    def _customize_template(
        self,
        template: Dict[str, Any],
        requirement: PolicyRequirement,
        provider: CloudProvider
    ) -> Dict[str, Any]:
        """Customize template based on requirement specifics"""
        policy = json.loads(json.dumps(template['template']))  # Deep copy
        
        # Extract parameters from requirement entities
        if provider == CloudProvider.AZURE:
            # Customize Azure policy
            if 'properties' in policy:
                # Update display name
                policy['properties']['displayName'] = requirement.text[:100]
                
                # Add parameters from entities
                if 'resource_type' in requirement.entities:
                    resource_types = requirement.entities['resource_type']
                    if resource_types and 'policyRule' in policy['properties']:
                        # Update resource type in policy rule
                        if 'if' in policy['properties']['policyRule']:
                            if 'field' in policy['properties']['policyRule']['if']:
                                if policy['properties']['policyRule']['if']['field'] == 'type':
                                    # Map to Azure resource type
                                    azure_type = self._map_to_azure_type(resource_types[0])
                                    policy['properties']['policyRule']['if']['equals'] = azure_type
                
                # Set effect based on severity
                if requirement.severity == 'critical':
                    if 'then' in policy['properties']['policyRule']:
                        policy['properties']['policyRule']['then']['effect'] = 'deny'
                elif requirement.severity == 'high':
                    if 'then' in policy['properties']['policyRule']:
                        policy['properties']['policyRule']['then']['effect'] = 'audit'
        
        elif provider == CloudProvider.AWS:
            # Customize AWS Config Rule
            if 'ConfigRuleName' in policy:
                policy['ConfigRuleName'] = self._generate_policy_name(requirement).replace(' ', '-').lower()
            
            # Add compliance resource types from entities
            if 'resource_type' in requirement.entities:
                resource_types = requirement.entities['resource_type']
                if resource_types and 'Scope' in policy:
                    aws_types = [self._map_to_aws_type(rt) for rt in resource_types]
                    policy['Scope']['ComplianceResourceTypes'] = aws_types
        
        elif provider == CloudProvider.GCP:
            # Customize GCP Organization Policy
            if 'constraint' in policy:
                policy['constraint']['displayName'] = requirement.text[:100]
                
                # Add resource types from entities
                if 'resource_type' in requirement.entities:
                    resource_types = requirement.entities['resource_type']
                    if resource_types and 'constraintDefault' in policy['constraint']:
                        gcp_types = [self._map_to_gcp_type(rt) for rt in resource_types]
                        policy['constraint']['constraintDefault']['resourceTypes'] = gcp_types
        
        return policy
    
    def _synthesize_from_scratch(
        self,
        requirement: PolicyRequirement,
        provider: CloudProvider
    ) -> Dict[str, Any]:
        """Synthesize policy from scratch when no template matches"""
        
        if provider == CloudProvider.AZURE:
            return self._synthesize_azure_policy(requirement)
        elif provider == CloudProvider.AWS:
            return self._synthesize_aws_policy(requirement)
        elif provider == CloudProvider.GCP:
            return self._synthesize_gcp_policy(requirement)
        else:
            return {}
    
    def _synthesize_azure_policy(self, requirement: PolicyRequirement) -> Dict[str, Any]:
        """Synthesize Azure Policy from scratch"""
        policy = {
            "properties": {
                "displayName": requirement.text[:100],
                "description": requirement.text,
                "policyType": "Custom",
                "mode": "All",
                "metadata": {
                    "category": requirement.intent,
                    "version": "1.0.0"
                },
                "parameters": {},
                "policyRule": {
                    "if": {
                        "allOf": []
                    },
                    "then": {
                        "effect": "audit"
                    }
                }
            }
        }
        
        # Build conditions from entities
        conditions = []
        
        if 'resource_type' in requirement.entities:
            for resource_type in requirement.entities['resource_type']:
                conditions.append({
                    "field": "type",
                    "equals": self._map_to_azure_type(resource_type)
                })
        
        if 'action' in requirement.entities:
            for action in requirement.entities['action']:
                if 'deny' in action.lower():
                    policy['properties']['policyRule']['then']['effect'] = 'deny'
                elif 'audit' in action.lower():
                    policy['properties']['policyRule']['then']['effect'] = 'audit'
        
        # Add constraints as conditions
        for constraint in requirement.constraints:
            if 'tag' in constraint.lower():
                conditions.append({
                    "field": "tags",
                    "exists": "true"
                })
            elif 'encrypt' in constraint.lower():
                conditions.append({
                    "field": "encryption.enabled",
                    "equals": "true"
                })
        
        if conditions:
            policy['properties']['policyRule']['if']['allOf'] = conditions
        
        return policy
    
    def _synthesize_aws_policy(self, requirement: PolicyRequirement) -> Dict[str, Any]:
        """Synthesize AWS Config Rule from scratch"""
        policy = {
            "ConfigRuleName": self._generate_policy_name(requirement).replace(' ', '-').lower(),
            "Description": requirement.text,
            "Source": {
                "Owner": "CUSTOM",
                "SourceDetails": [{
                    "EventSource": "aws.config",
                    "MessageType": "ConfigurationItemChangeNotification"
                }]
            },
            "Scope": {
                "ComplianceResourceTypes": []
            }
        }
        
        # Add resource types
        if 'resource_type' in requirement.entities:
            aws_types = [self._map_to_aws_type(rt) for rt in requirement.entities['resource_type']]
            policy['Scope']['ComplianceResourceTypes'] = aws_types
        
        # Add input parameters based on constraints
        input_params = {}
        for constraint in requirement.constraints:
            if 'encrypt' in constraint.lower():
                input_params['encryptionEnabled'] = 'true'
            elif 'tag' in constraint.lower():
                input_params['requiredTags'] = 'Environment,Owner,CostCenter'
        
        if input_params:
            policy['InputParameters'] = json.dumps(input_params)
        
        return policy
    
    def _synthesize_gcp_policy(self, requirement: PolicyRequirement) -> Dict[str, Any]:
        """Synthesize GCP Organization Policy from scratch"""
        policy = {
            "constraint": {
                "displayName": requirement.text[:100],
                "description": requirement.text,
                "constraintDefault": {
                    "resourceTypes": [],
                    "condition": {
                        "expression": "true",
                        "title": "Custom constraint",
                        "description": requirement.text
                    }
                }
            }
        }
        
        # Add resource types
        if 'resource_type' in requirement.entities:
            gcp_types = [self._map_to_gcp_type(rt) for rt in requirement.entities['resource_type']]
            policy['constraint']['constraintDefault']['resourceTypes'] = gcp_types
        
        # Build condition expression
        conditions = []
        for constraint in requirement.constraints:
            if 'encrypt' in constraint.lower():
                conditions.append("resource.encryptionKey != null")
            elif 'tag' in constraint.lower():
                conditions.append("resource.labels.size() > 0")
        
        if conditions:
            policy['constraint']['constraintDefault']['condition']['expression'] = ' && '.join(conditions)
        
        return policy
    
    def _validate_azure_policy(self, policy: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Validate Azure Policy structure and syntax"""
        errors = []
        
        # Check required fields
        if 'properties' not in policy:
            errors.append("Missing 'properties' field")
            return "invalid", errors
        
        props = policy['properties']
        
        if 'policyRule' not in props:
            errors.append("Missing 'policyRule' field")
        
        if 'displayName' not in props:
            errors.append("Missing 'displayName' field")
        
        # Validate policy rule structure
        if 'policyRule' in props:
            rule = props['policyRule']
            if 'if' not in rule:
                errors.append("Policy rule missing 'if' condition")
            if 'then' not in rule:
                errors.append("Policy rule missing 'then' action")
            
            # Validate effect
            if 'then' in rule:
                effect = rule['then'].get('effect', '')
                valid_effects = ['deny', 'audit', 'append', 'auditIfNotExists', 'deployIfNotExists']
                if effect not in valid_effects:
                    errors.append(f"Invalid effect: {effect}")
        
        return "valid" if not errors else "invalid", errors
    
    def _validate_aws_policy(self, policy: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Validate AWS Config Rule structure"""
        errors = []
        
        # Check required fields
        if 'ConfigRuleName' not in policy:
            errors.append("Missing 'ConfigRuleName' field")
        
        if 'Source' not in policy:
            errors.append("Missing 'Source' field")
        else:
            source = policy['Source']
            if 'Owner' not in source:
                errors.append("Source missing 'Owner' field")
            
            owner = source.get('Owner', '')
            if owner not in ['AWS', 'CUSTOM']:
                errors.append(f"Invalid Source Owner: {owner}")
        
        # Validate scope if present
        if 'Scope' in policy:
            scope = policy['Scope']
            if 'ComplianceResourceTypes' in scope:
                if not isinstance(scope['ComplianceResourceTypes'], list):
                    errors.append("ComplianceResourceTypes must be a list")
        
        return "valid" if not errors else "invalid", errors
    
    def _validate_gcp_policy(self, policy: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Validate GCP Organization Policy structure"""
        errors = []
        
        # Check required fields
        if 'constraint' not in policy:
            errors.append("Missing 'constraint' field")
            return "invalid", errors
        
        constraint = policy['constraint']
        
        if 'displayName' not in constraint:
            errors.append("Missing 'displayName' field")
        
        if 'constraintDefault' not in constraint:
            errors.append("Missing 'constraintDefault' field")
        else:
            default = constraint['constraintDefault']
            if 'condition' in default:
                condition = default['condition']
                if 'expression' not in condition:
                    errors.append("Condition missing 'expression' field")
        
        return "valid" if not errors else "invalid", errors
    
    def _generate_policy_name(self, requirement: PolicyRequirement) -> str:
        """Generate descriptive policy name from requirement"""
        # Extract key terms
        text = requirement.text.lower()
        
        # Common policy name patterns
        if 'encrypt' in text:
            return "Encryption Policy"
        elif 'tag' in text:
            return "Tagging Compliance Policy"
        elif 'network' in text:
            return "Network Security Policy"
        elif 'cost' in text:
            return "Cost Optimization Policy"
        elif 'access' in text:
            return "Access Control Policy"
        else:
            # Generic name from intent
            return f"{requirement.intent.replace('_', ' ').title()} Policy"
    
    def _determine_policy_type(self, requirement: PolicyRequirement) -> PolicyType:
        """Determine policy type from requirement"""
        text = requirement.text.lower()
        intent = requirement.intent.lower()
        
        if 'security' in text or 'encrypt' in text or 'secure' in text:
            return PolicyType.SECURITY
        elif 'compliance' in text or 'comply' in text or 'regulation' in text:
            return PolicyType.COMPLIANCE
        elif 'cost' in text or 'budget' in text or 'spend' in text:
            return PolicyType.COST
        elif 'access' in text or 'permission' in text or 'role' in text:
            return PolicyType.ACCESS
        elif 'network' in text or 'firewall' in text or 'subnet' in text:
            return PolicyType.NETWORK
        elif 'data' in text or 'storage' in text or 'database' in text:
            return PolicyType.DATA
        else:
            return PolicyType.RESOURCE
    
    def _map_to_azure_type(self, generic_type: str) -> str:
        """Map generic resource type to Azure type"""
        mappings = {
            'vm': 'Microsoft.Compute/virtualMachines',
            'storage': 'Microsoft.Storage/storageAccounts',
            'network': 'Microsoft.Network/virtualNetworks',
            'database': 'Microsoft.Sql/servers/databases',
            'container': 'Microsoft.ContainerInstance/containerGroups'
        }
        return mappings.get(generic_type.lower(), 'Microsoft.Resources/resources')
    
    def _map_to_aws_type(self, generic_type: str) -> str:
        """Map generic resource type to AWS type"""
        mappings = {
            'vm': 'AWS::EC2::Instance',
            'storage': 'AWS::S3::Bucket',
            'network': 'AWS::EC2::VPC',
            'database': 'AWS::RDS::DBInstance',
            'container': 'AWS::ECS::Service'
        }
        return mappings.get(generic_type.lower(), 'AWS::*::*')
    
    def _map_to_gcp_type(self, generic_type: str) -> str:
        """Map generic resource type to GCP type"""
        mappings = {
            'vm': 'compute.googleapis.com/Instance',
            'storage': 'storage.googleapis.com/Bucket',
            'network': 'compute.googleapis.com/Network',
            'database': 'sqladmin.googleapis.com/Instance',
            'container': 'container.googleapis.com/Cluster'
        }
        return mappings.get(generic_type.lower(), 'cloudresourcemanager.googleapis.com/Project')


class PolicyTranslationEngine:
    """Main engine for natural language to policy translation"""
    
    def __init__(self):
        self.synthesizer = PolicySynthesizer()
        self.policy_cache = {}
        logger.info("Policy Translation Engine initialized")
    
    def translate_requirement(
        self,
        natural_language: str,
        intent: str,
        entities: Dict[str, List[str]],
        provider: CloudProvider = CloudProvider.AZURE,
        context: Optional[Dict[str, Any]] = None
    ) -> TranslatedPolicy:
        """
        Translate natural language requirement to cloud policy
        
        Patent requirement: Complete translation pipeline
        """
        # Create policy requirement
        requirement = PolicyRequirement(
            text=natural_language,
            intent=intent,
            entities=entities,
            constraints=self._extract_constraints(natural_language),
            compliance_frameworks=self._extract_compliance_frameworks(natural_language),
            severity=self._determine_severity(natural_language, intent),
            user_context=context or {}
        )
        
        # Synthesize policy
        translated_policy = self.synthesizer.synthesize_policy(requirement, provider)
        
        # Cache policy
        cache_key = hashlib.md5(f"{natural_language}_{provider.value}".encode()).hexdigest()
        self.policy_cache[cache_key] = translated_policy
        
        logger.info(
            f"Translated policy: {translated_policy.name} "
            f"({translated_policy.validation_status})"
        )
        
        return translated_policy
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from natural language"""
        constraints = []
        
        # Encryption constraints
        if any(word in text.lower() for word in ['encrypt', 'encryption', 'encrypted']):
            constraints.append("encryption_required")
        
        # Tagging constraints
        if any(word in text.lower() for word in ['tag', 'label', 'metadata']):
            constraints.append("tagging_required")
        
        # Network constraints
        if any(word in text.lower() for word in ['private', 'public', 'internet', 'firewall']):
            constraints.append("network_restriction")
        
        # Access constraints
        if any(word in text.lower() for word in ['restrict', 'deny', 'prevent', 'block']):
            constraints.append("access_restriction")
        
        return constraints
    
    def _extract_compliance_frameworks(self, text: str) -> List[str]:
        """Extract compliance framework references"""
        frameworks = []
        
        framework_patterns = {
            'nist': r'\bNIST\b',
            'iso27001': r'\bISO\s*27001\b',
            'pci_dss': r'\bPCI[\s-]?DSS\b',
            'hipaa': r'\bHIPAA\b',
            'sox': r'\bSOX\b|\bSarbanes[\s-]?Oxley\b',
            'gdpr': r'\bGDPR\b',
            'fedramp': r'\bFedRAMP\b'
        }
        
        for framework, pattern in framework_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                frameworks.append(framework)
        
        return frameworks
    
    def _determine_severity(self, text: str, intent: str) -> str:
        """Determine policy severity level"""
        text_lower = text.lower()
        
        # Critical severity indicators
        if any(word in text_lower for word in ['critical', 'mandatory', 'must', 'require']):
            return 'critical'
        
        # High severity indicators
        if any(word in text_lower for word in ['important', 'should', 'high', 'enforce']):
            return 'high'
        
        # Medium severity (default)
        if any(word in text_lower for word in ['recommend', 'suggest', 'consider']):
            return 'medium'
        
        # Low severity
        if any(word in text_lower for word in ['optional', 'may', 'could']):
            return 'low'
        
        # Default based on intent
        if 'security' in intent.lower() or 'compliance' in intent.lower():
            return 'high'
        
        return 'medium'


# Export main components
__all__ = [
    'PolicyTranslationEngine',
    'PolicySynthesizer',
    'PolicyTemplateLibrary',
    'TranslatedPolicy',
    'PolicyRequirement',
    'CloudProvider',
    'PolicyType'
]