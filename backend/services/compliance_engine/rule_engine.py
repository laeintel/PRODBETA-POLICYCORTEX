"""
Compliance Rule Engine with Advanced Pattern Matching
Executes compliance rules and provides automated remediation
"""

import re
import json
import ast
import operator
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio

import structlog
from pydantic import BaseModel, Field, validator
import yaml

logger = structlog.get_logger(__name__)

class RuleType(str, Enum):
    """Types of compliance rules"""
    POLICY = "policy"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    TAGGING = "tagging"
    COST = "cost"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

class RuleOperator(str, Enum):
    """Operators for rule conditions"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"

class RuleAction(str, Enum):
    """Actions to take when rule matches"""
    ALERT = "alert"
    BLOCK = "block"
    REMEDIATE = "remediate"
    TAG = "tag"
    LOG = "log"
    NOTIFY = "notify"
    CUSTOM = "custom"

@dataclass
class RuleCondition:
    """Represents a single rule condition"""
    field: str
    operator: RuleOperator
    value: Any
    case_sensitive: bool = True
    
class ComplianceRule(BaseModel):
    """Defines a compliance rule"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    enabled: bool = True
    conditions: List[Dict[str, Any]]
    logical_operator: str = "AND"  # AND or OR
    actions: List[Dict[str, Any]]
    severity: str = "medium"
    tags: List[str] = Field(default_factory=list)
    remediation_script: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

class RuleExecutionResult(BaseModel):
    """Result of rule execution"""
    rule_id: str
    rule_name: str
    matched: bool
    execution_time_ms: float
    conditions_evaluated: int
    conditions_matched: int
    actions_taken: List[str] = Field(default_factory=list)
    remediation_applied: bool = False
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RuleEngine:
    """
    Advanced rule engine for compliance checking and automated remediation
    """
    
    def __init__(self):
        self.rules: Dict[str, ComplianceRule] = {}
        self.custom_functions: Dict[str, Callable] = {}
        self.execution_history: List[RuleExecutionResult] = []
        self.operators = self._initialize_operators()
        self._register_default_functions()
        
    def _initialize_operators(self) -> Dict[RuleOperator, Callable]:
        """Initialize operator functions"""
        return {
            RuleOperator.EQUALS: operator.eq,
            RuleOperator.NOT_EQUALS: operator.ne,
            RuleOperator.GREATER_THAN: operator.gt,
            RuleOperator.LESS_THAN: operator.lt,
            RuleOperator.GREATER_EQUAL: operator.ge,
            RuleOperator.LESS_EQUAL: operator.le,
            RuleOperator.CONTAINS: lambda a, b: b in str(a),
            RuleOperator.NOT_CONTAINS: lambda a, b: b not in str(a),
            RuleOperator.STARTS_WITH: lambda a, b: str(a).startswith(str(b)),
            RuleOperator.ENDS_WITH: lambda a, b: str(a).endswith(str(b)),
            RuleOperator.REGEX: lambda a, b: bool(re.match(b, str(a))),
            RuleOperator.IN: lambda a, b: a in b,
            RuleOperator.NOT_IN: lambda a, b: a not in b,
            RuleOperator.EXISTS: lambda a, _: a is not None,
            RuleOperator.NOT_EXISTS: lambda a, _: a is None
        }
        
    def _register_default_functions(self):
        """Register default custom functions"""
        self.register_custom_function('check_encryption', self._check_encryption)
        self.register_custom_function('check_public_access', self._check_public_access)
        self.register_custom_function('check_backup', self._check_backup)
        self.register_custom_function('check_tags', self._check_tags)
        self.register_custom_function('check_cost_threshold', self._check_cost_threshold)
        
    def add_rule(self, rule: ComplianceRule) -> None:
        """Add a rule to the engine"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.name} (ID: {rule.rule_id})")
        
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        return False
        
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
                    
            rule.updated_at = datetime.utcnow()
            rule.version += 1
            
            logger.info(f"Updated rule: {rule_id} (version: {rule.version})")
            return True
            
        return False
        
    def register_custom_function(self, name: str, function: Callable) -> None:
        """Register a custom function for rule conditions"""
        self.custom_functions[name] = function
        logger.info(f"Registered custom function: {name}")
        
    async def evaluate_rules(self,
                           resource: Dict[str, Any],
                           rule_ids: Optional[List[str]] = None,
                           execute_actions: bool = True) -> List[RuleExecutionResult]:
        """
        Evaluate rules against a resource
        
        Args:
            resource: Resource to evaluate
            rule_ids: Specific rules to evaluate (None for all)
            execute_actions: Whether to execute actions for matched rules
            
        Returns:
            List of execution results
        """
        results = []
        
        # Determine which rules to evaluate
        rules_to_evaluate = []
        if rule_ids:
            rules_to_evaluate = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            rules_to_evaluate = [r for r in self.rules.values() if r.enabled]
            
        # Evaluate each rule
        for rule in rules_to_evaluate:
            start_time = datetime.utcnow()
            
            try:
                result = await self._evaluate_single_rule(
                    rule,
                    resource,
                    execute_actions
                )
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                result.execution_time_ms = execution_time
                
                results.append(result)
                self.execution_history.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                
                error_result = RuleExecutionResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    matched=False,
                    execution_time_ms=0,
                    conditions_evaluated=0,
                    conditions_matched=0,
                    error=str(e)
                )
                
                results.append(error_result)
                
        return results
        
    async def _evaluate_single_rule(self,
                                   rule: ComplianceRule,
                                   resource: Dict[str, Any],
                                   execute_actions: bool) -> RuleExecutionResult:
        """Evaluate a single rule against a resource"""
        conditions_evaluated = 0
        conditions_matched = 0
        condition_results = []
        
        # Evaluate each condition
        for condition in rule.conditions:
            conditions_evaluated += 1
            
            if await self._evaluate_condition(condition, resource):
                conditions_matched += 1
                condition_results.append(True)
            else:
                condition_results.append(False)
                
        # Apply logical operator
        if rule.logical_operator == "AND":
            matched = all(condition_results) if condition_results else False
        elif rule.logical_operator == "OR":
            matched = any(condition_results) if condition_results else False
        else:
            matched = False
            
        # Execute actions if rule matched
        actions_taken = []
        remediation_applied = False
        
        if matched and execute_actions:
            for action in rule.actions:
                action_result = await self._execute_action(action, resource, rule)
                actions_taken.append(action_result['action'])
                
                if action_result.get('remediation_applied'):
                    remediation_applied = True
                    
        return RuleExecutionResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            matched=matched,
            execution_time_ms=0,  # Will be set by caller
            conditions_evaluated=conditions_evaluated,
            conditions_matched=conditions_matched,
            actions_taken=actions_taken,
            remediation_applied=remediation_applied,
            details={
                'resource_id': resource.get('id'),
                'resource_type': resource.get('type'),
                'rule_severity': rule.severity
            }
        )
        
    async def _evaluate_condition(self,
                                 condition: Dict[str, Any],
                                 resource: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get('type', 'standard')
        
        if condition_type == 'standard':
            return self._evaluate_standard_condition(condition, resource)
        elif condition_type == 'custom':
            return await self._evaluate_custom_condition(condition, resource)
        elif condition_type == 'complex':
            return self._evaluate_complex_condition(condition, resource)
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
            
    def _evaluate_standard_condition(self,
                                    condition: Dict[str, Any],
                                    resource: Dict[str, Any]) -> bool:
        """Evaluate standard field-operator-value condition"""
        field = condition.get('field')
        operator_str = condition.get('operator')
        value = condition.get('value')
        case_sensitive = condition.get('case_sensitive', True)
        
        # Get field value from resource
        field_value = self._get_field_value(resource, field)
        
        # Handle case sensitivity for string comparisons
        if not case_sensitive and isinstance(field_value, str) and isinstance(value, str):
            field_value = field_value.lower()
            value = value.lower()
            
        # Get operator function
        operator_func = self.operators.get(RuleOperator(operator_str))
        
        if not operator_func:
            logger.warning(f"Unknown operator: {operator_str}")
            return False
            
        try:
            return operator_func(field_value, value)
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
            
    async def _evaluate_custom_condition(self,
                                        condition: Dict[str, Any],
                                        resource: Dict[str, Any]) -> bool:
        """Evaluate custom condition using registered function"""
        function_name = condition.get('function')
        parameters = condition.get('parameters', {})
        
        if function_name not in self.custom_functions:
            logger.warning(f"Custom function not found: {function_name}")
            return False
            
        function = self.custom_functions[function_name]
        
        try:
            # Call function with resource and parameters
            if asyncio.iscoroutinefunction(function):
                return await function(resource, **parameters)
            else:
                return function(resource, **parameters)
        except Exception as e:
            logger.error(f"Error in custom function {function_name}: {e}")
            return False
            
    def _evaluate_complex_condition(self,
                                   condition: Dict[str, Any],
                                   resource: Dict[str, Any]) -> bool:
        """Evaluate complex condition with nested logic"""
        expression = condition.get('expression')
        
        if not expression:
            return False
            
        try:
            # Parse and evaluate the expression safely
            # This is a simplified version - in production, use a proper expression parser
            node = ast.parse(expression, mode='eval')
            
            # Create evaluation context
            context = {
                'resource': resource,
                'get': lambda obj, key, default=None: obj.get(key, default),
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool
            }
            
            # Evaluate expression
            compiled = compile(node, '<string>', 'eval')
            return eval(compiled, {'__builtins__': {}}, context)
            
        except Exception as e:
            logger.error(f"Error evaluating complex condition: {e}")
            return False
            
    def _get_field_value(self, obj: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        if not field_path:
            return obj
            
        parts = field_path.split('.')
        value = obj
        
        for part in parts:
            if isinstance(value, dict):
                # Handle array index notation like 'tags[0]'
                if '[' in part and ']' in part:
                    field_name = part[:part.index('[')]
                    index = int(part[part.index('[') + 1:part.index(']')])
                    
                    if field_name in value and isinstance(value[field_name], list):
                        if index < len(value[field_name]):
                            value = value[field_name][index]
                        else:
                            return None
                    else:
                        return None
                else:
                    value = value.get(part)
            else:
                return None
                
        return value
        
    async def _execute_action(self,
                             action: Dict[str, Any],
                             resource: Dict[str, Any],
                             rule: ComplianceRule) -> Dict[str, Any]:
        """Execute an action for a matched rule"""
        action_type = action.get('type')
        result = {'action': action_type, 'success': False}
        
        try:
            if action_type == RuleAction.ALERT.value:
                result.update(await self._action_alert(action, resource, rule))
                
            elif action_type == RuleAction.REMEDIATE.value:
                result.update(await self._action_remediate(action, resource, rule))
                result['remediation_applied'] = result.get('success', False)
                
            elif action_type == RuleAction.TAG.value:
                result.update(await self._action_tag(action, resource, rule))
                
            elif action_type == RuleAction.LOG.value:
                result.update(self._action_log(action, resource, rule))
                
            elif action_type == RuleAction.NOTIFY.value:
                result.update(await self._action_notify(action, resource, rule))
                
            elif action_type == RuleAction.BLOCK.value:
                result.update(await self._action_block(action, resource, rule))
                
            elif action_type == RuleAction.CUSTOM.value:
                result.update(await self._action_custom(action, resource, rule))
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            result['error'] = str(e)
            
        return result
        
    async def _action_alert(self,
                          action: Dict[str, Any],
                          resource: Dict[str, Any],
                          rule: ComplianceRule) -> Dict[str, Any]:
        """Create an alert"""
        alert_level = action.get('level', 'warning')
        message = action.get('message', f"Rule {rule.name} matched for resource {resource.get('id')}")
        
        logger.log(
            level=alert_level.upper(),
            msg=message,
            rule_id=rule.rule_id,
            resource_id=resource.get('id')
        )
        
        return {'success': True, 'alert_level': alert_level}
        
    async def _action_remediate(self,
                               action: Dict[str, Any],
                               resource: Dict[str, Any],
                               rule: ComplianceRule) -> Dict[str, Any]:
        """Apply remediation"""
        if rule.remediation_script:
            # Execute remediation script
            logger.info(f"Executing remediation for rule {rule.rule_id}")
            
            # In production, this would execute the actual remediation
            # For now, we'll simulate it
            await asyncio.sleep(0.1)  # Simulate remediation time
            
            return {
                'success': True,
                'remediation_type': 'script',
                'script_executed': True
            }
            
        elif 'steps' in action:
            # Execute remediation steps
            steps_executed = []
            
            for step in action['steps']:
                logger.info(f"Executing remediation step: {step}")
                steps_executed.append(step)
                
            return {
                'success': True,
                'remediation_type': 'steps',
                'steps_executed': steps_executed
            }
            
        return {'success': False, 'reason': 'No remediation defined'}
        
    async def _action_tag(self,
                        action: Dict[str, Any],
                        resource: Dict[str, Any],
                        rule: ComplianceRule) -> Dict[str, Any]:
        """Add tags to resource"""
        tags = action.get('tags', {})
        
        if 'tags' not in resource:
            resource['tags'] = {}
            
        resource['tags'].update(tags)
        resource['tags'][f'compliance_{rule.rule_id}'] = 'matched'
        
        return {'success': True, 'tags_added': len(tags)}
        
    def _action_log(self,
                   action: Dict[str, Any],
                   resource: Dict[str, Any],
                   rule: ComplianceRule) -> Dict[str, Any]:
        """Log rule match"""
        log_level = action.get('level', 'info')
        message = action.get('message', f"Rule {rule.name} matched")
        
        logger.log(
            level=log_level.upper(),
            msg=message,
            rule_id=rule.rule_id,
            resource_id=resource.get('id'),
            resource_type=resource.get('type')
        )
        
        return {'success': True, 'logged': True}
        
    async def _action_notify(self,
                           action: Dict[str, Any],
                           resource: Dict[str, Any],
                           rule: ComplianceRule) -> Dict[str, Any]:
        """Send notification"""
        channels = action.get('channels', ['email'])
        recipients = action.get('recipients', [])
        
        # In production, this would send actual notifications
        logger.info(
            f"Sending notifications via {channels} to {recipients}",
            rule_id=rule.rule_id
        )
        
        return {
            'success': True,
            'channels': channels,
            'recipients_count': len(recipients)
        }
        
    async def _action_block(self,
                          action: Dict[str, Any],
                          resource: Dict[str, Any],
                          rule: ComplianceRule) -> Dict[str, Any]:
        """Block resource or operation"""
        block_type = action.get('block_type', 'access')
        duration = action.get('duration', 'permanent')
        
        logger.warning(
            f"Blocking {block_type} for resource {resource.get('id')}",
            duration=duration,
            rule_id=rule.rule_id
        )
        
        return {
            'success': True,
            'block_type': block_type,
            'duration': duration
        }
        
    async def _action_custom(self,
                           action: Dict[str, Any],
                           resource: Dict[str, Any],
                           rule: ComplianceRule) -> Dict[str, Any]:
        """Execute custom action"""
        function_name = action.get('function')
        parameters = action.get('parameters', {})
        
        if function_name in self.custom_functions:
            function = self.custom_functions[function_name]
            
            try:
                if asyncio.iscoroutinefunction(function):
                    result = await function(resource, rule, **parameters)
                else:
                    result = function(resource, rule, **parameters)
                    
                return {'success': True, 'custom_result': result}
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
                
        return {'success': False, 'reason': f"Custom function {function_name} not found"}
        
    # Default custom functions
    def _check_encryption(self, resource: Dict[str, Any], **kwargs) -> bool:
        """Check if resource is encrypted"""
        encryption_config = resource.get('properties', {}).get('encryption', {})
        return encryption_config.get('enabled', False)
        
    def _check_public_access(self, resource: Dict[str, Any], **kwargs) -> bool:
        """Check if resource allows public access"""
        network_config = resource.get('properties', {}).get('networkAcls', {})
        return network_config.get('defaultAction') != 'Allow'
        
    def _check_backup(self, resource: Dict[str, Any], **kwargs) -> bool:
        """Check if backup is configured"""
        backup_config = resource.get('properties', {}).get('backup', {})
        return backup_config.get('enabled', False)
        
    def _check_tags(self, resource: Dict[str, Any], required_tags: List[str], **kwargs) -> bool:
        """Check if resource has required tags"""
        resource_tags = resource.get('tags', {})
        return all(tag in resource_tags for tag in required_tags)
        
    def _check_cost_threshold(self, resource: Dict[str, Any], threshold: float, **kwargs) -> bool:
        """Check if resource cost exceeds threshold"""
        cost = resource.get('cost', {}).get('monthly', 0)
        return float(cost) <= threshold
        
    def export_rules(self, format: str = 'json') -> str:
        """Export rules in specified format"""
        rules_data = {
            rule_id: rule.dict() for rule_id, rule in self.rules.items()
        }
        
        if format == 'json':
            return json.dumps(rules_data, indent=2, default=str)
        elif format == 'yaml':
            return yaml.dump(rules_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def import_rules(self, data: str, format: str = 'json') -> int:
        """Import rules from data"""
        if format == 'json':
            rules_data = json.loads(data)
        elif format == 'yaml':
            rules_data = yaml.safe_load(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        imported_count = 0
        
        for rule_id, rule_data in rules_data.items():
            try:
                rule = ComplianceRule(**rule_data)
                self.add_rule(rule)
                imported_count += 1
            except Exception as e:
                logger.error(f"Error importing rule {rule_id}: {e}")
                
        return imported_count
        
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get rule execution statistics"""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'average_execution_time_ms': 0,
                'match_rate': 0,
                'remediation_rate': 0
            }
            
        total = len(self.execution_history)
        matched = sum(1 for r in self.execution_history if r.matched)
        remediated = sum(1 for r in self.execution_history if r.remediation_applied)
        avg_time = sum(r.execution_time_ms for r in self.execution_history) / total
        
        return {
            'total_executions': total,
            'matched_count': matched,
            'match_rate': (matched / total) * 100,
            'remediated_count': remediated,
            'remediation_rate': (remediated / total) * 100,
            'average_execution_time_ms': avg_time,
            'rules_count': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled)
        }